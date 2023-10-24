use super::memory::obtain_ctx_mut;
use crate::concurrency::thread::EvalContextExt as ThreadEvalContextExt;
use crate::helpers::EvalContextExt as HelperEvalContextExt;
use crate::shims::foreign_items::EmulateByNameResult;
use crate::shims::foreign_items::EvalContextExt as ForeignEvalContextExt;
use crate::shims::llvm::convert::to_bytes::EvalContextExt as ToBytesEvalContextExt;
use crate::shims::llvm::convert::to_generic_value::EvalContextExt as ToGenericEvalContextExt;
use crate::shims::llvm::convert::to_opty::EvalContextExt as ToOpTyEvalContextExt;
use crate::shims::llvm::helpers::EvalContextExt as LLVMHelpersEvalContextExt;
use crate::shims::llvm::hooks::memcpy::eval_memcpy;
use crate::shims::llvm::hooks::memcpy::MemcpyMode;
use crate::shims::llvm::values::generic_value::GenericValueTy;
use crate::MiriInterpCx;
use crate::MiriInterpCxExt;
use crate::MiriMemoryKind;
use crate::Provenance;
use inkwell::types::BasicTypeEnum;
use inkwell::values::GenericValue;
use inkwell::values::GenericValueRef;
use llvm_sys::execution_engine::LLVMGenericValueArrayRef;
use llvm_sys::miri::MiriInterpCxOpaque;
use llvm_sys::miri::MiriPointer;
use llvm_sys::prelude::LLVMTypeRef;
use log::debug;
use rustc_const_eval::interpret::FnVal;
use rustc_const_eval::interpret::InterpResult;
use rustc_const_eval::interpret::MemoryKind;
use rustc_const_eval::interpret::PlaceTy;
use rustc_const_eval::interpret::Scalar;
use rustc_middle::ty;
use rustc_middle::ty::layout::HasParamEnv;
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::Instance;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::abi::Size;
use rustc_target::spec::abi::Abi;
use std::ffi::CStr;
use std::ffi::CString;
use std::time::SystemTime;

fn miri_call_by_instance_result<'tcx>(
    ctx: &mut MiriInterpCx<'_, 'tcx>,
    inst: Instance<'tcx>,
    mut function_args: Vec<GenericValueTy>,
    return_type: Option<BasicTypeEnum<'static>>,
) -> InterpResult<'tcx> {
    debug!("LLVM to Rust Call: {:?}", ctx.tcx.item_name(inst.def_id()));
    let signature = ctx.tcx.fn_sig(inst.def_id());

    let signature =
        ctx.tcx.subst_and_normalize_erasing_regions(inst.args, ctx.param_env(), signature);
    let signature = ctx.tcx.liberate_late_bound_regions(inst.def_id(), signature);

    let r_arg_types = signature.inputs();
    let mut op_ty_args = vec![];

    for (lli_arg, rust_arg_type) in function_args.drain(0..).zip(r_arg_types.iter()) {
        let rust_arg_layout = ctx.layout_of(*rust_arg_type)?;
        op_ty_args.push(ctx.generic_value_to_opty(lli_arg, rust_arg_layout)?)
    }
    let return_ty = signature.output();
    let return_layout = ctx.layout_of(return_ty)?;
    debug!("Expecting return type {:?}", return_ty);

    debug!("Starting callback thread");
    ctx.eval_context_mut().start_callback_thread(
        inst,
        op_ty_args.as_slice(),
        return_layout,
        return_type,
    )?;
    Ok(())
}

fn miri_call_by_pointer_result<'tcx>(
    ctx: &mut MiriInterpCx<'_, 'tcx>,
    fn_ptr: MiriPointer,
    args_ref: LLVMGenericValueArrayRef,
    fn_ty: LLVMTypeRef,
) -> InterpResult<'tcx> {
    let resolved_fn_ptr = ctx.lli_wrapped_pointer_to_resolved_pointer(fn_ptr)?;
    let function = ctx.get_ptr_fn(resolved_fn_ptr.ptr)?;
    let (function_args, f_return) = ctx.resolve_llvm_interface(fn_ty, args_ref)?;
    match function {
        FnVal::Instance(inst) => miri_call_by_instance_result(ctx, inst, function_args, f_return),
        FnVal::Other(_) => throw_interop_format!("cannot call .dlsym functions from LLVM"),
    }
}

fn perform_opty_conversion<'tcx, 'lli>(
    ctx: &mut MiriInterpCx<'_, 'tcx>,
    return_place: &PlaceTy<'tcx, Provenance>,
    return_type_opt: &'lli Option<BasicTypeEnum<'_>>,
) -> InterpResult<'tcx, GenericValue<'lli>> {
    if let Some(return_type) = return_type_opt {
        debug!("Preparing GV return value");
        let place_opty = ctx.place_to_op(return_place)?;
        Ok(ctx.op_to_generic_value(&place_opty, Some(*return_type))?)
    } else {
        debug!("Preparing void return value");
        Ok(ctx.void_generic_value())
    }
}

pub extern "C-unwind" fn miri_call_by_pointer(
    ctx_raw: *mut MiriInterpCxOpaque,
    fn_ptr: MiriPointer,
    args_ref: LLVMGenericValueArrayRef,
    tref: LLVMTypeRef,
) -> bool {
    let ctx = obtain_ctx_mut(ctx_raw);
    let result = miri_call_by_pointer_result(ctx, fn_ptr, args_ref, tref);
    let status = result.is_err();
    if let Err(e) = result {
        ctx.set_foreign_error(e);
    }
    return status;
}

pub extern "C-unwind" fn miri_call_by_name(
    ctx_raw: *mut MiriInterpCxOpaque,
    args_ref: LLVMGenericValueArrayRef,
    name: *const ::libc::c_char,
    name_length: u64,
    tref: LLVMTypeRef,
) -> bool {
    let ctx = obtain_ctx_mut(ctx_raw);
    let result = miri_call_by_name_result(ctx, args_ref, name, name_length, tref);
    let status = result.is_err();
    if let Err(e) = result {
        ctx.set_foreign_error(e);
    }
    status
}

#[allow(clippy::arithmetic_side_effects)]
fn miri_call_by_name_result<'tcx>(
    ctx: &mut MiriInterpCx<'_, 'tcx>,
    args_ref: LLVMGenericValueArrayRef,
    name: *const ::libc::c_char,
    name_length: u64,
    tref: LLVMTypeRef,
) -> InterpResult<'tcx> {
    let name_slice: &[u8] = unsafe {
        std::slice::from_raw_parts(name as *const u8, usize::try_from(name_length).unwrap())
    };
    let name_c_str = CString::new(name_slice).unwrap();
    let name_rust_str = name_c_str.to_str().unwrap();
    let inst_opt = ctx.machine.extern_instance_map.get(&name_rust_str.to_string());
    if let Some(inst) = inst_opt {
        let inst_val = *inst;
        let (function_args, ret_val) = ctx.resolve_llvm_interface(tref, args_ref)?;
        miri_call_by_instance_result(ctx, inst_val, function_args, ret_val)
    } else {
        let (mut function_args, return_type_opt) =
            ctx.resolve_llvm_interface_unchecked(tref, args_ref);
        debug!("LLVM to Shim: {:?}, TID: {:?}", name_rust_str, ctx.get_active_thread());

        let group_id =
            ctx.eval_context_ref().machine.threads.get_thread_group(ctx.get_active_thread());
        if let Some(logger) = &mut ctx.eval_context_mut().machine.llvm_logger {
            logger.log_llvm_call(name_rust_str, group_id, None);
        }

        let mut op_ty_args = vec![];
        let mut places_to_deallocate = vec![];
        let mut callback_places = vec![];
        for lli_arg in function_args.drain(0..) {
            let eqv_rust_layout = ctx.get_equivalent_rust_layout_for_value(&lli_arg)?;
            let (op_ty, allocated_place) = ctx.generic_value_to_opty(lli_arg, eqv_rust_layout)?;
            op_ty_args.push(op_ty);
            if let Some(place) = allocated_place {
                places_to_deallocate.push(place);
            }
        }
        let num_args = op_ty_args.len();
        let args = op_ty_args.as_slice();

        let gv_to_return = match name_rust_str {
            //void *malloc(size_t size);
            "malloc" | "_Znwm" =>
                if num_args == 1 {
                    let size_as_scalar = ctx.opty_as_scalar(&op_ty_args[0])?;
                    let size_value = size_as_scalar.to_u64()?;
                    let allocation = ctx.malloc(size_value, false, MiriMemoryKind::LLVMHeap)?;

                    let as_miri_ptr = ctx.pointer_to_lli_wrapped_pointer(allocation);
                    debug!(
                        "[llvm_user_malloc] allocated {:?} bytes at {:?}",
                        size_value, as_miri_ptr.prov.alloc_id
                    );
                    unsafe { GenericValue::create_generic_value_of_miri_pointer(as_miri_ptr) }
                } else {
                    throw_shim_argument_mismatch!(name_rust_str, 1, num_args);
                },
            //void free(void *address);
            "free" | "_Znam" =>
                if num_args == 1 {
                    let address = ctx.opty_as_scalar(&op_ty_args[0])?;
                    let as_pointer = address.to_pointer(ctx)?;
                    ctx.free(as_pointer, MiriMemoryKind::LLVMHeap)?;
                    ctx.void_generic_value()
                } else {
                    throw_shim_argument_mismatch!(name_rust_str, 1, num_args);
                },

            //void *calloc(int num, int size);
            "calloc" =>
                if num_args == 2 {
                    let num_as_scalar = ctx.opty_as_scalar(&op_ty_args[0])?;
                    let size_as_scalar = ctx.opty_as_scalar(&op_ty_args[1])?;
                    let num_value = num_as_scalar.to_u64()?;
                    let size_value = size_as_scalar.to_u64()?;
                    let allocation =
                        ctx.malloc(num_value * size_value, true, MiriMemoryKind::LLVMHeap)?;
                    let as_miri_ptr = ctx.pointer_to_lli_wrapped_pointer(allocation);

                    debug!(
                        "[llvm_user_calloc] allocated {:?} bytes at {:?}",
                        size_value, as_miri_ptr.prov.alloc_id,
                    );

                    unsafe { GenericValue::create_generic_value_of_miri_pointer(as_miri_ptr) }
                } else {
                    throw_shim_argument_mismatch!(name_rust_str, 2, num_args);
                },

            //void *realloc(void *address, int newsize);
            "realloc" =>
                if num_args == 2 {
                    let address = ctx.opty_as_scalar(&op_ty_args[0])?;
                    let as_pointer = address.to_pointer(ctx)?;
                    let num_as_scalar = ctx.opty_as_scalar(&op_ty_args[1])?;
                    let num_value = num_as_scalar.to_u64()?;
                    let allocation =
                        ctx.realloc(as_pointer, num_value, MiriMemoryKind::LLVMHeap)?;
                    let as_miri_ptr = ctx.pointer_to_lli_wrapped_pointer(allocation);
                    unsafe { GenericValue::create_generic_value_of_miri_pointer(as_miri_ptr) }
                } else {
                    throw_shim_argument_mismatch!(name_rust_str, 2, num_args);
                },
            "memcpy" =>
                eval_memcpy(
                    ctx,
                    name_rust_str,
                    &args[0],
                    &args[1],
                    Some(&args[2]),
                    MemcpyMode::DisjointOrEqual,
                )?,
            "__memcpy_chk" =>
                eval_memcpy(
                    ctx,
                    name_rust_str,
                    &args[0],
                    &args[1],
                    Some(&args[2]),
                    MemcpyMode::DisjointOrEqual,
                )?,

            "strcpy" =>
                eval_memcpy(ctx, name_rust_str, &args[0], &args[1], None, MemcpyMode::Disjoint)?,
            "__strcpy_chk" =>
                eval_memcpy(ctx, name_rust_str, &args[0], &args[1], None, MemcpyMode::Disjoint)?,

            //void * memcpy ( void * destination, const void * source, size_t num );
            "memmove" =>
                eval_memcpy(
                    ctx,
                    name_rust_str,
                    &args[0],
                    &args[1],
                    Some(&args[2]),
                    MemcpyMode::Overlapping,
                )?,
            "__memmove_chk" =>
                eval_memcpy(
                    ctx,
                    name_rust_str,
                    &args[0],
                    &args[1],
                    Some(&args[2]),
                    MemcpyMode::Overlapping,
                )?,

            //  (void * dest, int c, size_t len, size_t destlen);s
            "__memset_chk" =>
                if num_args != 4 {
                    throw_shim_argument_mismatch!(name_rust_str, 4, num_args);
                } else {
                    let dest = ctx.opty_as_scalar(&op_ty_args[0])?;
                    let dest_as_pointer = dest.to_pointer(ctx)?;

                    let c_as_scalar = ctx.opty_as_scalar(&op_ty_args[1])?;
                    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                    let c_value = c_as_scalar.to_i32()? as u8;

                    let len_as_scalar = ctx.opty_as_scalar(&op_ty_args[2])?;
                    let len_value = len_as_scalar.to_u64()?;

                    ctx.write_bytes_ptr(
                        dest_as_pointer,
                        std::iter::repeat(c_value).take(usize::try_from(len_value).unwrap()),
                    )?;
                    let as_miri_ptr = ctx.pointer_to_lli_wrapped_pointer(dest_as_pointer);
                    unsafe { GenericValue::create_generic_value_of_miri_pointer(as_miri_ptr) }
                },
            "time" =>
                if num_args != 1 {
                    throw_shim_argument_mismatch!(name_rust_str, 1, num_args);
                } else {
                    match SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
                        Ok(n) => {
                            let num_secs = n.as_secs();
                            let layout = ctx.layout_of(Ty::new_uint(
                                *ctx.tcx,
                                ty::uint_ty(rustc_ast::UintTy::U64),
                            ))?;
                            let dest = ctx.opty_as_scalar(&op_ty_args[0])?;
                            let dest_as_pointer = dest.to_pointer(ctx)?;
                            let bytes = ctx.scalar_to_bytes(Scalar::from_u64(num_secs), layout);
                            if !ctx.ptr_is_null(dest_as_pointer)? {
                                ctx.write_bytes_ptr(dest_as_pointer, bytes.clone())?;
                            }
                            GenericValue::from_byte_slice(&bytes)
                        }
                        Err(_) => panic!("SystemTime before UNIX EPOCH."),
                    }
                },
            "strchr" =>
                if num_args != 2 {
                    throw_shim_argument_mismatch!(name_rust_str, 2, num_args);
                } else {
                    let src = ctx.opty_as_scalar(&op_ty_args[0])?;
                    let src_as_pointer = src.to_pointer(ctx)?;
                    let src_as_slice = ctx.pointer_to_slice(src_as_pointer)?;
                    let src_as_cstring = CStr::from_bytes_until_nul(src_as_slice);

                    let c_as_scalar = ctx.opty_as_scalar(&op_ty_args[1])?;
                    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                    let c_value = c_as_scalar.to_i32()? as u8;
                    if let Ok(cstr) = src_as_cstring {
                        let result = cstr.to_bytes().iter().position(|&x| x == c_value);
                        if let Some(pos) = result {
                            let offset_pointer =
                                src_as_pointer.offset(Size::from_bytes(pos), ctx)?;
                            let as_miri_ptr = ctx.pointer_to_lli_wrapped_pointer(offset_pointer);
                            unsafe {
                                GenericValue::create_generic_value_of_miri_pointer(as_miri_ptr)
                            }
                        } else {
                            ctx.void_generic_value()
                        }
                    } else {
                        throw_interop_format!(
                            "expected a null-terminated string, but couldn't locate a null-terminator."
                        )
                    }
                },
            "printf" => {
                if num_args == 1 {
                    let src = ctx.opty_as_scalar(&op_ty_args[0])?;
                    let src_as_pointer = src.to_pointer(ctx)?;

                    if let Some(crate::Provenance::Concrete { alloc_id, .. }) =
                        src_as_pointer.provenance
                    {
                        let (size, _, _) = ctx.get_alloc_info(alloc_id);
                        let slice = ctx.read_bytes_ptr_strip_provenance(src_as_pointer, size)?;
                        let as_cstring = CStr::from_bytes_with_nul(slice).unwrap();
                        println!("{}", as_cstring.to_str().unwrap());
                    }
                }
                ctx.void_generic_value()
            }
            _ => {
                let rplace = match &return_type_opt {
                    Some(return_type) => {
                        let ret_eqv_rust_layout = ctx.get_equivalent_rust_layout(*return_type)?;
                        PlaceTy::from(ctx.allocate(
                            ret_eqv_rust_layout,
                            MemoryKind::Machine(MiriMemoryKind::LLVMInterop),
                        )?)
                    }
                    None =>
                        PlaceTy::from(ctx.allocate(
                            ctx.layout_of(ctx.tcx.types.unit)?,
                            MemoryKind::Machine(MiriMemoryKind::LLVMInterop),
                        )?),
                };
                let args_slice = op_ty_args.as_slice();
                let name_symbol = Symbol::intern(name_rust_str);
                debug!("Executing...");
                let emulation_result = ctx.emulate_foreign_item_by_name(
                    name_symbol,
                    Abi::C { unwind: false },
                    args_slice,
                    &rplace,
                )?;
                if let EmulateByNameResult::NotSupported = emulation_result {
                    let os = &ctx.tcx.sess.target.os;
                    throw_unsup_format!("can't call foreign function `{name_symbol}` on OS `{os}`");
                } else {
                    debug!("Finished executing, preparing return value...");
                    let return_gv = perform_opty_conversion(ctx, &rplace, &return_type_opt)?;
                    callback_places.push(rplace);
                    return_gv
                }
            }
        };

        for place in places_to_deallocate {
            ctx.deallocate_ptr(
                place.assert_mem_place().ptr(),
                Some((place.layout.size, place.align)),
                MemoryKind::Machine(MiriMemoryKind::LLVMInterop),
            )?;
        }
        for place in callback_places {
            ctx.deallocate_ptr(
                place.assert_mem_place().ptr(),
                Some((place.layout.size, place.align)),
                MemoryKind::Machine(MiriMemoryKind::LLVMInterop),
            )?;
        }
        ctx.set_pending_return_value(
            ctx.get_active_thread(),
            GenericValueRef::new(unsafe { gv_to_return.into_raw() }),
        );
        debug!("Returning to LLVM...");
        Ok(())
    }
}

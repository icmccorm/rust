extern crate either;
extern crate rustc_abi;
use super::to_bytes::EvalContextExt as ToBytesEvalExt;
use crate::shims::llvm::helpers::EvalContextExt as LLVMEvalExt;
use crate::shims::llvm_ffi_support::{ResolvedLLVMType, ResolvedRustArgument};
use crate::{MiriInterpCx, Provenance};
use either::Either::{Left, Right};
use inkwell::{types::BasicTypeEnum, values::GenericValue};
use log::debug;
use rustc_abi::{Size, VariantIdx};
use rustc_apfloat::Float;
use rustc_const_eval::interpret::{alloc_range, InterpResult, OpTy, Scalar};
use rustc_middle::ty::GenericArgsRef;
use rustc_middle::ty::{self, AdtDef};
use std::iter::repeat;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}

pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn void_generic_value(&self) -> GenericValue<'static> {
        GenericValue::from_byte_slice(&[0])
    }
    fn op_to_generic_value<'lli>(
        &mut self,
        arg: ResolvedRustArgument<'tcx>,
        bte: ResolvedLLVMType<'lli>,
    ) -> InterpResult<'tcx, GenericValue<'lli>> {
        let this = self.eval_context_mut();
        convert_opty_to_generic_value(this, arg, bte)
    }
}

#[allow(clippy::arithmetic_side_effects)]
fn convert_opty_to_generic_value<'tcx, 'lli>(
    ctx: &mut MiriInterpCx<'_, 'tcx>,
    arg: ResolvedRustArgument<'tcx>,
    bte: ResolvedLLVMType<'lli>,
) -> InterpResult<'tcx, GenericValue<'lli>> {
    if let Some(bte) = bte {
        debug!("[Op to GV]: {:?} -> {:?}", arg.ty(), bte.print_to_string());
        match bte {
            BasicTypeEnum::ArrayType(at) => {
                if !arg.ty().is_adt() {
                    throw_rust_type_mismatch!(arg.layout(), at);
                }
                let llvm_field_types =
                    repeat(at.get_element_type()).take(at.len() as usize).collect();

                Ok(convert_opty_to_aggregate(ctx, arg.opty(), llvm_field_types)?)
            }
            BasicTypeEnum::FloatType(ft) =>
                match arg.abi() {
                    rustc_abi::Abi::Scalar(_) => {
                        let scalar = ctx.read_scalar(arg.opty())?;
                        if let Scalar::Int(si) = scalar {
                            match bte.get_llvm_type_kind() {
                                llvm_sys::LLVMTypeKind::LLVMFloatTypeKind => {
                                    if si.size() != Size::from_bytes(std::mem::size_of::<f32>()) {
                                        throw_rust_type_mismatch!(arg.layout(), ft);
                                    }
                                    let bits = scalar.to_f32()?.to_bits();
                                    let bytes =
                                        ctx.to_vec_endian(bits, arg.value_size().bytes_usize());
                                    let float = f32::from_ne_bytes(bytes.try_into().unwrap());
                                    debug!("[Op to GV]: Float value: {}", float);
                                    Ok(GenericValue::new_f32(float))
                                }
                                llvm_sys::LLVMTypeKind::LLVMDoubleTypeKind => {
                                    if si.size() != Size::from_bytes(std::mem::size_of::<f64>()) {
                                        throw_rust_type_mismatch!(arg.layout(), ft);
                                    }
                                    let double = scalar.to_f64()?.to_bits();
                                    let bytes =
                                        ctx.to_vec_endian(double, arg.value_size().bytes_usize());
                                    let double = f64::from_ne_bytes(bytes.try_into().unwrap());
                                    debug!("[Op to GV]: Double value: {}", double);
                                    Ok(GenericValue::new_f64(double))
                                }
                                _ => {
                                    let bits = si.assert_bits(arg.value_size());
                                    let bytes =
                                        ctx.to_vec_endian(bits, arg.value_size().bytes_usize());
                                    Ok(GenericValue::from_byte_slice(&bytes))
                                }
                            }
                        } else {
                            throw_rust_type_mismatch!(arg.layout(), ft);
                        }
                    }
                    _ => {
                        throw_rust_type_mismatch!(arg.layout(), ft);
                    }
                },
            BasicTypeEnum::IntType(it) =>
                match arg.abi() {
                    rustc_abi::Abi::Scalar(sc) => {
                        let scalar = if let rustc_abi::Scalar::Union { value } = sc {
                            let mp = arg.opty().assert_mem_place();
                            let ptr = mp.ptr();
                            let alloc =
                                ctx.get_ptr_alloc(ptr, value.size(ctx), value.align(ctx).abi)?;
                            if let Some(a) = alloc {
                                a.read_scalar(alloc_range(Size::ZERO, value.size(ctx)), true)?
                            } else {
                                bug!("unable to resolve allocation for union scalar value.");
                            }
                        } else {
                            ctx.read_scalar(arg.opty())?
                        };
                        let bits = match scalar {
                            Scalar::Int(si) =>
                                si.to_bits(arg.value_size())
                                    .unwrap_or_else(|s| si.to_bits(s).unwrap()),
                            Scalar::Ptr(p, _) => p.into_parts().1.bits().into(),
                        };
                        debug!("[Op to GV]: Int value: {}", u64::try_from(bits).unwrap());
                        let bytes = ctx.to_vec_endian(bits, arg.value_size().bytes_usize());
                        let dest_size = ctx.resolve_llvm_type_size(bte)?;
                        if (bytes.len() as u64) < dest_size
                            && arg.padded_size() != Size::from_bytes(dest_size)
                        {
                            throw_rust_type_mismatch!(arg.layout(), it);
                        }
                        Ok(GenericValue::from_byte_slice(&bytes))
                    }
                    rustc_abi::Abi::ScalarPair(_, _)
                    | rustc_abi::Abi::Aggregate { sized: true } => {
                        let bytes = ctx.op_to_bytes(arg.opty())?;
                        Ok(GenericValue::from_byte_slice(bytes.as_slice()))
                    }
                    _ => {
                        throw_rust_type_mismatch!(arg.layout(), it);
                    }
                },
            BasicTypeEnum::PointerType(pt) => {
                let pointer_opty = if let ty::Adt(adef, sr) = arg.ty().kind() {
                    if let Some(vidx) = is_enum_of_nonnullable_ptr(ctx, *adef, sr) {
                        debug!("[Op to GV]: Enum of nonnullable pointer: {:?}", vidx);
                        let downcast_op = ctx.project_downcast(arg.opty(), vidx)?;
                        let field_op = ctx.project_field(&downcast_op, 0)?;
                        ctx.read_pointer(&field_op)?
                    } else {
                        if let Left(mp) = arg.opty().as_mplace_or_imm() {
                            mp.ptr()
                        } else {
                            ctx.read_pointer(arg.opty())?
                        }
                    }
                } else {
                    if arg.value_size() != ctx.tcx.data_layout.pointer_size {
                        throw_rust_type_mismatch!(arg.layout(), pt);
                    }
                    match arg.abi() {
                        rustc_abi::Abi::Scalar(_) => ctx.read_pointer(arg.opty())?,
                        _ =>
                            if let Left(mp) = arg.opty().as_mplace_or_imm() {
                                mp.ptr()
                            } else {
                                throw_rust_type_mismatch!(arg.layout(), pt);
                            },
                    }
                };
                let wrapped_pointer = ctx.pointer_to_lli_wrapped_pointer(pointer_opty);
                debug!(
                    "[Op to GV]: Pointer value: (AID: {}, Addr: {})",
                    wrapped_pointer.prov.alloc_id, wrapped_pointer.addr
                );
                let as_gv =
                    unsafe { GenericValue::create_generic_value_of_miri_pointer(wrapped_pointer) };
                return Ok(as_gv);
            }
            BasicTypeEnum::StructType(st) => {
                if !arg.ty().is_adt() {
                    throw_rust_type_mismatch!(arg.layout(), st);
                }
                let llvm_field_types = st.get_field_types();
                Ok(convert_opty_to_aggregate(ctx, arg.opty(), llvm_field_types)?)
            }
            BasicTypeEnum::VectorType(vt) =>
                match arg.abi() {
                    rustc_abi::Abi::Vector { element, count } => {
                        let pointer = match arg.opty().as_mplace_or_imm() {
                            Left(mp) => mp.ptr(),
                            Right(imm) => {
                                if let rustc_const_eval::interpret::Immediate::Scalar(
                                    Scalar::Ptr(ptr, ..),
                                ) = *imm
                                {
                                    ptr.into()
                                } else {
                                    bug!(
                                        "expected a pointer to the start of a vector, but found {:?}",
                                        *imm
                                    );
                                }
                            }
                        };
                        let wrapped_pointer = ctx.pointer_to_lli_wrapped_pointer(pointer);
                        if u64::from(vt.get_size()) != count {
                            throw_rust_type_mismatch!(arg.layout(), vt);
                        }
                        let rust_vector_element_size = element.size(&ctx.tcx).bytes();
                        let llvm_vector_element_size =
                            ctx.resolve_llvm_type_size(vt.get_element_type())?;
                        if rust_vector_element_size != llvm_vector_element_size {
                            throw_rust_type_mismatch!(arg.layout(), vt);
                        }
                        debug!(
                            "[Op to GV]: Vector pointer value: (AID: {}, Addr: {})",
                            wrapped_pointer.prov.alloc_id, wrapped_pointer.addr
                        );
                        let as_gv = unsafe {
                            GenericValue::create_generic_value_of_miri_pointer(wrapped_pointer)
                        };
                        return Ok(as_gv);
                    }
                    _ => {
                        throw_rust_type_mismatch!(arg.layout(), vt);
                    }
                },
        }
    } else {
        if let rustc_abi::Abi::Scalar(sc) = arg.abi() {
            match sc.primitive() {
                rustc_abi::Primitive::Int(_, _) => {
                    let scalar = ctx.read_scalar(arg.opty())?.assert_int();
                    let bits = scalar
                        .to_bits(arg.value_size())
                        .unwrap_or_else(|s| scalar.to_bits(s).unwrap());
                    debug!("[Op to GV]: Int value: {}", u64::try_from(bits).unwrap());
                    let bytes = ctx.to_vec_endian(bits, arg.value_size().bytes_usize());
                    Ok(GenericValue::from_byte_slice(&bytes))
                }
                rustc_abi::Primitive::F32 => {
                    let scalar = ctx.read_scalar(arg.opty())?;
                    let bits = scalar.to_f32()?.to_bits();
                    let bytes = ctx.to_vec_endian(bits, arg.value_size().bytes_usize());
                    let float = f32::from_ne_bytes(bytes.try_into().unwrap());
                    debug!("[Op to GV]: Float value: {}", float);
                    Ok(GenericValue::new_f32(float))
                }
                rustc_abi::Primitive::F64 => {
                    let scalar = ctx.read_scalar(arg.opty())?;
                    let double = scalar.to_f64()?.to_bits();
                    let bytes = ctx.to_vec_endian(double, arg.value_size().bytes_usize());
                    let double = f64::from_ne_bytes(bytes.try_into().unwrap());
                    debug!("[Op to GV]: Double value: {}", double);
                    Ok(GenericValue::new_f64(double))
                }
                rustc_abi::Primitive::Pointer(_) => {
                    let ptr = ctx.read_pointer(arg.opty())?;
                    let wrapped_pointer = ctx.pointer_to_lli_wrapped_pointer(ptr);
                    debug!(
                        "[Op to GV]: Pointer value: (AID: {}, Addr: {})",
                        wrapped_pointer.prov.alloc_id, wrapped_pointer.addr
                    );
                    let as_gv = unsafe {
                        GenericValue::create_generic_value_of_miri_pointer(wrapped_pointer)
                    };
                    return Ok(as_gv);
                }
            }
        } else {
            throw_unsup_var_arg!(arg.layout());
        }
    }
}

fn is_enum_of_nonnullable_ptr<'tcx>(
    ctx: &MiriInterpCx<'_, 'tcx>,
    adt_def: AdtDef<'tcx>,
    substs: GenericArgsRef<'tcx>,
) -> Option<VariantIdx> {
    if adt_def.repr().inhibit_enum_layout_opt() {
        return None;
    }
    let [var_one, var_two] = &adt_def.variants().raw[..] else {
        return None;
    };
    let (([], [field]) | ([field], [])) = (&var_one.fields.raw[..], &var_two.fields.raw[..]) else {
        return None;
    };
    matches!(field.ty(*ctx.tcx, substs).kind(), ty::FnPtr(..) | ty::Ref(..));
    let vidx: u32 =
        if let ([], [_field]) = (&var_one.fields.raw[..], &var_two.fields.raw[..]) { 1 } else { 0 };

    Some(VariantIdx::from_u32(vidx))
}

fn convert_opty_to_aggregate<'lli, 'tcx>(
    ctx: &mut MiriInterpCx<'_, 'tcx>,
    arg: &OpTy<'tcx, Provenance>,
    llvm_fields: Vec<BasicTypeEnum<'lli>>,
) -> InterpResult<'tcx, GenericValue<'lli>> {
    let mut arg = arg.clone();
    debug!("[Op to GV] Aggregate Conversion, rust_type: {:?}", arg.layout.ty,);
    while arg.layout.fields.count() == 1 && llvm_fields.len() > 1 {
        // the compiler optimized a #repr(C) struct with a single
        // field to be the same as the field when exposed to LLVM bytecode.
        arg = ctx.project_field(&arg, 0)?;
    }
    if arg.layout.fields.count() != llvm_fields.len() {
        throw_rust_field_mismatch!(arg.layout, llvm_fields.len());
    }
    debug!("Aggregate: {:?} fields to {:?} fields", arg.layout.fields.count(), llvm_fields.len());
    let mut gen_ag = GenericValue::new_aggregate(llvm_fields.len() as u64);
    let rust_field_iterator = arg.layout.fields.index_by_increasing_offset();
    let rust_llvm_field_pairs = rust_field_iterator.zip(llvm_fields);

    for (rust_field_idx, llvm_field) in rust_llvm_field_pairs {
        let padded_size = ctx.resolve_padded_size(&arg, rust_field_idx);
        let as_op = ctx.project_field(&arg, rust_field_idx)?;
        debug!(
            "Field {}, size: {}, padded size: {}",
            rust_field_idx,
            as_op.layout.size.bytes(),
            padded_size.bytes()
        );
        let as_gv = convert_opty_to_generic_value(
            ctx,
            ResolvedRustArgument::Padded(as_op, padded_size),
            Some(llvm_field),
        )?;
        gen_ag.append_aggregate_value(as_gv);
    }
    Ok(gen_ag)
}

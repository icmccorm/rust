extern crate either;
extern crate rustc_abi;
use super::values::resolved_ptr::ResolvedPointer;
use crate::alloc_addresses::EvalContextExt as _;
use crate::concurrency::thread::EvalContextExt as _;
use crate::eval::ForeignAlignmentCheckMode;
use crate::helpers::EvalContextExt as HelperEvalExt;
use crate::rustc_const_eval::interpret::AllocMap;
use crate::rustc_middle::ty::layout::LayoutOf;
use crate::shims::llvm::logging::LLVMFlag;
use crate::throw_unsup_llvm_type;
use crate::throw_unsup_shim_llvm_type;
use crate::*;
use either::Either::Right;
use inkwell::miri::StackTrace;
use inkwell::types::{AnyTypeEnum, BasicType, BasicTypeEnum};
use inkwell::values::GenericValueArrayRef;
use inkwell::values::GenericValueRef;
use llvm_sys::execution_engine::LLVMGenericValueArrayRef;
use llvm_sys::miri::{MiriPointer, MiriProvenance};
use llvm_sys::prelude::LLVMTypeRef;
use rustc_abi::Endian;
use rustc_const_eval::interpret::{AllocId, CheckInAllocMsg, InterpErrorInfo, InterpResult};
use rustc_middle::mir::Mutability;
use rustc_middle::ty::layout::{HasTyCtxt, TyAndLayout};
use rustc_middle::ty::AdtDef;
use rustc_middle::ty::GenericArgsRef;
use rustc_middle::ty::{self, Ty};
use rustc_span::FileNameDisplayPreference;
use rustc_target::abi::Align;
use rustc_target::abi::Size;
use rustc_target::abi::VariantIdx;
use std::num::NonZeroU64;
use tracing::debug;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}

pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn get_equivalent_rust_layout_for_value(
        &self,
        generic_value_ref: &GenericValueRef<'_>,
    ) -> InterpResult<'tcx, TyAndLayout<'tcx>> {
        let this = self.eval_context_ref();
        let type_tag = generic_value_ref.assert_type_tag();
        if let BasicTypeEnum::PointerType(_) = type_tag {
            let wrapped_pointer = generic_value_ref.as_miri_pointer();
            let mp = this.lli_wrapped_pointer_to_maybe_pointer(wrapped_pointer);

            if let Some(crate::Provenance::Concrete { alloc_id, .. }) = mp.provenance {
                let alloc_entry = this.memory.alloc_map().get(alloc_id);
                let (size, _, _) = this.get_alloc_info(alloc_id);
                if let Some((kind, _)) = alloc_entry {
                    if let rustc_const_eval::interpret::MemoryKind::Machine(
                        crate::MiriMemoryKind::LLVMStack | crate::MiriMemoryKind::LLVMStatic,
                    ) = kind
                    {
                        let base_address = this.addr_from_alloc_id(alloc_id, *kind)?;

                        #[allow(clippy::arithmetic_side_effects)]
                        let offset = mp.addr() - Size::from_bytes(base_address);
                        if offset == Size::ZERO {
                            if let Some(pointing_to) =
                                self.get_equivalent_rust_int_from_size(size)?
                            {
                                if let Some(ref logger) = this.machine.llvm_logger {
                                    logger.log_flag(LLVMFlag::SizeBasedTypeInference);
                                }
                                return this.raw_pointer_to(pointing_to.ty);
                            }
                        }
                    }
                }
            }
        }
        let primitive_layout = this.get_equivalent_rust_primitive_layout(type_tag)?;
        if let Some(primitive_layout) = primitive_layout {
            return Ok(primitive_layout);
        } else {
            throw_unsup_shim_llvm_type!(type_tag)
        }
    }

    fn is_enum_of_nonnullable_ptr(
        &self,
        adt_def: AdtDef<'tcx>,
        substs: GenericArgsRef<'tcx>,
    ) -> Option<VariantIdx> {
        let ctx = self.eval_context_ref();
        if adt_def.repr().inhibit_enum_layout_opt() {
            return None;
        }
        let [var_one, var_two] = &adt_def.variants().raw[..] else {
            return None;
        };
        let (([], [field]) | ([field], [])) = (&var_one.fields.raw[..], &var_two.fields.raw[..])
        else {
            return None;
        };
        matches!(field.ty(*ctx.tcx, substs).kind(), ty::FnPtr(..) | ty::Ref(..));
        let vidx: u32 = if let ([], [_field]) = (&var_one.fields.raw[..], &var_two.fields.raw[..]) {
            1
        } else {
            0
        };
        Some(VariantIdx::from_u32(vidx))
    }

    fn get_equivalent_rust_primitive_layout(
        &self,
        ty: BasicTypeEnum<'_>,
    ) -> InterpResult<'tcx, Option<TyAndLayout<'tcx>>> {
        let ctx = self.eval_context_ref();
        let resolved_ty = match ty {
            BasicTypeEnum::FloatType(_) =>
                match ty.get_llvm_type_kind() {
                    llvm_sys::LLVMTypeKind::LLVMDoubleTypeKind =>
                        Some(ctx.layout_of(ctx.tcx.types.f64)?),
                    llvm_sys::LLVMTypeKind::LLVMFloatTypeKind =>
                        Some(ctx.layout_of(ctx.tcx.types.f32)?),
                    _ => None,
                },
            BasicTypeEnum::IntType(it) =>
                ctx.get_equivalent_rust_int_from_size(Size::from_bits(it.get_bit_width()))?,
            BasicTypeEnum::PointerType(_) => Some(ctx.raw_pointer_to(ctx.tcx.types.u8)?),
            _ => None,
        };
        Ok(resolved_ty)
    }

    fn get_equivalent_rust_int_from_size(
        &self,
        integer_size: Size,
    ) -> InterpResult<'tcx, Option<TyAndLayout<'tcx>>> {
        let this = self.eval_context_ref();
        let result = match integer_size.bytes() {
            1 => Some(this.layout_of(this.tcx.types.u8)?),
            2 => Some(this.layout_of(this.tcx.types.u16)?),
            4 => Some(this.layout_of(this.tcx.types.u32)?),
            8 => Some(this.layout_of(this.tcx.types.u64)?),
            16 => Some(this.layout_of(this.tcx.types.u128)?),
            _ => None,
        };
        Ok(result)
    }

    fn resolve_llvm_type_size<'lli>(&self, bte: BasicTypeEnum<'lli>) -> InterpResult<'tcx, u64> {
        let this = self.eval_context_ref();
        let possible_llvm_size = bte.size_of().and_then(|ce| ce.get_zero_extended_constant());
        if let Some(size) = possible_llvm_size {
            return Ok(size);
        } else {
            let size = match bte {
                #[allow(clippy::arithmetic_side_effects)]
                BasicTypeEnum::ArrayType(at) =>
                    u64::from(at.len()) * self.resolve_llvm_type_size(at.get_element_type())?,

                BasicTypeEnum::FloatType(_) =>
                    match bte.get_llvm_type_kind() {
                        llvm_sys::LLVMTypeKind::LLVMDoubleTypeKind =>
                            this.layout_of(this.tcx.types.f64)?.size.bytes(),
                        llvm_sys::LLVMTypeKind::LLVMFloatTypeKind =>
                            this.layout_of(this.tcx.types.f32)?.size.bytes(),
                        _ => throw_unsup_llvm_type!(bte),
                    },

                BasicTypeEnum::IntType(it) => u64::from(it.get_bit_width() / 8),

                BasicTypeEnum::PointerType(_) =>
                    self.eval_context_ref().tcx().data_layout.pointer_size.bytes(),

                BasicTypeEnum::StructType(st) =>
                    st.get_field_types()
                        .iter()
                        .map(|ft| self.resolve_llvm_type_size(*ft))
                        .collect::<InterpResult<'_, Vec<_>>>()?
                        .iter()
                        .sum(),

                #[allow(clippy::arithmetic_side_effects)]
                BasicTypeEnum::VectorType(vt) =>
                    u64::from(vt.get_size()) * self.resolve_llvm_type_size(vt.get_element_type())?,
            };
            debug!("Resolved LLVM type size: {} bytes", size);
            Ok(size)
        }
    }

    fn resolve_llvm_interface(
        &self,
        fn_ty: LLVMTypeRef,
        args_ref: LLVMGenericValueArrayRef,
    ) -> InterpResult<'tcx, (Vec<GenericValueRef<'static>>, Option<BasicTypeEnum<'static>>)> {
        let fn_ty = unsafe { AnyTypeEnum::new(fn_ty).into_function_type() };
        let args = unsafe { GenericValueArrayRef::new(args_ref) };
        let ret_ty = fn_ty.get_return_type();
        let num_arguments_provided = args.len();
        let num_arguments_expected = u64::try_from(fn_ty.get_param_types().len()).unwrap();
        if num_arguments_provided != num_arguments_expected {
            throw_interop_format!(
                "expected {} arguments, but got {}.",
                num_arguments_expected,
                num_arguments_provided
            )
        }
        let args = (0..num_arguments_provided)
            .map(|idx| args.get_element_at(idx as u64).unwrap())
            .collect();
        Ok((args, ret_ty))
    }

    fn resolve_llvm_interface_unchecked(
        &self,
        fn_ty: LLVMTypeRef,
        args_ref: LLVMGenericValueArrayRef,
    ) -> (Vec<GenericValueRef<'static>>, Option<BasicTypeEnum<'static>>) {
        let fn_ty = unsafe { AnyTypeEnum::new(fn_ty).into_function_type() };
        let args = unsafe { inkwell::values::GenericValueArrayRef::new(args_ref) };
        let ret_ty = fn_ty.get_return_type();
        let args = (0..args.len()).map(|idx| args.get_element_at(idx as u64).unwrap()).collect();
        (args, ret_ty)
    }

    fn lli_wrapped_pointer_to_maybe_pointer(&self, mp: MiriPointer) -> crate::Pointer {
        if mp.addr == 0 {
            Pointer::null()
        } else {
            let alloc_id = mp.prov.alloc_id;
            let tag = mp.prov.tag;
            let pointer = Size::from_bytes(mp.addr);
            if alloc_id > 0 {
                let alloc_id = AllocId(NonZeroU64::new(alloc_id).unwrap());
                let prov = crate::Provenance::Concrete { alloc_id, tag: BorTag::new(tag).unwrap() };
                Pointer::new(Some(prov), pointer)
            } else {
                Pointer::new(Some(crate::Provenance::Wildcard), pointer)
            }
        }
    }

    fn pointer_to_lli_wrapped_pointer(&self, ptr: Pointer) -> MiriPointer {
        let (prov, _) = ptr.into_parts();
        let (alloc_id, tag) = if let Some(crate::Provenance::Concrete { alloc_id, tag }) = prov {
            (alloc_id.0.get(), tag.get())
        } else {
            (0, 0)
        };
        let addr = ptr.addr().bytes();
        MiriPointer { addr, prov: MiriProvenance { alloc_id, tag } }
    }

    fn get_foreign_error(&self) -> Option<InterpErrorInfo<'tcx>> {
        let this = self.eval_context_ref();
        let err_opt = this.machine.foreign_error.take();
        if err_opt.is_some() {
            eprintln!("\n\n---- Foreign Error Trace ----");
            if let Some(trace) = this.machine.foreign_error_trace.take() {
                eprintln!("{}", trace);
            }
            if let Some(call_location) = this.machine.foreign_error_rust_call_location.take() {
                let span_str = this
                    .tcx()
                    .sess
                    .source_map()
                    .span_to_string(call_location, FileNameDisplayPreference::Local);
                eprintln!("{}", span_str);
            }
            eprintln!("-----------------------------\n");
        }
        err_opt
    }

    fn set_pending_return_value(&mut self, id: ThreadId, val_ref: GenericValueRef<'static>) {
        let this = self.eval_context_mut();
        if id.to_u32() == 0 {
            this.machine.pending_return_values.insert(id, val_ref);
        } else {
            this.machine.pending_return_values.try_insert(id, val_ref).unwrap();
        }
    }

    fn get_pending_return_value(
        &mut self,
        id: ThreadId,
    ) -> InterpResult<'tcx, Option<GenericValueRef<'static>>> {
        let this = self.eval_context_mut();
        let pending_return = this.machine.pending_return_values.remove(&id);
        Ok(pending_return)
    }

    fn update_last_rust_call_location(&self) {
        let this = self.eval_context_ref();
        this.machine.foreign_error_rust_call_location.set(Some(this.cur_span()));
    }

    fn set_foreign_stack_trace(&self, trace: StackTrace) {
        let this = self.eval_context_ref();
        this.machine.foreign_error_trace.replace(Some(trace));
    }

    fn set_foreign_error(&self, info: InterpErrorInfo<'tcx>) {
        let this = self.eval_context_ref();
        this.machine.foreign_error.replace(Some(info));
    }

    fn is_pointer_convertible(&self, layout: &TyAndLayout<'tcx>) -> bool {
        let this = self.eval_context_ref();
        match layout.ty.kind() {
            ty::FnPtr(_) | ty::RawPtr(_, _) | ty::Ref(_, _, _) => return true,
            ty::Adt(adt_def, sr) =>
                if this.is_enum_of_nonnullable_ptr(*adt_def, sr).is_some() {
                    return true;
                },
            _ => {}
        }
        if layout.is_transparent::<MiriInterpCx<'tcx>>() {
            if let Some((_, field)) = layout.non_1zst_field(this) {
                return this.is_pointer_convertible(&field);
            }
        }
        false
    }

    #[allow(dead_code)]
    fn is_pointer_aligned(&self, ptr: Pointer, align: Align) -> bool {
        let this = self.eval_context_ref();
        match this.ptr_try_get_alloc_id(ptr) {
            Err(addr) => addr % align.bytes() == 0,
            Ok((alloc_id, offset, _)) =>
                if this.machine.check_alignment == AlignmentCheck::Int {
                    ptr.addr().bytes() % align.bytes() == 0
                } else {
                    let (_, alloc_align, _) = this.get_alloc_info(alloc_id);
                    alloc_align.bytes() >= align.bytes() && offset.bytes() % align.bytes() != 0
                },
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn lli_wrapped_pointer_to_resolved_pointer(
        &mut self,
        mp: MiriPointer,
    ) -> InterpResult<'tcx, ResolvedPointer> {
        let this = self.eval_context_mut();
        if mp.addr > 0 {
            let (provenance, alloc_id) = if mp.prov.alloc_id > 0 {
                let alloc_id = AllocId(NonZeroU64::new(mp.prov.alloc_id).unwrap());
                let prov = crate::Provenance::Concrete {
                    alloc_id,
                    tag: BorTag::new(mp.prov.tag).unwrap(),
                };
                (prov, alloc_id)
            } else {
                let resolved_alloc_id = this.alloc_id_from_addr(mp.addr);
                if let Some(ref logger) = &this.machine.llvm_logger {
                    logger.log_flag(LLVMFlag::LLVMOnResolve)
                }
                if let Some(alloc_id) = resolved_alloc_id {
                    (crate::Provenance::Wildcard, alloc_id)
                } else {
                    throw_ub!(DanglingIntPointer(mp.addr, CheckInAllocMsg::MemoryAccessTest))
                }
            };
            let (_, align, _) = this.get_alloc_info(alloc_id);
            let pointer = if this.should_check_alignment_in_llvm(Some(alloc_id)) {
                ResolvedPointer {
                    ptr: Pointer::new(Some(provenance), Size::from_bytes(mp.addr)),
                    alloc_id: Some(alloc_id),
                    align,
                    offset: Size::ZERO,
                }
            } else {
                let (kind, _) = this.memory.alloc_map().get(alloc_id).unwrap();
                let base_address = this.addr_from_alloc_id(alloc_id, *kind)?;
                let alignment_offset_multiple = (mp.addr - base_address) / align.bytes();
                let aligned_offset = alignment_offset_multiple * align.bytes();
                let offset = Size::from_bytes((mp.addr - base_address) - aligned_offset);
                let aligned_addr = Size::from_bytes(base_address + aligned_offset);
                ResolvedPointer {
                    ptr: Pointer::new(Some(provenance), aligned_addr),
                    alloc_id: Some(alloc_id),
                    align,
                    offset,
                }
            };
            Ok(pointer)
        } else {
            Ok(ResolvedPointer::null())
        }
    }

    fn opty_as_scalar(&self, opty: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        if let Right(imm) = opty.as_mplace_or_imm() {
            Ok(imm.to_scalar())
        } else {
            bug!("expected scalar, but got {:?}", opty.layout.ty)
        }
    }

    fn raw_pointer_to(&self, ty: Ty<'tcx>) -> InterpResult<'tcx, TyAndLayout<'tcx>> {
        let this = self.eval_context_ref();
        this.layout_of(this.tcx.mk_ty_from_kind(ty::RawPtr(ty, Mutability::Mut)))
    }
    #[allow(clippy::arithmetic_side_effects)]
    fn to_vec_endian(&self, bytes: u128, length: usize) -> Vec<u8> {
        let bytes = bytes.to_ne_bytes();
        match self.eval_context_ref().tcx.sess.target.endian {
            Endian::Little => bytes[..length].to_vec(),
            Endian::Big => bytes[bytes.len() - length..].to_vec(),
        }
    }

    fn can_dereference_into_singular_field(&self, layout: &TyAndLayout<'tcx>) -> bool {
        matches!(layout.ty.kind(), ty::Adt(_, _)) && layout.fields.count() == 1
    }

    fn dereference_into_singular_field(
        &mut self,
        arg: OpTy<'tcx>,
    ) -> InterpResult<'tcx, OpTy<'tcx>> {
        let this = self.eval_context_mut();
        let mut curr_arg = arg;
        while this.can_dereference_into_singular_field(&curr_arg.layout) {
            curr_arg = this.project_field(&curr_arg, 0)?;
        }
        Ok(curr_arg)
    }

    fn is_fieldless(&self, layout: &TyAndLayout<'tcx>) -> bool {
        matches!(
            &layout.fields,
            rustc_abi::FieldsShape::Union(_) | rustc_abi::FieldsShape::Primitive
        )
    }

    fn resolve_padded_size(&self, layout: &TyAndLayout<'tcx>, rust_field_idx: usize) -> Size {
        if layout.fields.count() <= 1 {
            layout.size
        } else {
            let curr_offset = layout.fields.offset(rust_field_idx);
            #[allow(clippy::arithmetic_side_effects)]
            if rust_field_idx + 1 == layout.fields.count() {
                layout.size - curr_offset
            } else {
                layout.fields.offset(rust_field_idx + 1) - curr_offset
            }
        }
    }

    fn is_llvm_managed_allocation(&self, alloc_id: AllocId) -> Option<bool> {
        let this = self.eval_context_ref();
        if let Some((kind, _)) = this.memory.alloc_map().get(alloc_id) {
            Some(matches!(
                kind,
                rustc_const_eval::interpret::MemoryKind::Machine(
                    crate::MiriMemoryKind::LLVMStack | crate::MiriMemoryKind::LLVMStatic
                )
            ))
        } else {
            None
        }
    }

    fn is_foreign_allocation(&self, alloc_id: AllocId) -> Option<bool> {
        self.is_llvm_managed_allocation(alloc_id).or(self.is_foreign_heap_allocation(alloc_id))
    }

    fn is_foreign_heap_allocation(&self, alloc_id: AllocId) -> Option<bool> {
        let this = self.eval_context_ref();
        if let Some((kind, _)) = this.memory.alloc_map().get(alloc_id) {
            Some(matches!(
                kind,
                rustc_const_eval::interpret::MemoryKind::Machine(crate::MiriMemoryKind::C)
            ))
        } else {
            None
        }
    }

    fn should_check_alignment_in_llvm(&self, alloc_id: Option<AllocId>) -> bool {
        let this = self.eval_context_ref();
        match this.machine.lli_config.alignment_check_mode {
            ForeignAlignmentCheckMode::Skip => false,
            ForeignAlignmentCheckMode::Check => true,
            ForeignAlignmentCheckMode::CheckRustOnly =>
                alloc_id.map(|id| !this.is_foreign_allocation(id).unwrap_or(false)).unwrap_or(false),
        }
    }

    fn in_llvm(&self) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_ref();
        Ok(this.active_thread_ref().is_llvm_thread())
    }

    fn strcmp(
        &mut self,
        left: &OpTy<'tcx>,
        right: &OpTy<'tcx>,
        n: Option<&OpTy<'tcx>>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();
        let left = this.read_pointer(left)?;
        let right = this.read_pointer(right)?;

        // C requires that this must always be a valid pointer (C18 §7.1.4).
        this.ptr_get_alloc_id(left)?;
        this.ptr_get_alloc_id(right)?;
        let mut left_bytes = this.read_c_str(left)?;
        let mut right_bytes = this.read_c_str(right)?;
        if let Some(n) = n {
            let n = this.read_target_usize(n)?.try_into().unwrap();
            if n <= left_bytes.len() {
                left_bytes = &left_bytes[..n];
            }
            if n <= right_bytes.len() {
                right_bytes = &right_bytes[..n];
            }
        }
        let result = {
            if left_bytes.len() > right_bytes.len() {
                return Ok(1);
            }
            if left_bytes.len() < right_bytes.len() {
                return Ok(-1);
            }
            use std::cmp::Ordering::*;
            match left_bytes.cmp(right_bytes) {
                Less => -1i32,
                Equal => 0,
                Greater => 1,
            }
        };
        Ok(result)
    }
}

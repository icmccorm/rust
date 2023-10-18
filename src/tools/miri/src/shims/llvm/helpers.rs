extern crate either;
extern crate rustc_abi;
use super::values::generic_value::GenericValueTy;
use super::values::resolved_ptr::ResolvedPointer;
use crate::rustc_middle::ty::layout::LayoutOf;
use crate::shims::llvm_ffi_support::EvalContextExt as LLVMEvalContextExt;
use crate::throw_unsup_llvm_type;
use crate::{intptrcast, BorTag, Provenance, ThreadId};
use either::Either::Right;
use inkwell::miri::StackTrace;
use inkwell::types::{AnyTypeEnum, BasicType, BasicTypeEnum};
use inkwell::values::GenericValueRef;
use llvm_sys::execution_engine::LLVMGenericValueArrayRef;
use llvm_sys::miri::{MiriPointer, MiriProvenance};
use llvm_sys::prelude::LLVMTypeRef;
use log::debug;
use rustc_abi::Endian;
use rustc_const_eval::interpret::{
    AllocId, CheckInAllocMsg, InterpErrorInfo, InterpResult, OpTy, Pointer, Scalar,
};
use rustc_middle::mir::Mutability;
use rustc_middle::ty::layout::{HasTyCtxt, TyAndLayout};
use rustc_middle::ty::{self, Ty, TypeAndMut};
use rustc_span::FileNameDisplayPreference;
use rustc_target::abi::Size;
use std::num::NonZeroU64;

#[macro_export]
macro_rules! throw_interop_format {
    ($($tt:tt)*) => { throw_machine_stop!($crate::TerminationInfo::InteroperationError { msg: format!($($tt)*) }) };
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}

pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn get_equivalent_rust_layout_for_value(
        &self,
        gvty: &GenericValueTy,
    ) -> InterpResult<'tcx, TyAndLayout<'tcx>> {
        let this = self.eval_context_ref();
        this.get_equivalent_rust_layout(gvty.ty())
    }

    fn get_equivalent_rust_layout(
        &self,
        ty: BasicTypeEnum<'_>,
    ) -> InterpResult<'tcx, TyAndLayout<'tcx>> {
        let ctx = self.eval_context_ref();
        match ty {
            BasicTypeEnum::FloatType(ft) =>
                match ty.get_llvm_type_kind() {
                    llvm_sys::LLVMTypeKind::LLVMDoubleTypeKind => ctx.layout_of(ctx.tcx.types.f64),
                    llvm_sys::LLVMTypeKind::LLVMFloatTypeKind => ctx.layout_of(ctx.tcx.types.f32),
                    _ => throw_unsup_llvm_type!(ft),
                },
            BasicTypeEnum::IntType(it) =>
                return match it.get_bit_width() {
                    8 => ctx.layout_of(ctx.tcx.types.u8),
                    16 => ctx.layout_of(ctx.tcx.types.u16),
                    32 => ctx.layout_of(ctx.tcx.types.u32),
                    64 => ctx.layout_of(ctx.tcx.types.u64),
                    128 => ctx.layout_of(ctx.tcx.types.u128),
                    _ => throw_unsup_llvm_type!(it),
                },
            BasicTypeEnum::PointerType(_) => ctx.raw_pointer_to(ctx.tcx.types.u8),
            _ => throw_unsup_llvm_type!(ty),
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn resolve_llvm_type_size<'lli>(&self, bte: BasicTypeEnum<'lli>) -> InterpResult<'tcx, u64> {
        let this = self.eval_context_ref();
        let possible_llvm_size = bte.size_of().and_then(|ce| ce.get_zero_extended_constant());
        if let Some(size) = possible_llvm_size {
            return Ok(size);
        } else {
            let size = match bte {
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
    ) -> InterpResult<'tcx, (Vec<GenericValueTy>, Option<BasicTypeEnum<'static>>)> {
        let fn_ty = unsafe { AnyTypeEnum::new(fn_ty).into_function_type() };
        let args = inkwell::values::GenericValueArrayRef::new(args_ref);
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

        let parameter_types = fn_ty.get_param_types();
        let args = parameter_types
            .iter()
            .enumerate()
            .map(|(idx, t)| GenericValueTy::new(*t, args.get_element_at(idx as u64).unwrap()))
            .collect();
        Ok((args, ret_ty))
    }

    fn resolve_llvm_interface_unchecked(
        &self,
        fn_ty: LLVMTypeRef,
        args_ref: LLVMGenericValueArrayRef,
    ) -> (Vec<GenericValueTy>, Option<BasicTypeEnum<'static>>) {
        let fn_ty = unsafe { AnyTypeEnum::new(fn_ty).into_function_type() };
        let args = inkwell::values::GenericValueArrayRef::new(args_ref);
        let ret_ty = fn_ty.get_return_type();
        let parameter_types = fn_ty.get_param_types();
        let args = parameter_types
            .iter()
            .enumerate()
            .map(|(idx, t)| GenericValueTy::new(*t, args.get_element_at(idx as u64).unwrap()))
            .collect();
        (args, ret_ty)
    }

    fn lli_wrapped_pointer_to_maybe_pointer(
        &self,
        mp: MiriPointer,
    ) -> Pointer<std::option::Option<crate::Provenance>> {
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

    fn pointer_to_lli_wrapped_pointer(
        &self,
        ptr: Pointer<Option<crate::Provenance>>,
    ) -> MiriPointer {
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

    fn set_pending_return_value(&mut self, id: ThreadId, val_ref: GenericValueRef) {
        let this = self.eval_context_mut();
        this.machine.pending_return_values.try_insert(id, val_ref).unwrap();
    }

    fn get_pending_return_value(
        &mut self,
        id: ThreadId,
    ) -> InterpResult<'tcx, Option<GenericValueRef>> {
        let this = self.eval_context_mut();
        if let Some(val_ref) = this.machine.pending_return_values.remove(&id) {
            Ok(Some(val_ref))
        } else {
            this.get_thread_exit_value(id)
        }
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

    #[allow(clippy::arithmetic_side_effects)]
    fn lli_wrapped_pointer_to_resolved_pointer(
        &self,
        mp: MiriPointer,
    ) -> InterpResult<'tcx, ResolvedPointer> {
        let this = self.eval_context_ref();
        if mp.addr > 0 {
            let (provenance, alloc_id) = if mp.prov.alloc_id > 0 {
                let alloc_id = AllocId(NonZeroU64::new(mp.prov.alloc_id).unwrap());
                let prov = crate::Provenance::Concrete {
                    alloc_id,
                    tag: BorTag::new(mp.prov.tag).unwrap(),
                };
                (prov, alloc_id)
            } else {
                let resolved_alloc_id =
                    intptrcast::GlobalStateInner::alloc_id_from_addr(this, mp.addr);
                if let Some(alloc_id) = resolved_alloc_id {
                    (crate::Provenance::Wildcard, alloc_id)
                } else {
                    throw_ub!(DanglingIntPointer(mp.addr, CheckInAllocMsg::MemoryAccessTest))
                }
            };
            let (_, align, _) = this.get_alloc_info(alloc_id);
            let base_address = intptrcast::GlobalStateInner::alloc_base_addr(this, alloc_id)?;
            let alignment_offset_multiple = (mp.addr - base_address) / align.bytes();
            let aligned_offset = alignment_offset_multiple * align.bytes();
            let offset = Size::from_bytes((mp.addr - base_address) - aligned_offset);
            let aligned_addr = Size::from_bytes(base_address + aligned_offset);
            Ok(ResolvedPointer {
                ptr: Pointer::new(Some(provenance), aligned_addr),
                alloc_id: Some(alloc_id),
                align,
                offset,
            })
        } else {
            Ok(ResolvedPointer::null())
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn pointer_to_slice<'l>(
        &'l self,
        ptr: Pointer<Option<crate::Provenance>>,
    ) -> InterpResult<'tcx, &'l [u8]>
    where
        'mir: 'l,
        'tcx: 'l,
    {
        let ctx = self.eval_context_ref();
        let alloc_id = if let Some(crate::Provenance::Concrete { alloc_id, .. }) = ptr.provenance {
            alloc_id
        } else {
            throw_interop_format!(
                "unable to resolve slice: invalid pointer to address {:?}",
                ptr.addr()
            );
        };
        let (alloc_size, _, _) = ctx.get_alloc_info(alloc_id);
        let base_address = intptrcast::GlobalStateInner::alloc_base_addr(ctx, alloc_id)?;
        let size_offset = ptr.addr() - Size::from_bytes(base_address);
        let size_remaining = alloc_size - size_offset;
        let slice = ctx.read_bytes_ptr_strip_provenance(ptr, size_remaining)?;
        Ok(slice)
    }
    fn opty_as_scalar(
        &self,
        opty: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        if let Right(imm) = opty.as_mplace_or_imm() {
            Ok(imm.to_scalar())
        } else {
            bug!("expected scalar, but got {:?}", opty.layout.ty)
        }
    }
    fn raw_pointer_to(&self, ty: Ty<'tcx>) -> InterpResult<'tcx, TyAndLayout<'tcx>> {
        let this = self.eval_context_ref();
        this.layout_of(
            this.tcx.mk_ty_from_kind(ty::RawPtr(TypeAndMut { ty, mutbl: Mutability::Mut })),
        )
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
        match &layout.fields {
            rustc_abi::FieldsShape::Array { stride: _, count: 1 } => true,
            rustc_abi::FieldsShape::Arbitrary { offsets, .. } => offsets.len() == 1,
            _ => false,
        }
    }

    fn is_fieldless(&self, layout: &TyAndLayout<'tcx>) -> bool {
        matches!(
            &layout.fields,
            rustc_abi::FieldsShape::Union(_) | rustc_abi::FieldsShape::Primitive
        )
    }

    fn resolve_scalar_pair(
        &mut self,
        arg: &OpTy<'tcx, crate::Provenance>,
        llvm_type: &BasicTypeEnum<'_>,
    ) -> InterpResult<'tcx, Option<(OpTy<'tcx, crate::Provenance>, OpTy<'tcx, crate::Provenance>)>>
    {
        let this = self.eval_context_mut();
        if !this.is_fieldless(&arg.layout) {
            let mut curr_arg = arg.clone();
            let single_field_dereferenced =
                this.can_dereference_into_singular_field(&curr_arg.layout);
            while this.can_dereference_into_singular_field(&curr_arg.layout) {
                curr_arg = this.project_field(&curr_arg, 0)?;
            }
            if matches!(curr_arg.layout.abi, rustc_target::abi::Abi::ScalarPair(_, _)) {
                let field_values = arg
                    .layout
                    .fields
                    .index_by_increasing_offset()
                    .map(|idx| this.project_field(arg, idx))
                    .collect::<InterpResult<'tcx, Vec<OpTy<'_, Provenance>>>>()?;
                let llvm_size = this.resolve_llvm_type_size(*llvm_type)?;
                for rust_field in field_values.iter() {
                    let rust_field_size = rust_field.layout.size.bytes();
                    if rust_field_size != llvm_size {
                        return Ok(None);
                    }
                }
                if let Some(logger) = &mut this.machine.llvm_logger {
                    logger.log_llvm_conversion(
                        curr_arg.layout,
                        single_field_dereferenced,
                        llvm_type,
                    )
                }
                return Ok(Some((
                    field_values.first().unwrap().clone(),
                    field_values.last().unwrap().clone(),
                )));
            }
        }
        Ok(None)
    }

    fn maybe_alloc_id(&self, mp: Pointer<Option<crate::Provenance>>) -> Option<AllocId> {
        let this = self.eval_context_ref();
        match mp.provenance {
            Some(crate::Provenance::Concrete { alloc_id, .. }) => Some(alloc_id),
            Some(crate::Provenance::Wildcard) =>
                intptrcast::GlobalStateInner::alloc_id_from_addr(this, mp.addr().bytes()),
            None => None,
        }
    }
}

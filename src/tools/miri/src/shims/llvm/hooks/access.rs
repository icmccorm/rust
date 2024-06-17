use crate::shims::llvm::helpers::EvalContextExt;
use crate::MiriInterpCx;
use inkwell::types::BasicTypeEnum;
use llvm_sys::LLVMTypeKind;
use rustc_const_eval::interpret::InterpResult;
use rustc_target::abi::{Align, Size};
use crate::*;

pub trait Source<T> {
    fn read_f32<'tcx>(&self, ctx: &MiriInterpCx<'tcx>, align: Align)
    -> InterpResult<'tcx, f32>;
    fn read_f64<'tcx>(&self, ctx: &MiriInterpCx<'tcx>, align: Align)
    -> InterpResult<'tcx, f64>;
    fn read_unsigned<'tcx>(
        &self,
        ctx: &mut MiriInterpCx<'tcx>,
        size: Size,
        align: Align,
    ) -> InterpResult<'tcx, u128>;
    fn read_pointer<'tcx>(
        &self,
        ctx: &MiriInterpCx<'tcx>,
        align: Align,
    ) -> InterpResult<'tcx, Pointer>;
    fn check_aggregate_size<'tcx>(
        &self,
        ctx: &MiriInterpCx<'tcx>,
        aggregate_size: u32,
    ) -> InterpResult<'tcx>;
    fn resolve_field<'tcx>(
        &self,
        ctx: &MiriInterpCx<'tcx>,
        size: Size,
        index: u32,
    ) -> InterpResult<'tcx, T>;
}
pub trait Destination<T> {
    fn write_f32<'tcx>(
        &mut self,
        ctx: &mut MiriInterpCx<'tcx>,
        value: f32,
        align: Align,
    ) -> InterpResult<'tcx>;
    fn write_f64<'tcx>(
        &mut self,
        ctx: &mut MiriInterpCx<'tcx>,
        value: f64,
        align: Align,
    ) -> InterpResult<'tcx>;
    fn write_unsigned<'tcx>(
        &mut self,
        ctx: &mut MiriInterpCx<'tcx>,
        value: u128,
        size: Size,
        align: Align,
    ) -> InterpResult<'tcx>;
    fn write_pointer<'tcx>(
        &mut self,
        ctx: &mut MiriInterpCx<'tcx>,
        pointer: Pointer,
        align: Align,
    ) -> InterpResult<'tcx>;
    fn resolve_field<'tcx>(
        &mut self,
        ctx: &mut MiriInterpCx<'tcx>,
        size: Size,
        index: u32,
    ) -> InterpResult<'tcx, T>;
    fn ensure_aggregate_size<'tcx>(
        &self,
        ctx: &MiriInterpCx<'tcx>,
        aggregate_size: u32,
    ) -> InterpResult<'tcx>;
}

#[allow(clippy::arithmetic_side_effects)]
pub fn memory_access_core<'tcx, S, D>(
    ctx: &mut MiriInterpCx<'tcx>,
    source: &dyn Source<S>,
    destination: &mut dyn Destination<D>,
    value_type: BasicTypeEnum<'_>,
    value_size: Size,
    value_align: Align,
) -> InterpResult<'tcx, ()>
where
    S: Source<S>,
    D: Destination<D>,
{
    match value_type {
        BasicTypeEnum::FloatType(ft) =>
            match value_type.get_llvm_type_kind() {
                LLVMTypeKind::LLVMFloatTypeKind =>
                    destination.write_f32(ctx, source.read_f32(&*ctx, value_align)?, value_align)?,
                LLVMTypeKind::LLVMDoubleTypeKind =>
                    destination.write_f64(ctx, source.read_f64(&*ctx, value_align)?, value_align)?,
                LLVMTypeKind::LLVMX86_FP80TypeKind
                | LLVMTypeKind::LLVMFP128TypeKind
                | LLVMTypeKind::LLVMPPC_FP128TypeKind => {
                    let value = source.read_unsigned(ctx, value_size, value_align)?;
                    destination.write_unsigned(ctx, value, value_size, value_align)?;
                }
                _ => throw_unsup_llvm_type!(ft),
            },
        BasicTypeEnum::IntType(_) => {
            let value = source.read_unsigned(ctx, value_size, value_align)?;
            destination.write_unsigned(ctx, value, value_size, value_align)?;
        }
        BasicTypeEnum::PointerType(_) => {
            destination.write_pointer(
                ctx,
                source.read_pointer(&*ctx, value_align)?,
                value_align,
            )?;
        }
        BasicTypeEnum::StructType(st) => {
            source.check_aggregate_size(ctx, st.count_fields())?;
            destination.ensure_aggregate_size(ctx, st.count_fields())?;

            for i in 0..st.count_fields() {
                let curr_field_ty = st.get_field_type_at_index(i).unwrap();
                let curr_field_size = ctx.resolve_llvm_type_size(curr_field_ty)?;
                let curr_field_size = Size::from_bytes(curr_field_size);
                let field_source = source.resolve_field(&*ctx, curr_field_size, i)?;
                let mut field_destination = destination.resolve_field(ctx, curr_field_size, i)?;
                memory_access_core(
                    ctx,
                    &field_source,
                    &mut field_destination,
                    curr_field_ty,
                    curr_field_size,
                    value_align,
                )?;
            }
        }
        BasicTypeEnum::ArrayType(at) => {
            access_aggregate(
                ctx,
                source,
                destination,
                at.get_element_type(),
                value_size,
                value_align,
                at.len(),
            )?;
        }
        BasicTypeEnum::VectorType(vt) =>
            access_aggregate(
                ctx,
                source,
                destination,
                vt.get_element_type(),
                value_size,
                value_align,
                vt.get_size(),
            )?,
    }
    Ok(())
}

#[allow(clippy::arithmetic_side_effects)]
fn access_aggregate<'tcx, S, D>(
    ctx: &mut MiriInterpCx<'tcx>,
    source: &dyn Source<S>,
    destination: &mut dyn Destination<D>,
    value_type: BasicTypeEnum<'_>,
    value_size: Size,
    value_align: Align,
    aggregate_length: u32,
) -> InterpResult<'tcx>
where
    S: Source<S>,
    D: Destination<D>,
{
    source.check_aggregate_size(ctx, aggregate_length)?;
    destination.ensure_aggregate_size(ctx, aggregate_length)?;
    let item_size = if aggregate_length > 0 {
        value_size.bytes() / u64::from(aggregate_length)
    } else {
        u64::from(aggregate_length)
    };
    let item_size = Size::from_bytes(item_size);
    for i in 0..aggregate_length {
        let field_source = source.resolve_field(ctx, item_size, i)?;
        let mut field_destination = destination.resolve_field(ctx, item_size, i)?;
        memory_access_core(
            ctx,
            &field_source,
            &mut field_destination,
            value_type,
            item_size,
            value_align,
        )?;
    }
    Ok(())
}

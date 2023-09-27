use crate::shims::llvm::helpers::EvalContextExt;
use crate::MiriInterpCx;
use inkwell::types::BasicTypeEnum;
use llvm_sys::LLVMTypeKind;
use rustc_const_eval::interpret::{InterpResult, Pointer};

pub trait Source<T> {
    fn read_f32<'tcx>(&self, ctx: &MiriInterpCx<'_, 'tcx>) -> InterpResult<'tcx, f32>;
    fn read_f64<'tcx>(&self, ctx: &MiriInterpCx<'_, 'tcx>) -> InterpResult<'tcx, f64>;
    fn read_unsigned<'tcx>(
        &self,
        ctx: &mut MiriInterpCx<'_, 'tcx>,
        bytes: u64,
    ) -> InterpResult<'tcx, u128>;
    fn read_pointer<'tcx>(
        &self,
        ctx: &MiriInterpCx<'_, 'tcx>,
    ) -> InterpResult<'tcx, Pointer<Option<crate::Provenance>>>;
    fn check_aggregate_size<'tcx>(
        &self,
        ctx: &MiriInterpCx<'_, 'tcx>,
        aggregate_size: u32,
    ) -> InterpResult<'tcx>;
    fn resolve_field<'tcx>(
        &self,
        ctx: &MiriInterpCx<'_, 'tcx>,
        size: u64,
        index: u32,
    ) -> InterpResult<'tcx, T>;
}
pub trait Destination<T> {
    fn write_f32<'tcx>(
        &mut self,
        ctx: &mut MiriInterpCx<'_, 'tcx>,
        value: f32,
    ) -> InterpResult<'tcx>;
    fn write_f64<'tcx>(
        &mut self,
        ctx: &mut MiriInterpCx<'_, 'tcx>,
        value: f64,
    ) -> InterpResult<'tcx>;
    fn write_unsigned<'tcx>(
        &mut self,
        ctx: &mut MiriInterpCx<'_, 'tcx>,
        value: u128,
        bytes: u64,
    ) -> InterpResult<'tcx>;
    fn write_pointer<'tcx>(
        &mut self,
        ctx: &mut MiriInterpCx<'_, 'tcx>,
        pointer: Pointer<Option<crate::Provenance>>,
    ) -> InterpResult<'tcx>;
    fn resolve_field<'tcx>(
        &mut self,
        ctx: &mut MiriInterpCx<'_, 'tcx>,
        size: u64,
        index: u32,
    ) -> InterpResult<'tcx, T>;
    fn ensure_aggregate_size<'tcx>(
        &self,
        ctx: &MiriInterpCx<'_, 'tcx>,
        aggregate_size: u32,
    ) -> InterpResult<'tcx>;
}

#[allow(clippy::arithmetic_side_effects)]
pub fn memory_access_core<'tcx, S, D>(
    ctx: &mut MiriInterpCx<'_, 'tcx>,
    source: &dyn Source<S>,
    destination: &mut dyn Destination<D>,
    value_type: BasicTypeEnum<'_>,
    value_size: u64,
) -> InterpResult<'tcx, ()>
where
    S: Source<S>,
    D: Destination<D>,
{
    match value_type {
        BasicTypeEnum::FloatType(ft) =>
            match value_type.get_llvm_type_kind() {
                LLVMTypeKind::LLVMFloatTypeKind =>
                    destination.write_f32(ctx, source.read_f32(&*ctx)?)?,
                LLVMTypeKind::LLVMDoubleTypeKind =>
                    destination.write_f64(ctx, source.read_f64(&*ctx)?)?,
                LLVMTypeKind::LLVMX86_FP80TypeKind
                | LLVMTypeKind::LLVMFP128TypeKind
                | LLVMTypeKind::LLVMPPC_FP128TypeKind => {
                    let value = source.read_unsigned(ctx, value_size)?;
                    destination.write_unsigned(ctx, value, value_size)?;
                }
                _ => throw_unsup_llvm_type!(ft),
            },
        BasicTypeEnum::IntType(_) => {
            let value = source.read_unsigned(ctx, value_size)?;
            destination.write_unsigned(ctx, value, value_size)?;
        }
        BasicTypeEnum::PointerType(_) => {
            destination.write_pointer(ctx, source.read_pointer(&*ctx)?)?;
        }
        BasicTypeEnum::StructType(st) => {
            source.check_aggregate_size(ctx, st.count_fields())?;
            destination.ensure_aggregate_size(ctx, st.count_fields())?;

            for i in 0..st.count_fields() {
                let curr_field_ty = st.get_field_type_at_index(i).unwrap();
                let curr_field_size = ctx.resolve_llvm_type_size(curr_field_ty)?;

                let field_source = source.resolve_field(&*ctx, curr_field_size, i)?;
                let mut field_destination = destination.resolve_field(ctx, curr_field_size, i)?;
                memory_access_core(
                    ctx,
                    &field_source,
                    &mut field_destination,
                    curr_field_ty,
                    curr_field_size,
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
                vt.get_size(),
            )?,
    }
    Ok(())
}

#[allow(clippy::arithmetic_side_effects)]
fn access_aggregate<'tcx, S, D>(
    ctx: &mut MiriInterpCx<'_, 'tcx>,
    source: &dyn Source<S>,
    destination: &mut dyn Destination<D>,
    value_type: BasicTypeEnum<'_>,
    value_size: u64,
    aggregate_length: u32,
) -> InterpResult<'tcx>
where
    S: Source<S>,
    D: Destination<D>,
{
    source.check_aggregate_size(ctx, aggregate_length)?;
    destination.ensure_aggregate_size(ctx, aggregate_length)?;
    let item_size = if aggregate_length > 0 {
        value_size / u64::from(aggregate_length)
    } else {
        u64::from(aggregate_length)
    };
    for i in 0..aggregate_length {
        let field_source = source.resolve_field(ctx, item_size, i)?;
        let mut field_destination = destination.resolve_field(ctx, item_size, i)?;
        memory_access_core(ctx, &field_source, &mut field_destination, value_type, item_size)?;
    }
    Ok(())
}

use crate::shims::llvm::helpers::EvalContextExt;
use crate::MiriInterpCx;
use inkwell::{types::BasicTypeEnum, values::GenericValueRef};
use llvm_sys::{
    execution_engine::LLVMGenericValueRef,
    miri::{MiriInterpCxOpaque, MiriPointer},
    prelude::LLVMTypeRef,
};
use rustc_const_eval::interpret::InterpResult;
use super::access::memory_access_core;
use super::memory::obtain_ctx_mut;

fn memory_load_result<'tcx>(
    ctx: &mut MiriInterpCx<'_, 'tcx>,
    destination_ref: LLVMGenericValueRef,
    source: MiriPointer,
    value_type_ref: LLVMTypeRef,
    value_size: u64,
) -> InterpResult<'tcx, ()> {
    let source = ctx.lli_wrapped_pointer_to_resolved_pointer(source)?;
    let mut destination = GenericValueRef::new(destination_ref);
    let value_type = unsafe { BasicTypeEnum::new(value_type_ref) };
    memory_access_core(ctx, &source, &mut destination, value_type, value_size)?;
    Ok(())
}

pub extern "C-unwind" fn miri_memory_load(
    ctx_raw: *mut MiriInterpCxOpaque,
    dest_value_ref: LLVMGenericValueRef,
    source: MiriPointer,
    dest_type_ref: LLVMTypeRef,
    dest_type_size: u64,
    _dest_type_alignment: u64,
) -> bool {
    let ctx = obtain_ctx_mut(ctx_raw);
    let result = memory_load_result(ctx, dest_value_ref, source, dest_type_ref, dest_type_size);
    let status = result.is_err();
    if let Err(e) = result {
        ctx.set_foreign_error(e);
    }
    status
}

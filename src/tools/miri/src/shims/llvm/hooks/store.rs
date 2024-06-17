use super::access::memory_access_core;
use super::memory::obtain_ctx_mut;
use crate::shims::llvm::helpers::EvalContextExt;
use rustc_target::abi::{Align, Size};
use crate::MiriInterpCx;
use inkwell::{types::BasicTypeEnum, values::GenericValueRef};
use llvm_sys::{
    execution_engine::LLVMGenericValueRef,
    miri::{MiriInterpCxOpaque, MiriPointer},
    prelude::LLVMTypeRef,
};
use rustc_const_eval::interpret::InterpResult;

fn memory_store_result<'tcx>(
    ctx: &mut MiriInterpCx<'tcx>,
    source_ref: LLVMGenericValueRef,
    destination_pointer: MiriPointer,
    value_type_ref: LLVMTypeRef,
    value_size: u64,
    value_alignment: u64,
) -> InterpResult<'tcx> {
    let mut destination = ctx.lli_wrapped_pointer_to_resolved_pointer(destination_pointer)?;
    let source = unsafe { GenericValueRef::new(source_ref) };
    let value_type = unsafe { BasicTypeEnum::new(value_type_ref) };
    let value_size = Size::from_bytes(value_size);
    let value_align = Align::from_bytes(value_alignment).unwrap();
    memory_access_core(ctx, &source, &mut destination, value_type, value_size, value_align)?;
    Ok(())
}

pub extern "C-unwind" fn miri_memory_store(
    ctx_raw: *mut MiriInterpCxOpaque,
    source_ref: LLVMGenericValueRef,
    destination_pointer: MiriPointer,
    value_type_ref: LLVMTypeRef,
    value_size: u64,
    value_alignment: u64,
) -> bool {
    let ctx = obtain_ctx_mut(ctx_raw);
    let store_result = memory_store_result(
        ctx,
        source_ref,
        destination_pointer,
        value_type_ref,
        value_size,
        value_alignment,
    );
    let status = store_result.is_err();
    if let Err(e) = store_result {
        ctx.set_foreign_error(e);
    }
    status
}

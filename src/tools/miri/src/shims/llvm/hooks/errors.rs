use super::{super::helpers::EvalContextExt, memory::obtain_ctx_mut};
use inkwell::execution_engine::MiriInterpCxOpaque;
use inkwell::miri::StackTrace;
use llvm_sys::miri::MiriErrorTrace;

pub extern "C-unwind" fn miri_error_trace_recorder(
    ctx_raw: *mut MiriInterpCxOpaque,
    error_traces: *const MiriErrorTrace,
    num_error_traces: u64,
) {
    let ctx = obtain_ctx_mut(ctx_raw);
    let trace_slice = unsafe {
        std::slice::from_raw_parts(error_traces, usize::try_from(num_error_traces).unwrap())
    };
    let stack_trace = StackTrace::new(trace_slice);
    ctx.set_foreign_stack_trace(stack_trace);
}

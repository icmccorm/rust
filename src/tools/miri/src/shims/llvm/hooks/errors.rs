use super::{super::helpers::EvalContextExt, memory::obtain_ctx_mut};
use inkwell::execution_engine::MiriInterpCxOpaque;
use inkwell::miri::StackTrace;
use llvm_sys::miri::MiriErrorTrace;
use std::ffi::CString;

pub extern "C-unwind" fn miri_error_trace_recorder(
    ctx_raw: *mut MiriInterpCxOpaque,
    error_traces: *const MiriErrorTrace,
    num_error_traces: u64,
    instruction_name: *const ::std::os::raw::c_char,
    instruction_name_length: u64,
) {
    let last_instruction = if instruction_name_length > 0 {
        unsafe {
            CString::new(std::slice::from_raw_parts(
                instruction_name as *const u8,
                usize::try_from(instruction_name_length).unwrap(),
            ))
            .ok()
            .map(|s| s.to_string_lossy().to_string())
        }
    } else {
        None
    };

    let ctx = obtain_ctx_mut(ctx_raw);
    let trace_slice = unsafe {
        std::slice::from_raw_parts(error_traces, usize::try_from(num_error_traces).unwrap())
    };
    let stack_trace = StackTrace::new(last_instruction, trace_slice);
    ctx.set_foreign_stack_trace(stack_trace);
}

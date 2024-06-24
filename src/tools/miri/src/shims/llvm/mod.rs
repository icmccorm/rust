pub mod convert;
mod helpers;

pub mod hooks;
mod lli;
pub use lli::LLI;
use lli::{ResolvedRustArgument, LLVM_INTERPRETER};

mod logging;
pub use logging::{LLVMFlag, LLVMLogger};

pub mod threads;
pub mod values;

use inkwell::{
    types::BasicTypeEnum,
    values::{GenericValue, GenericValueRef},
};

use crate::rustc_middle::ty::layout::HasTyCtxt;
use crate::*;
use rustc_const_eval::interpret::InterpResult;
use rustc_data_structures::fx::FxHashSet;
use rustc_span::{FileNameDisplayPreference, Symbol};

#[macro_export]
macro_rules! throw_interop_format {
    ($($tt:tt)*) => { throw_machine_stop!($crate::TerminationInfo::InteroperationError { msg: format!($($tt)*) }) };
}

#[macro_export]
macro_rules! err_interop_format {
    ($($tt:tt)*) => { Err($crate::InterpErrorInfo::MachineStop($crate::TerminationInfo::InteroperationError { msg: format!($($tt)*) })) };
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn with_lli_opt<T>(f: impl FnOnce(&LLI) -> T) -> Option<T> {
        LLVM_INTERPRETER.lock().borrow().as_ref().map(|lli| f(lli))
    }

    fn with_lli<T>(f: impl FnOnce(&LLI) -> T) -> T {
        f(LLVM_INTERPRETER.lock().borrow().as_ref().unwrap())
    }

    fn has_llvm_interpreter(&self) -> bool {
        LLVM_INTERPRETER.lock().borrow().is_some()
    }

    fn in_llvm(&self) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_ref();
        Ok(this.active_thread_ref().is_llvm_thread())
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

    fn get_pending_return_value(
        &mut self,
        id: ThreadId,
    ) -> InterpResult<'tcx, Option<GenericValueRef<'static>>> {
        let this = self.eval_context_mut();
        let pending_return = this.machine.pending_return_values.remove(&id);
        Ok(pending_return)
    }

    fn call_external_llvm_fct(
        &mut self,
        link_name: Symbol,
        dest: &PlaceTy<'tcx>,
        args: &[OpTy<'tcx>],
    ) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_mut();
        MiriInterpCx::<'_>::with_lli_opt(|ll| {
            ll.with_module(|module| {
                let function_opt = module.get_function(link_name.as_str());
                let result = function_opt.is_some();
                if let Some(function) = function_opt {
                    if let Some(ref logger) = &this.machine.llvm_logger {
                        logger.log_flag(LLVMFlag::LLVMEngaged)
                    }
                    ll.call_external_llvm_and_store_return(this, function, args, dest)?;
                }
                Ok(result)
            })
        })
        .unwrap_or(Ok(false))
    }

    fn perform_opty_conversion<'lli>(
        &mut self,
        return_place: &PlaceTy<'tcx>,
        return_type_opt: Option<BasicTypeEnum<'lli>>,
    ) -> InterpResult<'tcx, GenericValue<'lli>> {
        let this = self.eval_context_mut();
        if let Some(return_type) = return_type_opt {
            let place_opty = this.place_to_op(return_place)?;
            let place_resolved = ResolvedRustArgument::new(this, place_opty)?;
            place_resolved.to_generic_value(this, Some(return_type))
        } else {
            Ok(GenericValue::new_void())
        }
    }

    fn terminate_lli_thread(&self, id: ThreadId) {
        MiriInterpCx::<'_>::with_lli(|ll| {
            ll.with_engine(|engine| unsafe {
                engine.as_ref().unwrap().terminate_thread(id.into());
            });
        });
    }

    fn get_thread_exit_value(
        &self,
        id: ThreadId,
    ) -> InterpResult<'tcx, Option<GenericValueRef<'static>>> {
        let this = self.eval_context_ref();
        MiriInterpCx::<'_>::with_lli(|ll| {
            ll.with_engine(|engine| unsafe {
                let gv_opt = engine.as_ref().unwrap().get_thread_exit_value(id.into());
                if let Some(info) = this.get_foreign_error() { Err(info) } else { Ok(gv_opt) }
            })
        })
    }

    fn init_llvm_interpreter(&mut self, config: &MiriConfig) {
        let this = self.eval_context_mut();
        LLVM_INTERPRETER.lock().replace_with(|_| {
            let mut interpreter = if let Some(path_buff) = &config.singular_llvm_bc_file {
                let mut set = FxHashSet::default();
                set.insert(path_buff.clone());
                LLI::create(this, &set)
            } else {
                LLI::create(this, &config.external_bc_files)
            };
            interpreter.initialize_engine(this).unwrap();
            Some(interpreter)
        });
    }
    fn step_lli_thread(&mut self, id: ThreadId) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_mut();
        let pending_return_opt = this.get_pending_return_value(id)?;
        MiriInterpCx::<'_>::with_lli(|ll| {
            ll.with_engine(|engine| unsafe {
                let result = engine.as_ref().unwrap().step_thread(id.into(), pending_return_opt);
                if let Some(pending_return) = pending_return_opt {
                    drop(GenericValue::from_raw(pending_return.into_raw()));
                }
                if let Some(info) = this.get_foreign_error() { Err(info) } else { Ok(result) }
            })
        })
    }
}

#[macro_export]
macro_rules! throw_llvm_field_count_mismatch {
    ($llvm_field_count:expr, $rust_layout:expr) => {
        throw_interop_format!(
            "LLVM field count mismatch: cannot convert an LLVM value with {} fields to a Rust value of type `{}` which has {} fields.",
            $llvm_field_count,
            $rust_layout.ty,
            $rust_layout.fields.count()
        )
    };
}
#[macro_export]
macro_rules! throw_llvm_field_width_mismatch {
    ($llvm_field_width:expr, $rust_layout:expr) => {
        throw_interop_format!(
            "LLVM field width mismatch: cannot convert an LLVM field of width {} to a Rust field of type `{}` which has width {}",
            $llvm_field_width,
            $rust_layout.ty,
            $rust_layout.size.bytes()
        )
    };
}
#[macro_export]
macro_rules! throw_llvm_type_mismatch {
    ($llvm_type:expr, $rust_type:expr) => {
        throw_interop_format!(
            "LLVM type mismatch: cannot convert an LLVM value of type `{}` to a Rust value of type `{}`",
            $llvm_type.print_to_string().to_string(),
            $rust_type
        )
    };
}
#[macro_export]
macro_rules! throw_unsup_abi {
    ($abi_string:expr) => {
        throw_interop_format!("Unsupported target ABI: {}", $abi_string)
    };
}
#[macro_export]
macro_rules! throw_unsup_llvm_type {
    ($llvm_type:expr) => {
        throw_interop_format!("Unsupported LLVM Type: {}", $llvm_type.print_to_string().to_string())
    };
}

#[macro_export]
macro_rules! throw_unsup_shim_llvm_type {
    ($llvm_type:expr) => {
        throw_interop_format!(
            "LLVM Type `{}` is not supported for use in shims.",
            $llvm_type.print_to_string().to_string()
        )
    };
}

#[macro_export]
macro_rules! throw_rust_type_mismatch {
    ($rust_layout:expr, $llvm_type:expr) => {
        throw_interop_format!(
            "Rust type mismatch: cannot convert a Rust value of type `{}` to an LLVM value of type `{}`.",
            $rust_layout.ty,
            $llvm_type.print_to_string().to_string()
        )
    }
}
#[macro_export]
macro_rules! throw_unsup_var_arg {
    ($rust_layout:expr) => {
        throw_interop_format!(
            "Non-scalar variable arguments are not supported: `{}`.",
            $rust_layout.ty,
        )
    };
}
#[macro_export]
macro_rules! throw_rust_field_mismatch {
    ($rust_layout:expr, $llvm_field_count:expr) => {
        throw_interop_format!(
            "Rust field count mismatch: cannot convert a Rust value of type `{}` which has {} fields to an LLVM value with {} fields",
            $rust_layout.ty,
            $rust_layout.fields.count(),
            $llvm_field_count,
        )
    }
}

#[macro_export]
macro_rules! throw_shim_argument_mismatch {
    ($shim_name:expr, $arg_count:expr, $actual_arg_count:expr) => {
        throw_interop_format!(
            "shim argument mismatch: shim {} expects {} arguments but {} were provided",
            $shim_name,
            $arg_count,
            $actual_arg_count,
        )
    };
}

#[macro_export]
macro_rules! throw_llvm_argument_mismatch {
    ($function: expr, $rust_args: expr, $llvm_args: expr) => {
        throw_interop_format!(
            "argument count mismatch: LLVM function {:?} {} expects {}{} arguments, but Rust provided {}",
            $function.get_name(),
            $function.get_type().print_to_string().to_string(),
            (if $function.get_type().is_var_arg() { "at least " } else { "" }),
            $llvm_args,
            $rust_args
        )
    };
}

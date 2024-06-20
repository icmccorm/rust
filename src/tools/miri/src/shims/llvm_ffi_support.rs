use super::llvm::lli::LLVM_INTERPRETER;
use crate::shims::llvm::helpers::EvalContextExt as LLVMEvalContextExt;
use crate::shims::llvm::lli::ResolvedRustArgument;
use crate::shims::llvm::lli::LLI;
use crate::shims::llvm::logging::LLVMFlag;

use inkwell::types::BasicTypeEnum;
use inkwell::values::{GenericValue, GenericValueRef};
use rustc_const_eval::interpret::InterpResult;
use rustc_data_structures::fx::FxHashSet;
use rustc_span::Symbol;
use crate::{PlaceTy, OpTy};
use crate::*;

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

    /// [MiriLLI] Calls an external LLVM function and stores the return value in the destination.
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

    fn with_lli_opt<T>(f: impl FnOnce(&LLI) -> T) -> Option<T> {
        LLVM_INTERPRETER.lock().borrow().as_ref().map(|lli| f(lli))
    }

    fn with_lli<T>(f: impl FnOnce(&LLI) -> T) -> T {
        f(LLVM_INTERPRETER.lock().borrow().as_ref().unwrap())
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
    fn has_llvm_interpreter(&self) -> bool {
        LLVM_INTERPRETER.lock().borrow().is_some()
    }
}

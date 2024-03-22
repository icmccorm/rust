use super::llvm::lli::LLVM_INTERPRETER;
use crate::shims::llvm::helpers::EvalContextExt as LLVMEvalContextExt;
use crate::shims::llvm::lli::ResolvedRustArgument;
use crate::shims::llvm::logging::LLVMFlag;
use crate::Provenance;
use crate::ThreadId;
use inkwell::types::BasicTypeEnum;
use inkwell::values::{GenericValue, GenericValueRef};
use log::debug;
use rustc_const_eval::interpret::{InterpResult, OpTy, PlaceTy};
use rustc_span::Symbol;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}

pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn call_external_llvm_fct(
        &mut self,
        link_name: Symbol,
        dest: &PlaceTy<'tcx, Provenance>,
        args: &[OpTy<'tcx, Provenance>],
    ) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_mut();
        if let Some(ll) = LLVM_INTERPRETER.lock().borrow().as_ref() {
            return ll.with_module(|module| {
                let function_opt = module.get_function(link_name.as_str());
                let result = function_opt.is_some();
                if let Some(function) = function_opt {
                    if let Some(ref logger) = &this.machine.llvm_logger {
                        logger.log_flag(LLVMFlag::LLVMEngaged)
                    }
                    ll.call_external_llvm_and_store_return(this, function, args, dest)?;
                }
                Ok(result)
            });
        }
        Ok(false)
    }

    fn perform_opty_conversion<'lli>(
        &mut self,
        return_place: &PlaceTy<'tcx, Provenance>,
        return_type_opt: Option<BasicTypeEnum<'lli>>,
    ) -> InterpResult<'tcx, GenericValue<'lli>> {
        let this = self.eval_context_mut();
        if let Some(return_type) = return_type_opt {
            debug!("Preparing GV return value");
            let place_opty = this.place_to_op(return_place)?;
            let place_resolved = ResolvedRustArgument::new(this, place_opty)?;
            place_resolved.to_generic_value(this, Some(return_type))
        } else {
            debug!("Preparing void return value");
            Ok(GenericValue::new_void())
        }
    }

    fn terminate_lli_thread(&self, id: ThreadId) {
        if let Some(ll) = LLVM_INTERPRETER.lock().borrow().as_ref() {
            ll.with_engine(|engine| {
                unsafe {
                    engine.as_ref().unwrap().terminate_thread(id.into());
                }
            });
            return;
        }
        bug!("LLVM Interpreter not initialized")
    }

    fn thread_is_lli_thread(&self, id: ThreadId) -> InterpResult<'tcx, bool> {
        if let Some(ll) = LLVM_INTERPRETER.lock().borrow().as_ref() {
            return ll.with_engine(|engine| unsafe {
                if let Some(re) = engine.as_ref() {
                    Ok(re.has_thread(id.into()))
                } else {
                    Ok(false)
                }
            });
        }
        Ok(false)
    }

    fn get_thread_exit_value(
        &self,
        id: ThreadId,
    ) -> InterpResult<'tcx, Option<GenericValueRef<'static>>> {
        let this = self.eval_context_ref();
        if let Some(ll) = LLVM_INTERPRETER.lock().borrow().as_ref() {
            return ll.with_engine(|engine| unsafe {
                let gv_opt = engine.as_ref().unwrap().get_thread_exit_value(id.into());
                if let Some(info) = this.get_foreign_error() { Err(info) } else { Ok(gv_opt) }
            });
        }
        bug!("LLVM Interpreter not initialized")
    }

    fn step_lli_thread(&mut self, id: ThreadId) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_mut();
        let pending_return_opt = this.get_pending_return_value(id)?;

        if let Some(ll) = LLVM_INTERPRETER.lock().borrow().as_ref() {
            return ll.with_engine(|engine| unsafe {
                let result = engine.as_ref().unwrap().step_thread(id.into(), pending_return_opt);
                if let Some(pending_return) = pending_return_opt {
                    drop(GenericValue::from_raw(pending_return.into_raw()));
                }
                if let Some(info) = this.get_foreign_error() { Err(info) } else { Ok(result) }
            });
        }
        bug!("LLVM Interpreter not initialized")
    }

    fn has_llvm_interpreter(&self) -> bool {
        LLVM_INTERPRETER.lock().borrow().is_some()
    }
}

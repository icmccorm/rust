extern crate rustc_abi;
use super::llvm::lli::LLVM_INTERPRETER;
use crate::machine::MiriInterpCxExt;
use crate::shims::llvm::convert::to_generic_value::convert_opty_to_generic_value;
use crate::shims::llvm::helpers::EvalContextExt as LLVMHelperEvalExt;
use crate::shims::llvm::logging::LLVMFlag;
use crate::MiriInterpCx;
use crate::Provenance;
use crate::ThreadId;
use inkwell::values::GenericValueRef;
use rustc_abi::{Abi, Size};
use rustc_const_eval::interpret::{InterpResult, OpTy, PlaceTy};
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::abi::TyAndLayout;
use inkwell::types::BasicTypeEnum;
use inkwell::values::GenericValue;

pub struct ResolvedRustArgument<'tcx> {
    inner: ResolvedRustArgumentInner<'tcx>,
}

enum ResolvedRustArgumentInner<'tcx> {
    Default(OpTy<'tcx, Provenance>),
    Padded(OpTy<'tcx, Provenance>, Size),
}

impl<'tcx> ResolvedRustArgument<'tcx> {
    pub fn new(
        ctx: &mut MiriInterpCx<'_, 'tcx>,
        arg: OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Self> {
        let arg = ctx.dereference_into_singular_field(arg)?;
        Ok(Self { inner: ResolvedRustArgumentInner::Default(arg) })
    }
    pub fn new_padded(
        ctx: &mut MiriInterpCx<'_, 'tcx>,
        arg: OpTy<'tcx, Provenance>,
        padding: Size,
    ) -> InterpResult<'tcx, Self> {
        let arg = ctx.dereference_into_singular_field(arg)?;
        Ok(Self { inner: ResolvedRustArgumentInner::Padded(arg, padding) })
    }

    pub fn to_generic_value<'lli>(
        self,
        ctx: &mut MiriInterpCx<'_, 'tcx>,
        bte: ResolvedLLVMType<'lli>,
    ) -> InterpResult<'tcx, GenericValue<'lli>> {
        let this = ctx.eval_context_mut();
        convert_opty_to_generic_value(this, self, bte)
    }

    pub fn opty(&self) -> &OpTy<'tcx, Provenance> {
        match &self.inner {
            ResolvedRustArgumentInner::Default(op) => op,
            ResolvedRustArgumentInner::Padded(op, _) => op,
        }
    }
    pub fn padded_size(&self) -> Size {
        match self.inner {
            ResolvedRustArgumentInner::Default(_) => self.value_size(),
            ResolvedRustArgumentInner::Padded(_, padding) => padding,
        }
    }

    pub fn layout(&self) -> TyAndLayout<'tcx, Ty<'tcx>> {
        self.opty().layout
    }

    pub fn abi(&self) -> Abi {
        self.layout().abi
    }

    pub fn value_size(&self) -> Size {
        self.layout().size
    }

    pub fn ty(&self) -> Ty<'tcx> {
        self.layout().ty
    }
}

pub type ResolvedLLVMType<'a> = Option<BasicTypeEnum<'a>>;

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

    fn get_thread_exit_value(&self, id: ThreadId) -> InterpResult<'tcx, Option<GenericValueRef<'static>>> {
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
                if let Some(info) = this.get_foreign_error() { Err(info) } else { Ok(result) }
            });
        }
        bug!("LLVM Interpreter not initialized")
    }

    fn has_llvm_interpreter(&self) -> bool {
        LLVM_INTERPRETER.lock().borrow().is_some()
    }
}

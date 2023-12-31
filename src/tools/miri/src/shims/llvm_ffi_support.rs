extern crate either;
extern crate rustc_abi;
use super::llvm::lli::LLVM_INTERPRETER;
use crate::concurrency::thread::EvalContextExt as ThreadEvalContextExt;
use crate::machine::MiriInterpCxExt;
use crate::shims::llvm::convert::to_generic_value::convert_opty_to_generic_value;
use crate::shims::llvm::convert::to_generic_value::LLVMArgumentConverter;
use crate::shims::llvm::helpers::EvalContextExt as LLVMHelperEvalExt;
use crate::shims::llvm::logging::LLVMFlag;
use crate::shims::llvm::threads::link::ThreadLinkDestination;
use crate::MiriInterpCx;
use crate::Provenance;
use crate::ThreadId;
use either::Either;
use inkwell::{
    attributes::{Attribute, AttributeLoc},
    types::BasicTypeEnum,
    values::{FunctionValue, GenericValue, GenericValueRef},
};
use log::debug;
use rustc_abi::{Abi, Size};
use rustc_const_eval::interpret::{InterpResult, MemoryKind, OpTy, PlaceTy};
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::abi::TyAndLayout;

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
        if let Some(ll) = LLVM_INTERPRETER.lock().borrow().as_ref() {
            return ll.with_module(|m| {
                let function_opt = m.get_function(link_name.as_str());
                let result = function_opt.is_some();
                if let Some(function) = function_opt {
                    if let Some(ref logger) = &self.eval_context_ref().machine.llvm_logger {
                        logger.log_flag(LLVMFlag::LLVMEngaged)
                    }
                    self.call_external_llvm_and_store_return(function, args, dest)?;
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

    #[allow(clippy::arithmetic_side_effects)]
    fn call_external_llvm_and_store_return(
        &mut self,
        function: FunctionValue<'_>,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        debug!(
            "Calling {:?}: {:?}",
            function.get_name(),
            function.get_type().print_to_string().to_string()
        );
        let this = self.eval_context_mut();
        this.update_last_rust_call_location();
        let llvm_parameter_types = function.get_type().get_param_types();

        let has_sret = !llvm_parameter_types.is_empty()
            && function
                .attributes(AttributeLoc::Param(0))
                .iter()
                .any(|e| e.get_enum_kind_id() == Attribute::get_named_enum_kind_id("sret"));

        let (sret_argument, llvm_parameter_types, thread_link_destination): (
            Option<GenericValue<'_>>,
            &[BasicTypeEnum<'_>],
            ThreadLinkDestination<'tcx>,
        ) = if has_sret {
            let (head, tail) = llvm_parameter_types.split_first().unwrap();
            let (op, alloc) = match dest.as_mplace_or_local() {
                Either::Left(mp) => (OpTy::from(mp), ThreadLinkDestination::ToMiriStructReturn),
                Either::Right(_) => {
                    let allocation = this.allocate(
                        dest.layout,
                        MemoryKind::Machine(crate::MiriMemoryKind::LLVMInterop),
                    )?;
                    (
                        OpTy::from(allocation.clone()),
                        ThreadLinkDestination::ToMiriMirroredLocal(allocation, dest.clone()),
                    )
                }
            };
            let as_resolved = ResolvedRustArgument::new(this, op)?;
            let converted = as_resolved.to_generic_value(this, Some(*head))?;
            (Some(converted), tail, alloc)
        } else {
            (
                None,
                llvm_parameter_types.as_slice(),
                ThreadLinkDestination::ToMiriDefault(dest.clone()),
            )
        };
        debug!("link={}, sret={:?}", thread_link_destination, has_sret);
        let converter = LLVMArgumentConverter::new(this, function, args.to_vec(), llvm_parameter_types.to_vec())?;
        let mut generic_args = converter.convert(this, function)?;
        if let Some(sret_argument) = sret_argument {
            generic_args.insert(0, sret_argument);
        }
        let exposed = generic_args
            .drain(0..)
            .map(|gv| GenericValueRef::new(unsafe { gv.into_raw() }))
            .collect::<Vec<_>>();
        let lli_thread_id =
            this.start_rust_to_lli_thread(Some(thread_link_destination), function, exposed)?;
        debug!("Started LLI Thread, TID: {:?}", lli_thread_id);
        Ok(())
    }

    fn create_lli_thread(
        &self,
        id: ThreadId,
        fv: FunctionValue<'_>,
        args: &[GenericValueRef],
    ) -> InterpResult<'tcx> {
        debug!("Creating LLI Thread for {:?}, TID: {:?}", fv.get_name(), id);
        let this = self.eval_context_ref();
        if let Some(ll) = LLVM_INTERPRETER.lock().borrow().as_ref() {
            return ll.with_engine(|engine| {
                unsafe {
                    engine.as_ref().unwrap().create_thread(id.into(), fv, args);
                    if let Some(info) = this.get_foreign_error() {
                        return Err(info);
                    }
                    Ok(())
                }
            });
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

    fn get_thread_exit_value(&self, id: ThreadId) -> InterpResult<'tcx, Option<GenericValueRef>> {
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

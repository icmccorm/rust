extern crate rustc_hash;
use super::hooks;
use crate::concurrency::thread::EvalContextExt;
use crate::shims::llvm::{
    convert::to_generic_value::{convert_opty_to_generic_value, LLVMArgumentConverter},
    helpers::EvalContextExt as _,
    LLVMFlag,
    threads::ThreadLinkDestination,
};
use crate::*;
use either::Either;
use inkwell::{
    attributes::{Attribute, AttributeLoc},
    context::Context,
    execution_engine::{ExecutionEngine, MiriInterpCxOpaque},
    module::Module,
    types::BasicTypeEnum,
    values::{FunctionValue, GenericValue},
};
use ouroboros::self_referencing;
use parking_lot::ReentrantMutex;
use rustc_const_eval::interpret::InterpResult;
use rustc_hash::FxHashSet;
use rustc_middle::ty::{layout::TyAndLayout, Ty};
use rustc_target::abi::{Abi, Size};
use std::{cell::RefCell, path::PathBuf};
use tracing::debug;

#[self_referencing]
pub struct LLI {
    pub context: Context,
    #[borrows(context)]
    #[not_covariant]
    pub module: Module<'this>,
    #[borrows(context)]
    #[not_covariant]
    pub engine: Option<ExecutionEngine<'this>>,
}

unsafe impl Send for LLI {}
unsafe impl Sync for LLI {}

pub static LLVM_INTERPRETER: ReentrantMutex<RefCell<Option<LLI>>> =
    ReentrantMutex::new(RefCell::new(None));

impl LLI {
    pub fn create(miri: &mut MiriInterpCx<'_>, paths: &FxHashSet<PathBuf>) -> Self {
        LLITryBuilder {
            context: Context::create(),
            module_builder: |ctx| {
                let module = Context::create_module(ctx, "main");
                for path in paths.iter() {
                    debug!("LLVM: {}", path.to_string_lossy());
                    match Module::parse_bitcode_from_path(path.clone(), ctx) {
                        Ok(m) => {
                            let error = module.link_in_module(m.clone());
                            if let Some(logger) = &mut miri.eval_context_mut().machine.llvm_logger {
                                logger.log_bytecode(path, error);
                            }
                        }
                        Err(err) =>
                            return Err(format!(
                                "Unable to parse {}: {}",
                                path.to_string_lossy(),
                                err
                            )),
                    };
                }
                Ok(module)
            },
            engine_builder: |_| Ok(None),
        }
        .try_build()
        .expect("Unable to initialize interpreter")
    }
    pub fn initialize_engine<'tcx>(&mut self, miri: &mut MiriInterpCx<'tcx>) -> InterpResult<'tcx> {
        let initialization_result = self.with_mut(|lli| {
            let engine = match lli.module.create_interpreter_execution_engine() {
                Ok(engine) => engine,
                Err(err) => return Err(format!("Unable to initialize interpreter: {}", err)),
            };
            engine.set_miri_free(Some(hooks::llvm_free));
            engine.set_miri_malloc(Some(hooks::llvm_malloc));
            engine.set_miri_load(Some(hooks::miri_memory_load));
            engine.set_miri_store(Some(hooks::miri_memory_store));
            engine.set_miri_call_by_name(Some(hooks::miri_call_by_name));
            engine.set_miri_call_by_pointer(Some(hooks::miri_call_by_pointer));
            engine.set_miri_stack_trace_recorder(Some(hooks::miri_error_trace_recorder));
            engine.set_miri_memcpy(Some(hooks::miri_memcpy));
            engine.set_miri_memset(Some(hooks::miri_memset));
            engine.set_miri_inttoptr(Some(hooks::miri_inttoptr));
            engine.set_miri_ptrtoint(Some(hooks::miri_ptrtoint));
            engine.set_miri_get_element_pointer(Some(hooks::miri_get_element_pointer));
            engine.set_miri_register_global(Some(hooks::miri_register_global));
            engine.set_miri_interpcx_wrapper(miri as *mut _ as *mut MiriInterpCxOpaque);
            lli.engine.replace(engine);
            Ok(())
        });
        Ok(initialization_result.map_err(|s| err_unsup_format!("{}", s))?)
    }

    pub fn run_constructors<'tcx>(&self, miri: &mut MiriInterpCx<'tcx>) -> InterpResult<'tcx> {
        self.with_engine(|engine| {
            miri.active_thread_ref().set_llvm_thread(true);
            let engine = engine.as_ref().unwrap();
            let result = unsafe {
                let constructors = engine.get_constructors();
                if constructors.len() > 0 {
                    if let Some(logger) = &mut miri.eval_context_mut().machine.llvm_logger {
                        logger.log_flag(LLVMFlag::LLVMInvokedConstructor)
                    }
                }
                constructors
                    .iter()
                    .try_for_each(|cstor| miri.run_lli_function_to_completion(engine, *cstor))
            };
            miri.active_thread_ref().set_llvm_thread(false);
            result
        })
    }
    pub fn run_destructors<'tcx>(&self, miri: &mut MiriInterpCx<'tcx>) -> InterpResult<'tcx> {
        self.with_engine(|engine| {
            miri.active_thread_ref().set_llvm_thread(true);
            let engine = engine.as_ref().unwrap();
            let result = unsafe {
                let destructors = engine.get_destructors();
                if destructors.len() > 0 {
                    if let Some(logger) = &mut miri.eval_context_mut().machine.llvm_logger {
                        logger.log_flag(LLVMFlag::LLVMInvokedDestructor)
                    }
                }
                destructors
                    .iter()
                    .try_for_each(|dstor| miri.run_lli_function_to_completion(engine, *dstor))
            };
            miri.active_thread_ref().set_llvm_thread(false);
            result
        })
    }

    #[allow(clippy::arithmetic_side_effects)]
    pub fn call_external_llvm_and_store_return<'lli, 'tcx>(
        &self,
        ctx: &mut MiriInterpCx<'tcx>,
        function: FunctionValue<'lli>,
        args: &[OpTy<'tcx>],
        dest: &PlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        debug!(
            "Calling {:?}: {:?}",
            function.get_name(),
            function.get_type().print_to_string().to_string()
        );
        self.with_engine(|engine| {
            let engine = engine.as_ref().unwrap();
            let this = ctx.eval_context_mut();

            this.update_last_rust_call_location();

            let llvm_parameter_types = function.get_type().get_param_types();

            let has_sret = !llvm_parameter_types.is_empty()
                && function
                    .attributes(AttributeLoc::Param(0))
                    .iter()
                    .any(|e| e.get_enum_kind_id() == Attribute::get_named_enum_kind_id("sret"));

            let (sret_argument, llvm_parameter_types, thread_link_destination): (
                Option<GenericValue<'lli>>,
                &[BasicTypeEnum<'lli>],
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
            let converter = LLVMArgumentConverter::new(
                this,
                function,
                args.to_vec(),
                llvm_parameter_types.to_vec(),
            )?;
            let mut generic_args = converter.convert(this, function)?;
            if let Some(sret_argument) = sret_argument {
                generic_args.insert(0, sret_argument);
            }
            let lli_thread_id = this.start_rust_to_lli_thread(
                engine,
                Some(thread_link_destination),
                function,
                generic_args,
            )?;
            debug!("Started LLI Thread, TID: {:?}", lli_thread_id);
            Ok(())
        })
    }
}

pub struct ResolvedRustArgument<'tcx> {
    inner: ResolvedRustArgumentInner<'tcx>,
}
enum ResolvedRustArgumentInner<'tcx> {
    Default(OpTy<'tcx>),
    Padded(OpTy<'tcx>, Size),
}

impl<'tcx> ResolvedRustArgument<'tcx> {
    pub fn new(ctx: &mut MiriInterpCx<'tcx>, arg: OpTy<'tcx>) -> InterpResult<'tcx, Self> {
        let arg = ctx.dereference_into_singular_field(arg)?;
        Ok(Self { inner: ResolvedRustArgumentInner::Default(arg) })
    }
    pub fn new_padded(
        ctx: &mut MiriInterpCx<'tcx>,
        arg: OpTy<'tcx>,
        padding: Size,
    ) -> InterpResult<'tcx, Self> {
        let arg = ctx.dereference_into_singular_field(arg)?;
        Ok(Self { inner: ResolvedRustArgumentInner::Padded(arg, padding) })
    }

    pub fn to_generic_value<'lli>(
        self,
        ctx: &mut MiriInterpCx<'tcx>,
        bte: ResolvedLLVMType<'lli>,
    ) -> InterpResult<'tcx, GenericValue<'lli>> {
        let this = ctx.eval_context_mut();
        let mut value = GenericValue::new_void();
        convert_opty_to_generic_value(this, value.as_mut(), self, bte)?;
        Ok(value)
    }

    pub fn opty(&self) -> &OpTy<'tcx> {
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

    pub fn layout(&self) -> TyAndLayout<'tcx> {
        self.opty().layout
    }

    pub fn abi(&self) -> Abi {
        self.layout().abi
    }

    pub fn value_size(&self) -> Size {
        self.layout().size
    }

    pub fn is_immediate(&self) -> bool {
        self.opty().as_mplace_or_imm().is_right()
    }

    pub fn ty(&self) -> Ty<'tcx> {
        self.layout().ty
    }
}

pub type ResolvedLLVMType<'a> = Option<BasicTypeEnum<'a>>;

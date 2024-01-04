extern crate rustc_hash;
use super::hooks::calls::{miri_call_by_name, miri_call_by_pointer};
use super::hooks::errors::miri_error_trace_recorder;
use super::hooks::intptr::{miri_inttoptr, miri_ptrtoint, miri_get_element_pointer};
use super::hooks::load::miri_memory_load;
use super::hooks::memory::{
    llvm_free, llvm_malloc, miri_memcpy, miri_memset, miri_register_global,
};
use super::hooks::store::miri_memory_store;
use crate::concurrency::thread::EvalContextExt;
use crate::{MiriInterpCx, MiriInterpCxExt};
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, MiriInterpCxOpaque};
use inkwell::module::Module;
pub use inkwell::values::FunctionValue;
use log::debug;
use ouroboros::self_referencing;
use parking_lot::ReentrantMutex;
use rustc_const_eval::interpret::InterpResult;
use rustc_hash::FxHashSet;
use std::cell::RefCell;
use std::path::PathBuf;
use crate::shims::llvm::logging::LLVMFlag;
use crate::shims::llvm::helpers::EvalContextExt as LLVMEvalContextExt;
use crate::shims::llvm::convert::to_generic_value::LLVMArgumentConverter;
use crate::shims::llvm::threads::link::ThreadLinkDestination;
use rustc_const_eval::interpret::OpTy;
use crate::MemoryKind;
use crate::shims::either::Either;
use inkwell::types::BasicTypeEnum;
use inkwell::values::GenericValue;
use inkwell::attributes::Attribute;
use crate::Provenance;
use rustc_const_eval::interpret::PlaceTy;
use inkwell::attributes::AttributeLoc;
use crate::shims::llvm::convert::to_generic_value::convert_opty_to_generic_value;
use rustc_target::abi::Size;
use rustc_middle::ty::{layout::TyAndLayout, Ty};
use rustc_target::abi::Abi;

#[self_referencing]
pub struct LLI /*<'mir, 'tcx>*/ {
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
    pub fn create(miri: &mut MiriInterpCx<'_, '_>, paths: &FxHashSet<PathBuf>) -> Self {
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
    pub fn initialize_engine<'tcx>(
        &mut self,
        miri: &mut MiriInterpCx<'_, 'tcx>,
    ) -> InterpResult<'tcx> {
        let initialization_result = self.with_mut(|n| {
            let engine = match n.module.create_interpreter_execution_engine() {
                Ok(engine) => engine,
                Err(err) => return Err(format!("Unable to initialize interpreter: {}", err)),
            };
            engine.set_miri_free(Some(llvm_free));
            engine.set_miri_malloc(Some(llvm_malloc));
            engine.set_miri_load(Some(miri_memory_load));
            engine.set_miri_store(Some(miri_memory_store));
            engine.set_miri_call_by_name(Some(miri_call_by_name));
            engine.set_miri_call_by_pointer(Some(miri_call_by_pointer));
            engine.set_miri_stack_trace_recorder(Some(miri_error_trace_recorder));
            engine.set_miri_memcpy(Some(miri_memcpy));
            engine.set_miri_memset(Some(miri_memset));
            engine.set_miri_inttoptr(Some(miri_inttoptr));
            engine.set_miri_ptrtoint(Some(miri_ptrtoint));
            engine.set_miri_get_element_pointer(Some(miri_get_element_pointer));
            engine.set_miri_register_global(Some(miri_register_global));
            engine.set_miri_interpcx_wrapper(miri as *mut _ as *mut MiriInterpCxOpaque);
            n.engine.replace(engine);
            Ok(())
        });
        Ok(initialization_result.map_err(|s| err_unsup_format!("{}", s))?)
    }

    pub fn run_constructors<'tcx>(&self, miri: &mut MiriInterpCx<'_, 'tcx>) -> InterpResult<'tcx> {
        self.with_engine(|engine| {
            let engine = engine.as_ref().unwrap();
            unsafe {
                let constructors = engine.get_constructors();
                if constructors.len() > 0 {
                    if let Some(logger) = &mut miri.eval_context_mut().machine.llvm_logger {
                        logger.log_flag(LLVMFlag::LLVMInvokedConstructor)
                    }
                }
                constructors.iter()
                    .try_for_each(|cstor| miri.run_lli_function_to_completion(engine, *cstor))
            }
        })
    }
    pub fn run_destructors<'tcx>(&self, miri: &mut MiriInterpCx<'_, 'tcx>) -> InterpResult<'tcx> {
        self.with_engine(|engine| {
            let engine = engine.as_ref().unwrap();
            unsafe {
                let destructors = engine.get_destructors();
                if destructors.len() > 0 {
                    if let Some(logger) = &mut miri.eval_context_mut().machine.llvm_logger {
                        logger.log_flag(LLVMFlag::LLVMInvokedDestructor)
                    }
                }
                destructors.iter()
                    .try_for_each(|dstor| miri.run_lli_function_to_completion(engine, *dstor))
            }
        })
    }

    #[allow(clippy::arithmetic_side_effects)]
    pub fn call_external_llvm_and_store_return<'lli, 'tcx>(
        &self,
        ctx: &mut MiriInterpCx<'_, 'tcx>,
        function: FunctionValue<'lli>,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
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
            println!("generic_args: {:?}", generic_args);
            let lli_thread_id =
                this.start_rust_to_lli_thread(engine, Some(thread_link_destination), function, generic_args)?;
            debug!("Started LLI Thread, TID: {:?}", lli_thread_id);
            Ok(())
        })
    }
}


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
        let mut value = GenericValue::new_void();
        convert_opty_to_generic_value(this, value.as_mut(), self, bte)?;
        Ok(value)
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

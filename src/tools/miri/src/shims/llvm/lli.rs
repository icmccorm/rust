extern crate rustc_hash;
use super::hooks::calls::{miri_call_by_name, miri_call_by_pointer};
use super::hooks::errors::miri_error_trace_recorder;
use super::hooks::intptr::{miri_inttoptr, miri_ptrtoint};
use super::hooks::load::miri_memory_load;
use super::hooks::memory::{
    llvm_free, llvm_malloc, miri_memcpy, miri_memset, miri_register_global,
};
use super::hooks::store::miri_memory_store;
use crate::{MiriInterpCx, MiriInterpCxExt};
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, MiriInterpCxOpaque};
use inkwell::module::Module;
pub use inkwell::values::FunctionValue;
use log::debug;
use ouroboros::self_referencing;
use parking_lot::ReentrantMutex;
use rustc_hash::FxHashSet;
use std::cell::RefCell;
use std::path::PathBuf;

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
        let result = LLITryBuilder {
            context: Context::create(),
            module_builder: |ctx| {
                let module = Context::create_module(ctx, "main");

                for path in paths.iter() {
                    debug!("LLVM: {}", path.to_string_lossy());
                    match Module::parse_bitcode_from_path(path.clone(), ctx) {
                        Ok(m) => {
                            let error = module.link_in_module(m.clone());
                            if let Some(logger) = & mut miri.eval_context_mut().machine.llvm_logger {
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
        .try_build();

        match result {
            Ok(mut nllm) => {
                nllm.with_mut(|n| {
                    let engine = match n.module.create_interpreter_execution_engine() {
                        Ok(engine) => engine,
                        Err(err) =>
                            return Err(format!("Unable to initialize interpreter: {}", err)),
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
                    engine.set_miri_register_global(Some(miri_register_global));
                    engine.set_miri_interpcx_wrapper(miri as *mut _ as *mut MiriInterpCxOpaque);
                    n.engine.replace(engine);
                    Ok(())
                })
                .unwrap();
                nllm
            }
            Err(err) => {
                panic!("Unable to initialize interpreter: {}", err);
            }
        }
    }
}

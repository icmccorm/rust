extern crate either;
use super::llvm::lli::LLVM_INTERPRETER;
use crate::concurrency::thread::EvalContextExt as ThreadEvalContextExt;
use crate::shims::llvm::helpers::EvalContextExt as LLVMHelperEvalExt;
use crate::shims::llvm::threads::link::ThreadLinkDestination;
use crate::{shims::llvm::convert::to_generic_value::EvalContextExt as GenValEvalExt, Provenance};
use crate::{throw_interop_format, ThreadId};
use either::Either;
use inkwell::{
    attributes::{Attribute, AttributeLoc},
    types::BasicTypeEnum,
    values::{FunctionValue, GenericValueRef},
};
use log::debug;
use rustc_const_eval::interpret::{InterpResult, MemoryKind, OpTy, PlaceTy};
use rustc_span::Symbol;

macro_rules! throw_llvm_argument_mismatch {
    ($function: expr, $args: expr) => {
        throw_interop_format!(
            "argument count mismatch: LLVM function {:?} {} expects {}{} arguments, but Rust provided {}",
            $function.get_name(),
            $function.get_type().print_to_string().to_string(),
            (if $function.get_type().is_var_arg() { "at least " } else { "" }),
            $function.get_params().len(),
            $args
        )
    };
}

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
                    self.call_external_llvm_and_store_return(function, args, dest)?;
                }
                Ok(result)
            });
        }
        Ok(false)
    }

    fn terminate_lli_thread(&self, id: ThreadId) {
        if let Some(ll) = LLVM_INTERPRETER.lock().borrow_mut().as_mut() {
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

        let mut generic_args = vec![];
        let (llvm_parameter_types, thread_link_destination): (
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
            let converted = this.op_to_generic_value(&op, *head)?;
            generic_args.push(converted);
            (tail, alloc)
        } else {
            (llvm_parameter_types.as_slice(), ThreadLinkDestination::ToMiriDefault(dest.clone()))
        };

        if args.len() > llvm_parameter_types.len() {
            throw_llvm_argument_mismatch!(function, args.len());
        }

        let mut llvm_parameter_types = llvm_parameter_types.to_vec();
        llvm_parameter_types.reverse();

        let mut args = args.to_vec();
        args.reverse();

        let scalar_pair_expansion = args.len() < llvm_parameter_types.len();

        debug!("link={}, sret={:?}", thread_link_destination, has_sret);

        let last_type = llvm_parameter_types.first().copied();
        let original_arg_length = args.len();
        while let Some(current_arg) = args.pop() {
            if llvm_parameter_types.is_empty() {
                if function.get_type().is_var_arg() {
                    llvm_parameter_types.push(last_type.unwrap());
                } else {
                    throw_llvm_argument_mismatch!(function, original_arg_length);
                }
            }
            let num_types_remaining = llvm_parameter_types.len() - 1;
            let num_args_remaining = args.len();
            if scalar_pair_expansion
                && num_types_remaining >= 2
                && num_types_remaining > num_args_remaining
                && (*llvm_parameter_types.last().unwrap()
                    == llvm_parameter_types[llvm_parameter_types.len() - 1])
            {
                if let Some((first, second)) =
                    this.resolve_scalar_pair(&current_arg, llvm_parameter_types.last().unwrap())?
                {
                    args.push(second);
                    args.push(first);
                    continue;
                }
            }
            let first_type = llvm_parameter_types.pop().unwrap();
            generic_args.push(this.op_to_generic_value(&current_arg, first_type)?);
        }

        let exposed = generic_args
            .drain(0..)
            .map(|gv| GenericValueRef::new(unsafe { gv.into_raw() }))
            .collect::<Vec<_>>();
        let lli_thread_id =
            this.start_rust_to_lli_thread(thread_link_destination, function, exposed)?;
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

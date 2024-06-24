use crate::concurrency::thread::EvalContextExt as _;
use crate::shims::llvm::convert::EvalContextExt as _;
use crate::shims::llvm::helpers::EvalContextExt;
use crate::shims::llvm::EvalContextExt as _;
use crate::*;
use inkwell::{
    types::BasicTypeEnum,
    values::{GenericValue, GenericValueRef},
};
use llvm_sys::prelude::LLVMTypeRef;
use rustc_const_eval::interpret::{InterpResult, MemoryKind};
use tracing::debug;

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum ThreadLinkDestination<'tcx> {
    ToLLI(Option<BasicTypeEnum<'static>>),
    ToMiriStructReturn,
    ToMiriMirroredLocal(MPlaceTy<'tcx>, PlaceTy<'tcx>),
    ToMiriDefault(PlaceTy<'tcx>),
}

impl std::fmt::Display for ThreadLinkDestination<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThreadLinkDestination::ToLLI(Some(ty)) => {
                write!(f, "LLI ({})", ty.print_to_string().to_string())
            }
            ThreadLinkDestination::ToLLI(None) => write!(f, "LLI (void)"),
            ThreadLinkDestination::ToMiriStructReturn => write!(f, "Miri (sret)"),
            ThreadLinkDestination::ToMiriMirroredLocal(_, _) => write!(f, "Miri (Mirrored Local)"),
            ThreadLinkDestination::ToMiriDefault(_) => write!(f, "Miri (Default)"),
        }
    }
}

#[derive(Debug)]
pub enum ThreadLinkSource<'tcx> {
    FromMiri(MPlaceTy<'tcx>),
    FromLLI(Option<LLVMTypeRef>),
}

#[derive(Debug)]
pub struct ThreadLink<'tcx> {
    linked_id: ThreadId,
    id: ThreadId,
    link: ThreadLinkDestination<'tcx>,
    source: ThreadLinkSource<'tcx>,
    lli_allocations: Vec<MPlaceTy<'tcx>>,
}
impl<'tcx> ThreadLink<'tcx> {
    pub fn new(
        linked_id: ThreadId,
        id: ThreadId,
        link: ThreadLinkDestination<'tcx>,
        source: ThreadLinkSource<'tcx>,
    ) -> Self {
        Self { linked_id, id, link, source, lli_allocations: Vec::new() }
    }

    pub fn take_ownership(&mut self, to_deallocate: MPlaceTy<'tcx>) {
        self.lli_allocations.push(to_deallocate)
    }

    pub fn finalize(mut self, ctx: &mut MiriInterpCx<'tcx>) -> InterpResult<'tcx> {
        let this = ctx.eval_context_mut();
        debug!("[ThreadLink] Finalizing, TID:{:?}", self.id);
        let curr_exit_code = this.active_thread_mut().last_error.clone();
        let prev = this.machine.threads.set_active_thread_id(self.linked_id);
        this.active_thread_mut().last_error = curr_exit_code;
        this.machine.threads.set_active_thread_id(prev);

        match self.link {
            ThreadLinkDestination::ToLLI(dest_opt) =>
                match self.source {
                    ThreadLinkSource::FromMiri(place_src) => {
                        let return_value = if let Some(dest) = dest_opt {
                            let as_place = PlaceTy::from(place_src.clone());
                            this.perform_opty_conversion(&as_place, Some(dest))?
                        } else {
                            GenericValue::new_void()
                        };
                        this.set_pending_return_value(self.linked_id, unsafe {
                            GenericValueRef::new(return_value.into_raw())
                        });
                        this.deallocate_ptr(
                            place_src.ptr(),
                            Some((place_src.layout.size, place_src.layout.layout.align().abi)),
                            MemoryKind::Machine(crate::MiriMemoryKind::LLVMInterop),
                        )?;
                    }
                    ThreadLinkSource::FromLLI(_) => {
                        todo!("Threads within LLI aren't supported yet!")
                    }
                },
            ThreadLinkDestination::ToMiriMirroredLocal(mirror, local) => {
                let prev = this.machine.threads.set_active_thread_id(self.linked_id);
                let mirrored_place = PlaceTy::from(mirror.clone());
                let result_op = this.place_to_op(&mirrored_place)?;
                debug!("[ThreadLink] Copying mirrored Miri place into local.");

                this.copy_op(&result_op, &local)?;

                this.deallocate_ptr(
                    mirror.ptr(),
                    Some((mirror.layout.size, mirror.layout.layout.align().abi)),
                    MemoryKind::Machine(crate::MiriMemoryKind::LLVMInterop),
                )?;
                this.machine.threads.set_active_thread_id(prev);
            }
            ThreadLinkDestination::ToMiriDefault(place) =>
                if let ThreadLinkSource::FromLLI(gvty_opt) = self.source {
                    if let Some(return_type) = gvty_opt {
                        debug!("[ThreadLink] Writing generic value to Miri destination.");
                        let prev: ThreadId =
                            this.machine.threads.set_active_thread_id(self.linked_id);
                        let return_ref_opt = this.get_thread_exit_value(self.id)?;
                        if let Some(return_ref) = return_ref_opt {
                            return_ref.set_type_tag(unsafe { &BasicTypeEnum::new(return_type) });
                            this.write_generic_value(return_ref, place)?;
                            this.machine.threads.set_active_thread_id(prev);
                        } else {
                            bug!("Unable to resolve return value for thread {:?}.", self.id);
                        }
                    }
                } else {
                    bug!("Expected an LLVM source value.");
                },
            ThreadLinkDestination::ToMiriStructReturn => {}
        }
        self.lli_allocations.drain(0..).try_for_each(|mp| {
            this.deallocate_ptr(
                mp.ptr(),
                Some((mp.layout.size, mp.layout.layout.align().abi)),
                MemoryKind::Machine(crate::MiriMemoryKind::LLVMInterop),
            )
        })?;
        Ok(())
    }
}

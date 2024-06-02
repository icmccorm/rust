use crate::concurrency::thread::EvalContextExt as ConcurrencyExt;
use crate::shims::llvm::convert::to_opty::EvalContextExt as ToOpTyExt;
use crate::shims::llvm::helpers::EvalContextExt;
use crate::shims::llvm_ffi_support::EvalContextExt as FFIEvalContextExt;
use crate::MiriInterpCx;
use crate::ThreadId;
use inkwell::types::BasicTypeEnum;
use inkwell::values::GenericValue;
use inkwell::values::GenericValueRef;
use llvm_sys::prelude::LLVMTypeRef;
use rustc_const_eval::interpret::InterpResult;
use rustc_const_eval::interpret::MPlaceTy;
use rustc_const_eval::interpret::MemoryKind;
use rustc_const_eval::interpret::PlaceTy;

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum ThreadLinkDestination<'tcx> {
    ToLLI(Option<BasicTypeEnum<'static>>),
    ToMiriStructReturn,
    ToMiriMirroredLocal(MPlaceTy<'tcx, crate::Provenance>, PlaceTy<'tcx, crate::Provenance>),
    ToMiriDefault(PlaceTy<'tcx, crate::Provenance>),
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
    FromMiri(MPlaceTy<'tcx, crate::Provenance>),
    FromLLI(Option<LLVMTypeRef>),
}

#[derive(Debug)]
pub struct ThreadLink<'tcx> {
    linked_id: ThreadId,
    id: ThreadId,
    link: ThreadLinkDestination<'tcx>,
    source: ThreadLinkSource<'tcx>,
    lli_allocations: Vec<MPlaceTy<'tcx, crate::Provenance>>,
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

    pub fn take_ownership(&mut self, to_deallocate: MPlaceTy<'tcx, crate::Provenance>) {
        self.lli_allocations.push(to_deallocate)
    }

    pub fn finalize(mut self, ctx: &mut MiriInterpCx<'_, 'tcx>) -> InterpResult<'tcx> {
        debug!("[ThreadLink] Finalizing, TID:{:?}", self.id);
        let curr_exit_code = ctx.active_thread_mut().last_error.clone();
        let prev = ctx.set_active_thread(self.linked_id);
        ctx.active_thread_mut().last_error = curr_exit_code;
        ctx.set_active_thread(prev);

        match self.link {
            ThreadLinkDestination::ToLLI(dest_opt) =>
                match self.source {
                    ThreadLinkSource::FromMiri(place_src) => {
                        let return_value = if let Some(dest) = dest_opt {
                            let as_place = PlaceTy::from(place_src.clone());
                            ctx.perform_opty_conversion(&as_place, Some(dest))?
                        } else {
                            GenericValue::new_void()
                        };
                        ctx.set_pending_return_value(self.linked_id, unsafe {
                            GenericValueRef::new(return_value.into_raw())
                        });
                        ctx.deallocate_ptr(
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
                let prev = ctx.set_active_thread(self.linked_id);
                let mirrored_place = PlaceTy::from(mirror.clone());
                let result_op = ctx.place_to_op(&mirrored_place)?;
                debug!("[ThreadLink] Copying mirrored Miri place into local.");

                ctx.copy_op(&result_op, &local)?;

                ctx.deallocate_ptr(
                    mirror.ptr(),
                    Some((mirror.layout.size, mirror.layout.layout.align().abi)),
                    MemoryKind::Machine(crate::MiriMemoryKind::LLVMInterop),
                )?;
                ctx.set_active_thread(prev);
            }
            ThreadLinkDestination::ToMiriDefault(place) =>
                if let ThreadLinkSource::FromLLI(gvty_opt) = self.source {
                    if let Some(return_type) = gvty_opt {
                        debug!("[ThreadLink] Writing generic value to Miri destination.");
                        let prev: ThreadId = ctx.set_active_thread(self.linked_id);
                        let return_ref_opt = ctx.get_thread_exit_value(self.id)?;
                        if let Some(return_ref) = return_ref_opt {
                            return_ref.set_type_tag(unsafe { &BasicTypeEnum::new(return_type) });
                            ctx.write_generic_value(return_ref, place)?;
                            ctx.set_active_thread(prev);
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
            ctx.deallocate_ptr(
                mp.ptr(),
                Some((mp.layout.size, mp.layout.layout.align().abi)),
                MemoryKind::Machine(crate::MiriMemoryKind::LLVMInterop),
            )
        })?;
        Ok(())
    }
}

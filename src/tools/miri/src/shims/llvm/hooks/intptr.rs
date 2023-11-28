use super::memory::obtain_ctx_mut;
use crate::machine::MiriInterpCxExt;
use crate::rustc_const_eval::interpret::AllocMap;
use crate::InterpResult;
use crate::MiriInterpCx;
use crate::MiriMemoryKind;
use crate::Pointer;
use crate::Provenance::*;
use crate::{intptrcast, shims::llvm::helpers::EvalContextExt, BorTag};
use llvm_sys::miri::{MiriInterpCxOpaque, MiriPointer, MiriProvenance};
use log::debug;
use rustc_const_eval::interpret::AllocId;
use rustc_target::abi::Size;
use std::num::NonZeroU64;

pub extern "C-unwind" fn miri_ptrtoint(ctx_raw: *mut MiriInterpCxOpaque, mp: MiriPointer) -> u64 {
    let ctx = obtain_ctx_mut(ctx_raw);
    let addr_to_return = if mp.prov.alloc_id == 0 {
        if mp.addr == 0 { 0 } else { mp.addr }
    } else {
        let alloc_id = AllocId(NonZeroU64::new(mp.prov.alloc_id).unwrap());
        let borrow_tag = BorTag::new(mp.prov.tag).unwrap();
        if let Err(e) = intptrcast::GlobalStateInner::expose_ptr(ctx, alloc_id, borrow_tag) {
            debug!("Invalid pointer to int conversion occurred.");
            ctx.set_foreign_error(e);
            0
        } else {
            mp.addr
        }
    };
    if let Some(ref logger) = &ctx.machine.llvm_logger {
        logger.flags.log_ptrtoint_llvm();
    }
    addr_to_return
}

pub extern "C-unwind" fn miri_inttoptr(ctx_raw: *mut MiriInterpCxOpaque, addr: u64) -> MiriPointer {
    let ctx = obtain_ctx_mut(ctx_raw);

    let as_ptr = intptrcast::GlobalStateInner::ptr_from_addr_cast(ctx, addr);

    let as_miri_ptr = match as_ptr {
        Ok(p) => ctx.pointer_to_lli_wrapped_pointer(p),
        Err(e) => {
            ctx.set_foreign_error(e);
            MiriPointer { addr: 0, prov: MiriProvenance { alloc_id: 0, tag: 0 } }
        }
    };
    if let Some(ref logger) = &ctx.machine.llvm_logger {
        logger.flags.log_inttoptr_llvm();
    }
    as_miri_ptr
}

pub extern "C-unwind" fn miri_get_element_pointer(
    ctx_raw: *mut MiriInterpCxOpaque,
    base_ptr: MiriPointer,
    offset: u64,
) -> MiriPointer {
    let ctx = obtain_ctx_mut(ctx_raw);
    let ptr = ctx.lli_wrapped_pointer_to_maybe_pointer(base_ptr);
    let offset = Size::from_bytes(offset);
    match miri_get_element_pointer_result(ctx, ptr, offset) {
        Ok(ptr) => ctx.pointer_to_lli_wrapped_pointer(ptr),
        Err(e) => {
            ctx.set_foreign_error(e);
            MiriPointer { addr: 0, prov: MiriProvenance { alloc_id: 0, tag: 0 } }
        }
    }
}

fn miri_get_element_pointer_result<'tcx>(
    ctx: &mut MiriInterpCx<'_, 'tcx>,
    base_ptr: Pointer<Option<crate::Provenance>>,
    offset: Size,
) -> InterpResult<'tcx, Pointer<Option<crate::Provenance>>> {
    let this = ctx.eval_context_mut();
    let (ptr_offset, _) = base_ptr.overflowing_offset(offset, this);
    if let Some(Concrete { alloc_id, .. }) = base_ptr.provenance {
        let (size, _, _) = this.get_alloc_info(alloc_id);
        let base_address = intptrcast::GlobalStateInner::alloc_base_addr(this, alloc_id)?;
        let addr_upper_bound = base_address.checked_add(size.bytes());
        if let Some(addr_upper_bound) = addr_upper_bound {
            if ptr_offset.addr().bytes() >= addr_upper_bound {
                let provenance = intptrcast::GlobalStateInner::alloc_id_from_addr(
                    this,
                    ptr_offset.addr().bytes(),
                )
                .and_then(|offset_alloc_id| this.memory.alloc_map().get(offset_alloc_id))
                .and_then(|(kind, _)| {
                    match kind {
                        rustc_const_eval::interpret::MemoryKind::Machine(
                            MiriMemoryKind::LLVMStack | MiriMemoryKind::LLVMStatic,
                        ) => None,
                        _ => Some(Wildcard),
                    }
                });
                return Ok(Pointer::new(provenance, ptr_offset.addr()));
            }
        }
    }
    Ok(ptr_offset)
}

use super::memory::obtain_ctx_mut;
use crate::{intptrcast, shims::llvm::helpers::EvalContextExt, BorTag};
use llvm_sys::miri::{MiriInterpCxOpaque, MiriPointer, MiriProvenance};
use log::debug;
use rustc_const_eval::interpret::AllocId;
use std::num::NonZeroU64;

pub extern "C-unwind" fn miri_ptrtoint(ctx_raw: *mut MiriInterpCxOpaque, mp: MiriPointer) -> u64 {
    debug!("[ptrtoint] AID: {}, addr: {}", mp.prov.alloc_id, mp.addr);
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
    debug!("[inttoptr] AID: {}, addr: {}", as_miri_ptr.prov.alloc_id, addr);
    if let Some(ref logger) = &ctx.machine.llvm_logger {
        logger.flags.log_inttoptr_llvm();
    }
    as_miri_ptr
}

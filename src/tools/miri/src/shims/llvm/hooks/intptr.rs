use super::memory::obtain_ctx_mut;
use crate::shims::llvm::logging::LLVMFlag;
use crate::Provenance::*;
use crate::*;
use crate::shims::llvm::helpers::EvalContextExt as _;
use crate::alloc_addresses::EvalContextExt as _;
use llvm_sys::miri::{MiriInterpCxOpaque, MiriPointer, MiriProvenance};
use rustc_target::abi::Size;
use std::num::NonZeroU64;
use tracing::debug;
use crate::Pointer;

pub extern "C-unwind" fn miri_ptrtoint(ctx_raw: *mut MiriInterpCxOpaque, mp: MiriPointer) -> u64 {
    let ctx = obtain_ctx_mut(ctx_raw);
    let addr_to_return = if mp.prov.alloc_id == 0 {
        if mp.addr == 0 { 0 } else { mp.addr }
    } else {
        let alloc_id = AllocId(NonZeroU64::new(mp.prov.alloc_id).unwrap());
        let borrow_tag = BorTag::new(mp.prov.tag).unwrap();
        if let Err(e) = ctx.expose_ptr(alloc_id, borrow_tag) {
            debug!("Invalid pointer to int conversion occurred.");
            ctx.set_foreign_error(e);
            0
        } else {
            mp.addr
        }
    };
    if let Some(ref logger) = &ctx.machine.llvm_logger {
        logger.log_flag(LLVMFlag::LLVMPtrToInt);
    }
    addr_to_return
}

pub extern "C-unwind" fn miri_inttoptr(ctx_raw: *mut MiriInterpCxOpaque, addr: u64) -> MiriPointer {
    let ctx = obtain_ctx_mut(ctx_raw);

    let as_ptr = ctx.ptr_from_addr_cast(addr);

    let as_miri_ptr = match as_ptr {
        Ok(p) => ctx.pointer_to_lli_wrapped_pointer(p),
        Err(e) => {
            ctx.set_foreign_error(e);
            MiriPointer { addr: 0, prov: MiriProvenance { alloc_id: 0, tag: 0 } }
        }
    };
    if let Some(ref logger) = &ctx.machine.llvm_logger {
        logger.log_flag(LLVMFlag::LLVMIntToPtr);
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
    ctx: &mut MiriInterpCx<'tcx>,
    base_ptr: Pointer,
    offset: Size,
) -> InterpResult<'tcx, Pointer> {
    let this = ctx.eval_context_mut();
    let (ptr_offset, _) = base_ptr.overflowing_offset(offset, this);

    let source_alloc_id = base_ptr.provenance.map_or(
        this.alloc_id_from_addr(base_ptr.addr().bytes()),
        |p| p.get_alloc_id(),
    );
    let source_allocation =
        source_alloc_id.map(|alloc_id| this.memory.alloc_map().get(alloc_id)).flatten();
    let source_allocation_kind = source_allocation.map(|s| s.0);

    let dest_alloc_id =
        this.alloc_id_from_addr(ptr_offset.addr().bytes());
    let dest_allocation =
        dest_alloc_id.map(|alloc_id| this.memory.alloc_map().get(alloc_id)).flatten();
    let dest_allocation_kind = dest_allocation.map(|d| d.0);

    if let Some(Concrete { alloc_id, tag }) = base_ptr.provenance {
        let (size, _, _) = this.get_alloc_info(alloc_id);
        let base_address = this.addr_from_alloc_id(alloc_id, source_allocation_kind.unwrap())?;
        let addr_upper_bound = base_address.checked_add(size.bytes());
        if source_allocation.is_some() {
            if let Some(addr_upper_bound) = addr_upper_bound {
                if ptr_offset.addr().bytes() >= addr_upper_bound {
                    this.expose_ptr(alloc_id, tag)?;
                    return Ok(Pointer::new(Some(Wildcard), ptr_offset.addr()));
                }
            }
        }
    }
    debug!(
        "GEP - {:?}({:?}) to {:?}",
        source_allocation_kind, base_ptr.provenance, dest_allocation_kind
    );
    Ok(ptr_offset)
}

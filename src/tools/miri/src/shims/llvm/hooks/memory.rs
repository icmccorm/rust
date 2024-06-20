use crate::eval::ForeignMemoryMode;
use crate::shims::llvm::helpers::EvalContextExt as _;
use crate::*;
use inkwell::execution_engine::{MiriInterpCxOpaque, MiriPointer};
use llvm_sys::miri::MiriProvenance;
use rustc_const_eval::interpret::InterpResult;
use rustc_span::Symbol;
use rustc_target::abi::{Align, Size};
use std::{ffi::CString, iter, slice};
use tracing::debug;

pub fn obtain_ctx_mut(ctx_raw: *mut MiriInterpCxOpaque) -> &'static mut MiriInterpCx<'static> {
    debug_assert!(!ctx_raw.is_null(), "interpretation context is null");
    unsafe { (ctx_raw as *mut MiriInterpCx<'_>).as_mut() }.unwrap()
}

fn llvm_malloc_result<'tcx>(
    ctx: &mut MiriInterpCx<'tcx>,
    size: u64,
    alignment: u64,
    is_static: bool,
) -> InterpResult<'tcx, MiriPointer> {
    let this = ctx.eval_context_mut();

    let (kind, zero_init) = if is_static {
        (MiriMemoryKind::LLVMStatic, is_static)
    } else {
        (
            MiriMemoryKind::LLVMStack,
            matches!(this.machine.lli_config.memory_mode, ForeignMemoryMode::Zeroed),
        )
    };
    let align = Align::from_bytes(alignment).unwrap();
    let ptr = this.allocate_ptr(Size::from_bytes(size), align, kind.into())?;
    if zero_init {
        // We just allocated this, the access is definitely in-bounds and fits into our address space.
        this.write_bytes_ptr(ptr.into(), iter::repeat(0u8).take(usize::try_from(size).unwrap()))
            .unwrap();
    }
    if is_static {
        this.machine.static_roots.push(ptr.provenance.get_alloc_id().unwrap());
    }
    Ok(this.pointer_to_lli_wrapped_pointer(ptr.into()))
}

pub extern "C-unwind" fn llvm_malloc(
    ctx_raw: *mut MiriInterpCxOpaque,
    size: u64,
    align: u64,
    is_static: bool,
) -> MiriPointer {
    let this = obtain_ctx_mut(ctx_raw);
    match llvm_malloc_result(this, size, align, is_static) {
        Ok(ptr) => ptr,
        Err(e) => {
            this.set_foreign_error(e);
            MiriPointer { addr: 0, prov: MiriProvenance { alloc_id: 0, tag: 0 } }
        }
    }
}

fn llvm_free_result<'tcx>(ctx: &mut MiriInterpCx<'tcx>, ptr: MiriPointer) -> InterpResult<'tcx> {
    let this = ctx.eval_context_mut();
    let resolved = this.lli_wrapped_pointer_to_resolved_pointer(ptr)?;
    if !this.ptr_is_null(resolved.ptr)? {
        this.deallocate_ptr(resolved.ptr, None, MiriMemoryKind::LLVMStack.into())
    } else {
        Ok(())
    }
}

pub extern "C-unwind" fn llvm_free(ctx_raw: *mut MiriInterpCxOpaque, ptr: MiriPointer) -> bool {
    let ctx = obtain_ctx_mut(ctx_raw);
    let result = llvm_free_result(ctx, ptr);
    let status = result.is_err();
    if let Err(e) = result {
        ctx.set_foreign_error(e);
    };
    status
}

fn miri_memcpy_result<'tcx>(
    ctx: &mut MiriInterpCx<'tcx>,
    dest: MiriPointer,
    src_ptr: *const u8,
    src_len: u64,
) -> InterpResult<'tcx> {
    let dest_pointer = ctx.lli_wrapped_pointer_to_resolved_pointer(dest)?;
    let slice = unsafe { slice::from_raw_parts(src_ptr, usize::try_from(src_len).unwrap()) };
    ctx.write_bytes_ptr(dest_pointer.ptr.into(), slice.iter().copied())?;
    Ok(())
}

pub extern "C-unwind" fn miri_memcpy(
    ctx_raw: *mut MiriInterpCxOpaque,
    dest: MiriPointer,
    src_ptr: *const u8,
    src_len: u64,
) -> bool {
    let ctx = obtain_ctx_mut(ctx_raw);
    let result = miri_memcpy_result(ctx, dest, src_ptr, src_len);
    let status = result.is_err();
    if let Err(e) = result {
        ctx.set_foreign_error(e)
    }
    status
}

fn miri_memset_result<'tcx>(
    ctx: &mut MiriInterpCx<'tcx>,
    dest: MiriPointer,
    val: u8,
    len: u64,
) -> InterpResult<'tcx, ()> {
    let dest_pointer = ctx.lli_wrapped_pointer_to_resolved_pointer(dest)?;
    ctx.write_bytes_ptr(
        dest_pointer.ptr.into(),
        iter::repeat(val).take(usize::try_from(len).unwrap()),
    )?;
    Ok(())
}

pub extern "C-unwind" fn miri_memset(
    ctx_raw: *mut MiriInterpCxOpaque,
    dest: MiriPointer,
    val: u8,
    len: u64,
) -> bool {
    let ctx = obtain_ctx_mut(ctx_raw);
    let result = miri_memset_result(ctx, dest, val, len);
    let status = result.is_err();
    if let Err(e) = result {
        ctx.set_foreign_error(e);
    }
    status
}

pub extern "C-unwind" fn miri_register_global(
    ctx_raw: *mut MiriInterpCxOpaque,
    name: *const ::libc::c_char,
    name_length: u64,
    mp: MiriPointer,
) -> bool {
    let ctx = obtain_ctx_mut(ctx_raw);
    let name_slice: &[u8] = unsafe {
        std::slice::from_raw_parts(name as *const u8, usize::try_from(name_length).unwrap())
    };
    let name_c_str = CString::new(name_slice).unwrap();
    let name_rust_str = name_c_str.to_str().unwrap();
    let result = miri_register_global_result(ctx, name_rust_str, mp);
    let status = result.is_err();
    if let Err(e) = result {
        ctx.set_foreign_error(e);
    }
    status
}

fn miri_register_global_result<'tcx>(
    ctx: &mut MiriInterpCx<'tcx>,
    name: &str,
    mp: MiriPointer,
) -> InterpResult<'tcx, ()> {
    let pointer = ctx.lli_wrapped_pointer_to_resolved_pointer(mp)?;
    if let Some(valid_pointer) = pointer.with_provenance() {
        debug!("[LLVM Extern Static] Registering {} - {:?}", name, pointer.ptr);

        if ctx.machine.extern_statics.try_insert(Symbol::intern(name), valid_pointer).is_err() {
            throw_unsup_format!(
                "Unable to register LLVM global {} because it's already registered by Miri.",
                name
            )
        }
        Ok(())
    } else {
        panic!(
            "Attempted to register a pointer for static variable {name}, but it had invalid provenance."
        )
    }
}

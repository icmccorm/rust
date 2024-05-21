use crate::eval::ForeignMemoryMode;
use crate::shims::foreign_items::EvalContextExt as ForeignEvalContextExt;
use crate::shims::llvm::helpers::EvalContextExt;
use crate::{MiriInterpCx, MiriMemoryKind};
use inkwell::execution_engine::{MiriInterpCxOpaque, MiriPointer};
use rustc_const_eval::interpret::InterpResult;
use rustc_span::Symbol;
use rustc_target::abi::Align;
use std::ffi::CString;
use std::iter::repeat;
use std::panic::panic_any;

pub fn obtain_ctx_mut(
    ctx_raw: *mut MiriInterpCxOpaque,
) -> &'static mut MiriInterpCx<'static, 'static> {
    debug_assert!(!ctx_raw.is_null(), "interpretation context is null");
    unsafe { (ctx_raw as *mut MiriInterpCx<'_, '_>).as_mut() }.unwrap()
}

pub extern "C-unwind" fn llvm_malloc(
    ctx_raw: *mut MiriInterpCxOpaque,
    bytes: u64,
    alignment: u64,
    is_static: bool,
) -> MiriPointer {
    let ctx = obtain_ctx_mut(ctx_raw);
    let (kind, zero) = if is_static {
        (MiriMemoryKind::LLVMStatic, is_static)
    } else {
        (MiriMemoryKind::LLVMStack, matches!(ctx.machine.lli_config.memory_mode, ForeignMemoryMode::Zeroed))
    };
    let allocation = ctx.malloc_align(bytes, Align::from_bytes(alignment).unwrap(), zero, kind);
    if let Ok(ptr) = allocation {
        if let Some(crate::Provenance::Concrete { alloc_id, .. }) = ptr.provenance {
            if is_static {
                ctx.machine.static_roots.push(alloc_id);
            }
        }
        ctx.pointer_to_lli_wrapped_pointer(ptr)
    } else {
        panic_any("malloc failed");
    }
}

fn llvm_free_result<'tcx>(
    ctx: &mut MiriInterpCx<'_, 'tcx>,
    ptr: MiriPointer,
) -> InterpResult<'tcx, ()> {
    let pointer = ctx.lli_wrapped_pointer_to_resolved_pointer(ptr)?;
    ctx.free(pointer.ptr, MiriMemoryKind::LLVMStack)?;
    Ok(())
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
    ctx: &mut MiriInterpCx<'_, 'tcx>,
    dest: MiriPointer,
    src_ptr: *const u8,
    src_len: u64,
) -> InterpResult<'tcx> {
    let dest_pointer = ctx.lli_wrapped_pointer_to_resolved_pointer(dest)?;
    let slice = unsafe { std::slice::from_raw_parts(src_ptr, usize::try_from(src_len).unwrap()) };
    ctx.write_bytes_ptr(dest_pointer.ptr, slice.iter().copied())?;
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
    ctx: &mut MiriInterpCx<'_, 'tcx>,
    dest: MiriPointer,
    val: u8,
    len: u64,
) -> InterpResult<'tcx, ()> {
    let dest_pointer = ctx.lli_wrapped_pointer_to_resolved_pointer(dest)?;
    ctx.write_bytes_ptr(dest_pointer.ptr, repeat(val).take(usize::try_from(len).unwrap()))?;
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
    ctx: &mut MiriInterpCx<'_, 'tcx>,
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

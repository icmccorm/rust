extern crate rustc_abi;
use crate::helpers::EvalContextExt;
use crate::shims::llvm::helpers::EvalContextExt as HelperEvalExt;
use crate::MiriInterpCx;
use inkwell::values::GenericValue;
use rustc_abi::Size;
use rustc_const_eval::interpret::InterpResult;
use crate::*;

pub enum MemcpyMode {
    Disjoint,
    Overlapping,
    DisjointOrEqual,
}

#[allow(clippy::arithmetic_side_effects)]
pub fn eval_memcpy<'tcx>(
    ctx: &mut MiriInterpCx<'tcx>,
    function: &str,
    dest: &OpTy<'tcx>,
    source: &OpTy<'tcx>,
    source_length: Option<&OpTy<'tcx>>,
    mode: MemcpyMode,
) -> InterpResult<'tcx, GenericValue<'tcx>> {
    let dest = ctx.opty_as_scalar(dest)?;
    let dest_as_pointer = dest.to_pointer(ctx)?;

    let src = ctx.opty_as_scalar(source)?;
    let src_as_pointer = src.to_pointer(ctx)?;

    let length_value = if let Some(len) = source_length {
        let len_as_scalar = ctx.opty_as_scalar(len)?;
        len_as_scalar.to_target_usize(ctx)?
    } else {
        (ctx.read_c_str(src_as_pointer)?.len() + 1).try_into().unwrap()
    };
    let (src_prov, src_addr) = src_as_pointer.into_parts();
    let (dest_prov, dest_addr) = dest_as_pointer.into_parts();
    let addr_equal = src_addr == dest_addr;
    let provenance_equal = if let (Some(src_prov), Some(dest_prov)) = (src_prov, dest_prov) {
        match src_prov {
            crate::Provenance::Concrete { alloc_id, tag } => {
                if let crate::Provenance::Concrete { alloc_id: dest_alloc_id, tag: dest_tag } =
                    dest_prov
                {
                    alloc_id == dest_alloc_id && tag == dest_tag
                } else {
                    false
                }
            }
            crate::Provenance::Wildcard => {
                matches!(dest_prov, crate::Provenance::Wildcard)
            }
        }
    } else {
        dest_prov.is_none() && src_prov.is_none()
    };

    let nonoverlapping = matches!(mode, MemcpyMode::Disjoint | MemcpyMode::DisjointOrEqual);
    if !(addr_equal && provenance_equal && matches!(mode, MemcpyMode::DisjointOrEqual)) {
        if matches!(mode, MemcpyMode::DisjointOrEqual) && addr_equal && !provenance_equal {
            throw_interop_format!(
                "invalid memory copy with {}: cannot copy equal pointers with different provenance.",
                function
            )
        }
        ctx.mem_copy(
            src_as_pointer,
            dest_as_pointer,
            Size::from_bytes(length_value),
            nonoverlapping,
        )?;
    }
    let as_miri_ptr = ctx.pointer_to_lli_wrapped_pointer(dest_as_pointer);
    let as_gv = unsafe { GenericValue::create_generic_value_of_miri_pointer(as_miri_ptr) };
    Ok(as_gv)
}

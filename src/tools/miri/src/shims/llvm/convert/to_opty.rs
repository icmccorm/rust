extern crate itertools;
extern crate rustc_abi;
use crate::shims::llvm::helpers::EvalContextExt as LLVMEvalExt;

use crate::rustc_const_eval::interpret::Projectable;
use crate::shims::llvm::values::generic_value::GenericValueTy;
use crate::{intptrcast, MiriInterpCx, Provenance};
use inkwell::types::BasicTypeEnum;
use itertools::Itertools;
use log::debug;
use rustc_abi::{Endian, FieldIdx, VariantIdx};
use rustc_apfloat::{
    ieee::{Double, Single},
    Float,
};
use rustc_const_eval::interpret::{ImmTy, MemoryKind, OpTy, PlaceTy};
use rustc_const_eval::interpret::{InterpResult, Scalar};
use rustc_middle::{
    mir::{self, AggregateKind},
    ty::{self, layout::TyAndLayout, AdtKind},
};
use rustc_target::abi::FIRST_VARIANT;
use std::cell::Cell;
use std::fmt::Formatter;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}

#[derive(Debug, Clone)]
enum OpTySource {
    Generic(GenericValueTy),
    Bytes(FieldBytes),
}

impl std::fmt::Display for OpTySource {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            OpTySource::Generic(gv) => write!(f, "{:?}", gv.ty().print_to_string()),
            OpTySource::Bytes(b) => write!(f, "{}", b),
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct FieldBytes {
    bytes: [u8; 16],
    length: usize,
}

impl FieldBytes {
    fn new(bytes: [u8; 16], length: usize) -> Self {
        if length > bytes.len() {
            bug!("Offset must be greater than 0, less than {}", bytes.len());
        }
        FieldBytes { bytes, length }
    }
    fn bytes(&self, endian: Endian) -> &[u8] {
        match endian {
            Endian::Little => &self.bytes[..self.length],
            Endian::Big => &self.bytes[self.length..],
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn len(&self) -> usize {
        self.length
    }

    fn as_uint(&self) -> u128 {
        u128::from_ne_bytes(self.bytes)
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn field<'tcx>(
        &self,
        ctx: &MiriInterpCx<'_, 'tcx>,
        parent: TyAndLayout<'tcx>,
        index: usize,
    ) -> Self {
        let endian = ctx.tcx.data_layout.endian;
        let all_fields = self.bytes(endian);

        let field_offset = parent.fields.offset(index);
        let field_offset = usize::try_from(field_offset.bytes()).unwrap();

        let field_len = parent.field(ctx, index).layout.size().bytes();
        let field_len = usize::try_from(field_len).unwrap();

        let field_end = field_offset + field_len;

        let mut field_bytes = [0u8; 16];

        let subslice = &all_fields[field_offset..field_end];

        match endian {
            Endian::Little => {
                field_bytes[..subslice.len()].copy_from_slice(subslice);
            }
            Endian::Big => {
                field_bytes[self.len()..].copy_from_slice(subslice);
            }
        }

        FieldBytes::new(field_bytes, field_len)
    }
}

impl std::fmt::Display for FieldBytes {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let bytes_as_string = self.bytes.iter().map(|b| format!("{:02x}", b)).join("");
        write!(f, "FieldBytes({})", bytes_as_string)
    }
}

pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn generic_value_to_opty(
        &mut self,
        src: GenericValueTy,
        rust_layout: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, (OpTy<'tcx, Provenance>, Option<PlaceTy<'tcx, Provenance>>)> {
        let this = self.eval_context_mut();
        let mut converter = ConversionContext::new(src, rust_layout);
        convert_to_opty(this, &mut converter)
    }
    fn write_generic_value(
        &mut self,
        src: GenericValueTy,
        dest: PlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let mut converter = ConversionContext::new_to_place(src, dest);
        convert_to_opty(this, &mut converter)?;
        Ok(())
    }
}
struct ConversionContext<'tcx> {
    pub source: OpTySource,
    pub rust_layout: TyAndLayout<'tcx>,
    destination: Cell<Option<PlaceTy<'tcx, Provenance>>>,
}

impl<'tcx> ConversionContext<'tcx> {
    fn new(source: GenericValueTy, rust_layout: TyAndLayout<'tcx>) -> Self {
        let destination = Cell::new(None);

        let source = match source.ty() {
            BasicTypeEnum::IntType(_) if rust_layout.fields.count() > 0 => {
                let field_bytes = source.val_ref.as_int().to_ne_bytes();
                let field_width: usize = rust_layout.size.bytes().try_into().unwrap();
                let field_bytes = FieldBytes::new(field_bytes, field_width);
                OpTySource::Bytes(field_bytes)
            }
            _ => OpTySource::Generic(source),
        };
        ConversionContext { source, rust_layout, destination }
    }
    fn new_to_place(source: GenericValueTy, destination: PlaceTy<'tcx, Provenance>) -> Self {
        let context = Self::new(source, destination.layout);
        context.destination.set(Some(destination));
        context
    }

    fn new_from_field(source: OpTySource, destination: PlaceTy<'tcx, Provenance>) -> Self {
        ConversionContext {
            source,
            rust_layout: destination.layout,
            destination: Cell::new(Some(destination)),
        }
    }

    fn get_destination(&mut self) -> Option<PlaceTy<'tcx, Provenance>> {
        self.destination.get_mut().clone()
    }

    fn get_or_create_destination(
        &mut self,
        miri: &mut MiriInterpCx<'_, 'tcx>,
    ) -> InterpResult<'tcx, PlaceTy<'tcx, Provenance>> {
        let dest = self.destination.get_mut();
        if let Some(cvp) = dest {
            Ok(cvp.clone())
        } else {
            let fresh_place = PlaceTy::from(miri.allocate(
                self.rust_layout,
                MemoryKind::Machine(crate::MiriMemoryKind::LLVMInterop),
            )?);
            *dest = Some(fresh_place);
            Ok(dest.clone().unwrap())
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn get_discriminant(&self, miri: &MiriInterpCx<'_, 'tcx>) -> u32 {
        if self.rust_layout.fields.count() > 0 {
            match &self.source {
                OpTySource::Generic(gvr) => {
                    let first_field = gvr.val_ref.get_aggregate_value(0).unwrap();
                    return u32::try_from(first_field.as_int()).unwrap();
                }
                OpTySource::Bytes(b) => {
                    let field = b.field(miri, self.rust_layout, 0);
                    return u32::try_from(field.as_uint()).unwrap();
                }
            }
        } else {
            0
        }
    }
    fn get_aggregate_kind(&self, miri: &MiriInterpCx<'_, 'tcx>) -> Option<AggregateKind<'tcx>> {
        let rust_type = self.rust_layout.ty;
        if let ty::Adt(adt_def, sr) = rust_type.kind() {
            match adt_def.adt_kind() {
                AdtKind::Struct =>
                    return Some(AggregateKind::Adt(
                        adt_def.did(),
                        VariantIdx::from_u32(0),
                        sr,
                        None,
                        None,
                    )),
                AdtKind::Union =>
                    return Some(AggregateKind::Adt(
                        adt_def.did(),
                        VariantIdx::from_u32(0),
                        sr,
                        None,
                        Some(FieldIdx::from_u32(0)),
                    )),
                AdtKind::Enum =>
                    return Some(AggregateKind::Adt(
                        adt_def.did(),
                        VariantIdx::from_u32(self.get_discriminant(miri)),
                        sr,
                        None,
                        None,
                    )),
            }
        }
        None
    }
}

fn convert_to_opty<'tcx>(
    miri: &mut MiriInterpCx<'_, 'tcx>,
    ctx: &mut ConversionContext<'tcx>,
) -> InterpResult<'tcx, (OpTy<'tcx, Provenance>, Option<PlaceTy<'tcx, Provenance>>)> {
    debug!("[GV to Op]: {} to {:?}", ctx.source, ctx.rust_layout.ty);
    match ctx.rust_layout.abi {
        rustc_abi::Abi::Scalar(_) => {
            let scalar_op = OpTy::from(convert_to_immty(miri, ctx)?);
            if let Some(existing) = ctx.destination.get_mut() {
                let op_transmuted = scalar_op.transmute(existing.layout, miri)?;
                miri.copy_op(&op_transmuted, existing, false)?;
            }
            Ok((scalar_op, ctx.get_destination()))
        }
        rustc_abi::Abi::ScalarPair(_, _)
        | rustc_abi::Abi::Aggregate { sized: true }
        | rustc_abi::Abi::Vector { .. } => {
            let destination = ctx.get_or_create_destination(miri)?;
            let (variant_dest, active_field_index) = if let Some(agk) = ctx.get_aggregate_kind(miri)
            {
                let (variant_index, active_field_index) = match agk {
                    mir::AggregateKind::Adt(_, variant_index, _, _, active_field_index) =>
                        (variant_index, active_field_index),
                    _ => (FIRST_VARIANT, None),
                };
                let variant_dest = miri.project_downcast(&destination, variant_index)?;
                miri.write_discriminant(variant_index, &variant_dest)?;
                (variant_dest, active_field_index)
            } else {
                (destination.clone(), None)
            };
            let rust_fields = ctx
                .rust_layout
                .fields
                .index_by_increasing_offset()
                .map(|idx| {
                    let idx =
                        if let Some(afidx) = active_field_index { afidx.as_usize() } else { idx };
                    miri.project_field(&variant_dest, idx)
                })
                .collect::<InterpResult<'tcx, Vec<_>>>()?;
            match &ctx.source {
                OpTySource::Generic(generic) => {
                    let mut llvm_fields = generic.resolve_fields();
                    let rust_field_count = ctx.rust_layout.fields.count();
                    if rust_field_count == llvm_fields.len() {
                        for (llvm_field, rust_field) in llvm_fields.drain(0..).zip_eq(rust_fields) {
                            let mut new_context =
                                ConversionContext::new_to_place(llvm_field, rust_field);
                            convert_to_opty(miri, &mut new_context)?;
                        }
                    } else if miri.can_dereference_into_singular_field(&ctx.rust_layout)
                        && llvm_fields.len() > rust_field_count
                    {
                        let field = miri.project_field(&destination, 0)?;
                        let mut new_context =
                            ConversionContext::new_from_field(ctx.source.clone(), field);
                        convert_to_opty(miri, &mut new_context)?;
                    } else {
                        throw_llvm_field_count_mismatch!(llvm_fields.len(), ctx.rust_layout);
                    }
                }
                OpTySource::Bytes(fieldbytes) => {
                    if u64::try_from(fieldbytes.len()).unwrap() < ctx.rust_layout.size.bytes() {
                        throw_llvm_field_width_mismatch!(fieldbytes.len(), ctx.rust_layout);
                    } else {
                        for (idx, rust_field) in rust_fields.iter().enumerate() {
                            let llvm_field = fieldbytes.field(miri, ctx.rust_layout, idx);
                            let mut new_context = ConversionContext::new_from_field(
                                OpTySource::Bytes(llvm_field),
                                rust_field.clone(),
                            );
                            convert_to_opty(miri, &mut new_context)?;
                        }
                    }
                }
            }
            let operand = miri.place_to_op(&destination)?;
            Ok((operand, Some(destination)))
        }
        rustc_abi::Abi::Uninhabited => throw_unsup_abi!("Uninhabited"),
        rustc_abi::Abi::Aggregate { sized: false } => throw_unsup_abi!("unsized Aggregate"),
    }
}

#[allow(clippy::arithmetic_side_effects)]
fn convert_to_immty<'tcx>(
    miri: &MiriInterpCx<'_, 'tcx>,
    ctx: &ConversionContext<'tcx>,
) -> InterpResult<'tcx, ImmTy<'tcx, crate::Provenance>> {
    let truncate_to_pointer_size = |v: u128| -> u64 {
        let as_bytes: [u8; 16] = v.to_ne_bytes();
        let pointer_size = miri.tcx.data_layout.pointer_size;
        match miri.tcx.data_layout.endian {
            Endian::Little => {
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&as_bytes[..pointer_size.bytes().try_into().unwrap()]);
                u64::from_ne_bytes(bytes)
            }
            Endian::Big => {
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&as_bytes[8..]);
                u64::from_ne_bytes(bytes)
            }
        }
    };

    match &ctx.source {
        OpTySource::Generic(generic) => {
            let layouts = &miri.machine.layouts;
            let rust_type = ctx.rust_layout.ty;
            match generic.ty() {
                BasicTypeEnum::FloatType(_) =>
                    if let ty::Float(rft) = rust_type.kind() {
                        match rft {
                            ty::FloatTy::F32 => {
                                let val = generic.val_ref.as_f32();
                                debug!("[GV to Op]: Float value: {:?}", val);
                                let imm = ImmTy::from_scalar(
                                    Scalar::from_f32(Single::from_bits(val.to_bits().into())),
                                    layouts.f32,
                                );
                                return Ok(imm);
                            }
                            ty::FloatTy::F64 => {
                                let val = generic.val_ref.as_f64();
                                debug!("[GV to Op]: Float value: {:?}", val);
                                let imm = ImmTy::from_scalar(
                                    Scalar::from_f64(Double::from_bits(val.to_bits().into())),
                                    layouts.f64,
                                );
                                return Ok(imm);
                            }
                        }
                    },
                BasicTypeEnum::IntType(_) => {
                    let converted_int = generic.val_ref.as_int();
                    debug!("[GV to Op]: Int value: {:?}", converted_int);
                    let scalar = if let ty::RawPtr(_) | ty::FnPtr(_) = ctx.rust_layout.ty.kind() {
                        let first_word = truncate_to_pointer_size(converted_int);
                        let as_maybe_ptr =
                            intptrcast::GlobalStateInner::ptr_from_addr_cast(miri, first_word)?;
                        match as_maybe_ptr.into_pointer_or_addr() {
                            Ok(ptr) => Scalar::from_pointer(ptr, miri),
                            Err(addr) => Scalar::from_uint(addr.bytes(), ctx.rust_layout.size),
                        }
                    } else {
                        Scalar::from_uint(converted_int, ctx.rust_layout.size)
                    };
                    return Ok(ImmTy::from_scalar(scalar, ctx.rust_layout));
                }
                BasicTypeEnum::PointerType(_) => {
                    let wrapped_pointer = generic.val_ref.as_miri_pointer();
                    let mp = miri.lli_wrapped_pointer_to_maybe_pointer(wrapped_pointer);
                    debug!(
                        "[GV to Op]: Provenance: (AID: {}, Addr: {})",
                        wrapped_pointer.prov.alloc_id, wrapped_pointer.addr
                    );
                    // LLVM16 pointers are opaque; we cannot determine a pointee type.
                    // Figuring out the types required for a shim call at this stage would
                    // require a significant rewrite for the shim API. So instead,
                    // we use u8 when provenance is wildcard or undefined and calculate the
                    // maximum size up to a u128 otherwise.
                    let pointer_ty_layout =
                        if let Some(crate::Provenance::Concrete { alloc_id, .. }) = mp.provenance {
                            let (size, _, _) = miri.get_alloc_info(alloc_id);
                            let base =
                                intptrcast::GlobalStateInner::alloc_base_addr(miri, alloc_id)?;
                            let diff = base + size.bytes() - mp.addr().bytes();
                            nearest_pointer_type(miri, diff)?
                        } else {
                            ctx.rust_layout
                        };
                    debug!("Adjusted pointer type: {:?}", pointer_ty_layout.ty);
                    let scalar = Scalar::from_maybe_pointer(mp, miri);
                    let imm = ImmTy::from_scalar(scalar, pointer_ty_layout);
                    return Ok(imm);
                }
                _ => {}
            }
            throw_llvm_type_mismatch!(generic.ty(), rust_type);
        }
        OpTySource::Bytes(fieldbytes) => {
            let value = fieldbytes.as_uint();
            let layout = ctx.rust_layout;
            let scalar = match layout.ty.kind() {
                | ty::FnPtr(_) | ty::RawPtr(_) => {
                    if let ty::RawPtr(_) | ty::FnPtr(_) = ctx.rust_layout.ty.kind() {
                        let first_word = truncate_to_pointer_size(value);
                        let as_maybe_ptr =
                            intptrcast::GlobalStateInner::ptr_from_addr_cast(miri, first_word)?;
                        match as_maybe_ptr.into_pointer_or_addr() {
                            Ok(ptr) => Scalar::from_pointer(ptr, miri),
                            Err(addr) =>
                                Scalar::from_uint(addr.bytes(), miri.tcx.data_layout.pointer_size),
                        }
                    } else {
                        Scalar::from_uint(fieldbytes.as_uint(), layout.size)
                    }
                }
                _ => Scalar::from_uint(fieldbytes.as_uint(), layout.size),
            };
            return Ok(ImmTy::from_scalar(scalar, layout));
        }
    }
}

fn nearest_pointer_type<'tcx>(
    miri: &MiriInterpCx<'_, 'tcx>,
    diff: u64,
) -> InterpResult<'tcx, TyAndLayout<'tcx>> {
    let pointee_type = if diff >= 16 {
        miri.tcx.types.u128
    } else if diff >= 8 {
        miri.tcx.types.u64
    } else if diff >= 4 {
        miri.tcx.types.u32
    } else if diff >= 2 {
        miri.tcx.types.u16
    } else if diff >= 1 {
        miri.tcx.types.u8
    } else {
        miri.tcx.types.unit
    };
    miri.raw_pointer_to(pointee_type)
}

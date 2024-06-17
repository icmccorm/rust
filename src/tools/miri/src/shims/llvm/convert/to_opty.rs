extern crate itertools;
extern crate rustc_abi;
use super::field_bytes::FieldBytes;
use crate::alloc_addresses::EvalContextExt as _;
use crate::shims::llvm::helpers::EvalContextExt as LLVMEvalExt;
use crate::shims::llvm::logging::LLVMFlag;
use crate::*;
use inkwell::types::BasicTypeEnum;
use inkwell::values::GenericValueRef;
use itertools::Itertools;
use rustc_abi::{Endian, Size};
use rustc_apfloat::{
    ieee::{Double, Single},
    Float,
};
use rustc_const_eval::interpret::{InterpResult, MemoryKind};
use rustc_middle::{
    mir::{self, AggregateKind},
    ty::{self, layout::TyAndLayout, AdtKind},
};
use rustc_target::abi::FIRST_VARIANT;
use rustc_target::abi::{FieldIdx, VariantIdx};
use std::cell::Cell;
use std::fmt::Formatter;
use tracing::debug;

#[derive(Debug, Clone)]
enum OpTySource<'lli> {
    Generic(GenericValueRef<'lli>),
    Bytes(FieldBytes),
}

impl<'lli> std::fmt::Display for OpTySource<'lli> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            OpTySource::Generic(gv) => {
                let as_string = if let Some(type_tag) = gv.get_type_tag() {
                    type_tag.print_to_string().to_string()
                } else {
                    "None".to_string()
                };
                write!(f, "{:?}", as_string)
            }
            OpTySource::Bytes(b) => write!(f, "{}", b),
        }
    }
}

pub struct ConversionContext<'tcx, 'lli> {
    source: OpTySource<'lli>,
    rust_layout: TyAndLayout<'tcx>,
    padded_size: Size,
    destination: Cell<Option<PlaceTy<'tcx>>>,
}

impl<'tcx, 'lli> ConversionContext<'tcx, 'lli> {
    pub fn new(source: GenericValueRef<'lli>, rust_layout: TyAndLayout<'tcx>) -> Self {
        let padded_size = rust_layout.size;
        ConversionContext {
            source: OpTySource::Generic(source),
            rust_layout,
            padded_size,
            destination: Cell::new(None),
        }
    }

    pub fn new_to_place(source: GenericValueRef<'lli>, destination: PlaceTy<'tcx>) -> Self {
        let context = Self::new(source, destination.layout);
        context.destination.set(Some(destination));
        context
    }

    fn resolve_fields(
        &self,
        miri: &mut MiriInterpCx<'tcx>,
        destination: PlaceTy<'tcx>,
    ) -> InterpResult<'tcx, Vec<(PlaceTy<'tcx>, Size)>> {
        let (variant_dest, active_field_index) = if let Some(agk) = self.get_aggregate_kind(miri)? {
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
        Ok(self
            .rust_layout
            .fields
            .index_by_increasing_offset()
            .map(|idx| {
                let idx = if let Some(afidx) = active_field_index { afidx.as_usize() } else { idx };
                let padded_size = miri.resolve_padded_size(&self.rust_layout, idx);
                miri.project_field(&variant_dest, idx).map(|pt| (pt, padded_size))
            })
            .collect::<InterpResult<'tcx, Vec<_>>>()?)
    }

    fn new_from_field(
        source: OpTySource<'lli>,
        destination: PlaceTy<'tcx>,
        padded_size: Size,
    ) -> Self {
        ConversionContext {
            source,
            rust_layout: destination.layout,
            padded_size,
            destination: Cell::new(Some(destination)),
        }
    }

    fn get_destination(&mut self) -> Option<PlaceTy<'tcx>> {
        self.destination.get_mut().clone()
    }

    fn get_or_create_destination(
        &mut self,
        miri: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, PlaceTy<'tcx>> {
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
    fn get_discriminant(&self, miri: &MiriInterpCx<'tcx>) -> InterpResult<'tcx, u32> {
        if self.rust_layout.fields.count() > 0 {
            match &self.source {
                OpTySource::Generic(gvr) =>
                    if let Some(llvm_field) = gvr.get_field(0) {
                        Ok(u32::try_from(llvm_field.as_int()).unwrap())
                    } else {
                        throw_llvm_field_count_mismatch!(
                            gvr.get_aggregate_size(),
                            self.rust_layout
                        );
                    },
                OpTySource::Bytes(bytes) => {
                    let field = bytes.field(miri, self.rust_layout, 0);
                    Ok(u32::try_from(field.as_uint()).unwrap())
                }
            }
        } else {
            Ok(0)
        }
    }

    fn get_aggregate_kind(
        &self,
        miri: &MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Option<AggregateKind<'tcx>>> {
        let rust_type = self.rust_layout.ty;
        let kind = if let ty::Adt(adt_def, sr) = rust_type.kind() {
            match adt_def.adt_kind() {
                AdtKind::Struct =>
                    Some(AggregateKind::Adt(adt_def.did(), VariantIdx::from_u32(0), sr, None, None)),
                AdtKind::Union =>
                    Some(AggregateKind::Adt(
                        adt_def.did(),
                        VariantIdx::from_u32(0),
                        sr,
                        None,
                        Some(FieldIdx::from_u32(0)),
                    )),
                AdtKind::Enum =>
                    Some(AggregateKind::Adt(
                        adt_def.did(),
                        VariantIdx::from_u32(self.get_discriminant(miri)?),
                        sr,
                        None,
                        None,
                    )),
            }
        } else {
            None
        };
        Ok(kind)
    }

    pub fn convert_generic_value_to_opty(
        &mut self,
        miri: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, (OpTy<'tcx>, Option<PlaceTy<'tcx>>)> {
        debug!("[GV to Op]: {} to {:?}", self.source, self.rust_layout.ty);
        match self.rust_layout.abi {
            rustc_abi::Abi::Scalar(_) => {
                let scalar_op = OpTy::from(self.convert_to_immty(miri)?);
                if let Some(existing) = self.destination.get_mut() {
                    miri.copy_op(&scalar_op, existing)?;
                }
                Ok((scalar_op, self.get_destination()))
            }
            rustc_abi::Abi::ScalarPair(_, _)
            | rustc_abi::Abi::Aggregate { sized: true }
            | rustc_abi::Abi::Vector { .. } => {
                let destination = self.get_or_create_destination(miri)?;
                match &self.source {
                    OpTySource::Generic(generic) => {
                        let llvm_type = generic.assert_type_tag();
                        match llvm_type {
                            BasicTypeEnum::IntType(_) => {
                                let field_bytes = generic.as_int().to_ne_bytes();
                                let field_width: usize =
                                    self.rust_layout.size.bytes().try_into().unwrap();
                                let field_bytes = FieldBytes::new(field_bytes, field_width);
                                let mut new_context = ConversionContext::new_from_field(
                                    OpTySource::Bytes(field_bytes),
                                    destination.clone(),
                                    self.padded_size,
                                );
                                new_context.convert_generic_value_to_opty(miri)?;
                            }
                            _ => {
                                let mut llvm_fields = generic.assert_fields();
                                let rust_field_count = self.rust_layout.fields.count();
                                if rust_field_count == llvm_fields.len() {
                                    let rust_fields =
                                        self.resolve_fields(miri, destination.clone())?;
                                    for (llvm_field, (rust_field, padded_size)) in
                                        llvm_fields.drain(0..).zip_eq(rust_fields)
                                    {
                                        let mut new_context = ConversionContext::new_from_field(
                                            OpTySource::Generic(llvm_field),
                                            rust_field,
                                            padded_size,
                                        );
                                        new_context.convert_generic_value_to_opty(miri)?;
                                    }
                                } else if miri
                                    .can_dereference_into_singular_field(&self.rust_layout)
                                    && llvm_fields.len() != rust_field_count
                                {
                                    let mut field = miri.project_field(&destination, 0)?;
                                    while miri.can_dereference_into_singular_field(&field.layout) {
                                        field = miri.project_field(&field, 0)?;
                                    }
                                    let mut new_context = ConversionContext::new_from_field(
                                        self.source.clone(),
                                        field,
                                        self.padded_size,
                                    );
                                    new_context.convert_generic_value_to_opty(miri)?;
                                } else {
                                    throw_llvm_field_count_mismatch!(
                                        llvm_fields.len(),
                                        self.rust_layout
                                    );
                                }
                            }
                        }
                        let operand = miri.place_to_op(&destination)?;
                        return Ok((operand, Some(destination)));
                    }
                    OpTySource::Bytes(fieldbytes) => {
                        let mut rust_fields = self.resolve_fields(miri, destination.clone())?;
                        if u64::try_from(fieldbytes.len()).unwrap() != self.rust_layout.size.bytes()
                        {
                            throw_llvm_field_width_mismatch!(fieldbytes.len(), self.rust_layout);
                        } else {
                            for (idx, (rust_field, padded_size)) in
                                rust_fields.drain(..).enumerate()
                            {
                                let llvm_field = fieldbytes.field(miri, self.rust_layout, idx);
                                let mut new_context = ConversionContext::new_from_field(
                                    OpTySource::Bytes(llvm_field),
                                    rust_field,
                                    padded_size,
                                );
                                new_context.convert_generic_value_to_opty(miri)?;
                            }
                            let operand = miri.place_to_op(&destination)?;
                            return Ok((operand, Some(destination)));
                        }
                    }
                }
            }
            rustc_abi::Abi::Uninhabited => throw_unsup_abi!("Uninhabited"),
            rustc_abi::Abi::Aggregate { sized: false } => throw_unsup_abi!("unsized Aggregate"),
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn convert_to_immty(&self, miri: &MiriInterpCx<'tcx>) -> InterpResult<'tcx, ImmTy<'tcx>> {
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
        match &self.source {
            OpTySource::Generic(generic) => {
                let layouts = &miri.machine.layouts;
                let rust_type = self.rust_layout.ty;
                let llvm_type = generic.assert_type_tag();
                match llvm_type {
                    BasicTypeEnum::FloatType(_) =>
                        if let ty::Float(rft) = rust_type.kind() {
                            match rft {
                                ty::FloatTy::F32 => {
                                    let val = generic.as_f32();
                                    debug!("[GV to Op]: Float value: {:?}", val);
                                    let imm = ImmTy::from_scalar(
                                        Scalar::from_f32(Single::from_bits(val.to_bits().into())),
                                        layouts.f32,
                                    );
                                    return Ok(imm);
                                }
                                ty::FloatTy::F64 => {
                                    let val = generic.as_f64();
                                    debug!("[GV to Op]: Float value: {:?}", val);
                                    let imm = ImmTy::from_scalar(
                                        Scalar::from_f64(Double::from_bits(val.to_bits().into())),
                                        layouts.f64,
                                    );
                                    return Ok(imm);
                                }
                                _ => {
                                    throw_llvm_type_mismatch!(llvm_type, rust_type);
                                }
                            }
                        },
                    BasicTypeEnum::IntType(_) => {
                        let converted_int = generic.as_int();
                        debug!("[GV to Op]: Int value: {:?}", converted_int);
                        let byte_width = Size::from_bytes(u64::from(generic.int_width_bytes()));
                        if miri.is_pointer_convertible(&self.rust_layout) {
                            if byte_width == self.rust_layout.size {
                                let first_word = truncate_to_pointer_size(converted_int);
                                if let Some(logger) = &miri.machine.llvm_logger {
                                    logger.log_flag(LLVMFlag::CastPointerFromLLVMAtBoundary);
                                }
                                let as_maybe_ptr = miri.ptr_from_addr_cast(first_word)?;

                                let scalar = match as_maybe_ptr.into_pointer_or_addr() {
                                    Ok(ptr) => Scalar::from_pointer(ptr, miri),
                                    Err(addr) =>
                                        Scalar::from_uint(addr.bytes(), self.rust_layout.size),
                                };
                                return Ok(ImmTy::from_scalar(scalar, self.rust_layout));
                            }
                        } else if byte_width == self.rust_layout.size
                            || (byte_width > self.rust_layout.size
                                && byte_width == self.padded_size)
                        {
                            let scalar = Scalar::from_uint(converted_int, self.rust_layout.size);
                            return Ok(ImmTy::from_scalar(scalar, self.rust_layout));
                        } else {
                            throw_llvm_field_width_mismatch!(byte_width.bytes(), self.rust_layout);
                        }
                    }
                    BasicTypeEnum::PointerType(_) =>
                        if miri.is_pointer_convertible(&self.rust_layout) {
                            let wrapped_pointer = generic.as_miri_pointer();
                            let mp = miri.lli_wrapped_pointer_to_maybe_pointer(wrapped_pointer);
                            debug!(
                                "[GV to Op]: Provenance: (Tag: {}, AID: {}, Addr: {})",
                                wrapped_pointer.prov.tag,
                                wrapped_pointer.prov.alloc_id,
                                wrapped_pointer.addr
                            );
                            let pointer_ty_layout = self.rust_layout;
                            let scalar = Scalar::from_maybe_pointer(mp.into(), miri);
                            let imm = ImmTy::from_scalar(scalar, pointer_ty_layout);
                            return Ok(imm);
                        } else {
                            throw_llvm_type_mismatch!(llvm_type, rust_type);
                        },
                    _ => {}
                }
                throw_llvm_type_mismatch!(llvm_type, rust_type);
            }
            OpTySource::Bytes(fieldbytes) => {
                let value = fieldbytes.as_uint();
                let layout = self.rust_layout;
                let scalar = if miri.is_pointer_convertible(&layout) {
                    if let Some(logger) = &miri.machine.llvm_logger {
                        logger.log_flag(LLVMFlag::CastPointerFromLLVMAtBoundary);
                    }
                    let first_word = truncate_to_pointer_size(value);
                    let as_maybe_ptr = miri.ptr_from_addr_cast(first_word)?;
                    match as_maybe_ptr.into_pointer_or_addr() {
                        Ok(ptr) => Scalar::from_pointer(ptr, miri),
                        Err(addr) =>
                            Scalar::from_uint(addr.bytes(), miri.tcx.data_layout.pointer_size),
                    }
                } else {
                    Scalar::from_uint(fieldbytes.as_uint(), layout.size)
                };
                return Ok(ImmTy::from_scalar(scalar, layout));
            }
        }
    }
}

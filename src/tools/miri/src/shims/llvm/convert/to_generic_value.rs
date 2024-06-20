extern crate either;
extern crate rustc_abi;
use super::to_bytes::EvalContextExt as ToBytesEvalExt;
use crate::shims::llvm::helpers::EvalContextExt as LLVMEvalExt;
use crate::shims::llvm::lli::{ResolvedLLVMType, ResolvedRustArgument};
use crate::shims::llvm::logging::LLVMFlag;
use either::Either::{Left, Right};
use inkwell::values::FunctionValue;
use inkwell::values::GenericValueRef;
use inkwell::{types::BasicTypeEnum, values::GenericValue};
use rustc_abi::Size;
use rustc_apfloat::Float;
use rustc_const_eval::interpret::{alloc_range, InterpResult};
use rustc_middle::ty::{self};
use rustc_target::abi::call::HomogeneousAggregate;
use rustc_target::abi::FieldsShape;
use std::iter::repeat;
use tracing::debug;
use crate::*;


macro_rules! throw_llvm_argument_mismatch {
    ($function: expr, $rust_args: expr, $llvm_args: expr) => {
        throw_interop_format!(
            "argument count mismatch: LLVM function {:?} {} expects {}{} arguments, but Rust provided {}",
            $function.get_name(),
            $function.get_type().print_to_string().to_string(),
            (if $function.get_type().is_var_arg() { "at least " } else { "" }),
            $llvm_args,
            $rust_args
        )
    };
}

pub struct LLVMArgumentConverter<'tcx, 'lli> {
    function: FunctionValue<'lli>,
    rust_args: Vec<ResolvedRustArgument<'tcx>>,
    original_argument_counts: (usize, usize),
    llvm_types: Vec<ResolvedLLVMType<'lli>>,
}
#[allow(dead_code)]
impl<'tcx, 'lli> LLVMArgumentConverter<'tcx, 'lli> {
    pub fn new(
        ctx: &mut MiriInterpCx<'tcx>,
        function: FunctionValue<'lli>,
        rust_args: Vec<OpTy<'tcx>>,
        llvm_types: Vec<BasicTypeEnum<'lli>>,
    ) -> InterpResult<'tcx, Self> {
        let mut llvm_types: Vec<ResolvedLLVMType<'_>> =
            llvm_types.iter().map(|t| Some(*t)).collect();
        let mut rust_args = rust_args
            .into_iter()
            .map(|arg| ResolvedRustArgument::new(ctx, arg))
            .collect::<InterpResult<'tcx, Vec<_>>>()?;
        llvm_types.reverse();
        rust_args.reverse();
        let original_argument_counts = (rust_args.len(), llvm_types.len());
        Ok(Self { function, rust_args, original_argument_counts, llvm_types })
    }

    fn advance_arg(&mut self) -> Option<ResolvedRustArgument<'tcx>> {
        self.rust_args.pop()
    }

    fn advance_type(&mut self) -> Option<ResolvedLLVMType<'lli>> {
        self.llvm_types.pop()
    }

    fn peek_arg(&self) -> Option<&ResolvedRustArgument<'tcx>> {
        self.rust_args.last()
    }

    fn peek_type(&self) -> Option<&ResolvedLLVMType<'lli>> {
        self.llvm_types.last()
    }

    fn assert_type(&mut self) -> InterpResult<'tcx, ResolvedLLVMType<'lli>> {
        if let Some(llvm_type) = self.advance_type() { Ok(llvm_type) } else { self.error() }
    }

    fn error<A>(&self) -> InterpResult<'tcx, A> {
        let (rust_count, llvm_count) = self.original_argument_counts;
        throw_llvm_argument_mismatch!(self.function, llvm_count, rust_count)
    }

    pub fn convert(
        mut self,
        ctx: &mut MiriInterpCx<'tcx>,
        function: FunctionValue<'lli>,
    ) -> InterpResult<'tcx, Vec<GenericValue<'lli>>> {
        let this = ctx.eval_context_mut();
        let original_arg_length = self.rust_args.len();
        let is_var_arg = function.get_type().is_var_arg();
        if original_arg_length > self.llvm_types.len() && !is_var_arg {
            return self.error();
        }
        if is_var_arg {
            if let Some(logger) = &mut this.machine.llvm_logger {
                logger.log_flag(LLVMFlag::VarArgFunction);
            }
        }
        let mut generic_args = Vec::with_capacity(self.llvm_types.len());
        while let Some(current_arg) = self.advance_arg() {
            let mut next_llvm_type = if self.llvm_types.is_empty() && !is_var_arg {
                self.error()?
            } else {
                self.peek_type().map(|f| *f).flatten()
            };
            if let Some(llvm_type) = next_llvm_type {
                if self.can_expand_aggregate(this, &current_arg, &llvm_type)? {
                    if let Some(mut fields) = self.expand_aggregate(this, &current_arg)? {
                        self.rust_args.append(&mut fields);
                        continue;
                    }
                } else {
                    next_llvm_type = self.assert_type()?;
                }
            }
            generic_args.push(current_arg.to_generic_value(this, next_llvm_type)?);
        }
        if !self.llvm_types.is_empty() {
            println!("LLVM types remaining: {:?}", self.llvm_types.len());
            self.error()?;
        }
        Ok(generic_args)
    }

    fn can_expand_aggregate(
        &self,
        ctx: &MiriInterpCx<'tcx>,
        rust_arg: &ResolvedRustArgument<'tcx>,
        llvm_type: &BasicTypeEnum<'_>,
    ) -> InterpResult<'tcx, bool> {
        let this = ctx.eval_context_ref();
        let llvm_type_size = this.resolve_llvm_type_size(*llvm_type)?;
        let rust_type_size = rust_arg.layout().size.bytes();
        let num_types_remaining = self.llvm_types.len();
        if llvm_type_size < rust_type_size && num_types_remaining > 0 {
            if matches!(*llvm_type, BasicTypeEnum::PointerType(_)) && !rust_arg.is_immediate() {
                return Ok(false);
            }
            let is_homogenous = rust_arg.layout().homogeneous_aggregate(this);
            if let Ok(HomogeneousAggregate::Homogeneous(_)) = is_homogenous {
                let not_union = !(matches!(rust_arg.layout().fields, FieldsShape::Union(_)));
                let inhabited = !rust_arg.abi().is_uninhabited();
                let not_scalar = !rust_arg.abi().is_scalar();
                let field_count = rust_arg.layout().fields.count();
                if not_union && inhabited && not_scalar && field_count <= (num_types_remaining) {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    fn expand_aggregate(
        &mut self,
        ctx: &mut MiriInterpCx<'tcx>,
        arg: &ResolvedRustArgument<'tcx>,
    ) -> InterpResult<'tcx, Option<Vec<ResolvedRustArgument<'tcx>>>> {
        let this = ctx.eval_context_mut();
        if !this.is_fieldless(&arg.layout()) {
            let curr_arg = arg.opty().clone();
            let field_values = curr_arg
                .layout
                .fields
                .index_by_increasing_offset()
                .map(|idx| ResolvedRustArgument::new(this, this.project_field(arg.opty(), idx)?))
                .collect::<InterpResult<'tcx, Vec<ResolvedRustArgument<'tcx>>>>()?;
            if let Some(logger) = &mut this.machine.llvm_logger {
                logger.log_aggregate_expansion(curr_arg.layout)
            }
            return Ok(Some(field_values));
        }
        Ok(None)
    }
}

#[allow(clippy::arithmetic_side_effects)]
pub fn convert_opty_to_generic_value<'tcx, 'lli>(
    ctx: &mut MiriInterpCx<'tcx>,
    dest: &mut GenericValueRef<'lli>,
    arg: ResolvedRustArgument<'tcx>,
    bte: ResolvedLLVMType<'lli>,
) -> InterpResult<'tcx> {
    if let Some(bte) = bte {
        dest.set_type_tag(&bte);
        debug!("[Op to GV]: {:?} -> {:?}", arg.ty(), bte.print_to_string());
        match bte {
            BasicTypeEnum::ArrayType(at) => {
                let llvm_field_types =
                    repeat(at.get_element_type()).take(at.len() as usize).collect();
                convert_opty_to_aggregate(ctx, dest, arg.opty(), llvm_field_types)?;
            }
            BasicTypeEnum::FloatType(ft) =>
                if let rustc_abi::Abi::Scalar(..) = arg.abi() {
                    let type_size = ctx.resolve_llvm_type_size(bte)?;
                    if arg.value_size().bytes() != type_size {
                        if arg.padded_size().bytes() != type_size {
                            throw_rust_type_mismatch!(arg.layout(), ft);
                        }
                    }
                    let scalar = ctx.read_scalar(arg.opty())?;
                    if let Scalar::Int(si) = scalar {
                        match bte.get_llvm_type_kind() {
                            llvm_sys::LLVMTypeKind::LLVMFloatTypeKind => {
                                if si.size() != Size::from_bytes(std::mem::size_of::<f32>()) {
                                    throw_rust_type_mismatch!(arg.layout(), ft);
                                }
                                let bits = scalar.to_f32()?.to_bits();
                                let bytes = ctx.to_vec_endian(bits, arg.value_size().bytes_usize());
                                let float = f32::from_ne_bytes(bytes.try_into().unwrap());
                                debug!("[Op to GV]: Float value: {}", float);
                                dest.set_float_value(float);
                            }
                            llvm_sys::LLVMTypeKind::LLVMDoubleTypeKind => {
                                if si.size() != Size::from_bytes(std::mem::size_of::<f64>()) {
                                    throw_rust_type_mismatch!(arg.layout(), ft);
                                }
                                let double = scalar.to_f64()?.to_bits();
                                let bytes =
                                    ctx.to_vec_endian(double, arg.value_size().bytes_usize());
                                let double = f64::from_ne_bytes(bytes.try_into().unwrap());
                                debug!("[Op to GV]: Double value: {}", double);
                                dest.set_double_value(double);
                            }
                            _ => {
                                if si.size()
                                    != Size::from_bytes(
                                        ft.size_of().get_zero_extended_constant().unwrap(),
                                    )
                                {
                                    throw_rust_type_mismatch!(arg.layout(), ft);
                                }
                                let bits = si.to_bits(arg.value_size());
                                let bytes = ctx.to_vec_endian(bits, arg.value_size().bytes_usize());
                                dest.set_bytes(&bytes);
                            }
                        }
                    } else {
                        throw_rust_type_mismatch!(arg.layout(), ft);
                    }
                } else {
                    throw_rust_type_mismatch!(arg.layout(), ft);
                },

            BasicTypeEnum::IntType(it) => {
                if arg.value_size().bytes() != it.get_byte_width() as u64 {
                    if arg.padded_size().bytes() != it.get_byte_width() as u64 {
                        throw_rust_type_mismatch!(arg.layout(), it);
                    }
                }
                match arg.abi() {
                    rustc_abi::Abi::Scalar(sc) => {
                        let scalar = if let rustc_abi::Scalar::Union { value } = sc {
                            let mp = arg.opty().assert_mem_place();
                            let ptr = mp.ptr();
                            let alloc =
                                ctx.get_ptr_alloc(ptr, value.size(ctx))?;
                            if let Some(a) = alloc {
                                a.read_scalar(alloc_range(Size::ZERO, value.size(ctx)), true)?
                            } else {
                                bug!("unable to resolve allocation for union scalar value.");
                            }
                        } else {
                            ctx.read_scalar(arg.opty())?
                        };
                        let bytes = ctx.scalar_to_bytes(scalar, arg.layout())?;
                        dest.set_bytes(&bytes);
                    }
                    rustc_abi::Abi::ScalarPair(_, _) => {
                        let bytes = ctx.op_to_bytes(arg.opty())?;
                        dest.set_bytes(&bytes);
                    }

                    rustc_abi::Abi::Aggregate { sized: true } => {
                        if let Some(logger) = &ctx.machine.llvm_logger {
                            logger.log_flag(LLVMFlag::AggregateAsBytes);
                        }
                        let bytes = ctx.op_to_bytes(arg.opty())?;
                        dest.set_bytes(&bytes);
                    }
                    _ => {
                        throw_rust_type_mismatch!(arg.layout(), it);
                    }
                }
            }
            BasicTypeEnum::PointerType(pt) => {
                let pointer_opty = if let ty::Adt(adef, sr) = arg.ty().kind() {
                    if let Some(logger) = &ctx.machine.llvm_logger {
                        logger.log_flag(LLVMFlag::ADTAsPointerFromRust);
                    }
                    if let Some(vidx) = ctx.is_enum_of_nonnullable_ptr(*adef, sr) {
                        if let Some(logger) = &ctx.machine.llvm_logger {
                            logger.log_flag(LLVMFlag::EnumOfNonNullablePointer);
                        }
                        debug!("[Op to GV]: Enum of nonnullable pointer: {:?}", vidx);
                        let downcast_op = ctx.project_downcast(arg.opty(), vidx)?;
                        let field_op = ctx.project_field(&downcast_op, 0)?;
                        ctx.read_pointer(&field_op)?
                    } else {
                        if let Left(mp) = arg.opty().as_mplace_or_imm() {
                            mp.ptr()
                        } else {
                            ctx.read_pointer(arg.opty())?
                        }
                    }
                } else {
                    if arg.value_size() != ctx.tcx.data_layout.pointer_size {
                        throw_rust_type_mismatch!(arg.layout(), pt);
                    }
                    match arg.abi() {
                        rustc_abi::Abi::Scalar(_) => ctx.read_pointer(arg.opty())?,
                        _ =>
                            if let Left(mp) = arg.opty().as_mplace_or_imm() {
                                mp.ptr()
                            } else {
                                throw_rust_type_mismatch!(arg.layout(), pt);
                            },
                    }
                };
                let wrapped_pointer = ctx.pointer_to_lli_wrapped_pointer(pointer_opty);
                debug!(
                    "[Op to GV]: Pointer value: (AID: {}, Addr: {})",
                    wrapped_pointer.prov.alloc_id, wrapped_pointer.addr
                );
                dest.set_miri_pointer_value(wrapped_pointer);
            }
            BasicTypeEnum::StructType(st) => {
                let llvm_field_types = st.get_field_types();
                convert_opty_to_aggregate(ctx, dest, arg.opty(), llvm_field_types)?;
            }
            BasicTypeEnum::VectorType(vt) =>
                match arg.abi() {
                    rustc_abi::Abi::Vector { element, count } => {
                        let pointer = match arg.opty().as_mplace_or_imm() {
                            Left(mp) => mp.ptr(),
                            Right(imm) => {
                                if let rustc_const_eval::interpret::Immediate::Scalar(
                                    Scalar::Ptr(ptr, ..),
                                ) = *imm
                                {
                                    ptr.into()
                                } else {
                                    bug!(
                                        "expected a pointer to the start of a vector, but found {:?}",
                                        *imm
                                    );
                                }
                            }
                        };
                        let wrapped_pointer = ctx.pointer_to_lli_wrapped_pointer(pointer);
                        if u64::from(vt.get_size()) != count {
                            throw_rust_type_mismatch!(arg.layout(), vt);
                        }
                        let rust_vector_element_size = element.size(&ctx.tcx).bytes();
                        let llvm_vector_element_size =
                            ctx.resolve_llvm_type_size(vt.get_element_type())?;
                        if rust_vector_element_size != llvm_vector_element_size {
                            throw_rust_type_mismatch!(arg.layout(), vt);
                        }
                        debug!(
                            "[Op to GV]: Vector pointer value: (Tag: {}, AID: {}, Addr: {})",
                            wrapped_pointer.prov.tag,
                            wrapped_pointer.prov.alloc_id,
                            wrapped_pointer.addr
                        );
                        dest.set_miri_pointer_value(wrapped_pointer);
                    }
                    _ => {
                        throw_rust_type_mismatch!(arg.layout(), vt);
                    }
                },
        }
    } else {
        if let rustc_abi::Abi::Scalar(sc) = arg.abi() {
            match sc.primitive() {
                rustc_abi::Primitive::Int(_, _) => {
                    let scalar = ctx.read_scalar(arg.opty())?.assert_scalar_int();
                    let bits = scalar
                        .to_bits(arg.value_size());
                    debug!("[Op to GV]: Int value: {}", u64::try_from(bits).unwrap());
                    let bytes = ctx.to_vec_endian(bits, arg.value_size().bytes_usize());
                    dest.set_bytes(&bytes);
                }
                rustc_abi::Primitive::Float(fl) => {
                    match fl {
                        rustc_abi::Float::F32 => {
                            let scalar = ctx.read_scalar(arg.opty())?;
                            let bits = scalar.to_f32()?.to_bits();
                            let bytes = ctx.to_vec_endian(bits, arg.value_size().bytes_usize());
                            let float = f32::from_ne_bytes(bytes.try_into().unwrap());
                            debug!("[Op to GV]: Float value: {}", float);
                            dest.set_float_value(float);
                        }
                        rustc_abi::Float::F64 => {
                            let scalar = ctx.read_scalar(arg.opty())?;
                            let double = scalar.to_f64()?.to_bits();
                            let bytes = ctx.to_vec_endian(double, arg.value_size().bytes_usize());
                            let double = f64::from_ne_bytes(bytes.try_into().unwrap());
                            debug!("[Op to GV]: Double value: {}", double);
                            dest.set_double_value(double);
                        }
                        _ => {
                            throw_unsup_var_arg!(arg.layout());
                        }
                    }
                }
                rustc_abi::Primitive::Pointer(_) => {
                    let ptr = ctx.read_pointer(arg.opty())?;
                    let wrapped_pointer = ctx.pointer_to_lli_wrapped_pointer(ptr.into());
                    debug!(
                        "[Op to GV]: Pointer value: (AID: {}, Addr: {})",
                        wrapped_pointer.prov.alloc_id, wrapped_pointer.addr
                    );
                    dest.set_miri_pointer_value(wrapped_pointer);
                }
            }
        } else {
            throw_unsup_var_arg!(arg.layout());
        }
    }
    Ok(())
}

fn convert_opty_to_aggregate<'lli, 'tcx>(
    ctx: &mut MiriInterpCx<'tcx>,
    dest: &mut GenericValueRef<'lli>,
    arg: &OpTy<'tcx>,
    llvm_fields: Vec<BasicTypeEnum<'lli>>,
) -> InterpResult<'tcx> {
    debug!("[Op to GV] Aggregate Conversion, rust_type: {:?}", arg.layout.ty,);
    if arg.layout.fields.count() != llvm_fields.len() {
        throw_rust_field_mismatch!(arg.layout, llvm_fields.len());
    }
    debug!("Aggregate: {:?} fields to {:?} fields", arg.layout.fields.count(), llvm_fields.len());
    dest.ensure_capacity(llvm_fields.len().try_into().unwrap());

    let rust_field_iterator = arg.layout.fields.index_by_increasing_offset();
    let rust_llvm_field_pairs = rust_field_iterator.zip(llvm_fields);

    for (rust_field_idx, llvm_field) in rust_llvm_field_pairs {
        let padded_size = ctx.resolve_padded_size(&arg.layout, rust_field_idx);
        let as_op = ctx.project_field(arg, rust_field_idx)?;
        debug!(
            "Field {}, size: {}, padded size: {}",
            rust_field_idx,
            as_op.layout.size.bytes(),
            padded_size.bytes()
        );

        let mut dest_field = dest.assert_field(rust_field_idx.try_into().unwrap());
        let as_resolved = ResolvedRustArgument::new_padded(ctx, as_op, padded_size)?;
        convert_opty_to_generic_value(ctx, &mut dest_field, as_resolved, Some(llvm_field))?;
    }
    Ok(())
}

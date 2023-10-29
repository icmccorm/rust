use crate::shims::llvm::helpers::EvalContextExt;
use crate::shims::llvm::hooks::access::{Destination, Source};
use crate::shims::llvm::lli::LLI;
use crate::MiriInterpCx;
use inkwell::types::AsTypeRef;
use inkwell::{execution_engine::MiriPointer, values::GenericValue};
use inkwell::{types::BasicTypeEnum, values::GenericValueRef};
use llvm_sys::prelude::LLVMTypeRef;
use rustc_const_eval::interpret::{InterpResult, Pointer};

#[derive(Debug, Clone)]
pub struct GenericValueTy {
    pub type_ref: LLVMTypeRef,
    pub val_ref: GenericValueRef,
}
impl GenericValueTy {
    pub fn new(ty: BasicTypeEnum<'_>, val_ref: GenericValueRef) -> Self {
        Self { type_ref: ty.as_type_ref(), val_ref }
    }
    pub fn ty(&self) -> BasicTypeEnum<'static> {
        unsafe { BasicTypeEnum::new(self.type_ref) }
    }
    pub fn resolve_fields(&self) -> Vec<GenericValueTy> {
        let llvm_actual_field_count = usize::try_from(self.val_ref.get_aggregate_size()).unwrap();
        let field_array = match self.ty() {
            BasicTypeEnum::ArrayType(at) => {
                let array_length = usize::try_from(at.len()).unwrap();
                let array_types =
                    std::iter::repeat(at.get_element_type()).take(array_length).collect();
                Some(array_types)
            }
            BasicTypeEnum::StructType(st) => Some(st.get_field_types()),
            BasicTypeEnum::VectorType(vt) => {
                let vector_length = usize::try_from(vt.get_size()).unwrap();
                let vector_types =
                    std::iter::repeat(vt.get_element_type()).take(vector_length).collect();
                Some(vector_types)
            }
            _ => None,
        };
        if let Some(field_array) = field_array {
            if field_array.len() != llvm_actual_field_count {
                bug!(
                    "size mismatch: expected an LLVM aggregate value with {} fields, but received one with {} fields",
                    field_array.len(),
                    llvm_actual_field_count
                );
            }
            field_array
                .iter()
                .enumerate()
                .map(|(idx, at)| {
                    let element =
                        self.val_ref.get_aggregate_value(u64::try_from(idx).unwrap()).unwrap();
                    GenericValueTy::new(*at, element)
                })
                .collect::<Vec<GenericValueTy>>()
        } else {
            if llvm_actual_field_count != 0 {
                bug!(
                    "size mismatch: expected an LLVM scalar value, but received an aggregate value with {} fields",
                    llvm_actual_field_count
                );
            } else {
                vec![self.clone()]
            }
        }
    }
}

pub trait FromGeneric {
    fn from_generic(generic: GenericValue<'_>, ctx: &LLI) -> Self;
}

macro_rules! impl_from_generic_int {
    ($(($t:ty, $signed:expr)),*) => {
        $(
            impl FromGeneric for $t {
                fn from_generic(generic: GenericValue<'_>, _ctx: &LLI) -> $t {
                    generic.as_int() as $t
                }
            }
        )*
    };
}
impl_from_generic_int!(
    (i8, true),
    (i16, true),
    (i32, true),
    (i64, true),
    (i128, true),
    (isize, true),
    (u8, false),
    (u16, false),
    (u32, false),
    (u64, false),
    (u128, false),
    (usize, false)
);

#[allow(clippy::cast_possible_truncation)]
impl FromGeneric for f32 {
    fn from_generic(generic: GenericValue<'_>, interp: &LLI) -> f32 {
        generic.as_float(&interp.borrow_context().f32_type()) as f32
    }
}
impl FromGeneric for f64 {
    fn from_generic(generic: GenericValue<'_>, interp: &LLI) -> f64 {
        generic.as_float(&interp.borrow_context().f64_type())
    }
}

impl FromGeneric for () {
    fn from_generic(_generic: GenericValue<'_>, _ctx: &LLI) {}
}

impl FromGeneric for MiriPointer {
    fn from_generic(generic: GenericValue<'_>, _ctx: &LLI) -> MiriPointer {
        generic.as_miri_pointer()
    }
}

impl FromGeneric for bool {
    fn from_generic(generic: GenericValue<'_>, _ctx: &LLI) -> Self {
        generic.as_int() != 0
    }
}

impl Destination<GenericValueRef> for GenericValueRef {
    fn write_f32<'tcx>(
        &mut self,
        _ctx: &mut MiriInterpCx<'_, 'tcx>,
        value: f32,
    ) -> InterpResult<'tcx> {
        self.set_float_value(value);
        Ok(())
    }

    fn write_f64<'tcx>(
        &mut self,
        _ctx: &mut MiriInterpCx<'_, 'tcx>,
        value: f64,
    ) -> InterpResult<'tcx> {
        self.set_double_value(value);
        Ok(())
    }

    fn write_unsigned<'tcx>(
        &mut self,
        _ctx: &mut MiriInterpCx<'_, 'tcx>,
        value: u128,
        bytes: u64,
    ) -> InterpResult<'tcx> {
        self.set_int_value(value, bytes);
        Ok(())
    }

    fn write_pointer<'tcx>(
        &mut self,
        ctx: &mut MiriInterpCx<'_, 'tcx>,
        pointer: Pointer<Option<crate::Provenance>>,
    ) -> InterpResult<'tcx> {
        let pointer = ctx.pointer_to_lli_wrapped_pointer(pointer);
        self.set_miri_pointer_value(pointer);
        Ok(())
    }

    fn resolve_field<'tcx>(
        &mut self,
        _ctx: &mut MiriInterpCx<'_, 'tcx>,
        _size: u64,
        index: u32,
    ) -> InterpResult<'tcx, GenericValueRef> {
        if let Some(element) = self.get_aggregate_value(u64::from(index)) {
            Ok(element)
        } else {
            bug!("cannot resolve field of GenericValue at index {}", index)
        }
    }

    fn ensure_aggregate_size<'tcx>(
        &self,
        _ctx: &MiriInterpCx<'_, 'tcx>,
        aggregate_size: u32,
    ) -> InterpResult<'tcx> {
        self.ensure_capacity(u64::from(aggregate_size));
        Ok(())
    }
}

impl Source<GenericValueRef> for GenericValueRef {
    fn read_f32<'tcx>(&self, _ctx: &MiriInterpCx<'_, 'tcx>) -> InterpResult<'tcx, f32> {
        Ok(self.as_f32())
    }

    fn read_f64<'tcx>(&self, _ctx: &MiriInterpCx<'_, 'tcx>) -> InterpResult<'tcx, f64> {
        Ok(self.as_f64())
    }

    fn read_unsigned<'tcx>(
        &self,
        _ctx: &mut MiriInterpCx<'_, 'tcx>,
        _bytes: u64,
    ) -> InterpResult<'tcx, u128> {
        Ok(self.as_int())
    }

    fn read_pointer<'tcx>(
        &self,
        ctx: &MiriInterpCx<'_, 'tcx>,
    ) -> InterpResult<'tcx, Pointer<Option<crate::Provenance>>> {
        let pointer = self.as_miri_pointer();
        let pointer = ctx.lli_wrapped_pointer_to_maybe_pointer(pointer);
        Ok(pointer)
    }

    fn check_aggregate_size<'tcx>(
        &self,
        _ctx: &MiriInterpCx<'_, 'tcx>,
        aggregate_size: u32,
    ) -> InterpResult<'tcx> {
        if self.get_aggregate_size() != u64::from(aggregate_size) {
            bug!(
                "aggregate size mismatch; GenericValue has {} but {} fields were requested",
                self.get_aggregate_size(),
                aggregate_size
            )
        } else {
            Ok(())
        }
    }

    fn resolve_field<'tcx>(
        &self,
        _ctx: &MiriInterpCx<'_, 'tcx>,
        _size: u64,
        index: u32,
    ) -> InterpResult<'tcx, GenericValueRef> {
        if let Some(element) = self.get_aggregate_value(u64::from(index)) {
            Ok(element)
        } else {
            bug!("cannot resolve field of GenericValue at index {}", index)
        }
    }
}

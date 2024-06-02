use crate::shims::llvm::helpers::EvalContextExt;
use crate::shims::llvm::hooks::access::{Destination, Source};
use crate::MiriInterpCx;
use inkwell::values::GenericValueRef;
use rustc_const_eval::interpret::{InterpResult, Pointer};
use rustc_target::abi::{Size, Align};


impl<'lli> Destination<GenericValueRef<'lli>> for GenericValueRef<'lli> {
    fn write_f32<'tcx>(
        &mut self,
        _ctx: &mut MiriInterpCx<'_, 'tcx>,
        value: f32,
        _align: Align,
    ) -> InterpResult<'tcx> {
        self.set_float_value(value);
        Ok(())
    }

    fn write_f64<'tcx>(
        &mut self,
        _ctx: &mut MiriInterpCx<'_, 'tcx>,
        value: f64,
        _align: Align,
    ) -> InterpResult<'tcx> {
        self.set_double_value(value);
        Ok(())
    }

    fn write_unsigned<'tcx>(
        &mut self,
        _ctx: &mut MiriInterpCx<'_, 'tcx>,
        value: u128,
        size: Size,
        _align: Align,
    ) -> InterpResult<'tcx> {
        self.set_int_value(value, size.bytes());
        Ok(())
    }

    fn write_pointer<'tcx>(
        &mut self,
        ctx: &mut MiriInterpCx<'_, 'tcx>,
        pointer: Pointer<Option<crate::Provenance>>,
        _align: Align,
    ) -> InterpResult<'tcx> {
        let pointer = ctx.pointer_to_lli_wrapped_pointer(pointer);
        self.set_miri_pointer_value(pointer);
        Ok(())
    }

    fn resolve_field<'tcx>(
        &mut self,
        _ctx: &mut MiriInterpCx<'_, 'tcx>,
        _size: Size,
        index: u32,
    ) -> InterpResult<'tcx, GenericValueRef<'lli>> {
        Ok(self.assert_field(index.into()))
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

impl<'lli> Source<GenericValueRef<'lli>> for GenericValueRef<'lli> {
    fn read_f32<'tcx>(
        &self,
        _ctx: &MiriInterpCx<'_, 'tcx>,
        _align: Align,
    ) -> InterpResult<'tcx, f32> {
        Ok(self.as_f32())
    }

    fn read_f64<'tcx>(
        &self,
        _ctx: &MiriInterpCx<'_, 'tcx>,
        _align: Align,
    ) -> InterpResult<'tcx, f64> {
        Ok(self.as_f64())
    }

    fn read_unsigned<'tcx>(
        &self,
        _ctx: &mut MiriInterpCx<'_, 'tcx>,
        _size: Size,
        _align: Align,
    ) -> InterpResult<'tcx, u128> {
        Ok(self.as_int())
    }

    fn read_pointer<'tcx>(
        &self,
        ctx: &MiriInterpCx<'_, 'tcx>,
        _align: Align,
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
        _size: Size,
        index: u32,
    ) -> InterpResult<'tcx, GenericValueRef<'lli>> {
        Ok(self.assert_field(index.into()))
    }
}

extern crate rustc_abi;
use crate::machine::MiriInterpCxExt as _;
use crate::shims::llvm::logging::LLVMFlag;
use crate::{
    shims::llvm::{
        helpers::EvalContextExt,
        hooks::access::{Destination, Source},
    },
    MiriInterpCx,
};
use rustc_abi::{Align, Size};
use rustc_apfloat::{
    ieee::{Double, Single},
    Float,
};
use rustc_const_eval::interpret::{
    alloc_range, AllocId, AllocRange, AllocRef, AllocRefMut, InterpResult, Pointer, Scalar,
};
use crate::eval::ForeignMemoryMode;
use crate::alloc_addresses::EvalContextExt as _;

#[derive(Debug)]
pub struct ResolvedPointer {
    pub ptr: Pointer<Option<crate::Provenance>>,
    pub alloc_id: Option<AllocId>,
    pub align: Align,
    pub offset: Size,
}

impl ResolvedPointer {
    pub fn access_alloc<'tcx, 'a>(
        &'a self,
        ctx: &'a MiriInterpCx<'_, 'tcx>,
        access_size: Size,
        align: Align,
    ) -> InterpResult<
        'tcx,
        (AllocRef<'a, 'tcx, crate::Provenance, crate::AllocExtra<'tcx>>, AllocRange),
    > {
        let (size, _align, range) = self.get_access_size_range(ctx, access_size, align)?;
        let alloc_reference = unsafe { ctx.get_ptr_alloc_range(self.ptr, size, range)? };
        if let Some(ar) = alloc_reference {
            Ok((ar, range))
        } else {
            let addr = self.ptr.addr().bytes_usize();
            throw_ub_format!("unable to resolve allocation for pointer {addr}")
        }
    }

    pub fn access_alloc_mut<'tcx, 'a>(
        &'a self,
        ctx: &'a mut MiriInterpCx<'_, 'tcx>,
        access_size: Size,
        align: Align,
    ) -> InterpResult<
        'tcx,
        (AllocRefMut<'a, 'tcx, crate::Provenance, crate::AllocExtra<'tcx>>, AllocRange),
    > {
        let (size, _align, range) = self.get_access_size_range(ctx, access_size, align)?;
        let alloc_reference = unsafe { ctx.get_ptr_alloc_mut_range(self.ptr, size, range)? };
        if let Some(ar) = alloc_reference {
            Ok((ar, range))
        } else {
            let addr = self.ptr.addr().bytes_usize();
            throw_ub_format!("unable to resolve allocation for pointer {addr}")
        }
    }

    pub fn null() -> Self {
        ResolvedPointer {
            ptr: Pointer::null(),
            alloc_id: None,
            align: Align::ONE,
            offset: Size::from_bytes(0),
        }
    }

    pub fn with_provenance(&self) -> Option<Pointer<crate::Provenance>> {
        if let Some(p) = self.ptr.provenance {
            Some(Pointer::new(p, self.ptr.addr()))
        } else {
            None
        }
    }
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn get_access_size_range<'tcx>(
        &self,
        ctx: &MiriInterpCx<'_, 'tcx>,
        access_size: Size,
        access_align: Align,
    ) -> InterpResult<'tcx, (Size, Align, AllocRange)> {
        let this = ctx.eval_context_ref();
        if this.should_check_alignment_in_llvm(self.alloc_id) {
            Ok((access_size, access_align, alloc_range(Size::ZERO, access_size)))
        } else {
            Ok((self.offset + access_size, self.align, alloc_range(self.offset, access_size)))
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn offset_to_field<'tcx>(
        &self,
        ctx: &MiriInterpCx<'_, 'tcx>,
        size: Size,
        index: u32,
    ) -> InterpResult<'tcx, ResolvedPointer> {
        let this = ctx.eval_context_ref();
        let (ptr, offset) = if this.should_check_alignment_in_llvm(self.alloc_id) {
            let ptr = self.ptr.offset(Size::from_bytes(size.bytes() * u64::from(index)), ctx)?;
            (ptr, self.offset)
        } else {
            let new_offset = self.offset + Size::from_bytes(size.bytes() * u64::from(index));
            if new_offset.bytes() >= self.align.bytes() {
                let num_align = new_offset.bytes() / self.align.bytes();
                let ptr_by_align =
                    self.ptr.offset(Size::from_bytes(self.align.bytes() * num_align), ctx)?;
                let new_offset = Size::from_bytes(new_offset.bytes() % self.align.bytes());
                (ptr_by_align, new_offset)
            } else {
                (self.ptr, new_offset)
            }
        };
        Ok(Self { ptr, alloc_id: self.alloc_id, align: self.align, offset })
    }
}

impl Destination<ResolvedPointer> for ResolvedPointer {
    fn write_f32<'tcx>(
        &mut self,
        ctx: &mut MiriInterpCx<'_, 'tcx>,
        value: f32,
        align: Align,
    ) -> InterpResult<'tcx> {
        let size: Size = Size::from_bytes(std::mem::size_of::<f32>());
        let (mut alloc, range) = self.access_alloc_mut(ctx, size, align)?;
        alloc.write_scalar(range, Scalar::from_f32(Single::from_bits(value.to_bits().into())))?;
        Ok(())
    }

    fn write_f64<'tcx>(
        &mut self,
        ctx: &mut MiriInterpCx<'_, 'tcx>,
        value: f64,
        align: Align,
    ) -> InterpResult<'tcx> {
        let size = Size::from_bytes(std::mem::size_of::<f64>());
        let (mut alloc, range) = self.access_alloc_mut(ctx, size, align)?;
        alloc.write_scalar(range, Scalar::from_f64(Double::from_bits(value.to_bits().into())))?;
        Ok(())
    }

    fn write_unsigned<'tcx>(
        &mut self,
        ctx: &mut MiriInterpCx<'_, 'tcx>,
        value: u128,
        size: Size,
        align: Align,
    ) -> InterpResult<'tcx> {
        let (mut alloc, range) = self.access_alloc_mut(ctx, size, align)?;
        alloc.write_scalar(range, Scalar::from_uint(value, size))?;
        Ok(())
    }

    fn write_pointer<'tcx>(
        &mut self,
        ctx: &mut MiriInterpCx<'_, 'tcx>,
        pointer: Pointer<Option<crate::Provenance>>,
        align: Align,
    ) -> InterpResult<'tcx> {
        let pointer_size = ctx.tcx.data_layout.pointer_size;
        let pointer = Scalar::from_maybe_pointer(pointer, &*ctx);
        let (mut alloc, range) = self.access_alloc_mut(ctx, pointer_size, align)?;
        alloc.write_ptr_sized(range.start, pointer)?;
        Ok(())
    }

    fn resolve_field<'tcx>(
        &mut self,
        ctx: &mut MiriInterpCx<'_, 'tcx>,
        field_size: Size,
        index: u32,
    ) -> InterpResult<'tcx, ResolvedPointer> {
        self.offset_to_field(ctx, field_size, index)
    }

    fn ensure_aggregate_size<'tcx>(
        &self,
        _ctx: &MiriInterpCx<'_, 'tcx>,
        _aggregate_size: u32,
    ) -> InterpResult<'tcx> {
        Ok(())
    }
}

impl Source<ResolvedPointer> for ResolvedPointer {
    fn read_f32<'tcx>(
        &self,
        ctx: &MiriInterpCx<'_, 'tcx>,
        align: Align,
    ) -> InterpResult<'tcx, f32> {
        let size = Size::from_bytes(std::mem::size_of::<f32>());
        let (alloc, range) = self.access_alloc(ctx, size, align)?;
        let float_value = if matches!(ctx.machine.lli_config.memory_mode, ForeignMemoryMode::Uninit) {
            if alloc.is_uninit(range) {
                if let Some(logger) = &ctx.machine.llvm_logger {
                    logger.log_flag(LLVMFlag::LLVMReadUninit);
                }
            }
            alloc.read_scalar_uninit(range, false)?.to_f32()?
        } else {
            alloc.read_scalar(range, false)?.to_f32()?
        };
        let float_value = float_value.to_bits();
        let float_value = ctx.to_vec_endian(float_value, size.bytes_usize());
        let float_value = f32::from_ne_bytes(float_value.try_into().unwrap());
        Ok(float_value)
    }

    fn read_f64<'tcx>(
        &self,
        ctx: &MiriInterpCx<'_, 'tcx>,
        align: Align,
    ) -> InterpResult<'tcx, f64> {
        let size = Size::from_bytes(std::mem::size_of::<f64>());
        let (alloc, range) = self.access_alloc(ctx, size, align)?;
        let double_value = if matches!(ctx.machine.lli_config.memory_mode, ForeignMemoryMode::Uninit) {
            if alloc.is_uninit(range) {
                if let Some(logger) = &ctx.machine.llvm_logger {
                    logger.log_flag(LLVMFlag::LLVMReadUninit);
                }
            }
            alloc.read_scalar_uninit(range, false)?.to_f64()?
        } else {
            alloc.read_scalar(range, false)?.to_f64()?
        };
        let double_value = double_value.to_bits();
        let double_value = ctx.to_vec_endian(double_value, size.bytes_usize());
        let double_value = f64::from_ne_bytes(double_value.try_into().unwrap());
        Ok(double_value)
    }

    fn read_unsigned<'tcx>(
        &self,
        ctx: &mut MiriInterpCx<'_, 'tcx>,
        size: Size,
        align: Align,
    ) -> InterpResult<'tcx, u128> {
        let (alloc, range) = self.access_alloc(ctx, size, align)?;
        let int_value = if size == ctx.tcx.data_layout.pointer_size {
            let ptr_value = self.read_pointer(ctx, align)?;
            if let Some(crate::Provenance::Concrete { alloc_id, tag }) = ptr_value.provenance {
               ctx.expose_ptr(alloc_id, tag)?
            }
            u128::from(ptr_value.addr().bytes())
        } else {
            let scalar_value = if matches!(ctx.machine.lli_config.memory_mode, ForeignMemoryMode::Uninit) {
                if alloc.is_uninit(range) {
                    if let Some(logger) = &ctx.machine.llvm_logger {
                        logger.log_flag(LLVMFlag::LLVMReadUninit);
                    }
                }
                alloc.read_integer_uninit(range)?
            } else {
                alloc.read_integer(range)?
            };
            scalar_value.to_bits(size)?
        };
        Ok(int_value)
    }

    fn read_pointer<'tcx>(
        &self,
        ctx: &MiriInterpCx<'_, 'tcx>,
        align: Align,
    ) -> InterpResult<'tcx, Pointer<Option<crate::Provenance>>> {
        let (alloc, range) = self.access_alloc(ctx, ctx.tcx.data_layout.pointer_size, align)?;
        let pointer_val = if matches!(ctx.machine.lli_config.memory_mode, ForeignMemoryMode::Uninit) {
            if alloc.is_uninit(range) {
                if let Some(logger) = &ctx.machine.llvm_logger {
                    logger.log_flag(LLVMFlag::LLVMReadUninit);
                }
            }
            alloc.read_pointer_uninit(range.start)?
        } else {
            alloc.read_pointer(range.start)?
        };
        Ok(pointer_val.to_pointer(ctx)?)
    }

    fn check_aggregate_size<'tcx>(
        &self,
        _ctx: &MiriInterpCx<'_, 'tcx>,
        _aggregate_size: u32,
    ) -> InterpResult<'tcx> {
        Ok(())
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn resolve_field<'tcx>(
        &self,
        ctx: &MiriInterpCx<'_, 'tcx>,
        size: Size,
        index: u32,
    ) -> InterpResult<'tcx, ResolvedPointer> {
        self.offset_to_field(ctx, size, index)
    }
}

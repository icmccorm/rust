extern crate rustc_abi;
use crate::{
    intptrcast,
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

#[derive(Debug)]
pub struct ResolvedPointer {
    pub ptr: Pointer<Option<crate::Provenance>>,
    pub alloc_id: Option<AllocId>,
    pub align: Align,
    pub offset: Size,
}

impl ResolvedPointer {
    
    #[allow(dead_code)]
    fn access_range_as_string<'tcx>(
        &self,
        ctx: &MiriInterpCx<'_, 'tcx>,
        bytes: u64,
    ) -> InterpResult<'tcx, Option<String>> {
        if let Some(alloc_id) = self.alloc_id {
            let (_size, range) = self.get_access_size_range(bytes);
            let base_address = intptrcast::GlobalStateInner::alloc_base_addr(ctx, alloc_id)?;
            let offset = self.ptr.addr() - Size::from_bytes(base_address);
            let (size, _, _) = ctx.get_alloc_info(alloc_id);
            Ok(Some(format!(
                "alloc{} - [0..{}-[{},{}]-{}]",
                alloc_id.0.get(),
                offset.bytes(),
                offset.bytes() + range.start.bytes(),
                offset.bytes() + range.start.bytes() + range.size.bytes() - 1,
                size.bytes() - 1
            )))
        } else {
            Ok(None)
        }
    }

    pub fn access_alloc<'tcx, 'a>(
        &'a self,
        ctx: &'a MiriInterpCx<'_, 'tcx>,
        access_size: u64,
    ) -> InterpResult<
        'tcx,
        (AllocRef<'a, 'tcx, crate::Provenance, crate::AllocExtra<'tcx>>, AllocRange),
    > {
        let (size, range) = self.get_access_size_range(access_size);
        if let Some(ar) = ctx.get_ptr_alloc(self.ptr, size, self.align)? {
            Ok((ar, range))
        } else {
            let addr = self.ptr.addr().bytes_usize();
            throw_ub_format!("unable to resolve allocation for pointer {addr}")
        }
    }
    pub fn access_alloc_mut<'tcx, 'a>(
        &'a self,
        ctx: &'a mut MiriInterpCx<'_, 'tcx>,
        access_size: u64,
    ) -> InterpResult<
        'tcx,
        (AllocRefMut<'a, 'tcx, crate::Provenance, crate::AllocExtra<'tcx>>, AllocRange),
    > {
        let (size, range) = self.get_access_size_range(access_size);
        if let Some(ar) = ctx.get_ptr_alloc_mut(self.ptr, size, self.align)? {
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

    #[allow(clippy::arithmetic_side_effects)]
    pub fn get_access_size_range(&self, access_size: u64) -> (Size, AllocRange) {
        let alloc_range = alloc_range(self.offset, Size::from_bytes(access_size));
        (Size::from_bytes(self.offset.bytes() + access_size), alloc_range)
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn offset_to_field<'tcx>(
        &self,
        ctx: &MiriInterpCx<'_, 'tcx>,
        size: u64,
        index: u32,
    ) -> InterpResult<'tcx, ResolvedPointer> {
        let new_offset = self.offset + Size::from_bytes(size * u64::from(index));

        let (new_ptr, new_offset) = if new_offset.bytes() >= self.align.bytes() {
            let num_align = new_offset.bytes() / self.align.bytes();
            let ptr_by_align =
                self.ptr.offset(Size::from_bytes(self.align.bytes() * num_align), ctx)?;
            let new_offset = Size::from_bytes(new_offset.bytes() % self.align.bytes());
            (ptr_by_align, new_offset)
        } else {
            (self.ptr, new_offset)
        };

        Ok(Self { ptr: new_ptr, alloc_id: self.alloc_id, align: self.align, offset: new_offset })
    }
}

impl Destination<ResolvedPointer> for ResolvedPointer {
    fn write_f32<'tcx>(
        &mut self,
        ctx: &mut MiriInterpCx<'_, 'tcx>,
        value: f32,
    ) -> InterpResult<'tcx> {
        let size: Size = Size::from_bytes(std::mem::size_of::<f32>());
        let (mut alloc, range) = self.access_alloc_mut(ctx, size.bytes())?;
        alloc.write_scalar(range, Scalar::from_f32(Single::from_bits(value.to_bits().into())))?;
        Ok(())
    }

    fn write_f64<'tcx>(
        &mut self,
        ctx: &mut MiriInterpCx<'_, 'tcx>,
        value: f64,
    ) -> InterpResult<'tcx> {
        let size = Size::from_bytes(std::mem::size_of::<f64>());
        let (mut alloc, range) = self.access_alloc_mut(ctx, size.bytes())?;
        alloc.write_scalar(range, Scalar::from_f64(Double::from_bits(value.to_bits().into())))?;
        Ok(())
    }

    fn write_unsigned<'tcx>(
        &mut self,
        ctx: &mut MiriInterpCx<'_, 'tcx>,
        value: u128,
        bytes: u64,
    ) -> InterpResult<'tcx> {
        let (mut alloc, range) = self.access_alloc_mut(ctx, bytes)?;

        alloc.write_scalar(range, Scalar::from_uint(value, Size::from_bytes(bytes)))?;
        Ok(())
    }

    fn write_pointer<'tcx>(
        &mut self,
        ctx: &mut MiriInterpCx<'_, 'tcx>,
        pointer: Pointer<Option<crate::Provenance>>,
    ) -> InterpResult<'tcx> {
        let pointer_size = ctx.tcx.data_layout.pointer_size;
        let pointer = Scalar::from_maybe_pointer(pointer, &*ctx);
        let (mut alloc, range) = self.access_alloc_mut(ctx, pointer_size.bytes())?;
        alloc.write_ptr_sized(range.start, pointer)?;
        Ok(())
    }

    fn resolve_field<'tcx>(
        &mut self,
        ctx: &mut MiriInterpCx<'_, 'tcx>,
        size: u64,
        index: u32,
    ) -> InterpResult<'tcx, ResolvedPointer> {
        self.offset_to_field(ctx, size, index)
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
    fn read_f32<'tcx>(&self, ctx: &MiriInterpCx<'_, 'tcx>) -> InterpResult<'tcx, f32> {
        let size = Size::from_bytes(std::mem::size_of::<f32>());
        let (alloc, range) = self.access_alloc(ctx, size.bytes())?;
        let float_value = if ctx.machine.llvm_read_uninit {
            if alloc.is_uninit(range) {
                if let Some(logger) = &ctx.machine.llvm_logger {
                    logger.flags.log_llvm_read_uninit();
                }
            }
            alloc.read_scalar_uninit(range, false)?.to_f32()?
        }else{
            alloc.read_scalar(range, false)?.to_f32()?
        };
        let float_value = float_value.to_bits();
        let float_value = ctx.to_vec_endian(float_value, size.bytes_usize());
        let float_value = f32::from_ne_bytes(float_value.try_into().unwrap());
        Ok(float_value)
    }

    fn read_f64<'tcx>(&self, ctx: &MiriInterpCx<'_, 'tcx>) -> InterpResult<'tcx, f64> {
        let size = Size::from_bytes(std::mem::size_of::<f64>());
        let (alloc, range) = self.access_alloc(ctx, size.bytes())?;
        let double_value = if ctx.machine.llvm_read_uninit {
            if alloc.is_uninit(range) {
                if let Some(logger) = &ctx.machine.llvm_logger {
                    logger.flags.log_llvm_read_uninit();
                }
            }
            alloc.read_scalar_uninit(range, false)?.to_f64()?
        }else{
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
        bytes: u64,
    ) -> InterpResult<'tcx, u128> {
        let (alloc, range) = self.access_alloc(ctx, bytes)?;
        let int_value = if bytes == ctx.tcx.data_layout.pointer_size.bytes() {
            let ptr_value = self.read_pointer(ctx)?;
            if let Some(crate::Provenance::Concrete { alloc_id, tag }) = ptr_value.provenance {
                intptrcast::GlobalStateInner::expose_ptr(ctx, alloc_id, tag)?
            }
            u128::from(ptr_value.addr().bytes())
        } else {
            let scalar_value = if ctx.machine.llvm_read_uninit {
                if alloc.is_uninit(range) {
                    if let Some(logger) = &ctx.machine.llvm_logger {
                        logger.flags.log_llvm_read_uninit();
                    }
                }
                alloc.read_integer_uninit(range)?
            }else{
                alloc.read_integer(range)?
            };
            scalar_value.to_bits(Size::from_bytes(bytes))?
        };
        Ok(int_value)
    }

    fn read_pointer<'tcx>(
        &self,
        ctx: &MiriInterpCx<'_, 'tcx>,
    ) -> InterpResult<'tcx, Pointer<Option<crate::Provenance>>> {
        let (alloc, range) = self.access_alloc(ctx, ctx.tcx.data_layout.pointer_size.bytes())?;
        let pointer_val = if ctx.machine.llvm_read_uninit {
            if alloc.is_uninit(range) {
                if let Some(logger) = &ctx.machine.llvm_logger {
                    logger.flags.log_llvm_read_uninit();
                }
            }
            alloc.read_pointer_uninit(range.start)?
        }else{
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
        size: u64,
        index: u32,
    ) -> InterpResult<'tcx, ResolvedPointer> {
        self.offset_to_field(ctx, size, index)
    }
}

extern crate rustc_abi;
use crate::borrow_tracker::EvalContextExt as _;
use crate::shims::llvm::logging::LLVMFlag;
use crate::*;
use rustc_abi::Abi;
use rustc_const_eval::interpret::InterpResult;
use rustc_middle::ty::layout::TyAndLayout;
use std::iter::repeat;

mod field_bytes;

pub mod to_generic_value;
pub mod to_opty;
use to_opty::ConversionContext;

use inkwell::values::GenericValueRef;


impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn generic_value_to_opty<'lli>(
        &mut self,
        src: GenericValueRef<'lli>,
        rust_layout: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, (OpTy<'tcx>, Option<PlaceTy<'tcx>>)> {
        let this = self.eval_context_mut();
        let mut converter = ConversionContext::new(src, rust_layout);
        converter.convert_generic_value_to_opty(this)
    }

    fn write_generic_value<'lli>(
        &mut self,
        src: GenericValueRef<'lli>,
        dest: PlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let mut converter = ConversionContext::new_to_place(src, dest);
        converter.convert_generic_value_to_opty(this)?;
        Ok(())
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn scalar_to_bytes(
        &mut self,
        scalar: Scalar,
        layout: TyAndLayout<'_>,
    ) -> InterpResult<'tcx, Vec<u8>> {
        let this = self.eval_context_mut();
        let scalar_bits = match scalar {
            Scalar::Int(si) => si.to_bits(si.size()),
            Scalar::Ptr(p, _) => {
                if let crate::Provenance::Concrete { alloc_id, tag } = p.provenance {
                    if let Some(logger) = &mut this.machine.llvm_logger {
                        logger.log_flag(LLVMFlag::ExposedPointerFromRustAtBoundary);
                    }
                    this.expose_tag(alloc_id, tag)?
                }
                p.into_parts().1.bits().into()
            }
        };
        let length: usize = usize::try_from(layout.size.bytes()).unwrap();
        match this.tcx.sess.target.endian {
            rustc_abi::Endian::Little => {
                let as_byte_vec = scalar_bits.to_le_bytes();
                Ok(as_byte_vec.as_slice()[..length].to_vec())
            }
            rustc_abi::Endian::Big => {
                let as_byte_vec = scalar_bits.to_be_bytes();

                Ok(as_byte_vec.as_slice()[as_byte_vec.len() - length..].to_vec())
            }
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn op_to_bytes(&mut self, opty: &OpTy<'tcx>) -> InterpResult<'tcx, Vec<u8>> {
        let this = self.eval_context_mut();
        match opty.layout.abi {
            rustc_abi::Abi::Scalar(_) => {
                let as_scalar = this.read_scalar(opty)?;

                Ok(this.scalar_to_bytes(as_scalar, opty.layout)?)
            }
            Abi::ScalarPair(_, _) | Abi::Aggregate { sized: true } | Abi::Vector { .. } => {
                let mut data = Vec::new();
                for field_idx in opty.layout.fields.index_by_increasing_offset() {
                    let curr_field = this.project_field(opty, field_idx)?;

                    let mut curr_field_bytes = this.op_to_bytes(&curr_field)?;

                    let offset = opty.layout.fields.offset(field_idx);

                    let diff = offset.bytes() - u64::try_from(data.len()).unwrap();

                    data.append(
                        repeat(0)
                            .take(usize::try_from(diff).unwrap())
                            .collect::<Vec<u8>>()
                            .as_mut(),
                    );

                    data.append(curr_field_bytes.as_mut());
                }
                let data_length = u64::try_from(data.len()).unwrap();

                let remaining_diff = opty.layout.size.bytes() - data_length;
                data.append(
                    repeat(0)
                        .take(usize::try_from(remaining_diff).unwrap())
                        .collect::<Vec<u8>>()
                        .as_mut(),
                );
                Ok(data)
            }
            Abi::Aggregate { sized: false } => {
                throw_unsup_abi!("unsized Aggregate")
            }
            Abi::Uninhabited => throw_unsup_abi!("Uninhabited"),
        }
    }
}

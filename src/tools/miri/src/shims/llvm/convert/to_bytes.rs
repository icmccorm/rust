extern crate rustc_abi;
use crate::Provenance;
use rustc_abi::Abi;
use rustc_const_eval::interpret::{InterpResult, OpTy, Scalar};
use rustc_middle::ty::layout::TyAndLayout;
use std::iter::repeat;
impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}

pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    #![allow(clippy::arithmetic_side_effects)]
    fn scalar_to_bytes(&self, scalar: Scalar<Provenance>, layout: TyAndLayout<'_>) -> Vec<u8> {
        let this = self.eval_context_ref();
        let scalar_bits = scalar.assert_bits(layout.size);
        let length: usize = usize::try_from(layout.size.bytes()).unwrap();
        match this.tcx.sess.target.endian {
            rustc_abi::Endian::Little => {
                let as_byte_vec = scalar_bits.to_le_bytes();
                as_byte_vec.as_slice()[..length].to_vec()
            }
            rustc_abi::Endian::Big => {
                let as_byte_vec = scalar_bits.to_be_bytes();
                as_byte_vec.as_slice()[as_byte_vec.len() - length..].to_vec()
            }
        }
    }
    #[allow(clippy::arithmetic_side_effects)]
    fn op_to_bytes(&mut self, opty: &OpTy<'tcx, Provenance>) -> InterpResult<'tcx, Vec<u8>> {
        let this = self.eval_context_mut();
        match opty.layout.abi {
            rustc_abi::Abi::Scalar(_) => {
                let as_scalar = this.read_scalar(opty)?;
                Ok(this.scalar_to_bytes(as_scalar, opty.layout))
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

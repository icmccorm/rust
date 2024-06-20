extern crate itertools;
extern crate rustc_abi;
use itertools::Itertools;
use rustc_abi::Endian;
use rustc_middle::ty::layout::TyAndLayout;
use crate::*;


#[derive(Debug, Copy, Clone)]
pub struct FieldBytes {
    bytes: [u8; 16],
    length: usize,
}

impl FieldBytes {
    pub fn new(bytes: [u8; 16], length: usize) -> Self {
        if length > bytes.len() {
            bug!("Offset must be greater than 0, less than {}", bytes.len());
        }
        FieldBytes { bytes, length }
    }
    pub fn bytes(&self, endian: Endian) -> &[u8] {
        match endian {
            Endian::Little => &self.bytes[..self.length],
            Endian::Big => &self.bytes[self.length..],
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn as_uint(&self) -> u128 {
        u128::from_ne_bytes(self.bytes)
    }

    #[allow(clippy::arithmetic_side_effects)]
    pub fn field<'tcx>(
        &self,
        ctx: &MiriInterpCx<'tcx>,
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bytes_as_string = self.bytes.iter().map(|b| format!("{:02x}", b)).join("");
        write!(f, "FieldBytes({})", bytes_as_string)
    }
}

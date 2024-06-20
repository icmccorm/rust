pub mod lli;
pub mod helpers;
pub mod convert;
pub mod hooks;
pub mod threads;
pub mod values;
pub mod logging;

#[macro_export]
macro_rules! throw_llvm_field_count_mismatch {
    ($llvm_field_count:expr, $rust_layout:expr) => {
        throw_interop_format!(
            "LLVM field count mismatch: cannot convert an LLVM value with {} fields to a Rust value of type `{}` which has {} fields.",
            $llvm_field_count,
            $rust_layout.ty,
            $rust_layout.fields.count()
        )
    };
}
#[macro_export]
macro_rules! throw_llvm_field_width_mismatch {
    ($llvm_field_width:expr, $rust_layout:expr) => {
        throw_interop_format!(
            "LLVM field width mismatch: cannot convert an LLVM field of width {} to a Rust field of type `{}` which has width {}",
            $llvm_field_width,
            $rust_layout.ty,
            $rust_layout.size.bytes()
        )
    };
}
#[macro_export]
macro_rules! throw_llvm_type_mismatch {
    ($llvm_type:expr, $rust_type:expr) => {
        throw_interop_format!(
            "LLVM type mismatch: cannot convert an LLVM value of type `{}` to a Rust value of type `{}`",
            $llvm_type.print_to_string().to_string(),
            $rust_type
        )
    };
}
#[macro_export]
macro_rules! throw_unsup_abi {
    ($abi_string:expr) => {
        throw_interop_format!("Unsupported target ABI: {}", $abi_string)
    };
}
#[macro_export]
macro_rules! throw_unsup_llvm_type {
    ($llvm_type:expr) => {
        throw_interop_format!("Unsupported LLVM Type: {}", $llvm_type.print_to_string().to_string())
    };
}

#[macro_export]
macro_rules! throw_unsup_shim_llvm_type {
    ($llvm_type:expr) => {
        throw_interop_format!(
            "LLVM Type `{}` is not supported for use in shims.",
            $llvm_type.print_to_string().to_string()
        )
    };
}

#[macro_export]
macro_rules! throw_rust_type_mismatch {
    ($rust_layout:expr, $llvm_type:expr) => {
        throw_interop_format!(
            "Rust type mismatch: cannot convert a Rust value of type `{}` to an LLVM value of type `{}`.",
            $rust_layout.ty,
            $llvm_type.print_to_string().to_string()
        )
    }
}
#[macro_export]
macro_rules! throw_unsup_var_arg {
    ($rust_layout:expr) => {
        throw_interop_format!(
            "Non-scalar variable arguments are not supported: `{}`.",
            $rust_layout.ty,
        )
    };
}
#[macro_export]
macro_rules! throw_rust_field_mismatch {
    ($rust_layout:expr, $llvm_field_count:expr) => {
        throw_interop_format!(
            "Rust field count mismatch: cannot convert a Rust value of type `{}` which has {} fields to an LLVM value with {} fields",
            $rust_layout.ty,
            $rust_layout.fields.count(),
            $llvm_field_count,
        )
    }
}

#[macro_export]
macro_rules! throw_shim_argument_mismatch {
    ($shim_name:expr, $arg_count:expr, $actual_arg_count:expr) => {
        throw_interop_format!(
            "shim argument mismatch: shim {} expects {} arguments but {} were provided",
            $shim_name,
            $arg_count,
            $actual_arg_count,
        )
    };
}

use serde::Serialize;
use std::{
    cell::Cell,
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
};

use inkwell::support::LLVMString;
use rustc_middle::ty::Ty;
use rustc_target::abi::TyAndLayout;

use crate::eval::LLVMLoggingLevel;

pub struct LLVMLogger {
    bytecode: Option<File>,
    conversions: Option<File>,
    pub flags: LLVMFlags,
}

#[derive(Serialize, Debug)]
pub struct LLVMFlags {
    llvm_engaged: Cell<bool>,
    intptr_llvm: Cell<bool>,
    intptr_rust: Cell<bool>,
    llvm_inttoptr: Cell<bool>,
    llvm_ptrtoint: Cell<bool>,
    llvm_on_resolve: Cell<bool>,
    llvm_used_multithreading: Cell<bool>,
    scalar_pair_expansion: Cell<bool>,
    size_based_type_inference: Cell<bool>,
    integer_upcast: Cell<bool>,
}

impl LLVMFlags {
    pub fn new() -> Self {
        LLVMFlags {
            llvm_engaged: Cell::new(false),
            intptr_llvm: Cell::new(false),
            intptr_rust: Cell::new(false),
            llvm_inttoptr: Cell::new(false),
            llvm_ptrtoint: Cell::new(false),
            llvm_on_resolve: Cell::new(false),
            llvm_used_multithreading: Cell::new(false),
            scalar_pair_expansion: Cell::new(false),
            size_based_type_inference: Cell::new(false),
            integer_upcast: Cell::new(false),
        }
    }
    #[inline(always)]
    pub fn log_llvm_engaged(&self) {
        self.llvm_engaged.set(true)
    }
    #[inline(always)]
    pub fn log_inttoptr_llvm(&self) {
        self.llvm_inttoptr.set(true)
    }
    #[inline(always)]
    pub fn log_ptrtoint_llvm(&self) {
        self.llvm_ptrtoint.set(true)
    }
    #[inline(always)]
    pub fn log_from_addr_cast_rust(&self) {
        self.intptr_rust.set(true)
    }
    #[inline(always)]
    pub fn log_from_addr_cast_llvm(&self) {
        self.intptr_llvm.set(true)
    }
    #[inline(always)]
    pub fn log_llvm_on_resolve(&self) {
        self.llvm_on_resolve.set(true)
    }
    #[inline(always)]
    pub fn log_llvm_multithreading(&self) {
        self.llvm_used_multithreading.set(true)
    }
    #[inline(always)]
    pub fn log_size_based_type_inference(&self) {
        self.size_based_type_inference.set(true)
    }
}

impl Drop for LLVMFlags {
    fn drop(&mut self) {
        if self.llvm_engaged.get() {
            if let Ok(flags_json) = serde_json::to_string(&self) {
                let mut flags_path = std::env::current_dir().unwrap();
                flags_path.push("flags.json");
                let flags_log_file =
                    OpenOptions::new().create(true).write(true).truncate(true).open(flags_path);
                if let Ok(mut flags_log_file) = flags_log_file {
                    flags_log_file.write_all(flags_json.as_bytes()).unwrap_or(());
                    flags_log_file.flush().unwrap_or(());
                }
            }
        }
    }
}

impl LLVMLogger {
    pub fn new(level: LLVMLoggingLevel) -> Option<LLVMLogger> {
        let (bytecode, conversions) = if let LLVMLoggingLevel::Verbose = level {
            let cwd = std::env::current_dir().unwrap();
            if !(cwd.exists() && cwd.is_dir()) {
                return None;
            }
            let resolve_path = |path: &str| {
                let mut root_path = cwd.clone();
                root_path.push(path);
                if root_path.exists() {
                    std::fs::remove_file(&root_path).unwrap();
                }
                root_path
            };
            let bytecode_path = resolve_path("llvm_bc.csv");
            let bytecode_logs_file =
                OpenOptions::new().create(true).append(true).open(bytecode_path).unwrap();
            let conversion_logs_path = resolve_path("llvm_conversions.csv");
            let conversion_logs_file =
                OpenOptions::new().create(true).append(true).open(conversion_logs_path).unwrap();
            (Some(bytecode_logs_file), Some(conversion_logs_file))
        } else {
            (None, None)
        };
        Some(LLVMLogger { bytecode, conversions, flags: LLVMFlags::new() })
    }

    #[inline(always)]
    pub fn log_bytecode(&mut self, path: &Path, status: Result<(), LLVMString>) {
        if let Some(file) = &mut self.bytecode {
            let failed = u8::from(status.is_err());
            let message = status.as_ref().err().map(|e| e.to_string()).unwrap_or("".to_string());
            let entry = format!("{},{},{}\n", path.to_string_lossy(), failed, message);

            file.write_all(entry.as_bytes()).unwrap();
            file.flush().unwrap();
        }
    }
    #[inline(always)]
    pub fn log_llvm_conversion(
        &mut self,
        from: TyAndLayout<'_, Ty<'_>>,
        single_field_dereferenced: bool,
    ) {
        self.flags.scalar_pair_expansion.set(true);
        if let Some(file) = &mut self.conversions {
            let source_type = from.ty.to_string();
            let source_type_size = from.size.bytes();
            let line = format!("\"{source_type}\",{source_type_size},{single_field_dereferenced}");
            file.write_all(line.as_bytes()).unwrap();
            file.write_all(b"\n").unwrap();
            file.flush().unwrap();
        }
    }
}

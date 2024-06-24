use crate::eval::LLVMLoggingLevel;
use core::cell::UnsafeCell;
use inkwell::support::LLVMString;
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::ty::Ty;
use rustc_target::abi::TyAndLayout;
use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
};
use tracing::debug;

pub struct LLVMLogger {
    bytecode: Option<File>,
    conversions: Option<File>,
    flags: UnsafeCell<File>,
    visited: UnsafeCell<FxHashSet<LLVMFlag>>,
}

#[derive(Clone, Copy, Eq, Hash, PartialEq, Debug)]
pub enum LLVMFlag {
    EnumOfNonNullablePointer,
    ADTAsPointerFromRust,
    LLVMEngaged,
    LLVMIntToPtr,
    LLVMPtrToInt,
    LLVMOnResolve,
    LLVMMultithreading,
    SizeBasedTypeInference,
    LLVMReadUninit,
    LLVMInvokedConstructor,
    LLVMInvokedDestructor,
    Expansion,
    ExposedPointerFromRustAtBoundary,
    CastPointerFromLLVMAtBoundary,
    AggregateAsBytes,
    VarArgFunction,
    SizeMismatchInShim
}

impl std::fmt::Display for LLVMFlag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = match self {
            LLVMFlag::LLVMEngaged => "LLVMEngaged",
            LLVMFlag::LLVMIntToPtr => "LLVMIntToPtr",
            LLVMFlag::LLVMPtrToInt => "LLVMPtrToInt",
            LLVMFlag::LLVMOnResolve => "LLVMOnResolve",
            LLVMFlag::LLVMMultithreading => "LLVMMultithreading",
            LLVMFlag::SizeBasedTypeInference => "SizeBasedTypeInference",
            LLVMFlag::LLVMReadUninit => "LLVMReadUninit",
            LLVMFlag::LLVMInvokedConstructor => "LLVMInvokedConstructor",
            LLVMFlag::LLVMInvokedDestructor => "LLVMInvokedDestructor",
            LLVMFlag::Expansion => "Expansion",
            LLVMFlag::ExposedPointerFromRustAtBoundary => "ExposedPointerFromRustAtBoundary",
            LLVMFlag::CastPointerFromLLVMAtBoundary => "CastPointerFromLLVMAtBoundary",
            LLVMFlag::AggregateAsBytes => "AggregateAsBytes",
            LLVMFlag::VarArgFunction => "VarArgFunction",
            LLVMFlag::EnumOfNonNullablePointer => "EnumOfNonNullablePointer",
            LLVMFlag::ADTAsPointerFromRust => "ADTAsPointerFromRust",
            LLVMFlag::SizeMismatchInShim => "SizeMismatchInShim",
        };
        write!(f, "{}", string)
    }
}

impl LLVMLogger {
    pub fn log_flag(&self, flag: LLVMFlag) {
        debug!("Logging flag: {}", flag);
        let visited = unsafe { &mut *self.visited.get() };
        if !visited.contains(&flag) {
            visited.insert(flag);
            let file = unsafe { &mut *self.flags.get() };
            let line = format!("{}", flag);
            file.write_all(line.as_bytes()).unwrap();
            file.write_all(b"\n").unwrap();
            file.flush().unwrap();
        }
    }

    pub fn new(level: LLVMLoggingLevel) -> Option<LLVMLogger> {
        let cwd = std::env::current_dir().unwrap();
        if !(cwd.exists() && cwd.is_dir()) {
            return None;
        }
        let resolve_path = |path: &str| {
            let mut root_path = cwd.clone();
            root_path.push(path);
            root_path
        };
        let (bytecode, conversions) = if let LLVMLoggingLevel::Verbose = level {
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
        let flags = UnsafeCell::new(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(resolve_path("./llvm_flags.csv"))
                .unwrap(),
        );
        let visited = UnsafeCell::new(FxHashSet::default());
        Some(LLVMLogger { bytecode, conversions, flags, visited })
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
    pub fn log_aggregate_expansion(&mut self, from: TyAndLayout<'_, Ty<'_>>) {
        self.log_flag(LLVMFlag::Expansion);
        if let Some(file) = &mut self.conversions {
            let source_type = from.ty.to_string();
            let source_type_size = from.size.bytes();
            let line = format!("\"{source_type}\",{source_type_size}");
            file.write_all(line.as_bytes()).unwrap();
            file.write_all(b"\n").unwrap();
            file.flush().unwrap();
        }
    }
}

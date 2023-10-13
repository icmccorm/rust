use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
};

use inkwell::{support::LLVMString, types::BasicTypeEnum};
use rustc_middle::ty::Ty;
use rustc_target::abi::TyAndLayout;

use crate::ThreadId;

use super::threads::link::ThreadLinkDestination;

pub struct LLVMLogger {
    bytecode: File,
    calls: File,
    conversions: File,
}

impl LLVMLogger {
    pub fn new() -> Option<LLVMLogger> {
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
        let call_logs_path = resolve_path("llvm_calls.csv");
        let call_logs_file =
            OpenOptions::new().create(true).append(true).open(call_logs_path).unwrap();
        let conversion_logs_path = resolve_path("llvm_conversions.csv");
        let conversion_logs_file =
            OpenOptions::new().create(true).append(true).open(conversion_logs_path).unwrap();
        Some(LLVMLogger {
            bytecode: bytecode_logs_file,
            calls: call_logs_file,
            conversions: conversion_logs_file,
        })
    }

    pub fn log_bytecode(&mut self, path: &Path, status: Result<(), LLVMString>) {
        let failed = u8::from(status.is_err());
        let message = status.as_ref().err().map(|e| e.to_string()).unwrap_or("".to_string());
        let entry = format!("{},{},{}\n", path.to_string_lossy(), failed, message);
        let file = &mut self.bytecode;
        file.write_all(entry.as_bytes()).unwrap();
        file.flush().unwrap();
    }

    pub fn log_llvm_call(
        &mut self,
        function_name: &str,
        thread_group_id: ThreadId,
        link: Option<&ThreadLinkDestination<'_>>,
    ) {
        let function_origin = match link {
            Some(ThreadLinkDestination::ToLLI(_)) => "rust",
            Some(_) => "llvm",
            None => "shim",
        };
        let thread_id = thread_group_id.to_u32();

        let file = &mut self.calls;
        let line = format!("{thread_id},{function_name},{function_origin}");
        file.write_all(line.as_bytes()).unwrap();
        file.write_all(b"\n").unwrap();
        file.flush().unwrap();
    }

    pub fn log_llvm_conversion(
        &mut self,
        from: TyAndLayout<'_, Ty<'_>>,
        single_field_dereferenced: bool,
        llvm_field_type: &BasicTypeEnum<'_>,
    ) {
        let file = &mut self.conversions;
        let source_type = from.ty.to_string();
        let source_type_size = from.size.bytes();
        let line = format!(
            "\"{source_type}\",{source_type_size},{single_field_dereferenced},\"{llvm_field_type}\""
        );
        file.write_all(line.as_bytes()).unwrap();
        file.write_all(b"\n").unwrap();
        file.flush().unwrap();
    }
}

# Feature List

* Implement text-based flags and immediate CSV logging

* Implement clean termination from LLI

* Implement global timeout after loading LLVM?

* (done) Copy provenance in va_arg()

* (done) Correct behavior of getelementptr to only copy provenance when the offset lies within the bounds of the first allocation

* (done) Fix generic value conversion to automatically derererence ADTs with singular fields to their innermost type 

* (done) switch "can_dereference_into_singular_field" to return true for types that are ADTs with singular fields

* (TODO) implement variable args passed to shims

* (done) weakened requirements for llvm_malloc to allow retuning NULL pointers for zero-sized allocations.

* (done) limited updates to borrow state to the range being read or written to when accessed in LLVM

//! Compiler-assigned symbols; no unsafe export attributes or library changes.
#![forbid(unsafe_code)]
include!(concat!(env!("OUT_DIR"), "/replacement_probes.rs"));

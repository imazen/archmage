//! Public policy code generation probes.
#![forbid(unsafe_code)]
include!(concat!(env!("OUT_DIR"), "/policy_probes.rs"));

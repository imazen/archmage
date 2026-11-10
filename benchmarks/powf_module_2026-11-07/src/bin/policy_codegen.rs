//! Public policy code generation probes.
#![forbid(unsafe_code)]
mod highp_reference {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/highp_reference.rs"));
}
include!(concat!(env!("OUT_DIR"), "/policy_probes.rs"));

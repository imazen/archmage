// error: mismatched types
#![forbid(unsafe_code)]
#[artisanal_dispatch::bridge(v3)] fn raw() {}
pub fn f(t: artisanal_dispatch::V2) { raw_t(t); }

// error: mismatched types
#![forbid(unsafe_code)]
struct V3;
#[artisanal_dispatch::bridge(v3)] fn raw() {}
pub fn f() { raw_t(V3); }

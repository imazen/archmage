// error: error[E0133]
#![forbid(unsafe_code)]
#[artisanal_dispatch::tier(v3)] fn raw() {}
pub fn f() { raw(); }

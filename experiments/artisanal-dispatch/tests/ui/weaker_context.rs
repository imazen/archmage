// error: error[E0133]
#![forbid(unsafe_code)]
#[artisanal_dispatch::tier(v3)] fn raw() {}
#[artisanal_dispatch::tier(v2)] fn call() { raw(); }

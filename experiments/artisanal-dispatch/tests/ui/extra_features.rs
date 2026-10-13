// error: unsupported attribute on a feature body
#![forbid(unsafe_code)]
#[artisanal_dispatch::bridge(v3)] #[target_feature(enable="avx512f")] fn raw() {}

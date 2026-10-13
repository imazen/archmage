// error: unsupported attribute on a feature body
#![forbid(unsafe_code)]
#[target_feature(enable="avx512f")] #[artisanal_dispatch::bridge(v3)] fn work() {}

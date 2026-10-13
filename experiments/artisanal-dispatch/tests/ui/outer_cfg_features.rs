// error: unsupported attribute on a feature body
#![forbid(unsafe_code)]
#[cfg_attr(all(), target_feature(enable="avx512f"))] #[artisanal_dispatch::bridge(v3)] fn work() {}

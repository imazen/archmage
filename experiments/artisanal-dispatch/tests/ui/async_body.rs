// error: requires a safe, synchronous Rust function
#![forbid(unsafe_code)]
#[artisanal_dispatch::bridge(v3)] async fn raw() {}

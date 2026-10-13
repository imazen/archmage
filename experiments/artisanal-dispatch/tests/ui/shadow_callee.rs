// error: unsupported or reserved bridge argument
#![forbid(unsafe_code)]
#[artisanal_dispatch::bridge(v3)] fn raw(raw: unsafe fn()) {}

// error: bridge arguments must be identifiers
#![forbid(unsafe_code)]
#[artisanal_dispatch::bridge(v3)] fn raw((x,y): (u32,u32)) {}

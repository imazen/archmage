// error: only free functions are supported
#![forbid(unsafe_code)]
struct K; impl K { #[artisanal_dispatch::bridge(v3)] fn work(&self) {} }

// error: `const` items in this context need a name
#![forbid(unsafe_code)]
fn work() {} struct K; impl K { #[artisanal_dispatch::bridge(v3)] fn work() {} }

// error: `const` items in this context need a name
#![forbid(unsafe_code)]
trait K { #[artisanal_dispatch::bridge(scalar)] fn work() {} }

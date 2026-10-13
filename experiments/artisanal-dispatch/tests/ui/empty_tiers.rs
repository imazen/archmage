// error: one final scalar fallback
#![forbid(unsafe_code)]
pub fn f() { artisanal_dispatch::dispatch!(work();); }

// error: not inline(always)
#![forbid(unsafe_code)]
#[artisanal_dispatch::tier(v3)] #[inline(always)] fn raw() {}

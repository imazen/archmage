#![forbid(unsafe_code)]
#[archmage::attune(make(_v3_t))] fn work(x: u32) -> u32 { x }
fn main() { let _ = archmage::attuned!(work(1), [_v3]); }

#![forbid(unsafe_code)]
#[archmage::attune(make(_v3_t))] fn work(x: u32) -> u32 { x }
#[archmage::attune(v2)] fn caller() -> u32 { archmage::reattune!(work(1), [_v3]) }
fn main() {}

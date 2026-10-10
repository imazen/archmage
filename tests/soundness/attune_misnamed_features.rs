#![forbid(unsafe_code)]
#[archmage::attune(v3)] fn work_v2(x: u32) -> u32 { x }
#[archmage::attune(v2)] fn caller() -> u32 { archmage::attuned!(work(1), [_v2]) }
fn main() {}

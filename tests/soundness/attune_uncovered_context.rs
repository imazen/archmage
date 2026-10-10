#![forbid(unsafe_code)]
#[archmage::attune(v3)] fn work_v3(x: u32) -> u32 { x }
#[archmage::attune(v2)] fn caller() -> u32 { archmage::attuned!(work(1), [_v3]) }
fn main() {}

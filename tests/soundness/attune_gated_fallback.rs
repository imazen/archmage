#![forbid(unsafe_code)]
#[archmage::attune(make(_v3(avx512)))] fn work(x: u32) -> u32 { x }
#[archmage::attune(v3)] fn caller() -> u32 { archmage::attuned!(work(1), [_v3(avx512)]) }
fn main() {}

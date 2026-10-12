#![forbid(unsafe_code)]
#[archmage::attune(make(_v3_t))] fn work(x: u32) -> u32 { x }
#[archmage::attune(v3)] fn caller() -> u32 {
    fn nested() -> u32 { archmage::attuned!(work(1), [_v3]) }
    nested()
}
fn main() {}

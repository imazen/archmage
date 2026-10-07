#[target_feature(enable = "avx2")]
#[inline(always)]
pub fn body(x: u32) -> u32 { x + 1 }
fn main() {}

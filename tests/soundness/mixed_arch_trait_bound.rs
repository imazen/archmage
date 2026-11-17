// REGRESSION CASE: combining tier traits from conflicting CPU architectures on
// a single `#[arcane]` function must fail with an explicit architecture error.
#![forbid(unsafe_code)]
use archmage::{HasNeon, HasX64V2, arcane};

#[arcane]
fn impossible<T: HasX64V2 + HasNeon>(_token: T) -> u32 {
    0
}

fn main() {}

// An #[autoversion] body is the author's code: its nested items keep the
// author's lint levels in the SIMD variants (only forwarding wrappers and the
// scalar fallback carry an allow).
#![deny(dead_code)]
use archmage::prelude::*;

#[autoversion]
pub fn total(data: &[f32]) -> f32 {
    fn nested_unused() {}
    data.iter().sum()
}

fn main() {
    let _ = total(&[1.0]);
}

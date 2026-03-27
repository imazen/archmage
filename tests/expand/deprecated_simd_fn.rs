// #[simd_fn] — deprecated alias for #[arcane]
#[allow(deprecated)]
use archmage::{simd_fn, X64V3Token};

#[allow(deprecated)]
#[simd_fn]
fn legacy_process(token: X64V3Token, a: f32, b: f32) -> f32 {
    a + b
}

fn main() {}

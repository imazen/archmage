// simd_route! — deprecated alias for incant!
#[allow(deprecated)]
use archmage::{arcane, simd_route, X64V3Token, ScalarToken};

#[arcane]
fn compute_v3(_token: X64V3Token, x: f32) -> f32 {
    x * x
}

fn compute_scalar(_token: ScalarToken, x: f32) -> f32 {
    x * x
}

#[allow(deprecated)]
fn dispatch(x: f32) -> f32 {
    simd_route!(compute(x), [v3, scalar])
}

fn main() {}

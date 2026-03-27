// dispatch_variant! — descriptive alias for incant!
use archmage::{arcane, dispatch_variant, X64V3Token, ScalarToken};

#[arcane]
fn compute_v3(_token: X64V3Token, x: f32) -> f32 {
    x * x
}

fn compute_scalar(_token: ScalarToken, x: f32) -> f32 {
    x * x
}

fn dispatch(x: f32) -> f32 {
    dispatch_variant!(compute(x), [v3, scalar])
}

fn main() {}

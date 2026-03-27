// incant! with feature-gated tier
use archmage::{arcane, incant, X64V3Token, ScalarToken};

#[arcane]
fn work_v3(_token: X64V3Token, x: f32) -> f32 { x * 2.0 }

fn work_scalar(_token: ScalarToken, x: f32) -> f32 { x * 2.0 }

fn dispatch(x: f32) -> f32 {
    incant!(work(x), [v3(cfg(avx_opt)), scalar])
}

fn main() {}

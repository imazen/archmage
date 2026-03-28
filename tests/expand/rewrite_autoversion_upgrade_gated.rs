// incant! rewriting: autoversion V3 caller, V4 upgrade with cfg(avx512)
use archmage::{incant, autoversion, arcane, X64V3Token, X64V4Token, ScalarToken};

#[arcane]
fn inner_v4(_token: X64V4Token, x: f32) -> f32 { x * 4.0 }

#[arcane]
fn inner_v3(_token: X64V3Token, x: f32) -> f32 { x * 2.0 }

fn inner_scalar(_token: ScalarToken, x: f32) -> f32 { x * 2.0 }

#[autoversion(v3, scalar)]
fn outer(x: f32) -> f32 {
    incant!(inner(x), [v4(cfg(avx512)), v3, scalar])
}

fn main() {}

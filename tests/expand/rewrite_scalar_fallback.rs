// incant! rewriting: caller's arch doesn't match any callee tier — scalar fallback
use archmage::{arcane, X64V3Token, NeonToken, ScalarToken};

#[arcane]
fn inner_neon(_token: NeonToken, x: f32) -> f32 { x * 2.0 }

fn inner_scalar(_token: ScalarToken, x: f32) -> f32 { x * 2.0 }

#[arcane]
fn outer(token: X64V3Token, x: f32) -> f32 {
    // Callee only has neon + scalar — x86 caller gets scalar
    incant!(inner(x), [neon, scalar])
}

fn main() {}

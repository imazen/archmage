// incant! rewriting: rite multi-tier, each variant gets its own rewrite
use archmage::{incant, rite, arcane, X64V3Token, NeonToken, ScalarToken};

#[arcane]
fn inner_v3(_token: X64V3Token, x: f32) -> f32 { x * 2.0 }

#[arcane]
fn inner_neon(_token: NeonToken, x: f32) -> f32 { x * 2.0 }

fn inner_scalar(_token: ScalarToken, x: f32) -> f32 { x * 2.0 }

#[rite(v3, neon)]
fn outer(x: f32) -> f32 {
    incant!(inner(x), [v3, neon, scalar])
}

fn main() {}

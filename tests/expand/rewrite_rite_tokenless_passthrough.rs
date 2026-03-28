// incant! in tokenless #[rite(v3)] — should NOT be rewritten (no token to pass)
// incant! expands normally as a dispatcher
use archmage::{rite, incant, arcane, X64V3Token, ScalarToken};

#[arcane]
fn inner_v3(_token: X64V3Token, x: f32) -> f32 { x * 2.0 }

fn inner_scalar(_token: ScalarToken, x: f32) -> f32 { x * 2.0 }

#[rite(v3)]
fn outer(x: f32) -> f32 {
    incant!(inner(x), [v3, scalar])
}

fn main() {}

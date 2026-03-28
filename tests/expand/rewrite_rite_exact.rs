// incant! rewriting: rite with token, exact tier match
use archmage::{rite, arcane, X64V3Token, ScalarToken};

#[arcane]
fn inner_v3(_token: X64V3Token, x: f32) -> f32 { x * 2.0 }

fn inner_scalar(_token: ScalarToken, x: f32) -> f32 { x * 2.0 }

#[rite]
fn outer(token: X64V3Token, x: f32) -> f32 {
    incant!(inner(x), [v3, scalar])
}

fn main() {}

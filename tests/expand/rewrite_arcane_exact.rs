// incant! rewriting: arcane caller, exact tier match
use archmage::{arcane, X64V3Token, ScalarToken};

#[arcane]
fn inner_v3(_token: X64V3Token, x: f32) -> f32 { x * 2.0 }

fn inner_scalar(_token: ScalarToken, x: f32) -> f32 { x * 2.0 }

#[arcane]
fn outer(token: X64V3Token, x: f32) -> f32 {
    incant!(inner(x), [v3, scalar])
}

fn main() {}

// incant! rewriting: autoversion caller, exact tier match
// V3 body calling V3 variant — direct call, no summon
use archmage::{autoversion, arcane, incant, X64V3Token, ScalarToken};

#[arcane]
fn inner_v3(_token: X64V3Token, x: f32) -> f32 { x * 2.0 }

fn inner_scalar(_token: ScalarToken, x: f32) -> f32 { x * 2.0 }

#[autoversion(v3, scalar)]
fn outer(x: f32) -> f32 {
    incant!(inner(x), [v3, scalar])
}

fn main() {}

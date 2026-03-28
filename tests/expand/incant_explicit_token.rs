// incant! with explicit Token marker — token placed where user specifies
use archmage::{arcane, incant, X64V3Token, ScalarToken};

// Callee has token LAST
#[arcane]
fn inner_v3(x: f32, _token: X64V3Token) -> f32 { x * 2.0 }

fn inner_scalar(x: f32, _token: ScalarToken) -> f32 { x * 2.0 }

fn dispatch(x: f32) -> f32 {
    // Token marker at the end — matches callee signature
    incant!(inner(x, Token), [v3, scalar])
}

fn main() {}

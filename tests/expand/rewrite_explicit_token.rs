// incant! rewriting with explicit Token marker — token last in callee
use archmage::{arcane, X64V3Token, ScalarToken};

// Callee has token LAST
#[arcane]
fn inner_v3(x: f32, _token: X64V3Token) -> f32 { x * 2.0 }

fn inner_scalar(x: f32, _token: ScalarToken) -> f32 { x * 2.0 }

// Caller: #[arcane] with token, body uses incant! with explicit Token position
#[arcane]
fn outer(token: X64V3Token, x: f32) -> f32 {
    incant!(inner(x, Token), [v3, scalar])
}

fn main() {}

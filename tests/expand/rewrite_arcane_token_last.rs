// incant! rewriting where the CALLER has token last, but callees follow
// incant! convention (token first). The rewriter passes the caller's token
// in the first position, matching the callee's expected signature.
use archmage::{arcane, X64V3Token, ScalarToken};

#[arcane]
fn inner_v3(_token: X64V3Token, x: f32) -> f32 { x * 2.0 }

fn inner_scalar(_token: ScalarToken, x: f32) -> f32 { x * 2.0 }

// Caller has token last — that's fine for #[arcane], the token position
// in the CALLER doesn't affect how incant! calls the CALLEE
#[arcane]
fn outer(data: f32, token: X64V3Token) -> f32 {
    incant!(inner(data), [v3, scalar])
}

fn main() {}

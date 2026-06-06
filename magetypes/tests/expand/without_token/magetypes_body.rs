// Tests: `incant!(.. without token)` inside a #[magetypes] body rewrites to the
// per-tier tokenless variant call — including the scalar variant (which has no
// #[arcane] wrapper, so #[magetypes] itself resolves `without token` there).
use archmage::{incant, magetypes, rite};

#[rite(v3, scalar)]
fn dbl(x: f32) -> f32 {
    x * 2.0
}

#[magetypes(v3, scalar)]
fn run(token: Token, x: f32) -> f32 {
    let _ = token;
    incant!(dbl(x) without token) + 1.0
}

fn main() {}

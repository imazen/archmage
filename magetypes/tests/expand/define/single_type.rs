// Tests: #[magetypes(define(f32x8), ...)] injects a single type alias.
use archmage::magetypes;

#[magetypes(define(f32x8), v3, scalar)]
fn kernel(token: Token, data: &[f32; 8]) -> f32 {
    f32x8::load(token, data).reduce_add()
}

fn main() {}

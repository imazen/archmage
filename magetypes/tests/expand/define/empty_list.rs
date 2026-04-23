// Tests: define() with no types is accepted (no-op). Lets users comment out
// items without syntax errors.
use archmage::magetypes;

#[magetypes(define(), v3, scalar)]
fn kernel(_token: Token, x: f32) -> f32 {
    x * 2.0
}

fn main() {}

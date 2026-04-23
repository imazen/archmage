// Tests: define(...) between tier names in the attr list is accepted.
use archmage::magetypes;

#[magetypes(v3, define(f32x8), scalar)]
fn kernel(token: Token, data: &[f32; 8]) -> f32 {
    f32x8::load(token, data).reduce_add()
}

fn main() {}

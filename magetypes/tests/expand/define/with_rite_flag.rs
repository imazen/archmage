// Tests: define(...) combined with the rite flag — aliases injected into
// rite-flavored variants (direct #[target_feature] + #[inline], no trampoline).
use archmage::magetypes;

#[magetypes(rite, define(f32x8), v3, scalar)]
fn kernel(token: Token, data: &[f32; 8]) -> f32 {
    f32x8::load(token, data).reduce_add()
}

fn main() {}

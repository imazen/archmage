// Tests: #[magetypes(rite, ...)] emits #[rite]-style variants
// (direct #[target_feature] + #[inline]) instead of arcane wrappers.
use archmage::magetypes;

#[magetypes(rite, v3, scalar)]
fn kernel(token: Token, x: f32) -> f32 {
    let _ = token;
    x * x
}

fn main() {}

// Tests: rite-flavored magetypes with an actual magetypes body (references
// GenericF32x8<Token> etc). Without the rite flag this would wrap in arcane;
// with rite, each variant is a bare #[target_feature] fn.
use archmage::magetypes;
use magetypes::simd::generic::f32x8 as GenericF32x8;

#[magetypes(rite, v3, scalar)]
fn kernel(token: Token, data: &[f32; 8]) -> f32 {
    type Vec8 = GenericF32x8<Token>;
    Vec8::load(token, data).reduce_add()
}

fn main() {}

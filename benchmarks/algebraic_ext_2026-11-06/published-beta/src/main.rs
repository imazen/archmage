#![forbid(unsafe_code)]
use algebraic_magetypes_prototype::AlgebraicOps;
use archmage::SimdToken;
use magetypes::simd::generic::f32x8;
fn main() {
    let t = archmage::ScalarToken::summon().unwrap();
    let a = f32x8::splat_t(t, 6.0);
    let b = f32x8::splat_t(t, 2.0);
    assert_eq!(a.algebraic_add(b).to_array(), [8.0; 8]);
    assert_eq!(a.algebraic_sub(b).to_array(), [4.0; 8]);
    assert_eq!(a.algebraic_mul(b).to_array(), [12.0; 8]);
    assert_eq!(a.algebraic_div(b).to_array(), [3.0; 8]);
    assert_eq!(a.algebraic_rem(b).to_array(), [0.0; 8]);
}

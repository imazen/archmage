use archmage::{rite, arcane, incant, X64V3Token, NeonToken, ScalarToken};
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_inner_v3(_t: X64V3Token, x: f32) -> f32 {
    x * 2.0
}
#[inline(always)]
fn inner_v3(_t: X64V3Token, x: f32) -> f32 {
    {
        fn __archmage_verify(_: &::archmage::X64V3Token) {}
        __archmage_verify(&_t);
    }
    unsafe { __arcane_inner_v3(_t, x) }
}
fn inner_scalar(_t: ScalarToken, x: f32) -> f32 {
    x
}
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn outer_v3(x: f32) -> f32 {
    '__incant: {
        use archmage::SimdToken;
        {
            if let Some(__t) = archmage::X64V3Token::summon() {
                break '__incant inner_v3(__t, x);
            }
        }
        inner_scalar(archmage::ScalarToken, x)
    }
}
fn main() {}

use archmage::prelude::*;
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_inner_v3(token: archmage::X64V3Token, x: f32) -> f32 {
    let _ = token;
    x
}
#[inline(always)]
fn inner_v3(token: archmage::X64V3Token, x: f32) -> f32 {
    let _: () = <archmage::X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
    unsafe { __arcane_inner_v3(token, x) }
}
fn inner_scalar(token: archmage::ScalarToken, x: f32) -> f32 {
    let _ = token;
    x
}
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_outer_v3(token: archmage::X64V3Token, x: f32) -> f32 {
    inner_v3(token, x)
}
#[inline(always)]
fn outer_v3(token: archmage::X64V3Token, x: f32) -> f32 {
    let _: () = <archmage::X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
    unsafe { __arcane_outer_v3(token, x) }
}
fn outer_scalar(token: archmage::ScalarToken, x: f32) -> f32 {
    '__incant: {
        #[allow(unused_imports)]
        use archmage::SimdToken;
        {
            if let Some(__t) = archmage::X64V3Token::summon() {
                break '__incant inner_v3(__t, x);
            }
        }
        inner_scalar(archmage::ScalarToken, x)
    }
}
pub fn api(x: f32) -> f32 {
    '__incant: {
        #[allow(unused_imports)]
        use archmage::SimdToken;
        {
            if let Some(__t) = archmage::X64V3Token::summon() {
                break '__incant outer_v3(__t, x);
            }
        }
        outer_scalar(archmage::ScalarToken, x)
    }
}
fn main() {}

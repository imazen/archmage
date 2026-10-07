use archmage::prelude::*;
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
)]
#[inline]
fn __arcane_fast_v4(_t: X64V4Token, x: f32) -> f32 {
    x * 4.0
}
#[inline(always)]
fn fast_v4(_t: X64V4Token, x: f32) -> f32 {
    let _: () = <X64V4Token>::__ARCHMAGE_ASSERT_TIER_FE1B900C;
    unsafe { __arcane_fast_v4(_t, x) }
}
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_fast_v3(_t: X64V3Token, x: f32) -> f32 {
    x * 2.0
}
#[inline(always)]
fn fast_v3(_t: X64V3Token, x: f32) -> f32 {
    let _: () = <X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
    unsafe { __arcane_fast_v3(_t, x) }
}
fn fast_scalar(_t: ScalarToken, x: f32) -> f32 {
    x
}
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_v3_with_upgrade(t: X64V3Token, x: f32) -> f32 {
    '__incant_rewrite: {
        use archmage::SimdToken;
        if let Some(__t) = archmage::X64V4Token::summon() {
            break '__incant_rewrite fast_v4(__t, x);
        }
        fast_v3(t, x)
    }
}
#[inline(always)]
fn v3_with_upgrade(t: X64V3Token, x: f32) -> f32 {
    let _: () = <X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
    unsafe { __arcane_v3_with_upgrade(t, x) }
}
fn main() {}

use archmage::{incant, autoversion, arcane, X64V3Token, X64V4Token, ScalarToken};
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
)]
#[inline]
fn __arcane_inner_v4(_token: X64V4Token, x: f32) -> f32 {
    x * 4.0
}
#[inline(always)]
fn inner_v4(_token: X64V4Token, x: f32) -> f32 {
    unsafe { __arcane_inner_v4(_token, x) }
}
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_inner_v3(_token: X64V3Token, x: f32) -> f32 {
    x * 2.0
}
#[inline(always)]
fn inner_v3(_token: X64V3Token, x: f32) -> f32 {
    unsafe { __arcane_inner_v3(_token, x) }
}
fn inner_scalar(_token: ScalarToken, x: f32) -> f32 {
    x * 2.0
}
fn outer(x: f32) -> f32 {
    use archmage::SimdToken;
    {
        if let Some(__t) = archmage::X64V3Token::summon() {
            return outer_v3(__t, x);
        }
    }
    outer_scalar(archmage::ScalarToken, x)
}
#[doc(hidden)]
#[allow(dead_code)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_outer_v3(_token: archmage::X64V3Token, x: f32) -> f32 {
    '__incant_rewrite: {
        use archmage::SimdToken;
        inner_v3(_token, x)
    }
}
#[allow(dead_code)]
#[inline(always)]
fn outer_v3(_token: archmage::X64V3Token, x: f32) -> f32 {
    unsafe { __arcane_outer_v3(_token, x) }
}
#[allow(dead_code)]
fn outer_scalar(_token: archmage::ScalarToken, x: f32) -> f32 {
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

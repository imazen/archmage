use archmage::{arcane, incant, X64V3Token, X64V4Token, NeonToken, ScalarToken};
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
)]
#[inline]
fn __arcane_inner_v4(_t: X64V4Token, x: f32) -> f32 {
    x * 4.0
}
#[inline(always)]
fn inner_v4(_t: X64V4Token, x: f32) -> f32 {
    const _ARCHMAGE_TOKEN_MISMATCH: () = [
        (),
    ][!(<X64V4Token>::__ARCHMAGE_TIER_TAG == 4263219212u32) as usize];
    unsafe { __arcane_inner_v4(_t, x) }
}
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
    const _ARCHMAGE_TOKEN_MISMATCH: () = [
        (),
    ][!(<X64V3Token>::__ARCHMAGE_TIER_TAG == 4085983307u32) as usize];
    unsafe { __arcane_inner_v3(_t, x) }
}
fn inner_scalar(_t: ScalarToken, x: f32) -> f32 {
    x
}
fn dispatch(x: f32) -> f32 {
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

use archmage::{arcane, X64V3CryptoToken, X64V4Token, ScalarToken};
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,vpclmulqdq,vaes"
)]
#[inline]
fn __arcane_inner_v3_crypto(_t: X64V3CryptoToken, x: f32) -> f32 {
    x * 3.0
}
#[inline(always)]
fn inner_v3_crypto(_t: X64V3CryptoToken, x: f32) -> f32 {
    const _ARCHMAGE_TOKEN_MISMATCH: () = [
        (),
    ][!(<X64V3CryptoToken>::__ARCHMAGE_TIER_TAG == 32171784u32) as usize];
    unsafe { __arcane_inner_v3_crypto(_t, x) }
}
fn inner_scalar(_t: ScalarToken, x: f32) -> f32 {
    x
}
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
)]
#[inline]
fn __arcane_outer(token: X64V4Token, x: f32) -> f32 {
    '__incant_rewrite: {
        use archmage::SimdToken;
        if let Some(__t) = archmage::X64V3CryptoToken::summon() {
            break '__incant_rewrite inner_v3_crypto(__t, x);
        }
        inner_scalar(archmage::ScalarToken, x)
    }
}
#[inline(always)]
fn outer(token: X64V4Token, x: f32) -> f32 {
    const _ARCHMAGE_TOKEN_MISMATCH: () = [
        (),
    ][!(<X64V4Token>::__ARCHMAGE_TIER_TAG == 4263219212u32) as usize];
    unsafe { __arcane_outer(token, x) }
}
fn main() {}

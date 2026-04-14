use archmage::{arcane, X64V3Token, NeonToken, ScalarToken};
fn inner_scalar(_t: ScalarToken, x: f32) -> f32 {
    x
}
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_outer(token: X64V3Token, x: f32) -> f32 {
    inner_scalar(archmage::ScalarToken, x)
}
#[inline(always)]
fn outer(token: X64V3Token, x: f32) -> f32 {
    const _ARCHMAGE_TOKEN_MISMATCH: () = [
        (),
    ][!(<X64V3Token>::__ARCHMAGE_TIER_TAG == 4085983307u32) as usize];
    unsafe { __arcane_outer(token, x) }
}
fn main() {}

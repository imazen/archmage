use archmage::{incant, magetypes, rite};
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn dbl_v3(x: f32) -> f32 {
    x * 2.0
}
#[inline]
fn dbl_scalar(x: f32) -> f32 {
    x * 2.0
}
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_run_v3(token: archmage::X64V3Token, x: f32) -> f32 {
    let _ = token;
    dbl_v3(x) + 1.0
}
#[inline(always)]
fn run_v3(token: archmage::X64V3Token, x: f32) -> f32 {
    const _ARCHMAGE_TOKEN_MISMATCH: () = [
        (),
    ][!(<archmage::X64V3Token>::__ARCHMAGE_TIER_TAG == 4085983307u32) as usize];
    unsafe { __arcane_run_v3(token, x) }
}
fn run_scalar(token: archmage::ScalarToken, x: f32) -> f32 {
    let _ = token;
    dbl_scalar(x) + 1.0
}
fn main() {}

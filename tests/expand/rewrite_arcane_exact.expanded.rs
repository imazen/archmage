use archmage::{arcane, X64V3Token, ScalarToken};
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
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_outer(token: X64V3Token, x: f32) -> f32 {
    inner_v3(token, x)
}
#[inline(always)]
fn outer(token: X64V3Token, x: f32) -> f32 {
    unsafe { __arcane_outer(token, x) }
}
fn main() {}

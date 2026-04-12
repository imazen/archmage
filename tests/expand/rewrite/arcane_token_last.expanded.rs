use archmage::{arcane, X64V3Token, ScalarToken};
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_inner_v3(x: f32, _t: X64V3Token) -> f32 {
    x * 2.0
}
#[inline(always)]
fn inner_v3(x: f32, _t: X64V3Token) -> f32 {
    {
        fn __archmage_verify(_: &::archmage::X64V3Token) {}
        __archmage_verify(&_t);
    }
    unsafe { __arcane_inner_v3(x, _t) }
}
fn inner_scalar(x: f32, _t: ScalarToken) -> f32 {
    x
}
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_outer(token: X64V3Token, x: f32) -> f32 {
    inner_v3(x, token)
}
#[inline(always)]
fn outer(token: X64V3Token, x: f32) -> f32 {
    {
        fn __archmage_verify(_: &::archmage::X64V3Token) {}
        __archmage_verify(&token);
    }
    unsafe { __arcane_outer(token, x) }
}
fn main() {}

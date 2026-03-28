use archmage::{arcane, X64V3Token, X64V4Token, ScalarToken};
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
    unsafe { __arcane_inner_v3(_t, x) }
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
    inner_v3(token.v3(), x)
}
#[inline(always)]
fn outer(token: X64V4Token, x: f32) -> f32 {
    unsafe { __arcane_outer(token, x) }
}
fn main() {}

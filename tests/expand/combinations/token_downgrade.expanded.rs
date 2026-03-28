use archmage::{arcane, X64V3Token, X64V4Token};
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_v3_helper(_t: X64V3Token, x: f32) -> f32 {
    x
}
#[inline(always)]
fn v3_helper(_t: X64V3Token, x: f32) -> f32 {
    unsafe { __arcane_v3_helper(_t, x) }
}
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
)]
#[inline]
fn __arcane_v4_caller(token: X64V4Token, x: f32) -> f32 {
    v3_helper(token.v3(), x) + 1.0
}
#[inline(always)]
fn v4_caller(token: X64V4Token, x: f32) -> f32 {
    unsafe { __arcane_v4_caller(token, x) }
}
fn main() {}

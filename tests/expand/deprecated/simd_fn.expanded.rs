#[allow(deprecated)]
use archmage::{simd_fn, X64V3Token};
#[doc(hidden)]
#[allow(deprecated)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_process(token: X64V3Token, a: f32) -> f32 {
    a
}
#[allow(deprecated)]
#[inline(always)]
fn process(token: X64V3Token, a: f32) -> f32 {
    unsafe { __arcane_process(token, a) }
}
fn main() {}

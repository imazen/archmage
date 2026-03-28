use archmage::{arcane, X64V2Token};
#[doc(hidden)]
#[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b")]
#[inline]
fn __arcane_process(token: X64V2Token, a: f32) -> f32 {
    a + 1.0
}
#[inline(always)]
fn process(token: X64V2Token, a: f32) -> f32 {
    unsafe { __arcane_process(token, a) }
}
fn main() {}

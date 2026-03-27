use archmage::{arcane, HasX64V2};
#[doc(hidden)]
#[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b")]
#[inline]
fn __arcane_process_v2(token: impl HasX64V2, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}
#[inline(always)]
fn process_v2(token: impl HasX64V2, data: &[f32; 4]) -> f32 {
    unsafe { __arcane_process_v2(token, data) }
}
fn main() {}

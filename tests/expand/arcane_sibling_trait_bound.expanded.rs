use archmage::{arcane, HasX64V2};
#[doc(hidden)]
#[target_feature(enable = "sse")]
#[target_feature(enable = "sse2")]
#[target_feature(enable = "sse3")]
#[target_feature(enable = "ssse3")]
#[target_feature(enable = "sse4.1")]
#[target_feature(enable = "sse4.2")]
#[target_feature(enable = "popcnt")]
#[target_feature(enable = "cmpxchg16b")]
#[inline]
fn __arcane_process_v2(token: impl HasX64V2, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}
#[inline(always)]
fn process_v2(token: impl HasX64V2, data: &[f32; 4]) -> f32 {
    unsafe { __arcane_process_v2(token, data) }
}
fn main() {}

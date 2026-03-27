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
fn __arcane_process_generic<T: HasX64V2>(token: T, a: f32, b: f32) -> f32 {
    a + b
}
#[inline(always)]
fn process_generic<T: HasX64V2>(token: T, a: f32, b: f32) -> f32 {
    unsafe { __arcane_process_generic::<T>(token, a, b) }
}
fn main() {}

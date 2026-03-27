use archmage::{arcane, HasX64V2};
#[doc(hidden)]
#[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b")]
#[inline]
fn __arcane_process_generic<T: HasX64V2>(token: T, a: f32, b: f32) -> f32 {
    a + b
}
#[inline(always)]
fn process_generic<T: HasX64V2>(token: T, a: f32, b: f32) -> f32 {
    unsafe { __arcane_process_generic::<T>(token, a, b) }
}
fn main() {}

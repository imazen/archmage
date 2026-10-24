use archmage::{arcane, X64V3Token};
#[doc(hidden)]
#[track_caller]
#[expect(unused_variables)]
#[allow(clippy::too_many_arguments)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_process(token: X64V3Token, unused: f32) -> f32 {
    1.0
}
#[track_caller]
#[allow(unused_variables)]
#[allow(clippy::too_many_arguments)]
#[must_use]
#[inline(always)]
fn process(token: X64V3Token, unused: f32) -> f32 {
    let _: () = <X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
    unsafe { __arcane_process(token, unused) }
}
fn main() {}

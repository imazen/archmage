use archmage::{arcane, X64V3Token};
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_process(token: X64V3Token, callback: &dyn Fn(f32) -> f32, x: f32) -> f32 {
    let _ = token;
    callback(x)
}
#[inline(always)]
fn process(token: X64V3Token, callback: &dyn Fn(f32) -> f32, x: f32) -> f32 {
    const _: () = [()][!(<X64V3Token>::__ARCHMAGE_TIER_TAG == 4085983307u32) as usize];
    unsafe { __arcane_process(token, callback, x) }
}
fn main() {}

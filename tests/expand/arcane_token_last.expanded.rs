use archmage::{arcane, X64V3Token};
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_process(data: &[f32; 4], scale: f32, token: X64V3Token) -> f32 {
    let _ = token;
    data.iter().map(|&x| x * scale).sum()
}
#[inline(always)]
fn process(data: &[f32; 4], scale: f32, token: X64V3Token) -> f32 {
    unsafe { __arcane_process(data, scale, token) }
}
fn main() {}

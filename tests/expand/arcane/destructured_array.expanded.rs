use archmage::{arcane, X64V3Token};
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_process(token: X64V3Token, __archmage_arg_0: [f32; 4]) -> f32 {
    let [a, b, c, d]: [f32; 4] = __archmage_arg_0;
    a + b + c + d
}
#[inline(always)]
fn process(token: X64V3Token, __archmage_arg_0: [f32; 4]) -> f32 {
    unsafe { __arcane_process(token, __archmage_arg_0) }
}
fn main() {}

use archmage::prelude::*;
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn probe(__archmage_arg_0: (f32, f32), token: X64V3Token, x: f32) -> f32 {
    let (lo, hi): (f32, f32) = __archmage_arg_0;
    let _ = token;
    lo + hi + x
}
fn main() {}

use archmage::{arcane, X64V3Token};
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_sum_n<const N: usize>(token: X64V3Token, data: &[f32; N]) -> f32 {
    let _ = token;
    let mut s = 0.0f32;
    for &x in data {
        s += x;
    }
    s
}
#[inline(always)]
fn sum_n<const N: usize>(token: X64V3Token, data: &[f32; N]) -> f32 {
    {
        fn __archmage_verify(_: &::archmage::X64V3Token) {}
        __archmage_verify(&token);
    }
    unsafe { __arcane_sum_n::<N>(token, data) }
}
fn main() {}

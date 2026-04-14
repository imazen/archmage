use archmage::{arcane, X64V4Token};
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
)]
#[inline]
fn __arcane_process(token: X64V4Token, a: f32) -> f32 {
    a + 1.0
}
#[inline(always)]
fn process(token: X64V4Token, a: f32) -> f32 {
    const _: () = [()][!(<X64V4Token>::__ARCHMAGE_TIER_TAG == 4263219212u32) as usize];
    unsafe { __arcane_process(token, a) }
}
fn main() {}

use archmage::{arcane, X64V3Token};
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_process(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    #[allow(unused_imports)]
    use archmage::intrinsics::x86_64::*;
    let va = _mm256_loadu_ps(a);
    let vb = _mm256_loadu_ps(b);
    let s = _mm256_add_ps(va, vb);
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, s);
    out
}
#[inline(always)]
fn process(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    unsafe { __arcane_process(token, a, b) }
}
fn main() {}

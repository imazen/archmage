use archmage::{arcane, X64V3Token};
#[doc(hidden)]
#[target_feature(enable = "sse")]
#[target_feature(enable = "sse2")]
#[target_feature(enable = "sse3")]
#[target_feature(enable = "ssse3")]
#[target_feature(enable = "sse4.1")]
#[target_feature(enable = "sse4.2")]
#[target_feature(enable = "popcnt")]
#[target_feature(enable = "cmpxchg16b")]
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[target_feature(enable = "bmi1")]
#[target_feature(enable = "bmi2")]
#[target_feature(enable = "f16c")]
#[target_feature(enable = "lzcnt")]
#[target_feature(enable = "movbe")]
#[inline]
fn __arcane_load_and_add(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    #[allow(unused_imports)]
    use archmage::intrinsics::x86_64::*;
    let va = _mm256_loadu_ps(a);
    let vb = _mm256_loadu_ps(b);
    let sum = _mm256_add_ps(va, vb);
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, sum);
    out
}
#[inline(always)]
fn load_and_add(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    unsafe { __arcane_load_and_add(token, a, b) }
}
fn main() {}

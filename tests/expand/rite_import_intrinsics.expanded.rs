use archmage::{rite, X64V3Token};
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
fn add_vectors(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    #[allow(unused_imports)]
    use archmage::intrinsics::x86_64::*;
    let va = _mm256_loadu_ps(a);
    let vb = _mm256_loadu_ps(b);
    let sum = _mm256_add_ps(va, vb);
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, sum);
    out
}
fn main() {}

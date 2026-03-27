use archmage::{arcane, rite, X64V3Token};
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
fn normalize(v: &[f32; 4]) -> [f32; 4] {
    let mut sum = 0.0f32;
    for &x in v {
        sum += x * x;
    }
    let inv = 1.0 / sum.sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv, v[3] * inv]
}
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
fn __arcane_process(token: X64V3Token, data: &[f32; 4]) -> [f32; 4] {
    let _ = token;
    normalize(data)
}
#[inline(always)]
fn process(token: X64V3Token, data: &[f32; 4]) -> [f32; 4] {
    unsafe { __arcane_process(token, data) }
}
fn main() {}

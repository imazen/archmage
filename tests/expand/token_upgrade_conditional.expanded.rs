use archmage::{arcane, SimdToken, X64V3Token, X64V4Token};
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
#[target_feature(enable = "pclmulqdq")]
#[target_feature(enable = "aes")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512bw")]
#[target_feature(enable = "avx512cd")]
#[target_feature(enable = "avx512dq")]
#[target_feature(enable = "avx512vl")]
#[inline]
fn __arcane_v4_fast_path(_token: X64V4Token, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}
#[inline(always)]
fn v4_fast_path(_token: X64V4Token, data: &[f32; 4]) -> f32 {
    unsafe { __arcane_v4_fast_path(_token, data) }
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
fn __arcane_v3_with_upgrade(_token: X64V3Token, data: &[f32; 4]) -> f32 {
    if let Some(v4) = X64V4Token::summon() {
        v4_fast_path(v4, data)
    } else {
        data.iter().sum()
    }
}
#[inline(always)]
fn v3_with_upgrade(_token: X64V3Token, data: &[f32; 4]) -> f32 {
    unsafe { __arcane_v3_with_upgrade(_token, data) }
}
fn main() {}

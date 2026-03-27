use archmage::{autoversion, ScalarToken};
fn process(token: ScalarToken, data: &[f32; 4]) -> f32 {
    use archmage::SimdToken;
    {
        if let Some(__t) = archmage::X64V4Token::summon() {
            return process_v4(__t, data);
        }
        if let Some(__t) = archmage::X64V3Token::summon() {
            return process_v3(__t, data);
        }
    }
    process_scalar(archmage::ScalarToken, data)
}
#[doc(hidden)]
#[allow(dead_code)]
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
fn __arcane_process_v4(token: archmage::X64V4Token, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}
#[allow(dead_code)]
#[inline(always)]
fn process_v4(token: archmage::X64V4Token, data: &[f32; 4]) -> f32 {
    unsafe { __arcane_process_v4(token, data) }
}
#[doc(hidden)]
#[allow(dead_code)]
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
fn __arcane_process_v3(token: archmage::X64V3Token, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}
#[allow(dead_code)]
#[inline(always)]
fn process_v3(token: archmage::X64V3Token, data: &[f32; 4]) -> f32 {
    unsafe { __arcane_process_v3(token, data) }
}
#[allow(dead_code)]
fn process_scalar(token: archmage::ScalarToken, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}
fn main() {}

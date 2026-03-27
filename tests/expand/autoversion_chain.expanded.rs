use archmage::autoversion;
fn inner_work(data: &[f32; 4]) -> f32 {
    use archmage::SimdToken;
    {
        if let Some(__t) = archmage::X64V4Token::summon() {
            return inner_work_v4(__t, data);
        }
        if let Some(__t) = archmage::X64V3Token::summon() {
            return inner_work_v3(__t, data);
        }
    }
    inner_work_scalar(archmage::ScalarToken, data)
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
fn __arcane_inner_work_v4(_token: archmage::X64V4Token, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}
#[allow(dead_code)]
#[inline(always)]
fn inner_work_v4(_token: archmage::X64V4Token, data: &[f32; 4]) -> f32 {
    unsafe { __arcane_inner_work_v4(_token, data) }
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
fn __arcane_inner_work_v3(_token: archmage::X64V3Token, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}
#[allow(dead_code)]
#[inline(always)]
fn inner_work_v3(_token: archmage::X64V3Token, data: &[f32; 4]) -> f32 {
    unsafe { __arcane_inner_work_v3(_token, data) }
}
#[allow(dead_code)]
fn inner_work_scalar(_token: archmage::ScalarToken, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}
fn outer_work(data: &[f32; 4], scale: f32) -> f32 {
    use archmage::SimdToken;
    {
        if let Some(__t) = archmage::X64V4Token::summon() {
            return outer_work_v4(__t, data, scale);
        }
        if let Some(__t) = archmage::X64V3Token::summon() {
            return outer_work_v3(__t, data, scale);
        }
    }
    outer_work_scalar(archmage::ScalarToken, data, scale)
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
fn __arcane_outer_work_v4(
    _token: archmage::X64V4Token,
    data: &[f32; 4],
    scale: f32,
) -> f32 {
    inner_work(data) * scale
}
#[allow(dead_code)]
#[inline(always)]
fn outer_work_v4(_token: archmage::X64V4Token, data: &[f32; 4], scale: f32) -> f32 {
    unsafe { __arcane_outer_work_v4(_token, data, scale) }
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
fn __arcane_outer_work_v3(
    _token: archmage::X64V3Token,
    data: &[f32; 4],
    scale: f32,
) -> f32 {
    inner_work(data) * scale
}
#[allow(dead_code)]
#[inline(always)]
fn outer_work_v3(_token: archmage::X64V3Token, data: &[f32; 4], scale: f32) -> f32 {
    unsafe { __arcane_outer_work_v3(_token, data, scale) }
}
#[allow(dead_code)]
fn outer_work_scalar(_token: archmage::ScalarToken, data: &[f32; 4], scale: f32) -> f32 {
    inner_work(data) * scale
}
fn main() {}

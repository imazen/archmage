use archmage::{arcane, incant, X64V3Token, NeonToken, ScalarToken};
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
fn __arcane_process_v3(_token: X64V3Token, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}
#[inline(always)]
fn process_v3(_token: X64V3Token, data: &[f32; 4]) -> f32 {
    unsafe { __arcane_process_v3(_token, data) }
}
fn process_scalar(_token: ScalarToken, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}
fn dispatch(data: &[f32; 4]) -> f32 {
    '__incant: {
        use archmage::SimdToken;
        {
            if let Some(__t) = archmage::X64V3Token::summon() {
                break '__incant process_v3(__t, data);
            }
        }
        process_scalar(archmage::ScalarToken, data)
    }
}
fn main() {}

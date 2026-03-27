use archmage::{arcane, incant, X64V3Token, ScalarToken};
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
fn __arcane_compute_v3(_token: X64V3Token, x: f32) -> f32 {
    x * x
}
#[inline(always)]
fn compute_v3(_token: X64V3Token, x: f32) -> f32 {
    unsafe { __arcane_compute_v3(_token, x) }
}
fn compute_scalar(_token: ScalarToken, x: f32) -> f32 {
    x * x
}
fn dispatch(x: f32) -> f32 {
    '__incant: {
        use archmage::SimdToken;
        {
            if let Some(__t) = archmage::X64V3Token::summon() {
                break '__incant compute_v3(__t, x);
            }
        }
        compute_scalar(archmage::ScalarToken, x)
    }
}
fn main() {}

use archmage::{arcane, incant, IntoConcreteToken, X64V3Token, NeonToken, ScalarToken};
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
fn __arcane_inner_v3(_token: X64V3Token, x: f32) -> f32 {
    x * 2.0
}
#[inline(always)]
fn inner_v3(_token: X64V3Token, x: f32) -> f32 {
    unsafe { __arcane_inner_v3(_token, x) }
}
fn inner_scalar(_token: ScalarToken, x: f32) -> f32 {
    x * 2.0
}
fn pass_through<T: IntoConcreteToken>(token: T, x: f32) -> f32 {
    '__incant: {
        use archmage::IntoConcreteToken;
        let __incant_token = token;
        {
            if let Some(__t) = __incant_token.as_x64v3() {
                break '__incant inner_v3(__t, x);
            }
        }
        if let Some(__t) = __incant_token.as_scalar() {
            break '__incant inner_scalar(__t, x);
        }
        {
            ::core::panicking::panic_fmt(
                format_args!(
                    "internal error: entered unreachable code: {0}",
                    format_args!("Token did not match any known variant"),
                ),
            );
        }
    }
}
fn main() {}

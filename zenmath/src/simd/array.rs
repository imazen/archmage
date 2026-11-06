//! Accurate array operations. Selection occurs once per public call.
use archmage::prelude::*;

/// Correctly rounded exp; NaN payloads unspecified.
#[archmage::attune(
    _v3,
    _v3_t,
    _neon,
    _neon_t,
    _wasm128,
    _wasm128_t,
    _scalar,
    _scalar_t,
    _v4x(cfg(avx512)),
    _v4x_t(cfg(avx512))
)]
pub fn exp_f32x4(x: [f32; 4]) -> [f32; 4] {
    let token = Token::from_context();
    super::correct::exp(token, x)
}

/// Select exp once for this array. Runtime probes require `runtime-dispatch`.
#[inline]
pub fn exp_f32x4(x: [f32; 4]) -> [f32; 4] {
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(token) = super::available::<X64V4xToken>() {
        return exp_f32x4_v4x_t(token, x);
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = super::available::<X64V3Token>() {
        return exp_f32x4_v3_t(token, x);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(token) = super::available::<NeonToken>() {
        return exp_f32x4_neon_t(token, x);
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(token) = super::available::<Wasm128Token>() {
        return exp_f32x4_wasm128_t(token, x);
    }
    exp_f32x4_scalar_t(ScalarToken, x)
}

/// Correctly rounded ln; NaN payloads unspecified.
#[archmage::attune(
    _v3,
    _v3_t,
    _neon,
    _neon_t,
    _wasm128,
    _wasm128_t,
    _scalar,
    _scalar_t,
    _v4x(cfg(avx512)),
    _v4x_t(cfg(avx512))
)]
pub fn ln_f32x4(x: [f32; 4]) -> [f32; 4] {
    let token = Token::from_context();
    super::correct::log::<_, false>(token, x)
}

/// Select ln once for this array. Runtime probes require `runtime-dispatch`.
#[inline]
pub fn ln_f32x4(x: [f32; 4]) -> [f32; 4] {
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(token) = super::available::<X64V4xToken>() {
        return ln_f32x4_v4x_t(token, x);
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = super::available::<X64V3Token>() {
        return ln_f32x4_v3_t(token, x);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(token) = super::available::<NeonToken>() {
        return ln_f32x4_neon_t(token, x);
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(token) = super::available::<Wasm128Token>() {
        return ln_f32x4_wasm128_t(token, x);
    }
    ln_f32x4_scalar_t(ScalarToken, x)
}

/// Correctly rounded log2; NaN payloads unspecified.
#[archmage::attune(
    _v3,
    _v3_t,
    _neon,
    _neon_t,
    _wasm128,
    _wasm128_t,
    _scalar,
    _scalar_t,
    _v4x(cfg(avx512)),
    _v4x_t(cfg(avx512))
)]
pub fn log2_f32x4(x: [f32; 4]) -> [f32; 4] {
    let token = Token::from_context();
    super::correct::log::<_, true>(token, x)
}

/// Select log2 once for this array. Runtime probes require `runtime-dispatch`.
#[inline]
pub fn log2_f32x4(x: [f32; 4]) -> [f32; 4] {
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(token) = super::available::<X64V4xToken>() {
        return log2_f32x4_v4x_t(token, x);
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = super::available::<X64V3Token>() {
        return log2_f32x4_v3_t(token, x);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(token) = super::available::<NeonToken>() {
        return log2_f32x4_neon_t(token, x);
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(token) = super::available::<Wasm128Token>() {
        return log2_f32x4_wasm128_t(token, x);
    }
    log2_f32x4_scalar_t(ScalarToken, x)
}

/// Correctly rounded pow; NaN payloads unspecified.
#[archmage::attune(
    _v3,
    _v3_t,
    _neon,
    _neon_t,
    _wasm128,
    _wasm128_t,
    _scalar,
    _scalar_t,
    _v4x(cfg(avx512)),
    _v4x_t(cfg(avx512))
)]
pub fn pow_f32x4(x: [f32; 4], y: [f32; 4]) -> [f32; 4] {
    let token = Token::from_context();
    super::correct::pow(token, x, y)
}

/// Select pow once for this array. Runtime probes require `runtime-dispatch`.
#[inline]
pub fn pow_f32x4(x: [f32; 4], y: [f32; 4]) -> [f32; 4] {
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(token) = super::available::<X64V4xToken>() {
        return pow_f32x4_v4x_t(token, x, y);
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = super::available::<X64V3Token>() {
        return pow_f32x4_v3_t(token, x, y);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(token) = super::available::<NeonToken>() {
        return pow_f32x4_neon_t(token, x, y);
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(token) = super::available::<Wasm128Token>() {
        return pow_f32x4_wasm128_t(token, x, y);
    }
    pow_f32x4_scalar_t(ScalarToken, x, y)
}

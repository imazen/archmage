//! Accurate slice operations. Selection occurs once per public call.
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
pub fn exp_f32(data: &mut [f32]) {
    let token = Token::from_context();
    let (chunks, tail) = data.as_chunks_mut::<4>();
    for (i, chunk) in chunks.iter_mut().enumerate() {
        let _ = i;
        *chunk = super::correct::exp(token, *chunk);
    }
    if !tail.is_empty() {
        let mut x = [1.; 4];
        x[..tail.len()].copy_from_slice(tail);
        let out = super::correct::exp(token, x);
        tail.copy_from_slice(&out[..tail.len()]);
    }
}

/// Select exp once for this slice. Runtime probes require `runtime-dispatch`.
#[inline]
pub fn exp_f32(data: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = super::available::<X64V3Token>() {
        return exp_f32_v3_t(token, data);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(token) = super::available::<NeonToken>() {
        return exp_f32_neon_t(token, data);
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(token) = super::available::<Wasm128Token>() {
        return exp_f32_wasm128_t(token, data);
    }
    exp_f32_scalar_t(ScalarToken, data)
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
pub fn ln_f32(data: &mut [f32]) {
    let token = Token::from_context();
    let (chunks, tail) = data.as_chunks_mut::<4>();
    for (i, chunk) in chunks.iter_mut().enumerate() {
        let _ = i;
        *chunk = super::correct::log::<_, false, 4>(token, *chunk);
    }
    if !tail.is_empty() {
        let mut x = [1.; 4];
        x[..tail.len()].copy_from_slice(tail);
        let out = super::correct::log::<_, false, 4>(token, x);
        tail.copy_from_slice(&out[..tail.len()]);
    }
}

/// Select ln once for this slice. Runtime probes require `runtime-dispatch`.
#[inline]
pub fn ln_f32(data: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = super::available::<X64V3Token>() {
        return ln_f32_v3_t(token, data);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(token) = super::available::<NeonToken>() {
        return ln_f32_neon_t(token, data);
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(token) = super::available::<Wasm128Token>() {
        return ln_f32_wasm128_t(token, data);
    }
    ln_f32_scalar_t(ScalarToken, data)
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
pub fn log2_f32(data: &mut [f32]) {
    let token = Token::from_context();
    let (chunks, tail) = data.as_chunks_mut::<4>();
    for (i, chunk) in chunks.iter_mut().enumerate() {
        let _ = i;
        *chunk = super::correct::log::<_, true, 4>(token, *chunk);
    }
    if !tail.is_empty() {
        let mut x = [1.; 4];
        x[..tail.len()].copy_from_slice(tail);
        let out = super::correct::log::<_, true, 4>(token, x);
        tail.copy_from_slice(&out[..tail.len()]);
    }
}

/// Select log2 once for this slice. Runtime probes require `runtime-dispatch`.
#[inline]
pub fn log2_f32(data: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = super::available::<X64V3Token>() {
        return log2_f32_v3_t(token, data);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(token) = super::available::<NeonToken>() {
        return log2_f32_neon_t(token, data);
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(token) = super::available::<Wasm128Token>() {
        return log2_f32_wasm128_t(token, data);
    }
    log2_f32_scalar_t(ScalarToken, data)
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
pub fn pow_f32(data: &mut [f32], exponents: &[f32]) {
    let token = Token::from_context();
    assert_eq!(
        data.len(),
        exponents.len(),
        "power input lengths must match"
    );
    let (chunks, tail) = data.as_chunks_mut::<4>();
    for (i, chunk) in chunks.iter_mut().enumerate() {
        let y = core::array::from_fn(|j| exponents[4 * i + j]);
        *chunk = super::correct::pow(token, *chunk, y);
    }
    if !tail.is_empty() {
        let mut x = [1.; 4];
        x[..tail.len()].copy_from_slice(tail);
        let offset = chunks.len() * 4;
        let mut y = [1.; 4];
        y[..tail.len()].copy_from_slice(&exponents[offset..]);
        let out = super::correct::pow(token, x, y);
        tail.copy_from_slice(&out[..tail.len()]);
    }
}

/// Select pow once for this slice. Runtime probes require `runtime-dispatch`.
#[inline]
pub fn pow_f32(data: &mut [f32], exponents: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = super::available::<X64V3Token>() {
        return pow_f32_v3_t(token, data, exponents);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(token) = super::available::<NeonToken>() {
        return pow_f32_neon_t(token, data, exponents);
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(token) = super::available::<Wasm128Token>() {
        return pow_f32_wasm128_t(token, data, exponents);
    }
    pow_f32_scalar_t(ScalarToken, data, exponents)
}

/// Whole buffer using eight native AVX-512 binary64 intermediates.
#[archmage::attune(_v4x(cfg(avx512)), _v4x_t(cfg(avx512)))]
pub fn exp_f32_wide(data: &mut [f32]) {
    let token = Token::from_context();
    let (chunks, tail) = data.as_chunks_mut::<8>();
    for (i, chunk) in chunks.iter_mut().enumerate() {
        let _ = i;
        *chunk = super::correct::exp(token, *chunk);
    }
    if !tail.is_empty() {
        let mut x = [1.; 8];
        x[..tail.len()].copy_from_slice(tail);
        let out = super::correct::exp(token, x);
        tail.copy_from_slice(&out[..tail.len()]);
    }
}

/// Whole buffer using eight native AVX-512 binary64 intermediates.
#[archmage::attune(_v4x(cfg(avx512)), _v4x_t(cfg(avx512)))]
pub fn ln_f32_wide(data: &mut [f32]) {
    let token = Token::from_context();
    let (chunks, tail) = data.as_chunks_mut::<8>();
    for (i, chunk) in chunks.iter_mut().enumerate() {
        let _ = i;
        *chunk = super::correct::log::<_, false, 8>(token, *chunk);
    }
    if !tail.is_empty() {
        let mut x = [1.; 8];
        x[..tail.len()].copy_from_slice(tail);
        let out = super::correct::log::<_, false, 8>(token, x);
        tail.copy_from_slice(&out[..tail.len()]);
    }
}

/// Whole buffer using eight native AVX-512 binary64 intermediates.
#[archmage::attune(_v4x(cfg(avx512)), _v4x_t(cfg(avx512)))]
pub fn log2_f32_wide(data: &mut [f32]) {
    let token = Token::from_context();
    let (chunks, tail) = data.as_chunks_mut::<8>();
    for (i, chunk) in chunks.iter_mut().enumerate() {
        let _ = i;
        *chunk = super::correct::log::<_, true, 8>(token, *chunk);
    }
    if !tail.is_empty() {
        let mut x = [1.; 8];
        x[..tail.len()].copy_from_slice(tail);
        let out = super::correct::log::<_, true, 8>(token, x);
        tail.copy_from_slice(&out[..tail.len()]);
    }
}

/// Whole buffer using eight native AVX-512 binary64 intermediates.
#[archmage::attune(_v4x(cfg(avx512)), _v4x_t(cfg(avx512)))]
pub fn pow_f32_wide(data: &mut [f32], exponents: &[f32]) {
    let token = Token::from_context();
    assert_eq!(
        data.len(),
        exponents.len(),
        "power input lengths must match"
    );
    let (chunks, tail) = data.as_chunks_mut::<8>();
    for (i, chunk) in chunks.iter_mut().enumerate() {
        let y = core::array::from_fn(|j| exponents[i * 8 + j]);
        *chunk = super::correct::pow(token, *chunk, y);
    }
    if !tail.is_empty() {
        let mut x = [1.; 8];
        x[..tail.len()].copy_from_slice(tail);
        let mut y = [1.; 8];
        let offset = chunks.len() * 8;
        y[..tail.len()].copy_from_slice(&exponents[offset..]);
        let out = super::correct::pow(token, x, y);
        tail.copy_from_slice(&out[..tail.len()]);
    }
}

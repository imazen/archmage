//! Fixed-array operations: `[f32; 8]` in, `[f32; 8]` out.
//!
//! Each op is an `#[attune]` family: the public name dispatches at runtime,
//! `_v3_t`-style proof wrappers take a token, and `_v3`-style direct variants
//! are for callers already in that feature context. Each call is one load, one
//! kernel and one store; to keep a pipeline in registers across several
//! operations, call [`kernels`] from one `#[attune]` body instead.

crate::surface::array_surface!();

/// Correctly rounded exp, nearest-even with gradual underflow, per lane.
#[archmage::attune(_*, _*_t, dispatch(inline(hint)), +v4x(cfg(avx512)))]
pub fn exp_f32x8(a: [f32; 8]) -> [f32; 8] {
    let token = Token::from_context();
    let lo = crate::correct::exp(token, core::array::from_fn(|i| a[i]));
    let hi = crate::correct::exp(token, core::array::from_fn(|i| a[i + 4]));
    core::array::from_fn(|i| if i < 4 { lo[i] } else { hi[i - 4] })
}

/// Correctly rounded ln, nearest-even with gradual underflow, per lane.
#[archmage::attune(_*, _*_t, dispatch(inline(hint)), +v4x(cfg(avx512)))]
pub fn ln_f32x8(a: [f32; 8]) -> [f32; 8] {
    let token = Token::from_context();
    let lo = crate::correct::log::<_, false>(token, core::array::from_fn(|i| a[i]));
    let hi = crate::correct::log::<_, false>(token, core::array::from_fn(|i| a[i + 4]));
    core::array::from_fn(|i| if i < 4 { lo[i] } else { hi[i - 4] })
}

/// Correctly rounded log2, nearest-even with gradual underflow, per lane.
#[archmage::attune(_*, _*_t, dispatch(inline(hint)), +v4x(cfg(avx512)))]
pub fn log2_f32x8(a: [f32; 8]) -> [f32; 8] {
    let token = Token::from_context();
    let lo = crate::correct::log::<_, true>(token, core::array::from_fn(|i| a[i]));
    let hi = crate::correct::log::<_, true>(token, core::array::from_fn(|i| a[i + 4]));
    core::array::from_fn(|i| if i < 4 { lo[i] } else { hi[i - 4] })
}

/// Correctly rounded power with an independent exponent per lane.
#[archmage::attune(_*, _*_t, dispatch(inline(hint)), +v4x(cfg(avx512)))]
pub fn pow_f32x8(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
    let token = Token::from_context();
    let lo = crate::correct::pow(
        token,
        core::array::from_fn(|i| a[i]),
        core::array::from_fn(|i| b[i]),
    );
    let hi = crate::correct::pow(
        token,
        core::array::from_fn(|i| a[i + 4]),
        core::array::from_fn(|i| b[i + 4]),
    );
    core::array::from_fn(|i| if i < 4 { lo[i] } else { hi[i - 4] })
}

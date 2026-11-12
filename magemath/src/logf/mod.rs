//! Configurable f32 logarithms with explicit portability, precision and
//! validation, mirroring [`crate::powf`].
//!
//! MIDP and LOWP evaluate the existing `log2_midp`/`log2_lowp` approximations
//! on `log2(x)`, with `ln`/`log10` as scale-by-constant aliases. `Positive`
//! rejects `x <= 0`, subnormals, NaN and infinities under Checked; `Specials`
//! returns the IEEE specials (`0 -> -inf`, `<0 -> NaN`, NaN -> NaN,
//! `+inf -> +inf`). `AssumeValid` omits checks while retaining memory safety.
//! Legacy Magetypes logarithm methods remain available during deprecation.

#![doc = include_str!("permutations.md")]
pub(crate) mod const_api;
mod dispatch;
mod graph;
mod options;
#[cfg(test)]
mod tests;
pub use const_api::{
    try_ln_array_with, try_ln_runtime, try_ln_with, try_ln_with_token, try_log1p_array_runtime,
    try_log1p_array_with, try_log1p_runtime, try_log1p_with, try_log1p_with_token,
    try_log2_array_with, try_log2_runtime, try_log2_with, try_log2_with_token,
    try_log10_array_with, try_log10_runtime, try_log10_with, try_log10_with_token,
};
pub use options::{
    LogfDomain, LogfExecution, LogfOptions, LogfPortability, LogfPrecision, LogfValidation,
};

/// Reason a complete vector was rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LogfErrorKind {
    /// An input fails the domain: `x <= 0`, subnormal, NaN or infinite.
    InvalidInput,
    /// The array width is not 1, 4, 8, or enabled width 16.
    UnsupportedWidth,
    /// An unsupported combination of precision, portability, domain or features.
    UnsupportedOptions,
}

/// Whole-vector rejection. No partial result is returned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LogfError {
    /// Which part of the contract failed.
    pub kind: LogfErrorKind,
    /// Lanes failing the reported stage, low bit first.
    pub lane_mask: u16,
}

impl core::fmt::Display for LogfError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "logf {:?}, lane mask {:#06x}", self.kind, self.lane_mask)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LogfError {}

/// Positive-domain scalar log2 using the default checked MIDP domain.
#[inline(always)]
pub fn try_log2(x: f32) -> Result<f32, LogfError> {
    try_log2_with::<{ LogfOptions::DEFAULT }>(x)
}

/// Positive-domain scalar ln using the default checked MIDP domain.
#[inline(always)]
pub fn try_ln(x: f32) -> Result<f32, LogfError> {
    try_ln_with::<{ LogfOptions::DEFAULT }>(x)
}

/// Positive-domain scalar log10 using the default checked MIDP domain.
#[inline(always)]
pub fn try_log10(x: f32) -> Result<f32, LogfError> {
    try_log10_with::<{ LogfOptions::DEFAULT }>(x)
}

/// Configured log2 on a 4-lane vector, preserving the caller's feature proof.
/// Use the same options and token with the scalar `*_with_token` entry for tails.
#[inline(always)]
pub fn try_log2_f32x4_with<const OPTIONS: u32, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    input: magetypes::simd::generic::f32x4<T>,
) -> Result<magetypes::simd::generic::f32x4<T>, LogfError> {
    const_api::run4::<T, OPTIONS>(token, input)
}

/// Configured ln on a 4-lane vector, preserving the caller's feature proof.
/// Use the same options and token with the scalar `*_with_token` entry for tails.
#[inline(always)]
pub fn try_ln_f32x4_with<const OPTIONS: u32, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    input: magetypes::simd::generic::f32x4<T>,
) -> Result<magetypes::simd::generic::f32x4<T>, LogfError> {
    const_api::run4::<T, OPTIONS>(token, input).map(|value| {
        value * magetypes::simd::generic::f32x4::splat_t(token, core::f32::consts::LN_2)
    })
}

/// Configured log10 on a 4-lane vector, preserving the caller's feature proof.
/// Use the same options and token with the scalar `*_with_token` entry for tails.
#[inline(always)]
pub fn try_log10_f32x4_with<const OPTIONS: u32, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    input: magetypes::simd::generic::f32x4<T>,
) -> Result<magetypes::simd::generic::f32x4<T>, LogfError> {
    const_api::run4::<T, OPTIONS>(token, input).map(|value| {
        value * magetypes::simd::generic::f32x4::splat_t(token, core::f32::consts::LOG10_2)
    })
}

/// Configured log1p on a 4-lane vector, preserving the caller's feature proof.
/// Use the same options and token with the scalar `*_with_token` entry for tails.
#[inline(always)]
pub fn try_log1p_f32x4_with<const OPTIONS: u32, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    input: magetypes::simd::generic::f32x4<T>,
) -> Result<magetypes::simd::generic::f32x4<T>, LogfError> {
    let xs = input.to_array();
    let parts: [(f32, f32); 4] = xs.map(const_api::log1p_parts);
    let u = magetypes::simd::generic::f32x4::from_array_t(token, parts.map(|p| p.0));
    let c = magetypes::simd::generic::f32x4::from_array_t(token, parts.map(|p| p.1));
    const_api::run4::<T, OPTIONS>(token, u).map(|value| {
        let result = value.mul_add_fused(
            magetypes::simd::generic::f32x4::splat_t(token, core::f32::consts::LN_2),
            c,
        );
        magetypes::simd::generic::f32x4::blend(
            input.simd_eq(magetypes::simd::generic::f32x4::splat_t(token, 0.0)),
            input,
            result,
        )
    })
}

/// Configured log2 on a 8-lane vector, preserving the caller's feature proof.
/// Use the same options and token with the scalar `*_with_token` entry for tails.
#[inline(always)]
pub fn try_log2_f32x8_with<const OPTIONS: u32, T: magetypes::simd::backends::F32x8Convert>(
    token: T,
    input: magetypes::simd::generic::f32x8<T>,
) -> Result<magetypes::simd::generic::f32x8<T>, LogfError> {
    const_api::run8::<T, OPTIONS>(token, input)
}

/// Configured ln on a 8-lane vector, preserving the caller's feature proof.
/// Use the same options and token with the scalar `*_with_token` entry for tails.
#[inline(always)]
pub fn try_ln_f32x8_with<const OPTIONS: u32, T: magetypes::simd::backends::F32x8Convert>(
    token: T,
    input: magetypes::simd::generic::f32x8<T>,
) -> Result<magetypes::simd::generic::f32x8<T>, LogfError> {
    const_api::run8::<T, OPTIONS>(token, input).map(|value| {
        value * magetypes::simd::generic::f32x8::splat_t(token, core::f32::consts::LN_2)
    })
}

/// Configured log10 on a 8-lane vector, preserving the caller's feature proof.
/// Use the same options and token with the scalar `*_with_token` entry for tails.
#[inline(always)]
pub fn try_log10_f32x8_with<const OPTIONS: u32, T: magetypes::simd::backends::F32x8Convert>(
    token: T,
    input: magetypes::simd::generic::f32x8<T>,
) -> Result<magetypes::simd::generic::f32x8<T>, LogfError> {
    const_api::run8::<T, OPTIONS>(token, input).map(|value| {
        value * magetypes::simd::generic::f32x8::splat_t(token, core::f32::consts::LOG10_2)
    })
}

/// Configured log1p on a 8-lane vector, preserving the caller's feature proof.
/// Use the same options and token with the scalar `*_with_token` entry for tails.
#[inline(always)]
pub fn try_log1p_f32x8_with<const OPTIONS: u32, T: magetypes::simd::backends::F32x8Convert>(
    token: T,
    input: magetypes::simd::generic::f32x8<T>,
) -> Result<magetypes::simd::generic::f32x8<T>, LogfError> {
    let xs = input.to_array();
    let parts: [(f32, f32); 8] = xs.map(const_api::log1p_parts);
    let u = magetypes::simd::generic::f32x8::from_array_t(token, parts.map(|p| p.0));
    let c = magetypes::simd::generic::f32x8::from_array_t(token, parts.map(|p| p.1));
    const_api::run8::<T, OPTIONS>(token, u).map(|value| {
        let result = value.mul_add_fused(
            magetypes::simd::generic::f32x8::splat_t(token, core::f32::consts::LN_2),
            c,
        );
        magetypes::simd::generic::f32x8::blend(
            input.simd_eq(magetypes::simd::generic::f32x8::splat_t(token, 0.0)),
            input,
            result,
        )
    })
}

/// Configured log2 on a 16-lane vector, preserving the caller's feature proof.
/// Use the same options and token with the scalar `*_with_token` entry for tails.
#[cfg(feature = "w512")]
#[inline(always)]
pub fn try_log2_f32x16_with<const OPTIONS: u32, T: magetypes::simd::backends::F32x16Convert>(
    token: T,
    input: magetypes::simd::generic::f32x16<T>,
) -> Result<magetypes::simd::generic::f32x16<T>, LogfError> {
    const_api::run16::<T, OPTIONS>(token, input)
}

/// Configured ln on a 16-lane vector, preserving the caller's feature proof.
/// Use the same options and token with the scalar `*_with_token` entry for tails.
#[cfg(feature = "w512")]
#[inline(always)]
pub fn try_ln_f32x16_with<const OPTIONS: u32, T: magetypes::simd::backends::F32x16Convert>(
    token: T,
    input: magetypes::simd::generic::f32x16<T>,
) -> Result<magetypes::simd::generic::f32x16<T>, LogfError> {
    const_api::run16::<T, OPTIONS>(token, input).map(|value| {
        value * magetypes::simd::generic::f32x16::splat_t(token, core::f32::consts::LN_2)
    })
}

/// Configured log10 on a 16-lane vector, preserving the caller's feature proof.
/// Use the same options and token with the scalar `*_with_token` entry for tails.
#[cfg(feature = "w512")]
#[inline(always)]
pub fn try_log10_f32x16_with<const OPTIONS: u32, T: magetypes::simd::backends::F32x16Convert>(
    token: T,
    input: magetypes::simd::generic::f32x16<T>,
) -> Result<magetypes::simd::generic::f32x16<T>, LogfError> {
    const_api::run16::<T, OPTIONS>(token, input).map(|value| {
        value * magetypes::simd::generic::f32x16::splat_t(token, core::f32::consts::LOG10_2)
    })
}

/// Configured log1p on a 16-lane vector, preserving the caller's feature proof.
/// Use the same options and token with the scalar `*_with_token` entry for tails.
#[cfg(feature = "w512")]
#[inline(always)]
pub fn try_log1p_f32x16_with<const OPTIONS: u32, T: magetypes::simd::backends::F32x16Convert>(
    token: T,
    input: magetypes::simd::generic::f32x16<T>,
) -> Result<magetypes::simd::generic::f32x16<T>, LogfError> {
    let xs = input.to_array();
    let parts: [(f32, f32); 16] = xs.map(const_api::log1p_parts);
    let u = magetypes::simd::generic::f32x16::from_array_t(token, parts.map(|p| p.0));
    let c = magetypes::simd::generic::f32x16::from_array_t(token, parts.map(|p| p.1));
    const_api::run16::<T, OPTIONS>(token, u).map(|value| {
        let result = value.mul_add_fused(
            magetypes::simd::generic::f32x16::splat_t(token, core::f32::consts::LN_2),
            c,
        );
        magetypes::simd::generic::f32x16::blend(
            input.simd_eq(magetypes::simd::generic::f32x16::splat_t(token, 0.0)),
            input,
            result,
        )
    })
}

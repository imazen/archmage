//! Configurable f32 logarithms with explicit portability, precision and
//! validation, mirroring [`crate::powf`].
//!
//! MIDP and LOWP evaluate the existing `log2_midp`/`log2_lowp` approximations
//! on `log2(x)`, with `ln`/`log10` as scale-by-constant aliases and in-register
//! `log1p`. `Positive` rejects `x <= 0`, subnormals, NaN and infinities under
//! Checked; `Specials` returns IEEE specials (`0 -> -inf`, `<0 -> NaN`,
//! `NaN -> NaN`, `+inf -> +inf`). Declaring a [`LogfSpec::input_range`]
//! validates against `[lo, hi]` in one SIMD comparison pair. `AssumeValid`
//! omits checks while retaining memory safety.
//! The flat Magetypes logarithm methods are unchanged and remain supported.

pub(crate) mod const_api;
mod dispatch;
mod graph;
mod options;
mod profile;
#[cfg(test)]
mod tests;
pub use const_api::{
    try_ln_array_runtime, try_ln_array_with, try_ln_runtime, try_ln_slice_with,
    try_ln_slice_with_token, try_ln_with, try_ln_with_token, try_log1p_array_runtime,
    try_log1p_array_with, try_log1p_runtime, try_log1p_slice_with, try_log1p_slice_with_token,
    try_log1p_with, try_log1p_with_token, try_log2_array_runtime, try_log2_array_with,
    try_log2_runtime, try_log2_slice_with, try_log2_slice_with_token, try_log2_with,
    try_log2_with_token, try_log10_array_runtime, try_log10_array_with, try_log10_runtime,
    try_log10_slice_with, try_log10_slice_with_token, try_log10_with, try_log10_with_token,
};
pub use options::{
    LogfDomain, LogfExecution, LogfOptions, LogfPortability, LogfPrecision, LogfValidation,
};
pub use profile::{LogfProfile, LogfSpec, Options, presets};

/// Reason a complete vector was rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LogfErrorKind {
    /// An input fails the domain or declared input range.
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
    try_log2_with::<presets::Midp>(x)
}

/// Positive-domain scalar ln using the default checked MIDP domain.
#[inline(always)]
pub fn try_ln(x: f32) -> Result<f32, LogfError> {
    try_ln_with::<presets::Midp>(x)
}

/// Positive-domain scalar log10 using the default checked MIDP domain.
#[inline(always)]
pub fn try_log10(x: f32) -> Result<f32, LogfError> {
    try_log10_with::<presets::Midp>(x)
}

/// Positive-domain scalar log1p (`ln(1 + x)`) using the default checked MIDP domain.
#[inline(always)]
pub fn try_log1p(x: f32) -> Result<f32, LogfError> {
    try_log1p_with::<presets::Midp>(x)
}

/// Configured log2 on a 4-lane vector, preserving the caller's feature proof.
/// Use the same profile and token with [`try_log2_with_token`] for scalar tails.
#[inline(always)]
pub fn try_log2_f32x4_with<P: LogfProfile, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    input: magetypes::simd::generic::f32x4<T>,
) -> Result<magetypes::simd::generic::f32x4<T>, LogfError> {
    const_api::run4::<P, 0, T>(token, input)
}

/// Configured ln on a 4-lane vector, preserving the caller's feature proof.
/// Use the same profile and token with [`try_ln_with_token`] for scalar tails.
#[inline(always)]
pub fn try_ln_f32x4_with<P: LogfProfile, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    input: magetypes::simd::generic::f32x4<T>,
) -> Result<magetypes::simd::generic::f32x4<T>, LogfError> {
    const_api::run4::<P, 1, T>(token, input)
}

/// Configured log10 on a 4-lane vector, preserving the caller's feature proof.
/// Use the same profile and token with [`try_log10_with_token`] for scalar tails.
#[inline(always)]
pub fn try_log10_f32x4_with<P: LogfProfile, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    input: magetypes::simd::generic::f32x4<T>,
) -> Result<magetypes::simd::generic::f32x4<T>, LogfError> {
    const_api::run4::<P, 2, T>(token, input)
}

/// Configured log1p on a 4-lane vector, preserving the caller's feature proof.
/// Use the same profile and token with [`try_log1p_with_token`] for scalar tails.
#[inline(always)]
pub fn try_log1p_f32x4_with<P: LogfProfile, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    input: magetypes::simd::generic::f32x4<T>,
) -> Result<magetypes::simd::generic::f32x4<T>, LogfError> {
    const_api::run4::<P, 3, T>(token, input)
}

/// Configured log2 on an 8-lane vector, preserving the caller's feature proof.
/// Use the same profile and token with [`try_log2_with_token`] for scalar tails.
#[inline(always)]
pub fn try_log2_f32x8_with<P: LogfProfile, T: magetypes::simd::backends::F32x8Convert>(
    token: T,
    input: magetypes::simd::generic::f32x8<T>,
) -> Result<magetypes::simd::generic::f32x8<T>, LogfError> {
    const_api::run8::<P, 0, T>(token, input)
}

/// Configured ln on an 8-lane vector, preserving the caller's feature proof.
/// Use the same profile and token with [`try_ln_with_token`] for scalar tails.
#[inline(always)]
pub fn try_ln_f32x8_with<P: LogfProfile, T: magetypes::simd::backends::F32x8Convert>(
    token: T,
    input: magetypes::simd::generic::f32x8<T>,
) -> Result<magetypes::simd::generic::f32x8<T>, LogfError> {
    const_api::run8::<P, 1, T>(token, input)
}

/// Configured log10 on an 8-lane vector, preserving the caller's feature proof.
/// Use the same profile and token with [`try_log10_with_token`] for scalar tails.
#[inline(always)]
pub fn try_log10_f32x8_with<P: LogfProfile, T: magetypes::simd::backends::F32x8Convert>(
    token: T,
    input: magetypes::simd::generic::f32x8<T>,
) -> Result<magetypes::simd::generic::f32x8<T>, LogfError> {
    const_api::run8::<P, 2, T>(token, input)
}

/// Configured log1p on an 8-lane vector, preserving the caller's feature proof.
/// Use the same profile and token with [`try_log1p_with_token`] for scalar tails.
#[inline(always)]
pub fn try_log1p_f32x8_with<P: LogfProfile, T: magetypes::simd::backends::F32x8Convert>(
    token: T,
    input: magetypes::simd::generic::f32x8<T>,
) -> Result<magetypes::simd::generic::f32x8<T>, LogfError> {
    const_api::run8::<P, 3, T>(token, input)
}

/// Configured log2 on a 16-lane vector, preserving the caller's feature proof.
/// Use the same profile and token with [`try_log2_with_token`] for scalar tails.
#[cfg(feature = "w512")]
#[inline(always)]
pub fn try_log2_f32x16_with<P: LogfProfile, T: magetypes::simd::backends::F32x16Convert>(
    token: T,
    input: magetypes::simd::generic::f32x16<T>,
) -> Result<magetypes::simd::generic::f32x16<T>, LogfError> {
    const_api::run16::<P, 0, T>(token, input)
}

/// Configured ln on a 16-lane vector, preserving the caller's feature proof.
/// Use the same profile and token with [`try_ln_with_token`] for scalar tails.
#[cfg(feature = "w512")]
#[inline(always)]
pub fn try_ln_f32x16_with<P: LogfProfile, T: magetypes::simd::backends::F32x16Convert>(
    token: T,
    input: magetypes::simd::generic::f32x16<T>,
) -> Result<magetypes::simd::generic::f32x16<T>, LogfError> {
    const_api::run16::<P, 1, T>(token, input)
}

/// Configured log10 on a 16-lane vector, preserving the caller's feature proof.
/// Use the same profile and token with [`try_log10_with_token`] for scalar tails.
#[cfg(feature = "w512")]
#[inline(always)]
pub fn try_log10_f32x16_with<P: LogfProfile, T: magetypes::simd::backends::F32x16Convert>(
    token: T,
    input: magetypes::simd::generic::f32x16<T>,
) -> Result<magetypes::simd::generic::f32x16<T>, LogfError> {
    const_api::run16::<P, 2, T>(token, input)
}

/// Configured log1p on a 16-lane vector, preserving the caller's feature proof.
/// Use the same profile and token with [`try_log1p_with_token`] for scalar tails.
#[cfg(feature = "w512")]
#[inline(always)]
pub fn try_log1p_f32x16_with<P: LogfProfile, T: magetypes::simd::backends::F32x16Convert>(
    token: T,
    input: magetypes::simd::generic::f32x16<T>,
) -> Result<magetypes::simd::generic::f32x16<T>, LogfError> {
    const_api::run16::<P, 3, T>(token, input)
}

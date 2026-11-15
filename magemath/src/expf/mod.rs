//! Configurable f32 exponentials with explicit portability, precision and
//! validation, mirroring [`crate::powf`].
//!
//! MIDP and LOWP evaluate the existing `exp2_midp`/`exp2_lowp` approximations
//! on `exp2(x)` and `exp2(x * log2(e))`; BitExact selects one fused rounding
//! sequence on every backend, BackendDependent keeps the caller's relaxed
//! arithmetic. `Bounded` rejects NaN and infinities under Checked; `Full`
//! propagates them to their natural results (NaN -> NaN, `+inf` -> `+inf`,
//! `-inf` -> `+0`). `AssumeValid` omits checks while retaining memory safety.
//! Legacy Magetypes exponential methods remain available during deprecation.

pub(crate) mod const_api;
mod dispatch;
mod graph;
mod options;
#[cfg(test)]
mod tests;
pub use const_api::{
    try_exp_array_runtime, try_exp_array_with, try_exp_runtime, try_exp_with, try_exp_with_token,
    try_exp2_array_runtime, try_exp2_array_with, try_exp2_runtime, try_exp2_with,
    try_exp2_with_token,
};
pub use options::{
    ExpfDomain, ExpfExecution, ExpfOptions, ExpfPortability, ExpfPrecision, ExpfValidation,
};

/// Reason a complete vector was rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExpfErrorKind {
    /// An input is NaN or infinite in a domain that rejects non-finite inputs.
    InvalidInput,
    /// The array width is not 1, 4, 8, or enabled width 16.
    UnsupportedWidth,
    /// An unsupported combination of precision, portability, domain or features.
    UnsupportedOptions,
}

/// Whole-vector rejection. No partial result is returned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpfError {
    /// Which part of the contract failed.
    pub kind: ExpfErrorKind,
    /// Lanes failing the reported stage, low bit first.
    pub lane_mask: u16,
}

impl core::fmt::Display for ExpfError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "expf {:?}, lane mask {:#06x}", self.kind, self.lane_mask)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ExpfError {}

/// Bounded scalar exp2 using the default checked MIDP domain.
#[inline(always)]
pub fn try_exp2(x: f32) -> Result<f32, ExpfError> {
    try_exp2_with::<{ ExpfOptions::DEFAULT }>(x)
}

/// Bounded scalar exp using the default checked MIDP domain.
#[inline(always)]
pub fn try_exp(x: f32) -> Result<f32, ExpfError> {
    try_exp_with::<{ ExpfOptions::DEFAULT }>(x)
}

/// Configured exp2 on a 4-lane vector, preserving the caller's feature proof.
/// Use the same options and token with the scalar `*_with_token` entry for tails.
#[inline(always)]
pub fn try_exp2_f32x4_with<const OPTIONS: u32, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    input: magetypes::simd::generic::f32x4<T>,
) -> Result<magetypes::simd::generic::f32x4<T>, ExpfError> {
    const_api::run4::<T, OPTIONS>(token, input)
}

/// Configured exp on a 4-lane vector, preserving the caller's feature proof.
/// Use the same options and token with the scalar `*_with_token` entry for tails.
#[inline(always)]
pub fn try_exp_f32x4_with<const OPTIONS: u32, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    input: magetypes::simd::generic::f32x4<T>,
) -> Result<magetypes::simd::generic::f32x4<T>, ExpfError> {
    const_api::run4::<T, OPTIONS>(
        token,
        input * magetypes::simd::generic::f32x4::splat_t(token, core::f32::consts::LOG2_E),
    )
}

/// Configured exp2 on a 8-lane vector, preserving the caller's feature proof.
/// Use the same options and token with the scalar `*_with_token` entry for tails.
#[inline(always)]
pub fn try_exp2_f32x8_with<const OPTIONS: u32, T: magetypes::simd::backends::F32x8Convert>(
    token: T,
    input: magetypes::simd::generic::f32x8<T>,
) -> Result<magetypes::simd::generic::f32x8<T>, ExpfError> {
    const_api::run8::<T, OPTIONS>(token, input)
}

/// Configured exp on a 8-lane vector, preserving the caller's feature proof.
/// Use the same options and token with the scalar `*_with_token` entry for tails.
#[inline(always)]
pub fn try_exp_f32x8_with<const OPTIONS: u32, T: magetypes::simd::backends::F32x8Convert>(
    token: T,
    input: magetypes::simd::generic::f32x8<T>,
) -> Result<magetypes::simd::generic::f32x8<T>, ExpfError> {
    const_api::run8::<T, OPTIONS>(
        token,
        input * magetypes::simd::generic::f32x8::splat_t(token, core::f32::consts::LOG2_E),
    )
}

/// Configured exp2 on a 16-lane vector, preserving the caller's feature proof.
/// Use the same options and token with the scalar `*_with_token` entry for tails.
#[cfg(feature = "w512")]
#[inline(always)]
pub fn try_exp2_f32x16_with<const OPTIONS: u32, T: magetypes::simd::backends::F32x16Convert>(
    token: T,
    input: magetypes::simd::generic::f32x16<T>,
) -> Result<magetypes::simd::generic::f32x16<T>, ExpfError> {
    const_api::run16::<T, OPTIONS>(token, input)
}

/// Configured exp on a 16-lane vector, preserving the caller's feature proof.
/// Use the same options and token with the scalar `*_with_token` entry for tails.
#[cfg(feature = "w512")]
#[inline(always)]
pub fn try_exp_f32x16_with<const OPTIONS: u32, T: magetypes::simd::backends::F32x16Convert>(
    token: T,
    input: magetypes::simd::generic::f32x16<T>,
) -> Result<magetypes::simd::generic::f32x16<T>, ExpfError> {
    const_api::run16::<T, OPTIONS>(
        token,
        input * magetypes::simd::generic::f32x16::splat_t(token, core::f32::consts::LOG2_E),
    )
}

//! Configurable `f32` cube-root (`cbrtf`) approximations with in-register SIMD
//! Kahan initial seeds, explicit portability, precision, and domain validation.
//!
//! Replaces the scalar `to_array()` + integer `/ 3` + `from_array()` Kahan seed
//! in Magetypes' flat `cbrt_lowp` / `cbrt_midp` / `cbrt_midp_precise`
//! methods with a pure SIMD 4-step shift-add approximation to `u / 3` across all
//! vector lanes, followed by 1 (`Lowp`, <= 259 ULP) or 2 (`Midp`, <= 3.2 ULP)
//! Halley iterations.

pub(crate) mod const_api;
mod dispatch;
mod graph;
mod options;
mod profile;
#[cfg(test)]
mod tests;

pub use const_api::{
    try_cbrt_array_runtime, try_cbrt_array_with, try_cbrt_runtime, try_cbrt_slice_with,
    try_cbrt_slice_with_token, try_cbrt_with, try_cbrt_with_token,
};
pub use options::{
    CbrtfDomain, CbrtfExecution, CbrtfOptions, CbrtfPortability, CbrtfPrecision, CbrtfValidation,
};
pub use profile::{CbrtfProfile, CbrtfSpec, Options, presets};

/// Reason a complete vector was rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CbrtfErrorKind {
    /// An input is subnormal, `>= f32::MAX / 3`, infinite, NaN, or outside the declared input range.
    InvalidInput,
    /// The array width is not 1, 4, 8, or enabled width 16.
    UnsupportedWidth,
    /// An unsupported combination of precision, portability, domain, or features.
    UnsupportedOptions,
}

/// Whole-vector rejection. No partial result is returned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CbrtfError {
    /// Which part of the contract failed.
    pub kind: CbrtfErrorKind,
    /// Lanes failing the reported stage, low bit first.
    pub lane_mask: u16,
}

impl core::fmt::Display for CbrtfError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "cbrtf {:?}, lane mask {:#06x}",
            self.kind, self.lane_mask
        )
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CbrtfError {}

/// Bounded scalar `cbrt` using the default checked MIDP domain.
#[inline(always)]
pub fn try_cbrt(x: f32) -> Result<f32, CbrtfError> {
    try_cbrt_with::<presets::Midp>(x)
}

/// Configured `cbrt` on a 4-lane vector, preserving the caller's feature proof.
/// Use the same profile and token with [`try_cbrt_with_token`] for scalar tails.
#[inline(always)]
pub fn try_cbrt_f32x4_with<P: CbrtfProfile, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    input: magetypes::simd::generic::f32x4<T>,
) -> Result<magetypes::simd::generic::f32x4<T>, CbrtfError> {
    const_api::run4::<P, T>(token, input)
}

/// Configured `cbrt` on an 8-lane vector, preserving the caller's feature proof.
/// Use the same profile and token with [`try_cbrt_with_token`] for scalar tails.
#[inline(always)]
pub fn try_cbrt_f32x8_with<P: CbrtfProfile, T: magetypes::simd::backends::F32x8Convert>(
    token: T,
    input: magetypes::simd::generic::f32x8<T>,
) -> Result<magetypes::simd::generic::f32x8<T>, CbrtfError> {
    const_api::run8::<P, T>(token, input)
}

/// Configured `cbrt` on a 16-lane vector, preserving the caller's feature proof.
/// Use the same profile and token with [`try_cbrt_with_token`] for scalar tails.
#[cfg(feature = "w512")]
#[inline(always)]
pub fn try_cbrt_f32x16_with<P: CbrtfProfile, T: magetypes::simd::backends::F32x16Convert>(
    token: T,
    input: magetypes::simd::generic::f32x16<T>,
) -> Result<magetypes::simd::generic::f32x16<T>, CbrtfError> {
    const_api::run16::<P, T>(token, input)
}

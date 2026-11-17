//! Configurable `f32` logarithms (`log2`, `ln`, `log10`, `log1p`) with explicit
//! portability, precision, and validation, mirroring [`crate::powf`].
//!
//! `Midp` and `Lowp` evaluate the refitted `log2_midp`/`log2_lowp`
//! approximations on `log2(x)`, with `ln`/`log10` as scale-by-constant aliases
//! and in-register `log1p`. By default, `x <= 0`, subnormals, NaN, and
//! infinities are rejected under `Checked`; [`LogfOptions::full_domain`]
//! propagates IEEE specials (`0 -> -inf`, `<0 -> NaN`, `NaN -> NaN`,
//! `+inf -> +inf`). Declaring [`LogfOptions::input_range`] validates against
//! `[lo, hi]` in one SIMD comparison pair.
//!
//! ```
//! use magemath::logf::{Logf, LogfOptions, try_log2};
//!
//! const AUDIO: u128 = LogfOptions::new().input_range(1e-6, 1e6).lowp().bits();
//! assert_eq!(try_log2(8.0).unwrap(), 3.0);
//!
//! let src = [1.0, 2.0, 4.0, 8.0];
//! let mut dst = [0.0; 4];
//! Logf::<AUDIO>::log2_slice(&src, &mut dst).unwrap();
//! assert!(dst[0].abs() < 1e-5);
//! ```

pub(crate) mod const_api;
mod dispatch;
mod graph;
mod options;
#[cfg(test)]
mod tests;

pub use const_api::{Logf, LogfRuntime};
pub use options::{LogfOptions, LogfPrecision};

/// Reason a complete vector was rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LogfErrorKind {
    /// An input fails the domain or declared input range.
    InvalidInput,
    /// The array width is not `1`, `4`, `8`, or enabled width `16`.
    UnsupportedWidth,
    /// An unsupported combination of precision, portability, domain, or features.
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

/// Positive-domain scalar `log2(x)` using [`LogfOptions::new()`].
#[inline(always)]
pub fn try_log2(x: f32) -> Result<f32, LogfError> {
    <Logf>::log2(x)
}

/// Whole-slice `log2(src[i]) -> dst[i]` using [`LogfOptions::new()`].
#[inline(always)]
pub fn try_log2_slice(
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<LogfErrorKind>> {
    <Logf>::log2_slice(src, dst)
}

/// Positive-domain scalar `ln(x)` using [`LogfOptions::new()`].
#[inline(always)]
pub fn try_ln(x: f32) -> Result<f32, LogfError> {
    <Logf>::ln(x)
}

/// Whole-slice `ln(src[i]) -> dst[i]` using [`LogfOptions::new()`].
#[inline(always)]
pub fn try_ln_slice(src: &[f32], dst: &mut [f32]) -> Result<(), crate::SliceError<LogfErrorKind>> {
    <Logf>::ln_slice(src, dst)
}

/// Positive-domain scalar `log10(x)` using [`LogfOptions::new()`].
#[inline(always)]
pub fn try_log10(x: f32) -> Result<f32, LogfError> {
    <Logf>::log10(x)
}

/// Whole-slice `log10(src[i]) -> dst[i]` using [`LogfOptions::new()`].
#[inline(always)]
pub fn try_log10_slice(
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<LogfErrorKind>> {
    <Logf>::log10_slice(src, dst)
}

/// Positive-domain scalar `ln(1 + x)` using [`LogfOptions::new()`].
#[inline(always)]
pub fn try_log1p(x: f32) -> Result<f32, LogfError> {
    <Logf>::log1p(x)
}

/// Whole-slice `ln(1 + src[i]) -> dst[i]` using [`LogfOptions::new()`].
#[inline(always)]
pub fn try_log1p_slice(
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<LogfErrorKind>> {
    <Logf>::log1p_slice(src, dst)
}

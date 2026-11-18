//! Configurable `f32` exponentials (`exp2`, `exp`) with explicit portability,
//! precision, and validation, mirroring [`crate::powf`].
//!
//! `Midp` and `Lowp` evaluate the refitted `exp2_midp`/`exp2_lowp`
//! approximations on `exp2(x)` and `exp2(x * log2(e))`; `BitExact` selects one
//! fused rounding sequence on every backend, while `BackendDependent` keeps the
//! caller's relaxed arithmetic. Declaring a tight [`ExpfOptions::input_range`]
//! elides output clamping at compile time.
//!
//! ```
//! use magemath::expf::{Expf, ExpfOptions, try_exp2};
//!
//! const TIGHT: u128 = ExpfOptions::new().input_range(-10.0, 10.0).lowp().bits();
//! assert!(ExpfOptions::from_bits(TIGHT).unwrap().elides_clamp());
//! assert_eq!(try_exp2(3.0).unwrap(), 8.0);
//!
//! let src = [-1.0, 0.0, 1.0, 2.0];
//! let mut dst = [0.0; 4];
//! Expf::<TIGHT>::exp2_slice(&src, &mut dst).unwrap();
//! assert_eq!(dst[1], 1.0);
//! ```

pub(crate) mod const_api;
pub(crate) mod dispatch;
mod graph;
mod options;
#[cfg(test)]
mod tests;

pub use const_api::{Expf, ExpfRuntime};
pub use options::{ExpfOptions, ExpfPrecision};

/// Reason a complete vector was rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExpfErrorKind {
    /// An input is NaN, infinite, or outside the declared input range.
    InvalidInput,
    /// The array width is not `1`, `4`, `8`, or enabled width `16`.
    UnsupportedWidth,
    /// An unsupported combination of precision, portability, domain, or features.
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

/// Bounded scalar `2^x` using [`ExpfOptions::new()`].
#[inline(always)]
pub fn try_exp2(x: f32) -> Result<f32, ExpfError> {
    <Expf>::exp2(x)
}

/// Whole-slice `2^src[i] -> dst[i]` using [`ExpfOptions::new()`].
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline(always)]
pub fn try_exp2_slice(
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<ExpfErrorKind>> {
    <Expf>::exp2_slice(src, dst)
}

/// Bounded scalar `e^x` using [`ExpfOptions::new()`].
#[inline(always)]
pub fn try_exp(x: f32) -> Result<f32, ExpfError> {
    <Expf>::exp(x)
}

/// Whole-slice `e^src[i] -> dst[i]` using [`ExpfOptions::new()`].
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline(always)]
pub fn try_exp_slice(src: &[f32], dst: &mut [f32]) -> Result<(), crate::SliceError<ExpfErrorKind>> {
    <Expf>::exp_slice(src, dst)
}

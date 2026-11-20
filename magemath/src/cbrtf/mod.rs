//! Configurable `f32` cube-root (`cbrtf`) approximations with in-register SIMD
//! Kahan initial seeds, explicit portability, precision, and domain validation.
//!
//! Replaces the scalar `to_array()` + integer `/ 3` + `from_array()` Kahan seed
//! in Magetypes' flat `cbrt_lowp` / `cbrt_midp` / `cbrt_midp_precise` methods
//! with a pure SIMD 4-step shift-add approximation to `u / 3` across all vector
//! lanes, followed by `1` (`Lowp`, `<= 259 ULP`) or `2` (`Midp`, `<= 3.2 ULP`)
//! Halley iterations.
//!
//! ```
//! use magemath::cbrtf::{Cbrtf, CbrtfOptions, try_cbrt};
//!
//! const OKLAB: u128 = CbrtfOptions::new().input_range(1e-6, 4.0).lowp().bits();
//! assert!(CbrtfOptions::from_bits(OKLAB).unwrap().elides_denorm_check());
//! assert!((try_cbrt(8.0).unwrap() - 2.0).abs() < 1e-6);
//!
//! let src = [1.0, 2.0, 3.0, 4.0];
//! let mut dst = [0.0; 4];
//! Cbrtf::<OKLAB>::slice(&src, &mut dst).unwrap();
//! assert!((dst[0] - 1.0).abs() < 1e-4);
//! ```

pub(crate) mod const_api;
pub(crate) mod dispatch;
mod graph;
mod options;
#[cfg(test)]
mod tests;

pub use const_api::{Cbrtf, CbrtfRuntime};
pub use options::{CbrtfDomain, CbrtfOptions, CbrtfPrecision};

/// Reason a complete vector was rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CbrtfErrorKind {
    /// An input is subnormal, `>= f32::MAX / 3`, infinite, NaN, or outside the declared input range.
    InvalidInput,
    /// The array width is not `1`, `4`, `8`, or enabled width `16`.
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

/// Bounded scalar `cbrt(x)` using [`CbrtfOptions::new()`].
#[inline(always)]
pub fn try_cbrt(x: f32) -> Result<f32, CbrtfError> {
    <Cbrtf>::scalar(x)
}

/// Whole-slice `cbrt(src[i]) -> dst[i]` using [`CbrtfOptions::new()`].
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline(always)]
pub fn try_cbrt_slice(
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<CbrtfErrorKind>> {
    <Cbrtf>::slice(src, dst)
}

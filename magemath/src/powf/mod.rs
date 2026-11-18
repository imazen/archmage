//! Configurable `f32` power with explicit portability, precision, and validation.
//!
//! `Midp` and `Lowp` evaluate fixed approximations to `exp2(n * log2(x))`;
//! `BitExact` `Highp` evaluates a canonical binary64 graph; `BackendDependent`
//! `Highp` delegates to `std` when the `std` feature is enabled.
//! The default promises bit identity on accepted inputs; this is not correctly
//! rounded `powf`, and no whole-domain ULP bound is advertised.
//!
//! ```
//! use magemath::powf::{Powf, PowfOptions, try_powf};
//!
//! const ZERO_OK: u128 = PowfOptions::new().admit_zero().bits();
//! const GAMMA: u128 = PowfOptions::new().gamma().midp().bits();
//!
//! assert_eq!(try_powf(4.0, 0.5).unwrap(), 2.0);
//! assert!(try_powf(f32::NAN, 2.4).is_err());
//! assert_eq!(Powf::<ZERO_OK>::scalar(0.0, 2.4).unwrap(), 0.0);
//! let _ = Powf::<GAMMA>::scalar(0.25, 2.2).unwrap();
//! ```
//!
//! Invalid option words or inconsistent combinations fail at compile time:
//! ```compile_fail
//! let _ = magemath::powf::Powf::<3>::scalar(1.0, 2.4);
//! ```
//! ```compile_fail
//! use magemath::powf::{Powf, PowfOptions};
//! const BAD: u128 = PowfOptions::new().full_domain().bits();
//! let _ = Powf::<BAD>::scalar(1.0, 2.4);
//! ```

pub(crate) mod const_api;
mod dispatch;
mod graph;
mod highp;
mod highp_data;
mod highp_dispatch;
mod highp_scalar;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
mod highp_simd;
mod options;
mod std_probe;
#[cfg(test)]
mod tests;
#[cfg(target_arch = "wasm32")]
mod wasm;

pub use const_api::{Powf, PowfRuntime};
pub use options::{MAX_EXP_BOUND, PowfOptions, PowfPrecision};

/// Reason a complete vector was rejected.
///
/// Additional rejection reasons may be added. Callers must retain a fallback:
///
/// ```compile_fail
/// use magemath::powf::PowfErrorKind;
/// fn classify(kind: PowfErrorKind) -> u8 {
///     match kind {
///         PowfErrorKind::InvalidExponent => 1,
///         PowfErrorKind::InvalidBase => 2,
///         PowfErrorKind::OutsideApproximationRange => 3,
///         PowfErrorKind::UnsupportedWidth => 4,
///     }
/// }
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PowfErrorKind {
    /// An exponent is NaN, infinite, a nonzero subnormal, or outside the
    /// configured exponent bounds.
    InvalidExponent,
    /// A base is outside the configured input domain.
    InvalidBase,
    /// The approximation's computed `n * log2(x)` is outside `[-126, 127.5]`,
    /// or a restricted `Highp` result is outside its normal-output domain.
    OutsideApproximationRange,
    /// The array width is not `1`, `4`, `8`, or enabled width `16`.
    UnsupportedWidth,
    /// An unsupported combination of precision, portability, domain, or features.
    UnsupportedOptions,
}

/// Whole-vector rejection. No partial result is returned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PowfError {
    /// Which part of the contract failed.
    pub kind: PowfErrorKind,
    /// Lanes failing the reported stage, low bit first. Later stages are not
    /// examined. An invalid uniform exponent selects every lane; an unsupported
    /// array width uses zero.
    pub lane_mask: u16,
}

impl core::fmt::Display for PowfError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "powf {:?}, lane mask {:#06x}", self.kind, self.lane_mask)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PowfError {}

/// Bounded scalar `base^exponent` using [`PowfOptions::new()`].
#[inline(always)]
pub fn try_powf(base: f32, exponent: f32) -> Result<f32, PowfError> {
    <Powf>::scalar(base, exponent)
}

/// Whole-slice `src[i]^exponent -> dst[i]` using [`PowfOptions::new()`].
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline(always)]
pub fn try_powf_slice(
    src: &[f32],
    dst: &mut [f32],
    exponent: f32,
) -> Result<(), crate::SliceError<PowfErrorKind>> {
    <Powf>::slice(src, dst, exponent)
}

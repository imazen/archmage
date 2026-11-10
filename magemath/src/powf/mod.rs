//! Configurable f32 power with explicit portability, precision and validation.
//!
//! MIDP and LOWP evaluate fixed approximations to `exp2(n * log2(x))`;
//! BitExact HIGHP evaluates a canonical binary64 graph. BackendDependent HIGHP
//! delegates to std when the `std` feature is enabled.
//! The default promises bit identity on accepted inputs; this is not
//! correctly rounded `powf`, and no whole-domain ULP bound is advertised.
//! Round-to-nearest and gradual underflow are required. NaN/infinite inputs,
//! negative bases, subnormal inputs and unsupported output ranges are rejected
//! by restricted Checked presets. Full HIGHP permits all f32 inputs/results; NaN payloads are unspecified.
//! PositiveGamma excludes zero from Gamma and removes its result masks.
//! AssumeValid omits numerical checks while retaining memory safety.
//! Existing `pow_lowp`/`pow_midp` methods retain their existing behavior.
//!
//! ```
//! use magemath::powf::{try_powf, try_powf_with, PowfDomain, PowfOptions};
//! assert_eq!(try_powf(4.0, 0.5).unwrap(), 2.0);
//! assert!(try_powf(f32::NAN, 2.4).is_err());
//! assert_eq!(try_powf_with::<{ PowfOptions::new().with_domain(PowfDomain::NonNegativeNormal).bits() }>(0.0, 2.4).unwrap(), 0.0);
//! ```

pub(crate) mod const_api;
mod dispatch;
pub use const_api::{try_powf_array_with, try_powf_with, try_powf_with_token};
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
mod std_probe;
#[cfg(target_arch = "wasm32")]
mod wasm;

mod options;
pub use options::{
    PowfDomain, PowfExecution, PowfOptions, PowfPortability, PowfPrecision, PowfValidation,
};

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
    /// selected domain's exponent bounds.
    InvalidExponent,
    /// A base is outside the selected input domain.
    InvalidBase,
    /// The approximation's computed `n * log2(x)` is outside [-126,127.5],
    /// or a restricted HIGHP result is outside its normal-output domain.
    OutsideApproximationRange,
    /// The array width is not 1, 4, 8, or enabled width 16.
    UnsupportedWidth,
    /// An unsupported combination of precision, portability, domain or features.
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

/// Bounded scalar power using the default positive-normal domain.
#[inline(always)]
pub fn try_powf(base: f32, exponent: f32) -> Result<f32, PowfError> {
    try_powf_with::<{ PowfOptions::DEFAULT }>(base, exponent)
}

/// Scalar power with explicit runtime policy options.
///
/// MIDP/LOWP scalar entries share their graph with every vector width, including
/// range reduction, division, nearest-even splitting and fused Horner steps.
#[inline(always)]
pub fn try_powf_runtime(base: f32, exponent: f32, options: PowfOptions) -> Result<f32, PowfError> {
    try_powf_array_runtime([base], exponent, options).map(|result| result[0])
}

/// Version-independent array boundary for widths 1/4/8 and, with `w512`, 16.
///
/// Checked restricted presets require every lane to be accepted. Full HIGHP
/// permits all f32 inputs/results. AssumeValid removes numerical checks.
/// The exponent is uniform across the array; in restricted presets,
/// it must be finite and either normal or zero. The computed exp2 argument
/// must lie in [-126, 127.5], excluding overflow/subnormal-result regions.
/// This bound refers to the approximation's computed argument, not an exact
/// mathematical power or an independent accuracy guarantee.
#[inline(always)]
pub fn try_powf_array_runtime<const N: usize>(
    bases: [f32; N],
    exponent: f32,
    options: PowfOptions,
) -> Result<[f32; N], PowfError> {
    try_powf_token_runtime(archmage::ScalarToken, bases, exponent, options)
}

#[inline(always)]
pub(crate) fn try_powf_token_runtime<T: archmage::SimdToken, const N: usize>(
    token: T,
    bases: [f32; N],
    exponent: f32,
    options: PowfOptions,
) -> Result<[f32; N], PowfError> {
    const_api::runtime(token, bases, exponent, options)
}

/// Configured powf on a 4-lane vector, preserving the caller's feature proof.
/// Use the same options and token with the scalar `*_with_token` entry for tails.
#[inline(always)]
pub fn try_powf_f32x4_with<const OPTIONS: u32, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    input: magetypes::simd::generic::f32x4<T>,
    exponent: f32,
) -> Result<magetypes::simd::generic::f32x4<T>, PowfError> {
    const_api::run4::<T, OPTIONS>(token, input, exponent)
}

/// Configured powf on a 8-lane vector, preserving the caller's feature proof.
/// Use the same options and token with the scalar `*_with_token` entry for tails.
#[inline(always)]
pub fn try_powf_f32x8_with<const OPTIONS: u32, T: magetypes::simd::backends::F32x8Convert>(
    token: T,
    input: magetypes::simd::generic::f32x8<T>,
    exponent: f32,
) -> Result<magetypes::simd::generic::f32x8<T>, PowfError> {
    const_api::run8::<T, OPTIONS>(token, input, exponent)
}

/// Configured powf on a 16-lane vector, preserving the caller's feature proof.
/// Use the same options and token with the scalar `*_with_token` entry for tails.
#[cfg(feature = "w512")]
#[inline(always)]
pub fn try_powf_f32x16_with<const OPTIONS: u32, T: magetypes::simd::backends::F32x16Convert>(
    token: T,
    input: magetypes::simd::generic::f32x16<T>,
    exponent: f32,
) -> Result<magetypes::simd::generic::f32x16<T>, PowfError> {
    const_api::run16::<T, OPTIONS>(token, input, exponent)
}

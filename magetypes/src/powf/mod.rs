//! Bounded f32 power with one fused arithmetic graph across backends.
//!
//! These functions evaluate a fixed approximation to `exp2(n * log2(x))`.
//! Bit identity on accepted inputs is the portability contract; this is not
//! correctly rounded `powf`, and no whole-domain ULP bound is advertised.
//! Round-to-nearest and gradual underflow are required. NaN/infinite inputs,
//! negative bases, subnormal inputs and unsupported output ranges are rejected.
//! Existing `pow_lowp`/`pow_midp` methods retain their existing behavior.
//!
//! ```
//! use magetypes::powf::{try_powf, try_powf_with, PowfDomain, PowfOptions};
//! assert_eq!(try_powf(4.0, 0.5).unwrap(), 2.0);
//! assert!(try_powf(f32::NAN, 2.4).is_err());
//! let options = PowfOptions { domain: PowfDomain::NonNegativeNormal, ..PowfOptions::default() };
//! assert_eq!(try_powf_with(0.0, 2.4, options).unwrap(), 0.0);
//! ```

mod dispatch;
mod graph;
mod std_probe;
#[cfg(test)]
mod tests;
#[cfg(target_arch = "wasm32")]
mod wasm;

/// Supported base domain. All domains reject negative zero and subnormals.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PowfDomain {
    /// Strictly positive, finite, normal bases.
    #[default]
    PositiveNormal,
    /// Also admit positive zero when the exponent is nonnegative.
    /// `0^0` is 1; `0^n` for positive `n` is positive zero.
    NonNegativeNormal,
    /// Positive zero or a base in [2^-16, 2^16], with exponent in [0, 4].
    /// Includes normalized 8/16-bit integer-channel gamma inputs. The bounds prove the
    /// computed exp2 argument stays in range, removing its rejection branch.
    Gamma,
}

/// Control autonomous selection independently of the numerical contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PowfExecution {
    /// Reuse a capable caller token, otherwise select an available native tier.
    /// Scalar fallback can use std FMA after its cached correctness screen.
    /// Runtime token disabling is respected.
    #[default]
    Automatic,
    /// Preserve the caller's tier. Scalar callers use the exact software graph.
    /// This also provides a stable entry for backend tests without upgrades.
    Caller,
}

/// Numerical domain and execution selection for bounded f32 power.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PowfOptions {
    /// Accepted base domain.
    pub domain: PowfDomain,
    /// Whether a caller without native FMA can upgrade for the whole operation.
    pub execution: PowfExecution,
}

/// Reason a complete vector was rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PowfErrorKind {
    /// An exponent is NaN, infinite, a nonzero subnormal, or outside the
    /// selected domain's exponent bounds.
    InvalidExponent,
    /// A base is outside the selected input domain.
    InvalidBase,
    /// The computed `n * log2(x)` is outside the inclusive interval [-126, 127.5].
    OutsideApproximationRange,
    /// The array width is not 1, 4, 8, or enabled width 16.
    UnsupportedWidth,
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
    try_powf_with(base, exponent, PowfOptions::default())
}

/// Bounded scalar power with explicit domain/execution options.
///
/// The scalar entry shares the same graph as every vector width, including
/// range reduction, division, nearest-even splitting and fused Horner steps.
#[inline(always)]
pub fn try_powf_with(base: f32, exponent: f32, options: PowfOptions) -> Result<f32, PowfError> {
    try_powf_array_with([base], exponent, options).map(|result| result[0])
}

/// Version-independent array boundary for widths 1/4/8 and, with `w512`, 16.
///
/// Every lane must be accepted. The exponent is uniform across the array;
/// it must be finite and either normal or zero. The computed exp2 argument
/// must lie in [-126, 127.5], excluding overflow/subnormal-result regions.
/// This bound refers to the approximation's computed argument, not an exact
/// mathematical power or an independent accuracy guarantee.
#[inline(always)]
pub fn try_powf_array_with<const N: usize>(
    bases: [f32; N],
    exponent: f32,
    options: PowfOptions,
) -> Result<[f32; N], PowfError> {
    try_powf_token_with(archmage::ScalarToken, bases, exponent, options)
}

#[inline(always)]
pub(crate) fn try_powf_token_with<T: archmage::SimdToken, const N: usize>(
    token: T,
    bases: [f32; N],
    exponent: f32,
    options: PowfOptions,
) -> Result<[f32; N], PowfError> {
    if !matches!(N, 1 | 4 | 8) && !(cfg!(feature = "w512") && N == 16) {
        return Err(PowfError {
            kind: PowfErrorKind::UnsupportedWidth,
            lane_mask: 0,
        });
    }
    let bits = exponent.to_bits() & 0x7fff_ffff;
    if bits >= 0x7f80_0000
        || (bits != 0 && bits < 0x0080_0000)
        || (options.domain == PowfDomain::Gamma && !(0.0..=4.0).contains(&exponent))
    {
        return Err(PowfError {
            kind: PowfErrorKind::InvalidExponent,
            lane_mask: ((1_u32 << N) - 1) as u16,
        });
    }
    let plan = dispatch::select(token, options.execution);
    let mut output = [0.0; N];
    if N == 1 {
        let result =
            dispatch::run4(plan, [bases[0]; 4], exponent, options.domain).map_err(|error| {
                PowfError {
                    lane_mask: 1,
                    ..error
                }
            })?;
        output[0] = result[0];
    } else if N == 4 {
        let mut input = [0.0; 4];
        input.copy_from_slice(&bases);
        output.copy_from_slice(&dispatch::run4(plan, input, exponent, options.domain)?);
    } else if N == 8 {
        let mut input = [0.0; 8];
        input.copy_from_slice(&bases);
        output.copy_from_slice(&dispatch::run8(plan, input, exponent, options.domain)?);
    } else {
        #[cfg(feature = "w512")]
        {
            let mut input = [0.0; 16];
            input.copy_from_slice(&bases);
            output.copy_from_slice(&dispatch::run16(plan, input, exponent, options.domain)?);
        }
    }
    Ok(output)
}

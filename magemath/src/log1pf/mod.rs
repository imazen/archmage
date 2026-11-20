//! Configurable `f32` `ln(1 + x)` (`log1p`) on `(-1, +∞)` with explicit
//! portability, precision, subnormal policy, and validation.
//!
//! Separated from [`crate::logf`] because `log1p(x)` is centered at `x = 0`
//! (`ln(1 + 0) = 0`, singularity at `x = -1`) rather than `x > 0`.
//!
//! ```
//! use magemath::log1pf::{Log1pf, Log1pfDomain, Log1pfOptions, try_log1p};
//!
//! const SMALL: u128 = Log1pfOptions::new().domain(Log1pfDomain::AroundZero).bits();
//! assert!(Log1pfOptions::from_bits(SMALL).unwrap().elides_reduction());
//! assert_eq!(try_log1p(0.0).unwrap(), 0.0);
//!
//! let src = [-0.125, 0.0, 0.125, 0.25];
//! let mut dst = [0.0; 4];
//! Log1pf::<SMALL>::slice(&src, &mut dst).unwrap();
//! assert_eq!(dst[1], 0.0);
//! ```

pub(crate) mod const_api;
mod options;

pub use const_api::{Log1pf, Log1pfRuntime};
pub use options::{Log1pfDomain, Log1pfOptions, Log1pfPrecision};

/// Reason a complete vector was rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Log1pfErrorKind {
    /// An input fails the domain, subnormal policy, or declared input range.
    InvalidInput,
    /// The array width is not `1`, `4`, `8`, or enabled width `16`.
    UnsupportedWidth,
    /// An unsupported combination of precision, portability, domain, or features.
    UnsupportedOptions,
}

/// Whole-vector rejection. No partial result is returned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Log1pfError {
    /// Which part of the contract failed.
    pub kind: Log1pfErrorKind,
    /// Lanes failing the reported stage, low bit first.
    pub lane_mask: u16,
}

impl core::fmt::Display for Log1pfError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "log1pf {:?}, lane mask {:#06x}",
            self.kind, self.lane_mask
        )
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Log1pfError {}

/// Scalar `ln(1 + x)` using [`Log1pfOptions::new()`].
#[inline(always)]
pub fn try_log1p(x: f32) -> Result<f32, Log1pfError> {
    <Log1pf>::scalar(x)
}

/// Whole-slice `ln(1 + src[i]) -> dst[i]` using [`Log1pfOptions::new()`].
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline(always)]
pub fn try_log1p_slice(
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<Log1pfErrorKind>> {
    <Log1pf>::slice(src, dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log1p_basics_and_subnormal_policies() {
        const C: u128 = Log1pfOptions::new().caller().bits();
        const REJECT: u128 = Log1pfOptions::new().reject_subnormals().caller().bits();
        const NORM: u128 = Log1pfOptions::new().admit_subnormals().caller().bits();
        const S: u128 = Log1pfOptions::new().full_domain().caller().bits();
        assert_eq!(Log1pf::<C>::scalar(0.0), Ok(0.0));
        assert_eq!(
            Log1pf::<C>::scalar(-0.0).unwrap().to_bits(),
            (-0.0_f32).to_bits()
        );
        assert_eq!(
            Log1pf::<C>::scalar(1.0).unwrap() - core::f32::consts::LN_2,
            0.0
        );
        assert!((Log1pf::<C>::scalar(-0.5).unwrap() + core::f32::consts::LN_2).abs() < 1e-6);
        for x in [1e-8f32, -1e-8, 1e-20, 2.5e-5, -0.4142] {
            let got = Log1pf::<C>::scalar(x).unwrap();
            let r = (x as f64).ln_1p() as f32;
            assert_eq!(got, r, "log1p({x}): {got} vs {r}");
        }
        // Subnormals flush to signed zero by default, with explicit reject and normalize policies.
        assert_eq!(
            Log1pf::<REJECT>::scalar(f32::from_bits(1))
                .unwrap_err()
                .kind,
            Log1pfErrorKind::InvalidInput
        );
        assert_eq!(Log1pf::<C>::scalar(f32::from_bits(1)).unwrap().to_bits(), 0);
        assert_eq!(
            Log1pf::<C>::scalar(-f32::from_bits(1)).unwrap().to_bits(),
            0x8000_0000
        );
        assert_eq!(
            Log1pf::<NORM>::scalar(f32::from_bits(1)).unwrap().to_bits(),
            1
        );
        assert_eq!(
            Log1pf::<NORM>::scalar(-f32::from_bits(1))
                .unwrap()
                .to_bits(),
            0x8000_0001
        );
        // Full domain
        assert_eq!(Log1pf::<S>::scalar(-1.0), Ok(f32::NEG_INFINITY));
        assert!(Log1pf::<S>::scalar(-1.5).unwrap().is_nan());
        assert_eq!(Log1pf::<S>::scalar(f32::INFINITY), Ok(f32::INFINITY));
        assert_eq!(
            Log1pf::<C>::scalar(-1.0).unwrap_err().kind,
            Log1pfErrorKind::InvalidInput
        );
        assert_eq!(
            Log1pf::<C>::scalar(-2.0).unwrap_err().kind,
            Log1pfErrorKind::InvalidInput
        );
        assert_eq!(
            Log1pf::<C>::array::<1>([0.5]),
            Ok([f32::from_bits(0x3ecf_991f)])
        );
        assert_eq!(
            Log1pfOptions::new().runtime().scalar(1.0).unwrap(),
            core::f32::consts::LN_2
        );
    }
}

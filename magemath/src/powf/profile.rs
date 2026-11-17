//! Typed range-as-data profiles for the power family.
//!
//! A profile type implements [`PowfProfile`], declaring the accepted base and
//! exponent bounds in a `const` [`PowfSpec`]. When those ranges prove the
//! computed `n * log2(x)` stays inside the approximation's range, the kernel
//! elides the per-lane range check and clamp at compile time.
//!
//! ```
//! use magemath::powf::{PowfProfile, PowfSpec, try_powf_slice_with};
//!
//! struct Encode;
//! impl PowfProfile for Encode {
//!     const SPEC: PowfSpec = PowfSpec::new()
//!         .base_range(1.0 / 4096.0, 1.0)
//!         .exponent_range(0.25, 3.0)
//!         .admit_zero();
//! }
//! assert!(Encode::SPEC.elides_range_check());
//!
//! let src = [0.0, 0.25, 1.0];
//! let mut dst = [0.0; 3];
//! try_powf_slice_with::<Encode>(&src, &mut dst, 0.5).unwrap();
//! assert_eq!(dst, [0.0, 0.5, 1.0]);
//! ```

use super::{PowfDomain, PowfOptions, PowfPrecision};
use crate::{Execution, Portability, Validation};

/// Largest `|n * log2(x)|` a profile may reach before the kernel keeps its
/// range check. The approximation needs the argument in `[-126, 127.5]`.
const ELIDE_LIMIT: f32 = 120.0;

const fn abs(x: f32) -> f32 {
    if x < 0.0 { -x } else { x }
}

/// Unbiased binary exponent of a positive normal value.
const fn exponent_of(x: f32) -> i32 {
    ((x.to_bits() >> 23) & 0xff) as i32 - 127
}

/// A power configuration built in a `const`.
///
/// Fields are private so additional policy axes can be added without breaking
/// existing [`PowfProfile`] implementations.
#[derive(Clone, Copy, Debug)]
pub struct PowfSpec {
    base: Option<(f32, f32)>,
    exponent: Option<(f32, f32)>,
    zero: bool,
    full: bool,
    clamp: bool,
    precision: PowfPrecision,
    portability: Portability,
    validation: Validation,
    execution: Execution,
}

impl PowfSpec {
    /// Positive normal bases, any finite normal-or-zero exponent, MIDP,
    /// BitExact, Checked, Automatic execution.
    pub const fn new() -> Self {
        Self {
            base: None,
            exponent: None,
            zero: false,
            full: false,
            clamp: false,
            precision: PowfPrecision::Midp,
            portability: Portability::BitExact,
            validation: Validation::Checked,
            execution: Execution::Automatic,
        }
    }

    /// Translate an encoded [`PowfOptions`] value into an equivalent [`PowfSpec`].
    pub const fn from_options(options: PowfOptions) -> Self {
        let spec = Self::new()
            .precision(options.precision)
            .portability(options.portability)
            .validation(options.validation)
            .execution(options.execution);
        match options.domain {
            PowfDomain::PositiveNormal => spec,
            PowfDomain::NonNegativeNormal => spec.admit_zero(),
            PowfDomain::Gamma => spec
                .base_range(1.0 / 65_536.0, 65_536.0)
                .exponent_range(0.0, 4.0)
                .admit_zero(),
            PowfDomain::PositiveGamma => spec
                .base_range(1.0 / 65_536.0, 65_536.0)
                .exponent_range(0.0, 4.0),
            PowfDomain::Full => spec.full_domain(),
        }
    }

    pub(crate) const fn from_bits_const(bits: u32) -> Self {
        let Some(opts) = PowfOptions::from_bits(bits) else {
            panic!("invalid powf const options");
        };
        Self::from_options(opts)
    }

    /// Accept bases in `[lo, hi]`. Both must be positive normal finite `f32`s.
    pub const fn base_range(mut self, lo: f32, hi: f32) -> Self {
        assert!(
            lo >= f32::MIN_POSITIVE && lo <= hi && hi <= f32::MAX,
            "base range must be positive normal with lo <= hi"
        );
        self.base = Some((lo, hi));
        self
    }

    /// Accept exponents in `[lo, hi]`. Both must be finite `f32`s.
    pub const fn exponent_range(mut self, lo: f32, hi: f32) -> Self {
        assert!(
            lo <= hi && abs(lo) <= f32::MAX && abs(hi) <= f32::MAX,
            "exponent range must be finite with lo <= hi"
        );
        self.exponent = Some((lo, hi));
        self
    }

    /// Also accept `+0`: `0^0 = 1`, `0^positive = +0`. Negative exponents
    /// with a zero base remain rejected under [`Validation::Checked`].
    pub const fn admit_zero(mut self) -> Self {
        self.zero = true;
        self
    }

    /// Accept every `f32` input and output (for [`PowfPrecision::Highp`] only).
    pub const fn full_domain(mut self) -> Self {
        self.full = true;
        self
    }

    /// Clamp out-of-range MIDP/LOWP exp2 outputs instead of assuming range.
    pub const fn clamp_output(mut self) -> Self {
        self.clamp = true;
        self
    }

    /// Select the approximation family (`Midp`, `Lowp`, or `Highp`).
    pub const fn precision(mut self, value: PowfPrecision) -> Self {
        self.precision = value;
        self
    }

    /// Select cross-backend agreement.
    pub const fn portability(mut self, value: Portability) -> Self {
        self.portability = value;
        self
    }

    /// Select numerical precondition enforcement.
    pub const fn validation(mut self, value: Validation) -> Self {
        self.validation = value;
        self
    }

    /// Select whole-operation backend dispatch.
    pub const fn execution(mut self, value: Execution) -> Self {
        self.execution = value;
        self
    }

    /// Validate that this profile is self-consistent for the enabled features.
    pub const fn validate(&self) {
        assert!(
            !self.full || matches!(self.precision, PowfPrecision::Highp),
            "full_domain requires PowfPrecision::Highp"
        );
        assert!(
            !matches!(self.precision, PowfPrecision::Highp)
                || !matches!(self.portability, Portability::BackendDependent)
                || cfg!(feature = "std"),
            "BackendDependent Highp requires the `std` feature"
        );
        assert!(
            !(self.full && (self.base.is_some() || self.exponent.is_some())),
            "full_domain cannot be combined with base_range or exponent_range"
        );
        assert!(
            !(self.clamp && matches!(self.precision, PowfPrecision::Highp)),
            "clamp_output applies to Midp and Lowp, not Highp"
        );
    }

    /// Whether the declared ranges prove the computed `n * log2(x)` stays in
    /// range, so the kernel emits no range check or clamp.
    ///
    /// Needs both a base and an exponent range (and not `full_domain`), with
    /// `(E + 1) * N <= 120` and no admitted zero with negative exponents.
    pub const fn elides_range_check(&self) -> bool {
        if self.full {
            return false;
        }
        let (Some((lo, hi)), Some((nlo, nhi))) = (self.base, self.exponent) else {
            return false;
        };
        let (elo, ehi) = (exponent_of(lo), exponent_of(hi) + 1);
        let e = (if -elo > ehi { -elo } else { ehi }) + 1;
        let e = if e < 1 { 1 } else { e };
        let n = if abs(nlo) > abs(nhi) {
            abs(nlo)
        } else {
            abs(nhi)
        };
        (e as f32) * n <= ELIDE_LIMIT && !(self.zero && nlo < 0.0)
    }

    pub(crate) const fn base_bounds(&self) -> Option<(f32, f32)> {
        self.base
    }
    pub(crate) const fn exponent_bounds(&self) -> Option<(f32, f32)> {
        self.exponent
    }
    pub(crate) const fn admits_zero(&self) -> bool {
        self.zero
    }
    pub(crate) const fn is_full(&self) -> bool {
        self.full
    }
    pub(crate) const fn should_clamp(&self) -> bool {
        self.clamp
    }
    pub(crate) const fn is_lowp(&self) -> bool {
        matches!(self.precision, PowfPrecision::Lowp)
    }
    pub(crate) const fn is_highp(&self) -> bool {
        matches!(self.precision, PowfPrecision::Highp)
    }
    pub(crate) const fn is_backend(&self) -> bool {
        matches!(self.portability, Portability::BackendDependent)
    }
    pub(crate) const fn assume_valid(&self) -> bool {
        matches!(self.validation, Validation::AssumeValid)
    }
    pub(crate) const fn is_caller(&self) -> bool {
        matches!(self.execution, Execution::Caller)
    }
}

impl Default for PowfSpec {
    fn default() -> Self {
        Self::new()
    }
}

/// A named power configuration evaluated at compile time.
pub trait PowfProfile: 'static {
    /// The configuration specification.
    const SPEC: PowfSpec;
}

/// Profile adapter constructed from legacy [`PowfOptions::bits()`] integer bits.
#[derive(Clone, Copy, Debug)]
pub struct Options<const BITS: u32>;

impl<const BITS: u32> PowfProfile for Options<BITS> {
    const SPEC: PowfSpec = PowfSpec::from_bits_const(BITS);
}

impl<P: PowfProfile> PowfProfile for crate::Caller<P> {
    const SPEC: PowfSpec = P::SPEC.execution(Execution::Caller);
}

impl<P: PowfProfile> PowfProfile for crate::Unchecked<P> {
    const SPEC: PowfSpec = P::SPEC.validation(Validation::AssumeValid);
}

impl<P: PowfProfile> PowfProfile for crate::Backend<P> {
    const SPEC: PowfSpec = P::SPEC.portability(Portability::BackendDependent);
}

/// Built-in power profiles mirroring the standard presets.
pub mod presets {
    use super::{PowfOptions, PowfProfile, PowfSpec};

    /// BitExact MIDP on positive normal bases, checked, automatic execution.
    #[derive(Clone, Copy, Debug)]
    pub struct NormalMidp;
    impl PowfProfile for NormalMidp {
        const SPEC: PowfSpec = PowfSpec::from_options(PowfOptions::PORTABLE_NORMAL_MIDP);
    }

    /// BitExact LOWP on positive normal bases, checked, automatic execution.
    #[derive(Clone, Copy, Debug)]
    pub struct NormalLowp;
    impl PowfProfile for NormalLowp {
        const SPEC: PowfSpec = PowfSpec::from_options(
            PowfOptions::PORTABLE_NORMAL_MIDP.with_precision(super::PowfPrecision::Lowp),
        );
    }

    /// BitExact MIDP on nonnegative normal bases (`+0` and positive normals).
    #[derive(Clone, Copy, Debug)]
    pub struct NonNegativeNormalMidp;
    impl PowfProfile for NonNegativeNormalMidp {
        const SPEC: PowfSpec = PowfSpec::new().admit_zero();
    }

    /// BitExact MIDP on the Gamma domain (`[2^-16, 2^16]`, `+0`, exponent `[0, 4]`).
    #[derive(Clone, Copy, Debug)]
    pub struct GammaMidp;
    impl PowfProfile for GammaMidp {
        const SPEC: PowfSpec = PowfSpec::from_options(PowfOptions::PORTABLE_GAMMA_MIDP);
    }

    /// BitExact LOWP on the Gamma domain (`[2^-16, 2^16]`, `+0`, exponent `[0, 4]`).
    #[derive(Clone, Copy, Debug)]
    pub struct GammaLowp;
    impl PowfProfile for GammaLowp {
        const SPEC: PowfSpec = PowfSpec::from_options(PowfOptions::PORTABLE_GAMMA_LOWP);
    }

    /// BitExact MIDP on the positive Gamma domain (no zero handling).
    #[derive(Clone, Copy, Debug)]
    pub struct PositiveGammaMidp;
    impl PowfProfile for PositiveGammaMidp {
        const SPEC: PowfSpec = PowfSpec::from_options(PowfOptions::PORTABLE_POSITIVE_GAMMA_MIDP);
    }

    /// BitExact LOWP on the positive Gamma domain (no zero handling).
    #[derive(Clone, Copy, Debug)]
    pub struct PositiveGammaLowp;
    impl PowfProfile for PositiveGammaLowp {
        const SPEC: PowfSpec = PowfSpec::from_options(PowfOptions::PORTABLE_POSITIVE_GAMMA_LOWP);
    }

    /// Canonical BitExact HIGHP on the Gamma domain.
    #[derive(Clone, Copy, Debug)]
    pub struct GammaHighp;
    impl PowfProfile for GammaHighp {
        const SPEC: PowfSpec = PowfSpec::from_options(PowfOptions::PORTABLE_GAMMA_HIGHP);
    }

    /// Canonical BitExact HIGHP on the full `f32` domain.
    #[derive(Clone, Copy, Debug)]
    pub struct FullHighp;
    impl PowfProfile for FullHighp {
        const SPEC: PowfSpec = PowfSpec::from_options(PowfOptions::PORTABLE_HIGHP);
    }

    /// Backend-dependent std HIGHP on the full `f32` domain (`std` feature).
    #[cfg(feature = "std")]
    #[derive(Clone, Copy, Debug)]
    pub struct StdHighp;
    #[cfg(feature = "std")]
    impl PowfProfile for StdHighp {
        const SPEC: PowfSpec = PowfSpec::from_options(PowfOptions::STD_HIGHP);
    }
}

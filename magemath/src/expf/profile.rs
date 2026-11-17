//! Typed range-as-data profiles for the exponential family (`exp2`, `exp`).
//!
//! Declaring an [`ExpfSpec::input_range`] inside `[-126, 127.5]` (`127.99`
//! for LOWP) proves no overflow/underflow clamping is needed and elides the
//! clamp sequence at compile time.

use super::{ExpfDomain, ExpfOptions, ExpfPrecision};
use crate::{Execution, Portability, Validation};

const fn abs(x: f32) -> f32 {
    if x < 0.0 { -x } else { x }
}

/// An exponential configuration built in a `const`.
///
/// Fields are private so additional policy axes can be added without breaking
/// existing [`ExpfProfile`] implementations.
#[derive(Clone, Copy, Debug)]
pub struct ExpfSpec {
    input: Option<(f32, f32)>,
    full: bool,
    precision: ExpfPrecision,
    portability: Portability,
    validation: Validation,
    execution: Execution,
}

impl ExpfSpec {
    /// Bounded finite domain, MIDP, BitExact, Checked, Automatic execution.
    pub const fn new() -> Self {
        Self {
            input: None,
            full: false,
            precision: ExpfPrecision::Midp,
            portability: Portability::BitExact,
            validation: Validation::Checked,
            execution: Execution::Automatic,
        }
    }

    /// Translate an encoded [`ExpfOptions`] value into an equivalent [`ExpfSpec`].
    pub const fn from_options(options: ExpfOptions) -> Self {
        Self::new()
            .domain(options.domain)
            .precision(options.precision)
            .portability(options.portability)
            .validation(options.validation)
            .execution(options.execution)
    }

    pub(crate) const fn from_bits_const(bits: u32) -> Self {
        let Some(opts) = ExpfOptions::from_bits(bits) else {
            panic!("invalid expf const options");
        };
        Self::from_options(opts)
    }

    /// Accept inputs in `[lo, hi]`. Both must be finite `f32`s with `lo <= hi`.
    pub const fn input_range(mut self, lo: f32, hi: f32) -> Self {
        assert!(
            lo <= hi && abs(lo) <= f32::MAX && abs(hi) <= f32::MAX,
            "input range must be finite with lo <= hi"
        );
        self.input = Some((lo, hi));
        self
    }

    /// Select the domain (`Bounded` or `Full`).
    pub const fn domain(mut self, value: ExpfDomain) -> Self {
        self.full = matches!(value, ExpfDomain::Full);
        self
    }

    /// Accept every `f32` input (`NaN -> NaN`, `+inf -> +inf`, `-inf -> +0`).
    pub const fn full_domain(mut self) -> Self {
        self.full = true;
        self
    }

    /// Select the approximation family (`Midp` or `Lowp`).
    pub const fn precision(mut self, value: ExpfPrecision) -> Self {
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

    /// Validate that this profile is self-consistent.
    pub const fn validate(&self) {
        assert!(
            !(self.full && self.input.is_some()),
            "full_domain cannot be combined with input_range"
        );
    }

    /// Whether `exp2(x)` keeps `x` in the un-clamped range (`[-126, 127.5]`
    /// MIDP or `[-126, 127.99]` LOWP), allowing clamp elision.
    pub const fn elides_clamp(&self) -> bool {
        if self.full {
            return false;
        }
        let Some((lo, hi)) = self.input else {
            return false;
        };
        let hi_limit = if matches!(self.precision, ExpfPrecision::Lowp) {
            127.99
        } else {
            127.5
        };
        lo >= -126.0 && hi <= hi_limit
    }

    /// Whether `exp(x)` keeps `x * log2(e)` in the un-clamped range, allowing
    /// clamp elision.
    pub const fn elides_exp_clamp(&self) -> bool {
        if self.full {
            return false;
        }
        let Some((lo, hi)) = self.input else {
            return false;
        };
        let hi_limit = if matches!(self.precision, ExpfPrecision::Lowp) {
            127.99
        } else {
            127.5
        };
        let s_lo = lo * core::f32::consts::LOG2_E;
        let s_hi = hi * core::f32::consts::LOG2_E;
        s_lo >= -126.0 && s_hi <= hi_limit
    }

    pub(crate) const fn input_bounds(&self) -> Option<(f32, f32)> {
        self.input
    }
    pub(crate) const fn is_full(&self) -> bool {
        self.full
    }
    pub(crate) const fn is_lowp(&self) -> bool {
        matches!(self.precision, ExpfPrecision::Lowp)
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

impl Default for ExpfSpec {
    fn default() -> Self {
        Self::new()
    }
}

/// A named exponential configuration evaluated at compile time.
pub trait ExpfProfile: 'static {
    /// The configuration specification.
    const SPEC: ExpfSpec;
}

/// Profile adapter constructed from legacy [`ExpfOptions::bits()`] integer bits.
#[derive(Clone, Copy, Debug)]
pub struct Options<const BITS: u32>;

impl<const BITS: u32> ExpfProfile for Options<BITS> {
    const SPEC: ExpfSpec = ExpfSpec::from_bits_const(BITS);
}

impl<P: ExpfProfile> ExpfProfile for crate::Caller<P> {
    const SPEC: ExpfSpec = P::SPEC.execution(Execution::Caller);
}

impl<P: ExpfProfile> ExpfProfile for crate::Unchecked<P> {
    const SPEC: ExpfSpec = P::SPEC.validation(Validation::AssumeValid);
}

impl<P: ExpfProfile> ExpfProfile for crate::Backend<P> {
    const SPEC: ExpfSpec = P::SPEC.portability(Portability::BackendDependent);
}

/// Built-in exponential profiles mirroring the standard presets.
pub mod presets {
    use super::{ExpfOptions, ExpfProfile, ExpfSpec};

    /// BitExact MIDP, bounded finite domain, checked, automatic execution.
    #[derive(Clone, Copy, Debug)]
    pub struct Midp;
    impl ExpfProfile for Midp {
        const SPEC: ExpfSpec = ExpfSpec::from_options(ExpfOptions::PORTABLE_MIDP);
    }

    /// BitExact LOWP, bounded finite domain, checked, automatic execution.
    #[derive(Clone, Copy, Debug)]
    pub struct Lowp;
    impl ExpfProfile for Lowp {
        const SPEC: ExpfSpec = ExpfSpec::from_options(ExpfOptions::PORTABLE_LOWP);
    }

    /// BitExact MIDP accepting every `f32` input (`NaN`, `+inf`, `-inf`).
    #[derive(Clone, Copy, Debug)]
    pub struct MidpFull;
    impl ExpfProfile for MidpFull {
        const SPEC: ExpfSpec = ExpfSpec::from_options(ExpfOptions::PORTABLE_MIDP_FULL);
    }

    /// BitExact LOWP accepting every `f32` input (`NaN`, `+inf`, `-inf`).
    #[derive(Clone, Copy, Debug)]
    pub struct LowpFull;
    impl ExpfProfile for LowpFull {
        const SPEC: ExpfSpec = ExpfSpec::from_options(
            ExpfOptions::PORTABLE_MIDP_FULL.with_precision(super::ExpfPrecision::Lowp),
        );
    }
}

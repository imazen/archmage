//! Typed range-as-data profiles for the logarithm family
//! (`log2`, `ln`, `log10`, `log1p`).

use super::{LogfDomain, LogfOptions, LogfPrecision};
use crate::{Execution, Portability, Validation};

const fn abs(x: f32) -> f32 {
    if x < 0.0 { -x } else { x }
}

/// A logarithm configuration built in a `const`.
///
/// Fields are private so additional policy axes can be added without breaking
/// existing [`LogfProfile`] implementations.
#[derive(Clone, Copy, Debug)]
pub struct LogfSpec {
    input: Option<(f32, f32)>,
    specials: bool,
    precision: LogfPrecision,
    portability: Portability,
    validation: Validation,
    execution: Execution,
}

impl LogfSpec {
    /// Positive normal domain, MIDP, BitExact, Checked, Automatic execution.
    pub const fn new() -> Self {
        Self {
            input: None,
            specials: false,
            precision: LogfPrecision::Midp,
            portability: Portability::BitExact,
            validation: Validation::Checked,
            execution: Execution::Automatic,
        }
    }

    /// Translate an encoded [`LogfOptions`] value into an equivalent [`LogfSpec`].
    pub const fn from_options(options: LogfOptions) -> Self {
        Self::new()
            .domain(options.domain)
            .precision(options.precision)
            .portability(options.portability)
            .validation(options.validation)
            .execution(options.execution)
    }

    pub(crate) const fn from_bits_const(bits: u32) -> Self {
        let Some(opts) = LogfOptions::from_bits(bits) else {
            panic!("invalid logf const options");
        };
        Self::from_options(opts)
    }

    /// Accept inputs in `[lo, hi]`.
    ///
    /// For `log2`, `ln`, and `log10`, `lo` must be at least `f32::MIN_POSITIVE`.
    /// For `log1p`, `1 + lo` must be at least `f32::MIN_POSITIVE`. Both bounds
    /// must be finite with `lo <= hi`.
    pub const fn input_range(mut self, lo: f32, hi: f32) -> Self {
        assert!(
            lo > -1.0 && lo <= hi && abs(lo) <= f32::MAX && abs(hi) <= f32::MAX,
            "input range must be finite with -1 < lo <= hi"
        );
        self.input = Some((lo, hi));
        self
    }

    /// Select the domain (`Positive` or `Specials`).
    pub const fn domain(mut self, value: LogfDomain) -> Self {
        self.specials = matches!(value, LogfDomain::Specials);
        self
    }

    /// Enable IEEE specials (`0 -> -inf`, `<0 -> NaN`, `NaN -> NaN`, `+inf -> +inf`).
    pub const fn specials_domain(mut self) -> Self {
        self.specials = true;
        self
    }

    /// Select the approximation family (`Midp` or `Lowp`).
    pub const fn precision(mut self, value: LogfPrecision) -> Self {
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

    /// Validate domain-level invariants common to all log variants.
    pub const fn validate(&self) {
        assert!(
            !(self.specials && self.input.is_some()),
            "specials_domain cannot be combined with input_range"
        );
    }

    pub(crate) const fn validate_op(&self, op: u8) {
        self.validate();
        if let Some((lo, _)) = self.input {
            if op == 3 {
                assert!(
                    1.0 + lo >= f32::MIN_POSITIVE,
                    "log1p input_range lower bound must keep 1 + lo positive normal"
                );
            } else {
                assert!(
                    lo >= f32::MIN_POSITIVE,
                    "log input_range lower bound must be at least f32::MIN_POSITIVE"
                );
            }
        }
    }

    pub(crate) const fn input_bounds(&self) -> Option<(f32, f32)> {
        self.input
    }
    pub(crate) const fn is_specials(&self) -> bool {
        self.specials
    }
    pub(crate) const fn is_lowp(&self) -> bool {
        matches!(self.precision, LogfPrecision::Lowp)
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

impl Default for LogfSpec {
    fn default() -> Self {
        Self::new()
    }
}

/// A named logarithm configuration evaluated at compile time.
pub trait LogfProfile: 'static {
    /// The configuration specification.
    const SPEC: LogfSpec;
}

/// Profile adapter constructed from legacy [`LogfOptions::bits()`] integer bits.
#[derive(Clone, Copy, Debug)]
pub struct Options<const BITS: u32>;

impl<const BITS: u32> LogfProfile for Options<BITS> {
    const SPEC: LogfSpec = LogfSpec::from_bits_const(BITS);
}

impl<P: LogfProfile> LogfProfile for crate::Caller<P> {
    const SPEC: LogfSpec = P::SPEC.execution(Execution::Caller);
}

impl<P: LogfProfile> LogfProfile for crate::Unchecked<P> {
    const SPEC: LogfSpec = P::SPEC.validation(Validation::AssumeValid);
}

impl<P: LogfProfile> LogfProfile for crate::Backend<P> {
    const SPEC: LogfSpec = P::SPEC.portability(Portability::BackendDependent);
}

/// Built-in logarithm profiles mirroring the standard presets.
pub mod presets {
    use super::{LogfOptions, LogfProfile, LogfSpec};

    /// BitExact MIDP on positive normal inputs, checked, automatic execution.
    #[derive(Clone, Copy, Debug)]
    pub struct Midp;
    impl LogfProfile for Midp {
        const SPEC: LogfSpec = LogfSpec::from_options(LogfOptions::PORTABLE_MIDP);
    }

    /// BitExact LOWP on positive normal inputs, checked, automatic execution.
    #[derive(Clone, Copy, Debug)]
    pub struct Lowp;
    impl LogfProfile for Lowp {
        const SPEC: LogfSpec = LogfSpec::from_options(LogfOptions::PORTABLE_LOWP);
    }

    /// BitExact MIDP with IEEE specials (`0 -> -inf`, `<0 -> NaN`, `NaN -> NaN`, `+inf -> +inf`).
    #[derive(Clone, Copy, Debug)]
    pub struct MidpSpecials;
    impl LogfProfile for MidpSpecials {
        const SPEC: LogfSpec = LogfSpec::from_options(LogfOptions::PORTABLE_MIDP_SPECIALS);
    }

    /// BitExact LOWP with IEEE specials.
    #[derive(Clone, Copy, Debug)]
    pub struct LowpSpecials;
    impl LogfProfile for LowpSpecials {
        const SPEC: LogfSpec = LogfSpec::from_options(
            LogfOptions::PORTABLE_MIDP_SPECIALS.with_precision(super::LogfPrecision::Lowp),
        );
    }
}

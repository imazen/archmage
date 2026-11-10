//! Min/max policies describe values, not a floating-point accuracy grade.
/// NaN handling. Payloads, signs, signaling encoding and exception flags are unspecified.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MinMaxNaN {
    /// Return the number when exactly one operand is NaN; two NaNs yield NaN.
    #[default]
    Number,
    /// Either NaN operand makes the result NaN.
    Propagate,
    /// Backend-dependent NaN handling, including comparison-select and relaxed
    /// WASM. With one NaN either a number or NaN is allowed; two NaNs yield NaN.
    Backend,
}
/// How equal zeros are ordered.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MinMaxZero {
    /// Minimum chooses -0, maximum +0 when opposite zeros meet.
    #[default]
    Ordered,
    /// Native tie behavior; zero sign may depend on backend and operand order.
    Backend,
}
/// Accepted inputs for both operands.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MinMaxDomain {
    /// Every floating-point input, with the selected special-value policies.
    #[default]
    Full,
    /// Finite values, including zeros and subnormals.
    Finite,
    /// Finite normal values; excludes zeros, subnormals, infinities and NaNs.
    Normal,
}
/// Whether restricted-domain preconditions are checked.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MinMaxValidation {
    /// Reject the whole vector and report every invalid lane.
    #[default]
    Checked,
    /// Omit checks; invalid inputs have unspecified numerical results, never UB.
    AssumeValid,
}
/// Selection for scalar/array entries. Typed-vector and token-tail entries retain their caller.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MinMaxExecution {
    /// Select one backend for the complete operation.
    #[default]
    Automatic,
    /// Scalar/array entries stay scalar; explicit-token entries keep that token.
    Caller,
}
/// Const builder; use `.bits()` as the integer const argument.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MinMaxOptions {
    /// NaN policy.
    pub nan: MinMaxNaN,
    /// Signed-zero policy.
    pub zero: MinMaxZero,
    /// Input domain.
    pub domain: MinMaxDomain,
    /// Domain validation.
    pub validation: MinMaxValidation,
    /// Whole-operation selection.
    pub execution: MinMaxExecution,
}
impl MinMaxOptions {
    pub(crate) const ASSUME: u32 = 32;
    pub(crate) const CALLER: u32 = 64;
    /// Full domain, number-preferred NaNs, ordered zeros.
    pub const PORTABLE_NUMBER: Self = Self::new();
    /// Full domain, propagating NaNs, ordered zeros.
    pub const PORTABLE_PROPAGATE: Self = Self::new().with_nan(MinMaxNaN::Propagate);
    /// Full domain, native NaN/zero behavior; relaxed WASM is allowed.
    pub const RELAXED: Self = Self::new()
        .with_nan(MinMaxNaN::Backend)
        .with_zero(MinMaxZero::Backend);
    /// Normal inputs, checked; NaN/zero policies have nothing to repair.
    pub const NORMAL: Self = Self::new().with_domain(MinMaxDomain::Normal);
    /// Normal inputs, no validation or NaN/zero repairs.
    pub const UNCHECKED_NORMAL: Self = Self::NORMAL.with_validation(MinMaxValidation::AssumeValid);
    /// Full-domain, number-preferred, ordered-zero defaults.
    pub const fn new() -> Self {
        Self {
            nan: MinMaxNaN::Number,
            zero: MinMaxZero::Ordered,
            domain: MinMaxDomain::Full,
            validation: MinMaxValidation::Checked,
            execution: MinMaxExecution::Automatic,
        }
    }
    /// Set NaN behavior.
    pub const fn with_nan(self, nan: MinMaxNaN) -> Self {
        Self { nan, ..self }
    }
    /// Set zero tie behavior.
    pub const fn with_zero(self, zero: MinMaxZero) -> Self {
        Self { zero, ..self }
    }
    /// Set accepted input range.
    pub const fn with_domain(self, domain: MinMaxDomain) -> Self {
        Self { domain, ..self }
    }
    /// Set validation.
    pub const fn with_validation(self, validation: MinMaxValidation) -> Self {
        Self { validation, ..self }
    }
    /// Set scalar/array selection.
    pub const fn with_execution(self, execution: MinMaxExecution) -> Self {
        Self { execution, ..self }
    }
    /// Encode flags. Canonicalize vacuous full-domain validation, NaN choices
    /// on finite inputs, and NaN/zero choices on normal inputs.
    pub const fn bits(self) -> u32 {
        self.domain as u32
            | if matches!(self.domain, MinMaxDomain::Full) {
                (self.nan as u32) << 2
            } else {
                0
            }
            | if !matches!(self.domain, MinMaxDomain::Normal) {
                (self.zero as u32) << 4
            } else {
                0
            }
            | if matches!(self.validation, MinMaxValidation::AssumeValid)
                && !matches!(self.domain, MinMaxDomain::Full)
            {
                Self::ASSUME
            } else {
                0
            }
            | if matches!(self.execution, MinMaxExecution::Caller) {
                Self::CALLER
            } else {
                0
            }
    }
    pub(crate) const fn valid(flags: u32) -> bool {
        flags & !127 == 0 && flags & 3 < 3 && (flags >> 2) & 3 < 3
    }
}

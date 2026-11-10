//! Independent compile-time policy axes for the exp family.

/// Accepted input range and non-finite handling.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ExpfDomain {
    /// Finite inputs; the approximation's documented range applies
    /// (`[-126,127.99]` lowp, `[-126,128)` midp, clamped as before).
    /// NaN and infinities are invalid inputs.
    #[default]
    Bounded,
    /// Every f32 input produces a result: NaN propagates to a NaN result
    /// (payload unspecified), `+inf` maps to `+inf`, `-inf` maps to `+0`.
    /// Finite inputs are clamped exactly as `Bounded`.
    Full,
}

/// Whole-operation feature selection, independent of numerical policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ExpfExecution {
    /// Reuse a capable caller or select a native tier once per operation.
    #[default]
    Automatic,
    /// Retain the caller's Magetypes tier without autonomous upgrades.
    Caller,
}

/// Whether accepted results must agree bit for bit across backends.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ExpfPortability {
    /// One fused graph, with exact software FMA when necessary.
    #[default]
    BitExact,
    /// Native fused or separate graph; result bits may vary.
    BackendDependent,
}

/// Approximation family, not a global mathematical error bound.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ExpfPrecision {
    /// Existing MIDP coefficients and graph; ≤1.9 ULP below x = 127.5.
    #[default]
    Midp,
    /// Existing shorter LOWP coefficients; ≤0.56% relative error.
    Lowp,
}

/// Whether to enforce the selected numerical domain at runtime.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ExpfValidation {
    /// Reject the whole vector, returning the exact mask for failing lanes.
    #[default]
    Checked,
    /// Omit numerical input checks. The caller promises the same domain.
    /// Violations have unspecified numerical results, never memory unsoundness.
    /// Supported widths and valid option encodings are still enforced.
    AssumeValid,
}

/// Const builder ending in `.bits()`, also accepted by the runtime adapter.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExpfOptions {
    /// Input range and non-finite handling.
    pub domain: ExpfDomain,
    /// Whole-operation backend selection.
    pub execution: ExpfExecution,
    /// Cross-backend result agreement.
    pub portability: ExpfPortability,
    /// Approximation family.
    pub precision: ExpfPrecision,
    /// Numerical precondition enforcement.
    pub validation: ExpfValidation,
}

impl ExpfOptions {
    pub(crate) const DEFAULT: u32 = 0;
    pub(crate) const FULL: u32 = 1;
    pub(crate) const CALLER: u32 = 4;
    pub(crate) const LOWP: u32 = 8;
    pub(crate) const BACKEND: u32 = 16;
    pub(crate) const ASSUME: u32 = 32;
    pub(crate) const LEGACY: u32 = 64;

    /// BitExact MIDP, bounded domain, checked, automatic execution.
    pub const PORTABLE_MIDP: Self = Self::new();
    /// BitExact LOWP, bounded domain, checked, automatic execution.
    pub const PORTABLE_LOWP: Self = Self::new().with_precision(ExpfPrecision::Lowp);
    /// BitExact MIDP accepting every f32 input, checked.
    pub const PORTABLE_MIDP_FULL: Self = Self::new().with_domain(ExpfDomain::Full);
    /// Backend-dependent MIDP, bounded domain, checked.
    pub const BACKEND_MIDP: Self = Self::new().with_portability(ExpfPortability::BackendDependent);
    /// BitExact MIDP, bounded domain, unchecked.
    pub const UNCHECKED_MIDP: Self = Self::new().with_validation(ExpfValidation::AssumeValid);
    /// BitExact LOWP, bounded domain, unchecked.
    pub const UNCHECKED_LOWP: Self = Self::UNCHECKED_MIDP.with_precision(ExpfPrecision::Lowp);

    /// Build a configuration.
    pub const fn new() -> Self {
        Self {
            domain: ExpfDomain::Bounded,
            execution: ExpfExecution::Automatic,
            portability: ExpfPortability::BitExact,
            precision: ExpfPrecision::Midp,
            validation: ExpfValidation::Checked,
        }
    }
    /// Select input domain.
    pub const fn with_domain(self, v: ExpfDomain) -> Self {
        Self { domain: v, ..self }
    }
    /// Select backend execution.
    pub const fn with_execution(self, v: ExpfExecution) -> Self {
        Self {
            execution: v,
            ..self
        }
    }
    /// Select portability.
    pub const fn with_portability(self, v: ExpfPortability) -> Self {
        Self {
            portability: v,
            ..self
        }
    }
    /// Select precision.
    pub const fn with_precision(self, v: ExpfPrecision) -> Self {
        Self {
            precision: v,
            ..self
        }
    }
    /// Select validation.
    pub const fn with_validation(self, v: ExpfValidation) -> Self {
        Self {
            validation: v,
            ..self
        }
    }

    /// Encode as option bits for const-generic entry points.
    pub const fn bits(self) -> u32 {
        let mut bits = match self.domain {
            ExpfDomain::Bounded => 0,
            ExpfDomain::Full => Self::FULL,
        };
        if matches!(self.execution, ExpfExecution::Caller) {
            bits |= Self::CALLER;
        }
        if matches!(self.precision, ExpfPrecision::Lowp) {
            bits |= Self::LOWP;
        }
        if matches!(self.portability, ExpfPortability::BackendDependent) {
            bits |= Self::BACKEND;
        }
        if matches!(self.validation, ExpfValidation::AssumeValid) {
            bits |= Self::ASSUME;
        }
        bits
    }
    /// Decode option bits for the runtime adapter.
    pub(crate) const fn from_bits(bits: u32) -> Option<Self> {
        if !Self::valid(bits) {
            return None;
        }
        Some(Self {
            domain: if bits & 1 == 0 {
                ExpfDomain::Bounded
            } else {
                ExpfDomain::Full
            },
            execution: if bits & Self::CALLER == 0 {
                ExpfExecution::Automatic
            } else {
                ExpfExecution::Caller
            },
            portability: if bits & Self::BACKEND == 0 {
                ExpfPortability::BitExact
            } else {
                ExpfPortability::BackendDependent
            },
            precision: if bits & Self::LOWP == 0 {
                ExpfPrecision::Midp
            } else {
                ExpfPrecision::Lowp
            },
            validation: if bits & Self::ASSUME == 0 {
                ExpfValidation::Checked
            } else {
                ExpfValidation::AssumeValid
            },
        })
    }
    /// Reject encodings outside the supported space (const contexts).
    pub(crate) const fn valid(bits: u32) -> bool {
        bits & !0b11_1101 == 0 && bits & 3 != 2 && bits & 3 != 3
    }
}

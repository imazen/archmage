//! Independent compile-time policy axes for the log family.

/// Accepted input range and edge handling.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LogfDomain {
    /// Positive finite normal inputs only. `x <= 0`, subnormals, NaN and
    /// infinities are invalid inputs.
    #[default]
    Positive,
    /// IEEE specials: `0 -> -inf`, `x < 0 -> NaN`, NaN -> NaN,
    /// `+inf -> +inf`. Subnormal inputs keep their historical
    /// (inaccurate) low-accuracy result; they are not rejected.
    Specials,
}

/// Whole-operation feature selection, independent of numerical policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LogfExecution {
    /// Reuse a capable caller or select a native tier once per operation.
    #[default]
    Automatic,
    /// Retain the caller's Magetypes tier without autonomous upgrades.
    Caller,
}

/// Whether accepted results must agree bit for bit across backends.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LogfPortability {
    /// One fused graph, with exact software FMA when necessary.
    #[default]
    BitExact,
    /// Native fused or separate graph; result bits may vary.
    BackendDependent,
}

/// Approximation family, not a global mathematical error bound.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LogfPrecision {
    /// Existing MIDP coefficients and graph; at most 4.5 ULP.
    #[default]
    Midp,
    /// Existing shorter LOWP coefficients; ~6e-6 absolute error.
    Lowp,
    /// Original MIDP coefficients, preserved for bit-level compatibility.
    LegacyMidp,
    /// Original LOWP coefficients, preserved for bit-level compatibility.
    LegacyLowp,
}

/// Whether to enforce the selected numerical domain at runtime.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LogfValidation {
    /// Reject the whole vector, returning the exact mask for failing lanes.
    #[default]
    Checked,
    /// Omit numerical input checks. The caller promises the same domain.
    /// Violations have unspecified numerical results, never memory unsoundness.
    AssumeValid,
}

/// Const builder ending in `.bits()`, also accepted by the runtime adapter.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LogfOptions {
    /// Input range and edge handling.
    pub domain: LogfDomain,
    /// Whole-operation backend selection.
    pub execution: LogfExecution,
    /// Cross-backend result agreement.
    pub portability: LogfPortability,
    /// Approximation family.
    pub precision: LogfPrecision,
    /// Numerical precondition enforcement.
    pub validation: LogfValidation,
}

impl LogfOptions {
    pub(crate) const DEFAULT: u32 = 0;
    pub(crate) const SPECIALS: u32 = 1;
    pub(crate) const CALLER: u32 = 4;
    pub(crate) const LOWP: u32 = 8;
    pub(crate) const BACKEND: u32 = 16;
    pub(crate) const ASSUME: u32 = 32;
    pub(crate) const LEGACY: u32 = 64;

    /// BitExact MIDP, positive domain, checked, automatic execution.
    pub const PORTABLE_MIDP: Self = Self::new();
    /// BitExact LOWP, positive domain, checked.
    pub const PORTABLE_LOWP: Self = Self::new().with_precision(LogfPrecision::Lowp);
    /// BitExact MIDP with IEEE specials, checked.
    pub const PORTABLE_MIDP_SPECIALS: Self = Self::new().with_domain(LogfDomain::Specials);
    /// Backend-dependent MIDP, positive domain, checked.
    pub const BACKEND_MIDP: Self = Self::new().with_portability(LogfPortability::BackendDependent);
    /// BitExact MIDP, positive domain, unchecked.
    pub const UNCHECKED_MIDP: Self = Self::new().with_validation(LogfValidation::AssumeValid);
    /// BitExact LOWP, positive domain, unchecked.
    pub const UNCHECKED_LOWP: Self = Self::UNCHECKED_MIDP.with_precision(LogfPrecision::Lowp);

    /// Build a configuration.
    pub const fn new() -> Self {
        Self {
            domain: LogfDomain::Positive,
            execution: LogfExecution::Automatic,
            portability: LogfPortability::BitExact,
            precision: LogfPrecision::Midp,
            validation: LogfValidation::Checked,
        }
    }
    /// Select input domain.
    pub const fn with_domain(self, v: LogfDomain) -> Self {
        Self { domain: v, ..self }
    }
    /// Select backend execution.
    pub const fn with_execution(self, v: LogfExecution) -> Self {
        Self {
            execution: v,
            ..self
        }
    }
    /// Select portability.
    pub const fn with_portability(self, v: LogfPortability) -> Self {
        Self {
            portability: v,
            ..self
        }
    }
    /// Select precision.
    pub const fn with_precision(self, v: LogfPrecision) -> Self {
        Self {
            precision: v,
            ..self
        }
    }
    /// Select validation.
    pub const fn with_validation(self, v: LogfValidation) -> Self {
        Self {
            validation: v,
            ..self
        }
    }

    /// Encode as option bits for const-generic entry points.
    pub const fn bits(self) -> u32 {
        let mut bits = match self.domain {
            LogfDomain::Positive => 0,
            LogfDomain::Specials => Self::SPECIALS,
        };
        if matches!(self.execution, LogfExecution::Caller) {
            bits |= Self::CALLER;
        }
        if matches!(
            self.precision,
            LogfPrecision::Lowp | LogfPrecision::LegacyLowp
        ) {
            bits |= Self::LOWP;
        }
        if matches!(
            self.precision,
            LogfPrecision::LegacyMidp | LogfPrecision::LegacyLowp
        ) {
            bits |= Self::LEGACY;
        }
        if matches!(self.portability, LogfPortability::BackendDependent) {
            bits |= Self::BACKEND;
        }
        if matches!(self.validation, LogfValidation::AssumeValid) {
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
                LogfDomain::Positive
            } else {
                LogfDomain::Specials
            },
            execution: if bits & Self::CALLER == 0 {
                LogfExecution::Automatic
            } else {
                LogfExecution::Caller
            },
            portability: if bits & Self::BACKEND == 0 {
                LogfPortability::BitExact
            } else {
                LogfPortability::BackendDependent
            },
            precision: match ((bits & Self::LEGACY) != 0, (bits & Self::LOWP) != 0) {
                (false, false) => LogfPrecision::Midp,
                (false, true) => LogfPrecision::Lowp,
                (true, false) => LogfPrecision::LegacyMidp,
                (true, true) => LogfPrecision::LegacyLowp,
            },
            validation: if bits & Self::ASSUME == 0 {
                LogfValidation::Checked
            } else {
                LogfValidation::AssumeValid
            },
        })
    }
    /// Reject encodings outside the supported space (const contexts).
    pub(crate) const fn valid(bits: u32) -> bool {
        bits & !0b111_1101 == 0 && bits & 3 != 2 && bits & 3 != 3
    }
}

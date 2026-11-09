//! Independent compile-time policy axes. Existing encodings 0..=6 are preserved.

/// Input and output range. Restricted domains reject negative zero and subnormals.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PowfDomain {
    /// Positive finite normal bases; computed exp2 argument in [-126,127.5].
    #[default]
    PositiveNormal,
    /// Also admit +0 with nonnegative exponent: 0^0=1 and 0^positive=+0.
    NonNegativeNormal,
    /// +0 or positive normal [2^-16,2^16], exponent normal or zero in [0,4].
    /// The approximation's output is positive normal or +0 without a range check.
    Gamma,
    /// Full f32 input/output domain, available only for backend-dependent HIGHP.
    /// Uses std power semantics, including its NaNs, signed zeros and infinities.
    Full,
}

/// Whole-operation feature selection, independent of numerical policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PowfExecution {
    /// Reuse a capable caller or select a native tier once per power operation.
    #[default]
    Automatic,
    /// Retain the caller's tier without runtime upgrades.
    Caller,
}

/// Whether accepted results must agree bit for bit across backends.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PowfPortability {
    /// One fused graph, with exact software FMA when necessary. Not std power.
    #[default]
    BitExact,
    /// Native fused or separate graph; result bits and range rejection may vary.
    BackendDependent,
}

/// Approximation family, not a global mathematical error bound.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PowfPrecision {
    /// Existing MIDP coefficients and graph; no whole-domain ULP guarantee.
    #[default]
    Midp,
    /// Existing shorter LOWP coefficients; prefer a bounded Gamma preset.
    Lowp,
    /// Std scalar power, requiring `std` and BackendDependent portability.
    /// Inherits std's unspecified precision; this is not a ONE_ULP promise.
    Highp,
}

/// Whether to enforce the selected numerical domain at runtime.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PowfValidation {
    /// Reject the whole vector, returning the exact mask for the first failing stage.
    #[default]
    Checked,
    /// Omit numerical input/output checks. The caller promises the same domain.
    /// Violations have unspecified numerical results, never memory unsoundness.
    /// Supported widths and valid option encodings are still enforced.
    AssumeValid,
}

/// Const builder ending in `.bits()`, also accepted by the runtime adapter.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PowfOptions {
    /// Input and output range.
    pub domain: PowfDomain,
    /// Whole-operation backend selection.
    pub execution: PowfExecution,
    /// Cross-backend result agreement.
    pub portability: PowfPortability,
    /// Approximation family.
    pub precision: PowfPrecision,
    /// Numerical precondition enforcement.
    pub validation: PowfValidation,
}

impl PowfOptions {
    pub(crate) const DEFAULT: u32 = 0;
    pub(crate) const GAMMA: u32 = 2;
    pub(crate) const CALLER: u32 = 4;
    pub(crate) const LOWP: u32 = 8;
    pub(crate) const BACKEND: u32 = 16;
    pub(crate) const ASSUME: u32 = 32;
    pub(crate) const HIGHP: u32 = 64;

    /// BitExact MIDP, Gamma range, checked, automatic execution.
    pub const PORTABLE_GAMMA_MIDP: Self = Self::new().with_domain(PowfDomain::Gamma);
    /// BitExact LOWP, Gamma range, checked, automatic execution.
    pub const PORTABLE_GAMMA_LOWP: Self =
        Self::PORTABLE_GAMMA_MIDP.with_precision(PowfPrecision::Lowp);
    /// Backend-dependent MIDP, Gamma range, checked, automatic execution.
    pub const BACKEND_GAMMA_MIDP: Self =
        Self::PORTABLE_GAMMA_MIDP.with_portability(PowfPortability::BackendDependent);
    /// Backend-dependent LOWP, Gamma range, checked, automatic execution.
    pub const BACKEND_GAMMA_LOWP: Self =
        Self::BACKEND_GAMMA_MIDP.with_precision(PowfPrecision::Lowp);
    /// BitExact MIDP, positive-normal range, checked, automatic execution.
    pub const PORTABLE_NORMAL_MIDP: Self = Self::new();
    /// Std power, full f32 domain, backend-dependent, automatic execution.
    /// Requires the `std` feature. Full-domain validation has nothing to reject.
    pub const STD_HIGHP: Self = Self::new()
        .with_precision(PowfPrecision::Highp)
        .with_portability(PowfPortability::BackendDependent)
        .with_domain(PowfDomain::Full);

    /// Default positive-normal, BitExact MIDP, checked, automatic configuration.
    pub const fn new() -> Self {
        Self {
            domain: PowfDomain::PositiveNormal,
            execution: PowfExecution::Automatic,
            portability: PowfPortability::BitExact,
            precision: PowfPrecision::Midp,
            validation: PowfValidation::Checked,
        }
    }
    /// Select the input/output range.
    pub const fn with_domain(mut self, value: PowfDomain) -> Self {
        self.domain = value;
        self
    }
    /// Select whole-operation backend dispatch.
    pub const fn with_execution(mut self, value: PowfExecution) -> Self {
        self.execution = value;
        self
    }
    /// Select cross-backend agreement.
    pub const fn with_portability(mut self, value: PowfPortability) -> Self {
        self.portability = value;
        self
    }
    /// Select the approximation family.
    pub const fn with_precision(mut self, value: PowfPrecision) -> Self {
        self.precision = value;
        self
    }
    /// Select numerical precondition enforcement.
    pub const fn with_validation(mut self, value: PowfValidation) -> Self {
        self.validation = value;
        self
    }
    /// Encode all five choices as a monomorphized integer argument.
    /// Unsupported combinations cause a compile error in the const API and
    /// `UnsupportedOptions` in the runtime API.
    pub const fn bits(self) -> u32 {
        let domain = match self.domain {
            PowfDomain::PositiveNormal => 0,
            PowfDomain::NonNegativeNormal => 1,
            PowfDomain::Gamma => 2,
            PowfDomain::Full => 3,
        };
        domain
            | match self.execution {
                PowfExecution::Automatic => 0,
                PowfExecution::Caller => Self::CALLER,
            }
            | match self.portability {
                PowfPortability::BitExact => 0,
                PowfPortability::BackendDependent => Self::BACKEND,
            }
            | match self.precision {
                PowfPrecision::Midp => 0,
                PowfPrecision::Lowp => Self::LOWP,
                PowfPrecision::Highp => Self::HIGHP,
            }
            | match self.validation {
                PowfValidation::Checked => 0,
                PowfValidation::AssumeValid => Self::ASSUME,
            }
    }
    pub(crate) const fn valid(bits: u32) -> bool {
        if bits & !127 != 0 || bits & (Self::LOWP | Self::HIGHP) == Self::LOWP | Self::HIGHP {
            return false;
        }
        if bits & Self::HIGHP != 0 {
            cfg!(feature = "std") && bits & Self::BACKEND != 0
        } else {
            bits & 3 != 3
        }
    }
}

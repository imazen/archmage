//! Independent compile-time policy axes. Existing encodings 0..=6 are preserved.

/// Input and output range. Restricted domains reject negative zero and subnormals.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum PowfDomain {
    /// Positive finite normal bases; computed exp2 argument in [-126,127.5].
    #[default]
    PositiveNormal,
    /// Also admit +0 with nonnegative exponent: 0^0=1 and 0^positive=+0.
    NonNegativeNormal,
    /// +0 or positive normal `[2^-16, 2^16]`, exponent normal or zero in `[0, 4]`.
    /// The approximation's output is positive normal or +0 without a range check.
    Gamma,
    /// Positive normal `[2^-16, 2^16]`, exponent normal or zero in `[0, 4]`.
    /// Same nonzero graph as Gamma; excludes zero and removes its masks.
    PositiveGamma,
    /// Full f32 input/output domain for HIGHP.
    /// BitExact uses the canonical graph; BackendDependent uses std semantics.
    Full,
}

/// Whole-operation feature selection, independent of numerical policy.
pub use crate::Execution as PowfExecution;
/// Whether accepted results must agree bit for bit across backends.
pub use crate::Portability as PowfPortability;
/// Whether to enforce the selected numerical domain at runtime.
pub use crate::Validation as PowfValidation;

/// Approximation family, not a global mathematical error bound.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum PowfPrecision {
    /// Existing MIDP coefficients and graph; no whole-domain ULP guarantee.
    #[default]
    Midp,
    /// Existing shorter LOWP coefficients; prefer a bounded Gamma preset.
    Lowp,
    /// Table-based binary64 graph for BitExact; std power for BackendDependent.
    /// BackendDependent requires `std`. Neither is a global ONE_ULP promise.
    Highp,
}

/// Const builder ending in `.bits()`, also accepted by the runtime adapter.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
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
    pub(crate) const GAMMA: u32 = 2;
    pub(crate) const CALLER: u32 = 4;
    pub(crate) const LOWP: u32 = 8;
    pub(crate) const BACKEND: u32 = 16;
    pub(crate) const ASSUME: u32 = 32;
    pub(crate) const HIGHP: u32 = 64;
    pub(crate) const POSITIVE: u32 = 128;

    /// BitExact MIDP, Gamma range, checked, automatic execution.
    pub const PORTABLE_GAMMA_MIDP: Self = Self::new().with_domain(PowfDomain::Gamma);
    /// BitExact LOWP, Gamma range, checked, automatic execution.
    pub const PORTABLE_GAMMA_LOWP: Self =
        Self::PORTABLE_GAMMA_MIDP.with_precision(PowfPrecision::Lowp);
    /// Canonical HIGHP, Gamma range, checked, automatic execution.
    pub const PORTABLE_GAMMA_HIGHP: Self =
        Self::PORTABLE_GAMMA_MIDP.with_precision(PowfPrecision::Highp);
    /// Canonical HIGHP, full f32 domain, automatic execution.
    /// NaN classification is canonical; NaN payload and sign are unspecified.
    pub const PORTABLE_HIGHP: Self = Self::new()
        .with_precision(PowfPrecision::Highp)
        .with_domain(PowfDomain::Full);
    /// Backend-dependent MIDP, Gamma range, checked, automatic execution.
    pub const BACKEND_GAMMA_MIDP: Self =
        Self::PORTABLE_GAMMA_MIDP.with_portability(PowfPortability::BackendDependent);
    /// Backend-dependent LOWP, Gamma range, checked, automatic execution.
    pub const BACKEND_GAMMA_LOWP: Self =
        Self::BACKEND_GAMMA_MIDP.with_precision(PowfPrecision::Lowp);
    /// BitExact MIDP, positive Gamma range, checked, automatic execution.
    pub const PORTABLE_POSITIVE_GAMMA_MIDP: Self =
        Self::PORTABLE_GAMMA_MIDP.with_domain(PowfDomain::PositiveGamma);
    /// BitExact LOWP, positive Gamma range, checked, automatic execution.
    pub const PORTABLE_POSITIVE_GAMMA_LOWP: Self =
        Self::PORTABLE_POSITIVE_GAMMA_MIDP.with_precision(PowfPrecision::Lowp);
    /// Backend-dependent MIDP, positive Gamma range, checked, automatic execution.
    pub const BACKEND_POSITIVE_GAMMA_MIDP: Self =
        Self::PORTABLE_POSITIVE_GAMMA_MIDP.with_portability(PowfPortability::BackendDependent);
    /// Backend-dependent LOWP, positive Gamma range, checked, automatic execution.
    pub const BACKEND_POSITIVE_GAMMA_LOWP: Self =
        Self::BACKEND_POSITIVE_GAMMA_MIDP.with_precision(PowfPrecision::Lowp);
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
            PowfDomain::PositiveGamma => Self::POSITIVE | Self::GAMMA,
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
    pub(crate) const fn from_bits(bits: u32) -> Option<Self> {
        if !Self::valid(bits) {
            return None;
        }
        let domain = match bits & 3 {
            0 => PowfDomain::PositiveNormal,
            1 => PowfDomain::NonNegativeNormal,
            2 => {
                if bits & Self::POSITIVE != 0 {
                    PowfDomain::PositiveGamma
                } else {
                    PowfDomain::Gamma
                }
            }
            _ => PowfDomain::Full,
        };
        Some(Self {
            domain,
            execution: if bits & Self::CALLER == 0 {
                PowfExecution::Automatic
            } else {
                PowfExecution::Caller
            },
            portability: if bits & Self::BACKEND == 0 {
                PowfPortability::BitExact
            } else {
                PowfPortability::BackendDependent
            },
            precision: if bits & Self::LOWP != 0 {
                PowfPrecision::Lowp
            } else if bits & Self::HIGHP != 0 {
                PowfPrecision::Highp
            } else {
                PowfPrecision::Midp
            },
            validation: if bits & Self::ASSUME == 0 {
                PowfValidation::Checked
            } else {
                PowfValidation::AssumeValid
            },
        })
    }

    pub(crate) const fn valid(bits: u32) -> bool {
        if bits & !255 != 0
            || bits & (Self::LOWP | Self::HIGHP) == Self::LOWP | Self::HIGHP
            || (bits & Self::POSITIVE != 0 && bits & 3 != Self::GAMMA)
        {
            return false;
        }
        if bits & Self::HIGHP != 0 {
            bits & Self::BACKEND == 0 || cfg!(feature = "std")
        } else {
            bits & 3 != 3
        }
    }
}

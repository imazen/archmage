//! Independent compile-time policy axes for the cube-root (`cbrtf`) family.

/// Accepted input range and non-finite / subnormal handling.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CbrtfDomain {
    /// Zero (`±0.0`) and normal finite magnitudes strictly below `f32::MAX / 3`
    /// (`1.175_494_4e-38 <= |x| < 1.134_274_5e38`).
    ///
    /// Runs the direct unscaled Halley iteration with zero denormal/overflow
    /// pre-scaling overhead. Subnormals, magnitudes `>= f32::MAX / 3`, NaN, and
    /// infinities are rejected under [`CbrtfValidation::Checked`].
    #[default]
    Bounded,
    /// Every `f32` input produces a result: subnormals are rescaled by `2^24` /
    /// `2^-8`, large magnitudes (`|x| >= 1e36`) run on `x / 8` and double the
    /// result so `y^3 + 2x` cannot overflow, `±0.0` maps to `±0.0`, `±inf` maps
    /// to `±inf`, and NaN propagates to quiet NaN.
    Full,
}

/// Whole-operation feature selection, independent of numerical policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CbrtfExecution {
    /// Reuse a capable caller or select a native tier once per operation.
    #[default]
    Automatic,
    /// Retain the caller's Magetypes tier without autonomous upgrades.
    Caller,
}

/// Whether accepted results must agree bit for bit across backends.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CbrtfPortability {
    /// Identical IEEE-754 rounding graph across all backends.
    #[default]
    BitExact,
    /// Permits backend-specific fused multiply-add in the Halley step.
    BackendDependent,
}

/// Approximation family (1 vs 2 Halley iterations from the vectorized Kahan seed).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CbrtfPrecision {
    /// Vectorized Kahan seed + 2 Halley iterations; <= 3.2 ULP across accepted domain.
    #[default]
    Midp,
    /// Vectorized Kahan seed + 1 Halley iteration; <= 259 ULP (~15 bits, 3e-5 rel error).
    Lowp,
}

/// Whether to enforce the selected numerical domain at runtime.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CbrtfValidation {
    /// Reject the whole vector, returning the exact mask for failing lanes.
    #[default]
    Checked,
    /// Omit numerical input checks. The caller promises the same domain.
    /// Violations have unspecified numerical results, never memory unsoundness.
    AssumeValid,
}

/// Const builder ending in `.bits()`, also accepted by the runtime adapter.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CbrtfOptions {
    /// Input range and non-finite / subnormal handling.
    pub domain: CbrtfDomain,
    /// Whole-operation backend selection.
    pub execution: CbrtfExecution,
    /// Cross-backend result agreement.
    pub portability: CbrtfPortability,
    /// Approximation family.
    pub precision: CbrtfPrecision,
    /// Numerical precondition enforcement.
    pub validation: CbrtfValidation,
}

impl CbrtfOptions {
    pub(crate) const DEFAULT: u32 = 0;
    pub(crate) const FULL: u32 = 1;
    pub(crate) const CALLER: u32 = 4;
    pub(crate) const LOWP: u32 = 8;
    pub(crate) const BACKEND: u32 = 16;
    pub(crate) const ASSUME: u32 = 32;

    /// BitExact MIDP, bounded normal domain, checked, automatic execution.
    pub const PORTABLE_MIDP: Self = Self::new();
    /// BitExact LOWP, bounded normal domain, checked, automatic execution.
    pub const PORTABLE_LOWP: Self = Self::new().with_precision(CbrtfPrecision::Lowp);
    /// BitExact MIDP accepting every f32 input (subnormals, large normals, specials).
    pub const PORTABLE_MIDP_FULL: Self = Self::new().with_domain(CbrtfDomain::Full);
    /// BitExact LOWP accepting every f32 input.
    pub const PORTABLE_LOWP_FULL: Self =
        Self::PORTABLE_MIDP_FULL.with_precision(CbrtfPrecision::Lowp);
    /// Backend-dependent MIDP, bounded normal domain, checked.
    pub const BACKEND_MIDP: Self = Self::new().with_portability(CbrtfPortability::BackendDependent);
    /// BitExact MIDP, bounded normal domain, unchecked.
    pub const UNCHECKED_MIDP: Self = Self::new().with_validation(CbrtfValidation::AssumeValid);
    /// BitExact LOWP, bounded normal domain, unchecked.
    pub const UNCHECKED_LOWP: Self = Self::UNCHECKED_MIDP.with_precision(CbrtfPrecision::Lowp);

    /// Build a default configuration (`PORTABLE_MIDP`).
    pub const fn new() -> Self {
        Self {
            domain: CbrtfDomain::Bounded,
            execution: CbrtfExecution::Automatic,
            portability: CbrtfPortability::BitExact,
            precision: CbrtfPrecision::Midp,
            validation: CbrtfValidation::Checked,
        }
    }

    /// Select input domain.
    pub const fn with_domain(self, v: CbrtfDomain) -> Self {
        Self { domain: v, ..self }
    }

    /// Select backend execution.
    pub const fn with_execution(self, v: CbrtfExecution) -> Self {
        Self {
            execution: v,
            ..self
        }
    }

    /// Select portability.
    pub const fn with_portability(self, v: CbrtfPortability) -> Self {
        Self {
            portability: v,
            ..self
        }
    }

    /// Select precision.
    pub const fn with_precision(self, v: CbrtfPrecision) -> Self {
        Self {
            precision: v,
            ..self
        }
    }

    /// Select validation.
    pub const fn with_validation(self, v: CbrtfValidation) -> Self {
        Self {
            validation: v,
            ..self
        }
    }

    /// Encode as option bits for const-generic entry points.
    pub const fn bits(self) -> u32 {
        let mut bits = match self.domain {
            CbrtfDomain::Bounded => 0,
            CbrtfDomain::Full => Self::FULL,
        };
        if matches!(self.execution, CbrtfExecution::Caller) {
            bits |= Self::CALLER;
        }
        if matches!(self.precision, CbrtfPrecision::Lowp) {
            bits |= Self::LOWP;
        }
        if matches!(self.portability, CbrtfPortability::BackendDependent) {
            bits |= Self::BACKEND;
        }
        if matches!(self.validation, CbrtfValidation::AssumeValid) {
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
            domain: if bits & Self::FULL == 0 {
                CbrtfDomain::Bounded
            } else {
                CbrtfDomain::Full
            },
            execution: if bits & Self::CALLER == 0 {
                CbrtfExecution::Automatic
            } else {
                CbrtfExecution::Caller
            },
            portability: if bits & Self::BACKEND == 0 {
                CbrtfPortability::BitExact
            } else {
                CbrtfPortability::BackendDependent
            },
            precision: if bits & Self::LOWP == 0 {
                CbrtfPrecision::Midp
            } else {
                CbrtfPrecision::Lowp
            },
            validation: if bits & Self::ASSUME == 0 {
                CbrtfValidation::Checked
            } else {
                CbrtfValidation::AssumeValid
            },
        })
    }

    /// Reject encodings outside the supported space (const contexts).
    pub(crate) const fn valid(bits: u32) -> bool {
        bits & !0b11_1101 == 0
    }
}

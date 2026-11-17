//! Fluent `const` configuration builder for the logarithm family
//! (`log2`, `ln`, `log10`, `log1p`).

use crate::{Execution, Portability, Validation};

const MAGIC: u128 = 0x4C;
const HAS_RANGE: u128 = 1 << 8;
const FULL: u128 = 1 << 9;
const LOWP: u128 = 1 << 10;
const BACKEND: u128 = 1 << 11;
const ASSUME: u128 = 1 << 12;
const CALLER: u128 = 1 << 13;
const RESERVED_MASK: u128 = ((1_u128 << 50) - 1) << 14;

const fn abs(x: f32) -> f32 {
    if x < 0.0 { -x } else { x }
}

const fn canon_zero(x: f32) -> f32 {
    if x == 0.0 { 0.0 } else { x }
}

/// Approximation family, not a global mathematical error bound.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum LogfPrecision {
    /// Refitted `Midp` polynomial (`<= 4.5 ULP` on positive normals).
    #[default]
    Midp,
    /// Rational `P2/Q2` `Lowp` approximation (`~6e-6` absolute error).
    Lowp,
}

/// Fluent `const` configuration builder for [`super::Logf`] and [`crate::SimdMath`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LogfOptions {
    input: Option<(u32, u32)>,
    full: bool,
    precision: LogfPrecision,
    portability: Portability,
    validation: Validation,
    execution: Execution,
}

impl Default for LogfOptions {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl LogfOptions {
    /// Positive normal domain (`1 + x` positive normal for `log1p`), `Midp`,
    /// `BitExact`, `Checked`, `Execution::Upgrade`.
    pub const fn new() -> Self {
        Self {
            input: None,
            full: false,
            precision: LogfPrecision::Midp,
            portability: Portability::BitExact,
            validation: Validation::Checked,
            execution: Execution::Upgrade,
        }
    }

    /// Accept inputs in `[lo, hi]`.
    ///
    /// For `log2`, `ln`, and `log10`, `lo` must be at least `f32::MIN_POSITIVE`.
    /// For `log1p`, `1 + lo` must be at least `f32::MIN_POSITIVE`. Both bounds
    /// must be finite with `-1.0 < lo <= hi`.
    pub const fn input_range(mut self, lo: f32, hi: f32) -> Self {
        assert!(
            lo > -1.0 && lo <= hi && abs(lo) <= f32::MAX && abs(hi) <= f32::MAX,
            "input range must be finite with -1 < lo <= hi"
        );
        self.input = Some((canon_zero(lo).to_bits(), canon_zero(hi).to_bits()));
        self
    }

    /// Accept every `f32` input, propagating IEEE specials (`0 -> -inf`,
    /// `< 0 -> NaN`, `NaN -> NaN`, `+inf -> +inf`; shifted by `-1` for `log1p`).
    pub const fn full_domain(mut self) -> Self {
        self.full = true;
        self
    }

    /// Select the approximation family (`Midp` or `Lowp`).
    pub const fn precision(mut self, value: LogfPrecision) -> Self {
        self.precision = value;
        self
    }

    /// Shorthand for `.precision(LogfPrecision::Lowp)`.
    pub const fn lowp(self) -> Self {
        self.precision(LogfPrecision::Lowp)
    }

    /// Shorthand for `.precision(LogfPrecision::Midp)`.
    pub const fn midp(self) -> Self {
        self.precision(LogfPrecision::Midp)
    }

    /// Select cross-backend agreement.
    pub const fn portability(mut self, value: Portability) -> Self {
        self.portability = value;
        self
    }

    /// Shorthand for `.portability(Portability::BitExact)`.
    pub const fn bit_exact(self) -> Self {
        self.portability(Portability::BitExact)
    }

    /// Shorthand for `.portability(Portability::BackendDependent)`.
    pub const fn backend_dependent(self) -> Self {
        self.portability(Portability::BackendDependent)
    }

    /// Select numerical precondition enforcement.
    pub const fn validation(mut self, value: Validation) -> Self {
        self.validation = value;
        self
    }

    /// Shorthand for `.validation(Validation::Checked)`.
    pub const fn checked(self) -> Self {
        self.validation(Validation::Checked)
    }

    /// Shorthand for `.validation(Validation::AssumeValid)`.
    pub const fn unchecked(self) -> Self {
        self.validation(Validation::AssumeValid)
    }

    /// Select whole-operation backend dispatch.
    pub const fn execution(mut self, value: Execution) -> Self {
        self.execution = value;
        self
    }

    /// Shorthand for `.execution(Execution::Upgrade)`.
    pub const fn upgrade(self) -> Self {
        self.execution(Execution::Upgrade)
    }

    /// Shorthand for `.execution(Execution::Caller)`.
    pub const fn caller(self) -> Self {
        self.execution(Execution::Caller)
    }

    /// Whether domain-level invariants common to all log variants hold.
    pub const fn is_valid(&self) -> bool {
        !(self.full && self.input.is_some())
    }

    pub(crate) const fn is_valid_op(&self, op: u8) -> bool {
        if !self.is_valid() {
            return false;
        }
        if let Some((lo_b, _)) = self.input {
            let lo = f32::from_bits(lo_b);
            if op == 3 {
                if lo.is_nan() || 1.0 + lo < f32::MIN_POSITIVE {
                    return false;
                }
            } else if lo.is_nan() || lo < f32::MIN_POSITIVE {
                return false;
            }
        }
        true
    }

    /// Assert domain-level invariants common to all log variants.
    pub const fn validate(&self) {
        assert!(
            !(self.full && self.input.is_some()),
            "full_domain cannot be combined with input_range"
        );
    }

    pub(crate) const fn validate_op(&self, op: u8) {
        self.validate();
        if let Some((lo_b, _)) = self.input {
            let lo = f32::from_bits(lo_b);
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

    /// Whether this configuration can never return an input error
    /// (`.unchecked()` or `.full_domain()`).
    pub const fn is_infallible(&self) -> bool {
        self.assume_valid() || self.full
    }

    /// Pack this configuration into a `u128` const-generic argument.
    pub const fn bits(self) -> u128 {
        let mut out = MAGIC;
        if let Some((lo, hi)) = self.input {
            out |= HAS_RANGE | ((lo as u128) << 64) | ((hi as u128) << 96);
        }
        if self.full {
            out |= FULL;
        }
        if matches!(self.precision, LogfPrecision::Lowp) {
            out |= LOWP;
        }
        if matches!(self.portability, Portability::BackendDependent) {
            out |= BACKEND;
        }
        if matches!(self.validation, Validation::AssumeValid) {
            out |= ASSUME;
        }
        if matches!(self.execution, Execution::Caller) {
            out |= CALLER;
        }
        out
    }

    /// Decode a `u128` option word back into [`LogfOptions`].
    pub const fn from_bits(bits: u128) -> Option<Self> {
        if (bits & 0xff) != MAGIC || (bits & RESERVED_MASK) != 0 {
            return None;
        }
        let lo_b = ((bits >> 64) & 0xffff_ffff) as u32;
        let hi_b = ((bits >> 96) & 0xffff_ffff) as u32;
        let input = if (bits & HAS_RANGE) != 0 {
            if lo_b == 0x8000_0000 || hi_b == 0x8000_0000 {
                return None;
            }
            let lo = f32::from_bits(lo_b);
            let hi = f32::from_bits(hi_b);
            if !(lo > -1.0 && lo <= hi && abs(lo) <= f32::MAX && abs(hi) <= f32::MAX) {
                return None;
            }
            Some((lo_b, hi_b))
        } else {
            if lo_b != 0 || hi_b != 0 {
                return None;
            }
            None
        };
        let opts = Self {
            input,
            full: (bits & FULL) != 0,
            precision: if (bits & LOWP) != 0 {
                LogfPrecision::Lowp
            } else {
                LogfPrecision::Midp
            },
            portability: if (bits & BACKEND) != 0 {
                Portability::BackendDependent
            } else {
                Portability::BitExact
            },
            validation: if (bits & ASSUME) != 0 {
                Validation::AssumeValid
            } else {
                Validation::Checked
            },
            execution: if (bits & CALLER) != 0 {
                Execution::Caller
            } else {
                Execution::Upgrade
            },
        };
        if opts.is_valid() { Some(opts) } else { None }
    }

    pub(crate) const fn decode_const(bits: u128) -> Self {
        let Some(opts) = Self::from_bits(bits) else {
            panic!("invalid logf const options");
        };
        opts
    }

    /// Create a dynamic runtime evaluator for this [`LogfOptions`] value.
    #[inline(always)]
    pub const fn runtime(self) -> super::LogfRuntime {
        super::LogfRuntime::new(self)
    }

    pub(crate) const fn input_bounds(&self) -> Option<(f32, f32)> {
        match self.input {
            Some((lo, hi)) => Some((f32::from_bits(lo), f32::from_bits(hi))),
            None => None,
        }
    }
    pub(crate) const fn is_specials(&self) -> bool {
        self.full
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

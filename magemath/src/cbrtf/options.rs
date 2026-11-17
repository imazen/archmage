//! Fluent `const` configuration builder for the cube-root (`cbrtf`) family.

use crate::{Execution, Portability, Validation};

const MAX_BOUNDED: f32 = f32::MAX / 3.0;

const MAGIC: u128 = 0x43;
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
pub enum CbrtfPrecision {
    /// Two Halley iterations after the SIMD Kahan seed (`<= 3.2 ULP`).
    #[default]
    Midp,
    /// One Halley iteration after the SIMD Kahan seed (`<= 259 ULP`, `~3e-5` rel err).
    Lowp,
}

/// Fluent `const` configuration builder for [`super::Cbrtf`] and [`crate::SimdMath`].
///
/// Declaring [`CbrtfOptions::input_range`] with `lo >= f32::MIN_POSITIVE` (or
/// `hi <= -f32::MIN_POSITIVE`) proves subnormals cannot occur and elides the
/// per-lane subnormal check at compile time.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CbrtfOptions {
    input: Option<(u32, u32)>,
    full: bool,
    precision: CbrtfPrecision,
    portability: Portability,
    validation: Validation,
    execution: Execution,
}

impl Default for CbrtfOptions {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl CbrtfOptions {
    /// Bounded domain (`±0` and normals strictly below `f32::MAX / 3`), `Midp`,
    /// `BitExact`, `Checked`, `Execution::Upgrade`.
    pub const fn new() -> Self {
        Self {
            input: None,
            full: false,
            precision: CbrtfPrecision::Midp,
            portability: Portability::BitExact,
            validation: Validation::Checked,
            execution: Execution::Upgrade,
        }
    }

    /// Accept inputs in `[lo, hi]`. Both must be finite `f32`s strictly below
    /// `f32::MAX / 3` in magnitude with `lo <= hi`.
    pub const fn input_range(mut self, lo: f32, hi: f32) -> Self {
        assert!(
            lo <= hi && abs(lo) < MAX_BOUNDED && abs(hi) < MAX_BOUNDED,
            "cbrtf input range must satisfy -f32::MAX/3 < lo <= hi < f32::MAX/3"
        );
        self.input = Some((canon_zero(lo).to_bits(), canon_zero(hi).to_bits()));
        self
    }

    /// Accept every `f32` input (rescaling subnormals and large magnitudes,
    /// propagating `±0`, `±inf`, and `NaN`).
    pub const fn full_domain(mut self) -> Self {
        self.full = true;
        self
    }

    /// Select the approximation family (`Midp` or `Lowp`).
    pub const fn precision(mut self, value: CbrtfPrecision) -> Self {
        self.precision = value;
        self
    }

    /// Shorthand for `.precision(CbrtfPrecision::Lowp)`.
    pub const fn lowp(self) -> Self {
        self.precision(CbrtfPrecision::Lowp)
    }

    /// Shorthand for `.precision(CbrtfPrecision::Midp)`.
    pub const fn midp(self) -> Self {
        self.precision(CbrtfPrecision::Midp)
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

    /// Whether this configuration is self-consistent.
    pub const fn is_valid(&self) -> bool {
        !(self.full && self.input.is_some())
    }

    /// Assert that this configuration is self-consistent.
    pub const fn validate(&self) {
        assert!(
            !(self.full && self.input.is_some()),
            "full_domain cannot be combined with input_range"
        );
    }

    /// Whether the declared range excludes the subnormal band around zero,
    /// eliding the per-lane subnormal check.
    pub const fn elides_denorm_check(&self) -> bool {
        !self.denorm_possible()
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
        if matches!(self.precision, CbrtfPrecision::Lowp) {
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

    /// Decode a `u128` option word back into [`CbrtfOptions`].
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
            if !(lo <= hi && abs(lo) < MAX_BOUNDED && abs(hi) < MAX_BOUNDED) {
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
                CbrtfPrecision::Lowp
            } else {
                CbrtfPrecision::Midp
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
            panic!("invalid cbrtf const options");
        };
        opts
    }

    /// Create a dynamic runtime evaluator for this [`CbrtfOptions`] value.
    #[inline(always)]
    pub const fn runtime(self) -> super::CbrtfRuntime {
        super::CbrtfRuntime::new(self)
    }

    pub(crate) const fn input_bounds(&self) -> Option<(f32, f32)> {
        match self.input {
            Some((lo, hi)) => Some((f32::from_bits(lo), f32::from_bits(hi))),
            None => None,
        }
    }
    pub(crate) const fn denorm_possible(&self) -> bool {
        match self.input_bounds() {
            Some((lo, hi)) => lo < f32::MIN_POSITIVE && hi > -f32::MIN_POSITIVE,
            None => !self.full,
        }
    }
    pub(crate) const fn is_full(&self) -> bool {
        self.full
    }
    pub(crate) const fn is_lowp(&self) -> bool {
        matches!(self.precision, CbrtfPrecision::Lowp)
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

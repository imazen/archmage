//! Fluent `const` configuration builder for the exponential family (`exp2`, `exp`).

use crate::{Execution, Portability, Validation};

const MAGIC: u128 = 0x45;
const HAS_RANGE: u128 = 1 << 8;
const FULL: u128 = 1 << 9;
const LOWP: u128 = 1 << 10;
const BACKEND: u128 = 1 << 11;
const ASSUME: u128 = 1 << 12;
const CALLER: u128 = 1 << 13;
const CLAMP_OOB: u128 = 1 << 14;
const RESERVED_MASK: u128 = ((1_u128 << 49) - 1) << 15;

const fn abs(x: f32) -> f32 {
    if x < 0.0 { -x } else { x }
}

const fn canon_zero(x: f32) -> f32 {
    if x == 0.0 { 0.0 } else { x }
}

/// Approximation family, not a global mathematical error bound.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExpfPrecision {
    /// MIDP graph with refitted degree-6 coefficients (`<= 1.9 ULP` on
    /// `[-126, 127.5)`, `<= 0.78 ULP` on `[-0.5, 0.5]`).
    #[default]
    Midp,
    /// Shorter LOWP degree-3 graph with refitted coefficients (`<= 0.030%`
    /// relative error on `[-126, 127.99]`; up to `1.23%` when clamped at `127.99`).
    Lowp,
}

/// Fluent `const` configuration builder for [`super::Expf`] and [`crate::SimdMath`].
///
/// Options separate **what inputs are supported** ([`input_range`](Self::input_range),
/// [`small_exp2_range`](Self::small_exp2_range), [`small_exp_range`](Self::small_exp_range),
/// [`unclamped_exp2_range`](Self::unclamped_exp2_range),
/// [`unclamped_exp_range`](Self::unclamped_exp_range), and
/// [`admit_all_floats`](Self::admit_all_floats)) from **how out-of-bounds
/// inputs are handled**:
///
/// - [`error_on_out_of_bounds`](Self::error_on_out_of_bounds) (default, alias
///   [`checked`](Self::checked)): return `Err` on `NaN`, `±inf`, or inputs
///   outside `input_range`.
/// - [`clamp_out_of_bounds`](Self::clamp_out_of_bounds): clamp non-NaN inputs
///   into `[lo, hi]` (or `[-f32::MAX, f32::MAX]` when unranged) and propagate
///   `NaN` as `NaN` (infallible).
/// - [`assume_in_bounds`](Self::assume_in_bounds) (alias [`unchecked`](Self::unchecked)):
///   omit runtime checks and clamps when the caller already guarantees bounds
///   (infallible).
///
/// Declaring an [`input_range`](Self::input_range) inside `[-126, 127.5)`
/// (`[-126, 127.99]` for `Lowp`) elides overflow/underflow clamping, and
/// declaring one inside `[-0.5, 0.5]` (`[0, 1)` for `Lowp`) also elides integer
/// `2^k` range reduction at compile time.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpfOptions {
    input: Option<(u32, u32)>,
    full: bool,
    precision: ExpfPrecision,
    portability: Portability,
    validation: Validation,
    execution: Execution,
}

impl Default for ExpfOptions {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl ExpfOptions {
    /// Bounded finite domain, `Midp`, `BitExact`, `Checked`
    /// (`error_on_out_of_bounds`), `Execution::Upgrade`.
    pub const fn new() -> Self {
        Self {
            input: None,
            full: false,
            precision: ExpfPrecision::Midp,
            portability: Portability::BitExact,
            validation: Validation::Checked,
            execution: Execution::Upgrade,
        }
    }

    /// Accept inputs in `[lo, hi]`. Both must be finite `f32`s with `lo <= hi`.
    pub const fn input_range(mut self, lo: f32, hi: f32) -> Self {
        assert!(
            lo <= hi && abs(lo) <= f32::MAX && abs(hi) <= f32::MAX,
            "input range must be finite with lo <= hi"
        );
        self.input = Some((canon_zero(lo).to_bits(), canon_zero(hi).to_bits()));
        self
    }

    /// Single-octave `exp2` range `[-0.5, 0.5]`, eliding both output clamping
    /// and integer `2^k` reduction on `Midp` (`<= 0.78 ULP`).
    pub const fn small_exp2_range(self) -> Self {
        self.input_range(-0.5, 0.5)
    }

    /// Single-octave `exp` range `[-0.346_573_56, 0.346_573_56]` (`|x * log2(e)| <= 0.5`),
    /// eliding both output clamping and integer `2^k` reduction on `Midp`.
    pub const fn small_exp_range(self) -> Self {
        self.input_range(-0.346_573_56, 0.346_573_56)
    }

    /// Wide normal-output `exp2` range `[-126.0, 127.0]`, eliding overflow and
    /// underflow clamping on both `Midp` and `Lowp`.
    pub const fn unclamped_exp2_range(self) -> Self {
        self.input_range(-126.0, 127.0)
    }

    /// Wide normal-output `exp` range `[-87.0, 88.0]` (`x * log2(e)` in
    /// `[-125.52, 126.96]`), eliding overflow and underflow clamping on both
    /// `Midp` and `Lowp`.
    pub const fn unclamped_exp_range(self) -> Self {
        self.input_range(-87.0, 88.0)
    }

    /// Accept every `f32` input (`NaN -> NaN`, `+inf -> +inf`, `-inf -> +0`).
    pub const fn admit_all_floats(mut self) -> Self {
        self.full = true;
        self
    }

    /// Alias for [`admit_all_floats`](Self::admit_all_floats).
    pub const fn full_domain(self) -> Self {
        self.admit_all_floats()
    }

    /// Select the approximation family (`Midp` or `Lowp`).
    pub const fn precision(mut self, value: ExpfPrecision) -> Self {
        self.precision = value;
        self
    }

    /// Shorthand for `.precision(ExpfPrecision::Lowp)`.
    pub const fn lowp(self) -> Self {
        self.precision(ExpfPrecision::Lowp)
    }

    /// Shorthand for `.precision(ExpfPrecision::Midp)`.
    pub const fn midp(self) -> Self {
        self.precision(ExpfPrecision::Midp)
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

    /// Select how out-of-bounds inputs are handled.
    pub const fn validation(mut self, value: Validation) -> Self {
        self.validation = value;
        self
    }

    /// Return `Err` when any input falls outside the configured domain
    /// (`Validation::Checked`, default).
    pub const fn error_on_out_of_bounds(self) -> Self {
        self.validation(Validation::Checked)
    }

    /// Alias for [`error_on_out_of_bounds`](Self::error_on_out_of_bounds).
    pub const fn checked(self) -> Self {
        self.error_on_out_of_bounds()
    }

    /// Clamp finite/infinite out-of-bounds inputs into `[lo, hi]` (or
    /// `[-f32::MAX, f32::MAX]` when unranged) and propagate `NaN` as `NaN`
    /// (`Validation::ClampOutOfBounds`, infallible).
    pub const fn clamp_out_of_bounds(self) -> Self {
        self.validation(Validation::ClampOutOfBounds)
    }

    /// Omit numerical input checks and clamps; caller guarantees all inputs lie
    /// within the configured domain (`Validation::AssumeValid`, infallible).
    pub const fn assume_in_bounds(self) -> Self {
        self.validation(Validation::AssumeValid)
    }

    /// Alias for [`assume_in_bounds`](Self::assume_in_bounds).
    pub const fn unchecked(self) -> Self {
        self.assume_in_bounds()
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

    /// Whether `exp2(x)` keeps `x` in the un-clamped range (`[-126, 127.5)`
    /// `Midp` or `[-126, 127.99]` `Lowp`), allowing clamp elision.
    pub const fn elides_clamp(&self) -> bool {
        if self.full {
            return false;
        }
        let Some((lo_b, hi_b)) = self.input else {
            return false;
        };
        let (lo, hi) = (f32::from_bits(lo_b), f32::from_bits(hi_b));
        if matches!(self.precision, ExpfPrecision::Lowp) {
            lo >= -126.0 && hi <= 127.99
        } else {
            lo >= -126.0 && hi < 127.5
        }
    }

    /// Whether `exp(x)` keeps `x * log2(e)` in the un-clamped range (`[-126, 127.5)`
    /// `Midp` or `[-126, 127.99]` `Lowp`), allowing clamp elision.
    pub const fn elides_exp_clamp(&self) -> bool {
        if self.full {
            return false;
        }
        let Some((lo_b, hi_b)) = self.input else {
            return false;
        };
        let (lo, hi) = (f32::from_bits(lo_b), f32::from_bits(hi_b));
        let s_lo = lo * core::f32::consts::LOG2_E;
        let s_hi = hi * core::f32::consts::LOG2_E;
        if matches!(self.precision, ExpfPrecision::Lowp) {
            s_lo >= -126.0 && s_hi <= 127.99
        } else {
            s_lo >= -126.0 && s_hi < 127.5
        }
    }

    /// Whether `exp2(x)` stays in the fundamental single octave (`[-0.5, 0.5]`
    /// on `Midp` or `[0.0, 1.0)` on `Lowp`), eliding integer `2^k` reduction.
    pub const fn elides_reduction(&self) -> bool {
        if self.full {
            return false;
        }
        let Some((lo_b, hi_b)) = self.input else {
            return false;
        };
        let (lo, hi) = (f32::from_bits(lo_b), f32::from_bits(hi_b));
        if matches!(self.precision, ExpfPrecision::Lowp) {
            lo >= 0.0 && hi < 1.0
        } else {
            lo >= -0.5 && hi <= 0.5
        }
    }

    /// Whether `exp(x)` keeps `x * log2(e)` in the fundamental single octave
    /// (`[-0.5, 0.5]` on `Midp` or `[0.0, 1.0)` on `Lowp`), eliding integer
    /// `2^k` reduction.
    pub const fn elides_exp_reduction(&self) -> bool {
        if self.full {
            return false;
        }
        let Some((lo_b, hi_b)) = self.input else {
            return false;
        };
        let (lo, hi) = (f32::from_bits(lo_b), f32::from_bits(hi_b));
        let s_lo = lo * core::f32::consts::LOG2_E;
        let s_hi = hi * core::f32::consts::LOG2_E;
        if matches!(self.precision, ExpfPrecision::Lowp) {
            s_lo >= 0.0 && s_hi < 1.0
        } else {
            s_lo >= -0.5 && s_hi <= 0.5
        }
    }

    /// Whether this configuration can never return an input error
    /// (`.clamp_out_of_bounds()`, `.assume_in_bounds()` / `.unchecked()`, or
    /// `.admit_all_floats()` / `.full_domain()`).
    pub const fn is_infallible(&self) -> bool {
        self.assume_valid() || self.clamp_oob() || self.full
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
        if matches!(self.precision, ExpfPrecision::Lowp) {
            out |= LOWP;
        }
        if matches!(self.portability, Portability::BackendDependent) {
            out |= BACKEND;
        }
        match self.validation {
            Validation::Checked => {}
            Validation::ClampOutOfBounds => out |= CLAMP_OOB,
            Validation::AssumeValid => out |= ASSUME,
        }
        if matches!(self.execution, Execution::Caller) {
            out |= CALLER;
        }
        out
    }

    const fn from_bits_raw(bits: u128) -> Option<Self> {
        if (bits & 0xff) != MAGIC || (bits & RESERVED_MASK) != 0 {
            return None;
        }
        if (bits & ASSUME) != 0 && (bits & CLAMP_OOB) != 0 {
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
            if !(lo <= hi && abs(lo) <= f32::MAX && abs(hi) <= f32::MAX) {
                return None;
            }
            Some((lo_b, hi_b))
        } else {
            if lo_b != 0 || hi_b != 0 {
                return None;
            }
            None
        };
        let validation = if (bits & ASSUME) != 0 {
            Validation::AssumeValid
        } else if (bits & CLAMP_OOB) != 0 {
            Validation::ClampOutOfBounds
        } else {
            Validation::Checked
        };
        Some(Self {
            input,
            full: (bits & FULL) != 0,
            precision: if (bits & LOWP) != 0 {
                ExpfPrecision::Lowp
            } else {
                ExpfPrecision::Midp
            },
            portability: if (bits & BACKEND) != 0 {
                Portability::BackendDependent
            } else {
                Portability::BitExact
            },
            validation,
            execution: if (bits & CALLER) != 0 {
                Execution::Caller
            } else {
                Execution::Upgrade
            },
        })
    }

    /// Decode a `u128` option word back into [`ExpfOptions`].
    pub const fn from_bits(bits: u128) -> Option<Self> {
        let Some(opts) = Self::from_bits_raw(bits) else {
            return None;
        };
        if opts.is_valid() { Some(opts) } else { None }
    }

    pub(crate) const fn decode_const(bits: u128) -> Self {
        let Some(opts) = Self::from_bits_raw(bits) else {
            panic!("invalid expf const options bit pattern");
        };
        opts.validate();
        opts
    }

    /// Create a dynamic runtime evaluator for this [`ExpfOptions`] value.
    #[inline(always)]
    pub const fn runtime(self) -> super::ExpfRuntime {
        super::ExpfRuntime::new(self)
    }

    pub(crate) const fn input_bounds(&self) -> Option<(f32, f32)> {
        match self.input {
            Some((lo, hi)) => Some((f32::from_bits(lo), f32::from_bits(hi))),
            None => None,
        }
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
    pub(crate) const fn clamp_oob(&self) -> bool {
        matches!(self.validation, Validation::ClampOutOfBounds)
    }
    pub(crate) const fn assume_valid(&self) -> bool {
        matches!(self.validation, Validation::AssumeValid)
    }
    pub(crate) const fn is_caller(&self) -> bool {
        matches!(self.execution, Execution::Caller)
    }
}

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
const ZERO: u128 = 1 << 14;
const SUBNORM: u128 = 1 << 15;
const CLAMP_OOB: u128 = 1 << 16;
const RESERVED_MASK: u128 = ((1_u128 << 47) - 1) << 17;

/// Fundamental single-octave lower bound for `Midp` (`sqrt(0.5)` rounded up,
/// `0.70710677`).
const MIDP_OCTAVE_LO_BITS: u32 = 0x3f35_04f3;
/// Fundamental single-octave upper bound for `Midp` (`sqrt(2.0)` minus 1 ULP,
/// `1.4142134`).
const MIDP_OCTAVE_HI_BITS: u32 = 0x3fb5_04f2;
/// Fundamental single-octave lower bound for `Lowp` (`2/3` rounded up).
const LOWP_OCTAVE_LO_BITS: u32 = 0x3f2a_aaab;
/// Fundamental single-octave upper bound for `Lowp` (`4/3` minus 1 ULP).
const LOWP_OCTAVE_HI_BITS: u32 = 0x3faa_aaaa;

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
    /// Refitted `Midp` polynomial (`<= 4.5 ULP` on positive normals and
    /// normalized subnormals; `<= 2.9 ULP` on [`LogfOptions::near_one_range`]).
    #[default]
    Midp,
    /// Rational `P2/Q2` `Lowp` approximation (`~6e-6` absolute error).
    Lowp,
}

/// Fluent `const` configuration builder for [`super::Logf`] and [`crate::SimdMath`].
///
/// Options separate **what inputs are supported** ([`input_range`](Self::input_range),
/// [`near_one_range`](Self::near_one_range), [`small_log1p_range`](Self::small_log1p_range),
/// [`admit_zero`](Self::admit_zero), [`admit_subnormals`](Self::admit_subnormals),
/// and [`admit_all_floats`](Self::admit_all_floats)) from **how out-of-bounds
/// inputs are handled**:
///
/// - [`error_on_out_of_bounds`](Self::error_on_out_of_bounds) (default, alias
///   [`checked`](Self::checked)): return `Err` on out-of-domain inputs.
/// - [`clamp_out_of_bounds`](Self::clamp_out_of_bounds): clamp non-NaN inputs
///   into the configured domain and propagate `NaN` as `NaN` (infallible).
/// - [`assume_in_bounds`](Self::assume_in_bounds) (alias [`unchecked`](Self::unchecked)):
///   omit runtime checks and clamps when the caller already guarantees bounds
///   (infallible).
///
/// Domain granularity keeps hot loops fast: [`admit_zero`](Self::admit_zero)
/// adds `±0.0 -> -inf` (`log1p(-1.0) -> -inf`) with one off-critical-path blend,
/// whereas [`admit_subnormals`](Self::admit_subnormals) and
/// [`admit_all_floats`](Self::admit_all_floats) add `* 2^23` pre-normalization
/// before the rational reduction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LogfOptions {
    input: Option<(u32, u32)>,
    zero: bool,
    subnormals: bool,
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
    /// `BitExact`, `Checked` (`error_on_out_of_bounds`), `Execution::Upgrade`.
    pub const fn new() -> Self {
        Self {
            input: None,
            zero: false,
            subnormals: false,
            full: false,
            precision: LogfPrecision::Midp,
            portability: Portability::BitExact,
            validation: Validation::Checked,
            execution: Execution::Upgrade,
        }
    }

    /// Accept inputs in `[lo, hi]`.
    ///
    /// - For `log2`, `ln`, and `log10`, `lo` must be at least `f32::MIN_POSITIVE`
    ///   by default, positive subnormal when [`admit_subnormals`](Self::admit_subnormals)
    ///   is set, or `0.0` when both [`admit_zero`](Self::admit_zero) and
    ///   [`admit_subnormals`](Self::admit_subnormals) are set (`lo == hi == 0.0`
    ///   only requires [`admit_zero`](Self::admit_zero)).
    /// - For `log1p`, `lo > -1.0` by default, or `lo >= -1.0` when
    ///   [`admit_zero`](Self::admit_zero) is set.
    pub const fn input_range(mut self, lo: f32, hi: f32) -> Self {
        assert!(
            lo >= -1.0 && lo <= hi && abs(lo) <= f32::MAX && abs(hi) <= f32::MAX,
            "input range must be finite with -1 <= lo <= hi"
        );
        self.input = Some((canon_zero(lo).to_bits(), canon_zero(hi).to_bits()));
        self
    }

    /// Single-octave range around `1.0` (`[0.70710677, 1.4142134]`), eliding
    /// integer exponent extraction on `Midp` `log2` / `ln` / `log10` (`<= 2.9 ULP`).
    pub const fn near_one_range(self) -> Self {
        self.input_range(
            f32::from_bits(MIDP_OCTAVE_LO_BITS),
            f32::from_bits(MIDP_OCTAVE_HI_BITS),
        )
    }

    /// Single-octave `log1p` range `[-0.25, 0.25]` (`1 + x` in `[0.75, 1.25]`),
    /// eliding integer exponent extraction on both `Midp` and `Lowp` `log1p`.
    pub const fn small_log1p_range(self) -> Self {
        self.input_range(-0.25, 0.25)
    }

    /// Also accept `±0.0 -> -inf` (`-1.0 -> -inf` for `log1p`) without paying
    /// the subnormal pre-normalization cost.
    pub const fn admit_zero(mut self) -> Self {
        if !self.full {
            self.zero = true;
        }
        self
    }

    /// Also accept positive subnormals (`0 < x < f32::MIN_POSITIVE`), normalizing
    /// them accurately (`* 2^23` with `-23` exponent adjustment) before reduction.
    pub const fn admit_subnormals(mut self) -> Self {
        if !self.full {
            self.subnormals = true;
        }
        self
    }

    /// Accept every `f32` input, normalizing positive subnormals and propagating
    /// IEEE specials (`±0 -> -inf`, `< 0 -> NaN`, `NaN -> NaN`, `+inf -> +inf`;
    /// shifted by `-1` for `log1p`).
    pub const fn admit_all_floats(mut self) -> Self {
        self.full = true;
        self.zero = false;
        self.subnormals = false;
        self
    }

    /// Alias for [`admit_all_floats`](Self::admit_all_floats).
    pub const fn full_domain(self) -> Self {
        self.admit_all_floats()
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

    /// Clamp finite/infinite out-of-bounds inputs into the configured domain
    /// and propagate `NaN` as `NaN` (`Validation::ClampOutOfBounds`, infallible).
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

    /// Whether domain-level invariants common to all log variants hold.
    pub const fn is_valid(&self) -> bool {
        if self.full && self.input.is_some() {
            return false;
        }
        if let Some((lo_b, _)) = self.input {
            let lo = f32::from_bits(lo_b);
            if lo == -1.0 && !self.admits_zero() {
                return false;
            }
        }
        true
    }

    pub(crate) const fn is_valid_op(&self, op: u8) -> bool {
        if !self.is_valid() {
            return false;
        }
        if let Some((lo_b, hi_b)) = self.input {
            let lo = f32::from_bits(lo_b);
            let hi = f32::from_bits(hi_b);
            if lo.is_nan() || hi.is_nan() {
                return false;
            }
            if op == 3 {
                if lo == -1.0 {
                    if !self.admits_zero() {
                        return false;
                    }
                } else if 1.0 + lo < f32::MIN_POSITIVE && !self.admits_subnormals() {
                    return false;
                }
            } else if lo < 0.0 {
                return false;
            } else if lo == 0.0 {
                if !self.admits_zero() || (hi > 0.0 && !self.admits_subnormals()) {
                    return false;
                }
            } else if lo < f32::MIN_POSITIVE && !self.admits_subnormals() {
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
        if let Some((lo_b, _)) = self.input {
            let lo = f32::from_bits(lo_b);
            assert!(
                lo > -1.0 || self.admits_zero(),
                "log1p input_range lower bound of -1.0 requires .admit_zero()"
            );
        }
    }

    pub(crate) const fn validate_op(&self, op: u8) {
        self.validate();
        if let Some((lo_b, hi_b)) = self.input {
            let lo = f32::from_bits(lo_b);
            let hi = f32::from_bits(hi_b);
            if op == 3 {
                if lo == -1.0 {
                    assert!(
                        self.admits_zero(),
                        "log1p input_range lower bound of -1.0 requires .admit_zero()"
                    );
                } else {
                    assert!(
                        1.0 + lo >= f32::MIN_POSITIVE || self.admits_subnormals(),
                        "log1p input_range lower bound must keep 1 + lo positive normal (or enable .admit_subnormals())"
                    );
                }
            } else {
                assert!(
                    lo >= 0.0,
                    "log input_range lower bound must be non-negative"
                );
                if lo == 0.0 {
                    assert!(
                        self.admits_zero() && (hi == 0.0 || self.admits_subnormals()),
                        "log input_range starting at 0.0 requires .admit_zero().admit_subnormals() (or use f32::MIN_POSITIVE)"
                    );
                } else {
                    assert!(
                        lo >= f32::MIN_POSITIVE || self.admits_subnormals(),
                        "log input_range lower bound below f32::MIN_POSITIVE requires .admit_subnormals()"
                    );
                }
            }
        }
    }

    /// Whether `log2`/`ln`/`log10` stays inside the fundamental single octave
    /// (`[0.70710677, 1.4142134]` on `Midp` or `[0.6666667, 1.3333333]` on
    /// `Lowp`) without admitted zero or specials, eliding integer exponent
    /// extraction (`n = 0`).
    pub const fn elides_reduction(&self) -> bool {
        if self.full || self.zero || self.subnormals {
            return false;
        }
        let Some((lo_b, hi_b)) = self.input else {
            return false;
        };
        let (lo, hi) = (f32::from_bits(lo_b), f32::from_bits(hi_b));
        if matches!(self.precision, LogfPrecision::Lowp) {
            lo >= f32::from_bits(LOWP_OCTAVE_LO_BITS) && hi <= f32::from_bits(LOWP_OCTAVE_HI_BITS)
        } else {
            lo >= f32::from_bits(MIDP_OCTAVE_LO_BITS) && hi <= f32::from_bits(MIDP_OCTAVE_HI_BITS)
        }
    }

    /// Whether `log1p(x)` keeps `1 + x` inside the fundamental single octave,
    /// eliding integer exponent extraction and the `1 + x >= 2` Kahan branch.
    pub const fn elides_log1p_reduction(&self) -> bool {
        if self.full || self.zero || self.subnormals {
            return false;
        }
        let Some((lo_b, hi_b)) = self.input else {
            return false;
        };
        let u_lo = 1.0 + f32::from_bits(lo_b);
        let u_hi = 1.0 + f32::from_bits(hi_b);
        if matches!(self.precision, LogfPrecision::Lowp) {
            u_lo >= f32::from_bits(LOWP_OCTAVE_LO_BITS)
                && u_hi <= f32::from_bits(LOWP_OCTAVE_HI_BITS)
        } else {
            u_lo >= f32::from_bits(MIDP_OCTAVE_LO_BITS)
                && u_hi <= f32::from_bits(MIDP_OCTAVE_HI_BITS)
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
        if self.zero {
            out |= ZERO;
        }
        if self.subnormals {
            out |= SUBNORM;
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
        let full = (bits & FULL) != 0;
        let zero = (bits & ZERO) != 0;
        let subnormals = (bits & SUBNORM) != 0;
        if full && (zero || subnormals) {
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
            if !(lo >= -1.0 && lo <= hi && abs(lo) <= f32::MAX && abs(hi) <= f32::MAX) {
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
            zero,
            subnormals,
            full,
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
            validation,
            execution: if (bits & CALLER) != 0 {
                Execution::Caller
            } else {
                Execution::Upgrade
            },
        })
    }

    /// Decode a `u128` option word back into [`LogfOptions`].
    pub const fn from_bits(bits: u128) -> Option<Self> {
        let Some(opts) = Self::from_bits_raw(bits) else {
            return None;
        };
        if opts.is_valid() { Some(opts) } else { None }
    }

    pub(crate) const fn decode_const(bits: u128) -> Self {
        let Some(opts) = Self::from_bits_raw(bits) else {
            panic!("invalid logf const options bit pattern");
        };
        opts.validate();
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
    pub(crate) const fn admits_zero(&self) -> bool {
        self.zero || self.full
    }
    pub(crate) const fn admits_subnormals(&self) -> bool {
        self.subnormals || self.full
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

//! Fluent `const` configuration builder and runtime adapter for `powf`.

use crate::{Execution, Portability, Validation};

/// Largest `|n * log2(x)|` a configuration may reach before the kernel keeps
/// its range check. The approximation needs the argument in `[-126, 127.5]`.
const ELIDE_LIMIT: f32 = 120.0;

/// Largest finite magnitude representable without overflow in the 25-bit
/// outward-rounded exponent bound encoding (`~3.4028073e38`).
const MAX_EXP_BOUND: f32 = f32::from_bits(0x7f7f_ff80);

const MAGIC: u128 = 0xA;
const HAS_BASE: u128 = 1 << 4;
const HAS_EXP: u128 = 1 << 5;
const ZERO: u128 = 1 << 6;
const FULL: u128 = 1 << 7;
const CLAMP: u128 = 1 << 8;
const PREC_SHIFT: u32 = 9;
const BACKEND: u128 = 1 << 11;
const ASSUME: u128 = 1 << 12;
const CALLER: u128 = 1 << 13;
const BASE_LO_SHIFT: u32 = 14;
const BASE_HI_SHIFT: u32 = 46;
const EXP_LO_SHIFT: u32 = 78;
const EXP_HI_SHIFT: u32 = 103;
const MASK_25: u128 = (1 << 25) - 1;

const fn abs(x: f32) -> f32 {
    if x < 0.0 { -x } else { x }
}

const fn canon_zero(x: f32) -> f32 {
    if x == 0.0 { 0.0 } else { x }
}

/// Unbiased binary exponent of a positive normal value.
const fn exponent_of(x: f32) -> i32 {
    ((x.to_bits() >> 23) & 0xff) as i32 - 127
}

/// Largest 25-bit float (`sign + 8-bit exp + top 16 mantissa bits`) `<= x`.
const fn floor_25(x: f32) -> f32 {
    let x = canon_zero(x);
    let b = x.to_bits();
    if b == 0 {
        return 0.0;
    }
    if (b & 0x8000_0000) == 0 {
        f32::from_bits(b & !0x7f)
    } else if (b & 0x7f) == 0 {
        x
    } else {
        f32::from_bits((b & !0x7f) + 0x80)
    }
}

/// Smallest 25-bit float `>= x`.
const fn ceil_25(x: f32) -> f32 {
    canon_zero(-floor_25(-x))
}

/// Approximation family, not a global mathematical error bound.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum PowfPrecision {
    /// Existing MIDP coefficients and graph; no whole-domain ULP guarantee.
    #[default]
    Midp,
    /// Shorter LOWP coefficients; prefer a bounded `.gamma()` or custom range.
    Lowp,
    /// Table-based binary64 graph for [`Portability::BitExact`]; `std` `f32::powf`
    /// for [`Portability::BackendDependent`] (requires the `std` feature).
    Highp,
}

/// Fluent `const` configuration builder for [`super::Powf`] and [`crate::SimdMath`].
///
/// Call `.bits()` to pass as a `const CFG: u128` parameter for compile-time
/// specialization (eliding per-lane range checks and clamps when the declared
/// ranges stay inside `[-126, 127.5]`), or `.runtime()` to evaluate with
/// dynamic runtime options.
///
/// ```
/// use magemath::powf::{Powf, PowfOptions};
///
/// const ENCODE: u128 = PowfOptions::new()
///     .base_range(1.0 / 4096.0, 1.0)
///     .exponent_range(0.25, 3.0)
///     .admit_zero()
///     .bits();
///
/// assert!(PowfOptions::from_bits(ENCODE).unwrap().elides_range_check());
///
/// let src = [0.0, 0.25, 1.0];
/// let mut dst = [0.0; 3];
/// Powf::<ENCODE>::slice(&src, &mut dst, 0.5).unwrap();
/// assert_eq!(dst, [0.0, 0.5, 1.0]);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PowfOptions {
    base: Option<(u32, u32)>,
    exponent: Option<(u32, u32)>,
    zero: bool,
    full: bool,
    clamp: bool,
    precision: PowfPrecision,
    portability: Portability,
    validation: Validation,
    execution: Execution,
}

impl Default for PowfOptions {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl PowfOptions {
    /// Positive normal bases, any finite normal-or-zero exponent, `Midp`,
    /// `BitExact`, `Checked`, `Execution::Upgrade`.
    pub const fn new() -> Self {
        Self {
            base: None,
            exponent: None,
            zero: false,
            full: false,
            clamp: false,
            precision: PowfPrecision::Midp,
            portability: Portability::BitExact,
            validation: Validation::Checked,
            execution: Execution::Upgrade,
        }
    }

    /// Accept bases in `[lo, hi]`. Both must be positive normal finite `f32`s.
    pub const fn base_range(mut self, lo: f32, hi: f32) -> Self {
        assert!(
            lo >= f32::MIN_POSITIVE && lo <= hi && hi <= f32::MAX,
            "base range must be positive normal with lo <= hi"
        );
        self.base = Some((lo.to_bits(), hi.to_bits()));
        self
    }

    /// Accept exponents in `[lo, hi]`.
    ///
    /// Bounds are canonicalized outward to 25-bit floats (`sign + 8-bit exp +
    /// 16-bit mantissa`, exact for dyadic rationals such as `0.0`, `0.25`,
    /// `0.5`, `1.0`, `2.0`, `4.0`, `±8.0`) so `Powf::<{ opts.bits() }>` and
    /// `opts.runtime()` enforce the exact same bounds and always admit `lo` and `hi`.
    pub const fn exponent_range(mut self, lo: f32, hi: f32) -> Self {
        assert!(
            lo <= hi && abs(lo) <= MAX_EXP_BOUND && abs(hi) <= MAX_EXP_BOUND,
            "exponent range must be finite with lo <= hi"
        );
        let lo_c = floor_25(lo);
        let hi_c = ceil_25(hi);
        self.exponent = Some((lo_c.to_bits(), hi_c.to_bits()));
        self
    }

    /// Also accept `+0`: `0^0 = 1`, `0^positive = +0`. Negative exponents
    /// with a zero base remain rejected under [`Validation::Checked`].
    pub const fn admit_zero(mut self) -> Self {
        self.zero = true;
        self
    }

    /// Shorthand for the standard transfer-function gamma domain:
    /// `.base_range(1.0 / 65_536.0, 65_536.0).exponent_range(0.0, 4.0).admit_zero()`.
    pub const fn gamma(self) -> Self {
        self.base_range(1.0 / 65_536.0, 65_536.0)
            .exponent_range(0.0, 4.0)
            .admit_zero()
    }

    /// Shorthand for the strictly positive gamma domain (excludes `+0` and its
    /// zero masks): `.base_range(1.0 / 65_536.0, 65_536.0).exponent_range(0.0, 4.0)`.
    pub const fn positive_gamma(self) -> Self {
        self.base_range(1.0 / 65_536.0, 65_536.0)
            .exponent_range(0.0, 4.0)
    }

    /// Accept every `f32` input and output (requires [`PowfPrecision::Highp`]).
    pub const fn full_domain(mut self) -> Self {
        self.full = true;
        self
    }

    /// Clamp out-of-range `Midp`/`Lowp` `exp2` arguments instead of rejecting.
    pub const fn clamp_output(mut self) -> Self {
        self.clamp = true;
        self
    }

    /// Select the approximation family (`Midp`, `Lowp`, or `Highp`).
    pub const fn precision(mut self, value: PowfPrecision) -> Self {
        self.precision = value;
        self
    }

    /// Shorthand for `.precision(PowfPrecision::Lowp)`.
    pub const fn lowp(self) -> Self {
        self.precision(PowfPrecision::Lowp)
    }

    /// Shorthand for `.precision(PowfPrecision::Midp)`.
    pub const fn midp(self) -> Self {
        self.precision(PowfPrecision::Midp)
    }

    /// Shorthand for `.precision(PowfPrecision::Highp)`.
    pub const fn highp(self) -> Self {
        self.precision(PowfPrecision::Highp)
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

    /// Whether this configuration is self-consistent for the enabled crate features.
    pub const fn is_valid(&self) -> bool {
        if self.full && !matches!(self.precision, PowfPrecision::Highp) {
            return false;
        }
        if matches!(self.precision, PowfPrecision::Highp)
            && matches!(self.portability, Portability::BackendDependent)
            && !cfg!(feature = "std")
        {
            return false;
        }
        if self.full && (self.base.is_some() || self.exponent.is_some()) {
            return false;
        }
        if self.clamp && matches!(self.precision, PowfPrecision::Highp) {
            return false;
        }
        true
    }

    /// Assert that this configuration is self-consistent for the enabled features.
    pub const fn validate(&self) {
        assert!(
            !self.full || matches!(self.precision, PowfPrecision::Highp),
            "full_domain requires PowfPrecision::Highp"
        );
        assert!(
            !matches!(self.precision, PowfPrecision::Highp)
                || !matches!(self.portability, Portability::BackendDependent)
                || cfg!(feature = "std"),
            "BackendDependent Highp requires the `std` feature"
        );
        assert!(
            !(self.full && (self.base.is_some() || self.exponent.is_some())),
            "full_domain cannot be combined with base_range or exponent_range"
        );
        assert!(
            !(self.clamp && matches!(self.precision, PowfPrecision::Highp)),
            "clamp_output applies to Midp and Lowp, not Highp"
        );
    }

    /// Whether the declared ranges prove the computed `n * log2(x)` stays in
    /// range, so the kernel emits no range check or clamp.
    pub const fn elides_range_check(&self) -> bool {
        if self.full {
            return false;
        }
        let (Some((lo_b, hi_b)), Some((nlo_b, nhi_b))) = (self.base, self.exponent) else {
            return false;
        };
        let (lo, hi) = (f32::from_bits(lo_b), f32::from_bits(hi_b));
        let (nlo, nhi) = (f32::from_bits(nlo_b), f32::from_bits(nhi_b));
        let (elo, ehi) = (exponent_of(lo), exponent_of(hi) + 1);
        let e = (if -elo > ehi { -elo } else { ehi }) + 1;
        let e = if e < 1 { 1 } else { e };
        let n = if abs(nlo) > abs(nhi) {
            abs(nlo)
        } else {
            abs(nhi)
        };
        (e as f32) * n <= ELIDE_LIMIT && !(self.zero && nlo < 0.0)
    }

    /// Whether this configuration can never return an input/range error
    /// (`.unchecked()` or `.full_domain().highp()`).
    pub const fn is_infallible(&self) -> bool {
        self.assume_valid() || (self.full && matches!(self.precision, PowfPrecision::Highp))
    }

    /// Pack this configuration into a `u128` const-generic argument for [`super::Powf`]
    /// and [`crate::SimdMath`].
    pub const fn bits(self) -> u128 {
        let mut out = MAGIC;
        if let Some((lo, hi)) = self.base {
            out |= HAS_BASE | ((lo as u128) << BASE_LO_SHIFT) | ((hi as u128) << BASE_HI_SHIFT);
        }
        if let Some((nlo, nhi)) = self.exponent {
            let nlo_25 = ((nlo >> 7) as u128) & MASK_25;
            let nhi_25 = ((nhi >> 7) as u128) & MASK_25;
            out |= HAS_EXP | (nlo_25 << EXP_LO_SHIFT) | (nhi_25 << EXP_HI_SHIFT);
        }
        if self.zero {
            out |= ZERO;
        }
        if self.full {
            out |= FULL;
        }
        if self.clamp {
            out |= CLAMP;
        }
        let prec: u128 = match self.precision {
            PowfPrecision::Midp => 0,
            PowfPrecision::Lowp => 1,
            PowfPrecision::Highp => 2,
        };
        out |= prec << PREC_SHIFT;
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

    /// Decode a `u128` option word back into [`PowfOptions`], returning `None`
    /// if the tag, bounds, or feature combination is invalid.
    pub const fn from_bits(bits: u128) -> Option<Self> {
        if (bits & 0xf) != MAGIC {
            return None;
        }
        let precision = match (bits >> PREC_SHIFT) & 0b11 {
            0 => PowfPrecision::Midp,
            1 => PowfPrecision::Lowp,
            2 => PowfPrecision::Highp,
            _ => return None,
        };
        let has_base = (bits & HAS_BASE) != 0;
        let base_lo_b = ((bits >> BASE_LO_SHIFT) & 0xffff_ffff) as u32;
        let base_hi_b = ((bits >> BASE_HI_SHIFT) & 0xffff_ffff) as u32;
        let base = if has_base {
            let lo = f32::from_bits(base_lo_b);
            let hi = f32::from_bits(base_hi_b);
            if !(lo >= f32::MIN_POSITIVE && lo <= hi && hi <= f32::MAX) {
                return None;
            }
            Some((base_lo_b, base_hi_b))
        } else {
            if base_lo_b != 0 || base_hi_b != 0 {
                return None;
            }
            None
        };

        let has_exp = (bits & HAS_EXP) != 0;
        let exp_lo_25 = ((bits >> EXP_LO_SHIFT) & MASK_25) as u32;
        let exp_hi_25 = ((bits >> EXP_HI_SHIFT) & MASK_25) as u32;
        let exponent = if has_exp {
            let nlo_b = exp_lo_25 << 7;
            let nhi_b = exp_hi_25 << 7;
            let nlo = f32::from_bits(nlo_b);
            let nhi = f32::from_bits(nhi_b);
            // Reject -0.0 or non-ordered/out-of-range bounds.
            if nlo_b == 0x8000_0000
                || nhi_b == 0x8000_0000
                || !(nlo <= nhi && abs(nlo) <= MAX_EXP_BOUND && abs(nhi) <= MAX_EXP_BOUND)
            {
                return None;
            }
            Some((nlo_b, nhi_b))
        } else {
            if exp_lo_25 != 0 || exp_hi_25 != 0 {
                return None;
            }
            None
        };

        let opts = Self {
            base,
            exponent,
            zero: (bits & ZERO) != 0,
            full: (bits & FULL) != 0,
            clamp: (bits & CLAMP) != 0,
            precision,
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
            panic!("invalid powf const options");
        };
        opts
    }

    /// Create a dynamic runtime evaluator for this [`PowfOptions`] value.
    #[inline(always)]
    pub const fn runtime(self) -> super::PowfRuntime {
        super::PowfRuntime::new(self)
    }

    pub(crate) const fn base_bounds(&self) -> Option<(f32, f32)> {
        match self.base {
            Some((lo, hi)) => Some((f32::from_bits(lo), f32::from_bits(hi))),
            None => None,
        }
    }
    pub(crate) const fn exponent_bounds(&self) -> Option<(f32, f32)> {
        match self.exponent {
            Some((lo, hi)) => Some((f32::from_bits(lo), f32::from_bits(hi))),
            None => None,
        }
    }
    pub(crate) const fn admits_zero(&self) -> bool {
        self.zero
    }
    pub(crate) const fn is_full(&self) -> bool {
        self.full
    }
    pub(crate) const fn should_clamp(&self) -> bool {
        self.clamp
    }
    pub(crate) const fn is_lowp(&self) -> bool {
        matches!(self.precision, PowfPrecision::Lowp)
    }
    pub(crate) const fn is_highp(&self) -> bool {
        matches!(self.precision, PowfPrecision::Highp)
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

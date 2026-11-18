//! Fluent `const` configuration builder and runtime adapter for `powf`.

use crate::{Execution, Portability, Validation};

/// Largest `|n * log2(x)|` a configuration may reach before the kernel keeps
/// its range check. Keeping all four interval corners within `[-125.0, 125.0]`
/// stays strictly inside `Midp`/`Lowp`'s `[-126, 127.5)` domain and `Highp`'s
/// `|y * log2(x)| < 126` fast-pack domain.
const ELIDE_LIMIT: f32 = 125.0;

/// Largest finite magnitude representable without overflow in the 25-bit
/// outward-rounded exponent bound encoding (`~3.4028073e38`).
pub const MAX_EXP_BOUND: f32 = f32::from_bits(0x7f7f_ff80);

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
const SUBNORM: u128 = 1 << 14;
const CLAMP_OOB: u128 = 1 << 15;
const BASE_LO_SHIFT: u32 = 16;
const BASE_HI_SHIFT: u32 = 47;
const EXP_LO_SHIFT: u32 = 78;
const EXP_HI_SHIFT: u32 = 103;
const MASK_31: u128 = (1 << 31) - 1;
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

/// Whether `[lo, hi]` contains exclusively nonzero subnormals (excluding `0.0`
/// and every normal `f32`).
const fn is_nonzero_subnormal_only_interval(lo: f32, hi: f32) -> bool {
    (lo > 0.0 && hi < f32::MIN_POSITIVE) || (lo > -f32::MIN_POSITIVE && hi < 0.0)
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
    /// Shorter LOWP coefficients; prefer a bounded `.gamma_domain()` or custom range.
    Lowp,
    /// Table-based binary64 graph for [`Portability::BitExact`]; `std` `f32::powf`
    /// for [`Portability::BackendDependent`] (requires the `std` feature).
    Highp,
}

/// Fluent `const` configuration builder for [`super::Powf`] and [`crate::SimdMath`].
///
/// Options separate **what inputs are supported** (domain & range methods such as
/// [`base_range`](Self::base_range), [`exponent_range`](Self::exponent_range),
/// [`admit_zero`](Self::admit_zero), [`admit_subnormals`](Self::admit_subnormals),
/// [`gamma_domain`](Self::gamma_domain), [`srgb_encode_domain`](Self::srgb_encode_domain),
/// and [`admit_all_floats`](Self::admit_all_floats)) from **how out-of-bounds
/// inputs are handled**:
///
/// - [`error_on_out_of_bounds`](Self::error_on_out_of_bounds) (default, alias
///   [`checked`](Self::checked)): return `Err` on out-of-domain inputs or
///   out-of-range outputs.
/// - [`clamp_out_of_bounds`](Self::clamp_out_of_bounds): clamp finite out-of-bounds
///   bases/exponents into the configured domain, saturate overflow/underflow
///   outputs, and propagate `NaN` as `NaN` (infallible).
/// - [`assume_in_bounds`](Self::assume_in_bounds) (alias [`unchecked`](Self::unchecked)):
///   omit runtime checks and clamps when the caller already guarantees bounds
///   (infallible).
///
/// Call `.bits()` to pass as a `const CFG: u128` parameter for compile-time
/// specialization, or `.runtime()` to evaluate with dynamic runtime options.
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
    subnormals: bool,
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
    /// `BitExact`, `Checked` (`error_on_out_of_bounds`), `Execution::Upgrade`.
    pub const fn new() -> Self {
        Self {
            base: None,
            exponent: None,
            zero: false,
            subnormals: false,
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

    /// Restrict exponents to `[lo, hi]` within the active exponent domain
    /// (normal-or-zero unless [`admit_subnormals`](Self::admit_subnormals) is enabled).
    ///
    /// Both bounds must satisfy `|bound| <= MAX_EXP_BOUND` (`0x7f7f_ff80`,
    /// `~3.4028073e38`). Bounds are canonicalized outward to 25-bit floats
    /// (`sign + 8-bit exp + 16-bit mantissa`, exact for dyadic rationals such
    /// as `0.0`, `0.25`, `0.5`, `1.0`, `2.0`, `4.0`, `±8.0`) so
    /// `Powf::<{ opts.bits() }>` and `opts.runtime()` enforce the exact same
    /// interval. Intervals containing only nonzero subnormals require
    /// [`admit_subnormals`](Self::admit_subnormals).
    pub const fn exponent_range(mut self, lo: f32, hi: f32) -> Self {
        assert!(
            lo <= hi && abs(lo) <= MAX_EXP_BOUND && abs(hi) <= MAX_EXP_BOUND,
            "exponent range must be finite within ±MAX_EXP_BOUND (0x7f7f_ff80) with lo <= hi"
        );
        let lo_c = floor_25(lo);
        let hi_c = ceil_25(hi);
        self.exponent = Some((lo_c.to_bits(), hi_c.to_bits()));
        self
    }

    /// Also accept `+0.0` bases when `exponent >= 0`: `0^0 = 1`, `0^positive = +0`.
    /// Negative exponents with a zero base remain rejected under [`Validation::Checked`].
    pub const fn admit_zero(mut self) -> Self {
        if !self.full {
            self.zero = true;
        }
        self
    }

    /// Alias for [`admit_zero`](Self::admit_zero) clarifying that `+0.0` is
    /// added to the base domain (`0.0` exponents are already accepted by default).
    pub const fn admit_zero_base(self) -> Self {
        self.admit_zero()
    }

    /// Also accept positive subnormal bases (`0 < x < f32::MIN_POSITIVE`,
    /// normalized by `* 2^23` with `-23` exponent adjustment) and nonzero
    /// subnormal exponents.
    ///
    /// When combined with [`base_range`](Self::base_range), `lo` must be
    /// `f32::MIN_POSITIVE` so the accepted base domain has no normal gap.
    pub const fn admit_subnormals(mut self) -> Self {
        if !self.full {
            self.subnormals = true;
        }
        self
    }

    /// Standard transfer-function gamma domain:
    /// `.base_range(1.0 / 65_536.0, 65_536.0).exponent_range(0.0, 4.0).admit_zero()`.
    pub const fn gamma_domain(self) -> Self {
        let mut s = self
            .base_range(1.0 / 65_536.0, 65_536.0)
            .exponent_range(0.0, 4.0);
        if !s.full {
            s.zero = true;
            s.subnormals = false;
        }
        s
    }

    /// Shorthand alias for [`gamma_domain`](Self::gamma_domain).
    pub const fn gamma(self) -> Self {
        self.gamma_domain()
    }

    /// Strictly positive gamma domain (excludes `+0` and subnormals and their
    /// masks): `.base_range(1.0 / 65_536.0, 65_536.0).exponent_range(0.0, 4.0)`.
    pub const fn positive_gamma_domain(self) -> Self {
        let mut s = self
            .base_range(1.0 / 65_536.0, 65_536.0)
            .exponent_range(0.0, 4.0);
        s.zero = false;
        s.subnormals = false;
        s
    }

    /// Shorthand alias for [`positive_gamma_domain`](Self::positive_gamma_domain).
    pub const fn positive_gamma(self) -> Self {
        self.positive_gamma_domain()
    }

    /// Unit-normalized SDR transfer-function domain:
    /// `.base_range(1.0 / 65_536.0, 1.0).exponent_range(0.0, 4.0).admit_zero()`.
    pub const fn unit_gamma_domain(self) -> Self {
        let mut s = self
            .base_range(1.0 / 65_536.0, 1.0)
            .exponent_range(0.0, 4.0);
        if !s.full {
            s.zero = true;
            s.subnormals = false;
        }
        s
    }

    /// Encoding gamma domain (`1/2.4`, `1/2.2`, `sqrt`) across all positive
    /// normals in `(0, 1]` plus `+0.0`, eliding per-lane output range checks:
    /// `.base_range(f32::MIN_POSITIVE, 1.0).exponent_range(0.0, 0.99).admit_zero()`.
    pub const fn srgb_encode_domain(self) -> Self {
        let mut s = self
            .base_range(f32::MIN_POSITIVE, 1.0)
            .exponent_range(0.0, 0.99);
        if !s.full {
            s.zero = true;
            s.subnormals = false;
        }
        s
    }

    /// Accept every `f32` input and output (requires [`PowfPrecision::Highp`]).
    pub const fn admit_all_floats(mut self) -> Self {
        self.full = true;
        self.zero = false;
        self.subnormals = false;
        self
    }

    /// Alias for [`admit_all_floats`](Self::admit_all_floats) (requires [`PowfPrecision::Highp`]).
    pub const fn full_domain(self) -> Self {
        self.admit_all_floats()
    }

    /// Saturate out-of-range `Midp`/`Lowp` `exp2(n * log2(x))` outputs instead
    /// of returning `OutsideApproximationRange`, while preserving configured
    /// input validation.
    pub const fn saturate_output(mut self) -> Self {
        self.clamp = true;
        self
    }

    /// Alias for [`saturate_output`](Self::saturate_output).
    pub const fn clamp_output(self) -> Self {
        self.saturate_output()
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

    /// Select how out-of-bounds inputs and intermediate ranges are handled.
    pub const fn validation(mut self, value: Validation) -> Self {
        self.validation = value;
        self
    }

    /// Return `Err` when any input or un-clamped output falls outside the
    /// configured domain (`Validation::Checked`, default).
    pub const fn error_on_out_of_bounds(self) -> Self {
        self.validation(Validation::Checked)
    }

    /// Alias for [`error_on_out_of_bounds`](Self::error_on_out_of_bounds).
    pub const fn checked(self) -> Self {
        self.error_on_out_of_bounds()
    }

    /// Clamp finite out-of-bounds bases and exponents into the configured
    /// domain, saturate out-of-range outputs, and propagate `NaN` as `NaN`
    /// (`Validation::ClampOutOfBounds`, infallible).
    pub const fn clamp_out_of_bounds(self) -> Self {
        self.validation(Validation::ClampOutOfBounds)
    }

    /// Omit numerical input/output checks and clamps; caller guarantees all
    /// inputs lie within the configured domain (`Validation::AssumeValid`, infallible).
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
        if self.subnormals
            && let Some((lo_b, _)) = self.base
            && lo_b != f32::MIN_POSITIVE.to_bits()
        {
            return false;
        }
        if !self.subnormals
            && !self.full
            && let Some((nlo_b, nhi_b)) = self.exponent
            && is_nonzero_subnormal_only_interval(f32::from_bits(nlo_b), f32::from_bits(nhi_b))
        {
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
            "clamp_output / saturate_output applies to Midp and Lowp, not Highp"
        );
        if self.subnormals
            && let Some((lo_b, _)) = self.base
        {
            assert!(
                lo_b == f32::MIN_POSITIVE.to_bits(),
                "admit_subnormals with base_range requires lo == f32::MIN_POSITIVE so the domain has no normal gap"
            );
        }
        if !self.subnormals
            && !self.full
            && let Some((nlo_b, nhi_b)) = self.exponent
        {
            assert!(
                !is_nonzero_subnormal_only_interval(f32::from_bits(nlo_b), f32::from_bits(nhi_b)),
                "exponent_range contains only nonzero subnormals; add .admit_subnormals() or include 0.0 / normal bounds"
            );
        }
    }

    /// Whether the declared ranges prove the computed `n * log2(x)` stays in
    /// `[-125, 125]`, so the kernel emits no output range check or clamp.
    pub const fn elides_range_check(&self) -> bool {
        if self.full {
            return false;
        }
        let (Some((lo_b, hi_b)), Some((nlo_b, nhi_b))) = (self.base, self.exponent) else {
            return false;
        };
        let (lo, hi) = (f32::from_bits(lo_b), f32::from_bits(hi_b));
        let (nlo, nhi) = (f32::from_bits(nlo_b), f32::from_bits(nhi_b));
        if self.zero && nlo < 0.0 {
            return false;
        }
        let l_lo = if self.subnormals {
            -149.01
        } else {
            exponent_of(lo) as f32 - 0.01
        };
        let l_hi = (exponent_of(hi) + 1) as f32 + 0.01;
        let c0 = l_lo * nlo;
        let c1 = l_lo * nhi;
        let c2 = l_hi * nlo;
        let c3 = l_hi * nhi;
        abs(c0) <= ELIDE_LIMIT
            && abs(c1) <= ELIDE_LIMIT
            && abs(c2) <= ELIDE_LIMIT
            && abs(c3) <= ELIDE_LIMIT
    }

    /// Whether this configuration can never return an input/range error
    /// (`.clamp_out_of_bounds()`, `.assume_in_bounds()` / `.unchecked()`, or
    /// `.admit_all_floats().highp()`).
    pub const fn is_infallible(&self) -> bool {
        self.assume_valid()
            || self.clamp_oob()
            || (self.full && matches!(self.precision, PowfPrecision::Highp))
    }

    /// Pack this configuration into a `u128` const-generic argument for [`super::Powf`]
    /// and [`crate::SimdMath`].
    pub const fn bits(self) -> u128 {
        let mut out = MAGIC;
        if let Some((lo, hi)) = self.base {
            let lo_31 = (lo as u128) & MASK_31;
            let hi_31 = (hi as u128) & MASK_31;
            out |= HAS_BASE | (lo_31 << BASE_LO_SHIFT) | (hi_31 << BASE_HI_SHIFT);
        }
        if let Some((nlo, nhi)) = self.exponent {
            let nlo_25 = ((nlo >> 7) as u128) & MASK_25;
            let nhi_25 = ((nhi >> 7) as u128) & MASK_25;
            out |= HAS_EXP | (nlo_25 << EXP_LO_SHIFT) | (nhi_25 << EXP_HI_SHIFT);
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
        if (bits & 0xf) != MAGIC {
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
        let precision = match (bits >> PREC_SHIFT) & 0b11 {
            0 => PowfPrecision::Midp,
            1 => PowfPrecision::Lowp,
            2 => PowfPrecision::Highp,
            _ => return None,
        };
        let has_base = (bits & HAS_BASE) != 0;
        let base_lo_b = ((bits >> BASE_LO_SHIFT) & MASK_31) as u32;
        let base_hi_b = ((bits >> BASE_HI_SHIFT) & MASK_31) as u32;
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

        let validation = if (bits & ASSUME) != 0 {
            Validation::AssumeValid
        } else if (bits & CLAMP_OOB) != 0 {
            Validation::ClampOutOfBounds
        } else {
            Validation::Checked
        };

        Some(Self {
            base,
            exponent,
            zero,
            subnormals,
            full,
            clamp: (bits & CLAMP) != 0,
            precision,
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

    /// Decode a `u128` option word back into [`PowfOptions`], returning `None`
    /// if the tag, bounds, or feature combination is invalid.
    pub const fn from_bits(bits: u128) -> Option<Self> {
        let Some(opts) = Self::from_bits_raw(bits) else {
            return None;
        };
        if opts.is_valid() { Some(opts) } else { None }
    }

    pub(crate) const fn decode_const(bits: u128) -> Self {
        let Some(opts) = Self::from_bits_raw(bits) else {
            panic!("invalid powf const options bit pattern");
        };
        opts.validate();
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
        self.zero || self.full
    }
    pub(crate) const fn admits_subnormals(&self) -> bool {
        self.subnormals || self.full
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

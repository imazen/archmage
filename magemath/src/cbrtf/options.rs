//! Fluent `const` configuration builder, discrete [`CbrtfDomain`] enum,
//! [`Subnormals`] policy, and `bf16` custom range bounds for `cbrtf`.

use crate::{
    Execution, Portability, Subnormals, Validation,
    bf16::{MAX_CBRT_BF16, ceil_bf16, decode_bf16_finite, expand_bf16, floor_bf16},
};

const MAX_BOUNDED_INCLUSIVE: f32 = f32::from_bits((f32::MAX / 3.0).to_bits() - 1);

// Uniform u128 layout:
//   [0..4]   = MAGIC (0xD)
//   [4..9]   = CbrtfDomain (0..=4)
//   [9..12]  = Subnormals (0..=2)
//   [12..15] = Validation (0..=2)
//   [15..18] = CbrtfPrecision (0..=1)
//   [18..20] = Portability (0..=1)
//   [20..22] = Execution (0..=1)
//   [22]     = HAS_CUSTOM_RANGE
//   [23..64] = RESERVED_LO (41 bits, must be 0)
//   [64..80] = in_lo_bf16 (u16, 0 when !HAS_CUSTOM_RANGE)
//   [80..96] = in_hi_bf16 (u16, 0 when !HAS_CUSTOM_RANGE)
//   [96..128]= RESERVED_HI (32 bits, must be 0)
const MAGIC: u128 = 0xD;
const DOMAIN_SHIFT: u32 = 4;
const SUBNORM_SHIFT: u32 = 9;
const VALIDATION_SHIFT: u32 = 12;
const PREC_SHIFT: u32 = 15;
const PORTABILITY_SHIFT: u32 = 18;
const EXECUTION_SHIFT: u32 = 20;
const HAS_CUSTOM_RANGE: u128 = 1 << 22;
const RESERVED_LO_MASK: u128 = ((1_u128 << 41) - 1) << 23;
const IN_LO_SHIFT: u32 = 64;
const IN_HI_SHIFT: u32 = 80;
const RESERVED_HI_MASK: u128 = 0xffff_ffff_u128 << 96;

const fn abs(x: f32) -> f32 {
    if x < 0.0 { -x } else { x }
}

const fn is_nonzero_subnormal_only_interval(lo: f32, hi: f32) -> bool {
    (lo > 0.0 && hi < f32::MIN_POSITIVE) || (lo > -f32::MIN_POSITIVE && hi < 0.0)
}

/// Discrete input domain for [`CbrtfOptions`].
///
/// Subnormal normalization is **off by default** on every
/// domain except [`CbrtfDomain::AllFloats`]; opt in explicitly via
/// [`CbrtfOptions::admit_subnormals`] or shove subnormals to `0` / `MIN_POSITIVE`
/// via [`CbrtfOptions::flush_or_clamp_subnormals`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum CbrtfDomain {
    /// `±0.0` and normal floats strictly below `f32::MAX / 3.0`.
    #[default]
    Bounded,
    /// Non-negative unit interval `[0.0, 1.0]` (`{±0.0} ∪ [f32::MIN_POSITIVE, 1.0]` by default).
    Unit,
    /// Signed unit interval `[-1.0, 1.0]` (`[-1.0, -MIN_POSITIVE] ∪ {±0.0} ∪ [MIN_POSITIVE, 1.0]` by default).
    SignedUnit,
    /// Strictly positive normal floats `[f32::MIN_POSITIVE, f32::MAX / 3.0)`,
    /// eliding subnormal checks and zero restoration.
    PositiveNormals,
    /// Every `f32` input, including signed zeros, subnormals, infinities, and NaNs.
    AllFloats,
}

/// Approximation family, not a global mathematical error bound.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum CbrtfPrecision {
    /// Degree-seven polynomial for the factored mantissa correction (1.47 ULP
    /// maximum measured error with [`Portability::BitExact`]).
    #[default]
    Midp,
    /// Degree-three polynomial for the factored mantissa correction (180.67 ULP
    /// maximum measured error with [`Portability::BitExact`]).
    Lowp,
}

/// Fluent `const` configuration builder for [`super::Cbrtf`] and [`crate::SimdMath`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CbrtfOptions {
    domain: CbrtfDomain,
    custom_range: Option<(u16, u16)>,
    subnormals: Subnormals,
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
    /// `CbrtfDomain::Bounded`, `Subnormals::FlushOrClamp`, `Midp`, `BitExact`,
    /// `Validation::Checked` (`error_on_out_of_bounds`), `Execution::Upgrade`.
    pub const fn new() -> Self {
        Self {
            domain: CbrtfDomain::Bounded,
            custom_range: None,
            subnormals: Subnormals::FlushOrClamp,
            precision: CbrtfPrecision::Midp,
            portability: Portability::BitExact,
            validation: Validation::Checked,
            execution: Execution::Upgrade,
        }
    }

    /// Select the discrete [`CbrtfDomain`], clearing any custom [`input_range`](Self::input_range).
    pub const fn domain(mut self, domain: CbrtfDomain) -> Self {
        self.domain = domain;
        self.custom_range = None;
        if matches!(domain, CbrtfDomain::AllFloats) {
            self.subnormals = Subnormals::Reject;
        }
        self
    }

    /// Override input bounds with outward-rounded `bf16` `[lo, hi]`
    /// (`|bound| <= MAX_CBRT_BF16`, `lo <= hi`).
    pub const fn input_range(mut self, lo: f32, hi: f32) -> Self {
        assert!(
            lo <= hi && abs(lo) <= MAX_CBRT_BF16 && abs(hi) <= MAX_CBRT_BF16,
            "cbrtf input range must satisfy -MAX_CBRT_BF16 <= lo <= hi <= MAX_CBRT_BF16 (< f32::MAX / 3)"
        );
        self.custom_range = Some((floor_bf16(lo), ceil_bf16(hi)));
        self
    }

    /// Signed unit cube-root domain `[-1.0, 1.0]` (`CbrtfDomain::SignedUnit`;
    /// subnormals flush to signed zero by default).
    pub const fn unit_range(self) -> Self {
        self.domain(CbrtfDomain::SignedUnit)
    }

    /// Configure how nonzero subnormal inputs (`0 < |x| < f32::MIN_POSITIVE`) are handled.
    pub const fn subnormals(mut self, policy: Subnormals) -> Self {
        self.subnormals = policy;
        self
    }

    /// Treat nonzero subnormals as out-of-bounds (`Subnormals::Reject`).
    pub const fn reject_subnormals(self) -> Self {
        self.subnormals(Subnormals::Reject)
    }

    /// Shove nonzero subnormals out to `±0.0` (when `0.0` is in range) or the
    /// nearest normal range endpoint (when `0.0` is excluded), even under
    /// [`Validation::Checked`].
    pub const fn flush_or_clamp_subnormals(self) -> Self {
        self.subnormals(Subnormals::FlushOrClamp)
    }

    /// Opt in to exact subnormal normalization before the polynomial and
    /// compensation in the output exponent (`Subnormals::Normalize`).
    pub const fn admit_subnormals(self) -> Self {
        self.subnormals(Subnormals::Normalize)
    }

    /// Accept every `f32` input (`CbrtfDomain::AllFloats`).
    pub const fn admit_all_floats(self) -> Self {
        self.domain(CbrtfDomain::AllFloats)
    }

    /// Alias for [`admit_all_floats`](Self::admit_all_floats).
    pub const fn full_domain(self) -> Self {
        self.admit_all_floats()
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

    /// Select how out-of-bounds inputs are handled.
    pub const fn validation(mut self, value: Validation) -> Self {
        self.validation = value;
        self
    }

    /// Return `Err` when any input falls outside the configured domain (`Validation::Checked`, default).
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

    /// Clamp inputs and outputs as applicable. NaN inputs have unspecified
    /// numerical results (`Validation::ClampFinite`, infallible).
    pub const fn clamp_finite(self) -> Self {
        self.validation(Validation::ClampFinite)
    }

    /// Omit numerical input checks and clamps (`Validation::AssumeValid`, infallible).
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

    /// Active discrete [`CbrtfDomain`].
    pub const fn selected_domain(&self) -> CbrtfDomain {
        self.domain
    }

    /// Active [`Subnormals`] policy (`Normalize` when `CbrtfDomain::AllFloats`).
    pub const fn subnormals_policy(&self) -> Subnormals {
        if self.is_full() {
            Subnormals::Normalize
        } else {
            self.subnormals
        }
    }

    pub(crate) const fn is_full(&self) -> bool {
        matches!(self.domain, CbrtfDomain::AllFloats)
    }

    pub(crate) const fn input_bounds(&self) -> Option<(f32, f32)> {
        if self.is_full() {
            return None;
        }
        if let Some((lo, hi)) = self.custom_range {
            return Some((expand_bf16(lo), expand_bf16(hi)));
        }
        match self.domain {
            CbrtfDomain::Bounded | CbrtfDomain::AllFloats => None,
            CbrtfDomain::Unit => Some((0.0, 1.0)),
            CbrtfDomain::SignedUnit => Some((-1.0, 1.0)),
            CbrtfDomain::PositiveNormals => Some((
                if self.admits_subnormals() {
                    f32::from_bits(1)
                } else {
                    f32::MIN_POSITIVE
                },
                MAX_BOUNDED_INCLUSIVE,
            )),
        }
    }

    /// Whether this configuration is self-consistent.
    pub const fn is_valid(&self) -> bool {
        if self.is_full() {
            return self.custom_range.is_none() && matches!(self.subnormals, Subnormals::Reject);
        }
        if let Some((lo, hi)) = self.input_bounds() {
            if !self.admits_subnormals() && is_nonzero_subnormal_only_interval(lo, hi) {
                return false;
            }
            if matches!(self.subnormals, Subnormals::Normalize)
                && (lo >= f32::MIN_POSITIVE || hi <= -f32::MIN_POSITIVE || (lo == 0.0 && hi == 0.0))
            {
                return false;
            }
        }
        true
    }

    /// Assert that this configuration is self-consistent.
    #[doc(hidden)]
    pub const fn validate(&self) {
        if self.is_full() {
            assert!(
                self.custom_range.is_none() && matches!(self.subnormals, Subnormals::Reject),
                "CbrtfDomain::AllFloats / full_domain cannot be combined with custom input_range or subnormal overrides"
            );
            return;
        }
        if let Some((lo, hi)) = self.input_bounds() {
            assert!(
                !(!self.admits_subnormals() && is_nonzero_subnormal_only_interval(lo, hi)),
                "cbrtf input_range contains only nonzero subnormals; add .admit_subnormals()"
            );
            assert!(
                !(matches!(self.subnormals, Subnormals::Normalize)
                    && (lo >= f32::MIN_POSITIVE
                        || hi <= -f32::MIN_POSITIVE
                        || (lo == 0.0 && hi == 0.0))),
                "admit_subnormals requires input bounds to include nonzero subnormals around 0.0; use .flush_or_clamp_subnormals() on normal-only intervals"
            );
        }
    }

    const fn range_excludes_subnormals(&self) -> bool {
        match self.input_bounds() {
            Some((lo, hi)) => lo >= f32::MIN_POSITIVE || hi <= -f32::MIN_POSITIVE,
            None => false,
        }
    }

    /// Whether the declared range excludes the subnormal band around zero
    /// (or subnormals are admitted), eliding the per-lane subnormal rejection check.
    #[doc(hidden)]
    pub const fn elides_denorm_check(&self) -> bool {
        !self.denorm_possible()
    }

    pub(crate) const fn denorm_possible(&self) -> bool {
        !self.range_excludes_subnormals() && !self.admits_subnormals()
    }

    pub(crate) const fn scale_subnormals(&self) -> bool {
        !self.is_full()
            && matches!(self.subnormals, Subnormals::Normalize)
            && !self.range_excludes_subnormals()
    }

    pub(crate) const fn flushes_subnormals(&self) -> bool {
        !self.is_full() && matches!(self.subnormals, Subnormals::FlushOrClamp)
    }

    pub(crate) const fn admits_subnormals(&self) -> bool {
        self.is_full() || matches!(self.subnormals, Subnormals::Normalize)
    }

    /// Whether this configuration can never return an input error.
    pub const fn is_infallible(&self) -> bool {
        self.is_valid() && (self.assume_valid() || self.clamp_oob() || self.is_full())
    }

    /// Pack this configuration into a `u128` const-generic argument.
    pub const fn bits(&self) -> u128 {
        let dom: u128 = match self.domain {
            CbrtfDomain::Bounded => 0,
            CbrtfDomain::Unit => 1,
            CbrtfDomain::SignedUnit => 2,
            CbrtfDomain::PositiveNormals => 3,
            CbrtfDomain::AllFloats => 4,
        };
        let sub: u128 = match self.subnormals {
            Subnormals::Reject => 0,
            Subnormals::FlushOrClamp => 1,
            Subnormals::Normalize => 2,
        };
        let val: u128 = match self.validation {
            Validation::Checked => 0,
            Validation::ClampOutOfBounds => 1,
            Validation::AssumeValid => 2,
            Validation::ClampFinite => 3,
        };
        let prec: u128 = match self.precision {
            CbrtfPrecision::Midp => 0,
            CbrtfPrecision::Lowp => 1,
        };
        let port: u128 = match self.portability {
            Portability::BitExact => 0,
            Portability::BackendDependent => 1,
        };
        let exec: u128 = match self.execution {
            Execution::Upgrade => 0,
            Execution::Caller => 1,
        };
        let mut out = MAGIC
            | (dom << DOMAIN_SHIFT)
            | (sub << SUBNORM_SHIFT)
            | (val << VALIDATION_SHIFT)
            | (prec << PREC_SHIFT)
            | (port << PORTABILITY_SHIFT)
            | (exec << EXECUTION_SHIFT);
        if let Some((lo, hi)) = self.custom_range {
            out |= HAS_CUSTOM_RANGE | ((lo as u128) << IN_LO_SHIFT) | ((hi as u128) << IN_HI_SHIFT);
        }
        out
    }

    const fn from_bits_raw(bits: u128) -> Option<Self> {
        if (bits & 0xf) != MAGIC || (bits & RESERVED_LO_MASK) != 0 || (bits & RESERVED_HI_MASK) != 0
        {
            return None;
        }
        let domain = match (bits >> DOMAIN_SHIFT) & 0x1f {
            0 => CbrtfDomain::Bounded,
            1 => CbrtfDomain::Unit,
            2 => CbrtfDomain::SignedUnit,
            3 => CbrtfDomain::PositiveNormals,
            4 => CbrtfDomain::AllFloats,
            _ => return None,
        };
        let subnormals = match (bits >> SUBNORM_SHIFT) & 0x7 {
            0 => Subnormals::Reject,
            1 => Subnormals::FlushOrClamp,
            2 => Subnormals::Normalize,
            _ => return None,
        };
        let validation = match (bits >> VALIDATION_SHIFT) & 0x7 {
            0 => Validation::Checked,
            1 => Validation::ClampOutOfBounds,
            2 => Validation::AssumeValid,
            3 => Validation::ClampFinite,
            _ => return None,
        };
        let precision = match (bits >> PREC_SHIFT) & 0x7 {
            0 => CbrtfPrecision::Midp,
            1 => CbrtfPrecision::Lowp,
            _ => return None,
        };
        let portability = match (bits >> PORTABILITY_SHIFT) & 0x3 {
            0 => Portability::BitExact,
            1 => Portability::BackendDependent,
            _ => return None,
        };
        let execution = match (bits >> EXECUTION_SHIFT) & 0x3 {
            0 => Execution::Upgrade,
            1 => Execution::Caller,
            _ => return None,
        };
        let lo_u16 = ((bits >> IN_LO_SHIFT) & 0xffff) as u16;
        let hi_u16 = ((bits >> IN_HI_SHIFT) & 0xffff) as u16;
        let custom_range = if (bits & HAS_CUSTOM_RANGE) != 0 {
            let Some(lo) = decode_bf16_finite(lo_u16) else {
                return None;
            };
            let Some(hi) = decode_bf16_finite(hi_u16) else {
                return None;
            };
            if !(lo <= hi && abs(lo) <= MAX_CBRT_BF16 && abs(hi) <= MAX_CBRT_BF16) {
                return None;
            }
            Some((lo_u16, hi_u16))
        } else {
            if lo_u16 != 0 || hi_u16 != 0 {
                return None;
            }
            None
        };
        Some(Self {
            domain,
            custom_range,
            subnormals,
            precision,
            portability,
            validation,
            execution,
        })
    }

    /// Decode a `u128` option word back into [`CbrtfOptions`].
    pub const fn from_bits(bits: u128) -> Option<Self> {
        let Some(opts) = Self::from_bits_raw(bits) else {
            return None;
        };
        if opts.is_valid() { Some(opts) } else { None }
    }

    pub(crate) const fn decode_const(bits: u128) -> Self {
        let Some(opts) = Self::from_bits_raw(bits) else {
            panic!("invalid cbrtf const options bit pattern");
        };
        opts.validate();
        opts
    }

    /// Create a dynamic runtime evaluator for this [`CbrtfOptions`] value.
    #[inline(always)]
    pub const fn runtime(self) -> super::CbrtfRuntime {
        super::CbrtfRuntime::new(self)
    }

    pub(crate) const fn is_lowp(&self) -> bool {
        matches!(self.precision, CbrtfPrecision::Lowp)
    }
    pub(crate) const fn is_backend(&self) -> bool {
        matches!(self.portability, Portability::BackendDependent)
    }
    pub(crate) const fn clamp_oob(&self) -> bool {
        matches!(
            self.validation,
            Validation::ClampOutOfBounds | Validation::ClampFinite
        )
    }

    pub(crate) const fn preserve_clamped_nan(&self) -> bool {
        matches!(self.validation, Validation::ClampOutOfBounds)
    }
    pub(crate) const fn assume_valid(&self) -> bool {
        matches!(self.validation, Validation::AssumeValid)
    }
    pub(crate) const fn is_caller(&self) -> bool {
        matches!(self.execution, Execution::Caller)
    }
}

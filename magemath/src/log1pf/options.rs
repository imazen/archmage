//! Fluent `const` configuration builder, discrete [`Log1pfDomain`] enum,
//! [`Subnormals`] policy, and `bf16` custom range bounds for `log1p`.

use crate::{
    Execution, Portability, Subnormals, Validation,
    bf16::{MAX_BF16, ceil_bf16, decode_optional_bf16_range, expand_bf16, floor_bf16},
    logf::options::{
        LOWP_OCTAVE_HI_BITS, LOWP_OCTAVE_LO_BITS, MIDP_OCTAVE_HI_BITS, MIDP_OCTAVE_LO_BITS,
    },
};

// Uniform u128 layout:
//   [0..4]   = MAGIC (0xE)
//   [4..9]   = Log1pfDomain (0..=8)
//   [9..12]  = Subnormals (0..=2)
//   [12..15] = Validation (0..=2)
//   [15..18] = Log1pfPrecision (0..=1)
//   [18..20] = Portability (0..=1)
//   [20..22] = Execution (0..=1)
//   [22]     = HAS_CUSTOM_RANGE
//   [23]     = EXTRA_NEG_ONE
//   [24..64] = RESERVED_LO (40 bits, must be 0)
//   [64..80] = in_lo_bf16 (u16, 0 when !HAS_CUSTOM_RANGE)
//   [80..96] = in_hi_bf16 (u16, 0 when !HAS_CUSTOM_RANGE)
//   [96..128]= RESERVED_HI (32 bits, must be 0)
const MAGIC: u128 = 0xE;
const DOMAIN_SHIFT: u32 = 4;
const SUBNORM_SHIFT: u32 = 9;
const VALIDATION_SHIFT: u32 = 12;
const PREC_SHIFT: u32 = 15;
const PORTABILITY_SHIFT: u32 = 18;
const EXECUTION_SHIFT: u32 = 20;
const HAS_CUSTOM_RANGE: u128 = 1 << 22;
const EXTRA_NEG_ONE: u128 = 1 << 23;
const RESERVED_LO_MASK: u128 = ((1_u128 << 40) - 1) << 24;
const IN_LO_SHIFT: u32 = 64;
const IN_HI_SHIFT: u32 = 80;
const RESERVED_HI_MASK: u128 = 0xffff_ffff_u128 << 96;

const fn is_nonzero_subnormal_only_interval(lo: f32, hi: f32) -> bool {
    (lo > 0.0 && hi < f32::MIN_POSITIVE) || (lo > -f32::MIN_POSITIVE && hi < 0.0)
}

/// Discrete input domain for [`Log1pfOptions`] (`ln(1 + x)`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum Log1pfDomain {
    /// Strictly above `-1.0` (`(-1.0, f32::MAX]`, finite outputs only; `±0.0 -> ±0.0`).
    #[default]
    GreaterThanNegOne,
    /// Including `-1.0` (`[-1.0, f32::MAX]`; `-1.0 -> -inf`, `±0.0 -> ±0.0`).
    FromNegOne,
    /// Non-negative inputs `[0.0, f32::MAX]`.
    NonNegative,
    /// Unit interval `[0.0, 1.0]`.
    Unit,
    /// Open-below signed unit interval `(-1.0, 1.0]` (finite outputs only).
    OpenSignedUnit,
    /// Closed signed unit interval `[-1.0, 1.0]` (`-1.0 -> -inf`).
    SignedUnit,
    /// Small symmetric interval `[-0.25, 0.25]` with a direct polynomial in `x`
    /// for both `Midp` and `Lowp`.
    AroundZero,
    /// Fundamental single octave around `0.0` (`[-0.29289323, 0.41421342]` on `Midp`,
    /// `[-0.3333333, 0.3333333]` on `Lowp`).
    SingleOctave,
    /// Every `f32` input (`-1.0 -> -inf`, `< -1.0 -> NaN`, `±0.0 -> ±0.0`,
    /// subnormals preserved, `+inf -> +inf`, `NaN -> NaN`).
    AllFloats,
}

/// Approximation family for `log1p`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum Log1pfPrecision {
    /// Direct polynomial around zero, within 0.88 ULP in BitExact mode.
    /// Larger magnitudes use power-of-two range reduction.
    #[default]
    Midp,
    /// Shorter direct polynomial around zero. Larger magnitudes use
    /// the same polynomial after power-of-two range reduction.
    Lowp,
}

/// Fluent `const` configuration builder for [`super::Log1pf`] and [`crate::SimdMath`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Log1pfOptions {
    domain: Log1pfDomain,
    custom_range: Option<(u16, u16)>,
    extra_neg_one: bool,
    subnormals: Subnormals,
    precision: Log1pfPrecision,
    portability: Portability,
    validation: Validation,
    execution: Execution,
}

impl Default for Log1pfOptions {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Log1pfOptions {
    /// `Log1pfDomain::GreaterThanNegOne`, `Subnormals::FlushOrClamp`, `Midp`,
    /// `BitExact`, `Validation::Checked` (`error_on_out_of_bounds`), `Execution::Upgrade`.
    pub const fn new() -> Self {
        Self {
            domain: Log1pfDomain::GreaterThanNegOne,
            custom_range: None,
            extra_neg_one: false,
            subnormals: Subnormals::FlushOrClamp,
            precision: Log1pfPrecision::Midp,
            portability: Portability::BitExact,
            validation: Validation::Checked,
            execution: Execution::Upgrade,
        }
    }

    /// Select the discrete [`Log1pfDomain`], clearing any custom [`input_range`](Self::input_range).
    pub const fn domain(mut self, domain: Log1pfDomain) -> Self {
        self.domain = domain;
        self.custom_range = None;
        self.extra_neg_one = false;
        if matches!(domain, Log1pfDomain::AllFloats) {
            self.subnormals = Subnormals::Reject;
        }
        self
    }

    /// Override input bounds with outward-rounded `bf16` `[lo, hi]` (`-1.0 <= lo <= hi <= MAX_BF16`).
    pub const fn input_range(mut self, lo: f32, hi: f32) -> Self {
        assert!(
            lo >= -1.0 && lo <= hi && hi <= MAX_BF16,
            "log1p input range must satisfy -1.0 <= lo <= hi <= MAX_BF16 (0x7f7f_0000)"
        );
        self.custom_range = Some((floor_bf16(lo), ceil_bf16(hi)));
        self
    }

    /// Symmetric `[-0.25, 0.25]` domain (`Log1pfDomain::AroundZero`), eliding
    /// integer exponent extraction on both `Midp` and `Lowp`.
    pub const fn around_zero_range(self) -> Self {
        self.domain(Log1pfDomain::AroundZero)
    }

    /// Alias for [`around_zero_range`](Self::around_zero_range).
    pub const fn small_range(self) -> Self {
        self.around_zero_range()
    }

    /// Also accept `x == -1.0 -> -inf`.
    pub const fn admit_neg_one(mut self) -> Self {
        match self.domain {
            Log1pfDomain::GreaterThanNegOne => {
                if self.custom_range.is_some() {
                    self.extra_neg_one = true;
                } else {
                    self.domain = Log1pfDomain::FromNegOne;
                }
            }
            Log1pfDomain::OpenSignedUnit => {
                if self.custom_range.is_some() {
                    self.extra_neg_one = true;
                } else {
                    self.domain = Log1pfDomain::SignedUnit;
                }
            }
            Log1pfDomain::AllFloats | Log1pfDomain::FromNegOne | Log1pfDomain::SignedUnit => {}
            _ => {
                self.extra_neg_one = true;
            }
        }
        self
    }

    /// Alias for [`admit_neg_one`](Self::admit_neg_one) (`1 + (-1.0) == 0.0 -> -inf`).
    pub const fn admit_zero(self) -> Self {
        self.admit_neg_one()
    }

    /// Configure how nonzero subnormal inputs (`0 < |x| < f32::MIN_POSITIVE`) are handled.
    pub const fn subnormals(mut self, policy: Subnormals) -> Self {
        self.subnormals = policy;
        self
    }

    /// Treat nonzero subnormals (`0 < |x| < f32::MIN_POSITIVE`) as out-of-bounds
    /// (`Subnormals::Reject`).
    pub const fn reject_subnormals(self) -> Self {
        self.subnormals(Subnormals::Reject)
    }

    /// Shove nonzero subnormals (`0 < |x| < f32::MIN_POSITIVE`) out to `±0.0`
    /// (or the nearest normal range endpoint when `0.0` is excluded), even
    /// under [`Validation::Checked`].
    pub const fn flush_or_clamp_subnormals(self) -> Self {
        self.subnormals(Subnormals::FlushOrClamp)
    }

    /// Admit nonzero subnormals (`0 < |x| < f32::MIN_POSITIVE`), preserving
    /// `ln(1 + x) == x` (`Subnormals::Normalize`).
    pub const fn admit_subnormals(self) -> Self {
        self.subnormals(Subnormals::Normalize)
    }

    /// Accept every `f32` input (`Log1pfDomain::AllFloats`).
    pub const fn admit_all_floats(self) -> Self {
        self.domain(Log1pfDomain::AllFloats)
    }

    /// Alias for [`admit_all_floats`](Self::admit_all_floats).
    pub const fn full_domain(self) -> Self {
        self.admit_all_floats()
    }

    /// Select the approximation family (`Midp` or `Lowp`).
    pub const fn precision(mut self, value: Log1pfPrecision) -> Self {
        self.precision = value;
        self
    }

    /// Shorthand for `.precision(Log1pfPrecision::Lowp)`.
    pub const fn lowp(self) -> Self {
        self.precision(Log1pfPrecision::Lowp)
    }

    /// Shorthand for `.precision(Log1pfPrecision::Midp)`.
    pub const fn midp(self) -> Self {
        self.precision(Log1pfPrecision::Midp)
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

    /// Active discrete [`Log1pfDomain`].
    pub const fn selected_domain(&self) -> Log1pfDomain {
        self.domain
    }

    /// Active [`Subnormals`] policy (`Normalize` when `Log1pfDomain::AllFloats`).
    pub const fn subnormals_policy(&self) -> Subnormals {
        if self.is_specials() {
            Subnormals::Normalize
        } else {
            self.subnormals
        }
    }

    pub(crate) const fn is_specials(&self) -> bool {
        matches!(self.domain, Log1pfDomain::AllFloats)
    }

    pub(crate) const fn admits_neg_one(&self) -> bool {
        self.is_specials()
            || self.extra_neg_one
            || matches!(
                self.domain,
                Log1pfDomain::FromNegOne | Log1pfDomain::SignedUnit
            )
    }

    pub(crate) const fn admits_subnormals(&self) -> bool {
        self.is_specials() || matches!(self.subnormals, Subnormals::Normalize)
    }

    pub(crate) const fn flushes_subnormals(&self) -> bool {
        !self.is_specials() && matches!(self.subnormals, Subnormals::FlushOrClamp)
    }

    pub(crate) const fn input_bounds(&self) -> Option<(f32, f32)> {
        if self.is_specials() {
            return None;
        }
        if let Some((lo, hi)) = self.custom_range {
            return Some((expand_bf16(lo), expand_bf16(hi)));
        }
        match self.domain {
            Log1pfDomain::GreaterThanNegOne
            | Log1pfDomain::FromNegOne
            | Log1pfDomain::AllFloats => None,
            Log1pfDomain::NonNegative => Some((0.0, f32::MAX)),
            Log1pfDomain::Unit => Some((0.0, 1.0)),
            Log1pfDomain::OpenSignedUnit => Some((f32::from_bits(0xbf7f_ffff), 1.0)),
            Log1pfDomain::SignedUnit => Some((-1.0, 1.0)),
            Log1pfDomain::AroundZero => Some((-0.25, 0.25)),
            Log1pfDomain::SingleOctave => {
                if matches!(self.precision, Log1pfPrecision::Lowp) {
                    Some((
                        f32::from_bits(LOWP_OCTAVE_LO_BITS) - 1.0,
                        f32::from_bits(LOWP_OCTAVE_HI_BITS) - 1.0,
                    ))
                } else {
                    Some((
                        f32::from_bits(MIDP_OCTAVE_LO_BITS) - 1.0,
                        f32::from_bits(MIDP_OCTAVE_HI_BITS) - 1.0,
                    ))
                }
            }
        }
    }

    /// Whether this configuration is self-consistent.
    pub const fn is_valid(&self) -> bool {
        if self.is_specials() {
            return self.custom_range.is_none()
                && !self.extra_neg_one
                && matches!(self.subnormals, Subnormals::Reject);
        }
        if self.extra_neg_one
            && (matches!(
                self.domain,
                Log1pfDomain::FromNegOne | Log1pfDomain::SignedUnit
            ) || (self.custom_range.is_none()
                && matches!(
                    self.domain,
                    Log1pfDomain::GreaterThanNegOne | Log1pfDomain::OpenSignedUnit
                )))
        {
            return false;
        }
        if let Some((lo, hi)) = self.input_bounds() {
            if lo == -1.0 && !self.admits_neg_one() {
                return false;
            }
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
        if self.is_specials() {
            assert!(
                self.custom_range.is_none()
                    && !self.extra_neg_one
                    && matches!(self.subnormals, Subnormals::Reject),
                "Log1pfDomain::AllFloats / full_domain cannot be combined with custom input_range or subnormal overrides"
            );
            return;
        }
        assert!(
            !(self.extra_neg_one
                && (matches!(
                    self.domain,
                    Log1pfDomain::FromNegOne | Log1pfDomain::SignedUnit
                ) || (self.custom_range.is_none()
                    && matches!(
                        self.domain,
                        Log1pfDomain::GreaterThanNegOne | Log1pfDomain::OpenSignedUnit
                    )))),
            "redundant extra_neg_one flag"
        );
        if let Some((lo, hi)) = self.input_bounds() {
            assert!(
                lo > -1.0 || self.admits_neg_one(),
                "log1p input_range lower bound of -1.0 requires Log1pfDomain::FromNegOne, SignedUnit, or .admit_neg_one()"
            );
            assert!(
                !(!self.admits_subnormals() && is_nonzero_subnormal_only_interval(lo, hi)),
                "log1p input_range contains only nonzero subnormals; enable .admit_subnormals()"
            );
            assert!(
                !(matches!(self.subnormals, Subnormals::Normalize)
                    && (lo >= f32::MIN_POSITIVE
                        || hi <= -f32::MIN_POSITIVE
                        || (lo == 0.0 && hi == 0.0))),
                "admit_subnormals requires input range to include nonzero subnormals around 0.0; use .flush_or_clamp_subnormals() on normal-only intervals"
            );
        }
    }

    /// Whether `log1p(x)` keeps `1 + x` inside the fundamental single octave
    /// without `-1.0 -> -inf`. Restricted ranges around zero can use the
    /// polynomial directly.
    #[doc(hidden)]
    pub const fn elides_reduction(&self) -> bool {
        if self.is_specials() || self.admits_neg_one() {
            return false;
        }
        let Some((lo, hi)) = self.input_bounds() else {
            return false;
        };
        let u_lo = 1.0 + lo;
        let u_hi = 1.0 + hi;
        if matches!(self.precision, Log1pfPrecision::Lowp) {
            u_lo >= f32::from_bits(LOWP_OCTAVE_LO_BITS)
                && u_hi <= f32::from_bits(LOWP_OCTAVE_HI_BITS)
        } else {
            u_lo >= f32::from_bits(MIDP_OCTAVE_LO_BITS)
                && u_hi <= f32::from_bits(MIDP_OCTAVE_HI_BITS)
        }
    }

    /// Whether this configuration can never return an input error.
    pub const fn is_infallible(&self) -> bool {
        self.is_valid() && (self.assume_valid() || self.clamp_oob() || self.is_specials())
    }

    /// Pack this configuration into a `u128` const-generic argument.
    pub const fn bits(&self) -> u128 {
        let dom: u128 = match self.domain {
            Log1pfDomain::GreaterThanNegOne => 0,
            Log1pfDomain::FromNegOne => 1,
            Log1pfDomain::NonNegative => 2,
            Log1pfDomain::Unit => 3,
            Log1pfDomain::OpenSignedUnit => 4,
            Log1pfDomain::SignedUnit => 5,
            Log1pfDomain::AroundZero => 6,
            Log1pfDomain::SingleOctave => 7,
            Log1pfDomain::AllFloats => 8,
        };
        let prec: u128 = match self.precision {
            Log1pfPrecision::Midp => 0,
            Log1pfPrecision::Lowp => 1,
        };
        let mut out = MAGIC
            | (dom << DOMAIN_SHIFT)
            | (self.subnormals.encode() << SUBNORM_SHIFT)
            | (self.validation.encode() << VALIDATION_SHIFT)
            | (prec << PREC_SHIFT)
            | (self.portability.encode() << PORTABILITY_SHIFT)
            | (self.execution.encode() << EXECUTION_SHIFT);
        if self.extra_neg_one {
            out |= EXTRA_NEG_ONE;
        }
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
            0 => Log1pfDomain::GreaterThanNegOne,
            1 => Log1pfDomain::FromNegOne,
            2 => Log1pfDomain::NonNegative,
            3 => Log1pfDomain::Unit,
            4 => Log1pfDomain::OpenSignedUnit,
            5 => Log1pfDomain::SignedUnit,
            6 => Log1pfDomain::AroundZero,
            7 => Log1pfDomain::SingleOctave,
            8 => Log1pfDomain::AllFloats,
            _ => return None,
        };
        let Some(subnormals) = Subnormals::decode((bits >> SUBNORM_SHIFT) & 0x7) else {
            return None;
        };
        let Some(validation) = Validation::decode((bits >> VALIDATION_SHIFT) & 0x7) else {
            return None;
        };
        let precision = match (bits >> PREC_SHIFT) & 0x7 {
            0 => Log1pfPrecision::Midp,
            1 => Log1pfPrecision::Lowp,
            _ => return None,
        };
        let Some(portability) = Portability::decode((bits >> PORTABILITY_SHIFT) & 0x3) else {
            return None;
        };
        let Some(execution) = Execution::decode((bits >> EXECUTION_SHIFT) & 0x3) else {
            return None;
        };
        let lo_u16 = ((bits >> IN_LO_SHIFT) & 0xffff) as u16;
        let hi_u16 = ((bits >> IN_HI_SHIFT) & 0xffff) as u16;
        let Some(custom_range) = decode_optional_bf16_range(
            (bits & HAS_CUSTOM_RANGE) != 0,
            lo_u16,
            hi_u16,
            -1.0,
            MAX_BF16,
        ) else {
            return None;
        };
        Some(Self {
            domain,
            custom_range,
            extra_neg_one: (bits & EXTRA_NEG_ONE) != 0,
            subnormals,
            precision,
            portability,
            validation,
            execution,
        })
    }

    /// Decode a `u128` option word back into [`Log1pfOptions`].
    pub const fn from_bits(bits: u128) -> Option<Self> {
        let Some(opts) = Self::from_bits_raw(bits) else {
            return None;
        };
        if opts.is_valid() { Some(opts) } else { None }
    }

    pub(crate) const fn decode_const(bits: u128) -> Self {
        let Some(opts) = Self::from_bits_raw(bits) else {
            panic!("invalid log1pf const options bit pattern");
        };
        opts.validate();
        opts
    }

    /// Create a dynamic runtime evaluator for this [`Log1pfOptions`] value.
    #[inline(always)]
    pub const fn runtime(self) -> super::Log1pfRuntime {
        super::Log1pfRuntime::new(self)
    }

    pub(crate) const fn is_lowp(&self) -> bool {
        matches!(self.precision, Log1pfPrecision::Lowp)
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

    pub(crate) const fn to_kernel_cfg(self) -> crate::logf::graph::KernelCfg {
        let (in_lo, in_hi) = match self.input_bounds() {
            Some(bounds) => bounds,
            None => (0.0, 0.0),
        };
        crate::logf::graph::KernelCfg {
            ranged: self.input_bounds().is_some(),
            in_lo,
            in_hi,
            zero: self.admits_neg_one(),
            subnorm: self.admits_subnormals(),
            flush_sub: self.flushes_subnormals(),
            specials: self.is_specials(),
            clamp_oob: self.clamp_oob(),
            preserve_clamped_nan: self.preserve_clamped_nan(),
            lowp: self.is_lowp(),
            assume: self.assume_valid(),
            elide_red: false,
            elide_red_log1p: self.elides_reduction(),
        }
    }
}

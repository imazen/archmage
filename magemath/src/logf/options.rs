//! Fluent `const` configuration builder, discrete [`LogfDomain`] enum,
//! [`Subnormals`] policy, and `bf16` custom range bounds for `log2`, `ln`, and `log10`.

use crate::{
    Execution, Portability, Subnormals, Validation,
    bf16::{MAX_BF16, ceil_bf16, decode_optional_bf16_range, expand_bf16, floor_bf16},
};

// Uniform u128 layout:
//   [0..4]   = MAGIC (0xC)
//   [4..9]   = LogfDomain (0..=7)
//   [9..12]  = Subnormals (0..=2)
//   [12..15] = Validation (0..=2)
//   [15..18] = LogfPrecision (0..=1)
//   [18..20] = Portability (0..=1)
//   [20..22] = Execution (0..=1)
//   [22]     = HAS_CUSTOM_RANGE
//   [23]     = EXTRA_ZERO
//   [24..64] = RESERVED_LO (40 bits, must be 0)
//   [64..80] = in_lo_bf16 (u16, 0 when !HAS_CUSTOM_RANGE)
//   [80..96] = in_hi_bf16 (u16, 0 when !HAS_CUSTOM_RANGE)
//   [96..128]= RESERVED_HI (32 bits, must be 0)
const MAGIC: u128 = 0xC;
const DOMAIN_SHIFT: u32 = 4;
const SUBNORM_SHIFT: u32 = 9;
const VALIDATION_SHIFT: u32 = 12;
const PREC_SHIFT: u32 = 15;
const PORTABILITY_SHIFT: u32 = 18;
const EXECUTION_SHIFT: u32 = 20;
const HAS_CUSTOM_RANGE: u128 = 1 << 22;
const EXTRA_ZERO: u128 = 1 << 23;
const RESERVED_LO_MASK: u128 = ((1_u128 << 40) - 1) << 24;
const IN_LO_SHIFT: u32 = 64;
const IN_HI_SHIFT: u32 = 80;
const RESERVED_HI_MASK: u128 = 0xffff_ffff_u128 << 96;

/// Fundamental single-octave lower bound for `Midp` (`sqrt(0.5)` rounded up, `0.70710677`).
pub(crate) const MIDP_OCTAVE_LO_BITS: u32 = 0x3f35_04f3;
/// Fundamental single-octave upper bound for `Midp` (`sqrt(2.0)` minus 1 ULP, `1.4142134`).
pub(crate) const MIDP_OCTAVE_HI_BITS: u32 = 0x3fb5_04f2;
/// Fundamental single-octave lower bound for `Lowp` (`2/3` rounded up).
pub(crate) const LOWP_OCTAVE_LO_BITS: u32 = 0x3f2a_aaab;
/// Fundamental single-octave upper bound for `Lowp` (`4/3` minus 1 ULP).
pub(crate) const LOWP_OCTAVE_HI_BITS: u32 = 0x3faa_aaaa;

/// Discrete input domain for [`LogfOptions`] (`log2`, `ln`, `log10`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum LogfDomain {
    /// Positive normal inputs `[f32::MIN_POSITIVE, f32::MAX]`.
    #[default]
    PositiveNormals,
    /// Non-negative inputs `{±0.0} ∪ [f32::MIN_POSITIVE, f32::MAX]` (`±0.0 -> -inf`).
    NonNegative,
    /// SDR unit domain `{±0.0} ∪ [f32::MIN_POSITIVE, 1.0]` (`±0.0 -> -inf`).
    Unit,
    /// Strictly positive unit domain `[f32::MIN_POSITIVE, 1.0]` (finite outputs only).
    PositiveUnit,
    /// HDR luminance domain `{±0.0} ∪ [2^-16, 2^16]` (`±0.0 -> -inf`).
    Hdr,
    /// Strictly positive HDR domain `[2^-16, 2^16]` (finite outputs only).
    PositiveHdr,
    /// Fundamental single octave around `1.0` (`[0.70710677, 1.4142134]` on `Midp`,
    /// `[0.6666667, 1.3333333]` on `Lowp`), eliding integer exponent extraction (`n = 0`).
    SingleOctave,
    /// Every `f32` input (`±0 -> -inf`, subnormals normalized, `< 0 -> NaN`,
    /// `+inf -> +inf`, `NaN -> NaN`).
    AllFloats,
}

/// Approximation family, not a global mathematical error bound.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum LogfPrecision {
    /// Refitted `Midp` polynomial (`<= 4.5 ULP` on positive normals and
    /// normalized subnormals; `<= 2.9 ULP` on [`LogfDomain::SingleOctave`]).
    #[default]
    Midp,
    /// Rational `P2/Q2` `Lowp` approximation (`~6e-6` absolute error).
    Lowp,
}

/// Fluent `const` configuration builder for [`super::Logf`] and [`crate::SimdMath`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LogfOptions {
    domain: LogfDomain,
    custom_range: Option<(u16, u16)>,
    extra_zero: bool,
    subnormals: Subnormals,
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
    /// `LogfDomain::PositiveNormals`, `Subnormals::FlushOrClamp`, `Midp`, `BitExact`,
    /// `Validation::Checked` (`error_on_out_of_bounds`), `Execution::Upgrade`.
    pub const fn new() -> Self {
        Self {
            domain: LogfDomain::PositiveNormals,
            custom_range: None,
            extra_zero: false,
            subnormals: Subnormals::FlushOrClamp,
            precision: LogfPrecision::Midp,
            portability: Portability::BitExact,
            validation: Validation::Checked,
            execution: Execution::Upgrade,
        }
    }

    /// Select the discrete [`LogfDomain`], clearing any custom [`input_range`](Self::input_range).
    pub const fn domain(mut self, domain: LogfDomain) -> Self {
        self.domain = domain;
        self.custom_range = None;
        self.extra_zero = false;
        if matches!(domain, LogfDomain::AllFloats) {
            self.subnormals = Subnormals::Reject;
        }
        self
    }

    /// Override input bounds with outward-rounded `bf16` `[lo, hi]` (`0.0 <= lo <= hi <= MAX_BF16`).
    pub const fn input_range(mut self, lo: f32, hi: f32) -> Self {
        assert!(
            lo >= 0.0 && lo <= hi && hi <= MAX_BF16,
            "log input range must be non-negative finite within MAX_BF16 (0x7f7f_0000) with lo <= hi"
        );
        self.custom_range = Some((floor_bf16(lo), ceil_bf16(hi)));
        self
    }

    /// Single-octave domain around `1.0` (`LogfDomain::SingleOctave`), eliding
    /// integer exponent extraction (`<= 2.9 ULP` on `Midp`).
    pub const fn near_one_range(self) -> Self {
        self.domain(LogfDomain::SingleOctave)
    }

    /// Configure how nonzero positive subnormal inputs (`0 < x < f32::MIN_POSITIVE`) are handled.
    pub const fn subnormals(mut self, policy: Subnormals) -> Self {
        self.subnormals = policy;
        self
    }

    /// Treat positive subnormals as out-of-bounds (`Subnormals::Reject`).
    pub const fn reject_subnormals(self) -> Self {
        self.subnormals(Subnormals::Reject)
    }

    /// Shove positive subnormals out to `0.0` (`-> -inf`, when zero is admitted)
    /// or the lower positive normal bound (when zero is excluded), even under
    /// [`Validation::Checked`].
    pub const fn flush_or_clamp_subnormals(self) -> Self {
        self.subnormals(Subnormals::FlushOrClamp)
    }

    /// Also accept positive subnormals (`0 < x < f32::MIN_POSITIVE`), normalizing
    /// them accurately (`* 2^23` with `-23` exponent adjustment) before reduction.
    pub const fn admit_subnormals(self) -> Self {
        self.subnormals(Subnormals::Normalize)
    }

    /// Also accept `±0.0 -> -inf` without paying subnormal pre-normalization cost.
    pub const fn admit_zero(mut self) -> Self {
        match self.domain {
            LogfDomain::PositiveNormals => self.domain = LogfDomain::NonNegative,
            LogfDomain::PositiveUnit => self.domain = LogfDomain::Unit,
            LogfDomain::PositiveHdr => self.domain = LogfDomain::Hdr,
            LogfDomain::SingleOctave => self.extra_zero = true,
            _ => {}
        }
        self
    }

    /// Accept every `f32` input (`LogfDomain::AllFloats`).
    pub const fn admit_all_floats(self) -> Self {
        self.domain(LogfDomain::AllFloats)
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

    /// Active discrete [`LogfDomain`].
    pub const fn selected_domain(&self) -> LogfDomain {
        self.domain
    }

    /// Active [`Subnormals`] policy (`Normalize` when `LogfDomain::AllFloats`).
    pub const fn subnormals_policy(&self) -> Subnormals {
        if self.is_specials() {
            Subnormals::Normalize
        } else {
            self.subnormals
        }
    }

    pub(crate) const fn is_specials(&self) -> bool {
        matches!(self.domain, LogfDomain::AllFloats)
    }

    pub(crate) const fn admits_zero(&self) -> bool {
        self.is_specials()
            || self.extra_zero
            || matches!(
                self.domain,
                LogfDomain::NonNegative | LogfDomain::Unit | LogfDomain::Hdr
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
            LogfDomain::PositiveNormals | LogfDomain::NonNegative | LogfDomain::AllFloats => None,
            LogfDomain::Unit | LogfDomain::PositiveUnit => Some((
                if self.admits_subnormals() {
                    f32::from_bits(1)
                } else {
                    f32::MIN_POSITIVE
                },
                1.0,
            )),
            LogfDomain::Hdr | LogfDomain::PositiveHdr => Some((1.0 / 65_536.0, 65_536.0)),
            LogfDomain::SingleOctave => {
                if matches!(self.precision, LogfPrecision::Lowp) {
                    Some((
                        f32::from_bits(LOWP_OCTAVE_LO_BITS),
                        f32::from_bits(LOWP_OCTAVE_HI_BITS),
                    ))
                } else {
                    Some((
                        f32::from_bits(MIDP_OCTAVE_LO_BITS),
                        f32::from_bits(MIDP_OCTAVE_HI_BITS),
                    ))
                }
            }
        }
    }

    /// Whether this configuration is self-consistent.
    pub const fn is_valid(&self) -> bool {
        if self.is_specials() {
            return self.custom_range.is_none()
                && !self.extra_zero
                && matches!(self.subnormals, Subnormals::Reject);
        }
        if self.extra_zero && !matches!(self.domain, LogfDomain::SingleOctave) {
            return false;
        }
        if let Some((lo, hi)) = self.input_bounds() {
            if lo == 0.0 {
                if !self.admits_zero() {
                    return false;
                }
                if hi > 0.0 && matches!(self.subnormals, Subnormals::Reject) {
                    return false;
                }
                if hi == 0.0 && matches!(self.subnormals, Subnormals::Normalize) {
                    return false;
                }
            } else if lo < f32::MIN_POSITIVE {
                if matches!(self.subnormals, Subnormals::Reject) {
                    return false;
                }
                if hi < f32::MIN_POSITIVE && !self.admits_subnormals() {
                    return false;
                }
            }
            if matches!(self.subnormals, Subnormals::Normalize) && lo >= f32::MIN_POSITIVE {
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
                    && !self.extra_zero
                    && matches!(self.subnormals, Subnormals::Reject),
                "LogfDomain::AllFloats / full_domain cannot be combined with custom input_range or subnormal overrides"
            );
            return;
        }
        assert!(
            !(self.extra_zero && !matches!(self.domain, LogfDomain::SingleOctave)),
            "internal extra_zero flag only valid with LogfDomain::SingleOctave"
        );
        if let Some((lo, hi)) = self.input_bounds() {
            if lo == 0.0 {
                assert!(
                    self.admits_zero(),
                    "log input_range starting at 0.0 requires .admit_zero() (or LogfDomain::Unit / NonNegative)"
                );
                assert!(
                    hi == 0.0 || !matches!(self.subnormals, Subnormals::Reject),
                    "log input_range starting at 0.0 with hi > 0 requires .admit_subnormals() or .flush_or_clamp_subnormals() (or use f32::MIN_POSITIVE / LogfDomain::Unit)"
                );
                assert!(
                    !(hi == 0.0 && matches!(self.subnormals, Subnormals::Normalize)),
                    "log input_range [0.0, 0.0] contains no nonzero subnormals"
                );
            } else {
                assert!(
                    lo >= f32::MIN_POSITIVE || !matches!(self.subnormals, Subnormals::Reject),
                    "log input_range lower bound below f32::MIN_POSITIVE requires .admit_subnormals() or .flush_or_clamp_subnormals()"
                );
                assert!(
                    !(lo < f32::MIN_POSITIVE
                        && hi < f32::MIN_POSITIVE
                        && !self.admits_subnormals()),
                    "log input_range containing only nonzero subnormals requires .admit_subnormals()"
                );
            }
            assert!(
                !(matches!(self.subnormals, Subnormals::Normalize) && lo >= f32::MIN_POSITIVE),
                "admit_subnormals requires input lower bound < f32::MIN_POSITIVE (or LogfDomain::Unit / PositiveNormals); use .flush_or_clamp_subnormals() on normal-only domains"
            );
        }
    }

    /// Whether `log2`/`ln`/`log10` stays inside the fundamental single octave
    /// without admitted zero or subnormal rescaling, eliding integer exponent
    /// extraction (`n = 0`).
    #[doc(hidden)]
    pub const fn elides_reduction(&self) -> bool {
        if self.is_specials() || self.admits_zero() || self.admits_subnormals() {
            return false;
        }
        let Some((lo, hi)) = self.input_bounds() else {
            return false;
        };
        if matches!(self.precision, LogfPrecision::Lowp) {
            lo >= f32::from_bits(LOWP_OCTAVE_LO_BITS) && hi <= f32::from_bits(LOWP_OCTAVE_HI_BITS)
        } else {
            lo >= f32::from_bits(MIDP_OCTAVE_LO_BITS) && hi <= f32::from_bits(MIDP_OCTAVE_HI_BITS)
        }
    }

    /// Whether this configuration can never return an input error.
    pub const fn is_infallible(&self) -> bool {
        self.is_valid() && (self.assume_valid() || self.clamp_oob() || self.is_specials())
    }

    /// Pack this configuration into a `u128` const-generic argument.
    pub const fn bits(&self) -> u128 {
        let dom: u128 = match self.domain {
            LogfDomain::PositiveNormals => 0,
            LogfDomain::NonNegative => 1,
            LogfDomain::Unit => 2,
            LogfDomain::PositiveUnit => 3,
            LogfDomain::Hdr => 4,
            LogfDomain::PositiveHdr => 5,
            LogfDomain::SingleOctave => 6,
            LogfDomain::AllFloats => 7,
        };
        let prec: u128 = match self.precision {
            LogfPrecision::Midp => 0,
            LogfPrecision::Lowp => 1,
        };
        let mut out = MAGIC
            | (dom << DOMAIN_SHIFT)
            | (self.subnormals.encode() << SUBNORM_SHIFT)
            | (self.validation.encode() << VALIDATION_SHIFT)
            | (prec << PREC_SHIFT)
            | (self.portability.encode() << PORTABILITY_SHIFT)
            | (self.execution.encode() << EXECUTION_SHIFT);
        if self.extra_zero {
            out |= EXTRA_ZERO;
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
            0 => LogfDomain::PositiveNormals,
            1 => LogfDomain::NonNegative,
            2 => LogfDomain::Unit,
            3 => LogfDomain::PositiveUnit,
            4 => LogfDomain::Hdr,
            5 => LogfDomain::PositiveHdr,
            6 => LogfDomain::SingleOctave,
            7 => LogfDomain::AllFloats,
            _ => return None,
        };
        let Some(subnormals) = Subnormals::decode((bits >> SUBNORM_SHIFT) & 0x7) else {
            return None;
        };
        let Some(validation) = Validation::decode((bits >> VALIDATION_SHIFT) & 0x7) else {
            return None;
        };
        let precision = match (bits >> PREC_SHIFT) & 0x7 {
            0 => LogfPrecision::Midp,
            1 => LogfPrecision::Lowp,
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
            0.0,
            MAX_BF16,
        ) else {
            return None;
        };
        Some(Self {
            domain,
            custom_range,
            extra_zero: (bits & EXTRA_ZERO) != 0,
            subnormals,
            precision,
            portability,
            validation,
            execution,
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

    pub(crate) const fn is_lowp(&self) -> bool {
        matches!(self.precision, LogfPrecision::Lowp)
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

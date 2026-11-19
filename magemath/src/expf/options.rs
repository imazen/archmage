//! Fluent `const` configuration builder, discrete [`ExpfDomain`] enum, and
//! `bf16` custom range bounds for the exponential family (`exp2`, `exp`).

use crate::{
    Execution, Portability, Validation,
    bf16::{MAX_BF16, ceil_bf16, decode_bf16_finite, expand_bf16, floor_bf16},
};

// Uniform u128 layout:
//   [0..4]   = MAGIC (0xB)
//   [4..9]   = ExpfDomain (0..=6)
//   [9..12]  = Validation (0..=2)
//   [12..15] = ExpfPrecision (0..=1)
//   [15..17] = Portability (0..=1)
//   [17..19] = Execution (0..=1)
//   [19]     = HAS_CUSTOM_RANGE
//   [20..64] = RESERVED_LO (44 bits, must be 0)
//   [64..80] = in_lo_bf16 (u16, 0 when !HAS_CUSTOM_RANGE)
//   [80..96] = in_hi_bf16 (u16, 0 when !HAS_CUSTOM_RANGE)
//   [96..128]= RESERVED_HI (32 bits, must be 0)
const MAGIC: u128 = 0xB;
const DOMAIN_SHIFT: u32 = 4;
const VALIDATION_SHIFT: u32 = 9;
const PREC_SHIFT: u32 = 12;
const PORTABILITY_SHIFT: u32 = 15;
const EXECUTION_SHIFT: u32 = 17;
const HAS_CUSTOM_RANGE: u128 = 1 << 19;
const RESERVED_LO_MASK: u128 = ((1_u128 << 44) - 1) << 20;
const IN_LO_SHIFT: u32 = 64;
const IN_HI_SHIFT: u32 = 80;
const RESERVED_HI_MASK: u128 = 0xffff_ffff_u128 << 96;

const fn abs(x: f32) -> f32 {
    if x < 0.0 { -x } else { x }
}

/// Discrete input domain for [`ExpfOptions`] (`exp2` and `exp`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExpfDomain {
    /// All finite `f32` inputs `[-f32::MAX, f32::MAX]` (including `±0.0` and subnormals).
    #[default]
    Finite,
    /// Normal-output domain (`[-126.0, 127.0]` for `exp2`, `[-87.0, 88.0]` for `exp`),
    /// eliding both underflow and overflow output clamps.
    NormalOutput,
    /// Non-positive finite inputs `[-f32::MAX, 0.0]` (Gaussian weights, softmax `x - max`,
    /// decay kernels), eliding the `+inf` overflow clamp.
    NonPositive,
    /// Signed unit interval `[-1.0, 1.0]`, eliding underflow and overflow clamps.
    SignedUnit,
    /// Symmetric single-octave domain (`[-0.5, 0.5]` for `exp2`,
    /// `[-0.346_573_56, 0.346_573_56]` for `exp`), eliding both output clamps
    /// and integer `2^k` reduction on `Midp`.
    SingleOctave,
    /// Positive single-octave domain (`[0.0, 0.999_999_94]` for `exp2`,
    /// `[0.0, 0.693_147_1]` for `exp`), eliding both output clamps and integer
    /// `2^k` reduction on `Lowp`.
    PositiveOctave,
    /// Every `f32` input (`NaN -> NaN`, `+inf -> +inf`, `-inf -> +0.0`).
    AllFloats,
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
/// Options separate **domain & range** ([`domain`](Self::domain),
/// [`input_range`](Self::input_range)) from **out-of-bounds validation**
/// ([`error_on_out_of_bounds`](Self::error_on_out_of_bounds),
/// [`clamp_out_of_bounds`](Self::clamp_out_of_bounds),
/// [`assume_in_bounds`](Self::assume_in_bounds)).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpfOptions {
    domain: ExpfDomain,
    custom_range: Option<(u16, u16)>,
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
    /// `ExpfDomain::Finite`, `Midp`, `BitExact`, `Checked`
    /// (`error_on_out_of_bounds`), `Execution::Upgrade`.
    pub const fn new() -> Self {
        Self {
            domain: ExpfDomain::Finite,
            custom_range: None,
            precision: ExpfPrecision::Midp,
            portability: Portability::BitExact,
            validation: Validation::Checked,
            execution: Execution::Upgrade,
        }
    }

    /// Select the discrete [`ExpfDomain`], clearing any custom [`input_range`](Self::input_range).
    pub const fn domain(mut self, domain: ExpfDomain) -> Self {
        self.domain = domain;
        self.custom_range = None;
        self
    }

    /// Override input bounds with outward-rounded `bf16` `[lo, hi]` (`|bound| <= MAX_BF16`, `lo <= hi`).
    pub const fn input_range(mut self, lo: f32, hi: f32) -> Self {
        assert!(
            lo <= hi && abs(lo) <= MAX_BF16 && abs(hi) <= MAX_BF16,
            "input range must be finite within ±MAX_BF16 (0x7f7f_0000) with lo <= hi"
        );
        self.custom_range = Some((floor_bf16(lo), ceil_bf16(hi)));
        self
    }

    /// Single-octave `exp2` / `exp` domain (`ExpfDomain::SingleOctave`), eliding
    /// both output clamping and integer `2^k` reduction on `Midp` (`<= 0.78 ULP`).
    pub const fn small_exp2_range(self) -> Self {
        self.domain(ExpfDomain::SingleOctave)
    }

    /// Single-octave `exp` / `exp2` domain (`ExpfDomain::SingleOctave`), eliding
    /// both output clamping and integer `2^k` reduction on `Midp`.
    pub const fn small_exp_range(self) -> Self {
        self.domain(ExpfDomain::SingleOctave)
    }

    /// Normal-output `exp2` / `exp` domain (`ExpfDomain::NormalOutput`), eliding
    /// overflow and underflow clamping on both `Midp` and `Lowp`.
    pub const fn unclamped_exp2_range(self) -> Self {
        self.domain(ExpfDomain::NormalOutput)
    }

    /// Normal-output `exp` / `exp2` domain (`ExpfDomain::NormalOutput`), eliding
    /// overflow and underflow clamping on both `Midp` and `Lowp`.
    pub const fn unclamped_exp_range(self) -> Self {
        self.domain(ExpfDomain::NormalOutput)
    }

    /// Accept every `f32` input (`ExpfDomain::AllFloats`: `NaN -> NaN`, `+inf -> +inf`, `-inf -> +0`).
    pub const fn admit_all_floats(self) -> Self {
        self.domain(ExpfDomain::AllFloats)
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

    /// Clamp inputs and outputs as applicable. NaN inputs have unspecified
    /// numerical results (`Validation::ClampFinite`, infallible).
    pub const fn clamp_finite(self) -> Self {
        self.validation(Validation::ClampFinite)
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

    /// Active discrete [`ExpfDomain`].
    pub const fn selected_domain(&self) -> ExpfDomain {
        self.domain
    }

    /// Whether this configuration is self-consistent.
    pub const fn is_valid(&self) -> bool {
        !(self.is_full() && self.custom_range.is_some())
    }

    /// Assert that this configuration is self-consistent.
    pub const fn validate(&self) {
        assert!(
            !(self.is_full() && self.custom_range.is_some()),
            "ExpfDomain::AllFloats / full_domain cannot be combined with input_range"
        );
    }

    pub(crate) const fn exp2_bounds(&self) -> Option<(f32, f32)> {
        if self.is_full() {
            return None;
        }
        if let Some((lo, hi)) = self.custom_range {
            return Some((expand_bf16(lo), expand_bf16(hi)));
        }
        match self.domain {
            ExpfDomain::Finite | ExpfDomain::AllFloats => None,
            ExpfDomain::NormalOutput => Some((-126.0, 127.0)),
            ExpfDomain::NonPositive => Some((-f32::MAX, 0.0)),
            ExpfDomain::SignedUnit => Some((-1.0, 1.0)),
            ExpfDomain::SingleOctave => Some((-0.5, 0.5)),
            ExpfDomain::PositiveOctave => Some((0.0, 0.999_999_94)),
        }
    }

    pub(crate) const fn exp_bounds(&self) -> Option<(f32, f32)> {
        if self.is_full() {
            return None;
        }
        if let Some((lo, hi)) = self.custom_range {
            return Some((expand_bf16(lo), expand_bf16(hi)));
        }
        match self.domain {
            ExpfDomain::Finite | ExpfDomain::AllFloats => None,
            ExpfDomain::NormalOutput => Some((-87.0, 88.0)),
            ExpfDomain::NonPositive => Some((-f32::MAX, 0.0)),
            ExpfDomain::SignedUnit => Some((-1.0, 1.0)),
            ExpfDomain::SingleOctave => Some((-0.346_573_56, 0.346_573_56)),
            // 0x3f31_7217 = 0.693_147_1 (1 ULP below LN_2 so x * LOG2_E < 1.0).
            ExpfDomain::PositiveOctave => Some((0.0, f32::from_bits(0x3f31_7217))),
        }
    }

    /// Whether `exp2(x)` keeps `x` in the un-clamped range (`[-126, 127.5)`
    /// `Midp` or `[-126, 127.99]` `Lowp`), allowing clamp elision.
    pub const fn elides_clamp(&self) -> bool {
        let Some((lo, hi)) = self.exp2_bounds() else {
            return false;
        };
        if matches!(self.precision, ExpfPrecision::Lowp) {
            lo >= -126.0 && hi <= 127.99
        } else {
            lo >= -126.0 && hi < 127.5
        }
    }

    /// Whether `exp(x)` keeps `x * log2(e)` in the un-clamped range (`[-126, 127.5)`
    /// `Midp` or `[-126, 127.99]` `Lowp`), allowing clamp elision.
    pub const fn elides_exp_clamp(&self) -> bool {
        let Some((lo, hi)) = self.exp_bounds() else {
            return false;
        };
        let s_lo = lo * core::f32::consts::LOG2_E;
        let s_hi = hi * core::f32::consts::LOG2_E;
        if matches!(self.precision, ExpfPrecision::Lowp) {
            s_lo >= -126.0 && s_hi <= 127.99
        } else {
            s_lo >= -126.0 && s_hi < 127.5
        }
    }

    pub(crate) const fn elides_exp2_overflow(&self) -> bool {
        let Some((_, hi)) = self.exp2_bounds() else {
            return false;
        };
        if matches!(self.precision, ExpfPrecision::Lowp) {
            hi <= 127.99
        } else {
            hi < 127.5
        }
    }

    pub(crate) const fn elides_exp_overflow(&self) -> bool {
        let Some((_, hi)) = self.exp_bounds() else {
            return false;
        };
        let s_hi = hi * core::f32::consts::LOG2_E;
        if matches!(self.precision, ExpfPrecision::Lowp) {
            s_hi <= 127.99
        } else {
            s_hi < 127.5
        }
    }

    /// Whether `exp2(x)` stays in the fundamental single octave (`[-0.5, 0.5]`
    /// on `Midp` or `[0.0, 1.0)` on `Lowp`), eliding integer `2^k` reduction.
    pub const fn elides_reduction(&self) -> bool {
        let Some((lo, hi)) = self.exp2_bounds() else {
            return false;
        };
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
        let Some((lo, hi)) = self.exp_bounds() else {
            return false;
        };
        let s_lo = lo * core::f32::consts::LOG2_E;
        let s_hi = hi * core::f32::consts::LOG2_E;
        if matches!(self.precision, ExpfPrecision::Lowp) {
            s_lo >= 0.0 && s_hi < 1.0
        } else {
            s_lo >= -0.5 && s_hi <= 0.5
        }
    }

    /// Whether this configuration can never return an input error.
    pub const fn is_infallible(&self) -> bool {
        self.is_valid() && (self.assume_valid() || self.clamp_oob() || self.is_full())
    }

    /// Pack this configuration into a `u128` const-generic argument.
    pub const fn bits(&self) -> u128 {
        let dom: u128 = match self.domain {
            ExpfDomain::Finite => 0,
            ExpfDomain::NormalOutput => 1,
            ExpfDomain::NonPositive => 2,
            ExpfDomain::SignedUnit => 3,
            ExpfDomain::SingleOctave => 4,
            ExpfDomain::PositiveOctave => 5,
            ExpfDomain::AllFloats => 6,
        };
        let val: u128 = match self.validation {
            Validation::Checked => 0,
            Validation::ClampOutOfBounds => 1,
            Validation::AssumeValid => 2,
            Validation::ClampFinite => 3,
        };
        let prec: u128 = match self.precision {
            ExpfPrecision::Midp => 0,
            ExpfPrecision::Lowp => 1,
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
            0 => ExpfDomain::Finite,
            1 => ExpfDomain::NormalOutput,
            2 => ExpfDomain::NonPositive,
            3 => ExpfDomain::SignedUnit,
            4 => ExpfDomain::SingleOctave,
            5 => ExpfDomain::PositiveOctave,
            6 => ExpfDomain::AllFloats,
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
            0 => ExpfPrecision::Midp,
            1 => ExpfPrecision::Lowp,
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
            if !(lo <= hi && abs(lo) <= MAX_BF16 && abs(hi) <= MAX_BF16) {
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
            precision,
            portability,
            validation,
            execution,
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

    pub(crate) const fn is_full(&self) -> bool {
        matches!(self.domain, ExpfDomain::AllFloats)
    }
    pub(crate) const fn is_lowp(&self) -> bool {
        matches!(self.precision, ExpfPrecision::Lowp)
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

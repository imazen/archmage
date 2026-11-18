//! Fluent `const` configuration builder, discrete domain enum, `bf16` range
//! bounds, and compile-time `const_exponent` specialization for `powf`.

use crate::{
    Execution, Portability, Subnormals, Validation,
    bf16::{MAX_BF16, ceil_bf16, decode_bf16_finite, expand_bf16, floor_bf16},
};

/// Largest `|n * log2(x)|` a configuration may reach before the kernel keeps
/// its range check (`[-125.0, 125.0]` stays strictly inside `Midp`/`Lowp`'s
/// `[-126, 127.5)` domain and `Highp`'s `|y * log2(x)| < 126` fast-pack domain).
const ELIDE_LIMIT: f32 = 125.0;

/// Largest finite magnitude representable in custom `bf16` exponent/base bounds (`0x7f7f_0000`).
pub const MAX_EXP_BOUND: f32 = MAX_BF16;

// Uniform u128 layout:
//   [0..4]   = MAGIC (0xA)
//   [4..9]   = PowfDomain (0..=9)
//   [9..12]  = Subnormals (0..=2)
//   [12..15] = Validation (0..=2)
//   [15..18] = PowfPrecision (0..=2)
//   [18..20] = Portability (0..=1)
//   [20..22] = Execution (0..=1)
//   [22]     = HAS_CUSTOM_BASE
//   [23..25] = EXP_MODE (0 = DomainDefault, 1 = RangeBf16, 2 = ConstF32)
//   [25]     = SATURATE_OUTPUT
//   [26..64] = RESERVED (38 bits, must be 0)
//   [64..80] = base_lo_bf16 (u16, 0 when !HAS_CUSTOM_BASE)
//   [80..96] = base_hi_bf16 (u16, 0 when !HAS_CUSTOM_BASE)
//   [96..128]= exp slot: (exp_lo_bf16 | exp_hi_bf16 << 16) when EXP_MODE == 1,
//                        const_exp.to_bits() when EXP_MODE == 2, else 0.
const MAGIC: u128 = 0xA;
const DOMAIN_SHIFT: u32 = 4;
const SUBNORM_SHIFT: u32 = 9;
const VALIDATION_SHIFT: u32 = 12;
const PREC_SHIFT: u32 = 15;
const PORTABILITY_SHIFT: u32 = 18;
const EXECUTION_SHIFT: u32 = 20;
const HAS_CUSTOM_BASE: u128 = 1 << 22;
const EXP_MODE_SHIFT: u32 = 23;
const SATURATE_OUTPUT: u128 = 1 << 25;
const RESERVED_MASK: u128 = ((1_u128 << 38) - 1) << 26;
const BASE_LO_SHIFT: u32 = 64;
const BASE_HI_SHIFT: u32 = 80;
const EXP_SLOT_SHIFT: u32 = 96;

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

/// Discrete input-base and default-exponent domain for [`PowfOptions`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum PowfDomain {
    /// Positive normal bases (`[f32::MIN_POSITIVE, f32::MAX]`), any finite exponent.
    #[default]
    PositiveNormals,
    /// Non-negative bases (`{+0.0} ∪ [f32::MIN_POSITIVE, f32::MAX]`; `+0.0` admitted when `exponent >= 0`).
    NonNegative,
    /// SDR unit domain (`{+0.0} ∪ [f32::MIN_POSITIVE, 1.0]`; `+0.0` admitted when `exponent >= 0`).
    Unit,
    /// Strictly positive unit domain (`[f32::MIN_POSITIVE, 1.0]`, zero excluded; supports negative exponents).
    PositiveUnit,
    /// Encoding transfer domain (`{+0.0} ∪ [f32::MIN_POSITIVE, 1.0]`, default exponent `[0.0, 0.99]`),
    /// eliding internal output range checks across all positive normals `(0, 1]`.
    UnitEncode,
    /// SDR transfer-function gamma domain (`{+0.0} ∪ [2^-16, 1.0]`, default exponent `[0.0, 4.0]`),
    /// eliding internal output range checks.
    SdrGamma,
    /// HDR / extended transfer-function gamma domain (`{+0.0} ∪ [2^-16, 2^16]`, default exponent `[0.0, 4.0]`),
    /// eliding internal output range checks.
    HdrGamma,
    /// Strictly positive HDR gamma domain (`[2^-16, 2^16]`, zero excluded, default exponent `[0.0, 4.0]`).
    PositiveGamma,
    /// SMPTE ST 2084 (PQ) rational ratio domain (`[107.0 / 128.0, 1.0]`, default exponent `[0.0, 80.0]`),
    /// eliding internal output range checks up to `PQ_M2 = 78.84375`.
    PqRatio,
    /// Full IEEE-754 `f32` domain (`±0`, subnormals, negative bases at integer exponents, `±inf`, `NaN`).
    /// Requires [`PowfPrecision::Highp`].
    AllFloats,
}

/// Approximation family, not a global mathematical error bound.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum PowfPrecision {
    /// Existing MIDP coefficients and graph; no whole-domain ULP guarantee.
    #[default]
    Midp,
    /// Shorter LOWP coefficients; prefer a bounded gamma or unit domain.
    Lowp,
    /// Table-based binary64 graph for [`Portability::BitExact`]; `std` `f32::powf`
    /// for [`Portability::BackendDependent`] (requires the `std` feature;
    /// specialized algebraic [`PowfOptions::const_exponent`] values use the shared
    /// algebraic evaluator — with binary64 intermediates for compound steps — in
    /// both portability modes).
    Highp,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExpSpec {
    Default,
    RangeBf16(u16, u16),
    ConstF32(u32),
}

/// Fluent `const` configuration builder for [`super::Powf`] and [`crate::SimdMath`].
///
/// Options separate:
/// - **Domain & range** ([`domain`](Self::domain), [`base_range`](Self::base_range),
///   [`exponent_range`](Self::exponent_range), [`const_exponent`](Self::const_exponent));
/// - **Subnormal policy** ([`subnormals`](Self::subnormals): [`Subnormals::Reject`]
///   default, [`flush_or_clamp_subnormals`](Self::flush_or_clamp_subnormals) to
///   shove subnormals to `0` / `MIN_POSITIVE`, or [`admit_subnormals`](Self::admit_subnormals));
/// - **Out-of-bounds validation** ([`error_on_out_of_bounds`](Self::error_on_out_of_bounds),
///   [`clamp_out_of_bounds`](Self::clamp_out_of_bounds), [`assume_in_bounds`](Self::assume_in_bounds)).
///
/// ```
/// use magemath::powf::{Powf, PowfDomain, PowfOptions};
///
/// const SRGB_ENC: u128 = PowfOptions::new()
///     .domain(PowfDomain::Unit)
///     .const_exponent(1.0 / 2.4)
///     .flush_or_clamp_subnormals()
///     .bits();
///
/// assert!(PowfOptions::from_bits(SRGB_ENC).unwrap().elides_range_check());
///
/// let src = [0.0, 1.0e-41, 0.25, 1.0];
/// let mut dst = [0.0; 4];
/// Powf::<SRGB_ENC>::slice_const(&src, &mut dst).unwrap();
/// assert_eq!(dst[0], 0.0);
/// assert_eq!(dst[1], 0.0);
/// assert_eq!(dst[3], 1.0);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PowfOptions {
    domain: PowfDomain,
    custom_base: Option<(u16, u16)>,
    exp_spec: ExpSpec,
    subnormals: Subnormals,
    saturate: bool,
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
    /// `PowfDomain::PositiveNormals`, `Subnormals::Reject`, `Midp`, `BitExact`,
    /// `Validation::Checked` (`error_on_out_of_bounds`), `Execution::Upgrade`.
    pub const fn new() -> Self {
        Self {
            domain: PowfDomain::PositiveNormals,
            custom_base: None,
            exp_spec: ExpSpec::Default,
            subnormals: Subnormals::Reject,
            saturate: false,
            precision: PowfPrecision::Midp,
            portability: Portability::BitExact,
            validation: Validation::Checked,
            execution: Execution::Upgrade,
        }
    }

    /// Select the discrete [`PowfDomain`], clearing any custom [`base_range`](Self::base_range).
    pub const fn domain(mut self, domain: PowfDomain) -> Self {
        self.domain = domain;
        self.custom_base = None;
        if matches!(domain, PowfDomain::AllFloats) {
            self.exp_spec = ExpSpec::Default;
            self.subnormals = Subnormals::Reject;
        }
        self
    }

    /// Override base bounds with outward-rounded `bf16` `[lo, hi]` (`lo >= f32::MIN_POSITIVE`,
    /// `hi <= MAX_BF16`), retaining the active domain's zero-admission rule.
    pub const fn base_range(mut self, lo: f32, hi: f32) -> Self {
        assert!(
            lo >= f32::MIN_POSITIVE && lo <= hi && hi <= MAX_BF16,
            "base range must be positive normal within MAX_BF16 (0x7f7f_0000) with lo <= hi"
        );
        self.custom_base = Some((floor_bf16(lo), ceil_bf16(hi)));
        self
    }

    /// Restrict runtime exponents to outward-rounded `bf16` `[lo, hi]` (`|bound| <= MAX_BF16`),
    /// replacing any previous [`const_exponent`](Self::const_exponent).
    pub const fn exponent_range(mut self, lo: f32, hi: f32) -> Self {
        assert!(
            lo <= hi && abs(lo) <= MAX_BF16 && abs(hi) <= MAX_BF16,
            "exponent range must be finite within ±MAX_BF16 (0x7f7f_0000) with lo <= hi"
        );
        self.exp_spec = ExpSpec::RangeBf16(floor_bf16(lo), ceil_bf16(hi));
        self
    }

    /// Specialize the kernel for a compile-time constant finite exponent `n: f32`
    /// (enabling `powf_const`, `try_powf_const`, `scalar_const`, `array_const`,
    /// `slice_const`, compile-time sign/range proofs, and IEEE hardware fast-paths
    /// for `0, 0.25, ±0.5, ±1, 1.5, ±2, 3, 4`).
    pub const fn const_exponent(mut self, n: f32) -> Self {
        assert!(
            n.is_finite(),
            "const_exponent requires a finite f32 exponent"
        );
        self.exp_spec = ExpSpec::ConstF32(canon_zero(n).to_bits());
        self
    }

    /// Configure how nonzero subnormal inputs are handled.
    pub const fn subnormals(mut self, policy: Subnormals) -> Self {
        self.subnormals = policy;
        self
    }

    /// Treat nonzero subnormals as out-of-bounds (`Subnormals::Reject`, default).
    pub const fn reject_subnormals(self) -> Self {
        self.subnormals(Subnormals::Reject)
    }

    /// Shove nonzero subnormals out to `+0.0` (when zero is admitted) or the
    /// lower positive normal bound (when zero is excluded), even under
    /// [`Validation::Checked`].
    pub const fn flush_or_clamp_subnormals(self) -> Self {
        self.subnormals(Subnormals::FlushOrClamp)
    }

    /// Alias for [`flush_or_clamp_subnormals`](Self::flush_or_clamp_subnormals).
    pub const fn shove_subnormals(self) -> Self {
        self.flush_or_clamp_subnormals()
    }

    /// Opt in to exact subnormal base normalization (`* 2^23` with `-23` exponent
    /// adjustment) and nonzero subnormal exponents (`Subnormals::Normalize`).
    pub const fn admit_subnormals(self) -> Self {
        self.subnormals(Subnormals::Normalize)
    }

    /// Also accept `+0.0` bases when `exponent >= 0`: `0^0 = 1`, `0^positive = +0`.
    pub const fn admit_zero(mut self) -> Self {
        self.domain = match self.domain {
            PowfDomain::PositiveNormals => PowfDomain::NonNegative,
            PowfDomain::PositiveUnit => PowfDomain::Unit,
            PowfDomain::PositiveGamma => PowfDomain::HdrGamma,
            other => other,
        };
        self
    }

    /// Alias for [`admit_zero`](Self::admit_zero).
    pub const fn admit_zero_base(self) -> Self {
        self.admit_zero()
    }

    /// Standard HDR/extended transfer-function gamma domain (`PowfDomain::HdrGamma`,
    /// resetting `subnormals` to `Reject` and clearing `const_exponent` / custom `exponent_range`).
    pub const fn gamma_domain(mut self) -> Self {
        if !matches!(self.domain, PowfDomain::AllFloats) {
            self.domain = PowfDomain::HdrGamma;
            self.custom_base = None;
            self.exp_spec = ExpSpec::Default;
            self.subnormals = Subnormals::Reject;
        }
        self
    }

    /// Shorthand alias for [`gamma_domain`](Self::gamma_domain).
    pub const fn gamma(self) -> Self {
        self.gamma_domain()
    }

    /// Strictly positive gamma domain (`PowfDomain::PositiveGamma`, excluding `+0` and subnormals).
    pub const fn positive_gamma_domain(mut self) -> Self {
        if !matches!(self.domain, PowfDomain::AllFloats) {
            self.domain = PowfDomain::PositiveGamma;
            self.custom_base = None;
            self.exp_spec = ExpSpec::Default;
            self.subnormals = Subnormals::Reject;
        }
        self
    }

    /// Shorthand alias for [`positive_gamma_domain`](Self::positive_gamma_domain).
    pub const fn positive_gamma(self) -> Self {
        self.positive_gamma_domain()
    }

    /// Unit-normalized SDR transfer-function domain (`PowfDomain::SdrGamma`).
    pub const fn unit_gamma_domain(mut self) -> Self {
        if !matches!(self.domain, PowfDomain::AllFloats) {
            self.domain = PowfDomain::SdrGamma;
            self.custom_base = None;
            self.exp_spec = ExpSpec::Default;
            self.subnormals = Subnormals::Reject;
        }
        self
    }

    /// Encoding gamma domain (`PowfDomain::UnitEncode`: `{+0.0} ∪ [MIN_POSITIVE, 1.0]`,
    /// exponent `[0.0, 0.99]`), eliding per-lane output range checks.
    pub const fn srgb_encode_domain(mut self) -> Self {
        if !matches!(self.domain, PowfDomain::AllFloats) {
            self.domain = PowfDomain::UnitEncode;
            self.custom_base = None;
            self.exp_spec = ExpSpec::Default;
            self.subnormals = Subnormals::Reject;
        }
        self
    }

    /// Accept every `f32` input and output (`PowfDomain::AllFloats`; requires [`PowfPrecision::Highp`]).
    pub const fn admit_all_floats(self) -> Self {
        self.domain(PowfDomain::AllFloats)
    }

    /// Alias for [`admit_all_floats`](Self::admit_all_floats).
    pub const fn full_domain(self) -> Self {
        self.admit_all_floats()
    }

    /// Saturate out-of-range `Midp`/`Lowp` outputs instead of returning `OutsideApproximationRange`.
    pub const fn saturate_output(mut self) -> Self {
        self.saturate = true;
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

    /// Omit numerical input/output checks and clamps (`Validation::AssumeValid`, infallible).
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

    /// Active discrete [`PowfDomain`].
    pub const fn selected_domain(&self) -> PowfDomain {
        self.domain
    }

    /// Active [`Subnormals`] policy (`Normalize` when `PowfDomain::AllFloats`).
    pub const fn subnormals_policy(&self) -> Subnormals {
        if self.is_full() {
            Subnormals::Normalize
        } else {
            self.subnormals
        }
    }

    /// Whether [`const_exponent`](Self::const_exponent) was configured.
    pub const fn has_const_exponent(&self) -> bool {
        matches!(self.exp_spec, ExpSpec::ConstF32(_))
    }

    /// Configured constant exponent (flushing a nonzero subnormal to `0.0` when
    /// [`Subnormals::FlushOrClamp`] is active).
    pub const fn const_exponent_value(&self) -> Option<f32> {
        match self.exp_spec {
            ExpSpec::ConstF32(bits) => {
                let n = f32::from_bits(bits);
                if self.flushes_subnormals() {
                    let abs_b = bits & 0x7fff_ffff;
                    if abs_b != 0 && abs_b < 0x0080_0000 {
                        return Some(0.0);
                    }
                }
                Some(n)
            }
            _ => None,
        }
    }

    pub(crate) const fn is_full(&self) -> bool {
        matches!(self.domain, PowfDomain::AllFloats)
    }

    pub(crate) const fn admits_zero(&self) -> bool {
        matches!(
            self.domain,
            PowfDomain::NonNegative
                | PowfDomain::Unit
                | PowfDomain::UnitEncode
                | PowfDomain::SdrGamma
                | PowfDomain::HdrGamma
        )
    }

    pub(crate) const fn admits_subnormals(&self) -> bool {
        !self.is_full() && matches!(self.subnormals, Subnormals::Normalize)
    }

    pub(crate) const fn flushes_subnormals(&self) -> bool {
        !self.is_full() && matches!(self.subnormals, Subnormals::FlushOrClamp)
    }

    pub(crate) const fn base_bounds(&self) -> Option<(f32, f32)> {
        if self.is_full() {
            return None;
        }
        if let Some((lo, hi)) = self.custom_base {
            return Some((expand_bf16(lo), expand_bf16(hi)));
        }
        match self.domain {
            PowfDomain::PositiveNormals | PowfDomain::NonNegative | PowfDomain::AllFloats => None,
            PowfDomain::Unit | PowfDomain::PositiveUnit | PowfDomain::UnitEncode => {
                Some((f32::MIN_POSITIVE, 1.0))
            }
            PowfDomain::SdrGamma => Some((1.0 / 65_536.0, 1.0)),
            PowfDomain::HdrGamma | PowfDomain::PositiveGamma => Some((1.0 / 65_536.0, 65_536.0)),
            PowfDomain::PqRatio => Some((107.0 / 128.0, 1.0)),
        }
    }

    pub(crate) const fn exponent_bounds(&self) -> Option<(f32, f32)> {
        if self.is_full() {
            return None;
        }
        match self.exp_spec {
            ExpSpec::RangeBf16(lo, hi) => Some((expand_bf16(lo), expand_bf16(hi))),
            ExpSpec::ConstF32(_) => {
                let Some(n) = self.const_exponent_value() else {
                    return None;
                };
                Some((n, n))
            }
            ExpSpec::Default => match self.domain {
                PowfDomain::UnitEncode => Some((0.0, 0.99)),
                PowfDomain::SdrGamma | PowfDomain::HdrGamma | PowfDomain::PositiveGamma => {
                    Some((0.0, 4.0))
                }
                PowfDomain::PqRatio => Some((0.0, 80.0)),
                _ => None,
            },
        }
    }

    /// Algebraic hardware fast-path code (`0` = polynomial/table graph, `1..=11` = exact IEEE op).
    pub(crate) const fn alg_kind(&self) -> u8 {
        if self.is_full() {
            return 0;
        }
        let Some(n) = self.const_exponent_value() else {
            return 0;
        };
        if n == 0.0 {
            1
        } else if n == 1.0 {
            2
        } else if n == -1.0 {
            3
        } else if n == 2.0 {
            4
        } else if n == -2.0 {
            5
        } else if n == 3.0 {
            6
        } else if n == 4.0 {
            7
        } else if n == 0.5 {
            8
        } else if n == -0.5 {
            9
        } else if n == 1.5 {
            10
        } else if n == 0.25 {
            11
        } else {
            0
        }
    }

    /// Whether this configuration is self-consistent under current crate features.
    pub const fn is_valid(&self) -> bool {
        if self.is_full() {
            if !matches!(self.precision, PowfPrecision::Highp) {
                return false;
            }
            if self.custom_base.is_some()
                || !matches!(self.exp_spec, ExpSpec::Default)
                || !matches!(self.subnormals, Subnormals::Reject)
            {
                return false;
            }
        }
        if matches!(self.precision, PowfPrecision::Highp) && self.saturate {
            return false;
        }
        if matches!(
            (self.precision, self.portability),
            (PowfPrecision::Highp, Portability::BackendDependent)
        ) && !cfg!(feature = "std")
        {
            return false;
        }
        if self.admits_subnormals()
            && let Some((lo, _)) = self.base_bounds()
            && lo > f32::MIN_POSITIVE
        {
            return false;
        }
        match self.exp_spec {
            ExpSpec::RangeBf16(lo_b, hi_b) => {
                if !self.admits_subnormals()
                    && is_nonzero_subnormal_only_interval(expand_bf16(lo_b), expand_bf16(hi_b))
                {
                    return false;
                }
            }
            ExpSpec::ConstF32(bits) => {
                let abs_b = bits & 0x7fff_ffff;
                if matches!(self.subnormals, Subnormals::Reject)
                    && abs_b != 0
                    && abs_b < 0x0080_0000
                {
                    return false;
                }
                if self.admits_zero()
                    && matches!(self.validation, Validation::AssumeValid)
                    && let Some(n) = self.const_exponent_value()
                    && n < 0.0
                {
                    return false;
                }
            }
            ExpSpec::Default => {}
        }
        true
    }

    /// Assert that this configuration is self-consistent.
    pub const fn validate(&self) {
        if self.is_full() {
            assert!(
                matches!(self.precision, PowfPrecision::Highp),
                "PowfDomain::AllFloats / full_domain requires PowfPrecision::Highp"
            );
            assert!(
                self.custom_base.is_none()
                    && matches!(self.exp_spec, ExpSpec::Default)
                    && matches!(self.subnormals, Subnormals::Reject),
                "PowfDomain::AllFloats cannot be combined with custom ranges, const_exponent, or subnormal overrides"
            );
        }
        assert!(
            !(matches!(self.precision, PowfPrecision::Highp) && self.saturate),
            "saturate_output / clamp_output is only supported on Midp and Lowp"
        );
        assert!(
            !(matches!(
                (self.precision, self.portability),
                (PowfPrecision::Highp, Portability::BackendDependent)
            ) && !cfg!(feature = "std")),
            "BackendDependent Highp requires the `std` feature"
        );
        if self.admits_subnormals()
            && let Some((lo, _)) = self.base_bounds()
        {
            assert!(
                lo <= f32::MIN_POSITIVE,
                "admit_subnormals requires base lower bound == f32::MIN_POSITIVE (for example PowfDomain::Unit or PositiveNormals); use .flush_or_clamp_subnormals() on raised gamma bounds"
            );
        }
        match self.exp_spec {
            ExpSpec::RangeBf16(lo_b, hi_b) => {
                assert!(
                    !(!self.admits_subnormals()
                        && is_nonzero_subnormal_only_interval(
                            expand_bf16(lo_b),
                            expand_bf16(hi_b)
                        )),
                    "exponent_range contains only nonzero subnormals; enable .admit_subnormals(), or include 0.0 or a normal f32"
                );
            }
            ExpSpec::ConstF32(bits) => {
                let abs_b = bits & 0x7fff_ffff;
                assert!(
                    !(matches!(self.subnormals, Subnormals::Reject)
                        && abs_b != 0
                        && abs_b < 0x0080_0000),
                    "nonzero subnormal const_exponent requires .admit_subnormals() or .flush_or_clamp_subnormals()"
                );
                if let Some(n) = self.const_exponent_value() {
                    assert!(
                        !(self.admits_zero()
                            && matches!(self.validation, Validation::AssumeValid)
                            && n < 0.0),
                        "negative const_exponent with zero-admitting domain cannot use assume_in_bounds; use a positive-only domain or checked/clamp validation"
                    );
                }
            }
            ExpSpec::Default => {}
        }
    }

    /// Whether numerical evaluation cannot return `Err` for valid widths.
    pub const fn is_infallible(&self) -> bool {
        self.is_valid()
            && (matches!(
                self.validation,
                Validation::AssumeValid | Validation::ClampOutOfBounds
            ) || self.is_full())
    }

    /// Whether the configured base and exponent domains statically keep outputs
    /// inside the normal range, eliding per-lane output range checks.
    pub const fn elides_range_check(&self) -> bool {
        if self.is_full() {
            return false;
        }
        let alg = self.alg_kind();
        if matches!(alg, 1 | 8 | 11) || (alg == 2 && !self.admits_subnormals()) {
            return true;
        }
        let Some((blo, bhi)) = self.base_bounds() else {
            return false;
        };
        let Some((nlo, nhi)) = self.exponent_bounds() else {
            return false;
        };
        if self.admits_zero() && nlo < 0.0 {
            return false;
        }
        let eb_lo = if self.admits_subnormals() {
            -149.0_f32
        } else {
            exponent_of(blo) as f32
        };
        let eb_hi = (exponent_of(bhi) + 1) as f32;
        let p0 = eb_lo * nlo;
        let p1 = eb_lo * nhi;
        let p2 = eb_hi * nlo;
        let p3 = eb_hi * nhi;
        abs(p0) <= ELIDE_LIMIT
            && abs(p1) <= ELIDE_LIMIT
            && abs(p2) <= ELIDE_LIMIT
            && abs(p3) <= ELIDE_LIMIT
    }

    /// Encode this configuration into a `u128` for `Powf<CFG>` / `SimdMath`.
    pub const fn bits(&self) -> u128 {
        let dom: u128 = match self.domain {
            PowfDomain::PositiveNormals => 0,
            PowfDomain::NonNegative => 1,
            PowfDomain::Unit => 2,
            PowfDomain::PositiveUnit => 3,
            PowfDomain::UnitEncode => 4,
            PowfDomain::SdrGamma => 5,
            PowfDomain::HdrGamma => 6,
            PowfDomain::PositiveGamma => 7,
            PowfDomain::PqRatio => 8,
            PowfDomain::AllFloats => 9,
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
        };
        let prec: u128 = match self.precision {
            PowfPrecision::Midp => 0,
            PowfPrecision::Lowp => 1,
            PowfPrecision::Highp => 2,
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
        if let Some((lo, hi)) = self.custom_base {
            out |= HAS_CUSTOM_BASE;
            out |= (lo as u128) << BASE_LO_SHIFT;
            out |= (hi as u128) << BASE_HI_SHIFT;
        }
        match self.exp_spec {
            ExpSpec::Default => {}
            ExpSpec::RangeBf16(lo, hi) => {
                out |= 1_u128 << EXP_MODE_SHIFT;
                let packed = (lo as u32) | ((hi as u32) << 16);
                out |= (packed as u128) << EXP_SLOT_SHIFT;
            }
            ExpSpec::ConstF32(bits) => {
                out |= 2_u128 << EXP_MODE_SHIFT;
                out |= (bits as u128) << EXP_SLOT_SHIFT;
            }
        }
        if self.saturate {
            out |= SATURATE_OUTPUT;
        }
        out
    }

    const fn from_bits_raw(bits: u128) -> Option<Self> {
        if (bits & 0xf) != MAGIC || (bits & RESERVED_MASK) != 0 {
            return None;
        }
        let domain = match (bits >> DOMAIN_SHIFT) & 0x1f {
            0 => PowfDomain::PositiveNormals,
            1 => PowfDomain::NonNegative,
            2 => PowfDomain::Unit,
            3 => PowfDomain::PositiveUnit,
            4 => PowfDomain::UnitEncode,
            5 => PowfDomain::SdrGamma,
            6 => PowfDomain::HdrGamma,
            7 => PowfDomain::PositiveGamma,
            8 => PowfDomain::PqRatio,
            9 => PowfDomain::AllFloats,
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
            _ => return None,
        };
        let precision = match (bits >> PREC_SHIFT) & 0x7 {
            0 => PowfPrecision::Midp,
            1 => PowfPrecision::Lowp,
            2 => PowfPrecision::Highp,
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

        let has_base = (bits & HAS_CUSTOM_BASE) != 0;
        let base_lo_u16 = ((bits >> BASE_LO_SHIFT) & 0xffff) as u16;
        let base_hi_u16 = ((bits >> BASE_HI_SHIFT) & 0xffff) as u16;
        let custom_base = if has_base {
            let Some(lo) = decode_bf16_finite(base_lo_u16) else {
                return None;
            };
            let Some(hi) = decode_bf16_finite(base_hi_u16) else {
                return None;
            };
            if !(lo >= f32::MIN_POSITIVE && lo <= hi && hi <= MAX_BF16) {
                return None;
            }
            Some((base_lo_u16, base_hi_u16))
        } else {
            if base_lo_u16 != 0 || base_hi_u16 != 0 {
                return None;
            }
            None
        };

        let exp_slot = ((bits >> EXP_SLOT_SHIFT) & 0xffff_ffff) as u32;
        let exp_spec = match (bits >> EXP_MODE_SHIFT) & 0x3 {
            0 => {
                if exp_slot != 0 {
                    return None;
                }
                ExpSpec::Default
            }
            1 => {
                let lo_u16 = (exp_slot & 0xffff) as u16;
                let hi_u16 = (exp_slot >> 16) as u16;
                let Some(lo) = decode_bf16_finite(lo_u16) else {
                    return None;
                };
                let Some(hi) = decode_bf16_finite(hi_u16) else {
                    return None;
                };
                if !(lo <= hi && abs(lo) <= MAX_BF16 && abs(hi) <= MAX_BF16) {
                    return None;
                }
                ExpSpec::RangeBf16(lo_u16, hi_u16)
            }
            2 => {
                // Reject -0.0 (`0x8000_0000`) and non-finite (`±inf`, `NaN`).
                if exp_slot == 0x8000_0000 || (exp_slot & 0x7f80_0000) == 0x7f80_0000 {
                    return None;
                }
                ExpSpec::ConstF32(exp_slot)
            }
            _ => return None,
        };

        Some(Self {
            domain,
            custom_base,
            exp_spec,
            subnormals,
            saturate: (bits & SATURATE_OUTPUT) != 0,
            precision,
            portability,
            validation,
            execution,
        })
    }

    /// Decode a `u128` option word back into [`PowfOptions`], returning `None`
    /// if the tag, bounds, reserved bits, or feature combination is invalid.
    pub const fn from_bits(bits: u128) -> Option<Self> {
        let Some(opts) = Self::from_bits_raw(bits) else {
            return None;
        };
        if opts.is_valid() { Some(opts) } else { None }
    }

    pub(crate) const fn decode_const(bits: u128) -> Self {
        let Some(opts) = Self::from_bits_raw(bits) else {
            panic!("invalid PowfOptions bit pattern");
        };
        opts.validate();
        opts
    }

    /// Build a dynamic runtime evaluator from these options.
    #[inline]
    pub const fn runtime(self) -> super::PowfRuntime {
        super::PowfRuntime::new(self)
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

    pub(crate) const fn clamp_oob(&self) -> bool {
        matches!(self.validation, Validation::ClampOutOfBounds)
    }

    pub(crate) const fn should_clamp(&self) -> bool {
        self.saturate
    }

    pub(crate) const fn is_caller(&self) -> bool {
        matches!(self.execution, Execution::Caller)
    }

    #[allow(dead_code)]
    pub(crate) const fn selected_execution(&self) -> Execution {
        self.execution
    }
}

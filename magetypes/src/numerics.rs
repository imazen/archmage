//! Compile-time numerical policies for vector operations and scalar tails.
//!
//! Encode a policy with `bits()` and pass it to the same `*_with` operation
//! in the vector loop and its [scalar remainder](crate::scalar).
//!
//! | Policy | Vector example | Scalar equivalent |
//! | --- | --- | --- |
//! | [`IntConvert`] | [`f32x8::to_u16_with`](crate::simd::generic::f32x8::to_u16_with) | [`scalar::to_u16_with`](crate::scalar::to_u16_with) |
//! | [`HalfConvert`] | [`f32x8::to_f16_bits_with`](crate::simd::generic::f32x8::to_f16_bits_with) | [`scalar::to_f16_bits_with`](crate::scalar::to_f16_bits_with) |
//! | [`RecipOptions`] | [`f32x8::recip_with`](crate::simd::generic::f32x8::recip_with) | [`scalar::recip_f32_with`](crate::scalar::recip_f32_with) |
//! | [`ExtremaOptions`] | [`f32x8::min_with`](crate::simd::generic::f32x8::min_with) | [`scalar::min_f32_with`](crate::scalar::min_f32_with) |
//!
//! ```
//! use archmage::ScalarToken;
//! use magetypes::{numerics::IntConvert, scalar, simd::generic::f32x8};
//!
//! const CFG: u32 = IntConvert::new().trunc().saturate_all().bits();
//! let input = [1.9, -1.0, f32::NAN, 65_536.0, 255.5, 42.0, 0.0, 8.9];
//! let vector = f32x8::load_t(ScalarToken, &input);
//! assert_eq!(vector.to_u16_with::<CFG>(), input.map(scalar::to_u16_with::<CFG>));
//! assert_eq!(scalar::to_u16_with::<CFG>(1.9), 1);
//! ```
//!
//! Checked conversion methods use `try_*_with` and report every rejected lane
//! in a [`NumericsError`]. Integer ranges apply to the original input before
//! rounding. Half policies check input finiteness, leaving finite overflow
//! and subnormal handling to the selected conversion policy.
//!
//! Reserved bits are rejected at compile time.
//!
//! ```compile_fail
//! use magetypes::{numerics::IntConvert, scalar};
//! const RESERVED: u32 = IntConvert::new().bits() | 0x80;
//! let _ = scalar::try_to_i32_with::<RESERVED>(1.0);
//! ```
//!
//! A policy from another family is also rejected at compile time.
//!
//! ```compile_fail
//! use archmage::ScalarToken;
//! use magetypes::{numerics::IntConvert, simd::generic::f32x4};
//! let input = f32x4::splat_t(ScalarToken, 1.0);
//! let _ = input.try_to_f16_bits_with::<{IntConvert::new().bits()}>();
//! ```
//!
//! Checked reciprocal and reciprocal square root validate the normal input
//! domain for approximate policies. Exact and all-floats policies admit
//! every input, including IEEE special values.
//!
//! ```
//! use magetypes::{numerics::{NumericsErrorKind, RecipOptions}, scalar};
//! const NORMAL: u32 = RecipOptions::new().bits();
//! let invalid = scalar::try_rsqrt_f32_with::<NORMAL>(-4.0).unwrap_err();
//! assert_eq!(invalid.kind, NumericsErrorKind::RecipDomain);
//! assert_eq!(invalid.lane_mask, 1);
//! assert_eq!(scalar::try_recip_f32_with::<NORMAL>(4.0).unwrap(), 0.25);
//! ```
//!
//! ```compile_fail
//! use magetypes::{numerics::RecipOptions, scalar};
//! const RESERVED: u32 = RecipOptions::new().bits() | 0x80;
//! let _ = scalar::try_recip_f64_with::<RESERVED>(1.0);
//! ```
//!
//! ```compile_fail
//! use archmage::ScalarToken;
//! use magetypes::{numerics::HalfConvert, simd::generic::f32x4};
//! let input = f32x4::splat_t(ScalarToken, 1.0);
//! let _ = input.try_rsqrt_with::<{ HalfConvert::new().bits() }>();
//! ```

const FAMILY_MASK: u32 = 0xff00_0000;
const INT_FAMILY: u32 = 0x0100_0000;
const HALF_FAMILY: u32 = 0x0200_0000;
const RECIP_FAMILY: u32 = 0x0300_0000;
const EXTREMA_FAMILY: u32 = 0x0400_0000;

/// Input contract violated by a checked numerical operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NumericsErrorKind {
    /// A lane is outside the selected integer input bounds, including nonfinite lanes.
    IntegerBounds,
    /// A half-precision conversion requires finite input.
    NonFinite,
    /// A reciprocal input is outside the selected normal or positive-normal domain.
    RecipDomain,
}

/// Rejection of a checked scalar or SIMD numerical operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NumericsError {
    /// Which input contract failed.
    pub kind: NumericsErrorKind,
    /// Bit `i` identifies rejected lane `i`. Scalar calls use bit zero.
    pub lane_mask: u16,
}

impl core::fmt::Display for NumericsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?} (lane mask {:#06x})", self.kind, self.lane_mask)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for NumericsError {}

#[inline(always)]
fn check_lanes<T, const N: usize>(
    lanes: [T; N],
    valid: impl Fn(T) -> bool,
    kind: NumericsErrorKind,
) -> Result<(), NumericsError> {
    let lane_mask = lanes
        .into_iter()
        .enumerate()
        .fold(0, |mask, (i, lane)| mask | (u16::from(!valid(lane)) << i));
    if lane_mask == 0 {
        Ok(())
    } else {
        Err(NumericsError { kind, lane_mask })
    }
}

// Compare the original f32 to the destination domain before rounding. The
// inclusive upper constants for the 32-bit types are their largest in-range
// f32 values, since i32::MAX and u32::MAX round up when represented as f32.
pub(crate) const I32_MIN: f32 = -2_147_483_648.0;
pub(crate) const I32_MAX: f32 = 2_147_483_520.0;
pub(crate) const U32_MAX: f32 = 4_294_967_040.0;

/// Rounding before conversion to an integer.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
#[repr(u32)]
pub enum IntRounding {
    /// Round to the nearest integer, with ties going to the even integer.
    #[default]
    NearestEven,
    /// Round toward zero.
    Trunc,
}

/// Input contract for float-to-integer operations.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
#[repr(u32)]
pub enum IntBounds {
    /// Finite numbers in `[-2^31, 2^31)`. Unsigned outputs saturate at their rails.
    #[default]
    Finite,
    /// Finite numbers in the destination range. For i32/u32 the upper limit
    /// is exclusive (2^31 or 2^32 respectively).
    InRange,
    /// Nonnegative finite numbers below 2^31, saturating at the destination's upper rail.
    NonNegativeFinite,
    /// Round, then saturate as in a Rust cast. NaN maps to zero.
    SaturateAll,
}

/// Compile-time float-to-integer conversion policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct IntConvert {
    rounding: IntRounding,
    bounds: IntBounds,
}

impl IntConvert {
    /// Nearest-even rounding for finite inputs below the signed overflow threshold.
    pub const fn new() -> Self {
        Self {
            rounding: IntRounding::NearestEven,
            bounds: IntBounds::Finite,
        }
    }
    /// Select the rounding rule.
    pub const fn rounding(mut self, rounding: IntRounding) -> Self {
        self.rounding = rounding;
        self
    }
    /// Select the input contract.
    pub const fn bounds(mut self, bounds: IntBounds) -> Self {
        self.bounds = bounds;
        self
    }
    /// Round to the nearest even integer on ties.
    pub const fn nearest_even(self) -> Self {
        self.rounding(IntRounding::NearestEven)
    }
    /// Round toward zero.
    pub const fn trunc(self) -> Self {
        self.rounding(IntRounding::Trunc)
    }
    /// Use finite signed-conversion-range inputs.
    pub const fn finite(self) -> Self {
        self.bounds(IntBounds::Finite)
    }
    /// Use inputs already in the destination range.
    pub const fn in_range(self) -> Self {
        self.bounds(IntBounds::InRange)
    }
    /// Use nonnegative inputs below 2^31.
    pub const fn non_negative_finite(self) -> Self {
        self.bounds(IntBounds::NonNegativeFinite)
    }
    /// Saturate arbitrary inputs, mapping NaN to zero.
    pub const fn saturate_all(self) -> Self {
        self.bounds(IntBounds::SaturateAll)
    }
    /// Encode a policy for use as a const generic.
    pub const fn bits(self) -> u32 {
        INT_FAMILY | (self.bounds as u32) << 1 | self.rounding as u32
    }
    /// Decode an option word, rejecting reserved or foreign-family bits.
    pub const fn from_bits(bits: u32) -> Option<Self> {
        if bits & FAMILY_MASK != INT_FAMILY {
            return None;
        }
        let payload = bits & !FAMILY_MASK;
        if payload > 7 {
            return None;
        }
        let rounding = if payload & 1 == 0 {
            IntRounding::NearestEven
        } else {
            IntRounding::Trunc
        };
        let bounds = match payload >> 1 {
            0 => IntBounds::Finite,
            1 => IntBounds::InRange,
            2 => IntBounds::NonNegativeFinite,
            _ => IntBounds::SaturateAll,
        };
        Some(Self { rounding, bounds })
    }
    pub(crate) const fn decode(bits: u32) -> Self {
        match Self::from_bits(bits) {
            Some(opts) => opts,
            None => panic!("invalid IntConvert bits"),
        }
    }
    #[inline(always)]
    pub(crate) fn check<const N: usize>(
        self,
        lanes: impl FnOnce() -> [f32; N],
        min: f32,
        max: f32,
    ) -> Result<(), NumericsError> {
        let (min, max) = match self.bounds {
            IntBounds::Finite => (I32_MIN, I32_MAX),
            IntBounds::InRange => (min, max),
            IntBounds::NonNegativeFinite => (0.0, I32_MAX),
            IntBounds::SaturateAll => return Ok(()),
        };
        check_lanes(
            lanes(),
            |x| x >= min && x <= max,
            NumericsErrorKind::IntegerBounds,
        )
    }

    pub(crate) const fn rounds(self) -> bool {
        matches!(self.rounding, IntRounding::NearestEven)
    }
    pub(crate) const fn saturates_all(self) -> bool {
        matches!(self.bounds, IntBounds::SaturateAll)
    }
    #[cfg(any(
        all(target_arch = "x86_64", target_feature = "sse2"),
        target_arch = "aarch64",
        target_arch = "wasm32"
    ))]
    pub(crate) const fn is_in_range(self) -> bool {
        matches!(self.bounds, IntBounds::InRange)
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    pub(crate) const fn nonnegative(self) -> bool {
        matches!(
            self.bounds,
            IntBounds::InRange | IntBounds::NonNegativeFinite
        )
    }
}

/// Input classification for a half-precision conversion.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum HalfDomain {
    /// Finite inputs. Overflow of a finite input during narrowing follows the rounding mode.
    #[default]
    Finite,
    /// All IEEE inputs, including NaN and infinities. NaN payloads may vary.
    AllFloats,
}

/// Handling of subnormals in the narrower format.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum HalfSubnormals {
    /// Replace subnormals with signed zero, after rounding on narrowing.
    #[default]
    FlushOrClamp,
    /// Preserve subnormals and widen them exactly.
    Normalize,
}

/// Rounding for narrowing into a half-precision format.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum HalfRounding {
    /// Round to nearest with ties to even.
    #[default]
    NearestEven,
    /// Round toward zero. Finite binary16 overflow saturates to finite maximum.
    Trunc,
}

/// Compile-time policy for binary16 and bfloat16 register conversions.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HalfConvert {
    domain: HalfDomain,
    subnormals: HalfSubnormals,
    rounding: HalfRounding,
}

impl HalfConvert {
    /// Finite input, signed-zero flush, nearest-even rounding.
    pub const fn new() -> Self {
        Self {
            domain: HalfDomain::Finite,
            subnormals: HalfSubnormals::FlushOrClamp,
            rounding: HalfRounding::NearestEven,
        }
    }
    /// Select the input classification.
    pub const fn domain(mut self, domain: HalfDomain) -> Self {
        self.domain = domain;
        self
    }
    /// Select the subnormal mode.
    pub const fn subnormals(mut self, subnormals: HalfSubnormals) -> Self {
        self.subnormals = subnormals;
        self
    }
    /// Select the rounding rule for narrowing.
    pub const fn rounding(mut self, rounding: HalfRounding) -> Self {
        self.rounding = rounding;
        self
    }
    /// Admit every IEEE encoding.
    pub const fn all_floats(self) -> Self {
        self.domain(HalfDomain::AllFloats)
    }
    /// Preserve subnormals.
    pub const fn normalize(self) -> Self {
        self.subnormals(HalfSubnormals::Normalize)
    }
    /// Flush subnormals to signed zero.
    pub const fn flush_or_clamp(self) -> Self {
        self.subnormals(HalfSubnormals::FlushOrClamp)
    }
    /// Round toward zero when narrowing.
    pub const fn trunc(self) -> Self {
        self.rounding(HalfRounding::Trunc)
    }
    /// Encode a policy for use as a const generic.
    pub const fn bits(self) -> u32 {
        HALF_FAMILY
            | self.domain as u32
            | (self.subnormals as u32) << 1
            | (self.rounding as u32) << 2
    }
    /// Decode a policy, rejecting reserved or foreign-family bits.
    pub const fn from_bits(bits: u32) -> Option<Self> {
        if bits & FAMILY_MASK != HALF_FAMILY {
            return None;
        }
        let payload = bits & !FAMILY_MASK;
        if payload > 7 {
            return None;
        }
        let domain = if payload & 1 == 0 {
            HalfDomain::Finite
        } else {
            HalfDomain::AllFloats
        };
        let subnormals = if payload & 2 == 0 {
            HalfSubnormals::FlushOrClamp
        } else {
            HalfSubnormals::Normalize
        };
        let rounding = if payload & 4 == 0 {
            HalfRounding::NearestEven
        } else {
            HalfRounding::Trunc
        };
        Some(Self {
            domain,
            subnormals,
            rounding,
        })
    }
    pub(crate) const fn decode(bits: u32) -> Self {
        match Self::from_bits(bits) {
            Some(opts) => opts,
            None => panic!("invalid HalfConvert bits"),
        }
    }
    #[inline(always)]
    pub(crate) fn check_float<const N: usize>(
        self,
        lanes: impl FnOnce() -> [f32; N],
    ) -> Result<(), NumericsError> {
        if self.is_finite() {
            check_lanes(lanes(), f32::is_finite, NumericsErrorKind::NonFinite)?;
        }
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn check_f16<const N: usize>(self, bits: &[u16; N]) -> Result<(), NumericsError> {
        if self.is_finite() {
            check_lanes(
                *bits,
                |lane| lane & 0x7c00 != 0x7c00,
                NumericsErrorKind::NonFinite,
            )?;
        }
        Ok(())
    }

    pub(crate) const fn is_finite(self) -> bool {
        matches!(self.domain, HalfDomain::Finite)
    }
    pub(crate) const fn flushes(self) -> bool {
        matches!(self.subnormals, HalfSubnormals::FlushOrClamp)
    }
    pub(crate) const fn rounds(self) -> bool {
        matches!(self.rounding, HalfRounding::NearestEven)
    }
}

/// Reciprocal or reciprocal-square-root approximation precision.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
#[repr(u32)]
pub enum RecipPrecision {
    /// Native estimate where available, division otherwise.
    Approx = 1,
    /// Newton-refined reciprocal or reciprocal square root (`1` step on `f32`,
    /// `2` steps on `f64` where hardware estimates exist). When the
    /// reciprocal estimate uses division on Scalar or WASM, returns that quotient.
    #[default]
    Nr1 = 0,
    /// Refined reciprocal biased toward positive infinity so finite non-zero
    /// results satisfy `result >= 1 / x`. Exact division advances one
    /// representable step. SIMD estimates use a bias of `|r| * (2 * EPSILON)`.
    /// Reciprocal square root uses the `Nr1` path.
    Nr1Ceil = 2,
    /// Division, or division following square root.
    Exact = 3,
}

/// Reciprocal input and intermediate contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum RecipDomain {
    /// Finite normal inputs strictly below the reciprocal underflow threshold:
    /// `MIN_POSITIVE <= |x| < 1.0 / MIN_POSITIVE` (`[2^-126, 2^126)` for `f32`,
    /// `[2^-1022, 2^1022)` for `f64`; positive for reciprocal square root).
    #[default]
    FiniteNormal,
    /// IEEE special values, zeros and subnormal values are supported.
    AllFloats,
}

/// Compile-time reciprocal policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RecipOptions {
    precision: RecipPrecision,
    domain: RecipDomain,
}

impl RecipOptions {
    /// One Newton refinement for finite normal inputs.
    pub const fn new() -> Self {
        Self {
            precision: RecipPrecision::Nr1,
            domain: RecipDomain::FiniteNormal,
        }
    }
    /// Select precision.
    pub const fn precision(mut self, precision: RecipPrecision) -> Self {
        self.precision = precision;
        self
    }
    /// Select the input contract.
    pub const fn domain(mut self, domain: RecipDomain) -> Self {
        self.domain = domain;
        self
    }
    /// Use a hardware estimate when available.
    pub const fn approx(self) -> Self {
        self.precision(RecipPrecision::Approx)
    }
    /// Use one Newton step.
    pub const fn nr1(self) -> Self {
        self.precision(RecipPrecision::Nr1)
    }
    /// Bias finite non-zero reciprocals toward `+inf` so `result >= 1 / x`.
    pub const fn nr1_ceil(self) -> Self {
        self.precision(RecipPrecision::Nr1Ceil)
    }
    /// Use exact arithmetic for each operation.
    pub const fn exact(self) -> Self {
        self.precision(RecipPrecision::Exact)
    }
    /// Admit all IEEE input classes.
    pub const fn all_floats(self) -> Self {
        self.domain(RecipDomain::AllFloats)
    }
    /// Encode a policy for use as a const generic.
    pub const fn bits(self) -> u32 {
        RECIP_FAMILY | self.precision as u32 | (self.domain as u32) << 2
    }
    /// Decode a policy, rejecting reserved or foreign-family bits.
    pub const fn from_bits(bits: u32) -> Option<Self> {
        if bits & FAMILY_MASK != RECIP_FAMILY {
            return None;
        }
        let payload = bits & !FAMILY_MASK;
        if payload > 7 {
            return None;
        }
        let precision = match payload & 3 {
            0 => RecipPrecision::Nr1,
            1 => RecipPrecision::Approx,
            2 => RecipPrecision::Nr1Ceil,
            _ => RecipPrecision::Exact,
        };
        let domain = if payload & 4 == 0 {
            RecipDomain::FiniteNormal
        } else {
            RecipDomain::AllFloats
        };
        Some(Self { precision, domain })
    }
    pub(crate) const fn decode(bits: u32) -> Self {
        match Self::from_bits(bits) {
            Some(opts) => opts,
            None => panic!("invalid RecipOptions bits"),
        }
    }
    pub(crate) const fn selected_precision(self) -> RecipPrecision {
        self.precision
    }
    pub(crate) const fn admits_all(self) -> bool {
        matches!(self.domain, RecipDomain::AllFloats)
    }

    pub(crate) const fn checks_normal_domain(self) -> bool {
        !self.admits_all() && !matches!(self.precision, RecipPrecision::Exact)
    }
}

/// NaN handling for extrema.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
#[repr(u32)]
pub enum ExtremaDomain {
    /// Both operands finite. Zero ties may select either operand.
    #[default]
    Finite,
    /// The right operand is non-NaN and replaces a quiet NaN on the left.
    /// Signalling NaNs and zero ties may vary by backend.
    NonNanRhs,
    /// Prefer a number to a NaN. Signalling NaNs and zero ties may vary by backend.
    DropNan,
    /// Propagate NaNs and order negative zero before positive zero.
    PropagateNan,
}

/// Compile-time policy for min, max and clamp.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExtremaOptions(ExtremaDomain);

impl ExtremaOptions {
    /// Finite operands and backend-native zero ties.
    pub const fn new() -> Self {
        Self(ExtremaDomain::Finite)
    }
    /// Select the domain and NaN rule.
    pub const fn domain(self, domain: ExtremaDomain) -> Self {
        let _ = self;
        Self(domain)
    }
    /// Require only a non-NaN right operand.
    pub const fn non_nan_rhs(self) -> Self {
        self.domain(ExtremaDomain::NonNanRhs)
    }
    /// Prefer numbers over NaN.
    pub const fn drop_nan(self) -> Self {
        self.domain(ExtremaDomain::DropNan)
    }
    /// Propagate NaN and order signed zeros.
    pub const fn propagate_nan(self) -> Self {
        self.domain(ExtremaDomain::PropagateNan)
    }
    /// Encode a policy for use as a const generic.
    pub const fn bits(self) -> u32 {
        EXTREMA_FAMILY | self.0 as u32
    }
    /// Decode a policy, rejecting reserved or foreign-family bits.
    pub const fn from_bits(bits: u32) -> Option<Self> {
        if bits & FAMILY_MASK != EXTREMA_FAMILY {
            return None;
        }
        let domain = match bits & !FAMILY_MASK {
            0 => ExtremaDomain::Finite,
            1 => ExtremaDomain::NonNanRhs,
            2 => ExtremaDomain::DropNan,
            3 => ExtremaDomain::PropagateNan,
            _ => return None,
        };
        Some(Self(domain))
    }
    pub(crate) const fn decode(bits: u32) -> ExtremaDomain {
        match Self::from_bits(bits) {
            Some(opts) => opts.0,
            None => panic!("invalid ExtremaOptions bits"),
        }
    }
}

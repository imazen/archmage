//! Shared implementation bridge for the draft scalar/vector power policy.
//! Public solely so the companion magemath crate shares transformation order.

macro_rules! mode {
    ($name:ident, $doc:literal, $($variant:ident=$value:expr => $text:literal),+ $(,)?) => {
        #[doc=$doc]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(u8)]
        pub enum $name { $(#[doc=$text] $variant=$value),+ }
    };
}
mode!(PowPrecision,"Power algorithm; no full-range near-one-ULP preset is established.",
    CorrectlyRounded=0=>"CORE-MATH nearest-even with gradual underflow.",
    Midp=1=>"Existing midp polynomial within its working band, CORE rescue outside.",
    Lowp=2=>"Existing lowp polynomial within its working band, CORE rescue outside.");
mode!(PowConsistency,"Fixed polynomial evaluation sequence; CPU tier stays independent.",
    PortableFused=0=>"One rounding for each polynomial multiply-add on every target.",
    PortableSeparate=1=>"Two roundings on every target; approximation modes only.");
mode!(PowSign,"Base domain and output-sign semantics.",
    Mathematical=0=>"Negative finite bases require integer exponents; only odd integers produce negative results.",
    Magnitude=1=>"Compute power of the base's absolute value.",
    SignedMagnitude=2=>"Compute power of the absolute value, then restore the base sign even for even/noninteger exponents.");
mode!(SubnormalInputs,"Treatment of subnormal base AND exponent before power.",
    Preserve=0=>"Preserve both operands.",FlushSignedZero=1=>"Replace either subnormal operand with zero of its original sign.");
mode!(SubnormalOutputs,"Treatment of a subnormal power result.",
    Preserve=0=>"Preserve the result.",FlushSignedZero=1=>"Flush to zero while retaining its sign.");
mode!(PowNaN,"NaN result representation; special-case precedence is CORE-MATH's.",
    CoreMath=0=>"Preserve NaN-ness; payload and sign unspecified.",CanonicalResult=1=>"Canonicalize every NaN result to bits 0x7fc00000.");
mode!(PowInfinity,"Input-infinity handling after explicit input transformations.",
    Preserve=0=>"Use ordinary power semantics.",RejectInputsToNaN=1=>"Either infinite transformed operand forces NaN, including identity cases.");
mode!(PowRange,"Explicit input transformation or precondition, never an unchecked hint.",
    Full=0=>"Complete mathematical/special handling, with CORE rescue outside approximation working inputs.",
    ClampUnitInterval=1=>"Clamp the base to `[0,1]` using ordered comparisons; preserve NaNs and negative zero.",
    PositiveNormalPrecondition=2=>"Approximation only: positive normal base, finite exponent, polynomial exponent in `[-126,127.5]`. Release evaluates the named approximation formula without CORE rescue, also on violations.");
mode!(PowOutput,"Explicit final output transformation; NaNs are retained.",
    Preserve=0=>"Preserve the result.",SaturateFinite=1=>"Replace infinities with signed f32::MAX.",ClampUnitInterval=2=>"Clamp the result to `[0,1]`, preserving NaN and negative zero.");

/// Draft compile-time power contract, transported as a private packed `u64`.
///
/// Encoding is not a persistent/stable format. [`Self::to_u64`] and literal
/// const-generic entry points reject reserved bits and unsupported combinations.
/// Accuracy applies after the requested input transformations and before explicit
/// output transformations. Approximation presets have named-domain accuracy, not
/// a new uniform ULP guarantee. Both consistency modes pin arithmetic order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PowBehavior {
    bits: u64,
}
impl PowBehavior {
    /// Existing correctly rounded power contract, with no transformations.
    pub const CORRECT: Self = Self { bits: 0 };
    /// Fused midp polynomial with complete CORE special/out-of-band rescue.
    pub const MIDP: Self = Self { bits: 1 };
    /// Fused lowp polynomial with complete CORE special/out-of-band rescue.
    pub const LOWP: Self = Self { bits: 2 };
    const fn field(self, shift: u32, mask: u64, value: u64) -> Self {
        Self {
            bits: (self.bits & !(mask << shift)) | value << shift,
        }
    }
    /// Select accuracy/algorithm.
    pub const fn precision(self, v: PowPrecision) -> Self {
        self.field(0, 3, v as u64)
    }
    /// Select a fixed rounding sequence. Separate is not valid with CORE.
    pub const fn consistency(self, v: PowConsistency) -> Self {
        self.field(2, 1, v as u64)
    }
    /// Select mathematical, magnitude, or signed-magnitude semantics.
    pub const fn sign(self, v: PowSign) -> Self {
        self.field(3, 3, v as u64)
    }
    /// Transform subnormal base and exponent independently of output handling.
    pub const fn subnormal_inputs(self, v: SubnormalInputs) -> Self {
        self.field(5, 1, v as u64)
    }
    /// Transform a subnormal result.
    pub const fn subnormal_outputs(self, v: SubnormalOutputs) -> Self {
        self.field(6, 1, v as u64)
    }
    /// Select NaN result representation without changing identity precedence.
    pub const fn nan(self, v: PowNaN) -> Self {
        self.field(7, 1, v as u64)
    }
    /// Select input infinity handling after input transformations.
    pub const fn infinity(self, v: PowInfinity) -> Self {
        self.field(8, 1, v as u64)
    }
    /// Select explicit clamping or an approximation precondition.
    pub const fn range(self, v: PowRange) -> Self {
        self.field(9, 3, v as u64)
    }
    /// Select explicit output transformation.
    pub const fn output(self, v: PowOutput) -> Self {
        self.field(11, 3, v as u64)
    }
    /// Assert the precondition against ORIGINAL inputs in debug builds only.
    /// Valid only with PositiveNormalPrecondition; release evaluates the named
    /// formula without an input classifier/CORE fallback.
    pub const fn debug_assertions(self, v: bool) -> Self {
        self.field(13, 1, v as u64)
    }
    /// Validate and encode the policy for a const-generic operation.
    pub const fn to_u64(self) -> u64 {
        validate(self.bits);
        self.bits
    }
}
/// Validate literal callers as well as typed builders.
#[doc(hidden)]
pub const fn validate(bits: u64) {
    assert!(bits & !0x3fff == 0, "unknown PowBehavior bits");
    let precision = bits & 3;
    let consistency = (bits >> 2) & 1;
    let sign = (bits >> 3) & 3;
    let range = (bits >> 9) & 3;
    let output = (bits >> 11) & 3;
    assert!(
        precision < 3,
        "unsupported power precision; no proven near-one-ULP preset"
    );
    assert!(
        sign < 3 && range < 3 && output < 3,
        "reserved PowBehavior field value"
    );
    assert!(
        precision != 0 || consistency == 0,
        "correct rounding requires its CORE evaluation sequence"
    );
    assert!(
        bits & (1 << 13) == 0 || range == 2,
        "debug input assertions require an explicit precondition range"
    );
    if range == 2 {
        assert!(
            precision != 0,
            "precondition range is for named approximations"
        );
        assert!(
            bits & ((3 << 3) | (1 << 5) | (1 << 6) | (1 << 8) | (3 << 11)) == 0,
            "precondition range conflicts with input/sign/output transforms"
        );
    }
}
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct Prepared {
    pub x: f32,
    pub y: f32,
    pub sign: u32,
    pub reject: bool,
}
#[inline(always)]
fn flush(x: f32) -> f32 {
    if x.to_bits() & 0x7fffffff < 0x00800000 {
        f32::from_bits(x.to_bits() & 0x80000000)
    } else {
        x
    }
}
#[inline(always)]
fn unit(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}
/// Shared precondition check; original operands, structurally absent in release.
#[doc(hidden)]
#[cfg(debug_assertions)]
#[inline(always)]
pub fn debug_input<const P: u64>(x: f32, y: f32) {
    if P & (1 << 13) != 0 {
        let z = polynomial_exponent::<P>(x, y);
        debug_assert!(
            x.is_normal() && x > 0.0 && y.is_finite() && (-126.0..=127.5).contains(&z),
            "power input violates PositiveNormalPrecondition"
        );
    }
}
#[doc(hidden)]
#[cfg(not(debug_assertions))]
#[inline(always)]
pub fn debug_input<const P: u64>(_: f32, _: f32) {}
#[doc(hidden)]
#[inline(always)]
pub fn prepare<const P: u64>(mut x: f32, mut y: f32) -> Prepared {
    debug_input::<P>(x, y);
    if (P >> 9) & 3 == 1 {
        x = unit(x);
    }
    if P & (1 << 5) != 0 {
        x = flush(x);
        y = flush(y);
    }
    let reject = P & (1 << 8) != 0 && (x.is_infinite() || y.is_infinite());
    let sign = if (P >> 3) & 3 == 2 {
        x.to_bits() & 0x80000000
    } else {
        0
    };
    if (P >> 3) & 3 != 0 {
        x = x.abs();
    }
    Prepared { x, y, sign, reject }
}
#[doc(hidden)]
#[inline(always)]
pub fn finish<const P: u64>(input: Prepared, mut out: f32) -> f32 {
    if input.reject {
        out = f32::NAN;
    }
    if (P >> 3) & 3 == 2 && !out.is_nan() {
        out = f32::from_bits(out.to_bits() | input.sign);
    }
    if P & (1 << 6) != 0 {
        out = flush(out);
    }
    match (P >> 11) & 3 {
        1 => {
            if out.is_infinite() {
                out = f32::from_bits((out.to_bits() & 0x80000000) | 0x7f7fffff);
            }
        }
        2 => out = unit(out),
        _ => {}
    }
    if P & (1 << 7) != 0 && out.is_nan() {
        out = f32::from_bits(0x7fc00000);
    }
    out
}
/// (integer, odd integer), with all finite f32 integers >=2^24 even.
#[doc(hidden)]
#[inline(always)]
pub fn integer_parity(y: f32) -> (bool, bool) {
    let b = y.to_bits() & 0x7fffffff;
    let e = (b >> 23) as i32 - 127;
    if e >= 24 {
        return (b < 0x7f800000, false);
    }
    if e < 0 {
        return (b == 0, false);
    }
    let shift = (23 - e) as u32;
    let integer = b & ((1 << shift) - 1) == 0;
    (integer, integer && ((b >> shift) & 1) != 0)
}
#[doc(hidden)]
#[inline(always)]
pub fn working_input(p: Prepared) -> bool {
    p.x.abs().is_normal()
        && p.x.abs() != 1.0
        && p.y.is_finite()
        && p.y != 0.0
        && p.y != 1.0
        && (!p.x.is_sign_negative() || integer_parity(p.y).0)
}
#[doc(hidden)]
#[inline(always)]
pub fn mathematical_sign(p: Prepared, out: f32) -> f32 {
    let sign = (p.x.to_bits() & 0x80000000) & 0u32.wrapping_sub(u32::from(integer_parity(p.y).1));
    f32::from_bits(out.to_bits() | sign)
}
#[doc(hidden)]
#[inline(always)]
pub fn polynomial_exponent<const P: u64>(x: f32, y: f32) -> f32 {
    match (P & 3, P & (1 << 2) != 0) {
        (1, false) => crate::scalar::log2_midp_f32(x) * y,
        (1, true) => crate::scalar::unfused::log2_midp_f32(x) * y,
        (2, false) => crate::scalar::log2_lowp_f32(x) * y,
        (2, true) => crate::scalar::unfused::log2_lowp_f32(x) * y,
        _ => 0.0,
    }
}
#[inline(always)]
fn exponential<const P: u64>(z: f32) -> f32 {
    match (P & 3, P & (1 << 2) != 0) {
        (1, false) => crate::scalar::exp2_midp_f32(z),
        (1, true) => crate::scalar::unfused::exp2_midp_f32(z),
        (2, false) => crate::scalar::exp2_lowp_f32(z),
        (2, true) => crate::scalar::unfused::exp2_lowp_f32(z),
        _ => unreachable!(),
    }
}
#[inline(always)]
fn formula<const P: u64>(x: f32, y: f32) -> f32 {
    match (P & 3, P & (1 << 2) != 0) {
        (1, false) => crate::scalar::pow_midp_f32(x, y),
        (1, true) => crate::scalar::unfused::pow_midp_f32(x, y),
        (2, false) => crate::scalar::pow_lowp_f32(x, y),
        (2, true) => crate::scalar::unfused::pow_lowp_f32(x, y),
        _ => unreachable!(),
    }
}
/// Power with a validated compile-time behavior. Default preserves [`super::scalar::powf`].
///
/// ```
/// use zenmath::{PowBehavior,PowSign};
/// const SIGNED:u64=PowBehavior::CORRECT.sign(PowSign::SignedMagnitude).to_u64();
/// assert_eq!(zenmath::scalar::powf_with::<SIGNED>(-4.0,0.5),-2.0);
/// ```
/// Unknown literal bits cannot bypass validation:
/// ```compile_fail
/// let _=zenmath::scalar::powf_with::<{1u64<<63}>(2.0,3.0);
/// ```
/// No unproven near-one-ULP mode is exposed:
/// ```compile_fail
/// let _=zenmath::scalar::powf_with::<3>(2.0,3.0);
/// ```
/// Reserved field values and conflicting preconditions also fail:
/// ```compile_fail
/// let _=zenmath::scalar::powf_with::<{3u64<<3}>(2.0,3.0);
/// ```
/// ```compile_fail
/// use zenmath::{PowBehavior,PowRange,PowSign};
/// const P:u64=PowBehavior::MIDP.range(PowRange::PositiveNormalPrecondition).sign(PowSign::SignedMagnitude).to_u64();
/// let _=zenmath::scalar::powf_with::<P>(2.0,3.0);
/// ```
/// ```compile_fail
/// use zenmath::PowBehavior;
/// const P:u64=PowBehavior::CORRECT.debug_assertions(true).to_u64();
/// let _=zenmath::scalar::powf_with::<P>(2.0,3.0);
/// ```
/// Correct rounding cannot select a different polynomial rounding sequence:
/// ```compile_fail
/// use zenmath::{PowBehavior,PowConsistency};
/// const P:u64=PowBehavior::CORRECT.consistency(PowConsistency::PortableSeparate).to_u64();
/// let _=zenmath::scalar::powf_with::<P>(2.0,3.0);
/// ```
#[inline(always)]
pub fn powf_with<const P: u64>(x: f32, y: f32) -> f32 {
    const {
        validate(P);
    }
    if P == 0 {
        return crate::scalar::powf(x, y);
    }
    let p = prepare::<P>(x, y);
    let out = if P & 3 == 0 {
        crate::scalar::powf(p.x, p.y)
    } else if (P >> 9) & 3 == 2 {
        formula::<P>(p.x, p.y)
    } else if working_input(p) {
        let z = polynomial_exponent::<P>(p.x.abs(), p.y);
        if (-126.0..=127.5).contains(&z) {
            mathematical_sign(p, exponential::<P>(z))
        } else {
            crate::scalar::powf(p.x, p.y)
        }
    } else {
        crate::scalar::powf(p.x, p.y)
    };
    finish::<P>(p, out)
}

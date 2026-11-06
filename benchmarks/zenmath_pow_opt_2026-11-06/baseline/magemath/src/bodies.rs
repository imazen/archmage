//! One body per op, generic over how a polynomial step rounds.
//!
//! The public `kernels` modules are thin wrappers over these, one module per
//! [`Step`]. The parameter stays private: a caller picks the contract with an
//! import path, not a type argument.
//!
//! Polynomials and operand order are magetypes' generated transcendentals. Two
//! rules keep special-value handling independent of the ISA. Finite polynomial
//! results still depend on the selected `Step`:
//!
//! - NaN input gives NaN output without depending on anything that differs by
//!   ISA. `min`, `max` and `simd_ne` treat NaN differently per backend, so a NaN
//!   lane here reaches the result through arithmetic or an ordered comparison.
//! - No float negation whose sign of zero reaches the result: `-(+0.0)` differs
//!   by ISA (`docs/site/content/magetypes/isa-quirks.md`).

use magetypes::simd::backends::F32x8Convert;
use magetypes::simd::generic::{f32x8, i32x8};

/// One polynomial step, `a * b + c`.
pub(crate) trait Step {
    fn madd<T: F32x8Convert>(a: f32x8<T>, b: f32x8<T>, c: f32x8<T>) -> f32x8<T>;
}

/// One rounding on every tier: hardware FMA or software.
pub(crate) struct Fused;

impl Step for Fused {
    #[inline(always)]
    fn madd<T: F32x8Convert>(a: f32x8<T>, b: f32x8<T>, c: f32x8<T>) -> f32x8<T> {
        a.mul_add_fused(b, c)
    }
}

/// Two roundings on every tier: IEEE multiply, then IEEE add.
pub(crate) struct Unfused;

impl Step for Unfused {
    #[inline(always)]
    fn madd<T: F32x8Convert>(a: f32x8<T>, b: f32x8<T>, c: f32x8<T>) -> f32x8<T> {
        a * b + c
    }
}

/// Backend rounding: one rounding on x86 v3/v4x and NEON, two on
/// the scalar tier and strict WASM, the engine's choice on relaxed WASM.
pub(crate) struct Relaxed;

impl Step for Relaxed {
    #[inline(always)]
    fn madd<T: F32x8Convert>(a: f32x8<T>, b: f32x8<T>, c: f32x8<T>) -> f32x8<T> {
        a.mul_add_relaxed(b, c)
    }
}

#[inline(always)]
fn splat_f32<T: F32x8Convert>(token: T, v: f32) -> f32x8<T> {
    f32x8::splat_t(token, v)
}

#[inline(always)]
fn splat_i32<T: F32x8Convert>(token: T, v: i32) -> i32x8<T> {
    i32x8::splat_t(token, v)
}

/// 0 → −inf, negative or NaN → NaN, +inf → +inf.
#[inline(always)]
fn log_specials<T: F32x8Convert>(token: T, x: f32x8<T>, r: f32x8<T>) -> f32x8<T> {
    let zero = splat_f32(token, 0.0);
    let inf = splat_f32(token, f32::INFINITY);
    // `x >= 0` is false for NaN, so one comparison handles NaN and negatives.
    let r = f32x8::blend(x.simd_ge(zero), r, splat_f32(token, f32::NAN));
    let r = f32x8::blend(x.simd_eq(zero), splat_f32(token, f32::NEG_INFINITY), r);
    f32x8::blend(x.simd_eq(inf), inf, r)
}

// ====== Exactly rounded primitives: no multiply-add, so `S` changes nothing ======

#[inline(always)]
pub(crate) fn sqrt<S: Step, T: F32x8Convert>(_token: T, x: f32x8<T>) -> f32x8<T> {
    x.sqrt()
}

#[inline(always)]
pub(crate) fn floor<S: Step, T: F32x8Convert>(_token: T, x: f32x8<T>) -> f32x8<T> {
    x.floor()
}

#[inline(always)]
pub(crate) fn ceil<S: Step, T: F32x8Convert>(_token: T, x: f32x8<T>) -> f32x8<T> {
    x.ceil()
}

#[inline(always)]
pub(crate) fn roundeven<S: Step, T: F32x8Convert>(_token: T, x: f32x8<T>) -> f32x8<T> {
    x.round()
}

// ====== lowp ======

#[inline(always)]
pub(crate) fn log2_lowp_unchecked<S: Step, T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    const OFFSET: i32 = 0x3f2a_aaab;
    const P0: f32 = -1.850_383_3e-6;
    const P1: f32 = 1.428_716_1;
    const P2: f32 = 0.742_458_7;
    const Q0: f32 = 0.990_328_14;
    const Q1: f32 = 1.009_671_8;
    const Q2: f32 = 0.174_093_43;

    let x_bits = x.bitcast_to_i32();
    let exp_shifted = (x_bits - splat_i32(token, OFFSET)).shr_arithmetic_const::<23>();
    let mantissa = (x_bits - exp_shifted.shl_const::<23>()).bitcast_to_f32();
    let m = mantissa - splat_f32(token, 1.0);
    let yp = S::madd(splat_f32(token, P2), m, splat_f32(token, P1));
    let yp = S::madd(yp, m, splat_f32(token, P0));
    let yq = S::madd(splat_f32(token, Q2), m, splat_f32(token, Q1));
    let yq = S::madd(yq, m, splat_f32(token, Q0));
    yp / yq + exp_shifted.to_f32()
}

#[inline(always)]
pub(crate) fn log2_lowp<S: Step, T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    log_specials(token, x, log2_lowp_unchecked::<S, T>(token, x))
}

#[inline(always)]
pub(crate) fn exp2_lowp_unchecked<S: Step, T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    const C1: f32 = core::f32::consts::LN_2;
    const C2: f32 = 0.240_226_5;
    const C3: f32 = 0.055_504_11;

    let xi = x.floor();
    let xf = x - xi;
    let poly = S::madd(splat_f32(token, C3), xf, splat_f32(token, C2));
    let poly = S::madd(poly, xf, splat_f32(token, C1));
    let poly = S::madd(poly, xf, splat_f32(token, 1.0));
    let scale = (xi.to_i32_round() + splat_i32(token, 127))
        .shl_const::<23>()
        .bitcast_to_f32();
    poly * scale
}

#[inline(always)]
pub(crate) fn exp2_lowp<S: Step, T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    let lo = splat_f32(token, -126.0);
    let hi = splat_f32(token, 127.99);
    // Comparisons, not `max`/`min`: a NaN lane stays NaN through the clamp.
    let clamped = f32x8::blend(x.simd_lt(lo), lo, x);
    let clamped = f32x8::blend(x.simd_gt(hi), hi, clamped);
    exp2_lowp_unchecked::<S, T>(token, clamped)
}

#[inline(always)]
pub(crate) fn ln_lowp_unchecked<S: Step, T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    log2_lowp_unchecked::<S, T>(token, x) * splat_f32(token, core::f32::consts::LN_2)
}

#[inline(always)]
pub(crate) fn ln_lowp<S: Step, T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    log2_lowp::<S, T>(token, x) * splat_f32(token, core::f32::consts::LN_2)
}

#[inline(always)]
pub(crate) fn exp_lowp_unchecked<S: Step, T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    exp2_lowp_unchecked::<S, T>(token, x * splat_f32(token, core::f32::consts::LOG2_E))
}

#[inline(always)]
pub(crate) fn exp_lowp<S: Step, T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    exp2_lowp::<S, T>(token, x * splat_f32(token, core::f32::consts::LOG2_E))
}

#[inline(always)]
pub(crate) fn log10_lowp_unchecked<S: Step, T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    log2_lowp_unchecked::<S, T>(token, x)
        * splat_f32(token, core::f32::consts::LN_2 / core::f32::consts::LN_10)
}

#[inline(always)]
pub(crate) fn log10_lowp<S: Step, T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    log2_lowp::<S, T>(token, x)
        * splat_f32(token, core::f32::consts::LN_2 / core::f32::consts::LN_10)
}

#[inline(always)]
pub(crate) fn pow_lowp_unchecked<S: Step, T: F32x8Convert>(
    token: T,
    x: f32x8<T>,
    n: f32,
) -> f32x8<T> {
    exp2_lowp_unchecked::<S, T>(
        token,
        log2_lowp_unchecked::<S, T>(token, x) * splat_f32(token, n),
    )
}

#[inline(always)]
pub(crate) fn pow_lowp<S: Step, T: F32x8Convert>(token: T, x: f32x8<T>, n: f32) -> f32x8<T> {
    let zero = splat_f32(token, 0.0);
    let r = exp2_lowp::<S, T>(token, log2_lowp::<S, T>(token, x) * splat_f32(token, n));
    f32x8::blend(x.simd_eq(zero), zero, r)
}

#[inline(always)]
pub(crate) fn cbrt_lowp<S: Step, T: F32x8Convert>(_token: T, x: f32x8<T>) -> f32x8<T> {
    // No multiply-add: plain IEEE operations already give the same bits
    // on every backend.
    x.cbrt_lowp()
}

// ====== midp ======

#[inline(always)]
pub(crate) fn log2_midp_unchecked<S: Step, T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    const SQRT2_OVER_2: i32 = 0x3f35_04f3;
    const ONE_BITS: i32 = 0x3f80_0000;
    const MANTISSA_MASK: i32 = 0x007f_ffff;
    const C0: f32 = 2.885_39;
    const C1: f32 = 0.961_800_76;
    const C2: f32 = 0.576_974_45;
    const C3: f32 = 0.434_411_97;

    let adjusted = x.bitcast_to_i32() + splat_i32(token, ONE_BITS - SQRT2_OVER_2);
    let n = (adjusted.shr_arithmetic_const::<23>() - splat_i32(token, 127)).to_f32();
    let a = ((adjusted & splat_i32(token, MANTISSA_MASK)) + splat_i32(token, SQRT2_OVER_2))
        .bitcast_to_f32();
    let one = splat_f32(token, 1.0);
    let y = (a - one) / (a + one);
    let y2 = y * y;
    let poly = S::madd(splat_f32(token, C3), y2, splat_f32(token, C2));
    let poly = S::madd(poly, y2, splat_f32(token, C1));
    let poly = S::madd(poly, y2, splat_f32(token, C0));
    S::madd(poly, y, n)
}

#[inline(always)]
pub(crate) fn log2_midp<S: Step, T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    log_specials(token, x, log2_midp_unchecked::<S, T>(token, x))
}

#[inline(always)]
pub(crate) fn exp2_midp_unchecked<S: Step, T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    const C1: f32 = core::f32::consts::LN_2;
    const C2: f32 = 0.240_226_46;
    const C3: f32 = 0.055_504_545;
    const C4: f32 = 0.009_618_055;
    const C5: f32 = 0.001_333_37;
    const C6: f32 = 0.000_154_47;

    let xi = x.round().min(splat_f32(token, 127.0));
    let xf = x - xi;
    let poly = S::madd(splat_f32(token, C6), xf, splat_f32(token, C5));
    let poly = S::madd(poly, xf, splat_f32(token, C4));
    let poly = S::madd(poly, xf, splat_f32(token, C3));
    let poly = S::madd(poly, xf, splat_f32(token, C2));
    let poly = S::madd(poly, xf, splat_f32(token, C1));
    let poly = S::madd(poly, xf, splat_f32(token, 1.0));
    let scale = (xi.to_i32_round() + splat_i32(token, 127))
        .shl_const::<23>()
        .bitcast_to_f32();
    poly * scale
}

#[inline(always)]
pub(crate) fn exp2_midp<S: Step, T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    // No clamp before the polynomial: the blends below replace every lane a clamp
    // would change, whatever it computed. A NaN lane stays NaN without help, since
    // `xf` is NaN on every tier however `min` treated it.
    let r = exp2_midp_unchecked::<S, T>(token, x);
    let r = f32x8::blend(
        x.simd_lt(splat_f32(token, -126.0)),
        splat_f32(token, 0.0),
        r,
    );
    f32x8::blend(
        x.simd_ge(splat_f32(token, 128.0)),
        splat_f32(token, f32::INFINITY),
        r,
    )
}

#[inline(always)]
pub(crate) fn ln_midp_unchecked<S: Step, T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    log2_midp_unchecked::<S, T>(token, x) * splat_f32(token, core::f32::consts::LN_2)
}

#[inline(always)]
pub(crate) fn ln_midp<S: Step, T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    log2_midp::<S, T>(token, x) * splat_f32(token, core::f32::consts::LN_2)
}

#[inline(always)]
pub(crate) fn exp_midp_unchecked<S: Step, T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    exp2_midp_unchecked::<S, T>(token, x * splat_f32(token, core::f32::consts::LOG2_E))
}

#[inline(always)]
pub(crate) fn exp_midp<S: Step, T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    exp2_midp::<S, T>(token, x * splat_f32(token, core::f32::consts::LOG2_E))
}

#[inline(always)]
pub(crate) fn log10_midp_unchecked<S: Step, T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    log2_midp_unchecked::<S, T>(token, x)
        * splat_f32(token, core::f32::consts::LN_2 / core::f32::consts::LN_10)
}

#[inline(always)]
pub(crate) fn log10_midp<S: Step, T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    log2_midp::<S, T>(token, x)
        * splat_f32(token, core::f32::consts::LN_2 / core::f32::consts::LN_10)
}

#[inline(always)]
pub(crate) fn pow_midp_unchecked<S: Step, T: F32x8Convert>(
    token: T,
    x: f32x8<T>,
    n: f32,
) -> f32x8<T> {
    exp2_midp_unchecked::<S, T>(
        token,
        log2_midp_unchecked::<S, T>(token, x) * splat_f32(token, n),
    )
}

#[inline(always)]
pub(crate) fn pow_midp<S: Step, T: F32x8Convert>(token: T, x: f32x8<T>, n: f32) -> f32x8<T> {
    exp2_midp::<S, T>(token, log2_midp::<S, T>(token, x) * splat_f32(token, n))
}

#[inline(always)]
pub(crate) fn cbrt_midp<S: Step, T: F32x8Convert>(_token: T, x: f32x8<T>) -> f32x8<T> {
    // See `cbrt_lowp`: no multiply-add to fuse.
    x.cbrt_midp()
}

#[inline(always)]
pub(crate) fn cbrt_midp_precise<S: Step, T: F32x8Convert>(_token: T, x: f32x8<T>) -> f32x8<T> {
    // See `cbrt_lowp`: no multiply-add to fuse.
    x.cbrt_midp_precise()
}

#[inline(always)]
pub(crate) fn sigmoid_midp<S: Step, T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    // The sign of a zero `-x` cannot reach the result: exp(±0) is exactly 1.
    let one = splat_f32(token, 1.0);
    one / (exp_midp::<S, T>(token, -x) + one)
}

#[inline(always)]
pub(crate) fn silu_midp<S: Step, T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    x * sigmoid_midp::<S, T>(token, x)
}

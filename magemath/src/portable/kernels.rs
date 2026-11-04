//! Token-generic vector kernels: the single implementation behind
//! [`slice`](super::slice) and [`array`](super::array).
//!
//! Each function takes the token and an [`f32x8`] and is `#[inline(always)]`,
//! so a chain of them called inside one `#[attune]` body becomes one feature
//! region and the values stay in vector registers:
//!
//! ```
//! use archmage::prelude::*;
//! use magemath::portable::kernels;
//!
//! #[attune(_*_t, define(f32x8))]
//! fn gamma(token: Token, a: [f32; 8]) -> [f32; 8] {
//!     let x = f32x8::from_array_t(token, a);
//!     kernels::exp2_midp(token, kernels::log2_midp(token, x) * f32x8::splat_t(token, 2.4))
//!         .to_array()
//! }
//!
//! let out = attuned!(gamma([0.5; 8]));
//! assert_eq!(out[0], zenmath::scalar::pow_midp_f32(0.5, 2.4));
//! ```
//!
//! ## What makes these portable
//!
//! Polynomials and operand order are magetypes' generated transcendentals, so
//! on the tiers where `mul_add_relaxed` fuses (x86 v3/v4x, NEON) in-domain
//! results equal the `f32x8` methods bit for bit. Two things differ on purpose:
//!
//! - Every polynomial step is [`mul_add_fused`](f32x8::mul_add_fused), which
//!   rounds once on every backend. On x86 and NEON it is the same instruction
//!   as `mul_add_relaxed`. On the scalar tier and strict WASM it is software,
//!   and relaxed WASM uses the engine's madd once a cached probe shows it fuses.
//! - NaN input gives NaN output, without depending on anything that differs by
//!   ISA. `min`, `max` and `simd_ne` treat NaN differently per backend, so a NaN
//!   lane here reaches the result through arithmetic or an ordered comparison.
//!
//! Do not add a float negation whose sign of zero reaches the result, either:
//! `-(+0.0)` differs by ISA (`docs/site/content/magetypes/isa-quirks.md`).
//!
//! ## `_unchecked`
//!
//! The `*_unchecked` kernels skip edge handling. Use them in a hot loop whose
//! inputs you know are in the stated domain; `docs/DESIGN.md` has what that saves.
//! In the domain they return the same bits as the checked kernel on every tier.
//! Outside it the result is unspecified and differs by tier; it is never undefined
//! behavior.

use magetypes::simd::backends::F32x8Convert;
use magetypes::simd::generic::{f32x8, i32x8};

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

// ====== Exactly rounded primitives ======

/// Square root, correctly rounded.
#[inline(always)]
pub fn sqrt<T: F32x8Convert>(_token: T, x: f32x8<T>) -> f32x8<T> {
    x.sqrt()
}

/// Round toward negative infinity.
#[inline(always)]
pub fn floor<T: F32x8Convert>(_token: T, x: f32x8<T>) -> f32x8<T> {
    x.floor()
}

/// Round toward positive infinity.
#[inline(always)]
pub fn ceil<T: F32x8Convert>(_token: T, x: f32x8<T>) -> f32x8<T> {
    x.ceil()
}

/// Round to nearest, ties to even. Not `f32::round`, which ties away from zero.
#[inline(always)]
pub fn roundeven<T: F32x8Convert>(_token: T, x: f32x8<T>) -> f32x8<T> {
    x.round()
}

// ====== lowp ======

/// [`log2_lowp`] for positive finite `x`.
#[inline(always)]
pub fn log2_lowp_unchecked<T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
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
    let yp = splat_f32(token, P2)
        .mul_add_fused(m, splat_f32(token, P1))
        .mul_add_fused(m, splat_f32(token, P0));
    let yq = splat_f32(token, Q2)
        .mul_add_fused(m, splat_f32(token, Q1))
        .mul_add_fused(m, splat_f32(token, Q0));
    yp / yq + exp_shifted.to_f32()
}

/// `zenmath::scalar::log2_lowp_f32` per lane.
#[inline(always)]
pub fn log2_lowp<T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    log_specials(token, x, log2_lowp_unchecked(token, x))
}

/// [`exp2_lowp`] for `x` in [-126, 127.99].
#[inline(always)]
pub fn exp2_lowp_unchecked<T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    const C1: f32 = core::f32::consts::LN_2;
    const C2: f32 = 0.240_226_5;
    const C3: f32 = 0.055_504_11;

    let xi = x.floor();
    let xf = x - xi;
    let poly = splat_f32(token, C3)
        .mul_add_fused(xf, splat_f32(token, C2))
        .mul_add_fused(xf, splat_f32(token, C1))
        .mul_add_fused(xf, splat_f32(token, 1.0));
    let scale = (xi.to_i32_round() + splat_i32(token, 127))
        .shl_const::<23>()
        .bitcast_to_f32();
    poly * scale
}

/// `zenmath::scalar::exp2_lowp_f32` per lane.
#[inline(always)]
pub fn exp2_lowp<T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    let lo = splat_f32(token, -126.0);
    let hi = splat_f32(token, 127.99);
    // Comparisons, not `max`/`min`: a NaN lane stays NaN through the clamp.
    let clamped = f32x8::blend(x.simd_lt(lo), lo, x);
    let clamped = f32x8::blend(x.simd_gt(hi), hi, clamped);
    exp2_lowp_unchecked(token, clamped)
}

/// [`ln_lowp`] for positive finite `x`.
#[inline(always)]
pub fn ln_lowp_unchecked<T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    log2_lowp_unchecked(token, x) * splat_f32(token, core::f32::consts::LN_2)
}

/// `zenmath::scalar::ln_lowp_f32` per lane.
#[inline(always)]
pub fn ln_lowp<T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    log2_lowp(token, x) * splat_f32(token, core::f32::consts::LN_2)
}

/// [`exp_lowp`] for `x · log2(e)` in [-126, 127.99].
#[inline(always)]
pub fn exp_lowp_unchecked<T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    exp2_lowp_unchecked(token, x * splat_f32(token, core::f32::consts::LOG2_E))
}

/// `zenmath::scalar::exp_lowp_f32` per lane.
#[inline(always)]
pub fn exp_lowp<T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    exp2_lowp(token, x * splat_f32(token, core::f32::consts::LOG2_E))
}

/// [`log10_lowp`] for positive finite `x`.
#[inline(always)]
pub fn log10_lowp_unchecked<T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    log2_lowp_unchecked(token, x)
        * splat_f32(token, core::f32::consts::LN_2 / core::f32::consts::LN_10)
}

/// `zenmath::scalar::log10_lowp_f32` per lane.
#[inline(always)]
pub fn log10_lowp<T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    log2_lowp(token, x) * splat_f32(token, core::f32::consts::LN_2 / core::f32::consts::LN_10)
}

/// [`pow_lowp`] for positive finite `x` with `log2(x) · n` in [-126, 127.99].
#[inline(always)]
pub fn pow_lowp_unchecked<T: F32x8Convert>(token: T, x: f32x8<T>, n: f32) -> f32x8<T> {
    exp2_lowp_unchecked(token, log2_lowp_unchecked(token, x) * splat_f32(token, n))
}

/// `zenmath::scalar::pow_lowp_f32` per lane, with one exponent for all lanes.
#[inline(always)]
pub fn pow_lowp<T: F32x8Convert>(token: T, x: f32x8<T>, n: f32) -> f32x8<T> {
    let zero = splat_f32(token, 0.0);
    let r = exp2_lowp(token, log2_lowp(token, x) * splat_f32(token, n));
    f32x8::blend(x.simd_eq(zero), zero, r)
}

/// `zenmath::scalar::cbrt_lowp_f32` per lane.
#[inline(always)]
pub fn cbrt_lowp<T: F32x8Convert>(_token: T, x: f32x8<T>) -> f32x8<T> {
    // No multiply-add: plain IEEE operations already give the same bits
    // on every backend.
    x.cbrt_lowp()
}

// ====== midp ======

/// [`log2_midp`] for positive finite `x`.
#[inline(always)]
pub fn log2_midp_unchecked<T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
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
    let poly = splat_f32(token, C3)
        .mul_add_fused(y2, splat_f32(token, C2))
        .mul_add_fused(y2, splat_f32(token, C1))
        .mul_add_fused(y2, splat_f32(token, C0));
    poly.mul_add_fused(y, n)
}

/// `zenmath::scalar::log2_midp_f32` per lane.
#[inline(always)]
pub fn log2_midp<T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    log_specials(token, x, log2_midp_unchecked(token, x))
}

/// [`exp2_midp`] for `x` in [-126, 128).
#[inline(always)]
pub fn exp2_midp_unchecked<T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    const C1: f32 = core::f32::consts::LN_2;
    const C2: f32 = 0.240_226_46;
    const C3: f32 = 0.055_504_545;
    const C4: f32 = 0.009_618_055;
    const C5: f32 = 0.001_333_37;
    const C6: f32 = 0.000_154_47;

    let xi = x.round().min(splat_f32(token, 127.0));
    let xf = x - xi;
    let poly = splat_f32(token, C6)
        .mul_add_fused(xf, splat_f32(token, C5))
        .mul_add_fused(xf, splat_f32(token, C4))
        .mul_add_fused(xf, splat_f32(token, C3))
        .mul_add_fused(xf, splat_f32(token, C2))
        .mul_add_fused(xf, splat_f32(token, C1))
        .mul_add_fused(xf, splat_f32(token, 1.0));
    let scale = (xi.to_i32_round() + splat_i32(token, 127))
        .shl_const::<23>()
        .bitcast_to_f32();
    poly * scale
}

/// `zenmath::scalar::exp2_midp_f32` per lane.
#[inline(always)]
pub fn exp2_midp<T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    // No clamp before the polynomial: the blends below replace every lane a clamp
    // would change, whatever it computed. A NaN lane stays NaN without help, since
    // `xf` is NaN on every tier however `min` treated it.
    let r = exp2_midp_unchecked(token, x);
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

/// [`ln_midp`] for positive finite `x`.
#[inline(always)]
pub fn ln_midp_unchecked<T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    log2_midp_unchecked(token, x) * splat_f32(token, core::f32::consts::LN_2)
}

/// `zenmath::scalar::ln_midp_f32` per lane.
#[inline(always)]
pub fn ln_midp<T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    log2_midp(token, x) * splat_f32(token, core::f32::consts::LN_2)
}

/// [`exp_midp`] for `x · log2(e)` in [-126, 128).
#[inline(always)]
pub fn exp_midp_unchecked<T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    exp2_midp_unchecked(token, x * splat_f32(token, core::f32::consts::LOG2_E))
}

/// `zenmath::scalar::exp_midp_f32` per lane.
#[inline(always)]
pub fn exp_midp<T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    exp2_midp(token, x * splat_f32(token, core::f32::consts::LOG2_E))
}

/// [`log10_midp`] for positive finite `x`.
#[inline(always)]
pub fn log10_midp_unchecked<T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    log2_midp_unchecked(token, x)
        * splat_f32(token, core::f32::consts::LN_2 / core::f32::consts::LN_10)
}

/// `zenmath::scalar::log10_midp_f32` per lane.
#[inline(always)]
pub fn log10_midp<T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    log2_midp(token, x) * splat_f32(token, core::f32::consts::LN_2 / core::f32::consts::LN_10)
}

/// [`pow_midp`] for positive finite `x` with `log2(x) · n` in [-126, 128).
#[inline(always)]
pub fn pow_midp_unchecked<T: F32x8Convert>(token: T, x: f32x8<T>, n: f32) -> f32x8<T> {
    exp2_midp_unchecked(token, log2_midp_unchecked(token, x) * splat_f32(token, n))
}

/// `zenmath::scalar::pow_midp_f32` per lane, with one exponent for all lanes.
#[inline(always)]
pub fn pow_midp<T: F32x8Convert>(token: T, x: f32x8<T>, n: f32) -> f32x8<T> {
    exp2_midp(token, log2_midp(token, x) * splat_f32(token, n))
}

/// `zenmath::scalar::cbrt_midp_f32` per lane.
#[inline(always)]
pub fn cbrt_midp<T: F32x8Convert>(_token: T, x: f32x8<T>) -> f32x8<T> {
    // See `cbrt_lowp`: no multiply-add to fuse.
    x.cbrt_midp()
}

/// `zenmath::scalar::cbrt_midp_precise_f32` per lane.
#[inline(always)]
pub fn cbrt_midp_precise<T: F32x8Convert>(_token: T, x: f32x8<T>) -> f32x8<T> {
    // See `cbrt_lowp`: no multiply-add to fuse.
    x.cbrt_midp_precise()
}

/// `zenmath::scalar::sigmoid_midp_f32` per lane.
#[inline(always)]
pub fn sigmoid_midp<T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    // The sign of a zero `-x` cannot reach the result: exp(±0) is exactly 1.
    let one = splat_f32(token, 1.0);
    one / (exp_midp(token, -x) + one)
}

/// `zenmath::scalar::silu_midp_f32` per lane.
#[inline(always)]
pub fn silu_midp<T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
    x * sigmoid_midp(token, x)
}

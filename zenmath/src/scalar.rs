//! Portable scalar math for `no_std` — the zenmath scalar core.
//!
//! Ported from `magetypes::nostd_math` (itself derived from
//! [compiler-builtins/libm](https://github.com/rust-lang/compiler-builtins/blob/main/libm/src/math/generic/sqrt.rs),
//! MIT license, origin: musl `src/math/sqrt.c`). These functions exist so
//! callers never need `std` or a platform `libm`: every result is defined
//! bit-for-bit on every target.
//!
//! Contract: every function is bit-exact with its `std` counterpart for all
//! finite inputs and ±Inf where "bit-exact" is claimed (sqrt, floor, ceil,
//! round, roundeven). NaN propagation preserves NaN-ness but payloads may
//! differ (hardware ops quiet sNaN→qNaN; software implementations don't).
//! The transcendental approximations (`*_lowp_f64`, `*_lowp_f32`, `*_midp_f32`,
//! `cbrt_*`) carry their own documented error bounds — they are approximation
//! tiers, not libm replacements.
//!
//! The f32 `*_lowp`/`*_midp` functions are the scalar form of
//! `magemath::portable`: every polynomial step is one [`fmaf`] and every
//! result, specials included, matches those kernels bit for bit on every tier
//! (NaN results match in NaN-ness only). `tests/parity.rs` and `tests/golden.rs`
//! in `magemath` enforce that.
//!
//! `fmaf` is correctly rounded on every target: `std` builds with compile-time x86 FMA or AArch64 NEON use
//! hardware `mul_add`; other builds use a software path (exact f64
//! product, TwoSum error recovery, round-to-odd narrowing — ported from
//! `magetypes::soft_fma`, MIT). `fma` (f64) lands with the f64 midp
//! mirrors that need it.
//!
//! [`unfused`] holds the same functions with each polynomial step a plain
//! multiply and then add: two roundings, no FMA, on every target.

// ============================================================================
// Square root — Goldschmidt iteration
// ============================================================================

/// High half of widening u32 multiply: `(a * b) >> 32`.
#[inline(always)]
fn wmulh32(a: u32, b: u32) -> u32 {
    ((a as u64).wrapping_mul(b as u64) >> 32) as u32
}

/// High half of widening u64 multiply: `(a * b) >> 64`.
#[inline(always)]
fn wmulh64(a: u64, b: u64) -> u64 {
    ((a as u128).wrapping_mul(b as u128) >> 64) as u64
}

/// Reciprocal square root lookup table (U0.16). Index is 7 bits:
/// 1 bit from exponent + 6 bits from significand.
#[rustfmt::skip]
static RSQRT_TAB: [u16; 128] = [
    0xb451, 0xb2f0, 0xb196, 0xb044, 0xaef9, 0xadb6, 0xac79, 0xab43,
    0xaa14, 0xa8eb, 0xa7c8, 0xa6aa, 0xa592, 0xa480, 0xa373, 0xa26b,
    0xa168, 0xa06a, 0x9f70, 0x9e7b, 0x9d8a, 0x9c9d, 0x9bb5, 0x9ad1,
    0x99f0, 0x9913, 0x983a, 0x9765, 0x9693, 0x95c4, 0x94f8, 0x9430,
    0x936b, 0x92a9, 0x91ea, 0x912e, 0x9075, 0x8fbe, 0x8f0a, 0x8e59,
    0x8daa, 0x8cfe, 0x8c54, 0x8bac, 0x8b07, 0x8a64, 0x89c4, 0x8925,
    0x8889, 0x87ee, 0x8756, 0x86c0, 0x862b, 0x8599, 0x8508, 0x8479,
    0x83ec, 0x8361, 0x82d8, 0x8250, 0x81c9, 0x8145, 0x80c2, 0x8040,
    0xff02, 0xfd0e, 0xfb25, 0xf947, 0xf773, 0xf5aa, 0xf3ea, 0xf234,
    0xf087, 0xeee3, 0xed47, 0xebb3, 0xea27, 0xe8a3, 0xe727, 0xe5b2,
    0xe443, 0xe2dc, 0xe17a, 0xe020, 0xdecb, 0xdd7d, 0xdc34, 0xdaf1,
    0xd9b3, 0xd87b, 0xd748, 0xd61a, 0xd4f1, 0xd3cd, 0xd2ad, 0xd192,
    0xd07b, 0xcf69, 0xce5b, 0xcd51, 0xcc4a, 0xcb48, 0xca4a, 0xc94f,
    0xc858, 0xc764, 0xc674, 0xc587, 0xc49d, 0xc3b7, 0xc2d4, 0xc1f4,
    0xc116, 0xc03c, 0xbf65, 0xbe90, 0xbdbe, 0xbcef, 0xbc23, 0xbb59,
    0xba91, 0xb9cc, 0xb90a, 0xb84a, 0xb78c, 0xb6d0, 0xb617, 0xb560,
];

/// f32 square root — correctly rounded, bit-exact with hardware sqrtss.
///
/// Goldschmidt iteration in u32 integer arithmetic: table lookup for initial
/// `1/sqrt(m)` estimate, then 3 iterations computing `sqrt(m)` and `1/sqrt(m)`
/// simultaneously. No floating-point division required.
#[inline(always)]
pub fn sqrtf(x: f32) -> f32 {
    let mut ix = x.to_bits();

    // Special cases: subnormal, zero, negative, NaN, infinity.
    // Test: (biased_exp - 1) >= (max_exp - 1) catches exp=0 and exp=255.
    if ix.wrapping_sub(0x0080_0000) >= 0x7F00_0000 {
        if ix << 1 == 0 {
            return x; // ±0
        }
        if ix == 0x7F80_0000 {
            return x; // +inf
        }
        if ix > 0x7F80_0000 {
            return f32::NAN; // NaN or negative (including -inf)
        }
        // Subnormal: normalize by multiplying by 2^23, then adjust exponent
        let scaled = x * f32::from_bits((23 + 127) << 23);
        ix = scaled.to_bits();
        ix = ix.wrapping_sub(23 << 23);
    }

    // Argument reduction: x = 4^e * m, where m ∈ [1, 4).
    // `even` is true when the exponent's lowest bit is set.
    let even = ix & (1 << 23) != 0;

    // Result exponent: halve the input exponent with bias correction.
    let mut ey = ix >> 1;
    ey = ey.wrapping_add((0x7F80_0000 ^ 0x4000_0000) >> 1);
    ey &= 0x7F80_0000;

    // Fixed-point U2.30 mantissa (2 integer bits, 30 fractional bits).
    // Even exponent: m ∈ [1, 2), shifted one less to keep in range.
    // Odd exponent: m ∈ [2, 4), shifted fully with implicit bit restored.
    let m = if even {
        (ix << 7) & 0x7FFF_FFFF
    } else {
        (ix << 8) | 0x8000_0000
    };

    // Table lookup: 7-bit index from lowest exponent bit + top 6 significand bits.
    let i = ((ix >> 17) & 0x7F) as usize;

    // Initial estimates in U0.32 (reciprocal) and U2.30 (sqrt).
    let mut r: u32 = (RSQRT_TAB[i] as u32) << 16;
    let mut s: u32 = m;

    // 3 Goldschmidt iterations at u32 width.
    // r approaches 1/sqrt(m), s approaches sqrt(m).
    let three: u32 = 3 << 30; // 3.0 in U2.30
    let mut u = r; // first iteration uses r as initial u

    // Unrolled: the compiler needs to see constant iteration counts.
    // Iteration 0: s = m*r (first uses u=r)
    s = wmulh32(s, u);
    let d = wmulh32(s, r);
    u = three.wrapping_sub(d);
    r = wmulh32(r, u) << 1;

    // Iteration 1: s = s*u/2
    s = wmulh32(s, u) << 1; // non-final, non-first: shift
    let d = wmulh32(s, r);
    u = three.wrapping_sub(d);
    r = wmulh32(r, u) << 1;

    // Iteration 2 (final): s = s*u/2, but skip the /2 (combined with later shift)
    s = wmulh32(s, u); // final iteration: no shift
    let d = wmulh32(s, r);
    u = three.wrapping_sub(d);
    r = wmulh32(r, u) << 1;
    let _ = (r, u); // suppress unused warnings

    // Shift from U3.29 to mantissa position (EXP_BITS - 2 = 6).
    let mut result = s >> 6;

    // Rounding: compare (result)² against the original mantissa to decide
    // whether to round up. Pure integer comparison, no float ops.
    let d0 = (m << 16).wrapping_sub(result.wrapping_mul(result));
    let d1 = result.wrapping_sub(d0);
    result = result.wrapping_add(d1 >> 31);
    result &= 0x007F_FFFF; // SIG_MASK

    // Combine mantissa with exponent.
    result |= ey;
    f32::from_bits(result)
}

/// f64 square root — correctly rounded, bit-exact with hardware sqrtsd.
///
/// Goldschmidt iteration: 2 rounds at u32 width, then 2 rounds at u64 width.
/// Uses a 128-entry reciprocal sqrt lookup table for the initial estimate.
#[inline(always)]
pub fn sqrt(x: f64) -> f64 {
    let mut ix = x.to_bits();

    // Extract top 12 bits (sign + exponent) for fast special-case detection.
    let mut top = (ix >> 52) as u32;
    if top.wrapping_sub(1) >= 0x7FE {
        // ±0
        if ix << 1 == 0 {
            return x;
        }
        // +inf
        if ix == 0x7FF0_0000_0000_0000 {
            return x;
        }
        // NaN or negative
        if ix > 0x7FF0_0000_0000_0000 {
            return f64::NAN;
        }
        // Subnormal: normalize by multiplying by 2^52
        let scaled = x * f64::from_bits(((52 + 1023) as u64) << 52);
        ix = scaled.to_bits();
        top = (ix >> 52) as u32;
        top = top.wrapping_sub(52);
    }

    // Argument reduction: x = 4^e * m, m ∈ [1, 4) as U2.62 fixed point.
    let mut e = top;
    let mut m: u64 = (ix | 0x0010_0000_0000_0000) << 11; // implicit bit + shift to U2.62
    if e & 1 != 0 {
        m >>= 1; // odd exponent: m ∈ [1, 2) instead of [2, 4)
    }
    e = (e.wrapping_add(0x3FF)) >> 1; // result exponent: (e + bias) / 2

    // Table lookup: 7-bit index from lowest exponent bit + top 6 significand bits.
    let i = ((ix >> 46) & 0x7F) as usize;

    // Phase 1: 2 Goldschmidt iterations at u32 width.
    let mut r32: u32 = (RSQRT_TAB[i] as u32) << 16;
    let mut s32: u32 = (m >> 32) as u32; // upper 32 bits of m
    let three32: u32 = 3 << 30;
    let mut u32v = r32;

    // Iteration 0
    s32 = wmulh32(s32, u32v);
    let d = wmulh32(s32, r32);
    u32v = three32.wrapping_sub(d);
    r32 = wmulh32(r32, u32v) << 1;

    // Iteration 1 (non-final set: shift s)
    s32 = wmulh32(s32, u32v) << 1;
    let d = wmulh32(s32, r32);
    u32v = three32.wrapping_sub(d);
    r32 = wmulh32(r32, u32v) << 1;
    let _ = (s32, u32v); // only r32 carries forward

    // Phase 2: widen to u64, 2 Goldschmidt iterations at full width.
    let mut r64: u64 = (r32 as u64) << 32;
    let mut s64: u64 = m; // full U2.62 mantissa
    let three64: u64 = 3 << 62;
    let mut u64v = r64;

    // Iteration 0
    s64 = wmulh64(s64, u64v);
    let d = wmulh64(s64, r64);
    u64v = three64.wrapping_sub(d);
    r64 = wmulh64(r64, u64v) << 1;

    // Iteration 1 (final: no shift on s)
    s64 = wmulh64(s64, u64v); // final iteration: skip /2
    let d = wmulh64(s64, r64);
    u64v = three64.wrapping_sub(d);
    r64 = wmulh64(r64, u64v) << 1;
    let _ = (r64, u64v);

    // Shift from U3.61 to mantissa position (EXP_BITS - 2 = 9).
    let mut result = s64 >> 9;

    // Rounding via integer comparison of result² against original mantissa.
    let d0 = (m << 42).wrapping_sub(result.wrapping_mul(result));
    let d1 = result.wrapping_sub(d0);
    result = result.wrapping_add(d1 >> 63);
    result &= 0x000F_FFFF_FFFF_FFFF; // SIG_MASK

    // Combine mantissa with exponent.
    result |= (e as u64) << 52;
    f64::from_bits(result)
}

/// f32 floor: largest integer ≤ x.
#[inline(always)]
pub fn floorf(x: f32) -> f32 {
    if x.is_nan() || x.is_infinite() {
        return x;
    }
    // Preserve sign of zero (i32 cast loses -0.0)
    if x == 0.0 {
        return x;
    }

    // For values beyond i32 range, f32 can't represent fractional parts anyway
    // (f32 has 24 mantissa bits, so |x| >= 2^23 means x is already integral)
    const LIMIT: f32 = (1u32 << 23) as f32; // 8388608.0
    if x >= LIMIT || x <= -LIMIT {
        return x;
    }

    let trunc = x as i32 as f32;
    if trunc > x { trunc - 1.0 } else { trunc }
}

/// f64 floor: largest integer ≤ x.
#[inline(always)]
pub fn floor(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return x;
    }
    if x == 0.0 {
        return x;
    }

    const LIMIT: f64 = (1u64 << 52) as f64;
    if x >= LIMIT || x <= -LIMIT {
        return x;
    }

    let trunc = x as i64 as f64;
    if trunc > x { trunc - 1.0 } else { trunc }
}

/// f32 ceil: smallest integer ≥ x.
#[inline(always)]
pub fn ceilf(x: f32) -> f32 {
    -floorf(-x)
}

/// f64 ceil: smallest integer ≥ x.
#[inline(always)]
pub fn ceil(x: f64) -> f64 {
    -floor(-x)
}

/// f32 round: round to nearest, ties away from zero (matches `std` behavior).
///
/// Uses truncation + fractional comparison to avoid the precision issue where
/// `x + 0.5` rounds up incorrectly at the boundary (e.g., 0.49999997 + 0.5 = 1.0).
#[inline(always)]
pub fn roundf(x: f32) -> f32 {
    if x.is_nan() || x.is_infinite() {
        return x;
    }
    if x == 0.0 {
        return x;
    }

    // Work with absolute value, restore sign at end
    let abs_x = f32::from_bits(x.to_bits() & 0x7FFF_FFFF);
    let t = floorf(abs_x); // truncation toward zero for positive values
    let frac = abs_x - t; // exact for small values (Sterbenz), always in [0, 1)

    let abs_result = if frac >= 0.5 { t + 1.0 } else { t };

    // Restore original sign
    f32::from_bits(abs_result.to_bits() | (x.to_bits() & 0x8000_0000))
}

/// f64 round: round to nearest, ties away from zero.
#[inline(always)]
pub fn round(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return x;
    }
    if x == 0.0 {
        return x;
    }

    let abs_x = f64::from_bits(x.to_bits() & 0x7FFF_FFFF_FFFF_FFFF);
    let t = floor(abs_x);
    let frac = abs_x - t;

    let abs_result = if frac >= 0.5 { t + 1.0 } else { t };

    f64::from_bits(abs_result.to_bits() | (x.to_bits() & 0x8000_0000_0000_0000))
}

/// f32 round to nearest, ties to even (IEEE 754 default rounding mode).
///
/// Matches the rounding behavior of hardware SIMD instructions:
/// SSE/AVX `cvtps2dq`/`roundps`, NEON `vrndnq_f32`, WASM `f32x4.nearest`.
///
/// Uses the magic-number trick: adding 2^23 forces the mantissa to snap to an
/// integer (IEEE 754 addition uses round-to-nearest-even by default), then
/// subtracting 2^23 recovers the original scale. For |x| >= 2^23, f32 values
/// are already exact integers.
#[inline(always)]
pub fn roundevenf(x: f32) -> f32 {
    const MAGIC: f32 = 8_388_608.0; // 2^23
    if x.is_nan() || x.is_infinite() {
        return x;
    }
    if x == 0.0 {
        return x; // preserve sign of zero
    }
    let abs_x = f32::from_bits(x.to_bits() & 0x7FFF_FFFF);
    if abs_x >= MAGIC {
        return x; // already an integer
    }
    let result = if x > 0.0 {
        (x + MAGIC) - MAGIC
    } else {
        (x - MAGIC) + MAGIC
    };
    // Preserve the sign of the input when the result is zero.
    // The magic-number trick can produce +0.0 for negative values that round to zero
    // (e.g., -0.5 → 0.0), but IEEE 754 roundToIntegralTiesToEven preserves the sign.
    if result == 0.0 {
        f32::from_bits(result.to_bits() | (x.to_bits() & 0x8000_0000))
    } else {
        result
    }
}

/// f64 round to nearest, ties to even (IEEE 754 default rounding mode).
///
/// See [`roundevenf`] for the f32 version and design rationale.
#[inline(always)]
pub fn roundeven(x: f64) -> f64 {
    const MAGIC: f64 = 4_503_599_627_370_496.0; // 2^52
    if x.is_nan() || x.is_infinite() {
        return x;
    }
    if x == 0.0 {
        return x; // preserve sign of zero
    }
    let abs_x = f64::from_bits(x.to_bits() & 0x7FFF_FFFF_FFFF_FFFF);
    if abs_x >= MAGIC {
        return x; // already an integer
    }
    let result = if x > 0.0 {
        (x + MAGIC) - MAGIC
    } else {
        (x - MAGIC) + MAGIC
    };
    if result == 0.0 {
        f64::from_bits(result.to_bits() | (x.to_bits() & 0x8000_0000_0000_0000))
    } else {
        result
    }
}

// ============================================================================
// Transcendental approximations (lowp tier)
// ============================================================================
//
// Plain multiply-then-add polynomials on f64, ported from
// `magetypes::nostd_math`. "lowp": not bit-exact with std; about 1% maximum
// relative error. No vector form exists yet (f64 vector math waits on a
// caller), so these have no fused/unfused split to reconcile.

/// Low-precision base-2 logarithm for f64.
///
/// Rational polynomial on the mantissa.
/// Only valid for positive, finite, non-zero inputs; edge cases are handled
/// (0 → −inf, <0 → NaN, +inf → +inf, NaN → NaN).
#[inline(always)]
pub fn log2_lowp_f64(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        return if x == 0.0 {
            f64::NEG_INFINITY
        } else {
            f64::NAN
        };
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }

    const P0: f64 = -1.850_383_340_051_831e-6;
    const P1: f64 = 1.428_716_047_008_376;
    const P2: f64 = 0.742_458_733_278_206;
    const Q0: f64 = 0.990_328_142_775_907;
    const Q1: f64 = 1.009_671_857_224_115;
    const Q2: f64 = 0.174_093_430_036_669;
    const OFFSET: u64 = 0x3fe6a09e667f3bcd; // 2/3 in f64 bits

    let x_bits = x.to_bits() as i64;
    let offset = OFFSET as i64;
    let exp_bits = x_bits.wrapping_sub(offset);
    let exp_shifted = exp_bits >> 52;
    let mantissa_bits = x_bits - (exp_shifted << 52);
    let mantissa = f64::from_bits(mantissa_bits as u64);
    let exp_val = exp_shifted as f64;
    let m = mantissa - 1.0;

    let yp = P2 * m + P1;
    let yp = yp * m + P0;
    let yq = Q2 * m + Q1;
    let yq = yq * m + Q0;

    yp / yq + exp_val
}

/// Low-precision base-2 exponential (2^x) for f64.
///
/// Degree-4 polynomial on the fractional part. Inputs between the zero/inf
/// rails are clamped to [-1022, 1022]; this is not a full-range exponential
/// approximation (for example, 1023 returns 2^1022).
#[inline(always)]
pub fn exp2_lowp_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x >= 1024.0 {
        return f64::INFINITY;
    }
    if x <= -1075.0 {
        return 0.0;
    }

    const C0: f64 = 1.0;
    const C1: f64 = core::f64::consts::LN_2;
    const C2: f64 = 0.240_226_506_959_101;
    const C3: f64 = 0.055_504_108_664_822;
    const C4: f64 = 0.009_618_129_107_629;

    let x = x.clamp(-1022.0, 1022.0);

    let xi = floor(x);
    let xf = x - xi;

    let poly = C4 * xf + C3;
    let poly = poly * xf + C2;
    let poly = poly * xf + C1;
    let poly = poly * xf + C0;

    let scale = f64::from_bits(((xi as i64 + 1023) << 52) as u64);
    poly * scale
}

/// Low-precision natural logarithm for f64.
#[inline(always)]
pub fn ln_lowp_f64(x: f64) -> f64 {
    log2_lowp_f64(x) * core::f64::consts::LN_2
}

/// Low-precision natural exponential (e^x) for f64.
#[inline(always)]
pub fn exp_lowp_f64(x: f64) -> f64 {
    exp2_lowp_f64(x * core::f64::consts::LOG2_E)
}

/// Low-precision base-10 logarithm for f64.
#[inline(always)]
pub fn log10_lowp_f64(x: f64) -> f64 {
    log2_lowp_f64(x) * core::f64::consts::LOG10_2
}

/// Low-precision power function (x^n) for f64.
#[inline(always)]
pub fn pow_lowp_f64(x: f64, n: f64) -> f64 {
    exp2_lowp_f64(n * log2_lowp_f64(x))
}

// ============================================================================
// Fused multiply-add — correctly rounded everywhere
// ============================================================================
//
// Ported from `magetypes::soft_fma` (MIT). Use std mul_add only with
// compile-time hardware FMA. WASI and native no-FMA std both fail the
// independent subnormal regression in tests/scalar.rs. Other builds use the
// software path: the f32 product is exact in f64, the f64 sum's error is
// recovered by TwoSum, and round-to-odd narrowing avoids double rounding.
// Both paths produce the unique correctly-rounded result, so bits match.

/// Low 29 bits of an f64 sum's mantissa — the bits an f32 rounding drops.
#[cfg(not(all(
    feature = "std",
    any(
        target_feature = "fma",
        all(target_arch = "aarch64", target_feature = "neon")
    )
)))]
const F32_DROPPED: u32 = (1 << 29) - 1;
/// A sum sitting exactly on an f32 halfway point (bit 28 of the dropped field).
#[cfg(not(all(
    feature = "std",
    any(
        target_feature = "fma",
        all(target_arch = "aarch64", target_feature = "neon")
    )
)))]
const F32_HALFWAY: u32 = 1 << 28;
/// High word of the smallest normal f32's exponent field, biased.
#[cfg(not(all(
    feature = "std",
    any(
        target_feature = "fma",
        all(target_arch = "aarch64", target_feature = "neon")
    )
)))]
const F32_MIN_NORMAL_HIGH: u32 = ((1023 - 126) << 20) as u32;

/// Knuth's TwoSum: `s + e == a + b` exactly, `s` rounded to nearest.
#[inline(always)]
#[cfg(not(all(
    feature = "std",
    any(
        target_feature = "fma",
        all(target_arch = "aarch64", target_feature = "neon")
    )
)))]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let virtual_b = s - a;
    let e = (a - (s - virtual_b)) + (b - virtual_b);
    (s, e)
}

/// Rounds `sum + error` (an exact TwoSum pair) to odd, branch-free.
#[inline(always)]
#[cfg(not(all(
    feature = "std",
    any(
        target_feature = "fma",
        all(target_arch = "aarch64", target_feature = "neon")
    )
)))]
fn round_to_odd(sum: f64, error: f64, apply: bool) -> f64 {
    let inexact = apply & (error != 0.0);
    let overshot = inexact & ((error > 0.0) != (sum > 0.0));
    let bits = sum.to_bits() - u64::from(overshot);
    f64::from_bits(bits | u64::from(inexact))
}

/// Exact f32 FMA for the rare lanes the fast path cannot settle: TwoSum
/// recovers the f64 sum's error and round-to-odd avoids double rounding,
/// including at underflow.
#[cold]
#[inline(never)]
#[cfg(not(all(
    feature = "std",
    any(
        target_feature = "fma",
        all(target_arch = "aarch64", target_feature = "neon")
    )
)))]
fn fmaf_exact(a: f32, b: f32, c: f32) -> f32 {
    let product = f64::from(a) * f64::from(b);
    let (sum, error) = two_sum(product, f64::from(c));
    round_to_odd(sum, error, sum.is_finite()) as f32
}

/// Software f32 fused multiply-add, correctly rounded, no CPU detection.
#[inline(always)]
#[cfg(not(all(
    feature = "std",
    any(
        target_feature = "fma",
        all(target_arch = "aarch64", target_feature = "neon")
    )
)))]
fn fmaf_soft(a: f32, b: f32, c: f32) -> f32 {
    let sum = f64::from(a) * f64::from(b) + f64::from(c);
    let bits = sum.to_bits();
    // 32-bit compares: the dropped bits all sit in the low word.
    let halfway = (bits as u32) & F32_DROPPED == F32_HALFWAY;
    // Exact results are multiples of 2^-298, so the high word is zero only
    // for a zero sum; one unsigned compare finds f32-subnormal magnitudes.
    let high = (bits >> 32) as u32 & 0x7fff_ffff;
    let subnormal = high.wrapping_sub(1) < F32_MIN_NORMAL_HIGH - 1;
    if halfway | subnormal {
        fmaf_exact(a, b, c)
    } else {
        sum as f32
    }
}

/// Fused multiply-add `a * b + c`, correctly rounded (single rounding).
///
/// `std` builds with compile-time x86 FMA or AArch64 NEON use hardware
/// `f32::mul_add`; all other builds use software, without a runtime CPU probe. Both produce the unique correctly-rounded result, so
/// bits are identical — only speed differs. NaN payloads/signs are
/// unspecified; assumes round-to-nearest and gradual underflow.
#[inline(always)]
pub fn fmaf(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(all(
        feature = "std",
        any(
            target_feature = "fma",
            all(target_arch = "aarch64", target_feature = "neon")
        )
    ))]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(all(
        feature = "std",
        any(
            target_feature = "fma",
            all(target_arch = "aarch64", target_feature = "neon")
        )
    )))]
    {
        fmaf_soft(a, b, c)
    }
}

// ============================================================================
// f32 mirrors of the `magemath::portable` kernels
// ============================================================================
//
// Same polynomials and operand order as magetypes' generated transcendentals,
// with two deliberate differences that make the results the same on every
// tier: each polynomial step is one fused `fmaf` (magetypes' `mul_add_relaxed`
// rounds twice on its scalar backend and strict WASM), and a NaN input returns
// NaN. magetypes leaves NaN to `min`/`max`, which disagree across ISAs: its
// `exp2_midp(NaN)` is 2^-126 on x86 and its scalar backend, NaN on NEON and
// WASM; its `log2_midp(NaN)` is a finite value on all of them.

/// One polynomial step, `a * b + c`.
trait Step {
    fn madd(a: f32, b: f32, c: f32) -> f32;
}

/// One rounding: [`fmaf`].
struct Fused;

impl Step for Fused {
    #[inline(always)]
    fn madd(a: f32, b: f32, c: f32) -> f32 {
        fmaf(a, b, c)
    }
}

/// Two roundings: multiply, then add. Rust never contracts these into an FMA.
struct Unfused;

impl Step for Unfused {
    #[inline(always)]
    fn madd(a: f32, b: f32, c: f32) -> f32 {
        a * b + c
    }
}

/// Specials shared by the log family: 0 → −inf, negative or NaN → NaN, +inf → +inf.
#[inline(always)]
fn log_specials(x: f32, r: f32) -> f32 {
    if x == f32::INFINITY {
        x
    } else if x > 0.0 {
        r
    } else if x == 0.0 {
        f32::NEG_INFINITY
    } else {
        f32::NAN
    }
}

// ====== lowp ======

/// Low-precision `log2`: at most 6.4e-6 absolute error over positive normals;
/// relative error grows without bound near x = 1. (2,2) rational polynomial
/// on the mantissa. 0 → −inf, negative or NaN → NaN, +inf → +inf. Subnormal
/// inputs are outside the domain: finite, but inaccurate.
#[inline]
pub fn log2_lowp_f32(x: f32) -> f32 {
    log2_lowp_with::<Fused>(x)
}

#[inline]
fn log2_lowp_with<S: Step>(x: f32) -> f32 {
    log2_lowp_evaluate(x, S::madd)
}

// One shared polynomial body; proof contexts supply the same fused or separate
// operations and exact rounding primitive without per-step runtime selection.
#[inline(always)]
#[allow(clippy::manual_clamp)]
pub(crate) fn log2_lowp_evaluate(x: f32, madd: impl Fn(f32, f32, f32) -> f32) -> f32 {
    const OFFSET: i32 = 0x3f2a_aaab;
    const P0: f32 = -1.850_383_3e-6;
    const P1: f32 = 1.428_716_1;
    const P2: f32 = 0.742_458_7;
    const Q0: f32 = 0.990_328_14;
    const Q1: f32 = 1.009_671_8;
    const Q2: f32 = 0.174_093_43;

    let x_bits = x.to_bits() as i32;
    let exp_shifted = x_bits.wrapping_sub(OFFSET) >> 23;
    let mantissa = f32::from_bits(x_bits.wrapping_sub(exp_shifted << 23) as u32);
    let m = mantissa - 1.0;
    let yp = madd(madd(P2, m, P1), m, P0);
    let yq = madd(madd(Q2, m, Q1), m, Q0);
    log_specials(x, yp / yq + exp_shifted as f32)
}

/// Low-precision `2^x`: at most 0.56% relative error for x in [-126, 127.99],
/// 1.23% above. Clamps its input to that range instead of saturating, so
/// x < -126 returns 2^-126 and x ≥ 127.99 stays finite. NaN → NaN.
#[inline]
pub fn exp2_lowp_f32(x: f32) -> f32 {
    exp2_lowp_with::<Fused>(x)
}

#[inline]
#[allow(clippy::manual_clamp)] // Written as the vector kernel's .max().min().
fn exp2_lowp_with<S: Step>(x: f32) -> f32 {
    exp2_lowp_evaluate(x, S::madd, floorf)
}

// One shared polynomial body; proof contexts supply the same fused or separate
// operations and exact rounding primitive without per-step runtime selection.
#[inline(always)]
#[allow(clippy::manual_clamp)]
pub(crate) fn exp2_lowp_evaluate(
    x: f32,
    madd: impl Fn(f32, f32, f32) -> f32,
    round: impl Fn(f32) -> f32,
) -> f32 {
    const C1: f32 = core::f32::consts::LN_2;
    const C2: f32 = 0.240_226_5;
    const C3: f32 = 0.055_504_11;

    if x.is_nan() {
        return f32::NAN;
    }
    let clamped = x.max(-126.0).min(127.99);
    let xi = round(clamped);
    let xf = clamped - xi;
    let poly = madd(madd(madd(C3, xf, C2), xf, C1), xf, 1.0);
    poly * f32::from_bits(((xi as i32).wrapping_add(127) << 23) as u32)
}

/// Low-precision `ln`: `log2_lowp · LN_2` (at most 8.5e-6 absolute error).
#[inline]
pub fn ln_lowp_f32(x: f32) -> f32 {
    ln_lowp_with::<Fused>(x)
}

#[inline]
fn ln_lowp_with<S: Step>(x: f32) -> f32 {
    log2_lowp_with::<S>(x) * core::f32::consts::LN_2
}

/// Low-precision `e^x`: `exp2_lowp(x · LOG2_E)`.
#[inline]
pub fn exp_lowp_f32(x: f32) -> f32 {
    exp_lowp_with::<Fused>(x)
}

#[inline]
fn exp_lowp_with<S: Step>(x: f32) -> f32 {
    exp2_lowp_with::<S>(x * core::f32::consts::LOG2_E)
}

/// Low-precision `log10`: `log2_lowp · (LN_2 / LN_10)` (at most 5.8e-6 absolute error).
#[inline]
pub fn log10_lowp_f32(x: f32) -> f32 {
    log10_lowp_with::<Fused>(x)
}

#[inline]
fn log10_lowp_with<S: Step>(x: f32) -> f32 {
    log2_lowp_with::<S>(x) * (core::f32::consts::LN_2 / core::f32::consts::LN_10)
}

/// Low-precision `x^n`: `exp2_lowp(log2_lowp(x) · n)`. Returns +0 for ±0 at
/// every `n`, NaN for negative or NaN `x` and for +inf with `n` = 0.
#[inline]
pub fn pow_lowp_f32(x: f32, n: f32) -> f32 {
    pow_lowp_with::<Fused>(x, n)
}

#[inline]
fn pow_lowp_with<S: Step>(x: f32, n: f32) -> f32 {
    let r = exp2_lowp_with::<S>(log2_lowp_with::<S>(x) * n);
    if x == 0.0 { 0.0 } else { r }
}

/// Low-precision cube root: at most 259 ULP, 3e-5 relative error, for
/// magnitudes in [2^-126, `f32::MAX` / 3]. Kahan's bit hack plus one Halley
/// iteration. ±0 → ±0, NaN → NaN. Subnormals are inaccurate, and larger
/// magnitudes and ±inf return NaN or ±inf.
#[inline(always)]
pub fn cbrt_lowp_f32(x: f32) -> f32 {
    let sign = x.signum();
    let ax = x.abs();
    let mut y = f32::from_bits((ax.to_bits() / 3) + 0x2a50_8c2d);
    let y3 = y * y * y;
    y *= (y3 + 2.0 * ax) / (2.0 * y3 + ax);
    // Zero masking: bit hack gives garbage for zero, but ax==0 check
    // compiles to cmov (branchless) preserving auto-vectorizability.
    if ax == 0.0 { x } else { sign * y }
}

// ====== midp ======

/// Mid-precision `log2`: at most 4.5 ULP over positive normals. (a−1)/(a+1)
/// transform with an odd polynomial. 0 → −inf, negative or NaN → NaN,
/// +inf → +inf. Subnormal inputs are outside the domain: for 1e-42 it returns
/// about -127, not -139.5.
#[inline]
pub fn log2_midp_f32(x: f32) -> f32 {
    log2_midp_with::<Fused>(x)
}

#[inline]
fn log2_midp_with<S: Step>(x: f32) -> f32 {
    log2_midp_evaluate(x, S::madd)
}

// One shared polynomial body; proof contexts supply the same fused or separate
// operations and exact rounding primitive without per-step runtime selection.
#[inline(always)]
#[allow(clippy::manual_clamp)]
pub(crate) fn log2_midp_evaluate(x: f32, madd: impl Fn(f32, f32, f32) -> f32) -> f32 {
    const SQRT2_OVER_2: u32 = 0x3f35_04f3;
    const ONE_BITS: u32 = 0x3f80_0000;
    const MANTISSA_MASK: i32 = 0x007f_ffff;
    const C0: f32 = 2.885_39;
    const C1: f32 = 0.961_800_76;
    const C2: f32 = 0.576_974_45;
    const C3: f32 = 0.434_411_97;

    let x_bits = x.to_bits() as i32;
    let adjusted = x_bits.wrapping_add((ONE_BITS - SQRT2_OVER_2) as i32);
    let n = ((adjusted >> 23).wrapping_sub(127)) as f32;
    let a = f32::from_bits(((adjusted & MANTISSA_MASK) as u32).wrapping_add(SQRT2_OVER_2));
    let y = (a - 1.0) / (a + 1.0);
    let y2 = y * y;
    let poly = madd(madd(madd(C3, y2, C2), y2, C1), y2, C0);
    log_specials(x, madd(poly, y, n))
}

/// Mid-precision `2^x`: at most 1.9 ULP for x below 127.5, up to 134.1 ULP in
/// [127.5, 128). Ties-to-even split (|frac| ≤ 0.5), degree-6 Horner, exponent
/// bits. x < -126 → 0, x ≥ 128 → +inf, NaN → NaN.
#[inline]
pub fn exp2_midp_f32(x: f32) -> f32 {
    exp2_midp_with::<Fused>(x)
}

#[inline]
#[allow(clippy::manual_clamp)] // Written as the vector kernel's .max().min().
fn exp2_midp_with<S: Step>(x: f32) -> f32 {
    exp2_midp_evaluate(x, S::madd, roundevenf)
}

// One shared polynomial body; proof contexts supply the same fused or separate
// operations and exact rounding primitive without per-step runtime selection.
#[inline(always)]
#[allow(clippy::manual_clamp)]
pub(crate) fn exp2_midp_evaluate(
    x: f32,
    madd: impl Fn(f32, f32, f32) -> f32,
    round: impl Fn(f32) -> f32,
) -> f32 {
    const C1: f32 = core::f32::consts::LN_2;
    const C2: f32 = 0.240_226_46;
    const C3: f32 = 0.055_504_545;
    const C4: f32 = 0.009_618_055;
    const C5: f32 = 0.001_333_37;
    const C6: f32 = 0.000_154_47;

    if x.is_nan() {
        return f32::NAN;
    }
    if x < -126.0 {
        return 0.0;
    }
    if x >= 128.0 {
        return f32::INFINITY;
    }
    let xi = round(x).min(127.0);
    let xf = x - xi;
    let poly = madd(
        madd(
            madd(madd(madd(madd(C6, xf, C5), xf, C4), xf, C3), xf, C2),
            xf,
            C1,
        ),
        xf,
        1.0,
    );
    poly * f32::from_bits(((xi as i32).wrapping_add(127) << 23) as u32)
}

/// Mid-precision `ln`: `log2_midp · LN_2` (at most 4.1 ULP).
#[inline]
pub fn ln_midp_f32(x: f32) -> f32 {
    ln_midp_with::<Fused>(x)
}

#[inline]
fn ln_midp_with<S: Step>(x: f32) -> f32 {
    log2_midp_with::<S>(x) * core::f32::consts::LN_2
}

/// Mid-precision `e^x`: `exp2_midp(x · LOG2_E)`. Error grows with |x|: 2.0 ULP
/// for |x| ≤ 1, 8.2 for |x| ≤ 10, 64.1 on [-87, 88.5].
#[inline]
pub fn exp_midp_f32(x: f32) -> f32 {
    exp_midp_with::<Fused>(x)
}

#[inline]
fn exp_midp_with<S: Step>(x: f32) -> f32 {
    exp2_midp_with::<S>(x * core::f32::consts::LOG2_E)
}

/// Mid-precision `log10`: `log2_midp · (LN_2 / LN_10)` (at most 4.5 ULP).
#[inline]
pub fn log10_midp_f32(x: f32) -> f32 {
    log10_midp_with::<Fused>(x)
}

#[inline]
fn log10_midp_with<S: Step>(x: f32) -> f32 {
    log2_midp_with::<S>(x) * (core::f32::consts::LN_2 / core::f32::consts::LN_10)
}

/// Mid-precision `x^n`: `exp2_midp(log2_midp(x) · n)`. Error grows with
/// |n · log2(x)|: 27.8 ULP for n = 2.4 on [2^-8, 2^8]. NaN for negative or NaN
/// `x`, and where `log2(x) · n` is 0 · inf: `x` = 0 or +inf with `n` = 0.
#[inline]
pub fn pow_midp_f32(x: f32, n: f32) -> f32 {
    pow_midp_with::<Fused>(x, n)
}

#[inline]
fn pow_midp_with<S: Step>(x: f32, n: f32) -> f32 {
    exp2_midp_with::<S>(log2_midp_with::<S>(x) * n)
}

/// Mid-precision cube root: at most 3.2 ULP for magnitudes in
/// [2^-126, `f32::MAX` / 3]. Kahan's bit hack plus two Halley iterations.
/// ±0 → ±0, NaN → NaN. Subnormals are inaccurate, and larger magnitudes and
/// ±inf return NaN or ±inf: use [`cbrt_midp_precise_f32`] for those.
#[inline(always)]
pub fn cbrt_midp_f32(x: f32) -> f32 {
    let sign = x.signum();
    let ax = x.abs();
    let mut y = f32::from_bits((ax.to_bits() / 3) + 0x2a50_8c2d);
    for _ in 0..2 {
        let y3 = y * y * y;
        y *= (y3 + 2.0 * ax) / (2.0 * y3 + ax);
    }
    // Zero masking: bit hack gives garbage for zero, but ax==0 check
    // compiles to cmov (branchless) preserving auto-vectorizability.
    if ax == 0.0 { x } else { sign * y }
}

/// Mid-precision cube root over the whole f32 range (at most 3.2 ULP):
/// subnormals and huge magnitudes are rescaled around [`cbrt_midp_f32`].
/// ±0 and ±inf return themselves, NaN → NaN.
#[inline]
pub fn cbrt_midp_precise_f32(x: f32) -> f32 {
    let ax = x.abs();
    if x == 0.0 || ax == f32::INFINITY {
        return x;
    }
    let (pre, post) = if ax < 1.175_494_4e-38 {
        (16_777_216.0, 1.0 / 256.0)
    } else if ax >= 1.0e36 {
        (0.125, 2.0)
    } else {
        (1.0, 1.0)
    };
    cbrt_midp_f32(x * pre) * post
}

/// Mid-precision logistic sigmoid `1 / (1 + e^-x)`: saturated inputs give
/// exactly 0 or 1 because the division is exact. NaN → NaN.
#[inline]
pub fn sigmoid_midp_f32(x: f32) -> f32 {
    sigmoid_midp_with::<Fused>(x)
}

#[inline]
fn sigmoid_midp_with<S: Step>(x: f32) -> f32 {
    1.0 / (exp_midp_with::<S>(-x) + 1.0)
}

/// Mid-precision SiLU / swish `x · sigmoid(x)`. NaN → NaN.
#[inline]
pub fn silu_midp_f32(x: f32) -> f32 {
    silu_midp_with::<Fused>(x)
}

#[inline]
fn silu_midp_with<S: Step>(x: f32) -> f32 {
    x * sigmoid_midp_with::<S>(x)
}

/// The same functions with each polynomial step a plain multiply and then add:
/// two roundings, on every target. No `fmaf`, so no software path and no
/// dependence on `std`. Specials match the functions in the parent module, and the
/// accuracy bounds in `docs/transcendentals.md` hold for both roundings; finite
/// results can differ in the last bits.
///
/// The functions with no multiply-add step are the same either way, and are
/// re-exported so this module is a complete set of mirrors.
pub mod unfused {
    pub use super::{
        cbrt_lowp_f32, cbrt_midp_f32, cbrt_midp_precise_f32, ceilf, floorf, roundevenf, sqrtf,
    };

    /// [`super::log2_lowp_f32`] with two roundings per polynomial step.
    #[inline]
    pub fn log2_lowp_f32(x: f32) -> f32 {
        super::log2_lowp_with::<super::Unfused>(x)
    }

    /// [`super::exp2_lowp_f32`] with two roundings per polynomial step.
    #[inline]
    pub fn exp2_lowp_f32(x: f32) -> f32 {
        super::exp2_lowp_with::<super::Unfused>(x)
    }

    /// [`super::ln_lowp_f32`] with two roundings per polynomial step.
    #[inline]
    pub fn ln_lowp_f32(x: f32) -> f32 {
        super::ln_lowp_with::<super::Unfused>(x)
    }

    /// [`super::exp_lowp_f32`] with two roundings per polynomial step.
    #[inline]
    pub fn exp_lowp_f32(x: f32) -> f32 {
        super::exp_lowp_with::<super::Unfused>(x)
    }

    /// [`super::log10_lowp_f32`] with two roundings per polynomial step.
    #[inline]
    pub fn log10_lowp_f32(x: f32) -> f32 {
        super::log10_lowp_with::<super::Unfused>(x)
    }

    /// [`super::pow_lowp_f32`] with two roundings per polynomial step.
    #[inline]
    pub fn pow_lowp_f32(x: f32, n: f32) -> f32 {
        super::pow_lowp_with::<super::Unfused>(x, n)
    }

    /// [`super::log2_midp_f32`] with two roundings per polynomial step.
    #[inline]
    pub fn log2_midp_f32(x: f32) -> f32 {
        super::log2_midp_with::<super::Unfused>(x)
    }

    /// [`super::exp2_midp_f32`] with two roundings per polynomial step.
    #[inline]
    pub fn exp2_midp_f32(x: f32) -> f32 {
        super::exp2_midp_with::<super::Unfused>(x)
    }

    /// [`super::ln_midp_f32`] with two roundings per polynomial step.
    #[inline]
    pub fn ln_midp_f32(x: f32) -> f32 {
        super::ln_midp_with::<super::Unfused>(x)
    }

    /// [`super::exp_midp_f32`] with two roundings per polynomial step.
    #[inline]
    pub fn exp_midp_f32(x: f32) -> f32 {
        super::exp_midp_with::<super::Unfused>(x)
    }

    /// [`super::log10_midp_f32`] with two roundings per polynomial step.
    #[inline]
    pub fn log10_midp_f32(x: f32) -> f32 {
        super::log10_midp_with::<super::Unfused>(x)
    }

    /// [`super::pow_midp_f32`] with two roundings per polynomial step.
    #[inline]
    pub fn pow_midp_f32(x: f32, n: f32) -> f32 {
        super::pow_midp_with::<super::Unfused>(x, n)
    }

    /// [`super::sigmoid_midp_f32`] with two roundings per polynomial step.
    #[inline]
    pub fn sigmoid_midp_f32(x: f32) -> f32 {
        super::sigmoid_midp_with::<super::Unfused>(x)
    }

    /// [`super::silu_midp_f32`] with two roundings per polynomial step.
    #[inline]
    pub fn silu_midp_f32(x: f32) -> f32 {
        super::silu_midp_with::<super::Unfused>(x)
    }
}

// Correctly rounded full-range functions, separate from the approximation tiers.
pub use crate::core_math::powf;
pub use crate::core_math::{atanf, fma, log10f, tanhf};
pub use crate::core_math::{cbrtf, exp2f, expf, log2f, logf};
pub use crate::core_math::{cosf, sinf};
/// Correctly rounded sine and cosine (nearest-even).
pub fn sincosf(x: f32) -> (f32, f32) {
    (sinf(x), cosf(x))
}
pub use crate::core_math::atan;
pub use crate::core_math::atan2f;
pub use crate::core_math::cbrt;
pub use crate::core_math::cos;
pub use crate::core_math::exp;
pub use crate::core_math::exp2;
pub use crate::core_math::log;
pub use crate::core_math::log1p;
pub use crate::core_math::log2;
pub use crate::core_math::log10;
pub use crate::core_math::sin;
pub use crate::core_math::tanf;
pub use crate::core_math::{expm1f, log1pf};

pub use crate::core_math::atan2;

pub use crate::core_math::asinh;
pub use crate::core_math::atanh;
pub use crate::core_math::pow;
pub use crate::core_math::tan;
pub use crate::core_math::tanh;
pub use crate::core_math::{acosf, hypotf};

#[doc(inline)]
pub use crate::__pow_policy::powf_with;

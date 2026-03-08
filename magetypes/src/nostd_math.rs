//! Pure-Rust math functions for `no_std` scalar backends.
//!
//! These replace `f32::sqrt()`, `f32::floor()`, etc. which require `std`.
//! On platforms with hardware SIMD, intrinsics handle these operations directly.
//! These are only used by the scalar fallback backend.
//!
//! Every function here must be bit-exact with its `std` counterpart for all
//! finite inputs and ±Inf. NaN propagation preserves NaN-ness but payloads
//! may differ (hardware ops quiet sNaN→qNaN; software implementations don't).
//!
//! The sqrt algorithm is Goldschmidt iteration with a lookup table, ported from
//! [compiler-builtins/libm](https://github.com/rust-lang/compiler-builtins/blob/main/libm/src/math/generic/sqrt.rs)
//! (MIT license, origin: musl `src/math/sqrt.c`). It uses pure integer arithmetic
//! — no floating-point division — making it much faster than Newton-Raphson on
//! targets without hardware sqrt.

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

/// f32 fused multiply-add (non-fused fallback: `a * b + c`).
///
/// In a scalar no_std context there's no hardware FMA instruction to use,
/// so this is just the unfused version. The precision difference vs true FMA
/// is acceptable for a fallback path.
#[inline(always)]
pub fn fmaf(a: f32, b: f32, c: f32) -> f32 {
    a * b + c
}

/// f64 fused multiply-add (non-fused fallback: `a * b + c`).
#[inline(always)]
pub fn fma(a: f64, b: f64, c: f64) -> f64 {
    a * b + c
}

// ============================================================================
// Transcendental approximations for no_std f64 scalar fallbacks
// ============================================================================

// These match the polynomial coefficients used in the SIMD implementations
// (x86 f64x2 log2_lowp/exp2_lowp). They are "lowp" — not bit-exact with std,
// but good enough for the lowp tier (~1% max relative error).

/// Low-precision base-2 logarithm for f64.
///
/// Uses the same rational polynomial as the x86 f64x2 SIMD implementation.
/// Only valid for positive, finite, non-zero inputs.
#[inline(always)]
pub fn log2_f64(x: f64) -> f64 {
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
/// Uses the same polynomial as the x86 f64x2 SIMD implementation.
#[inline(always)]
pub fn exp2_f64(x: f64) -> f64 {
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
pub fn ln_f64(x: f64) -> f64 {
    log2_f64(x) * core::f64::consts::LN_2
}

/// Low-precision natural exponential (e^x) for f64.
#[inline(always)]
pub fn exp_f64(x: f64) -> f64 {
    exp2_f64(x * core::f64::consts::LOG2_E)
}

/// Low-precision base-10 logarithm for f64.
#[inline(always)]
pub fn log10_f64(x: f64) -> f64 {
    log2_f64(x) * core::f64::consts::LOG10_2
}

/// Low-precision power function (x^n) for f64.
#[inline(always)]
pub fn powf_f64(x: f64, n: f64) -> f64 {
    exp2_f64(n * log2_f64(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Brute-force f32 tests: every bit pattern
    // =========================================================================

    /// Compare two f32 results for bit-exactness.
    /// For NaN outputs: both must be NaN, but payloads may differ.
    /// Hardware ops (roundss, sqrtss) quiet signaling NaN → quiet NaN,
    /// but software implementations (our target) pass NaN through unchanged.
    /// NaN payloads are implementation-defined per IEEE 754.
    fn f32_match(ours: f32, std_result: f32) -> bool {
        if ours.is_nan() && std_result.is_nan() {
            return true;
        }
        ours.to_bits() == std_result.to_bits()
    }

    // Quick spot-check tests — always run (including under Miri)
    #[test]
    fn spot_check_f32_floor() {
        let cases: &[(f32, f32)] = &[
            (0.0, 0.0),
            (-0.0, -0.0),
            (1.5, 1.0),
            (-1.5, -2.0),
            (2.9, 2.0),
            (-2.9, -3.0),
            (1e10, 1e10),
            (-1e10, -1e10),
            (f32::INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, f32::NEG_INFINITY),
        ];
        for &(input, expected) in cases {
            let result = floorf(input);
            assert!(
                f32_match(result, expected),
                "floorf({input}) = {result}, expected {expected}"
            );
        }
        assert!(floorf(f32::NAN).is_nan());
    }

    #[test]
    fn spot_check_f32_ceil() {
        let cases: &[(f32, f32)] = &[
            (0.0, 0.0),
            (-0.0, -0.0),
            (1.5, 2.0),
            (-1.5, -1.0),
            (2.1, 3.0),
            (-2.1, -2.0),
            (1e10, 1e10),
        ];
        for &(input, expected) in cases {
            let result = ceilf(input);
            assert!(
                f32_match(result, expected),
                "ceilf({input}) = {result}, expected {expected}"
            );
        }
        assert!(ceilf(f32::NAN).is_nan());
    }

    #[test]
    fn spot_check_f32_round() {
        let cases: &[(f32, f32)] = &[
            (0.0, 0.0),
            (-0.0, -0.0),
            (0.5, 1.0),
            (-0.5, -1.0),
            (1.5, 2.0),
            (-1.5, -2.0),
            (2.4, 2.0),
            (-2.4, -2.0),
            (2.6, 3.0),
            (-2.6, -3.0),
            (1e10, 1e10),
        ];
        for &(input, expected) in cases {
            let result = roundf(input);
            assert!(
                f32_match(result, expected),
                "roundf({input}) = {result}, expected {expected}"
            );
        }
        assert!(roundf(f32::NAN).is_nan());
    }

    #[test]
    fn spot_check_f32_sqrt() {
        let cases: &[(f32, f32)] = &[
            (0.0, 0.0),
            (1.0, 1.0),
            (4.0, 2.0),
            (9.0, 3.0),
            (16.0, 4.0),
            (2.0, core::f32::consts::SQRT_2),
            (0.25, 0.5),
        ];
        for &(input, expected) in cases {
            let result = sqrtf(input);
            assert!(
                f32_match(result, expected),
                "sqrtf({input}) = {result}, expected {expected}"
            );
        }
        assert!(sqrtf(f32::NAN).is_nan());
        assert!(sqrtf(-1.0).is_nan());
        assert!(sqrtf(f32::INFINITY).is_infinite());
    }

    #[test]
    fn spot_check_f64_sqrt() {
        let cases: &[(f64, f64)] = &[
            (0.0, 0.0),
            (1.0, 1.0),
            (4.0, 2.0),
            (9.0, 3.0),
            (16.0, 4.0),
            (2.0, core::f64::consts::SQRT_2),
            (0.25, 0.5),
        ];
        for &(input, expected) in cases {
            let result = sqrt(input);
            assert_eq!(
                result.to_bits(),
                expected.to_bits(),
                "sqrt({input}) = {result}, expected {expected}"
            );
        }
        assert!(sqrt(f64::NAN).is_nan());
        assert!(sqrt(-1.0).is_nan());
        assert!(sqrt(f64::INFINITY).is_infinite());
    }

    #[test]
    #[ignore] // 4B iterations — run with `cargo test -- --ignored`
    fn brute_force_f32_floor() {
        for bits in 0..=u32::MAX {
            let x = f32::from_bits(bits);
            let ours = floorf(x);
            let std_result = x.floor();
            assert!(
                f32_match(ours, std_result),
                "floorf mismatch at bits={bits:#010x} ({x}): ours={ours}, std={std_result}"
            );
        }
    }

    #[test]
    #[ignore] // 4B iterations — run with `cargo test -- --ignored`
    fn brute_force_f32_ceil() {
        for bits in 0..=u32::MAX {
            let x = f32::from_bits(bits);
            let ours = ceilf(x);
            let std_result = x.ceil();
            assert!(
                f32_match(ours, std_result),
                "ceilf mismatch at bits={bits:#010x} ({x}): ours={ours}, std={std_result}"
            );
        }
    }

    #[test]
    #[ignore] // 4B iterations — run with `cargo test -- --ignored`
    fn brute_force_f32_round() {
        for bits in 0..=u32::MAX {
            let x = f32::from_bits(bits);
            let ours = roundf(x);
            let std_result = x.round();
            assert!(
                f32_match(ours, std_result),
                "roundf mismatch at bits={bits:#010x} ({x}): ours={ours}, std={std_result}"
            );
        }
    }

    #[test]
    #[ignore] // 4B iterations — run with `cargo test -- --ignored`
    fn brute_force_f32_sqrt() {
        for bits in 0..=u32::MAX {
            let x = f32::from_bits(bits);
            let ours = sqrtf(x);
            let std_result = x.sqrt();
            assert!(
                f32_match(ours, std_result),
                "sqrtf mismatch at bits={bits:#010x} ({x}): ours={ours} ({:#010x}), std={std_result} ({:#010x})",
                ours.to_bits(),
                std_result.to_bits()
            );
        }
    }

    // =========================================================================
    // Sampled f64 tests (2^64 is too many — test critical ranges)
    // =========================================================================

    /// Test f64 across critical ranges: denorms, small, medium, large, special
    fn f64_test_values() -> impl Iterator<Item = f64> {
        let mut values = alloc::vec::Vec::new();

        // Special values
        values.push(0.0);
        values.push(-0.0);
        values.push(f64::NAN);
        values.push(f64::INFINITY);
        values.push(f64::NEG_INFINITY);
        values.push(f64::MIN);
        values.push(f64::MAX);
        values.push(f64::MIN_POSITIVE);
        values.push(f64::EPSILON);

        // Small integers and near-integers
        for i in -1000..=1000 {
            let f = i as f64;
            values.push(f);
            values.push(f + 0.1);
            values.push(f + 0.25);
            values.push(f + 0.49999999999999994);
            values.push(f + 0.5);
            values.push(f + 0.50000000000000006);
            values.push(f + 0.75);
            values.push(f + 0.9);
            values.push(f - 0.1);
            values.push(f - 0.5);
        }

        // Powers of 2 and neighbors
        for exp in -1022..=1023i32 {
            let base = 2.0f64.powi(exp);
            values.push(base);
            values.push(-base);
            values.push(base * 1.5);
            values.push(-base * 1.5);
        }

        // Near the i64 truncation boundary (2^52)
        let boundary = (1u64 << 52) as f64;
        for offset in -100..=100 {
            values.push(boundary + offset as f64);
            values.push(-boundary + offset as f64);
        }

        // Random-ish bit patterns across the f64 range
        // Under Miri, use fewer samples to keep runtime reasonable
        let sample_count = if cfg!(miri) { 10_000 } else { 10_000_000 };
        let stride = u64::MAX / sample_count;
        let mut bits = 0u64;
        loop {
            values.push(f64::from_bits(bits));
            match bits.checked_add(stride) {
                Some(next) => bits = next,
                None => break,
            }
        }

        values.into_iter()
    }

    /// Compare two f64 results (same NaN tolerance as f32_match).
    fn f64_match(ours: f64, std_result: f64) -> bool {
        if ours.is_nan() && std_result.is_nan() {
            return true;
        }
        ours.to_bits() == std_result.to_bits()
    }

    #[test]
    fn sampled_f64_floor() {
        for x in f64_test_values() {
            let ours = floor(x);
            let std_result = x.floor();
            assert!(
                f64_match(ours, std_result),
                "floor mismatch at {x} (bits={:#018x}): ours={ours}, std={std_result}",
                x.to_bits()
            );
        }
    }

    #[test]
    fn sampled_f64_ceil() {
        for x in f64_test_values() {
            let ours = ceil(x);
            let std_result = x.ceil();
            assert!(
                f64_match(ours, std_result),
                "ceil mismatch at {x} (bits={:#018x}): ours={ours}, std={std_result}",
                x.to_bits()
            );
        }
    }

    #[test]
    fn sampled_f64_round() {
        for x in f64_test_values() {
            let ours = round(x);
            let std_result = x.round();
            assert!(
                f64_match(ours, std_result),
                "round mismatch at {x} (bits={:#018x}): ours={ours}, std={std_result}",
                x.to_bits()
            );
        }
    }

    #[test]
    fn sampled_f64_sqrt() {
        for x in f64_test_values() {
            let ours = sqrt(x);
            let std_result = x.sqrt();
            assert!(
                f64_match(ours, std_result),
                "sqrt mismatch at {x} (bits={:#018x}): ours={ours} ({:#018x}), std={std_result} ({:#018x})",
                x.to_bits(),
                ours.to_bits(),
                std_result.to_bits()
            );
        }
    }
}

//! WebAssembly SIMD transcendental operations (log, exp, pow).
//!
//! Pure polynomial approximations - no native transcendental intrinsics in WASM SIMD.
//! All operations use v128 type directly since that's what WASM intrinsics expect.
//!
//! Provides both unchecked (fast) and checked (handles edge cases) variants.

use super::types::{ElementType, SimdType, SimdWidth};
use std::fmt::Write;

/// Generate WASM transcendental operations for float types
pub fn generate_wasm_transcendental_ops(ty: &SimdType) -> String {
    let mut code = String::new();

    // Only for float types
    if !ty.elem.is_float() {
        return code;
    }

    assert!(
        ty.width == SimdWidth::W128,
        "WASM only supports 128-bit vectors"
    );

    writeln!(
        code,
        "    // ========== Transcendental Operations (Polynomial Approximations) =========="
    )
    .unwrap();
    writeln!(code, "    //").unwrap();
    writeln!(
        code,
        "    // WASM SIMD has no native transcendental intrinsics."
    )
    .unwrap();
    writeln!(
        code,
        "    // Provides _unchecked (fast) and checked (handles edge cases) variants.\n"
    )
    .unwrap();

    if ty.elem == ElementType::F32 {
        generate_f32_transcendentals(&mut code);
    } else if ty.elem == ElementType::F64 {
        generate_f64_transcendentals(&mut code);
    }

    code
}

fn generate_f32_transcendentals(code: &mut String) {
    // ===== F32 log2_lowp_unchecked =====
    writeln!(
        code,
        "    /// Low-precision base-2 logarithm - unchecked variant."
    )
    .unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN)."
    )
    .unwrap();
    writeln!(code, "    /// Use `log2_lowp()` for correct IEEE behavior.").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn log2_lowp_unchecked(self) -> Self {{").unwrap();
    writeln!(
        code,
        "        // Rational polynomial coefficients from butteraugli/jpegli"
    )
    .unwrap();
    writeln!(code, "        const P0: f32 = -1.850_383_34e-6;").unwrap();
    writeln!(code, "        const P1: f32 = 1.428_716_05;").unwrap();
    writeln!(code, "        const P2: f32 = 0.742_458_73;").unwrap();
    writeln!(code, "        const Q0: f32 = 0.990_328_14;").unwrap();
    writeln!(code, "        const Q1: f32 = 1.009_671_86;").unwrap();
    writeln!(code, "        const Q2: f32 = 0.174_093_43;").unwrap();
    writeln!(code, "        const OFFSET: u32 = 0x3f2aaaab;").unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        let x = self.0;").unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        // Extract exponent via bit manipulation").unwrap();
    writeln!(code, "        let offset = u32x4_splat(OFFSET);").unwrap();
    writeln!(code, "        let exp_bits = i32x4_sub(x, offset);").unwrap();
    writeln!(code, "        let exp_shifted = i32x4_shr(exp_bits, 23);").unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        // Reconstruct mantissa in [1, 2) range").unwrap();
    writeln!(
        code,
        "        let mantissa_bits = i32x4_sub(x, i32x4_shl(exp_shifted, 23));"
    )
    .unwrap();
    writeln!(
        code,
        "        let exp_val = f32x4_convert_i32x4(exp_shifted);"
    )
    .unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        // m = mantissa - 1").unwrap();
    writeln!(code, "        let one = f32x4_splat(1.0);").unwrap();
    writeln!(code, "        let m = f32x4_sub(mantissa_bits, one);").unwrap();
    writeln!(code, "").unwrap();
    writeln!(
        code,
        "        // Horner's for numerator: P2*m^2 + P1*m + P0"
    )
    .unwrap();
    writeln!(code, "        let m2 = f32x4_mul(m, m);").unwrap();
    writeln!(code, "        let p2_m2 = f32x4_mul(f32x4_splat(P2), m2);").unwrap();
    writeln!(code, "        let p1_m = f32x4_mul(f32x4_splat(P1), m);").unwrap();
    writeln!(
        code,
        "        let yp = f32x4_add(p2_m2, f32x4_add(p1_m, f32x4_splat(P0)));"
    )
    .unwrap();
    writeln!(code, "").unwrap();
    writeln!(
        code,
        "        // Horner's for denominator: Q2*m^2 + Q1*m + Q0"
    )
    .unwrap();
    writeln!(code, "        let q2_m2 = f32x4_mul(f32x4_splat(Q2), m2);").unwrap();
    writeln!(code, "        let q1_m = f32x4_mul(f32x4_splat(Q1), m);").unwrap();
    writeln!(
        code,
        "        let yq = f32x4_add(q2_m2, f32x4_add(q1_m, f32x4_splat(Q0)));"
    )
    .unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        Self(f32x4_add(f32x4_div(yp, yq), exp_val))").unwrap();
    writeln!(code, "    }}\n").unwrap();

    // ===== F32 log2_lowp (checked) =====
    writeln!(
        code,
        "    /// Low-precision base-2 logarithm (~7.7e-5 max relative error)."
    )
    .unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// Handles edge cases: log2(0) = -inf, log2(negative) = NaN,"
    )
    .unwrap();
    writeln!(code, "    /// log2(+inf) = +inf, log2(NaN) = NaN.").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn log2_lowp(self) -> Self {{").unwrap();
    writeln!(code, "        let result = self.log2_lowp_unchecked();").unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        // Edge case masks").unwrap();
    writeln!(code, "        let zero = f32x4_splat(0.0);").unwrap();
    writeln!(code, "        let is_zero = f32x4_eq(self.0, zero);").unwrap();
    writeln!(code, "        let is_neg = f32x4_lt(self.0, zero);").unwrap();
    writeln!(
        code,
        "        let is_inf = f32x4_eq(self.0, f32x4_splat(f32::INFINITY));"
    )
    .unwrap();
    writeln!(code, "        // NaN: x != x").unwrap();
    writeln!(
        code,
        "        let is_nan = v128_not(f32x4_eq(self.0, self.0));"
    )
    .unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        // Apply corrections using bitselect").unwrap();
    writeln!(
        code,
        "        let neg_inf = f32x4_splat(f32::NEG_INFINITY);"
    )
    .unwrap();
    writeln!(code, "        let pos_inf = f32x4_splat(f32::INFINITY);").unwrap();
    writeln!(code, "        let nan = f32x4_splat(f32::NAN);").unwrap();
    writeln!(code, "").unwrap();
    writeln!(
        code,
        "        let r = v128_bitselect(neg_inf, result.0, is_zero);  // 0 -> -inf"
    )
    .unwrap();
    writeln!(
        code,
        "        let r = v128_bitselect(nan, r, is_neg);              // neg -> NaN"
    )
    .unwrap();
    writeln!(
        code,
        "        let r = v128_bitselect(pos_inf, r, is_inf);          // inf -> inf"
    )
    .unwrap();
    writeln!(
        code,
        "        let r = v128_bitselect(nan, r, is_nan);              // NaN -> NaN"
    )
    .unwrap();
    writeln!(code, "        Self(r)").unwrap();
    writeln!(code, "    }}\n").unwrap();

    // ===== F32 exp2_lowp_unchecked =====
    writeln!(
        code,
        "    /// Low-precision base-2 exponential - unchecked variant."
    )
    .unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// **Warning**: Clamps to [-126, 126]. Does not return inf for overflow."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn exp2_lowp_unchecked(self) -> Self {{").unwrap();
    writeln!(code, "        const C0: f32 = 1.0;").unwrap();
    writeln!(code, "        const C1: f32 = core::f32::consts::LN_2;").unwrap();
    writeln!(code, "        const C2: f32 = 0.240_226_5;").unwrap();
    writeln!(code, "        const C3: f32 = 0.055_504_11;").unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        // Clamp to safe range").unwrap();
    writeln!(
        code,
        "        let x = f32x4_pmax(self.0, f32x4_splat(-126.0));"
    )
    .unwrap();
    writeln!(code, "        let x = f32x4_pmin(x, f32x4_splat(126.0));").unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        let xi = f32x4_floor(x);").unwrap();
    writeln!(code, "        let xf = f32x4_sub(x, xi);").unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        // Polynomial for 2^frac").unwrap();
    writeln!(
        code,
        "        let poly = f32x4_add(f32x4_mul(f32x4_splat(C3), xf), f32x4_splat(C2));"
    )
    .unwrap();
    writeln!(
        code,
        "        let poly = f32x4_add(f32x4_mul(poly, xf), f32x4_splat(C1));"
    )
    .unwrap();
    writeln!(
        code,
        "        let poly = f32x4_add(f32x4_mul(poly, xf), f32x4_splat(C0));"
    )
    .unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        // Scale by 2^integer").unwrap();
    writeln!(code, "        let xi_i32 = i32x4_trunc_sat_f32x4(xi);").unwrap();
    writeln!(code, "        let bias = i32x4_splat(127);").unwrap();
    writeln!(
        code,
        "        let scale_bits = i32x4_shl(i32x4_add(xi_i32, bias), 23);"
    )
    .unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        Self(f32x4_mul(poly, scale_bits))").unwrap();
    writeln!(code, "    }}\n").unwrap();

    // ===== F32 exp2_lowp (checked) =====
    writeln!(
        code,
        "    /// Low-precision base-2 exponential (~5.5e-3 max relative error)."
    )
    .unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// Handles edge cases: exp2(x > 128) = +inf, exp2(x < -150) = 0,"
    )
    .unwrap();
    writeln!(code, "    /// exp2(NaN) = NaN.").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn exp2_lowp(self) -> Self {{").unwrap();
    writeln!(code, "        let result = self.exp2_lowp_unchecked();").unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        // Edge case masks").unwrap();
    writeln!(
        code,
        "        let is_overflow = f32x4_ge(self.0, f32x4_splat(128.0));"
    )
    .unwrap();
    writeln!(
        code,
        "        let is_underflow = f32x4_lt(self.0, f32x4_splat(-150.0));"
    )
    .unwrap();
    writeln!(
        code,
        "        let is_nan = v128_not(f32x4_eq(self.0, self.0));"
    )
    .unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        let pos_inf = f32x4_splat(f32::INFINITY);").unwrap();
    writeln!(code, "        let zero = f32x4_splat(0.0);").unwrap();
    writeln!(code, "        let nan = f32x4_splat(f32::NAN);").unwrap();
    writeln!(code, "").unwrap();
    writeln!(
        code,
        "        let r = v128_bitselect(pos_inf, result.0, is_overflow);"
    )
    .unwrap();
    writeln!(
        code,
        "        let r = v128_bitselect(zero, r, is_underflow);"
    )
    .unwrap();
    writeln!(code, "        let r = v128_bitselect(nan, r, is_nan);").unwrap();
    writeln!(code, "        Self(r)").unwrap();
    writeln!(code, "    }}\n").unwrap();

    // ===== F32 ln_lowp =====
    writeln!(code, "    /// Low-precision natural logarithm.").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(code, "    /// Computed as `log2_lowp(x) * ln(2)`.").unwrap();
    writeln!(code, "    /// Handles edge cases correctly.").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn ln_lowp(self) -> Self {{").unwrap();
    writeln!(code, "        const LN2: f32 = core::f32::consts::LN_2;").unwrap();
    writeln!(
        code,
        "        Self(f32x4_mul(self.log2_lowp().0, f32x4_splat(LN2)))"
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    // ===== F32 exp_lowp =====
    writeln!(code, "    /// Low-precision natural exponential (e^x).").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(code, "    /// Computed as `exp2_lowp(x * log2(e))`.").unwrap();
    writeln!(code, "    /// Handles edge cases correctly.").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn exp_lowp(self) -> Self {{").unwrap();
    writeln!(
        code,
        "        const LOG2_E: f32 = core::f32::consts::LOG2_E;"
    )
    .unwrap();
    writeln!(
        code,
        "        Self(f32x4_mul(self.0, f32x4_splat(LOG2_E))).exp2_lowp()"
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    // ===== F32 log10_lowp =====
    writeln!(code, "    /// Low-precision base-10 logarithm.").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(code, "    /// Computed as `log2_lowp(x) / log2(10)`.").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn log10_lowp(self) -> Self {{").unwrap();
    writeln!(
        code,
        "        const LOG10_2: f32 = core::f32::consts::LOG10_2;"
    )
    .unwrap();
    writeln!(
        code,
        "        Self(f32x4_mul(self.log2_lowp().0, f32x4_splat(LOG10_2)))"
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    // ===== F32 pow_lowp =====
    writeln!(code, "    /// Low-precision power function (self^n).").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// Computed as `exp2_lowp(n * log2_lowp(self))`."
    )
    .unwrap();
    writeln!(code, "    /// Note: Only valid for positive self values.").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn pow_lowp(self, n: f32) -> Self {{").unwrap();
    writeln!(
        code,
        "        Self(f32x4_mul(self.log2_lowp().0, f32x4_splat(n))).exp2_lowp()"
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    // ========== Mid-Precision ==========
    writeln!(
        code,
        "    // ========== Mid-Precision Transcendental Operations ==========\n"
    )
    .unwrap();

    // ===== F32 log2_midp_unchecked =====
    writeln!(
        code,
        "    /// Mid-precision base-2 logarithm - unchecked variant."
    )
    .unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN, denormals)."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn log2_midp_unchecked(self) -> Self {{").unwrap();
    writeln!(code, "        const SQRT2_OVER_2: u32 = 0x3f3504f3;").unwrap();
    writeln!(code, "        const ONE: u32 = 0x3f800000;").unwrap();
    writeln!(code, "        const MANTISSA_MASK: u32 = 0x007fffff;").unwrap();
    writeln!(code, "        const EXPONENT_BIAS: i32 = 127;").unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        // Coefficients for odd polynomial").unwrap();
    writeln!(code, "        const C0: f32 = 2.885_390_08;").unwrap();
    writeln!(code, "        const C1: f32 = 0.961_800_76;").unwrap();
    writeln!(code, "        const C2: f32 = 0.576_974_45;").unwrap();
    writeln!(code, "        const C3: f32 = 0.434_411_97;").unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        let x = self.0;").unwrap();
    writeln!(code, "").unwrap();
    writeln!(
        code,
        "        // Normalize mantissa to [sqrt(2)/2, sqrt(2)]"
    )
    .unwrap();
    writeln!(
        code,
        "        let offset = u32x4_splat(ONE - SQRT2_OVER_2);"
    )
    .unwrap();
    writeln!(code, "        let adjusted = i32x4_add(x, offset);").unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        // Extract exponent").unwrap();
    writeln!(code, "        let exp_raw = i32x4_shr(adjusted, 23);").unwrap();
    writeln!(
        code,
        "        let exp_biased = i32x4_sub(exp_raw, i32x4_splat(EXPONENT_BIAS));"
    )
    .unwrap();
    writeln!(code, "        let n = f32x4_convert_i32x4(exp_biased);").unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        // Reconstruct normalized mantissa").unwrap();
    writeln!(
        code,
        "        let mantissa_bits = v128_and(adjusted, u32x4_splat(MANTISSA_MASK));"
    )
    .unwrap();
    writeln!(
        code,
        "        let a_bits = i32x4_add(mantissa_bits, u32x4_splat(SQRT2_OVER_2));"
    )
    .unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        // y = (a - 1) / (a + 1)").unwrap();
    writeln!(code, "        let one = f32x4_splat(1.0);").unwrap();
    writeln!(code, "        let a_minus_1 = f32x4_sub(a_bits, one);").unwrap();
    writeln!(code, "        let a_plus_1 = f32x4_add(a_bits, one);").unwrap();
    writeln!(code, "        let y = f32x4_div(a_minus_1, a_plus_1);").unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        let y2 = f32x4_mul(y, y);").unwrap();
    writeln!(code, "").unwrap();
    writeln!(
        code,
        "        // Polynomial: c0 + y^2*(c1 + y^2*(c2 + y^2*c3))"
    )
    .unwrap();
    writeln!(
        code,
        "        let poly = f32x4_add(f32x4_mul(f32x4_splat(C3), y2), f32x4_splat(C2));"
    )
    .unwrap();
    writeln!(
        code,
        "        let poly = f32x4_add(f32x4_mul(poly, y2), f32x4_splat(C1));"
    )
    .unwrap();
    writeln!(
        code,
        "        let poly = f32x4_add(f32x4_mul(poly, y2), f32x4_splat(C0));"
    )
    .unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        Self(f32x4_add(f32x4_mul(y, poly), n))").unwrap();
    writeln!(code, "    }}\n").unwrap();

    // ===== F32 log2_midp (checked) =====
    writeln!(
        code,
        "    /// Mid-precision base-2 logarithm (~3 ULP max error)."
    )
    .unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// Handles edge cases: log2(0) = -inf, log2(negative) = NaN,"
    )
    .unwrap();
    writeln!(code, "    /// log2(+inf) = +inf, log2(NaN) = NaN.").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// Note: Does not handle denormals. Use `log2_midp_precise()` for full IEEE."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn log2_midp(self) -> Self {{").unwrap();
    writeln!(code, "        let result = self.log2_midp_unchecked();").unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        let zero = f32x4_splat(0.0);").unwrap();
    writeln!(code, "        let is_zero = f32x4_eq(self.0, zero);").unwrap();
    writeln!(code, "        let is_neg = f32x4_lt(self.0, zero);").unwrap();
    writeln!(
        code,
        "        let is_inf = f32x4_eq(self.0, f32x4_splat(f32::INFINITY));"
    )
    .unwrap();
    writeln!(
        code,
        "        let is_nan = v128_not(f32x4_eq(self.0, self.0));"
    )
    .unwrap();
    writeln!(code, "").unwrap();
    writeln!(
        code,
        "        let neg_inf = f32x4_splat(f32::NEG_INFINITY);"
    )
    .unwrap();
    writeln!(code, "        let pos_inf = f32x4_splat(f32::INFINITY);").unwrap();
    writeln!(code, "        let nan = f32x4_splat(f32::NAN);").unwrap();
    writeln!(code, "").unwrap();
    writeln!(
        code,
        "        let r = v128_bitselect(neg_inf, result.0, is_zero);"
    )
    .unwrap();
    writeln!(code, "        let r = v128_bitselect(nan, r, is_neg);").unwrap();
    writeln!(code, "        let r = v128_bitselect(pos_inf, r, is_inf);").unwrap();
    writeln!(code, "        let r = v128_bitselect(nan, r, is_nan);").unwrap();
    writeln!(code, "        Self(r)").unwrap();
    writeln!(code, "    }}\n").unwrap();

    // ===== F32 log2_midp_precise (handles denormals) =====
    writeln!(
        code,
        "    /// Mid-precision base-2 logarithm with full IEEE compliance."
    )
    .unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(code, "    /// Handles all edge cases including denormals.").unwrap();
    writeln!(
        code,
        "    /// About 50% slower than `log2_midp()` due to denormal scaling."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn log2_midp_precise(self) -> Self {{").unwrap();
    writeln!(code, "        // Scale factor for denormals: 2^24").unwrap();
    writeln!(code, "        const SCALE_UP: f32 = 16777216.0;  // 2^24").unwrap();
    writeln!(
        code,
        "        const SCALE_ADJUST: f32 = 24.0;    // log2(2^24)"
    )
    .unwrap();
    writeln!(
        code,
        "        const DENORM_LIMIT: f32 = 1.17549435e-38;  // Smallest normal f32"
    )
    .unwrap();
    writeln!(code, "").unwrap();
    writeln!(
        code,
        "        // Detect denormals (positive values smaller than smallest normal)"
    )
    .unwrap();
    writeln!(code, "        let zero = f32x4_splat(0.0);").unwrap();
    writeln!(code, "        let is_positive = f32x4_gt(self.0, zero);").unwrap();
    writeln!(
        code,
        "        let is_small = f32x4_lt(self.0, f32x4_splat(DENORM_LIMIT));"
    )
    .unwrap();
    writeln!(
        code,
        "        let is_denorm = v128_and(is_positive, is_small);"
    )
    .unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        // Scale up denormals").unwrap();
    writeln!(
        code,
        "        let scaled_x = f32x4_mul(self.0, f32x4_splat(SCALE_UP));"
    )
    .unwrap();
    writeln!(
        code,
        "        let x_for_log = v128_bitselect(scaled_x, self.0, is_denorm);"
    )
    .unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        // Compute log2 with edge case handling").unwrap();
    writeln!(code, "        let result = Self(x_for_log).log2_midp();").unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        // Adjust result for denormals (subtract 24)").unwrap();
    writeln!(
        code,
        "        let adjusted = f32x4_sub(result.0, f32x4_splat(SCALE_ADJUST));"
    )
    .unwrap();
    writeln!(
        code,
        "        Self(v128_bitselect(adjusted, result.0, is_denorm))"
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    // ===== F32 exp2_midp_unchecked =====
    writeln!(
        code,
        "    /// Mid-precision base-2 exponential - unchecked variant."
    )
    .unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// **Warning**: Clamps to finite range. Does not return inf for overflow."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn exp2_midp_unchecked(self) -> Self {{").unwrap();
    writeln!(code, "        const C0: f32 = 1.0;").unwrap();
    writeln!(code, "        const C1: f32 = 0.693_147_180_559_945;").unwrap();
    writeln!(code, "        const C2: f32 = 0.240_226_506_959_101;").unwrap();
    writeln!(code, "        const C3: f32 = 0.055_504_108_664_822;").unwrap();
    writeln!(code, "        const C4: f32 = 0.009_618_129_107_629;").unwrap();
    writeln!(code, "        const C5: f32 = 0.001_333_355_814_497;").unwrap();
    writeln!(code, "        const C6: f32 = 0.000_154_035_303_933;").unwrap();
    writeln!(code, "").unwrap();
    writeln!(
        code,
        "        let x = f32x4_pmax(self.0, f32x4_splat(-126.0));"
    )
    .unwrap();
    writeln!(code, "        let x = f32x4_pmin(x, f32x4_splat(126.0));").unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        let xi = f32x4_floor(x);").unwrap();
    writeln!(code, "        let xf = f32x4_sub(x, xi);").unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        // Horner's method with 6 coefficients").unwrap();
    writeln!(
        code,
        "        let poly = f32x4_add(f32x4_mul(f32x4_splat(C6), xf), f32x4_splat(C5));"
    )
    .unwrap();
    writeln!(
        code,
        "        let poly = f32x4_add(f32x4_mul(poly, xf), f32x4_splat(C4));"
    )
    .unwrap();
    writeln!(
        code,
        "        let poly = f32x4_add(f32x4_mul(poly, xf), f32x4_splat(C3));"
    )
    .unwrap();
    writeln!(
        code,
        "        let poly = f32x4_add(f32x4_mul(poly, xf), f32x4_splat(C2));"
    )
    .unwrap();
    writeln!(
        code,
        "        let poly = f32x4_add(f32x4_mul(poly, xf), f32x4_splat(C1));"
    )
    .unwrap();
    writeln!(
        code,
        "        let poly = f32x4_add(f32x4_mul(poly, xf), f32x4_splat(C0));"
    )
    .unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        let xi_i32 = i32x4_trunc_sat_f32x4(xi);").unwrap();
    writeln!(code, "        let bias = i32x4_splat(127);").unwrap();
    writeln!(
        code,
        "        let scale_bits = i32x4_shl(i32x4_add(xi_i32, bias), 23);"
    )
    .unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        Self(f32x4_mul(poly, scale_bits))").unwrap();
    writeln!(code, "    }}\n").unwrap();

    // ===== F32 exp2_midp (checked) =====
    writeln!(
        code,
        "    /// Mid-precision base-2 exponential (~8e-6 max relative error)."
    )
    .unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// Handles edge cases: exp2(x > 128) = +inf, exp2(x < -150) = 0,"
    )
    .unwrap();
    writeln!(code, "    /// exp2(NaN) = NaN.").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn exp2_midp(self) -> Self {{").unwrap();
    writeln!(code, "        let result = self.exp2_midp_unchecked();").unwrap();
    writeln!(code, "").unwrap();
    writeln!(
        code,
        "        let is_overflow = f32x4_ge(self.0, f32x4_splat(128.0));"
    )
    .unwrap();
    writeln!(
        code,
        "        let is_underflow = f32x4_lt(self.0, f32x4_splat(-150.0));"
    )
    .unwrap();
    writeln!(
        code,
        "        let is_nan = v128_not(f32x4_eq(self.0, self.0));"
    )
    .unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "        let pos_inf = f32x4_splat(f32::INFINITY);").unwrap();
    writeln!(code, "        let zero = f32x4_splat(0.0);").unwrap();
    writeln!(code, "        let nan = f32x4_splat(f32::NAN);").unwrap();
    writeln!(code, "").unwrap();
    writeln!(
        code,
        "        let r = v128_bitselect(pos_inf, result.0, is_overflow);"
    )
    .unwrap();
    writeln!(
        code,
        "        let r = v128_bitselect(zero, r, is_underflow);"
    )
    .unwrap();
    writeln!(code, "        let r = v128_bitselect(nan, r, is_nan);").unwrap();
    writeln!(code, "        Self(r)").unwrap();
    writeln!(code, "    }}\n").unwrap();

    // ===== F32 ln_midp =====
    writeln!(code, "    /// Mid-precision natural logarithm.").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(code, "    /// Computed as `log2_midp(x) * ln(2)`.").unwrap();
    writeln!(code, "    /// Handles edge cases correctly.").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn ln_midp(self) -> Self {{").unwrap();
    writeln!(code, "        const LN2: f32 = core::f32::consts::LN_2;").unwrap();
    writeln!(
        code,
        "        Self(f32x4_mul(self.log2_midp().0, f32x4_splat(LN2)))"
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    // ===== F32 exp_midp =====
    writeln!(code, "    /// Mid-precision natural exponential (e^x).").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(code, "    /// Computed as `exp2_midp(x * log2(e))`.").unwrap();
    writeln!(code, "    /// Handles edge cases correctly.").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn exp_midp(self) -> Self {{").unwrap();
    writeln!(
        code,
        "        const LOG2_E: f32 = core::f32::consts::LOG2_E;"
    )
    .unwrap();
    writeln!(
        code,
        "        Self(f32x4_mul(self.0, f32x4_splat(LOG2_E))).exp2_midp()"
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    // ===== F32 pow_midp =====
    writeln!(code, "    /// Mid-precision power function (self^n).").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// Computed as `exp2_midp(n * log2_midp(self))`."
    )
    .unwrap();
    writeln!(
        code,
        "    /// Achieves 100% exact round-trips for 8-bit, 10-bit, and 12-bit values."
    )
    .unwrap();
    writeln!(code, "    /// Note: Only valid for positive self values.").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn pow_midp(self, n: f32) -> Self {{").unwrap();
    writeln!(
        code,
        "        Self(f32x4_mul(self.log2_midp().0, f32x4_splat(n))).exp2_midp()"
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();
}

fn generate_f64_transcendentals(code: &mut String) {
    // For f64, use scalar fallback since WASM f64x2 bit manipulation is more complex
    // and there are only 2 lanes anyway

    // ===== F64 log2_lowp =====
    writeln!(code, "    /// Low-precision base-2 logarithm.").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// Uses scalar fallback for simplicity (only 2 lanes)."
    )
    .unwrap();
    writeln!(code, "    /// Handles edge cases correctly via std.").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn log2_lowp(self) -> Self {{").unwrap();
    writeln!(code, "        let arr = self.to_array();").unwrap();
    writeln!(code, "        let result = [arr[0].log2(), arr[1].log2()];").unwrap();
    writeln!(code, "        Self::from(result)").unwrap();
    writeln!(code, "    }}\n").unwrap();

    // ===== F64 exp2_lowp =====
    writeln!(code, "    /// Low-precision base-2 exponential (2^x).").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(code, "    /// Handles edge cases correctly via std.").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn exp2_lowp(self) -> Self {{").unwrap();
    writeln!(code, "        let arr = self.to_array();").unwrap();
    writeln!(code, "        let result = [arr[0].exp2(), arr[1].exp2()];").unwrap();
    writeln!(code, "        Self::from(result)").unwrap();
    writeln!(code, "    }}\n").unwrap();

    // ===== F64 ln_lowp =====
    writeln!(code, "    /// Low-precision natural logarithm.").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(code, "    /// Handles edge cases correctly via std.").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn ln_lowp(self) -> Self {{").unwrap();
    writeln!(code, "        let arr = self.to_array();").unwrap();
    writeln!(code, "        let result = [arr[0].ln(), arr[1].ln()];").unwrap();
    writeln!(code, "        Self::from(result)").unwrap();
    writeln!(code, "    }}\n").unwrap();

    // ===== F64 exp_lowp =====
    writeln!(code, "    /// Low-precision natural exponential (e^x).").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(code, "    /// Handles edge cases correctly via std.").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn exp_lowp(self) -> Self {{").unwrap();
    writeln!(code, "        let arr = self.to_array();").unwrap();
    writeln!(code, "        let result = [arr[0].exp(), arr[1].exp()];").unwrap();
    writeln!(code, "        Self::from(result)").unwrap();
    writeln!(code, "    }}\n").unwrap();

    // ===== F64 pow_lowp =====
    writeln!(code, "    /// Low-precision power function (self^n).").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(code, "    /// Handles edge cases correctly via std.").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn pow_lowp(self, n: f64) -> Self {{").unwrap();
    writeln!(code, "        let arr = self.to_array();").unwrap();
    writeln!(
        code,
        "        let result = [arr[0].powf(n), arr[1].powf(n)];"
    )
    .unwrap();
    writeln!(code, "        Self::from(result)").unwrap();
    writeln!(code, "    }}\n").unwrap();
}

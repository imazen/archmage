//! WebAssembly SIMD transcendental operations (log, exp, pow).
//!
//! Pure polynomial approximations - no native transcendental intrinsics in WASM SIMD.
//! All operations use v128 type directly since that's what WASM intrinsics expect.
//!
//! Provides both unchecked (fast) and checked (handles edge cases) variants.

use super::types::{ElementType, SimdType, SimdWidth};
use indoc::formatdoc;

/// Generate WASM transcendental operations for float types
pub fn generate_wasm_transcendental_ops(ty: &SimdType) -> String {
    // Only for float types
    if !ty.elem.is_float() {
        return String::new();
    }

    assert!(
        ty.width == SimdWidth::W128,
        "WASM only supports 128-bit vectors"
    );

    let mut code = formatdoc! {"
        // ========== Transcendental Operations (Polynomial Approximations) ==========
        //
        // WASM SIMD has no native transcendental intrinsics.
        // Provides _unchecked (fast) and checked (handles edge cases) variants.

    "};

    if ty.elem == ElementType::F32 {
        code.push_str(&generate_f32_transcendentals());
    } else if ty.elem == ElementType::F64 {
        code.push_str(&generate_f64_transcendentals());
    }

    code
}

fn generate_f32_transcendentals() -> String {
    formatdoc! {r#"
        /// Low-precision base-2 logarithm - unchecked variant.
        ///
        /// **Warning**: Does not handle edge cases (0, negative, inf, NaN).
        /// Use `log2_lowp()` for correct IEEE behavior.
        #[inline(always)]
        pub fn log2_lowp_unchecked(self) -> Self {{
        // Rational polynomial coefficients from butteraugli/jpegli
        const P0: f32 = -1.850_383_34e-6;
        const P1: f32 = 1.428_716_05;
        const P2: f32 = 0.742_458_73;
        const Q0: f32 = 0.990_328_14;
        const Q1: f32 = 1.009_671_86;
        const Q2: f32 = 0.174_093_43;
        const OFFSET: u32 = 0x3f2aaaab;

        let x = self.0;

        // Extract exponent via bit manipulation
        let offset = u32x4_splat(OFFSET);
        let exp_bits = i32x4_sub(x, offset);
        let exp_shifted = i32x4_shr(exp_bits, 23);

        // Reconstruct mantissa in [1, 2) range
        let mantissa_bits = i32x4_sub(x, i32x4_shl(exp_shifted, 23));
        let exp_val = f32x4_convert_i32x4(exp_shifted);

        // m = mantissa - 1
        let one = f32x4_splat(1.0);
        let m = f32x4_sub(mantissa_bits, one);

        // Horner's for numerator: P2*m^2 + P1*m + P0
        let m2 = f32x4_mul(m, m);
        let p2_m2 = f32x4_mul(f32x4_splat(P2), m2);
        let p1_m = f32x4_mul(f32x4_splat(P1), m);
        let yp = f32x4_add(p2_m2, f32x4_add(p1_m, f32x4_splat(P0)));

        // Horner's for denominator: Q2*m^2 + Q1*m + Q0
        let q2_m2 = f32x4_mul(f32x4_splat(Q2), m2);
        let q1_m = f32x4_mul(f32x4_splat(Q1), m);
        let yq = f32x4_add(q2_m2, f32x4_add(q1_m, f32x4_splat(Q0)));

        Self(f32x4_add(f32x4_div(yp, yq), exp_val))
        }}

        /// Low-precision base-2 logarithm (~7.7e-5 max relative error).
        ///
        /// Handles edge cases: log2(0) = -inf, log2(negative) = NaN,
        /// log2(+inf) = +inf, log2(NaN) = NaN.
        #[inline(always)]
        pub fn log2_lowp(self) -> Self {{
        let result = self.log2_lowp_unchecked();

        // Edge case masks
        let zero = f32x4_splat(0.0);
        let is_zero = f32x4_eq(self.0, zero);
        let is_neg = f32x4_lt(self.0, zero);
        let is_inf = f32x4_eq(self.0, f32x4_splat(f32::INFINITY));
        // NaN: x != x
        let is_nan = v128_not(f32x4_eq(self.0, self.0));

        // Apply corrections using bitselect
        let neg_inf = f32x4_splat(f32::NEG_INFINITY);
        let pos_inf = f32x4_splat(f32::INFINITY);
        let nan = f32x4_splat(f32::NAN);

        let r = v128_bitselect(neg_inf, result.0, is_zero);  // 0 -> -inf
        let r = v128_bitselect(nan, r, is_neg);              // neg -> NaN
        let r = v128_bitselect(pos_inf, r, is_inf);          // inf -> inf
        let r = v128_bitselect(nan, r, is_nan);              // NaN -> NaN
        Self(r)
        }}

        /// Low-precision base-2 exponential - unchecked variant.
        ///
        /// **Warning**: Clamps to [-126, 126]. Does not return inf for overflow.
        #[inline(always)]
        pub fn exp2_lowp_unchecked(self) -> Self {{
        const C0: f32 = 1.0;
        const C1: f32 = core::f32::consts::LN_2;
        const C2: f32 = 0.240_226_5;
        const C3: f32 = 0.055_504_11;

        // Clamp to safe range
        let x = f32x4_pmax(self.0, f32x4_splat(-126.0));
        let x = f32x4_pmin(x, f32x4_splat(126.0));

        let xi = f32x4_floor(x);
        let xf = f32x4_sub(x, xi);

        // Polynomial for 2^frac
        let poly = f32x4_add(f32x4_mul(f32x4_splat(C3), xf), f32x4_splat(C2));
        let poly = f32x4_add(f32x4_mul(poly, xf), f32x4_splat(C1));
        let poly = f32x4_add(f32x4_mul(poly, xf), f32x4_splat(C0));

        // Scale by 2^integer
        let xi_i32 = i32x4_trunc_sat_f32x4(xi);
        let bias = i32x4_splat(127);
        let scale_bits = i32x4_shl(i32x4_add(xi_i32, bias), 23);

        Self(f32x4_mul(poly, scale_bits))
        }}

        /// Low-precision base-2 exponential (~5.5e-3 max relative error).
        ///
        /// Handles edge cases: exp2(x > 128) = +inf, exp2(x < -150) = 0,
        /// exp2(NaN) = NaN.
        #[inline(always)]
        pub fn exp2_lowp(self) -> Self {{
        let result = self.exp2_lowp_unchecked();

        // Edge case masks
        let is_overflow = f32x4_ge(self.0, f32x4_splat(128.0));
        let is_underflow = f32x4_lt(self.0, f32x4_splat(-150.0));
        let is_nan = v128_not(f32x4_eq(self.0, self.0));

        let pos_inf = f32x4_splat(f32::INFINITY);
        let zero = f32x4_splat(0.0);
        let nan = f32x4_splat(f32::NAN);

        let r = v128_bitselect(pos_inf, result.0, is_overflow);
        let r = v128_bitselect(zero, r, is_underflow);
        let r = v128_bitselect(nan, r, is_nan);
        Self(r)
        }}

        /// Low-precision natural logarithm - unchecked variant.
        ///
        /// **Warning**: Does not handle edge cases (0, negative, inf, NaN).
        #[inline(always)]
        pub fn ln_lowp_unchecked(self) -> Self {{
        const LN2: f32 = core::f32::consts::LN_2;
        Self(f32x4_mul(self.log2_lowp_unchecked().0, f32x4_splat(LN2)))
        }}

        /// Low-precision natural logarithm.
        ///
        /// Computed as `log2_lowp(x) * ln(2)`.
        /// Handles edge cases correctly.
        #[inline(always)]
        pub fn ln_lowp(self) -> Self {{
        const LN2: f32 = core::f32::consts::LN_2;
        Self(f32x4_mul(self.log2_lowp().0, f32x4_splat(LN2)))
        }}

        /// Low-precision natural exponential - unchecked variant.
        ///
        /// **Warning**: Clamps to finite range. Does not return inf for overflow.
        #[inline(always)]
        pub fn exp_lowp_unchecked(self) -> Self {{
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        Self(f32x4_mul(self.0, f32x4_splat(LOG2_E))).exp2_lowp_unchecked()
        }}

        /// Low-precision natural exponential (e^x).
        ///
        /// Computed as `exp2_lowp(x * log2(e))`.
        /// Handles edge cases correctly.
        #[inline(always)]
        pub fn exp_lowp(self) -> Self {{
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        Self(f32x4_mul(self.0, f32x4_splat(LOG2_E))).exp2_lowp()
        }}

        /// Low-precision base-10 logarithm - unchecked variant.
        ///
        /// **Warning**: Does not handle edge cases (0, negative, inf, NaN).
        #[inline(always)]
        pub fn log10_lowp_unchecked(self) -> Self {{
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        Self(f32x4_mul(self.log2_lowp_unchecked().0, f32x4_splat(LOG10_2)))
        }}

        /// Low-precision base-10 logarithm.
        ///
        /// Computed as `log2_lowp(x) * log10(2)`.
        /// Handles edge cases correctly.
        #[inline(always)]
        pub fn log10_lowp(self) -> Self {{
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        Self(f32x4_mul(self.log2_lowp().0, f32x4_splat(LOG10_2)))
        }}

        /// Low-precision power function - unchecked variant.
        ///
        /// **Warning**: Does not handle edge cases. Only valid for positive self values.
        #[inline(always)]
        pub fn pow_lowp_unchecked(self, n: f32) -> Self {{
        Self(f32x4_mul(self.log2_lowp_unchecked().0, f32x4_splat(n))).exp2_lowp_unchecked()
        }}

        /// Low-precision power function (self^n).
        ///
        /// Computed as `exp2_lowp(n * log2_lowp(self))`.
        /// Note: Only valid for positive self values.
        #[inline(always)]
        pub fn pow_lowp(self, n: f32) -> Self {{
        Self(f32x4_mul(self.log2_lowp().0, f32x4_splat(n))).exp2_lowp()
        }}

        // ========== Mid-Precision Transcendental Operations ==========

        /// Mid-precision base-2 logarithm - unchecked variant.
        ///
        /// **Warning**: Does not handle edge cases (0, negative, inf, NaN, denormals).
        #[inline(always)]
        pub fn log2_midp_unchecked(self) -> Self {{
        const SQRT2_OVER_2: u32 = 0x3f3504f3;
        const ONE: u32 = 0x3f800000;
        const MANTISSA_MASK: u32 = 0x007fffff;
        const EXPONENT_BIAS: i32 = 127;

        // Coefficients for odd polynomial
        const C0: f32 = 2.885_390_08;
        const C1: f32 = 0.961_800_76;
        const C2: f32 = 0.576_974_45;
        const C3: f32 = 0.434_411_97;

        let x = self.0;

        // Normalize mantissa to [sqrt(2)/2, sqrt(2)]
        let offset = u32x4_splat(ONE - SQRT2_OVER_2);
        let adjusted = i32x4_add(x, offset);

        // Extract exponent
        let exp_raw = i32x4_shr(adjusted, 23);
        let exp_biased = i32x4_sub(exp_raw, i32x4_splat(EXPONENT_BIAS));
        let n = f32x4_convert_i32x4(exp_biased);

        // Reconstruct normalized mantissa
        let mantissa_bits = v128_and(adjusted, u32x4_splat(MANTISSA_MASK));
        let a_bits = i32x4_add(mantissa_bits, u32x4_splat(SQRT2_OVER_2));

        // y = (a - 1) / (a + 1)
        let one = f32x4_splat(1.0);
        let a_minus_1 = f32x4_sub(a_bits, one);
        let a_plus_1 = f32x4_add(a_bits, one);
        let y = f32x4_div(a_minus_1, a_plus_1);

        let y2 = f32x4_mul(y, y);

        // Polynomial: c0 + y^2*(c1 + y^2*(c2 + y^2*c3))
        let poly = f32x4_add(f32x4_mul(f32x4_splat(C3), y2), f32x4_splat(C2));
        let poly = f32x4_add(f32x4_mul(poly, y2), f32x4_splat(C1));
        let poly = f32x4_add(f32x4_mul(poly, y2), f32x4_splat(C0));

        Self(f32x4_add(f32x4_mul(y, poly), n))
        }}

        /// Mid-precision base-2 logarithm (~3 ULP max error).
        ///
        /// Handles edge cases: log2(0) = -inf, log2(negative) = NaN,
        /// log2(+inf) = +inf, log2(NaN) = NaN.
        ///
        /// Note: Does not handle denormals. Use `log2_midp_precise()` for full IEEE.
        #[inline(always)]
        pub fn log2_midp(self) -> Self {{
        let result = self.log2_midp_unchecked();

        let zero = f32x4_splat(0.0);
        let is_zero = f32x4_eq(self.0, zero);
        let is_neg = f32x4_lt(self.0, zero);
        let is_inf = f32x4_eq(self.0, f32x4_splat(f32::INFINITY));
        let is_nan = v128_not(f32x4_eq(self.0, self.0));

        let neg_inf = f32x4_splat(f32::NEG_INFINITY);
        let pos_inf = f32x4_splat(f32::INFINITY);
        let nan = f32x4_splat(f32::NAN);

        let r = v128_bitselect(neg_inf, result.0, is_zero);
        let r = v128_bitselect(nan, r, is_neg);
        let r = v128_bitselect(pos_inf, r, is_inf);
        let r = v128_bitselect(nan, r, is_nan);
        Self(r)
        }}

        /// Mid-precision base-2 logarithm with full IEEE compliance.
        ///
        /// Handles all edge cases including denormals.
        /// About 50% slower than `log2_midp()` due to denormal scaling.
        #[inline(always)]
        pub fn log2_midp_precise(self) -> Self {{
        // Scale factor for denormals: 2^24
        const SCALE_UP: f32 = 16777216.0;  // 2^24
        const SCALE_ADJUST: f32 = 24.0;    // log2(2^24)
        const DENORM_LIMIT: f32 = 1.17549435e-38;  // Smallest normal f32

        // Detect denormals (positive values smaller than smallest normal)
        let zero = f32x4_splat(0.0);
        let is_positive = f32x4_gt(self.0, zero);
        let is_small = f32x4_lt(self.0, f32x4_splat(DENORM_LIMIT));
        let is_denorm = v128_and(is_positive, is_small);

        // Scale up denormals
        let scaled_x = f32x4_mul(self.0, f32x4_splat(SCALE_UP));
        let x_for_log = v128_bitselect(scaled_x, self.0, is_denorm);

        // Compute log2 with edge case handling
        let result = Self(x_for_log).log2_midp();

        // Adjust result for denormals (subtract 24)
        let adjusted = f32x4_sub(result.0, f32x4_splat(SCALE_ADJUST));
        Self(v128_bitselect(adjusted, result.0, is_denorm))
        }}

        /// Mid-precision base-2 exponential - unchecked variant.
        ///
        /// **Warning**: Clamps to finite range. Does not return inf for overflow.
        #[inline(always)]
        pub fn exp2_midp_unchecked(self) -> Self {{
        const C0: f32 = 1.0;
        const C1: f32 = 0.693_147_180_559_945;
        const C2: f32 = 0.240_226_506_959_101;
        const C3: f32 = 0.055_504_108_664_822;
        const C4: f32 = 0.009_618_129_107_629;
        const C5: f32 = 0.001_333_355_814_497;
        const C6: f32 = 0.000_154_035_303_933;

        let x = f32x4_pmax(self.0, f32x4_splat(-126.0));
        let x = f32x4_pmin(x, f32x4_splat(126.0));

        let xi = f32x4_floor(x);
        let xf = f32x4_sub(x, xi);

        // Horner's method with 6 coefficients
        let poly = f32x4_add(f32x4_mul(f32x4_splat(C6), xf), f32x4_splat(C5));
        let poly = f32x4_add(f32x4_mul(poly, xf), f32x4_splat(C4));
        let poly = f32x4_add(f32x4_mul(poly, xf), f32x4_splat(C3));
        let poly = f32x4_add(f32x4_mul(poly, xf), f32x4_splat(C2));
        let poly = f32x4_add(f32x4_mul(poly, xf), f32x4_splat(C1));
        let poly = f32x4_add(f32x4_mul(poly, xf), f32x4_splat(C0));

        let xi_i32 = i32x4_trunc_sat_f32x4(xi);
        let bias = i32x4_splat(127);
        let scale_bits = i32x4_shl(i32x4_add(xi_i32, bias), 23);

        Self(f32x4_mul(poly, scale_bits))
        }}

        /// Mid-precision base-2 exponential (~8e-6 max relative error).
        ///
        /// Handles edge cases: exp2(x > 128) = +inf, exp2(x < -150) = 0,
        /// exp2(NaN) = NaN.
        #[inline(always)]
        pub fn exp2_midp(self) -> Self {{
        let result = self.exp2_midp_unchecked();

        let is_overflow = f32x4_ge(self.0, f32x4_splat(128.0));
        let is_underflow = f32x4_lt(self.0, f32x4_splat(-150.0));
        let is_nan = v128_not(f32x4_eq(self.0, self.0));

        let pos_inf = f32x4_splat(f32::INFINITY);
        let zero = f32x4_splat(0.0);
        let nan = f32x4_splat(f32::NAN);

        let r = v128_bitselect(pos_inf, result.0, is_overflow);
        let r = v128_bitselect(zero, r, is_underflow);
        let r = v128_bitselect(nan, r, is_nan);
        Self(r)
        }}

        /// Mid-precision natural logarithm - unchecked variant.
        ///
        /// **Warning**: Does not handle edge cases (0, negative, inf, NaN, denormals).
        #[inline(always)]
        pub fn ln_midp_unchecked(self) -> Self {{
        const LN2: f32 = core::f32::consts::LN_2;
        Self(f32x4_mul(self.log2_midp_unchecked().0, f32x4_splat(LN2)))
        }}

        /// Mid-precision natural logarithm.
        ///
        /// Computed as `log2_midp(x) * ln(2)`.
        /// Handles edge cases correctly.
        ///
        /// Note: Does not handle denormals. Use `ln_midp_precise()` for full IEEE.
        #[inline(always)]
        pub fn ln_midp(self) -> Self {{
        const LN2: f32 = core::f32::consts::LN_2;
        Self(f32x4_mul(self.log2_midp().0, f32x4_splat(LN2)))
        }}

        /// Mid-precision natural logarithm with full IEEE compliance.
        ///
        /// Handles all edge cases including denormals.
        /// About 50% slower than `ln_midp()` due to denormal scaling.
        #[inline(always)]
        pub fn ln_midp_precise(self) -> Self {{
        const LN2: f32 = core::f32::consts::LN_2;
        Self(f32x4_mul(self.log2_midp_precise().0, f32x4_splat(LN2)))
        }}

        /// Mid-precision natural exponential - unchecked variant.
        ///
        /// **Warning**: Clamps to finite range. Does not return inf for overflow.
        #[inline(always)]
        pub fn exp_midp_unchecked(self) -> Self {{
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        Self(f32x4_mul(self.0, f32x4_splat(LOG2_E))).exp2_midp_unchecked()
        }}

        /// Mid-precision natural exponential (e^x).
        ///
        /// Computed as `exp2_midp(x * log2(e))`.
        /// Handles edge cases correctly.
        #[inline(always)]
        pub fn exp_midp(self) -> Self {{
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        Self(f32x4_mul(self.0, f32x4_splat(LOG2_E))).exp2_midp()
        }}

        /// Mid-precision base-10 logarithm - unchecked variant.
        ///
        /// **Warning**: Does not handle edge cases (0, negative, inf, NaN, denormals).
        #[inline(always)]
        pub fn log10_midp_unchecked(self) -> Self {{
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        Self(f32x4_mul(self.log2_midp_unchecked().0, f32x4_splat(LOG10_2)))
        }}

        /// Mid-precision base-10 logarithm.
        ///
        /// Computed as `log2_midp(x) * log10(2)`.
        /// Handles edge cases correctly.
        ///
        /// Note: Does not handle denormals. Use `log10_midp_precise()` for full IEEE.
        #[inline(always)]
        pub fn log10_midp(self) -> Self {{
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        Self(f32x4_mul(self.log2_midp().0, f32x4_splat(LOG10_2)))
        }}

        /// Mid-precision base-10 logarithm with full IEEE compliance.
        ///
        /// Handles all edge cases including denormals.
        /// About 50% slower than `log10_midp()` due to denormal scaling.
        #[inline(always)]
        pub fn log10_midp_precise(self) -> Self {{
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        Self(f32x4_mul(self.log2_midp_precise().0, f32x4_splat(LOG10_2)))
        }}

        /// Mid-precision power function - unchecked variant.
        ///
        /// **Warning**: Does not handle edge cases. Only valid for positive self values.
        #[inline(always)]
        pub fn pow_midp_unchecked(self, n: f32) -> Self {{
        Self(f32x4_mul(self.log2_midp_unchecked().0, f32x4_splat(n))).exp2_midp_unchecked()
        }}

        /// Mid-precision power function (self^n).
        ///
        /// Computed as `exp2_midp(n * log2_midp(self))`.
        /// Achieves 100% exact round-trips for 8-bit, 10-bit, and 12-bit values.
        /// Note: Only valid for positive self values.
        #[inline(always)]
        pub fn pow_midp(self, n: f32) -> Self {{
        Self(f32x4_mul(self.log2_midp().0, f32x4_splat(n))).exp2_midp()
        }}

        /// Mid-precision power function with denormal handling.
        ///
        /// Uses `log2_midp_precise()` to handle denormal inputs correctly.
        /// Note: Only valid for positive self values.
        #[inline(always)]
        pub fn pow_midp_precise(self, n: f32) -> Self {{
        Self(f32x4_mul(self.log2_midp_precise().0, f32x4_splat(n))).exp2_midp()
        }}

    "#}
}

fn generate_f64_transcendentals() -> String {
    // For f64, use scalar fallback since WASM f64x2 bit manipulation is more complex
    // and there are only 2 lanes anyway
    formatdoc! {r#"
        /// Low-precision base-2 logarithm.
        ///
        /// Uses scalar fallback for simplicity (only 2 lanes).
        /// Handles edge cases correctly via std.
        #[inline(always)]
        pub fn log2_lowp(self) -> Self {{
        let arr = self.to_array();
        let result = [arr[0].log2(), arr[1].log2()];
        Self::from(result)
        }}

        /// Low-precision base-2 exponential (2^x).
        ///
        /// Handles edge cases correctly via std.
        #[inline(always)]
        pub fn exp2_lowp(self) -> Self {{
        let arr = self.to_array();
        let result = [arr[0].exp2(), arr[1].exp2()];
        Self::from(result)
        }}

        /// Low-precision natural logarithm.
        ///
        /// Handles edge cases correctly via std.
        #[inline(always)]
        pub fn ln_lowp(self) -> Self {{
        let arr = self.to_array();
        let result = [arr[0].ln(), arr[1].ln()];
        Self::from(result)
        }}

        /// Low-precision natural exponential (e^x).
        ///
        /// Handles edge cases correctly via std.
        #[inline(always)]
        pub fn exp_lowp(self) -> Self {{
        let arr = self.to_array();
        let result = [arr[0].exp(), arr[1].exp()];
        Self::from(result)
        }}

        /// Low-precision power function (self^n).
        ///
        /// Handles edge cases correctly via std.
        #[inline(always)]
        pub fn pow_lowp(self, n: f64) -> Self {{
        let arr = self.to_array();
        let result = [arr[0].powf(n), arr[1].powf(n)];
        Self::from(result)
        }}

    "#}
}

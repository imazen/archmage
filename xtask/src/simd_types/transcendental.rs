//! Transcendental operations (log, exp, pow, cbrt).

use super::types::{ElementType, SimdType, SimdWidth};
use std::fmt::Write;

pub fn generate_transcendental_ops(ty: &SimdType) -> String {
    let mut code = String::new();

    // Only for float types
    if !ty.elem.is_float() {
        return code;
    }

    let prefix = ty.width.x86_prefix();
    let suffix = ty.elem.x86_suffix();
    let _bits = ty.width.bits();

    // Determine integer suffix for cast operations
    let int_suffix = if ty.elem == ElementType::F32 {
        "si256"
    } else {
        "si256"
    };
    let int_suffix_512 = if ty.elem == ElementType::F32 {
        "si512"
    } else {
        "si512"
    };
    let int_suffix_128 = if ty.elem == ElementType::F32 {
        "si128"
    } else {
        "si128"
    };

    let actual_int_suffix = match ty.width {
        SimdWidth::W128 => int_suffix_128,
        SimdWidth::W256 => int_suffix,
        SimdWidth::W512 => int_suffix_512,
    };

    writeln!(
        code,
        "    // ========== Transcendental Operations ==========\n"
    )
    .unwrap();

    if ty.elem == ElementType::F32 {
        // ===== F32 log2_lowp =====
        writeln!(
            code,
            "    /// Low-precision base-2 logarithm (~7.7e-5 max relative error)."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(code, "    /// Uses rational polynomial approximation. Fast but not suitable for color-accurate work.").unwrap();
        writeln!(code, "    /// For higher precision, use `log2_midp()`.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn log2_lowp(self) -> Self {{").unwrap();
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
        writeln!(code, "").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            let x_bits = {}_cast{}_{}(self.0);",
            prefix, suffix, actual_int_suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let offset = {}_set1_epi32(0x3f2aaaab_u32 as i32);",
            prefix
        )
        .unwrap();
        writeln!(
            code,
            "            let exp_bits = {}_sub_epi32(x_bits, offset);",
            prefix
        )
        .unwrap();
        writeln!(
            code,
            "            let exp_shifted = {}_srai_epi32::<23>(exp_bits);",
            prefix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            let mantissa_bits = {}_sub_epi32(x_bits, {}_slli_epi32::<23>(exp_shifted));", prefix, prefix).unwrap();
        writeln!(
            code,
            "            let mantissa = {}_cast{}_{}(mantissa_bits);",
            prefix, actual_int_suffix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let exp_val = {}_cvtepi32_{}(exp_shifted);",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            let one = {}_set1_{}(1.0);",
            prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let m = {}_sub_{}(mantissa, one);",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            // Horner's for numerator: P2*m^2 + P1*m + P0"
        )
        .unwrap();
        writeln!(
            code,
            "            let yp = {}_fmadd_{}({}_set1_{}(P2), m, {}_set1_{}(P1));",
            prefix, suffix, prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let yp = {}_fmadd_{}(yp, m, {}_set1_{}(P0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            // Horner's for denominator: Q2*m^2 + Q1*m + Q0"
        )
        .unwrap();
        writeln!(
            code,
            "            let yq = {}_fmadd_{}({}_set1_{}(Q2), m, {}_set1_{}(Q1));",
            prefix, suffix, prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let yq = {}_fmadd_{}(yq, m, {}_set1_{}(Q0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            Self({}_add_{}({}_div_{}(yp, yq), exp_val))",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 exp2_lowp =====
        writeln!(
            code,
            "    /// Low-precision base-2 exponential (~5.5e-3 max relative error)."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(code, "    /// Uses degree-3 polynomial approximation. Fast but not suitable for color-accurate work.").unwrap();
        writeln!(code, "    /// For higher precision, use `exp2_midp()`.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn exp2_lowp(self) -> Self {{").unwrap();
        writeln!(code, "        // Polynomial coefficients").unwrap();
        writeln!(code, "        const C0: f32 = 1.0;").unwrap();
        writeln!(code, "        const C1: f32 = core::f32::consts::LN_2;").unwrap();
        writeln!(code, "        const C2: f32 = 0.240_226_5;").unwrap();
        writeln!(code, "        const C3: f32 = 0.055_504_11;").unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(code, "            // Clamp to safe range").unwrap();
        writeln!(
            code,
            "            let x = {}_max_{}(self.0, {}_set1_{}(-126.0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let x = {}_min_{}(x, {}_set1_{}(126.0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            // Split into integer and fractional parts"
        )
        .unwrap();

        // Use appropriate floor/round intrinsic based on width
        if ty.width == SimdWidth::W512 {
            writeln!(
                code,
                "            let xi = {}_roundscale_{}::<0x01>(x); // floor",
                prefix, suffix
            )
            .unwrap();
        } else {
            writeln!(code, "            let xi = {}_floor_{}(x);", prefix, suffix).unwrap();
        }

        writeln!(
            code,
            "            let xf = {}_sub_{}(x, xi);",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // Polynomial for 2^frac").unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}({}_set1_{}(C3), xf, {}_set1_{}(C2));",
            prefix, suffix, prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}(poly, xf, {}_set1_{}(C1));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}(poly, xf, {}_set1_{}(C0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            // Scale by 2^integer using bit manipulation"
        )
        .unwrap();
        writeln!(
            code,
            "            let xi_i32 = {}_cvt{}_epi32(xi);",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "            let bias = {}_set1_epi32(127);", prefix).unwrap();
        writeln!(
            code,
            "            let scale_bits = {}_slli_epi32::<23>({}_add_epi32(xi_i32, bias));",
            prefix, prefix
        )
        .unwrap();
        writeln!(
            code,
            "            let scale = {}_cast{}_{}(scale_bits);",
            prefix, actual_int_suffix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(poly, scale))",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 ln_lowp (natural log) =====
        writeln!(code, "    /// Low-precision natural logarithm.").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Computed as `log2_lowp(x) * ln(2)`. For higher precision, use `ln_midp()`."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn ln_lowp(self) -> Self {{").unwrap();
        writeln!(code, "        const LN2: f32 = core::f32::consts::LN_2;").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.log2_lowp().0, {}_set1_{}(LN2)))",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 exp_lowp (natural exp) =====
        writeln!(code, "    /// Low-precision natural exponential (e^x).").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Computed as `exp2_lowp(x * log2(e))`. For higher precision, use `exp_midp()`."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn exp_lowp(self) -> Self {{").unwrap();
        writeln!(
            code,
            "        const LOG2_E: f32 = core::f32::consts::LOG2_E;"
        )
        .unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.0, {}_set1_{}(LOG2_E))).exp2_lowp()",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 log10_lowp =====
        writeln!(code, "    /// Low-precision base-10 logarithm.").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(code, "    /// Computed as `log2_lowp(x) / log2(10)`.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn log10_lowp(self) -> Self {{").unwrap();
        writeln!(
            code,
            "        const LOG10_2: f32 = core::f32::consts::LOG10_2; // 1/log2(10)"
        )
        .unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.log2_lowp().0, {}_set1_{}(LOG10_2)))",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 pow_lowp =====
        writeln!(code, "    /// Low-precision power function (self^n).").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(code, "    /// Computed as `exp2_lowp(n * log2_lowp(self))`. For higher precision, use `pow_midp()`.").unwrap();
        writeln!(code, "    /// Note: Only valid for positive self values.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn pow_lowp(self, n: f32) -> Self {{").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.log2_lowp().0, {}_set1_{}(n))).exp2_lowp()",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ========== Mid-Precision Transcendental Operations ==========
        writeln!(
            code,
            "    // ========== Mid-Precision Transcendental Operations ==========\n"
        )
        .unwrap();

        // ===== F32 log2_midp_unchecked =====
        writeln!(
            code,
            "    /// Mid-precision base-2 logarithm (~3 ULP max error) - unchecked variant."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Uses (a-1)/(a+1) transform with degree-6 odd polynomial."
        )
        .unwrap();
        writeln!(
            code,
            "    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN)."
        )
        .unwrap();
        writeln!(
            code,
            "    /// Use `log2_midp()` for correct IEEE behavior on edge cases."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn log2_midp_unchecked(self) -> Self {{").unwrap();
        writeln!(code, "        // Constants for range reduction").unwrap();
        writeln!(
            code,
            "        const SQRT2_OVER_2: u32 = 0x3f3504f3; // sqrt(2)/2 in f32 bits"
        )
        .unwrap();
        writeln!(
            code,
            "        const ONE: u32 = 0x3f800000;          // 1.0 in f32 bits"
        )
        .unwrap();
        writeln!(
            code,
            "        const MANTISSA_MASK: i32 = 0x007fffff_u32 as i32;"
        )
        .unwrap();
        writeln!(code, "        const EXPONENT_BIAS: i32 = 127;").unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "        // Coefficients for odd polynomial on y = (a-1)/(a+1)"
        )
        .unwrap();
        writeln!(code, "        const C0: f32 = 2.885_390_08;  // 2/ln(2)").unwrap();
        writeln!(
            code,
            "        const C1: f32 = 0.961_800_76;  // y^2 coefficient"
        )
        .unwrap();
        writeln!(
            code,
            "        const C2: f32 = 0.576_974_45;  // y^4 coefficient"
        )
        .unwrap();
        writeln!(
            code,
            "        const C3: f32 = 0.434_411_97;  // y^6 coefficient"
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            let x_bits = {}_cast{}_{}(self.0);",
            prefix, suffix, actual_int_suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            // Normalize mantissa to [sqrt(2)/2, sqrt(2)]"
        )
        .unwrap();
        writeln!(
            code,
            "            let offset = {}_set1_epi32((ONE - SQRT2_OVER_2) as i32);",
            prefix
        )
        .unwrap();
        writeln!(
            code,
            "            let adjusted = {}_add_epi32(x_bits, offset);",
            prefix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // Extract exponent").unwrap();
        writeln!(
            code,
            "            let exp_raw = {}_srai_epi32::<23>(adjusted);",
            prefix
        )
        .unwrap();
        writeln!(
            code,
            "            let exp_biased = {}_sub_epi32(exp_raw, {}_set1_epi32(EXPONENT_BIAS));",
            prefix, prefix
        )
        .unwrap();
        writeln!(
            code,
            "            let n = {}_cvtepi32_{}(exp_biased);",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // Reconstruct normalized mantissa").unwrap();
        writeln!(
            code,
            "            let mantissa_bits = {}_and_{}(adjusted, {}_set1_epi32(MANTISSA_MASK));",
            prefix, actual_int_suffix, prefix
        )
        .unwrap();
        writeln!(code, "            let a_bits = {}_add_epi32(mantissa_bits, {}_set1_epi32(SQRT2_OVER_2 as i32));", prefix, prefix).unwrap();
        writeln!(
            code,
            "            let a = {}_cast{}_{}(a_bits);",
            prefix, actual_int_suffix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // y = (a - 1) / (a + 1)").unwrap();
        writeln!(
            code,
            "            let one = {}_set1_{}(1.0);",
            prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let a_minus_1 = {}_sub_{}(a, one);",
            prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let a_plus_1 = {}_add_{}(a, one);",
            prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let y = {}_div_{}(a_minus_1, a_plus_1);",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // y^2").unwrap();
        writeln!(
            code,
            "            let y2 = {}_mul_{}(y, y);",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            // Polynomial: c0 + y^2*(c1 + y^2*(c2 + y^2*c3))"
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}({}_set1_{}(C3), y2, {}_set1_{}(C2));",
            prefix, suffix, prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}(poly, y2, {}_set1_{}(C1));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}(poly, y2, {}_set1_{}(C0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // Result: y * poly + n").unwrap();
        writeln!(
            code,
            "            Self({}_fmadd_{}(y, poly, n))",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 log2_midp (checked, default) =====
        writeln!(
            code,
            "    /// Mid-precision base-2 logarithm (~3 ULP max error)."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Uses (a-1)/(a+1) transform with degree-6 odd polynomial."
        )
        .unwrap();
        writeln!(
            code,
            "    /// Suitable for 8-bit, 10-bit, and 12-bit color processing."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Handles edge cases correctly: log2(0) = -inf, log2(negative) = NaN,"
        )
        .unwrap();
        writeln!(code, "    /// log2(+inf) = +inf, log2(NaN) = NaN.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn log2_midp(self) -> Self {{").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(code, "            let result = self.log2_midp_unchecked();").unwrap();
        writeln!(code, "").unwrap();

        if ty.width == SimdWidth::W512 {
            // AVX-512 uses mask registers
            writeln!(
                code,
                "            // Edge case masks (AVX-512 uses mask registers)"
            )
            .unwrap();
            writeln!(
                code,
                "            let zero = {}_setzero_{}();",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let is_zero = {}_cmp_{}_mask::<_CMP_EQ_OQ>(self.0, zero);",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let is_neg = {}_cmp_{}_mask::<_CMP_LT_OQ>(self.0, zero);",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let is_inf = {}_cmp_{}_mask::<_CMP_EQ_OQ>(self.0, {}_set1_{}(f32::INFINITY));",
                prefix, suffix, prefix, suffix
            ).unwrap();
            writeln!(
                code,
                "            let is_nan = {}_cmp_{}_mask::<_CMP_UNORD_Q>(self.0, self.0);",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "").unwrap();
            writeln!(code, "            // Apply corrections using mask blend").unwrap();
            writeln!(
                code,
                "            let neg_inf = {}_set1_{}(f32::NEG_INFINITY);",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let pos_inf = {}_set1_{}(f32::INFINITY);",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let nan = {}_set1_{}(f32::NAN);",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "").unwrap();
            writeln!(
                code,
                "            let r = {}_mask_blend_{}(is_zero, result.0, neg_inf);",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let r = {}_mask_blend_{}(is_neg, r, nan);",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let r = {}_mask_blend_{}(is_inf, r, pos_inf);",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let r = {}_mask_blend_{}(is_nan, r, nan);",
                prefix, suffix
            )
            .unwrap();
        } else {
            // SSE/AVX use vector masks
            writeln!(code, "            // Edge case masks").unwrap();
            writeln!(
                code,
                "            let zero = {}_setzero_{}();",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let is_zero = {}_cmp_{}::<_CMP_EQ_OQ>(self.0, zero);",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let is_neg = {}_cmp_{}::<_CMP_LT_OQ>(self.0, zero);",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let is_inf = {}_cmp_{}::<_CMP_EQ_OQ>(self.0, {}_set1_{}(f32::INFINITY));",
                prefix, suffix, prefix, suffix
            ).unwrap();
            writeln!(
                code,
                "            let is_nan = {}_cmp_{}::<_CMP_UNORD_Q>(self.0, self.0);",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "").unwrap();
            writeln!(code, "            // Apply corrections").unwrap();
            writeln!(
                code,
                "            let neg_inf = {}_set1_{}(f32::NEG_INFINITY);",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let pos_inf = {}_set1_{}(f32::INFINITY);",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let nan = {}_set1_{}(f32::NAN);",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "").unwrap();
            writeln!(
                code,
                "            let r = {}_blendv_{}(result.0, neg_inf, is_zero);",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let r = {}_blendv_{}(r, nan, is_neg);",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let r = {}_blendv_{}(r, pos_inf, is_inf);",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let r = {}_blendv_{}(r, nan, is_nan);",
                prefix, suffix
            )
            .unwrap();
        }
        writeln!(code, "            Self(r)").unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 exp2_midp_unchecked =====
        writeln!(
            code,
            "    /// Mid-precision base-2 exponential (~140 ULP, ~8e-6 max relative error) - unchecked variant."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(code, "    /// Uses degree-6 minimax polynomial.").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// **Warning**: Clamps output to finite range. Does not return infinity for overflow."
        )
        .unwrap();
        writeln!(
            code,
            "    /// Use `exp2_midp()` for correct IEEE behavior on edge cases."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn exp2_midp_unchecked(self) -> Self {{").unwrap();
        writeln!(
            code,
            "        // Degree-6 minimax polynomial for 2^x on [0, 1]"
        )
        .unwrap();
        writeln!(code, "        const C0: f32 = 1.0;").unwrap();
        writeln!(code, "        const C1: f32 = 0.693_147_180_559_945;").unwrap();
        writeln!(code, "        const C2: f32 = 0.240_226_506_959_101;").unwrap();
        writeln!(code, "        const C3: f32 = 0.055_504_108_664_822;").unwrap();
        writeln!(code, "        const C4: f32 = 0.009_618_129_107_629;").unwrap();
        writeln!(code, "        const C5: f32 = 0.001_333_355_814_497;").unwrap();
        writeln!(code, "        const C6: f32 = 0.000_154_035_303_933;").unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(code, "            // Clamp to safe range").unwrap();
        writeln!(
            code,
            "            let x = {}_max_{}(self.0, {}_set1_{}(-126.0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let x = {}_min_{}(x, {}_set1_{}(126.0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();

        // Use appropriate floor intrinsic based on width
        if ty.width == SimdWidth::W512 {
            writeln!(
                code,
                "            let xi = {}_roundscale_{}::<0x01>(x); // floor",
                prefix, suffix
            )
            .unwrap();
        } else {
            writeln!(code, "            let xi = {}_floor_{}(x);", prefix, suffix).unwrap();
        }

        writeln!(
            code,
            "            let xf = {}_sub_{}(x, xi);",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // Horner's method with 6 coefficients").unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}({}_set1_{}(C6), xf, {}_set1_{}(C5));",
            prefix, suffix, prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}(poly, xf, {}_set1_{}(C4));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}(poly, xf, {}_set1_{}(C3));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}(poly, xf, {}_set1_{}(C2));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}(poly, xf, {}_set1_{}(C1));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}(poly, xf, {}_set1_{}(C0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // Scale by 2^integer").unwrap();
        writeln!(
            code,
            "            let xi_i32 = {}_cvt{}_epi32(xi);",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "            let bias = {}_set1_epi32(127);", prefix).unwrap();
        writeln!(
            code,
            "            let scale_bits = {}_slli_epi32::<23>({}_add_epi32(xi_i32, bias));",
            prefix, prefix
        )
        .unwrap();
        writeln!(
            code,
            "            let scale = {}_cast{}_{}(scale_bits);",
            prefix, actual_int_suffix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(poly, scale))",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 exp2_midp (checked, default) =====
        writeln!(
            code,
            "    /// Mid-precision base-2 exponential (~140 ULP, ~8e-6 max relative error)."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(code, "    /// Uses degree-6 minimax polynomial.").unwrap();
        writeln!(
            code,
            "    /// Suitable for 8-bit, 10-bit, and 12-bit color processing."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Handles edge cases correctly: exp2(x > 128) = +inf, exp2(x < -150) = 0,"
        )
        .unwrap();
        writeln!(code, "    /// exp2(NaN) = NaN.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn exp2_midp(self) -> Self {{").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(code, "            let result = self.exp2_midp_unchecked();").unwrap();
        writeln!(code, "").unwrap();

        if ty.width == SimdWidth::W512 {
            // AVX-512 uses mask registers
            writeln!(
                code,
                "            // Edge case masks (AVX-512 uses mask registers)"
            )
            .unwrap();
            writeln!(
                code,
                "            let is_overflow = {}_cmp_{}_mask::<_CMP_GE_OQ>(self.0, {}_set1_{}(128.0));",
                prefix, suffix, prefix, suffix
            ).unwrap();
            writeln!(
                code,
                "            let is_underflow = {}_cmp_{}_mask::<_CMP_LT_OQ>(self.0, {}_set1_{}(-150.0));",
                prefix, suffix, prefix, suffix
            ).unwrap();
            writeln!(
                code,
                "            let is_nan = {}_cmp_{}_mask::<_CMP_UNORD_Q>(self.0, self.0);",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "").unwrap();
            writeln!(code, "            // Apply corrections using mask blend").unwrap();
            writeln!(
                code,
                "            let pos_inf = {}_set1_{}(f32::INFINITY);",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let zero = {}_setzero_{}();",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let nan = {}_set1_{}(f32::NAN);",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "").unwrap();
            writeln!(
                code,
                "            let r = {}_mask_blend_{}(is_overflow, result.0, pos_inf);",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let r = {}_mask_blend_{}(is_underflow, r, zero);",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let r = {}_mask_blend_{}(is_nan, r, nan);",
                prefix, suffix
            )
            .unwrap();
        } else {
            // SSE/AVX use vector masks
            writeln!(code, "            // Edge case masks").unwrap();
            writeln!(
                code,
                "            let is_overflow = {}_cmp_{}::<_CMP_GE_OQ>(self.0, {}_set1_{}(128.0));",
                prefix, suffix, prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let is_underflow = {}_cmp_{}::<_CMP_LT_OQ>(self.0, {}_set1_{}(-150.0));",
                prefix, suffix, prefix, suffix
            ).unwrap();
            writeln!(
                code,
                "            let is_nan = {}_cmp_{}::<_CMP_UNORD_Q>(self.0, self.0);",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "").unwrap();
            writeln!(code, "            // Apply corrections").unwrap();
            writeln!(
                code,
                "            let pos_inf = {}_set1_{}(f32::INFINITY);",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let zero = {}_setzero_{}();",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let nan = {}_set1_{}(f32::NAN);",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "").unwrap();
            writeln!(
                code,
                "            let r = {}_blendv_{}(result.0, pos_inf, is_overflow);",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let r = {}_blendv_{}(r, zero, is_underflow);",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let r = {}_blendv_{}(r, nan, is_nan);",
                prefix, suffix
            )
            .unwrap();
        }
        writeln!(code, "            Self(r)").unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 pow_midp_unchecked =====
        writeln!(
            code,
            "    /// Mid-precision power function (self^n) - unchecked variant."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Computed as `exp2_midp_unchecked(n * log2_midp_unchecked(self))`."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN)."
        )
        .unwrap();
        writeln!(
            code,
            "    /// Use `pow_midp()` for correct IEEE behavior on edge cases."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(
            code,
            "    pub fn pow_midp_unchecked(self, n: f32) -> Self {{"
        )
        .unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.log2_midp_unchecked().0, {}_set1_{}(n))).exp2_midp_unchecked()",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
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
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Handles edge cases: pow(0, n) = 0 (n>0), pow(inf, n) = inf (n>0)."
        )
        .unwrap();
        writeln!(code, "    /// Note: Only valid for positive self values.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn pow_midp(self, n: f32) -> Self {{").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.log2_midp().0, {}_set1_{}(n))).exp2_midp()",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 ln_midp_unchecked =====
        writeln!(
            code,
            "    /// Mid-precision natural logarithm - unchecked variant."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Computed as `log2_midp_unchecked(x) * ln(2)`."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN)."
        )
        .unwrap();
        writeln!(
            code,
            "    /// Use `ln_midp()` for correct IEEE behavior on edge cases."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn ln_midp_unchecked(self) -> Self {{").unwrap();
        writeln!(code, "        const LN2: f32 = core::f32::consts::LN_2;").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.log2_midp_unchecked().0, {}_set1_{}(LN2)))",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 ln_midp =====
        writeln!(code, "    /// Mid-precision natural logarithm.").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(code, "    /// Computed as `log2_midp(x) * ln(2)`.").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Handles edge cases: ln(0) = -inf, ln(negative) = NaN, ln(inf) = inf."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn ln_midp(self) -> Self {{").unwrap();
        writeln!(code, "        const LN2: f32 = core::f32::consts::LN_2;").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.log2_midp().0, {}_set1_{}(LN2)))",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 exp_midp_unchecked =====
        writeln!(
            code,
            "    /// Mid-precision natural exponential (e^x) - unchecked variant."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Computed as `exp2_midp_unchecked(x * log2(e))`."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// **Warning**: Clamps output to finite range. Does not return infinity for overflow."
        )
        .unwrap();
        writeln!(
            code,
            "    /// Use `exp_midp()` for correct IEEE behavior on edge cases."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn exp_midp_unchecked(self) -> Self {{").unwrap();
        writeln!(
            code,
            "        const LOG2_E: f32 = core::f32::consts::LOG2_E;"
        )
        .unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.0, {}_set1_{}(LOG2_E))).exp2_midp_unchecked()",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 exp_midp =====
        writeln!(code, "    /// Mid-precision natural exponential (e^x).").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(code, "    /// Computed as `exp2_midp(x * log2(e))`.").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Handles edge cases: exp(x>88) = inf, exp(x<-103) = 0, exp(NaN) = NaN."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn exp_midp(self) -> Self {{").unwrap();
        writeln!(
            code,
            "        const LOG2_E: f32 = core::f32::consts::LOG2_E;"
        )
        .unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.0, {}_set1_{}(LOG2_E))).exp2_midp()",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ========== Cube Root ==========
        writeln!(code, "    // ========== Cube Root ==========\n").unwrap();

        // ===== F32 cbrt_midp_unchecked =====
        writeln!(
            code,
            "    /// Mid-precision cube root (x^(1/3)) - unchecked variant."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Uses scalar extraction for initial guess + Newton-Raphson."
        )
        .unwrap();
        writeln!(
            code,
            "    /// Handles negative values correctly (returns -cbrt(|x|))."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// **Warning**: Does not handle edge cases (0, inf, NaN, denormals)."
        )
        .unwrap();
        writeln!(
            code,
            "    /// Use `cbrt_midp()` for correct IEEE behavior on edge cases."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn cbrt_midp_unchecked(self) -> Self {{").unwrap();
        writeln!(
            code,
            "        // B1 magic constant for cube root initial approximation"
        )
        .unwrap();
        writeln!(
            code,
            "        // B1 = (127 - 127.0/3 - 0.03306235651) * 2^23 = 709958130"
        )
        .unwrap();
        writeln!(code, "        const B1: u32 = 709_958_130;").unwrap();
        writeln!(code, "        const ONE_THIRD: f32 = 1.0 / 3.0;").unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            // Extract to array for initial approximation (scalar division by 3)"
        )
        .unwrap();
        writeln!(
            code,
            "            let x_arr: [f32; {}] = core::mem::transmute(self.0);",
            ty.lanes()
        )
        .unwrap();
        writeln!(
            code,
            "            let mut y_arr = [0.0f32; {}];",
            ty.lanes()
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            for i in 0..{} {{", ty.lanes()).unwrap();
        writeln!(code, "                let xi = x_arr[i];").unwrap();
        writeln!(code, "                let ui = xi.to_bits();").unwrap();
        writeln!(
            code,
            "                let hx = ui & 0x7FFF_FFFF; // abs bits"
        )
        .unwrap();
        writeln!(
            code,
            "                // Initial approximation: bits/3 + B1 (always positive)"
        )
        .unwrap();
        writeln!(code, "                let approx = hx / 3 + B1;").unwrap();
        writeln!(code, "                y_arr[i] = f32::from_bits(approx);").unwrap();
        writeln!(code, "            }}").unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            let abs_x = {}_andnot_{}({}_set1_{}(-0.0), self.0);",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let sign_bits = {}_and_{}(self.0, {}_set1_{}(-0.0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let mut y = core::mem::transmute::<_, _>(y_arr);"
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            // Newton-Raphson: y = y * (2*x + y^3) / (x + 2*y^3)"
        )
        .unwrap();
        writeln!(code, "            // Two iterations for full f32 precision").unwrap();
        writeln!(
            code,
            "            let two = {}_set1_{}(2.0);",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // Iteration 1").unwrap();
        writeln!(
            code,
            "            let y3 = {}_mul_{}({}_mul_{}(y, y), y);",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let num = {}_fmadd_{}(two, abs_x, y3);",
            prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let den = {}_fmadd_{}(two, y3, abs_x);",
            prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            y = {}_mul_{}(y, {}_div_{}(num, den));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // Iteration 2").unwrap();
        writeln!(
            code,
            "            let y3 = {}_mul_{}({}_mul_{}(y, y), y);",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let num = {}_fmadd_{}(two, abs_x, y3);",
            prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let den = {}_fmadd_{}(two, y3, abs_x);",
            prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            y = {}_mul_{}(y, {}_div_{}(num, den));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // Restore sign").unwrap();
        writeln!(
            code,
            "            Self({}_or_{}(y, sign_bits))",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 cbrt_midp (checked, default) =====
        writeln!(code, "    /// Mid-precision cube root (x^(1/3)).").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Uses scalar extraction for initial guess + Newton-Raphson."
        )
        .unwrap();
        writeln!(
            code,
            "    /// Handles negative values correctly (returns -cbrt(|x|))."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Handles edge cases: cbrt(0) = 0, cbrt(±inf) = ±inf, cbrt(NaN) = NaN."
        )
        .unwrap();
        writeln!(
            code,
            "    /// Does not handle denormals (use `cbrt_midp_precise()` for full IEEE compliance)."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn cbrt_midp(self) -> Self {{").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(code, "            let result = self.cbrt_midp_unchecked();").unwrap();
        writeln!(code, "").unwrap();

        if ty.width == SimdWidth::W512 {
            // AVX-512 uses mask registers
            writeln!(
                code,
                "            // Edge case masks (AVX-512 uses mask registers)"
            )
            .unwrap();
            writeln!(
                code,
                "            let zero = {}_setzero_{}();",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let is_zero = {}_cmp_{}_mask::<_CMP_EQ_OQ>(self.0, zero);",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let abs_x = {}_andnot_{}({}_set1_{}(-0.0), self.0);",
                prefix, suffix, prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let is_inf = {}_cmp_{}_mask::<_CMP_EQ_OQ>(abs_x, {}_set1_{}(f32::INFINITY));",
                prefix, suffix, prefix, suffix
            ).unwrap();
            writeln!(
                code,
                "            let is_nan = {}_cmp_{}_mask::<_CMP_UNORD_Q>(self.0, self.0);",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "").unwrap();
            writeln!(code, "            // Apply corrections using mask blend (use self.0 for zero to preserve sign)").unwrap();
            writeln!(
                code,
                "            let r = {}_mask_blend_{}(is_zero, result.0, self.0);  // ±0 -> ±0",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let r = {}_mask_blend_{}(is_inf, r, self.0);  // ±inf -> ±inf",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let r = {}_mask_blend_{}(is_nan, r, {}_set1_{}(f32::NAN));",
                prefix, suffix, prefix, suffix
            )
            .unwrap();
        } else {
            // SSE/AVX use vector masks
            writeln!(code, "            // Edge case masks").unwrap();
            writeln!(
                code,
                "            let zero = {}_setzero_{}();",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let is_zero = {}_cmp_{}::<_CMP_EQ_OQ>(self.0, zero);",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let abs_x = {}_andnot_{}({}_set1_{}(-0.0), self.0);",
                prefix, suffix, prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let is_inf = {}_cmp_{}::<_CMP_EQ_OQ>(abs_x, {}_set1_{}(f32::INFINITY));",
                prefix, suffix, prefix, suffix
            ).unwrap();
            writeln!(
                code,
                "            let is_nan = {}_cmp_{}::<_CMP_UNORD_Q>(self.0, self.0);",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "").unwrap();
            writeln!(
                code,
                "            // Apply corrections (use self.0 for zero to preserve sign)"
            )
            .unwrap();
            writeln!(
                code,
                "            let r = {}_blendv_{}(result.0, self.0, is_zero);  // ±0 -> ±0",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let r = {}_blendv_{}(r, self.0, is_inf);  // ±inf -> ±inf",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let r = {}_blendv_{}(r, {}_set1_{}(f32::NAN), is_nan);",
                prefix, suffix, prefix, suffix
            )
            .unwrap();
        }
        writeln!(code, "            Self(r)").unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 cbrt_midp_precise =====
        writeln!(
            code,
            "    /// Precise cube root (x^(1/3)) with full IEEE compliance."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Uses scalar extraction for initial guess + Newton-Raphson."
        )
        .unwrap();
        writeln!(
            code,
            "    /// Handles negative values correctly (returns -cbrt(|x|))."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Handles all edge cases including denormals. About 67% slower than `cbrt_midp()`."
        )
        .unwrap();
        writeln!(
            code,
            "    /// Use `cbrt_midp()` if denormal support is not needed (most image processing)."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn cbrt_midp_precise(self) -> Self {{").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(code, "            // Scale factor for denormals: 2^24").unwrap();
        writeln!(
            code,
            "            const SCALE_UP: f32 = 16777216.0;  // 2^24"
        )
        .unwrap();
        writeln!(
            code,
            "            const SCALE_DOWN: f32 = 0.00390625;  // 2^(-8) = cbrt(2^(-24))"
        )
        .unwrap();
        writeln!(
            code,
            "            const DENORM_LIMIT: f32 = 1.17549435e-38;  // Smallest normal f32"
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            let abs_x = {}_andnot_{}({}_set1_{}(-0.0), self.0);",
            prefix, suffix, prefix, suffix
        )
        .unwrap();

        if ty.width == SimdWidth::W512 {
            // AVX-512 uses mask registers
            writeln!(
                code,
                "            let is_denorm = {}_cmp_{}_mask::<_CMP_LT_OQ>(abs_x, {}_set1_{}(DENORM_LIMIT));",
                prefix, suffix, prefix, suffix
            ).unwrap();
            writeln!(code, "").unwrap();
            writeln!(code, "            // Scale up denormals").unwrap();
            writeln!(
                code,
                "            let scaled_x = {}_mul_{}(self.0, {}_set1_{}(SCALE_UP));",
                prefix, suffix, prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let x_for_cbrt = {}_mask_blend_{}(is_denorm, self.0, scaled_x);",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "").unwrap();
            writeln!(code, "            // Compute cbrt with edge case handling").unwrap();
            writeln!(
                code,
                "            let result = Self(x_for_cbrt).cbrt_midp();"
            )
            .unwrap();
            writeln!(code, "").unwrap();
            writeln!(
                code,
                "            // Scale down results from denormal inputs"
            )
            .unwrap();
            writeln!(
                code,
                "            let scaled_result = {}_mul_{}(result.0, {}_set1_{}(SCALE_DOWN));",
                prefix, suffix, prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            Self({}_mask_blend_{}(is_denorm, result.0, scaled_result))",
                prefix, suffix
            )
            .unwrap();
        } else {
            // SSE/AVX use vector masks
            writeln!(
                code,
                "            let is_denorm = {}_cmp_{}::<_CMP_LT_OQ>(abs_x, {}_set1_{}(DENORM_LIMIT));",
                prefix, suffix, prefix, suffix
            ).unwrap();
            writeln!(code, "").unwrap();
            writeln!(code, "            // Scale up denormals").unwrap();
            writeln!(
                code,
                "            let scaled_x = {}_mul_{}(self.0, {}_set1_{}(SCALE_UP));",
                prefix, suffix, prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let x_for_cbrt = {}_blendv_{}(self.0, scaled_x, is_denorm);",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "").unwrap();
            writeln!(code, "            // Compute cbrt with edge case handling").unwrap();
            writeln!(
                code,
                "            let result = Self(x_for_cbrt).cbrt_midp();"
            )
            .unwrap();
            writeln!(code, "").unwrap();
            writeln!(
                code,
                "            // Scale down results from denormal inputs"
            )
            .unwrap();
            writeln!(
                code,
                "            let scaled_result = {}_mul_{}(result.0, {}_set1_{}(SCALE_DOWN));",
                prefix, suffix, prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            Self({}_blendv_{}(result.0, scaled_result, is_denorm))",
                prefix, suffix
            )
            .unwrap();
        }
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();
    } else if ty.elem == ElementType::F64 {
        // ===== F64 log2_lowp =====
        // For f64, we use a similar algorithm but with f64 constants
        writeln!(code, "    /// Low-precision base-2 logarithm.").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Uses polynomial approximation. For natural log, use `ln_lowp()`."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn log2_lowp(self) -> Self {{").unwrap();
        writeln!(code, "        // Polynomial coefficients for f64").unwrap();
        writeln!(code, "        const P0: f64 = -1.850_383_340_051_831e-6;").unwrap();
        writeln!(code, "        const P1: f64 = 1.428_716_047_008_376;").unwrap();
        writeln!(code, "        const P2: f64 = 0.742_458_733_278_206;").unwrap();
        writeln!(code, "        const Q0: f64 = 0.990_328_142_775_907;").unwrap();
        writeln!(code, "        const Q1: f64 = 1.009_671_857_224_115;").unwrap();
        writeln!(code, "        const Q2: f64 = 0.174_093_430_036_669;").unwrap();
        writeln!(
            code,
            "        const OFFSET: i64 = 0x3fe6a09e667f3bcd_u64 as i64; // 2/3 in f64 bits"
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            let x_bits = {}_cast{}_{}(self.0);",
            prefix, suffix, actual_int_suffix
        )
        .unwrap();

        // For 64-bit integers, we need different intrinsics
        // For set1 with i64, SSE/AVX use epi64x, AVX-512 uses epi64
        let epi64_suffix = if ty.width == SimdWidth::W512 {
            "epi64"
        } else {
            "epi64x"
        };

        writeln!(
            code,
            "            let offset = {}_set1_{}(OFFSET);",
            prefix, epi64_suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let exp_bits = {}_sub_epi64(x_bits, offset);",
            prefix
        )
        .unwrap();
        // _mm_srai_epi64 / _mm256_srai_epi64 require AVX-512F.
        // For W128/W256, polyfill via scalar extraction.
        if ty.width == SimdWidth::W512 {
            writeln!(
                code,
                "            let exp_shifted = {}_srai_epi64::<52>(exp_bits);",
                prefix
            )
            .unwrap();
        } else {
            let lanes = ty.lanes();
            writeln!(
                code,
                "            let exp_arr_raw: [i64; {lanes}] = core::mem::transmute(exp_bits);"
            )
            .unwrap();
            match ty.width {
                SimdWidth::W128 => {
                    writeln!(
                        code,
                        "            let exp_shifted = _mm_set_epi64x(exp_arr_raw[1] >> 52, exp_arr_raw[0] >> 52);"
                    )
                    .unwrap();
                }
                SimdWidth::W256 => {
                    writeln!(
                        code,
                        "            let exp_shifted = _mm256_set_epi64x(exp_arr_raw[3] >> 52, exp_arr_raw[2] >> 52, exp_arr_raw[1] >> 52, exp_arr_raw[0] >> 52);"
                    )
                    .unwrap();
                }
                _ => unreachable!(),
            }
        }
        writeln!(code, "").unwrap();
        writeln!(code, "            let mantissa_bits = {}_sub_epi64(x_bits, {}_slli_epi64::<52>(exp_shifted));", prefix, prefix).unwrap();
        writeln!(
            code,
            "            let mantissa = {}_cast{}_{}(mantissa_bits);",
            prefix, actual_int_suffix, suffix
        )
        .unwrap();

        // Convert i64 to f64 - extract and convert via scalar
        writeln!(code, "            // Convert exponent to f64").unwrap();
        writeln!(
            code,
            "            let exp_arr: [i64; {}] = core::mem::transmute(exp_shifted);",
            ty.lanes()
        )
        .unwrap();
        writeln!(code, "            let exp_f64: [f64; {}] = [", ty.lanes()).unwrap();
        for i in 0..ty.lanes() {
            if i > 0 {
                write!(code, ", ").unwrap();
            }
            write!(code, "exp_arr[{}] as f64", i).unwrap();
        }
        writeln!(code, "];").unwrap();
        writeln!(
            code,
            "            let exp_val = {}_loadu_{}(exp_f64.as_ptr());",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            let one = {}_set1_{}(1.0);",
            prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let m = {}_sub_{}(mantissa, one);",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // Horner's for numerator").unwrap();
        writeln!(
            code,
            "            let yp = {}_fmadd_{}({}_set1_{}(P2), m, {}_set1_{}(P1));",
            prefix, suffix, prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let yp = {}_fmadd_{}(yp, m, {}_set1_{}(P0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // Horner's for denominator").unwrap();
        writeln!(
            code,
            "            let yq = {}_fmadd_{}({}_set1_{}(Q2), m, {}_set1_{}(Q1));",
            prefix, suffix, prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let yq = {}_fmadd_{}(yq, m, {}_set1_{}(Q0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            Self({}_add_{}({}_div_{}(yp, yq), exp_val))",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F64 exp2_lowp =====
        writeln!(code, "    /// Low-precision base-2 exponential (2^x).").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Uses polynomial approximation. For natural exp, use `exp_lowp()`."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn exp2_lowp(self) -> Self {{").unwrap();
        writeln!(code, "        const C0: f64 = 1.0;").unwrap();
        writeln!(code, "        const C1: f64 = core::f64::consts::LN_2;").unwrap();
        writeln!(code, "        const C2: f64 = 0.240_226_506_959_101;").unwrap();
        writeln!(code, "        const C3: f64 = 0.055_504_108_664_822;").unwrap();
        writeln!(code, "        const C4: f64 = 0.009_618_129_107_629;").unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(code, "            // Clamp to safe range").unwrap();
        writeln!(
            code,
            "            let x = {}_max_{}(self.0, {}_set1_{}(-1022.0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let x = {}_min_{}(x, {}_set1_{}(1022.0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();

        if ty.width == SimdWidth::W512 {
            writeln!(
                code,
                "            let xi = {}_roundscale_{}::<0x01>(x); // floor",
                prefix, suffix
            )
            .unwrap();
        } else {
            writeln!(code, "            let xi = {}_floor_{}(x);", prefix, suffix).unwrap();
        }

        writeln!(
            code,
            "            let xf = {}_sub_{}(x, xi);",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // Polynomial for 2^frac").unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}({}_set1_{}(C4), xf, {}_set1_{}(C3));",
            prefix, suffix, prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}(poly, xf, {}_set1_{}(C2));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}(poly, xf, {}_set1_{}(C1));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}(poly, xf, {}_set1_{}(C0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            // Scale by 2^integer - extract, convert, scale"
        )
        .unwrap();
        writeln!(
            code,
            "            let xi_arr: [f64; {}] = core::mem::transmute(xi);",
            ty.lanes()
        )
        .unwrap();
        writeln!(code, "            let scale_arr: [f64; {}] = [", ty.lanes()).unwrap();
        for i in 0..ty.lanes() {
            if i > 0 {
                write!(code, ", ").unwrap();
            }
            write!(
                code,
                "f64::from_bits(((xi_arr[{}] as i64 + 1023) << 52) as u64)",
                i
            )
            .unwrap();
        }
        writeln!(code, "];").unwrap();
        writeln!(
            code,
            "            let scale = {}_loadu_{}(scale_arr.as_ptr());",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(poly, scale))",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F64 ln_lowp =====
        writeln!(code, "    /// Low-precision natural logarithm.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn ln_lowp(self) -> Self {{").unwrap();
        writeln!(code, "        const LN2: f64 = core::f64::consts::LN_2;").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.log2_lowp().0, {}_set1_{}(LN2)))",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F64 exp_lowp =====
        writeln!(code, "    /// Low-precision natural exponential (e^x).").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn exp_lowp(self) -> Self {{").unwrap();
        writeln!(
            code,
            "        const LOG2_E: f64 = core::f64::consts::LOG2_E;"
        )
        .unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.0, {}_set1_{}(LOG2_E))).exp2_lowp()",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F64 log10_lowp =====
        writeln!(code, "    /// Low-precision base-10 logarithm.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn log10_lowp(self) -> Self {{").unwrap();
        writeln!(
            code,
            "        const LOG10_2: f64 = core::f64::consts::LOG10_2;"
        )
        .unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.log2_lowp().0, {}_set1_{}(LOG10_2)))",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F64 pow_lowp =====
        writeln!(code, "    /// Low-precision power function (self^n).").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn pow_lowp(self, n: f64) -> Self {{").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.log2_lowp().0, {}_set1_{}(n))).exp2_lowp()",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    code
}

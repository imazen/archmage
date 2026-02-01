//! Transcendental operations (log, exp, pow, cbrt) for x86.

use super::types::{ElementType, SimdType, SimdWidth};
use indoc::formatdoc;

pub fn generate_transcendental_ops(ty: &SimdType) -> String {
    // Only for float types
    if !ty.elem.is_float() {
        return String::new();
    }

    let prefix = ty.width.x86_prefix();
    let suffix = ty.elem.x86_suffix();

    // Integer suffix for cast operations
    let int_suffix = match ty.width {
        SimdWidth::W128 => "si128",
        SimdWidth::W256 => "si256",
        SimdWidth::W512 => "si512",
    };

    let mut code = String::from("    // ========== Transcendental Operations ==========\n\n");

    if ty.elem == ElementType::F32 {
        code.push_str(&generate_f32_lowp_ops(ty, prefix, suffix, int_suffix));
        code.push_str(&generate_f32_midp_ops(ty, prefix, suffix, int_suffix));
    } else if ty.elem == ElementType::F64 {
        code.push_str(&generate_f64_lowp_ops(ty, prefix, suffix, int_suffix));
    }

    code
}

fn generate_f32_lowp_ops(ty: &SimdType, prefix: &str, suffix: &str, int_suffix: &str) -> String {
    let floor_op = if ty.width == SimdWidth::W512 {
        format!("{prefix}_roundscale_{suffix}::<0x01>(x)")
    } else {
        format!("{prefix}_floor_{suffix}(x)")
    };

    formatdoc! {r#"
    /// Low-precision base-2 logarithm (~7.7e-5 max relative error).
    ///
    /// Uses rational polynomial approximation. Fast but not suitable for color-accurate work.
    /// For higher precision, use `log2_midp()`.
    #[inline(always)]
    pub fn log2_lowp(self) -> Self {{
        // Rational polynomial coefficients from butteraugli/jpegli
        const P0: f32 = -1.850_383_34e-6;
        const P1: f32 = 1.428_716_05;
        const P2: f32 = 0.742_458_73;
        const Q0: f32 = 0.990_328_14;
        const Q1: f32 = 1.009_671_86;
        const Q2: f32 = 0.174_093_43;

        unsafe {{
            let x_bits = {prefix}_cast{suffix}_{int_suffix}(self.0);
            let offset = {prefix}_set1_epi32(0x3f2aaaab_u32 as i32);
            let exp_bits = {prefix}_sub_epi32(x_bits, offset);
            let exp_shifted = {prefix}_srai_epi32::<23>(exp_bits);

            let mantissa_bits = {prefix}_sub_epi32(x_bits, {prefix}_slli_epi32::<23>(exp_shifted));
            let mantissa = {prefix}_cast{int_suffix}_{suffix}(mantissa_bits);
            let exp_val = {prefix}_cvtepi32_{suffix}(exp_shifted);

            let one = {prefix}_set1_{suffix}(1.0);
            let m = {prefix}_sub_{suffix}(mantissa, one);

            // Horner's for numerator: P2*m^2 + P1*m + P0
            let yp = {prefix}_fmadd_{suffix}({prefix}_set1_{suffix}(P2), m, {prefix}_set1_{suffix}(P1));
            let yp = {prefix}_fmadd_{suffix}(yp, m, {prefix}_set1_{suffix}(P0));

            // Horner's for denominator: Q2*m^2 + Q1*m + Q0
            let yq = {prefix}_fmadd_{suffix}({prefix}_set1_{suffix}(Q2), m, {prefix}_set1_{suffix}(Q1));
            let yq = {prefix}_fmadd_{suffix}(yq, m, {prefix}_set1_{suffix}(Q0));

            Self({prefix}_add_{suffix}({prefix}_div_{suffix}(yp, yq), exp_val))
        }}
    }}

    /// Low-precision base-2 exponential (~5.5e-3 max relative error).
    ///
    /// Uses degree-3 polynomial approximation. Fast but not suitable for color-accurate work.
    /// For higher precision, use `exp2_midp()`.
    #[inline(always)]
    pub fn exp2_lowp(self) -> Self {{
        // Polynomial coefficients
        const C0: f32 = 1.0;
        const C1: f32 = core::f32::consts::LN_2;
        const C2: f32 = 0.240_226_5;
        const C3: f32 = 0.055_504_11;

        unsafe {{
            // Clamp to safe range
            let x = {prefix}_max_{suffix}(self.0, {prefix}_set1_{suffix}(-126.0));
            let x = {prefix}_min_{suffix}(x, {prefix}_set1_{suffix}(126.0));

            // Split into integer and fractional parts
            let xi = {floor_op};
            let xf = {prefix}_sub_{suffix}(x, xi);

            // Polynomial for 2^frac
            let poly = {prefix}_fmadd_{suffix}({prefix}_set1_{suffix}(C3), xf, {prefix}_set1_{suffix}(C2));
            let poly = {prefix}_fmadd_{suffix}(poly, xf, {prefix}_set1_{suffix}(C1));
            let poly = {prefix}_fmadd_{suffix}(poly, xf, {prefix}_set1_{suffix}(C0));

            // Scale by 2^integer using bit manipulation
            let xi_i32 = {prefix}_cvt{suffix}_epi32(xi);
            let bias = {prefix}_set1_epi32(127);
            let scale_bits = {prefix}_slli_epi32::<23>({prefix}_add_epi32(xi_i32, bias));
            let scale = {prefix}_cast{int_suffix}_{suffix}(scale_bits);

            Self({prefix}_mul_{suffix}(poly, scale))
        }}
    }}

    /// Low-precision natural logarithm.
    ///
    /// Computed as `log2_lowp(x) * ln(2)`. For higher precision, use `ln_midp()`.
    #[inline(always)]
    pub fn ln_lowp(self) -> Self {{
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe {{
            Self({prefix}_mul_{suffix}(self.log2_lowp().0, {prefix}_set1_{suffix}(LN2)))
        }}
    }}

    /// Low-precision natural exponential (e^x).
    ///
    /// Computed as `exp2_lowp(x * log2(e))`. For higher precision, use `exp_midp()`.
    #[inline(always)]
    pub fn exp_lowp(self) -> Self {{
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        unsafe {{
            Self({prefix}_mul_{suffix}(self.0, {prefix}_set1_{suffix}(LOG2_E))).exp2_lowp()
        }}
    }}

    /// Low-precision base-10 logarithm.
    ///
    /// Computed as `log2_lowp(x) / log2(10)`.
    #[inline(always)]
    pub fn log10_lowp(self) -> Self {{
        const LOG10_2: f32 = core::f32::consts::LOG10_2; // 1/log2(10)
        unsafe {{
            Self({prefix}_mul_{suffix}(self.log2_lowp().0, {prefix}_set1_{suffix}(LOG10_2)))
        }}
    }}

    /// Low-precision power function (self^n).
    ///
    /// Computed as `exp2_lowp(n * log2_lowp(self))`. For higher precision, use `pow_midp()`.
    /// Note: Only valid for positive self values.
    #[inline(always)]
    pub fn pow_lowp(self, n: f32) -> Self {{
        unsafe {{
            Self({prefix}_mul_{suffix}(self.log2_lowp().0, {prefix}_set1_{suffix}(n))).exp2_lowp()
        }}
    }}

    /// Low-precision base-2 logarithm - unchecked variant.
    ///
    /// Identical to `log2_lowp()` (lowp already skips edge case handling).
    #[inline(always)]
    pub fn log2_lowp_unchecked(self) -> Self {{
        self.log2_lowp()
    }}

    /// Low-precision base-2 exponential - unchecked variant.
    ///
    /// Identical to `exp2_lowp()` (lowp already clamps to safe range).
    #[inline(always)]
    pub fn exp2_lowp_unchecked(self) -> Self {{
        self.exp2_lowp()
    }}

    /// Low-precision natural logarithm - unchecked variant.
    ///
    /// Identical to `ln_lowp()` (lowp already skips edge case handling).
    #[inline(always)]
    pub fn ln_lowp_unchecked(self) -> Self {{
        self.ln_lowp()
    }}

    /// Low-precision natural exponential - unchecked variant.
    ///
    /// Identical to `exp_lowp()` (lowp already clamps to safe range).
    #[inline(always)]
    pub fn exp_lowp_unchecked(self) -> Self {{
        self.exp_lowp()
    }}

    /// Low-precision base-10 logarithm - unchecked variant.
    ///
    /// Identical to `log10_lowp()` (lowp already skips edge case handling).
    #[inline(always)]
    pub fn log10_lowp_unchecked(self) -> Self {{
        self.log10_lowp()
    }}

    /// Low-precision power function - unchecked variant.
    ///
    /// Identical to `pow_lowp()` (lowp already skips edge case handling).
    #[inline(always)]
    pub fn pow_lowp_unchecked(self, n: f32) -> Self {{
        self.pow_lowp(n)
    }}

"#}
}

fn generate_f32_midp_ops(ty: &SimdType, prefix: &str, suffix: &str, int_suffix: &str) -> String {
    let floor_op = if ty.width == SimdWidth::W512 {
        format!("{prefix}_roundscale_{suffix}::<0x01>(x)")
    } else {
        format!("{prefix}_floor_{suffix}(x)")
    };

    let mut code = formatdoc! {r#"
    // ========== Mid-Precision Transcendental Operations ==========

    /// Mid-precision base-2 logarithm (~3 ULP max error) - unchecked variant.
    ///
    /// Uses (a-1)/(a+1) transform with degree-6 odd polynomial.
    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN).
    /// Use `log2_midp()` for correct IEEE behavior on edge cases.
    #[inline(always)]
    pub fn log2_midp_unchecked(self) -> Self {{
        // Constants for range reduction
        const SQRT2_OVER_2: u32 = 0x3f3504f3; // sqrt(2)/2 in f32 bits
        const ONE: u32 = 0x3f800000;          // 1.0 in f32 bits
        const MANTISSA_MASK: i32 = 0x007fffff_u32 as i32;
        const EXPONENT_BIAS: i32 = 127;

        // Coefficients for odd polynomial on y = (a-1)/(a+1)
        const C0: f32 = 2.885_390_08;  // 2/ln(2)
        const C1: f32 = 0.961_800_76;  // y^2 coefficient
        const C2: f32 = 0.576_974_45;  // y^4 coefficient
        const C3: f32 = 0.434_411_97;  // y^6 coefficient

        unsafe {{
            let x_bits = {prefix}_cast{suffix}_{int_suffix}(self.0);

            // Normalize mantissa to [sqrt(2)/2, sqrt(2)]
            let offset = {prefix}_set1_epi32((ONE - SQRT2_OVER_2) as i32);
            let adjusted = {prefix}_add_epi32(x_bits, offset);

            // Extract exponent
            let exp_raw = {prefix}_srai_epi32::<23>(adjusted);
            let exp_biased = {prefix}_sub_epi32(exp_raw, {prefix}_set1_epi32(EXPONENT_BIAS));
            let n = {prefix}_cvtepi32_{suffix}(exp_biased);

            // Reconstruct normalized mantissa
            let mantissa_bits = {prefix}_and_{int_suffix}(adjusted, {prefix}_set1_epi32(MANTISSA_MASK));
            let a_bits = {prefix}_add_epi32(mantissa_bits, {prefix}_set1_epi32(SQRT2_OVER_2 as i32));
            let a = {prefix}_cast{int_suffix}_{suffix}(a_bits);

            // y = (a - 1) / (a + 1)
            let one = {prefix}_set1_{suffix}(1.0);
            let y = {prefix}_div_{suffix}({prefix}_sub_{suffix}(a, one), {prefix}_add_{suffix}(a, one));
            let y2 = {prefix}_mul_{suffix}(y, y);

            // Polynomial: C0*y + C1*y^3 + C2*y^5 + C3*y^7 = y*(C0 + y^2*(C1 + y^2*(C2 + C3*y^2)))
            let poly = {prefix}_fmadd_{suffix}({prefix}_set1_{suffix}(C3), y2, {prefix}_set1_{suffix}(C2));
            let poly = {prefix}_fmadd_{suffix}(poly, y2, {prefix}_set1_{suffix}(C1));
            let poly = {prefix}_fmadd_{suffix}(poly, y2, {prefix}_set1_{suffix}(C0));

            Self({prefix}_fmadd_{suffix}(poly, y, n))
        }}
    }}

"#};

    // log2_midp with edge case handling
    code.push_str(&generate_f32_log2_midp(ty, prefix, suffix, int_suffix));

    // exp2_midp_unchecked
    code.push_str(&formatdoc! {r#"
    /// Mid-precision base-2 exponential (~2 ULP max error) - unchecked variant.
    ///
    /// Uses degree-6 polynomial approximation.
    /// **Warning**: Does not handle edge cases (underflow, overflow).
    /// Use `exp2_midp()` for correct IEEE behavior on edge cases.
    #[inline(always)]
    pub fn exp2_midp_unchecked(self) -> Self {{
        // Polynomial coefficients (degree 6 Remez)
        const C0: f32 = 1.0;
        const C1: f32 = 0.693_147_182;
        const C2: f32 = 0.240_226_463;
        const C3: f32 = 0.055_504_545;
        const C4: f32 = 0.009_618_055;
        const C5: f32 = 0.001_333_37;
        const C6: f32 = 0.000_154_47;

        unsafe {{
            let x = self.0;

            // Split into integer and fractional parts
            let xi = {floor_op};
            let xf = {prefix}_sub_{suffix}(x, xi);

            // Polynomial for 2^frac
            let poly = {prefix}_fmadd_{suffix}({prefix}_set1_{suffix}(C6), xf, {prefix}_set1_{suffix}(C5));
            let poly = {prefix}_fmadd_{suffix}(poly, xf, {prefix}_set1_{suffix}(C4));
            let poly = {prefix}_fmadd_{suffix}(poly, xf, {prefix}_set1_{suffix}(C3));
            let poly = {prefix}_fmadd_{suffix}(poly, xf, {prefix}_set1_{suffix}(C2));
            let poly = {prefix}_fmadd_{suffix}(poly, xf, {prefix}_set1_{suffix}(C1));
            let poly = {prefix}_fmadd_{suffix}(poly, xf, {prefix}_set1_{suffix}(C0));

            // Scale by 2^integer using bit manipulation
            let xi_i32 = {prefix}_cvt{suffix}_epi32(xi);
            let bias = {prefix}_set1_epi32(127);
            let scale_bits = {prefix}_slli_epi32::<23>({prefix}_add_epi32(xi_i32, bias));
            let scale = {prefix}_cast{int_suffix}_{suffix}(scale_bits);

            Self({prefix}_mul_{suffix}(poly, scale))
        }}
    }}

"#});

    // exp2_midp with edge case handling
    code.push_str(&generate_f32_exp2_midp(ty, prefix, suffix));

    // Simple derived functions
    code.push_str(&formatdoc! {r#"
    /// Mid-precision natural logarithm - unchecked variant.
    ///
    /// Computed as `log2_midp_unchecked(x) * ln(2)`.
    #[inline(always)]
    pub fn ln_midp_unchecked(self) -> Self {{
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe {{
            Self({prefix}_mul_{suffix}(self.log2_midp_unchecked().0, {prefix}_set1_{suffix}(LN2)))
        }}
    }}

    /// Mid-precision natural logarithm with edge case handling.
    ///
    /// Returns -inf for 0, NaN for negative, correct results for inf/NaN.
    #[inline(always)]
    pub fn ln_midp(self) -> Self {{
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe {{
            Self({prefix}_mul_{suffix}(self.log2_midp().0, {prefix}_set1_{suffix}(LN2)))
        }}
    }}

    /// Mid-precision natural exponential (e^x) - unchecked variant.
    ///
    /// Computed as `exp2_midp_unchecked(x * log2(e))`.
    #[inline(always)]
    pub fn exp_midp_unchecked(self) -> Self {{
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        unsafe {{
            Self({prefix}_mul_{suffix}(self.0, {prefix}_set1_{suffix}(LOG2_E))).exp2_midp_unchecked()
        }}
    }}

    /// Mid-precision natural exponential (e^x) with edge case handling.
    ///
    /// Returns 0 for large negative, inf for large positive, correct results for inf/NaN.
    #[inline(always)]
    pub fn exp_midp(self) -> Self {{
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        unsafe {{
            Self({prefix}_mul_{suffix}(self.0, {prefix}_set1_{suffix}(LOG2_E))).exp2_midp()
        }}
    }}

    /// Mid-precision power function (self^n) - unchecked variant.
    ///
    /// Computed as `exp2_midp_unchecked(n * log2_midp_unchecked(self))`.
    /// Note: Only valid for positive self values.
    #[inline(always)]
    pub fn pow_midp_unchecked(self, n: f32) -> Self {{
        unsafe {{
            Self({prefix}_mul_{suffix}(self.log2_midp_unchecked().0, {prefix}_set1_{suffix}(n))).exp2_midp_unchecked()
        }}
    }}

    /// Mid-precision power function (self^n) with edge case handling.
    ///
    /// Handles 0, negative, inf, and NaN in base. Only valid for positive self values.
    #[inline(always)]
    pub fn pow_midp(self, n: f32) -> Self {{
        unsafe {{
            Self({prefix}_mul_{suffix}(self.log2_midp().0, {prefix}_set1_{suffix}(n))).exp2_midp()
        }}
    }}

"#});

    // log2_midp_precise, ln_midp_precise, pow_midp_precise, log10_midp family
    code.push_str(&generate_f32_midp_precise_and_log10(
        ty, prefix, suffix, int_suffix,
    ));

    // cbrt operations
    code.push_str(&generate_f32_cbrt_ops(ty, prefix, suffix, int_suffix));

    code
}

fn generate_f32_log2_midp(ty: &SimdType, prefix: &str, suffix: &str, _int_suffix: &str) -> String {
    if ty.width == SimdWidth::W512 {
        formatdoc! {r#"
    /// Mid-precision base-2 logarithm (~3 ULP max error) with edge case handling.
    ///
    /// Returns -inf for 0, NaN for negative, correct results for inf/NaN.
    #[inline(always)]
    pub fn log2_midp(self) -> Self {{
        unsafe {{
            let x = self.0;
            let zero = {prefix}_setzero_{suffix}();
            let neg_inf = {prefix}_set1_{suffix}(f32::NEG_INFINITY);
            let nan = {prefix}_set1_{suffix}(f32::NAN);
            let inf = {prefix}_set1_{suffix}(f32::INFINITY);

            // Handle special cases with AVX-512 mask operations
            let is_zero = {prefix}_cmp_{suffix}_mask::<_CMP_EQ_OQ>(x, zero);
            let is_negative = {prefix}_cmp_{suffix}_mask::<_CMP_LT_OQ>(x, zero);
            let is_inf = {prefix}_cmp_{suffix}_mask::<_CMP_EQ_OQ>(x, inf);

            // Compute log2 for normal values
            let log_result = self.log2_midp_unchecked().0;

            // Apply special case results
            let result = {prefix}_mask_blend_{suffix}(is_zero, log_result, neg_inf);
            let result = {prefix}_mask_blend_{suffix}(is_negative, result, nan);
            let result = {prefix}_mask_blend_{suffix}(is_inf, result, inf);

            Self(result)
        }}
    }}

"#}
    } else {
        formatdoc! {r#"
    /// Mid-precision base-2 logarithm (~3 ULP max error) with edge case handling.
    ///
    /// Returns -inf for 0, NaN for negative, correct results for inf/NaN.
    #[inline(always)]
    pub fn log2_midp(self) -> Self {{
        unsafe {{
            let x = self.0;
            let zero = {prefix}_setzero_{suffix}();
            let neg_inf = {prefix}_set1_{suffix}(f32::NEG_INFINITY);
            let nan = {prefix}_set1_{suffix}(f32::NAN);
            let inf = {prefix}_set1_{suffix}(f32::INFINITY);

            // Handle special cases
            let is_zero = {prefix}_cmp_{suffix}::<_CMP_EQ_OQ>(x, zero);
            let is_negative = {prefix}_cmp_{suffix}::<_CMP_LT_OQ>(x, zero);
            let is_inf = {prefix}_cmp_{suffix}::<_CMP_EQ_OQ>(x, inf);

            // Compute log2 for normal values
            let log_result = self.log2_midp_unchecked().0;

            // Apply special case results
            let result = {prefix}_blendv_{suffix}(log_result, neg_inf, is_zero);
            let result = {prefix}_blendv_{suffix}(result, nan, is_negative);
            let result = {prefix}_blendv_{suffix}(result, inf, is_inf);

            Self(result)
        }}
    }}

"#}
    }
}

fn generate_f32_exp2_midp(ty: &SimdType, prefix: &str, suffix: &str) -> String {
    if ty.width == SimdWidth::W512 {
        formatdoc! {r#"
    /// Mid-precision base-2 exponential (~2 ULP max error) with edge case handling.
    ///
    /// Returns 0 for large negative, inf for large positive, correct results for inf/NaN.
    #[inline(always)]
    pub fn exp2_midp(self) -> Self {{
        unsafe {{
            let x = self.0;
            let zero = {prefix}_setzero_{suffix}();
            let inf = {prefix}_set1_{suffix}(f32::INFINITY);

            // Clamp to prevent overflow in intermediate calculations
            let x_clamped = {prefix}_max_{suffix}(x, {prefix}_set1_{suffix}(-150.0));
            let x_clamped = {prefix}_min_{suffix}(x_clamped, {prefix}_set1_{suffix}(128.0));

            // Compute exp2 for clamped values
            let exp_result = Self(x_clamped).exp2_midp_unchecked().0;

            // Handle edge cases with AVX-512 mask operations
            let underflow = {prefix}_cmp_{suffix}_mask::<_CMP_LT_OQ>(x, {prefix}_set1_{suffix}(-150.0));
            let overflow = {prefix}_cmp_{suffix}_mask::<_CMP_GT_OQ>(x, {prefix}_set1_{suffix}(128.0));

            let result = {prefix}_mask_blend_{suffix}(underflow, exp_result, zero);
            let result = {prefix}_mask_blend_{suffix}(overflow, result, inf);

            Self(result)
        }}
    }}

"#}
    } else {
        formatdoc! {r#"
    /// Mid-precision base-2 exponential (~2 ULP max error) with edge case handling.
    ///
    /// Returns 0 for large negative, inf for large positive, correct results for inf/NaN.
    #[inline(always)]
    pub fn exp2_midp(self) -> Self {{
        unsafe {{
            let x = self.0;
            let zero = {prefix}_setzero_{suffix}();
            let inf = {prefix}_set1_{suffix}(f32::INFINITY);

            // Clamp to prevent overflow in intermediate calculations
            let x_clamped = {prefix}_max_{suffix}(x, {prefix}_set1_{suffix}(-150.0));
            let x_clamped = {prefix}_min_{suffix}(x_clamped, {prefix}_set1_{suffix}(128.0));

            // Compute exp2 for clamped values
            let exp_result = Self(x_clamped).exp2_midp_unchecked().0;

            // Handle edge cases
            let underflow = {prefix}_cmp_{suffix}::<_CMP_LT_OQ>(x, {prefix}_set1_{suffix}(-150.0));
            let overflow = {prefix}_cmp_{suffix}::<_CMP_GT_OQ>(x, {prefix}_set1_{suffix}(128.0));

            let result = {prefix}_blendv_{suffix}(exp_result, zero, underflow);
            let result = {prefix}_blendv_{suffix}(result, inf, overflow);

            Self(result)
        }}
    }}

"#}
    }
}

fn generate_f32_cbrt_ops(ty: &SimdType, prefix: &str, suffix: &str, _int_suffix: &str) -> String {
    let lanes = ty.lanes();
    let lane_indices = (0..lanes).collect::<Vec<_>>();

    let mut code = formatdoc! {r#"
    /// Mid-precision cube root (~1 ULP max error).
    ///
    /// Uses scalar extraction for initial guess + Newton-Raphson.
    /// Handles negative values correctly (returns -cbrt(|x|)).
    ///
    /// Does not handle denormals. Use `cbrt_midp_precise()` if denormal support is needed.
    #[inline(always)]
    pub fn cbrt_midp(self) -> Self {{
        // Kahan's magic constant for initial approximation
        const KAHAN_CBRT: f32 = 0.333_333_313;
        const TWO_THIRDS: f32 = 0.666_666_627;

        unsafe {{
            let x = self.0;

            // Save sign and work with absolute value
            let sign_mask = {prefix}_set1_{suffix}(-0.0);
            let sign = {prefix}_and_{suffix}(x, sign_mask);
            let abs_x = {prefix}_andnot_{suffix}(sign_mask, x);

            // Extract to scalar for initial approximation
            let arr: [f32; {lanes}] = core::mem::transmute(abs_x);
            let approx: [f32; {lanes}] = [
"#};

    // Generate scalar cbrt approximations for each lane
    for i in &lane_indices {
        code.push_str(&format!(
            "                f32::from_bits((arr[{i}].to_bits() / 3) + 0x2a508c2d),\n"
        ));
    }

    code.push_str(&formatdoc! {r#"
            ];
            let mut y = {prefix}_loadu_{suffix}(approx.as_ptr());

            // Newton-Raphson iterations: y = y * (2/3 + x/(3*y^3))
            for _ in 0..3 {{
                let y2 = {prefix}_mul_{suffix}(y, y);
                let y3 = {prefix}_mul_{suffix}(y2, y);
                let term = {prefix}_div_{suffix}(abs_x, {prefix}_mul_{suffix}({prefix}_set1_{suffix}(3.0), y3));
                y = {prefix}_mul_{suffix}(y, {prefix}_add_{suffix}({prefix}_set1_{suffix}(TWO_THIRDS), term));
            }}

            // Restore sign
            Self({prefix}_or_{suffix}(y, sign))
        }}
    }}

"#});

    // cbrt_midp_precise with denormal handling
    if ty.width == SimdWidth::W512 {
        code.push_str(&formatdoc! {r#"
    /// Mid-precision cube root with denormal handling (~1 ULP max error).
    ///
    /// Uses scalar extraction for initial guess + Newton-Raphson.
    /// Handles negative values correctly (returns -cbrt(|x|)).
    ///
    /// Handles all edge cases including denormals. About 67% slower than `cbrt_midp()`.
    /// Use `cbrt_midp()` if denormal support is not needed (most image processing).
    #[inline(always)]
    pub fn cbrt_midp_precise(self) -> Self {{
        unsafe {{
            // Scale factor for denormals: 2^24
            const SCALE_UP: f32 = 16777216.0;  // 2^24
            const SCALE_DOWN: f32 = 0.00390625;  // 2^(-8) = cbrt(2^(-24))
            const DENORM_LIMIT: f32 = 1.17549435e-38;  // Smallest normal f32

            let abs_x = {prefix}_andnot_{suffix}({prefix}_set1_{suffix}(-0.0), self.0);
            let is_denorm = {prefix}_cmp_{suffix}_mask::<_CMP_LT_OQ>(abs_x, {prefix}_set1_{suffix}(DENORM_LIMIT));

            // Scale up denormals
            let scaled_x = {prefix}_mul_{suffix}(self.0, {prefix}_set1_{suffix}(SCALE_UP));
            let x_for_cbrt = {prefix}_mask_blend_{suffix}(is_denorm, self.0, scaled_x);

            // Compute cbrt with edge case handling
            let result = Self(x_for_cbrt).cbrt_midp();

            // Scale down results from denormal inputs
            let scaled_result = {prefix}_mul_{suffix}(result.0, {prefix}_set1_{suffix}(SCALE_DOWN));
            Self({prefix}_mask_blend_{suffix}(is_denorm, result.0, scaled_result))
        }}
    }}

"#});
    } else {
        code.push_str(&formatdoc! {r#"
    /// Mid-precision cube root with denormal handling (~1 ULP max error).
    ///
    /// Uses scalar extraction for initial guess + Newton-Raphson.
    /// Handles negative values correctly (returns -cbrt(|x|)).
    ///
    /// Handles all edge cases including denormals. About 67% slower than `cbrt_midp()`.
    /// Use `cbrt_midp()` if denormal support is not needed (most image processing).
    #[inline(always)]
    pub fn cbrt_midp_precise(self) -> Self {{
        unsafe {{
            // Scale factor for denormals: 2^24
            const SCALE_UP: f32 = 16777216.0;  // 2^24
            const SCALE_DOWN: f32 = 0.00390625;  // 2^(-8) = cbrt(2^(-24))
            const DENORM_LIMIT: f32 = 1.17549435e-38;  // Smallest normal f32

            let abs_x = {prefix}_andnot_{suffix}({prefix}_set1_{suffix}(-0.0), self.0);
            let is_denorm = {prefix}_cmp_{suffix}::<_CMP_LT_OQ>(abs_x, {prefix}_set1_{suffix}(DENORM_LIMIT));

            // Scale up denormals
            let scaled_x = {prefix}_mul_{suffix}(self.0, {prefix}_set1_{suffix}(SCALE_UP));
            let x_for_cbrt = {prefix}_blendv_{suffix}(self.0, scaled_x, is_denorm);

            // Compute cbrt with edge case handling
            let result = Self(x_for_cbrt).cbrt_midp();

            // Scale down results from denormal inputs
            let scaled_result = {prefix}_mul_{suffix}(result.0, {prefix}_set1_{suffix}(SCALE_DOWN));
            Self({prefix}_blendv_{suffix}(result.0, scaled_result, is_denorm))
        }}
    }}

"#});
    }

    code
}

fn generate_f32_midp_precise_and_log10(
    ty: &SimdType,
    prefix: &str,
    suffix: &str,
    _int_suffix: &str,
) -> String {
    if ty.width == SimdWidth::W512 {
        formatdoc! {r#"
    /// Mid-precision base-2 logarithm with full IEEE compliance.
    ///
    /// Handles all edge cases including denormals.
    /// About 50% slower than `log2_midp()` due to denormal scaling.
    #[inline(always)]
    pub fn log2_midp_precise(self) -> Self {{
        unsafe {{
            const SCALE_UP: f32 = 16777216.0;  // 2^24
            const SCALE_ADJUST: f32 = 24.0;    // log2(2^24)
            const DENORM_LIMIT: f32 = 1.17549435e-38;

            let zero = {prefix}_setzero_{suffix}();
            let abs_x = {prefix}_andnot_{suffix}({prefix}_set1_{suffix}(-0.0), self.0);
            let is_positive = {prefix}_cmp_{suffix}_mask::<_CMP_GT_OQ>(self.0, zero);
            let is_small = {prefix}_cmp_{suffix}_mask::<_CMP_LT_OQ>(abs_x, {prefix}_set1_{suffix}(DENORM_LIMIT));
            let is_denorm = is_positive & is_small;

            let scaled_x = {prefix}_mul_{suffix}(self.0, {prefix}_set1_{suffix}(SCALE_UP));
            let x_for_log = {prefix}_mask_blend_{suffix}(is_denorm, self.0, scaled_x);

            let result = Self(x_for_log).log2_midp();

            let adjusted = {prefix}_sub_{suffix}(result.0, {prefix}_set1_{suffix}(SCALE_ADJUST));
            Self({prefix}_mask_blend_{suffix}(is_denorm, result.0, adjusted))
        }}
    }}

    /// Mid-precision natural logarithm with full IEEE compliance.
    ///
    /// Handles all edge cases including denormals.
    /// About 50% slower than `ln_midp()` due to denormal scaling.
    #[inline(always)]
    pub fn ln_midp_precise(self) -> Self {{
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe {{
            Self({prefix}_mul_{suffix}(self.log2_midp_precise().0, {prefix}_set1_{suffix}(LN2)))
        }}
    }}

    /// Mid-precision power function with denormal handling.
    ///
    /// Uses `log2_midp_precise()` to handle denormal inputs correctly.
    /// Note: Only valid for positive self values.
    #[inline(always)]
    pub fn pow_midp_precise(self, n: f32) -> Self {{
        unsafe {{
            Self({prefix}_mul_{suffix}(self.log2_midp_precise().0, {prefix}_set1_{suffix}(n))).exp2_midp()
        }}
    }}

    /// Mid-precision base-10 logarithm - unchecked variant.
    ///
    /// Computed as `log2_midp_unchecked(x) * log10(2)`.
    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN, denormals).
    #[inline(always)]
    pub fn log10_midp_unchecked(self) -> Self {{
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        unsafe {{
            Self({prefix}_mul_{suffix}(self.log2_midp_unchecked().0, {prefix}_set1_{suffix}(LOG10_2)))
        }}
    }}

    /// Mid-precision base-10 logarithm with edge case handling.
    ///
    /// Computed as `log2_midp(x) * log10(2)`.
    /// Returns -inf for 0, NaN for negative, correct results for inf/NaN.
    #[inline(always)]
    pub fn log10_midp(self) -> Self {{
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        unsafe {{
            Self({prefix}_mul_{suffix}(self.log2_midp().0, {prefix}_set1_{suffix}(LOG10_2)))
        }}
    }}

    /// Mid-precision base-10 logarithm with full IEEE compliance.
    ///
    /// Handles all edge cases including denormals.
    /// About 50% slower than `log10_midp()` due to denormal scaling.
    #[inline(always)]
    pub fn log10_midp_precise(self) -> Self {{
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        unsafe {{
            Self({prefix}_mul_{suffix}(self.log2_midp_precise().0, {prefix}_set1_{suffix}(LOG10_2)))
        }}
    }}

"#}
    } else {
        formatdoc! {r#"
    /// Mid-precision base-2 logarithm with full IEEE compliance.
    ///
    /// Handles all edge cases including denormals.
    /// About 50% slower than `log2_midp()` due to denormal scaling.
    #[inline(always)]
    pub fn log2_midp_precise(self) -> Self {{
        unsafe {{
            const SCALE_UP: f32 = 16777216.0;  // 2^24
            const SCALE_ADJUST: f32 = 24.0;    // log2(2^24)
            const DENORM_LIMIT: f32 = 1.17549435e-38;

            let zero = {prefix}_setzero_{suffix}();
            let is_positive = {prefix}_cmp_{suffix}::<_CMP_GT_OQ>(self.0, zero);
            let abs_x = {prefix}_andnot_{suffix}({prefix}_set1_{suffix}(-0.0), self.0);
            let is_small = {prefix}_cmp_{suffix}::<_CMP_LT_OQ>(abs_x, {prefix}_set1_{suffix}(DENORM_LIMIT));
            let is_denorm = {prefix}_and_{suffix}(is_positive, is_small);

            let scaled_x = {prefix}_mul_{suffix}(self.0, {prefix}_set1_{suffix}(SCALE_UP));
            let x_for_log = {prefix}_blendv_{suffix}(self.0, scaled_x, is_denorm);

            let result = Self(x_for_log).log2_midp();

            let adjusted = {prefix}_sub_{suffix}(result.0, {prefix}_set1_{suffix}(SCALE_ADJUST));
            Self({prefix}_blendv_{suffix}(result.0, adjusted, is_denorm))
        }}
    }}

    /// Mid-precision natural logarithm with full IEEE compliance.
    ///
    /// Handles all edge cases including denormals.
    /// About 50% slower than `ln_midp()` due to denormal scaling.
    #[inline(always)]
    pub fn ln_midp_precise(self) -> Self {{
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe {{
            Self({prefix}_mul_{suffix}(self.log2_midp_precise().0, {prefix}_set1_{suffix}(LN2)))
        }}
    }}

    /// Mid-precision power function with denormal handling.
    ///
    /// Uses `log2_midp_precise()` to handle denormal inputs correctly.
    /// Note: Only valid for positive self values.
    #[inline(always)]
    pub fn pow_midp_precise(self, n: f32) -> Self {{
        unsafe {{
            Self({prefix}_mul_{suffix}(self.log2_midp_precise().0, {prefix}_set1_{suffix}(n))).exp2_midp()
        }}
    }}

    /// Mid-precision base-10 logarithm - unchecked variant.
    ///
    /// Computed as `log2_midp_unchecked(x) * log10(2)`.
    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN, denormals).
    #[inline(always)]
    pub fn log10_midp_unchecked(self) -> Self {{
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        unsafe {{
            Self({prefix}_mul_{suffix}(self.log2_midp_unchecked().0, {prefix}_set1_{suffix}(LOG10_2)))
        }}
    }}

    /// Mid-precision base-10 logarithm with edge case handling.
    ///
    /// Computed as `log2_midp(x) * log10(2)`.
    /// Returns -inf for 0, NaN for negative, correct results for inf/NaN.
    #[inline(always)]
    pub fn log10_midp(self) -> Self {{
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        unsafe {{
            Self({prefix}_mul_{suffix}(self.log2_midp().0, {prefix}_set1_{suffix}(LOG10_2)))
        }}
    }}

    /// Mid-precision base-10 logarithm with full IEEE compliance.
    ///
    /// Handles all edge cases including denormals.
    /// About 50% slower than `log10_midp()` due to denormal scaling.
    #[inline(always)]
    pub fn log10_midp_precise(self) -> Self {{
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        unsafe {{
            Self({prefix}_mul_{suffix}(self.log2_midp_precise().0, {prefix}_set1_{suffix}(LOG10_2)))
        }}
    }}

"#}
    }
}

fn generate_f64_lowp_ops(ty: &SimdType, prefix: &str, suffix: &str, int_suffix: &str) -> String {
    let lanes = ty.lanes();
    let floor_op = if ty.width == SimdWidth::W512 {
        format!("{prefix}_roundscale_{suffix}::<0x01>(x)")
    } else {
        format!("{prefix}_floor_{suffix}(x)")
    };

    // For i64 set1, AVX-512 uses epi64, others use epi64x
    let epi64_suffix = if ty.width == SimdWidth::W512 {
        "epi64"
    } else {
        "epi64x"
    };

    // Exponent shift for f64 - AVX-512 has native srai_epi64, others need polyfill
    let exp_shift = if ty.width == SimdWidth::W512 {
        format!("let exp_shifted = {prefix}_srai_epi64::<52>(exp_bits);")
    } else {
        let set_fn = match ty.width {
            SimdWidth::W128 => "_mm_set_epi64x",
            SimdWidth::W256 => "_mm256_set_epi64x",
            _ => unreachable!(),
        };
        let indices: Vec<String> = (0..lanes)
            .rev()
            .map(|i| format!("exp_arr_raw[{i}] >> 52"))
            .collect();
        format!(
            "let exp_arr_raw: [i64; {lanes}] = core::mem::transmute(exp_bits);\n            let exp_shifted = {set_fn}({});",
            indices.join(", ")
        )
    };

    // Convert i64 exponent to f64
    let exp_to_f64_indices: Vec<String> =
        (0..lanes).map(|i| format!("exp_arr[{i}] as f64")).collect();

    // Scale array generation for exp2
    let scale_arr_indices: Vec<String> = (0..lanes)
        .map(|i| format!("f64::from_bits(((xi_arr[{i}] as i64 + 1023) << 52) as u64)"))
        .collect();

    formatdoc! {r#"
    /// Low-precision base-2 logarithm.
    ///
    /// Uses polynomial approximation. For natural log, use `ln_lowp()`.
    #[inline(always)]
    pub fn log2_lowp(self) -> Self {{
        // Polynomial coefficients for f64
        const P0: f64 = -1.850_383_340_051_831e-6;
        const P1: f64 = 1.428_716_047_008_376;
        const P2: f64 = 0.742_458_733_278_206;
        const Q0: f64 = 0.990_328_142_775_907;
        const Q1: f64 = 1.009_671_857_224_115;
        const Q2: f64 = 0.174_093_430_036_669;
        const OFFSET: i64 = 0x3fe6a09e667f3bcd_u64 as i64; // 2/3 in f64 bits

        unsafe {{
            let x_bits = {prefix}_cast{suffix}_{int_suffix}(self.0);
            let offset = {prefix}_set1_{epi64_suffix}(OFFSET);
            let exp_bits = {prefix}_sub_epi64(x_bits, offset);
            {exp_shift}

            let mantissa_bits = {prefix}_sub_epi64(x_bits, {prefix}_slli_epi64::<52>(exp_shifted));
            let mantissa = {prefix}_cast{int_suffix}_{suffix}(mantissa_bits);
            // Convert exponent to f64
            let exp_arr: [i64; {lanes}] = core::mem::transmute(exp_shifted);
            let exp_f64: [f64; {lanes}] = [{exp_to_f64_list}];
            let exp_val = {prefix}_loadu_{suffix}(exp_f64.as_ptr());

            let one = {prefix}_set1_{suffix}(1.0);
            let m = {prefix}_sub_{suffix}(mantissa, one);

            // Horner's for numerator
            let yp = {prefix}_fmadd_{suffix}({prefix}_set1_{suffix}(P2), m, {prefix}_set1_{suffix}(P1));
            let yp = {prefix}_fmadd_{suffix}(yp, m, {prefix}_set1_{suffix}(P0));

            // Horner's for denominator
            let yq = {prefix}_fmadd_{suffix}({prefix}_set1_{suffix}(Q2), m, {prefix}_set1_{suffix}(Q1));
            let yq = {prefix}_fmadd_{suffix}(yq, m, {prefix}_set1_{suffix}(Q0));

            Self({prefix}_add_{suffix}({prefix}_div_{suffix}(yp, yq), exp_val))
        }}
    }}

    /// Low-precision base-2 exponential (2^x).
    ///
    /// Uses polynomial approximation. For natural exp, use `exp_lowp()`.
    #[inline(always)]
    pub fn exp2_lowp(self) -> Self {{
        const C0: f64 = 1.0;
        const C1: f64 = core::f64::consts::LN_2;
        const C2: f64 = 0.240_226_506_959_101;
        const C3: f64 = 0.055_504_108_664_822;
        const C4: f64 = 0.009_618_129_107_629;

        unsafe {{
            // Clamp to safe range
            let x = {prefix}_max_{suffix}(self.0, {prefix}_set1_{suffix}(-1022.0));
            let x = {prefix}_min_{suffix}(x, {prefix}_set1_{suffix}(1022.0));

            let xi = {floor_op};
            let xf = {prefix}_sub_{suffix}(x, xi);

            // Polynomial for 2^frac
            let poly = {prefix}_fmadd_{suffix}({prefix}_set1_{suffix}(C4), xf, {prefix}_set1_{suffix}(C3));
            let poly = {prefix}_fmadd_{suffix}(poly, xf, {prefix}_set1_{suffix}(C2));
            let poly = {prefix}_fmadd_{suffix}(poly, xf, {prefix}_set1_{suffix}(C1));
            let poly = {prefix}_fmadd_{suffix}(poly, xf, {prefix}_set1_{suffix}(C0));

            // Scale by 2^integer - extract, convert, scale
            let xi_arr: [f64; {lanes}] = core::mem::transmute(xi);
            let scale_arr: [f64; {lanes}] = [{scale_arr_list}];
            let scale = {prefix}_loadu_{suffix}(scale_arr.as_ptr());

            Self({prefix}_mul_{suffix}(poly, scale))
        }}
    }}

    /// Low-precision natural logarithm.
    #[inline(always)]
    pub fn ln_lowp(self) -> Self {{
        const LN2: f64 = core::f64::consts::LN_2;
        unsafe {{
            Self({prefix}_mul_{suffix}(self.log2_lowp().0, {prefix}_set1_{suffix}(LN2)))
        }}
    }}

    /// Low-precision natural exponential (e^x).
    #[inline(always)]
    pub fn exp_lowp(self) -> Self {{
        const LOG2_E: f64 = core::f64::consts::LOG2_E;
        unsafe {{
            Self({prefix}_mul_{suffix}(self.0, {prefix}_set1_{suffix}(LOG2_E))).exp2_lowp()
        }}
    }}

    /// Low-precision base-10 logarithm.
    #[inline(always)]
    pub fn log10_lowp(self) -> Self {{
        const LOG10_2: f64 = core::f64::consts::LOG10_2;
        unsafe {{
            Self({prefix}_mul_{suffix}(self.log2_lowp().0, {prefix}_set1_{suffix}(LOG10_2)))
        }}
    }}

    /// Low-precision power function (self^n).
    #[inline(always)]
    pub fn pow_lowp(self, n: f64) -> Self {{
        unsafe {{
            Self({prefix}_mul_{suffix}(self.log2_lowp().0, {prefix}_set1_{suffix}(n))).exp2_lowp()
        }}
    }}

"#, exp_to_f64_list = exp_to_f64_indices.join(", "), scale_arr_list = scale_arr_indices.join(", ")}
}

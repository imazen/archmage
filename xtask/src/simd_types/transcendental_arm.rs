//! ARM NEON transcendental operations (log, exp, pow, cbrt) for f32x4 and f64x2.
//!
//! Uses same polynomial coefficients as x86/WASM — only intrinsic names differ.
//! NEON has native FMA (vfmaq_f32) and floor (vrndmq_f32) which map cleanly.
//!
//! **FMA arg order**: NEON `vfmaq_f32(a,b,c)` = a + b*c (accumulator first).
//! x86 `_mm_fmadd_ps(a,b,c)` = a*b + c (multiply operands first).

use super::types::{ElementType, SimdType, SimdWidth};
use indoc::formatdoc;

/// Generate ARM NEON transcendental operations for float types
pub fn generate_arm_transcendental_ops(ty: &SimdType) -> String {
    // Only for float types
    if !ty.elem.is_float() {
        return String::new();
    }

    assert!(
        ty.width == SimdWidth::W128,
        "NEON only supports 128-bit vectors"
    );

    let mut code = formatdoc! {"
        // ========== Transcendental Operations (Polynomial Approximations) ==========

    "};

    if ty.elem == ElementType::F32 {
        code.push_str(&generate_f32_lowp_ops());
        code.push_str(&generate_f32_cbrt_ops());
    } else if ty.elem == ElementType::F64 {
        code.push_str(&generate_f64_lowp_ops());
    }

    code
}

fn generate_f32_lowp_ops() -> String {
    formatdoc! {r#"
    /// Low-precision base-2 logarithm - unchecked variant (~7.7e-5 max relative error).
    ///
    /// Uses rational polynomial approximation.
    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN).
    /// Use `log2_lowp()` for correct IEEE behavior.
    #[inline(always)]
    pub fn log2_lowp_unchecked(self) -> Self {{
        const P0: f32 = -1.850_383_34e-6;
        const P1: f32 = 1.428_716_05;
        const P2: f32 = 0.742_458_73;
        const Q0: f32 = 0.990_328_14;
        const Q1: f32 = 1.009_671_86;
        const Q2: f32 = 0.174_093_43;
        const OFFSET: u32 = 0x3f2aaaab;

        unsafe {{
            let x = self.0;

            // Extract exponent via bit manipulation
            let x_bits = vreinterpretq_u32_f32(x);
            let offset = vdupq_n_u32(OFFSET);
            let exp_bits = vsubq_u32(x_bits, offset);
            let exp_shifted = vreinterpretq_s32_u32(vshrq_n_u32::<23>(exp_bits));

            // Reconstruct mantissa
            let shift_back = vreinterpretq_u32_s32(vshlq_n_s32::<23>(exp_shifted));
            let mantissa_bits = vsubq_u32(x_bits, shift_back);
            let mantissa = vreinterpretq_f32_u32(mantissa_bits);
            let exp_val = vcvtq_f32_s32(exp_shifted);

            let one = vdupq_n_f32(1.0);
            let m = vsubq_f32(mantissa, one);

            // Horner's for numerator: P2*m^2 + P1*m + P0
            // vfmaq_f32(a, b, c) = a + b*c
            let yp = vfmaq_f32(vdupq_n_f32(P1), vdupq_n_f32(P2), m);
            let yp = vfmaq_f32(vdupq_n_f32(P0), yp, m);

            // Horner's for denominator: Q2*m^2 + Q1*m + Q0
            let yq = vfmaq_f32(vdupq_n_f32(Q1), vdupq_n_f32(Q2), m);
            let yq = vfmaq_f32(vdupq_n_f32(Q0), yq, m);

            Self(vaddq_f32(vdivq_f32(yp, yq), exp_val))
        }}
    }}

    /// Low-precision base-2 logarithm (~7.7e-5 max relative error).
    ///
    /// Handles edge cases: log2(0) = -inf, log2(negative) = NaN,
    /// log2(+inf) = +inf, log2(NaN) = NaN.
    #[inline(always)]
    pub fn log2_lowp(self) -> Self {{
        let result = self.log2_lowp_unchecked();

        unsafe {{
            let zero = vdupq_n_f32(0.0);
            let is_zero = vceqq_f32(self.0, zero);
            let is_neg = vcltq_f32(self.0, zero);
            let is_inf = vceqq_f32(self.0, vdupq_n_f32(f32::INFINITY));
            // NaN: x != x
            let is_nan = vmvnq_u32(vceqq_f32(self.0, self.0));

            let neg_inf = vdupq_n_f32(f32::NEG_INFINITY);
            let pos_inf = vdupq_n_f32(f32::INFINITY);
            let nan = vdupq_n_f32(f32::NAN);

            // vbslq_f32(mask, true_val, false_val)
            let r = vbslq_f32(is_zero, neg_inf, result.0);
            let r = vbslq_f32(is_neg, nan, r);
            let r = vbslq_f32(is_inf, pos_inf, r);
            let r = vbslq_f32(is_nan, nan, r);
            Self(r)
        }}
    }}

    /// Low-precision base-2 exponential - unchecked variant (~5.5e-3 max relative error).
    ///
    /// **Warning**: Clamps to [-126, 126]. Does not return inf for overflow.
    #[inline(always)]
    pub fn exp2_lowp_unchecked(self) -> Self {{
        const C0: f32 = 1.0;
        const C1: f32 = core::f32::consts::LN_2;
        const C2: f32 = 0.240_226_5;
        const C3: f32 = 0.055_504_11;

        unsafe {{
            // Clamp to safe range
            let x = vmaxq_f32(self.0, vdupq_n_f32(-126.0));
            let x = vminq_f32(x, vdupq_n_f32(126.0));

            let xi = vrndmq_f32(x);
            let xf = vsubq_f32(x, xi);

            // Polynomial for 2^frac using FMA
            // vfmaq_f32(a, b, c) = a + b*c
            let poly = vfmaq_f32(vdupq_n_f32(C2), vdupq_n_f32(C3), xf);
            let poly = vfmaq_f32(vdupq_n_f32(C1), poly, xf);
            let poly = vfmaq_f32(vdupq_n_f32(C0), poly, xf);

            // Scale by 2^integer using bit manipulation
            let xi_i32 = vcvtq_s32_f32(xi);
            let bias = vdupq_n_s32(127);
            let scale_bits = vshlq_n_s32::<23>(vaddq_s32(xi_i32, bias));
            let scale = vreinterpretq_f32_s32(scale_bits);

            Self(vmulq_f32(poly, scale))
        }}
    }}

    /// Low-precision base-2 exponential (~5.5e-3 max relative error).
    ///
    /// Handles edge cases: exp2(x > 128) = +inf, exp2(x < -150) = 0,
    /// exp2(NaN) = NaN.
    #[inline(always)]
    pub fn exp2_lowp(self) -> Self {{
        let result = self.exp2_lowp_unchecked();

        unsafe {{
            let is_overflow = vcgeq_f32(self.0, vdupq_n_f32(128.0));
            let is_underflow = vcltq_f32(self.0, vdupq_n_f32(-150.0));
            let is_nan = vmvnq_u32(vceqq_f32(self.0, self.0));

            let pos_inf = vdupq_n_f32(f32::INFINITY);
            let zero = vdupq_n_f32(0.0);
            let nan = vdupq_n_f32(f32::NAN);

            let r = vbslq_f32(is_overflow, pos_inf, result.0);
            let r = vbslq_f32(is_underflow, zero, r);
            let r = vbslq_f32(is_nan, nan, r);
            Self(r)
        }}
    }}

    /// Low-precision natural logarithm - unchecked variant.
    ///
    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN).
    #[inline(always)]
    pub fn ln_lowp_unchecked(self) -> Self {{
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe {{
            Self(vmulq_f32(self.log2_lowp_unchecked().0, vdupq_n_f32(LN2)))
        }}
    }}

    /// Low-precision natural logarithm.
    ///
    /// Computed as `log2_lowp(x) * ln(2)`.
    /// Handles edge cases correctly.
    #[inline(always)]
    pub fn ln_lowp(self) -> Self {{
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe {{
            Self(vmulq_f32(self.log2_lowp().0, vdupq_n_f32(LN2)))
        }}
    }}

    /// Low-precision natural exponential - unchecked variant.
    ///
    /// **Warning**: Clamps to finite range. Does not return inf for overflow.
    #[inline(always)]
    pub fn exp_lowp_unchecked(self) -> Self {{
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        unsafe {{
            Self(vmulq_f32(self.0, vdupq_n_f32(LOG2_E))).exp2_lowp_unchecked()
        }}
    }}

    /// Low-precision natural exponential (e^x).
    ///
    /// Computed as `exp2_lowp(x * log2(e))`.
    /// Handles edge cases correctly.
    #[inline(always)]
    pub fn exp_lowp(self) -> Self {{
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        unsafe {{
            Self(vmulq_f32(self.0, vdupq_n_f32(LOG2_E))).exp2_lowp()
        }}
    }}

    /// Low-precision base-10 logarithm - unchecked variant.
    ///
    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN).
    #[inline(always)]
    pub fn log10_lowp_unchecked(self) -> Self {{
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        unsafe {{
            Self(vmulq_f32(self.log2_lowp_unchecked().0, vdupq_n_f32(LOG10_2)))
        }}
    }}

    /// Low-precision base-10 logarithm.
    ///
    /// Computed as `log2_lowp(x) * log10(2)`.
    /// Handles edge cases correctly.
    #[inline(always)]
    pub fn log10_lowp(self) -> Self {{
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        unsafe {{
            Self(vmulq_f32(self.log2_lowp().0, vdupq_n_f32(LOG10_2)))
        }}
    }}

    /// Low-precision power function - unchecked variant.
    ///
    /// **Warning**: Does not handle edge cases. Only valid for positive self values.
    #[inline(always)]
    pub fn pow_lowp_unchecked(self, n: f32) -> Self {{
        unsafe {{
            Self(vmulq_f32(self.log2_lowp_unchecked().0, vdupq_n_f32(n))).exp2_lowp_unchecked()
        }}
    }}

    /// Low-precision power function (self^n).
    ///
    /// Computed as `exp2_lowp(n * log2_lowp(self))`.
    /// Note: Only valid for positive self values.
    #[inline(always)]
    pub fn pow_lowp(self, n: f32) -> Self {{
        unsafe {{
            Self(vmulq_f32(self.log2_lowp().0, vdupq_n_f32(n))).exp2_lowp()
        }}
    }}

    // ========== Mid-Precision Transcendental Operations ==========

    /// Mid-precision base-2 logarithm (~3 ULP max error) - unchecked variant.
    ///
    /// Uses (a-1)/(a+1) transform with degree-6 odd polynomial.
    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN).
    #[inline(always)]
    pub fn log2_midp_unchecked(self) -> Self {{
        const SQRT2_OVER_2: u32 = 0x3f3504f3;
        const ONE: u32 = 0x3f800000;
        const MANTISSA_MASK: u32 = 0x007fffff;
        const EXPONENT_BIAS: i32 = 127;

        const C0: f32 = 2.885_390_08;
        const C1: f32 = 0.961_800_76;
        const C2: f32 = 0.576_974_45;
        const C3: f32 = 0.434_411_97;

        unsafe {{
            let x_bits = vreinterpretq_u32_f32(self.0);

            // Normalize mantissa to [sqrt(2)/2, sqrt(2)]
            let offset = vdupq_n_u32(ONE - SQRT2_OVER_2);
            let adjusted = vaddq_u32(x_bits, offset);

            // Extract exponent
            let exp_raw = vreinterpretq_s32_u32(vshrq_n_u32::<23>(adjusted));
            let exp_biased = vsubq_s32(exp_raw, vdupq_n_s32(EXPONENT_BIAS));
            let n = vcvtq_f32_s32(exp_biased);

            // Reconstruct normalized mantissa
            let mantissa_bits = vandq_u32(adjusted, vdupq_n_u32(MANTISSA_MASK));
            let a_bits = vaddq_u32(mantissa_bits, vdupq_n_u32(SQRT2_OVER_2));
            let a = vreinterpretq_f32_u32(a_bits);

            // y = (a - 1) / (a + 1)
            let one = vdupq_n_f32(1.0);
            let y = vdivq_f32(vsubq_f32(a, one), vaddq_f32(a, one));
            let y2 = vmulq_f32(y, y);

            // Polynomial: y*(C0 + y^2*(C1 + y^2*(C2 + C3*y^2)))
            // vfmaq_f32(a, b, c) = a + b*c
            let poly = vfmaq_f32(vdupq_n_f32(C2), vdupq_n_f32(C3), y2);
            let poly = vfmaq_f32(vdupq_n_f32(C1), poly, y2);
            let poly = vfmaq_f32(vdupq_n_f32(C0), poly, y2);

            // result = y * poly + n
            Self(vfmaq_f32(n, poly, y))
        }}
    }}

    /// Mid-precision base-2 logarithm (~3 ULP max error).
    ///
    /// Handles edge cases: log2(0) = -inf, log2(negative) = NaN,
    /// log2(+inf) = +inf, log2(NaN) = NaN.
    #[inline(always)]
    pub fn log2_midp(self) -> Self {{
        let result = self.log2_midp_unchecked();

        unsafe {{
            let zero = vdupq_n_f32(0.0);
            let is_zero = vceqq_f32(self.0, zero);
            let is_neg = vcltq_f32(self.0, zero);
            let is_inf = vceqq_f32(self.0, vdupq_n_f32(f32::INFINITY));
            let is_nan = vmvnq_u32(vceqq_f32(self.0, self.0));

            let neg_inf = vdupq_n_f32(f32::NEG_INFINITY);
            let pos_inf = vdupq_n_f32(f32::INFINITY);
            let nan = vdupq_n_f32(f32::NAN);

            let r = vbslq_f32(is_zero, neg_inf, result.0);
            let r = vbslq_f32(is_neg, nan, r);
            let r = vbslq_f32(is_inf, pos_inf, r);
            let r = vbslq_f32(is_nan, nan, r);
            Self(r)
        }}
    }}

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

            let zero = vdupq_n_f32(0.0);
            let is_positive = vcgtq_f32(self.0, zero);
            let abs_x = vabsq_f32(self.0);
            let is_small = vcltq_f32(abs_x, vdupq_n_f32(DENORM_LIMIT));
            let is_denorm = vandq_u32(is_positive, is_small);

            let scaled_x = vmulq_f32(self.0, vdupq_n_f32(SCALE_UP));
            let x_for_log = vbslq_f32(is_denorm, scaled_x, self.0);

            let result = Self(x_for_log).log2_midp();

            let adjusted = vsubq_f32(result.0, vdupq_n_f32(SCALE_ADJUST));
            Self(vbslq_f32(is_denorm, adjusted, result.0))
        }}
    }}

    /// Mid-precision base-2 exponential (~2 ULP max error) - unchecked variant.
    ///
    /// Uses degree-6 polynomial approximation.
    /// **Warning**: Does not handle edge cases (underflow, overflow).
    #[inline(always)]
    pub fn exp2_midp_unchecked(self) -> Self {{
        const C0: f32 = 1.0;
        const C1: f32 = 0.693_147_182;
        const C2: f32 = 0.240_226_463;
        const C3: f32 = 0.055_504_545;
        const C4: f32 = 0.009_618_055;
        const C5: f32 = 0.001_333_37;
        const C6: f32 = 0.000_154_47;

        unsafe {{
            let x = self.0;

            let xi = vrndmq_f32(x);
            let xf = vsubq_f32(x, xi);

            // Horner's method with 6 coefficients
            let poly = vfmaq_f32(vdupq_n_f32(C5), vdupq_n_f32(C6), xf);
            let poly = vfmaq_f32(vdupq_n_f32(C4), poly, xf);
            let poly = vfmaq_f32(vdupq_n_f32(C3), poly, xf);
            let poly = vfmaq_f32(vdupq_n_f32(C2), poly, xf);
            let poly = vfmaq_f32(vdupq_n_f32(C1), poly, xf);
            let poly = vfmaq_f32(vdupq_n_f32(C0), poly, xf);

            // Scale by 2^integer
            let xi_i32 = vcvtq_s32_f32(xi);
            let bias = vdupq_n_s32(127);
            let scale_bits = vshlq_n_s32::<23>(vaddq_s32(xi_i32, bias));
            let scale = vreinterpretq_f32_s32(scale_bits);

            Self(vmulq_f32(poly, scale))
        }}
    }}

    /// Mid-precision base-2 exponential (~2 ULP max error).
    ///
    /// Handles edge cases: exp2(x > 128) = +inf, exp2(x < -150) = 0,
    /// exp2(NaN) = NaN.
    #[inline(always)]
    pub fn exp2_midp(self) -> Self {{
        unsafe {{
            let x = self.0;

            // Clamp to prevent overflow in intermediate calculations
            let x_clamped = vmaxq_f32(x, vdupq_n_f32(-150.0));
            let x_clamped = vminq_f32(x_clamped, vdupq_n_f32(128.0));

            let exp_result = Self(x_clamped).exp2_midp_unchecked().0;

            let is_underflow = vcltq_f32(x, vdupq_n_f32(-150.0));
            let is_overflow = vcgtq_f32(x, vdupq_n_f32(128.0));

            let zero = vdupq_n_f32(0.0);
            let inf = vdupq_n_f32(f32::INFINITY);

            let r = vbslq_f32(is_underflow, zero, exp_result);
            let r = vbslq_f32(is_overflow, inf, r);
            Self(r)
        }}
    }}

    /// Mid-precision natural logarithm - unchecked variant.
    ///
    /// Computed as `log2_midp_unchecked(x) * ln(2)`.
    #[inline(always)]
    pub fn ln_midp_unchecked(self) -> Self {{
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe {{
            Self(vmulq_f32(self.log2_midp_unchecked().0, vdupq_n_f32(LN2)))
        }}
    }}

    /// Mid-precision natural logarithm with edge case handling.
    ///
    /// Returns -inf for 0, NaN for negative, correct results for inf/NaN.
    #[inline(always)]
    pub fn ln_midp(self) -> Self {{
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe {{
            Self(vmulq_f32(self.log2_midp().0, vdupq_n_f32(LN2)))
        }}
    }}

    /// Mid-precision natural logarithm with full IEEE compliance.
    ///
    /// Handles all edge cases including denormals.
    #[inline(always)]
    pub fn ln_midp_precise(self) -> Self {{
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe {{
            Self(vmulq_f32(self.log2_midp_precise().0, vdupq_n_f32(LN2)))
        }}
    }}

    /// Mid-precision natural exponential (e^x) - unchecked variant.
    ///
    /// Computed as `exp2_midp_unchecked(x * log2(e))`.
    #[inline(always)]
    pub fn exp_midp_unchecked(self) -> Self {{
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        unsafe {{
            Self(vmulq_f32(self.0, vdupq_n_f32(LOG2_E))).exp2_midp_unchecked()
        }}
    }}

    /// Mid-precision natural exponential (e^x) with edge case handling.
    ///
    /// Returns 0 for large negative, inf for large positive, correct results for inf/NaN.
    #[inline(always)]
    pub fn exp_midp(self) -> Self {{
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        unsafe {{
            Self(vmulq_f32(self.0, vdupq_n_f32(LOG2_E))).exp2_midp()
        }}
    }}

    /// Mid-precision base-10 logarithm - unchecked variant.
    ///
    /// Computed as `log2_midp_unchecked(x) * log10(2)`.
    #[inline(always)]
    pub fn log10_midp_unchecked(self) -> Self {{
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        unsafe {{
            Self(vmulq_f32(self.log2_midp_unchecked().0, vdupq_n_f32(LOG10_2)))
        }}
    }}

    /// Mid-precision base-10 logarithm with edge case handling.
    ///
    /// Computed as `log2_midp(x) * log10(2)`.
    #[inline(always)]
    pub fn log10_midp(self) -> Self {{
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        unsafe {{
            Self(vmulq_f32(self.log2_midp().0, vdupq_n_f32(LOG10_2)))
        }}
    }}

    /// Mid-precision base-10 logarithm with full IEEE compliance.
    ///
    /// Handles all edge cases including denormals.
    #[inline(always)]
    pub fn log10_midp_precise(self) -> Self {{
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        unsafe {{
            Self(vmulq_f32(self.log2_midp_precise().0, vdupq_n_f32(LOG10_2)))
        }}
    }}

    /// Mid-precision power function (self^n) - unchecked variant.
    ///
    /// Computed as `exp2_midp_unchecked(n * log2_midp_unchecked(self))`.
    /// Note: Only valid for positive self values.
    #[inline(always)]
    pub fn pow_midp_unchecked(self, n: f32) -> Self {{
        unsafe {{
            Self(vmulq_f32(self.log2_midp_unchecked().0, vdupq_n_f32(n))).exp2_midp_unchecked()
        }}
    }}

    /// Mid-precision power function (self^n) with edge case handling.
    ///
    /// Handles 0, negative, inf, and NaN in base. Only valid for positive self values.
    #[inline(always)]
    pub fn pow_midp(self, n: f32) -> Self {{
        unsafe {{
            Self(vmulq_f32(self.log2_midp().0, vdupq_n_f32(n))).exp2_midp()
        }}
    }}

    /// Mid-precision power function with denormal handling.
    ///
    /// Uses `log2_midp_precise()` to handle denormal inputs correctly.
    /// Note: Only valid for positive self values.
    #[inline(always)]
    pub fn pow_midp_precise(self, n: f32) -> Self {{
        unsafe {{
            Self(vmulq_f32(self.log2_midp_precise().0, vdupq_n_f32(n))).exp2_midp()
        }}
    }}

"#}
}

fn generate_f32_cbrt_ops() -> String {
    formatdoc! {r#"
    /// Mid-precision cube root (~1 ULP max error).
    ///
    /// Uses scalar extraction for initial guess + Newton-Raphson.
    /// Handles negative values correctly (returns -cbrt(|x|)).
    ///
    /// Does not handle denormals. Use `cbrt_midp_precise()` if denormal support is needed.
    #[inline(always)]
    pub fn cbrt_midp(self) -> Self {{
        const TWO_THIRDS: f32 = 0.666_666_627;

        unsafe {{
            let x = self.0;

            // Save sign and work with absolute value
            let abs_x = vabsq_f32(x);
            let sign_mask = vdupq_n_f32(-0.0);
            let sign = vreinterpretq_f32_u32(vandq_u32(
                vreinterpretq_u32_f32(x),
                vreinterpretq_u32_f32(sign_mask),
            ));

            // Extract to scalar for initial approximation
            let arr: [f32; 4] = core::mem::transmute(abs_x);
            let approx: [f32; 4] = [
                f32::from_bits((arr[0].to_bits() / 3) + 0x2a508c2d),
                f32::from_bits((arr[1].to_bits() / 3) + 0x2a508c2d),
                f32::from_bits((arr[2].to_bits() / 3) + 0x2a508c2d),
                f32::from_bits((arr[3].to_bits() / 3) + 0x2a508c2d),
            ];
            let mut y = vld1q_f32(approx.as_ptr());

            // Newton-Raphson iterations: y = y * (2/3 + x/(3*y^3))
            for _ in 0..3 {{
                let y2 = vmulq_f32(y, y);
                let y3 = vmulq_f32(y2, y);
                let term = vdivq_f32(abs_x, vmulq_f32(vdupq_n_f32(3.0), y3));
                y = vmulq_f32(y, vaddq_f32(vdupq_n_f32(TWO_THIRDS), term));
            }}

            // Restore sign
            Self(vreinterpretq_f32_u32(vorrq_u32(
                vreinterpretq_u32_f32(y),
                vreinterpretq_u32_f32(sign),
            )))
        }}
    }}

    /// Mid-precision cube root with denormal handling (~1 ULP max error).
    ///
    /// Handles negative values correctly (returns -cbrt(|x|)).
    /// Handles all edge cases including denormals. About 67% slower than `cbrt_midp()`.
    #[inline(always)]
    pub fn cbrt_midp_precise(self) -> Self {{
        unsafe {{
            const SCALE_UP: f32 = 16777216.0;  // 2^24
            const SCALE_DOWN: f32 = 0.00390625;  // 2^(-8) = cbrt(2^(-24))
            const DENORM_LIMIT: f32 = 1.17549435e-38;

            let abs_x = vabsq_f32(self.0);
            let is_denorm = vcltq_f32(abs_x, vdupq_n_f32(DENORM_LIMIT));
            // Exclude zeros from denormal handling
            let is_zero = vceqq_f32(self.0, vdupq_n_f32(0.0));
            let is_denorm = vandq_u32(is_denorm, vmvnq_u32(is_zero));

            let scaled_x = vmulq_f32(self.0, vdupq_n_f32(SCALE_UP));
            let x_for_cbrt = vbslq_f32(is_denorm, scaled_x, self.0);

            let result = Self(x_for_cbrt).cbrt_midp();

            let scaled_result = vmulq_f32(result.0, vdupq_n_f32(SCALE_DOWN));
            Self(vbslq_f32(is_denorm, scaled_result, result.0))
        }}
    }}

"#}
}

fn generate_f64_lowp_ops() -> String {
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

    /// Low-precision base-10 logarithm.
    ///
    /// Handles edge cases correctly via std.
    #[inline(always)]
    pub fn log10_lowp(self) -> Self {{
        let arr = self.to_array();
        let result = [arr[0].log10(), arr[1].log10()];
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

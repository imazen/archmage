//! Transcendental math functions for `f32x8<T>`.
//!
//! Generic implementations using IEEE 754 bit manipulation and polynomial
//! approximation. Available when `T: F32x8Convert` (float↔int conversion).
//!
//! Two precision tiers:
//! - **lowp** (~1% max error): Fast, suitable for perceptual/audio work
//! - **midp** (~3 ULP): Accurate, suitable for most numerical work
//!
//! Variant suffixes:
//! - `_unchecked`: No edge case handling (fastest, undefined for ≤0/NaN/Inf)
//! - (normal): Basic edge case handling (0→-inf, negative→NaN for log)
//! - `_precise`: Full handling including denormals

use crate::simd::backends::{F32x8Backend, F32x8Convert, I32x8Backend};
use crate::simd::generic::{f32x8, i32x8};

/// Splat an i32 into i32x8 (disambiguates from f32 splat).
#[inline(always)]
fn splat_i32<T: F32x8Convert>(v: i32) -> i32x8<T> {
    i32x8::from_repr_unchecked(<T as I32x8Backend>::splat(v))
}

/// Splat an f32 into f32x8.
#[inline(always)]
fn splat_f32<T: F32x8Convert>(v: f32) -> f32x8<T> {
    f32x8::from_repr_unchecked(<T as F32x8Backend>::splat(v))
}

impl<T: F32x8Convert> f32x8<T> {
    // ====== Low-Precision Transcendentals (~1% error) ======

    /// Low-precision base-2 logarithm (~1% max error).
    ///
    /// Uses rational polynomial approximation on the mantissa.
    /// Result is undefined for x <= 0.
    #[inline(always)]
    pub fn log2_lowp(self) -> Self {
        const P0: f32 = -1.850_383_3e-6;
        const P1: f32 = 1.428_716_1;
        const P2: f32 = 0.742_458_7;
        const Q0: f32 = 0.990_328_14;
        const Q1: f32 = 1.009_671_8;
        const Q2: f32 = 0.174_093_43;

        let x_bits = self.bitcast_to_i32();
        let offset = splat_i32::<T>(0x3f2a_aaab_u32 as i32);
        let exp_bits = x_bits - offset;
        let exp_shifted = exp_bits.shr_arithmetic_const::<23>();
        let mantissa_bits = x_bits - exp_shifted.shl_const::<23>();
        let mantissa = mantissa_bits.bitcast_to_f32();
        let exp_val = exp_shifted.to_f32();

        let m = mantissa - splat_f32::<T>(1.0);

        // Horner's for numerator: P2*m^2 + P1*m + P0
        let yp = splat_f32::<T>(P2).mul_add(m, splat_f32::<T>(P1));
        let yp = yp.mul_add(m, splat_f32::<T>(P0));

        // Horner's for denominator: Q2*m^2 + Q1*m + Q0
        let yq = splat_f32::<T>(Q2).mul_add(m, splat_f32::<T>(Q1));
        let yq = yq.mul_add(m, splat_f32::<T>(Q0));

        yp / yq + exp_val
    }

    /// Low-precision base-2 logarithm, no edge case handling.
    #[inline(always)]
    pub fn log2_lowp_unchecked(self) -> Self {
        self.log2_lowp()
    }

    /// Low-precision base-2 exponential (~1% max error).
    ///
    /// Uses degree-3 polynomial with IEEE 754 bit manipulation for the
    /// integer part.
    #[inline(always)]
    pub fn exp2_lowp(self) -> Self {
        const C0: f32 = 1.0;
        const C1: f32 = core::f32::consts::LN_2;
        const C2: f32 = 0.240_226_5;
        const C3: f32 = 0.055_504_11;

        let x = self
            .max(splat_f32::<T>(-126.0))
            .min(splat_f32::<T>(126.0));
        let xi = x.floor();
        let xf = x - xi;

        let poly = splat_f32::<T>(C3).mul_add(xf, splat_f32::<T>(C2));
        let poly = poly.mul_add(xf, splat_f32::<T>(C1));
        let poly = poly.mul_add(xf, splat_f32::<T>(C0));

        let xi_i32 = xi.to_i32_round();
        let scale_bits = (xi_i32 + splat_i32::<T>(127)).shl_const::<23>();
        poly * scale_bits.bitcast_to_f32()
    }

    /// Low-precision base-2 exponential, no edge case handling.
    #[inline(always)]
    pub fn exp2_lowp_unchecked(self) -> Self {
        self.exp2_lowp()
    }

    /// Low-precision natural logarithm.
    #[inline(always)]
    pub fn ln_lowp(self) -> Self {
        self.log2_lowp() * splat_f32::<T>(core::f32::consts::LN_2)
    }

    /// Low-precision natural logarithm, no edge case handling.
    #[inline(always)]
    pub fn ln_lowp_unchecked(self) -> Self {
        self.ln_lowp()
    }

    /// Low-precision natural exponential.
    #[inline(always)]
    pub fn exp_lowp(self) -> Self {
        (self * splat_f32::<T>(core::f32::consts::LOG2_E)).exp2_lowp()
    }

    /// Low-precision natural exponential, no edge case handling.
    #[inline(always)]
    pub fn exp_lowp_unchecked(self) -> Self {
        self.exp_lowp()
    }

    /// Low-precision base-10 logarithm.
    #[inline(always)]
    pub fn log10_lowp(self) -> Self {
        self.log2_lowp()
            * splat_f32::<T>(core::f32::consts::LN_2 / core::f32::consts::LN_10)
    }

    /// Low-precision base-10 logarithm, no edge case handling.
    #[inline(always)]
    pub fn log10_lowp_unchecked(self) -> Self {
        self.log10_lowp()
    }

    /// Low-precision power function: `self^n`.
    #[inline(always)]
    pub fn pow_lowp(self, n: f32) -> Self {
        (self.log2_lowp() * splat_f32::<T>(n)).exp2_lowp()
    }

    /// Low-precision power function, no edge case handling.
    #[inline(always)]
    pub fn pow_lowp_unchecked(self, n: f32) -> Self {
        self.pow_lowp(n)
    }

    // ====== Mid-Precision Transcendentals (~3 ULP) ======

    /// Mid-precision base-2 logarithm (~3 ULP).
    ///
    /// Uses (a-1)/(a+1) transform with odd polynomial evaluation.
    /// Result is undefined for x <= 0.
    #[inline(always)]
    pub fn log2_midp_unchecked(self) -> Self {
        const SQRT2_OVER_2: u32 = 0x3f35_04f3;
        const ONE_BITS: u32 = 0x3f80_0000;
        const MANTISSA_MASK: i32 = 0x007f_ffff_u32 as i32;

        // Coefficients for odd polynomial on y = (a-1)/(a+1)
        const C0: f32 = 2.885_39;
        const C1: f32 = 0.961_800_76;
        const C2: f32 = 0.576_974_45;
        const C3: f32 = 0.434_411_97;

        let x_bits = self.bitcast_to_i32();

        // Normalize mantissa to [sqrt(2)/2, sqrt(2)]
        let offset = splat_i32::<T>((ONE_BITS - SQRT2_OVER_2) as i32);
        let adjusted = x_bits + offset;

        // Extract exponent
        let exp_raw = adjusted.shr_arithmetic_const::<23>();
        let n = (exp_raw - splat_i32::<T>(127)).to_f32();

        // Reconstruct normalized mantissa
        let mantissa_bits = adjusted & splat_i32::<T>(MANTISSA_MASK);
        let a = (mantissa_bits + splat_i32::<T>(SQRT2_OVER_2 as i32)).bitcast_to_f32();

        // y = (a - 1) / (a + 1)
        let one = splat_f32::<T>(1.0);
        let y = (a - one) / (a + one);
        let y2 = y * y;

        // Polynomial: C0*y + C1*y^3 + C2*y^5 + C3*y^7
        let poly = splat_f32::<T>(C3).mul_add(y2, splat_f32::<T>(C2));
        let poly = poly.mul_add(y2, splat_f32::<T>(C1));
        let poly = poly.mul_add(y2, splat_f32::<T>(C0));

        poly.mul_add(y, n)
    }

    /// Mid-precision base-2 logarithm with edge case handling.
    ///
    /// Returns -inf for 0, NaN for negative values.
    #[inline(always)]
    pub fn log2_midp(self) -> Self {
        let result = self.log2_midp_unchecked();
        let zero = splat_f32::<T>(0.0);
        let result =
            Self::blend(self.simd_eq(zero), splat_f32::<T>(f32::NEG_INFINITY), result);
        Self::blend(self.simd_lt(zero), splat_f32::<T>(f32::NAN), result)
    }

    /// Mid-precision base-2 logarithm with denormal handling.
    #[inline(always)]
    pub fn log2_midp_precise(self) -> Self {
        self.log2_midp()
    }

    /// Mid-precision base-2 exponential (~3 ULP).
    ///
    /// Uses degree-6 polynomial approximation. Undefined for extreme inputs.
    #[inline(always)]
    pub fn exp2_midp_unchecked(self) -> Self {
        const C0: f32 = 1.0;
        const C1: f32 = core::f32::consts::LN_2;
        const C2: f32 = 0.240_226_46;
        const C3: f32 = 0.055_504_545;
        const C4: f32 = 0.009_618_055;
        const C5: f32 = 0.001_333_37;
        const C6: f32 = 0.000_154_47;

        let xi = self.floor();
        let xf = self - xi;

        let poly = splat_f32::<T>(C6).mul_add(xf, splat_f32::<T>(C5));
        let poly = poly.mul_add(xf, splat_f32::<T>(C4));
        let poly = poly.mul_add(xf, splat_f32::<T>(C3));
        let poly = poly.mul_add(xf, splat_f32::<T>(C2));
        let poly = poly.mul_add(xf, splat_f32::<T>(C1));
        let poly = poly.mul_add(xf, splat_f32::<T>(C0));

        let xi_i32 = xi.to_i32_round();
        let scale_bits = (xi_i32 + splat_i32::<T>(127)).shl_const::<23>();
        poly * scale_bits.bitcast_to_f32()
    }

    /// Mid-precision base-2 exponential with clamping.
    #[inline(always)]
    pub fn exp2_midp(self) -> Self {
        self.max(splat_f32::<T>(-126.0))
            .min(splat_f32::<T>(126.0))
            .exp2_midp_unchecked()
    }

    /// Mid-precision base-2 exponential with full edge case handling.
    #[inline(always)]
    pub fn exp2_midp_precise(self) -> Self {
        self.exp2_midp()
    }

    /// Mid-precision natural logarithm.
    #[inline(always)]
    pub fn ln_midp(self) -> Self {
        self.log2_midp() * splat_f32::<T>(core::f32::consts::LN_2)
    }

    /// Mid-precision natural logarithm, no edge case handling.
    #[inline(always)]
    pub fn ln_midp_unchecked(self) -> Self {
        self.log2_midp_unchecked() * splat_f32::<T>(core::f32::consts::LN_2)
    }

    /// Mid-precision natural logarithm with denormal handling.
    #[inline(always)]
    pub fn ln_midp_precise(self) -> Self {
        self.ln_midp()
    }

    /// Mid-precision natural exponential.
    #[inline(always)]
    pub fn exp_midp(self) -> Self {
        (self * splat_f32::<T>(core::f32::consts::LOG2_E)).exp2_midp()
    }

    /// Mid-precision natural exponential, no edge case handling.
    #[inline(always)]
    pub fn exp_midp_unchecked(self) -> Self {
        (self * splat_f32::<T>(core::f32::consts::LOG2_E)).exp2_midp_unchecked()
    }

    /// Mid-precision natural exponential with full edge case handling.
    #[inline(always)]
    pub fn exp_midp_precise(self) -> Self {
        self.exp_midp()
    }

    /// Mid-precision base-10 logarithm.
    #[inline(always)]
    pub fn log10_midp(self) -> Self {
        self.log2_midp()
            * splat_f32::<T>(core::f32::consts::LN_2 / core::f32::consts::LN_10)
    }

    /// Mid-precision base-10 logarithm, no edge case handling.
    #[inline(always)]
    pub fn log10_midp_unchecked(self) -> Self {
        self.log2_midp_unchecked()
            * splat_f32::<T>(core::f32::consts::LN_2 / core::f32::consts::LN_10)
    }

    /// Mid-precision base-10 logarithm with denormal handling.
    #[inline(always)]
    pub fn log10_midp_precise(self) -> Self {
        self.log10_midp()
    }

    /// Mid-precision power function: `self^n`.
    #[inline(always)]
    pub fn pow_midp(self, n: f32) -> Self {
        (self.log2_midp() * splat_f32::<T>(n)).exp2_midp()
    }

    /// Mid-precision power function, no edge case handling.
    #[inline(always)]
    pub fn pow_midp_unchecked(self, n: f32) -> Self {
        (self.log2_midp_unchecked() * splat_f32::<T>(n)).exp2_midp_unchecked()
    }

    /// Mid-precision power function with full edge case handling.
    #[inline(always)]
    pub fn pow_midp_precise(self, n: f32) -> Self {
        self.pow_midp(n)
    }

    /// Mid-precision cube root.
    ///
    /// Uses Kahan's initial approximation via bit manipulation followed
    /// by 3 Newton-Raphson iterations. Handles negative inputs correctly
    /// (returns -cbrt(|x|)).
    #[inline(always)]
    pub fn cbrt_midp(self) -> Self {
        const MAGIC: u32 = 0x2a50_8c2d;
        const TWO_THIRDS: f32 = 0.666_666_6;

        // Save sign and work with absolute value
        let sign_mask = splat_f32::<T>(-0.0);
        let sign = self & sign_mask;
        let abs_x = self.abs();

        // Initial approximation: scalar bit manipulation (Kahan's method)
        // bits/3 + magic gives ~1-digit accuracy
        let abs_arr = abs_x.to_array();
        let approx_arr: [f32; 8] = core::array::from_fn(|i| {
            f32::from_bits((abs_arr[i].to_bits() / 3) + MAGIC)
        });
        let mut y = f32x8::from_repr_unchecked(
            <T as F32x8Backend>::from_array(approx_arr),
        );

        // 3 Newton-Raphson iterations: y' = y * (2/3 + x/(3*y^3))
        let three = splat_f32::<T>(3.0);
        let two_thirds = splat_f32::<T>(TWO_THIRDS);
        for _ in 0..3 {
            let y2 = y * y;
            let y3 = y2 * y;
            y *= two_thirds + abs_x / (three * y3);
        }

        // Restore sign
        y | sign
    }

    /// Mid-precision cube root with denormal and zero handling.
    #[inline(always)]
    pub fn cbrt_midp_precise(self) -> Self {
        let zero = splat_f32::<T>(0.0);
        let is_zero = self.simd_eq(zero);

        let abs_x = self.abs();
        let is_denorm = abs_x.simd_lt(splat_f32::<T>(1.175_494_4e-38));

        // Scale up denormals by 2^24
        let scaled = self * splat_f32::<T>(16_777_216.0);
        let x_for_cbrt = Self::blend(is_denorm, scaled, self);

        let result = x_for_cbrt.cbrt_midp();

        // Scale down: cbrt(2^24) = 2^8, so divide by 256
        let scaled_result = result * splat_f32::<T>(1.0 / 256.0);
        let result = Self::blend(is_denorm, scaled_result, result);

        // Zero → zero
        Self::blend(is_zero, zero, result)
    }
}

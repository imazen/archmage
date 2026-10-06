//! Transcendental math functions for `f32x8<T>`.
//!
//! Generic implementations using IEEE 754 bit manipulation and polynomial
//! approximation. Available when `T: F32x8Convert` (float↔int conversion).
//!
//! Two precision tiers. Errors are maxima measured over every f32 in each
//! function's normal range against an f64 reference, on backends with and
//! without hardware FMA:
//! - **lowp**: fast, for perceptual and audio work. `exp2`, `exp` and `pow`
//!   stay within 0.56% relative error, 1.23% for results above 2^127.99;
//!   `log2`, `ln` and `log10` within 8.5e-6 absolute error, so their
//!   relative error grows without bound near x = 1; `cbrt` within 3e-5
//!   relative error.
//! - **midp**: for most numerical work. `log2`, `ln` and `log10` at most
//!   4.5 ULP, `cbrt` 3.2 ULP, `exp2` 1.9 ULP below x = 127.5 and 134.1
//!   ULP above. `exp` and `pow` lose accuracy as the exponent grows; each
//!   function gives its figures.
//!
//! Variant suffixes:
//! - `_unchecked`: No edge case handling (fastest, undefined for ≤0/NaN/Inf)
//! - (normal): Basic edge case handling (0→-inf, negative→NaN for log)
//! - `_precise`: Full handling including denormals

use crate::simd::backends::{F32x8Backend, F32x8Convert, I32x8Backend};
use crate::simd::generic::{f32x8, i32x8};

/// Splat an i32 into i32x8 (disambiguates from f32 splat).
#[inline(always)]
fn splat_i32<T: F32x8Convert>(token: T, v: i32) -> i32x8<T> {
    i32x8::from_repr_unchecked(token, <T as I32x8Backend>::splat(token, v))
}

/// Splat an f32 into f32x8.
#[inline(always)]
fn splat_f32<T: F32x8Convert>(token: T, v: f32) -> f32x8<T> {
    f32x8::from_repr_unchecked(token, <T as F32x8Backend>::splat(token, v))
}

impl<T: F32x8Convert> f32x8<T> {
    // ====== Low-Precision Transcendentals ======

    /// Low-precision base-2 logarithm (absolute error at most 6.4e-6).
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
        let offset = splat_i32::<T>(self.1, 0x3f2a_aaab_u32 as i32);
        let exp_bits = x_bits - offset;
        let exp_shifted = exp_bits.shr_arithmetic_const::<23>();
        let mantissa_bits = x_bits - exp_shifted.shl_const::<23>();
        let mantissa = mantissa_bits.bitcast_to_f32();
        let exp_val = exp_shifted.to_f32();

        let m = mantissa - splat_f32::<T>(self.1, 1.0);

        let yp = splat_f32::<T>(self.1, P2).mul_add(m, splat_f32::<T>(self.1, P1));
        let yp = yp.mul_add(m, splat_f32::<T>(self.1, P0));

        let yq = splat_f32::<T>(self.1, Q2).mul_add(m, splat_f32::<T>(self.1, Q1));
        let yq = yq.mul_add(m, splat_f32::<T>(self.1, Q0));

        yp / yq + exp_val
    }

    /// Low-precision base-2 logarithm, no edge case handling.
    #[inline(always)]
    pub fn log2_lowp_unchecked(self) -> Self {
        self.log2_lowp()
    }

    /// Low-precision base-2 exponential: relative error at most 0.56% for x up
    /// to 127.99 and 1.23% above, where the input is clamped to 127.99.
    #[inline(always)]
    pub fn exp2_lowp(self) -> Self {
        const C0: f32 = 1.0;
        const C1: f32 = core::f32::consts::LN_2;
        const C2: f32 = 0.240_226_5;
        const C3: f32 = 0.055_504_11;

        // floor(x) <= 127 keeps the bit trick (n+127)<<23 in range; at 127.99
        // the result stays below f32::MAX despite the polynomial's error.
        let x = self
            .max(splat_f32::<T>(self.1, -126.0))
            .min(splat_f32::<T>(self.1, 127.99));
        let xi = x.floor();
        let xf = x - xi;

        let poly = splat_f32::<T>(self.1, C3).mul_add(xf, splat_f32::<T>(self.1, C2));
        let poly = poly.mul_add(xf, splat_f32::<T>(self.1, C1));
        let poly = poly.mul_add(xf, splat_f32::<T>(self.1, C0));

        let xi_i32 = xi.to_i32_round();
        let scale_bits = (xi_i32 + splat_i32::<T>(self.1, 127)).shl_const::<23>();
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
        self.log2_lowp() * splat_f32::<T>(self.1, core::f32::consts::LN_2)
    }

    /// Low-precision natural logarithm, no edge case handling.
    #[inline(always)]
    pub fn ln_lowp_unchecked(self) -> Self {
        self.ln_lowp()
    }

    /// Low-precision natural exponential.
    #[inline(always)]
    pub fn exp_lowp(self) -> Self {
        (self * splat_f32::<T>(self.1, core::f32::consts::LOG2_E)).exp2_lowp()
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
            * splat_f32::<T>(self.1, core::f32::consts::LN_2 / core::f32::consts::LN_10)
    }

    /// Low-precision base-10 logarithm, no edge case handling.
    #[inline(always)]
    pub fn log10_lowp_unchecked(self) -> Self {
        self.log10_lowp()
    }

    /// Low-precision power function: `self^n`. Returns 0 for zero input.
    #[inline(always)]
    pub fn pow_lowp(self, n: f32) -> Self {
        let result = (self.log2_lowp() * splat_f32::<T>(self.1, n)).exp2_lowp();
        // Zero masking: pow(0, n) = 0 for n > 0
        let is_zero = self.simd_eq(splat_f32::<T>(self.1, 0.0));
        Self::blend(is_zero, splat_f32::<T>(self.1, 0.0), result)
    }

    /// Low-precision power function, no edge case handling.
    #[inline(always)]
    pub fn pow_lowp_unchecked(self, n: f32) -> Self {
        self.pow_lowp(n)
    }

    // ====== Mid-Precision Transcendentals ======

    /// Mid-precision base-2 logarithm (at most 4.5 ULP).
    ///
    /// Uses (a-1)/(a+1) transform with odd polynomial evaluation.
    /// Result is undefined for x <= 0.
    #[inline(always)]
    pub fn log2_midp_unchecked(self) -> Self {
        const SQRT2_OVER_2: u32 = 0x3f35_04f3;
        const ONE_BITS: u32 = 0x3f80_0000;
        const MANTISSA_MASK: i32 = 0x007f_ffff_u32 as i32;

        const C0: f32 = 2.885_39;
        const C1: f32 = 0.961_800_76;
        const C2: f32 = 0.576_974_45;
        const C3: f32 = 0.434_411_97;

        let x_bits = self.bitcast_to_i32();

        let offset = splat_i32::<T>(self.1, (ONE_BITS - SQRT2_OVER_2) as i32);
        let adjusted = x_bits + offset;

        let exp_raw = adjusted.shr_arithmetic_const::<23>();
        let n = (exp_raw - splat_i32::<T>(self.1, 127)).to_f32();

        let mantissa_bits = adjusted & splat_i32::<T>(self.1, MANTISSA_MASK);
        let a = (mantissa_bits + splat_i32::<T>(self.1, SQRT2_OVER_2 as i32)).bitcast_to_f32();

        let one = splat_f32::<T>(self.1, 1.0);
        let y = (a - one) / (a + one);
        let y2 = y * y;

        let poly = splat_f32::<T>(self.1, C3).mul_add(y2, splat_f32::<T>(self.1, C2));
        let poly = poly.mul_add(y2, splat_f32::<T>(self.1, C1));
        let poly = poly.mul_add(y2, splat_f32::<T>(self.1, C0));

        poly.mul_add(y, n)
    }

    /// Mid-precision base-2 logarithm with edge case handling.
    ///
    /// Returns -inf for 0, NaN for negative values, +inf for +inf.
    #[inline(always)]
    pub fn log2_midp(self) -> Self {
        let result = self.log2_midp_unchecked();
        let zero = splat_f32::<T>(self.1, 0.0);
        let result = Self::blend(
            self.simd_eq(zero),
            splat_f32::<T>(self.1, f32::NEG_INFINITY),
            result,
        );
        let result = Self::blend(self.simd_lt(zero), splat_f32::<T>(self.1, f32::NAN), result);
        // +inf -> +inf; the polynomial otherwise returns inf's raw
        // unbiased exponent (128) for an infinite input.
        let inf = splat_f32::<T>(self.1, f32::INFINITY);
        Self::blend(self.simd_eq(inf), inf, result)
    }

    /// Mid-precision base-2 logarithm with denormal handling.
    #[inline(always)]
    pub fn log2_midp_precise(self) -> Self {
        self.log2_midp()
    }

    /// Mid-precision base-2 exponential: at most 1.9 ULP for x below 127.5 and
    /// up to 134.1 ULP in [127.5, 128). Undefined outside [-126, 128).
    ///
    /// Uses round-to-nearest splitting to keep |frac| <= 0.5, giving
    /// ~1000x less polynomial truncation error than floor-based splitting.
    #[inline(always)]
    pub fn exp2_midp_unchecked(self) -> Self {
        const C0: f32 = 1.0;
        const C1: f32 = core::f32::consts::LN_2;
        const C2: f32 = 0.240_226_46;
        const C3: f32 = 0.055_504_545;
        const C4: f32 = 0.009_618_055;
        const C5: f32 = 0.001_333_37;
        const C6: f32 = 0.000_154_47;

        // Round-to-nearest keeps |frac| <= 0.5 (vs floor's [0,1))
        // Clamp xi to 127 so the bit trick (n+127)<<23 doesn't overflow.
        // For x in [127.5, 128) that leaves |frac| up to 1, outside the
        // polynomial's range: up to 134.1 ULP there. Folding in the missing
        // factor of two measured 7-12% slower for exp2/exp/pow with AVX2 and
        // 4-8% slower on the scalar backend.
        let xi = self.round().min(splat_f32::<T>(self.1, 127.0));
        let xf = self - xi;

        let poly = splat_f32::<T>(self.1, C6).mul_add(xf, splat_f32::<T>(self.1, C5));
        let poly = poly.mul_add(xf, splat_f32::<T>(self.1, C4));
        let poly = poly.mul_add(xf, splat_f32::<T>(self.1, C3));
        let poly = poly.mul_add(xf, splat_f32::<T>(self.1, C2));
        let poly = poly.mul_add(xf, splat_f32::<T>(self.1, C1));
        let poly = poly.mul_add(xf, splat_f32::<T>(self.1, C0));

        let xi_i32 = xi.to_i32_round();
        let scale_bits = (xi_i32 + splat_i32::<T>(self.1, 127)).shl_const::<23>();
        poly * scale_bits.bitcast_to_f32()
    }

    /// Mid-precision base-2 exponential with clamping.
    ///
    /// At most 1.9 ULP for x below 127.5 and up to 134.1 ULP in [127.5, 128).
    /// Returns 0 for x < -126 (denormal results can't be constructed),
    /// inf for x >= 128.
    #[inline(always)]
    pub fn exp2_midp(self) -> Self {
        let underflow_limit = splat_f32::<T>(self.1, -126.0);
        let overflow_limit = splat_f32::<T>(self.1, 128.0);

        // Clamp to prevent overflow in intermediate calculations
        let clamped = self.max(underflow_limit).min(overflow_limit);
        let result = clamped.exp2_midp_unchecked();

        // Handle edge cases: large negative → 0, large positive → inf
        // 2^128 > f32::MAX, so >= 128 must return inf
        let is_underflow = self.simd_lt(underflow_limit);
        let is_overflow = self.simd_ge(overflow_limit);
        let zero = splat_f32::<T>(self.1, 0.0);
        let inf = splat_f32::<T>(self.1, f32::INFINITY);
        let result = Self::blend(is_underflow, zero, result);
        Self::blend(is_overflow, inf, result)
    }

    /// Mid-precision base-2 exponential with full edge case handling.
    #[inline(always)]
    pub fn exp2_midp_precise(self) -> Self {
        self.exp2_midp()
    }

    /// Mid-precision natural logarithm (at most 4.1 ULP).
    #[inline(always)]
    pub fn ln_midp(self) -> Self {
        self.log2_midp() * splat_f32::<T>(self.1, core::f32::consts::LN_2)
    }

    /// Mid-precision natural logarithm, no edge case handling.
    #[inline(always)]
    pub fn ln_midp_unchecked(self) -> Self {
        self.log2_midp_unchecked() * splat_f32::<T>(self.1, core::f32::consts::LN_2)
    }

    /// Mid-precision natural logarithm with denormal handling.
    #[inline(always)]
    pub fn ln_midp_precise(self) -> Self {
        self.ln_midp()
    }

    /// Mid-precision natural exponential.
    ///
    /// Computed as `exp2(x * log2(e))`, so the rounding of that product grows
    /// with |x|: at most 2.0 ULP for |x| <= 1, 8.2 ULP for |x| <= 10,
    /// 31.3 ULP for |x| <= 40, 64.1 ULP over [-87, 88.5] and 197.1 ULP above
    /// 88.5, where the product nears 128 (see `exp2_midp`).
    #[inline(always)]
    pub fn exp_midp(self) -> Self {
        (self * splat_f32::<T>(self.1, core::f32::consts::LOG2_E)).exp2_midp()
    }

    /// Mid-precision natural exponential, no edge case handling.
    #[inline(always)]
    pub fn exp_midp_unchecked(self) -> Self {
        (self * splat_f32::<T>(self.1, core::f32::consts::LOG2_E)).exp2_midp_unchecked()
    }

    /// Mid-precision natural exponential with full edge case handling.
    #[inline(always)]
    pub fn exp_midp_precise(self) -> Self {
        self.exp_midp()
    }

    /// Mid-precision logistic sigmoid: `1 / (1 + exp(-x))`, in `[0, 1]`.
    ///
    /// Rail-safe by construction (issue #64): the division is exact
    /// IEEE, so the saturated exponential behaves — for `x` ≲ -88,
    /// `exp_midp(-x)` overflows to `inf` and the result is `1/inf = 0`;
    /// for large positive `x` it underflows to `0` and the result is
    /// exactly `1`. Conv pre-activations at ±100 produce clean 0/1
    /// lanes, not NaN.
    #[inline(always)]
    pub fn sigmoid_midp(self) -> Self {
        let one = splat_f32::<T>(self.1, 1.0);
        // Exact IEEE division, deliberately NOT `.recip()`. recip()'s
        // working tier now carries exact rails too, but its <= 4 ULP
        // slack breaks a pinned nicety: recip(1.0) lands ~1 ULP under
        // 1.0, so saturated sigmoid would return 0.9999999 instead of
        // exactly 1.0 (sigmoid_silu.rs pins the exact 0/1 rails).
        // Division costs ~nothing here — exp_midp dominates.
        one / ((-self).exp_midp() + one)
    }

    /// Mid-precision SiLU / swish: `x * sigmoid(x)` = `x / (1 + exp(-x))`.
    ///
    /// Same rails as [`sigmoid_midp`](Self::sigmoid_midp): for `x` ≲ -88
    /// the result is `x * 0 = -0.0`, and for large positive `x` it is
    /// exactly `x`.
    #[inline(always)]
    pub fn silu_midp(self) -> Self {
        self * self.sigmoid_midp()
    }

    /// Mid-precision base-10 logarithm (at most 4.5 ULP).
    #[inline(always)]
    pub fn log10_midp(self) -> Self {
        self.log2_midp()
            * splat_f32::<T>(self.1, core::f32::consts::LN_2 / core::f32::consts::LN_10)
    }

    /// Mid-precision base-10 logarithm, no edge case handling.
    #[inline(always)]
    pub fn log10_midp_unchecked(self) -> Self {
        self.log2_midp_unchecked()
            * splat_f32::<T>(self.1, core::f32::consts::LN_2 / core::f32::consts::LN_10)
    }

    /// Mid-precision base-10 logarithm with denormal handling.
    #[inline(always)]
    pub fn log10_midp_precise(self) -> Self {
        self.log10_midp()
    }

    /// Mid-precision power function: `self^n`.
    ///
    /// Computed as `exp2(n * log2(self))`, so the error grows with
    /// |n * log2(self)|. Measured for n = 2.4: at most 27.8 ULP (mean 3.7) on
    /// [2^-8, 2^8], 64.8 ULP on [2^-20, 2^20] and 144.9 ULP on [2^-50, 2^50].
    #[inline(always)]
    pub fn pow_midp(self, n: f32) -> Self {
        (self.log2_midp() * splat_f32::<T>(self.1, n)).exp2_midp()
    }

    /// Mid-precision power function, no edge case handling.
    #[inline(always)]
    pub fn pow_midp_unchecked(self, n: f32) -> Self {
        (self.log2_midp_unchecked() * splat_f32::<T>(self.1, n)).exp2_midp_unchecked()
    }

    /// Mid-precision power function with full edge case handling.
    #[inline(always)]
    pub fn pow_midp_precise(self, n: f32) -> Self {
        self.pow_midp(n)
    }

    /// Low-precision cube root (~15 bits, ~4.5 decimal digits).
    ///
    /// Uses Kahan's bit-hack initial approximation followed by 1 Halley
    /// iteration. Over every positive normal f32 below `f32::MAX / 3`: max
    /// 259 ULP vs `std::f32::cbrt` (relative error 3e-5), mean 56 ULP.
    /// Larger magnitudes overflow an intermediate and return NaN (±inf for
    /// some near the limit); use `cbrt_midp_precise` for the whole range.
    ///
    /// Fastest cbrt variant — 1.8x faster than `cbrt_midp` (1 division
    /// vs 2). Suitable for perceptual color (Oklab/XYB) targeting 8-bit
    /// output, or any context where ~4.5 decimal digits suffice.
    ///
    /// Returns ±0 for ±0 input. Does not handle denormals or infinity — use
    /// `cbrt_midp_precise` for those.
    #[inline(always)]
    pub fn cbrt_lowp(self) -> Self {
        const MAGIC: u32 = 0x2a50_8c2d;

        let sign_mask = splat_f32::<T>(self.1, -0.0);
        let sign = self & sign_mask;
        let abs_x = self.abs();

        let abs_arr = abs_x.to_array();
        let approx_arr: [f32; 8] =
            core::array::from_fn(|i| f32::from_bits((abs_arr[i].to_bits() / 3) + MAGIC));
        let mut y =
            f32x8::from_repr_unchecked(self.1, <T as F32x8Backend>::from_array(self.1, approx_arr));

        // Halley iteration: y *= (y³ + 2x) / (2y³ + x)
        // y³ + 2x reaches 3x, so this overflows above f32::MAX / 3.
        // Triples bits of precision: ~5 → ~15
        let two = splat_f32::<T>(self.1, 2.0);
        let y3 = y * y * y;
        y *= (y3 + two * abs_x) / (two * y3 + abs_x);

        let result = y | sign;

        // Zero masking: cbrt(±0) = ±0 (bit hack gives garbage for zero)
        let is_zero = self.simd_eq(splat_f32::<T>(self.1, 0.0));
        Self::blend(is_zero, self, result)
    }

    /// Mid-precision cube root (max 3.2 ULP vs `std::f32::cbrt`).
    ///
    /// Uses Kahan's bit-hack initial approximation followed by 2 Halley
    /// iterations. Each Halley step triples precision: ~5 → ~15 → ~45
    /// bits, saturating f32's 24-bit mantissa. Over every positive normal
    /// f32 below `f32::MAX / 3`: max 3.2 ULP, mean 0.53 ULP.
    ///
    /// Uses 2 divisions (vs 3 for Newton-Raphson at equivalent accuracy),
    /// making it ~35% faster at equal or better precision.
    ///
    /// Returns ±0 for ±0 input. Magnitudes above `f32::MAX / 3` (1.13e38)
    /// overflow an intermediate and return NaN (±inf for a few just above
    /// the limit), and denormals and infinity are not handled: use
    /// `cbrt_midp_precise` for those.
    #[inline(always)]
    pub fn cbrt_midp(self) -> Self {
        const MAGIC: u32 = 0x2a50_8c2d;

        let sign_mask = splat_f32::<T>(self.1, -0.0);
        let sign = self & sign_mask;
        let abs_x = self.abs();

        let abs_arr = abs_x.to_array();
        let approx_arr: [f32; 8] =
            core::array::from_fn(|i| f32::from_bits((abs_arr[i].to_bits() / 3) + MAGIC));
        let mut y =
            f32x8::from_repr_unchecked(self.1, <T as F32x8Backend>::from_array(self.1, approx_arr));

        // 2 Halley iterations: y *= (y³ + 2x) / (2y³ + x)
        // y³ + 2x reaches 3x, so this overflows above f32::MAX / 3;
        // cbrt_midp_precise rescales those inputs.
        let two = splat_f32::<T>(self.1, 2.0);
        for _ in 0..2 {
            let y3 = y * y * y;
            y *= (y3 + two * abs_x) / (two * y3 + abs_x);
        }

        let result = y | sign;

        // Zero masking: cbrt(±0) = ±0 (bit hack gives garbage for zero)
        let is_zero = self.simd_eq(splat_f32::<T>(self.1, 0.0));
        Self::blend(is_zero, self, result)
    }

    /// Mid-precision cube root over the whole f32 range (max 3.2 ULP).
    ///
    /// Wraps `cbrt_midp()`: denormals are scaled up first, magnitudes from
    /// 1e36 up run on x/8 with the result doubled so the Halley step cannot
    /// overflow, and ±0 and ±inf return themselves.
    #[inline(always)]
    pub fn cbrt_midp_precise(self) -> Self {
        let zero = splat_f32::<T>(self.1, 0.0);
        let one = splat_f32::<T>(self.1, 1.0);
        let abs_x = self.abs();
        // ±0 and ±inf are their own cube roots.
        let keep = self.simd_eq(zero) | abs_x.simd_eq(splat_f32::<T>(self.1, f32::INFINITY));
        let is_denorm = abs_x.simd_lt(splat_f32::<T>(self.1, 1.175_494_4e-38));
        // cbrt_midp overflows above f32::MAX / 3, so magnitudes from 1e36 up
        // run on x/8 and get doubled; denormals run on x * 2^24 and get
        // scaled by 2^-8. All four scalings are exact.
        let is_big = abs_x.simd_ge(splat_f32::<T>(self.1, 1.0e36));
        let pre = Self::blend(
            is_denorm,
            splat_f32::<T>(self.1, 16_777_216.0),
            Self::blend(is_big, splat_f32::<T>(self.1, 0.125), one),
        );
        let post = Self::blend(
            is_denorm,
            splat_f32::<T>(self.1, 1.0 / 256.0),
            Self::blend(is_big, splat_f32::<T>(self.1, 2.0), one),
        );

        let result = (self * pre).cbrt_midp() * post;
        Self::blend(keep, self, result)
    }
}

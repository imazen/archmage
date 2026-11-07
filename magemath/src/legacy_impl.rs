//! Exact legacy backend rounding and specials, moved from Magetypes.
//! Canonical Magemath arithmetic, initially imported from the 35 legacy bodies.
//! No scalarization: each width uses its original vector operations.
macro_rules! legacy_bodies {
    ($float:ident,$int:ident,$float_backend:ident,$convert:ident,$int_backend:ident,$lanes:literal) => {
        use magetypes::simd::backends::$convert as FloatConvert;
        use magetypes::simd::generic::{$float as FloatVec, $int as IntVec};
        #[inline(always)]
        fn splat_i32<T: FloatConvert>(token: T, v: i32) -> IntVec<T> {
            IntVec::splat_t(token, v)
        }
        #[inline(always)]
        fn splat_f32<T: FloatConvert>(token: T, v: f32) -> FloatVec<T> {
            FloatVec::splat_t(token, v)
        }

        #[inline(always)]
        pub(crate) fn log2_lowp<T: FloatConvert>(token: T, x: FloatVec<T>) -> FloatVec<T> {
            const P0: f32 = -1.850_383_3e-6;
            const P1: f32 = 1.428_716_1;
            const P2: f32 = 0.742_458_7;
            const Q0: f32 = 0.990_328_14;
            const Q1: f32 = 1.009_671_8;
            const Q2: f32 = 0.174_093_43;

            let x_bits = x.bitcast_to_i32();
            let offset = splat_i32::<T>(token, 0x3f2a_aaab_u32 as i32);
            let exp_bits = x_bits - offset;
            let exp_shifted = exp_bits.shr_arithmetic_const::<23>();
            let mantissa_bits = x_bits - exp_shifted.shl_const::<23>();
            let mantissa = mantissa_bits.bitcast_to_f32();
            let exp_val = exp_shifted.to_f32();

            let m = mantissa - splat_f32::<T>(token, 1.0);

            let yp = splat_f32::<T>(token, P2).mul_add_relaxed(m, splat_f32::<T>(token, P1));
            let yp = yp.mul_add_relaxed(m, splat_f32::<T>(token, P0));

            let yq = splat_f32::<T>(token, Q2).mul_add_relaxed(m, splat_f32::<T>(token, Q1));
            let yq = yq.mul_add_relaxed(m, splat_f32::<T>(token, Q0));

            yp / yq + exp_val
        }

        #[inline(always)]
        pub(crate) fn log2_lowp_unchecked<T: FloatConvert>(
            token: T,
            x: FloatVec<T>,
        ) -> FloatVec<T> {
            log2_lowp::<T>(token, x)
        }

        #[inline(always)]
        pub(crate) fn exp2_lowp<T: FloatConvert>(token: T, x: FloatVec<T>) -> FloatVec<T> {
            const C0: f32 = 1.0;
            const C1: f32 = core::f32::consts::LN_2;
            const C2: f32 = 0.240_226_5;
            const C3: f32 = 0.055_504_11;

            // floor(x) <= 127 keeps the bit trick (n+127)<<23 in range; at 127.99
            // the result stays below f32::MAX despite the polynomial's error.
            let x = x
                .max(splat_f32::<T>(token, -126.0))
                .min(splat_f32::<T>(token, 127.99));
            let xi = x.floor();
            let xf = x - xi;

            let poly = splat_f32::<T>(token, C3).mul_add_relaxed(xf, splat_f32::<T>(token, C2));
            let poly = poly.mul_add_relaxed(xf, splat_f32::<T>(token, C1));
            let poly = poly.mul_add_relaxed(xf, splat_f32::<T>(token, C0));

            let xi_i32 = xi.to_i32_round();
            let scale_bits = (xi_i32 + splat_i32::<T>(token, 127)).shl_const::<23>();
            poly * scale_bits.bitcast_to_f32()
        }

        #[inline(always)]
        pub(crate) fn exp2_lowp_unchecked<T: FloatConvert>(
            token: T,
            x: FloatVec<T>,
        ) -> FloatVec<T> {
            exp2_lowp::<T>(token, x)
        }

        #[inline(always)]
        pub(crate) fn ln_lowp<T: FloatConvert>(token: T, x: FloatVec<T>) -> FloatVec<T> {
            log2_lowp::<T>(token, x) * splat_f32::<T>(token, core::f32::consts::LN_2)
        }

        #[inline(always)]
        pub(crate) fn ln_lowp_unchecked<T: FloatConvert>(token: T, x: FloatVec<T>) -> FloatVec<T> {
            ln_lowp::<T>(token, x)
        }

        #[inline(always)]
        pub(crate) fn exp_lowp<T: FloatConvert>(token: T, x: FloatVec<T>) -> FloatVec<T> {
            exp2_lowp::<T>(token, x * splat_f32::<T>(token, core::f32::consts::LOG2_E))
        }

        #[inline(always)]
        pub(crate) fn exp_lowp_unchecked<T: FloatConvert>(token: T, x: FloatVec<T>) -> FloatVec<T> {
            exp_lowp::<T>(token, x)
        }

        #[inline(always)]
        pub(crate) fn log10_lowp<T: FloatConvert>(token: T, x: FloatVec<T>) -> FloatVec<T> {
            log2_lowp::<T>(token, x)
                * splat_f32::<T>(token, core::f32::consts::LN_2 / core::f32::consts::LN_10)
        }

        #[inline(always)]
        pub(crate) fn log10_lowp_unchecked<T: FloatConvert>(
            token: T,
            x: FloatVec<T>,
        ) -> FloatVec<T> {
            log10_lowp::<T>(token, x)
        }

        #[inline(always)]
        pub(crate) fn pow_lowp<T: FloatConvert>(token: T, x: FloatVec<T>, n: f32) -> FloatVec<T> {
            let result = exp2_lowp::<T>(token, log2_lowp::<T>(token, x) * splat_f32::<T>(token, n));
            // Zero masking: pow(0, n) = 0 for n > 0
            let is_zero = x.simd_eq(splat_f32::<T>(token, 0.0));
            FloatVec::blend(is_zero, splat_f32::<T>(token, 0.0), result)
        }

        #[inline(always)]
        pub(crate) fn pow_lowp_unchecked<T: FloatConvert>(
            token: T,
            x: FloatVec<T>,
            n: f32,
        ) -> FloatVec<T> {
            pow_lowp::<T>(token, x, n)
        }

        #[inline(always)]
        pub(crate) fn log2_midp_unchecked<T: FloatConvert>(
            token: T,
            x: FloatVec<T>,
        ) -> FloatVec<T> {
            const SQRT2_OVER_2: u32 = 0x3f35_04f3;
            const ONE_BITS: u32 = 0x3f80_0000;
            const MANTISSA_MASK: i32 = 0x007f_ffff_u32 as i32;

            const C0: f32 = 2.885_39;
            const C1: f32 = 0.961_800_76;
            const C2: f32 = 0.576_974_45;
            const C3: f32 = 0.434_411_97;

            let x_bits = x.bitcast_to_i32();

            let offset = splat_i32::<T>(token, (ONE_BITS - SQRT2_OVER_2) as i32);
            let adjusted = x_bits + offset;

            let exp_raw = adjusted.shr_arithmetic_const::<23>();
            let n = (exp_raw - splat_i32::<T>(token, 127)).to_f32();

            let mantissa_bits = adjusted & splat_i32::<T>(token, MANTISSA_MASK);
            let a = (mantissa_bits + splat_i32::<T>(token, SQRT2_OVER_2 as i32)).bitcast_to_f32();

            let one = splat_f32::<T>(token, 1.0);
            let y = (a - one) / (a + one);
            let y2 = y * y;

            let poly = splat_f32::<T>(token, C3).mul_add_relaxed(y2, splat_f32::<T>(token, C2));
            let poly = poly.mul_add_relaxed(y2, splat_f32::<T>(token, C1));
            let poly = poly.mul_add_relaxed(y2, splat_f32::<T>(token, C0));

            poly.mul_add_relaxed(y, n)
        }

        #[inline(always)]
        pub(crate) fn log2_midp<T: FloatConvert>(token: T, x: FloatVec<T>) -> FloatVec<T> {
            let result = log2_midp_unchecked::<T>(token, x);
            let zero = splat_f32::<T>(token, 0.0);
            let result = FloatVec::blend(
                x.simd_eq(zero),
                splat_f32::<T>(token, f32::NEG_INFINITY),
                result,
            );
            let result = FloatVec::blend(x.simd_lt(zero), splat_f32::<T>(token, f32::NAN), result);
            // +inf -> +inf; the polynomial otherwise returns inf's raw
            // unbiased exponent (128) for an infinite input.
            let inf = splat_f32::<T>(token, f32::INFINITY);
            FloatVec::blend(x.simd_eq(inf), inf, result)
        }

        #[inline(always)]
        pub(crate) fn log2_midp_precise<T: FloatConvert>(token: T, x: FloatVec<T>) -> FloatVec<T> {
            log2_midp::<T>(token, x)
        }

        #[inline(always)]
        pub(crate) fn exp2_midp_unchecked<T: FloatConvert>(
            token: T,
            x: FloatVec<T>,
        ) -> FloatVec<T> {
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
            let xi = x.round().min(splat_f32::<T>(token, 127.0));
            let xf = x - xi;

            let poly = splat_f32::<T>(token, C6).mul_add_relaxed(xf, splat_f32::<T>(token, C5));
            let poly = poly.mul_add_relaxed(xf, splat_f32::<T>(token, C4));
            let poly = poly.mul_add_relaxed(xf, splat_f32::<T>(token, C3));
            let poly = poly.mul_add_relaxed(xf, splat_f32::<T>(token, C2));
            let poly = poly.mul_add_relaxed(xf, splat_f32::<T>(token, C1));
            let poly = poly.mul_add_relaxed(xf, splat_f32::<T>(token, C0));

            let xi_i32 = xi.to_i32_round();
            let scale_bits = (xi_i32 + splat_i32::<T>(token, 127)).shl_const::<23>();
            poly * scale_bits.bitcast_to_f32()
        }

        #[inline(always)]
        pub(crate) fn exp2_midp<T: FloatConvert>(token: T, x: FloatVec<T>) -> FloatVec<T> {
            let underflow_limit = splat_f32::<T>(token, -126.0);
            let overflow_limit = splat_f32::<T>(token, 128.0);

            // Clamp to prevent overflow in intermediate calculations
            let clamped = x.max(underflow_limit).min(overflow_limit);
            let result = exp2_midp_unchecked::<T>(token, clamped);

            // Handle edge cases: large negative → 0, large positive → inf
            // 2^128 > f32::MAX, so >= 128 must return inf
            let is_underflow = x.simd_lt(underflow_limit);
            let is_overflow = x.simd_ge(overflow_limit);
            let zero = splat_f32::<T>(token, 0.0);
            let inf = splat_f32::<T>(token, f32::INFINITY);
            let result = FloatVec::blend(is_underflow, zero, result);
            FloatVec::blend(is_overflow, inf, result)
        }

        #[inline(always)]
        pub(crate) fn exp2_midp_precise<T: FloatConvert>(token: T, x: FloatVec<T>) -> FloatVec<T> {
            exp2_midp::<T>(token, x)
        }

        #[inline(always)]
        pub(crate) fn ln_midp<T: FloatConvert>(token: T, x: FloatVec<T>) -> FloatVec<T> {
            log2_midp::<T>(token, x) * splat_f32::<T>(token, core::f32::consts::LN_2)
        }

        #[inline(always)]
        pub(crate) fn ln_midp_unchecked<T: FloatConvert>(token: T, x: FloatVec<T>) -> FloatVec<T> {
            log2_midp_unchecked::<T>(token, x) * splat_f32::<T>(token, core::f32::consts::LN_2)
        }

        #[inline(always)]
        pub(crate) fn ln_midp_precise<T: FloatConvert>(token: T, x: FloatVec<T>) -> FloatVec<T> {
            ln_midp::<T>(token, x)
        }

        #[inline(always)]
        pub(crate) fn exp_midp<T: FloatConvert>(token: T, x: FloatVec<T>) -> FloatVec<T> {
            exp2_midp::<T>(token, x * splat_f32::<T>(token, core::f32::consts::LOG2_E))
        }

        #[inline(always)]
        pub(crate) fn exp_midp_unchecked<T: FloatConvert>(token: T, x: FloatVec<T>) -> FloatVec<T> {
            exp2_midp_unchecked::<T>(token, x * splat_f32::<T>(token, core::f32::consts::LOG2_E))
        }

        #[inline(always)]
        pub(crate) fn exp_midp_precise<T: FloatConvert>(token: T, x: FloatVec<T>) -> FloatVec<T> {
            exp_midp::<T>(token, x)
        }

        #[inline(always)]
        pub(crate) fn sigmoid_midp<T: FloatConvert>(token: T, x: FloatVec<T>) -> FloatVec<T> {
            let one = splat_f32::<T>(token, 1.0);
            // Exact IEEE division, deliberately NOT `.recip()`. recip()'s
            // working tier now carries exact rails too, but its <= 4 ULP
            // slack breaks a pinned nicety: recip(1.0) lands ~1 ULP under
            // 1.0, so saturated sigmoid would return 0.9999999 instead of
            // exactly 1.0 (sigmoid_silu.rs pins the exact 0/1 rails).
            // Division costs ~nothing here — exp_midp dominates.
            one / (exp_midp::<T>(token, -x) + one)
        }

        #[inline(always)]
        pub(crate) fn silu_midp<T: FloatConvert>(token: T, x: FloatVec<T>) -> FloatVec<T> {
            x * sigmoid_midp::<T>(token, x)
        }

        #[inline(always)]
        pub(crate) fn log10_midp<T: FloatConvert>(token: T, x: FloatVec<T>) -> FloatVec<T> {
            log2_midp::<T>(token, x)
                * splat_f32::<T>(token, core::f32::consts::LN_2 / core::f32::consts::LN_10)
        }

        #[inline(always)]
        pub(crate) fn log10_midp_unchecked<T: FloatConvert>(
            token: T,
            x: FloatVec<T>,
        ) -> FloatVec<T> {
            log2_midp_unchecked::<T>(token, x)
                * splat_f32::<T>(token, core::f32::consts::LN_2 / core::f32::consts::LN_10)
        }

        #[inline(always)]
        pub(crate) fn log10_midp_precise<T: FloatConvert>(token: T, x: FloatVec<T>) -> FloatVec<T> {
            log10_midp::<T>(token, x)
        }

        #[inline(always)]
        pub(crate) fn pow_midp<T: FloatConvert>(token: T, x: FloatVec<T>, n: f32) -> FloatVec<T> {
            exp2_midp::<T>(token, log2_midp::<T>(token, x) * splat_f32::<T>(token, n))
        }

        #[inline(always)]
        pub(crate) fn pow_midp_unchecked<T: FloatConvert>(
            token: T,
            x: FloatVec<T>,
            n: f32,
        ) -> FloatVec<T> {
            exp2_midp_unchecked::<T>(
                token,
                log2_midp_unchecked::<T>(token, x) * splat_f32::<T>(token, n),
            )
        }

        #[inline(always)]
        pub(crate) fn pow_midp_precise<T: FloatConvert>(
            token: T,
            x: FloatVec<T>,
            n: f32,
        ) -> FloatVec<T> {
            pow_midp::<T>(token, x, n)
        }

        #[inline(always)]
        pub(crate) fn cbrt_lowp<T: FloatConvert>(token: T, x: FloatVec<T>) -> FloatVec<T> {
            const MAGIC: u32 = 0x2a50_8c2d;

            let sign_mask = splat_f32::<T>(token, -0.0);
            let sign = x & sign_mask;
            let abs_x = x.abs();

            let abs_arr = abs_x.to_array();
            let approx_arr: [f32; $lanes] =
                core::array::from_fn(|i| f32::from_bits((abs_arr[i].to_bits() / 3) + MAGIC));
            let mut y = FloatVec::from_array_t(token, approx_arr);

            // Halley iteration: y *= (y³ + 2x) / (2y³ + x)
            // y³ + 2x reaches 3x, so this overflows above f32::MAX / 3.
            // Triples bits of precision: ~5 → ~15
            let two = splat_f32::<T>(token, 2.0);
            let y3 = y * y * y;
            y *= (y3 + two * abs_x) / (two * y3 + abs_x);

            let result = y | sign;

            // Zero masking: cbrt(±0) = ±0 (bit hack gives garbage for zero)
            let is_zero = x.simd_eq(splat_f32::<T>(token, 0.0));
            FloatVec::blend(is_zero, x, result)
        }

        #[inline(always)]
        pub(crate) fn cbrt_midp<T: FloatConvert>(token: T, x: FloatVec<T>) -> FloatVec<T> {
            const MAGIC: u32 = 0x2a50_8c2d;

            let sign_mask = splat_f32::<T>(token, -0.0);
            let sign = x & sign_mask;
            let abs_x = x.abs();

            let abs_arr = abs_x.to_array();
            let approx_arr: [f32; $lanes] =
                core::array::from_fn(|i| f32::from_bits((abs_arr[i].to_bits() / 3) + MAGIC));
            let mut y = FloatVec::from_array_t(token, approx_arr);

            // 2 Halley iterations: y *= (y³ + 2x) / (2y³ + x)
            // y³ + 2x reaches 3x, so this overflows above f32::MAX / 3;
            // cbrt_midp_precise rescales those inputs.
            let two = splat_f32::<T>(token, 2.0);
            for _ in 0..2 {
                let y3 = y * y * y;
                y *= (y3 + two * abs_x) / (two * y3 + abs_x);
            }

            let result = y | sign;

            // Zero masking: cbrt(±0) = ±0 (bit hack gives garbage for zero)
            let is_zero = x.simd_eq(splat_f32::<T>(token, 0.0));
            FloatVec::blend(is_zero, x, result)
        }

        #[inline(always)]
        pub(crate) fn cbrt_midp_precise<T: FloatConvert>(token: T, x: FloatVec<T>) -> FloatVec<T> {
            let zero = splat_f32::<T>(token, 0.0);
            let one = splat_f32::<T>(token, 1.0);
            let abs_x = x.abs();
            // ±0 and ±inf are their own cube roots.
            let keep = x.simd_eq(zero) | abs_x.simd_eq(splat_f32::<T>(token, f32::INFINITY));
            let is_denorm = abs_x.simd_lt(splat_f32::<T>(token, 1.175_494_4e-38));
            // cbrt_midp overflows above f32::MAX / 3, so magnitudes from 1e36 up
            // run on x/8 and get doubled; denormals run on x * 2^24 and get
            // scaled by 2^-8. All four scalings are exact.
            let is_big = abs_x.simd_ge(splat_f32::<T>(token, 1.0e36));
            let pre = FloatVec::blend(
                is_denorm,
                splat_f32::<T>(token, 16_777_216.0),
                FloatVec::blend(is_big, splat_f32::<T>(token, 0.125), one),
            );
            let post = FloatVec::blend(
                is_denorm,
                splat_f32::<T>(token, 1.0 / 256.0),
                FloatVec::blend(is_big, splat_f32::<T>(token, 2.0), one),
            );

            let result = cbrt_midp::<T>(token, x * pre) * post;
            FloatVec::blend(keep, x, result)
        }
    };
}
pub(crate) mod width4 {
    legacy_bodies!(f32x4, i32x4, F32x4Backend, F32x4Convert, I32x4Backend, 4);
}
pub(crate) mod width8 {
    legacy_bodies!(f32x8, i32x8, F32x8Backend, F32x8Convert, I32x8Backend, 8);
}
#[cfg(feature = "w512")]
pub(crate) mod width16 {
    legacy_bodies!(
        f32x16,
        i32x16,
        F32x16Backend,
        F32x16Convert,
        I32x16Backend,
        16
    );
}

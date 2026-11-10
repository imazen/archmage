// Frozen from imazen/archmage 91a13739b73f46213ecfe7083c77297c531fc8ad; regenerate with freeze.py.
pub mod x4 {
    use magetypes::simd::backends::F32x4Convert;
    use magetypes::simd::generic::{f32x4, i32x4};

    /// Splat an i32 into i32x4 (disambiguates from f32 splat).
    #[inline(always)]
    fn splat_i32<T: F32x4Convert>(token: T, v: i32) -> i32x4<T> {
        i32x4::splat_t(token, v)
    }

    /// Splat an f32 into f32x4.
    #[inline(always)]
    fn splat_f32<T: F32x4Convert>(token: T, v: f32) -> f32x4<T> {
        f32x4::splat_t(token, v)
    }

    pub trait Frozen<T> {
        fn old_exp2_midp(self, token: T) -> Self;
        fn old_exp2_lowp(self, token: T) -> Self;
        fn old_exp2_midp_unchecked(self, token: T) -> Self;
        fn old_log2_midp(self, token: T) -> Self;
        fn old_log2_lowp(self, token: T) -> Self;
        fn old_log2_midp_unchecked(self, token: T) -> Self;
    }
    impl<T: F32x4Convert> Frozen<T> for f32x4<T> {
        #[inline(always)]
        fn old_exp2_midp(self, token: T) -> Self {
            let underflow_limit = splat_f32::<T>(token, -126.0);
            let overflow_limit = splat_f32::<T>(token, 128.0);

            // Clamp to prevent overflow in intermediate calculations
            let clamped = self.max(underflow_limit).min(overflow_limit);
            let result = clamped.old_exp2_midp_unchecked(token);

            // Handle edge cases: large negative → 0, large positive → inf
            // 2^128 > f32::MAX, so >= 128 must return inf
            let is_underflow = self.simd_lt(underflow_limit);
            let is_overflow = self.simd_ge(overflow_limit);
            let zero = splat_f32::<T>(token, 0.0);
            let inf = splat_f32::<T>(token, f32::INFINITY);
            let result = Self::blend(is_underflow, zero, result);
            Self::blend(is_overflow, inf, result)
        }
        #[inline(always)]
        fn old_exp2_lowp(self, token: T) -> Self {
            const C0: f32 = 1.0;
            const C1: f32 = core::f32::consts::LN_2;
            const C2: f32 = 0.240_226_5;
            const C3: f32 = 0.055_504_11;

            // floor(x) <= 127 keeps the bit trick (n+127)<<23 in range; at 127.99
            // the result stays below f32::MAX despite the polynomial's error.
            let x = self
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
        fn old_exp2_midp_unchecked(self, token: T) -> Self {
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
            let xi = self.round().min(splat_f32::<T>(token, 127.0));
            let xf = self - xi;

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
        fn old_log2_midp(self, token: T) -> Self {
            let result = self.old_log2_midp_unchecked(token);
            let zero = splat_f32::<T>(token, 0.0);
            let result = Self::blend(
                self.simd_eq(zero),
                splat_f32::<T>(token, f32::NEG_INFINITY),
                result,
            );
            let result = Self::blend(self.simd_lt(zero), splat_f32::<T>(token, f32::NAN), result);
            // +inf -> +inf; the polynomial otherwise returns inf's raw
            // unbiased exponent (128) for an infinite input.
            let inf = splat_f32::<T>(token, f32::INFINITY);
            Self::blend(self.simd_eq(inf), inf, result)
        }
        #[inline(always)]
        fn old_log2_lowp(self, token: T) -> Self {
            const P0: f32 = -1.850_383_3e-6;
            const P1: f32 = 1.428_716_1;
            const P2: f32 = 0.742_458_7;
            const Q0: f32 = 0.990_328_14;
            const Q1: f32 = 1.009_671_8;
            const Q2: f32 = 0.174_093_43;

            let x_bits = self.bitcast_to_i32();
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
        fn old_log2_midp_unchecked(self, token: T) -> Self {
            const SQRT2_OVER_2: u32 = 0x3f35_04f3;
            const ONE_BITS: u32 = 0x3f80_0000;
            const MANTISSA_MASK: i32 = 0x007f_ffff_u32 as i32;

            const C0: f32 = 2.885_39;
            const C1: f32 = 0.961_800_76;
            const C2: f32 = 0.576_974_45;
            const C3: f32 = 0.434_411_97;

            let x_bits = self.bitcast_to_i32();

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
    }
}
pub mod x8 {
    use magetypes::simd::backends::F32x8Convert;
    use magetypes::simd::generic::{f32x8, i32x8};

    /// Splat an i32 into i32x8 (disambiguates from f32 splat).
    #[inline(always)]
    fn splat_i32<T: F32x8Convert>(token: T, v: i32) -> i32x8<T> {
        i32x8::splat_t(token, v)
    }

    /// Splat an f32 into f32x8.
    #[inline(always)]
    fn splat_f32<T: F32x8Convert>(token: T, v: f32) -> f32x8<T> {
        f32x8::splat_t(token, v)
    }

    pub trait Frozen<T> {
        fn old_exp2_midp(self, token: T) -> Self;
        fn old_exp2_lowp(self, token: T) -> Self;
        fn old_exp2_midp_unchecked(self, token: T) -> Self;
        fn old_log2_midp(self, token: T) -> Self;
        fn old_log2_lowp(self, token: T) -> Self;
        fn old_log2_midp_unchecked(self, token: T) -> Self;
    }
    impl<T: F32x8Convert> Frozen<T> for f32x8<T> {
        #[inline(always)]
        fn old_exp2_midp(self, token: T) -> Self {
            let underflow_limit = splat_f32::<T>(token, -126.0);
            let overflow_limit = splat_f32::<T>(token, 128.0);

            // Clamp to prevent overflow in intermediate calculations
            let clamped = self.max(underflow_limit).min(overflow_limit);
            let result = clamped.old_exp2_midp_unchecked(token);

            // Handle edge cases: large negative → 0, large positive → inf
            // 2^128 > f32::MAX, so >= 128 must return inf
            let is_underflow = self.simd_lt(underflow_limit);
            let is_overflow = self.simd_ge(overflow_limit);
            let zero = splat_f32::<T>(token, 0.0);
            let inf = splat_f32::<T>(token, f32::INFINITY);
            let result = Self::blend(is_underflow, zero, result);
            Self::blend(is_overflow, inf, result)
        }
        #[inline(always)]
        fn old_exp2_lowp(self, token: T) -> Self {
            const C0: f32 = 1.0;
            const C1: f32 = core::f32::consts::LN_2;
            const C2: f32 = 0.240_226_5;
            const C3: f32 = 0.055_504_11;

            // floor(x) <= 127 keeps the bit trick (n+127)<<23 in range; at 127.99
            // the result stays below f32::MAX despite the polynomial's error.
            let x = self
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
        fn old_exp2_midp_unchecked(self, token: T) -> Self {
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
            let xi = self.round().min(splat_f32::<T>(token, 127.0));
            let xf = self - xi;

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
        fn old_log2_midp(self, token: T) -> Self {
            let result = self.old_log2_midp_unchecked(token);
            let zero = splat_f32::<T>(token, 0.0);
            let result = Self::blend(
                self.simd_eq(zero),
                splat_f32::<T>(token, f32::NEG_INFINITY),
                result,
            );
            let result = Self::blend(self.simd_lt(zero), splat_f32::<T>(token, f32::NAN), result);
            // +inf -> +inf; the polynomial otherwise returns inf's raw
            // unbiased exponent (128) for an infinite input.
            let inf = splat_f32::<T>(token, f32::INFINITY);
            Self::blend(self.simd_eq(inf), inf, result)
        }
        #[inline(always)]
        fn old_log2_lowp(self, token: T) -> Self {
            const P0: f32 = -1.850_383_3e-6;
            const P1: f32 = 1.428_716_1;
            const P2: f32 = 0.742_458_7;
            const Q0: f32 = 0.990_328_14;
            const Q1: f32 = 1.009_671_8;
            const Q2: f32 = 0.174_093_43;

            let x_bits = self.bitcast_to_i32();
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
        fn old_log2_midp_unchecked(self, token: T) -> Self {
            const SQRT2_OVER_2: u32 = 0x3f35_04f3;
            const ONE_BITS: u32 = 0x3f80_0000;
            const MANTISSA_MASK: i32 = 0x007f_ffff_u32 as i32;

            const C0: f32 = 2.885_39;
            const C1: f32 = 0.961_800_76;
            const C2: f32 = 0.576_974_45;
            const C3: f32 = 0.434_411_97;

            let x_bits = self.bitcast_to_i32();

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
    }
}
pub mod x16 {
    use magetypes::simd::backends::F32x16Convert;
    use magetypes::simd::generic::{f32x16, i32x16};

    /// Splat an i32 into i32x16 (disambiguates from f32 splat).
    #[inline(always)]
    fn splat_i32<T: F32x16Convert>(token: T, v: i32) -> i32x16<T> {
        i32x16::splat_t(token, v)
    }

    /// Splat an f32 into f32x16.
    #[inline(always)]
    fn splat_f32<T: F32x16Convert>(token: T, v: f32) -> f32x16<T> {
        f32x16::splat_t(token, v)
    }

    pub trait Frozen<T> {
        fn old_exp2_midp(self, token: T) -> Self;
        fn old_exp2_lowp(self, token: T) -> Self;
        fn old_exp2_midp_unchecked(self, token: T) -> Self;
        fn old_log2_midp(self, token: T) -> Self;
        fn old_log2_lowp(self, token: T) -> Self;
        fn old_log2_midp_unchecked(self, token: T) -> Self;
    }
    impl<T: F32x16Convert> Frozen<T> for f32x16<T> {
        #[inline(always)]
        fn old_exp2_midp(self, token: T) -> Self {
            let underflow_limit = splat_f32::<T>(token, -126.0);
            let overflow_limit = splat_f32::<T>(token, 128.0);

            // Clamp to prevent overflow in intermediate calculations
            let clamped = self.max(underflow_limit).min(overflow_limit);
            let result = clamped.old_exp2_midp_unchecked(token);

            // Handle edge cases: large negative → 0, large positive → inf
            // 2^128 > f32::MAX, so >= 128 must return inf
            let is_underflow = self.simd_lt(underflow_limit);
            let is_overflow = self.simd_ge(overflow_limit);
            let zero = splat_f32::<T>(token, 0.0);
            let inf = splat_f32::<T>(token, f32::INFINITY);
            let result = Self::blend(is_underflow, zero, result);
            Self::blend(is_overflow, inf, result)
        }
        #[inline(always)]
        fn old_exp2_lowp(self, token: T) -> Self {
            const C0: f32 = 1.0;
            const C1: f32 = core::f32::consts::LN_2;
            const C2: f32 = 0.240_226_5;
            const C3: f32 = 0.055_504_11;

            // floor(x) <= 127 keeps the bit trick (n+127)<<23 in range; at 127.99
            // the result stays below f32::MAX despite the polynomial's error.
            let x = self
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
        fn old_exp2_midp_unchecked(self, token: T) -> Self {
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
            let xi = self.round().min(splat_f32::<T>(token, 127.0));
            let xf = self - xi;

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
        fn old_log2_midp(self, token: T) -> Self {
            let result = self.old_log2_midp_unchecked(token);
            let zero = splat_f32::<T>(token, 0.0);
            let result = Self::blend(
                self.simd_eq(zero),
                splat_f32::<T>(token, f32::NEG_INFINITY),
                result,
            );
            let result = Self::blend(self.simd_lt(zero), splat_f32::<T>(token, f32::NAN), result);
            // +inf -> +inf; the polynomial otherwise returns inf's raw
            // unbiased exponent (128) for an infinite input.
            let inf = splat_f32::<T>(token, f32::INFINITY);
            Self::blend(self.simd_eq(inf), inf, result)
        }
        #[inline(always)]
        fn old_log2_lowp(self, token: T) -> Self {
            const P0: f32 = -1.850_383_3e-6;
            const P1: f32 = 1.428_716_1;
            const P2: f32 = 0.742_458_7;
            const Q0: f32 = 0.990_328_14;
            const Q1: f32 = 1.009_671_8;
            const Q2: f32 = 0.174_093_43;

            let x_bits = self.bitcast_to_i32();
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
        fn old_log2_midp_unchecked(self, token: T) -> Self {
            const SQRT2_OVER_2: u32 = 0x3f35_04f3;
            const ONE_BITS: u32 = 0x3f80_0000;
            const MANTISSA_MASK: i32 = 0x007f_ffff_u32 as i32;

            const C0: f32 = 2.885_39;
            const C1: f32 = 0.961_800_76;
            const C2: f32 = 0.576_974_45;
            const C3: f32 = 0.434_411_97;

            let x_bits = self.bitcast_to_i32();

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
    }
}

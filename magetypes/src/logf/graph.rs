//! Const-configured log2 graphs, instantiated per width and arithmetic provider.
use super::{LogfError, LogfErrorKind, LogfOptions};


#[cfg(debug_assertions)]
const KERNEL_FLAGS: u32 =
    LogfOptions::LOWP | LogfOptions::ASSUME | LogfOptions::SPECIALS | LogfOptions::LEGACY;

#[cfg(debug_assertions)]
macro_rules! shared_kernel {
    ($vector:ident, $t:ty, $options:ident, $token:ident, $input:ident) => {
        match $options & KERNEL_FLAGS {
            0 => $vector::<$t, 0>($token, $input),
            1 => $vector::<$t, 1>($token, $input),
            8 => $vector::<$t, 8>($token, $input),
            9 => $vector::<$t, 9>($token, $input),
            32 => $vector::<$t, 32>($token, $input),
            33 => $vector::<$t, 33>($token, $input),
            40 => $vector::<$t, 40>($token, $input),
            41 => $vector::<$t, 41>($token, $input),
            _ => unreachable!("private arithmetic flags are validated at the boundary"),
        }
    };
}

macro_rules! graph {
    ($vector:ident, $profile:ident, $const_entry:ident, $float:ident, $int:ident, $bound:ident, $lanes:literal) => {
        #[cfg_attr(debug_assertions, inline(never))]
        #[cfg_attr(not(debug_assertions), inline(always))]
        fn $vector<T: crate::simd::backends::$bound, const OPTIONS: u32>(
            token: T,
            input: crate::simd::generic::$float<T>,
        ) -> Result<crate::simd::generic::$float<T>, LogfError> {
            use crate::simd::generic::{$float as V, $int as I};
            let sp = |x| V::splat_t(token, x);
            let ip = |x| I::splat_t(token, x);
            let reject = |mask: I<T>, kind| {
                let lane_mask = mask
                    .to_array()
                    .iter()
                    .enumerate()
                    .fold(0_u16, |m, (i, &v)| m | (u16::from(v != 0) << i));
                LogfError { kind, lane_mask }
            };
            let bits = input.bitcast_to_i32();
            let assume = OPTIONS & LogfOptions::ASSUME != 0;
            if !assume && OPTIONS & LogfOptions::SPECIALS == 0 {
                // Positive domain: x <= 0, subnormal, NaN, infinite are invalid.
                // bits <= 0 covers -0..=0 and every negative; exponent 0 covers
                // subnormals; exponent 0x7f80 covers NaN/inf.
                let exp_bits = bits & ip(0x7f80_0000);
                let invalid = bits.simd_le(ip(0))
                    | exp_bits.simd_eq(ip(0))
                    | exp_bits.simd_eq(ip(0x7f80_0000));
                if invalid.any_true() {
                    return Err(reject(invalid, LogfErrorKind::InvalidInput));
                }
            }
            let madd = |a: V<T>, b: V<T>, c: V<T>| -> V<T> {
                if OPTIONS & LogfOptions::BACKEND != 0 {
                    a.mul_add_relaxed(b, c)
                } else {
                    a.mul_add_fused(b, c)
                }
            };
            let result = if OPTIONS & LogfOptions::LOWP != 0 {
                // Rational P2/Q2 — beats the deg-2 atanh alternative on
                // measured speed (~20% slower on Zen5 and NEON) while the
                // atanh's accuracy edge is unneeded at this tier.
                let offset = ip(0x3f2a_aaab_u32 as i32);
                let exp_bits_v = bits - offset;
                let exp_shifted = exp_bits_v.shr_arithmetic_const::<23>();
                let mantissa_bits = bits - exp_shifted.shl_const::<23>();
                let m = mantissa_bits.bitcast_to_f32() - sp(1.0);
                let yp = madd(sp(0.742_458_7), m, sp(1.428_716_1));
                let yp = madd(yp, m, sp(-1.850_383_3e-6));
                let yq = madd(sp(0.174_093_43), m, sp(1.009_671_8));
                let yq = madd(yq, m, sp(0.990_328_14));
                yp / yq + exp_shifted.to_f32()
            } else {
                // Same coefficients and arithmetic order as log2_midp.
                let adjusted = bits + ip((0x3f80_0000_u32 - 0x3f35_04f3) as i32);
                let n = (adjusted.shr_arithmetic_const::<23>() - ip(127)).to_f32();
                let a = ((adjusted & ip(0x007f_ffff)) + ip(0x3f35_04f3)).bitcast_to_f32();
                let y = (a - sp(1.0)) / (a + sp(1.0));
                let y2 = y * y;
                let (c3, c2, c1, c0) = if OPTIONS & LogfOptions::LEGACY != 0 {
                    (0.434_411_97, 0.576_974_45, 0.961_800_76, 2.885_39)
                } else {
                    (
                        0.430_468_597_034_945_7,
                        0.576_767_602_130_619_1,
                        0.961_798_223_457_557_8,
                        2.885_390_081_777_926_8,
                    )
                };
                let p = madd(sp(c3), y2, sp(c2));
                let p = madd(p, y2, sp(c1));
                let p = madd(p, y2, sp(c0));
                madd(p, y, n)
            };
            let result = if OPTIONS & LogfOptions::SPECIALS != 0 && !assume {
                // Specials domain: 0 -> -inf, <0 -> NaN, NaN -> NaN,
                // +inf -> +inf (the raw exponent bias the polynomial
                // produces for an infinite input is replaced).
                // Ordered >= accepts both zero signs and positive inputs.
                // Selecting NaN on its false side also covers every NaN:
                // one comparison replaces separate negative/NaN predicates.
                let zero = input.simd_eq(sp(0.0));
                let nonnegative = input.simd_ge(sp(0.0));
                let pos_inf = input.simd_eq(sp(f32::INFINITY));
                let result = V::blend(zero, sp(f32::NEG_INFINITY), result);
                let result = V::blend(nonnegative, result, sp(f32::NAN));
                V::blend(pos_inf, sp(f32::INFINITY), result)
            } else {
                result
            };
            Ok(result)
        }

        #[inline(always)]
        pub(super) fn $profile<T: crate::simd::backends::$bound, const OPTIONS: u32>(
            token: T,
            input: crate::simd::generic::$float<T>,
        ) -> Result<crate::simd::generic::$float<T>, LogfError> {
            #[cfg(debug_assertions)]
            return shared_kernel!($vector, T, OPTIONS, token, input);
            #[cfg(not(debug_assertions))]
            $vector::<T, OPTIONS>(token, input)
        }

        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
        pub(super) fn $const_entry<const OPTIONS: u32>(
            token: Token,
            input: [f32; $lanes],
        ) -> Result<[f32; $lanes], LogfError> {
            $profile::<Token, OPTIONS>(
                token,
                crate::simd::generic::$float::from_array_t(token, input),
            )
            .map(|result| result.to_array())
        }
    };
}

graph!(vector4, profile4, kernel4_with, f32x4, i32x4, F32x4Convert, 4);
graph!(vector8, profile8, kernel8_with, f32x8, i32x8, F32x8Convert, 8);
#[cfg(feature = "w512")]
graph!(vector16, profile16, kernel16_with, f32x16, i32x16, F32x16Convert, 16);

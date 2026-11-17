//! Const-configured log2 graphs, instantiated per width and arithmetic provider.
#![allow(clippy::excessive_precision)] // Preserve the fitted f32 coefficient spellings.
use super::{LogfError, LogfErrorKind, LogfOptions};

macro_rules! graph {
    ($vector:ident, $eval_op:ident, $profile:ident, $const_entry:ident, $float:ident, $int:ident, $bound:ident, $lanes:literal) => {
        #[cfg_attr(debug_assertions, inline(never))]
        #[cfg_attr(not(debug_assertions), inline(always))]
        fn $vector<T: magetypes::simd::backends::$bound, const FIXED_OPTIONS: u32>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
            options: u32,
        ) -> Result<magetypes::simd::generic::$float<T>, LogfError> {
            // Debug shares one graph per token/width without discarding any flag.
            // Optimized builds keep the integer const through the arithmetic.
            #[cfg(not(debug_assertions))]
            let options = {
                let _ = options;
                FIXED_OPTIONS
            };
            use magetypes::simd::generic::{$float as V, $int as I};
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
            let assume = options & LogfOptions::ASSUME != 0;
            if !assume && options & LogfOptions::SPECIALS == 0 {
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
                if options & LogfOptions::BACKEND != 0 {
                    a.mul_add_relaxed(b, c)
                } else {
                    a.mul_add_fused(b, c)
                }
            };
            let result = if options & LogfOptions::LOWP != 0 {
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
                let (c3, c2, c1, c0) = if options & LogfOptions::LEGACY != 0 {
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
            let result = if options & LogfOptions::SPECIALS != 0 {
                // Specials domain: 0 -> -inf, <0 -> NaN, NaN -> NaN,
                // +inf -> +inf (the raw exponent bias the polynomial
                // produces for an infinite input is replaced).
                // Float-side masks, same op profile as the generated method.
                // Ordered >= rejects negatives and NaNs with one mask.
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

        #[cfg_attr(debug_assertions, inline(never))]
        #[cfg_attr(not(debug_assertions), inline(always))]
        fn $eval_op<
            T: magetypes::simd::backends::$bound,
            const FIXED_OP: u8,
            const FIXED_OPTIONS: u32,
        >(
            token: T,
            input: magetypes::simd::generic::$float<T>,
            op: u8,
            options: u32,
        ) -> Result<magetypes::simd::generic::$float<T>, LogfError> {
            #[cfg(not(debug_assertions))]
            let (op, options) = {
                let _ = (op, options);
                (FIXED_OP, FIXED_OPTIONS)
            };
            use magetypes::simd::generic::$float as V;
            if op == 3 {
                let sp = |x| V::splat_t(token, x);
                let one = sp(1.0);
                let two = sp(2.0);
                let u = input + one;
                let finite_pos = u.simd_gt(sp(0.0)) & u.simd_lt(sp(f32::INFINITY));
                let c_raw = V::blend(
                    u.simd_ge(two),
                    (one - (u - input)) / u,
                    (input - (u - one)) / u,
                );
                let c = V::blend(finite_pos, c_raw, sp(0.0));
                let log2_u = $vector::<T, FIXED_OPTIONS>(token, u, options)?;
                let ln2 = sp(core::f32::consts::LN_2);
                let res = if options & LogfOptions::BACKEND != 0 {
                    log2_u.mul_add_relaxed(ln2, c)
                } else {
                    log2_u.mul_add_fused(ln2, c)
                };
                Ok(V::blend(input.simd_eq(sp(0.0)), input, res))
            } else {
                let log2_v = $vector::<T, FIXED_OPTIONS>(token, input, options)?;
                Ok(match op {
                    1 => log2_v * V::splat_t(token, core::f32::consts::LN_2),
                    2 => log2_v * V::splat_t(token, core::f32::consts::LOG10_2),
                    _ => log2_v,
                })
            }
        }

        #[inline(always)]
        pub(super) fn $profile<
            T: magetypes::simd::backends::$bound,
            const OP: u8,
            const OPTIONS: u32,
        >(
            token: T,
            input: magetypes::simd::generic::$float<T>,
        ) -> Result<magetypes::simd::generic::$float<T>, LogfError> {
            #[cfg(debug_assertions)]
            return $eval_op::<T, 0, 0>(token, input, OP, OPTIONS);
            #[cfg(not(debug_assertions))]
            $eval_op::<T, OP, OPTIONS>(token, input, OP, OPTIONS)
        }

        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
        pub(super) fn $const_entry<const OP: u8, const OPTIONS: u32>(
            token: Token,
            input: [f32; $lanes],
        ) -> Result<[f32; $lanes], LogfError> {
            $profile::<Token, OP, OPTIONS>(
                token,
                magetypes::simd::generic::$float::from_array_t(token, input),
            )
            .map(|result| result.to_array())
        }
    };
}

graph!(
    vector4,
    eval_op4,
    profile4,
    kernel4_with,
    f32x4,
    i32x4,
    F32x4Convert,
    4
);
graph!(
    vector8,
    eval_op8,
    profile8,
    kernel8_with,
    f32x8,
    i32x8,
    F32x8Convert,
    8
);
#[cfg(feature = "w512")]
graph!(
    vector16,
    eval_op16,
    profile16,
    kernel16_with,
    f32x16,
    i32x16,
    F32x16Convert,
    16
);

/// Whole-slice kernel (`OP`: `0=log2`, `1=ln`, `2=log10`, `3=log1p`): one
/// feature region around the loop, the 8-lane graph per chunk, and one padded
/// vector for the remainder. The pad value 1.0 is valid for both `log2(1.0)`
/// and `log1p(1.0) = ln(2)`, so only real lanes can be rejected.
#[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
pub(super) fn slice8_with<const OP: u8, const OPTIONS: u32>(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, LogfError)> {
    use magetypes::simd::generic::f32x8 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        profile8::<Token, OP, OPTIONS>(token, V::load_t(token, s))
            .map_err(|e| (i * 8, e))?
            .store(d);
    }
    if !tail.is_empty() {
        profile8::<Token, OP, OPTIONS>(token, V::load_padded_t(token, tail, 1.0))
            .map_err(|e| (chunks.len() * 8, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

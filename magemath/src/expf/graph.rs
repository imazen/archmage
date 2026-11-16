//! Const-configured exp2 graphs, instantiated per width and arithmetic provider.
#![allow(clippy::excessive_precision)] // Preserve the fitted f32 coefficient spellings.
use super::{ExpfError, ExpfErrorKind, ExpfOptions};

// Private flags (not accepted public encodings): infallible methods use them.
pub(super) const CLAMP: u32 = 256;

macro_rules! graph {
    ($vector:ident, $profile:ident, $const_entry:ident, $float:ident, $int:ident, $bound:ident, $lanes:literal) => {
        #[cfg_attr(debug_assertions, inline(never))]
        #[cfg_attr(not(debug_assertions), inline(always))]
        fn $vector<T: magetypes::simd::backends::$bound, const FIXED_OPTIONS: u32>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
            options: u32,
        ) -> Result<magetypes::simd::generic::$float<T>, ExpfError> {
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
                ExpfError { kind, lane_mask }
            };
            let bits = input.bitcast_to_i32();
            let abs = bits & ip(0x7fff_ffff_u32 as i32);
            if options & ExpfOptions::ASSUME == 0 && options & ExpfOptions::FULL == 0 {
                // Bounded domain: NaN and infinities are invalid inputs.
                let invalid = abs.simd_ge(ip(0x7f80_0000));
                if invalid.any_true() {
                    return Err(reject(invalid, ExpfErrorKind::InvalidInput));
                }
            }
            let madd = |a: V<T>, b: V<T>, c: V<T>| -> V<T> {
                if options & ExpfOptions::BACKEND != 0 {
                    a.mul_add_relaxed(b, c)
                } else {
                    a.mul_add_fused(b, c)
                }
            };
            let result = if options & ExpfOptions::LOWP != 0 {
                // Same coefficients and arithmetic order as exp2_lowp.
                let x = if options & CLAMP == 0 {
                    input.max(sp(-126.0)).min(sp(127.99))
                } else {
                    input
                };
                let xi = x.floor();
                let xf = x - xi;
                let (c3, c2) = if options & ExpfOptions::LEGACY != 0 {
                    (0.055_504_11, 0.240_226_5)
                } else {
                    (0.071_916_997_023_492_97, 0.234_336_384_644_832_92)
                };
                let p = madd(sp(c3), xf, sp(c2));
                let p = madd(p, xf, sp(core::f32::consts::LN_2));
                let p = madd(p, xf, sp(1.0));
                let integer = if core::any::TypeId::of::<T>()
                    == core::any::TypeId::of::<archmage::ScalarToken>()
                {
                    xi.to_i32()
                } else {
                    xi.to_i32_round()
                };
                p * (integer + ip(127)).shl_const::<23>().bitcast_to_f32()
            } else {
                // Same coefficients and arithmetic order as exp2_midp:
                // round-to-nearest split, xi clamped to 127.
                let underflow_limit = sp(-126.0);
                let overflow_limit = sp(128.0);
                let reduced = if options & CLAMP == 0 {
                    input.max(underflow_limit).min(overflow_limit)
                } else {
                    input
                };
                let xi = reduced.round().min(sp(127.0));
                let xf = reduced - xi;
                let (c6, c5, c4, c3, c2) = if options & ExpfOptions::LEGACY != 0 {
                    (
                        0.000_154_47,
                        0.001_333_37,
                        0.009_618_055,
                        0.055_504_545,
                        0.240_226_46,
                    )
                } else {
                    (
                        0.000_152_527_896_498_79,
                        0.001_338_604_727_826_21,
                        0.009_618_715_779_149_75,
                        0.055_503_695_542_993_11,
                        0.240_226_463_934_714_0,
                    )
                };
                let p = madd(sp(c6), xf, sp(c5));
                let p = madd(p, xf, sp(c4));
                let p = madd(p, xf, sp(c3));
                let p = madd(p, xf, sp(c2));
                let p = madd(p, xf, sp(core::f32::consts::LN_2));
                let p = madd(p, xf, sp(1.0));
                let integer = if core::any::TypeId::of::<T>()
                    == core::any::TypeId::of::<archmage::ScalarToken>()
                {
                    xi.to_i32()
                } else {
                    xi.to_i32_round()
                };
                let scale = (integer + ip(127)).shl_const::<23>().bitcast_to_f32();
                let result = p * scale;
                if options & CLAMP == 0 {
                    let is_underflow = input.simd_lt(underflow_limit);
                    let is_overflow = input.simd_ge(overflow_limit);
                    let result = V::blend(is_underflow, sp(0.0), result);
                    V::blend(is_overflow, sp(f32::INFINITY), result)
                } else {
                    result
                }
            };
            let result = if options & ExpfOptions::FULL != 0 {
                // Full domain: non-finite inputs produce specials, not errors.
                let nan = input.simd_ne(input);
                let pos_inf = input.simd_eq(sp(f32::INFINITY));
                let neg_inf = input.simd_eq(sp(f32::NEG_INFINITY));
                let result = V::blend(nan, sp(f32::NAN), result);
                let result = V::blend(pos_inf, sp(f32::INFINITY), result);
                V::blend(neg_inf, sp(0.0), result)
            } else {
                result
            };
            Ok(result)
        }

        #[inline(always)]
        pub(super) fn $profile<T: magetypes::simd::backends::$bound, const OPTIONS: u32>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
        ) -> Result<magetypes::simd::generic::$float<T>, ExpfError> {
            #[cfg(debug_assertions)]
            return $vector::<T, 0>(token, input, OPTIONS);
            #[cfg(not(debug_assertions))]
            $vector::<T, OPTIONS>(token, input, OPTIONS)
        }

        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
        pub(super) fn $const_entry<const OPTIONS: u32>(
            token: Token,
            input: [f32; $lanes],
        ) -> Result<[f32; $lanes], ExpfError> {
            $profile::<Token, OPTIONS>(
                token,
                magetypes::simd::generic::$float::from_array_t(token, input),
            )
            .map(|result| result.to_array())
        }
    };
}

graph!(
    vector4,
    profile4,
    kernel4_with,
    f32x4,
    i32x4,
    F32x4Convert,
    4
);
graph!(
    vector8,
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
    profile16,
    kernel16_with,
    f32x16,
    i32x16,
    F32x16Convert,
    16
);

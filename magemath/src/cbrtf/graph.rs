//! Const-configured cube-root (`cbrtf`) graphs.
use super::{CbrtfError, CbrtfErrorKind, CbrtfOptions};

const MAGIC: i32 = 0x2a50_8c2d;
const MIN_NORMAL_BITS: i32 = 0x0080_0000;
const INF_BITS: i32 = 0x7f80_0000;
const MAX_BOUNDED_BITS: i32 = (f32::MAX / 3.0).to_bits() as i32;
const BIG_SCALE_THRESHOLD_BITS: i32 = 1.0e36_f32.to_bits() as i32;

macro_rules! graph {
    ($vector:ident, $profile:ident, $const_entry:ident, $float:ident, $int:ident, $bound:ident, $lanes:literal) => {
        #[cfg_attr(debug_assertions, inline(never))]
        #[cfg_attr(not(debug_assertions), inline(always))]
        fn $vector<T: magetypes::simd::backends::$bound, const FIXED_OPTIONS: u32>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
            options: u32,
        ) -> Result<magetypes::simd::generic::$float<T>, CbrtfError> {
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
                CbrtfError { kind, lane_mask }
            };

            let bits = input.bitcast_to_i32();
            let abs_bits = bits & ip(0x7fff_ffff_u32 as i32);

            if options & CbrtfOptions::ASSUME == 0 && options & CbrtfOptions::FULL == 0 {
                // Bounded domain: ±0.0 or normal finite magnitudes < f32::MAX / 3.
                let is_denorm = abs_bits.simd_gt(ip(0)) & abs_bits.simd_lt(ip(MIN_NORMAL_BITS));
                let is_too_large_or_nonfinite = abs_bits.simd_ge(ip(MAX_BOUNDED_BITS));
                let invalid = is_denorm | is_too_large_or_nonfinite;
                if invalid.any_true() {
                    return Err(reject(invalid, CbrtfErrorKind::InvalidInput));
                }
            }

            let (work_input, post_scale) = if options & CbrtfOptions::FULL != 0 {
                let one = sp(1.0);
                let is_denorm = (abs_bits.simd_gt(ip(0)) & abs_bits.simd_lt(ip(MIN_NORMAL_BITS)))
                    .bitcast_to_f32();
                let is_big = abs_bits
                    .simd_ge(ip(BIG_SCALE_THRESHOLD_BITS))
                    .bitcast_to_f32();
                let pre = V::blend(
                    is_denorm,
                    sp(16_777_216.0),
                    V::blend(is_big, sp(0.125), one),
                );
                let post = V::blend(is_denorm, sp(1.0 / 256.0), V::blend(is_big, sp(2.0), one));
                (input * pre, Some(post))
            } else {
                (input, None)
            };

            let sign = work_input & sp(-0.0);
            let abs_x = work_input.abs();

            // Kahan bit-hack initial seed (0.9.31 scalar `/ 3` baseline):
            let mut arr = abs_x.to_array();
            for elem in &mut arr {
                *elem = f32::from_bits((elem.to_bits() / 3) + MAGIC as u32);
            }
            let mut y = V::from_array_t(token, arr);

            // Halley iteration: y *= (y^3 + 2x) / (2y^3 + x)
            // Triples precision per step: ~5 -> ~15 bits (LOWP) -> ~45 bits (MIDP).
            let two = sp(2.0);
            let steps = if options & CbrtfOptions::LOWP != 0 {
                1
            } else {
                2
            };
            for _ in 0..steps {
                let y3 = y * y * y;
                if options & CbrtfOptions::BACKEND != 0 {
                    y *= two.mul_add_relaxed(abs_x, y3) / two.mul_add_relaxed(y3, abs_x);
                } else {
                    y *= (y3 + two * abs_x) / (two * y3 + abs_x);
                }
            }

            let mut result = y | sign;
            if let Some(post) = post_scale {
                result *= post;
                // Preserve ±0.0 and ±inf verbatim, and map NaN inputs to quiet NaN.
                let keep =
                    (abs_bits.simd_eq(ip(0)) | abs_bits.simd_eq(ip(INF_BITS))).bitcast_to_f32();
                let is_nan = abs_bits.simd_gt(ip(INF_BITS)).bitcast_to_f32();
                result = V::blend(keep, input, result);
                result = V::blend(is_nan, sp(f32::NAN), result);
            } else {
                // Bounded domain: cbrt(±0.0) = ±0.0 (preserves sign of zero).
                let is_zero = abs_bits.simd_eq(ip(0)).bitcast_to_f32();
                result = V::blend(is_zero, input, result);
            }

            Ok(result)
        }

        #[inline(always)]
        pub(super) fn $profile<T: magetypes::simd::backends::$bound, const OPTIONS: u32>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
        ) -> Result<magetypes::simd::generic::$float<T>, CbrtfError> {
            #[cfg(debug_assertions)]
            return $vector::<T, 0>(token, input, OPTIONS);
            #[cfg(not(debug_assertions))]
            $vector::<T, OPTIONS>(token, input, OPTIONS)
        }

        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
        pub(super) fn $const_entry<const OPTIONS: u32>(
            token: Token,
            input: [f32; $lanes],
        ) -> Result<[f32; $lanes], CbrtfError> {
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

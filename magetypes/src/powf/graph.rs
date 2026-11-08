//! One expression graph, instantiated per logical width and arithmetic backend.
use super::{PowfError, PowfErrorKind};

// Arithmetic mode is chosen once at the operation boundary. Mode 0 is used
// only with a token whose native madd fuses, or a verified relaxed-WASM probe.
pub(super) const NATIVE: usize = 0;
pub(super) const SOFTWARE: usize = 1;
#[cfg(target_arch = "wasm32")]
pub(super) const WASM_SOFTWARE: usize = 3;
#[cfg(all(feature = "std", not(miri)))]
pub(super) const STD: usize = 2;

macro_rules! graph {
    ($core:ident, $vector:ident, $profile:ident, $entry:ident, $const_entry:ident, $float:ident, $int:ident, $bound:ident, $lanes:literal) => {
        #[inline(always)]
        fn $core<
            T: crate::simd::backends::$bound,
            const ARITHMETIC: usize,
            const ZERO: bool,
            const GAMMA: bool,
        >(
            token: T,
            bases: [f32; $lanes],
            exponent: f32,
        ) -> Result<[f32; $lanes], PowfError> {
            $vector::<T, ARITHMETIC, ZERO, GAMMA>(
                token,
                crate::simd::generic::$float::from_array_t(token, bases),
                exponent,
            )
            .map(|result| result.to_array())
        }

        #[inline(always)]
        fn $vector<
            T: crate::simd::backends::$bound,
            const ARITHMETIC: usize,
            const ZERO: bool,
            const GAMMA: bool,
        >(
            token: T,
            input: crate::simd::generic::$float<T>,
            exponent: f32,
        ) -> Result<crate::simd::generic::$float<T>, PowfError> {
            use crate::simd::generic::{$float as V, $int as I};
            let sp = |x| V::splat_t(token, x);
            let ip = |x| I::splat_t(token, x);
            let bits = input.bitcast_to_i32();
            let zero = bits.simd_eq(ip(0));
            let invalid = if GAMMA {
                let within = input.simd_ge(sp(1.0 / 65_536.0)) & input.simd_le(sp(65_536.0));
                (within.bitcast_to_i32() | zero) ^ ip(-1)
            } else {
                let exp_bits = bits & ip(0x7f80_0000);
                let invalid = bits.simd_lt(ip(0)) | exp_bits.simd_eq(ip(0x7f80_0000));
                let tiny = exp_bits.simd_eq(ip(0));
                invalid
                    | if ZERO && exponent >= 0.0 {
                        tiny & (zero ^ ip(-1))
                    } else {
                        tiny
                    }
            };
            let reject = |mask: I<T>, kind| {
                let lane_mask = mask
                    .to_array()
                    .iter()
                    .enumerate()
                    .fold(0_u16, |m, (i, &v)| m | (u16::from(v != 0) << i));
                PowfError { kind, lane_mask }
            };
            if invalid.any_true() {
                return Err(reject(invalid, PowfErrorKind::InvalidBase));
            }
            let x = if ZERO {
                V::blend(zero.bitcast_to_f32(), sp(1.0), input)
            } else {
                input
            };
            let madd = |a: V<T>, b: V<T>, c: V<T>| -> V<T> {
                if ARITHMETIC == NATIVE {
                    a.mul_add_relaxed(b, c)
                } else {
                    let (a, b, c) = (a.to_array(), b.to_array(), c.to_array());
                    #[cfg(target_arch = "wasm32")]
                    if ARITHMETIC == WASM_SOFTWARE {
                        let any: &dyn core::any::Any = &token;
                        let proof = any
                            .downcast_ref::<archmage::Wasm128Token>()
                            .expect("private WASM mode requires its matching proof");
                        return V::from_array_t(
                            token,
                            super::wasm::software_lanes(*proof, a, b, c),
                        );
                    }
                    #[cfg(all(feature = "std", not(miri)))]
                    if ARITHMETIC == STD {
                        return V::from_array_t(
                            token,
                            core::array::from_fn(|i| a[i].mul_add(b[i], c[i])),
                        );
                    }
                    V::from_array_t(token, crate::soft_fma::fmaf_soft_lanes(a, b, c))
                }
            };
            // Coefficients and arithmetic order from the existing midp log2
            // and exp2 graph; every former relaxed madd rounds exactly once.
            let adjusted = x.bitcast_to_i32() + ip((0x3f80_0000_u32 - 0x3f35_04f3) as i32);
            let n = (adjusted.shr_arithmetic_const::<23>() - ip(127)).to_f32();
            let a = ((adjusted & ip(0x007f_ffff)) + ip(0x3f35_04f3)).bitcast_to_f32();
            let y = (a - sp(1.0)) / (a + sp(1.0));
            let y2 = y * y;
            let p = madd(sp(0.434_411_97), y2, sp(0.576_974_45));
            let p = madd(p, y2, sp(0.961_800_76));
            let p = madd(p, y2, sp(2.885_39));
            let t = madd(p, y, n) * sp(exponent);
            // Gamma: |integer exponent| <= 16, |y| < 1/4, and the
            // positive Horner polynomial is < 2.96. Thus |log2| < 17,
            // and |n*log2| < 68 for n in [0,4]. xtask/powf_vectors.py
            // checks an independent rational/RNE interval certificate.
            if !GAMMA {
                let outside = t.simd_lt(sp(-126.0)) | t.simd_gt(sp(127.5));
                if outside.bitcast_to_i32().any_true() {
                    return Err(reject(
                        outside.bitcast_to_i32(),
                        PowfErrorKind::OutsideApproximationRange,
                    ));
                }
            }
            let xi = t.round().min(sp(127.0));
            let xf = t - xi;
            let p = madd(sp(0.000_154_47), xf, sp(0.001_333_37));
            let p = madd(p, xf, sp(0.009_618_055));
            let p = madd(p, xf, sp(0.055_504_545));
            let p = madd(p, xf, sp(0.240_226_46));
            let p = madd(p, xf, sp(core::f32::consts::LN_2));
            let p = madd(p, xf, sp(1.0));
            let scale = (xi.to_i32_round() + ip(127))
                .shl_const::<23>()
                .bitcast_to_f32();
            let result = p * scale;
            let result = if ZERO {
                V::blend(
                    zero.bitcast_to_f32(),
                    sp(if exponent == 0.0 { 1.0 } else { 0.0 }),
                    result,
                )
            } else {
                result
            };
            Ok(result)
        }

        // These flags remain const parameters through the generic caller and
        // native feature entry. No runtime domain enum enters this path.
        #[inline(always)]
        pub(super) fn $profile<
            T: crate::simd::backends::$bound,
            const ARITHMETIC: usize,
            const OPTIONS: u32,
        >(
            token: T,
            input: crate::simd::generic::$float<T>,
            exponent: f32,
        ) -> Result<crate::simd::generic::$float<T>, PowfError> {
            match OPTIONS & 3 {
                0 => $vector::<T, ARITHMETIC, false, false>(token, input, exponent),
                1 => $vector::<T, ARITHMETIC, true, false>(token, input, exponent),
                2 => $vector::<T, ARITHMETIC, true, true>(token, input, exponent),
                _ => unreachable!("validated const domain"),
            }
        }

        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
        pub(super) fn $const_entry<const ARITHMETIC: usize, const OPTIONS: u32>(
            token: Token,
            bases: [f32; $lanes],
            exponent: f32,
        ) -> Result<[f32; $lanes], PowfError> {
            $profile::<Token, ARITHMETIC, OPTIONS>(
                token,
                crate::simd::generic::$float::from_array_t(token, bases),
                exponent,
            )
            .map(|result| result.to_array())
        }

        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
        pub(super) fn $entry<const ARITHMETIC: usize, const DOMAIN: usize>(
            token: Token,
            bases: [f32; $lanes],
            exponent: f32,
        ) -> Result<[f32; $lanes], PowfError> {
            match DOMAIN {
                0 => $core::<Token, ARITHMETIC, false, false>(token, bases, exponent),
                1 => $core::<Token, ARITHMETIC, true, false>(token, bases, exponent),
                2 => $core::<Token, ARITHMETIC, true, true>(token, bases, exponent),
                _ => unreachable!("private domain specialization"),
            }
        }
    };
}

graph!(
    core4,
    vector4,
    profile4,
    kernel4,
    kernel4_with,
    f32x4,
    i32x4,
    F32x4Convert,
    4
);
graph!(
    core8,
    vector8,
    profile8,
    kernel8,
    kernel8_with,
    f32x8,
    i32x8,
    F32x8Convert,
    8
);
#[cfg(feature = "w512")]
graph!(
    core16,
    vector16,
    profile16,
    kernel16,
    kernel16_with,
    f32x16,
    i32x16,
    F32x16Convert,
    16
);

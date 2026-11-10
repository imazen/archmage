//! Const-configured power graphs, instantiated per width and arithmetic provider.
use super::{PowfError, PowfErrorKind, PowfOptions};

// Arithmetic mode is chosen once at the operation boundary. Mode 0 uses the
// token-native relaxed madd; BitExact proves fusion before selecting it, while
// BackendDependent permits either rounding choice.
pub(super) const NATIVE: usize = 0;
pub(super) const SOFTWARE: usize = 1;
pub(super) const RELAXED: usize = 4;
// Internal approximate-power configurations. These are not accepted public
// option encodings: the existing infallible methods select them at compile time.
pub(super) const CLAMP: u32 = 256;
pub(super) const INPUT_EDGES: u32 = 512;
// Dispatch and domain flags have already selected the provider and ZERO/GAMMA
// parameters. Keeping them in the arithmetic instantiation duplicates identical
// debug graphs for Caller/Automatic and portable/backend-dependent adapters.
#[cfg(debug_assertions)]
const KERNEL_FLAGS: u32 = PowfOptions::LOWP | PowfOptions::ASSUME | CLAMP | INPUT_EDGES;
#[cfg(target_arch = "wasm32")]
pub(super) const WASM_SOFTWARE: usize = 3;
#[cfg(all(feature = "std", not(miri)))]
pub(super) const STD: usize = 2;

// Literal arms support the MSRV without generic const expressions. Only debug
// builds use this adapter; release keeps the original caller-visible flags.
#[cfg(debug_assertions)]
macro_rules! shared_kernel {
    ($vector:ident, $t:ty, $arithmetic:ident, $zero:literal, $gamma:literal,
     $options:ident, $token:ident, $input:ident, $exponent:ident) => {
        match $options & KERNEL_FLAGS {
            0 => $vector::<$t, $arithmetic, $zero, $gamma, 0>($token, $input, $exponent),
            8 => $vector::<$t, $arithmetic, $zero, $gamma, 8>($token, $input, $exponent),
            32 => $vector::<$t, $arithmetic, $zero, $gamma, 32>($token, $input, $exponent),
            40 => $vector::<$t, $arithmetic, $zero, $gamma, 40>($token, $input, $exponent),
            296 => $vector::<$t, $arithmetic, $zero, $gamma, 296>($token, $input, $exponent),
            800 => $vector::<$t, $arithmetic, $zero, $gamma, 800>($token, $input, $exponent),
            _ => unreachable!("private arithmetic flags are validated at the boundary"),
        }
    };
}

macro_rules! graph {
    ($core:ident, $vector:ident, $profile:ident, $entry:ident, $const_entry:ident, $float:ident, $int:ident, $bound:ident, $lanes:literal) => {
        // Debug callers share each instantiated graph rather than duplicating
        // every Horner node through the const/runtime adapters. Release keeps
        // the caller-visible graph and its measured feature-region inlining.
        #[cfg_attr(debug_assertions, inline(never))]
        #[cfg_attr(not(debug_assertions), inline(always))]
        fn $vector<
            T: crate::simd::backends::$bound,
            const ARITHMETIC: usize,
            const ZERO: bool,
            const GAMMA: bool,
            const OPTIONS: u32,
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
                let valid = if ZERO {
                    within.bitcast_to_i32() | zero
                } else {
                    within.bitcast_to_i32()
                };
                valid ^ ip(-1)
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
            if OPTIONS & PowfOptions::ASSUME == 0 && invalid.any_true() {
                return Err(reject(invalid, PowfErrorKind::InvalidBase));
            }
            // Gamma's zero sentinel can follow the ordinary reduction safely:
            // its integer exponent is -127, and multiplying by n in [0,4]
            // stays within [-512,0]. The final mask supplies the specified
            // zero result. General nonnegative inputs still need substitution
            // before multiplying by an arbitrarily large admitted exponent.
            let x = if ZERO && !GAMMA {
                V::blend(zero.bitcast_to_f32(), sp(1.0), input)
            } else {
                input
            };
            let madd = |a: V<T>, b: V<T>, c: V<T>| -> V<T> {
                if ARITHMETIC == NATIVE || ARITHMETIC == RELAXED {
                    a.mul_add_relaxed(b, c)
                } else if ARITHMETIC == SOFTWARE {
                    // Preserve the scalar representation, while allowing the
                    // existing fused primitive to use compile-proven FMA (for
                    // example baseline AArch64). It adds no runtime f32 probe.
                    a.mul_add_fused(b, c)
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
            let result = if OPTIONS & PowfOptions::LOWP != 0 {
                let log = if OPTIONS & CLAMP != 0 {
                    input.log2_lowp()
                } else {
                    let xb = x.bitcast_to_i32();
                    let offset = ip(0x3f2a_aaab_u32 as i32);
                    let shifted = (xb - offset).shr_arithmetic_const::<23>();
                    let m = (xb - shifted.shl_const::<23>()).bitcast_to_f32() - sp(1.0);
                    let yp = madd(sp(0.742_458_7), m, sp(1.428_716_1));
                    let yp = madd(yp, m, sp(-1.850_383_3e-6));
                    let yq = madd(sp(0.174_093_43), m, sp(1.009_671_8));
                    let yq = madd(yq, m, sp(0.990_328_14));
                    yp / yq + shifted.to_f32()
                };
                let t = log * sp(exponent);
                if OPTIONS & PowfOptions::ASSUME == 0 && !GAMMA {
                    let outside = (t.simd_lt(sp(-126.0)) | t.simd_gt(sp(127.5))).bitcast_to_i32();
                    // Admitted zero has an explicit result, independent of
                    // LOWP's nonzero log2(1) bias at very large exponents.
                    let outside = if ZERO {
                        outside & (zero ^ ip(-1))
                    } else {
                        outside
                    };
                    if outside.any_true() {
                        return Err(reject(outside, PowfErrorKind::OutsideApproximationRange));
                    }
                }
                // PositiveNormal already promises t in [-126,127.5]; Gamma's
                // nonzero lanes have |t| < 68. Only general nonnegative zero
                // needs the legacy clamp (LOWP log2(1) has a small bias).
                let t = if (ZERO && !GAMMA) || OPTIONS & CLAMP != 0 {
                    t.max(sp(-126.0)).min(sp(127.99))
                } else {
                    t
                };
                let xi = t.floor();
                let xf = t - xi;
                let p = madd(sp(0.055_504_11), xf, sp(0.240_226_5));
                let p = madd(p, xf, sp(core::f32::consts::LN_2));
                let p = madd(p, xf, sp(1.0));
                p * (xi.to_i32_round() + ip(127))
                    .shl_const::<23>()
                    .bitcast_to_f32()
            } else {
                // BEGIN MIDP GRAPH
                // Coefficients and arithmetic order from the existing midp log2
                // and exp2 graph; BitExact rounds every madd once, while the
                // infallible approximation retains its caller's relaxed arithmetic.
                let log = if OPTIONS & INPUT_EDGES != 0 {
                    input.log2_midp()
                } else {
                    let adjusted = x.bitcast_to_i32() + ip((0x3f80_0000_u32 - 0x3f35_04f3) as i32);
                    let n = (adjusted.shr_arithmetic_const::<23>() - ip(127)).to_f32();
                    let a = ((adjusted & ip(0x007f_ffff)) + ip(0x3f35_04f3)).bitcast_to_f32();
                    let y = (a - sp(1.0)) / (a + sp(1.0));
                    let y2 = y * y;
                    let p = madd(sp(0.434_411_97), y2, sp(0.576_974_45));
                    let p = madd(p, y2, sp(0.961_800_76));
                    let p = madd(p, y2, sp(2.885_39));
                    madd(p, y, n)
                };
                let t = log * sp(exponent);
                // Gamma: |integer exponent| <= 16, |y| < 1/4, and the
                // positive Horner polynomial is < 2.96. Thus |log2| < 17,
                // and |n*log2| < 68 for n in [0,4]. xtask/powf_vectors.py
                // checks an independent rational/RNE interval certificate.
                if OPTIONS & PowfOptions::ASSUME == 0 && !GAMMA {
                    let outside = t.simd_lt(sp(-126.0)) | t.simd_gt(sp(127.5));
                    if outside.bitcast_to_i32().any_true() {
                        return Err(reject(
                            outside.bitcast_to_i32(),
                            PowfErrorKind::OutsideApproximationRange,
                        ));
                    }
                }
                let underflow_limit = sp(-126.0);
                let overflow_limit = sp(128.0);
                let reduced = if OPTIONS & CLAMP != 0 {
                    t.max(underflow_limit).min(overflow_limit)
                } else {
                    t
                };
                let xi = if GAMMA {
                    reduced.round()
                } else {
                    reduced.round().min(sp(127.0))
                };
                let xf = reduced - xi;
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
                if OPTIONS & CLAMP != 0 {
                    let is_underflow = t.simd_lt(underflow_limit);
                    let is_overflow = t.simd_ge(overflow_limit);
                    let zero = sp(0.0);
                    let inf = sp(f32::INFINITY);
                    let result = V::blend(is_underflow, zero, result);
                    V::blend(is_overflow, inf, result)
                } else {
                    result
                }
                // END MIDP GRAPH
            };
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
            #[cfg(debug_assertions)]
            return match OPTIONS & 3 {
                0 => shared_kernel!(
                    $vector, T, ARITHMETIC, false, false, OPTIONS, token, input, exponent
                ),
                1 => shared_kernel!(
                    $vector, T, ARITHMETIC, true, false, OPTIONS, token, input, exponent
                ),
                2 => {
                    if OPTIONS & PowfOptions::POSITIVE != 0 {
                        shared_kernel!(
                            $vector, T, ARITHMETIC, false, true, OPTIONS, token, input, exponent
                        )
                    } else {
                        shared_kernel!(
                            $vector, T, ARITHMETIC, true, true, OPTIONS, token, input, exponent
                        )
                    }
                }
                _ => unreachable!("validated const domain"),
            };
            #[cfg(not(debug_assertions))]
            match OPTIONS & 3 {
                0 => $vector::<T, ARITHMETIC, false, false, OPTIONS>(token, input, exponent),
                1 => $vector::<T, ARITHMETIC, true, false, OPTIONS>(token, input, exponent),
                2 => {
                    if OPTIONS & PowfOptions::POSITIVE != 0 {
                        $vector::<T, ARITHMETIC, false, true, OPTIONS>(token, input, exponent)
                    } else {
                        $vector::<T, ARITHMETIC, true, true, OPTIONS>(token, input, exponent)
                    }
                }
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

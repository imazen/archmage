//! Configured logarithm graphs (`log2`, `ln`, `log10`, `log1p`), instantiated
//! per width, kernel configuration, op, and arithmetic mode.
#![allow(clippy::excessive_precision)] // Preserve the fitted f32 coefficient spellings.
use super::{LogfError, LogfErrorKind};

pub(super) const NATIVE: usize = 0;
pub(super) const SOFTWARE: usize = 1;
#[cfg(target_arch = "wasm32")]
pub(super) const WASM_SOFTWARE: usize = 2;

pub(super) trait Kernel: 'static {
    const RANGED: bool;
    const IN_LO: f32;
    const IN_HI: f32;
    const ZERO: bool;
    const SUBNORM: bool;
    const SPECIALS: bool;
    const CLAMP_OOB: bool;
    const LOWP: bool;
    const ASSUME: bool;
    const ELIDE_RED: bool;
    const ELIDE_RED_LOG1P: bool;
}

#[derive(Clone, Copy)]
pub(super) struct KernelCfg {
    pub(super) ranged: bool,
    pub(super) in_lo: f32,
    pub(super) in_hi: f32,
    pub(super) zero: bool,
    pub(super) subnorm: bool,
    pub(super) specials: bool,
    pub(super) clamp_oob: bool,
    pub(super) lowp: bool,
    pub(super) assume: bool,
    pub(super) elide_red: bool,
    pub(super) elide_red_log1p: bool,
}

impl KernelCfg {
    pub(super) const fn from_options(opts: super::LogfOptions) -> Self {
        let (in_lo, in_hi) = match opts.input_bounds() {
            Some(bounds) => bounds,
            None => (0.0, 0.0),
        };
        Self {
            ranged: opts.input_bounds().is_some(),
            in_lo,
            in_hi,
            zero: opts.admits_zero(),
            subnorm: opts.admits_subnormals(),
            specials: opts.is_specials(),
            clamp_oob: opts.clamp_oob(),
            lowp: opts.is_lowp(),
            assume: opts.assume_valid(),
            elide_red: opts.elides_reduction(),
            elide_red_log1p: opts.elides_log1p_reduction(),
        }
    }
}

#[cfg(debug_assertions)]
#[inline(always)]
pub(super) const fn cfg_of<K: Kernel>() -> KernelCfg {
    KernelCfg {
        ranged: K::RANGED,
        in_lo: K::IN_LO,
        in_hi: K::IN_HI,
        zero: K::ZERO,
        subnorm: K::SUBNORM,
        specials: K::SPECIALS,
        clamp_oob: K::CLAMP_OOB,
        lowp: K::LOWP,
        assume: K::ASSUME,
        elide_red: K::ELIDE_RED,
        elide_red_log1p: K::ELIDE_RED_LOG1P,
    }
}

macro_rules! make_madd {
    ($V:ident, $token:ident, $arithmetic:expr) => {
        |a: $V<T>, b: $V<T>, c: $V<T>| -> $V<T> {
            if $arithmetic == NATIVE {
                a.mul_add_relaxed(b, c)
            } else if $arithmetic == SOFTWARE {
                a.mul_add_fused(b, c)
            } else {
                #[cfg(target_arch = "wasm32")]
                if $arithmetic == WASM_SOFTWARE {
                    let (a, b, c) = (a.to_array(), b.to_array(), c.to_array());
                    let proof = $token
                        .as_wasm128()
                        .expect("private WASM mode requires its matching proof");
                    return $V::from_array_t(
                        $token,
                        crate::wasm_fma::software_lanes(proof, a, b, c),
                    );
                }
                let (a, b, c) = (a.to_array(), b.to_array(), c.to_array());
                $V::from_array_t($token, crate::soft_fma::fmaf_soft_lanes(a, b, c))
            }
        }
    };
}

macro_rules! eval_log2 {
    (
        $V:ident,
        $I:ident,
        $token:ident,
        $input:ident,
        $arithmetic:expr,
        $ranged:expr,
        $in_lo:expr,
        $in_hi:expr,
        $zero:expr,
        $subnorm:expr,
        $specials:expr,
        $clamp_oob:expr,
        $skip_clamp:expr,
        $lowp:expr,
        $assume:expr,
        $elide_red:expr
    ) => {{
        let sp = |x| $V::splat_t($token, x);
        let ip = |x| $I::splat_t($token, x);
        let reject = |mask: $I<T>, kind| {
            let lane_mask = mask
                .to_array()
                .iter()
                .enumerate()
                .fold(0_u16, |m, (i, &v)| m | (u16::from(v != 0) << i));
            LogfError { kind, lane_mask }
        };
        if !$assume && !$clamp_oob {
            if $ranged {
                let within =
                    ($input.simd_ge(sp($in_lo)) & $input.simd_le(sp($in_hi))).bitcast_to_i32();
                let valid = if $zero {
                    within | $input.simd_eq(sp(0.0)).bitcast_to_i32()
                } else {
                    within
                };
                let invalid = valid ^ ip(-1);
                if invalid.any_true() {
                    return Err(reject(invalid, LogfErrorKind::InvalidInput));
                }
            } else if !$specials {
                let bits0 = $input.bitcast_to_i32();
                let exp_bits = bits0 & ip(0x7f80_0000);
                let is_zero_f = $input.simd_eq(sp(0.0)).bitcast_to_i32();
                let neg = if $zero {
                    bits0.simd_lt(ip(0)) & (is_zero_f ^ ip(-1))
                } else {
                    bits0.simd_le(ip(0))
                };
                let nonfinite = exp_bits.simd_eq(ip(0x7f80_0000));
                let tiny_bad = if $subnorm {
                    if $zero { ip(0) } else { is_zero_f }
                } else if $zero {
                    exp_bits.simd_eq(ip(0)) & (is_zero_f ^ ip(-1))
                } else {
                    exp_bits.simd_eq(ip(0))
                };
                let invalid = neg | nonfinite | tiny_bad;
                if invalid.any_true() {
                    return Err(reject(invalid, LogfErrorKind::InvalidInput));
                }
            }
        }
        let is_nan = if $clamp_oob && !$specials && !$skip_clamp {
            $input.simd_ne($input)
        } else {
            ip(0).bitcast_to_f32()
        };
        let work_in = if $clamp_oob && !$specials && !$skip_clamp {
            let lo_bound = if $ranged {
                $in_lo
            } else if $subnorm {
                f32::from_bits(1)
            } else {
                f32::MIN_POSITIVE
            };
            let hi_bound = if $ranged { $in_hi } else { f32::MAX };
            let c = if $zero {
                let nonpos = $input.simd_le(sp(0.0));
                let lo_c = $V::blend(
                    nonpos,
                    sp(0.0),
                    $V::blend($input.simd_lt(sp(lo_bound)), sp(lo_bound), $input),
                );
                $V::blend($input.simd_gt(sp(hi_bound)), sp(hi_bound), lo_c)
            } else {
                $V::blend(
                    $input.simd_lt(sp(lo_bound)),
                    sp(lo_bound),
                    $V::blend($input.simd_gt(sp(hi_bound)), sp(hi_bound), $input),
                )
            };
            let safe = if $ranged && lo_bound > 0.0 {
                lo_bound
            } else {
                1.0
            };
            $V::blend(is_nan, sp(safe), c)
        } else {
            $input
        };

        let raw_bits = work_in.bitcast_to_i32();
        let is_subnorm = if $subnorm && !$elide_red {
            raw_bits.simd_gt(ip(0)) & raw_bits.simd_lt(ip(0x0080_0000))
        } else {
            ip(0)
        };
        let norm_in = if $subnorm && !$elide_red {
            $V::blend(
                is_subnorm.bitcast_to_f32(),
                work_in * sp(8_388_608.0),
                work_in,
            )
        } else {
            work_in
        };
        let bits = norm_in.bitcast_to_i32();
        let madd = make_madd!($V, $token, $arithmetic);

        let result = if $lowp {
            if $elide_red {
                let m = work_in - sp(1.0);
                let yp = madd(sp(0.742_458_7), m, sp(1.428_716_1));
                let yp = madd(yp, m, sp(-1.850_383_3e-6));
                let yq = madd(sp(0.174_093_43), m, sp(1.009_671_8));
                let yq = madd(yq, m, sp(0.990_328_14));
                yp / yq
            } else {
                let offset = ip(0x3f2a_aaab_u32 as i32);
                let exp_bits_v = bits - offset;
                let exp_shifted = exp_bits_v.shr_arithmetic_const::<23>();
                let mantissa_bits = bits - exp_shifted.shl_const::<23>();
                let m = mantissa_bits.bitcast_to_f32() - sp(1.0);
                let yp = madd(sp(0.742_458_7), m, sp(1.428_716_1));
                let yp = madd(yp, m, sp(-1.850_383_3e-6));
                let yq = madd(sp(0.174_093_43), m, sp(1.009_671_8));
                let yq = madd(yq, m, sp(0.990_328_14));
                let exp_adj = if $subnorm {
                    exp_shifted - (is_subnorm & ip(23))
                } else {
                    exp_shifted
                };
                yp / yq + exp_adj.to_f32()
            }
        } else if $elide_red {
            let y = (work_in - sp(1.0)) / (work_in + sp(1.0));
            let y2 = y * y;
            let p = madd(sp(0.430_468_597_034_945_7), y2, sp(0.576_767_602_130_619_1));
            let p = madd(p, y2, sp(0.961_798_223_457_557_8));
            let p = madd(p, y2, sp(2.885_390_081_777_926_8));
            p * y
        } else {
            let adjusted = bits + ip((0x3f80_0000_u32 - 0x3f35_04f3) as i32);
            let raw_exp = adjusted.shr_arithmetic_const::<23>() - ip(127);
            let n = (if $subnorm {
                raw_exp - (is_subnorm & ip(23))
            } else {
                raw_exp
            })
            .to_f32();
            let a = ((adjusted & ip(0x007f_ffff)) + ip(0x3f35_04f3)).bitcast_to_f32();
            let y = (a - sp(1.0)) / (a + sp(1.0));
            let y2 = y * y;
            let p = madd(sp(0.430_468_597_034_945_7), y2, sp(0.576_767_602_130_619_1));
            let p = madd(p, y2, sp(0.961_798_223_457_557_8));
            let p = madd(p, y2, sp(2.885_390_081_777_926_8));
            madd(p, y, n)
        };
        let result = if $specials {
            let zero = work_in.simd_eq(sp(0.0));
            let nonnegative = work_in.simd_ge(sp(0.0));
            let pos_inf = work_in.simd_eq(sp(f32::INFINITY));
            let result = $V::blend(zero, sp(f32::NEG_INFINITY), result);
            let result = $V::blend(nonnegative, result, sp(f32::NAN));
            $V::blend(pos_inf, sp(f32::INFINITY), result)
        } else {
            let result = if $zero {
                $V::blend(work_in.simd_eq(sp(0.0)), sp(f32::NEG_INFINITY), result)
            } else {
                result
            };
            if $clamp_oob && !$skip_clamp {
                $V::blend(is_nan, sp(f32::NAN), result)
            } else {
                result
            }
        };
        Ok(result)
    }};
}

macro_rules! eval_log_op {
    (
        $V:ident,
        $I:ident,
        $token:ident,
        $input:ident,
        $op:expr,
        $arithmetic:expr,
        $ranged:expr,
        $in_lo:expr,
        $in_hi:expr,
        $zero:expr,
        $specials:expr,
        $clamp_oob:expr,
        $assume:expr,
        $elide_red_log1p:expr,
        $log2_fn:expr
    ) => {{
        let sp = |x| $V::splat_t($token, x);
        let ip = |x| $I::splat_t($token, x);
        if $op == 3 {
            if !$assume && !$clamp_oob && $ranged {
                let within =
                    ($input.simd_ge(sp($in_lo)) & $input.simd_le(sp($in_hi))).bitcast_to_i32();
                let valid = if $zero {
                    within | $input.simd_eq(sp(-1.0)).bitcast_to_i32()
                } else {
                    within
                };
                let invalid = valid ^ ip(-1);
                if invalid.any_true() {
                    let lane_mask = invalid
                        .to_array()
                        .iter()
                        .enumerate()
                        .fold(0_u16, |m, (i, &v)| m | (u16::from(v != 0) << i));
                    return Err(LogfError {
                        kind: LogfErrorKind::InvalidInput,
                        lane_mask,
                    });
                }
            }
            let is_nan = if $clamp_oob && !$specials {
                $input.simd_ne($input)
            } else {
                ip(0).bitcast_to_f32()
            };
            let work_x = if $clamp_oob && !$specials {
                let lo_bound = if $ranged {
                    $in_lo
                } else if $zero {
                    -1.0
                } else {
                    f32::from_bits(0xbf7f_ffff)
                };
                let hi_bound = if $ranged { $in_hi } else { f32::MAX };
                let c = if $zero {
                    let le_neg_one = $input.simd_le(sp(-1.0));
                    let lo_c = $V::blend(
                        le_neg_one,
                        sp(-1.0),
                        $V::blend($input.simd_lt(sp(lo_bound)), sp(lo_bound), $input),
                    );
                    $V::blend($input.simd_gt(sp(hi_bound)), sp(hi_bound), lo_c)
                } else {
                    $V::blend(
                        $input.simd_lt(sp(lo_bound)),
                        sp(lo_bound),
                        $V::blend($input.simd_gt(sp(hi_bound)), sp(hi_bound), $input),
                    )
                };
                let safe = if $ranged && lo_bound > -1.0 {
                    lo_bound
                } else {
                    0.0
                };
                $V::blend(is_nan, sp(safe), c)
            } else {
                $input
            };
            let one = sp(1.0);
            let u = work_x + one;
            let c = if $elide_red_log1p {
                (work_x - (u - one)) / u
            } else {
                let two = sp(2.0);
                let finite_pos = u.simd_gt(sp(0.0)) & u.simd_lt(sp(f32::INFINITY));
                let c_raw = $V::blend(
                    u.simd_ge(two),
                    (one - (u - work_x)) / u,
                    (work_x - (u - one)) / u,
                );
                $V::blend(finite_pos, c_raw, sp(0.0))
            };
            let log2_u = $log2_fn(u)?;
            let ln2 = sp(core::f32::consts::LN_2);
            let madd = make_madd!($V, $token, $arithmetic);
            let res = madd(log2_u, ln2, c);
            let res = $V::blend(work_x.simd_eq(sp(0.0)), work_x, res);
            if $clamp_oob && !$specials {
                Ok($V::blend(is_nan, sp(f32::NAN), res))
            } else {
                Ok(res)
            }
        } else {
            let log2_v = $log2_fn($input)?;
            Ok(match $op {
                1 => log2_v * sp(core::f32::consts::LN_2),
                2 => log2_v * sp(core::f32::consts::LOG10_2),
                _ => log2_v,
            })
        }
    }};
}

macro_rules! graph {
    ($vector_dbg:ident, $vector_rel:ident, $eval_dbg:ident, $eval_rel:ident, $profile:ident, $cfg_entry:ident, $const_entry:ident, $float:ident, $int:ident, $bound:ident, $lanes:literal) => {
        #[cfg_attr(debug_assertions, inline(never))]
        #[cfg_attr(not(debug_assertions), inline)]
        fn $vector_dbg<T: magetypes::simd::backends::$bound, const ARITHMETIC: usize>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
            skip_clamp: bool,
            cfg: KernelCfg,
        ) -> Result<magetypes::simd::generic::$float<T>, LogfError> {
            use magetypes::simd::generic::{$float as V, $int as I};
            eval_log2!(
                V,
                I,
                token,
                input,
                ARITHMETIC,
                cfg.ranged,
                cfg.in_lo,
                cfg.in_hi,
                cfg.zero,
                cfg.subnorm,
                cfg.specials,
                cfg.clamp_oob,
                skip_clamp,
                cfg.lowp,
                cfg.assume,
                cfg.elide_red
            )
        }

        #[cfg(not(debug_assertions))]
        #[inline(always)]
        fn $vector_rel<
            T: magetypes::simd::backends::$bound,
            K: Kernel,
            const ARITHMETIC: usize,
            const FOR_LOG1P: bool,
        >(
            token: T,
            input: magetypes::simd::generic::$float<T>,
        ) -> Result<magetypes::simd::generic::$float<T>, LogfError> {
            use magetypes::simd::generic::{$float as V, $int as I};
            eval_log2!(
                V,
                I,
                token,
                input,
                ARITHMETIC,
                if FOR_LOG1P { false } else { K::RANGED },
                K::IN_LO,
                K::IN_HI,
                K::ZERO,
                K::SUBNORM,
                K::SPECIALS,
                if FOR_LOG1P { false } else { K::CLAMP_OOB },
                FOR_LOG1P,
                K::LOWP,
                if FOR_LOG1P {
                    K::ASSUME || K::RANGED || K::CLAMP_OOB
                } else {
                    K::ASSUME
                },
                if FOR_LOG1P {
                    K::ELIDE_RED_LOG1P
                } else {
                    K::ELIDE_RED
                }
            )
        }

        #[cfg_attr(debug_assertions, inline(never))]
        #[cfg_attr(not(debug_assertions), inline)]
        fn $eval_dbg<T: magetypes::simd::backends::$bound, const ARITHMETIC: usize>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
            op: u8,
            cfg: KernelCfg,
        ) -> Result<magetypes::simd::generic::$float<T>, LogfError> {
            use magetypes::simd::generic::{$float as V, $int as I};
            let u_cfg = if op == 3 {
                KernelCfg {
                    ranged: false,
                    clamp_oob: false,
                    assume: cfg.assume || cfg.ranged || cfg.clamp_oob,
                    elide_red: cfg.elide_red_log1p,
                    ..cfg
                }
            } else {
                cfg
            };
            eval_log_op!(
                V,
                I,
                token,
                input,
                op,
                ARITHMETIC,
                cfg.ranged,
                cfg.in_lo,
                cfg.in_hi,
                cfg.zero,
                cfg.specials,
                cfg.clamp_oob,
                cfg.assume,
                cfg.elide_red_log1p,
                |arg| $vector_dbg::<T, ARITHMETIC>(token, arg, op == 3, u_cfg)
            )
        }

        #[cfg(not(debug_assertions))]
        #[inline(always)]
        fn $eval_rel<
            T: magetypes::simd::backends::$bound,
            K: Kernel,
            const OP: u8,
            const ARITHMETIC: usize,
        >(
            token: T,
            input: magetypes::simd::generic::$float<T>,
        ) -> Result<magetypes::simd::generic::$float<T>, LogfError> {
            use magetypes::simd::generic::{$float as V, $int as I};
            eval_log_op!(
                V,
                I,
                token,
                input,
                OP,
                ARITHMETIC,
                K::RANGED,
                K::IN_LO,
                K::IN_HI,
                K::ZERO,
                K::SPECIALS,
                K::CLAMP_OOB,
                K::ASSUME,
                K::ELIDE_RED_LOG1P,
                |arg| if OP == 3 {
                    $vector_rel::<T, K, ARITHMETIC, true>(token, arg)
                } else {
                    $vector_rel::<T, K, ARITHMETIC, false>(token, arg)
                }
            )
        }

        #[inline(always)]
        pub(super) fn $profile<
            T: magetypes::simd::backends::$bound,
            K: Kernel,
            const OP: u8,
            const ARITHMETIC: usize,
        >(
            token: T,
            input: magetypes::simd::generic::$float<T>,
        ) -> Result<magetypes::simd::generic::$float<T>, LogfError> {
            #[cfg(debug_assertions)]
            return $eval_dbg::<T, ARITHMETIC>(token, input, OP, cfg_of::<K>());
            #[cfg(not(debug_assertions))]
            $eval_rel::<T, K, OP, ARITHMETIC>(token, input)
        }

        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
        pub(super) fn $cfg_entry<const ARITHMETIC: usize>(
            token: Token,
            input: [f32; $lanes],
            op: u8,
            cfg: KernelCfg,
        ) -> Result<[f32; $lanes], LogfError> {
            $eval_dbg::<Token, ARITHMETIC>(
                token,
                magetypes::simd::generic::$float::from_array_t(token, input),
                op,
                cfg,
            )
            .map(|result| result.to_array())
        }

        #[cfg(not(debug_assertions))]
        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
        pub(super) fn $const_entry<K: Kernel, const OP: u8, const ARITHMETIC: usize>(
            token: Token,
            input: [f32; $lanes],
        ) -> Result<[f32; $lanes], LogfError> {
            $profile::<Token, K, OP, ARITHMETIC>(
                token,
                magetypes::simd::generic::$float::from_array_t(token, input),
            )
            .map(|result| result.to_array())
        }
    };
}

graph!(
    vector4_dbg,
    vector4_rel,
    eval_op4_dbg,
    eval_op4_rel,
    profile4,
    kernel4_cfg,
    kernel4_with,
    f32x4,
    i32x4,
    F32x4Convert,
    4
);
graph!(
    vector8_dbg,
    vector8_rel,
    eval_op8_dbg,
    eval_op8_rel,
    profile8,
    kernel8_cfg,
    kernel8_with,
    f32x8,
    i32x8,
    F32x8Convert,
    8
);
#[cfg(feature = "w512")]
graph!(
    vector16_dbg,
    vector16_rel,
    eval_op16_dbg,
    eval_op16_rel,
    profile16,
    kernel16_cfg,
    kernel16_with,
    f32x16,
    i32x16,
    F32x16Convert,
    16
);

/// Whole-slice kernel (`op`: `0=log2`, `1=ln`, `2=log10`, `3=log1p`)
/// parameterized on `KernelCfg`.
#[archmage::magetypes(v3, neon, wasm128, scalar)]
pub(super) fn slice8_cfg<const ARITHMETIC: usize>(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
    op: u8,
    cfg: KernelCfg,
) -> Result<(), (usize, LogfError)> {
    use magetypes::simd::generic::f32x8 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        eval_op8_dbg::<Token, ARITHMETIC>(token, V::load_t(token, s), op, cfg)
            .map_err(|e| (i * 8, e))?
            .store(d);
    }
    if !tail.is_empty() {
        eval_op8_dbg::<Token, ARITHMETIC>(token, V::load_padded_t(token, tail, tail[0]), op, cfg)
            .map_err(|e| (chunks.len() * 8, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[inline(always)]
fn slice16_cfg_impl<T: magetypes::simd::backends::F32x16Convert, const ARITHMETIC: usize>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
    op: u8,
    cfg: KernelCfg,
) -> Result<(), (usize, LogfError)> {
    use magetypes::simd::generic::f32x16 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        eval_op16_dbg::<T, ARITHMETIC>(token, V::load_t(token, s), op, cfg)
            .map_err(|e| (i * 16, e))?
            .store(d);
    }
    if !tail.is_empty() {
        eval_op16_dbg::<T, ARITHMETIC>(token, V::load_padded_t(token, tail, tail[0]), op, cfg)
            .map_err(|e| (chunks.len() * 16, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

/// 16-lane AVX-512 (`V4`) whole-slice kernel parameterized on `KernelCfg`.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
pub(super) fn slice16_cfg_v4<const ARITHMETIC: usize>(
    token: archmage::X64V4Token,
    src: &[f32],
    dst: &mut [f32],
    op: u8,
    cfg: KernelCfg,
) -> Result<(), (usize, LogfError)> {
    slice16_cfg_impl::<_, ARITHMETIC>(token, src, dst, op, cfg)
}

/// 16-lane AVX-512 (`V4x`) whole-slice kernel parameterized on `KernelCfg`.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
pub(super) fn slice16_cfg_v4x<const ARITHMETIC: usize>(
    token: archmage::X64V4xToken,
    src: &[f32],
    dst: &mut [f32],
    op: u8,
    cfg: KernelCfg,
) -> Result<(), (usize, LogfError)> {
    slice16_cfg_impl::<_, ARITHMETIC>(token, src, dst, op, cfg)
}

/// Whole-slice kernel (`OP`: `0=log2`, `1=ln`, `2=log10`, `3=log1p`): one
/// feature region around the loop, the 8-lane graph per chunk, and `tail[0]`
/// padding so ranges excluding `1.0` never false-reject.
#[cfg(not(debug_assertions))]
#[archmage::magetypes(v3, neon, wasm128, scalar)]
pub(super) fn slice8_with<K: Kernel, const OP: u8, const ARITHMETIC: usize>(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, LogfError)> {
    use magetypes::simd::generic::f32x8 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        profile8::<Token, K, OP, ARITHMETIC>(token, V::load_t(token, s))
            .map_err(|e| (i * 8, e))?
            .store(d);
    }
    if !tail.is_empty() {
        profile8::<Token, K, OP, ARITHMETIC>(token, V::load_padded_t(token, tail, tail[0]))
            .map_err(|e| (chunks.len() * 8, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

#[cfg(all(not(debug_assertions), target_arch = "x86_64", feature = "avx512"))]
#[inline(always)]
fn slice16_with_impl<
    T: magetypes::simd::backends::F32x16Convert,
    K: Kernel,
    const OP: u8,
    const ARITHMETIC: usize,
>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, LogfError)> {
    use magetypes::simd::generic::f32x16 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        profile16::<T, K, OP, ARITHMETIC>(token, V::load_t(token, s))
            .map_err(|e| (i * 16, e))?
            .store(d);
    }
    if !tail.is_empty() {
        profile16::<T, K, OP, ARITHMETIC>(token, V::load_padded_t(token, tail, tail[0]))
            .map_err(|e| (chunks.len() * 16, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

/// 16-lane AVX-512 (`V4`) whole-slice kernel (`zmm`).
#[cfg(all(not(debug_assertions), target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
pub(super) fn slice16_with_v4<K: Kernel, const OP: u8, const ARITHMETIC: usize>(
    token: archmage::X64V4Token,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, LogfError)> {
    slice16_with_impl::<_, K, OP, ARITHMETIC>(token, src, dst)
}

/// 16-lane AVX-512 (`V4x`) whole-slice kernel (`zmm`).
#[cfg(all(not(debug_assertions), target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
pub(super) fn slice16_with_v4x<K: Kernel, const OP: u8, const ARITHMETIC: usize>(
    token: archmage::X64V4xToken,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, LogfError)> {
    slice16_with_impl::<_, K, OP, ARITHMETIC>(token, src, dst)
}

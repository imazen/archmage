//! Configured exp2/exp graphs, instantiated per width, kernel config, and arithmetic mode.
#![allow(clippy::excessive_precision)] // Preserve the fitted f32 coefficient spellings.
use super::{ExpfError, ExpfErrorKind};

pub(super) const NATIVE: usize = 0;
pub(super) const SOFTWARE: usize = 1;
#[cfg(target_arch = "wasm32")]
pub(super) const WASM_SOFTWARE: usize = 2;

pub(super) trait Kernel: 'static {
    const RANGED: bool;
    const IN_LO_EXP2: f32;
    const IN_HI_EXP2: f32;
    const IN_LO_EXP: f32;
    const IN_HI_EXP: f32;
    const FULL: bool;
    const CLAMP_OOB: bool;
    const PRESERVE_CLAMPED_NAN: bool;
    const LOWP: bool;
    const ASSUME: bool;
    const ELIDE_EXP2: bool;
    const ELIDE_EXP: bool;
    const ELIDE_OVERFLOW_EXP2: bool;
    const ELIDE_OVERFLOW_EXP: bool;
    const ELIDE_RED_EXP2: bool;
    const ELIDE_RED_EXP: bool;
}

#[derive(Clone, Copy)]
pub(super) struct KernelCfg {
    pub(super) ranged: bool,
    pub(super) in_lo_exp2: f32,
    pub(super) in_hi_exp2: f32,
    pub(super) in_lo_exp: f32,
    pub(super) in_hi_exp: f32,
    pub(super) full: bool,
    pub(super) clamp_oob: bool,
    pub(crate) preserve_clamped_nan: bool,
    pub(super) lowp: bool,
    pub(super) assume: bool,
    pub(super) elide_exp2: bool,
    pub(super) elide_exp: bool,
    pub(super) elide_overflow_exp2: bool,
    pub(super) elide_overflow_exp: bool,
    pub(super) elide_red_exp2: bool,
    pub(super) elide_red_exp: bool,
}

impl KernelCfg {
    pub(super) const fn from_options(opts: super::ExpfOptions) -> Self {
        let (in_lo_exp2, in_hi_exp2) = match opts.exp2_bounds() {
            Some(bounds) => bounds,
            None => (0.0, 0.0),
        };
        let (in_lo_exp, in_hi_exp) = match opts.exp_bounds() {
            Some(bounds) => bounds,
            None => (0.0, 0.0),
        };
        Self {
            ranged: opts.exp2_bounds().is_some(),
            in_lo_exp2,
            in_hi_exp2,
            in_lo_exp,
            in_hi_exp,
            full: opts.is_full(),
            clamp_oob: opts.clamp_oob(),
            preserve_clamped_nan: opts.preserve_clamped_nan(),
            lowp: opts.is_lowp(),
            assume: opts.assume_valid(),
            elide_exp2: opts.elides_clamp(),
            elide_exp: opts.elides_exp_clamp(),
            elide_overflow_exp2: opts.elides_exp2_overflow(),
            elide_overflow_exp: opts.elides_exp_overflow(),
            elide_red_exp2: opts.elides_reduction(),
            elide_red_exp: opts.elides_exp_reduction(),
        }
    }
}

#[cfg(debug_assertions)]
#[inline(always)]
pub(super) const fn cfg_of<K: Kernel>() -> KernelCfg {
    KernelCfg {
        ranged: K::RANGED,
        in_lo_exp2: K::IN_LO_EXP2,
        in_hi_exp2: K::IN_HI_EXP2,
        in_lo_exp: K::IN_LO_EXP,
        in_hi_exp: K::IN_HI_EXP,
        full: K::FULL,
        clamp_oob: K::CLAMP_OOB,
        preserve_clamped_nan: K::PRESERVE_CLAMPED_NAN,
        lowp: K::LOWP,
        assume: K::ASSUME,
        elide_exp2: K::ELIDE_EXP2,
        elide_exp: K::ELIDE_EXP,
        elide_overflow_exp2: K::ELIDE_OVERFLOW_EXP2,
        elide_overflow_exp: K::ELIDE_OVERFLOW_EXP,
        elide_red_exp2: K::ELIDE_RED_EXP2,
        elide_red_exp: K::ELIDE_RED_EXP,
    }
}

// Relative minimax cubic, constrained to 1 and 2 at the octave endpoints.
pub(crate) const LOWP_COEFFICIENTS: [f32; 3] = [0.6954243475, 0.2263076823, 0.07826797018];

macro_rules! exp2_lowp {
    ($x:ident, $sp:ident, $madd:ident) => {{
        let [c1, c2, c3] = $crate::expf::graph::LOWP_COEFFICIENTS;
        let p = $madd($sp(c3), $x, $sp(c2));
        let p = $madd(p, $x, $sp(c1));
        $madd(p, $x, $sp(1.0))
    }};
}
pub(crate) use exp2_lowp;

macro_rules! eval_expf {
    (
        $V:ident,
        $I:ident,
        $token:ident,
        $input:ident,
        $op:expr,
        $arithmetic:expr,
        $ranged:expr,
        $in_lo_exp2:expr,
        $in_hi_exp2:expr,
        $in_lo_exp:expr,
        $in_hi_exp:expr,
        $full:expr,
        $clamp_oob:expr,
        $preserve_clamped_nan:expr,
        $lowp:expr,
        $assume:expr,
        $elide_exp2:expr,
        $elide_exp:expr,
        $elide_overflow_exp2:expr,
        $elide_overflow_exp:expr,
        $elide_red_exp2:expr,
        $elide_red_exp:expr
    ) => {{
        let sp = |x| $V::splat_t($token, x);
        let ip = |x| $I::splat_t($token, x);
        let in_lo = if $op == 1 { $in_lo_exp } else { $in_lo_exp2 };
        let in_hi = if $op == 1 { $in_hi_exp } else { $in_hi_exp2 };
        let reject = |mask: $I<T>, kind| {
            let lane_mask = mask
                .to_array()
                .iter()
                .enumerate()
                .fold(0_u16, |m, (i, &v)| m | (u16::from(v != 0) << i));
            ExpfError { kind, lane_mask }
        };
        if !$assume && !$clamp_oob {
            if $ranged {
                let invalid = ($input.simd_ge(sp(in_lo)) & $input.simd_le(sp(in_hi)))
                    .bitcast_to_i32()
                    ^ ip(-1);
                if invalid.any_true() {
                    return Err(reject(invalid, ExpfErrorKind::InvalidInput));
                }
            } else if !$full {
                let bits = $input.bitcast_to_i32();
                let abs = bits & ip(0x7fff_ffff_u32 as i32);
                let invalid = abs.simd_ge(ip(0x7f80_0000));
                if invalid.any_true() {
                    return Err(reject(invalid, ExpfErrorKind::InvalidInput));
                }
            }
        }
        let is_nan = if $preserve_clamped_nan && !$full {
            $input.simd_ne($input)
        } else {
            ip(0).bitcast_to_f32()
        };
        let work_in = if $clamp_oob && !$full {
            let (lo_b, hi_b, safe) = if $ranged {
                (in_lo, in_hi, in_lo)
            } else {
                (-f32::MAX, f32::MAX, 0.0)
            };
            let c = $V::blend(
                $input.simd_lt(sp(lo_b)),
                sp(lo_b),
                $V::blend($input.simd_gt(sp(hi_b)), sp(hi_b), $input),
            );
            if $preserve_clamped_nan {
                $V::blend(is_nan, sp(safe), c)
            } else {
                c
            }
        } else {
            $input
        };
        let arg = if $op == 1 {
            work_in * sp(core::f32::consts::LOG2_E)
        } else {
            work_in
        };
        let elide_clamp = if $op == 1 { $elide_exp } else { $elide_exp2 };
        let elide_overflow = if $op == 1 {
            $elide_overflow_exp
        } else {
            $elide_overflow_exp2
        };
        let elide_red = if $op == 1 {
            $elide_red_exp
        } else {
            $elide_red_exp2
        };
        let madd = |a: $V<T>, b: $V<T>, c: $V<T>| -> $V<T> {
            if $arithmetic == NATIVE {
                a.mul_add_relaxed(b, c)
            } else if $arithmetic == SOFTWARE {
                a.mul_add_fused(b, c)
            } else {
                #[cfg(target_arch = "wasm32")]
                if $arithmetic == WASM_SOFTWARE {
                    let (a, b, c) = (a.to_array(), b.to_array(), c.to_array());
                    let proof = $token
                        .try_wasm128()
                        .expect("private WASM mode requires its matching proof");
                    return $V::from_array_t(
                        $token,
                        crate::wasm_fma::software_lanes(proof, a, b, c),
                    );
                }
                let (a, b, c) = (a.to_array(), b.to_array(), c.to_array());
                $V::from_array_t($token, crate::soft_fma::fmaf_soft_lanes(a, b, c))
            }
        };
        let max_lower = |x: $V<T>, lo| {
            if $clamp_oob && !$full && !$preserve_clamped_nan {
                x.max(lo)
            } else {
                x.max_fast(lo)
            }
        };
        let result = if $lowp {
            if elide_red {
                exp2_lowp!(arg, sp, madd)
            } else {
                let x = if elide_clamp {
                    arg
                } else if elide_overflow {
                    max_lower(arg, sp(-126.0))
                } else {
                    max_lower(arg, sp(-126.0)).min_fast(sp(127.99))
                };
                let xi = x.floor();
                let xf = x - xi;
                let p = exp2_lowp!(xf, sp, madd);
                let integer = if $token.as_scalar().is_some() {
                    xi.to_i32()
                } else {
                    xi.to_i32_round()
                };
                p * (integer + ip(127)).shl_const::<23>().bitcast_to_f32()
            }
        } else if elide_red {
            let xf = arg;
            let p = madd(
                sp(0.000_152_527_896_498_79),
                xf,
                sp(0.001_338_604_727_826_21),
            );
            let p = madd(p, xf, sp(0.009_618_715_779_149_75));
            let p = madd(p, xf, sp(0.055_503_695_542_993_11));
            let p = madd(p, xf, sp(0.240_226_463_934_714_0));
            let p = madd(p, xf, sp(core::f32::consts::LN_2));
            madd(p, xf, sp(1.0))
        } else {
            let underflow_limit = sp(-126.0);
            let overflow_limit = sp(128.0);
            let reduced = if elide_clamp {
                arg
            } else if elide_overflow {
                max_lower(arg, underflow_limit)
            } else {
                max_lower(arg, underflow_limit).min_fast(overflow_limit)
            };
            let xi = if elide_clamp || elide_overflow {
                reduced.round()
            } else {
                reduced.round().min_fast(sp(127.0))
            };
            let xf = reduced - xi;
            let p = madd(
                sp(0.000_152_527_896_498_79),
                xf,
                sp(0.001_338_604_727_826_21),
            );
            let p = madd(p, xf, sp(0.009_618_715_779_149_75));
            let p = madd(p, xf, sp(0.055_503_695_542_993_11));
            let p = madd(p, xf, sp(0.240_226_463_934_714_0));
            let p = madd(p, xf, sp(core::f32::consts::LN_2));
            let p = madd(p, xf, sp(1.0));
            let integer = if $token.as_scalar().is_some() {
                xi.to_i32()
            } else {
                xi.to_i32_round()
            };
            let scale = (integer + ip(127)).shl_const::<23>().bitcast_to_f32();
            let result = p * scale;
            if elide_clamp {
                result
            } else if elide_overflow {
                let is_underflow = arg.simd_lt(underflow_limit);
                $V::blend(is_underflow, sp(0.0), result)
            } else {
                let is_underflow = arg.simd_lt(underflow_limit);
                let is_overflow = arg.simd_ge(overflow_limit);
                let result = $V::blend(is_underflow, sp(0.0), result);
                $V::blend(is_overflow, sp(f32::INFINITY), result)
            }
        };
        let result = if $full {
            let nan = $input.simd_ne($input);
            let pos_inf = $input.simd_eq(sp(f32::INFINITY));
            let neg_inf = $input.simd_eq(sp(f32::NEG_INFINITY));
            let result = $V::blend(nan, sp(f32::NAN), result);
            let result = $V::blend(pos_inf, sp(f32::INFINITY), result);
            $V::blend(neg_inf, sp(0.0), result)
        } else if $preserve_clamped_nan {
            $V::blend(is_nan, sp(f32::NAN), result)
        } else {
            result
        };
        Ok(result)
    }};
}

macro_rules! graph {
    ($vector_dbg:ident, $vector_rel:ident, $profile:ident, $cfg_entry:ident, $const_entry:ident, $float:ident, $int:ident, $bound:ident, $lanes:literal) => {
        #[cfg_attr(debug_assertions, inline(never))]
        #[cfg_attr(not(debug_assertions), inline)]
        fn $vector_dbg<T: magetypes::simd::backends::$bound, const ARITHMETIC: usize>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
            op: u8,
            cfg: KernelCfg,
        ) -> Result<magetypes::simd::generic::$float<T>, ExpfError> {
            use magetypes::simd::generic::{$float as V, $int as I};
            eval_expf!(
                V,
                I,
                token,
                input,
                op,
                ARITHMETIC,
                cfg.ranged,
                cfg.in_lo_exp2,
                cfg.in_hi_exp2,
                cfg.in_lo_exp,
                cfg.in_hi_exp,
                cfg.full,
                cfg.clamp_oob,
                cfg.preserve_clamped_nan,
                cfg.lowp,
                cfg.assume,
                cfg.elide_exp2,
                cfg.elide_exp,
                cfg.elide_overflow_exp2,
                cfg.elide_overflow_exp,
                cfg.elide_red_exp2,
                cfg.elide_red_exp
            )
        }

        #[cfg(not(debug_assertions))]
        #[inline(always)]
        fn $vector_rel<
            T: magetypes::simd::backends::$bound,
            K: Kernel,
            const OP: u8,
            const ARITHMETIC: usize,
        >(
            token: T,
            input: magetypes::simd::generic::$float<T>,
        ) -> Result<magetypes::simd::generic::$float<T>, ExpfError> {
            use magetypes::simd::generic::{$float as V, $int as I};
            eval_expf!(
                V,
                I,
                token,
                input,
                OP,
                ARITHMETIC,
                K::RANGED,
                K::IN_LO_EXP2,
                K::IN_HI_EXP2,
                K::IN_LO_EXP,
                K::IN_HI_EXP,
                K::FULL,
                K::CLAMP_OOB,
                K::PRESERVE_CLAMPED_NAN,
                K::LOWP,
                K::ASSUME,
                K::ELIDE_EXP2,
                K::ELIDE_EXP,
                K::ELIDE_OVERFLOW_EXP2,
                K::ELIDE_OVERFLOW_EXP,
                K::ELIDE_RED_EXP2,
                K::ELIDE_RED_EXP
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
        ) -> Result<magetypes::simd::generic::$float<T>, ExpfError> {
            #[cfg(debug_assertions)]
            return $vector_dbg::<T, ARITHMETIC>(token, input, OP, cfg_of::<K>());
            #[cfg(not(debug_assertions))]
            $vector_rel::<T, K, OP, ARITHMETIC>(token, input)
        }

        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
        pub(super) fn $cfg_entry<const ARITHMETIC: usize>(
            token: Token,
            input: [f32; $lanes],
            op: u8,
            cfg: KernelCfg,
        ) -> Result<[f32; $lanes], ExpfError> {
            $vector_dbg::<Token, ARITHMETIC>(
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
        ) -> Result<[f32; $lanes], ExpfError> {
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
    profile16,
    kernel16_cfg,
    kernel16_with,
    f32x16,
    i32x16,
    F32x16Convert,
    16
);

/// Whole-slice kernel (`op = 0`: `exp2`, `op = 1`: `exp`) parameterized on `KernelCfg`.
#[archmage::magetypes(v3, neon, wasm128, scalar)]
pub(super) fn slice8_cfg<const ARITHMETIC: usize>(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
    op: u8,
    cfg: KernelCfg,
) -> Result<(), (usize, ExpfError)> {
    use magetypes::simd::generic::f32x8 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        vector8_dbg::<Token, ARITHMETIC>(token, V::load_t(token, s), op, cfg)
            .map_err(|e| (i * 8, e))?
            .store(d);
    }
    if !tail.is_empty() {
        vector8_dbg::<Token, ARITHMETIC>(token, V::load_padded_t(token, tail, tail[0]), op, cfg)
            .map_err(|e| (chunks.len() * 8, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
#[inline(always)]
fn slice16_cfg_impl<T: magetypes::simd::backends::F32x16Convert, const ARITHMETIC: usize>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
    op: u8,
    cfg: KernelCfg,
) -> Result<(), (usize, ExpfError)> {
    use magetypes::simd::generic::f32x16 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        vector16_dbg::<T, ARITHMETIC>(token, V::load_t(token, s), op, cfg)
            .map_err(|e| (i * 16, e))?
            .store(d);
    }
    if !tail.is_empty() {
        vector16_dbg::<T, ARITHMETIC>(token, V::load_padded_t(token, tail, tail[0]), op, cfg)
            .map_err(|e| (chunks.len() * 16, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

/// 16-lane AVX-512 (`V4`) whole-slice kernel parameterized on `KernelCfg`.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
#[archmage::arcane]
pub(super) fn slice16_cfg_v4<const ARITHMETIC: usize>(
    token: archmage::X64V4Token,
    src: &[f32],
    dst: &mut [f32],
    op: u8,
    cfg: KernelCfg,
) -> Result<(), (usize, ExpfError)> {
    slice16_cfg_impl::<_, ARITHMETIC>(token, src, dst, op, cfg)
}

/// 16-lane AVX-512 (`V4x`) whole-slice kernel parameterized on `KernelCfg`.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
#[archmage::arcane]
pub(super) fn slice16_cfg_v4x<const ARITHMETIC: usize>(
    token: archmage::X64V4xToken,
    src: &[f32],
    dst: &mut [f32],
    op: u8,
    cfg: KernelCfg,
) -> Result<(), (usize, ExpfError)> {
    slice16_cfg_impl::<_, ARITHMETIC>(token, src, dst, op, cfg)
}

/// Whole-slice kernel (`OP = 0`: `exp2`, `OP = 1`: `exp`): one feature region
/// around the loop, the 8-lane graph per chunk, and `tail[0]` padding.
#[cfg(not(debug_assertions))]
#[archmage::magetypes(v3, neon, wasm128, scalar)]
pub(super) fn slice8_with<K: Kernel, const OP: u8, const ARITHMETIC: usize>(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, ExpfError)> {
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

#[cfg(all(
    not(debug_assertions),
    all(target_arch = "x86_64", target_feature = "sse2"),
    feature = "avx512"
))]
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
) -> Result<(), (usize, ExpfError)> {
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
#[cfg(all(
    not(debug_assertions),
    all(target_arch = "x86_64", target_feature = "sse2"),
    feature = "avx512"
))]
#[archmage::arcane]
pub(super) fn slice16_with_v4<K: Kernel, const OP: u8, const ARITHMETIC: usize>(
    token: archmage::X64V4Token,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, ExpfError)> {
    slice16_with_impl::<_, K, OP, ARITHMETIC>(token, src, dst)
}

/// 16-lane AVX-512 (`V4x`) whole-slice kernel (`zmm`).
#[cfg(all(
    not(debug_assertions),
    all(target_arch = "x86_64", target_feature = "sse2"),
    feature = "avx512"
))]
#[archmage::arcane]
pub(super) fn slice16_with_v4x<K: Kernel, const OP: u8, const ARITHMETIC: usize>(
    token: archmage::X64V4xToken,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, ExpfError)> {
    slice16_with_impl::<_, K, OP, ARITHMETIC>(token, src, dst)
}

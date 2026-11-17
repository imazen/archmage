//! Configured exp2/exp graphs, instantiated per width, kernel config, and portability.
#![allow(clippy::excessive_precision)] // Preserve the fitted f32 coefficient spellings.
use super::{ExpfError, ExpfErrorKind};

pub(super) trait Kernel: 'static {
    const RANGED: bool;
    const IN_LO: f32;
    const IN_HI: f32;
    const FULL: bool;
    const LOWP: bool;
    const ASSUME: bool;
    const ELIDE_EXP2: bool;
    const ELIDE_EXP: bool;
}

#[derive(Clone, Copy)]
pub(super) struct KernelCfg {
    pub(super) ranged: bool,
    pub(super) in_lo: f32,
    pub(super) in_hi: f32,
    pub(super) full: bool,
    pub(super) lowp: bool,
    pub(super) assume: bool,
    pub(super) elide_exp2: bool,
    pub(super) elide_exp: bool,
}

impl KernelCfg {
    pub(super) const fn from_options(opts: super::ExpfOptions) -> Self {
        let (in_lo, in_hi) = match opts.input_bounds() {
            Some(bounds) => bounds,
            None => (0.0, 0.0),
        };
        Self {
            ranged: opts.input_bounds().is_some(),
            in_lo,
            in_hi,
            full: opts.is_full(),
            lowp: opts.is_lowp(),
            assume: opts.assume_valid(),
            elide_exp2: opts.elides_clamp(),
            elide_exp: opts.elides_exp_clamp(),
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
        full: K::FULL,
        lowp: K::LOWP,
        assume: K::ASSUME,
        elide_exp2: K::ELIDE_EXP2,
        elide_exp: K::ELIDE_EXP,
    }
}

macro_rules! eval_expf {
    (
        $V:ident,
        $I:ident,
        $token:ident,
        $input:ident,
        $op:expr,
        $backend:expr,
        $ranged:expr,
        $in_lo:expr,
        $in_hi:expr,
        $full:expr,
        $lowp:expr,
        $assume:expr,
        $elide_exp2:expr,
        $elide_exp:expr
    ) => {{
        let sp = |x| $V::splat_t($token, x);
        let ip = |x| $I::splat_t($token, x);
        let reject = |mask: $I<T>, kind| {
            let lane_mask = mask
                .to_array()
                .iter()
                .enumerate()
                .fold(0_u16, |m, (i, &v)| m | (u16::from(v != 0) << i));
            ExpfError { kind, lane_mask }
        };
        if !$assume && $ranged {
            let invalid =
                ($input.simd_ge(sp($in_lo)) & $input.simd_le(sp($in_hi))).bitcast_to_i32() ^ ip(-1);
            if invalid.any_true() {
                return Err(reject(invalid, ExpfErrorKind::InvalidInput));
            }
        }
        let arg = if $op == 1 {
            $input * sp(core::f32::consts::LOG2_E)
        } else {
            $input
        };
        if !$assume && !$ranged && !$full {
            let bits = arg.bitcast_to_i32();
            let abs = bits & ip(0x7fff_ffff_u32 as i32);
            let invalid = abs.simd_ge(ip(0x7f80_0000));
            if invalid.any_true() {
                return Err(reject(invalid, ExpfErrorKind::InvalidInput));
            }
        }
        let elide_clamp = if $op == 1 { $elide_exp } else { $elide_exp2 };
        let madd = |a: $V<T>, b: $V<T>, c: $V<T>| -> $V<T> {
            if $backend {
                a.mul_add_relaxed(b, c)
            } else {
                a.mul_add_fused(b, c)
            }
        };
        let result = if $lowp {
            let x = if !elide_clamp {
                arg.max(sp(-126.0)).min(sp(127.99))
            } else {
                arg
            };
            let xi = x.floor();
            let xf = x - xi;
            let p = madd(
                sp(0.071_916_997_023_492_97),
                xf,
                sp(0.234_336_384_644_832_92),
            );
            let p = madd(p, xf, sp(core::f32::consts::LN_2));
            let p = madd(p, xf, sp(1.0));
            let integer = if $token.as_scalar().is_some() {
                xi.to_i32()
            } else {
                xi.to_i32_round()
            };
            p * (integer + ip(127)).shl_const::<23>().bitcast_to_f32()
        } else {
            let underflow_limit = sp(-126.0);
            let overflow_limit = sp(128.0);
            let reduced = if !elide_clamp {
                arg.max(underflow_limit).min(overflow_limit)
            } else {
                arg
            };
            let xi = if elide_clamp {
                reduced.round()
            } else {
                reduced.round().min(sp(127.0))
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
            if !elide_clamp {
                let is_underflow = arg.simd_lt(underflow_limit);
                let is_overflow = arg.simd_ge(overflow_limit);
                let result = $V::blend(is_underflow, sp(0.0), result);
                $V::blend(is_overflow, sp(f32::INFINITY), result)
            } else {
                result
            }
        };
        let result = if $full {
            let nan = $input.simd_ne($input);
            let pos_inf = $input.simd_eq(sp(f32::INFINITY));
            let neg_inf = $input.simd_eq(sp(f32::NEG_INFINITY));
            let result = $V::blend(nan, sp(f32::NAN), result);
            let result = $V::blend(pos_inf, sp(f32::INFINITY), result);
            $V::blend(neg_inf, sp(0.0), result)
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
        fn $vector_dbg<T: magetypes::simd::backends::$bound>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
            op: u8,
            backend: bool,
            cfg: KernelCfg,
        ) -> Result<magetypes::simd::generic::$float<T>, ExpfError> {
            use magetypes::simd::generic::{$float as V, $int as I};
            eval_expf!(
                V,
                I,
                token,
                input,
                op,
                backend,
                cfg.ranged,
                cfg.in_lo,
                cfg.in_hi,
                cfg.full,
                cfg.lowp,
                cfg.assume,
                cfg.elide_exp2,
                cfg.elide_exp
            )
        }

        #[cfg(not(debug_assertions))]
        #[inline(always)]
        fn $vector_rel<
            T: magetypes::simd::backends::$bound,
            K: Kernel,
            const OP: u8,
            const BACKEND: bool,
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
                BACKEND,
                K::RANGED,
                K::IN_LO,
                K::IN_HI,
                K::FULL,
                K::LOWP,
                K::ASSUME,
                K::ELIDE_EXP2,
                K::ELIDE_EXP
            )
        }

        #[inline(always)]
        pub(super) fn $profile<
            T: magetypes::simd::backends::$bound,
            K: Kernel,
            const OP: u8,
            const BACKEND: bool,
        >(
            token: T,
            input: magetypes::simd::generic::$float<T>,
        ) -> Result<magetypes::simd::generic::$float<T>, ExpfError> {
            #[cfg(debug_assertions)]
            return $vector_dbg::<T>(token, input, OP, BACKEND, cfg_of::<K>());
            #[cfg(not(debug_assertions))]
            $vector_rel::<T, K, OP, BACKEND>(token, input)
        }

        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
        pub(super) fn $cfg_entry(
            token: Token,
            input: [f32; $lanes],
            op: u8,
            backend: bool,
            cfg: KernelCfg,
        ) -> Result<[f32; $lanes], ExpfError> {
            $vector_dbg::<Token>(
                token,
                magetypes::simd::generic::$float::from_array_t(token, input),
                op,
                backend,
                cfg,
            )
            .map(|result| result.to_array())
        }

        #[cfg(not(debug_assertions))]
        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
        pub(super) fn $const_entry<K: Kernel, const OP: u8, const BACKEND: bool>(
            token: Token,
            input: [f32; $lanes],
        ) -> Result<[f32; $lanes], ExpfError> {
            $profile::<Token, K, OP, BACKEND>(
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
pub(super) fn slice8_cfg(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
    op: u8,
    backend: bool,
    cfg: KernelCfg,
) -> Result<(), (usize, ExpfError)> {
    use magetypes::simd::generic::f32x8 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        vector8_dbg::<Token>(token, V::load_t(token, s), op, backend, cfg)
            .map_err(|e| (i * 8, e))?
            .store(d);
    }
    if !tail.is_empty() {
        vector8_dbg::<Token>(
            token,
            V::load_padded_t(token, tail, tail[0]),
            op,
            backend,
            cfg,
        )
        .map_err(|e| (chunks.len() * 8, e))?
        .store_prefix(out_tail);
    }
    Ok(())
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[inline(always)]
fn slice16_cfg_impl<T: magetypes::simd::backends::F32x16Convert>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
    op: u8,
    backend: bool,
    cfg: KernelCfg,
) -> Result<(), (usize, ExpfError)> {
    use magetypes::simd::generic::f32x16 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        vector16_dbg::<T>(token, V::load_t(token, s), op, backend, cfg)
            .map_err(|e| (i * 16, e))?
            .store(d);
    }
    if !tail.is_empty() {
        vector16_dbg::<T>(
            token,
            V::load_padded_t(token, tail, tail[0]),
            op,
            backend,
            cfg,
        )
        .map_err(|e| (chunks.len() * 16, e))?
        .store_prefix(out_tail);
    }
    Ok(())
}

/// 16-lane AVX-512 (`V4`) whole-slice kernel parameterized on `KernelCfg`.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
pub(super) fn slice16_cfg_v4(
    token: archmage::X64V4Token,
    src: &[f32],
    dst: &mut [f32],
    op: u8,
    backend: bool,
    cfg: KernelCfg,
) -> Result<(), (usize, ExpfError)> {
    slice16_cfg_impl(token, src, dst, op, backend, cfg)
}

/// 16-lane AVX-512 (`V4x`) whole-slice kernel parameterized on `KernelCfg`.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
pub(super) fn slice16_cfg_v4x(
    token: archmage::X64V4xToken,
    src: &[f32],
    dst: &mut [f32],
    op: u8,
    backend: bool,
    cfg: KernelCfg,
) -> Result<(), (usize, ExpfError)> {
    slice16_cfg_impl(token, src, dst, op, backend, cfg)
}

/// Whole-slice kernel (`OP = 0`: `exp2`, `OP = 1`: `exp`): one feature region
/// around the loop, the 8-lane graph per chunk, and `tail[0]` padding.
#[cfg(not(debug_assertions))]
#[archmage::magetypes(v3, neon, wasm128, scalar)]
pub(super) fn slice8_with<K: Kernel, const OP: u8, const BACKEND: bool>(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, ExpfError)> {
    use magetypes::simd::generic::f32x8 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        profile8::<Token, K, OP, BACKEND>(token, V::load_t(token, s))
            .map_err(|e| (i * 8, e))?
            .store(d);
    }
    if !tail.is_empty() {
        profile8::<Token, K, OP, BACKEND>(token, V::load_padded_t(token, tail, tail[0]))
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
    const BACKEND: bool,
>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, ExpfError)> {
    use magetypes::simd::generic::f32x16 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        profile16::<T, K, OP, BACKEND>(token, V::load_t(token, s))
            .map_err(|e| (i * 16, e))?
            .store(d);
    }
    if !tail.is_empty() {
        profile16::<T, K, OP, BACKEND>(token, V::load_padded_t(token, tail, tail[0]))
            .map_err(|e| (chunks.len() * 16, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

/// 16-lane AVX-512 (`V4`) whole-slice kernel (`zmm`).
#[cfg(all(not(debug_assertions), target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
pub(super) fn slice16_with_v4<K: Kernel, const OP: u8, const BACKEND: bool>(
    token: archmage::X64V4Token,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, ExpfError)> {
    slice16_with_impl::<_, K, OP, BACKEND>(token, src, dst)
}

/// 16-lane AVX-512 (`V4x`) whole-slice kernel (`zmm`).
#[cfg(all(not(debug_assertions), target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
pub(super) fn slice16_with_v4x<K: Kernel, const OP: u8, const BACKEND: bool>(
    token: archmage::X64V4xToken,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, ExpfError)> {
    slice16_with_impl::<_, K, OP, BACKEND>(token, src, dst)
}

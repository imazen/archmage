//! Configured logarithm graphs (`log2`, `ln`, `log10`, `log1p`), instantiated
//! per width, kernel configuration, op, and portability.
#![allow(clippy::excessive_precision)] // Preserve the fitted f32 coefficient spellings.
use super::{LogfError, LogfErrorKind};

pub(super) trait Kernel: 'static {
    const RANGED: bool;
    const IN_LO: f32;
    const IN_HI: f32;
    const SPECIALS: bool;
    const LOWP: bool;
    const ASSUME: bool;
}

#[derive(Clone, Copy)]
pub(super) struct KernelCfg {
    pub(super) ranged: bool,
    pub(super) in_lo: f32,
    pub(super) in_hi: f32,
    pub(super) specials: bool,
    pub(super) lowp: bool,
    pub(super) assume: bool,
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
            specials: opts.is_specials(),
            lowp: opts.is_lowp(),
            assume: opts.assume_valid(),
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
        specials: K::SPECIALS,
        lowp: K::LOWP,
        assume: K::ASSUME,
    }
}

macro_rules! eval_log2 {
    (
        $V:ident,
        $I:ident,
        $token:ident,
        $input:ident,
        $backend:expr,
        $ranged:expr,
        $in_lo:expr,
        $in_hi:expr,
        $specials:expr,
        $lowp:expr,
        $assume:expr
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
        let bits = $input.bitcast_to_i32();
        if !$assume {
            if $ranged {
                let invalid = ($input.simd_ge(sp($in_lo)) & $input.simd_le(sp($in_hi)))
                    .bitcast_to_i32()
                    ^ ip(-1);
                if invalid.any_true() {
                    return Err(reject(invalid, LogfErrorKind::InvalidInput));
                }
            } else if !$specials {
                let exp_bits = bits & ip(0x7f80_0000);
                let invalid = bits.simd_le(ip(0))
                    | exp_bits.simd_eq(ip(0))
                    | exp_bits.simd_eq(ip(0x7f80_0000));
                if invalid.any_true() {
                    return Err(reject(invalid, LogfErrorKind::InvalidInput));
                }
            }
        }
        let madd = |a: $V<T>, b: $V<T>, c: $V<T>| -> $V<T> {
            if $backend {
                a.mul_add_relaxed(b, c)
            } else {
                a.mul_add_fused(b, c)
            }
        };
        let result = if $lowp {
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
            let adjusted = bits + ip((0x3f80_0000_u32 - 0x3f35_04f3) as i32);
            let n = (adjusted.shr_arithmetic_const::<23>() - ip(127)).to_f32();
            let a = ((adjusted & ip(0x007f_ffff)) + ip(0x3f35_04f3)).bitcast_to_f32();
            let y = (a - sp(1.0)) / (a + sp(1.0));
            let y2 = y * y;
            let p = madd(sp(0.430_468_597_034_945_7), y2, sp(0.576_767_602_130_619_1));
            let p = madd(p, y2, sp(0.961_798_223_457_557_8));
            let p = madd(p, y2, sp(2.885_390_081_777_926_8));
            madd(p, y, n)
        };
        let result = if $specials {
            let zero = $input.simd_eq(sp(0.0));
            let nonnegative = $input.simd_ge(sp(0.0));
            let pos_inf = $input.simd_eq(sp(f32::INFINITY));
            let result = $V::blend(zero, sp(f32::NEG_INFINITY), result);
            let result = $V::blend(nonnegative, result, sp(f32::NAN));
            $V::blend(pos_inf, sp(f32::INFINITY), result)
        } else {
            result
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
        $backend:expr,
        $ranged:expr,
        $in_lo:expr,
        $in_hi:expr,
        $specials:expr,
        $lowp:expr,
        $assume:expr,
        $log2_fn:expr
    ) => {{
        let sp = |x| $V::splat_t($token, x);
        let ip = |x| $I::splat_t($token, x);
        if $op == 3 {
            if !$assume && $ranged {
                let invalid = ($input.simd_ge(sp($in_lo)) & $input.simd_le(sp($in_hi)))
                    .bitcast_to_i32()
                    ^ ip(-1);
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
            let one = sp(1.0);
            let two = sp(2.0);
            let u = $input + one;
            let finite_pos = u.simd_gt(sp(0.0)) & u.simd_lt(sp(f32::INFINITY));
            let c_raw = $V::blend(
                u.simd_ge(two),
                (one - (u - $input)) / u,
                ($input - (u - one)) / u,
            );
            let c = $V::blend(finite_pos, c_raw, sp(0.0));
            let log2_u = $log2_fn(u)?;
            let ln2 = sp(core::f32::consts::LN_2);
            let res = if $backend {
                log2_u.mul_add_relaxed(ln2, c)
            } else {
                log2_u.mul_add_fused(ln2, c)
            };
            Ok($V::blend($input.simd_eq(sp(0.0)), $input, res))
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
        fn $vector_dbg<T: magetypes::simd::backends::$bound>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
            backend: bool,
            cfg: KernelCfg,
        ) -> Result<magetypes::simd::generic::$float<T>, LogfError> {
            use magetypes::simd::generic::{$float as V, $int as I};
            eval_log2!(
                V,
                I,
                token,
                input,
                backend,
                cfg.ranged,
                cfg.in_lo,
                cfg.in_hi,
                cfg.specials,
                cfg.lowp,
                cfg.assume
            )
        }

        #[cfg(not(debug_assertions))]
        #[inline(always)]
        fn $vector_rel<
            T: magetypes::simd::backends::$bound,
            K: Kernel,
            const BACKEND: bool,
            const SKIP_U_RANGE: bool,
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
                BACKEND,
                if SKIP_U_RANGE { false } else { K::RANGED },
                K::IN_LO,
                K::IN_HI,
                K::SPECIALS,
                K::LOWP,
                if SKIP_U_RANGE {
                    K::ASSUME || K::RANGED
                } else {
                    K::ASSUME
                }
            )
        }

        #[cfg_attr(debug_assertions, inline(never))]
        #[cfg_attr(not(debug_assertions), inline)]
        fn $eval_dbg<T: magetypes::simd::backends::$bound>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
            op: u8,
            backend: bool,
            cfg: KernelCfg,
        ) -> Result<magetypes::simd::generic::$float<T>, LogfError> {
            use magetypes::simd::generic::{$float as V, $int as I};
            let u_cfg = if op == 3 && cfg.ranged {
                KernelCfg {
                    ranged: false,
                    assume: true,
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
                backend,
                cfg.ranged,
                cfg.in_lo,
                cfg.in_hi,
                cfg.specials,
                cfg.lowp,
                cfg.assume,
                |arg| $vector_dbg::<T>(token, arg, backend, u_cfg)
            )
        }

        #[cfg(not(debug_assertions))]
        #[inline(always)]
        fn $eval_rel<
            T: magetypes::simd::backends::$bound,
            K: Kernel,
            const OP: u8,
            const BACKEND: bool,
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
                BACKEND,
                K::RANGED,
                K::IN_LO,
                K::IN_HI,
                K::SPECIALS,
                K::LOWP,
                K::ASSUME,
                |arg| if OP == 3 {
                    $vector_rel::<T, K, BACKEND, true>(token, arg)
                } else {
                    $vector_rel::<T, K, BACKEND, false>(token, arg)
                }
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
        ) -> Result<magetypes::simd::generic::$float<T>, LogfError> {
            #[cfg(debug_assertions)]
            return $eval_dbg::<T>(token, input, OP, BACKEND, cfg_of::<K>());
            #[cfg(not(debug_assertions))]
            $eval_rel::<T, K, OP, BACKEND>(token, input)
        }

        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
        pub(super) fn $cfg_entry(
            token: Token,
            input: [f32; $lanes],
            op: u8,
            backend: bool,
            cfg: KernelCfg,
        ) -> Result<[f32; $lanes], LogfError> {
            $eval_dbg::<Token>(
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
        ) -> Result<[f32; $lanes], LogfError> {
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
pub(super) fn slice8_cfg(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
    op: u8,
    backend: bool,
    cfg: KernelCfg,
) -> Result<(), (usize, LogfError)> {
    use magetypes::simd::generic::f32x8 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        eval_op8_dbg::<Token>(token, V::load_t(token, s), op, backend, cfg)
            .map_err(|e| (i * 8, e))?
            .store(d);
    }
    if !tail.is_empty() {
        eval_op8_dbg::<Token>(
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
) -> Result<(), (usize, LogfError)> {
    use magetypes::simd::generic::f32x16 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        eval_op16_dbg::<T>(token, V::load_t(token, s), op, backend, cfg)
            .map_err(|e| (i * 16, e))?
            .store(d);
    }
    if !tail.is_empty() {
        eval_op16_dbg::<T>(
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
) -> Result<(), (usize, LogfError)> {
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
) -> Result<(), (usize, LogfError)> {
    slice16_cfg_impl(token, src, dst, op, backend, cfg)
}

/// Whole-slice kernel (`OP`: `0=log2`, `1=ln`, `2=log10`, `3=log1p`): one
/// feature region around the loop, the 8-lane graph per chunk, and `tail[0]`
/// padding so ranges excluding `1.0` never false-reject.
#[cfg(not(debug_assertions))]
#[archmage::magetypes(v3, neon, wasm128, scalar)]
pub(super) fn slice8_with<K: Kernel, const OP: u8, const BACKEND: bool>(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, LogfError)> {
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
) -> Result<(), (usize, LogfError)> {
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
) -> Result<(), (usize, LogfError)> {
    slice16_with_impl::<_, K, OP, BACKEND>(token, src, dst)
}

/// 16-lane AVX-512 (`V4x`) whole-slice kernel (`zmm`).
#[cfg(all(not(debug_assertions), target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
pub(super) fn slice16_with_v4x<K: Kernel, const OP: u8, const BACKEND: bool>(
    token: archmage::X64V4xToken,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, LogfError)> {
    slice16_with_impl::<_, K, OP, BACKEND>(token, src, dst)
}

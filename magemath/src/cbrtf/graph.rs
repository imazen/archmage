//! Configured cube-root (`cbrtf`) graphs with division-free exponent reduction.
use super::{CbrtfError, CbrtfErrorKind};
use crate::lanes::{F32Lanes, I32Lanes};

const MIN_NORMAL_BITS: i32 = 0x0080_0000;
const INF_BITS: i32 = 0x7f80_0000;
const MAX_BOUNDED_BITS: i32 = (f32::MAX / 3.0).to_bits() as i32;
const MAX_BOUNDED_INCLUSIVE: f32 = f32::from_bits((f32::MAX / 3.0).to_bits() - 1);
const MANTISSA_MASK: i32 = MIN_NORMAL_BITS - 1;
const CBRT_2_MANTISSA: i32 = 0x0021_4518;
const CBRT_4_MANTISSA: i32 = 0x004b_2ff5;

pub(super) trait Kernel: 'static {
    const RANGED: bool;
    const IN_LO: f32;
    const IN_HI: f32;
    const DENORM_POSSIBLE: bool;
    const SCALE_SUBNORM: bool;
    const FLUSH_SUB: bool;
    const FULL: bool;
    const CLAMP_OOB: bool;
    const PRESERVE_CLAMPED_NAN: bool;
    const LOWP: bool;
    const ASSUME: bool;
}

#[derive(Clone, Copy)]
pub(super) struct KernelCfg {
    pub(super) ranged: bool,
    pub(super) in_lo: f32,
    pub(super) in_hi: f32,
    pub(super) denorm_possible: bool,
    pub(super) scale_subnorm: bool,
    pub(super) flush_sub: bool,
    pub(super) full: bool,
    pub(super) clamp_oob: bool,
    pub(crate) preserve_clamped_nan: bool,
    pub(super) lowp: bool,
    pub(super) assume: bool,
}

impl KernelCfg {
    pub(super) const fn from_options(opts: super::CbrtfOptions) -> Self {
        let (in_lo, in_hi) = match opts.input_bounds() {
            Some(bounds) => bounds,
            None => (0.0, 0.0),
        };
        Self {
            ranged: opts.input_bounds().is_some(),
            in_lo,
            in_hi,
            denorm_possible: opts.denorm_possible(),
            scale_subnorm: opts.scale_subnormals(),
            flush_sub: opts.flushes_subnormals(),
            full: opts.is_full(),
            clamp_oob: opts.clamp_oob(),
            preserve_clamped_nan: opts.preserve_clamped_nan(),
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
        denorm_possible: K::DENORM_POSSIBLE,
        scale_subnorm: K::SCALE_SUBNORM,
        flush_sub: K::FLUSH_SUB,
        full: K::FULL,
        clamp_oob: K::CLAMP_OOB,
        preserve_clamped_nan: K::PRESERVE_CLAMPED_NAN,
        lowp: K::LOWP,
        assume: K::ASSUME,
    }
}

trait CfgSource: Copy {
    fn backend(self) -> bool;
    fn ranged(self) -> bool;
    fn in_lo(self) -> f32;
    fn in_hi(self) -> f32;
    fn denorm_possible(self) -> bool;
    fn scale_subnorm(self) -> bool;
    fn flush_sub(self) -> bool;
    fn full(self) -> bool;
    fn clamp_oob(self) -> bool;
    fn preserve_clamped_nan(self) -> bool;
    fn lowp(self) -> bool;
    fn assume(self) -> bool;
}

#[derive(Clone, Copy)]
struct RuntimeCfg {
    backend: bool,
    cfg: KernelCfg,
}

impl CfgSource for RuntimeCfg {
    #[inline(always)]
    fn backend(self) -> bool {
        self.backend
    }
    #[inline(always)]
    fn ranged(self) -> bool {
        self.cfg.ranged
    }
    #[inline(always)]
    fn in_lo(self) -> f32 {
        self.cfg.in_lo
    }
    #[inline(always)]
    fn in_hi(self) -> f32 {
        self.cfg.in_hi
    }
    #[inline(always)]
    fn denorm_possible(self) -> bool {
        self.cfg.denorm_possible
    }
    #[inline(always)]
    fn scale_subnorm(self) -> bool {
        self.cfg.scale_subnorm
    }
    #[inline(always)]
    fn flush_sub(self) -> bool {
        self.cfg.flush_sub
    }
    #[inline(always)]
    fn full(self) -> bool {
        self.cfg.full
    }
    #[inline(always)]
    fn clamp_oob(self) -> bool {
        self.cfg.clamp_oob
    }
    #[inline(always)]
    fn preserve_clamped_nan(self) -> bool {
        self.cfg.preserve_clamped_nan
    }
    #[inline(always)]
    fn lowp(self) -> bool {
        self.cfg.lowp
    }
    #[inline(always)]
    fn assume(self) -> bool {
        self.cfg.assume
    }
}

#[cfg(not(debug_assertions))]
struct ConstCfg<K: Kernel, const BACKEND: bool>(core::marker::PhantomData<K>);

#[cfg(not(debug_assertions))]
impl<K: Kernel, const BACKEND: bool> Clone for ConstCfg<K, BACKEND> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
#[cfg(not(debug_assertions))]
impl<K: Kernel, const BACKEND: bool> Copy for ConstCfg<K, BACKEND> {}

#[cfg(not(debug_assertions))]
impl<K: Kernel, const BACKEND: bool> CfgSource for ConstCfg<K, BACKEND> {
    #[inline(always)]
    fn backend(self) -> bool {
        BACKEND
    }
    #[inline(always)]
    fn ranged(self) -> bool {
        K::RANGED
    }
    #[inline(always)]
    fn in_lo(self) -> f32 {
        K::IN_LO
    }
    #[inline(always)]
    fn in_hi(self) -> f32 {
        K::IN_HI
    }
    #[inline(always)]
    fn denorm_possible(self) -> bool {
        K::DENORM_POSSIBLE
    }
    #[inline(always)]
    fn scale_subnorm(self) -> bool {
        K::SCALE_SUBNORM
    }
    #[inline(always)]
    fn flush_sub(self) -> bool {
        K::FLUSH_SUB
    }
    #[inline(always)]
    fn full(self) -> bool {
        K::FULL
    }
    #[inline(always)]
    fn clamp_oob(self) -> bool {
        K::CLAMP_OOB
    }
    #[inline(always)]
    fn preserve_clamped_nan(self) -> bool {
        K::PRESERVE_CLAMPED_NAN
    }
    #[inline(always)]
    fn lowp(self) -> bool {
        K::LOWP
    }
    #[inline(always)]
    fn assume(self) -> bool {
        K::ASSUME
    }
}

#[inline(always)]
fn eval_cbrtf<V: F32Lanes, C: CfgSource>(
    token: V::Token,
    input: V,
    cfg: C,
) -> Result<V, CbrtfError> {
    let sp = |x| V::splat(token, x);
    let ip = |x| V::I32::splat(token, x);
    let reject = |mask: V::I32, kind| CbrtfError {
        kind,
        lane_mask: mask.lane_mask(),
    };

    let (lo_b, hi_b) = if cfg.ranged() {
        (cfg.in_lo(), cfg.in_hi())
    } else {
        (-MAX_BOUNDED_INCLUSIVE, MAX_BOUNDED_INCLUSIVE)
    };
    // Bounded admits zero of either sign. Its subnormals can be flushed
    // when restoring zero, after validation and polynomial evaluation.
    let flush_bounded =
        !cfg.ranged() && !cfg.assume() && !cfg.clamp_oob() && !cfg.full() && cfg.flush_sub();
    let pre_in =
        if cfg.ranged() && !cfg.assume() && !cfg.clamp_oob() && !cfg.full() && cfg.flush_sub() {
            let in_bits = input.bitcast_to_i32();
            let in_abs = in_bits & ip(0x7fff_ffff_u32 as i32);
            let is_denorm = in_abs.simd_gt(ip(0)) & in_abs.simd_lt(ip(MIN_NORMAL_BITS));
            if (lo_b == 0.0 && hi_b == 0.0) || (lo_b < 0.0 && hi_b > 0.0) {
                V::blend(is_denorm.bitcast_to_f32(), input & sp(-0.0), input)
            } else if lo_b == 0.0 && hi_b > 0.0 {
                let pos_sub = in_bits.simd_gt(ip(0)) & in_bits.simd_lt(ip(MIN_NORMAL_BITS));
                V::blend(pos_sub.bitcast_to_f32(), sp(0.0), input)
            } else if hi_b == 0.0 && lo_b < 0.0 {
                let neg_sub = in_bits.simd_lt(ip(0)) & in_abs.simd_lt(ip(MIN_NORMAL_BITS));
                V::blend(neg_sub.bitcast_to_f32(), sp(-0.0), input)
            } else if lo_b > 0.0 {
                let pos_sub = in_bits.simd_gt(ip(0)) & in_bits.simd_lt(ip(MIN_NORMAL_BITS));
                V::blend(
                    pos_sub.bitcast_to_f32(),
                    sp(lo_b.max(f32::MIN_POSITIVE)),
                    input,
                )
            } else {
                let neg_sub = in_bits.simd_lt(ip(0)) & in_abs.simd_lt(ip(MIN_NORMAL_BITS));
                V::blend(
                    neg_sub.bitcast_to_f32(),
                    sp(hi_b.min(-f32::MIN_POSITIVE)),
                    input,
                )
            }
        } else {
            input
        };

    if !cfg.assume() && !cfg.clamp_oob() {
        let bits = pre_in.bitcast_to_i32();
        let abs_bits = bits & ip(0x7fff_ffff_u32 as i32);
        if cfg.ranged() {
            let outside = (pre_in.simd_ge(sp(cfg.in_lo())) & pre_in.simd_le(sp(cfg.in_hi())))
                .bitcast_to_i32()
                ^ ip(-1);
            let invalid = if cfg.denorm_possible() {
                outside | (abs_bits.simd_gt(ip(0)) & abs_bits.simd_lt(ip(MIN_NORMAL_BITS)))
            } else {
                outside
            };
            if invalid.any_true() {
                return Err(reject(invalid, CbrtfErrorKind::InvalidInput));
            }
        } else if !cfg.full() {
            let is_too_large_or_nonfinite = abs_bits.simd_ge(ip(MAX_BOUNDED_BITS));
            let invalid = if cfg.denorm_possible() && !flush_bounded {
                let is_denorm = abs_bits.simd_gt(ip(0)) & abs_bits.simd_lt(ip(MIN_NORMAL_BITS));
                is_denorm | is_too_large_or_nonfinite
            } else {
                is_too_large_or_nonfinite
            };
            if invalid.any_true() {
                return Err(reject(invalid, CbrtfErrorKind::InvalidInput));
            }
        }
    }

    let is_nan = if cfg.preserve_clamped_nan() && !cfg.full() {
        input.simd_ne(input)
    } else {
        ip(0).bitcast_to_f32()
    };
    let base_in = if cfg.clamp_oob() && !cfg.full() {
        let mut c = V::blend(
            input.simd_lt(sp(lo_b)),
            sp(lo_b),
            V::blend(input.simd_gt(sp(hi_b)), sp(hi_b), input),
        );
        if cfg.denorm_possible() {
            let c_abs = c.bitcast_to_i32() & ip(0x7fff_ffff_u32 as i32);
            let c_denorm =
                (c_abs.simd_gt(ip(0)) & c_abs.simd_lt(ip(MIN_NORMAL_BITS))).bitcast_to_f32();
            let sub_target = if cfg.ranged() && lo_b > 0.0 && lo_b < f32::MIN_POSITIVE {
                sp(f32::MIN_POSITIVE)
            } else if cfg.ranged() && hi_b < 0.0 && hi_b > -f32::MIN_POSITIVE {
                sp(-f32::MIN_POSITIVE)
            } else {
                c & sp(-0.0)
            };
            c = V::blend(c_denorm, sub_target, c);
        }
        let safe = if cfg.ranged() {
            if cfg.denorm_possible() && lo_b > 0.0 && lo_b < f32::MIN_POSITIVE {
                f32::MIN_POSITIVE
            } else if cfg.denorm_possible() && hi_b < 0.0 && hi_b > -f32::MIN_POSITIVE {
                -f32::MIN_POSITIVE
            } else {
                lo_b
            }
        } else {
            0.0
        };
        if cfg.preserve_clamped_nan() {
            V::blend(is_nan, sp(safe), c)
        } else {
            c
        }
    } else {
        pre_in
    };

    let bits = base_in.bitcast_to_i32();
    let abs_bits = bits & ip(i32::MAX);
    let tiny = abs_bits.simd_lt(ip(MIN_NORMAL_BITS));
    let (work_bits, scale_adjust) = if (cfg.full() || cfg.scale_subnorm()) && tiny.any_true() {
        // Putting the fraction bits at 1.0 and subtracting 1.0 gives
        // |x| * 2^126 exactly. Subtract 42 from the output exponent.
        let fraction = ((abs_bits & ip(MANTISSA_MASK)) | ip(1.0f32.to_bits() as i32))
            .bitcast_to_f32()
            - sp(1.0);
        (
            V::I32::blend(tiny, fraction.bitcast_to_i32(), abs_bits),
            tiny & ip(42),
        )
    } else {
        (abs_bits, ip(0))
    };

    // For n = biased exponent + 2, write n = 3q + r. Then 1366*n
    // equals 4096*q + (2*q + 1366*r). For n <= 257, the low 12
    // bits encode r = 0, 1, or 2 in bits 10 and 11. The upper bits
    // encode q, biased by 127 - 43 for the output scale.
    let biased_exponent = work_bits.shr_logical_const::<23>();
    let product = biased_exponent * ip(1366) + ip(((127 - 43) << 12) + 2 * 1366);
    let mantissa = ((product & ip(0x0400)).simd_ne(ip(0)) & ip(CBRT_2_MANTISSA))
        | ((product & ip(0x0800)).simd_ne(ip(0)) & ip(CBRT_4_MANTISSA));
    let exponent = (product.shl_const::<11>() & ip(INF_BITS)) - scale_adjust.shl_const::<23>();
    let scale = (exponent | mantissa | (bits & ip(i32::MIN))).bitcast_to_f32();

    let madd = |a: V, b: V, c: V| {
        if cfg.backend() {
            a.mul_add_relaxed(b, c)
        } else {
            a.mul_add_fused(b, c)
        }
    };
    let m = ((work_bits & ip(MANTISSA_MASK)) | ip(1.0f32.to_bits() as i32)).bitcast_to_f32();
    let delta = m - sp(1.0);
    // Weighted minimax fits for (cbrt(m) - 1) / (m - 1). The factor
    // m - 1 anchors both graphs at exact cube roots of powers of eight.
    let mut result = if cfg.lowp() {
        // Horner in delta keeps the polynomial independent of the scale.
        let p23 = madd(sp(-0.010_966_721), delta, sp(0.044_462_245));
        let p123 = madd(p23, delta, sp(-0.106_520_32));
        let p = madd(p123, delta, sp(0.332_932_38));
        madd(delta, p, sp(1.0)) * scale
    } else {
        // Estrin in the centered mantissa. The final FMA scales the
        // correction before rounding, without rounding cbrt(m) first.
        let t = m - sp(1.5);
        let t2 = t * t;
        let p01 = madd(sp(-0.070_095_22), t, sp(0.289_428_47));
        let p23 = madd(sp(-0.012_384_874), t, sp(0.027_132_858));
        let p45 = madd(sp(-0.003_293_586_4), t, sp(0.006_144_125));
        let p67 = madd(sp(-0.000_931_472_75), t, sp(0.002_000_763_8));
        let p03 = madd(p23, t2, p01);
        let p47 = madd(p67, t2, p45);
        let p = madd(p47, t2 * t2, p03);
        madd(delta * scale, p, scale)
    };

    if cfg.full() {
        let keep = (abs_bits.simd_eq(ip(0)) | abs_bits.simd_eq(ip(INF_BITS))).bitcast_to_f32();
        let nan = abs_bits.simd_gt(ip(INF_BITS)).bitcast_to_f32();
        result = V::blend(keep, base_in, result);
        result = V::blend(nan, sp(f32::NAN), result);
    } else if flush_bounded {
        result = V::blend(tiny.bitcast_to_f32(), base_in & sp(-0.0), result);
    } else if cfg.scale_subnorm() || !cfg.ranged() || (cfg.in_lo() <= 0.0 && cfg.in_hi() >= 0.0) {
        let is_zero = abs_bits.simd_eq(ip(0)).bitcast_to_f32();
        result = V::blend(is_zero, base_in, result);
    }
    if cfg.preserve_clamped_nan() && !cfg.full() {
        result = V::blend(is_nan, sp(f32::NAN), result);
    }

    Ok(result)
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline)]
fn eval_dbg<V: F32Lanes>(
    token: V::Token,
    input: V,
    backend: bool,
    cfg: KernelCfg,
) -> Result<V, CbrtfError> {
    eval_cbrtf(token, input, RuntimeCfg { backend, cfg })
}

#[inline(always)]
fn eval_profile<V: F32Lanes, K: Kernel, const BACKEND: bool>(
    token: V::Token,
    input: V,
) -> Result<V, CbrtfError> {
    #[cfg(debug_assertions)]
    return eval_dbg(token, input, BACKEND, cfg_of::<K>());
    #[cfg(not(debug_assertions))]
    eval_cbrtf(
        token,
        input,
        ConstCfg::<K, BACKEND>(core::marker::PhantomData),
    )
}

macro_rules! kernel_entries {
    ($profile:ident, $cfg_entry:ident, $const_entry:ident, $float:ident, $bound:ident, $lanes:literal) => {
        #[inline(always)]
        pub(super) fn $profile<
            T: magetypes::simd::backends::$bound,
            K: Kernel,
            const BACKEND: bool,
        >(
            token: T,
            input: magetypes::simd::generic::$float<T>,
        ) -> Result<magetypes::simd::generic::$float<T>, CbrtfError> {
            eval_profile::<_, K, BACKEND>(token, input)
        }

        #[archmage::attune(_*_t, +v4(cfg(avx512)), +v4x(cfg(avx512)))]
        pub(super) fn $cfg_entry(
            token: Token,
            input: [f32; $lanes],
            backend: bool,
            cfg: KernelCfg,
        ) -> Result<[f32; $lanes], CbrtfError> {
            eval_dbg(
                token,
                magetypes::simd::generic::$float::from_array_t(token, input),
                backend,
                cfg,
            )
            .map(|result| result.to_array())
        }

        #[cfg(not(debug_assertions))]
        #[archmage::attune(_*_t, +v4(cfg(avx512)), +v4x(cfg(avx512)))]
        pub(super) fn $const_entry<K: Kernel, const BACKEND: bool>(
            token: Token,
            input: [f32; $lanes],
        ) -> Result<[f32; $lanes], CbrtfError> {
            eval_profile::<_, K, BACKEND>(
                token,
                magetypes::simd::generic::$float::from_array_t(token, input),
            )
            .map(|result| result.to_array())
        }
    };
}

kernel_entries!(profile4, kernel4_cfg, kernel4_with, f32x4, F32x4Convert, 4);
kernel_entries!(profile8, kernel8_cfg, kernel8_with, f32x8, F32x8Convert, 8);
#[cfg(feature = "w512")]
kernel_entries!(
    profile16,
    kernel16_cfg,
    kernel16_with,
    f32x16,
    F32x16Convert,
    16
);

#[inline(always)]
pub(super) fn kernel1_cfg(input: f32, backend: bool, cfg: KernelCfg) -> Result<f32, CbrtfError> {
    use crate::scalar1::ScalarF32;
    use archmage::ScalarToken;
    eval_dbg(
        ScalarToken,
        ScalarF32::splat_t(ScalarToken, input),
        backend,
        cfg,
    )
    .map(|value| value.0)
}

#[inline(always)]
pub(super) fn kernel1_with<K: Kernel, const BACKEND: bool>(input: f32) -> Result<f32, CbrtfError> {
    use crate::scalar1::ScalarF32;
    use archmage::ScalarToken;
    eval_profile::<_, K, BACKEND>(ScalarToken, ScalarF32::splat_t(ScalarToken, input))
        .map(|value| value.0)
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline)]
pub(super) fn slice1_cfg_scalar(
    src: &[f32],
    dst: &mut [f32],
    backend: bool,
    cfg: KernelCfg,
) -> Result<(), (usize, CbrtfError)> {
    for (i, (&x, d)) in src.iter().zip(dst.iter_mut()).enumerate() {
        *d = kernel1_cfg(x, backend, cfg).map_err(|e| (i, e))?;
    }
    Ok(())
}

#[cfg(not(debug_assertions))]
#[inline]
pub(super) fn slice1_with_scalar<K: Kernel, const BACKEND: bool>(
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, CbrtfError)> {
    for (i, (&x, d)) in src.iter().zip(dst.iter_mut()).enumerate() {
        *d = kernel1_with::<K, BACKEND>(x).map_err(|e| (i, e))?;
    }
    Ok(())
}

/// Whole-slice kernel parameterized on `KernelCfg`.
#[archmage::attune(_v3_t, _neon_t)]
pub(super) fn slice8_cfg(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
    backend: bool,
    cfg: KernelCfg,
) -> Result<(), (usize, CbrtfError)> {
    use magetypes::simd::generic::f32x8 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        eval_dbg(token, V::load_t(token, s), backend, cfg)
            .map_err(|e| (i * 8, e))?
            .store(d);
    }
    if !tail.is_empty() {
        eval_dbg(token, V::load_padded_t(token, tail, tail[0]), backend, cfg)
            .map_err(|e| (chunks.len() * 8, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

#[archmage::attune(_wasm128_t)]
pub(super) fn slice4_cfg(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
    relaxed: bool,
    cfg: KernelCfg,
) -> Result<(), (usize, CbrtfError)> {
    use magetypes::simd::generic::f32x4 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        eval_dbg(token, V::load_t(token, s), relaxed, cfg)
            .map_err(|e| (i * 4, e))?
            .store(d);
    }
    if !tail.is_empty() {
        eval_dbg(token, V::load_padded_t(token, tail, tail[0]), relaxed, cfg)
            .map_err(|e| (chunks.len() * 4, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

/// 16-lane AVX-512 (`V4`/`V4x`) whole-slice kernel parameterized on `KernelCfg`.
#[archmage::attune(_v4_t(cfg(avx512)), _v4x_t(cfg(avx512)))]
pub(super) fn slice16_cfg(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
    backend: bool,
    cfg: KernelCfg,
) -> Result<(), (usize, CbrtfError)> {
    use magetypes::simd::generic::f32x16 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        eval_dbg(token, V::load_t(token, s), backend, cfg)
            .map_err(|e| (i * 16, e))?
            .store(d);
    }
    if !tail.is_empty() {
        eval_dbg(token, V::load_padded_t(token, tail, tail[0]), backend, cfg)
            .map_err(|e| (chunks.len() * 16, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

/// Whole-slice kernel: one feature region around the loop, the 8-lane graph
/// per chunk, and `tail[0]` padding.
#[cfg(not(debug_assertions))]
#[archmage::attune(_v3_t, _neon_t)]
pub(super) fn slice8_with<K: Kernel, const BACKEND: bool>(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, CbrtfError)> {
    use magetypes::simd::generic::f32x8 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        eval_profile::<_, K, BACKEND>(token, V::load_t(token, s))
            .map_err(|e| (i * 8, e))?
            .store(d);
    }
    if !tail.is_empty() {
        eval_profile::<_, K, BACKEND>(token, V::load_padded_t(token, tail, tail[0]))
            .map_err(|e| (chunks.len() * 8, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

#[cfg(not(debug_assertions))]
#[archmage::attune(_wasm128_t)]
pub(super) fn slice4_with<K: Kernel, const RELAXED: bool>(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, CbrtfError)> {
    use magetypes::simd::generic::f32x4 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        eval_profile::<_, K, RELAXED>(token, V::load_t(token, s))
            .map_err(|e| (i * 4, e))?
            .store(d);
    }
    if !tail.is_empty() {
        eval_profile::<_, K, RELAXED>(token, V::load_padded_t(token, tail, tail[0]))
            .map_err(|e| (chunks.len() * 4, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

/// 16-lane AVX-512 (`V4`/`V4x`) whole-slice kernel (`zmm`).
#[cfg(not(debug_assertions))]
#[archmage::attune(_v4_t(cfg(avx512)), _v4x_t(cfg(avx512)))]
pub(super) fn slice16_with<K: Kernel, const BACKEND: bool>(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, CbrtfError)> {
    use magetypes::simd::generic::f32x16 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        eval_profile::<_, K, BACKEND>(token, V::load_t(token, s))
            .map_err(|e| (i * 16, e))?
            .store(d);
    }
    if !tail.is_empty() {
        eval_profile::<_, K, BACKEND>(token, V::load_padded_t(token, tail, tail[0]))
            .map_err(|e| (chunks.len() * 16, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

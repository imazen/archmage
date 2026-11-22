//! Configured exp2/exp graphs, instantiated per width, kernel config, and arithmetic mode.
#![allow(clippy::excessive_precision)] // Preserve the fitted f32 coefficient spellings.
use super::{ExpfError, ExpfErrorKind};
#[cfg(target_arch = "wasm32")]
pub(super) use crate::lanes::arith::WASM_SOFTWARE;
pub(super) use crate::lanes::arith::{NATIVE, SOFTWARE};
use crate::lanes::{F32Lanes, I32Lanes, madd};
use archmage::IntoConcreteToken as _;

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

trait CfgSource: Copy {
    fn op(self) -> u8;
    fn ranged(self) -> bool;
    fn in_lo_exp2(self) -> f32;
    fn in_hi_exp2(self) -> f32;
    fn in_lo_exp(self) -> f32;
    fn in_hi_exp(self) -> f32;
    fn full(self) -> bool;
    fn clamp_oob(self) -> bool;
    fn preserve_clamped_nan(self) -> bool;
    fn lowp(self) -> bool;
    fn assume(self) -> bool;
    fn elide_exp2(self) -> bool;
    fn elide_exp(self) -> bool;
    fn elide_overflow_exp2(self) -> bool;
    fn elide_overflow_exp(self) -> bool;
    fn elide_red_exp2(self) -> bool;
    fn elide_red_exp(self) -> bool;
}

#[derive(Clone, Copy)]
struct RuntimeCfg {
    op: u8,
    cfg: KernelCfg,
}

impl CfgSource for RuntimeCfg {
    #[inline(always)]
    fn op(self) -> u8 {
        self.op
    }
    #[inline(always)]
    fn ranged(self) -> bool {
        self.cfg.ranged
    }
    #[inline(always)]
    fn in_lo_exp2(self) -> f32 {
        self.cfg.in_lo_exp2
    }
    #[inline(always)]
    fn in_hi_exp2(self) -> f32 {
        self.cfg.in_hi_exp2
    }
    #[inline(always)]
    fn in_lo_exp(self) -> f32 {
        self.cfg.in_lo_exp
    }
    #[inline(always)]
    fn in_hi_exp(self) -> f32 {
        self.cfg.in_hi_exp
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
    #[inline(always)]
    fn elide_exp2(self) -> bool {
        self.cfg.elide_exp2
    }
    #[inline(always)]
    fn elide_exp(self) -> bool {
        self.cfg.elide_exp
    }
    #[inline(always)]
    fn elide_overflow_exp2(self) -> bool {
        self.cfg.elide_overflow_exp2
    }
    #[inline(always)]
    fn elide_overflow_exp(self) -> bool {
        self.cfg.elide_overflow_exp
    }
    #[inline(always)]
    fn elide_red_exp2(self) -> bool {
        self.cfg.elide_red_exp2
    }
    #[inline(always)]
    fn elide_red_exp(self) -> bool {
        self.cfg.elide_red_exp
    }
}

#[cfg(not(debug_assertions))]
struct ConstCfg<K: Kernel, const OP: u8>(core::marker::PhantomData<K>);

#[cfg(not(debug_assertions))]
impl<K: Kernel, const OP: u8> Clone for ConstCfg<K, OP> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
#[cfg(not(debug_assertions))]
impl<K: Kernel, const OP: u8> Copy for ConstCfg<K, OP> {}

#[cfg(not(debug_assertions))]
impl<K: Kernel, const OP: u8> CfgSource for ConstCfg<K, OP> {
    #[inline(always)]
    fn op(self) -> u8 {
        OP
    }
    #[inline(always)]
    fn ranged(self) -> bool {
        K::RANGED
    }
    #[inline(always)]
    fn in_lo_exp2(self) -> f32 {
        K::IN_LO_EXP2
    }
    #[inline(always)]
    fn in_hi_exp2(self) -> f32 {
        K::IN_HI_EXP2
    }
    #[inline(always)]
    fn in_lo_exp(self) -> f32 {
        K::IN_LO_EXP
    }
    #[inline(always)]
    fn in_hi_exp(self) -> f32 {
        K::IN_HI_EXP
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
    #[inline(always)]
    fn elide_exp2(self) -> bool {
        K::ELIDE_EXP2
    }
    #[inline(always)]
    fn elide_exp(self) -> bool {
        K::ELIDE_EXP
    }
    #[inline(always)]
    fn elide_overflow_exp2(self) -> bool {
        K::ELIDE_OVERFLOW_EXP2
    }
    #[inline(always)]
    fn elide_overflow_exp(self) -> bool {
        K::ELIDE_OVERFLOW_EXP
    }
    #[inline(always)]
    fn elide_red_exp2(self) -> bool {
        K::ELIDE_RED_EXP2
    }
    #[inline(always)]
    fn elide_red_exp(self) -> bool {
        K::ELIDE_RED_EXP
    }
}

// Relative minimax cubic, constrained to 1 and 2 at the octave endpoints.
pub(crate) const LOWP_COEFFICIENTS: [f32; 3] = [0.6954243475, 0.2263076823, 0.07826797018];

/// Evaluate the low-precision `2^x` octave polynomial on `x in [0, 1]`.
#[inline(always)]
pub(crate) fn exp2_lowp<V: F32Lanes, const ARITHMETIC: usize>(token: V::Token, x: V) -> V {
    let sp = |v| V::splat(token, v);
    let [c1, c2, c3] = LOWP_COEFFICIENTS;
    let p = madd::<V, ARITHMETIC>(token, sp(c3), x, sp(c2));
    let p = madd::<V, ARITHMETIC>(token, p, x, sp(c1));
    madd::<V, ARITHMETIC>(token, p, x, sp(1.0))
}

#[inline(always)]
fn eval_expf<V: F32Lanes, C: CfgSource, const ARITHMETIC: usize>(
    token: V::Token,
    input: V,
    cfg: C,
) -> Result<V, ExpfError> {
    let sp = |x| V::splat(token, x);
    let ip = |x| V::I32::splat(token, x);
    let op = cfg.op();
    let in_lo = if op == 1 {
        cfg.in_lo_exp()
    } else {
        cfg.in_lo_exp2()
    };
    let in_hi = if op == 1 {
        cfg.in_hi_exp()
    } else {
        cfg.in_hi_exp2()
    };
    let reject = |mask: V::I32, kind| ExpfError {
        kind,
        lane_mask: mask.lane_mask(),
    };
    if !cfg.assume() && !cfg.clamp_oob() {
        if cfg.ranged() {
            let invalid =
                (input.simd_ge(sp(in_lo)) & input.simd_le(sp(in_hi))).bitcast_to_i32() ^ ip(-1);
            if invalid.any_true() {
                return Err(reject(invalid, ExpfErrorKind::InvalidInput));
            }
        } else if !cfg.full() {
            let bits = input.bitcast_to_i32();
            let abs = bits & ip(0x7fff_ffff_u32 as i32);
            let invalid = abs.simd_ge(ip(0x7f80_0000));
            if invalid.any_true() {
                return Err(reject(invalid, ExpfErrorKind::InvalidInput));
            }
        }
    }
    let is_nan = if cfg.preserve_clamped_nan() && !cfg.full() {
        input.simd_ne(input)
    } else {
        ip(0).bitcast_to_f32()
    };
    let work_in = if cfg.clamp_oob() && !cfg.full() {
        let (lo_b, hi_b, safe) = if cfg.ranged() {
            (in_lo, in_hi, in_lo)
        } else {
            (-f32::MAX, f32::MAX, 0.0)
        };
        let c = V::blend(
            input.simd_lt(sp(lo_b)),
            sp(lo_b),
            V::blend(input.simd_gt(sp(hi_b)), sp(hi_b), input),
        );
        if cfg.preserve_clamped_nan() {
            V::blend(is_nan, sp(safe), c)
        } else {
            c
        }
    } else {
        input
    };
    let arg = if op == 1 {
        work_in * sp(core::f32::consts::LOG2_E)
    } else {
        work_in
    };
    let elide_clamp = if op == 1 {
        cfg.elide_exp()
    } else {
        cfg.elide_exp2()
    };
    let elide_overflow = if op == 1 {
        cfg.elide_overflow_exp()
    } else {
        cfg.elide_overflow_exp2()
    };
    let elide_red = if op == 1 {
        cfg.elide_red_exp()
    } else {
        cfg.elide_red_exp2()
    };
    let max_lower = |x: V, lo| {
        if cfg.clamp_oob() && !cfg.full() && !cfg.preserve_clamped_nan() {
            x.max(lo)
        } else {
            x.max_fast(lo)
        }
    };
    let result = if cfg.lowp() {
        if elide_red {
            exp2_lowp::<V, ARITHMETIC>(token, arg)
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
            let p = exp2_lowp::<V, ARITHMETIC>(token, xf);
            let integer = if token.as_scalar().is_some() {
                xi.to_i32()
            } else {
                xi.to_i32_round()
            };
            p * (integer + ip(127)).shl_const::<23>().bitcast_to_f32()
        }
    } else if elide_red {
        let xf = arg;
        let p = madd::<V, ARITHMETIC>(
            token,
            sp(0.000_152_527_896_498_79),
            xf,
            sp(0.001_338_604_727_826_21),
        );
        let p = madd::<V, ARITHMETIC>(token, p, xf, sp(0.009_618_715_779_149_75));
        let p = madd::<V, ARITHMETIC>(token, p, xf, sp(0.055_503_695_542_993_11));
        let p = madd::<V, ARITHMETIC>(token, p, xf, sp(0.240_226_463_934_714_0));
        let p = madd::<V, ARITHMETIC>(token, p, xf, sp(core::f32::consts::LN_2));
        madd::<V, ARITHMETIC>(token, p, xf, sp(1.0))
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
        let p = madd::<V, ARITHMETIC>(
            token,
            sp(0.000_152_527_896_498_79),
            xf,
            sp(0.001_338_604_727_826_21),
        );
        let p = madd::<V, ARITHMETIC>(token, p, xf, sp(0.009_618_715_779_149_75));
        let p = madd::<V, ARITHMETIC>(token, p, xf, sp(0.055_503_695_542_993_11));
        let p = madd::<V, ARITHMETIC>(token, p, xf, sp(0.240_226_463_934_714_0));
        let p = madd::<V, ARITHMETIC>(token, p, xf, sp(core::f32::consts::LN_2));
        let p = madd::<V, ARITHMETIC>(token, p, xf, sp(1.0));
        let integer = if token.as_scalar().is_some() {
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
            V::blend(is_underflow, sp(0.0), result)
        } else {
            let is_underflow = arg.simd_lt(underflow_limit);
            let is_overflow = arg.simd_ge(overflow_limit);
            let result = V::blend(is_underflow, sp(0.0), result);
            V::blend(is_overflow, sp(f32::INFINITY), result)
        }
    };
    let result = if cfg.full() {
        let nan = input.simd_ne(input);
        let pos_inf = input.simd_eq(sp(f32::INFINITY));
        let neg_inf = input.simd_eq(sp(f32::NEG_INFINITY));
        let result = V::blend(nan, sp(f32::NAN), result);
        let result = V::blend(pos_inf, sp(f32::INFINITY), result);
        V::blend(neg_inf, sp(0.0), result)
    } else if cfg.preserve_clamped_nan() {
        V::blend(is_nan, sp(f32::NAN), result)
    } else {
        result
    };
    Ok(result)
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline)]
fn eval_dbg<V: F32Lanes, const ARITHMETIC: usize>(
    token: V::Token,
    input: V,
    op: u8,
    cfg: KernelCfg,
) -> Result<V, ExpfError> {
    eval_expf::<V, RuntimeCfg, ARITHMETIC>(token, input, RuntimeCfg { op, cfg })
}

#[inline(always)]
fn eval_profile<V: F32Lanes, K: Kernel, const OP: u8, const ARITHMETIC: usize>(
    token: V::Token,
    input: V,
) -> Result<V, ExpfError> {
    #[cfg(debug_assertions)]
    return eval_dbg::<V, ARITHMETIC>(token, input, OP, cfg_of::<K>());
    #[cfg(not(debug_assertions))]
    eval_expf::<V, ConstCfg<K, OP>, ARITHMETIC>(
        token,
        input,
        ConstCfg::<K, OP>(core::marker::PhantomData),
    )
}

macro_rules! kernel_entries {
    ($profile:ident, $cfg_entry:ident, $const_entry:ident, $float:ident, $bound:ident, $lanes:literal) => {
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
            eval_profile::<_, K, OP, ARITHMETIC>(token, input)
        }

        #[archmage::attune(_*_t, +v4(cfg(avx512)), +v4x(cfg(avx512)))]
        pub(super) fn $cfg_entry<const ARITHMETIC: usize>(
            token: Token,
            input: [f32; $lanes],
            op: u8,
            cfg: KernelCfg,
        ) -> Result<[f32; $lanes], ExpfError> {
            eval_dbg::<_, ARITHMETIC>(
                token,
                magetypes::simd::generic::$float::from_array_t(token, input),
                op,
                cfg,
            )
            .map(|result| result.to_array())
        }

        #[cfg(not(debug_assertions))]
        #[archmage::attune(_*_t, +v4(cfg(avx512)), +v4x(cfg(avx512)))]
        pub(super) fn $const_entry<K: Kernel, const OP: u8, const ARITHMETIC: usize>(
            token: Token,
            input: [f32; $lanes],
        ) -> Result<[f32; $lanes], ExpfError> {
            eval_profile::<_, K, OP, ARITHMETIC>(
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
pub(super) fn kernel1_cfg<const ARITHMETIC: usize>(
    input: f32,
    op: u8,
    cfg: KernelCfg,
) -> Result<f32, ExpfError> {
    use crate::scalar1::ScalarF32;
    use archmage::ScalarToken;
    eval_dbg::<_, ARITHMETIC>(ScalarToken, ScalarF32::splat_t(ScalarToken, input), op, cfg)
        .map(|value| value.0)
}

#[inline(always)]
pub(super) fn kernel1_with<K: Kernel, const OP: u8, const ARITHMETIC: usize>(
    input: f32,
) -> Result<f32, ExpfError> {
    use crate::scalar1::ScalarF32;
    use archmage::ScalarToken;
    eval_profile::<_, K, OP, ARITHMETIC>(ScalarToken, ScalarF32::splat_t(ScalarToken, input))
        .map(|value| value.0)
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline)]
pub(super) fn slice1_cfg_scalar<const ARITHMETIC: usize>(
    src: &[f32],
    dst: &mut [f32],
    op: u8,
    cfg: KernelCfg,
) -> Result<(), (usize, ExpfError)> {
    for (i, (&x, d)) in src.iter().zip(dst.iter_mut()).enumerate() {
        *d = kernel1_cfg::<ARITHMETIC>(x, op, cfg).map_err(|e| (i, e))?;
    }
    Ok(())
}

#[cfg(not(debug_assertions))]
#[inline]
pub(super) fn slice1_with_scalar<K: Kernel, const OP: u8, const ARITHMETIC: usize>(
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, ExpfError)> {
    for (i, (&x, d)) in src.iter().zip(dst.iter_mut()).enumerate() {
        *d = kernel1_with::<K, OP, ARITHMETIC>(x).map_err(|e| (i, e))?;
    }
    Ok(())
}

/// Whole-slice kernel (`op = 0`: `exp2`, `op = 1`: `exp`) parameterized on `KernelCfg`.
#[archmage::attune(_v3_t, _neon_t)]
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
        eval_dbg::<_, ARITHMETIC>(token, V::load_t(token, s), op, cfg)
            .map_err(|e| (i * 8, e))?
            .store(d);
    }
    if !tail.is_empty() {
        eval_dbg::<_, ARITHMETIC>(token, V::load_padded_t(token, tail, tail[0]), op, cfg)
            .map_err(|e| (chunks.len() * 8, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

#[archmage::attune(_wasm128_t)]
pub(super) fn slice4_cfg<const ARITHMETIC: usize>(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
    op: u8,
    cfg: KernelCfg,
) -> Result<(), (usize, ExpfError)> {
    use magetypes::simd::generic::f32x4 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        eval_dbg::<_, ARITHMETIC>(token, V::load_t(token, s), op, cfg)
            .map_err(|e| (i * 4, e))?
            .store(d);
    }
    if !tail.is_empty() {
        eval_dbg::<_, ARITHMETIC>(token, V::load_padded_t(token, tail, tail[0]), op, cfg)
            .map_err(|e| (chunks.len() * 4, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

/// 16-lane AVX-512 (`V4`/`V4x`) whole-slice kernel parameterized on `KernelCfg`.
#[archmage::attune(_v4_t(cfg(avx512)), _v4x_t(cfg(avx512)))]
pub(super) fn slice16_cfg<const ARITHMETIC: usize>(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
    op: u8,
    cfg: KernelCfg,
) -> Result<(), (usize, ExpfError)> {
    use magetypes::simd::generic::f32x16 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        eval_dbg::<_, ARITHMETIC>(token, V::load_t(token, s), op, cfg)
            .map_err(|e| (i * 16, e))?
            .store(d);
    }
    if !tail.is_empty() {
        eval_dbg::<_, ARITHMETIC>(token, V::load_padded_t(token, tail, tail[0]), op, cfg)
            .map_err(|e| (chunks.len() * 16, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

/// Whole-slice kernel (`OP = 0`: `exp2`, `OP = 1`: `exp`): one feature region
/// around the loop, the 8-lane graph per chunk, and `tail[0]` padding.
#[cfg(not(debug_assertions))]
#[archmage::attune(_v3_t, _neon_t)]
pub(super) fn slice8_with<K: Kernel, const OP: u8, const ARITHMETIC: usize>(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, ExpfError)> {
    use magetypes::simd::generic::f32x8 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        eval_profile::<_, K, OP, ARITHMETIC>(token, V::load_t(token, s))
            .map_err(|e| (i * 8, e))?
            .store(d);
    }
    if !tail.is_empty() {
        eval_profile::<_, K, OP, ARITHMETIC>(token, V::load_padded_t(token, tail, tail[0]))
            .map_err(|e| (chunks.len() * 8, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

#[cfg(not(debug_assertions))]
#[archmage::attune(_wasm128_t)]
pub(super) fn slice4_with<K: Kernel, const OP: u8, const ARITHMETIC: usize>(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, ExpfError)> {
    use magetypes::simd::generic::f32x4 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        eval_profile::<_, K, OP, ARITHMETIC>(token, V::load_t(token, s))
            .map_err(|e| (i * 4, e))?
            .store(d);
    }
    if !tail.is_empty() {
        eval_profile::<_, K, OP, ARITHMETIC>(token, V::load_padded_t(token, tail, tail[0]))
            .map_err(|e| (chunks.len() * 4, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

/// 16-lane AVX-512 (`V4`/`V4x`) whole-slice kernel (`zmm`).
#[cfg(not(debug_assertions))]
#[archmage::attune(_v4_t(cfg(avx512)), _v4x_t(cfg(avx512)))]
pub(super) fn slice16_with<K: Kernel, const OP: u8, const ARITHMETIC: usize>(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, ExpfError)> {
    use magetypes::simd::generic::f32x16 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        eval_profile::<_, K, OP, ARITHMETIC>(token, V::load_t(token, s))
            .map_err(|e| (i * 16, e))?
            .store(d);
    }
    if !tail.is_empty() {
        eval_profile::<_, K, OP, ARITHMETIC>(token, V::load_padded_t(token, tail, tail[0]))
            .map_err(|e| (chunks.len() * 16, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

//! Configured power graphs, instantiated per width, kernel configuration,
//! and arithmetic provider.
use super::{PowfError, PowfErrorKind};
#[cfg(all(feature = "std", not(miri)))]
pub(super) use crate::lanes::arith::STD;
#[cfg(target_arch = "wasm32")]
pub(super) use crate::lanes::arith::WASM_SOFTWARE;
pub(super) use crate::lanes::arith::{NATIVE, RELAXED, SOFTWARE};
use crate::lanes::{F32Lanes, I32Lanes, madd};
use archmage::IntoConcreteToken as _;

/// Arithmetic and domain knobs needed inside the vector graph. Execution and
/// portability select an arithmetic provider before entering the graph.
pub(super) trait Kernel: 'static {
    /// `+0` is admitted and restored after the graph.
    const ZERO: bool;
    /// Positive subnormal bases (`0 < x < MIN_POSITIVE`) are normalized.
    const SUBNORM: bool;
    /// Positive subnormal bases (`0 < x < MIN_POSITIVE`) are flushed to `0.0`
    /// (when zero is admitted) or `BASE_LO`/`MIN_POSITIVE` (when zero is excluded).
    const FLUSH_SUB: bool;
    /// Bases are validated against `[BASE_LO, BASE_HI]`; otherwise against
    /// the positive-normal (or positive-finite) bit pattern.
    const RANGED: bool;
    const BASE_LO: f32;
    const BASE_HI: f32;
    /// Declared base and exponent ranges keep `n * log2(x)` inside the
    /// approximation's range, so no per-lane range check is emitted.
    const ELIDE: bool;
    /// Explicit output range saturation in MIDP/LOWP paths.
    const CLAMP: bool;
    /// Out-of-bounds inputs are clamped into domain (`NaN -> NaN`).
    const CLAMP_OOB: bool;
    const PRESERVE_CLAMPED_NAN: bool;
    const LOWP: bool;
    const ASSUME: bool;
    /// Algebraic hardware fast-path (`0` = log2/exp2 graph, `1..=11` = IEEE op).
    const ALG_KIND: u8;
    const HAS_CONST_EXP: bool;
    const CONST_EXP: f32;
}

#[derive(Clone, Copy)]
pub(super) struct KernelCfg {
    pub(super) zero: bool,
    pub(super) subnorm: bool,
    pub(super) flush_sub: bool,
    pub(super) ranged: bool,
    pub(super) base_lo: f32,
    pub(super) base_hi: f32,
    pub(super) elide: bool,
    pub(super) clamp: bool,
    pub(super) clamp_oob: bool,
    pub(crate) preserve_clamped_nan: bool,
    pub(super) lowp: bool,
    pub(super) assume: bool,
    pub(super) alg_kind: u8,
    pub(super) has_const_exp: bool,
    pub(super) const_exp: f32,
}

impl KernelCfg {
    pub(super) const fn from_options(opts: super::PowfOptions) -> Self {
        let (base_lo, base_hi) = match opts.base_bounds() {
            Some(bounds) => bounds,
            None => (f32::MIN_POSITIVE, f32::MAX),
        };
        let const_exp = match opts.const_exponent_value() {
            Some(v) => v,
            None => 0.0,
        };
        Self {
            zero: opts.admits_zero(),
            subnorm: opts.admits_subnormals(),
            flush_sub: opts.flushes_subnormals(),
            ranged: opts.base_bounds().is_some(),
            base_lo,
            base_hi,
            elide: opts.elides_range_check(),
            clamp: opts.should_clamp(),
            clamp_oob: opts.clamp_oob(),
            preserve_clamped_nan: opts.preserve_clamped_nan(),
            lowp: opts.is_lowp(),
            assume: opts.assume_valid(),
            alg_kind: opts.alg_kind(),
            has_const_exp: opts.has_const_exponent(),
            const_exp,
        }
    }
}

#[cfg(debug_assertions)]
#[inline(always)]
pub(super) const fn cfg_of<K: Kernel>() -> KernelCfg {
    KernelCfg {
        zero: K::ZERO,
        subnorm: K::SUBNORM,
        flush_sub: K::FLUSH_SUB,
        ranged: K::RANGED,
        base_lo: K::BASE_LO,
        base_hi: K::BASE_HI,
        elide: K::ELIDE,
        clamp: K::CLAMP,
        clamp_oob: K::CLAMP_OOB,
        preserve_clamped_nan: K::PRESERVE_CLAMPED_NAN,
        lowp: K::LOWP,
        assume: K::ASSUME,
        alg_kind: K::ALG_KIND,
        has_const_exp: K::HAS_CONST_EXP,
        const_exp: K::CONST_EXP,
    }
}

trait CfgSource: Copy {
    fn zero(self) -> bool;
    fn subnorm(self) -> bool;
    fn flush_sub(self) -> bool;
    fn ranged(self) -> bool;
    fn base_lo(self) -> f32;
    fn base_hi(self) -> f32;
    fn elide(self) -> bool;
    fn clamp(self) -> bool;
    fn clamp_oob(self) -> bool;
    fn preserve_clamped_nan(self) -> bool;
    fn lowp(self) -> bool;
    fn assume(self) -> bool;
    fn alg_kind(self) -> u8;
    fn has_const_exp(self) -> bool;
    fn const_exp(self) -> f32;
}

impl CfgSource for KernelCfg {
    #[inline(always)]
    fn zero(self) -> bool {
        self.zero
    }
    #[inline(always)]
    fn subnorm(self) -> bool {
        self.subnorm
    }
    #[inline(always)]
    fn flush_sub(self) -> bool {
        self.flush_sub
    }
    #[inline(always)]
    fn ranged(self) -> bool {
        self.ranged
    }
    #[inline(always)]
    fn base_lo(self) -> f32 {
        self.base_lo
    }
    #[inline(always)]
    fn base_hi(self) -> f32 {
        self.base_hi
    }
    #[inline(always)]
    fn elide(self) -> bool {
        self.elide
    }
    #[inline(always)]
    fn clamp(self) -> bool {
        self.clamp
    }
    #[inline(always)]
    fn clamp_oob(self) -> bool {
        self.clamp_oob
    }
    #[inline(always)]
    fn preserve_clamped_nan(self) -> bool {
        self.preserve_clamped_nan
    }
    #[inline(always)]
    fn lowp(self) -> bool {
        self.lowp
    }
    #[inline(always)]
    fn assume(self) -> bool {
        self.assume
    }
    #[inline(always)]
    fn alg_kind(self) -> u8 {
        self.alg_kind
    }
    #[inline(always)]
    fn has_const_exp(self) -> bool {
        self.has_const_exp
    }
    #[inline(always)]
    fn const_exp(self) -> f32 {
        self.const_exp
    }
}

#[cfg(not(debug_assertions))]
struct ConstCfg<K: Kernel>(core::marker::PhantomData<K>);

#[cfg(not(debug_assertions))]
impl<K: Kernel> Clone for ConstCfg<K> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
#[cfg(not(debug_assertions))]
impl<K: Kernel> Copy for ConstCfg<K> {}

#[cfg(not(debug_assertions))]
impl<K: Kernel> CfgSource for ConstCfg<K> {
    #[inline(always)]
    fn zero(self) -> bool {
        K::ZERO
    }
    #[inline(always)]
    fn subnorm(self) -> bool {
        K::SUBNORM
    }
    #[inline(always)]
    fn flush_sub(self) -> bool {
        K::FLUSH_SUB
    }
    #[inline(always)]
    fn ranged(self) -> bool {
        K::RANGED
    }
    #[inline(always)]
    fn base_lo(self) -> f32 {
        K::BASE_LO
    }
    #[inline(always)]
    fn base_hi(self) -> f32 {
        K::BASE_HI
    }
    #[inline(always)]
    fn elide(self) -> bool {
        K::ELIDE
    }
    #[inline(always)]
    fn clamp(self) -> bool {
        K::CLAMP
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
    fn alg_kind(self) -> u8 {
        K::ALG_KIND
    }
    #[inline(always)]
    fn has_const_exp(self) -> bool {
        K::HAS_CONST_EXP
    }
    #[inline(always)]
    fn const_exp(self) -> f32 {
        K::CONST_EXP
    }
}

#[inline(always)]
fn eval_powf<V: F32Lanes, C: CfgSource, const ARITHMETIC: usize>(
    token: V::Token,
    input: V,
    exponent: f32,
    cfg: C,
) -> Result<V, PowfError> {
    let sp = |x| V::splat(token, x);
    let ip = |x| V::I32::splat(token, x);
    let eff_exp: f32 = if cfg.has_const_exp() {
        cfg.const_exp()
    } else {
        exponent
    };
    let lo_bound = if cfg.subnorm() {
        f32::from_bits(1)
    } else if cfg.ranged() {
        cfg.base_lo()
    } else {
        f32::MIN_POSITIVE
    };
    let hi_bound = if cfg.ranged() {
        cfg.base_hi()
    } else {
        f32::MAX
    };
    let zero_ok = cfg.zero() && (cfg.elide() || eff_exp >= 0.0);
    let is_nan = if cfg.preserve_clamped_nan() {
        input.simd_ne(input)
    } else {
        ip(0).bitcast_to_f32()
    };
    let work_in = if cfg.clamp_oob() {
        let c = if !cfg.preserve_clamped_nan() && !zero_ok {
            input.max_fast(sp(lo_bound)).min_fast(sp(hi_bound))
        } else if zero_ok {
            if cfg.subnorm() {
                let nonpos = input.simd_le(sp(0.0));
                let lo_clamped = V::blend(nonpos, sp(0.0), input);
                V::blend(input.simd_gt(sp(hi_bound)), sp(hi_bound), lo_clamped)
            } else if !cfg.ranged() || cfg.base_lo() == f32::MIN_POSITIVE {
                if !cfg.preserve_clamped_nan() {
                    let normal = input & input.simd_ge(sp(f32::MIN_POSITIVE));
                    normal.min_fast(sp(hi_bound))
                } else {
                    let lo_clamped = V::blend(input.simd_lt(sp(f32::MIN_POSITIVE)), sp(0.0), input);
                    V::blend(input.simd_gt(sp(hi_bound)), sp(hi_bound), lo_clamped)
                }
            } else {
                let zero_or_sub = input.simd_lt(sp(f32::MIN_POSITIVE));
                let lo_clamped = V::blend(
                    zero_or_sub,
                    sp(0.0),
                    V::blend(input.simd_lt(sp(lo_bound)), sp(lo_bound), input),
                );
                V::blend(input.simd_gt(sp(hi_bound)), sp(hi_bound), lo_clamped)
            }
        } else {
            V::blend(
                input.simd_lt(sp(lo_bound)),
                sp(lo_bound),
                V::blend(input.simd_gt(sp(hi_bound)), sp(hi_bound), input),
            )
        };
        if cfg.preserve_clamped_nan() {
            V::blend(is_nan, sp(1.0), c)
        } else {
            c
        }
    } else if cfg.flush_sub() && !cfg.assume() {
        let in_bits = input.bitcast_to_i32();
        let pos_sub = in_bits.simd_gt(ip(0)) & in_bits.simd_lt(ip(0x0080_0000));
        let target = if zero_ok { 0.0 } else { lo_bound };
        V::blend(pos_sub.bitcast_to_f32(), sp(target), input)
    } else {
        input
    };
    let bits = work_in.bitcast_to_i32();
    let zero_mask = bits.simd_eq(ip(0));
    if !cfg.assume() && !cfg.clamp_oob() {
        let invalid = if cfg.ranged() {
            if cfg.subnorm() {
                let le_hi = work_in.simd_le(sp(cfg.base_hi())).bitcast_to_i32();
                let pos_ok = if zero_ok {
                    bits.simd_ge(ip(0))
                } else {
                    bits.simd_gt(ip(0))
                };
                (pos_ok & le_hi) ^ ip(-1)
            } else {
                let within =
                    work_in.simd_ge(sp(cfg.base_lo())) & work_in.simd_le(sp(cfg.base_hi()));
                let valid = if zero_ok {
                    within.bitcast_to_i32() | zero_mask
                } else {
                    within.bitcast_to_i32()
                };
                valid ^ ip(-1)
            }
        } else if cfg.subnorm() {
            if zero_ok {
                bits.simd_lt(ip(0)) | bits.simd_ge(ip(0x7f80_0000))
            } else {
                bits.simd_le(ip(0)) | bits.simd_ge(ip(0x7f80_0000))
            }
        } else {
            let exp_bits = bits & ip(0x7f80_0000);
            let invalid = bits.simd_lt(ip(0)) | exp_bits.simd_eq(ip(0x7f80_0000));
            let tiny = exp_bits.simd_eq(ip(0));
            invalid
                | if zero_ok {
                    tiny & (zero_mask ^ ip(-1))
                } else {
                    tiny
                }
        };
        if invalid.any_true() {
            return Err(PowfError {
                kind: PowfErrorKind::InvalidBase,
                lane_mask: invalid.lane_mask(),
            });
        }
    }
    // Admitted zero uses a safe 1.0 sentinel on checking paths where an
    // arbitrary exponent could otherwise amplify log2(0) or bias outside
    // [-126, 127.5]; elided-range paths keep 0 directly and overwrite below.
    let x = if cfg.zero() && !cfg.elide() {
        V::blend(zero_mask.bitcast_to_f32(), sp(1.0), work_in)
    } else {
        work_in
    };
    let clamp_eff = cfg.clamp() || (cfg.clamp_oob() && !cfg.elide());
    let result = if cfg.alg_kind() != 0 {
        let alg = cfg.alg_kind();
        let raw_alg = if alg == 1 {
            sp(1.0)
        } else if alg == 2 {
            x
        } else if alg == 3 {
            sp(1.0) / x
        } else if alg == 4 {
            x * x
        } else if alg == 5 {
            sp(1.0) / (x * x)
        } else if alg == 6 {
            (x * x) * x
        } else if alg == 7 {
            let s = x * x;
            s * s
        } else if alg == 8 {
            x.sqrt()
        } else if alg == 9 {
            sp(1.0) / x.sqrt()
        } else if alg == 10 {
            x * x.sqrt()
        } else {
            x.sqrt().sqrt()
        };
        if !cfg.assume() && !cfg.elide() && !clamp_eff {
            let rb = raw_alg.bitcast_to_i32();
            let outside = rb.simd_lt(ip(0x0080_0000)) | rb.simd_ge(ip(0x7f80_0000));
            if outside.any_true() {
                return Err(PowfError {
                    kind: PowfErrorKind::OutsideApproximationRange,
                    lane_mask: outside.lane_mask(),
                });
            }
        }
        if clamp_eff {
            V::blend(raw_alg.simd_lt(sp(f32::MIN_POSITIVE)), sp(0.0), raw_alg)
        } else {
            raw_alg
        }
    } else {
        let is_subnorm = if cfg.subnorm() {
            let x_bits = x.bitcast_to_i32();
            x_bits.simd_gt(ip(0)) & x_bits.simd_lt(ip(0x0080_0000))
        } else {
            ip(0)
        };
        let x = if cfg.subnorm() {
            V::blend(is_subnorm.bitcast_to_f32(), x * sp(8_388_608.0), x)
        } else {
            x
        };
        let fma = |a: V, b: V, c: V| madd::<V, ARITHMETIC>(token, a, b, c);
        let max_lower = |x: V, lo| {
            if cfg.clamp_oob() && !cfg.preserve_clamped_nan() {
                x.max(lo)
            } else {
                x.max_fast(lo)
            }
        };
        if cfg.lowp() {
            let log = {
                let xb = x.bitcast_to_i32();
                let offset = ip(0x3f2a_aaab_u32 as i32);
                let shifted = (xb - offset).shr_arithmetic_const::<23>();
                let m = (xb - shifted.shl_const::<23>()).bitcast_to_f32() - sp(1.0);
                let yp = fma(sp(0.742_458_7), m, sp(1.428_716_1));
                let yp = fma(yp, m, sp(-1.850_383_3e-6));
                let yq = fma(sp(0.174_093_43), m, sp(1.009_671_8));
                let yq = fma(yq, m, sp(0.990_328_14));
                let exp_adj = if cfg.subnorm() {
                    shifted - (is_subnorm & ip(23))
                } else {
                    shifted
                };
                yp / yq + exp_adj.to_f32()
            };
            let t = log * sp(eff_exp);
            if !cfg.assume() && !cfg.elide() && !clamp_eff {
                let outside = (t.simd_lt(sp(-126.0)) | t.simd_gt(sp(127.5))).bitcast_to_i32();
                let outside = if cfg.zero() {
                    outside & (zero_mask ^ ip(-1))
                } else {
                    outside
                };
                if outside.any_true() {
                    return Err(PowfError {
                        kind: PowfErrorKind::OutsideApproximationRange,
                        lane_mask: outside.lane_mask(),
                    });
                }
            }
            let t = if (cfg.zero() && !cfg.elide()) || clamp_eff {
                max_lower(t, sp(-126.0)).min_fast(sp(127.99))
            } else {
                t
            };
            let xi = t.floor();
            let xf = t - xi;
            let p = crate::expf::graph::exp2_lowp::<V, ARITHMETIC>(token, xf);
            let integer = if token.as_scalar().is_some() {
                xi.to_i32()
            } else {
                xi.to_i32_round()
            };
            p * (integer + ip(127)).shl_const::<23>().bitcast_to_f32()
        } else {
            let log = {
                let adjusted = x.bitcast_to_i32() + ip((0x3f80_0000_u32 - 0x3f35_04f3) as i32);
                let raw_exp = adjusted.shr_arithmetic_const::<23>() - ip(127);
                let n = (if cfg.subnorm() {
                    raw_exp - (is_subnorm & ip(23))
                } else {
                    raw_exp
                })
                .to_f32();
                let a = ((adjusted & ip(0x007f_ffff)) + ip(0x3f35_04f3)).bitcast_to_f32();
                let y = (a - sp(1.0)) / (a + sp(1.0));
                let y2 = y * y;
                let p = fma(sp(0.434_411_97), y2, sp(0.576_974_45));
                let p = fma(p, y2, sp(0.961_800_76));
                let p = fma(p, y2, sp(2.885_39));
                fma(p, y, n)
            };
            let t = log * sp(eff_exp);
            if !cfg.assume() && !cfg.elide() && !clamp_eff {
                let outside = t.simd_lt(sp(-126.0)) | t.simd_gt(sp(127.5));
                if outside.bitcast_to_i32().any_true() {
                    return Err(PowfError {
                        kind: PowfErrorKind::OutsideApproximationRange,
                        lane_mask: outside.bitcast_to_i32().lane_mask(),
                    });
                }
            }
            let underflow_limit = sp(-126.0);
            let overflow_limit = sp(128.0);
            let reduced = if clamp_eff {
                max_lower(t, underflow_limit).min_fast(overflow_limit)
            } else {
                t
            };
            let xi = if cfg.elide() {
                reduced.round()
            } else {
                reduced.round().min_fast(sp(127.0))
            };
            let xf = reduced - xi;
            let p = fma(sp(0.000_154_47), xf, sp(0.001_333_37));
            let p = fma(p, xf, sp(0.009_618_055));
            let p = fma(p, xf, sp(0.055_504_545));
            let p = fma(p, xf, sp(0.240_226_46));
            let p = fma(p, xf, sp(core::f32::consts::LN_2));
            let p = fma(p, xf, sp(1.0));
            let integer = if token.as_scalar().is_some() {
                xi.to_i32()
            } else {
                xi.to_i32_round()
            };
            let scale = (integer + ip(127)).shl_const::<23>().bitcast_to_f32();
            let result = p * scale;
            if clamp_eff {
                let is_underflow = t.simd_lt(underflow_limit);
                let is_overflow = t.simd_ge(overflow_limit);
                let zero = sp(0.0);
                let inf = sp(f32::INFINITY);
                let result = V::blend(is_underflow, zero, result);
                V::blend(is_overflow, inf, result)
            } else {
                result
            }
        }
    };
    let result = if cfg.zero()
        && (!cfg.elide() || cfg.alg_kind() == 0)
        && (cfg.elide() || eff_exp >= 0.0 || !cfg.clamp_oob())
    {
        V::blend(
            zero_mask.bitcast_to_f32(),
            sp(if eff_exp == 0.0 { 1.0 } else { 0.0 }),
            result,
        )
    } else {
        result
    };
    let result = if cfg.preserve_clamped_nan() {
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
    exponent: f32,
    cfg: KernelCfg,
) -> Result<V, PowfError> {
    eval_powf::<V, KernelCfg, ARITHMETIC>(token, input, exponent, cfg)
}

#[inline(always)]
fn eval_profile<V: F32Lanes, K: Kernel, const ARITHMETIC: usize>(
    token: V::Token,
    input: V,
    exponent: f32,
) -> Result<V, PowfError> {
    #[cfg(debug_assertions)]
    return eval_dbg::<V, ARITHMETIC>(token, input, exponent, cfg_of::<K>());
    #[cfg(not(debug_assertions))]
    eval_powf::<V, ConstCfg<K>, ARITHMETIC>(
        token,
        input,
        exponent,
        ConstCfg::<K>(core::marker::PhantomData),
    )
}

macro_rules! kernel_entries {
    ($profile:ident, $cfg_entry:ident, $const_entry:ident, $float:ident, $bound:ident, $lanes:literal) => {
        #[inline(always)]
        pub(super) fn $profile<
            T: magetypes::simd::backends::$bound,
            K: Kernel,
            const ARITHMETIC: usize,
        >(
            token: T,
            input: magetypes::simd::generic::$float<T>,
            exponent: f32,
        ) -> Result<magetypes::simd::generic::$float<T>, PowfError> {
            eval_profile::<_, K, ARITHMETIC>(token, input, exponent)
        }

        #[archmage::attune(_*_t, +v4(cfg(avx512)), +v4x(cfg(avx512)))]
        pub(super) fn $cfg_entry<const ARITHMETIC: usize>(
            token: Token,
            bases: [f32; $lanes],
            exponent: f32,
            cfg: KernelCfg,
        ) -> Result<[f32; $lanes], PowfError> {
            eval_dbg::<_, ARITHMETIC>(
                token,
                magetypes::simd::generic::$float::from_array_t(token, bases),
                exponent,
                cfg,
            )
            .map(|result| result.to_array())
        }

        #[cfg(not(debug_assertions))]
        #[archmage::attune(_*_t, +v4(cfg(avx512)), +v4x(cfg(avx512)))]
        pub(super) fn $const_entry<K: Kernel, const ARITHMETIC: usize>(
            token: Token,
            bases: [f32; $lanes],
            exponent: f32,
        ) -> Result<[f32; $lanes], PowfError> {
            eval_profile::<_, K, ARITHMETIC>(
                token,
                magetypes::simd::generic::$float::from_array_t(token, bases),
                exponent,
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
    exponent: f32,
    cfg: KernelCfg,
) -> Result<f32, PowfError> {
    use crate::scalar1::ScalarF32;
    use archmage::ScalarToken;
    eval_dbg::<_, ARITHMETIC>(
        ScalarToken,
        ScalarF32::splat_t(ScalarToken, input),
        exponent,
        cfg,
    )
    .map(|value| value.0)
}

#[inline(always)]
pub(super) fn kernel1_with<K: Kernel, const ARITHMETIC: usize>(
    input: f32,
    exponent: f32,
) -> Result<f32, PowfError> {
    use crate::scalar1::ScalarF32;
    use archmage::ScalarToken;
    eval_profile::<_, K, ARITHMETIC>(
        ScalarToken,
        ScalarF32::splat_t(ScalarToken, input),
        exponent,
    )
    .map(|value| value.0)
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline)]
pub(super) fn slice1_cfg_scalar<const ARITHMETIC: usize>(
    src: &[f32],
    dst: &mut [f32],
    exponent: f32,
    cfg: KernelCfg,
) -> Result<(), (usize, PowfError)> {
    for (i, (&x, d)) in src.iter().zip(dst.iter_mut()).enumerate() {
        *d = kernel1_cfg::<ARITHMETIC>(x, exponent, cfg).map_err(|e| (i, e))?;
    }
    Ok(())
}

#[cfg(not(debug_assertions))]
#[inline]
pub(super) fn slice1_with_scalar<K: Kernel, const ARITHMETIC: usize>(
    src: &[f32],
    dst: &mut [f32],
    exponent: f32,
) -> Result<(), (usize, PowfError)> {
    for (i, (&x, d)) in src.iter().zip(dst.iter_mut()).enumerate() {
        *d = kernel1_with::<K, ARITHMETIC>(x, exponent).map_err(|e| (i, e))?;
    }
    Ok(())
}

/// Whole-slice kernel parameterized on `KernelCfg` (used by `.runtime()` and
/// debug builds): one feature region around the loop.
#[archmage::attune(_v3_t, _neon_t)]
pub(super) fn slice8_cfg<const ARITHMETIC: usize>(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
    exponent: f32,
    cfg: KernelCfg,
) -> Result<(), (usize, PowfError)> {
    use magetypes::simd::generic::f32x8 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        eval_dbg::<_, ARITHMETIC>(token, V::load_t(token, s), exponent, cfg)
            .map_err(|e| (i * 8, e))?
            .store(d);
    }
    if !tail.is_empty() {
        eval_dbg::<_, ARITHMETIC>(token, V::load_padded_t(token, tail, tail[0]), exponent, cfg)
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
    exponent: f32,
    cfg: KernelCfg,
) -> Result<(), (usize, PowfError)> {
    use magetypes::simd::generic::f32x4 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        eval_dbg::<_, ARITHMETIC>(token, V::load_t(token, s), exponent, cfg)
            .map_err(|e| (i * 4, e))?
            .store(d);
    }
    if !tail.is_empty() {
        eval_dbg::<_, ARITHMETIC>(token, V::load_padded_t(token, tail, tail[0]), exponent, cfg)
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
    exponent: f32,
    cfg: KernelCfg,
) -> Result<(), (usize, PowfError)> {
    use magetypes::simd::generic::f32x16 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        eval_dbg::<_, ARITHMETIC>(token, V::load_t(token, s), exponent, cfg)
            .map_err(|e| (i * 16, e))?
            .store(d);
    }
    if !tail.is_empty() {
        eval_dbg::<_, ARITHMETIC>(token, V::load_padded_t(token, tail, tail[0]), exponent, cfg)
            .map_err(|e| (chunks.len() * 16, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

/// Whole-slice kernel: one feature region around the loop, the 8-lane graph
/// per chunk, and one tail vector padded with `tail[0]` so any range that
/// excludes `1.0` validates accurately and still reports exact first failure.
#[cfg(not(debug_assertions))]
#[archmage::attune(_v3_t, _neon_t)]
pub(super) fn slice8_with<K: Kernel, const ARITHMETIC: usize>(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
    exponent: f32,
) -> Result<(), (usize, PowfError)> {
    use magetypes::simd::generic::f32x8 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        eval_profile::<_, K, ARITHMETIC>(token, V::load_t(token, s), exponent)
            .map_err(|e| (i * 8, e))?
            .store(d);
    }
    if !tail.is_empty() {
        eval_profile::<_, K, ARITHMETIC>(token, V::load_padded_t(token, tail, tail[0]), exponent)
            .map_err(|e| (chunks.len() * 8, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

#[cfg(not(debug_assertions))]
#[archmage::attune(_wasm128_t)]
pub(super) fn slice4_with<K: Kernel, const ARITHMETIC: usize>(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
    exponent: f32,
) -> Result<(), (usize, PowfError)> {
    use magetypes::simd::generic::f32x4 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        eval_profile::<_, K, ARITHMETIC>(token, V::load_t(token, s), exponent)
            .map_err(|e| (i * 4, e))?
            .store(d);
    }
    if !tail.is_empty() {
        eval_profile::<_, K, ARITHMETIC>(token, V::load_padded_t(token, tail, tail[0]), exponent)
            .map_err(|e| (chunks.len() * 4, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

/// 16-lane AVX-512 (`V4`/`V4x`) whole-slice kernel (`zmm` native registers).
#[cfg(not(debug_assertions))]
#[archmage::attune(_v4_t(cfg(avx512)), _v4x_t(cfg(avx512)))]
pub(super) fn slice16_with<K: Kernel, const ARITHMETIC: usize>(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
    exponent: f32,
) -> Result<(), (usize, PowfError)> {
    use magetypes::simd::generic::f32x16 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        eval_profile::<_, K, ARITHMETIC>(token, V::load_t(token, s), exponent)
            .map_err(|e| (i * 16, e))?
            .store(d);
    }
    if !tail.is_empty() {
        eval_profile::<_, K, ARITHMETIC>(token, V::load_padded_t(token, tail, tail[0]), exponent)
            .map_err(|e| (chunks.len() * 16, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

//! Configured logarithm graphs (`log2`, `ln`, `log10`, and shared `log1p`),
//! instantiated per width, kernel configuration, op, and arithmetic mode.
#![allow(clippy::excessive_precision)] // Preserve the fitted f32 coefficient spellings.
use super::{LogfError, LogfErrorKind};
#[cfg(target_arch = "wasm32")]
pub(crate) use crate::lanes::arith::WASM_SOFTWARE;
pub(crate) use crate::lanes::arith::{NATIVE, SOFTWARE};
use crate::lanes::{F32Lanes, I32Lanes, madd};

pub(crate) trait Kernel: 'static {
    const RANGED: bool;
    const IN_LO: f32;
    const IN_HI: f32;
    const ZERO: bool;
    const SUBNORM: bool;
    const FLUSH_SUB: bool;
    const SPECIALS: bool;
    const CLAMP_OOB: bool;
    const PRESERVE_CLAMPED_NAN: bool;
    const LOWP: bool;
    const ASSUME: bool;
    const ELIDE_RED: bool;
    const ELIDE_RED_LOG1P: bool;
}

#[derive(Clone, Copy)]
pub(crate) struct KernelCfg {
    pub(crate) ranged: bool,
    pub(crate) in_lo: f32,
    pub(crate) in_hi: f32,
    pub(crate) zero: bool,
    pub(crate) subnorm: bool,
    pub(crate) flush_sub: bool,
    pub(crate) specials: bool,
    pub(crate) clamp_oob: bool,
    pub(crate) preserve_clamped_nan: bool,
    pub(crate) lowp: bool,
    pub(crate) assume: bool,
    pub(crate) elide_red: bool,
    pub(crate) elide_red_log1p: bool,
}

impl KernelCfg {
    pub(crate) const fn from_options(opts: super::LogfOptions) -> Self {
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
            flush_sub: opts.flushes_subnormals(),
            specials: opts.is_specials(),
            clamp_oob: opts.clamp_oob(),
            preserve_clamped_nan: opts.preserve_clamped_nan(),
            lowp: opts.is_lowp(),
            assume: opts.assume_valid(),
            elide_red: opts.elides_reduction(),
            elide_red_log1p: false,
        }
    }
}

#[cfg(debug_assertions)]
#[inline(always)]
pub(crate) const fn cfg_of<K: Kernel>() -> KernelCfg {
    KernelCfg {
        ranged: K::RANGED,
        in_lo: K::IN_LO,
        in_hi: K::IN_HI,
        zero: K::ZERO,
        subnorm: K::SUBNORM,
        flush_sub: K::FLUSH_SUB,
        specials: K::SPECIALS,
        clamp_oob: K::CLAMP_OOB,
        preserve_clamped_nan: K::PRESERVE_CLAMPED_NAN,
        lowp: K::LOWP,
        assume: K::ASSUME,
        elide_red: K::ELIDE_RED,
        elide_red_log1p: K::ELIDE_RED_LOG1P,
    }
}

trait CfgSource: Copy {
    fn ranged(self) -> bool;
    fn in_lo(self) -> f32;
    fn in_hi(self) -> f32;
    fn zero(self) -> bool;
    fn subnorm(self) -> bool;
    fn flush_sub(self) -> bool;
    fn specials(self) -> bool;
    fn clamp_oob(self) -> bool;
    fn preserve_clamped_nan(self) -> bool;
    fn lowp(self) -> bool;
    fn assume(self) -> bool;
    fn elide_red(self) -> bool;
    fn elide_red_log1p(self) -> bool;
}

impl CfgSource for KernelCfg {
    #[inline(always)]
    fn ranged(self) -> bool {
        self.ranged
    }
    #[inline(always)]
    fn in_lo(self) -> f32 {
        self.in_lo
    }
    #[inline(always)]
    fn in_hi(self) -> f32 {
        self.in_hi
    }
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
    fn specials(self) -> bool {
        self.specials
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
    fn elide_red(self) -> bool {
        self.elide_red
    }
    #[inline(always)]
    fn elide_red_log1p(self) -> bool {
        self.elide_red_log1p
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
    fn specials(self) -> bool {
        K::SPECIALS
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
    fn elide_red(self) -> bool {
        K::ELIDE_RED
    }
    #[inline(always)]
    fn elide_red_log1p(self) -> bool {
        K::ELIDE_RED_LOG1P
    }
}

// Keep `eval_log2` forced-inline on native targets (so it enters the caller's
// `#[target_feature]` region cleanly), while letting WASM `opt-z` share one
// outlined copy across chunk + tail loops.
#[cfg_attr(not(target_arch = "wasm32"), inline(always))]
#[cfg_attr(target_arch = "wasm32", inline)]
fn eval_log2<V: F32Lanes, C: CfgSource, const ARITHMETIC: usize>(
    token: V::Token,
    input: V,
    cfg: C,
) -> Result<V, LogfError> {
    let sp = |x| V::splat(token, x);
    let ip = |x| V::I32::splat(token, x);
    let reject = |mask: V::I32, kind| LogfError {
        kind,
        lane_mask: mask.lane_mask(),
    };
    let lo_bound = if cfg.ranged() {
        cfg.in_lo()
    } else if cfg.subnorm() {
        f32::from_bits(1)
    } else {
        f32::MIN_POSITIVE
    };
    let hi_bound = if cfg.ranged() { cfg.in_hi() } else { f32::MAX };
    let pre_in = if !cfg.assume() && !cfg.clamp_oob() && !cfg.specials() && cfg.flush_sub() {
        let in_bits = input.bitcast_to_i32();
        let pos_sub = in_bits.simd_gt(ip(0)) & in_bits.simd_lt(ip(0x0080_0000));
        let target = if cfg.zero() {
            0.0
        } else {
            lo_bound.max(f32::MIN_POSITIVE)
        };
        V::blend(pos_sub.bitcast_to_f32(), sp(target), input)
    } else {
        input
    };
    if !cfg.assume() && !cfg.clamp_oob() {
        if cfg.ranged() {
            let within = (pre_in.simd_ge(sp(cfg.in_lo())) & pre_in.simd_le(sp(cfg.in_hi())))
                .bitcast_to_i32();
            let valid = if cfg.zero() {
                within | pre_in.simd_eq(sp(0.0)).bitcast_to_i32()
            } else {
                within
            };
            let invalid = valid ^ ip(-1);
            if invalid.any_true() {
                return Err(reject(invalid, LogfErrorKind::InvalidInput));
            }
        } else if !cfg.specials() {
            let bits0 = pre_in.bitcast_to_i32();
            let exp_bits = bits0 & ip(0x7f80_0000);
            let is_zero_f = pre_in.simd_eq(sp(0.0)).bitcast_to_i32();
            let neg = if cfg.zero() {
                bits0.simd_lt(ip(0)) & (is_zero_f ^ ip(-1))
            } else {
                bits0.simd_le(ip(0))
            };
            let nonfinite = exp_bits.simd_eq(ip(0x7f80_0000));
            let tiny_bad = if cfg.subnorm() {
                if cfg.zero() { ip(0) } else { is_zero_f }
            } else if cfg.zero() {
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
    let is_nan = if cfg.preserve_clamped_nan() && !cfg.specials() {
        input.simd_ne(input)
    } else {
        ip(0).bitcast_to_f32()
    };
    let work_in = if cfg.clamp_oob() && !cfg.specials() {
        let eff_lo = if !cfg.subnorm() && lo_bound > 0.0 && lo_bound < f32::MIN_POSITIVE {
            f32::MIN_POSITIVE
        } else {
            lo_bound
        };
        let c = if !cfg.preserve_clamped_nan() && !cfg.zero() {
            input.max_fast(sp(eff_lo)).min_fast(sp(hi_bound))
        } else if cfg.zero() {
            if cfg.subnorm() {
                let nonpos = input.simd_le(sp(0.0));
                let lo_c = V::blend(
                    nonpos,
                    sp(0.0),
                    V::blend(input.simd_lt(sp(eff_lo)), sp(eff_lo), input),
                );
                V::blend(input.simd_gt(sp(hi_bound)), sp(hi_bound), lo_c)
            } else if eff_lo <= f32::MIN_POSITIVE {
                if !cfg.preserve_clamped_nan() {
                    let normal = input & input.simd_ge(sp(f32::MIN_POSITIVE));
                    normal.min_fast(sp(hi_bound))
                } else {
                    let lo_c = V::blend(input.simd_lt(sp(f32::MIN_POSITIVE)), sp(0.0), input);
                    V::blend(input.simd_gt(sp(hi_bound)), sp(hi_bound), lo_c)
                }
            } else {
                let zero_or_sub = input.simd_lt(sp(f32::MIN_POSITIVE));
                let lo_c = V::blend(
                    zero_or_sub,
                    sp(0.0),
                    V::blend(input.simd_lt(sp(eff_lo)), sp(eff_lo), input),
                );
                V::blend(input.simd_gt(sp(hi_bound)), sp(hi_bound), lo_c)
            }
        } else {
            V::blend(
                input.simd_lt(sp(eff_lo)),
                sp(eff_lo),
                V::blend(input.simd_gt(sp(hi_bound)), sp(hi_bound), input),
            )
        };
        let safe = if cfg.ranged() && eff_lo > 0.0 {
            eff_lo
        } else {
            1.0
        };
        if cfg.preserve_clamped_nan() {
            V::blend(is_nan, sp(safe), c)
        } else {
            c
        }
    } else {
        pre_in
    };

    let raw_bits = work_in.bitcast_to_i32();
    let is_subnorm = if cfg.subnorm() && !cfg.elide_red() {
        raw_bits.simd_gt(ip(0)) & raw_bits.simd_lt(ip(0x0080_0000))
    } else {
        ip(0)
    };
    let norm_in = if cfg.subnorm() && !cfg.elide_red() {
        V::blend(
            is_subnorm.bitcast_to_f32(),
            work_in * sp(8_388_608.0),
            work_in,
        )
    } else {
        work_in
    };
    let bits = norm_in.bitcast_to_i32();

    let result = if cfg.lowp() {
        if cfg.elide_red() {
            let m = work_in - sp(1.0);
            let yp = madd::<V, ARITHMETIC>(token, sp(0.742_458_7), m, sp(1.428_716_1));
            let yp = madd::<V, ARITHMETIC>(token, yp, m, sp(-1.850_383_3e-6));
            let yq = madd::<V, ARITHMETIC>(token, sp(0.174_093_43), m, sp(1.009_671_8));
            let yq = madd::<V, ARITHMETIC>(token, yq, m, sp(0.990_328_14));
            yp / yq
        } else {
            let offset = ip(0x3f2a_aaab_u32 as i32);
            let exp_bits_v = bits - offset;
            let exp_shifted = exp_bits_v.shr_arithmetic_const::<23>();
            let mantissa_bits = bits - exp_shifted.shl_const::<23>();
            let m = mantissa_bits.bitcast_to_f32() - sp(1.0);
            let yp = madd::<V, ARITHMETIC>(token, sp(0.742_458_7), m, sp(1.428_716_1));
            let yp = madd::<V, ARITHMETIC>(token, yp, m, sp(-1.850_383_3e-6));
            let yq = madd::<V, ARITHMETIC>(token, sp(0.174_093_43), m, sp(1.009_671_8));
            let yq = madd::<V, ARITHMETIC>(token, yq, m, sp(0.990_328_14));
            let exp_adj = if cfg.subnorm() {
                exp_shifted - (is_subnorm & ip(23))
            } else {
                exp_shifted
            };
            yp / yq + exp_adj.to_f32()
        }
    } else if cfg.elide_red() {
        let y = (work_in - sp(1.0)) / (work_in + sp(1.0));
        let y2 = y * y;
        let p = madd::<V, ARITHMETIC>(
            token,
            sp(0.430_468_597_034_945_7),
            y2,
            sp(0.576_767_602_130_619_1),
        );
        let p = madd::<V, ARITHMETIC>(token, p, y2, sp(0.961_798_223_457_557_8));
        let p = madd::<V, ARITHMETIC>(token, p, y2, sp(2.885_390_081_777_926_8));
        p * y
    } else {
        let adjusted = bits + ip((0x3f80_0000_u32 - 0x3f35_04f3) as i32);
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
        let p = madd::<V, ARITHMETIC>(
            token,
            sp(0.430_468_597_034_945_7),
            y2,
            sp(0.576_767_602_130_619_1),
        );
        let p = madd::<V, ARITHMETIC>(token, p, y2, sp(0.961_798_223_457_557_8));
        let p = madd::<V, ARITHMETIC>(token, p, y2, sp(2.885_390_081_777_926_8));
        madd::<V, ARITHMETIC>(token, p, y, n)
    };
    let result = if cfg.specials() {
        let zero = work_in.simd_eq(sp(0.0));
        let nonnegative = work_in.simd_ge(sp(0.0));
        let pos_inf = work_in.simd_eq(sp(f32::INFINITY));
        let result = V::blend(zero, sp(f32::NEG_INFINITY), result);
        let result = V::blend(nonnegative, result, sp(f32::NAN));
        V::blend(pos_inf, sp(f32::INFINITY), result)
    } else {
        let result = if cfg.zero() {
            V::blend(work_in.simd_eq(sp(0.0)), sp(f32::NEG_INFINITY), result)
        } else {
            result
        };
        if cfg.preserve_clamped_nan() {
            V::blend(is_nan, sp(f32::NAN), result)
        } else {
            result
        }
    };
    Ok(result)
}

// Fits to (ln(1 + x) - x) / x², with an exact linear term at zero.
const LOG1P_LOW: [f32; 6] = [
    -0.5000584211,
    0.3331853231,
    -0.2473258626,
    0.2025821466,
    -0.2009205163,
    0.1448155509,
];
const LOG1P_MID: [f32; 9] = [
    -0.4999998684,
    0.3333332149,
    -0.2500173438,
    0.2000235959,
    -0.1660544601,
    0.1417161584,
    -0.1326462578,
    0.1298266121,
    -0.07664833665,
];

#[inline(always)]
fn log1p_tail<V: F32Lanes, C: CfgSource, const ARITHMETIC: usize>(
    token: V::Token,
    x: V,
    cfg: C,
) -> V {
    let sp = |v| V::splat(token, v);
    let c: &[f32] = if cfg.lowp() { &LOG1P_LOW } else { &LOG1P_MID };
    let x2 = x * x;
    let x4 = x2 * x2;
    let p01 = madd::<V, ARITHMETIC>(token, sp(c[1]), x, sp(c[0]));
    let p23 = madd::<V, ARITHMETIC>(token, sp(c[3]), x, sp(c[2]));
    let p45 = madd::<V, ARITHMETIC>(token, sp(c[5]), x, sp(c[4]));
    let p48 = if cfg.lowp() {
        p45
    } else {
        let p67 = madd::<V, ARITHMETIC>(token, sp(c[7]), x, sp(c[6]));
        sp(c[8]) * x4 + (p67 * x2 + p45)
    };
    // Keep fusion on the leading terms and round the smaller terms
    // separately to avoid extra software FMAs on strict WASM.
    let p = p48 * x4 + madd::<V, ARITHMETIC>(token, p23, x2, p01);
    x2 * p
}

#[inline(always)]
fn eval_log1p<V: F32Lanes, C: CfgSource, const ARITHMETIC: usize>(
    token: V::Token,
    input: V,
    cfg: C,
) -> Result<V, LogfError> {
    let sp = |x| V::splat(token, x);
    let ip = |x| V::I32::splat(token, x);
    let lo_bound = if cfg.ranged() {
        cfg.in_lo()
    } else if cfg.zero() {
        -1.0
    } else {
        f32::from_bits(0xbf7f_ffff)
    };
    let hi_bound = if cfg.ranged() { cfg.in_hi() } else { f32::MAX };
    let pre_x = if !cfg.assume() && !cfg.clamp_oob() && !cfg.specials() && cfg.flush_sub() {
        let in_bits = input.bitcast_to_i32();
        let abs_bits = in_bits & ip(0x7fff_ffff_u32 as i32);
        let is_sub = abs_bits.simd_gt(ip(0)) & abs_bits.simd_lt(ip(0x0080_0000));
        if (lo_bound == 0.0 && hi_bound == 0.0) || (lo_bound < 0.0 && hi_bound > 0.0) {
            let signed_zero = (in_bits & ip(0x8000_0000_u32 as i32)).bitcast_to_f32();
            V::blend(is_sub.bitcast_to_f32(), signed_zero, input)
        } else if lo_bound == 0.0 && hi_bound > 0.0 {
            let pos_sub = in_bits.simd_gt(ip(0)) & in_bits.simd_lt(ip(0x0080_0000));
            V::blend(pos_sub.bitcast_to_f32(), sp(0.0), input)
        } else if hi_bound == 0.0 && lo_bound < 0.0 {
            let neg_sub = in_bits.simd_lt(ip(0)) & abs_bits.simd_lt(ip(0x0080_0000));
            V::blend(neg_sub.bitcast_to_f32(), sp(-0.0), input)
        } else if lo_bound > 0.0 {
            let pos_sub = in_bits.simd_gt(ip(0)) & in_bits.simd_lt(ip(0x0080_0000));
            V::blend(
                pos_sub.bitcast_to_f32(),
                sp(lo_bound.max(f32::MIN_POSITIVE)),
                input,
            )
        } else {
            let neg_sub = in_bits.simd_lt(ip(0)) & abs_bits.simd_lt(ip(0x0080_0000));
            V::blend(
                neg_sub.bitcast_to_f32(),
                sp(hi_bound.min(-f32::MIN_POSITIVE)),
                input,
            )
        }
    } else {
        input
    };
    if !cfg.assume() && !cfg.clamp_oob() && !cfg.specials() {
        let range_ok = if cfg.ranged() {
            let within =
                (pre_x.simd_ge(sp(cfg.in_lo())) & pre_x.simd_le(sp(cfg.in_hi()))).bitcast_to_i32();
            if cfg.zero() {
                within | pre_x.simd_eq(sp(-1.0)).bitcast_to_i32()
            } else {
                within
            }
        } else {
            let lo_ok = if cfg.zero() {
                pre_x.simd_ge(sp(-1.0))
            } else {
                pre_x.simd_gt(sp(-1.0))
            };
            (lo_ok & pre_x.simd_le(sp(f32::MAX))).bitcast_to_i32()
        };
        let sub_bad = if !cfg.subnorm() && !cfg.flush_sub() {
            let abs_b = pre_x.bitcast_to_i32() & ip(0x7fff_ffff_u32 as i32);
            abs_b.simd_gt(ip(0)) & abs_b.simd_lt(ip(0x0080_0000))
        } else {
            ip(0)
        };
        let invalid = (range_ok ^ ip(-1)) | sub_bad;
        if invalid.any_true() {
            return Err(LogfError {
                kind: LogfErrorKind::InvalidInput,
                lane_mask: invalid.lane_mask(),
            });
        }
    }
    let is_nan = if cfg.preserve_clamped_nan() && !cfg.specials() {
        input.simd_ne(input)
    } else {
        ip(0).bitcast_to_f32()
    };
    let work_x = if cfg.clamp_oob() && !cfg.specials() {
        let eff_lo = if !cfg.subnorm() && lo_bound > 0.0 && lo_bound < f32::MIN_POSITIVE {
            f32::MIN_POSITIVE
        } else {
            lo_bound
        };
        let eff_hi = if !cfg.subnorm() && hi_bound < 0.0 && hi_bound > -f32::MIN_POSITIVE {
            -f32::MIN_POSITIVE
        } else {
            hi_bound
        };
        let c = if cfg.zero() {
            let le_neg_one = input.simd_le(sp(-1.0));
            let lo_c = V::blend(
                le_neg_one,
                sp(-1.0),
                V::blend(input.simd_lt(sp(eff_lo)), sp(eff_lo), input),
            );
            V::blend(input.simd_gt(sp(eff_hi)), sp(eff_hi), lo_c)
        } else {
            V::blend(
                input.simd_lt(sp(eff_lo)),
                sp(eff_lo),
                V::blend(input.simd_gt(sp(eff_hi)), sp(eff_hi), input),
            )
        };
        let c = if !cfg.subnorm() {
            let cb = c.bitcast_to_i32();
            let abs_c = cb & ip(0x7fff_ffff_u32 as i32);
            let is_sub = abs_c.simd_gt(ip(0)) & abs_c.simd_lt(ip(0x0080_0000));
            let z = if eff_lo == 0.0 {
                sp(0.0)
            } else {
                (cb & ip(0x8000_0000_u32 as i32)).bitcast_to_f32()
            };
            V::blend(is_sub.bitcast_to_f32(), z, c)
        } else {
            c
        };
        let safe = if cfg.ranged() && eff_lo > -1.0 {
            eff_lo
        } else {
            0.0
        };
        if cfg.preserve_clamped_nan() {
            V::blend(is_nan, sp(safe), c)
        } else {
            c
        }
    } else {
        pre_x
    };
    let magnitude = work_x.bitcast_to_i32() & ip(i32::MAX);
    let res = if (cfg.elide_red_log1p() && cfg.in_lo() >= -0.25 && cfg.in_hi() <= 0.25)
        || magnitude.simd_le(ip(0.25_f32.to_bits() as i32)).all_true()
    {
        work_x + log1p_tail::<V, C, ARITHMETIC>(token, work_x, cfg)
    } else {
        let one = sp(1.0);
        let u = work_x + one;
        let adjusted = u.bitcast_to_i32() + ip(0x3f80_0000 - 0x3f35_04f3);
        let exponent = adjusted.shr_arithmetic_const::<23>() - ip(127);
        let scale_exp = ip(127) - exponent;
        let normal_exp = scale_exp.max(ip(1));
        let scale = normal_exp.shl_const::<23>().bitcast_to_f32();
        let tail_scale = (scale_exp - normal_exp + ip(127))
            .shl_const::<23>()
            .bitcast_to_f32();
        // Reduce 1 + x into [sqrt(0.5), sqrt(2)). Apply the exact
        // power-of-two scale to x before adding the offset so rounding
        // 1 + x cannot erase small inputs. Splitting the scale keeps
        // both factors normal at exponents 127 and 128.
        let f = (work_x * scale) * tail_scale + (scale - one);
        let n = exponent.to_f32();
        let tail = log1p_tail::<V, C, ARITHMETIC>(token, f, cfg);
        // The high part has enough trailing zeros for n * ln2_hi to be exact.
        let ln2_hi = sp(f32::from_bits(0x3f31_7200));
        let ln2_lo = sp(1.428_606_8e-6);
        let result = n * ln2_hi + (f + (tail + n * ln2_lo));
        let result = if cfg.zero() {
            V::blend(work_x.simd_eq(sp(-1.0)), sp(f32::NEG_INFINITY), result)
        } else {
            result
        };
        if cfg.specials() {
            let result = V::blend(work_x.simd_ge(sp(-1.0)), result, sp(f32::NAN));
            V::blend(work_x.simd_eq(sp(f32::INFINITY)), sp(f32::INFINITY), result)
        } else {
            result
        }
    };
    let preserve_x = if cfg.subnorm() || cfg.specials() {
        ((work_x.bitcast_to_i32() & ip(0x7f80_0000)).simd_eq(ip(0))).bitcast_to_f32()
    } else {
        work_x.simd_eq(sp(0.0))
    };
    let res = V::blend(preserve_x, work_x, res);
    if cfg.preserve_clamped_nan() && !cfg.specials() {
        Ok(V::blend(is_nan, sp(f32::NAN), res))
    } else {
        Ok(res)
    }
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline)]
fn eval_dbg<V: F32Lanes, const ARITHMETIC: usize>(
    token: V::Token,
    input: V,
    op: u8,
    cfg: KernelCfg,
) -> Result<V, LogfError> {
    if op == 3 {
        eval_log1p::<V, KernelCfg, ARITHMETIC>(token, input, cfg)
    } else {
        let log2_v = eval_log2::<V, KernelCfg, ARITHMETIC>(token, input, cfg)?;
        Ok(match op {
            1 => log2_v * V::splat(token, core::f32::consts::LN_2),
            2 => log2_v * V::splat(token, core::f32::consts::LOG10_2),
            _ => log2_v,
        })
    }
}

#[inline(always)]
fn eval_profile<V: F32Lanes, K: Kernel, const OP: u8, const ARITHMETIC: usize>(
    token: V::Token,
    input: V,
) -> Result<V, LogfError> {
    #[cfg(debug_assertions)]
    return eval_dbg::<V, ARITHMETIC>(token, input, OP, cfg_of::<K>());
    #[cfg(not(debug_assertions))]
    {
        let cfg = ConstCfg::<K>(core::marker::PhantomData);
        if OP == 3 {
            eval_log1p::<V, ConstCfg<K>, ARITHMETIC>(token, input, cfg)
        } else {
            let log2_v = eval_log2::<V, ConstCfg<K>, ARITHMETIC>(token, input, cfg)?;
            Ok(match OP {
                1 => log2_v * V::splat(token, core::f32::consts::LN_2),
                2 => log2_v * V::splat(token, core::f32::consts::LOG10_2),
                _ => log2_v,
            })
        }
    }
}

macro_rules! kernel_entries {
    ($profile:ident, $cfg_entry:ident, $const_entry:ident, $float:ident, $bound:ident, $lanes:literal) => {
        #[inline(always)]
        pub(crate) fn $profile<
            T: magetypes::simd::backends::$bound,
            K: Kernel,
            const OP: u8,
            const ARITHMETIC: usize,
        >(
            token: T,
            input: magetypes::simd::generic::$float<T>,
        ) -> Result<magetypes::simd::generic::$float<T>, LogfError> {
            eval_profile::<_, K, OP, ARITHMETIC>(token, input)
        }

        #[archmage::attune(_*_t, +v4(cfg(avx512)), +v4x(cfg(avx512)))]
        pub(super) fn $cfg_entry<const ARITHMETIC: usize>(
            token: Token,
            input: [f32; $lanes],
            op: u8,
            cfg: KernelCfg,
        ) -> Result<[f32; $lanes], LogfError> {
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
        ) -> Result<[f32; $lanes], LogfError> {
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
pub(crate) fn kernel1_cfg<const ARITHMETIC: usize>(
    input: f32,
    op: u8,
    cfg: KernelCfg,
) -> Result<f32, LogfError> {
    use crate::scalar1::ScalarF32;
    use archmage::ScalarToken;
    eval_dbg::<_, ARITHMETIC>(ScalarToken, ScalarF32::splat_t(ScalarToken, input), op, cfg)
        .map(|value| value.0)
}

#[inline(always)]
pub(crate) fn kernel1_with<K: Kernel, const OP: u8, const ARITHMETIC: usize>(
    input: f32,
) -> Result<f32, LogfError> {
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
) -> Result<(), (usize, LogfError)> {
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
) -> Result<(), (usize, LogfError)> {
    for (i, (&x, d)) in src.iter().zip(dst.iter_mut()).enumerate() {
        *d = kernel1_with::<K, OP, ARITHMETIC>(x).map_err(|e| (i, e))?;
    }
    Ok(())
}

/// Whole-slice kernel (`op`: `0=log2`, `1=ln`, `2=log10`, `3=log1p`)
/// parameterized on `KernelCfg`.
#[archmage::attune(_v3_t, _neon_t)]
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
) -> Result<(), (usize, LogfError)> {
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
) -> Result<(), (usize, LogfError)> {
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

/// Whole-slice kernel (`OP`: `0=log2`, `1=ln`, `2=log10`, `3=log1p`): one
/// feature region around the loop, the 8-lane graph per chunk, and `tail[0]`
/// padding so ranges excluding `1.0` never false-reject.
#[cfg(not(debug_assertions))]
#[archmage::attune(_v3_t, _neon_t)]
pub(super) fn slice8_with<K: Kernel, const OP: u8, const ARITHMETIC: usize>(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, LogfError)> {
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
) -> Result<(), (usize, LogfError)> {
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
) -> Result<(), (usize, LogfError)> {
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

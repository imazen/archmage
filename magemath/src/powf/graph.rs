//! Configured power graphs, instantiated per width, kernel configuration,
//! and arithmetic provider.
use super::{PowfError, PowfErrorKind};

// Arithmetic mode is chosen once at the operation boundary. Mode 0 uses the
// token-native relaxed madd; BitExact proves fusion before selecting it, while
// BackendDependent permits either rounding choice.
pub(super) const NATIVE: usize = 0;
pub(super) const SOFTWARE: usize = 1;
pub(super) const RELAXED: usize = 4;
#[cfg(target_arch = "wasm32")]
pub(super) const WASM_SOFTWARE: usize = 3;
#[cfg(all(feature = "std", not(miri)))]
pub(super) const STD: usize = 2;

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

macro_rules! eval_powf {
    (
        $V:ident,
        $I:ident,
        $token:ident,
        $input:ident,
        $exponent:ident,
        $arithmetic:expr,
        $zero:expr,
        $subnorm:expr,
        $flush_sub:expr,
        $ranged:expr,
        $base_lo:expr,
        $base_hi:expr,
        $elide:expr,
        $clamp:expr,
        $clamp_oob:expr,
        $preserve_clamped_nan:expr,
        $lowp:expr,
        $assume:expr,
        $alg_kind:expr,
        $has_const_exp:expr,
        $const_exp:expr
    ) => {{
        let sp = |x| $V::splat_t($token, x);
        let ip = |x| $I::splat_t($token, x);
        let eff_exp: f32 = if $has_const_exp {
            $const_exp
        } else {
            $exponent
        };
        let lo_bound = if $subnorm {
            f32::from_bits(1)
        } else if $ranged {
            $base_lo
        } else {
            f32::MIN_POSITIVE
        };
        let hi_bound = if $ranged { $base_hi } else { f32::MAX };
        let zero_ok = $zero && ($elide || eff_exp >= 0.0);
        let is_nan = if $preserve_clamped_nan {
            $input.simd_ne($input)
        } else {
            ip(0).bitcast_to_f32()
        };
        let work_in = if $clamp_oob {
            let c = if !$preserve_clamped_nan && !zero_ok {
                $input.max_fast(sp(lo_bound)).min_fast(sp(hi_bound))
            } else if zero_ok {
                if $subnorm {
                    let nonpos = $input.simd_le(sp(0.0));
                    let lo_clamped = $V::blend(nonpos, sp(0.0), $input);
                    $V::blend($input.simd_gt(sp(hi_bound)), sp(hi_bound), lo_clamped)
                } else if !$ranged || $base_lo == f32::MIN_POSITIVE {
                    if !$preserve_clamped_nan {
                        let normal = $input & $input.simd_ge(sp(f32::MIN_POSITIVE));
                        normal.min_fast(sp(hi_bound))
                    } else {
                        let lo_clamped =
                            $V::blend($input.simd_lt(sp(f32::MIN_POSITIVE)), sp(0.0), $input);
                        $V::blend($input.simd_gt(sp(hi_bound)), sp(hi_bound), lo_clamped)
                    }
                } else {
                    let zero_or_sub = $input.simd_lt(sp(f32::MIN_POSITIVE));
                    let lo_clamped = $V::blend(
                        zero_or_sub,
                        sp(0.0),
                        $V::blend($input.simd_lt(sp(lo_bound)), sp(lo_bound), $input),
                    );
                    $V::blend($input.simd_gt(sp(hi_bound)), sp(hi_bound), lo_clamped)
                }
            } else {
                $V::blend(
                    $input.simd_lt(sp(lo_bound)),
                    sp(lo_bound),
                    $V::blend($input.simd_gt(sp(hi_bound)), sp(hi_bound), $input),
                )
            };
            if $preserve_clamped_nan {
                $V::blend(is_nan, sp(1.0), c)
            } else {
                c
            }
        } else if $flush_sub && !$assume {
            let in_bits = $input.bitcast_to_i32();
            let pos_sub = in_bits.simd_gt(ip(0)) & in_bits.simd_lt(ip(0x0080_0000));
            let target = if zero_ok { 0.0 } else { lo_bound };
            $V::blend(pos_sub.bitcast_to_f32(), sp(target), $input)
        } else {
            $input
        };
        let bits = work_in.bitcast_to_i32();
        let zero_mask = bits.simd_eq(ip(0));
        let reject = |mask: $I<T>, kind| {
            let lane_mask = mask
                .to_array()
                .iter()
                .enumerate()
                .fold(0_u16, |m, (i, &v)| m | (u16::from(v != 0) << i));
            PowfError { kind, lane_mask }
        };
        if !$assume && !$clamp_oob {
            let invalid = if $ranged {
                if $subnorm {
                    let le_hi = work_in.simd_le(sp($base_hi)).bitcast_to_i32();
                    let pos_ok = if zero_ok {
                        bits.simd_ge(ip(0))
                    } else {
                        bits.simd_gt(ip(0))
                    };
                    (pos_ok & le_hi) ^ ip(-1)
                } else {
                    let within = work_in.simd_ge(sp($base_lo)) & work_in.simd_le(sp($base_hi));
                    let valid = if zero_ok {
                        within.bitcast_to_i32() | zero_mask
                    } else {
                        within.bitcast_to_i32()
                    };
                    valid ^ ip(-1)
                }
            } else if $subnorm {
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
                return Err(reject(invalid, PowfErrorKind::InvalidBase));
            }
        }
        // Admitted zero uses a safe 1.0 sentinel on checking paths where an
        // arbitrary exponent could otherwise amplify log2(0) or bias outside
        // [-126, 127.5]; elided-range paths keep 0 directly and overwrite below.
        let x = if $zero && !$elide {
            $V::blend(zero_mask.bitcast_to_f32(), sp(1.0), work_in)
        } else {
            work_in
        };
        let clamp_eff = $clamp || ($clamp_oob && !$elide);
        let result = if $alg_kind != 0 {
            let raw_alg = if $alg_kind == 1 {
                sp(1.0)
            } else if $alg_kind == 2 {
                x
            } else if $alg_kind == 3 {
                sp(1.0) / x
            } else if $alg_kind == 4 {
                x * x
            } else if $alg_kind == 5 {
                sp(1.0) / (x * x)
            } else if $alg_kind == 6 {
                (x * x) * x
            } else if $alg_kind == 7 {
                let s = x * x;
                s * s
            } else if $alg_kind == 8 {
                x.sqrt()
            } else if $alg_kind == 9 {
                sp(1.0) / x.sqrt()
            } else if $alg_kind == 10 {
                x * x.sqrt()
            } else {
                x.sqrt().sqrt()
            };
            if !$assume && !$elide && !clamp_eff {
                let rb = raw_alg.bitcast_to_i32();
                let outside = rb.simd_lt(ip(0x0080_0000)) | rb.simd_ge(ip(0x7f80_0000));
                if outside.any_true() {
                    return Err(reject(outside, PowfErrorKind::OutsideApproximationRange));
                }
            }
            if clamp_eff {
                $V::blend(raw_alg.simd_lt(sp(f32::MIN_POSITIVE)), sp(0.0), raw_alg)
            } else {
                raw_alg
            }
        } else {
            let is_subnorm = if $subnorm {
                let x_bits = x.bitcast_to_i32();
                x_bits.simd_gt(ip(0)) & x_bits.simd_lt(ip(0x0080_0000))
            } else {
                ip(0)
            };
            let x = if $subnorm {
                $V::blend(is_subnorm.bitcast_to_f32(), x * sp(8_388_608.0), x)
            } else {
                x
            };
            let madd = |a: $V<T>, b: $V<T>, c: $V<T>| -> $V<T> {
                if $arithmetic == NATIVE || $arithmetic == RELAXED {
                    a.mul_add_relaxed(b, c)
                } else if $arithmetic == SOFTWARE {
                    a.mul_add_fused(b, c)
                } else {
                    let (a, b, c) = (a.to_array(), b.to_array(), c.to_array());
                    #[cfg(target_arch = "wasm32")]
                    if $arithmetic == WASM_SOFTWARE {
                        let proof = $token
                            .as_wasm128()
                            .expect("private WASM mode requires its matching proof");
                        return $V::from_array_t(
                            $token,
                            super::wasm::software_lanes(proof, a, b, c),
                        );
                    }
                    #[cfg(all(feature = "std", not(miri)))]
                    if $arithmetic == STD {
                        return $V::from_array_t(
                            $token,
                            core::array::from_fn(|i| a[i].mul_add(b[i], c[i])),
                        );
                    }
                    $V::from_array_t($token, crate::soft_fma::fmaf_soft_lanes(a, b, c))
                }
            };
            if $lowp {
                let log = {
                    let xb = x.bitcast_to_i32();
                    let offset = ip(0x3f2a_aaab_u32 as i32);
                    let shifted = (xb - offset).shr_arithmetic_const::<23>();
                    let m = (xb - shifted.shl_const::<23>()).bitcast_to_f32() - sp(1.0);
                    let yp = madd(sp(0.742_458_7), m, sp(1.428_716_1));
                    let yp = madd(yp, m, sp(-1.850_383_3e-6));
                    let yq = madd(sp(0.174_093_43), m, sp(1.009_671_8));
                    let yq = madd(yq, m, sp(0.990_328_14));
                    let exp_adj = if $subnorm {
                        shifted - (is_subnorm & ip(23))
                    } else {
                        shifted
                    };
                    yp / yq + exp_adj.to_f32()
                };
                let t = log * sp(eff_exp);
                if !$assume && !$elide && !clamp_eff {
                    let outside = (t.simd_lt(sp(-126.0)) | t.simd_gt(sp(127.5))).bitcast_to_i32();
                    let outside = if $zero {
                        outside & (zero_mask ^ ip(-1))
                    } else {
                        outside
                    };
                    if outside.any_true() {
                        return Err(reject(outside, PowfErrorKind::OutsideApproximationRange));
                    }
                }
                let t = if ($zero && !$elide) || clamp_eff {
                    t.max(sp(-126.0)).min(sp(127.99))
                } else {
                    t
                };
                let xi = t.floor();
                let xf = t - xi;
                let p = madd(sp(0.055_504_11), xf, sp(0.240_226_5));
                let p = madd(p, xf, sp(core::f32::consts::LN_2));
                let p = madd(p, xf, sp(1.0));
                let integer = if $token.as_scalar().is_some() {
                    xi.to_i32()
                } else {
                    xi.to_i32_round()
                };
                p * (integer + ip(127)).shl_const::<23>().bitcast_to_f32()
            } else {
                let log = {
                    let adjusted = x.bitcast_to_i32() + ip((0x3f80_0000_u32 - 0x3f35_04f3) as i32);
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
                    let p = madd(sp(0.434_411_97), y2, sp(0.576_974_45));
                    let p = madd(p, y2, sp(0.961_800_76));
                    let p = madd(p, y2, sp(2.885_39));
                    madd(p, y, n)
                };
                let t = log * sp(eff_exp);
                if !$assume && !$elide && !clamp_eff {
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
                let reduced = if clamp_eff {
                    t.max(underflow_limit).min(overflow_limit)
                } else {
                    t
                };
                let xi = if $elide {
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
                let integer = if $token.as_scalar().is_some() {
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
                    let result = $V::blend(is_underflow, zero, result);
                    $V::blend(is_overflow, inf, result)
                } else {
                    result
                }
            }
        };
        let result =
            if $zero && (!$elide || $alg_kind == 0) && ($elide || eff_exp >= 0.0 || !$clamp_oob) {
                $V::blend(
                    zero_mask.bitcast_to_f32(),
                    sp(if eff_exp == 0.0 { 1.0 } else { 0.0 }),
                    result,
                )
            } else {
                result
            };
        let result = if $preserve_clamped_nan {
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
            exponent: f32,
            cfg: KernelCfg,
        ) -> Result<magetypes::simd::generic::$float<T>, PowfError> {
            use magetypes::simd::generic::{$float as V, $int as I};
            eval_powf!(
                V,
                I,
                token,
                input,
                exponent,
                ARITHMETIC,
                cfg.zero,
                cfg.subnorm,
                cfg.flush_sub,
                cfg.ranged,
                cfg.base_lo,
                cfg.base_hi,
                cfg.elide,
                cfg.clamp,
                cfg.clamp_oob,
                cfg.preserve_clamped_nan,
                cfg.lowp,
                cfg.assume,
                cfg.alg_kind,
                cfg.has_const_exp,
                cfg.const_exp
            )
        }

        #[cfg(not(debug_assertions))]
        #[inline(always)]
        fn $vector_rel<T: magetypes::simd::backends::$bound, K: Kernel, const ARITHMETIC: usize>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
            exponent: f32,
        ) -> Result<magetypes::simd::generic::$float<T>, PowfError> {
            use magetypes::simd::generic::{$float as V, $int as I};
            eval_powf!(
                V,
                I,
                token,
                input,
                exponent,
                ARITHMETIC,
                K::ZERO,
                K::SUBNORM,
                K::FLUSH_SUB,
                K::RANGED,
                K::BASE_LO,
                K::BASE_HI,
                K::ELIDE,
                K::CLAMP,
                K::CLAMP_OOB,
                K::PRESERVE_CLAMPED_NAN,
                K::LOWP,
                K::ASSUME,
                K::ALG_KIND,
                K::HAS_CONST_EXP,
                K::CONST_EXP
            )
        }

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
            #[cfg(debug_assertions)]
            return $vector_dbg::<T, ARITHMETIC>(token, input, exponent, cfg_of::<K>());
            #[cfg(not(debug_assertions))]
            $vector_rel::<T, K, ARITHMETIC>(token, input, exponent)
        }

        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
        pub(super) fn $cfg_entry<const ARITHMETIC: usize>(
            token: Token,
            bases: [f32; $lanes],
            exponent: f32,
            cfg: KernelCfg,
        ) -> Result<[f32; $lanes], PowfError> {
            $vector_dbg::<Token, ARITHMETIC>(
                token,
                magetypes::simd::generic::$float::from_array_t(token, bases),
                exponent,
                cfg,
            )
            .map(|result| result.to_array())
        }

        #[cfg(not(debug_assertions))]
        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
        pub(super) fn $const_entry<K: Kernel, const ARITHMETIC: usize>(
            token: Token,
            bases: [f32; $lanes],
            exponent: f32,
        ) -> Result<[f32; $lanes], PowfError> {
            $profile::<Token, K, ARITHMETIC>(
                token,
                magetypes::simd::generic::$float::from_array_t(token, bases),
                exponent,
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

/// Whole-slice kernel parameterized on `KernelCfg` (used by `.runtime()` and
/// debug builds): one feature region around the loop.
#[archmage::magetypes(v3, neon, wasm128, scalar)]
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
        vector8_dbg::<Token, ARITHMETIC>(token, V::load_t(token, s), exponent, cfg)
            .map_err(|e| (i * 8, e))?
            .store(d);
    }
    if !tail.is_empty() {
        vector8_dbg::<Token, ARITHMETIC>(
            token,
            V::load_padded_t(token, tail, tail[0]),
            exponent,
            cfg,
        )
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
    exponent: f32,
    cfg: KernelCfg,
) -> Result<(), (usize, PowfError)> {
    use magetypes::simd::generic::f32x16 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        vector16_dbg::<T, ARITHMETIC>(token, V::load_t(token, s), exponent, cfg)
            .map_err(|e| (i * 16, e))?
            .store(d);
    }
    if !tail.is_empty() {
        vector16_dbg::<T, ARITHMETIC>(token, V::load_padded_t(token, tail, tail[0]), exponent, cfg)
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
    exponent: f32,
    cfg: KernelCfg,
) -> Result<(), (usize, PowfError)> {
    slice16_cfg_impl::<_, ARITHMETIC>(token, src, dst, exponent, cfg)
}

/// 16-lane AVX-512 (`V4x`) whole-slice kernel parameterized on `KernelCfg`.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
#[archmage::arcane]
pub(super) fn slice16_cfg_v4x<const ARITHMETIC: usize>(
    token: archmage::X64V4xToken,
    src: &[f32],
    dst: &mut [f32],
    exponent: f32,
    cfg: KernelCfg,
) -> Result<(), (usize, PowfError)> {
    slice16_cfg_impl::<_, ARITHMETIC>(token, src, dst, exponent, cfg)
}

/// Whole-slice kernel: one feature region around the loop, the 8-lane graph
/// per chunk, and one tail vector padded with `tail[0]` so any range that
/// excludes `1.0` validates accurately and still reports exact first failure.
#[cfg(not(debug_assertions))]
#[archmage::magetypes(v3, neon, wasm128, scalar)]
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
        profile8::<Token, K, ARITHMETIC>(token, V::load_t(token, s), exponent)
            .map_err(|e| (i * 8, e))?
            .store(d);
    }
    if !tail.is_empty() {
        profile8::<Token, K, ARITHMETIC>(token, V::load_padded_t(token, tail, tail[0]), exponent)
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
    const ARITHMETIC: usize,
>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
    exponent: f32,
) -> Result<(), (usize, PowfError)> {
    use magetypes::simd::generic::f32x16 as V;
    let (chunks, tail) = V::partition_slice_t(token, src);
    let (out_chunks, out_tail) = V::partition_slice_mut_t(token, dst);
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        profile16::<T, K, ARITHMETIC>(token, V::load_t(token, s), exponent)
            .map_err(|e| (i * 16, e))?
            .store(d);
    }
    if !tail.is_empty() {
        profile16::<T, K, ARITHMETIC>(token, V::load_padded_t(token, tail, tail[0]), exponent)
            .map_err(|e| (chunks.len() * 16, e))?
            .store_prefix(out_tail);
    }
    Ok(())
}

/// 16-lane AVX-512 (`V4`) whole-slice kernel (`zmm` native registers).
#[cfg(all(
    not(debug_assertions),
    all(target_arch = "x86_64", target_feature = "sse2"),
    feature = "avx512"
))]
#[archmage::arcane]
pub(super) fn slice16_with_v4<K: Kernel, const ARITHMETIC: usize>(
    token: archmage::X64V4Token,
    src: &[f32],
    dst: &mut [f32],
    exponent: f32,
) -> Result<(), (usize, PowfError)> {
    slice16_with_impl::<_, K, ARITHMETIC>(token, src, dst, exponent)
}

/// 16-lane AVX-512 (`V4x`) whole-slice kernel (`zmm` native registers).
#[cfg(all(
    not(debug_assertions),
    all(target_arch = "x86_64", target_feature = "sse2"),
    feature = "avx512"
))]
#[archmage::arcane]
pub(super) fn slice16_with_v4x<K: Kernel, const ARITHMETIC: usize>(
    token: archmage::X64V4xToken,
    src: &[f32],
    dst: &mut [f32],
    exponent: f32,
) -> Result<(), (usize, PowfError)> {
    slice16_with_impl::<_, K, ARITHMETIC>(token, src, dst, exponent)
}

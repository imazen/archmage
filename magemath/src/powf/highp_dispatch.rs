//! One whole-operation feature selection for the canonical HIGHP graph.
#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
use super::highp_simd;
use super::{PowfError, PowfOptions, dispatch::Plan, graph, highp, highp_scalar};
use crate::Execution;
use archmage::ScalarToken;
use highp::{HighpCfg, RUNTIME};

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
fn run<
    const CFG: u128,
    const IN_CONTEXT: bool,
    const ELIDE: bool,
    const ZERO: bool,
    const N: usize,
>(
    plan: Plan,
    cfg: HighpCfg,
    bases: [f32; N],
    n: f32,
) -> Result<[f32; N], PowfError> {
    let cfg = cfg.resolve::<CFG>();
    let (bases, nan_mask) = highp::sanitize_bases_cfg::<N>(cfg, bases, n);
    highp::check_bases_cfg(cfg, &bases, n)?;
    let mut out = if cfg.alg_kind != 0 {
        match plan {
            #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
            Plan::V3(t) => {
                if IN_CONTEXT {
                    highp::eval_alg_token::<_, N>(t, cfg.alg_kind, bases)
                } else {
                    v3_alg::<CFG, N>(t, cfg, bases)
                }
            }
            #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
            Plan::V4(t) => {
                if IN_CONTEXT {
                    highp::eval_alg_token::<_, N>(t, cfg.alg_kind, bases)
                } else {
                    v4_alg::<CFG, N>(t, cfg, bases)
                }
            }
            #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
            Plan::V4x(t) => {
                if IN_CONTEXT {
                    highp::eval_alg_token::<_, N>(t, cfg.alg_kind, bases)
                } else {
                    v4x_alg::<CFG, N>(t, cfg, bases)
                }
            }
            #[cfg(target_arch = "aarch64")]
            Plan::Neon(t) => {
                if IN_CONTEXT {
                    highp::eval_alg_token::<_, N>(t, cfg.alg_kind, bases)
                } else {
                    neon_alg::<CFG, N>(t, cfg, bases)
                }
            }
            #[cfg(target_arch = "wasm32")]
            Plan::Wasm(t, _) => {
                if IN_CONTEXT {
                    highp::eval_alg_token::<_, N>(t, cfg.alg_kind, bases)
                } else {
                    wasm_alg::<CFG, N>(t, cfg, bases)
                }
            }
            _ => highp::eval_alg_token::<_, N>(ScalarToken, cfg.alg_kind, bases),
        }
    } else {
        match plan {
            #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
            Plan::V3(t) => {
                if IN_CONTEXT {
                    highp_simd::packed4::<archmage::X64V3Token, ELIDE, ZERO, N>(t, bases, n)
                } else {
                    v3::<ELIDE, ZERO, N>(t, bases, n)
                }
            }
            #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
            Plan::V4(t) => {
                if IN_CONTEXT {
                    highp_simd::wide::<archmage::X64V4Token, ELIDE, ZERO, N>(t, bases, n)
                } else {
                    v4::<ELIDE, ZERO, N>(t, bases, n)
                }
            }
            #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
            Plan::V4x(t) => {
                if IN_CONTEXT {
                    highp_simd::wide::<archmage::X64V4xToken, ELIDE, ZERO, N>(t, bases, n)
                } else {
                    v4x::<ELIDE, ZERO, N>(t, bases, n)
                }
            }
            #[cfg(target_arch = "aarch64")]
            Plan::Neon(t) => {
                if IN_CONTEXT {
                    highp_simd::packed2::<archmage::NeonToken, ELIDE, ZERO, N>(t, bases, n)
                } else {
                    neon::<ELIDE, ZERO, N>(t, bases, n)
                }
            }
            #[cfg(target_arch = "wasm32")]
            Plan::Wasm(t, true) => {
                if IN_CONTEXT {
                    highp_simd::packed2::<archmage::Wasm128Token, ELIDE, ZERO, N>(t, bases, n)
                } else {
                    wasm::<ELIDE, ZERO, N>(t, bases, n)
                }
            }
            _ => scalar::<ScalarToken, { graph::SOFTWARE }, ELIDE, N>(ScalarToken, bases, n),
        }
    };
    highp::check_results_cfg(cfg, &bases, &out, n)?;
    highp::restore_nans(&mut out, nan_mask);
    Ok(out)
}

#[inline]
fn scalar<
    T: magetypes::simd::backends::F64x2Backend,
    const MODE: usize,
    const ELIDE: bool,
    const N: usize,
>(
    t: T,
    x: [f32; N],
    n: f32,
) -> [f32; N] {
    let mut out = [0.; N];
    let mut i = 0;
    while i < N {
        out[i] = highp_scalar::pow::<T, MODE, ELIDE>(t, x[i], n);
        i += 1;
    }
    out
}

#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
macro_rules! entry {
    ($name:ident,$alg_name:ident,$token:ident,$packed:ident) => {
        #[archmage::arcane]
        fn $name<const ELIDE: bool, const ZERO: bool, const N: usize>(
            t: archmage::$token,
            x: [f32; N],
            n: f32,
        ) -> [f32; N] {
            highp_simd::$packed::<_, ELIDE, ZERO, N>(t, x, n)
        }

        #[archmage::arcane]
        fn $alg_name<const CFG: u128, const N: usize>(
            t: archmage::$token,
            cfg: HighpCfg,
            x: [f32; N],
        ) -> [f32; N] {
            highp::eval_alg_token::<_, N>(t, cfg.resolve::<CFG>().alg_kind, x)
        }
    };
}
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
entry!(v3, v3_alg, X64V3Token, packed4);
#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
entry!(v4, v4_alg, X64V4Token, wide);
#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
entry!(v4x, v4x_alg, X64V4xToken, wide);
#[cfg(target_arch = "aarch64")]
entry!(neon, neon_alg, NeonToken, packed2);
#[cfg(target_arch = "wasm32")]
entry!(wasm, wasm_alg, Wasm128Token, packed2);

#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
fn eval_alg_slice_loop<T: magetypes::simd::backends::F64x2Backend>(
    t: T,
    cfg: HighpCfg,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
) -> Result<(), (usize, PowfError)> {
    let (chunks, tail) = src.as_chunks::<8>();
    let (out_chunks, out_tail) = dst.as_chunks_mut::<8>();
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        let (bases, nan_mask) = highp::sanitize_bases_cfg::<8>(cfg, *s, n);
        highp::check_bases_cfg(cfg, &bases, n).map_err(|e| (i * 8, e))?;
        let mut out = highp::eval_alg_token::<T, 8>(t, cfg.alg_kind, bases);
        highp::check_results_cfg(cfg, &bases, &out, n).map_err(|e| (i * 8, e))?;
        highp::restore_nans(&mut out, nan_mask);
        *d = out;
    }
    let tail_base = chunks.len() * 8;
    for (i, (&x, d)) in tail.iter().zip(out_tail.iter_mut()).enumerate() {
        let ([bx], nan_mask) = highp::sanitize_bases_cfg::<1>(cfg, [x], n);
        highp::check_bases_cfg(cfg, &[bx], n)
            .map_err(|e| (tail_base + i, PowfError { lane_mask: 1, ..e }))?;
        let mut out = highp::eval_alg_token::<T, 1>(t, cfg.alg_kind, [bx]);
        highp::check_results_cfg(cfg, &[bx], &out, n)
            .map_err(|e| (tail_base + i, PowfError { lane_mask: 1, ..e }))?;
        highp::restore_nans(&mut out, nan_mask);
        *d = out[0];
    }
    Ok(())
}

#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
fn eval_slice_loop<
    const CFG: u128,
    T: magetypes::simd::backends::F64x2Backend,
    const MODE: usize,
    const ELIDE: bool,
    F: FnMut(T, [f32; 8], f32) -> [f32; 8],
>(
    t: T,
    cfg: HighpCfg,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
    mut eval8: F,
) -> Result<(), (usize, PowfError)> {
    let cfg = cfg.resolve::<CFG>();
    if cfg.alg_kind != 0 {
        return eval_alg_slice_loop(t, cfg, src, dst, n);
    }
    let (chunks, tail) = src.as_chunks::<8>();
    let (out_chunks, out_tail) = dst.as_chunks_mut::<8>();
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        let (bases, nan_mask) = highp::sanitize_bases_cfg::<8>(cfg, *s, n);
        highp::check_bases_cfg(cfg, &bases, n).map_err(|e| (i * 8, e))?;
        let mut out = eval8(t, bases, n);
        highp::check_results_cfg(cfg, &bases, &out, n).map_err(|e| (i * 8, e))?;
        highp::restore_nans(&mut out, nan_mask);
        *d = out;
    }
    let tail_base = chunks.len() * 8;
    for (i, (&x, d)) in tail.iter().zip(out_tail.iter_mut()).enumerate() {
        let ([bx], nan_mask) = highp::sanitize_bases_cfg::<1>(cfg, [x], n);
        highp::check_bases_cfg(cfg, &[bx], n)
            .map_err(|e| (tail_base + i, PowfError { lane_mask: 1, ..e }))?;
        let mut out = [highp_scalar::pow::<T, MODE, ELIDE>(t, bx, n)];
        highp::check_results_cfg(cfg, &[bx], &out, n)
            .map_err(|e| (tail_base + i, PowfError { lane_mask: 1, ..e }))?;
        highp::restore_nans(&mut out, nan_mask);
        *d = out[0];
    }
    Ok(())
}

#[inline(never)]
fn scalar_slice<const CFG: u128, const ELIDE: bool>(
    cfg: HighpCfg,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
) -> Result<(), (usize, PowfError)> {
    let cfg = cfg.resolve::<CFG>();
    for (i, (&x, d)) in src.iter().zip(dst.iter_mut()).enumerate() {
        let ([base], nan_mask) = highp::sanitize_bases_cfg(cfg, [x], n);
        highp::check_bases_cfg(cfg, &[base], n).map_err(|e| (i, e))?;
        let mut out = if cfg.alg_kind == 0 {
            [highp_scalar::pow::<_, { graph::SOFTWARE }, ELIDE>(
                ScalarToken,
                base,
                n,
            )]
        } else {
            highp::eval_alg_token(ScalarToken, cfg.alg_kind, [base])
        };
        highp::check_results_cfg(cfg, &[base], &out, n).map_err(|e| (i, e))?;
        highp::restore_nans(&mut out, nan_mask);
        *d = out[0];
    }
    Ok(())
}

#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
macro_rules! slice_entry {
    ($name:ident, $token:ident, $packed:ident) => {
        #[archmage::arcane]
        fn $name<const CFG: u128, const ELIDE: bool, const ZERO: bool>(
            t: archmage::$token,
            cfg: HighpCfg,
            src: &[f32],
            dst: &mut [f32],
            n: f32,
        ) -> Result<(), (usize, PowfError)> {
            eval_slice_loop::<CFG, _, { graph::NATIVE }, ELIDE, _>(
                t,
                cfg,
                src,
                dst,
                n,
                highp_simd::$packed::<_, ELIDE, ZERO, 8>,
            )
        }
    };
}
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
slice_entry!(v3_slice, X64V3Token, packed4);
#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
slice_entry!(v4_slice, X64V4Token, wide);
#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
slice_entry!(v4x_slice, X64V4xToken, wide);
#[cfg(target_arch = "aarch64")]
slice_entry!(neon_slice, NeonToken, packed2);
#[cfg(target_arch = "wasm32")]
slice_entry!(wasm_slice, Wasm128Token, packed2);

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
fn run_slice<const CFG: u128, const IN_CONTEXT: bool, const ELIDE: bool, const ZERO: bool>(
    plan: Plan,
    cfg: HighpCfg,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
) -> Result<(), (usize, PowfError)> {
    match plan {
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        Plan::V3(t) => {
            if IN_CONTEXT {
                eval_slice_loop::<CFG, _, { graph::NATIVE }, ELIDE, _>(
                    t,
                    cfg,
                    src,
                    dst,
                    n,
                    highp_simd::packed4::<archmage::X64V3Token, ELIDE, ZERO, 8>,
                )
            } else {
                v3_slice::<CFG, ELIDE, ZERO>(t, cfg, src, dst, n)
            }
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
        Plan::V4(t) => {
            if IN_CONTEXT {
                eval_slice_loop::<CFG, _, { graph::NATIVE }, ELIDE, _>(
                    t,
                    cfg,
                    src,
                    dst,
                    n,
                    highp_simd::wide::<archmage::X64V4Token, ELIDE, ZERO, 8>,
                )
            } else {
                v4_slice::<CFG, ELIDE, ZERO>(t, cfg, src, dst, n)
            }
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
        Plan::V4x(t) => {
            if IN_CONTEXT {
                eval_slice_loop::<CFG, _, { graph::NATIVE }, ELIDE, _>(
                    t,
                    cfg,
                    src,
                    dst,
                    n,
                    highp_simd::wide::<archmage::X64V4xToken, ELIDE, ZERO, 8>,
                )
            } else {
                v4x_slice::<CFG, ELIDE, ZERO>(t, cfg, src, dst, n)
            }
        }
        #[cfg(target_arch = "aarch64")]
        Plan::Neon(t) => {
            if IN_CONTEXT {
                eval_slice_loop::<CFG, _, { graph::NATIVE }, ELIDE, _>(
                    t,
                    cfg,
                    src,
                    dst,
                    n,
                    highp_simd::packed2::<archmage::NeonToken, ELIDE, ZERO, 8>,
                )
            } else {
                neon_slice::<CFG, ELIDE, ZERO>(t, cfg, src, dst, n)
            }
        }
        #[cfg(target_arch = "wasm32")]
        Plan::Wasm(t, true) => {
            if IN_CONTEXT {
                eval_slice_loop::<CFG, _, { graph::NATIVE }, ELIDE, _>(
                    t,
                    cfg,
                    src,
                    dst,
                    n,
                    highp_simd::packed2::<archmage::Wasm128Token, ELIDE, ZERO, 8>,
                )
            } else {
                wasm_slice::<CFG, ELIDE, ZERO>(t, cfg, src, dst, n)
            }
        }
        _ => scalar_slice::<CFG, ELIDE>(cfg, src, dst, n),
    }
}

// The direct form inlines into an already capable vector caller. Upgrades
// enter one arcane boundary for the complete graph instead.
#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline)]
pub(super) fn for_token_cfg<T: archmage::IntoConcreteToken, const N: usize>(
    t: T,
    opts: PowfOptions,
    bases: [f32; N],
    n: f32,
) -> Result<[f32; N], PowfError> {
    let cfg = HighpCfg::from_options(opts);
    #[cfg(feature = "std")]
    if opts.is_backend() {
        return highp::run_cfg::<N>(cfg, bases, n);
    }
    let caller = super::dispatch::select_highp(t, Execution::Caller, false);
    let plan = if opts.is_caller() {
        caller
    } else {
        super::dispatch::select_highp(t, Execution::Upgrade, N >= 8)
    };
    let in_context = !matches!(caller, Plan::Software) && plan.same_tier(caller);
    let elide = opts.elides_range_check() && !opts.admits_subnormals();
    match (in_context, elide, opts.admits_zero()) {
        (true, true, true) => run::<RUNTIME, true, true, true, N>(plan, cfg, bases, n),
        (true, true, false) => run::<RUNTIME, true, true, false, N>(plan, cfg, bases, n),
        (true, false, _) => run::<RUNTIME, true, false, false, N>(plan, cfg, bases, n),
        (false, true, true) => run::<RUNTIME, false, true, true, N>(plan, cfg, bases, n),
        (false, true, false) => run::<RUNTIME, false, true, false, N>(plan, cfg, bases, n),
        (false, false, _) => run::<RUNTIME, false, false, false, N>(plan, cfg, bases, n),
    }
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline)]
pub(super) fn for_token_slice_cfg<T: archmage::IntoConcreteToken>(
    t: T,
    opts: PowfOptions,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
) -> Result<(), (usize, PowfError)> {
    let cfg = HighpCfg::from_options(opts);
    #[cfg(feature = "std")]
    if opts.is_backend() {
        return std_slice(cfg, src, dst, n);
    }
    let caller = super::dispatch::select_highp(t, Execution::Caller, false);
    let plan = if opts.is_caller() {
        caller
    } else {
        super::dispatch::select_highp(t, Execution::Upgrade, true)
    };
    let in_context = !matches!(caller, Plan::Software) && plan.same_tier(caller);
    let elide = opts.elides_range_check() && !opts.admits_subnormals();
    match (in_context, elide, opts.admits_zero()) {
        (true, true, true) => run_slice::<RUNTIME, true, true, true>(plan, cfg, src, dst, n),
        (true, true, false) => run_slice::<RUNTIME, true, true, false>(plan, cfg, src, dst, n),
        (true, false, _) => run_slice::<RUNTIME, true, false, false>(plan, cfg, src, dst, n),
        (false, true, true) => run_slice::<RUNTIME, false, true, true>(plan, cfg, src, dst, n),
        (false, true, false) => run_slice::<RUNTIME, false, true, false>(plan, cfg, src, dst, n),
        (false, false, _) => run_slice::<RUNTIME, false, false, false>(plan, cfg, src, dst, n),
    }
}

#[cfg(feature = "std")]
#[inline(always)]
fn std_slice(
    cfg: HighpCfg,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
) -> Result<(), (usize, PowfError)> {
    let (chunks, tail) = src.as_chunks::<8>();
    let (out_chunks, out_tail) = dst.as_chunks_mut::<8>();
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        *d = highp::run_cfg::<8>(cfg, *s, n).map_err(|e| (i * 8, e))?;
    }
    let tail_base = chunks.len() * 8;
    for (i, (&x, d)) in tail.iter().zip(out_tail.iter_mut()).enumerate() {
        *d = highp::run_cfg::<1>(cfg, [x], n)
            .map_err(|e| (tail_base + i, PowfError { lane_mask: 1, ..e }))?[0];
    }
    Ok(())
}

macro_rules! run_const {
    ($run:ident, $cfg_bits:ident, $in_context:expr, [$($width:ident)?], $($args:expr),+ $(,)?) => {{
        if const {
            let opts = PowfOptions::decode_const($cfg_bits);
            opts.elides_range_check() && !opts.admits_subnormals()
        } {
            if const { PowfOptions::decode_const($cfg_bits).admits_zero() } {
                if $in_context {
                    $run::<$cfg_bits, true, true, true $(, $width)?>($($args),+)
                } else {
                    $run::<$cfg_bits, false, true, true $(, $width)?>($($args),+)
                }
            } else if $in_context {
                $run::<$cfg_bits, true, true, false $(, $width)?>($($args),+)
            } else {
                $run::<$cfg_bits, false, true, false $(, $width)?>($($args),+)
            }
        } else if $in_context {
            $run::<$cfg_bits, true, false, false $(, $width)?>($($args),+)
        } else {
            $run::<$cfg_bits, false, false, false $(, $width)?>($($args),+)
        }
    }};
}

// Resolve the policy at compile time before entering a feature boundary.
// Caller plans and the numerical policy can then fold even at opt-level=z.
#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn for_token_const<const CFG: u128, T: archmage::IntoConcreteToken, const N: usize>(
    t: T,
    bases: [f32; N],
    n: f32,
) -> Result<[f32; N], PowfError> {
    let cfg = const { HighpCfg::from_options(PowfOptions::decode_const(CFG)) };
    #[cfg(feature = "std")]
    if const { PowfOptions::decode_const(CFG).is_backend() } {
        return highp::run_cfg::<N>(cfg, bases, n);
    }
    let caller = super::dispatch::select_highp(t, Execution::Caller, false);
    let plan = if const { PowfOptions::decode_const(CFG).is_caller() } {
        caller
    } else {
        super::dispatch::select_highp(t, Execution::Upgrade, N >= 8)
    };
    let in_context = !matches!(caller, Plan::Software) && plan.same_tier(caller);
    run_const!(run, CFG, in_context, [N], plan, cfg, bases, n)
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn for_token_slice_const<const CFG: u128, T: archmage::IntoConcreteToken>(
    t: T,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
) -> Result<(), (usize, PowfError)> {
    let cfg = const { HighpCfg::from_options(PowfOptions::decode_const(CFG)) };
    #[cfg(feature = "std")]
    if const { PowfOptions::decode_const(CFG).is_backend() } {
        return std_slice(cfg, src, dst, n);
    }
    let caller = super::dispatch::select_highp(t, Execution::Caller, false);
    let plan = if const { PowfOptions::decode_const(CFG).is_caller() } {
        caller
    } else {
        super::dispatch::select_highp(t, Execution::Upgrade, true)
    };
    let in_context = !matches!(caller, Plan::Software) && plan.same_tier(caller);
    run_const!(run_slice, CFG, in_context, [], plan, cfg, src, dst, n)
}

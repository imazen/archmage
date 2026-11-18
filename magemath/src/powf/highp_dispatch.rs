//! One whole-operation feature selection for the canonical HIGHP graph.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
use super::highp_simd;
use super::{PowfError, PowfOptions, dispatch::Plan, graph, highp, highp_scalar};
use crate::Execution;
use archmage::ScalarToken;

#[inline(always)]
fn run<const IN_CONTEXT: bool, const ELIDE: bool, const ZERO: bool, const N: usize>(
    plan: Plan,
    cfg: highp::HighpCfg,
    bases: [f32; N],
    n: f32,
) -> Result<[f32; N], PowfError> {
    let (bases, nan_mask) = highp::sanitize_bases_cfg::<N>(cfg, bases, n);
    highp::check_bases_cfg(cfg, &bases, n)?;
    let mut out = match plan {
        #[cfg(target_arch = "x86_64")]
        Plan::V3(t) => {
            if IN_CONTEXT {
                highp_simd::packed4::<archmage::X64V3Token, ELIDE, ZERO, N>(t, bases, n)
            } else {
                v3::<ELIDE, ZERO, N>(t, bases, n)
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Plan::V4(t) => {
            if IN_CONTEXT {
                highp_simd::wide::<archmage::X64V4Token, ELIDE, ZERO, N>(t, bases, n)
            } else {
                v4::<ELIDE, ZERO, N>(t, bases, n)
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
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
    };
    highp::check_results_cfg(cfg, &bases, &out, n)?;
    highp::restore_nans(&mut out, nan_mask);
    Ok(out)
}

#[inline(always)]
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
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
macro_rules! entry {
    ($name:ident,$token:ident,$packed:ident) => {
        #[archmage::arcane]
        fn $name<const ELIDE: bool, const ZERO: bool, const N: usize>(
            t: archmage::$token,
            x: [f32; N],
            n: f32,
        ) -> [f32; N] {
            highp_simd::$packed::<_, ELIDE, ZERO, N>(t, x, n)
        }
    };
}
#[cfg(target_arch = "x86_64")]
entry!(v3, X64V3Token, packed4);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
entry!(v4, X64V4Token, wide);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
entry!(v4x, X64V4xToken, wide);
#[cfg(target_arch = "aarch64")]
entry!(neon, NeonToken, packed2);
#[cfg(target_arch = "wasm32")]
entry!(wasm, Wasm128Token, packed2);

#[inline(always)]
fn eval_slice_loop<
    T: magetypes::simd::backends::F64x2Backend,
    const MODE: usize,
    const ELIDE: bool,
    F: FnMut(T, [f32; 8], f32) -> [f32; 8],
>(
    t: T,
    cfg: highp::HighpCfg,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
    mut eval8: F,
) -> Result<(), (usize, PowfError)> {
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

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
macro_rules! slice_entry {
    ($name:ident, $token:ident, $packed:ident) => {
        #[archmage::arcane]
        fn $name<const ELIDE: bool, const ZERO: bool>(
            t: archmage::$token,
            cfg: highp::HighpCfg,
            src: &[f32],
            dst: &mut [f32],
            n: f32,
        ) -> Result<(), (usize, PowfError)> {
            eval_slice_loop::<_, { graph::NATIVE }, ELIDE, _>(
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
#[cfg(target_arch = "x86_64")]
slice_entry!(v3_slice, X64V3Token, packed4);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
slice_entry!(v4_slice, X64V4Token, wide);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
slice_entry!(v4x_slice, X64V4xToken, wide);
#[cfg(target_arch = "aarch64")]
slice_entry!(neon_slice, NeonToken, packed2);
#[cfg(target_arch = "wasm32")]
slice_entry!(wasm_slice, Wasm128Token, packed2);

#[inline(always)]
fn run_slice<const IN_CONTEXT: bool, const ELIDE: bool, const ZERO: bool>(
    plan: Plan,
    cfg: highp::HighpCfg,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
) -> Result<(), (usize, PowfError)> {
    match plan {
        #[cfg(target_arch = "x86_64")]
        Plan::V3(t) => {
            if IN_CONTEXT {
                eval_slice_loop::<_, { graph::NATIVE }, ELIDE, _>(
                    t,
                    cfg,
                    src,
                    dst,
                    n,
                    highp_simd::packed4::<archmage::X64V3Token, ELIDE, ZERO, 8>,
                )
            } else {
                v3_slice::<ELIDE, ZERO>(t, cfg, src, dst, n)
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Plan::V4(t) => {
            if IN_CONTEXT {
                eval_slice_loop::<_, { graph::NATIVE }, ELIDE, _>(
                    t,
                    cfg,
                    src,
                    dst,
                    n,
                    highp_simd::wide::<archmage::X64V4Token, ELIDE, ZERO, 8>,
                )
            } else {
                v4_slice::<ELIDE, ZERO>(t, cfg, src, dst, n)
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Plan::V4x(t) => {
            if IN_CONTEXT {
                eval_slice_loop::<_, { graph::NATIVE }, ELIDE, _>(
                    t,
                    cfg,
                    src,
                    dst,
                    n,
                    highp_simd::wide::<archmage::X64V4xToken, ELIDE, ZERO, 8>,
                )
            } else {
                v4x_slice::<ELIDE, ZERO>(t, cfg, src, dst, n)
            }
        }
        #[cfg(target_arch = "aarch64")]
        Plan::Neon(t) => {
            if IN_CONTEXT {
                eval_slice_loop::<_, { graph::NATIVE }, ELIDE, _>(
                    t,
                    cfg,
                    src,
                    dst,
                    n,
                    highp_simd::packed2::<archmage::NeonToken, ELIDE, ZERO, 8>,
                )
            } else {
                neon_slice::<ELIDE, ZERO>(t, cfg, src, dst, n)
            }
        }
        #[cfg(target_arch = "wasm32")]
        Plan::Wasm(t, true) => {
            if IN_CONTEXT {
                eval_slice_loop::<_, { graph::NATIVE }, ELIDE, _>(
                    t,
                    cfg,
                    src,
                    dst,
                    n,
                    highp_simd::packed2::<archmage::Wasm128Token, ELIDE, ZERO, 8>,
                )
            } else {
                wasm_slice::<ELIDE, ZERO>(t, cfg, src, dst, n)
            }
        }
        _ => eval_slice_loop::<_, { graph::SOFTWARE }, ELIDE, _>(
            ScalarToken,
            cfg,
            src,
            dst,
            n,
            scalar::<ScalarToken, { graph::SOFTWARE }, ELIDE, 8>,
        ),
    }
}

// The direct form inlines into an already capable vector caller. Upgrades
// enter one arcane boundary for the complete graph instead.
#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn for_token_cfg<T: archmage::IntoConcreteToken, const N: usize>(
    t: T,
    opts: PowfOptions,
    bases: [f32; N],
    n: f32,
) -> Result<[f32; N], PowfError> {
    let cfg = highp::HighpCfg::from_options(opts);
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
        (true, true, true) => run::<true, true, true, N>(plan, cfg, bases, n),
        (true, true, false) => run::<true, true, false, N>(plan, cfg, bases, n),
        (true, false, _) => run::<true, false, false, N>(plan, cfg, bases, n),
        (false, true, true) => run::<false, true, true, N>(plan, cfg, bases, n),
        (false, true, false) => run::<false, true, false, N>(plan, cfg, bases, n),
        (false, false, _) => run::<false, false, false, N>(plan, cfg, bases, n),
    }
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn for_token_slice_cfg<T: archmage::IntoConcreteToken>(
    t: T,
    opts: PowfOptions,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
) -> Result<(), (usize, PowfError)> {
    let cfg = highp::HighpCfg::from_options(opts);
    #[cfg(feature = "std")]
    if opts.is_backend() {
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
        return Ok(());
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
        (true, true, true) => run_slice::<true, true, true>(plan, cfg, src, dst, n),
        (true, true, false) => run_slice::<true, true, false>(plan, cfg, src, dst, n),
        (true, false, _) => run_slice::<true, false, false>(plan, cfg, src, dst, n),
        (false, true, true) => run_slice::<false, true, true>(plan, cfg, src, dst, n),
        (false, true, false) => run_slice::<false, true, false>(plan, cfg, src, dst, n),
        (false, false, _) => run_slice::<false, false, false>(plan, cfg, src, dst, n),
    }
}

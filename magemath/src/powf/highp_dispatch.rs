//! One whole-operation feature selection for the canonical HIGHP graph.
#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
use super::highp_simd;
use super::{
    PowfError, PowfErrorKind, PowfOptions, const_api::refine_slice_error, dispatch::Plan, graph,
    highp, highp_scalar,
};
use crate::{Execution, SliceError};
use archmage::ScalarToken;
use highp::HighpCfg;

type SliceResult = Result<(), (usize, PowfError)>;

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn for_token_const<const CFG: u128, T: archmage::IntoConcreteToken, const N: usize>(
    t: T,
    bases: [f32; N],
    n: f32,
) -> Result<[f32; N], PowfError> {
    let (opts, cfg) = const {
        let opts = PowfOptions::decode_const(CFG);
        (opts, HighpCfg::from_options(opts))
    };
    #[cfg(feature = "std")]
    if opts.is_backend() {
        return highp::run_cfg(cfg, bases, n);
    }
    let (plan, in_context) = select(t, opts.is_caller(), N >= 8);
    eval_array(plan, in_context, cfg, bases, n)
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn for_token_slice_const<const CFG: u128, T: archmage::IntoConcreteToken>(
    t: T,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
) -> Result<(), SliceError<PowfErrorKind>> {
    let (opts, cfg) = const {
        let opts = PowfOptions::decode_const(CFG);
        (opts, HighpCfg::from_options(opts))
    };
    slice_with_options(t, opts, cfg, src, dst, n)
}

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
        return highp::run_cfg(cfg, bases, n);
    }
    let (plan, in_context) = select(t, opts.is_caller(), N >= 8);
    eval_array(plan, in_context, cfg, bases, n)
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline)]
pub(super) fn for_token_slice_cfg<T: archmage::IntoConcreteToken>(
    t: T,
    opts: PowfOptions,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
) -> Result<(), SliceError<PowfErrorKind>> {
    let cfg = HighpCfg::from_options(opts);
    slice_with_options(t, opts, cfg, src, dst, n)
}

#[inline(always)]
fn slice_with_options(
    t: impl archmage::IntoConcreteToken,
    opts: PowfOptions,
    cfg: HighpCfg,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
) -> Result<(), SliceError<PowfErrorKind>> {
    #[cfg(feature = "std")]
    if opts.is_backend() {
        return std_slice(cfg, src, dst, n);
    }
    let (plan, in_context) = select(t, opts.is_caller(), true);
    eval_slice(plan, in_context, cfg, src, dst, n).map_err(|(base, error)| {
        let width = plan.midp_slice_width().min(8);
        if opts.is_infallible() || width == 1 || src.len() - base < width {
            SliceError {
                kind: error.kind,
                index: base,
            }
        } else {
            refine_highp_chunk(plan, in_context, cfg, src, base, error, n)
        }
    })
}

#[inline(always)]
fn prepare<const VALIDATE: bool, const N: usize>(
    cfg: HighpCfg,
    bases: [f32; N],
    n: f32,
) -> Result<([f32; N], u16), PowfError> {
    if !VALIDATE {
        return Ok((bases, 0));
    }
    let (bases, nan_mask) = highp::sanitize_bases_cfg(cfg, bases, n);
    highp::check_bases_cfg(cfg, &bases, n)?;
    Ok((bases, nan_mask))
}

#[inline(always)]
fn finish<const VALIDATE: bool, const N: usize>(
    cfg: HighpCfg,
    bases: &[f32; N],
    mut out: [f32; N],
    n: f32,
    nan_mask: u16,
) -> Result<[f32; N], PowfError> {
    if VALIDATE {
        highp::check_results_cfg(cfg, bases, &out, n)?;
        highp::restore_nans(&mut out, nan_mask);
    }
    Ok(out)
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
fn run<const ELIDE: bool, const ZERO: bool, const N: usize>(
    plan: Plan,
    in_context: bool,
    cfg: HighpCfg,
    bases: [f32; N],
    n: f32,
) -> Result<[f32; N], PowfError> {
    let _ = in_context;
    let (bases, nan_mask) = prepare::<true, N>(cfg, bases, n)?;
    let out = if cfg.alg_kind != 0 {
        match plan {
            #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
            Plan::V3(t) => {
                if in_context {
                    highp::eval_alg_token::<_, N>(t, cfg.alg_kind, bases)
                } else {
                    v3_alg::<N>(t, cfg.alg_kind, bases)
                }
            }
            #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
            Plan::V4(t) => {
                if in_context {
                    highp::eval_alg_token::<_, N>(t, cfg.alg_kind, bases)
                } else {
                    v4_alg::<N>(t, cfg.alg_kind, bases)
                }
            }
            #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
            Plan::V4x(t) => {
                if in_context {
                    highp::eval_alg_token::<_, N>(t, cfg.alg_kind, bases)
                } else {
                    v4x_alg::<N>(t, cfg.alg_kind, bases)
                }
            }
            #[cfg(target_arch = "aarch64")]
            Plan::Neon(t) => {
                if in_context {
                    highp::eval_alg_token::<_, N>(t, cfg.alg_kind, bases)
                } else {
                    neon_alg::<N>(t, cfg.alg_kind, bases)
                }
            }
            #[cfg(target_arch = "wasm32")]
            Plan::Wasm(t, _) => {
                if in_context {
                    highp::eval_alg_token::<_, N>(t, cfg.alg_kind, bases)
                } else {
                    wasm_alg::<N>(t, cfg.alg_kind, bases)
                }
            }
            _ => highp::eval_alg_token::<_, N>(ScalarToken, cfg.alg_kind, bases),
        }
    } else {
        match plan {
            #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
            Plan::V3(t) => {
                if in_context {
                    highp_simd::packed4::<archmage::X64V3Token, ELIDE, ZERO, N>(t, bases, n)
                } else {
                    v3::<ELIDE, ZERO, N>(t, bases, n)
                }
            }
            #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
            Plan::V4(t) => {
                if in_context {
                    highp_simd::wide::<archmage::X64V4Token, ELIDE, ZERO, N>(t, bases, n)
                } else {
                    v4::<ELIDE, ZERO, N>(t, bases, n)
                }
            }
            #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
            Plan::V4x(t) => {
                if in_context {
                    highp_simd::wide::<archmage::X64V4xToken, ELIDE, ZERO, N>(t, bases, n)
                } else {
                    v4x::<ELIDE, ZERO, N>(t, bases, n)
                }
            }
            #[cfg(target_arch = "aarch64")]
            Plan::Neon(t) => {
                if in_context {
                    highp_simd::packed2::<archmage::NeonToken, ELIDE, ZERO, N>(t, bases, n)
                } else {
                    neon::<ELIDE, ZERO, N>(t, bases, n)
                }
            }
            #[cfg(target_arch = "wasm32")]
            Plan::Wasm(t, true) => {
                if in_context {
                    highp_simd::packed2::<archmage::Wasm128Token, ELIDE, ZERO, N>(t, bases, n)
                } else {
                    wasm::<ELIDE, ZERO, N>(t, bases, n)
                }
            }
            _ => scalar::<ScalarToken, { graph::SOFTWARE }, ELIDE, N>(ScalarToken, bases, n),
        }
    };
    finish::<true, N>(cfg, &bases, out, n, nan_mask)
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
macro_rules! packed_backend {
    ($token:ident, $packed:ident, $array:ident, $alg:ident, $slice:ident, $slice_boundary:ident, $width:literal) => {
        #[archmage::attune(wrap)]
        fn $array<const ELIDE: bool, const ZERO: bool, const N: usize>(
            t: archmage::$token,
            x: [f32; N],
            n: f32,
        ) -> [f32; N] {
            highp_simd::$packed::<_, ELIDE, ZERO, N>(t, x, n)
        }

        #[archmage::attune(wrap)]
        fn $alg<const N: usize>(t: archmage::$token, kind: u8, x: [f32; N]) -> [f32; N] {
            highp::eval_alg_token(t, kind, x)
        }

        #[inline(always)]
        fn $slice<
            const ELIDE: bool,
            const ZERO: bool,
            const VALIDATE: bool,
            const ALGEBRAIC: bool,
        >(
            t: archmage::$token,
            in_context: bool,
            cfg: HighpCfg,
            src: &[f32],
            dst: &mut [f32],
            n: f32,
        ) -> SliceResult {
            if in_context {
                eval_slice_loop::<_, { graph::NATIVE }, ELIDE, VALIDATE, ALGEBRAIC, $width, _>(
                    t,
                    cfg,
                    src,
                    dst,
                    n,
                    highp_simd::$packed::<_, ELIDE, ZERO, $width>,
                )
            } else {
                $slice_boundary::<ELIDE, ZERO, VALIDATE, ALGEBRAIC>(t, cfg, src, dst, n)
            }
        }

        #[archmage::attune(wrap)]
        fn $slice_boundary<
            const ELIDE: bool,
            const ZERO: bool,
            const VALIDATE: bool,
            const ALGEBRAIC: bool,
        >(
            t: archmage::$token,
            cfg: HighpCfg,
            src: &[f32],
            dst: &mut [f32],
            n: f32,
        ) -> SliceResult {
            eval_slice_loop::<_, { graph::NATIVE }, ELIDE, VALIDATE, ALGEBRAIC, $width, _>(
                t,
                cfg,
                src,
                dst,
                n,
                highp_simd::$packed::<_, ELIDE, ZERO, $width>,
            )
        }
    };
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
packed_backend!(X64V3Token, packed4, v3, v3_alg, v3_slice, v3_slice_t, 8);
#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
packed_backend!(X64V4Token, wide, v4, v4_alg, v4_slice, v4_slice_t, 8);
#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
packed_backend!(X64V4xToken, wide, v4x, v4x_alg, v4x_slice, v4x_slice_t, 8);
#[cfg(target_arch = "aarch64")]
packed_backend!(
    NeonToken,
    packed2,
    neon,
    neon_alg,
    neon_slice,
    neon_slice_t,
    8
);
#[cfg(target_arch = "wasm32")]
packed_backend!(
    Wasm128Token,
    packed2,
    wasm,
    wasm_alg,
    wasm_slice,
    wasm_slice_t,
    4
);

#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
fn eval_alg_slice_loop<T: magetypes::simd::backends::F64x2Backend, const N: usize>(
    t: T,
    cfg: HighpCfg,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
) -> Result<(), (usize, PowfError)> {
    let (chunks, tail) = src.as_chunks::<N>();
    let (out_chunks, out_tail) = dst.as_chunks_mut::<N>();
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        let (bases, nan_mask) = highp::sanitize_bases_cfg::<N>(cfg, *s, n);
        highp::check_bases_cfg(cfg, &bases, n).map_err(|e| (i * N, e))?;
        let mut out = highp::eval_alg_token::<T, N>(t, cfg.alg_kind, bases);
        highp::check_results_cfg(cfg, &bases, &out, n).map_err(|e| (i * N, e))?;
        highp::restore_nans(&mut out, nan_mask);
        *d = out;
    }
    let tail_base = chunks.len() * N;
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

// Keep the packed result assignment in this loop. Factoring each chunk
// through the scalar validation helpers produced extra stores on x86.
#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
fn eval_slice_loop<
    T: magetypes::simd::backends::F64x2Backend,
    const MODE: usize,
    const ELIDE: bool,
    const VALIDATE: bool,
    const ALGEBRAIC: bool,
    const N: usize,
    F: FnMut(T, [f32; N], f32) -> [f32; N],
>(
    t: T,
    cfg: HighpCfg,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
    mut eval_chunk: F,
) -> Result<(), (usize, PowfError)> {
    if ALGEBRAIC {
        return eval_alg_slice_loop::<T, N>(t, cfg, src, dst, n);
    }
    let cfg = if VALIDATE {
        cfg
    } else {
        const { HighpCfg::from_options(PowfOptions::new().full_domain().highp()) }
    };
    let (chunks, tail) = src.as_chunks::<N>();
    let (out_chunks, out_tail) = dst.as_chunks_mut::<N>();
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        let (bases, nan_mask) = highp::sanitize_bases_cfg::<N>(cfg, *s, n);
        highp::check_bases_cfg(cfg, &bases, n).map_err(|e| (i * N, e))?;
        let mut out = eval_chunk(t, bases, n);
        highp::check_results_cfg(cfg, &bases, &out, n).map_err(|e| (i * N, e))?;
        highp::restore_nans(&mut out, nan_mask);
        *d = out;
    }
    let tail_base = chunks.len() * N;
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

#[inline(always)]
fn eval_lane<
    T: magetypes::simd::backends::F64x2Backend,
    const MODE: usize,
    const ELIDE: bool,
    const VALIDATE: bool,
    const ALGEBRAIC: bool,
>(
    t: T,
    cfg: HighpCfg,
    x: f32,
    n: f32,
) -> Result<f32, PowfError> {
    let (bases, nan_mask) = prepare::<VALIDATE, 1>(cfg, [x], n)?;
    let out = if ALGEBRAIC {
        highp::eval_alg_token(t, cfg.alg_kind, bases)
    } else {
        [highp_scalar::pow::<T, MODE, ELIDE>(t, bases[0], n)]
    };
    Ok(finish::<VALIDATE, 1>(cfg, &bases, out, n, nan_mask)?[0])
}

#[inline(never)]
fn scalar_slice<const ELIDE: bool, const VALIDATE: bool, const ALGEBRAIC: bool>(
    cfg: HighpCfg,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
) -> SliceResult {
    for (i, (&x, d)) in src.iter().zip(dst.iter_mut()).enumerate() {
        *d =
            eval_lane::<_, { graph::SOFTWARE }, ELIDE, VALIDATE, ALGEBRAIC>(ScalarToken, cfg, x, n)
                .map_err(|e| (i, e))?;
    }
    Ok(())
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
fn run_slice<const ELIDE: bool, const ZERO: bool, const VALIDATE: bool, const ALGEBRAIC: bool>(
    plan: Plan,
    in_context: bool,
    cfg: HighpCfg,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
) -> SliceResult {
    let _ = in_context;
    match plan {
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        Plan::V3(t) => {
            v3_slice::<ELIDE, ZERO, VALIDATE, ALGEBRAIC>(t, in_context, cfg, src, dst, n)
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
        Plan::V4(t) => {
            v4_slice::<ELIDE, ZERO, VALIDATE, ALGEBRAIC>(t, in_context, cfg, src, dst, n)
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
        Plan::V4x(t) => {
            v4x_slice::<ELIDE, ZERO, VALIDATE, ALGEBRAIC>(t, in_context, cfg, src, dst, n)
        }
        #[cfg(target_arch = "aarch64")]
        Plan::Neon(t) => {
            neon_slice::<ELIDE, ZERO, VALIDATE, ALGEBRAIC>(t, in_context, cfg, src, dst, n)
        }
        #[cfg(target_arch = "wasm32")]
        Plan::Wasm(t, true) => {
            wasm_slice::<ELIDE, ZERO, VALIDATE, ALGEBRAIC>(t, in_context, cfg, src, dst, n)
        }
        _ => scalar_slice::<ELIDE, VALIDATE, ALGEBRAIC>(cfg, src, dst, n),
    }
}

// A caller already inside the selected tier can inline the numerical graph.
// An upgrade uses one proof boundary around the whole operation.
#[inline(always)]
fn select(t: impl archmage::IntoConcreteToken, caller_only: bool, wide: bool) -> (Plan, bool) {
    let caller = super::dispatch::select_highp(t, Execution::Caller, false);
    let plan = if caller_only {
        caller
    } else {
        super::dispatch::select_highp(t, Execution::Upgrade, wide)
    };
    (
        plan,
        !matches!(caller, Plan::Software) && plan.same_tier(caller),
    )
}

#[cold]
#[inline(never)]
fn refine_highp_chunk(
    plan: Plan,
    in_context: bool,
    cfg: HighpCfg,
    src: &[f32],
    base: usize,
    error: PowfError,
    n: f32,
) -> SliceError<PowfErrorKind> {
    refine_slice_error(src, base, plan.midp_slice_width().min(8), error, |x| {
        eval_array(plan, in_context, cfg, [x], n).map(|value| value[0])
    })
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
fn eval_array<const N: usize>(
    plan: Plan,
    in_context: bool,
    cfg: HighpCfg,
    bases: [f32; N],
    n: f32,
) -> Result<[f32; N], PowfError> {
    if !cfg.elide || cfg.subnormals {
        return run::<false, false, N>(plan, in_context, cfg, bases, n);
    }
    if cfg.zero {
        run::<true, true, N>(plan, in_context, cfg, bases, n)
    } else {
        run::<true, false, N>(plan, in_context, cfg, bases, n)
    }
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
fn eval_slice(
    plan: Plan,
    in_context: bool,
    cfg: HighpCfg,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
) -> SliceResult {
    if cfg.full || cfg.assume {
        eval_slice_with_validation::<false>(plan, in_context, cfg, src, dst, n)
    } else {
        eval_slice_with_validation::<true>(plan, in_context, cfg, src, dst, n)
    }
}

#[inline(always)]
fn eval_slice_with_validation<const VALIDATE: bool>(
    plan: Plan,
    in_context: bool,
    cfg: HighpCfg,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
) -> SliceResult {
    if cfg.alg_kind != 0 {
        return run_slice::<false, false, VALIDATE, true>(plan, in_context, cfg, src, dst, n);
    }
    if !cfg.elide || cfg.subnormals {
        return run_slice::<false, false, VALIDATE, false>(plan, in_context, cfg, src, dst, n);
    }
    if cfg.zero {
        run_slice::<true, true, VALIDATE, false>(plan, in_context, cfg, src, dst, n)
    } else {
        run_slice::<true, false, VALIDATE, false>(plan, in_context, cfg, src, dst, n)
    }
}

#[cfg(feature = "std")]
#[inline(always)]
fn std_slice(
    cfg: HighpCfg,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
) -> Result<(), SliceError<PowfErrorKind>> {
    let (chunks, tail) = src.as_chunks::<8>();
    let (out_chunks, out_tail) = dst.as_chunks_mut::<8>();
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        *d = highp::run_cfg::<8>(cfg, *s, n)
            .map_err(|error| refine_std_chunk(cfg, src, i * 8, error, n))?;
    }
    let tail_base = chunks.len() * 8;
    for (i, (&x, d)) in tail.iter().zip(out_tail.iter_mut()).enumerate() {
        *d = highp::run_cfg::<1>(cfg, [x], n).map_err(|error| SliceError {
            kind: error.kind,
            index: tail_base + i,
        })?[0];
    }
    Ok(())
}

#[cfg(feature = "std")]
#[cold]
#[inline(never)]
fn refine_std_chunk(
    cfg: HighpCfg,
    src: &[f32],
    base: usize,
    error: PowfError,
    n: f32,
) -> SliceError<PowfErrorKind> {
    refine_slice_error(src, base, 8, error, |x| {
        highp::run_cfg(cfg, [x], n).map(|value| value[0])
    })
}

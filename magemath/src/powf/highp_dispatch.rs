//! One whole-operation feature selection for the canonical HIGHP graph.
use super::PowfOptions;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
use super::highp_simd;
use super::{PowfError, dispatch::Plan, graph, highp, highp_scalar};
use archmage::ScalarToken;

#[inline(always)]
fn run<const IN_CONTEXT: bool, const OPTIONS: u32, const N: usize>(
    plan: Plan,
    bases: [f32; N],
    n: f32,
) -> Result<[f32; N], PowfError> {
    #[cfg(feature = "std")]
    if OPTIONS & PowfOptions::BACKEND != 0 {
        return highp::run::<OPTIONS, N>(bases, n);
    }
    highp::check_bases::<OPTIONS, N>(&bases, n)?;
    let out = match plan {
        #[cfg(target_arch = "x86_64")]
        Plan::V3(t) => {
            if IN_CONTEXT {
                highp_simd::packed4::<archmage::X64V3Token, OPTIONS, N>(t, bases, n)
            } else {
                v3::<OPTIONS, N>(t, bases, n)
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Plan::V4(t) => {
            if IN_CONTEXT {
                highp_simd::wide::<archmage::X64V4Token, OPTIONS, N>(t, bases, n)
            } else {
                v4::<OPTIONS, N>(t, bases, n)
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Plan::V4x(t) => {
            if IN_CONTEXT {
                highp_simd::wide::<archmage::X64V4xToken, OPTIONS, N>(t, bases, n)
            } else {
                v4x::<OPTIONS, N>(t, bases, n)
            }
        }
        #[cfg(target_arch = "aarch64")]
        Plan::Neon(t) => {
            if IN_CONTEXT {
                highp_simd::packed2::<archmage::NeonToken, OPTIONS, N>(t, bases, n)
            } else {
                neon::<OPTIONS, N>(t, bases, n)
            }
        }
        #[cfg(target_arch = "wasm32")]
        Plan::Wasm(t, true) => {
            if IN_CONTEXT {
                highp_simd::packed2::<archmage::Wasm128Token, OPTIONS, N>(t, bases, n)
            } else {
                wasm::<OPTIONS, N>(t, bases, n)
            }
        }
        _ => scalar::<ScalarToken, { graph::SOFTWARE }, OPTIONS, N>(ScalarToken, bases, n),
    };
    highp::check_results::<OPTIONS, N>(&bases, &out, n)?;
    Ok(out)
}
#[inline(always)]
fn scalar<
    T: magetypes::simd::backends::F64x2Backend,
    const MODE: usize,
    const OPTIONS: u32,
    const N: usize,
>(
    t: T,
    x: [f32; N],
    n: f32,
) -> [f32; N] {
    let mut out = [0.; N];
    let mut i = 0;
    while i < N {
        out[i] = if OPTIONS & 3 == 2 {
            highp_scalar::pow::<T, MODE, true>(t, x[i], n)
        } else {
            highp_scalar::pow::<T, MODE, false>(t, x[i], n)
        };
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
        fn $name<const OPTIONS: u32, const N: usize>(
            t: archmage::$token,
            x: [f32; N],
            n: f32,
        ) -> [f32; N] {
            highp_simd::$packed::<_, OPTIONS, N>(t, x, n)
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
    const OPTIONS: u32,
    F: FnMut(T, [f32; 8], f32) -> [f32; 8],
>(
    t: T,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
    mut eval8: F,
) -> Result<(), (usize, PowfError)> {
    let (chunks, tail) = src.as_chunks::<8>();
    let (out_chunks, out_tail) = dst.as_chunks_mut::<8>();
    for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
        highp::check_bases::<OPTIONS, 8>(s, n).map_err(|e| (i * 8, e))?;
        let out = eval8(t, *s, n);
        highp::check_results::<OPTIONS, 8>(s, &out, n).map_err(|e| (i * 8, e))?;
        *d = out;
    }
    let tail_base = chunks.len() * 8;
    for (i, (&x, d)) in tail.iter().zip(out_tail.iter_mut()).enumerate() {
        highp::check_bases::<OPTIONS, 1>(&[x], n)
            .map_err(|e| (tail_base + i, PowfError { lane_mask: 1, ..e }))?;
        let out = if OPTIONS & 3 == 2 {
            highp_scalar::pow::<T, MODE, true>(t, x, n)
        } else {
            highp_scalar::pow::<T, MODE, false>(t, x, n)
        };
        highp::check_results::<OPTIONS, 1>(&[x], &[out], n)
            .map_err(|e| (tail_base + i, PowfError { lane_mask: 1, ..e }))?;
        *d = out;
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
        fn $name<const OPTIONS: u32>(
            t: archmage::$token,
            src: &[f32],
            dst: &mut [f32],
            n: f32,
        ) -> Result<(), (usize, PowfError)> {
            eval_slice_loop::<_, { graph::NATIVE }, OPTIONS, _>(
                t,
                src,
                dst,
                n,
                highp_simd::$packed::<_, OPTIONS, 8>,
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
fn run_slice<const IN_CONTEXT: bool, const OPTIONS: u32>(
    plan: Plan,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
) -> Result<(), (usize, PowfError)> {
    match plan {
        #[cfg(target_arch = "x86_64")]
        Plan::V3(t) => {
            if IN_CONTEXT {
                eval_slice_loop::<_, { graph::NATIVE }, OPTIONS, _>(
                    t,
                    src,
                    dst,
                    n,
                    highp_simd::packed4::<archmage::X64V3Token, OPTIONS, 8>,
                )
            } else {
                v3_slice::<OPTIONS>(t, src, dst, n)
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Plan::V4(t) => {
            if IN_CONTEXT {
                eval_slice_loop::<_, { graph::NATIVE }, OPTIONS, _>(
                    t,
                    src,
                    dst,
                    n,
                    highp_simd::wide::<archmage::X64V4Token, OPTIONS, 8>,
                )
            } else {
                v4_slice::<OPTIONS>(t, src, dst, n)
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Plan::V4x(t) => {
            if IN_CONTEXT {
                eval_slice_loop::<_, { graph::NATIVE }, OPTIONS, _>(
                    t,
                    src,
                    dst,
                    n,
                    highp_simd::wide::<archmage::X64V4xToken, OPTIONS, 8>,
                )
            } else {
                v4x_slice::<OPTIONS>(t, src, dst, n)
            }
        }
        #[cfg(target_arch = "aarch64")]
        Plan::Neon(t) => {
            if IN_CONTEXT {
                eval_slice_loop::<_, { graph::NATIVE }, OPTIONS, _>(
                    t,
                    src,
                    dst,
                    n,
                    highp_simd::packed2::<archmage::NeonToken, OPTIONS, 8>,
                )
            } else {
                neon_slice::<OPTIONS>(t, src, dst, n)
            }
        }
        #[cfg(target_arch = "wasm32")]
        Plan::Wasm(t, true) => {
            if IN_CONTEXT {
                eval_slice_loop::<_, { graph::NATIVE }, OPTIONS, _>(
                    t,
                    src,
                    dst,
                    n,
                    highp_simd::packed2::<archmage::Wasm128Token, OPTIONS, 8>,
                )
            } else {
                wasm_slice::<OPTIONS>(t, src, dst, n)
            }
        }
        _ => eval_slice_loop::<_, { graph::SOFTWARE }, OPTIONS, _>(
            ScalarToken,
            src,
            dst,
            n,
            scalar::<ScalarToken, { graph::SOFTWARE }, OPTIONS, 8>,
        ),
    }
}

// The direct form inlines into an already capable vector caller. Automatic
// upgrades enter one arcane boundary for the complete graph instead.
#[inline(always)]
pub(super) fn for_token<T: archmage::IntoConcreteToken, const OPTIONS: u32, const N: usize>(
    t: T,
    bases: [f32; N],
    n: f32,
) -> Result<[f32; N], PowfError> {
    #[cfg(feature = "std")]
    if OPTIONS & PowfOptions::BACKEND != 0 {
        return highp::run::<OPTIONS, N>(bases, n);
    }
    let caller = super::dispatch::select_highp(t, super::PowfExecution::Caller);
    if !matches!(caller, Plan::Software) {
        return run::<true, OPTIONS, N>(caller, bases, n);
    }
    let plan = if OPTIONS & PowfOptions::CALLER == 0 {
        super::dispatch::select_highp(t, super::PowfExecution::Automatic)
    } else {
        caller
    };
    run::<false, OPTIONS, N>(plan, bases, n)
}

/// Whole-slice HIGHP dispatcher: enters the selected target-feature region
/// once around all 8-element chunks and the scalar remainder.
#[inline(always)]
pub(super) fn for_token_slice<T: archmage::IntoConcreteToken, const OPTIONS: u32>(
    t: T,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
) -> Result<(), (usize, PowfError)> {
    #[cfg(feature = "std")]
    if OPTIONS & PowfOptions::BACKEND != 0 {
        let (chunks, tail) = src.as_chunks::<8>();
        let (out_chunks, out_tail) = dst.as_chunks_mut::<8>();
        for (i, (s, d)) in chunks.iter().zip(out_chunks.iter_mut()).enumerate() {
            *d = highp::run::<OPTIONS, 8>(*s, n).map_err(|e| (i * 8, e))?;
        }
        let tail_base = chunks.len() * 8;
        for (i, (&x, d)) in tail.iter().zip(out_tail.iter_mut()).enumerate() {
            *d = highp::run::<OPTIONS, 1>([x], n)
                .map_err(|e| (tail_base + i, PowfError { lane_mask: 1, ..e }))?[0];
        }
        return Ok(());
    }
    let caller = super::dispatch::select_highp(t, super::PowfExecution::Caller);
    if !matches!(caller, Plan::Software) {
        return run_slice::<true, OPTIONS>(caller, src, dst, n);
    }
    let plan = if OPTIONS & PowfOptions::CALLER == 0 {
        super::dispatch::select_highp(t, super::PowfExecution::Automatic)
    } else {
        caller
    };
    run_slice::<false, OPTIONS>(plan, src, dst, n)
}

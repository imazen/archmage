//! ARM (NEON) throughput comparison of three reciprocal / reciprocal-sqrt
//! implementation families, run on real aarch64 hardware (a Hetzner Ampere /
//! Neoverse box), not QEMU:
//!
//!   1. **original**   — the pre-change methods: raw `vrsqrteq`/`vrecpeq` 8-bit
//!                       estimate, then (for full precision) two *manual* Newton
//!                       steps built from `mul`/`sub` + 0.5/3.0/2.0 splats.
//!   2. **frsqrts**    — the current shipped fast methods: the native NEON
//!                       Newton-Raphson assist instructions FRSQRTS/FRECPS, with
//!                       `_approx` taking one step (~16-bit) and full taking two.
//!   3. **portable**   — the deterministic `_portable` family: an integer
//!                       bit-trick seed + non-FMA Newton step(s), full precision
//!                       via IEEE div/sqrt.
//!
//! Each kernel streams an L1-resident f32 array (load → op → store). Load/store
//! is identical across kernels, so the deltas reflect the op cost.
//!
//! Run on the ARM box:
//!   cargo bench -p magetypes --bench rsqrt_arm --features std

use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

#[cfg(target_arch = "aarch64")]
fn bench_rsqrt(c: &mut Criterion) {
    use archmage::{NeonToken, SimdToken};
    use core::arch::aarch64::{vrecpeq_f32, vrsqrteq_f32};
    use magetypes::simd::generic::f32x4;

    const N: usize = 2048; // 8 KB → L1-resident, compute-bound
    const REPS: usize = 16; // passes per measured iteration (lift well above timer resolution)

    let Some(token) = NeonToken::summon() else {
        return;
    };
    // Positive, well-conditioned inputs across a couple of octaves.
    let input: Vec<f32> = (0..N).map(|i| 0.1 + (i % 997) as f32 * 0.1).collect();
    let mut out = vec![0.0f32; N];

    // Generic streaming loop; `op` is monomorphized per kernel, load/store shared.
    #[inline(always)]
    fn process<F>(token: NeonToken, input: &[f32], out: &mut [f32], op: F)
    where
        F: Fn(f32x4<NeonToken>) -> f32x4<NeonToken>,
    {
        for (ci, co) in input.chunks_exact(4).zip(out.chunks_exact_mut(4)) {
            let v = f32x4::<NeonToken>::from_array(token, ci.try_into().unwrap());
            op(v).store(co.try_into().unwrap());
        }
    }

    // --- "original" kernels (raw hardware seed; full = + 2 manual Newton steps) ---
    let rsqrt_orig = |v: f32x4<NeonToken>| {
        let seed = f32x4::from_repr(token, unsafe { vrsqrteq_f32(v.into_repr()) });
        let half = f32x4::splat(token, 0.5);
        let three = f32x4::splat(token, 3.0);
        let step = |y: f32x4<NeonToken>| half * y * (three - v * y * y);
        step(step(seed))
    };
    let recip_orig = |v: f32x4<NeonToken>| {
        let seed = f32x4::from_repr(token, unsafe { vrecpeq_f32(v.into_repr()) });
        let two = f32x4::splat(token, 2.0);
        let step = |r: f32x4<NeonToken>| r * (two - v * r);
        step(step(seed))
    };
    let rsqrt_orig_approx =
        |v: f32x4<NeonToken>| f32x4::from_repr(token, unsafe { vrsqrteq_f32(v.into_repr()) });
    let recip_orig_approx =
        |v: f32x4<NeonToken>| f32x4::from_repr(token, unsafe { vrecpeq_f32(v.into_repr()) });

    macro_rules! bench {
        ($name:expr, $op:expr) => {
            c.bench_function($name, |b| {
                b.iter(|| {
                    for _ in 0..REPS {
                        process(token, black_box(&input), &mut out, $op);
                    }
                    black_box(&out);
                })
            });
        };
    }

    // rsqrt — estimate tier
    bench!("rsqrt_approx/1_original_raw_8bit", rsqrt_orig_approx);
    bench!("rsqrt_approx/2_frsqrts_16bit", |v: f32x4<NeonToken>| v
        .rsqrt_approx());
    bench!("rsqrt_approx/3_portable_8bit", |v: f32x4<NeonToken>| v
        .rsqrt_approx_portable());
    // rsqrt — full precision
    bench!("rsqrt_full/1_original_2manual", rsqrt_orig);
    bench!("rsqrt_full/2_frsqrts_2step", |v: f32x4<NeonToken>| v
        .rsqrt());
    bench!("rsqrt_full/3_portable_divsqrt", |v: f32x4<NeonToken>| v
        .rsqrt_portable());

    // rcp — estimate tier
    bench!("rcp_approx/1_original_raw_8bit", recip_orig_approx);
    bench!("rcp_approx/2_frsqrts_16bit", |v: f32x4<NeonToken>| v
        .rcp_approx());
    bench!("rcp_approx/3_portable_8bit", |v: f32x4<NeonToken>| v
        .rcp_approx_portable());
    // rcp — full precision
    bench!("rcp_full/1_original_2manual", recip_orig);
    bench!("rcp_full/2_frsqrts_2step", |v: f32x4<NeonToken>| v.recip());
    bench!("rcp_full/3_portable_div", |v: f32x4<NeonToken>| v
        .recip_portable());
}

/// f64 `recip`/`rsqrt`: the shipped exact form vs the estimate-and-refine form
/// it replaced.
///
/// The f64 NEON backend used `vrecpeq_f64`/`vrsqrteq_f64` + **three** fused
/// Newton steps until 2026-08-01. That missed the documented "full precision"
/// contract by 1 ULP, so it was replaced with exact FDIV / FDIV+FSQRT. The
/// correctness case is settled; this measures what the exactness costs, because
/// the f32 answer is famously core-dependent and does NOT transfer:
/// exact is faster on Apple M-series and slower on Neoverse-N1 (see
/// `benchmarks/rsqrt_arm_neoverse-n1_2026-06-21.md`). Run it on both.
#[cfg(target_arch = "aarch64")]
fn bench_rsqrt_f64(c: &mut Criterion) {
    use archmage::{NeonToken, SimdToken};
    use core::arch::aarch64::{vmulq_f64, vrecpeq_f64, vrecpsq_f64, vrsqrteq_f64, vrsqrtsq_f64};
    use magetypes::simd::generic::f64x2;

    const N: usize = 1024; // 8 KB → L1-resident, matches the f32 group's footprint
    const REPS: usize = 16;

    let Some(token) = NeonToken::summon() else {
        return;
    };
    let input: Vec<f64> = (0..N).map(|i| 0.1 + (i % 997) as f64 * 0.1).collect();
    let mut out = vec![0.0f64; N];

    #[inline(always)]
    fn process<F>(token: NeonToken, input: &[f64], out: &mut [f64], op: F)
    where
        F: Fn(f64x2<NeonToken>) -> f64x2<NeonToken>,
    {
        for (ci, co) in input.chunks_exact(2).zip(out.chunks_exact_mut(2)) {
            let v = f64x2::<NeonToken>::from_array(token, ci.try_into().unwrap());
            op(v).store(co.try_into().unwrap());
        }
    }

    // The replaced bodies, reproduced exactly: raw estimate + 3 fused steps.
    let recip_3step = |v: f64x2<NeonToken>| {
        let a = v.into_repr();
        f64x2::from_repr(token, unsafe {
            let y = vrecpeq_f64(a);
            let y = vmulq_f64(vrecpsq_f64(a, y), y);
            let y = vmulq_f64(vrecpsq_f64(a, y), y);
            vmulq_f64(vrecpsq_f64(a, y), y)
        })
    };
    let rsqrt_3step = |v: f64x2<NeonToken>| {
        let a = v.into_repr();
        f64x2::from_repr(token, unsafe {
            let y = vrsqrteq_f64(a);
            let y = vmulq_f64(vrsqrtsq_f64(vmulq_f64(a, y), y), y);
            let y = vmulq_f64(vrsqrtsq_f64(vmulq_f64(a, y), y), y);
            vmulq_f64(vrsqrtsq_f64(vmulq_f64(a, y), y), y)
        })
    };

    macro_rules! bench64 {
        ($name:expr, $op:expr) => {
            c.bench_function($name, |b| {
                b.iter(|| {
                    for _ in 0..REPS {
                        process(token, black_box(&input), &mut out, $op);
                    }
                    black_box(&out);
                })
            });
        };
    }

    // 1_* is the form that was replaced, 2_* is what ships now.
    bench64!("f64_rcp_full/1_vrecpe_3step_1ulp", recip_3step);
    bench64!("f64_rcp_full/2_exact_fdiv", |v: f64x2<NeonToken>| v.recip());
    bench64!("f64_rsqrt_full/1_vrsqrte_3step_1ulp", rsqrt_3step);
    bench64!("f64_rsqrt_full/2_exact_fdiv_fsqrt", |v: f64x2<
        NeonToken,
    >| v.rsqrt());
}

#[cfg(not(target_arch = "aarch64"))]
fn bench_rsqrt(_c: &mut Criterion) {
    // aarch64-only benchmark; nothing to run on other targets.
}

#[cfg(not(target_arch = "aarch64"))]
fn bench_rsqrt_f64(_c: &mut Criterion) {
    // aarch64-only benchmark; nothing to run on other targets.
}

criterion_group!(benches, bench_rsqrt, bench_rsqrt_f64);
criterion_main!(benches);

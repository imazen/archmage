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

#[cfg(not(target_arch = "aarch64"))]
fn bench_rsqrt(_c: &mut Criterion) {
    // aarch64-only benchmark; nothing to run on other targets.
}

criterion_group!(benches, bench_rsqrt);
criterion_main!(benches);

//! x86 throughput comparison of the reciprocal / reciprocal-sqrt tier grid on
//! `f32x8` (AVX2, the mainstream x86 tier). This bench produced the Zen 5
//! numbers behind the tier split (see
//! `benchmarks/recip_x86_zen5-9950x3d_2026-09-03.md`).
//!
//!   1. **newton**   — hand-rolled `rcpps`/`rsqrtps` + one Newton step from
//!      `mul`/`sub` splats: the reference the bare names must match.
//!   2. **bare**     — `recip()`/`rsqrt()`, the working tier (<= 4 ULP,
//!      estimate + Newton on x86). Expected == kernel 1 to the digit.
//!   3. **approx**   — the raw estimate tier (`rcp_approx`/`rsqrt_approx`).
//!   4. **portable** — `recip_portable()`/`rsqrt_portable()`, the precise
//!      tier: exact IEEE division, 0 ULP, IEEE rails (issue #64).
//!
//! Each kernel streams an L1-resident f32 array (load → op → store) inside one
//! `#[arcane]` region so the generic ops inline into AVX2 codegen the way real
//! callers get them. Load/store is identical across kernels.
//!
//! Run on real x86_64 (no `-Ctarget-cpu=native` — bench what users get):
//!   cargo bench -p magetypes --bench recip_x86 --features std

use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

#[cfg(target_arch = "x86_64")]
mod kernels {
    use archmage::{X64V3Token, arcane};
    use magetypes::simd::generic::f32x8;

    macro_rules! streaming_kernel {
        ($name:ident, $tok:ident, $v:ident, $body:expr) => {
            #[arcane]
            pub fn $name($tok: X64V3Token, input: &[f32], out: &mut [f32]) {
                for (ci, co) in input.chunks_exact(8).zip(out.chunks_exact_mut(8)) {
                    let $v = f32x8::<X64V3Token>::from_array($tok, ci.try_into().unwrap());
                    let r: f32x8<X64V3Token> = $body;
                    r.store(co.try_into().unwrap());
                }
            }
        };
    }

    streaming_kernel!(recip_bare, token, v, v.recip());
    streaming_kernel!(recip_newton, token, v, {
        let two = f32x8::splat(token, 2.0);
        let r = v.rcp_approx();
        r * (two - v * r)
    });
    streaming_kernel!(recip_approx, token, v, v.rcp_approx());
    streaming_kernel!(recip_portable_k, token, v, v.recip_portable());
    streaming_kernel!(rsqrt_bare, token, v, v.rsqrt());
    streaming_kernel!(rsqrt_newton, token, v, {
        let half = f32x8::splat(token, 0.5);
        let three = f32x8::splat(token, 3.0);
        let y = v.rsqrt_approx();
        half * y * (three - v * y * y)
    });
    streaming_kernel!(rsqrt_approx, token, v, v.rsqrt_approx());
    streaming_kernel!(rsqrt_portable_k, token, v, v.rsqrt_portable());
}

#[cfg(target_arch = "x86_64")]
fn bench_recip_x86(c: &mut Criterion) {
    use archmage::{SimdToken, X64V3Token};

    const N: usize = 2048; // 8 KB → L1-resident, compute-bound
    const REPS: usize = 16; // passes per measured iteration

    let Some(token) = X64V3Token::summon() else {
        return;
    };
    // Positive, well-conditioned inputs across a couple of octaves.
    let input: Vec<f32> = (0..N).map(|i| 0.1 + (i % 997) as f32 * 0.1).collect();
    let mut out = vec![0.0f32; N];

    macro_rules! bench {
        ($name:expr, $kernel:path) => {
            c.bench_function($name, |b| {
                b.iter(|| {
                    for _ in 0..REPS {
                        $kernel(token, black_box(&input), &mut out);
                    }
                    black_box(&out);
                })
            });
        };
    }

    // 1_* is the form that was replaced, 2_* is what ships now, 3_* the estimate.
    bench!("f32x8_rcp_full/1_rcpps_newton1", kernels::recip_newton);
    bench!("f32x8_rcp_full/2_recip_bare", kernels::recip_bare);
    bench!("f32x8_rcp/3_rcp_approx_raw", kernels::recip_approx);
    bench!("f32x8_rcp_full/4_recip_portable", kernels::recip_portable_k);
    bench!("f32x8_rsqrt_full/1_rsqrtps_newton1", kernels::rsqrt_newton);
    bench!("f32x8_rsqrt_full/2_rsqrt_bare", kernels::rsqrt_bare);
    bench!("f32x8_rsqrt/3_rsqrt_approx_raw", kernels::rsqrt_approx);
    bench!(
        "f32x8_rsqrt_full/4_rsqrt_portable",
        kernels::rsqrt_portable_k
    );
}

#[cfg(not(target_arch = "x86_64"))]
fn bench_recip_x86(_c: &mut Criterion) {
    // x86_64-only benchmark; nothing to run on other targets.
}

criterion_group!(benches, bench_recip_x86);
criterion_main!(benches);

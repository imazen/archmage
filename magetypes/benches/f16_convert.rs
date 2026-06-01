//! f16 ↔ f32 slice-conversion throughput on the *production* dispatch path.
//!
//! Times the whole `F16Convert::f16_to_f32_slice` / `f32_to_f16_slice` method —
//! including the per-call `Arm64V2Token::summon()` fp16 dispatch — so the
//! numbers reflect what a real caller pays, not an inlined kernel microbench.
//!
//! ## A/B design (run the SAME binary on two toolchains on ONE aarch64 box)
//!
//! The NEON-f16 hardware kernel is `#[rustversion::since(1.94)]`-gated, so the
//! toolchain version selects the path a `NeonToken` takes — no source change:
//!   * built with **rustc ≥ 1.94** → `NeonToken` takes the native `vcvt_f32_f16`
//!     hardware path.
//!   * built with **rustc < 1.94** (e.g. 1.93) → `NeonToken` takes the branchless
//!     NEON-f32x4 software kernel.
//! `ScalarToken` is the pure-scalar branchless baseline in both builds (and the
//! direct answer to "is the HW path even worth it vs scalar?").
//!
//! So: `cargo +1.96 bench` gives HW-vcvt + scalar; `cargo +1.93 bench` gives
//! NEON-software + scalar. Comparing the two `neon/*` columns across builds
//! isolates `vcvt` vs branchless-NEON on identical hardware; the `scalar/*`
//! column must match across builds (sanity).
//!
//! Sizes span the regimes: tiny (call-overhead-dominated) → large (memory-
//! bandwidth-bound, where a 1-instruction `vcvt` and a ~10-op branchless kernel
//! both saturate the load/store path and any compute win disappears).

use archmage::ScalarToken;
#[cfg(target_arch = "aarch64")]
use archmage::SimdToken;
use magetypes::simd::generic::F16Convert;
use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

const SIZES: &[usize] = &[16, 256, 4096, 65536, 1_048_576];

/// Deterministic spread of finite f16 bit patterns (normals + subnormals),
/// avoiding Inf/NaN so every lane takes the common arithmetic path.
fn f16_input(n: usize) -> Vec<u16> {
    (0..n)
        .map(|i| {
            // walk exponent+mantissa over the finite range, sign varied
            let v = ((i.wrapping_mul(2_654_435_761)) & 0x7BFF) as u16;
            let sign = ((i & 1) as u16) << 15;
            sign | v
        })
        .collect()
}

fn f32_input(n: usize) -> Vec<f32> {
    let f16 = f16_input(n);
    let mut out = vec![0.0f32; n];
    ScalarToken.f16_to_f32_slice(&f16, &mut out);
    out
}

fn bench_decode(c: &mut Criterion) {
    for &n in SIZES {
        let input = f16_input(n);
        let mut out = vec![0.0f32; n];

        c.bench_function(&format!("decode/scalar/{n}"), |b| {
            b.iter(|| ScalarToken.f16_to_f32_slice(black_box(&input), black_box(&mut out)))
        });

        #[cfg(target_arch = "aarch64")]
        if let Some(nt) = archmage::NeonToken::summon() {
            c.bench_function(&format!("decode/neon/{n}"), |b| {
                b.iter(|| nt.f16_to_f32_slice(black_box(&input), black_box(&mut out)))
            });
        }
    }
}

fn bench_encode(c: &mut Criterion) {
    for &n in SIZES {
        let input = f32_input(n);
        let mut out = vec![0u16; n];

        c.bench_function(&format!("encode/scalar/{n}"), |b| {
            b.iter(|| ScalarToken.f32_to_f16_slice(black_box(&input), black_box(&mut out)))
        });

        #[cfg(target_arch = "aarch64")]
        if let Some(nt) = archmage::NeonToken::summon() {
            c.bench_function(&format!("encode/neon/{n}"), |b| {
                b.iter(|| nt.f32_to_f16_slice(black_box(&input), black_box(&mut out)))
            });
        }
    }
}

criterion_group!(benches, bench_decode, bench_encode);
criterion_main!(benches);

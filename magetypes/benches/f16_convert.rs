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
//! ## x86-64 paths
//!
//! With `--features "std avx512"` this binary measures, same process:
//!   * `scalar/*` — branchless software baseline.
//!   * `f16c/*`   — `X64V3Token` slice (the production path; summons-up to the
//!                  best tier, so 16-wide AVX-512F on a V4 CPU, else 8-wide F16C).
//!   * `v4/*`     — `X64V4Token` slice (the plain V4 path; 16-wide directly, no
//!                  probe). `f16c` ≈ `v4` on a V4 CPU shows the summon is free.
//!
//! The 8-wide-vs-16-wide *width* isolation lives in the dedicated microbench in
//! `benchmarks/f16_convert_zen4-7950x_2026-06-01.md` (the production path here
//! auto-upgrades, so it can't pin 8-wide on a V4 box).
//!
//! Sizes span the regimes: tiny (call-overhead-dominated) → large (memory-
//! bandwidth-bound, where the wider convert saturates the load/store path and
//! the compute win shrinks toward parity).

use archmage::ScalarToken;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
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

        // x86-64: `f16c` = X64V3Token slice (production path, summons-up to best
        // tier); `v4` = X64V4Token slice (plain V4 path, 16-wide direct). On a V4
        // CPU both run 16-wide — `f16c ≈ v4` shows the summon is amortized away.
        #[cfg(target_arch = "x86_64")]
        if let Some(t) = archmage::X64V3Token::summon() {
            c.bench_function(&format!("decode/f16c/{n}"), |b| {
                b.iter(|| t.f16_to_f32_slice(black_box(&input), black_box(&mut out)))
            });
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if let Some(t) = archmage::X64V4Token::summon() {
            c.bench_function(&format!("decode/v4/{n}"), |b| {
                b.iter(|| t.f16_to_f32_slice(black_box(&input), black_box(&mut out)))
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

        // x86-64: `f16c` = V3 slice (production, summons-up); `v4` = V4 slice
        // (plain path, direct). See decode.
        #[cfg(target_arch = "x86_64")]
        if let Some(t) = archmage::X64V3Token::summon() {
            c.bench_function(&format!("encode/f16c/{n}"), |b| {
                b.iter(|| t.f32_to_f16_slice(black_box(&input), black_box(&mut out)))
            });
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if let Some(t) = archmage::X64V4Token::summon() {
            c.bench_function(&format!("encode/v4/{n}"), |b| {
                b.iter(|| t.f32_to_f16_slice(black_box(&input), black_box(&mut out)))
            });
        }
    }
}

criterion_group!(benches, bench_decode, bench_encode);
criterion_main!(benches);

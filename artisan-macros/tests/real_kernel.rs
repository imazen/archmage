//! Real SIMD kernel exercising `#[cpu_tier]` + `#[chain]` end-to-end.
//!
//! Sums an `&[f32]` using AVX2 + FMA on x86_64 and NEON on aarch64, falling
//! back to a scalar loop on any other arch (including during CI if the host
//! CPU lacks the required features).
//!
//! Unlike `tests/smoke.rs` (which uses scalar bodies in every tier), this
//! test actually calls arch-specific intrinsics. The tier bodies use `unsafe`
//! for memory loads (raw-pointer intrinsics); `#[cpu_tier]`'s target_feature
//! attribute makes the value-based intrinsics (_mm256_add_ps, vaddq_f32)
//! safe to call inside the tier body.
//!
//! `#![forbid(unsafe_code)]`-compat is NOT claimed for the tier bodies
//! themselves — users who want it pull in `safe_unaligned_simd` or similar.
//! The claim is about the macro expansions (trampoline, entry, test hooks),
//! which remain forbid-compatible regardless of what tier bodies do.

#![allow(dead_code)]

use artisan_macros::{chain, cpu_tier};

// ---------------- Scalar fallback ----------------

fn sum_scalar(data: &[f32]) -> f32 {
    data.iter().sum()
}

// ---------------- x86_64 / AVX2+FMA ----------------

#[cpu_tier(enable = "avx2,fma")]
fn sum_v3(data: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::*;
        // Accumulate in an 8-lane vector.
        let mut acc = _mm256_setzero_ps();
        let chunks = data.chunks_exact(8);
        let rem = chunks.remainder();
        for chunk in chunks {
            // SAFETY: chunks_exact(8) guarantees 8 f32 values are readable at
            // chunk.as_ptr(); alignment is unconstrained but _mm256_loadu_ps
            // tolerates unaligned loads.
            let v = unsafe { _mm256_loadu_ps(chunk.as_ptr()) };
            acc = _mm256_add_ps(acc, v);
        }
        // Horizontal sum of 8 lanes.
        let mut tmp = [0f32; 8];
        // SAFETY: tmp has room for 8 f32 values; _mm256_storeu_ps tolerates
        // unaligned writes.
        unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), acc) };
        tmp.iter().sum::<f32>() + rem.iter().sum::<f32>()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = data;
        unreachable!("sum_v3 is x86_64-only; #[cpu_tier] cfg-gates it out elsewhere")
    }
}

// ---------------- aarch64 / NEON ----------------

#[cpu_tier(enable = "neon")]
fn sum_neon(data: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;
        // Accumulate in a 4-lane vector.
        let mut acc = vdupq_n_f32(0.0);
        let chunks = data.chunks_exact(4);
        let rem = chunks.remainder();
        for chunk in chunks {
            // SAFETY: chunks_exact(4) guarantees 4 f32 values at chunk.as_ptr();
            // vld1q_f32 takes a *const f32 and accepts unaligned addresses.
            let v = unsafe { vld1q_f32(chunk.as_ptr()) };
            acc = vaddq_f32(acc, v);
        }
        let lane_sum = vaddvq_f32(acc);
        lane_sum + rem.iter().sum::<f32>()
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = data;
        unreachable!("sum_neon is aarch64-only; #[cpu_tier] cfg-gates it out elsewhere")
    }
}

// ---------------- Dispatch chain ----------------

#[chain(
    x86_64 = [
        sum_v3 = "avx2,fma",
    ],
    aarch64 = [
        sum_neon = "neon",
    ],
    default = sum_scalar,
)]
/// Sum an `&[f32]` using the best available SIMD tier for the host CPU.
pub fn sum(data: &[f32]) -> f32 {}

// ---------------- Tests ----------------

fn reference_sum(data: &[f32]) -> f32 {
    // Kahan-ish reference: the scalar fallback is already iter().sum(), which
    // matches Rust's standard summation. Using the same reference keeps the
    // assertion bit-exact.
    data.iter().sum()
}

#[test]
fn sum_matches_scalar_small() {
    let data: Vec<f32> = (0..4).map(|i| i as f32).collect();
    assert_eq!(sum(&data), reference_sum(&data));
}

#[test]
fn sum_matches_scalar_aligned_block() {
    let data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.5).collect();
    assert_eq!(sum(&data), reference_sum(&data));
}

#[test]
fn sum_matches_scalar_with_remainder() {
    // 37 elements: one AVX2 iter (8 * 4 = 32) plus 5 remainder for x86,
    // nine NEON iters (4 * 9 = 36) plus 1 remainder for aarch64.
    let data: Vec<f32> = (0..37).map(|i| (i as f32) - 10.0).collect();
    assert_eq!(sum(&data), reference_sum(&data));
}

#[test]
fn sum_matches_scalar_empty() {
    let data: &[f32] = &[];
    assert_eq!(sum(data), 0.0);
}

#[test]
fn sum_matches_scalar_large() {
    let n = 10_000;
    let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.001).sin()).collect();
    // Tolerance: lane-reordering in SIMD sums produces different rounding vs
    // sequential scalar sum. We accept a tiny relative error.
    let actual = sum(&data);
    let expected = reference_sum(&data);
    let rel = ((actual - expected).abs()) / expected.abs().max(1e-6);
    assert!(
        rel < 1e-5,
        "relative error {rel:.2e} exceeds tolerance (actual={actual}, expected={expected})"
    );
}

// ---------------- Test-hook exercise on real tiers ----------------

#[cfg(any(test, feature = "artisan_test_hooks"))]
#[test]
fn force_scalar_matches_reference() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32) * 0.25).collect();
    {
        let _scope = sum_force_max_tier(SumTier::Default);
        // Forced to default — scalar path runs.
        assert_eq!(sum(&data), reference_sum(&data));
    }
    // Normal dispatch resumes.
    let _ = sum(&data);
}

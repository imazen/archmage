//! Benchmark: target-feature boundary overhead vs wrapper overhead.
//!
//! Run with: cargo bench --bench asm_inspection
//!
//! ## Key finding: the overhead is from target-feature mismatch, NOT wrappers
//!
//! LLVM cannot inline a `#[target_feature(enable = "avx2")]` function into a
//! caller that lacks those features. This creates an optimization boundary per
//! call — LLVM can't hoist loads, sink stores, or vectorize across it.
//!
//! Proof:
//! - Patterns 1, 4, & 7 all cross the boundary per iteration → same speed (~2.2 µs)
//! - Pattern 4 has NO wrapper (calls `#[rite]` directly) — still slow
//! - Pattern 7 is bare `#[target_feature]` with no archmage at all — same speed as 1
//! - Patterns 5 & 6 use wrappers but WITHOUT target-feature mismatch → fast (~545 ns)
//!
//! Conclusion: `#[arcane]`'s overhead equals a bare `#[target_feature]` call.
//! The cost is from LLVM's inability to inline across mismatched target features,
//! not from wrapper functions or archmage abstractions.

#![cfg(target_arch = "x86_64")]

use archmage::{Desktop64, SimdToken, arcane, rite};
use std::arch::x86_64::*;

// ============================================================================
// Pattern 1: #[arcane] in loop — target-feature boundary each iteration
// ============================================================================

#[arcane]
fn process_chunk_arcane(_token: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    unsafe {
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let sum = _mm256_add_ps(va, vb);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), sum);
        out
    }
}

#[inline(never)]
pub fn loop_with_arcane_in_loop(token: Desktop64, data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    let mut total = 0.0f32;
    for (a, b) in data.iter().zip(other.iter()) {
        let result = process_chunk_arcane(token, a, b);
        total += result[0];
    }
    total
}

// ============================================================================
// Pattern 2: Loop inside #[arcane], #[rite] inlines (features match)
// ============================================================================

#[rite]
fn process_chunk_rite(_token: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    unsafe {
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let sum = _mm256_add_ps(va, vb);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), sum);
        out
    }
}

#[arcane]
fn loop_inner_rite(token: Desktop64, data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    let mut total = 0.0f32;
    for (a, b) in data.iter().zip(other.iter()) {
        let result = process_chunk_rite(token, a, b);
        total += result[0];
    }
    total
}

#[inline(never)]
pub fn loop_inside_arcane_with_rite(
    token: Desktop64,
    data: &[[f32; 8]],
    other: &[[f32; 8]],
) -> f32 {
    loop_inner_rite(token, data, other)
}

// ============================================================================
// Pattern 3: Manual inline for comparison (best possible)
// ============================================================================

#[arcane]
fn loop_manual_inline(_token: Desktop64, data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    let mut total = 0.0f32;
    for (a, b) in data.iter().zip(other.iter()) {
        unsafe {
            let va = _mm256_loadu_ps(a.as_ptr());
            let vb = _mm256_loadu_ps(b.as_ptr());
            let sum = _mm256_add_ps(va, vb);
            let mut out = [0.0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), sum);
            total += out[0];
        }
    }
    total
}

#[inline(never)]
pub fn loop_inside_arcane_manual(token: Desktop64, data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    loop_manual_inline(token, data, other)
}

// ============================================================================
// Pattern 4: #[rite] called directly from non-target_feature context (NO wrapper)
// Same speed as pattern 1 — proving the overhead is NOT from the wrapper
// ============================================================================

#[inline(never)]
pub fn loop_calling_rite_directly(token: Desktop64, data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    let mut total = 0.0f32;
    for (a, b) in data.iter().zip(other.iter()) {
        // SAFETY: We have the token, so CPU supports the features
        let result = unsafe { process_chunk_rite(token, a, b) };
        total += result[0];
    }
    total
}

// ============================================================================
// Pattern 5: Wrapper WITHOUT target-feature mismatch (scalar fallback)
// Proves that wrapper call overhead itself is negligible — LLVM inlines it.
// ============================================================================

#[inline]
fn process_chunk_scalar(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    let mut out = [0.0f32; 8];
    for i in 0..8 {
        out[i] = a[i] + b[i];
    }
    out
}

/// Wrapper function (no target_feature) calling scalar helper (no target_feature).
/// LLVM inlines this freely — no feature mismatch to block it.
#[inline(never)]
pub fn loop_scalar_wrapper(data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    let mut total = 0.0f32;
    for (a, b) in data.iter().zip(other.iter()) {
        let result = process_chunk_scalar(a, b);
        total += result[0];
    }
    total
}

// ============================================================================
// Pattern 6: Scalar code inlined directly (baseline for pattern 5 comparison)
// ============================================================================

#[inline(never)]
pub fn loop_scalar_inline(data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    let mut total = 0.0f32;
    for (a, b) in data.iter().zip(other.iter()) {
        let mut out = [0.0f32; 8];
        for i in 0..8 {
            out[i] = a[i] + b[i];
        }
        total += out[0];
    }
    total
}

// ============================================================================
// Pattern 7: Bare #[target_feature] — no archmage, no wrapper, no token
// Identical cost to pattern 1 (#[arcane]). Proves archmage adds zero overhead
// vs hand-written #[target_feature] + unsafe.
// ============================================================================

#[target_feature(enable = "avx2")]
unsafe fn process_chunk_bare_target_feature(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    unsafe {
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let sum = _mm256_add_ps(va, vb);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), sum);
        out
    }
}

#[inline(never)]
pub fn loop_bare_target_feature(data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    let mut total = 0.0f32;
    for (a, b) in data.iter().zip(other.iter()) {
        // SAFETY: benchmark only runs when Desktop64::summon() succeeds
        let result = unsafe { process_chunk_bare_target_feature(a, b) };
        total += result[0];
    }
    total
}

// ============================================================================
// Criterion benchmark
// ============================================================================

use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_patterns(c: &mut Criterion) {
    let data: Vec<[f32; 8]> = (0..1000).map(|i| [i as f32; 8]).collect();
    let other: Vec<[f32; 8]> = (0..1000).map(|i| [i as f32 * 2.0; 8]).collect();

    if let Some(token) = Desktop64::summon() {
        c.bench_function("1_arcane_in_loop", |b| {
            b.iter(|| loop_with_arcane_in_loop(token, black_box(&data), black_box(&other)))
        });

        c.bench_function("2_rite_in_arcane", |b| {
            b.iter(|| loop_inside_arcane_with_rite(token, black_box(&data), black_box(&other)))
        });

        c.bench_function("3_manual_inline", |b| {
            b.iter(|| loop_inside_arcane_manual(token, black_box(&data), black_box(&other)))
        });

        c.bench_function("4_rite_direct_unsafe", |b| {
            b.iter(|| loop_calling_rite_directly(token, black_box(&data), black_box(&other)))
        });

        c.bench_function("7_bare_target_feature", |b| {
            b.iter(|| loop_bare_target_feature(black_box(&data), black_box(&other)))
        });
    } else {
        eprintln!("Desktop64 not available, skipping benchmarks");
    }

    // These don't need a token — proving wrapper overhead is negligible
    c.bench_function("5_scalar_wrapper", |b| {
        b.iter(|| loop_scalar_wrapper(black_box(&data), black_box(&other)))
    });

    c.bench_function("6_scalar_inline", |b| {
        b.iter(|| loop_scalar_inline(black_box(&data), black_box(&other)))
    });
}

criterion_group!(benches, bench_patterns);
criterion_main!(benches);

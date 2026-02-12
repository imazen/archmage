//! Zero-overhead proof: safe_unaligned_simd vs raw unsafe vs magetypes.
//!
//! All three approaches should produce identical benchmark times (within noise)
//! because they all compile to the same `vmovups` instruction.
//!
//! Run: cargo bench --bench safe_memory_overhead
//! ASM: cargo asm --bench safe_memory_overhead --rust

#![cfg(target_arch = "x86_64")]

use archmage::{Desktop64, SimdToken, arcane};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use magetypes::simd::f32x8;
use std::arch::x86_64::*;

// =============================================================================
// Pattern 1: safe_unaligned_simd load in hot loop inside #[arcane]
// =============================================================================

#[arcane]
fn safe_load_loop_inner(_token: Desktop64, data: &[[f32; 8]]) -> f32 {
    let mut acc = _mm256_setzero_ps();
    for chunk in data {
        let v = safe_unaligned_simd::x86_64::_mm256_loadu_ps(chunk);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(v, v));
    }
    // Horizontal sum
    let hi = _mm256_extractf128_ps::<1>(acc);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let total = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(total)
}

#[inline(never)]
pub fn safe_load_loop(token: Desktop64, data: &[[f32; 8]]) -> f32 {
    safe_load_loop_inner(token, data)
}

// =============================================================================
// Pattern 2: raw unsafe pointer load in hot loop inside #[arcane]
// =============================================================================

#[arcane]
fn unsafe_load_loop_inner(_token: Desktop64, data: &[[f32; 8]]) -> f32 {
    let mut acc = _mm256_setzero_ps();
    for chunk in data {
        let v = unsafe { _mm256_loadu_ps(chunk.as_ptr()) };
        acc = _mm256_add_ps(acc, _mm256_mul_ps(v, v));
    }
    let hi = _mm256_extractf128_ps::<1>(acc);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let total = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(total)
}

#[inline(never)]
pub fn unsafe_load_loop(token: Desktop64, data: &[[f32; 8]]) -> f32 {
    unsafe_load_loop_inner(token, data)
}

// =============================================================================
// Pattern 3: magetypes f32x8 load in hot loop inside #[arcane]
// =============================================================================

#[arcane]
fn magetypes_load_loop_inner(token: Desktop64, data: &[[f32; 8]]) -> f32 {
    let mut acc = f32x8::splat(token, 0.0);
    for chunk in data {
        let v = f32x8::load(token, chunk);
        acc = acc + v * v;
    }
    acc.reduce_add()
}

#[inline(never)]
pub fn magetypes_load_loop(token: Desktop64, data: &[[f32; 8]]) -> f32 {
    magetypes_load_loop_inner(token, data)
}

// =============================================================================
// Standalone functions for cargo asm inspection
// =============================================================================

/// Inspect with: cargo asm safe_memory_overhead::safe_load_single
#[unsafe(no_mangle)]
#[arcane]
pub fn safe_load_single(_token: Desktop64, data: &[f32; 8]) -> __m256 {
    safe_unaligned_simd::x86_64::_mm256_loadu_ps(data)
}

/// Inspect with: cargo asm safe_memory_overhead::unsafe_load_single
#[unsafe(no_mangle)]
#[arcane]
pub fn unsafe_load_single(_token: Desktop64, data: &[f32; 8]) -> __m256 {
    unsafe { _mm256_loadu_ps(data.as_ptr()) }
}

// =============================================================================
// Criterion benchmark
// =============================================================================

fn bench_memory_overhead(c: &mut Criterion) {
    let data: Vec<[f32; 8]> = (0..1000).map(|i| [i as f32; 8]).collect();

    if let Some(token) = Desktop64::summon() {
        c.bench_function("1_safe_load_loop", |b| {
            b.iter(|| safe_load_loop(token, black_box(&data)))
        });

        c.bench_function("2_unsafe_load_loop", |b| {
            b.iter(|| unsafe_load_loop(token, black_box(&data)))
        });

        c.bench_function("3_magetypes_load_loop", |b| {
            b.iter(|| magetypes_load_loop(token, black_box(&data)))
        });

        // Verify all three produce the same result
        let safe_result = safe_load_loop(token, &data);
        let unsafe_result = unsafe_load_loop(token, &data);
        let magetypes_result = magetypes_load_loop(token, &data);
        assert!(
            (safe_result - unsafe_result).abs() < 1e-3,
            "safe ({safe_result}) != unsafe ({unsafe_result})"
        );
        assert!(
            (safe_result - magetypes_result).abs() < 1e-3,
            "safe ({safe_result}) != magetypes ({magetypes_result})"
        );
    } else {
        eprintln!("Desktop64 not available, skipping benchmarks");
    }
}

criterion_group!(benches, bench_memory_overhead);
criterion_main!(benches);

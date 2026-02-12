//! ASM pattern verification for slice→SIMD load patterns.
//!
//! Each function is `#[unsafe(no_mangle)]` + `#[arcane]` so `cargo asm` can find it.
//! Verified by `scripts/verify-asm.sh`.
//!
//! Run: cargo asm -p archmage --bench asm_patterns --features "std macros avx512"

#![cfg(target_arch = "x86_64")]

use archmage::{Desktop64, SimdToken, arcane};
use std::arch::x86_64::*;

// ============================================================================
// 256-bit float loads
// ============================================================================

/// Baseline: load from array reference → vmovups
#[unsafe(no_mangle)]
#[arcane]
fn load_array_ref(_t: Desktop64, data: &[f32; 8]) -> __m256 {
    safe_unaligned_simd::x86_64::_mm256_loadu_ps(data)
}

/// Slice via .first_chunk() → should produce same vmovups
#[unsafe(no_mangle)]
#[arcane]
fn load_first_chunk(_t: Desktop64, data: &[f32]) -> __m256 {
    let arr: &[f32; 8] = data.first_chunk().unwrap();
    safe_unaligned_simd::x86_64::_mm256_loadu_ps(arr)
}

/// Slice via try_into → should produce same vmovups
#[unsafe(no_mangle)]
#[arcane]
fn load_try_into(_t: Desktop64, data: &[f32]) -> __m256 {
    let arr: &[f32; 8] = data[..8].try_into().unwrap();
    safe_unaligned_simd::x86_64::_mm256_loadu_ps(arr)
}

// ============================================================================
// 256-bit integer load
// ============================================================================

/// Integer load via first_chunk → vmovdqu
#[unsafe(no_mangle)]
#[arcane]
fn load_first_chunk_i(_t: Desktop64, data: &[u8]) -> __m256i {
    let arr: &[u8; 32] = data.first_chunk().unwrap();
    safe_unaligned_simd::x86_64::_mm256_loadu_si256(arr)
}

// ============================================================================
// 128-bit float load
// ============================================================================

/// 128-bit first_chunk → vmovups (128-bit)
#[unsafe(no_mangle)]
#[arcane]
fn load_first_chunk_128(_t: Desktop64, data: &[f32]) -> __m128 {
    let arr: &[f32; 4] = data.first_chunk().unwrap();
    safe_unaligned_simd::x86_64::_mm_loadu_ps(arr)
}

// ============================================================================
// Store patterns
// ============================================================================

/// Store via first_chunk_mut → vmovups (store)
#[unsafe(no_mangle)]
#[arcane]
fn store_first_chunk_mut(_t: Desktop64, v: __m256, out: &mut [f32]) {
    let arr: &mut [f32; 8] = out.first_chunk_mut().unwrap();
    safe_unaligned_simd::x86_64::_mm256_storeu_ps(arr, v);
}

// ============================================================================
// Magetypes patterns
// ============================================================================

/// magetypes from_slice → should produce vmovups
#[unsafe(no_mangle)]
#[arcane]
fn load_f32x8_from_slice(_t: Desktop64, data: &[f32]) -> __m256 {
    use magetypes::simd::f32x8;
    let v = f32x8::from_slice(_t, data);
    v.raw()
}

/// magetypes load via first_chunk → should produce vmovups
#[unsafe(no_mangle)]
#[arcane]
fn load_f32x8_first_chunk(_t: Desktop64, data: &[f32]) -> __m256 {
    use magetypes::simd::f32x8;
    let arr: &[f32; 8] = data.first_chunk().unwrap();
    let v = f32x8::load(_t, arr);
    v.raw()
}

// ============================================================================
// Criterion benchmark (required for cargo asm --bench to work)
// ============================================================================

use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_load_patterns(c: &mut Criterion) {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let slice: &[f32] = &data;
    let bytes = [0u8; 32];
    let byte_slice: &[u8] = &bytes;

    if let Some(token) = Desktop64::summon() {
        c.bench_function("load_array_ref", |b| {
            b.iter(|| load_array_ref(token, black_box(&data)))
        });

        c.bench_function("load_first_chunk", |b| {
            b.iter(|| load_first_chunk(token, black_box(slice)))
        });

        c.bench_function("load_try_into", |b| {
            b.iter(|| load_try_into(token, black_box(slice)))
        });

        c.bench_function("load_first_chunk_i", |b| {
            b.iter(|| load_first_chunk_i(token, black_box(byte_slice)))
        });

        c.bench_function("load_first_chunk_128", |b| {
            b.iter(|| load_first_chunk_128(token, black_box(slice)))
        });

        c.bench_function("load_f32x8_from_slice", |b| {
            b.iter(|| load_f32x8_from_slice(token, black_box(slice)))
        });

        c.bench_function("load_f32x8_first_chunk", |b| {
            b.iter(|| load_f32x8_first_chunk(token, black_box(slice)))
        });
    } else {
        eprintln!("Desktop64 not available, skipping benchmarks");
    }
}

criterion_group!(benches, bench_load_patterns);
criterion_main!(benches);

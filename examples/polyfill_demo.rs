//! Demonstration of polyfilled SIMD types.
//!
//! Run with: `cargo run --example polyfill_demo --release`
//!
//! This example shows how to use the polyfill module to write code
//! targeting AVX2-width vectors (f32x8) while running on SSE hardware.
//!
//! The polyfilled types use two SSE operations for each AVX2-equivalent
//! operation, so they're slower than native AVX2, but faster than scalar
//! and allow writing code targeting a single width.

#![cfg(target_arch = "x86_64")]

use archmage::simd::polyfill::sse as poly;
use archmage::{SimdToken, Sse41Token, arcane};
use std::time::Instant;

const N: usize = 64 * 1024;
const ITERATIONS: u32 = 1000;

/// Sum using polyfilled f32x8 on SSE
#[arcane]
fn sum_polyfill(token: Sse41Token, data: &[f32]) -> f32 {
    let mut acc = poly::f32xN::zero(token);
    let chunks = data.chunks_exact(poly::LANES_F32);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let arr: &[f32; 8] = chunk.try_into().unwrap();
        let v = poly::f32xN::load(token, arr);
        acc += v;
    }

    let mut sum = acc.reduce_add();
    for &x in remainder {
        sum += x;
    }
    sum
}

/// Sum using native SSE f32x4
#[arcane]
fn sum_native_sse(token: archmage::Sse41Token, data: &[f32]) -> f32 {
    use archmage::simd::f32x4;

    let mut acc = f32x4::zero(token);
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let arr: &[f32; 4] = chunk.try_into().unwrap();
        let v = f32x4::load(token, arr);
        acc += v;
    }

    let mut sum = acc.reduce_add();
    for &x in remainder {
        sum += x;
    }
    sum
}

/// Sum using native AVX2 f32x8
#[arcane]
fn sum_native_avx2(token: archmage::Avx2FmaToken, data: &[f32]) -> f32 {
    use archmage::simd::f32x8;

    let mut acc = f32x8::zero(token);
    let chunks = data.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let arr: &[f32; 8] = chunk.try_into().unwrap();
        let v = f32x8::load(token, arr);
        acc += v;
    }

    let mut sum = acc.reduce_add();
    for &x in remainder {
        sum += x;
    }
    sum
}

/// Scalar sum
fn sum_scalar(data: &[f32]) -> f32 {
    data.iter().sum()
}

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║           Polyfill SIMD Demonstration                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let data: Vec<f32> = (0..N).map(|i| (i as f32) * 0.001).collect();

    // Verify correctness
    let expected = sum_scalar(&data);
    println!("Expected sum: {:.2}\n", expected);

    if let Some(token) = archmage::Sse41Token::try_new() {
        let polyfill_result = sum_polyfill(token, &data);
        let native_sse_result = sum_native_sse(token, &data);

        println!("Polyfilled f32x8 on SSE: {:.2}", polyfill_result);
        println!("Native SSE f32x4:        {:.2}", native_sse_result);

        assert!(
            (polyfill_result - expected).abs() / expected.abs() < 0.001,
            "Polyfill result mismatch"
        );
    }

    if let Some(token) = archmage::Avx2FmaToken::try_new() {
        let native_avx2_result = sum_native_avx2(token, &data);
        println!("Native AVX2 f32x8:       {:.2}", native_avx2_result);
    }

    println!(
        "\n=== Benchmarks ({} elements, {} iterations) ===\n",
        N, ITERATIONS
    );

    // Scalar baseline
    let start = Instant::now();
    let mut result;
    for _ in 0..ITERATIONS {
        result = sum_scalar(&data);
        std::hint::black_box(result);
    }
    let scalar_time = start.elapsed();
    println!(
        "  Scalar:           {:>8.2} ms",
        scalar_time.as_secs_f64() * 1000.0
    );

    // SSE polyfill (f32x8 emulated with 2x f32x4)
    if let Some(token) = archmage::Sse41Token::try_new() {
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            result = sum_polyfill(token, &data);
            std::hint::black_box(result);
        }
        let polyfill_time = start.elapsed();
        println!(
            "  Polyfill f32x8:   {:>8.2} ms ({:.1}x faster than scalar)",
            polyfill_time.as_secs_f64() * 1000.0,
            scalar_time.as_secs_f64() / polyfill_time.as_secs_f64()
        );

        // Native SSE f32x4
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            result = sum_native_sse(token, &data);
            std::hint::black_box(result);
        }
        let native_sse_time = start.elapsed();
        println!(
            "  Native SSE f32x4: {:>8.2} ms ({:.1}x faster than scalar)",
            native_sse_time.as_secs_f64() * 1000.0,
            scalar_time.as_secs_f64() / native_sse_time.as_secs_f64()
        );
    }

    // Native AVX2 f32x8
    if let Some(token) = archmage::Avx2FmaToken::try_new() {
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            result = sum_native_avx2(token, &data);
            std::hint::black_box(result);
        }
        let native_avx2_time = start.elapsed();
        println!(
            "  Native AVX2 f32x8:{:>8.2} ms ({:.1}x faster than scalar)",
            native_avx2_time.as_secs_f64() * 1000.0,
            scalar_time.as_secs_f64() / native_avx2_time.as_secs_f64()
        );
    }

    println!("\nDone!");
}

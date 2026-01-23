//! Comprehensive demonstration of the #[multiwidth] macro.
//!
//! Run with: `cargo run --example multiwidth_demo --release`
//! With AVX-512: `cargo run --example multiwidth_demo --release --features avx512`
//!
//! This example shows:
//! - Writing width-agnostic SIMD code
//! - Automatic specialization for SSE, AVX2, and AVX-512
//! - Runtime dispatch to best available implementation
//! - Performance comparison across widths

#![cfg(target_arch = "x86_64")]
#![allow(clippy::approx_constant)]

use archmage::multiwidth;
use std::time::Instant;

// Test data sizes
const SMALL_N: usize = 1024;
const LARGE_N: usize = 64 * 1024;
const BENCH_ITERATIONS: u32 = 1000;

// ============================================================================
// Width-agnostic SIMD kernels using #[multiwidth]
// ============================================================================

#[multiwidth]
mod kernels {
    use magetypes::simd::*;

    /// Sum all elements in a slice using SIMD accumulation.
    pub fn sum(token: Token, data: &[f32]) -> f32 {
        let mut acc = f32xN::zero(token);
        let chunks = data.chunks_exact(LANES_F32);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let arr: &[f32; LANES_F32] = chunk.try_into().unwrap();
            let v = f32xN::load(token, arr);
            acc += v;
        }

        let mut sum = acc.reduce_add();
        for &x in remainder {
            sum += x;
        }
        sum
    }

    /// Dot product of two vectors.
    pub fn dot_product(token: Token, a: &[f32], b: &[f32]) -> f32 {
        let mut acc = f32xN::zero(token);
        let a_chunks = a.chunks_exact(LANES_F32);
        let b_chunks = b.chunks_exact(LANES_F32);
        let a_rem = a_chunks.remainder();
        let b_rem = b_chunks.remainder();

        for (ca, cb) in a_chunks.zip(b_chunks) {
            let arr_a: &[f32; LANES_F32] = ca.try_into().unwrap();
            let arr_b: &[f32; LANES_F32] = cb.try_into().unwrap();
            let va = f32xN::load(token, arr_a);
            let vb = f32xN::load(token, arr_b);
            acc = va.mul_add(vb, acc); // va * vb + acc using FMA
        }

        let mut sum = acc.reduce_add();
        for (&x, &y) in a_rem.iter().zip(b_rem.iter()) {
            sum += x * y;
        }
        sum
    }

    /// Scale all elements by a constant factor (in-place).
    pub fn scale(token: Token, data: &mut [f32], factor: f32) {
        let factor_v = f32xN::splat(token, factor);
        let (chunks, remainder) = data.split_at_mut(data.len() - data.len() % LANES_F32);

        for chunk in chunks.chunks_exact_mut(LANES_F32) {
            let arr: &[f32; LANES_F32] = (&*chunk).try_into().unwrap();
            let v = f32xN::load(token, arr);
            let scaled = v * factor_v;
            let out_arr: &mut [f32; LANES_F32] = chunk.try_into().unwrap();
            scaled.store(out_arr);
        }

        for x in remainder {
            *x *= factor;
        }
    }

    /// Add two arrays element-wise, storing result in output.
    pub fn add_arrays(token: Token, a: &[f32], b: &[f32], out: &mut [f32]) {
        let len = a.len().min(b.len()).min(out.len());
        let simd_len = len - len % LANES_F32;

        for i in (0..simd_len).step_by(LANES_F32) {
            let arr_a: &[f32; LANES_F32] = (&a[i..i + LANES_F32]).try_into().unwrap();
            let arr_b: &[f32; LANES_F32] = (&b[i..i + LANES_F32]).try_into().unwrap();
            let va = f32xN::load(token, arr_a);
            let vb = f32xN::load(token, arr_b);
            let result = va + vb;
            let out_arr: &mut [f32; LANES_F32] = (&mut out[i..i + LANES_F32]).try_into().unwrap();
            result.store(out_arr);
        }

        for i in simd_len..len {
            out[i] = a[i] + b[i];
        }
    }

    /// Find the maximum value in an array.
    pub fn max_value(token: Token, data: &[f32]) -> f32 {
        if data.is_empty() {
            return f32::NEG_INFINITY;
        }

        let mut acc = f32xN::splat(token, f32::NEG_INFINITY);
        let chunks = data.chunks_exact(LANES_F32);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let arr: &[f32; LANES_F32] = chunk.try_into().unwrap();
            let v = f32xN::load(token, arr);
            acc = acc.max(v);
        }

        let mut max_val = acc.reduce_max();
        for &x in remainder {
            if x > max_val {
                max_val = x;
            }
        }
        max_val
    }

    /// Find the minimum value in an array.
    pub fn min_value(token: Token, data: &[f32]) -> f32 {
        if data.is_empty() {
            return f32::INFINITY;
        }

        let mut acc = f32xN::splat(token, f32::INFINITY);
        let chunks = data.chunks_exact(LANES_F32);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let arr: &[f32; LANES_F32] = chunk.try_into().unwrap();
            let v = f32xN::load(token, arr);
            acc = acc.min(v);
        }

        let mut min_val = acc.reduce_min();
        for &x in remainder {
            if x < min_val {
                min_val = x;
            }
        }
        min_val
    }

    /// Compute element-wise square root (in-place).
    pub fn sqrt_array(token: Token, data: &mut [f32]) {
        let (chunks, remainder) = data.split_at_mut(data.len() - data.len() % LANES_F32);

        for chunk in chunks.chunks_exact_mut(LANES_F32) {
            let arr: &[f32; LANES_F32] = (&*chunk).try_into().unwrap();
            let v = f32xN::load(token, arr);
            let result = v.sqrt();
            let out_arr: &mut [f32; LANES_F32] = chunk.try_into().unwrap();
            result.store(out_arr);
        }

        for x in remainder {
            *x = x.sqrt();
        }
    }

    /// Clamp all values to a range (in-place).
    pub fn clamp_array(token: Token, data: &mut [f32], min: f32, max: f32) {
        let min_v = f32xN::splat(token, min);
        let max_v = f32xN::splat(token, max);
        let (chunks, remainder) = data.split_at_mut(data.len() - data.len() % LANES_F32);

        for chunk in chunks.chunks_exact_mut(LANES_F32) {
            let arr: &[f32; LANES_F32] = (&*chunk).try_into().unwrap();
            let v = f32xN::load(token, arr);
            let clamped = v.clamp(min_v, max_v);
            let out_arr: &mut [f32; LANES_F32] = chunk.try_into().unwrap();
            clamped.store(out_arr);
        }

        for x in remainder {
            *x = x.clamp(min, max);
        }
    }

    /// Normalize array to [0, 1] range based on min/max.
    pub fn normalize(token: Token, data: &mut [f32]) {
        let min_val = min_value(token, data);
        let max_val = max_value(token, data);
        let range = max_val - min_val;

        if range == 0.0 {
            return;
        }

        let min_v = f32xN::splat(token, min_val);
        let inv_range = f32xN::splat(token, 1.0 / range);
        let (chunks, remainder) = data.split_at_mut(data.len() - data.len() % LANES_F32);

        for chunk in chunks.chunks_exact_mut(LANES_F32) {
            let arr: &[f32; LANES_F32] = (&*chunk).try_into().unwrap();
            let v = f32xN::load(token, arr);
            let normalized = (v - min_v) * inv_range;
            let out_arr: &mut [f32; LANES_F32] = chunk.try_into().unwrap();
            normalized.store(out_arr);
        }

        for x in remainder {
            *x = (*x - min_val) / range;
        }
    }

    /// Compute L2 norm (Euclidean length) of a vector.
    pub fn l2_norm(token: Token, data: &[f32]) -> f32 {
        let mut acc = f32xN::zero(token);
        let chunks = data.chunks_exact(LANES_F32);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let arr: &[f32; LANES_F32] = chunk.try_into().unwrap();
            let v = f32xN::load(token, arr);
            acc = v.mul_add(v, acc); // v * v + acc using FMA
        }

        let mut sum = acc.reduce_add();
        for &x in remainder {
            sum += x * x;
        }
        sum.sqrt()
    }
}

// ============================================================================
// Scalar baseline implementations for comparison
// ============================================================================

fn scalar_sum(data: &[f32]) -> f32 {
    data.iter().sum()
}

fn scalar_dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn scalar_scale(data: &mut [f32], factor: f32) {
    for x in data {
        *x *= factor;
    }
}

fn scalar_max(data: &[f32]) -> f32 {
    data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

fn scalar_l2_norm(data: &[f32]) -> f32 {
    data.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ============================================================================
// Testing
// ============================================================================

fn test_correctness() {
    println!("=== Correctness Tests ===\n");

    // Test data
    let data: Vec<f32> = (0..SMALL_N).map(|i| (i as f32) * 0.1).collect();
    let data2: Vec<f32> = (0..SMALL_N).map(|i| (i as f32) * 0.05 + 1.0).collect();

    // Test sum
    let expected_sum = scalar_sum(&data);
    let simd_sum = kernels::sum(&data);
    assert!(
        (simd_sum - expected_sum).abs() < 0.01,
        "sum: expected {}, got {}",
        expected_sum,
        simd_sum
    );
    println!(
        "  sum:         PASS (expected {:.2}, got {:.2})",
        expected_sum, simd_sum
    );

    // Test dot product
    let expected_dot = scalar_dot_product(&data, &data2);
    let simd_dot = kernels::dot_product(&data, &data2);
    assert!(
        (simd_dot - expected_dot).abs() / expected_dot.abs() < 0.0001,
        "dot_product: expected {}, got {}",
        expected_dot,
        simd_dot
    );
    println!(
        "  dot_product: PASS (expected {:.2}, got {:.2})",
        expected_dot, simd_dot
    );

    // Test scale
    let mut data_clone = data.clone();
    let mut expected_clone = data.clone();
    kernels::scale(&mut data_clone, 2.5);
    scalar_scale(&mut expected_clone, 2.5);
    assert!(
        data_clone
            .iter()
            .zip(expected_clone.iter())
            .all(|(a, b)| (a - b).abs() < 0.001),
        "scale: mismatch"
    );
    println!("  scale:       PASS");

    // Test max
    let expected_max = scalar_max(&data);
    let simd_max = kernels::max_value(&data);
    assert!(
        (simd_max - expected_max).abs() < 0.001,
        "max_value: expected {}, got {}",
        expected_max,
        simd_max
    );
    println!(
        "  max_value:   PASS (expected {:.2}, got {:.2})",
        expected_max, simd_max
    );

    // Test L2 norm
    let expected_norm = scalar_l2_norm(&data);
    let simd_norm = kernels::l2_norm(&data);
    assert!(
        (simd_norm - expected_norm).abs() / expected_norm.abs() < 0.0001,
        "l2_norm: expected {}, got {}",
        expected_norm,
        simd_norm
    );
    println!(
        "  l2_norm:     PASS (expected {:.2}, got {:.2})",
        expected_norm, simd_norm
    );

    println!("\nAll correctness tests passed!\n");
}

// ============================================================================
// Benchmarking
// ============================================================================

fn bench_sum() {
    println!(
        "=== Benchmarking sum ({} elements, {} iterations) ===\n",
        LARGE_N, BENCH_ITERATIONS
    );

    let data: Vec<f32> = (0..LARGE_N).map(|i| (i as f32) * 0.001).collect();

    // Scalar baseline
    let start = Instant::now();
    let mut result = 0.0f32;
    for _ in 0..BENCH_ITERATIONS {
        result = scalar_sum(&data);
        std::hint::black_box(result);
    }
    let scalar_time = start.elapsed();
    println!(
        "  Scalar:     {:>8.2} ms (result: {:.2})",
        scalar_time.as_secs_f64() * 1000.0,
        result
    );

    // Auto-dispatch (picks best)
    let start = Instant::now();
    for _ in 0..BENCH_ITERATIONS {
        result = kernels::sum(&data);
        std::hint::black_box(result);
    }
    let dispatch_time = start.elapsed();
    println!(
        "  Dispatch:   {:>8.2} ms ({:.1}x faster)",
        dispatch_time.as_secs_f64() * 1000.0,
        scalar_time.as_secs_f64() / dispatch_time.as_secs_f64()
    );

    // Width-specific implementations
    use archmage::SimdToken;

    if let Some(token) = archmage::Sse41Token::try_new() {
        let start = Instant::now();
        for _ in 0..BENCH_ITERATIONS {
            result = kernels::sse::sum(token, &data);
            std::hint::black_box(result);
        }
        let sse_time = start.elapsed();
        println!(
            "  SSE (4x):   {:>8.2} ms ({:.1}x faster)",
            sse_time.as_secs_f64() * 1000.0,
            scalar_time.as_secs_f64() / sse_time.as_secs_f64()
        );
    }

    if let Some(token) = archmage::Avx2FmaToken::try_new() {
        let start = Instant::now();
        for _ in 0..BENCH_ITERATIONS {
            result = kernels::avx2::sum(token, &data);
            std::hint::black_box(result);
        }
        let avx2_time = start.elapsed();
        println!(
            "  AVX2 (8x):  {:>8.2} ms ({:.1}x faster)",
            avx2_time.as_secs_f64() * 1000.0,
            scalar_time.as_secs_f64() / avx2_time.as_secs_f64()
        );
    }

    #[cfg(feature = "avx512")]
    if let Some(token) = archmage::X64V4Token::try_new() {
        let start = Instant::now();
        for _ in 0..BENCH_ITERATIONS {
            result = kernels::avx512::sum(token, &data);
            std::hint::black_box(result);
        }
        let avx512_time = start.elapsed();
        println!(
            "  AVX512(16x):{:>8.2} ms ({:.1}x faster)",
            avx512_time.as_secs_f64() * 1000.0,
            scalar_time.as_secs_f64() / avx512_time.as_secs_f64()
        );
    }

    println!();
}

fn bench_dot_product() {
    println!(
        "=== Benchmarking dot_product ({} elements, {} iterations) ===\n",
        LARGE_N, BENCH_ITERATIONS
    );

    let a: Vec<f32> = (0..LARGE_N).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..LARGE_N)
        .map(|i| ((LARGE_N - i) as f32) * 0.001)
        .collect();

    // Scalar baseline
    let start = Instant::now();
    let mut result = 0.0f32;
    for _ in 0..BENCH_ITERATIONS {
        result = scalar_dot_product(&a, &b);
        std::hint::black_box(result);
    }
    let scalar_time = start.elapsed();
    println!(
        "  Scalar:     {:>8.2} ms (result: {:.2})",
        scalar_time.as_secs_f64() * 1000.0,
        result
    );

    // Auto-dispatch
    let start = Instant::now();
    for _ in 0..BENCH_ITERATIONS {
        result = kernels::dot_product(&a, &b);
        std::hint::black_box(result);
    }
    let dispatch_time = start.elapsed();
    println!(
        "  Dispatch:   {:>8.2} ms ({:.1}x faster)",
        dispatch_time.as_secs_f64() * 1000.0,
        scalar_time.as_secs_f64() / dispatch_time.as_secs_f64()
    );

    use archmage::SimdToken;

    if let Some(token) = archmage::Sse41Token::try_new() {
        let start = Instant::now();
        for _ in 0..BENCH_ITERATIONS {
            result = kernels::sse::dot_product(token, &a, &b);
            std::hint::black_box(result);
        }
        let sse_time = start.elapsed();
        println!(
            "  SSE (4x):   {:>8.2} ms ({:.1}x faster)",
            sse_time.as_secs_f64() * 1000.0,
            scalar_time.as_secs_f64() / sse_time.as_secs_f64()
        );
    }

    if let Some(token) = archmage::Avx2FmaToken::try_new() {
        let start = Instant::now();
        for _ in 0..BENCH_ITERATIONS {
            result = kernels::avx2::dot_product(token, &a, &b);
            std::hint::black_box(result);
        }
        let avx2_time = start.elapsed();
        println!(
            "  AVX2 (8x):  {:>8.2} ms ({:.1}x faster)",
            avx2_time.as_secs_f64() * 1000.0,
            scalar_time.as_secs_f64() / avx2_time.as_secs_f64()
        );
    }

    #[cfg(feature = "avx512")]
    if let Some(token) = archmage::X64V4Token::try_new() {
        let start = Instant::now();
        for _ in 0..BENCH_ITERATIONS {
            result = kernels::avx512::dot_product(token, &a, &b);
            std::hint::black_box(result);
        }
        let avx512_time = start.elapsed();
        println!(
            "  AVX512(16x):{:>8.2} ms ({:.1}x faster)",
            avx512_time.as_secs_f64() * 1000.0,
            scalar_time.as_secs_f64() / avx512_time.as_secs_f64()
        );
    }

    println!();
}

fn bench_scale() {
    println!(
        "=== Benchmarking scale ({} elements, {} iterations) ===\n",
        LARGE_N, BENCH_ITERATIONS
    );

    let original: Vec<f32> = (0..LARGE_N).map(|i| (i as f32) * 0.001).collect();

    // Scalar baseline
    let mut data = original.clone();
    let start = Instant::now();
    for _ in 0..BENCH_ITERATIONS {
        data.copy_from_slice(&original);
        scalar_scale(&mut data, 2.5);
        std::hint::black_box(&data);
    }
    let scalar_time = start.elapsed();
    println!(
        "  Scalar:     {:>8.2} ms",
        scalar_time.as_secs_f64() * 1000.0
    );

    // Auto-dispatch
    let start = Instant::now();
    for _ in 0..BENCH_ITERATIONS {
        data.copy_from_slice(&original);
        kernels::scale(&mut data, 2.5);
        std::hint::black_box(&data);
    }
    let dispatch_time = start.elapsed();
    println!(
        "  Dispatch:   {:>8.2} ms ({:.1}x faster)",
        dispatch_time.as_secs_f64() * 1000.0,
        scalar_time.as_secs_f64() / dispatch_time.as_secs_f64()
    );

    use archmage::SimdToken;

    if let Some(token) = archmage::Sse41Token::try_new() {
        let start = Instant::now();
        for _ in 0..BENCH_ITERATIONS {
            data.copy_from_slice(&original);
            kernels::sse::scale(token, &mut data, 2.5);
            std::hint::black_box(&data);
        }
        let sse_time = start.elapsed();
        println!(
            "  SSE (4x):   {:>8.2} ms ({:.1}x faster)",
            sse_time.as_secs_f64() * 1000.0,
            scalar_time.as_secs_f64() / sse_time.as_secs_f64()
        );
    }

    if let Some(token) = archmage::Avx2FmaToken::try_new() {
        let start = Instant::now();
        for _ in 0..BENCH_ITERATIONS {
            data.copy_from_slice(&original);
            kernels::avx2::scale(token, &mut data, 2.5);
            std::hint::black_box(&data);
        }
        let avx2_time = start.elapsed();
        println!(
            "  AVX2 (8x):  {:>8.2} ms ({:.1}x faster)",
            avx2_time.as_secs_f64() * 1000.0,
            scalar_time.as_secs_f64() / avx2_time.as_secs_f64()
        );
    }

    #[cfg(feature = "avx512")]
    if let Some(token) = archmage::X64V4Token::try_new() {
        let start = Instant::now();
        for _ in 0..BENCH_ITERATIONS {
            data.copy_from_slice(&original);
            kernels::avx512::scale(token, &mut data, 2.5);
            std::hint::black_box(&data);
        }
        let avx512_time = start.elapsed();
        println!(
            "  AVX512(16x):{:>8.2} ms ({:.1}x faster)",
            avx512_time.as_secs_f64() * 1000.0,
            scalar_time.as_secs_f64() / avx512_time.as_secs_f64()
        );
    }

    println!();
}

fn bench_l2_norm() {
    println!(
        "=== Benchmarking l2_norm ({} elements, {} iterations) ===\n",
        LARGE_N, BENCH_ITERATIONS
    );

    let data: Vec<f32> = (0..LARGE_N).map(|i| (i as f32) * 0.001).collect();

    // Scalar baseline
    let start = Instant::now();
    let mut result = 0.0f32;
    for _ in 0..BENCH_ITERATIONS {
        result = scalar_l2_norm(&data);
        std::hint::black_box(result);
    }
    let scalar_time = start.elapsed();
    println!(
        "  Scalar:     {:>8.2} ms (result: {:.2})",
        scalar_time.as_secs_f64() * 1000.0,
        result
    );

    // Auto-dispatch
    let start = Instant::now();
    for _ in 0..BENCH_ITERATIONS {
        result = kernels::l2_norm(&data);
        std::hint::black_box(result);
    }
    let dispatch_time = start.elapsed();
    println!(
        "  Dispatch:   {:>8.2} ms ({:.1}x faster)",
        dispatch_time.as_secs_f64() * 1000.0,
        scalar_time.as_secs_f64() / dispatch_time.as_secs_f64()
    );

    use archmage::SimdToken;

    if let Some(token) = archmage::Sse41Token::try_new() {
        let start = Instant::now();
        for _ in 0..BENCH_ITERATIONS {
            result = kernels::sse::l2_norm(token, &data);
            std::hint::black_box(result);
        }
        let sse_time = start.elapsed();
        println!(
            "  SSE (4x):   {:>8.2} ms ({:.1}x faster)",
            sse_time.as_secs_f64() * 1000.0,
            scalar_time.as_secs_f64() / sse_time.as_secs_f64()
        );
    }

    if let Some(token) = archmage::Avx2FmaToken::try_new() {
        let start = Instant::now();
        for _ in 0..BENCH_ITERATIONS {
            result = kernels::avx2::l2_norm(token, &data);
            std::hint::black_box(result);
        }
        let avx2_time = start.elapsed();
        println!(
            "  AVX2 (8x):  {:>8.2} ms ({:.1}x faster)",
            avx2_time.as_secs_f64() * 1000.0,
            scalar_time.as_secs_f64() / avx2_time.as_secs_f64()
        );
    }

    #[cfg(feature = "avx512")]
    if let Some(token) = archmage::X64V4Token::try_new() {
        let start = Instant::now();
        for _ in 0..BENCH_ITERATIONS {
            result = kernels::avx512::l2_norm(token, &data);
            std::hint::black_box(result);
        }
        let avx512_time = start.elapsed();
        println!(
            "  AVX512(16x):{:>8.2} ms ({:.1}x faster)",
            avx512_time.as_secs_f64() * 1000.0,
            scalar_time.as_secs_f64() / avx512_time.as_secs_f64()
        );
    }

    println!();
}

fn show_available_widths() {
    println!("=== Available SIMD Widths ===\n");

    use archmage::SimdToken;

    print!("  SSE4.1 (128-bit, 4 lanes):  ");
    if archmage::Sse41Token::try_new().is_some() {
        println!("AVAILABLE");
    } else {
        println!("not available");
    }

    print!("  AVX2+FMA (256-bit, 8 lanes): ");
    if archmage::Avx2FmaToken::try_new().is_some() {
        println!("AVAILABLE");
    } else {
        println!("not available");
    }

    #[cfg(feature = "avx512")]
    {
        print!("  AVX-512 (512-bit, 16 lanes): ");
        if archmage::X64V4Token::try_new().is_some() {
            println!("AVAILABLE");
        } else {
            println!("not available");
        }
    }

    #[cfg(not(feature = "avx512"))]
    println!("  AVX-512: compile with --features avx512 to enable");

    println!();
}

fn warmup() {
    // Warm up SIMD units by running some operations
    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    for _ in 0..100 {
        std::hint::black_box(kernels::sum(&data));
        std::hint::black_box(kernels::l2_norm(&data));
    }
}

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║           Multiwidth SIMD Demonstration                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    show_available_widths();

    println!("Warming up SIMD units...\n");
    warmup();

    test_correctness();
    bench_sum();
    bench_dot_product();
    bench_scale();
    bench_l2_norm();

    println!("Done!");
}

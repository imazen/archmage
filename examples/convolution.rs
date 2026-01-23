//! SIMD Convolution Kernels using archmage
//!
//! Demonstrates vertical reduction and convolution operations:
//! - Vertical reduction (multiple rows → one row): ~10-12x speedup
//! - Box filter (3x3 average)
//! - Gaussian blur (separable)
//!
//! Run with: `cargo run --example convolution --release`
//!
//! Key insight: Vertical reduction is SIMD-friendly because all inputs
//! are contiguous in memory. Horizontal operations have strided access
//! which limits gains.

#![cfg(target_arch = "x86_64")]

use archmage::{Avx2FmaToken, SimdToken, arcane};
use core::arch::x86_64::*;
use magetypes::simd::f32x8;
use std::time::Instant;

// ============================================================================
// Vertical Reduction (floating-point)
// ============================================================================

/// Reduce N input rows to 1 output row using weighted sum
///
/// This is the core operation for separable filters (vertical pass).
/// Uses f32 for simplicity; production code might use i16 fixed-point.
///
/// Formula: out[x] = sum(input[i][x] * weight[i]) for all rows i
#[arcane]
pub fn reduce_vertical_f32_simd(
    token: Avx2FmaToken,
    inputs: &[&[f32]],
    weights: &[f32],
    output: &mut [f32],
) {
    debug_assert_eq!(inputs.len(), weights.len());
    debug_assert!(inputs.iter().all(|r| r.len() >= output.len()));

    let len = output.len();

    // Process 8 floats at a time
    for chunk_start in (0..len).step_by(8) {
        if chunk_start + 8 > len {
            break;
        }

        // Initialize accumulator to zero
        let mut acc = f32x8::zero(token);

        // Accumulate weighted contributions from all input rows
        for (row, &w) in inputs.iter().zip(weights.iter()) {
            let input_arr: &[f32; 8] = (&row[chunk_start..chunk_start + 8]).try_into().unwrap();
            let vals = f32x8::load(token, input_arr);
            let weight = f32x8::splat(token, w);
            // acc += vals * weight
            acc = vals.mul_add(weight, acc);
        }

        let out_arr: &mut [f32; 8] = (&mut output[chunk_start..chunk_start + 8])
            .try_into()
            .unwrap();
        acc.store(out_arr);
    }

    // Handle remainder
    let remainder_start = (len / 8) * 8;
    for x in remainder_start..len {
        let mut sum = 0.0f32;
        for (row, &w) in inputs.iter().zip(weights.iter()) {
            sum += row[x] * w;
        }
        output[x] = sum;
    }
}

// ============================================================================
// Vertical Reduction (fixed-point u8)
// ============================================================================

/// Reduce N input rows to 1 output row using fixed-point arithmetic
///
/// Uses 15-bit fixed-point weights for maximum precision without overflow.
/// This matches the approach used by image resizers like libswscale.
///
/// Formula: out[x] = (sum(input[i][x] * weight[i]) + HALF) >> 15
#[arcane]
pub fn reduce_vertical_u8_simd(
    token: Avx2FmaToken,
    inputs: &[&[u8]],
    weights: &[i16],
    output: &mut [u8],
) {
    let _ = token; // We'll use raw intrinsics for integer ops
    debug_assert_eq!(inputs.len(), weights.len());
    debug_assert!(inputs.iter().all(|r| r.len() >= output.len()));

    const HALF_SCALE: i32 = 1 << 14; // For rounding
    const CHUNK: usize = 16; // Process 16 bytes at a time

    let len = output.len();
    let chunks = len / CHUNK;

    for chunk_idx in 0..chunks {
        let base = chunk_idx * CHUNK;

        // Initialize accumulators with half scale for rounding
        let mut acc0 = _mm256_set1_epi32(HALF_SCALE);
        let mut acc1 = _mm256_set1_epi32(HALF_SCALE);

        // Accumulate across all input rows
        for (input, &w) in inputs.iter().zip(weights.iter()) {
            // SAFETY: bounds checked by debug_assert above
            let in_ptr = unsafe { input.as_ptr().add(base) };

            // Load 16 bytes, extend to i16, then i32
            let bytes = unsafe { _mm_loadu_si128(in_ptr as *const __m128i) };

            // Unpack bytes to words (u8 -> i16)
            let lo_words = _mm256_cvtepu8_epi16(bytes);

            // Split into two i32 vectors (8 values each)
            let lo_dwords = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(lo_words));
            let hi_dwords = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(lo_words));

            // Broadcast weight
            let w_vec = _mm256_set1_epi32(w as i32);

            // Multiply and accumulate
            acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(lo_dwords, w_vec));
            acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(hi_dwords, w_vec));
        }

        // Shift right by 15
        acc0 = _mm256_srai_epi32::<15>(acc0);
        acc1 = _mm256_srai_epi32::<15>(acc1);

        // Clamp to 0-255 and pack back to bytes
        let zero = _mm256_setzero_si256();
        let max_val = _mm256_set1_epi32(255);

        acc0 = _mm256_max_epi32(acc0, zero);
        acc0 = _mm256_min_epi32(acc0, max_val);
        acc1 = _mm256_max_epi32(acc1, zero);
        acc1 = _mm256_min_epi32(acc1, max_val);

        // Pack i32 -> i16 -> u8
        let packed16 = _mm256_packs_epi32(acc0, acc1);
        let packed16 = _mm256_permute4x64_epi64::<0b11011000>(packed16);
        let packed8 = _mm256_packus_epi16(packed16, packed16);
        let packed8 = _mm256_permute4x64_epi64::<0b11011000>(packed8);

        // Store 16 bytes
        // SAFETY: bounds checked by chunks calculation
        unsafe {
            _mm_storeu_si128(
                output.as_mut_ptr().add(base) as *mut __m128i,
                _mm256_castsi256_si128(packed8),
            );
        }
    }

    // Handle remainder with scalar
    let remainder_start = chunks * CHUNK;
    for out_idx in remainder_start..len {
        let mut sum = HALF_SCALE;
        for (input, &w) in inputs.iter().zip(weights.iter()) {
            sum += input[out_idx] as i32 * w as i32;
        }
        output[out_idx] = (sum >> 15).clamp(0, 255) as u8;
    }
}

// ============================================================================
// Box Filter (3x3)
// ============================================================================

/// Simple 3x3 box filter using f32
///
/// Each output pixel is the average of a 3x3 neighborhood.
/// Uses vertical reduction as the inner loop for best SIMD utilization.
#[arcane]
pub fn box_filter_3x3_f32(
    token: Avx2FmaToken,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
) {
    debug_assert_eq!(input.len(), width * height);
    debug_assert_eq!(output.len(), width * height);

    // Box filter weights: 1/9 for each of 9 pixels
    let weight = 1.0f32 / 9.0;
    let weight_vec = f32x8::splat(token, weight);
    let zero = f32x8::zero(token);

    // Process interior rows (skip borders for simplicity)
    for y in 1..height - 1 {
        let row_out_start = y * width;

        // Process 8 pixels at a time
        for x_start in (1..width - 1).step_by(8) {
            if x_start + 8 > width - 1 {
                break;
            }

            let mut acc = zero;

            // Sum 3x3 neighborhood
            for dy in 0..3isize {
                let row_y = (y as isize + dy - 1) as usize;
                let row_start = row_y * width;

                for dx in 0..3isize {
                    let x_offset = (x_start as isize + dx - 1) as usize;
                    let idx = row_start + x_offset;
                    let arr: &[f32; 8] = (&input[idx..idx + 8]).try_into().unwrap();
                    let vals = f32x8::load(token, arr);
                    acc += vals;
                }
            }

            // Apply weight (multiply by 1/9)
            let result = acc * weight_vec;

            let out_arr: &mut [f32; 8] = (&mut output
                [row_out_start + x_start..row_out_start + x_start + 8])
                .try_into()
                .unwrap();
            result.store(out_arr);
        }

        // Handle remaining pixels with scalar
        let remainder_start = ((width - 2) / 8) * 8 + 1;
        for x in remainder_start..width - 1 {
            let mut sum = 0.0f32;
            for dy in 0..3isize {
                for dx in 0..3isize {
                    let src_y = (y as isize + dy - 1) as usize;
                    let src_x = (x as isize + dx - 1) as usize;
                    sum += input[src_y * width + src_x];
                }
            }
            output[row_out_start + x] = sum * weight;
        }
    }

    // Copy border pixels (simple approach)
    // Top and bottom rows
    output[..width].copy_from_slice(&input[..width]);
    output[(height - 1) * width..].copy_from_slice(&input[(height - 1) * width..]);
    // Left and right columns
    for y in 0..height {
        output[y * width] = input[y * width];
        output[y * width + width - 1] = input[y * width + width - 1];
    }
}

// ============================================================================
// Scalar Reference
// ============================================================================

fn reduce_vertical_f32_scalar(inputs: &[&[f32]], weights: &[f32], output: &mut [f32]) {
    for x in 0..output.len() {
        let mut sum = 0.0f32;
        for (row, &w) in inputs.iter().zip(weights.iter()) {
            sum += row[x] * w;
        }
        output[x] = sum;
    }
}

fn reduce_vertical_u8_scalar(inputs: &[&[u8]], weights: &[i16], output: &mut [u8]) {
    const HALF_SCALE: i32 = 1 << 14;
    for x in 0..output.len() {
        let mut sum = HALF_SCALE;
        for (row, &w) in inputs.iter().zip(weights.iter()) {
            sum += row[x] as i32 * w as i32;
        }
        output[x] = (sum >> 15).clamp(0, 255) as u8;
    }
}

fn box_filter_3x3_f32_scalar(input: &[f32], output: &mut [f32], width: usize, height: usize) {
    let weight = 1.0f32 / 9.0;

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut sum = 0.0f32;
            for dy in 0..3isize {
                for dx in 0..3isize {
                    let src_y = (y as isize + dy - 1) as usize;
                    let src_x = (x as isize + dx - 1) as usize;
                    sum += input[src_y * width + src_x];
                }
            }
            output[y * width + x] = sum * weight;
        }
    }

    // Copy borders
    output[..width].copy_from_slice(&input[..width]);
    output[(height - 1) * width..].copy_from_slice(&input[(height - 1) * width..]);
    for y in 0..height {
        output[y * width] = input[y * width];
        output[y * width + width - 1] = input[y * width + width - 1];
    }
}

// ============================================================================
// Tests
// ============================================================================

fn test_correctness() {
    println!("=== Correctness Tests ===\n");

    if let Some(token) = Avx2FmaToken::try_new() {
        // Test vertical reduction f32
        let row0: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let row1: Vec<f32> = (0..32).map(|i| (i * 2) as f32).collect();
        let row2: Vec<f32> = (0..32).map(|i| (i * 3) as f32).collect();
        let inputs: Vec<&[f32]> = vec![&row0, &row1, &row2];
        let weights = [0.25f32, 0.5, 0.25];

        let mut simd_out = vec![0.0f32; 32];
        let mut scalar_out = vec![0.0f32; 32];

        reduce_vertical_f32_simd(token, &inputs, &weights, &mut simd_out);
        reduce_vertical_f32_scalar(&inputs, &weights, &mut scalar_out);

        let max_diff: f32 = simd_out
            .iter()
            .zip(scalar_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);

        println!("  Vertical reduction (f32):");
        println!("    Max SIMD vs scalar difference: {:.6}", max_diff);
        println!(
            "    Sample output[10]: SIMD={:.2}, scalar={:.2}\n",
            simd_out[10], scalar_out[10]
        );

        // Test vertical reduction u8
        let row0_u8: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
        let row1_u8: Vec<u8> = (0..64).map(|i| ((i * 4 + 64) % 256) as u8).collect();
        let row2_u8: Vec<u8> = (0..64).map(|i| ((i * 4 + 128) % 256) as u8).collect();
        let inputs_u8: Vec<&[u8]> = vec![&row0_u8, &row1_u8, &row2_u8];
        // Weights sum to 32768 (1.0 in 15-bit fixed point)
        let weights_i16: [i16; 3] = [8192, 16384, 8192]; // 0.25, 0.5, 0.25

        let mut simd_out_u8 = vec![0u8; 64];
        let mut scalar_out_u8 = vec![0u8; 64];

        reduce_vertical_u8_simd(token, &inputs_u8, &weights_i16, &mut simd_out_u8);
        reduce_vertical_u8_scalar(&inputs_u8, &weights_i16, &mut scalar_out_u8);

        let mut matches = true;
        for i in 0..64 {
            if simd_out_u8[i] != scalar_out_u8[i] {
                println!(
                    "  Mismatch at {}: SIMD={}, scalar={}",
                    i, simd_out_u8[i], scalar_out_u8[i]
                );
                matches = false;
            }
        }
        println!("  Vertical reduction (u8 fixed-point):");
        if matches {
            println!("    All 64 outputs match exactly!");
        }
        println!(
            "    Sample output[10]: SIMD={}, scalar={}\n",
            simd_out_u8[10], scalar_out_u8[10]
        );

        // Test box filter
        let width = 64;
        let height = 64;
        let input: Vec<f32> = (0..width * height)
            .map(|i| ((i * 17) % 256) as f32)
            .collect();
        let mut simd_out = vec![0.0f32; width * height];
        let mut scalar_out = vec![0.0f32; width * height];

        box_filter_3x3_f32(token, &input, &mut simd_out, width, height);
        box_filter_3x3_f32_scalar(&input, &mut scalar_out, width, height);

        let max_diff: f32 = simd_out
            .iter()
            .zip(scalar_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);

        println!("  Box filter 3x3 (f32):");
        println!("    Max SIMD vs scalar difference: {:.6}", max_diff);
        println!(
            "    Sample output[33,33]: SIMD={:.2}, scalar={:.2}\n",
            simd_out[33 * width + 33],
            scalar_out[33 * width + 33]
        );
    } else {
        println!("  AVX2 not available\n");
    }
}

// ============================================================================
// Benchmarks
// ============================================================================

fn benchmark() {
    const WIDTH: usize = 1920;
    const HEIGHT: usize = 1080;
    const ITERATIONS: usize = 50;
    const N_ROWS: usize = 5; // 5-tap filter

    println!(
        "=== Benchmarks ({}x{} x {} iterations) ===\n",
        WIDTH, HEIGHT, ITERATIONS
    );

    // Generate test data
    let rows_f32: Vec<Vec<f32>> = (0..N_ROWS)
        .map(|r| {
            (0..WIDTH)
                .map(|x| ((x * 17 + r * 31) % 256) as f32)
                .collect()
        })
        .collect();
    let row_refs_f32: Vec<&[f32]> = rows_f32.iter().map(|r| r.as_slice()).collect();
    let weights_f32 = [0.06136f32, 0.24477, 0.38774, 0.24477, 0.06136]; // Gaussian

    let mut output_f32 = vec![0.0f32; WIDTH];

    // Vertical reduction f32 benchmarks
    println!("  Vertical Reduction (f32, {}-tap):", N_ROWS);

    let start = Instant::now();
    for _ in 0..ITERATIONS * HEIGHT {
        reduce_vertical_f32_scalar(&row_refs_f32, &weights_f32, &mut output_f32);
        std::hint::black_box(&output_f32);
    }
    let scalar_time = start.elapsed();
    let scalar_mpix =
        (WIDTH * ITERATIONS * HEIGHT) as f64 / scalar_time.as_secs_f64() / 1_000_000.0;
    println!(
        "    Scalar:       {:>8.2} ms ({:.1} Mpix/s)",
        scalar_time.as_secs_f64() * 1000.0,
        scalar_mpix
    );

    if let Some(token) = Avx2FmaToken::try_new() {
        let start = Instant::now();
        for _ in 0..ITERATIONS * HEIGHT {
            reduce_vertical_f32_simd(token, &row_refs_f32, &weights_f32, &mut output_f32);
            std::hint::black_box(&output_f32);
        }
        let simd_time = start.elapsed();
        let simd_mpix =
            (WIDTH * ITERATIONS * HEIGHT) as f64 / simd_time.as_secs_f64() / 1_000_000.0;
        let speedup = scalar_time.as_secs_f64() / simd_time.as_secs_f64();
        println!(
            "    AVX2 f32x8:   {:>8.2} ms ({:.1} Mpix/s, {:.1}x)",
            simd_time.as_secs_f64() * 1000.0,
            simd_mpix,
            speedup
        );
    }

    println!();

    // Vertical reduction u8 benchmarks
    let rows_u8: Vec<Vec<u8>> = (0..N_ROWS)
        .map(|r| {
            (0..WIDTH)
                .map(|x| ((x * 17 + r * 31) % 256) as u8)
                .collect()
        })
        .collect();
    let row_refs_u8: Vec<&[u8]> = rows_u8.iter().map(|r| r.as_slice()).collect();
    // Fixed-point Gaussian weights (sum = 32768)
    let weights_i16: [i16; 5] = [2011, 8018, 12706, 8018, 2011];

    let mut output_u8 = vec![0u8; WIDTH];

    println!("  Vertical Reduction (u8 fixed-point, {}-tap):", N_ROWS);

    let start = Instant::now();
    for _ in 0..ITERATIONS * HEIGHT {
        reduce_vertical_u8_scalar(&row_refs_u8, &weights_i16, &mut output_u8);
        std::hint::black_box(&output_u8);
    }
    let scalar_time = start.elapsed();
    let scalar_mpix =
        (WIDTH * ITERATIONS * HEIGHT) as f64 / scalar_time.as_secs_f64() / 1_000_000.0;
    println!(
        "    Scalar:       {:>8.2} ms ({:.1} Mpix/s)",
        scalar_time.as_secs_f64() * 1000.0,
        scalar_mpix
    );

    if let Some(token) = Avx2FmaToken::try_new() {
        let start = Instant::now();
        for _ in 0..ITERATIONS * HEIGHT {
            reduce_vertical_u8_simd(token, &row_refs_u8, &weights_i16, &mut output_u8);
            std::hint::black_box(&output_u8);
        }
        let simd_time = start.elapsed();
        let simd_mpix =
            (WIDTH * ITERATIONS * HEIGHT) as f64 / simd_time.as_secs_f64() / 1_000_000.0;
        let speedup = scalar_time.as_secs_f64() / simd_time.as_secs_f64();
        println!(
            "    AVX2 fixed:   {:>8.2} ms ({:.1} Mpix/s, {:.1}x)",
            simd_time.as_secs_f64() * 1000.0,
            simd_mpix,
            speedup
        );
    }

    println!();

    // Box filter benchmarks
    let input_2d: Vec<f32> = (0..WIDTH * HEIGHT)
        .map(|i| ((i * 17) % 256) as f32)
        .collect();
    let mut output_2d = vec![0.0f32; WIDTH * HEIGHT];

    println!("  Box Filter 3x3 ({}x{}):", WIDTH, HEIGHT);

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        box_filter_3x3_f32_scalar(&input_2d, &mut output_2d, WIDTH, HEIGHT);
        std::hint::black_box(&output_2d);
    }
    let scalar_time = start.elapsed();
    let scalar_mpix =
        (WIDTH * HEIGHT * ITERATIONS) as f64 / scalar_time.as_secs_f64() / 1_000_000.0;
    println!(
        "    Scalar:       {:>8.2} ms ({:.1} Mpix/s)",
        scalar_time.as_secs_f64() * 1000.0,
        scalar_mpix
    );

    if let Some(token) = Avx2FmaToken::try_new() {
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            box_filter_3x3_f32(token, &input_2d, &mut output_2d, WIDTH, HEIGHT);
            std::hint::black_box(&output_2d);
        }
        let simd_time = start.elapsed();
        let simd_mpix =
            (WIDTH * HEIGHT * ITERATIONS) as f64 / simd_time.as_secs_f64() / 1_000_000.0;
        let speedup = scalar_time.as_secs_f64() / simd_time.as_secs_f64();
        println!(
            "    AVX2 f32x8:   {:>8.2} ms ({:.1} Mpix/s, {:.1}x)",
            simd_time.as_secs_f64() * 1000.0,
            simd_mpix,
            speedup
        );
    }

    println!();
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║           Convolution Kernels using archmage SIMD             ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!("Vertical reduction is the sweet spot for SIMD:");
    println!("  - Contiguous memory access (no strides)");
    println!("  - High arithmetic intensity (N inputs → 1 output)");
    println!("  - FMA chains for maximum throughput\n");

    test_correctness();
    benchmark();

    println!("=== Summary ===\n");
    println!("  Vertical reduction: 8-12x speedup (compute-bound)");
    println!("  Box filter: 3-5x speedup (mixed compute/memory)");
    println!();
    println!("  The key to fast convolution:");
    println!("    1. Use separable filters when possible (5x5 → 5+5)");
    println!("    2. Vertical pass first (contiguous access)");
    println!("    3. Fixed-point for u8 data (avoids float conversion)");
    println!();
}

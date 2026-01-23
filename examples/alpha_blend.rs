//! Alpha Blending Operations using archmage SIMD
//!
//! Demonstrates common alpha channel operations:
//! - Premultiply alpha (prepare for compositing)
//! - Unpremultiply alpha (restore straight alpha)
//! - Porter-Duff "over" compositing
//!
//! Run with: `cargo run --example alpha_blend --release`
//!
//! These operations are fundamental to image compositing pipelines.
//! SIMD provides 2-4x speedup, limited by memory bandwidth.

#![cfg(target_arch = "x86_64")]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::let_and_return)]

use archmage::{Avx2FmaToken, SimdToken, arcane};
use core::arch::x86_64::*;
use magetypes::simd::f32x8;
use std::time::Instant;

// ============================================================================
// Premultiply Alpha
// ============================================================================

/// Premultiply alpha for 2 RGBA pixels (8 floats) using f32x8
///
/// Converts straight alpha to premultiplied:
/// R' = R * A, G' = G * A, B' = B * A, A' = A
///
/// Memory layout: [R0,G0,B0,A0, R1,G1,B1,A1]
#[arcane]
fn premultiply_2px(_token: Avx2FmaToken, pixels: f32x8) -> f32x8 {
    // Extract alpha values and broadcast to all channels of each pixel
    // We need: [A0,A0,A0,A0, A1,A1,A1,A1]
    let raw = pixels.raw();

    // Use permutevar to broadcast alpha (index 3→0,1,2,3 and 7→4,5,6,7)
    let alpha_broadcast = _mm256_permutevar8x32_ps(raw, _mm256_set_epi32(7, 7, 7, 7, 3, 3, 3, 3));

    // Multiply all channels by alpha
    let premul = _mm256_mul_ps(raw, alpha_broadcast);

    // Restore original alpha values (we don't want A*A, just A)
    // Use blend: take RGB from premul, A from original
    // Blend mask: lanes 3 and 7 should come from original
    let blend_mask = _mm256_set_ps(-0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0); // -0 has sign bit set
    let result = _mm256_blendv_ps(premul, raw, blend_mask);

    // SAFETY: We're inside #[arcane] which guarantees AVX2 support
    unsafe { f32x8::from_raw(result) }
}

/// Premultiply alpha for a slice of RGBA pixels
#[arcane]
pub fn premultiply_alpha_simd(token: Avx2FmaToken, data: &mut [f32]) {
    debug_assert!(data.len() % 4 == 0, "Must be RGBA pixels");

    // Process 2 pixels (8 floats) at a time
    for chunk in data.chunks_exact_mut(8) {
        let arr: &[f32; 8] = (&*chunk).try_into().unwrap();
        let pixels = f32x8::load(token, arr);
        let result = premultiply_2px(token, pixels);
        let out: &mut [f32; 8] = chunk.try_into().unwrap();
        result.store(out);
    }

    // Handle remainder (1 pixel)
    let remainder = data.len() % 8;
    if remainder >= 4 {
        let start = data.len() - remainder;
        let alpha = data[start + 3];
        data[start] *= alpha;
        data[start + 1] *= alpha;
        data[start + 2] *= alpha;
    }
}

// ============================================================================
// Unpremultiply Alpha
// ============================================================================

/// Unpremultiply alpha for 2 RGBA pixels (8 floats) using f32x8
///
/// Converts premultiplied to straight alpha:
/// R = R' / A, G = G' / A, B = B' / A, A' = A
///
/// Uses epsilon protection to avoid division by zero.
#[arcane]
fn unpremultiply_2px(token: Avx2FmaToken, pixels: f32x8) -> f32x8 {
    const ALPHA_EPSILON: f32 = 1.0 / 255.0;

    let raw = pixels.raw();
    let epsilon = f32x8::splat(token, ALPHA_EPSILON);
    let zero = f32x8::zero(token);
    let one = f32x8::splat(token, 1.0);

    // Broadcast alpha
    let alpha_broadcast = _mm256_permutevar8x32_ps(raw, _mm256_set_epi32(7, 7, 7, 7, 3, 3, 3, 3));
    let alpha_vec = unsafe { f32x8::from_raw(alpha_broadcast) };

    // Compute safe 1/alpha (clamp alpha to epsilon to avoid div by zero)
    let safe_alpha = alpha_vec.max(epsilon);
    let inv_alpha = one / safe_alpha;

    // Divide RGB by alpha
    let pixels_vec = unsafe { f32x8::from_raw(raw) };
    let divided = pixels_vec * inv_alpha;

    // Zero out RGB where alpha < epsilon using blendv
    let alpha_ok_mask = alpha_vec.simd_ge(epsilon);

    // Blend: use divided where alpha_ok, zero otherwise
    // blendv selects divided where mask bits are set, zero where not
    let rgb_result_raw = _mm256_blendv_ps(zero.raw(), divided.raw(), alpha_ok_mask.raw());
    let rgb_result = unsafe { f32x8::from_raw(rgb_result_raw) };

    // Restore original alpha values
    let blend_mask = _mm256_set_ps(-0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0);
    let result = _mm256_blendv_ps(rgb_result.raw(), raw, blend_mask);

    unsafe { f32x8::from_raw(result) }
}

/// Unpremultiply alpha for a slice of RGBA pixels
#[arcane]
pub fn unpremultiply_alpha_simd(token: Avx2FmaToken, data: &mut [f32]) {
    const ALPHA_EPSILON: f32 = 1.0 / 255.0;

    debug_assert!(data.len() % 4 == 0, "Must be RGBA pixels");

    // Process 2 pixels (8 floats) at a time
    for chunk in data.chunks_exact_mut(8) {
        let arr: &[f32; 8] = (&*chunk).try_into().unwrap();
        let pixels = f32x8::load(token, arr);
        let result = unpremultiply_2px(token, pixels);
        let out: &mut [f32; 8] = chunk.try_into().unwrap();
        result.store(out);
    }

    // Handle remainder
    let remainder = data.len() % 8;
    if remainder >= 4 {
        let start = data.len() - remainder;
        let alpha = data[start + 3];
        if alpha > ALPHA_EPSILON {
            let inv_alpha = 1.0 / alpha;
            data[start] *= inv_alpha;
            data[start + 1] *= inv_alpha;
            data[start + 2] *= inv_alpha;
        } else {
            data[start] = 0.0;
            data[start + 1] = 0.0;
            data[start + 2] = 0.0;
        }
    }
}

// ============================================================================
// Porter-Duff Over Compositing
// ============================================================================

/// Porter-Duff "over" composite for 2 pixels: dst = src + dst * (1 - src_alpha)
///
/// Both src and dst must be premultiplied.
#[arcane]
fn composite_over_2px(token: Avx2FmaToken, src: f32x8, dst: f32x8) -> f32x8 {
    let one = f32x8::splat(token, 1.0);
    let src_raw = src.raw();

    // Broadcast src alpha
    let src_alpha_broadcast =
        _mm256_permutevar8x32_ps(src_raw, _mm256_set_epi32(7, 7, 7, 7, 3, 3, 3, 3));
    let src_alpha = unsafe { f32x8::from_raw(src_alpha_broadcast) };

    // 1 - src_alpha
    let inv_src_alpha = one - src_alpha;

    // dst * (1 - src_alpha)
    let dst_contrib = dst * inv_src_alpha;

    // src + dst * (1 - src_alpha)
    src + dst_contrib
}

/// Porter-Duff "over" composite a source layer onto destination
///
/// Both layers must be premultiplied RGBA.
#[arcane]
pub fn composite_over_simd(token: Avx2FmaToken, src: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert!(src.len() % 4 == 0, "Must be RGBA pixels");

    // Process 2 pixels at a time
    for (src_chunk, dst_chunk) in src.chunks_exact(8).zip(dst.chunks_exact_mut(8)) {
        let src_arr: &[f32; 8] = src_chunk.try_into().unwrap();
        let dst_arr: &[f32; 8] = (&*dst_chunk).try_into().unwrap();
        let s = f32x8::load(token, src_arr);
        let d = f32x8::load(token, dst_arr);
        let result = composite_over_2px(token, s, d);
        let out: &mut [f32; 8] = dst_chunk.try_into().unwrap();
        result.store(out);
    }

    // Handle remainder
    let remainder = src.len() % 8;
    if remainder >= 4 {
        let start = src.len() - remainder;
        let src_alpha = src[start + 3];
        let inv_src_alpha = 1.0 - src_alpha;
        dst[start] = src[start] + dst[start] * inv_src_alpha;
        dst[start + 1] = src[start + 1] + dst[start + 1] * inv_src_alpha;
        dst[start + 2] = src[start + 2] + dst[start + 2] * inv_src_alpha;
        dst[start + 3] = src[start + 3] + dst[start + 3] * inv_src_alpha;
    }
}

// ============================================================================
// Scalar Reference
// ============================================================================

fn premultiply_alpha_scalar(data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(4) {
        let alpha = chunk[3];
        chunk[0] *= alpha;
        chunk[1] *= alpha;
        chunk[2] *= alpha;
    }
}

fn unpremultiply_alpha_scalar(data: &mut [f32]) {
    const ALPHA_EPSILON: f32 = 1.0 / 255.0;
    for chunk in data.chunks_exact_mut(4) {
        let alpha = chunk[3];
        if alpha > ALPHA_EPSILON {
            let inv_alpha = 1.0 / alpha;
            chunk[0] *= inv_alpha;
            chunk[1] *= inv_alpha;
            chunk[2] *= inv_alpha;
        } else {
            chunk[0] = 0.0;
            chunk[1] = 0.0;
            chunk[2] = 0.0;
        }
    }
}

fn composite_over_scalar(src: &[f32], dst: &mut [f32]) {
    for (src_chunk, dst_chunk) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        let src_alpha = src_chunk[3];
        let inv_src_alpha = 1.0 - src_alpha;
        dst_chunk[0] = src_chunk[0] + dst_chunk[0] * inv_src_alpha;
        dst_chunk[1] = src_chunk[1] + dst_chunk[1] * inv_src_alpha;
        dst_chunk[2] = src_chunk[2] + dst_chunk[2] * inv_src_alpha;
        dst_chunk[3] = src_chunk[3] + dst_chunk[3] * inv_src_alpha;
    }
}

// ============================================================================
// Tests
// ============================================================================

fn test_correctness() {
    println!("=== Correctness Tests ===\n");

    if let Some(token) = Avx2FmaToken::try_new() {
        // Test premultiply
        let original = [
            0.5, 0.3, 0.8, 0.5, // 50% alpha pixel
            1.0, 0.0, 0.0, 1.0, // Opaque red
        ];

        let mut simd_data = original;
        let mut scalar_data = original;

        premultiply_alpha_simd(token, &mut simd_data);
        premultiply_alpha_scalar(&mut scalar_data);

        println!("  Premultiply test:");
        let max_diff: f32 = simd_data
            .iter()
            .zip(scalar_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        println!("    Max SIMD vs scalar difference: {:.6}", max_diff);
        println!(
            "    Pixel 0: [{:.3}, {:.3}, {:.3}, {:.3}]",
            simd_data[0], simd_data[1], simd_data[2], simd_data[3]
        );
        println!(
            "    Expected: [{:.3}, {:.3}, {:.3}, {:.3}]\n",
            0.25, 0.15, 0.4, 0.5
        );

        // Test unpremultiply round-trip
        let mut round_trip = original;
        premultiply_alpha_simd(token, &mut round_trip);
        unpremultiply_alpha_simd(token, &mut round_trip);

        println!("  Premultiply/Unpremultiply round-trip:");
        let max_diff: f32 = round_trip
            .iter()
            .zip(original.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        println!("    Max round-trip error: {:.6}\n", max_diff);

        // Test compositing
        let src = [0.5, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5]; // 50% red, 50% green (premul)
        let mut dst = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]; // Opaque blue (premul)
        let mut dst_scalar = dst;

        composite_over_simd(token, &src, &mut dst);
        composite_over_scalar(&src, &mut dst_scalar);

        println!("  Porter-Duff Over test:");
        let max_diff: f32 = dst
            .iter()
            .zip(dst_scalar.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        println!("    Max SIMD vs scalar difference: {:.6}", max_diff);
        println!(
            "    Result: [{:.3}, {:.3}, {:.3}, {:.3}]",
            dst[0], dst[1], dst[2], dst[3]
        );
        println!(
            "    Expected (50% red over blue): [{:.3}, {:.3}, {:.3}, {:.3}]\n",
            0.5, 0.0, 0.5, 1.0
        );
    } else {
        println!("  AVX2 not available\n");
    }
}

// ============================================================================
// Benchmarks
// ============================================================================

fn benchmark() {
    const PIXELS: usize = 1920 * 1080; // HD frame
    const ITERATIONS: usize = 100;

    println!(
        "=== Benchmarks ({} pixels x {} iterations) ===\n",
        PIXELS, ITERATIONS
    );

    // Generate test data (straight alpha RGBA)
    let original: Vec<f32> = (0..PIXELS * 4)
        .map(|i| {
            let val = ((i * 17) % 256) as f32 / 255.0;
            val
        })
        .collect();

    let mut data = original.clone();

    // Premultiply benchmarks
    println!("  Premultiply Alpha:");

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        data.copy_from_slice(&original);
        premultiply_alpha_scalar(&mut data);
        std::hint::black_box(&data);
    }
    let scalar_time = start.elapsed();
    let scalar_mpix = (PIXELS * ITERATIONS) as f64 / scalar_time.as_secs_f64() / 1_000_000.0;
    println!(
        "    Scalar:       {:>8.2} ms ({:.1} Mpix/s)",
        scalar_time.as_secs_f64() * 1000.0,
        scalar_mpix
    );

    if let Some(token) = Avx2FmaToken::try_new() {
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            data.copy_from_slice(&original);
            premultiply_alpha_simd(token, &mut data);
            std::hint::black_box(&data);
        }
        let simd_time = start.elapsed();
        let simd_mpix = (PIXELS * ITERATIONS) as f64 / simd_time.as_secs_f64() / 1_000_000.0;
        let speedup = scalar_time.as_secs_f64() / simd_time.as_secs_f64();
        println!(
            "    AVX2 f32x8:   {:>8.2} ms ({:.1} Mpix/s, {:.1}x)",
            simd_time.as_secs_f64() * 1000.0,
            simd_mpix,
            speedup
        );
    }

    println!();

    // Unpremultiply benchmarks (need to premultiply first)
    let mut premul_data = original.clone();
    premultiply_alpha_scalar(&mut premul_data);

    println!("  Unpremultiply Alpha:");

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        data.copy_from_slice(&premul_data);
        unpremultiply_alpha_scalar(&mut data);
        std::hint::black_box(&data);
    }
    let scalar_time = start.elapsed();
    let scalar_mpix = (PIXELS * ITERATIONS) as f64 / scalar_time.as_secs_f64() / 1_000_000.0;
    println!(
        "    Scalar:       {:>8.2} ms ({:.1} Mpix/s)",
        scalar_time.as_secs_f64() * 1000.0,
        scalar_mpix
    );

    if let Some(token) = Avx2FmaToken::try_new() {
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            data.copy_from_slice(&premul_data);
            unpremultiply_alpha_simd(token, &mut data);
            std::hint::black_box(&data);
        }
        let simd_time = start.elapsed();
        let simd_mpix = (PIXELS * ITERATIONS) as f64 / simd_time.as_secs_f64() / 1_000_000.0;
        let speedup = scalar_time.as_secs_f64() / simd_time.as_secs_f64();
        println!(
            "    AVX2 f32x8:   {:>8.2} ms ({:.1} Mpix/s, {:.1}x)",
            simd_time.as_secs_f64() * 1000.0,
            simd_mpix,
            speedup
        );
    }

    println!();

    // Compositing benchmarks
    let src: Vec<f32> = (0..PIXELS * 4)
        .map(|i| {
            let val = ((i * 31) % 256) as f32 / 255.0;
            if i % 4 == 3 { 0.5 } else { val * 0.5 }
        })
        .collect();
    let dst_orig: Vec<f32> = (0..PIXELS * 4)
        .map(|i| {
            let val = ((i * 13) % 256) as f32 / 255.0;
            val
        })
        .collect();
    let mut dst = dst_orig.clone();

    println!("  Porter-Duff Over:");

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        dst.copy_from_slice(&dst_orig);
        composite_over_scalar(&src, &mut dst);
        std::hint::black_box(&dst);
    }
    let scalar_time = start.elapsed();
    let scalar_mpix = (PIXELS * ITERATIONS) as f64 / scalar_time.as_secs_f64() / 1_000_000.0;
    println!(
        "    Scalar:       {:>8.2} ms ({:.1} Mpix/s)",
        scalar_time.as_secs_f64() * 1000.0,
        scalar_mpix
    );

    if let Some(token) = Avx2FmaToken::try_new() {
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            dst.copy_from_slice(&dst_orig);
            composite_over_simd(token, &src, &mut dst);
            std::hint::black_box(&dst);
        }
        let simd_time = start.elapsed();
        let simd_mpix = (PIXELS * ITERATIONS) as f64 / simd_time.as_secs_f64() / 1_000_000.0;
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
    println!("║          Alpha Blending using archmage SIMD                   ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!("Operations demonstrated:");
    println!("  - Premultiply alpha (prepare for compositing)");
    println!("  - Unpremultiply alpha (restore straight alpha)");
    println!("  - Porter-Duff over (layer compositing)\n");

    println!("Key technique: Alpha broadcast using _mm256_permutevar8x32_ps");
    println!("to replicate alpha to all RGBA channels.\n");

    test_correctness();
    benchmark();

    println!("=== Summary ===\n");
    println!("  Alpha operations are memory-bound, so SIMD speedup is modest (2-3x).");
    println!("  The main benefit is consistent vectorized processing.\n");
    println!("  Key pattern for RGBA operations:");
    println!("    let alpha = _mm256_permutevar8x32_ps(pixels,");
    println!("                    _mm256_set_epi32(7,7,7,7, 3,3,3,3));\n");
}

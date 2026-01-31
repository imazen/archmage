//! Color Space Conversion using archmage SIMD
//!
//! Demonstrates YUV↔RGB conversion with two approaches:
//! 1. High-level f32x8 API (clean, readable)
//! 2. Direct intrinsics for fixed-point (matching libwebp's 14-bit arithmetic)
//!
//! Run with: `cargo run --example color_convert --release`
//!
//! The fixed-point version matches libwebp/image-webp exactly for bit-perfect
//! decoding. The float version is simpler but may have minor rounding differences.

#![cfg(target_arch = "x86_64")]

use archmage::{SimdToken, X64V2Token, X64V3Token, arcane};
use core::arch::x86_64::*;
use magetypes::simd::f32x8;
use std::time::Instant;

// ============================================================================
// Floating-Point YUV→RGB (Clean API demonstration)
// ============================================================================

/// BT.601 YUV→RGB matrix coefficients (full range)
///
/// R = Y + 1.402 * (V - 128)
/// G = Y - 0.344136 * (U - 128) - 0.714136 * (V - 128)
/// B = Y + 1.772 * (U - 128)
mod bt601 {
    pub const KR_V: f32 = 1.402;
    pub const KG_U: f32 = -0.344136;
    pub const KG_V: f32 = -0.714136;
    pub const KB_U: f32 = 1.772;
    pub const OFFSET: f32 = 128.0;
}

/// Convert 8 YUV pixels to RGB using floating-point f32x8
///
/// Each input vector contains 8 values for Y, U, and V channels.
/// Output: 3 vectors for R, G, B channels.
#[arcane]
fn yuv_to_rgb_f32x8(token: X64V3Token, y: f32x8, u: f32x8, v: f32x8) -> (f32x8, f32x8, f32x8) {
    let offset = f32x8::splat(token, bt601::OFFSET);
    let zero = f32x8::splat(token, 0.0);
    let max = f32x8::splat(token, 255.0);

    // Shift U and V from [0,255] to [-128,127]
    let u_shifted = u - offset;
    let v_shifted = v - offset;

    // Load coefficients
    let kr_v = f32x8::splat(token, bt601::KR_V);
    let kg_u = f32x8::splat(token, bt601::KG_U);
    let kg_v = f32x8::splat(token, bt601::KG_V);
    let kb_u = f32x8::splat(token, bt601::KB_U);

    // R = Y + 1.402 * V'
    let r = y + v_shifted * kr_v;

    // G = Y - 0.344 * U' - 0.714 * V'
    let g = u_shifted.mul_add(kg_u, v_shifted.mul_add(kg_v, y));

    // B = Y + 1.772 * U'
    let b = y + u_shifted * kb_u;

    // Clamp to [0, 255]
    (r.clamp(zero, max), g.clamp(zero, max), b.clamp(zero, max))
}

/// Convert 8 RGB pixels to YUV using floating-point f32x8
///
/// BT.601 RGB→YUV:
/// Y = 0.299*R + 0.587*G + 0.114*B
/// U = -0.169*R - 0.331*G + 0.5*B + 128
/// V = 0.5*R - 0.419*G - 0.081*B + 128
#[arcane]
fn rgb_to_yuv_f32x8(token: X64V3Token, r: f32x8, g: f32x8, b: f32x8) -> (f32x8, f32x8, f32x8) {
    let offset = f32x8::splat(token, 128.0);
    let zero = f32x8::splat(token, 0.0);
    let max = f32x8::splat(token, 255.0);

    // Y coefficients
    let ky_r = f32x8::splat(token, 0.299);
    let ky_g = f32x8::splat(token, 0.587);
    let ky_b = f32x8::splat(token, 0.114);

    // U coefficients
    let ku_r = f32x8::splat(token, -0.168736);
    let ku_g = f32x8::splat(token, -0.331264);
    let ku_b = f32x8::splat(token, 0.5);

    // V coefficients
    let kv_r = f32x8::splat(token, 0.5);
    let kv_g = f32x8::splat(token, -0.418688);
    let kv_b = f32x8::splat(token, -0.081312);

    // Y = 0.299*R + 0.587*G + 0.114*B
    let y = r.mul_add(ky_r, g.mul_add(ky_g, b * ky_b));

    // U = -0.169*R - 0.331*G + 0.5*B + 128
    let u = r.mul_add(ku_r, g.mul_add(ku_g, b.mul_add(ku_b, offset)));

    // V = 0.5*R - 0.419*G - 0.081*B + 128
    let v = r.mul_add(kv_r, g.mul_add(kv_g, b.mul_add(kv_b, offset)));

    // Clamp to [0, 255]
    (y.clamp(zero, max), u.clamp(zero, max), v.clamp(zero, max))
}

// ============================================================================
// Fixed-Point YUV→RGB (libwebp-compatible, exact)
// ============================================================================

/// Fixed-point YUV to RGB conversion constants (14-bit, matching libwebp)
///
/// R = (19077 * y             + 26149 * v - 14234) >> 6
/// G = (19077 * y -  6419 * u - 13320 * v +  8708) >> 6
/// B = (19077 * y + 33050 * u             - 17685) >> 6
///
/// These are in the upper 8 bits of 16-bit words for efficient multiplication.
mod fixed_point {
    pub const K_Y: i16 = 19077;
    pub const K_V_R: i16 = 26149;
    pub const K_U_G: i16 = 6419;
    pub const K_V_G: i16 = 13320;
    pub const K_U_B: i16 = 33050u16 as i16; // Unsigned, but fits in epu16
    pub const OFFSET_R: i16 = 14234;
    pub const OFFSET_G: i16 = -8708i16;
    pub const OFFSET_B: i16 = 17685;
}

/// Load 8 bytes into upper 8 bits of 16-bit words (equivalent to << 8)
///
/// Used for fixed-point YUV conversion where we want maximum precision
/// from 8-bit inputs in 16-bit arithmetic.
#[arcane]
#[inline]
fn load_hi_16(token: X64V2Token, src: &[u8]) -> __m128i {
    let _ = token;
    debug_assert!(src.len() >= 8);
    let zero = _mm_setzero_si128();
    // Load 8 bytes into low 64 bits
    // SAFETY: src.len() >= 8 checked above, pointer is valid
    let data = unsafe { _mm_loadl_epi64(src.as_ptr() as *const __m128i) };
    // Interleave with zeros to get [0,y0,0,y1,0,y2,...] = y << 8
    _mm_unpacklo_epi8(zero, data)
}

/// Convert 8 YUV pixels to RGB using fixed-point arithmetic (SSE2)
///
/// This matches libwebp's exact output for bit-perfect decoding.
/// Input Y, U, V are 8-bit values. Output R, G, B are 8-bit clamped.
#[arcane]
fn yuv_to_rgb_fixed_8(
    token: X64V2Token,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    r_out: &mut [u8],
    g_out: &mut [u8],
    b_out: &mut [u8],
) {
    let _ = token;
    debug_assert!(y.len() >= 8 && u.len() >= 8 && v.len() >= 8);
    debug_assert!(r_out.len() >= 8 && g_out.len() >= 8 && b_out.len() >= 8);

    // Load Y, U, V into upper 8 bits of 16-bit words
    let y_hi = load_hi_16(token, y);
    let u_hi = load_hi_16(token, u);
    let v_hi = load_hi_16(token, v);

    // Load constants
    let k_y = _mm_set1_epi16(fixed_point::K_Y);
    let k_v_r = _mm_set1_epi16(fixed_point::K_V_R);
    let k_u_g = _mm_set1_epi16(fixed_point::K_U_G);
    let k_v_g = _mm_set1_epi16(fixed_point::K_V_G);
    let k_u_b = _mm_set1_epi16(fixed_point::K_U_B);
    let offset_r = _mm_set1_epi16(fixed_point::OFFSET_R);
    let offset_g = _mm_set1_epi16(fixed_point::OFFSET_G);
    let offset_b = _mm_set1_epi16(fixed_point::OFFSET_B);

    // Y contribution (same for all channels)
    // mulhi_epu16 gives upper 16 bits of unsigned 16x16→32 multiply
    let y1 = _mm_mulhi_epu16(y_hi, k_y);

    // R = Y1 + V*k_v_r - offset_r
    let r0 = _mm_mulhi_epu16(v_hi, k_v_r);
    let r1 = _mm_sub_epi16(y1, offset_r);
    let r2 = _mm_add_epi16(r1, r0);

    // G = Y1 - U*k_u_g - V*k_v_g - offset_g (note offset_g is negative)
    let g0 = _mm_mulhi_epu16(u_hi, k_u_g);
    let g1 = _mm_mulhi_epu16(v_hi, k_v_g);
    let g2 = _mm_sub_epi16(y1, offset_g); // Subtract negative = add
    let g3 = _mm_add_epi16(g0, g1);
    let g4 = _mm_sub_epi16(g2, g3);

    // B = Y1 + U*k_u_b - offset_b (using unsigned saturation for large coefficient)
    let b0 = _mm_mulhi_epu16(u_hi, k_u_b);
    let b1 = _mm_adds_epu16(b0, y1);
    let b2 = _mm_subs_epu16(b1, offset_b);

    // Final shift by 6 and clamp
    // R and G can be negative, use arithmetic shift
    // B is always positive (due to unsigned ops), use logical shift
    let r_final = _mm_srai_epi16(r2, 6);
    let g_final = _mm_srai_epi16(g4, 6);
    let b_final = _mm_srli_epi16(b2, 6);

    // Pack to 8-bit with saturation
    let r_packed = _mm_packus_epi16(r_final, _mm_setzero_si128());
    let g_packed = _mm_packus_epi16(g_final, _mm_setzero_si128());
    let b_packed = _mm_packus_epi16(b_final, _mm_setzero_si128());

    // Store results (lower 8 bytes only)
    // SAFETY: Output slices have len >= 8 as checked by debug_assert above
    unsafe {
        _mm_storel_epi64(r_out.as_mut_ptr() as *mut __m128i, r_packed);
        _mm_storel_epi64(g_out.as_mut_ptr() as *mut __m128i, g_packed);
        _mm_storel_epi64(b_out.as_mut_ptr() as *mut __m128i, b_packed);
    }
}

// ============================================================================
// Scalar Reference
// ============================================================================

fn yuv_to_rgb_scalar(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    let y = y as f32;
    let u = u as f32 - 128.0;
    let v = v as f32 - 128.0;

    let r = y + 1.402 * v;
    let g = y - 0.344136 * u - 0.714136 * v;
    let b = y + 1.772 * u;

    (
        r.clamp(0.0, 255.0) as u8,
        g.clamp(0.0, 255.0) as u8,
        b.clamp(0.0, 255.0) as u8,
    )
}

/// Fixed-point scalar (matches libwebp exactly)
fn yuv_to_rgb_fixed_scalar(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    fn mulhi(val: u8, coeff: u16) -> i32 {
        ((u32::from(val) * u32::from(coeff)) >> 8) as i32
    }
    fn clip(v: i32) -> u8 {
        (v >> 6).clamp(0, 255) as u8
    }
    let r = clip(mulhi(y, 19077) + mulhi(v, 26149) - 14234);
    let g = clip(mulhi(y, 19077) - mulhi(u, 6419) - mulhi(v, 13320) + 8708);
    let b = clip(mulhi(y, 19077) + mulhi(u, 33050) - 17685);
    (r, g, b)
}

#[allow(dead_code)]
fn rgb_to_yuv_scalar(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = r as f32;
    let g = g as f32;
    let b = b as f32;

    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let u = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
    let v = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;

    (
        y.clamp(0.0, 255.0) as u8,
        u.clamp(0.0, 255.0) as u8,
        v.clamp(0.0, 255.0) as u8,
    )
}

// ============================================================================
// Tests
// ============================================================================

fn test_correctness() {
    println!("=== Correctness Tests ===\n");

    // Test YUV→RGB float vs scalar
    if let Some(token) = X64V3Token::try_new() {
        let y_vals = [16.0, 128.0, 235.0, 100.0, 200.0, 50.0, 180.0, 80.0];
        let u_vals = [128.0, 128.0, 128.0, 90.0, 180.0, 200.0, 60.0, 150.0];
        let v_vals = [128.0, 128.0, 128.0, 200.0, 60.0, 100.0, 180.0, 80.0];

        let y = f32x8::from_array(token, y_vals);
        let u = f32x8::from_array(token, u_vals);
        let v = f32x8::from_array(token, v_vals);

        let (r_simd, g_simd, b_simd) = yuv_to_rgb_f32x8(token, y, u, v);
        let r_arr = r_simd.to_array();
        let g_arr = g_simd.to_array();
        let b_arr = b_simd.to_array();

        println!("  Float YUV→RGB comparison:");
        let mut max_diff = 0.0f32;
        for i in 0..8 {
            let (r_s, g_s, b_s) =
                yuv_to_rgb_scalar(y_vals[i] as u8, u_vals[i] as u8, v_vals[i] as u8);
            let diff_r = (r_arr[i] - r_s as f32).abs();
            let diff_g = (g_arr[i] - g_s as f32).abs();
            let diff_b = (b_arr[i] - b_s as f32).abs();
            max_diff = max_diff.max(diff_r).max(diff_g).max(diff_b);
        }
        println!("    Max difference from scalar: {:.3}", max_diff);
        println!("    (Expected: <1.0 due to rounding)\n");
    }

    // Test fixed-point vs scalar (should match exactly)
    if let Some(token) = X64V2Token::try_new() {
        let y = [16, 128, 235, 100, 200, 50, 180, 80];
        let u = [128, 128, 128, 90, 180, 200, 60, 150];
        let v = [128, 128, 128, 200, 60, 100, 180, 80];

        let mut r_simd = [0u8; 8];
        let mut g_simd = [0u8; 8];
        let mut b_simd = [0u8; 8];

        yuv_to_rgb_fixed_8(token, &y, &u, &v, &mut r_simd, &mut g_simd, &mut b_simd);

        println!("  Fixed-point YUV→RGB comparison:");
        let mut matches = true;
        for i in 0..8 {
            let (r_s, g_s, b_s) = yuv_to_rgb_fixed_scalar(y[i], u[i], v[i]);
            if r_simd[i] != r_s || g_simd[i] != g_s || b_simd[i] != b_s {
                println!(
                    "    Mismatch at {}: SIMD=({},{},{}) scalar=({},{},{})",
                    i, r_simd[i], g_simd[i], b_simd[i], r_s, g_s, b_s
                );
                matches = false;
            }
        }
        if matches {
            println!("    All 8 pixels match exactly!\n");
        }
    }

    // Test round-trip RGB→YUV→RGB
    if let Some(token) = X64V3Token::try_new() {
        let r_orig = [255.0, 0.0, 0.0, 128.0, 64.0, 192.0, 100.0, 200.0];
        let g_orig = [0.0, 255.0, 0.0, 128.0, 128.0, 64.0, 150.0, 100.0];
        let b_orig = [0.0, 0.0, 255.0, 128.0, 192.0, 128.0, 50.0, 150.0];

        let r = f32x8::from_array(token, r_orig);
        let g = f32x8::from_array(token, g_orig);
        let b = f32x8::from_array(token, b_orig);

        let (y, u, v) = rgb_to_yuv_f32x8(token, r, g, b);
        let (r_back, g_back, b_back) = yuv_to_rgb_f32x8(token, y, u, v);

        let r_arr = r_back.to_array();
        let g_arr = g_back.to_array();
        let b_arr = b_back.to_array();

        println!("  Round-trip RGB→YUV→RGB:");
        let mut max_diff = 0.0f32;
        for i in 0..8 {
            let diff_r = (r_arr[i] - r_orig[i]).abs();
            let diff_g = (g_arr[i] - g_orig[i]).abs();
            let diff_b = (b_arr[i] - b_orig[i]).abs();
            max_diff = max_diff.max(diff_r).max(diff_g).max(diff_b);
        }
        println!("    Max round-trip error: {:.3}", max_diff);
        println!("    (Expected: <2.0 due to chroma subsampling approx)\n");
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

    // Generate test data
    let y_data: Vec<f32> = (0..PIXELS).map(|i| ((i * 7) % 256) as f32).collect();
    let u_data: Vec<f32> = (0..PIXELS)
        .map(|i| (((i * 11) + 64) % 256) as f32)
        .collect();
    let v_data: Vec<f32> = (0..PIXELS)
        .map(|i| (((i * 13) + 128) % 256) as f32)
        .collect();

    let mut r_out = vec![0.0f32; PIXELS];
    let mut g_out = vec![0.0f32; PIXELS];
    let mut b_out = vec![0.0f32; PIXELS];

    // Scalar baseline
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        for i in 0..PIXELS {
            let (r, g, b) = yuv_to_rgb_scalar(y_data[i] as u8, u_data[i] as u8, v_data[i] as u8);
            r_out[i] = r as f32;
            g_out[i] = g as f32;
            b_out[i] = b as f32;
        }
        std::hint::black_box(&r_out);
    }
    let scalar_time = start.elapsed();
    let scalar_mpix_sec = (PIXELS * ITERATIONS) as f64 / scalar_time.as_secs_f64() / 1_000_000.0;
    println!(
        "  Scalar:         {:>8.2} ms ({:.1} Mpix/s)",
        scalar_time.as_secs_f64() * 1000.0,
        scalar_mpix_sec
    );

    // SIMD f32x8 version
    if let Some(token) = X64V3Token::try_new() {
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            for chunk_start in (0..PIXELS).step_by(8) {
                if chunk_start + 8 <= PIXELS {
                    let y = f32x8::load(
                        token,
                        (&y_data[chunk_start..chunk_start + 8]).try_into().unwrap(),
                    );
                    let u = f32x8::load(
                        token,
                        (&u_data[chunk_start..chunk_start + 8]).try_into().unwrap(),
                    );
                    let v = f32x8::load(
                        token,
                        (&v_data[chunk_start..chunk_start + 8]).try_into().unwrap(),
                    );

                    let (r, g, b) = yuv_to_rgb_f32x8(token, y, u, v);

                    r.store(
                        (&mut r_out[chunk_start..chunk_start + 8])
                            .try_into()
                            .unwrap(),
                    );
                    g.store(
                        (&mut g_out[chunk_start..chunk_start + 8])
                            .try_into()
                            .unwrap(),
                    );
                    b.store(
                        (&mut b_out[chunk_start..chunk_start + 8])
                            .try_into()
                            .unwrap(),
                    );
                }
            }
            std::hint::black_box(&r_out);
        }
        let simd_time = start.elapsed();
        let simd_mpix_sec = (PIXELS * ITERATIONS) as f64 / simd_time.as_secs_f64() / 1_000_000.0;
        let speedup = scalar_time.as_secs_f64() / simd_time.as_secs_f64();
        println!(
            "  AVX2 f32x8:     {:>8.2} ms ({:.1} Mpix/s, {:.1}x)",
            simd_time.as_secs_f64() * 1000.0,
            simd_mpix_sec,
            speedup
        );
    }

    // Fixed-point SSE2 version
    if let Some(token) = X64V2Token::try_new() {
        let y_u8: Vec<u8> = y_data.iter().map(|&v| v as u8).collect();
        let u_u8: Vec<u8> = u_data.iter().map(|&v| v as u8).collect();
        let v_u8: Vec<u8> = v_data.iter().map(|&v| v as u8).collect();
        let mut r_u8 = vec![0u8; PIXELS];
        let mut g_u8 = vec![0u8; PIXELS];
        let mut b_u8 = vec![0u8; PIXELS];

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            for chunk_start in (0..PIXELS).step_by(8) {
                if chunk_start + 8 <= PIXELS {
                    yuv_to_rgb_fixed_8(
                        token,
                        &y_u8[chunk_start..],
                        &u_u8[chunk_start..],
                        &v_u8[chunk_start..],
                        &mut r_u8[chunk_start..],
                        &mut g_u8[chunk_start..],
                        &mut b_u8[chunk_start..],
                    );
                }
            }
            std::hint::black_box(&r_u8);
        }
        let fixed_time = start.elapsed();
        let fixed_mpix_sec = (PIXELS * ITERATIONS) as f64 / fixed_time.as_secs_f64() / 1_000_000.0;
        let speedup = scalar_time.as_secs_f64() / fixed_time.as_secs_f64();
        println!(
            "  SSE2 fixed-pt:  {:>8.2} ms ({:.1} Mpix/s, {:.1}x)",
            fixed_time.as_secs_f64() * 1000.0,
            fixed_mpix_sec,
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
    println!("║        Color Space Conversion using archmage SIMD             ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!("Two approaches demonstrated:");
    println!("  1. Float f32x8 - Clean API, easy to read, uses FMA");
    println!("  2. Fixed-point SSE2 - Matches libwebp exactly for decoders\n");

    test_correctness();
    benchmark();

    println!("=== Summary ===\n");
    println!("  The f32x8 API provides clean, readable SIMD code:");
    println!();
    println!("    let (r, g, b) = yuv_to_rgb_f32x8(token, y, u, v);");
    println!();
    println!("  For bit-exact codec compatibility, use direct intrinsics");
    println!("  inside #[arcane] functions with the appropriate token.\n");
}

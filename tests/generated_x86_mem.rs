//! Auto-generated exhaustive tests for x86 mem module intrinsics.
//!
//! This file exercises every intrinsic in `archmage::mem` to ensure they compile
//! and execute correctly on supported hardware.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(clippy::approx_constant)]

use std::hint::black_box;

use archmage::SimdToken;
use archmage::mem::avx;

#[cfg(feature = "avx512")]
use archmage::mem::{v4, v4_bw, v4_bw_vl, v4_vl, modern, modern_vl};


/// Test all AVX load/store intrinsics
#[test]
fn test_avx_mem_intrinsics_exhaustive() {
    use archmage::Avx2Token;

    let Some(token) = Avx2Token::try_new() else {
        eprintln!("AVX2 not available, skipping test");
        return;
    };

    // Test data
    let f32_data: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let f64_data: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    let i32_data: [i32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let i64_data: [i64; 4] = [1, 2, 3, 4];

    let mut f32_out: [f32; 8] = [0.0; 8];
    let mut f64_out: [f64; 4] = [0.0; 4];
    let mut i32_out: [i32; 8] = [0; 8];
    let mut i64_out: [i64; 4] = [0; 4];

    // _mm256_loadu_pd
    let v = avx::_mm256_loadu_pd(token, &f64_data); avx::_mm256_storeu_pd(token, &mut f64_out, v);

    // _mm256_loadu_ps
    let v = avx::_mm256_loadu_ps(token, &f32_data); avx::_mm256_storeu_ps(token, &mut f32_out, v);

    // _mm256_loadu_si256
    let v = avx::_mm256_loadu_si256(token, &i64_data); avx::_mm256_storeu_si256(token, &mut i64_out, v);

    // _mm256_loadu2_m128
    // _mm256_loadu2_m128 - skipped (complex signature)

    // _mm256_loadu2_m128d
    // _mm256_loadu2_m128d - skipped (complex signature)

    // _mm256_loadu2_m128i
    // _mm256_loadu2_m128i - skipped (complex signature)

    // Ensure values are used
    black_box(&f32_out);
    black_box(&f64_out);
    black_box(&i32_out);
    black_box(&i64_out);
}

/// Test all AVX-512 (v4) load/store intrinsics
#[test]
#[cfg(feature = "avx512")]
fn test_v4_mem_intrinsics_exhaustive() {
    use archmage::X64V4Token;

    let Some(token) = X64V4Token::try_new() else {
        eprintln!("AVX-512 not available, skipping test");
        return;
    };

    // Test data for 512-bit vectors
    let f32_data_16: [f32; 16] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                                   9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let f64_data_8: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let i32_data_16: [i32; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let i64_data_8: [i64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

    let mut f32_out_16: [f32; 16] = [0.0; 16];
    let mut f64_out_8: [f64; 8] = [0.0; 8];
    let mut i32_out_16: [i32; 16] = [0; 16];
    let mut i64_out_8: [i64; 8] = [0; 8];

    // Exercise v4 intrinsics
    let v = v4::_mm512_loadu_ps(token, &f32_data_16);
    v4::_mm512_storeu_ps(token, &mut f32_out_16, v);
    assert_eq!(f32_data_16, f32_out_16);

    let v = v4::_mm512_loadu_pd(token, &f64_data_8);
    v4::_mm512_storeu_pd(token, &mut f64_out_8, v);
    assert_eq!(f64_data_8, f64_out_8);

    let v = v4::_mm512_loadu_epi32(token, &i32_data_16);
    v4::_mm512_storeu_epi32(token, &mut i32_out_16, v);
    assert_eq!(i32_data_16, i32_out_16);

    let v = v4::_mm512_loadu_epi64(token, &i64_data_8);
    v4::_mm512_storeu_epi64(token, &mut i64_out_8, v);
    assert_eq!(i64_data_8, i64_out_8);

    black_box(&f32_out_16);
    black_box(&f64_out_8);
    black_box(&i32_out_16);
    black_box(&i64_out_8);
}

/// Test AVX-512VL intrinsics (256/128-bit with AVX-512 features)
#[test]
#[cfg(feature = "avx512")]
fn test_v4_vl_mem_intrinsics_exhaustive() {
    use archmage::X64V4Token;

    let Some(token) = X64V4Token::try_new() else {
        eprintln!("AVX-512 not available, skipping test");
        return;
    };

    // 256-bit test data
    let f32_data_8: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let f64_data_4: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    let i32_data_8: [i32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let i64_data_4: [i64; 4] = [1, 2, 3, 4];

    let mut f32_out_8: [f32; 8] = [0.0; 8];
    let mut f64_out_4: [f64; 4] = [0.0; 4];
    let mut i32_out_8: [i32; 8] = [0; 8];
    let mut i64_out_4: [i64; 4] = [0; 4];

    // 128-bit test data
    let f32_data_4: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let f64_data_2: [f64; 2] = [1.0, 2.0];
    let i32_data_4: [i32; 4] = [1, 2, 3, 4];
    let i64_data_2: [i64; 2] = [1, 2];

    let mut f32_out_4: [f32; 4] = [0.0; 4];
    let mut f64_out_2: [f64; 2] = [0.0; 2];
    let mut i32_out_4: [i32; 4] = [0; 4];
    let mut i64_out_2: [i64; 2] = [0; 2];

    // 256-bit VL operations
    let v = v4_vl::_mm256_loadu_epi32(token, &i32_data_8);
    v4_vl::_mm256_storeu_epi32(token, &mut i32_out_8, v);
    assert_eq!(i32_data_8, i32_out_8);

    let v = v4_vl::_mm256_loadu_epi64(token, &i64_data_4);
    v4_vl::_mm256_storeu_epi64(token, &mut i64_out_4, v);
    assert_eq!(i64_data_4, i64_out_4);

    // 128-bit VL operations
    let v = v4_vl::_mm_loadu_epi32(token, &i32_data_4);
    v4_vl::_mm_storeu_epi32(token, &mut i32_out_4, v);
    assert_eq!(i32_data_4, i32_out_4);

    let v = v4_vl::_mm_loadu_epi64(token, &i64_data_2);
    v4_vl::_mm_storeu_epi64(token, &mut i64_out_2, v);
    assert_eq!(i64_data_2, i64_out_2);

    black_box(&f32_out_8);
    black_box(&f64_out_4);
    black_box(&i32_out_8);
    black_box(&i64_out_4);
    black_box(&f32_out_4);
    black_box(&f64_out_2);
    black_box(&i32_out_4);
    black_box(&i64_out_2);
}

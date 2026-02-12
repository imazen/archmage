//! Tests proving safe memory operations are correct and match unsafe equivalents.
//!
//! Every test compares safe_unaligned_simd output against core::arch output
//! bit-for-bit to verify zero semantic difference.

#![cfg(all(target_arch = "x86_64", feature = "safe_unaligned_simd"))]

use archmage::{Desktop64, SimdToken, X64V2Token, arcane, rite};
use std::arch::x86_64::*;

// =============================================================================
// Correctness: safe == unsafe, bit-for-bit
// =============================================================================

#[arcane]
fn safe_loadu_ps_256(_token: Desktop64, data: &[f32; 8]) -> __m256 {
    safe_unaligned_simd::x86_64::_mm256_loadu_ps(data)
}

#[arcane]
fn safe_storeu_ps_256(_token: Desktop64, v: __m256, out: &mut [f32; 8]) {
    safe_unaligned_simd::x86_64::_mm256_storeu_ps(out, v)
}

#[test]
fn safe_loadu_ps_256_matches_unsafe() {
    if let Some(token) = Desktop64::summon() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let safe_v = safe_loadu_ps_256(token, &data);
        let unsafe_v = unsafe { _mm256_loadu_ps(data.as_ptr()) };

        let mut safe_out = [0.0f32; 8];
        let mut unsafe_out = [0.0f32; 8];
        unsafe {
            _mm256_storeu_ps(safe_out.as_mut_ptr(), safe_v);
            _mm256_storeu_ps(unsafe_out.as_mut_ptr(), unsafe_v);
        }
        assert_eq!(safe_out, unsafe_out);
    }
}

#[test]
fn safe_storeu_ps_256_matches_unsafe() {
    if let Some(token) = Desktop64::summon() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v = safe_loadu_ps_256(token, &data);

        let mut safe_out = [0.0f32; 8];
        safe_storeu_ps_256(token, v, &mut safe_out);

        let mut unsafe_out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(unsafe_out.as_mut_ptr(), v) };

        assert_eq!(safe_out, unsafe_out);
    }
}

#[arcane]
fn safe_loadu_si256(_token: Desktop64, data: &[u8; 32]) -> __m256i {
    safe_unaligned_simd::x86_64::_mm256_loadu_si256(data)
}

#[test]
fn safe_loadu_si256_matches_unsafe() {
    if let Some(token) = Desktop64::summon() {
        let data: [u8; 32] = core::array::from_fn(|i| i as u8);

        let safe_v = safe_loadu_si256(token, &data);
        let unsafe_v = unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) };

        let mut safe_out = [0u8; 32];
        let mut unsafe_out = [0u8; 32];
        unsafe {
            _mm256_storeu_si256(safe_out.as_mut_ptr() as *mut __m256i, safe_v);
            _mm256_storeu_si256(unsafe_out.as_mut_ptr() as *mut __m256i, unsafe_v);
        }
        assert_eq!(safe_out, unsafe_out);
    }
}

#[arcane]
fn safe_loadu_ps_128(_token: X64V2Token, data: &[f32; 4]) -> __m128 {
    safe_unaligned_simd::x86_64::_mm_loadu_ps(data)
}

#[test]
fn safe_loadu_ps_128_matches_unsafe() {
    if let Some(token) = X64V2Token::summon() {
        let data = [1.0f32, 2.0, 3.0, 4.0];

        let safe_v = safe_loadu_ps_128(token, &data);
        let unsafe_v = unsafe { _mm_loadu_ps(data.as_ptr()) };

        let mut safe_out = [0.0f32; 4];
        let mut unsafe_out = [0.0f32; 4];
        unsafe {
            _mm_storeu_ps(safe_out.as_mut_ptr(), safe_v);
            _mm_storeu_ps(unsafe_out.as_mut_ptr(), unsafe_v);
        }
        assert_eq!(safe_out, unsafe_out);
    }
}

#[arcane]
fn safe_loadu_pd_128(_token: X64V2Token, data: &[f64; 2]) -> __m128d {
    safe_unaligned_simd::x86_64::_mm_loadu_pd(data)
}

#[test]
fn safe_loadu_pd_128_matches_unsafe() {
    if let Some(token) = X64V2Token::summon() {
        let data = [1.0f64, 2.0];

        let safe_v = safe_loadu_pd_128(token, &data);
        let unsafe_v = unsafe { _mm_loadu_pd(data.as_ptr()) };

        let mut safe_out = [0.0f64; 2];
        let mut unsafe_out = [0.0f64; 2];
        unsafe {
            _mm_storeu_pd(safe_out.as_mut_ptr(), safe_v);
            _mm_storeu_pd(unsafe_out.as_mut_ptr(), unsafe_v);
        }
        assert_eq!(safe_out, unsafe_out);
    }
}

// =============================================================================
// Zero unsafe: complete #[arcane] functions with no unsafe blocks
// =============================================================================

/// A real FMA algorithm with ZERO unsafe blocks.
/// Loads, computes, stores — all safe.
#[arcane]
fn fma_no_unsafe(_token: Desktop64, a: &[f32; 8], b: &[f32; 8], c: &[f32; 8]) -> [f32; 8] {
    let va = safe_unaligned_simd::x86_64::_mm256_loadu_ps(a);
    let vb = safe_unaligned_simd::x86_64::_mm256_loadu_ps(b);
    let vc = safe_unaligned_simd::x86_64::_mm256_loadu_ps(c);
    let result = _mm256_fmadd_ps(va, vb, vc);
    let mut out = [0.0f32; 8];
    safe_unaligned_simd::x86_64::_mm256_storeu_ps(&mut out, result);
    out
}

#[test]
fn arcane_function_with_zero_unsafe_blocks() {
    if let Some(token) = Desktop64::summon() {
        let a = [2.0f32; 8];
        let b = [3.0f32; 8];
        let c = [1.0f32; 8];
        let result = fma_no_unsafe(token, &a, &b, &c);
        // 2*3 + 1 = 7
        assert_eq!(result, [7.0f32; 8]);
    }
}

/// Inner #[rite] helper also with zero unsafe.
#[rite]
fn square_rite(_token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
    let v = safe_unaligned_simd::x86_64::_mm256_loadu_ps(data);
    let sq = _mm256_mul_ps(v, v);
    let mut out = [0.0f32; 8];
    safe_unaligned_simd::x86_64::_mm256_storeu_ps(&mut out, sq);
    out
}

#[arcane]
fn square_outer(token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
    square_rite(token, data)
}

#[test]
fn rite_helper_with_zero_unsafe() {
    if let Some(token) = Desktop64::summon() {
        let data = [3.0f32; 8];
        let result = square_outer(token, &data);
        assert_eq!(result, [9.0f32; 8]);
    }
}

// =============================================================================
// Loop correctness
// =============================================================================

#[arcane]
fn process_loop(_token: Desktop64, chunks: &[[f32; 8]]) -> Vec<f32> {
    let mut results = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let v = safe_unaligned_simd::x86_64::_mm256_loadu_ps(chunk);
        let squared = _mm256_mul_ps(v, v);
        // Horizontal sum via shuffle+add
        let hi = _mm256_extractf128_ps::<1>(squared);
        let lo = _mm256_castps256_ps128(squared);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let sum_scalar = _mm_add_ss(sums, shuf2);
        results.push(_mm_cvtss_f32(sum_scalar));
    }
    results
}

#[test]
fn safe_load_in_processing_loop() {
    if let Some(token) = Desktop64::summon() {
        let chunks: Vec<[f32; 8]> = (0..1000)
            .map(|i| {
                let v = i as f32;
                [v; 8]
            })
            .collect();

        let results = process_loop(token, &chunks);
        assert_eq!(results.len(), 1000);

        for (i, &result) in results.iter().enumerate() {
            let v = i as f32;
            let expected = v * v * 8.0; // sum of 8 copies of v^2
            assert!(
                (result - expected).abs() < 1e-3 * expected.abs().max(1.0),
                "chunk {}: got {}, expected {}",
                i,
                result,
                expected
            );
        }
    }
}

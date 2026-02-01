//! Dot Product Operations
//!
//! SIMD-accelerated dot product using FMA instructions.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::simd_fn;
use crate::tokens::X64V3Token;

/// Compute dot product of two f32 slices using AVX2+FMA.
///
/// Uses fused multiply-add for maximum performance and accuracy.
///
/// # Example
///
/// ```rust,ignore
/// use archmage::{X64V3Token, SimdToken, composite::dot_product_f32};
///
/// if let Some(token) = X64V3Token::try_new() {
///     let a = vec![1.0f32; 1024];
///     let b = vec![2.0f32; 1024];
///     let result = dot_product_f32(token, &a, &b);
///     assert!((result - 2048.0).abs() < 0.001);
/// }
/// ```
#[simd_fn]
#[inline]
pub fn dot_product_f32(_token: X64V3Token, a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Slice lengths must match");

    // Accumulate in 4 vectors to hide FMA latency
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    let chunks = a.len() / 32;
    for i in 0..chunks {
        let base = i * 32;

        let a0 = unsafe { _mm256_loadu_ps(a.as_ptr().add(base)) };
        let b0 = unsafe { _mm256_loadu_ps(b.as_ptr().add(base)) };
        sum0 = _mm256_fmadd_ps(a0, b0, sum0);

        let a1 = unsafe { _mm256_loadu_ps(a.as_ptr().add(base + 8)) };
        let b1 = unsafe { _mm256_loadu_ps(b.as_ptr().add(base + 8)) };
        sum1 = _mm256_fmadd_ps(a1, b1, sum1);

        let a2 = unsafe { _mm256_loadu_ps(a.as_ptr().add(base + 16)) };
        let b2 = unsafe { _mm256_loadu_ps(b.as_ptr().add(base + 16)) };
        sum2 = _mm256_fmadd_ps(a2, b2, sum2);

        let a3 = unsafe { _mm256_loadu_ps(a.as_ptr().add(base + 24)) };
        let b3 = unsafe { _mm256_loadu_ps(b.as_ptr().add(base + 24)) };
        sum3 = _mm256_fmadd_ps(a3, b3, sum3);
    }

    // Combine accumulators
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);

    // Handle remaining chunks of 8
    let remaining_start = chunks * 32;
    let remaining_chunks = (a.len() - remaining_start) / 8;
    for i in 0..remaining_chunks {
        let base = remaining_start + i * 8;
        let av = unsafe { _mm256_loadu_ps(a.as_ptr().add(base)) };
        let bv = unsafe { _mm256_loadu_ps(b.as_ptr().add(base)) };
        sum0 = _mm256_fmadd_ps(av, bv, sum0);
    }

    // Horizontal sum
    let arr: [f32; 8] = unsafe { core::mem::transmute(sum0) };
    let mut result: f32 = arr.iter().sum();

    // Handle remaining elements
    let scalar_start = remaining_start + remaining_chunks * 8;
    for i in scalar_start..a.len() {
        result += a[i] * b[i];
    }

    result
}

/// Compute squared L2 norm (sum of squares) using AVX2+FMA.
#[inline]
pub fn norm_squared_f32(token: X64V3Token, a: &[f32]) -> f32 {
    dot_product_f32(token, a, a)
}

/// Compute L2 norm using AVX2+FMA.
///
/// Requires `std` feature (uses `f32::sqrt`).
#[cfg(feature = "std")]
#[inline]
pub fn norm_f32(token: X64V3Token, a: &[f32]) -> f32 {
    norm_squared_f32(token, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::SimdToken;

    #[test]
    fn test_dot_product() {
        if let Some(token) = X64V3Token::try_new() {
            let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
            let b: Vec<f32> = vec![1.0; 64];

            let result = dot_product_f32(token, &a, &b);
            let expected: f32 = (0..64).map(|i| i as f32).sum();

            assert!(
                (result - expected).abs() < 0.001,
                "Expected {}, got {}",
                expected,
                result
            );
        }
    }

    #[test]
    fn test_dot_product_small() {
        if let Some(token) = X64V3Token::try_new() {
            // Test with non-multiple of 8
            let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
            let b = vec![2.0f32, 2.0, 2.0, 2.0, 2.0];

            let result = dot_product_f32(token, &a, &b);
            let expected = 2.0 + 4.0 + 6.0 + 8.0 + 10.0; // 30

            assert!(
                (result - expected).abs() < 0.001,
                "Expected {}, got {}",
                expected,
                result
            );
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_norm() {
        if let Some(token) = X64V3Token::try_new() {
            let a = vec![3.0f32, 4.0];
            let norm = norm_f32(token, &a);
            assert!((norm - 5.0).abs() < 0.001); // 3-4-5 triangle
        }
    }
}

//! Dot Product Operations
//!
//! SIMD-accelerated dot product using FMA instructions.

use crate::ops::x86::*;
use crate::tokens::x86::{Avx2FmaToken, Avx2Token};

/// Compute dot product of two f32 slices using AVX2+FMA.
///
/// Uses fused multiply-add for maximum performance and accuracy.
///
/// # Example
///
/// ```rust,ignore
/// use archmage::{Avx2FmaToken, composite::dot_product_f32};
///
/// if let Some(token) = Avx2FmaToken::try_new() {
///     let a = vec![1.0f32; 1024];
///     let b = vec![2.0f32; 1024];
///     let result = dot_product_f32(token, &a, &b);
///     assert!((result - 2048.0).abs() < 0.001);
/// }
/// ```
#[inline]
pub fn dot_product_f32(token: Avx2FmaToken, a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Slice lengths must match");

    let avx2 = token.avx2();
    let fma = token.fma();

    // Accumulate in 4 vectors to hide FMA latency
    let mut sum0 = zero_f32x8(avx2);
    let mut sum1 = zero_f32x8(avx2);
    let mut sum2 = zero_f32x8(avx2);
    let mut sum3 = zero_f32x8(avx2);

    let chunks = a.len() / 32;
    for i in 0..chunks {
        let base = i * 32;

        let a0 = load_f32x8(avx2, a[base..][..8].try_into().unwrap());
        let b0 = load_f32x8(avx2, b[base..][..8].try_into().unwrap());
        sum0 = fmadd_f32x8(fma, a0, b0, sum0);

        let a1 = load_f32x8(avx2, a[base + 8..][..8].try_into().unwrap());
        let b1 = load_f32x8(avx2, b[base + 8..][..8].try_into().unwrap());
        sum1 = fmadd_f32x8(fma, a1, b1, sum1);

        let a2 = load_f32x8(avx2, a[base + 16..][..8].try_into().unwrap());
        let b2 = load_f32x8(avx2, b[base + 16..][..8].try_into().unwrap());
        sum2 = fmadd_f32x8(fma, a2, b2, sum2);

        let a3 = load_f32x8(avx2, a[base + 24..][..8].try_into().unwrap());
        let b3 = load_f32x8(avx2, b[base + 24..][..8].try_into().unwrap());
        sum3 = fmadd_f32x8(fma, a3, b3, sum3);
    }

    // Combine accumulators
    sum0 = add_f32x8(avx2, sum0, sum1);
    sum2 = add_f32x8(avx2, sum2, sum3);
    sum0 = add_f32x8(avx2, sum0, sum2);

    // Handle remaining chunks of 8
    let remaining_start = chunks * 32;
    let remaining_chunks = (a.len() - remaining_start) / 8;
    for i in 0..remaining_chunks {
        let base = remaining_start + i * 8;
        let av = load_f32x8(avx2, a[base..][..8].try_into().unwrap());
        let bv = load_f32x8(avx2, b[base..][..8].try_into().unwrap());
        sum0 = fmadd_f32x8(fma, av, bv, sum0);
    }

    // Horizontal sum
    let arr = to_array_f32x8(sum0);
    let mut result: f32 = arr.iter().sum();

    // Handle remaining elements
    let scalar_start = remaining_start + remaining_chunks * 8;
    for i in scalar_start..a.len() {
        result += a[i] * b[i];
    }

    result
}

/// Compute dot product without FMA (AVX2 only).
///
/// Slightly less accurate than FMA version due to separate multiply and add.
#[inline]
pub fn dot_product_f32_no_fma(token: Avx2Token, a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let mut sum = zero_f32x8(token);

    let chunks = a.len() / 8;
    for i in 0..chunks {
        let av = load_f32x8(token, a[i * 8..][..8].try_into().unwrap());
        let bv = load_f32x8(token, b[i * 8..][..8].try_into().unwrap());
        let prod = mul_f32x8(token, av, bv);
        sum = add_f32x8(token, sum, prod);
    }

    let arr = to_array_f32x8(sum);
    let mut result: f32 = arr.iter().sum();

    for i in (chunks * 8)..a.len() {
        result += a[i] * b[i];
    }

    result
}

/// Compute squared L2 norm (sum of squares) using AVX2+FMA.
#[inline]
pub fn norm_squared_f32(token: Avx2FmaToken, a: &[f32]) -> f32 {
    dot_product_f32(token, a, a)
}

/// Compute L2 norm using AVX2+FMA.
#[inline]
pub fn norm_f32(token: Avx2FmaToken, a: &[f32]) -> f32 {
    norm_squared_f32(token, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::SimdToken;

    #[test]
    fn test_dot_product() {
        if let Some(token) = Avx2FmaToken::try_new() {
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
        if let Some(token) = Avx2FmaToken::try_new() {
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
    fn test_dot_product_no_fma() {
        if let Some(token) = Avx2Token::try_new() {
            let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
            let b: Vec<f32> = vec![1.0; 64];

            let result = dot_product_f32_no_fma(token, &a, &b);
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
    fn test_norm() {
        if let Some(token) = Avx2FmaToken::try_new() {
            let a = vec![3.0f32, 4.0];
            let norm = norm_f32(token, &a);
            assert!((norm - 5.0).abs() < 0.001); // 3-4-5 triangle
        }
    }
}

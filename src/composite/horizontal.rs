//! Horizontal SIMD Operations
//!
//! Operations that reduce a vector to a scalar.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::ops::x86::*;
use crate::tokens::x86::Avx2Token;

/// Horizontal sum of 8 f32s using AVX2.
///
/// Reduces a 256-bit vector to a single f32 sum.
#[inline]
pub fn hsum_f32x8(token: Avx2Token, v: __m256) -> f32 {
    // Extract high and low 128-bit lanes
    let lo = extractf128_f32x8::<0>(token, v);
    let hi = extractf128_f32x8::<1>(token, v);

    // Add the two lanes
    let sum128 = unsafe { _mm_add_ps(lo, hi) };

    // Horizontal add within 128 bits
    let sum64 = unsafe { _mm_hadd_ps(sum128, sum128) };
    let sum32 = unsafe { _mm_hadd_ps(sum64, sum64) };

    // Extract scalar
    unsafe { _mm_cvtss_f32(sum32) }
}

/// Horizontal maximum of 8 f32s using AVX2.
#[inline]
pub fn hmax_f32x8(token: Avx2Token, v: __m256) -> f32 {
    let lo = extractf128_f32x8::<0>(token, v);
    let hi = extractf128_f32x8::<1>(token, v);

    let max128 = unsafe { _mm_max_ps(lo, hi) };

    // Shuffle and max to reduce
    let shuf = unsafe { _mm_movehdup_ps(max128) }; // [1,1,3,3]
    let max64 = unsafe { _mm_max_ps(max128, shuf) };
    let shuf2 = unsafe { _mm_movehl_ps(max64, max64) }; // [2,3,2,3]
    let max32 = unsafe { _mm_max_ss(max64, shuf2) };

    unsafe { _mm_cvtss_f32(max32) }
}

/// Horizontal minimum of 8 f32s using AVX2.
#[inline]
pub fn hmin_f32x8(token: Avx2Token, v: __m256) -> f32 {
    let lo = extractf128_f32x8::<0>(token, v);
    let hi = extractf128_f32x8::<1>(token, v);

    let min128 = unsafe { _mm_min_ps(lo, hi) };

    let shuf = unsafe { _mm_movehdup_ps(min128) };
    let min64 = unsafe { _mm_min_ps(min128, shuf) };
    let shuf2 = unsafe { _mm_movehl_ps(min64, min64) };
    let min32 = unsafe { _mm_min_ss(min64, shuf2) };

    unsafe { _mm_cvtss_f32(min32) }
}

/// Sum all elements of an f32 slice using AVX2.
#[inline]
pub fn sum_f32_slice(token: Avx2Token, data: &[f32]) -> f32 {
    let mut sum = zero_f32x8(token);

    let chunks = data.len() / 8;
    for i in 0..chunks {
        let v = load_f32x8(token, data[i * 8..][..8].try_into().unwrap());
        sum = add_f32x8(token, sum, v);
    }

    let mut result = hsum_f32x8(token, sum);

    // Handle remainder
    for &val in &data[chunks * 8..] {
        result += val;
    }

    result
}

/// Find maximum element in an f32 slice using AVX2.
#[inline]
pub fn max_f32_slice(token: Avx2Token, data: &[f32]) -> f32 {
    if data.is_empty() {
        return f32::NEG_INFINITY;
    }

    let mut max_vec = set1_f32x8(token, f32::NEG_INFINITY);

    let chunks = data.len() / 8;
    for i in 0..chunks {
        let v = load_f32x8(token, data[i * 8..][..8].try_into().unwrap());
        max_vec = max_f32x8(token, max_vec, v);
    }

    let mut result = hmax_f32x8(token, max_vec);

    // Handle remainder
    for &val in &data[chunks * 8..] {
        if val > result {
            result = val;
        }
    }

    result
}

/// Find minimum element in an f32 slice using AVX2.
#[inline]
pub fn min_f32_slice(token: Avx2Token, data: &[f32]) -> f32 {
    if data.is_empty() {
        return f32::INFINITY;
    }

    let mut min_vec = set1_f32x8(token, f32::INFINITY);

    let chunks = data.len() / 8;
    for i in 0..chunks {
        let v = load_f32x8(token, data[i * 8..][..8].try_into().unwrap());
        min_vec = min_f32x8(token, min_vec, v);
    }

    let mut result = hmin_f32x8(token, min_vec);

    // Handle remainder
    for &val in &data[chunks * 8..] {
        if val < result {
            result = val;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::SimdToken;

    #[test]
    fn test_hsum() {
        if let Some(token) = Avx2Token::try_new() {
            let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let v = load_f32x8(token, &data);
            let sum = hsum_f32x8(token, v);
            assert!((sum - 36.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_hmax() {
        if let Some(token) = Avx2Token::try_new() {
            let data = [1.0f32, 8.0, 3.0, 4.0, 5.0, 2.0, 7.0, 6.0];
            let v = load_f32x8(token, &data);
            let max = hmax_f32x8(token, v);
            assert!((max - 8.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_hmin() {
        if let Some(token) = Avx2Token::try_new() {
            let data = [3.0f32, 8.0, 1.0, 4.0, 5.0, 2.0, 7.0, 6.0];
            let v = load_f32x8(token, &data);
            let min = hmin_f32x8(token, v);
            assert!((min - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_sum_slice() {
        if let Some(token) = Avx2Token::try_new() {
            let data: Vec<f32> = (1..=100).map(|i| i as f32).collect();
            let sum = sum_f32_slice(token, &data);
            let expected = 5050.0; // n*(n+1)/2 for n=100
            assert!((sum - expected).abs() < 0.001);
        }
    }

    #[test]
    fn test_max_slice() {
        if let Some(token) = Avx2Token::try_new() {
            let data = vec![1.0f32, 5.0, 3.0, 9.0, 2.0, 8.0, 4.0, 7.0, 6.0, 10.0];
            let max = max_f32_slice(token, &data);
            assert!((max - 10.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_min_slice() {
        if let Some(token) = Avx2Token::try_new() {
            let data = vec![5.0f32, 3.0, 9.0, 1.0, 8.0, 4.0, 7.0, 6.0, 2.0, 10.0];
            let min = min_f32_slice(token, &data);
            assert!((min - 1.0).abs() < 0.001);
        }
    }
}

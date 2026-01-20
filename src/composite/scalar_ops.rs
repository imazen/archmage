//! Operation traits with scalar fallbacks.
//!
//! These traits provide default scalar implementations that any token can use.
//! Tokens with SIMD support can override with optimized implementations.
//!
//! The `_or_scalar` suffix on methods indicates the operation may fall back
//! to scalar code if the token doesn't provide a SIMD override.
//!
//! For guaranteed SIMD performance, see [`super::simd_ops`].

use crate::tokens::SimdToken;

/// 8x8 matrix transpose with scalar fallback.
///
/// Tokens with SIMD support (e.g., `Avx2Token`) can override with optimized code.
/// Other tokens use the default scalar implementation.
pub trait Transpose8x8OrScalar: SimdToken {
    /// Transpose an 8x8 f32 matrix in-place.
    #[inline(always)]
    fn transpose_8x8_or_scalar(&self, block: &mut [f32; 64]) {
        // Scalar fallback: swap elements across the diagonal
        for i in 0..8 {
            for j in (i + 1)..8 {
                block.swap(i * 8 + j, j * 8 + i);
            }
        }
    }

    /// Transpose an 8x8 f32 matrix from input to output.
    #[inline(always)]
    fn transpose_8x8_copy_or_scalar(&self, input: &[f32; 64], output: &mut [f32; 64]) {
        for i in 0..8 {
            for j in 0..8 {
                output[j * 8 + i] = input[i * 8 + j];
            }
        }
    }
}

/// Dot product with scalar fallback.
///
/// Tokens with SIMD+FMA support can override with optimized code.
/// Other tokens use the default scalar implementation.
pub trait DotProductOrScalar: SimdToken {
    /// Compute dot product of two f32 slices.
    #[inline(always)]
    fn dot_product_f32_or_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Slice lengths must match");
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Compute squared L2 norm (sum of squares).
    #[inline(always)]
    fn norm_squared_f32_or_scalar(&self, a: &[f32]) -> f32 {
        self.dot_product_f32_or_scalar(a, a)
    }

    /// Compute L2 norm.
    ///
    /// Requires `std` feature (uses `f32::sqrt`).
    #[cfg(feature = "std")]
    #[inline(always)]
    fn norm_f32_or_scalar(&self, a: &[f32]) -> f32 {
        self.norm_squared_f32_or_scalar(a).sqrt()
    }
}

/// Horizontal reductions with scalar fallback.
///
/// Tokens with SIMD support can override with optimized code.
/// Other tokens use the default scalar implementation.
pub trait HorizontalOpsOrScalar: SimdToken {
    /// Sum all elements of an f32 slice.
    #[inline(always)]
    fn sum_f32_or_scalar(&self, data: &[f32]) -> f32 {
        data.iter().sum()
    }

    /// Find maximum element in an f32 slice.
    #[inline(always)]
    fn max_f32_or_scalar(&self, data: &[f32]) -> f32 {
        data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Find minimum element in an f32 slice.
    #[inline(always)]
    fn min_f32_or_scalar(&self, data: &[f32]) -> f32 {
        data.iter().copied().fold(f32::INFINITY, f32::min)
    }
}

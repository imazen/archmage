//! SIMD-optimized operation traits.
//!
//! These traits require explicit implementation for each token type.
//! Use these when you need guaranteed SIMD performance.
//!
//! For operations that work with any token (with scalar fallback),
//! see [`super::scalar_ops`].

use super::SimdToken;

/// 8x8 matrix transpose operation (SIMD-optimized).
///
/// This is critical for DCT transforms in image/video codecs.
pub trait Transpose8x8: SimdToken {
    /// Transpose an 8x8 f32 matrix in-place.
    fn transpose_8x8(&self, block: &mut [f32; 64]);

    /// Transpose an 8x8 f32 matrix from input to output.
    fn transpose_8x8_copy(&self, input: &[f32; 64], output: &mut [f32; 64]);
}

/// Dot product operations (SIMD-optimized).
pub trait DotProduct: SimdToken {
    /// Compute dot product of two f32 slices.
    fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> f32;

    /// Compute squared L2 norm (sum of squares).
    #[inline(always)]
    fn norm_squared_f32(&self, a: &[f32]) -> f32 {
        self.dot_product_f32(a, a)
    }

    /// Compute L2 norm.
    #[inline(always)]
    fn norm_f32(&self, a: &[f32]) -> f32 {
        self.norm_squared_f32(a).sqrt()
    }
}

/// Horizontal reduction operations (SIMD-optimized).
pub trait HorizontalOps: SimdToken {
    /// Sum all elements of an f32 slice.
    fn sum_f32(&self, data: &[f32]) -> f32;

    /// Find maximum element in an f32 slice.
    fn max_f32(&self, data: &[f32]) -> f32;

    /// Find minimum element in an f32 slice.
    fn min_f32(&self, data: &[f32]) -> f32;
}

//! SIMD capability tokens
//!
//! Tokens are zero-sized proof types that demonstrate a CPU feature is available.
//! They can only be constructed via unsafe `new_unchecked()` or fallible `try_new()`.

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod arm;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

// Re-export platform-specific tokens
#[cfg(target_arch = "x86_64")]
pub use x86::*;

#[cfg(target_arch = "aarch64")]
pub use arm::*;

#[cfg(target_arch = "wasm32")]
pub use wasm::*;

/// Marker trait for SIMD capability tokens.
///
/// All tokens implement this trait, enabling generic code over different
/// SIMD feature levels.
///
/// # Safety
///
/// Implementors must ensure that `new_unchecked()` is only called when
/// the corresponding CPU feature is actually available.
pub trait SimdToken: Copy + Clone + Send + Sync + 'static {
    /// Human-readable name for diagnostics and error messages.
    const NAME: &'static str;

    /// Attempt to create a token with runtime feature detection.
    ///
    /// Returns `Some(token)` if the CPU supports this feature, `None` otherwise.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if let Some(token) = Avx2Token::try_new() {
    ///     // Use AVX2 operations
    /// } else {
    ///     // Fallback path
    /// }
    /// ```
    fn try_new() -> Option<Self>;

    /// Create a token without runtime checks.
    ///
    /// # Safety
    ///
    /// Caller must guarantee the CPU feature is available via one of:
    /// - Runtime detection (`is_x86_feature_detected!` or equivalent)
    /// - Compile-time guarantee (`#[target_feature]` attribute)
    /// - Target CPU specification (`-C target-cpu=native` or similar)
    /// - Being inside a multiversioned function variant
    unsafe fn new_unchecked() -> Self;
}

/// Trait for tokens that can be decomposed into sub-tokens.
///
/// Combined tokens like `Avx2FmaToken` implement this to provide
/// access to their component tokens.
pub trait CompositeToken: SimdToken {
    /// The component token types
    type Components;

    /// Decompose into component tokens
    fn components(&self) -> Self::Components;
}

// ============================================================================
// Capability Marker Traits
// ============================================================================
//
// These traits indicate what capabilities a token provides, enabling generic
// code to constrain which tokens are accepted. They don't provide operations
// directly - use raw intrinsics via `#[simd_fn]` for that.

/// Marker trait for tokens that provide 128-bit SIMD.
///
/// Implemented by: `Sse2Token`, `NeonToken`, `Simd128Token`
pub trait Has128BitSimd: SimdToken {}

/// Marker trait for tokens that provide 256-bit SIMD.
///
/// Implemented by: `Avx2Token`, `X64V3Token`, etc.
pub trait Has256BitSimd: Has128BitSimd {}

/// Marker trait for tokens that provide 512-bit SIMD.
///
/// Implemented by: `Avx512fToken`, `X64V4Token`
pub trait Has512BitSimd: Has256BitSimd {}

/// Marker trait for tokens that provide FMA (fused multiply-add).
///
/// Implemented by: `FmaToken`, `Avx2FmaToken`, `X64V3Token`, `NeonToken`
pub trait HasFma: SimdToken {}

/// Marker trait for tokens that provide scalable vectors (variable width).
///
/// Implemented by: `SveToken`, `Sve2Token`
pub trait HasScalableVectors: SimdToken {}

// ============================================================================
// Operation Traits - Generic interface with specialized implementations
// ============================================================================
//
// These traits provide a generic interface that can be used with any token,
// while each token provides its own optimized implementation using the
// appropriate architecture-specific intrinsics.

/// 8x8 matrix transpose operation.
///
/// This is critical for DCT transforms in image/video codecs.
pub trait Transpose8x8: SimdToken {
    /// Transpose an 8x8 f32 matrix in-place.
    fn transpose_8x8(&self, block: &mut [f32; 64]);

    /// Transpose an 8x8 f32 matrix from input to output.
    fn transpose_8x8_copy(&self, input: &[f32; 64], output: &mut [f32; 64]);
}

/// Dot product operations.
pub trait DotProduct: SimdToken {
    /// Compute dot product of two f32 slices.
    fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> f32;

    /// Compute squared L2 norm (sum of squares).
    fn norm_squared_f32(&self, a: &[f32]) -> f32 {
        self.dot_product_f32(a, a)
    }

    /// Compute L2 norm.
    fn norm_f32(&self, a: &[f32]) -> f32 {
        self.norm_squared_f32(a).sqrt()
    }
}

/// Horizontal reduction operations on vectors.
pub trait HorizontalOps: SimdToken {
    /// Sum all elements of an f32 slice.
    fn sum_f32(&self, data: &[f32]) -> f32;

    /// Find maximum element in an f32 slice.
    fn max_f32(&self, data: &[f32]) -> f32;

    /// Find minimum element in an f32 slice.
    fn min_f32(&self, data: &[f32]) -> f32;
}

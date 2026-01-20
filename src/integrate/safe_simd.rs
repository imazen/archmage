//! Integration with `safe_unaligned_simd` crate
//!
//! Provides safe load/store operations via references instead of raw pointers.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use crate::tokens::x86::*;

#[cfg(all(feature = "safe_unaligned_simd", target_arch = "x86_64"))]
use safe_unaligned_simd::x86_64 as safe_simd;

// ============================================================================
// Safe Load/Store Operations
// ============================================================================

/// Safe unaligned load of 8 f32s using safe_unaligned_simd
///
/// This is the safest way to load SIMD data:
/// - Token proves AVX2 is available
/// - Reference proves memory is valid
/// - No raw pointers
#[cfg(all(feature = "safe_unaligned_simd", target_arch = "x86_64"))]
#[inline(always)]
pub fn safe_load_f32x8(_token: Avx2Token, data: &[f32; 8]) -> __m256 {
    // SAFETY: token proves AVX is available
    unsafe { safe_simd::_mm256_loadu_ps(data) }
}

/// Safe unaligned store of 8 f32s using safe_unaligned_simd
///
/// # Safety note
/// The unsafe block is required because safe_simd functions have `#[target_feature]`.
/// The token proves the feature is available at runtime.
#[cfg(all(feature = "safe_unaligned_simd", target_arch = "x86_64"))]
#[inline(always)]
pub fn safe_store_f32x8(_token: Avx2Token, data: &mut [f32; 8], v: __m256) {
    // SAFETY: token proves AVX2 is available
    unsafe { safe_simd::_mm256_storeu_ps(data, v) };
}

/// Safe unaligned load of 4 f64s
#[cfg(all(feature = "safe_unaligned_simd", target_arch = "x86_64"))]
#[inline(always)]
pub fn safe_load_f64x4(_token: Avx2Token, data: &[f64; 4]) -> __m256d {
    // SAFETY: token proves AVX is available
    unsafe { safe_simd::_mm256_loadu_pd(data) }
}

/// Safe unaligned store of 4 f64s
#[cfg(all(feature = "safe_unaligned_simd", target_arch = "x86_64"))]
#[inline(always)]
pub fn safe_store_f64x4(_token: Avx2Token, data: &mut [f64; 4], v: __m256d) {
    // SAFETY: token proves AVX is available
    unsafe { safe_simd::_mm256_storeu_pd(data, v) };
}

/// Safe unaligned load of 8 i32s
#[cfg(all(feature = "safe_unaligned_simd", target_arch = "x86_64"))]
#[inline(always)]
pub fn safe_load_i32x8(_token: Avx2Token, data: &[i32; 8]) -> __m256i {
    // SAFETY: token proves AVX2 is available
    unsafe { safe_simd::_mm256_loadu_si256(data) }
}

/// Safe unaligned store of 8 i32s
#[cfg(all(feature = "safe_unaligned_simd", target_arch = "x86_64"))]
#[inline(always)]
pub fn safe_store_i32x8(_token: Avx2Token, data: &mut [i32; 8], v: __m256i) {
    // SAFETY: token proves AVX2 is available
    unsafe { safe_simd::_mm256_storeu_si256(data, v) };
}

// SSE variants (128-bit)

/// Safe unaligned load of 4 f32s (SSE)
#[cfg(all(feature = "safe_unaligned_simd", target_arch = "x86_64"))]
#[inline(always)]
pub fn safe_load_f32x4(_token: Sse2Token, data: &[f32; 4]) -> __m128 {
    // SAFETY: token proves SSE is available (baseline on x86_64)
    unsafe { safe_simd::_mm_loadu_ps(data) }
}

/// Safe unaligned store of 4 f32s (SSE)
#[cfg(all(feature = "safe_unaligned_simd", target_arch = "x86_64"))]
#[inline(always)]
pub fn safe_store_f32x4(_token: Sse2Token, data: &mut [f32; 4], v: __m128) {
    // SAFETY: token proves SSE is available (baseline on x86_64)
    unsafe { safe_simd::_mm_storeu_ps(data, v) };
}

/// Safe unaligned load of 2 f64s (SSE2)
#[cfg(all(feature = "safe_unaligned_simd", target_arch = "x86_64"))]
#[inline(always)]
pub fn safe_load_f64x2(_token: Sse2Token, data: &[f64; 2]) -> __m128d {
    // SAFETY: token proves SSE2 is available (baseline on x86_64)
    unsafe { safe_simd::_mm_loadu_pd(data) }
}

/// Safe unaligned store of 2 f64s (SSE2)
#[cfg(all(feature = "safe_unaligned_simd", target_arch = "x86_64"))]
#[inline(always)]
pub fn safe_store_f64x2(_token: Sse2Token, data: &mut [f64; 2], v: __m128d) {
    // SAFETY: token proves SSE2 is available (baseline on x86_64)
    unsafe { safe_simd::_mm_storeu_pd(data, v) };
}

/// Safe unaligned load of 4 i32s (SSE2)
#[cfg(all(feature = "safe_unaligned_simd", target_arch = "x86_64"))]
#[inline(always)]
pub fn safe_load_i32x4(_token: Sse2Token, data: &[i32; 4]) -> __m128i {
    // SAFETY: token proves SSE2 is available (baseline on x86_64)
    unsafe { safe_simd::_mm_loadu_si128(data) }
}

/// Safe unaligned store of 4 i32s (SSE2)
#[cfg(all(feature = "safe_unaligned_simd", target_arch = "x86_64"))]
#[inline(always)]
pub fn safe_store_i32x4(_token: Sse2Token, data: &mut [i32; 4], v: __m128i) {
    // SAFETY: token proves SSE2 is available (baseline on x86_64)
    unsafe { safe_simd::_mm_storeu_si128(data, v) };
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "safe_unaligned_simd", target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::tokens::SimdToken;

    #[test]
    fn test_safe_load_store_f32x8() {
        if let Some(token) = Avx2Token::try_new() {
            let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let v = safe_load_f32x8(token, &data);

            let mut out = [0.0f32; 8];
            safe_store_f32x8(token, &mut out, v);

            assert_eq!(data, out);
        }
    }

    #[test]
    fn test_safe_load_store_i32x8() {
        if let Some(token) = Avx2Token::try_new() {
            let data = [1i32, 2, 3, 4, 5, 6, 7, 8];
            let v = safe_load_i32x8(token, &data);

            let mut out = [0i32; 8];
            safe_store_i32x8(token, &mut out, v);

            assert_eq!(data, out);
        }
    }

    #[test]
    fn test_safe_load_store_sse() {
        // SSE2 is always available on x86_64
        let token = Sse2Token::try_new().unwrap();

        let data = [1.0f32, 2.0, 3.0, 4.0];
        let v = safe_load_f32x4(token, &data);

        let mut out = [0.0f32; 4];
        safe_store_f32x4(token, &mut out, v);

        assert_eq!(data, out);
    }
}

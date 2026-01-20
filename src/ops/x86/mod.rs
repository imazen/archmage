//! Safe load/store operations for x86_64 SIMD.
//!
//! These operations use references instead of raw pointers, making memory
//! access safe when combined with a capability token.
//!
//! For all other intrinsics (arithmetic, shuffle, compare, etc.), use
//! `#[simd_fn]` to make them safe inside `#[target_feature]` context.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::tokens::x86::*;

// ============================================================================
// Safe Load/Store Operations (256-bit)
// ============================================================================

/// Load 8 f32s from memory safely.
///
/// Uses a reference instead of a raw pointer, guaranteeing valid memory access.
#[inline(always)]
pub fn load_f32x8(_token: Avx2Token, data: &[f32; 8]) -> __m256 {
    // SAFETY: Token proves AVX2 is available, reference proves valid memory
    unsafe { _mm256_loadu_ps(data.as_ptr()) }
}

/// Store 8 f32s to memory safely.
#[inline(always)]
pub fn store_f32x8(_token: Avx2Token, data: &mut [f32; 8], v: __m256) {
    // SAFETY: Token proves AVX2 is available, reference proves valid memory
    unsafe { _mm256_storeu_ps(data.as_mut_ptr(), v) }
}

/// Load 8 i32s from memory safely.
#[inline(always)]
pub fn load_i32x8(_token: Avx2Token, data: &[i32; 8]) -> __m256i {
    // SAFETY: Token proves AVX2 is available, reference proves valid memory
    unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) }
}

/// Store 8 i32s to memory safely.
#[inline(always)]
pub fn store_i32x8(_token: Avx2Token, data: &mut [i32; 8], v: __m256i) {
    // SAFETY: Token proves AVX2 is available, reference proves valid memory
    unsafe { _mm256_storeu_si256(data.as_mut_ptr() as *mut __m256i, v) }
}

/// Load 4 f64s from memory safely.
#[inline(always)]
pub fn load_f64x4(_token: Avx2Token, data: &[f64; 4]) -> __m256d {
    // SAFETY: Token proves AVX2 is available, reference proves valid memory
    unsafe { _mm256_loadu_pd(data.as_ptr()) }
}

/// Store 4 f64s to memory safely.
#[inline(always)]
pub fn store_f64x4(_token: Avx2Token, data: &mut [f64; 4], v: __m256d) {
    // SAFETY: Token proves AVX2 is available, reference proves valid memory
    unsafe { _mm256_storeu_pd(data.as_mut_ptr(), v) }
}

// ============================================================================
// Safe Load/Store Operations (128-bit SSE)
// ============================================================================

/// Load 4 f32s from memory safely (SSE).
#[inline(always)]
pub fn load_f32x4(_token: Sse2Token, data: &[f32; 4]) -> __m128 {
    // SAFETY: Token proves SSE is available, reference proves valid memory
    unsafe { _mm_loadu_ps(data.as_ptr()) }
}

/// Store 4 f32s to memory safely (SSE).
#[inline(always)]
pub fn store_f32x4(_token: Sse2Token, data: &mut [f32; 4], v: __m128) {
    // SAFETY: Token proves SSE is available, reference proves valid memory
    unsafe { _mm_storeu_ps(data.as_mut_ptr(), v) }
}

/// Load 4 i32s from memory safely (SSE2).
#[inline(always)]
pub fn load_i32x4(_token: Sse2Token, data: &[i32; 4]) -> __m128i {
    // SAFETY: Token proves SSE2 is available, reference proves valid memory
    unsafe { _mm_loadu_si128(data.as_ptr() as *const __m128i) }
}

/// Store 4 i32s to memory safely (SSE2).
#[inline(always)]
pub fn store_i32x4(_token: Sse2Token, data: &mut [i32; 4], v: __m128i) {
    // SAFETY: Token proves SSE2 is available, reference proves valid memory
    unsafe { _mm_storeu_si128(data.as_mut_ptr() as *mut __m128i, v) }
}

/// Load 2 f64s from memory safely (SSE2).
#[inline(always)]
pub fn load_f64x2(_token: Sse2Token, data: &[f64; 2]) -> __m128d {
    // SAFETY: Token proves SSE2 is available, reference proves valid memory
    unsafe { _mm_loadu_pd(data.as_ptr()) }
}

/// Store 2 f64s to memory safely (SSE2).
#[inline(always)]
pub fn store_f64x2(_token: Sse2Token, data: &mut [f64; 2], v: __m128d) {
    // SAFETY: Token proves SSE2 is available, reference proves valid memory
    unsafe { _mm_storeu_pd(data.as_mut_ptr(), v) }
}

// ============================================================================
// Conversion Utilities
// ============================================================================

/// Convert __m256 to array (no token needed - just reinterpret bits).
#[inline(always)]
pub fn to_array_f32x8(v: __m256) -> [f32; 8] {
    // SAFETY: __m256 and [f32; 8] have the same memory layout
    unsafe { core::mem::transmute(v) }
}

/// Convert __m256i to i32 array (no token needed - just reinterpret bits).
#[inline(always)]
pub fn to_array_i32x8(v: __m256i) -> [i32; 8] {
    // SAFETY: __m256i and [i32; 8] have the same memory layout
    unsafe { core::mem::transmute(v) }
}

/// Convert __m128 to array (no token needed - just reinterpret bits).
#[inline(always)]
pub fn to_array_f32x4(v: __m128) -> [f32; 4] {
    // SAFETY: __m128 and [f32; 4] have the same memory layout
    unsafe { core::mem::transmute(v) }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::SimdToken;

    #[test]
    fn test_load_store_f32x8() {
        if let Some(token) = Avx2Token::try_new() {
            let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let v = load_f32x8(token, &data);
            let mut out = [0.0f32; 8];
            store_f32x8(token, &mut out, v);
            assert_eq!(data, out);
        }
    }

    #[test]
    fn test_load_store_i32x8() {
        if let Some(token) = Avx2Token::try_new() {
            let data = [1i32, 2, 3, 4, 5, 6, 7, 8];
            let v = load_i32x8(token, &data);
            let mut out = [0i32; 8];
            store_i32x8(token, &mut out, v);
            assert_eq!(data, out);
        }
    }

    #[test]
    fn test_load_store_sse() {
        // SSE2 is always available on x86_64
        let token = Sse2Token::try_new().unwrap();

        let data = [1.0f32, 2.0, 3.0, 4.0];
        let v = load_f32x4(token, &data);
        let mut out = [0.0f32; 4];
        store_f32x4(token, &mut out, v);
        assert_eq!(data, out);
    }

    #[test]
    fn test_to_array() {
        if let Some(token) = Avx2Token::try_new() {
            let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let v = load_f32x8(token, &data);
            let arr = to_array_f32x8(v);
            assert_eq!(data, arr);
        }
    }
}

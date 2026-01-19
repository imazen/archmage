//! x86_64 SIMD operations gated by capability tokens

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::tokens::x86::*;

// ============================================================================
// AVX2 Operations (256-bit f32)
// ============================================================================

/// Load 8 f32s from memory - requires AVX2 token as proof
#[inline(always)]
pub fn load_f32x8(_token: Avx2Token, data: &[f32; 8]) -> __m256 {
    // SAFETY: Token proves AVX2 is available
    unsafe { _mm256_loadu_ps(data.as_ptr()) }
}

/// Store 8 f32s to memory - requires AVX2 token as proof
#[inline(always)]
pub fn store_f32x8(_token: Avx2Token, data: &mut [f32; 8], v: __m256) {
    // SAFETY: Token proves AVX2 is available
    unsafe { _mm256_storeu_ps(data.as_mut_ptr(), v) }
}

/// Create zero vector - requires AVX2 token
#[inline(always)]
pub fn zero_f32x8(_token: Avx2Token) -> __m256 {
    unsafe { _mm256_setzero_ps() }
}

/// Broadcast scalar to all lanes - requires AVX2 token
#[inline(always)]
pub fn set1_f32x8(_token: Avx2Token, value: f32) -> __m256 {
    unsafe { _mm256_set1_ps(value) }
}

/// Add two vectors - requires AVX2 token
#[inline(always)]
pub fn add_f32x8(_token: Avx2Token, a: __m256, b: __m256) -> __m256 {
    unsafe { _mm256_add_ps(a, b) }
}

/// Subtract two vectors - requires AVX2 token
#[inline(always)]
pub fn sub_f32x8(_token: Avx2Token, a: __m256, b: __m256) -> __m256 {
    unsafe { _mm256_sub_ps(a, b) }
}

/// Multiply two vectors - requires AVX2 token
#[inline(always)]
pub fn mul_f32x8(_token: Avx2Token, a: __m256, b: __m256) -> __m256 {
    unsafe { _mm256_mul_ps(a, b) }
}

/// Divide two vectors - requires AVX2 token
#[inline(always)]
pub fn div_f32x8(_token: Avx2Token, a: __m256, b: __m256) -> __m256 {
    unsafe { _mm256_div_ps(a, b) }
}

/// Minimum of two vectors - requires AVX2 token
#[inline(always)]
pub fn min_f32x8(_token: Avx2Token, a: __m256, b: __m256) -> __m256 {
    unsafe { _mm256_min_ps(a, b) }
}

/// Maximum of two vectors - requires AVX2 token
#[inline(always)]
pub fn max_f32x8(_token: Avx2Token, a: __m256, b: __m256) -> __m256 {
    unsafe { _mm256_max_ps(a, b) }
}

/// Square root - requires AVX2 token
#[inline(always)]
pub fn sqrt_f32x8(_token: Avx2Token, a: __m256) -> __m256 {
    unsafe { _mm256_sqrt_ps(a) }
}

/// Approximate reciprocal - requires AVX2 token
#[inline(always)]
pub fn rcp_f32x8(_token: Avx2Token, a: __m256) -> __m256 {
    unsafe { _mm256_rcp_ps(a) }
}

/// Approximate reciprocal square root - requires AVX2 token
#[inline(always)]
pub fn rsqrt_f32x8(_token: Avx2Token, a: __m256) -> __m256 {
    unsafe { _mm256_rsqrt_ps(a) }
}

/// Floor - requires AVX2 token
#[inline(always)]
pub fn floor_f32x8(_token: Avx2Token, a: __m256) -> __m256 {
    unsafe { _mm256_floor_ps(a) }
}

/// Ceiling - requires AVX2 token
#[inline(always)]
pub fn ceil_f32x8(_token: Avx2Token, a: __m256) -> __m256 {
    unsafe { _mm256_ceil_ps(a) }
}

/// Bitwise AND - requires AVX2 token
#[inline(always)]
pub fn and_f32x8(_token: Avx2Token, a: __m256, b: __m256) -> __m256 {
    unsafe { _mm256_and_ps(a, b) }
}

/// Bitwise OR - requires AVX2 token
#[inline(always)]
pub fn or_f32x8(_token: Avx2Token, a: __m256, b: __m256) -> __m256 {
    unsafe { _mm256_or_ps(a, b) }
}

/// Bitwise XOR - requires AVX2 token
#[inline(always)]
pub fn xor_f32x8(_token: Avx2Token, a: __m256, b: __m256) -> __m256 {
    unsafe { _mm256_xor_ps(a, b) }
}

/// Bitwise AND-NOT (a AND (NOT b)) - requires AVX2 token
#[inline(always)]
pub fn andnot_f32x8(_token: Avx2Token, a: __m256, b: __m256) -> __m256 {
    unsafe { _mm256_andnot_ps(a, b) }
}

// ============================================================================
// FMA Operations
// ============================================================================

/// Fused multiply-add: a * b + c - requires FMA token
#[inline(always)]
pub fn fmadd_f32x8(_token: FmaToken, a: __m256, b: __m256, c: __m256) -> __m256 {
    unsafe { _mm256_fmadd_ps(a, b, c) }
}

/// Fused multiply-subtract: a * b - c - requires FMA token
#[inline(always)]
pub fn fmsub_f32x8(_token: FmaToken, a: __m256, b: __m256, c: __m256) -> __m256 {
    unsafe { _mm256_fmsub_ps(a, b, c) }
}

/// Fused negated multiply-add: -(a * b) + c - requires FMA token
#[inline(always)]
pub fn fnmadd_f32x8(_token: FmaToken, a: __m256, b: __m256, c: __m256) -> __m256 {
    unsafe { _mm256_fnmadd_ps(a, b, c) }
}

/// Fused negated multiply-subtract: -(a * b) - c - requires FMA token
#[inline(always)]
pub fn fnmsub_f32x8(_token: FmaToken, a: __m256, b: __m256, c: __m256) -> __m256 {
    unsafe { _mm256_fnmsub_ps(a, b, c) }
}

// ============================================================================
// Shuffle/Permute Operations
// ============================================================================

/// Unpack low elements - requires AVX2 token
#[inline(always)]
pub fn unpacklo_f32x8(_token: Avx2Token, a: __m256, b: __m256) -> __m256 {
    unsafe { _mm256_unpacklo_ps(a, b) }
}

/// Unpack high elements - requires AVX2 token
#[inline(always)]
pub fn unpackhi_f32x8(_token: Avx2Token, a: __m256, b: __m256) -> __m256 {
    unsafe { _mm256_unpackhi_ps(a, b) }
}

/// Shuffle elements within 128-bit lanes - requires AVX2 token
#[inline(always)]
pub fn shuffle_f32x8<const MASK: i32>(_token: Avx2Token, a: __m256, b: __m256) -> __m256 {
    unsafe { _mm256_shuffle_ps::<MASK>(a, b) }
}

/// Permute elements within each 128-bit lane - requires AVX2 token
#[inline(always)]
pub fn permute_f32x8<const MASK: i32>(_token: Avx2Token, a: __m256) -> __m256 {
    unsafe { _mm256_permute_ps::<MASK>(a) }
}

/// Permute 128-bit lanes - requires AVX2 token
#[inline(always)]
pub fn permute2_f32x8<const MASK: i32>(_token: Avx2Token, a: __m256, b: __m256) -> __m256 {
    unsafe { _mm256_permute2f128_ps::<MASK>(a, b) }
}

/// Blend elements based on immediate mask - requires AVX2 token
#[inline(always)]
pub fn blend_f32x8<const MASK: i32>(_token: Avx2Token, a: __m256, b: __m256) -> __m256 {
    unsafe { _mm256_blend_ps::<MASK>(a, b) }
}

/// Blend elements based on vector mask - requires AVX2 token
#[inline(always)]
pub fn blendv_f32x8(_token: Avx2Token, a: __m256, b: __m256, mask: __m256) -> __m256 {
    unsafe { _mm256_blendv_ps(a, b, mask) }
}

/// Broadcast single element to all lanes - requires AVX2 token
#[inline(always)]
pub fn broadcast_f32x8(_token: Avx2Token, value: &f32) -> __m256 {
    unsafe { _mm256_broadcast_ss(value) }
}

// ============================================================================
// Comparison Operations
// ============================================================================

/// Compare equal - requires AVX2 token
#[inline(always)]
pub fn cmpeq_f32x8(_token: Avx2Token, a: __m256, b: __m256) -> __m256 {
    unsafe { _mm256_cmp_ps::<_CMP_EQ_OQ>(a, b) }
}

/// Compare not equal - requires AVX2 token
#[inline(always)]
pub fn cmpneq_f32x8(_token: Avx2Token, a: __m256, b: __m256) -> __m256 {
    unsafe { _mm256_cmp_ps::<_CMP_NEQ_OQ>(a, b) }
}

/// Compare less than - requires AVX2 token
#[inline(always)]
pub fn cmplt_f32x8(_token: Avx2Token, a: __m256, b: __m256) -> __m256 {
    unsafe { _mm256_cmp_ps::<_CMP_LT_OQ>(a, b) }
}

/// Compare less than or equal - requires AVX2 token
#[inline(always)]
pub fn cmple_f32x8(_token: Avx2Token, a: __m256, b: __m256) -> __m256 {
    unsafe { _mm256_cmp_ps::<_CMP_LE_OQ>(a, b) }
}

/// Compare greater than - requires AVX2 token
#[inline(always)]
pub fn cmpgt_f32x8(_token: Avx2Token, a: __m256, b: __m256) -> __m256 {
    unsafe { _mm256_cmp_ps::<_CMP_GT_OQ>(a, b) }
}

/// Compare greater than or equal - requires AVX2 token
#[inline(always)]
pub fn cmpge_f32x8(_token: Avx2Token, a: __m256, b: __m256) -> __m256 {
    unsafe { _mm256_cmp_ps::<_CMP_GE_OQ>(a, b) }
}

/// Move mask - extract sign bits - requires AVX2 token
#[inline(always)]
pub fn movemask_f32x8(_token: Avx2Token, a: __m256) -> i32 {
    unsafe { _mm256_movemask_ps(a) }
}

// ============================================================================
// Conversion Operations
// ============================================================================

/// Convert to array - no token needed (just reinterpret)
#[inline(always)]
pub fn to_array_f32x8(v: __m256) -> [f32; 8] {
    // SAFETY: __m256 and [f32; 8] have the same layout
    unsafe { core::mem::transmute(v) }
}

/// Convert from array - requires AVX2 token
#[inline(always)]
pub fn from_array_f32x8(_token: Avx2Token, arr: [f32; 8]) -> __m256 {
    // SAFETY: Token proves AVX2 is available
    unsafe { core::mem::transmute(arr) }
}

/// Convert i32x8 to f32x8 - requires AVX2 token
#[inline(always)]
pub fn cvt_i32x8_f32x8(_token: Avx2Token, a: __m256i) -> __m256 {
    unsafe { _mm256_cvtepi32_ps(a) }
}

/// Convert f32x8 to i32x8 (truncate) - requires AVX2 token
#[inline(always)]
pub fn cvtt_f32x8_i32x8(_token: Avx2Token, a: __m256) -> __m256i {
    unsafe { _mm256_cvttps_epi32(a) }
}

/// Convert f32x8 to i32x8 (round) - requires AVX2 token
#[inline(always)]
pub fn cvt_f32x8_i32x8(_token: Avx2Token, a: __m256) -> __m256i {
    unsafe { _mm256_cvtps_epi32(a) }
}

// ============================================================================
// Integer Operations (i32x8)
// ============================================================================

/// Load 8 i32s from memory - requires AVX2 token
#[inline(always)]
pub fn load_i32x8(_token: Avx2Token, data: &[i32; 8]) -> __m256i {
    unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) }
}

/// Store 8 i32s to memory - requires AVX2 token
#[inline(always)]
pub fn store_i32x8(_token: Avx2Token, data: &mut [i32; 8], v: __m256i) {
    unsafe { _mm256_storeu_si256(data.as_mut_ptr() as *mut __m256i, v) }
}

/// Create zero vector - requires AVX2 token
#[inline(always)]
pub fn zero_i32x8(_token: Avx2Token) -> __m256i {
    unsafe { _mm256_setzero_si256() }
}

/// Broadcast scalar to all lanes - requires AVX2 token
#[inline(always)]
pub fn set1_i32x8(_token: Avx2Token, value: i32) -> __m256i {
    unsafe { _mm256_set1_epi32(value) }
}

/// Add two vectors - requires AVX2 token
#[inline(always)]
pub fn add_i32x8(_token: Avx2Token, a: __m256i, b: __m256i) -> __m256i {
    unsafe { _mm256_add_epi32(a, b) }
}

/// Subtract two vectors - requires AVX2 token
#[inline(always)]
pub fn sub_i32x8(_token: Avx2Token, a: __m256i, b: __m256i) -> __m256i {
    unsafe { _mm256_sub_epi32(a, b) }
}

/// Multiply low 32 bits, produce 32-bit result - requires AVX2 token
#[inline(always)]
pub fn mullo_i32x8(_token: Avx2Token, a: __m256i, b: __m256i) -> __m256i {
    unsafe { _mm256_mullo_epi32(a, b) }
}

/// Bitwise AND - requires AVX2 token
#[inline(always)]
pub fn and_i32x8(_token: Avx2Token, a: __m256i, b: __m256i) -> __m256i {
    unsafe { _mm256_and_si256(a, b) }
}

/// Bitwise OR - requires AVX2 token
#[inline(always)]
pub fn or_i32x8(_token: Avx2Token, a: __m256i, b: __m256i) -> __m256i {
    unsafe { _mm256_or_si256(a, b) }
}

/// Bitwise XOR - requires AVX2 token
#[inline(always)]
pub fn xor_i32x8(_token: Avx2Token, a: __m256i, b: __m256i) -> __m256i {
    unsafe { _mm256_xor_si256(a, b) }
}

/// Shift left (immediate) - requires AVX2 token
#[inline(always)]
pub fn slli_i32x8<const IMM: i32>(_token: Avx2Token, a: __m256i) -> __m256i {
    unsafe { _mm256_slli_epi32::<IMM>(a) }
}

/// Shift right logical (immediate) - requires AVX2 token
#[inline(always)]
pub fn srli_i32x8<const IMM: i32>(_token: Avx2Token, a: __m256i) -> __m256i {
    unsafe { _mm256_srli_epi32::<IMM>(a) }
}

/// Shift right arithmetic (immediate) - requires AVX2 token
#[inline(always)]
pub fn srai_i32x8<const IMM: i32>(_token: Avx2Token, a: __m256i) -> __m256i {
    unsafe { _mm256_srai_epi32::<IMM>(a) }
}

// ============================================================================
// Horizontal Operations
// ============================================================================

/// Horizontal add (add adjacent pairs) - requires AVX2 token
#[inline(always)]
pub fn hadd_f32x8(_token: Avx2Token, a: __m256, b: __m256) -> __m256 {
    unsafe { _mm256_hadd_ps(a, b) }
}

/// Horizontal subtract - requires AVX2 token
#[inline(always)]
pub fn hsub_f32x8(_token: Avx2Token, a: __m256, b: __m256) -> __m256 {
    unsafe { _mm256_hsub_ps(a, b) }
}

/// Add-subtract alternating - requires AVX2 token
#[inline(always)]
pub fn addsub_f32x8(_token: Avx2Token, a: __m256, b: __m256) -> __m256 {
    unsafe { _mm256_addsub_ps(a, b) }
}

// ============================================================================
// Extract/Insert Operations
// ============================================================================

/// Extract 128-bit lane - requires AVX2 token
#[inline(always)]
pub fn extractf128_f32x8<const IMM: i32>(_token: Avx2Token, a: __m256) -> __m128 {
    unsafe { _mm256_extractf128_ps::<IMM>(a) }
}

/// Insert 128-bit lane - requires AVX2 token
#[inline(always)]
pub fn insertf128_f32x8<const IMM: i32>(_token: Avx2Token, a: __m256, b: __m128) -> __m256 {
    unsafe { _mm256_insertf128_ps::<IMM>(a, b) }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::SimdToken;

    #[test]
    fn test_load_store() {
        if let Some(token) = Avx2Token::try_new() {
            let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let v = load_f32x8(token, &data);
            let mut out = [0.0f32; 8];
            store_f32x8(token, &mut out, v);
            assert_eq!(data, out);
        }
    }

    #[test]
    fn test_arithmetic() {
        if let Some(token) = Avx2Token::try_new() {
            let a = [1.0f32; 8];
            let b = [2.0f32; 8];
            let va = load_f32x8(token, &a);
            let vb = load_f32x8(token, &b);

            let sum = add_f32x8(token, va, vb);
            let result = to_array_f32x8(sum);
            assert_eq!(result, [3.0f32; 8]);

            let product = mul_f32x8(token, va, vb);
            let result = to_array_f32x8(product);
            assert_eq!(result, [2.0f32; 8]);
        }
    }

    #[test]
    fn test_fma() {
        if let Some(token) = Avx2FmaToken::try_new() {
            let a = [2.0f32; 8];
            let b = [3.0f32; 8];
            let c = [1.0f32; 8];

            let va = load_f32x8(token.avx2(), &a);
            let vb = load_f32x8(token.avx2(), &b);
            let vc = load_f32x8(token.avx2(), &c);

            // 2 * 3 + 1 = 7
            let result = fmadd_f32x8(token.fma(), va, vb, vc);
            let arr = to_array_f32x8(result);
            assert_eq!(arr, [7.0f32; 8]);
        }
    }

    #[test]
    fn test_shuffle() {
        if let Some(token) = Avx2Token::try_new() {
            let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let va = load_f32x8(token, &a);

            // Test unpack operations
            let lo = unpacklo_f32x8(token, va, va);
            let hi = unpackhi_f32x8(token, va, va);

            // Just verify they don't crash
            let _ = to_array_f32x8(lo);
            let _ = to_array_f32x8(hi);
        }
    }
}

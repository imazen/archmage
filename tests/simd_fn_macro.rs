//! Tests for the #[simd_fn] attribute macro.

#![allow(unused)]

#[cfg(target_arch = "x86_64")]
mod x86_tests {
    use archmage::{Avx2FmaToken, Avx2Token, SimdToken, X64V3Token, simd_fn};
    use std::arch::x86_64::*;

    /// Basic test: simd_fn with Avx2Token
    #[simd_fn]
    fn double_values(token: Avx2Token, data: &[f32; 8]) -> [f32; 8] {
        let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };
        let doubled = _mm256_add_ps(v, v);
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
        out
    }

    #[test]
    fn test_simd_fn_basic() {
        if let Some(token) = Avx2Token::try_new() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let output = double_values(token, &input);
            assert_eq!(output, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        }
    }

    /// Test with FMA token
    #[simd_fn]
    fn fma_operation(token: Avx2FmaToken, a: &[f32; 8], b: &[f32; 8], c: &[f32; 8]) -> [f32; 8] {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr()) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr()) };
        let vc = unsafe { _mm256_loadu_ps(c.as_ptr()) };
        // a * b + c
        let result = _mm256_fmadd_ps(va, vb, vc);
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), result) };
        out
    }

    #[test]
    fn test_simd_fn_fma() {
        if let Some(token) = Avx2FmaToken::try_new() {
            let a = [2.0f32; 8];
            let b = [3.0f32; 8];
            let c = [1.0f32; 8];
            let output = fma_operation(token, &a, &b, &c);
            // 2 * 3 + 1 = 7
            assert_eq!(output, [7.0f32; 8]);
        }
    }

    /// Test with profile token (X64V3Token)
    #[simd_fn]
    fn profile_token_test(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
        let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };
        // Use both AVX2 and FMA instructions
        let squared = _mm256_mul_ps(v, v);
        let result = _mm256_fmadd_ps(v, v, squared); // v*v + v*v = 2*v*v
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), result) };
        out
    }

    #[test]
    fn test_simd_fn_profile_token() {
        if let Some(token) = X64V3Token::try_new() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let output = profile_token_test(token, &input);
            // 2 * v * v
            let expected: [f32; 8] = input.map(|x| 2.0 * x * x);
            assert_eq!(output, expected);
        }
    }

    /// Test with multiple parameters
    #[simd_fn]
    fn multi_param(token: Avx2Token, a: &[f32; 8], b: &[f32; 8], scale: f32) -> [f32; 8] {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr()) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr()) };
        let vs = _mm256_set1_ps(scale);
        let sum = _mm256_add_ps(va, vb);
        let result = _mm256_mul_ps(sum, vs);
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), result) };
        out
    }

    #[test]
    fn test_simd_fn_multi_param() {
        if let Some(token) = Avx2Token::try_new() {
            let a = [1.0f32; 8];
            let b = [2.0f32; 8];
            let output = multi_param(token, &a, &b, 3.0);
            // (1 + 2) * 3 = 9
            assert_eq!(output, [9.0f32; 8]);
        }
    }

    /// Test with return type that's not an array
    #[simd_fn]
    fn horizontal_sum(token: Avx2Token, data: &[f32; 8]) -> f32 {
        let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };
        // Horizontal add within 128-bit lanes
        let sum1 = _mm256_hadd_ps(v, v);
        let sum2 = _mm256_hadd_ps(sum1, sum1);
        // Extract and add the two 128-bit lane results
        let low = _mm256_castps256_ps128(sum2);
        let high = _mm256_extractf128_ps::<1>(sum2);
        let final_sum = _mm_add_ss(low, high);
        unsafe { _mm_cvtss_f32(final_sum) }
    }

    #[test]
    fn test_simd_fn_scalar_return() {
        if let Some(token) = Avx2Token::try_new() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let sum = horizontal_sum(token, &input);
            assert_eq!(sum, 36.0); // 1+2+3+4+5+6+7+8 = 36
        }
    }

    /// Test that value-based intrinsics are safe (no unsafe block needed in body)
    #[simd_fn]
    fn safe_value_ops(token: Avx2Token, a: __m256, b: __m256) -> __m256 {
        // All these are safe in target_feature context (Rust 1.92+)
        let sum = _mm256_add_ps(a, b);
        let product = _mm256_mul_ps(a, b);
        let blended = _mm256_blend_ps::<0b10101010>(sum, product);
        _mm256_shuffle_ps::<0b00_01_10_11>(blended, blended)
    }

    #[test]
    fn test_simd_fn_value_ops() {
        if let Some(token) = Avx2Token::try_new() {
            let a = unsafe { _mm256_set1_ps(1.0) };
            let b = unsafe { _mm256_set1_ps(2.0) };
            let _result = safe_value_ops(token, a, b);
            // Just verify it compiles and runs
        }
    }
}

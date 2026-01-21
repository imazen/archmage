//! Tests for the #[simd_fn] attribute macro.

#![allow(unused)]

#[cfg(target_arch = "x86_64")]
mod x86_tests {
    use archmage::{
        Avx2FmaToken, Avx2Token, Desktop64, HasAvx, HasAvx2, HasFma, Server64, SimdToken,
        X64V3Token, simd_fn,
    };
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

    // =====================================================================
    // Tests for impl Trait and generic type parameters
    // =====================================================================

    /// Test with impl Trait bound
    #[simd_fn]
    fn impl_trait_test(token: impl HasAvx2, data: &[f32; 8]) -> [f32; 8] {
        let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };
        let doubled = _mm256_add_ps(v, v);
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
        out
    }

    #[test]
    fn test_simd_fn_impl_trait() {
        if let Some(token) = Avx2Token::try_new() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let output = impl_trait_test(token, &input);
            assert_eq!(output, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        }
    }

    /// Test that impl Trait accepts different concrete tokens
    #[test]
    fn test_simd_fn_impl_trait_accepts_x64v3() {
        if let Some(token) = X64V3Token::try_new() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            // X64V3Token implements HasAvx2, so this should work
            let output = impl_trait_test(token, &input);
            assert_eq!(output, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        }
    }

    /// Test with generic type parameter (inline bounds)
    #[simd_fn]
    fn generic_inline_bounds<T: HasAvx2>(token: T, data: &[f32; 8]) -> [f32; 8] {
        let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };
        let doubled = _mm256_add_ps(v, v);
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
        out
    }

    #[test]
    fn test_simd_fn_generic_inline_bounds() {
        if let Some(token) = Avx2Token::try_new() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let output = generic_inline_bounds(token, &input);
            assert_eq!(output, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        }
    }

    /// Test with generic type parameter (where clause)
    #[simd_fn]
    fn generic_where_clause<T>(token: T, data: &[f32; 8]) -> [f32; 8]
    where
        T: HasAvx2,
    {
        let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };
        let doubled = _mm256_add_ps(v, v);
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
        out
    }

    #[test]
    fn test_simd_fn_generic_where_clause() {
        if let Some(token) = Avx2Token::try_new() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let output = generic_where_clause(token, &input);
            assert_eq!(output, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        }
    }

    /// Test with multiple trait bounds using impl Trait
    #[simd_fn]
    fn impl_trait_multi_bounds(
        token: impl HasAvx2 + HasFma,
        a: &[f32; 8],
        b: &[f32; 8],
        c: &[f32; 8],
    ) -> [f32; 8] {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr()) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr()) };
        let vc = unsafe { _mm256_loadu_ps(c.as_ptr()) };
        // a * b + c using FMA
        let result = _mm256_fmadd_ps(va, vb, vc);
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), result) };
        out
    }

    #[test]
    fn test_simd_fn_impl_trait_multi_bounds() {
        // X64V3Token provides both HasAvx2 and HasFma
        if let Some(token) = X64V3Token::try_new() {
            let a = [2.0f32; 8];
            let b = [3.0f32; 8];
            let c = [1.0f32; 8];
            let output = impl_trait_multi_bounds(token, &a, &b, &c);
            // 2 * 3 + 1 = 7
            assert_eq!(output, [7.0f32; 8]);
        }
    }

    /// Test with multiple trait bounds using generic type parameter
    #[simd_fn]
    fn generic_multi_bounds<T: HasAvx2 + HasFma>(
        token: T,
        a: &[f32; 8],
        b: &[f32; 8],
        c: &[f32; 8],
    ) -> [f32; 8] {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr()) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr()) };
        let vc = unsafe { _mm256_loadu_ps(c.as_ptr()) };
        let result = _mm256_fmadd_ps(va, vb, vc);
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), result) };
        out
    }

    #[test]
    fn test_simd_fn_generic_multi_bounds() {
        if let Some(token) = X64V3Token::try_new() {
            let a = [2.0f32; 8];
            let b = [3.0f32; 8];
            let c = [1.0f32; 8];
            let output = generic_multi_bounds(token, &a, &b, &c);
            assert_eq!(output, [7.0f32; 8]);
        }
    }

    /// Test using HasAvx (lower bound) with AVX2 token
    #[simd_fn]
    fn lower_bound_test(token: impl HasAvx, data: &[f32; 8]) -> [f32; 8] {
        let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };
        // AVX instruction (not AVX2)
        let doubled = _mm256_add_ps(v, v);
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
        out
    }

    #[test]
    fn test_simd_fn_lower_bound_accepts_higher_token() {
        // Avx2Token should work with HasAvx bound
        if let Some(token) = Avx2Token::try_new() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let output = lower_bound_test(token, &input);
            assert_eq!(output, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        }
    }

    // =====================================================================
    // Tests for friendly aliases (Desktop64, Server64)
    // =====================================================================

    /// Test Desktop64 alias with simd_fn macro
    #[simd_fn]
    fn desktop64_test(token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
        let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };
        // Use both AVX2 and FMA (Desktop64 = X64V3Token = AVX2+FMA+BMI2)
        let result = _mm256_fmadd_ps(v, v, v); // v*v + v
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), result) };
        out
    }

    #[test]
    fn test_simd_fn_desktop64_alias() {
        if let Some(token) = Desktop64::try_new() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let output = desktop64_test(token, &input);
            // v*v + v
            let expected: [f32; 8] = input.map(|x| x * x + x);
            assert_eq!(output, expected);
        }
    }

    /// Test that Desktop64 is interchangeable with X64V3Token
    #[test]
    fn test_desktop64_is_x64v3() {
        // Desktop64 is a type alias for X64V3Token, so they should be interchangeable
        if let Some(token) = Desktop64::try_new() {
            // Can pass Desktop64 to function expecting X64V3Token
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let output = profile_token_test(token, &input);
            let expected: [f32; 8] = input.map(|x| 2.0 * x * x);
            assert_eq!(output, expected);
        }
    }

    /// Test that Desktop64 works with impl HasAvx2 bounds
    #[test]
    fn test_desktop64_with_impl_trait() {
        if let Some(token) = Desktop64::try_new() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            // Desktop64 implements HasAvx2
            let output = impl_trait_test(token, &input);
            assert_eq!(output, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        }
    }

    /// Test Server64 alias (only runs on machines with AVX-512)
    #[simd_fn]
    fn server64_test(token: Server64, data: &[f32; 8]) -> [f32; 8] {
        // Server64 = X64V4Token = AVX-512, but we'll just use AVX2 ops for simplicity
        let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };
        let doubled = _mm256_add_ps(v, v);
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
        out
    }

    #[test]
    fn test_simd_fn_server64_alias() {
        // This test only runs on machines with AVX-512
        if let Some(token) = Server64::try_new() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let output = server64_test(token, &input);
            assert_eq!(output, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        }
    }

    // =====================================================================
    // Tests for summon() alias
    // =====================================================================

    /// Test that summon() works as an alias for try_new()
    #[test]
    fn test_summon_alias() {
        // summon() should behave identically to try_new()
        let via_try_new = Desktop64::try_new();
        let via_summon = Desktop64::summon();

        assert_eq!(via_try_new.is_some(), via_summon.is_some());

        if let Some(token) = Desktop64::summon() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let output = desktop64_test(token, &input);
            let expected: [f32; 8] = input.map(|x| x * x + x);
            assert_eq!(output, expected);
        }
    }

    // =========================================================================
    // Tests for _self = Type support (trait methods with self receivers)
    // =========================================================================

    use archmage::arcane;

    /// A simple wrapper type for testing self receiver support
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct SimdVec8([f32; 8]);

    impl SimdVec8 {
        fn new(data: [f32; 8]) -> Self {
            Self(data)
        }

        fn as_array(&self) -> &[f32; 8] {
            &self.0
        }
    }

    /// Trait with all three self receiver types
    trait SimdOps {
        fn double(&self, token: impl HasAvx2) -> Self;
        fn square(self, token: impl HasAvx2) -> Self;
        fn scale(&mut self, token: impl HasAvx2, factor: f32);
    }

    impl SimdOps for SimdVec8 {
        #[arcane(_self = SimdVec8)]
        fn double(&self, _token: impl HasAvx2) -> Self {
            let v = unsafe { _mm256_loadu_ps(_self.0.as_ptr()) };
            let doubled = _mm256_add_ps(v, v);
            let mut out = [0.0f32; 8];
            unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
            SimdVec8(out)
        }

        #[arcane(_self = SimdVec8)]
        fn square(self, _token: impl HasAvx2) -> Self {
            let v = unsafe { _mm256_loadu_ps(_self.0.as_ptr()) };
            let squared = _mm256_mul_ps(v, v);
            let mut out = [0.0f32; 8];
            unsafe { _mm256_storeu_ps(out.as_mut_ptr(), squared) };
            SimdVec8(out)
        }

        #[arcane(_self = SimdVec8)]
        fn scale(&mut self, _token: impl HasAvx2, factor: f32) {
            let v = unsafe { _mm256_loadu_ps(_self.0.as_ptr()) };
            let scale = _mm256_set1_ps(factor);
            let scaled = _mm256_mul_ps(v, scale);
            unsafe { _mm256_storeu_ps(_self.0.as_mut_ptr(), scaled) };
        }
    }

    #[test]
    fn test_self_receiver_ref() {
        if let Some(token) = Desktop64::summon() {
            let v = SimdVec8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            let result = v.double(token);
            assert_eq!(
                result.as_array(),
                &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
            );
        }
    }

    #[test]
    fn test_self_receiver_owned() {
        if let Some(token) = Desktop64::summon() {
            let v = SimdVec8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            let result = v.square(token);
            assert_eq!(
                result.as_array(),
                &[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]
            );
        }
    }

    #[test]
    fn test_self_receiver_mut_ref() {
        if let Some(token) = Desktop64::summon() {
            let mut v = SimdVec8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            v.scale(token, 2.0);
            assert_eq!(v.as_array(), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        }
    }
}

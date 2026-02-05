//! Tests for the #[arcane] attribute macro.

#![allow(unused)]

#[cfg(target_arch = "x86_64")]
mod x86_tests {
    #[cfg(feature = "avx512")]
    use archmage::Avx512Token;
    use archmage::{Avx2FmaToken, Desktop64, Has256BitSimd, SimdToken, X64V3Token, arcane};
    use std::arch::x86_64::*;

    /// Basic test: arcane with X64V3Token
    #[arcane]
    fn double_values(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
        let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };
        let doubled = _mm256_add_ps(v, v);
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
        out
    }

    #[test]
    fn test_arcane_basic() {
        if let Some(token) = X64V3Token::summon() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let output = double_values(token, &input);
            assert_eq!(output, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        }
    }

    /// Test with FMA token (Avx2FmaToken is now alias for X64V3Token)
    #[arcane]
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
    fn test_arcane_fma() {
        if let Some(token) = Avx2FmaToken::summon() {
            let a = [2.0f32; 8];
            let b = [3.0f32; 8];
            let c = [1.0f32; 8];
            let output = fma_operation(token, &a, &b, &c);
            // 2 * 3 + 1 = 7
            assert_eq!(output, [7.0f32; 8]);
        }
    }

    /// Test with profile token (X64V3Token)
    #[arcane]
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
    fn test_arcane_profile_token() {
        if let Some(token) = X64V3Token::summon() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let output = profile_token_test(token, &input);
            // 2 * v * v
            let expected: [f32; 8] = input.map(|x| 2.0 * x * x);
            assert_eq!(output, expected);
        }
    }

    /// Test with multiple parameters
    #[arcane]
    fn multi_param(token: X64V3Token, a: &[f32; 8], b: &[f32; 8], scale: f32) -> [f32; 8] {
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
    fn test_arcane_multi_param() {
        if let Some(token) = X64V3Token::summon() {
            let a = [1.0f32; 8];
            let b = [2.0f32; 8];
            let output = multi_param(token, &a, &b, 3.0);
            // (1 + 2) * 3 = 9
            assert_eq!(output, [9.0f32; 8]);
        }
    }

    /// Test with return type that's not an array
    #[arcane]
    fn horizontal_sum(token: X64V3Token, data: &[f32; 8]) -> f32 {
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
    fn test_arcane_scalar_return() {
        if let Some(token) = X64V3Token::summon() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let sum = horizontal_sum(token, &input);
            assert_eq!(sum, 36.0); // 1+2+3+4+5+6+7+8 = 36
        }
    }

    /// Test that value-based intrinsics are safe (no unsafe block needed in body)
    #[arcane]
    fn safe_value_ops(token: X64V3Token, a: __m256, b: __m256) -> __m256 {
        // All these are safe in target_feature context (Rust 1.92+)
        let sum = _mm256_add_ps(a, b);
        let product = _mm256_mul_ps(a, b);
        let blended = _mm256_blend_ps::<0b10101010>(sum, product);
        _mm256_shuffle_ps::<0b00_01_10_11>(blended, blended)
    }

    #[test]
    fn test_arcane_value_ops() {
        if let Some(token) = X64V3Token::summon() {
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
    #[arcane]
    fn impl_trait_test(token: impl Has256BitSimd, data: &[f32; 8]) -> [f32; 8] {
        let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };
        let doubled = _mm256_add_ps(v, v);
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
        out
    }

    #[test]
    fn test_arcane_impl_trait() {
        if let Some(token) = X64V3Token::summon() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let output = impl_trait_test(token, &input);
            assert_eq!(output, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        }
    }

    /// Test that impl Trait accepts different concrete tokens
    #[test]
    fn test_arcane_impl_trait_accepts_x64v3() {
        if let Some(token) = X64V3Token::summon() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            // X64V3Token implements Has256BitSimd, so this should work
            let output = impl_trait_test(token, &input);
            assert_eq!(output, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        }
    }

    /// Test with generic type parameter (inline bounds)
    #[arcane]
    fn generic_inline_bounds<T: Has256BitSimd>(token: T, data: &[f32; 8]) -> [f32; 8] {
        let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };
        let doubled = _mm256_add_ps(v, v);
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
        out
    }

    #[test]
    fn test_arcane_generic_inline_bounds() {
        if let Some(token) = X64V3Token::summon() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let output = generic_inline_bounds(token, &input);
            assert_eq!(output, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        }
    }

    /// Test with generic type parameter (where clause)
    #[arcane]
    fn generic_where_clause<T>(token: T, data: &[f32; 8]) -> [f32; 8]
    where
        T: Has256BitSimd,
    {
        let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };
        let doubled = _mm256_add_ps(v, v);
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
        out
    }

    #[test]
    fn test_arcane_generic_where_clause() {
        if let Some(token) = X64V3Token::summon() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let output = generic_where_clause(token, &input);
            assert_eq!(output, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        }
    }

    /// Test with multiple trait bounds using Avx2FmaToken
    #[arcane]
    fn impl_trait_multi_bounds(
        token: Avx2FmaToken,
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
    fn test_arcane_impl_trait_multi_bounds() {
        // Avx2FmaToken provides AVX2 + FMA
        if let Some(token) = Avx2FmaToken::summon() {
            let a = [2.0f32; 8];
            let b = [3.0f32; 8];
            let c = [1.0f32; 8];
            let output = impl_trait_multi_bounds(token, &a, &b, &c);
            // 2 * 3 + 1 = 7
            assert_eq!(output, [7.0f32; 8]);
        }
    }

    /// Test with Avx2FmaToken (provides both 256-bit SIMD and FMA)
    #[arcane]
    fn generic_multi_bounds(
        token: Avx2FmaToken,
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
    fn test_arcane_generic_multi_bounds() {
        if let Some(token) = Avx2FmaToken::summon() {
            let a = [2.0f32; 8];
            let b = [3.0f32; 8];
            let c = [1.0f32; 8];
            let output = generic_multi_bounds(token, &a, &b, &c);
            assert_eq!(output, [7.0f32; 8]);
        }
    }

    /// Test using Has256BitSimd (lower bound) with X64V3Token
    #[arcane]
    fn lower_bound_test(token: impl Has256BitSimd, data: &[f32; 8]) -> [f32; 8] {
        let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };
        // AVX instruction (256-bit)
        let doubled = _mm256_add_ps(v, v);
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
        out
    }

    #[test]
    fn test_arcane_lower_bound_accepts_higher_token() {
        // X64V3Token should work with Has256BitSimd bound
        if let Some(token) = X64V3Token::summon() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let output = lower_bound_test(token, &input);
            assert_eq!(output, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        }
    }

    // =====================================================================
    // Tests for friendly aliases (Desktop64, Avx512Token)
    // =====================================================================

    /// Test Desktop64 alias with arcane macro
    #[arcane]
    fn desktop64_test(token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
        let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };
        // Use both AVX2 and FMA (Desktop64 = X64V3Token = AVX2+FMA+BMI2)
        let result = _mm256_fmadd_ps(v, v, v); // v*v + v
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), result) };
        out
    }

    #[test]
    fn test_arcane_desktop64_alias() {
        if let Some(token) = Desktop64::summon() {
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
        if let Some(token) = Desktop64::summon() {
            // Can pass Desktop64 to function expecting X64V3Token
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let output = profile_token_test(token, &input);
            let expected: [f32; 8] = input.map(|x| 2.0 * x * x);
            assert_eq!(output, expected);
        }
    }

    /// Test that Desktop64 works with impl Has256BitSimd bounds
    #[test]
    fn test_desktop64_with_impl_trait() {
        if let Some(token) = Desktop64::summon() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            // Desktop64 implements Has256BitSimd
            let output = impl_trait_test(token, &input);
            assert_eq!(output, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        }
    }

    /// Test Avx512Token alias (only runs on machines with AVX-512)
    #[cfg(feature = "avx512")]
    #[arcane]
    fn server64_test(token: Avx512Token, data: &[f32; 8]) -> [f32; 8] {
        // Avx512Token = X64V4Token = AVX-512, but we'll just use AVX2 ops for simplicity
        let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };
        let doubled = _mm256_add_ps(v, v);
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
        out
    }

    #[cfg(feature = "avx512")]
    #[test]
    fn test_arcane_server64_alias() {
        // This test only runs on machines with AVX-512
        if let Some(token) = Avx512Token::summon() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let output = server64_test(token, &input);
            assert_eq!(output, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        }
    }

    // =====================================================================
    // Tests for summon() alias
    // =====================================================================

    /// Test that summon() works as an alias for summon()
    #[test]
    fn test_summon_alias() {
        // summon() should behave identically to summon()
        let via_summon = Desktop64::summon();
        let via_summon = Desktop64::summon();

        assert_eq!(via_summon.is_some(), via_summon.is_some());

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

    // arcane already imported

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
        fn double(&self, token: impl Has256BitSimd) -> Self;
        fn square(self, token: impl Has256BitSimd) -> Self;
        fn scale(&mut self, token: impl Has256BitSimd, factor: f32);
    }

    impl SimdOps for SimdVec8 {
        #[arcane(_self = SimdVec8)]
        fn double(&self, _token: impl Has256BitSimd) -> Self {
            let v = unsafe { _mm256_loadu_ps(_self.0.as_ptr()) };
            let doubled = _mm256_add_ps(v, v);
            let mut out = [0.0f32; 8];
            unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
            SimdVec8(out)
        }

        #[arcane(_self = SimdVec8)]
        fn square(self, _token: impl Has256BitSimd) -> Self {
            let v = unsafe { _mm256_loadu_ps(_self.0.as_ptr()) };
            let squared = _mm256_mul_ps(v, v);
            let mut out = [0.0f32; 8];
            unsafe { _mm256_storeu_ps(out.as_mut_ptr(), squared) };
            SimdVec8(out)
        }

        #[arcane(_self = SimdVec8)]
        fn scale(&mut self, _token: impl Has256BitSimd, factor: f32) {
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

// =============================================================================
// Cross-architecture stub tests
// =============================================================================
// Verify that #[arcane] with wrong-arch tokens compiles (generates stub)

#[cfg(target_arch = "x86_64")]
mod cross_arch_stub_tests {
    use archmage::{arcane, NeonToken, SimdToken};

    /// This function uses an ARM token on x86 - should compile to a stub
    #[arcane]
    fn arm_function_on_x86(_token: NeonToken, data: &[f32]) -> f32 {
        // This body uses no intrinsics so it would compile either way,
        // but the key is that #[arcane] generates a stub instead of
        // trying to enable neon features on x86
        data.iter().sum()
    }

    #[test]
    fn stub_function_compiles() {
        // We can't call the function (NeonToken::summon() returns None on x86)
        // but the code compiles, which is the point
        assert!(NeonToken::summon().is_none());
    }
}

#[cfg(target_arch = "aarch64")]
mod cross_arch_stub_tests_arm {
    use archmage::{arcane, X64V3Token, SimdToken};

    /// This function uses an x86 token on ARM - should compile to a stub
    #[arcane]
    fn x86_function_on_arm(_token: X64V3Token, data: &[f32]) -> f32 {
        data.iter().sum()
    }

    #[test]
    fn stub_function_compiles() {
        assert!(X64V3Token::summon().is_none());
    }
}

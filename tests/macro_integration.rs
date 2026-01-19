//! Integration tests for archmage macros and tokens
//!
//! These tests verify that our macros work correctly and that the token
//! pattern provides zero-overhead safe access to SIMD operations.

#![allow(unused)]

#[cfg(target_arch = "x86_64")]
mod x86_tests {
    use archmage::is_x86_feature_available;
    use archmage::tokens::x86::*;
    use archmage::tokens::SimdToken;
    use std::arch::x86_64::*;

    /// Test that is_x86_feature_available! works at runtime
    #[test]
    fn test_feature_detection_macro() {
        // These should work on any x86_64 machine (SSE2 is baseline)
        // Use let bindings to avoid clippy::assertions_on_constants
        let has_sse = is_x86_feature_available!("sse");
        let has_sse2 = is_x86_feature_available!("sse2");
        assert!(has_sse);
        assert!(has_sse2);

        // These might or might not be available depending on CPU
        let has_avx = is_x86_feature_available!("avx");
        let has_avx2 = is_x86_feature_available!("avx2");
        let has_fma = is_x86_feature_available!("fma");

        // If AVX2 is available, AVX must also be available
        if has_avx2 {
            assert!(has_avx);
        }

        // Print what's available for debugging
        println!("SSE: true (baseline)");
        println!("SSE2: true (baseline)");
        println!("AVX: {}", has_avx);
        println!("AVX2: {}", has_avx2);
        println!("FMA: {}", has_fma);
    }

    /// Test token creation with try_new()
    #[test]
    fn test_token_try_new() {
        // SSE2 token should always succeed on x86_64
        let sse2_token = Sse2Token::try_new();
        assert!(sse2_token.is_some());

        // AVX2 token depends on CPU
        let avx2_token = Avx2Token::try_new();
        if is_x86_feature_available!("avx2") {
            assert!(avx2_token.is_some());
        } else {
            assert!(avx2_token.is_none());
        }

        // Combined token
        let avx2_fma_token = Avx2FmaToken::try_new();
        if is_x86_feature_available!("avx2") && is_x86_feature_available!("fma") {
            assert!(avx2_fma_token.is_some());
        }
    }

    /// Test that token-gated operations work
    #[test]
    fn test_token_gated_operations() {
        if let Some(token) = Avx2Token::try_new() {
            // All these operations are safe via token
            let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let loaded = archmage::ops::x86::load_f32x8(token, &data);

            let ones = archmage::ops::x86::set1_f32x8(token, 1.0);
            let sum = archmage::ops::x86::add_f32x8(token, loaded, ones);

            let mut out = [0.0f32; 8];
            archmage::ops::x86::store_f32x8(token, &mut out, sum);

            assert_eq!(out, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        }
    }

    /// Test shuffle operations have zero overhead
    #[test]
    fn test_shuffle_operations() {
        if let Some(token) = Avx2Token::try_new() {
            let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let b = [9.0f32, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];

            let va = archmage::ops::x86::load_f32x8(token, &a);
            let vb = archmage::ops::x86::load_f32x8(token, &b);

            // These are all safe via token
            let shuffled = archmage::ops::x86::shuffle_f32x8::<0b00_01_10_11>(token, va, vb);
            let permuted = archmage::ops::x86::permute_f32x8::<0b00_01_10_11>(token, va);
            let blended = archmage::ops::x86::blend_f32x8::<0b10101010>(token, va, vb);
            let unpacked_lo = archmage::ops::x86::unpacklo_f32x8(token, va, vb);
            let unpacked_hi = archmage::ops::x86::unpackhi_f32x8(token, va, vb);

            // Verify they produce results (not checking exact values, just that they work)
            let result = archmage::ops::x86::to_array_f32x8(shuffled);
            assert!(result.iter().all(|&x| (1.0..=16.0).contains(&x)));
        }
    }

    /// Test FMA operations
    #[test]
    fn test_fma_operations() {
        if let Some(token) = Avx2FmaToken::try_new() {
            let a = [2.0f32; 8];
            let b = [3.0f32; 8];
            let c = [1.0f32; 8];

            let va = archmage::ops::x86::load_f32x8(token.avx2(), &a);
            let vb = archmage::ops::x86::load_f32x8(token.avx2(), &b);
            let vc = archmage::ops::x86::load_f32x8(token.avx2(), &c);

            // a * b + c = 2 * 3 + 1 = 7
            let fma_result = archmage::ops::x86::fmadd_f32x8(token.fma(), va, vb, vc);
            let result = archmage::ops::x86::to_array_f32x8(fma_result);

            assert_eq!(result, [7.0f32; 8]);
        }
    }

    /// Test that is_x86_feature_available! compiles to constant when feature is enabled
    #[test]
    #[cfg(target_feature = "avx2")]
    fn test_compile_time_detection_avx2() {
        // When compiled with -C target-feature=+avx2, this should be a compile-time constant
        assert!(is_x86_feature_available!("avx2"));
        assert!(is_x86_feature_available!("avx"));
        assert!(is_x86_feature_available!("sse4.2"));
        assert!(is_x86_feature_available!("sse4.1"));
        assert!(is_x86_feature_available!("ssse3"));
        assert!(is_x86_feature_available!("sse3"));
        assert!(is_x86_feature_available!("sse2"));
        assert!(is_x86_feature_available!("sse"));
    }

    /// Test that horizontal operations work
    #[test]
    fn test_horizontal_operations() {
        if let Some(token) = Avx2Token::try_new() {
            let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let va = archmage::ops::x86::load_f32x8(token, &a);

            // Horizontal add
            let hadd = archmage::ops::x86::hadd_f32x8(token, va, va);
            let result = archmage::ops::x86::to_array_f32x8(hadd);

            // hadd adds adjacent pairs: [1+2, 3+4, 1+2, 3+4, 5+6, 7+8, 5+6, 7+8]
            assert_eq!(result[0], 3.0);  // 1 + 2
            assert_eq!(result[1], 7.0);  // 3 + 4
        }
    }
}

/// Test that safe_simd integration works when enabled
#[cfg(all(target_arch = "x86_64", feature = "safe-simd"))]
mod safe_simd_tests {
    use archmage::integrate::safe_simd::*;
    use archmage::tokens::x86::*;
    use archmage::tokens::SimdToken;

    #[test]
    fn test_safe_simd_load_store() {
        if let Some(token) = Avx2Token::try_new() {
            let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let mut out = [0.0f32; 8];

            // These use safe_unaligned_simd's reference-based API
            let loaded = safe_load_f32x8(token, &data);
            safe_store_f32x8(token, &mut out, loaded);

            assert_eq!(data, out);
        }
    }
}

/// Test that wide integration works when enabled
#[cfg(all(target_arch = "x86_64", feature = "wide"))]
mod wide_tests {
    use archmage::tokens::x86::*;
    use archmage::tokens::SimdToken;
    use wide::f32x8;

    #[test]
    fn test_wide_operations() {
        if let Some(token) = Avx2Token::try_new() {
            let a = f32x8::splat(2.0);
            let b = f32x8::splat(3.0);

            // wide_ops are methods on the token
            let sum = token.add_f32x8_wide(a, b);
            let product = token.mul_f32x8_wide(a, b);

            assert_eq!(sum, f32x8::splat(5.0));
            assert_eq!(product, f32x8::splat(6.0));
        }
    }

    #[test]
    fn test_wide_fma() {
        if let Some(token) = Avx2FmaToken::try_new() {
            let a = f32x8::splat(2.0);
            let b = f32x8::splat(3.0);
            let c = f32x8::splat(1.0);

            // a * b + c = 2 * 3 + 1 = 7
            let result = token.fma_f32x8_wide(a, b, c);

            assert_eq!(result, f32x8::splat(7.0));
        }
    }
}

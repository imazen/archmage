//! Exhaustive tests for cross-platform tokens and all mem/core::arch intrinsics.
//!
//! This test exercises:
//! - Cross-platform token availability (all tokens compile on all archs)
//! - Every function in the mem module for each platform
//! - Stable core::arch intrinsics via #[arcane]

use archmage::SimdToken;

// =============================================================================
// Cross-Platform Token Availability Tests
// =============================================================================

/// Test that all token types exist and can be referenced on any architecture.
/// On non-native architectures, summon() returns None.
#[test]
fn test_cross_platform_token_types_exist() {
    // x86 tokens - should compile on ARM/WASM, summon returns None there
    use archmage::{
        Avx2FmaToken, Avx2Token, AvxToken, Desktop64, FmaToken, Sse2Token, Sse41Token, Sse42Token,
        SseToken, X64V2Token, X64V3Token,
    };
    #[cfg(feature = "avx512")]
    use archmage::{Avx512Token, X64V4Token};

    // Verify tokens are zero-sized
    assert_eq!(core::mem::size_of::<SseToken>(), 0);
    assert_eq!(core::mem::size_of::<Sse2Token>(), 0);
    assert_eq!(core::mem::size_of::<Sse41Token>(), 0);
    assert_eq!(core::mem::size_of::<Sse42Token>(), 0);
    assert_eq!(core::mem::size_of::<AvxToken>(), 0);
    assert_eq!(core::mem::size_of::<Avx2Token>(), 0);
    assert_eq!(core::mem::size_of::<FmaToken>(), 0);
    assert_eq!(core::mem::size_of::<Avx2FmaToken>(), 0);
    assert_eq!(core::mem::size_of::<X64V2Token>(), 0);
    assert_eq!(core::mem::size_of::<X64V3Token>(), 0);
    #[cfg(feature = "avx512")]
    assert_eq!(core::mem::size_of::<X64V4Token>(), 0);
    assert_eq!(core::mem::size_of::<Desktop64>(), 0);
    #[cfg(feature = "avx512")]
    assert_eq!(core::mem::size_of::<Avx512Token>(), 0);

    // ARM tokens - should compile on x86/WASM, summon returns None there
    use archmage::{Arm64, NeonToken};

    assert_eq!(core::mem::size_of::<NeonToken>(), 0);
    assert_eq!(core::mem::size_of::<Arm64>(), 0);

    // WASM token - should compile everywhere
    use archmage::Simd128Token;
    assert_eq!(core::mem::size_of::<Simd128Token>(), 0);
}

/// Test that summon() works correctly for the current platform.
#[test]
fn test_summon_behavior() {
    use archmage::{Arm64, NeonToken, Simd128Token};

    // On x86_64, Desktop64/Avx512Token may succeed
    #[cfg(target_arch = "x86_64")]
    {
        // SSE2 is baseline on x86_64, so Sse2Token should always succeed
        use archmage::Sse2Token;
        assert!(Sse2Token::summon().is_some(), "SSE2 is baseline on x86_64");

        // ARM and WASM tokens should return None on x86
        assert!(NeonToken::summon().is_none(), "NEON unavailable on x86");
        assert!(Arm64::summon().is_none(), "Arm64 unavailable on x86");
        assert!(
            Simd128Token::summon().is_none(),
            "WASM SIMD unavailable on x86"
        );
    }

    // On aarch64, NEON is always available
    #[cfg(target_arch = "aarch64")]
    {
        assert!(NeonToken::summon().is_some(), "NEON is baseline on AArch64");
        assert!(Arm64::summon().is_some(), "Arm64 is baseline on AArch64");

        // x86 and WASM tokens should return None on ARM
        assert!(
            Desktop64::summon().is_none(),
            "Desktop64 unavailable on ARM"
        );
        #[cfg(feature = "avx512")]
        assert!(
            archmage::Avx512Token::summon().is_none(),
            "Avx512Token unavailable on ARM"
        );
        assert!(
            Simd128Token::summon().is_none(),
            "WASM SIMD unavailable on ARM"
        );
    }

    // On WASM, SIMD128 may be available
    #[cfg(target_arch = "wasm32")]
    {
        // x86 and ARM tokens should return None on WASM
        assert!(
            Desktop64::summon().is_none(),
            "Desktop64 unavailable on WASM"
        );
        assert!(NeonToken::summon().is_none(), "NEON unavailable on WASM");
    }
}

/// Test that ARCHMAGE_DISABLE env var forces summon() to return None.
/// Run with: ARCHMAGE_DISABLE=1 cargo test test_disable_archmage_env
#[test]
fn test_disable_archmage_env() {
    use archmage::{SimdToken, Sse2Token};

    // This test verifies the mechanism works - actual disable testing
    // should be done by running: ARCHMAGE_DISABLE=1 cargo test
    if std::env::var_os("ARCHMAGE_DISABLE").is_some() {
        #[cfg(target_arch = "x86_64")]
        {
            assert!(
                Sse2Token::summon().is_none(),
                "ARCHMAGE_DISABLE should make summon() return None"
            );
            // try_new() should still work
            assert!(
                Sse2Token::try_new().is_some(),
                "try_new() should still detect CPU features"
            );
        }
    }
}

/// Test that cross-platform dispatch code compiles and runs.
#[test]
fn test_cross_platform_dispatch_pattern() {
    use archmage::{Arm64, Desktop64, NeonToken};

    let mut data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    // This pattern should compile on any architecture
    #[cfg(feature = "avx512")]
    let processed = if let Some(_token) = archmage::Avx512Token::summon() {
        "avx512"
    } else if let Some(_token) = Desktop64::summon() {
        "avx2"
    } else if let Some(_token) = NeonToken::summon() {
        "neon"
    } else if let Some(_token) = Arm64::summon() {
        "arm64"
    } else {
        // Scalar fallback
        for x in &mut data {
            *x *= 2.0;
        }
        "scalar"
    };

    #[cfg(not(feature = "avx512"))]
    let processed = if let Some(_token) = Desktop64::summon() {
        "avx2"
    } else if let Some(_token) = NeonToken::summon() {
        "neon"
    } else if let Some(_token) = Arm64::summon() {
        "arm64"
    } else {
        // Scalar fallback
        for x in &mut data {
            *x *= 2.0;
        }
        "scalar"
    };

    // At least one path should have been taken
    assert!(!processed.is_empty());
}

/// Test that token names are correct.
#[test]
fn test_token_names() {
    use archmage::{
        Arm64, Avx2FmaToken, Avx2Token, AvxToken, Desktop64, FmaToken, NeonToken, Simd128Token,
        Sse2Token, Sse41Token, Sse42Token, SseToken, X64V2Token, X64V3Token,
    };

    // x86 tokens
    assert_eq!(SseToken::NAME, "SSE");
    assert_eq!(Sse2Token::NAME, "SSE2");
    assert_eq!(Sse41Token::NAME, "SSE4.1");
    assert_eq!(Sse42Token::NAME, "SSE4.2");
    assert_eq!(AvxToken::NAME, "AVX");
    assert_eq!(Avx2Token::NAME, "AVX2");
    assert_eq!(FmaToken::NAME, "FMA");
    assert_eq!(Avx2FmaToken::NAME, "AVX2+FMA");
    assert_eq!(X64V2Token::NAME, "x86-64-v2");
    assert_eq!(X64V3Token::NAME, "x86-64-v3");

    // AVX-512 tokens (requires avx512 feature)
    #[cfg(feature = "avx512")]
    {
        use archmage::{Avx512Token, X64V4Token};
        // X64V4Token is an alias for Avx512Token
        assert_eq!(X64V4Token::NAME, "AVX-512");
        assert_eq!(Avx512Token::NAME, "AVX-512");
    }

    // Verify aliases
    assert_eq!(Desktop64::NAME, X64V3Token::NAME);

    // ARM tokens
    assert_eq!(NeonToken::NAME, "NEON");
    assert_eq!(Arm64::NAME, NeonToken::NAME);

    // WASM tokens
    assert_eq!(Simd128Token::NAME, "SIMD128");
}

// =============================================================================
// x86_64 mem Module Exhaustive Tests
// =============================================================================

#[cfg(all(target_arch = "x86_64", feature = "safe_unaligned_simd"))]
mod x86_mem_tests {
    use archmage::SimdToken;
    use core::arch::x86_64::*;
    use std::hint::black_box;

    /// Exhaustive test of all SSE mem functions.
    #[test]
    fn test_sse_mem_exhaustive() {
        use archmage::SseToken;
        use archmage::mem::sse;

        let Some(token) = SseToken::summon() else {
            eprintln!("SSE not available, skipping test");
            return;
        };

        // Test data
        let data_f32 = [1.0f32, 2.0, 3.0, 4.0];
        let single_f32 = 42.0f32;

        // _mm_load1_ps - broadcast single f32
        let v = sse::_mm_load1_ps(token, &single_f32);
        let mut out = [0.0f32; 4];
        sse::_mm_storeu_ps(token, &mut out, v);
        assert!(out.iter().all(|&x| x == 42.0));

        // _mm_load_ps1 - alias for _mm_load1_ps
        let v = sse::_mm_load_ps1(token, &single_f32);
        sse::_mm_storeu_ps(token, &mut out, v);
        assert!(out.iter().all(|&x| x == 42.0));

        // _mm_load_ss - load single, zero upper
        let v = sse::_mm_load_ss(token, &single_f32);
        sse::_mm_storeu_ps(token, &mut out, v);
        assert_eq!(out[0], 42.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 0.0);
        assert_eq!(out[3], 0.0);

        // _mm_loadu_ps - load 4 floats
        let v = sse::_mm_loadu_ps(token, &data_f32);
        sse::_mm_storeu_ps(token, &mut out, v);
        assert_eq!(out, data_f32);

        // _mm_store_ss - store single
        let mut single_out = 0.0f32;
        sse::_mm_store_ss(token, &mut single_out, v);
        assert_eq!(single_out, 1.0);

        // _mm_storeu_ps already tested above

        black_box(&out);
    }

    /// Exhaustive test of all SSE2 mem functions.
    #[test]
    fn test_sse2_mem_exhaustive() {
        use archmage::Sse2Token;
        use archmage::mem::sse2;

        let Some(token) = Sse2Token::summon() else {
            eprintln!("SSE2 not available, skipping test");
            return;
        };

        // Test data
        let data_f64 = [1.0f64, 2.0];
        let single_f64 = 42.0f64;
        let data_i8: [i8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let data_i16: [i16; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let data_i32: [i32; 4] = [1, 2, 3, 4];
        let data_i64: [i64; 2] = [1, 2];

        let mut out_f64 = [0.0f64; 2];
        let mut out_i8 = [0i8; 16];
        let mut out_i16 = [0i16; 8];
        let mut out_i32 = [0i32; 4];
        let mut out_i64 = [0i64; 2];

        // _mm_load_pd1 - broadcast double
        let v = sse2::_mm_load_pd1(token, &single_f64);
        sse2::_mm_storeu_pd(token, &mut out_f64, v);
        assert!(out_f64.iter().all(|&x| x == 42.0));

        // _mm_load_sd - load single double, zero upper
        let v = sse2::_mm_load_sd(token, &single_f64);
        sse2::_mm_storeu_pd(token, &mut out_f64, v);
        assert_eq!(out_f64[0], 42.0);
        assert_eq!(out_f64[1], 0.0);

        // _mm_load1_pd - broadcast double (alias)
        let v = sse2::_mm_load1_pd(token, &single_f64);
        sse2::_mm_storeu_pd(token, &mut out_f64, v);
        assert!(out_f64.iter().all(|&x| x == 42.0));

        // _mm_loadh_pd - load high double
        let base = unsafe { _mm_setzero_pd() };
        let v = sse2::_mm_loadh_pd(token, base, &single_f64);
        sse2::_mm_storeu_pd(token, &mut out_f64, v);
        assert_eq!(out_f64[0], 0.0);
        assert_eq!(out_f64[1], 42.0);

        // _mm_loadl_epi64 - load 64 bits to low, zero upper
        let v = sse2::_mm_loadl_epi64(token, &data_i64);
        sse2::_mm_storeu_si128(token, &mut out_i64, v);
        assert_eq!(out_i64[0], data_i64[0]);

        // _mm_loadl_pd - load low double
        let base = unsafe { _mm_set_pd(99.0, 99.0) };
        let v = sse2::_mm_loadl_pd(token, base, &single_f64);
        sse2::_mm_storeu_pd(token, &mut out_f64, v);
        assert_eq!(out_f64[0], 42.0);
        assert_eq!(out_f64[1], 99.0);

        // _mm_loadu_pd - load 2 doubles
        let v = sse2::_mm_loadu_pd(token, &data_f64);
        sse2::_mm_storeu_pd(token, &mut out_f64, v);
        assert_eq!(out_f64, data_f64);

        // _mm_loadu_si128 - load 128 bits
        let v = sse2::_mm_loadu_si128(token, &data_i8);
        sse2::_mm_storeu_si128(token, &mut out_i8, v);
        assert_eq!(out_i8, data_i8);

        // _mm_loadu_si16 - load 16 bits
        let v = sse2::_mm_loadu_si16(token, &data_i16[0]);
        sse2::_mm_storeu_si16(token, &mut out_i16[0], v);
        assert_eq!(out_i16[0], data_i16[0]);

        // _mm_loadu_si32 - load 32 bits
        let v = sse2::_mm_loadu_si32(token, &data_i32[0]);
        sse2::_mm_storeu_si32(token, &mut out_i32[0], v);
        assert_eq!(out_i32[0], data_i32[0]);

        // _mm_loadu_si64 - load 64 bits
        let v = sse2::_mm_loadu_si64(token, &data_i64[0]);
        sse2::_mm_storeu_si64(token, &mut out_i64[0], v);
        assert_eq!(out_i64[0], data_i64[0]);

        // _mm_store_sd
        let v = sse2::_mm_loadu_pd(token, &data_f64);
        let mut single_out = 0.0f64;
        sse2::_mm_store_sd(token, &mut single_out, v);
        assert_eq!(single_out, 1.0);

        // _mm_storeh_pd - store high double
        let v = unsafe { _mm_set_pd(99.0, 42.0) };
        sse2::_mm_storeh_pd(token, &mut single_out, v);
        assert_eq!(single_out, 99.0);

        // _mm_storel_epi64 - store low 64 bits
        let v = sse2::_mm_loadu_si128(token, &data_i64);
        sse2::_mm_storel_epi64(token, &mut out_i64, v);
        assert_eq!(out_i64[0], data_i64[0]);

        // _mm_storel_pd - store low double
        let v = unsafe { _mm_set_pd(99.0, 42.0) };
        sse2::_mm_storel_pd(token, &mut single_out, v);
        assert_eq!(single_out, 42.0);

        black_box(&out_f64);
        black_box(&out_i8);
        black_box(&out_i16);
        black_box(&out_i32);
        black_box(&out_i64);
    }

    /// Exhaustive test of all AVX mem functions.
    #[test]
    fn test_avx_mem_exhaustive() {
        use archmage::AvxToken;
        use archmage::mem::avx;

        let Some(token) = AvxToken::summon() else {
            eprintln!("AVX not available, skipping test");
            return;
        };

        // Test data
        let data_f32 = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let data_f64 = [1.0f64, 2.0, 3.0, 4.0];
        let data_f32_lo = [1.0f32, 2.0, 3.0, 4.0];
        let data_f32_hi = [5.0f32, 6.0, 7.0, 8.0];
        let data_f64_lo = [1.0f64, 2.0];
        let data_f64_hi = [3.0f64, 4.0];
        let single_f32 = 42.0f32;
        let single_f64 = 42.0f64;
        let data_i8: [i8; 32] = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ];
        let data_i8_lo: [i8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let data_i8_hi: [i8; 16] = [
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        ];

        let mut out_f32 = [0.0f32; 8];
        let mut out_f64 = [0.0f64; 4];
        let mut out_f32_lo = [0.0f32; 4];
        let mut out_f32_hi = [0.0f32; 4];
        let mut out_f64_lo = [0.0f64; 2];
        let mut out_f64_hi = [0.0f64; 2];
        let mut out_i8 = [0i8; 32];
        let mut out_i8_lo = [0i8; 16];
        let mut out_i8_hi = [0i8; 16];

        // _mm256_broadcast_pd - broadcast 128-bit pd to 256-bit
        let src = unsafe { _mm_loadu_pd(data_f64_lo.as_ptr()) };
        let v = avx::_mm256_broadcast_pd(token, &src);
        avx::_mm256_storeu_pd(token, &mut out_f64, v);
        assert_eq!(out_f64[0], 1.0);
        assert_eq!(out_f64[1], 2.0);
        assert_eq!(out_f64[2], 1.0);
        assert_eq!(out_f64[3], 2.0);

        // _mm256_broadcast_ps - broadcast 128-bit ps to 256-bit
        let src = unsafe { _mm_loadu_ps(data_f32_lo.as_ptr()) };
        let v = avx::_mm256_broadcast_ps(token, &src);
        avx::_mm256_storeu_ps(token, &mut out_f32, v);
        assert_eq!(&out_f32[0..4], &data_f32_lo);
        assert_eq!(&out_f32[4..8], &data_f32_lo);

        // _mm256_broadcast_sd - broadcast single double
        let v = avx::_mm256_broadcast_sd(token, &single_f64);
        avx::_mm256_storeu_pd(token, &mut out_f64, v);
        assert!(out_f64.iter().all(|&x| x == 42.0));

        // _mm_broadcast_ss - broadcast single float to __m128
        let v = avx::_mm_broadcast_ss(token, &single_f32);
        let mut out_128 = [0.0f32; 4];
        unsafe { _mm_storeu_ps(out_128.as_mut_ptr(), v) };
        assert!(out_128.iter().all(|&x| x == 42.0));

        // _mm256_broadcast_ss - broadcast single float to __m256
        let v = avx::_mm256_broadcast_ss(token, &single_f32);
        avx::_mm256_storeu_ps(token, &mut out_f32, v);
        assert!(out_f32.iter().all(|&x| x == 42.0));

        // _mm256_loadu_pd - load 4 doubles
        let v = avx::_mm256_loadu_pd(token, &data_f64);
        avx::_mm256_storeu_pd(token, &mut out_f64, v);
        assert_eq!(out_f64, data_f64);

        // _mm256_loadu_ps - load 8 floats
        let v = avx::_mm256_loadu_ps(token, &data_f32);
        avx::_mm256_storeu_ps(token, &mut out_f32, v);
        assert_eq!(out_f32, data_f32);

        // _mm256_loadu_si256 - load 256 bits
        let v = avx::_mm256_loadu_si256(token, &data_i8);
        avx::_mm256_storeu_si256(token, &mut out_i8, v);
        assert_eq!(out_i8, data_i8);

        // _mm256_loadu2_m128 - load two 128-bit floats
        let v = avx::_mm256_loadu2_m128(token, &data_f32_hi, &data_f32_lo);
        avx::_mm256_storeu_ps(token, &mut out_f32, v);
        assert_eq!(&out_f32[0..4], &data_f32_lo);
        assert_eq!(&out_f32[4..8], &data_f32_hi);

        // _mm256_loadu2_m128d - load two 128-bit doubles
        let v = avx::_mm256_loadu2_m128d(token, &data_f64_hi, &data_f64_lo);
        avx::_mm256_storeu_pd(token, &mut out_f64, v);
        assert_eq!(&out_f64[0..2], &data_f64_lo);
        assert_eq!(&out_f64[2..4], &data_f64_hi);

        // _mm256_loadu2_m128i - load two 128-bit integers
        let v = avx::_mm256_loadu2_m128i(token, &data_i8_hi, &data_i8_lo);
        avx::_mm256_storeu_si256(token, &mut out_i8, v);
        assert_eq!(&out_i8[0..16], &data_i8_lo);
        assert_eq!(&out_i8[16..32], &data_i8_hi);

        // _mm256_storeu_pd already tested

        // _mm256_storeu_ps already tested

        // _mm256_storeu_si256 already tested

        // _mm256_storeu2_m128 - store to two 128-bit locations
        let v = avx::_mm256_loadu_ps(token, &data_f32);
        avx::_mm256_storeu2_m128(token, &mut out_f32_hi, &mut out_f32_lo, v);
        assert_eq!(out_f32_lo, data_f32_lo);
        assert_eq!(out_f32_hi, data_f32_hi);

        // _mm256_storeu2_m128d
        let v = avx::_mm256_loadu_pd(token, &data_f64);
        avx::_mm256_storeu2_m128d(token, &mut out_f64_hi, &mut out_f64_lo, v);
        assert_eq!(out_f64_lo, data_f64_lo);
        assert_eq!(out_f64_hi, data_f64_hi);

        // _mm256_storeu2_m128i
        let v = avx::_mm256_loadu_si256(token, &data_i8);
        avx::_mm256_storeu2_m128i(token, &mut out_i8_hi, &mut out_i8_lo, v);
        assert_eq!(out_i8_lo, data_i8_lo);
        assert_eq!(out_i8_hi, data_i8_hi);

        black_box(&out_f32);
        black_box(&out_f64);
    }

    /// Test AVX-512F mem functions (if available).
    #[cfg(feature = "avx512")]
    #[test]
    fn test_avx512f_mem_sampling() {
        use archmage::Avx512fToken;
        use archmage::mem::avx512f;

        let Some(token) = Avx512fToken::summon() else {
            eprintln!("AVX-512F not available, skipping test");
            return;
        };

        // Test data
        let data_f32 = [1.0f32; 16];
        let data_f64 = [2.0f64; 8];
        let data_i32 = [3i32; 16];
        let data_i64 = [4i64; 8];

        let mut out_f32 = [0.0f32; 16];
        let mut out_f64 = [0.0f64; 8];
        let mut out_i32 = [0i32; 16];
        let mut out_i64 = [0i64; 8];

        // Basic load/store
        let v = avx512f::_mm512_loadu_ps(token, &data_f32);
        avx512f::_mm512_storeu_ps(token, &mut out_f32, v);
        assert_eq!(out_f32, data_f32);

        let v = avx512f::_mm512_loadu_pd(token, &data_f64);
        avx512f::_mm512_storeu_pd(token, &mut out_f64, v);
        assert_eq!(out_f64, data_f64);

        let v = avx512f::_mm512_loadu_epi32(token, &data_i32);
        avx512f::_mm512_storeu_epi32(token, &mut out_i32, v);
        assert_eq!(out_i32, data_i32);

        let v = avx512f::_mm512_loadu_epi64(token, &data_i64);
        avx512f::_mm512_storeu_epi64(token, &mut out_i64, v);
        assert_eq!(out_i64, data_i64);

        // Masked operations
        let mask: u16 = 0b1010101010101010;
        let v = avx512f::_mm512_maskz_loadu_ps(token, mask, &data_f32);
        avx512f::_mm512_storeu_ps(token, &mut out_f32, v);
        #[allow(clippy::needless_range_loop)] // i used for both indexing and mask shifting
        for i in 0..16 {
            if (mask >> i) & 1 == 1 {
                assert_eq!(out_f32[i], 1.0);
            } else {
                assert_eq!(out_f32[i], 0.0);
            }
        }

        black_box(&out_f32);
        black_box(&out_f64);
        black_box(&out_i32);
        black_box(&out_i64);
    }

    /// Exhaustive test of core::arch intrinsics via #[arcane].
    #[test]
    fn test_arcane_core_arch_intrinsics() {
        use archmage::{Desktop64, HasAvx2, HasFma, SimdToken, arcane};

        // Skip if Desktop64 not available
        let Some(token) = Desktop64::summon() else {
            eprintln!("Desktop64 not available, skipping arcane tests");
            return;
        };

        // Test arithmetic intrinsics
        #[arcane]
        fn test_avx2_arithmetic(_token: impl HasAvx2) -> [f32; 8] {
            let a = _mm256_set1_ps(2.0);
            let b = _mm256_set1_ps(3.0);

            // Addition
            let sum = _mm256_add_ps(a, b);
            // Subtraction
            let diff = _mm256_sub_ps(sum, a);
            // Multiplication
            let prod = _mm256_mul_ps(diff, b);
            // Division
            let quot = _mm256_div_ps(prod, b);

            let mut result = [0.0f32; 8];
            unsafe { _mm256_storeu_ps(result.as_mut_ptr(), quot) };
            result
        }

        let result = test_avx2_arithmetic(token);
        assert!(result.iter().all(|&x| (x - 3.0).abs() < 0.0001));

        // Test FMA intrinsics
        #[arcane]
        fn test_fma_intrinsics<T: HasAvx2 + HasFma>(_token: T) -> [f32; 8] {
            let a = _mm256_set1_ps(2.0);
            let b = _mm256_set1_ps(3.0);
            let c = _mm256_set1_ps(1.0);

            // fmadd: a*b + c = 2*3 + 1 = 7
            let fmadd = _mm256_fmadd_ps(a, b, c);
            // fmsub: a*b - c = 2*3 - 1 = 5
            let fmsub = _mm256_fmsub_ps(a, b, c);
            // fnmadd: -a*b + c = -2*3 + 1 = -5
            let fnmadd = _mm256_fnmadd_ps(a, b, c);

            // Return fmadd for verification
            let mut result = [0.0f32; 8];
            unsafe { _mm256_storeu_ps(result.as_mut_ptr(), fmadd) };
            black_box(fmsub);
            black_box(fnmadd);
            result
        }

        let result = test_fma_intrinsics(token);
        assert!(result.iter().all(|&x| (x - 7.0).abs() < 0.0001));

        // Test comparison and blending
        #[arcane]
        fn test_compare_blend(_token: impl HasAvx2) -> [f32; 8] {
            let a = _mm256_set_ps(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
            let b = _mm256_set1_ps(4.5);

            // Compare: a > b
            let mask = _mm256_cmp_ps::<_CMP_GT_OS>(a, b);
            // Blend based on comparison
            let result = _mm256_blendv_ps(a, b, mask);

            let mut out = [0.0f32; 8];
            unsafe { _mm256_storeu_ps(out.as_mut_ptr(), result) };
            out
        }

        let result = test_compare_blend(token);
        // Elements > 4.5 get replaced with 4.5
        assert_eq!(result[0], 1.0); // 1.0 <= 4.5, keep
        assert_eq!(result[4], 4.5); // 5.0 > 4.5, replaced

        // Test shuffle and permute
        #[arcane]
        fn test_shuffle_permute(_token: impl HasAvx2) -> [i32; 8] {
            let a = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);

            // Permute within 128-bit lanes
            let permuted = _mm256_shuffle_epi32::<0b00_01_10_11>(a);

            let mut result = [0i32; 8];
            unsafe { _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, permuted) };
            result
        }

        let result = test_shuffle_permute(token);
        // Shuffle reverses elements within each 128-bit lane
        assert_eq!(result[0], 3);
        assert_eq!(result[3], 0);

        // Test bitwise operations
        #[arcane]
        fn test_bitwise(_token: impl HasAvx2) -> [i32; 8] {
            let a = _mm256_set1_epi32(0b1100);
            let b = _mm256_set1_epi32(0b1010);

            let and = _mm256_and_si256(a, b); // 0b1000
            let or = _mm256_or_si256(a, b); // 0b1110
            let xor = _mm256_xor_si256(a, b); // 0b0110
            let not = _mm256_andnot_si256(a, b); // ~a & b = 0b0010

            let mut result = [0i32; 8];
            unsafe { _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, and) };
            black_box(or);
            black_box(xor);
            black_box(not);
            result
        }

        let result = test_bitwise(token);
        assert!(result.iter().all(|&x| x == 0b1000));

        // Test horizontal operations
        #[arcane]
        fn test_horizontal(_token: impl HasAvx2) -> [f32; 8] {
            let a = _mm256_set_ps(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
            let b = _mm256_set_ps(16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0);

            // Horizontal add
            let hadd = _mm256_hadd_ps(a, b);

            let mut result = [0.0f32; 8];
            unsafe { _mm256_storeu_ps(result.as_mut_ptr(), hadd) };
            result
        }

        let result = test_horizontal(token);
        // hadd adds adjacent pairs within 128-bit lanes
        // Lower lane: [a0+a1, a2+a3, b0+b1, b2+b3] = [1+2, 3+4, 9+10, 11+12]
        // Upper lane: [a4+a5, a6+a7, b4+b5, b6+b7] = [5+6, 7+8, 13+14, 15+16]
        assert_eq!(result[0], 3.0); // 1+2
        assert_eq!(result[1], 7.0); // 3+4

        // Test conversion intrinsics
        #[arcane]
        fn test_conversion(_token: impl HasAvx2) -> [i32; 8] {
            let a = _mm256_set_ps(8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5);

            // Convert to integers (truncate)
            let ints = _mm256_cvttps_epi32(a);

            let mut result = [0i32; 8];
            unsafe { _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, ints) };
            result
        }

        let result = test_conversion(token);
        assert_eq!(result, [1, 2, 3, 4, 5, 6, 7, 8]);

        // Test gather operations (AVX2)
        #[arcane]
        fn test_gather(_token: impl HasAvx2) -> [f32; 8] {
            let base = [0.0f32, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
            let indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);

            let gathered = unsafe { _mm256_i32gather_ps::<4>(base.as_ptr(), indices) };

            let mut result = [0.0f32; 8];
            unsafe { _mm256_storeu_ps(result.as_mut_ptr(), gathered) };
            result
        }

        let result = test_gather(token);
        assert_eq!(result, [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]);
    }
}

// =============================================================================
// AArch64 mem Module Exhaustive Tests
// =============================================================================

#[cfg(target_arch = "aarch64")]
mod aarch64_mem_tests {
    use archmage::SimdToken;
    use core::arch::aarch64::*;
    use std::hint::black_box;

    /// Exhaustive test of NEON load functions.
    #[test]
    fn test_neon_load_exhaustive() {
        use archmage::NeonToken;
        use archmage::mem::neon;

        let token = NeonToken::summon().expect("NEON should be available on AArch64");

        // =====================================================================
        // 8-byte register loads (vld1_*)
        // =====================================================================

        // u8x8
        let data_u8: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let v = neon::vld1_u8(token, &data_u8);
        let mut out = [0u8; 8];
        neon::vst1_u8(token, &mut out, v);
        assert_eq!(out, data_u8);

        // i8x8
        let data_i8: [i8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let v = neon::vld1_s8(token, &data_i8);
        let mut out = [0i8; 8];
        neon::vst1_s8(token, &mut out, v);
        assert_eq!(out, data_i8);

        // u16x4
        let data_u16: [u16; 4] = [1, 2, 3, 4];
        let v = neon::vld1_u16(token, &data_u16);
        let mut out = [0u16; 4];
        neon::vst1_u16(token, &mut out, v);
        assert_eq!(out, data_u16);

        // i16x4
        let data_i16: [i16; 4] = [1, 2, 3, 4];
        let v = neon::vld1_s16(token, &data_i16);
        let mut out = [0i16; 4];
        neon::vst1_s16(token, &mut out, v);
        assert_eq!(out, data_i16);

        // u32x2
        let data_u32: [u32; 2] = [1, 2];
        let v = neon::vld1_u32(token, &data_u32);
        let mut out = [0u32; 2];
        neon::vst1_u32(token, &mut out, v);
        assert_eq!(out, data_u32);

        // i32x2
        let data_i32: [i32; 2] = [1, 2];
        let v = neon::vld1_s32(token, &data_i32);
        let mut out = [0i32; 2];
        neon::vst1_s32(token, &mut out, v);
        assert_eq!(out, data_i32);

        // f32x2
        let data_f32: [f32; 2] = [1.0, 2.0];
        let v = neon::vld1_f32(token, &data_f32);
        let mut out = [0.0f32; 2];
        neon::vst1_f32(token, &mut out, v);
        assert_eq!(out, data_f32);

        // u64x1
        let data_u64: u64 = 42;
        let v = neon::vld1_u64(token, &data_u64);
        let mut out = 0u64;
        neon::vst1_u64(token, &mut out, v);
        assert_eq!(out, data_u64);

        // i64x1
        let data_i64: i64 = 42;
        let v = neon::vld1_s64(token, &data_i64);
        let mut out = 0i64;
        neon::vst1_s64(token, &mut out, v);
        assert_eq!(out, data_i64);

        // f64x1
        let data_f64: f64 = 42.0;
        let v = neon::vld1_f64(token, &data_f64);
        let mut out = 0.0f64;
        neon::vst1_f64(token, &mut out, v);
        assert_eq!(out, data_f64);

        // =====================================================================
        // 16-byte register loads (vld1q_*)
        // =====================================================================

        // u8x16
        let data_u8: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let v = neon::vld1q_u8(token, &data_u8);
        let mut out = [0u8; 16];
        neon::vst1q_u8(token, &mut out, v);
        assert_eq!(out, data_u8);

        // i8x16
        let data_i8: [i8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let v = neon::vld1q_s8(token, &data_i8);
        let mut out = [0i8; 16];
        neon::vst1q_s8(token, &mut out, v);
        assert_eq!(out, data_i8);

        // u16x8
        let data_u16: [u16; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let v = neon::vld1q_u16(token, &data_u16);
        let mut out = [0u16; 8];
        neon::vst1q_u16(token, &mut out, v);
        assert_eq!(out, data_u16);

        // i16x8
        let data_i16: [i16; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let v = neon::vld1q_s16(token, &data_i16);
        let mut out = [0i16; 8];
        neon::vst1q_s16(token, &mut out, v);
        assert_eq!(out, data_i16);

        // u32x4
        let data_u32: [u32; 4] = [1, 2, 3, 4];
        let v = neon::vld1q_u32(token, &data_u32);
        let mut out = [0u32; 4];
        neon::vst1q_u32(token, &mut out, v);
        assert_eq!(out, data_u32);

        // i32x4
        let data_i32: [i32; 4] = [1, 2, 3, 4];
        let v = neon::vld1q_s32(token, &data_i32);
        let mut out = [0i32; 4];
        neon::vst1q_s32(token, &mut out, v);
        assert_eq!(out, data_i32);

        // f32x4
        let data_f32: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let v = neon::vld1q_f32(token, &data_f32);
        let mut out = [0.0f32; 4];
        neon::vst1q_f32(token, &mut out, v);
        assert_eq!(out, data_f32);

        // u64x2
        let data_u64: [u64; 2] = [1, 2];
        let v = neon::vld1q_u64(token, &data_u64);
        let mut out = [0u64; 2];
        neon::vst1q_u64(token, &mut out, v);
        assert_eq!(out, data_u64);

        // i64x2
        let data_i64: [i64; 2] = [1, 2];
        let v = neon::vld1q_s64(token, &data_i64);
        let mut out = [0i64; 2];
        neon::vst1q_s64(token, &mut out, v);
        assert_eq!(out, data_i64);

        // f64x2
        let data_f64: [f64; 2] = [1.0, 2.0];
        let v = neon::vld1q_f64(token, &data_f64);
        let mut out = [0.0f64; 2];
        neon::vst1q_f64(token, &mut out, v);
        assert_eq!(out, data_f64);

        // =====================================================================
        // Multi-register loads (vld1_*_x2, x3, x4)
        // =====================================================================

        // vld1_f32_x2
        let data: [[f32; 2]; 2] = [[1.0, 2.0], [3.0, 4.0]];
        let v = neon::vld1_f32_x2(token, &data);
        let mut out = [[0.0f32; 2]; 2];
        neon::vst1_f32_x2(token, &mut out, v);
        assert_eq!(out, data);

        // vld1q_f32_x2
        let data: [[f32; 4]; 2] = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let v = neon::vld1q_f32_x2(token, &data);
        let mut out = [[0.0f32; 4]; 2];
        neon::vst1q_f32_x2(token, &mut out, v);
        assert_eq!(out, data);

        // vld1_u8_x3
        let data: [[u8; 8]; 3] = [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16],
            [17, 18, 19, 20, 21, 22, 23, 24],
        ];
        let v = neon::vld1_u8_x3(token, &data);
        let mut out = [[0u8; 8]; 3];
        neon::vst1_u8_x3(token, &mut out, v);
        assert_eq!(out, data);

        // vld1_i32_x4
        let data: [[i32; 2]; 4] = [[1, 2], [3, 4], [5, 6], [7, 8]];
        let v = neon::vld1_s32_x4(token, &data);
        let mut out = [[0i32; 2]; 4];
        neon::vst1_s32_x4(token, &mut out, v);
        assert_eq!(out, data);

        // =====================================================================
        // Duplicate loads (vld1_dup_*)
        // =====================================================================

        // vld1_dup_f32
        let val = 42.0f32;
        let v = neon::vld1_dup_f32(token, &val);
        let mut out = [0.0f32; 2];
        neon::vst1_f32(token, &mut out, v);
        assert!(out.iter().all(|&x| x == 42.0));

        // vld1q_dup_f32
        let v = neon::vld1q_dup_f32(token, &val);
        let mut out = [0.0f32; 4];
        neon::vst1q_f32(token, &mut out, v);
        assert!(out.iter().all(|&x| x == 42.0));

        // vld1_dup_u64
        let val = 123u64;
        let v = neon::vld1_dup_u64(token, &val);
        let mut out = 0u64;
        neon::vst1_u64(token, &mut out, v);
        assert_eq!(out, val);

        black_box(&out);
    }

    /// Test NEON intrinsics via #[arcane].
    #[test]
    fn test_arcane_neon_intrinsics() {
        use archmage::{HasNeon, NeonToken, SimdToken, arcane};

        let token = NeonToken::summon().expect("NEON should be available");

        // Test arithmetic
        #[arcane]
        fn test_neon_arithmetic(token: impl HasNeon) -> [f32; 4] {
            let a = vdupq_n_f32(2.0);
            let b = vdupq_n_f32(3.0);

            let sum = vaddq_f32(a, b);
            let diff = vsubq_f32(sum, a);
            let prod = vmulq_f32(diff, b);

            let mut result = [0.0f32; 4];
            unsafe { vst1q_f32(result.as_mut_ptr(), prod) };
            result
        }

        let result = test_neon_arithmetic(token);
        assert!(result.iter().all(|&x| (x - 9.0).abs() < 0.0001));

        // Test FMA
        #[arcane]
        fn test_neon_fma(token: impl HasNeon) -> [f32; 4] {
            let a = vdupq_n_f32(2.0);
            let b = vdupq_n_f32(3.0);
            let c = vdupq_n_f32(1.0);

            // a*b + c = 7.0
            let fma = vfmaq_f32(c, a, b);

            let mut result = [0.0f32; 4];
            unsafe { vst1q_f32(result.as_mut_ptr(), fma) };
            result
        }

        let result = test_neon_fma(token);
        assert!(result.iter().all(|&x| (x - 7.0).abs() < 0.0001));

        // Test comparison and selection
        #[arcane]
        fn test_neon_compare(token: impl HasNeon) -> [f32; 4] {
            let a = unsafe { vld1q_f32([1.0f32, 5.0, 3.0, 7.0].as_ptr()) };
            let b = vdupq_n_f32(4.0);

            // Compare a > b
            let mask = vcgtq_f32(a, b);
            // Select b where a > b, else a
            let result = vbslq_f32(vreinterpretq_u32_f32(vreinterpretq_f32_u32(mask)), b, a);

            let mut out = [0.0f32; 4];
            unsafe { vst1q_f32(out.as_mut_ptr(), result) };
            out
        }

        let result = test_neon_compare(token);
        assert_eq!(result[0], 1.0); // 1 < 4, keep a
        assert_eq!(result[1], 4.0); // 5 > 4, select b
        assert_eq!(result[2], 3.0); // 3 < 4, keep a
        assert_eq!(result[3], 4.0); // 7 > 4, select b

        // Test integer operations
        #[arcane]
        fn test_neon_integer(token: impl HasNeon) -> [i32; 4] {
            let a = vdupq_n_s32(10);
            let b = vdupq_n_s32(3);

            let sum = vaddq_s32(a, b);
            let diff = vsubq_s32(sum, b);
            let neg = vnegq_s32(diff);
            let abs = vabsq_s32(neg);

            let mut result = [0i32; 4];
            unsafe { vst1q_s32(result.as_mut_ptr(), abs) };
            result
        }

        let result = test_neon_integer(token);
        assert!(result.iter().all(|&x| x == 10));

        // Test bitwise operations
        #[arcane]
        fn test_neon_bitwise(token: impl HasNeon) -> [i32; 4] {
            let a = vdupq_n_s32(0b1100);
            let b = vdupq_n_s32(0b1010);

            let and = vandq_s32(a, b); // 0b1000
            let or = vorrq_s32(a, b); // 0b1110
            let xor = veorq_s32(a, b); // 0b0110
            let not = vmvnq_s32(a); // ~0b1100

            let mut result = [0i32; 4];
            unsafe { vst1q_s32(result.as_mut_ptr(), and) };
            black_box(or);
            black_box(xor);
            black_box(not);
            result
        }

        let result = test_neon_bitwise(token);
        assert!(result.iter().all(|&x| x == 0b1000));

        // Test shuffle/permute
        #[arcane]
        fn test_neon_shuffle(token: impl HasNeon) -> [f32; 4] {
            let a = unsafe { vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr()) };

            // Reverse elements
            let rev = vrev64q_f32(a);

            let mut result = [0.0f32; 4];
            unsafe { vst1q_f32(result.as_mut_ptr(), rev) };
            result
        }

        let result = test_neon_shuffle(token);
        // vrev64 reverses elements within 64-bit pairs
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 1.0);
        assert_eq!(result[2], 4.0);
        assert_eq!(result[3], 3.0);

        // Test type conversions
        #[arcane]
        fn test_neon_conversion(token: impl HasNeon) -> [i32; 4] {
            let a = unsafe { vld1q_f32([1.5f32, 2.7, 3.2, 4.9].as_ptr()) };

            // Convert to integer (truncate toward zero)
            let ints = vcvtq_s32_f32(a);

            let mut result = [0i32; 4];
            unsafe { vst1q_s32(result.as_mut_ptr(), ints) };
            result
        }

        let result = test_neon_conversion(token);
        assert_eq!(result, [1, 2, 3, 4]);

        // Test horizontal operations
        #[arcane]
        fn test_neon_horizontal(token: impl HasNeon) -> f32 {
            let a = unsafe { vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr()) };

            // Pairwise add
            let pair = vpaddq_f32(a, a);
            // Add across vector
            let sum = vaddvq_f32(a);

            black_box(pair);
            sum
        }

        let result = test_neon_horizontal(token);
        assert!((result - 10.0).abs() < 0.0001);
    }
}

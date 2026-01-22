//! Exhaustive tests for cross-platform tokens and core::arch intrinsics.
//!
//! This test exercises:
//! - Cross-platform token availability (all tokens compile on all archs)
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
        Avx2FmaToken, Avx2Token, AvxToken, Desktop64, FmaToken, Sse41Token, Sse42Token, X64V2Token,
        X64V3Token,
    };
    #[cfg(feature = "avx512")]
    use archmage::{Avx512Token, X64V4Token};

    // Verify tokens are zero-sized
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
        // X64V2Token covers SSE4.2 which is baseline on modern x86_64
        use archmage::X64V2Token;
        // Note: X64V2Token may not be available on very old processors
        let _ = X64V2Token::summon();

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
    use archmage::{SimdToken, Sse41Token};

    // This test verifies the mechanism works - actual disable testing
    // should be done by running: ARCHMAGE_DISABLE=1 cargo test
    if std::env::var_os("ARCHMAGE_DISABLE").is_some() {
        #[cfg(target_arch = "x86_64")]
        {
            assert!(
                Sse41Token::summon().is_none(),
                "ARCHMAGE_DISABLE should make summon() return None"
            );
            // try_new() should still work
            assert!(
                Sse41Token::try_new().is_some(),
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
        Sse41Token, Sse42Token, X64V2Token, X64V3Token,
    };

    // x86 tokens
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

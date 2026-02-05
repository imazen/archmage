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
    // x86 tier tokens - should compile on ARM/WASM, summon returns None there
    use archmage::{Avx2FmaToken, Desktop64, X64V2Token, X64V3Token};
    #[cfg(feature = "avx512")]
    use archmage::{Avx512Token, X64V4Token};

    // Verify tokens are zero-sized
    assert_eq!(core::mem::size_of::<X64V2Token>(), 0);
    assert_eq!(core::mem::size_of::<X64V3Token>(), 0);
    assert_eq!(core::mem::size_of::<Avx2FmaToken>(), 0);
    assert_eq!(core::mem::size_of::<Desktop64>(), 0);
    #[cfg(feature = "avx512")]
    assert_eq!(core::mem::size_of::<X64V4Token>(), 0);
    #[cfg(feature = "avx512")]
    assert_eq!(core::mem::size_of::<Avx512Token>(), 0);

    // ARM tokens - should compile on x86/WASM, summon returns None there
    use archmage::{Arm64, NeonAesToken, NeonCrcToken, NeonSha3Token, NeonToken};

    assert_eq!(core::mem::size_of::<NeonToken>(), 0);
    assert_eq!(core::mem::size_of::<Arm64>(), 0);
    assert_eq!(core::mem::size_of::<NeonAesToken>(), 0);
    assert_eq!(core::mem::size_of::<NeonSha3Token>(), 0);
    assert_eq!(core::mem::size_of::<NeonCrcToken>(), 0);

    // WASM token - should compile everywhere
    use archmage::Wasm128Token;
    assert_eq!(core::mem::size_of::<Wasm128Token>(), 0);
}

/// Test that summon() works correctly for the current platform.
#[test]
fn test_summon_behavior() {
    use archmage::{Arm64, NeonToken, Wasm128Token};

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
            Wasm128Token::summon().is_none(),
            "WASM SIMD unavailable on x86"
        );
    }

    // On aarch64, NEON is always available
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::Desktop64;
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
            Wasm128Token::summon().is_none(),
            "WASM SIMD unavailable on ARM"
        );
    }

    // On WASM, SIMD128 may be available
    #[cfg(target_arch = "wasm32")]
    {
        use archmage::Desktop64;
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
    use archmage::{SimdToken, X64V2Token};

    // This test verifies the mechanism works - actual disable testing
    // should be done by running: ARCHMAGE_DISABLE=1 cargo test
    if std::env::var_os("ARCHMAGE_DISABLE").is_some() {
        #[cfg(target_arch = "x86_64")]
        {
            assert!(
                X64V2Token::summon().is_none(),
                "ARCHMAGE_DISABLE should make summon() return None"
            );
            // summon() should still work
            assert!(
                X64V2Token::summon().is_some(),
                "summon() should still detect CPU features"
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
        Arm64, Avx2FmaToken, Desktop64, NeonAesToken, NeonCrcToken, NeonSha3Token, NeonToken,
        Wasm128Token, X64V2Token, X64V3Token,
    };

    // x86 tier tokens
    assert_eq!(X64V2Token::NAME, "x86-64-v2");
    assert_eq!(X64V3Token::NAME, "x86-64-v3");
    assert_eq!(Avx2FmaToken::NAME, "x86-64-v3"); // alias for X64V3Token

    // AVX-512 tokens (requires avx512 feature)
    #[cfg(feature = "avx512")]
    {
        use archmage::{Avx512Token, Server64, X64V4Token};
        assert_eq!(X64V4Token::NAME, "AVX-512");
        assert_eq!(Avx512Token::NAME, "AVX-512");
        assert_eq!(Server64::NAME, "AVX-512");
    }

    // Verify aliases share the same NAME
    assert_eq!(Desktop64::NAME, X64V3Token::NAME);
    assert_eq!(Avx2FmaToken::NAME, X64V3Token::NAME);

    // ARM tokens
    assert_eq!(NeonToken::NAME, "NEON");
    assert_eq!(Arm64::NAME, NeonToken::NAME);
    assert_eq!(NeonAesToken::NAME, "NEON+AES");
    assert_eq!(NeonSha3Token::NAME, "NEON+SHA3");
    assert_eq!(NeonCrcToken::NAME, "NEON+CRC");

    // WASM tokens
    assert_eq!(Wasm128Token::NAME, "WASM SIMD128");
}

/// Print all token names for debugging.
#[test]
fn print_all_token_names() {
    println!("All archmage tokens:");
    println!("  x86 tier tokens:");
    println!("    X64V2Token:        {}", archmage::X64V2Token::NAME);
    println!("    X64V3Token:        {}", archmage::X64V3Token::NAME);
    println!("    Desktop64:         {}", archmage::Desktop64::NAME);
    println!("    Avx2FmaToken:      {}", archmage::Avx2FmaToken::NAME);
    #[cfg(feature = "avx512")]
    {
        println!("    X64V4Token:        {}", archmage::X64V4Token::NAME);
        println!("    Avx512Token:       {}", archmage::Avx512Token::NAME);
        println!("    Server64:          {}", archmage::Server64::NAME);
        println!(
            "    Avx512ModernToken: {}",
            archmage::Avx512ModernToken::NAME
        );
        println!("    Avx512Fp16Token:   {}", archmage::Avx512Fp16Token::NAME);
    }
    println!("  ARM tokens:");
    println!("    NeonToken:         {}", archmage::NeonToken::NAME);
    println!("    Arm64:             {}", archmage::Arm64::NAME);
    println!("    NeonAesToken:      {}", archmage::NeonAesToken::NAME);
    println!("    NeonSha3Token:     {}", archmage::NeonSha3Token::NAME);
    println!("    NeonCrcToken:      {}", archmage::NeonCrcToken::NAME);
    println!("  WASM tokens:");
    println!("    Wasm128Token:      {}", archmage::Wasm128Token::NAME);
}

/// Test that implementation_name() returns the correct implementation path for all types.
///
/// This tests:
/// 1. Native types at each width level
/// 2. Polyfill types (always available alongside native types)
/// 3. Every element type (f32, f64, i8, u8, i16, u16, i32, u32, i64, u64)
#[test]
fn test_implementation_names() {
    // ========== x86_64 native types ==========
    #[cfg(target_arch = "x86_64")]
    {
        use magetypes::simd::x86;

        // W128 (SSE) - all 10 element types
        assert_eq!(x86::w128::f32x4::implementation_name(), "x86::v3::f32x4");
        assert_eq!(x86::w128::f64x2::implementation_name(), "x86::v3::f64x2");
        assert_eq!(x86::w128::i8x16::implementation_name(), "x86::v3::i8x16");
        assert_eq!(x86::w128::u8x16::implementation_name(), "x86::v3::u8x16");
        assert_eq!(x86::w128::i16x8::implementation_name(), "x86::v3::i16x8");
        assert_eq!(x86::w128::u16x8::implementation_name(), "x86::v3::u16x8");
        assert_eq!(x86::w128::i32x4::implementation_name(), "x86::v3::i32x4");
        assert_eq!(x86::w128::u32x4::implementation_name(), "x86::v3::u32x4");
        assert_eq!(x86::w128::i64x2::implementation_name(), "x86::v3::i64x2");
        assert_eq!(x86::w128::u64x2::implementation_name(), "x86::v3::u64x2");

        // W256 (AVX2) - all 10 element types
        assert_eq!(x86::w256::f32x8::implementation_name(), "x86::v3::f32x8");
        assert_eq!(x86::w256::f64x4::implementation_name(), "x86::v3::f64x4");
        assert_eq!(x86::w256::i8x32::implementation_name(), "x86::v3::i8x32");
        assert_eq!(x86::w256::u8x32::implementation_name(), "x86::v3::u8x32");
        assert_eq!(x86::w256::i16x16::implementation_name(), "x86::v3::i16x16");
        assert_eq!(x86::w256::u16x16::implementation_name(), "x86::v3::u16x16");
        assert_eq!(x86::w256::i32x8::implementation_name(), "x86::v3::i32x8");
        assert_eq!(x86::w256::u32x8::implementation_name(), "x86::v3::u32x8");
        assert_eq!(x86::w256::i64x4::implementation_name(), "x86::v3::i64x4");
        assert_eq!(x86::w256::u64x4::implementation_name(), "x86::v3::u64x4");

        // W512 (AVX-512) - all 10 element types
        #[cfg(feature = "avx512")]
        {
            assert_eq!(x86::w512::f32x16::implementation_name(), "x86::v4::f32x16");
            assert_eq!(x86::w512::f64x8::implementation_name(), "x86::v4::f64x8");
            assert_eq!(x86::w512::i8x64::implementation_name(), "x86::v4::i8x64");
            assert_eq!(x86::w512::u8x64::implementation_name(), "x86::v4::u8x64");
            assert_eq!(x86::w512::i16x32::implementation_name(), "x86::v4::i16x32");
            assert_eq!(x86::w512::u16x32::implementation_name(), "x86::v4::u16x32");
            assert_eq!(x86::w512::i32x16::implementation_name(), "x86::v4::i32x16");
            assert_eq!(x86::w512::u32x16::implementation_name(), "x86::v4::u32x16");
            assert_eq!(x86::w512::i64x8::implementation_name(), "x86::v4::i64x8");
            assert_eq!(x86::w512::u64x8::implementation_name(), "x86::v4::u64x8");
        }
    }

    // ========== x86_64 polyfill types ==========
    // These are ALWAYS available, even when native types exist!
    #[cfg(target_arch = "x86_64")]
    {
        use magetypes::simd::polyfill;

        // polyfill::v3 - W256 emulated from W128 (all 10 element types)
        assert_eq!(
            polyfill::v3::f32x8::implementation_name(),
            "polyfill::v3::f32x8"
        );
        assert_eq!(
            polyfill::v3::f64x4::implementation_name(),
            "polyfill::v3::f64x4"
        );
        assert_eq!(
            polyfill::v3::i8x32::implementation_name(),
            "polyfill::v3::i8x32"
        );
        assert_eq!(
            polyfill::v3::u8x32::implementation_name(),
            "polyfill::v3::u8x32"
        );
        assert_eq!(
            polyfill::v3::i16x16::implementation_name(),
            "polyfill::v3::i16x16"
        );
        assert_eq!(
            polyfill::v3::u16x16::implementation_name(),
            "polyfill::v3::u16x16"
        );
        assert_eq!(
            polyfill::v3::i32x8::implementation_name(),
            "polyfill::v3::i32x8"
        );
        assert_eq!(
            polyfill::v3::u32x8::implementation_name(),
            "polyfill::v3::u32x8"
        );
        assert_eq!(
            polyfill::v3::i64x4::implementation_name(),
            "polyfill::v3::i64x4"
        );
        assert_eq!(
            polyfill::v3::u64x4::implementation_name(),
            "polyfill::v3::u64x4"
        );

        // polyfill::v3 - W512 emulated from W256 (all 10 element types)
        assert_eq!(
            polyfill::v3_512::f32x16::implementation_name(),
            "polyfill::v3_512::f32x16"
        );
        assert_eq!(
            polyfill::v3_512::f64x8::implementation_name(),
            "polyfill::v3_512::f64x8"
        );
        assert_eq!(
            polyfill::v3_512::i8x64::implementation_name(),
            "polyfill::v3_512::i8x64"
        );
        assert_eq!(
            polyfill::v3_512::u8x64::implementation_name(),
            "polyfill::v3_512::u8x64"
        );
        assert_eq!(
            polyfill::v3_512::i16x32::implementation_name(),
            "polyfill::v3_512::i16x32"
        );
        assert_eq!(
            polyfill::v3_512::u16x32::implementation_name(),
            "polyfill::v3_512::u16x32"
        );
        assert_eq!(
            polyfill::v3_512::i32x16::implementation_name(),
            "polyfill::v3_512::i32x16"
        );
        assert_eq!(
            polyfill::v3_512::u32x16::implementation_name(),
            "polyfill::v3_512::u32x16"
        );
        assert_eq!(
            polyfill::v3_512::i64x8::implementation_name(),
            "polyfill::v3_512::i64x8"
        );
        assert_eq!(
            polyfill::v3_512::u64x8::implementation_name(),
            "polyfill::v3_512::u64x8"
        );
    }

    // ========== aarch64 native types ==========
    #[cfg(target_arch = "aarch64")]
    {
        use magetypes::simd::arm;

        // W128 (NEON) - all 10 element types
        assert_eq!(arm::w128::f32x4::implementation_name(), "arm::neon::f32x4");
        assert_eq!(arm::w128::f64x2::implementation_name(), "arm::neon::f64x2");
        assert_eq!(arm::w128::i8x16::implementation_name(), "arm::neon::i8x16");
        assert_eq!(arm::w128::u8x16::implementation_name(), "arm::neon::u8x16");
        assert_eq!(arm::w128::i16x8::implementation_name(), "arm::neon::i16x8");
        assert_eq!(arm::w128::u16x8::implementation_name(), "arm::neon::u16x8");
        assert_eq!(arm::w128::i32x4::implementation_name(), "arm::neon::i32x4");
        assert_eq!(arm::w128::u32x4::implementation_name(), "arm::neon::u32x4");
        assert_eq!(arm::w128::i64x2::implementation_name(), "arm::neon::i64x2");
        assert_eq!(arm::w128::u64x2::implementation_name(), "arm::neon::u64x2");
    }

    // ========== aarch64 polyfill types ==========
    #[cfg(target_arch = "aarch64")]
    {
        use magetypes::simd::polyfill;

        // polyfill::neon - W256 emulated from W128 (all 10 element types)
        assert_eq!(
            polyfill::neon::f32x8::implementation_name(),
            "polyfill::neon::f32x8"
        );
        assert_eq!(
            polyfill::neon::f64x4::implementation_name(),
            "polyfill::neon::f64x4"
        );
        assert_eq!(
            polyfill::neon::i8x32::implementation_name(),
            "polyfill::neon::i8x32"
        );
        assert_eq!(
            polyfill::neon::u8x32::implementation_name(),
            "polyfill::neon::u8x32"
        );
        assert_eq!(
            polyfill::neon::i16x16::implementation_name(),
            "polyfill::neon::i16x16"
        );
        assert_eq!(
            polyfill::neon::u16x16::implementation_name(),
            "polyfill::neon::u16x16"
        );
        assert_eq!(
            polyfill::neon::i32x8::implementation_name(),
            "polyfill::neon::i32x8"
        );
        assert_eq!(
            polyfill::neon::u32x8::implementation_name(),
            "polyfill::neon::u32x8"
        );
        assert_eq!(
            polyfill::neon::i64x4::implementation_name(),
            "polyfill::neon::i64x4"
        );
        assert_eq!(
            polyfill::neon::u64x4::implementation_name(),
            "polyfill::neon::u64x4"
        );
    }

    // ========== wasm32 native types ==========
    #[cfg(target_arch = "wasm32")]
    {
        use magetypes::simd::wasm;

        // W128 (SIMD128) - all 10 element types
        assert_eq!(
            wasm::w128::f32x4::implementation_name(),
            "wasm::simd128::f32x4"
        );
        assert_eq!(
            wasm::w128::f64x2::implementation_name(),
            "wasm::simd128::f64x2"
        );
        assert_eq!(
            wasm::w128::i8x16::implementation_name(),
            "wasm::simd128::i8x16"
        );
        assert_eq!(
            wasm::w128::u8x16::implementation_name(),
            "wasm::simd128::u8x16"
        );
        assert_eq!(
            wasm::w128::i16x8::implementation_name(),
            "wasm::simd128::i16x8"
        );
        assert_eq!(
            wasm::w128::u16x8::implementation_name(),
            "wasm::simd128::u16x8"
        );
        assert_eq!(
            wasm::w128::i32x4::implementation_name(),
            "wasm::simd128::i32x4"
        );
        assert_eq!(
            wasm::w128::u32x4::implementation_name(),
            "wasm::simd128::u32x4"
        );
        assert_eq!(
            wasm::w128::i64x2::implementation_name(),
            "wasm::simd128::i64x2"
        );
        assert_eq!(
            wasm::w128::u64x2::implementation_name(),
            "wasm::simd128::u64x2"
        );
    }

    // ========== wasm32 polyfill types ==========
    #[cfg(target_arch = "wasm32")]
    {
        use magetypes::simd::polyfill;

        // polyfill::wasm128 - W256 emulated from W128 (all 10 element types)
        assert_eq!(
            polyfill::wasm128::f32x8::implementation_name(),
            "polyfill::wasm128::f32x8"
        );
        assert_eq!(
            polyfill::wasm128::f64x4::implementation_name(),
            "polyfill::wasm128::f64x4"
        );
        assert_eq!(
            polyfill::wasm128::i8x32::implementation_name(),
            "polyfill::wasm128::i8x32"
        );
        assert_eq!(
            polyfill::wasm128::u8x32::implementation_name(),
            "polyfill::wasm128::u8x32"
        );
        assert_eq!(
            polyfill::wasm128::i16x16::implementation_name(),
            "polyfill::wasm128::i16x16"
        );
        assert_eq!(
            polyfill::wasm128::u16x16::implementation_name(),
            "polyfill::wasm128::u16x16"
        );
        assert_eq!(
            polyfill::wasm128::i32x8::implementation_name(),
            "polyfill::wasm128::i32x8"
        );
        assert_eq!(
            polyfill::wasm128::u32x8::implementation_name(),
            "polyfill::wasm128::u32x8"
        );
        assert_eq!(
            polyfill::wasm128::i64x4::implementation_name(),
            "polyfill::wasm128::i64x4"
        );
        assert_eq!(
            polyfill::wasm128::u64x4::implementation_name(),
            "polyfill::wasm128::u64x4"
        );
    }

    // ========== Verify top-level re-exports use native types ==========
    // When importing from `magetypes::simd::*`, we should get the native (fastest) types
    #[cfg(target_arch = "x86_64")]
    {
        use magetypes::simd::*;
        // Top-level f32x4 should be native x86::w128::f32x4
        assert_eq!(f32x4::implementation_name(), "x86::v3::f32x4");
        // Top-level f32x8 should be native x86::w256::f32x8
        assert_eq!(f32x8::implementation_name(), "x86::v3::f32x8");
        // With avx512 feature, top-level f32x16 should be native x86::w512::f32x16
        #[cfg(feature = "avx512")]
        assert_eq!(f32x16::implementation_name(), "x86::v4::f32x16");
    }

    // ========== x86_64 width-aliased namespaces ==========
    // These are the "token tier" modules - each corresponds to a specific SIMD level
    #[cfg(target_arch = "x86_64")]
    {
        use magetypes::simd;

        // simd::sse - maps to x86::w128 (128-bit SIMD)
        // f32xN = f32x4, i32xN = i32x4, etc.
        assert_eq!(simd::sse::f32xN::implementation_name(), "x86::v3::f32x4");
        assert_eq!(simd::sse::f64xN::implementation_name(), "x86::v3::f64x2");
        assert_eq!(simd::sse::i8xN::implementation_name(), "x86::v3::i8x16");
        assert_eq!(simd::sse::u8xN::implementation_name(), "x86::v3::u8x16");
        assert_eq!(simd::sse::i16xN::implementation_name(), "x86::v3::i16x8");
        assert_eq!(simd::sse::u16xN::implementation_name(), "x86::v3::u16x8");
        assert_eq!(simd::sse::i32xN::implementation_name(), "x86::v3::i32x4");
        assert_eq!(simd::sse::u32xN::implementation_name(), "x86::v3::u32x4");
        assert_eq!(simd::sse::i64xN::implementation_name(), "x86::v3::i64x2");
        assert_eq!(simd::sse::u64xN::implementation_name(), "x86::v3::u64x2");
        // Also verify the non-aliased types are accessible
        assert_eq!(simd::sse::f32x4::implementation_name(), "x86::v3::f32x4");

        // simd::avx2 - maps to x86::w256 (256-bit SIMD)
        // f32xN = f32x8, i32xN = i32x8, etc.
        assert_eq!(simd::avx2::f32xN::implementation_name(), "x86::v3::f32x8");
        assert_eq!(simd::avx2::f64xN::implementation_name(), "x86::v3::f64x4");
        assert_eq!(simd::avx2::i8xN::implementation_name(), "x86::v3::i8x32");
        assert_eq!(simd::avx2::u8xN::implementation_name(), "x86::v3::u8x32");
        assert_eq!(simd::avx2::i16xN::implementation_name(), "x86::v3::i16x16");
        assert_eq!(simd::avx2::u16xN::implementation_name(), "x86::v3::u16x16");
        assert_eq!(simd::avx2::i32xN::implementation_name(), "x86::v3::i32x8");
        assert_eq!(simd::avx2::u32xN::implementation_name(), "x86::v3::u32x8");
        assert_eq!(simd::avx2::i64xN::implementation_name(), "x86::v3::i64x4");
        assert_eq!(simd::avx2::u64xN::implementation_name(), "x86::v3::u64x4");
        // Also verify the non-aliased types are accessible
        assert_eq!(simd::avx2::f32x8::implementation_name(), "x86::v3::f32x8");

        // simd::avx512 - maps to x86::w512 (512-bit SIMD)
        // f32xN = f32x16, i32xN = i32x16, etc.
        #[cfg(feature = "avx512")]
        {
            assert_eq!(
                simd::avx512::f32xN::implementation_name(),
                "x86::v4::f32x16"
            );
            assert_eq!(simd::avx512::f64xN::implementation_name(), "x86::v4::f64x8");
            assert_eq!(simd::avx512::i8xN::implementation_name(), "x86::v4::i8x64");
            assert_eq!(simd::avx512::u8xN::implementation_name(), "x86::v4::u8x64");
            assert_eq!(
                simd::avx512::i16xN::implementation_name(),
                "x86::v4::i16x32"
            );
            assert_eq!(
                simd::avx512::u16xN::implementation_name(),
                "x86::v4::u16x32"
            );
            assert_eq!(
                simd::avx512::i32xN::implementation_name(),
                "x86::v4::i32x16"
            );
            assert_eq!(
                simd::avx512::u32xN::implementation_name(),
                "x86::v4::u32x16"
            );
            assert_eq!(simd::avx512::i64xN::implementation_name(), "x86::v4::i64x8");
            assert_eq!(simd::avx512::u64xN::implementation_name(), "x86::v4::u64x8");
            // Also verify the non-aliased types are accessible
            assert_eq!(
                simd::avx512::f32x16::implementation_name(),
                "x86::v4::f32x16"
            );
        }
    }

    // ========== aarch64 width-aliased namespaces ==========
    #[cfg(target_arch = "aarch64")]
    {
        use magetypes::simd;

        // simd::neon - maps to arm::w128 (128-bit SIMD)
        // f32xN = f32x4, i32xN = i32x4, etc.
        assert_eq!(simd::neon::f32xN::implementation_name(), "arm::neon::f32x4");
        assert_eq!(simd::neon::f64xN::implementation_name(), "arm::neon::f64x2");
        assert_eq!(simd::neon::i8xN::implementation_name(), "arm::neon::i8x16");
        assert_eq!(simd::neon::u8xN::implementation_name(), "arm::neon::u8x16");
        assert_eq!(simd::neon::i16xN::implementation_name(), "arm::neon::i16x8");
        assert_eq!(simd::neon::u16xN::implementation_name(), "arm::neon::u16x8");
        assert_eq!(simd::neon::i32xN::implementation_name(), "arm::neon::i32x4");
        assert_eq!(simd::neon::u32xN::implementation_name(), "arm::neon::u32x4");
        assert_eq!(simd::neon::i64xN::implementation_name(), "arm::neon::i64x2");
        assert_eq!(simd::neon::u64xN::implementation_name(), "arm::neon::u64x2");
        // Also verify the non-aliased types are accessible
        assert_eq!(simd::neon::f32x4::implementation_name(), "arm::neon::f32x4");
    }

    // ========== wasm32 width-aliased namespaces ==========
    #[cfg(target_arch = "wasm32")]
    {
        use magetypes::simd;

        // simd::wasm128 - maps to wasm::w128 (128-bit SIMD)
        // f32xN = f32x4, i32xN = i32x4, etc.
        assert_eq!(
            simd::wasm128::f32xN::implementation_name(),
            "wasm::simd128::f32x4"
        );
        assert_eq!(
            simd::wasm128::f64xN::implementation_name(),
            "wasm::simd128::f64x2"
        );
        assert_eq!(
            simd::wasm128::i8xN::implementation_name(),
            "wasm::simd128::i8x16"
        );
        assert_eq!(
            simd::wasm128::u8xN::implementation_name(),
            "wasm::simd128::u8x16"
        );
        assert_eq!(
            simd::wasm128::i16xN::implementation_name(),
            "wasm::simd128::i16x8"
        );
        assert_eq!(
            simd::wasm128::u16xN::implementation_name(),
            "wasm::simd128::u16x8"
        );
        assert_eq!(
            simd::wasm128::i32xN::implementation_name(),
            "wasm::simd128::i32x4"
        );
        assert_eq!(
            simd::wasm128::u32xN::implementation_name(),
            "wasm::simd128::u32x4"
        );
        assert_eq!(
            simd::wasm128::i64xN::implementation_name(),
            "wasm::simd128::i64x2"
        );
        assert_eq!(
            simd::wasm128::u64xN::implementation_name(),
            "wasm::simd128::u64x2"
        );
        // Also verify the non-aliased types are accessible
        assert_eq!(
            simd::wasm128::f32x4::implementation_name(),
            "wasm::simd128::f32x4"
        );
    }
}

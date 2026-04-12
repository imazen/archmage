//! Exhaustive tests for magetypes SIMD type implementation names.
//!
//! Verifies that implementation_name() returns the correct backend path for:
//! - Native types at each width level (x86, ARM, WASM)
//! - Polyfill types (always available alongside native types)
//! - Every element type (f32, f64, i8, u8, i16, u16, i32, u32, i64, u64)
//! - Width-aliased namespaces (v3, v4, neon, wasm128)
//!
//! For cross-platform token availability tests (summon, token names, dispatch),
//! see archmage's own exhaustive_intrinsics test.

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
        // Top-level f32x16 is the V3 polyfill (2×256-bit)
        #[cfg(feature = "avx512")]
        assert_eq!(f32x16::implementation_name(), "polyfill::v3_512::f32x16");
    }

    // ========== x86_64 width-aliased namespaces ==========
    // These are the "token tier" modules - each corresponds to a specific SIMD level
    #[cfg(target_arch = "x86_64")]
    {
        use magetypes::simd;

        // Note: No simd::sse namespace - v2 is legacy. 128-bit types available via simd::x86::w128::*

        // simd::v3 - maps to x86::w256 (256-bit SIMD)
        // f32xN = f32x8, i32xN = i32x8, etc.
        assert_eq!(simd::v3::f32xN::implementation_name(), "x86::v3::f32x8");
        assert_eq!(simd::v3::f64xN::implementation_name(), "x86::v3::f64x4");
        assert_eq!(simd::v3::i8xN::implementation_name(), "x86::v3::i8x32");
        assert_eq!(simd::v3::u8xN::implementation_name(), "x86::v3::u8x32");
        assert_eq!(simd::v3::i16xN::implementation_name(), "x86::v3::i16x16");
        assert_eq!(simd::v3::u16xN::implementation_name(), "x86::v3::u16x16");
        assert_eq!(simd::v3::i32xN::implementation_name(), "x86::v3::i32x8");
        assert_eq!(simd::v3::u32xN::implementation_name(), "x86::v3::u32x8");
        assert_eq!(simd::v3::i64xN::implementation_name(), "x86::v3::i64x4");
        assert_eq!(simd::v3::u64xN::implementation_name(), "x86::v3::u64x4");
        // Also verify the non-aliased types are accessible
        assert_eq!(simd::v3::f32x8::implementation_name(), "x86::v3::f32x8");

        // simd::v4 - maps to 512-bit types (native AVX-512 via X64V4Token)
        // f32xN = f32x16, i32xN = i32x16, etc.
        #[cfg(feature = "avx512")]
        {
            assert_eq!(simd::v4::f32xN::implementation_name(), "x86::v4::f32x16");
            assert_eq!(simd::v4::f64xN::implementation_name(), "x86::v4::f64x8");
            assert_eq!(simd::v4::i8xN::implementation_name(), "x86::v4::i8x64");
            assert_eq!(simd::v4::u8xN::implementation_name(), "x86::v4::u8x64");
            assert_eq!(simd::v4::i16xN::implementation_name(), "x86::v4::i16x32");
            assert_eq!(simd::v4::u16xN::implementation_name(), "x86::v4::u16x32");
            assert_eq!(simd::v4::i32xN::implementation_name(), "x86::v4::i32x16");
            assert_eq!(simd::v4::u32xN::implementation_name(), "x86::v4::u32x16");
            assert_eq!(simd::v4::i64xN::implementation_name(), "x86::v4::i64x8");
            assert_eq!(simd::v4::u64xN::implementation_name(), "x86::v4::u64x8");
            // Also verify the non-aliased types are accessible
            assert_eq!(simd::v4::f32x16::implementation_name(), "x86::v4::f32x16");
        }

        // simd::v4x - maps to 512-bit types (X64V4xToken, superset of V4)
        // Same widths as V4, but types get extension methods like .popcnt()
        #[cfg(feature = "avx512")]
        {
            assert_eq!(simd::v4x::f32xN::implementation_name(), "x86::v4x::f32x16");
            assert_eq!(simd::v4x::f64xN::implementation_name(), "x86::v4x::f64x8");
            assert_eq!(simd::v4x::i8xN::implementation_name(), "x86::v4x::i8x64");
            assert_eq!(simd::v4x::u8xN::implementation_name(), "x86::v4x::u8x64");
            assert_eq!(simd::v4x::i16xN::implementation_name(), "x86::v4x::i16x32");
            assert_eq!(simd::v4x::u16xN::implementation_name(), "x86::v4x::u16x32");
            assert_eq!(simd::v4x::i32xN::implementation_name(), "x86::v4x::i32x16");
            assert_eq!(simd::v4x::u32xN::implementation_name(), "x86::v4x::u32x16");
            assert_eq!(simd::v4x::i64xN::implementation_name(), "x86::v4x::i64x8");
            assert_eq!(simd::v4x::u64xN::implementation_name(), "x86::v4x::u64x8");
            // Also verify the non-aliased types are accessible
            assert_eq!(simd::v4x::f32x16::implementation_name(), "x86::v4x::f32x16");
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

//! Exhaustive tests for magetypes SIMD type implementation names.
//!
//! Verifies that `implementation_name()` returns the correct backend path for
//! every per-token namespace and element type. Since the legacy concrete
//! per-platform structs were retired, every type is the generic
//! `generic::TYPE<Token>` reached through a tier namespace (`v3`, `v4`, `v4x`,
//! `neon`, `wasm128`) or the bare `simd::*` aliases.
//!
//! For cross-platform token availability tests (summon, token names, dispatch),
//! see archmage's own exhaustive_intrinsics test.

/// Test that `implementation_name()` returns the correct implementation path for
/// every namespace × width × element type.
#[test]
fn test_implementation_names() {
    // ========== x86_64: v3 (128/256-bit AVX2), v4/v4x (512-bit AVX-512) ==========
    #[cfg(target_arch = "x86_64")]
    {
        use magetypes::simd;

        // v3 — 128-bit, native AVX2 (the sole x86 ≤256-bit backend)
        assert_eq!(simd::v3::f32x4::implementation_name(), "x86::v3::f32x4");
        assert_eq!(simd::v3::f64x2::implementation_name(), "x86::v3::f64x2");
        assert_eq!(simd::v3::i8x16::implementation_name(), "x86::v3::i8x16");
        assert_eq!(simd::v3::u8x16::implementation_name(), "x86::v3::u8x16");
        assert_eq!(simd::v3::i16x8::implementation_name(), "x86::v3::i16x8");
        assert_eq!(simd::v3::u16x8::implementation_name(), "x86::v3::u16x8");
        assert_eq!(simd::v3::i32x4::implementation_name(), "x86::v3::i32x4");
        assert_eq!(simd::v3::u32x4::implementation_name(), "x86::v3::u32x4");
        assert_eq!(simd::v3::i64x2::implementation_name(), "x86::v3::i64x2");
        assert_eq!(simd::v3::u64x2::implementation_name(), "x86::v3::u64x2");

        // v3 — 256-bit, native AVX2
        assert_eq!(simd::v3::f32x8::implementation_name(), "x86::v3::f32x8");
        assert_eq!(simd::v3::f64x4::implementation_name(), "x86::v3::f64x4");
        assert_eq!(simd::v3::i8x32::implementation_name(), "x86::v3::i8x32");
        assert_eq!(simd::v3::u8x32::implementation_name(), "x86::v3::u8x32");
        assert_eq!(simd::v3::i16x16::implementation_name(), "x86::v3::i16x16");
        assert_eq!(simd::v3::u16x16::implementation_name(), "x86::v3::u16x16");
        assert_eq!(simd::v3::i32x8::implementation_name(), "x86::v3::i32x8");
        assert_eq!(simd::v3::u32x8::implementation_name(), "x86::v3::u32x8");
        assert_eq!(simd::v3::i64x4::implementation_name(), "x86::v3::i64x4");
        assert_eq!(simd::v3::u64x4::implementation_name(), "x86::v3::u64x4");

        // v3 — 512-bit, polyfilled as two 256-bit halves (no native AVX-512)
        #[cfg(feature = "w512")]
        {
            assert_eq!(
                simd::v3::f32x16::implementation_name(),
                "polyfill::v3_512::f32x16"
            );
            assert_eq!(
                simd::v3::f64x8::implementation_name(),
                "polyfill::v3_512::f64x8"
            );
            assert_eq!(
                simd::v3::i8x64::implementation_name(),
                "polyfill::v3_512::i8x64"
            );
            assert_eq!(
                simd::v3::u8x64::implementation_name(),
                "polyfill::v3_512::u8x64"
            );
            assert_eq!(
                simd::v3::i16x32::implementation_name(),
                "polyfill::v3_512::i16x32"
            );
            assert_eq!(
                simd::v3::u16x32::implementation_name(),
                "polyfill::v3_512::u16x32"
            );
            assert_eq!(
                simd::v3::i32x16::implementation_name(),
                "polyfill::v3_512::i32x16"
            );
            assert_eq!(
                simd::v3::u32x16::implementation_name(),
                "polyfill::v3_512::u32x16"
            );
            assert_eq!(
                simd::v3::i64x8::implementation_name(),
                "polyfill::v3_512::i64x8"
            );
            assert_eq!(
                simd::v3::u64x8::implementation_name(),
                "polyfill::v3_512::u64x8"
            );
        }

        // v4 — 512-bit, native AVX-512
        #[cfg(feature = "avx512")]
        {
            assert_eq!(simd::v4::f32x16::implementation_name(), "x86::v4::f32x16");
            assert_eq!(simd::v4::f64x8::implementation_name(), "x86::v4::f64x8");
            assert_eq!(simd::v4::i8x64::implementation_name(), "x86::v4::i8x64");
            assert_eq!(simd::v4::u8x64::implementation_name(), "x86::v4::u8x64");
            assert_eq!(simd::v4::i16x32::implementation_name(), "x86::v4::i16x32");
            assert_eq!(simd::v4::u16x32::implementation_name(), "x86::v4::u16x32");
            assert_eq!(simd::v4::i32x16::implementation_name(), "x86::v4::i32x16");
            assert_eq!(simd::v4::u32x16::implementation_name(), "x86::v4::u32x16");
            assert_eq!(simd::v4::i64x8::implementation_name(), "x86::v4::i64x8");
            assert_eq!(simd::v4::u64x8::implementation_name(), "x86::v4::u64x8");
            // xN alias (natural width = 512-bit)
            assert_eq!(simd::v4::f32xN::implementation_name(), "x86::v4::f32x16");

            // v4x — superset of v4 (extension methods like .popcnt())
            assert_eq!(simd::v4x::f32x16::implementation_name(), "x86::v4x::f32x16");
            assert_eq!(simd::v4x::i32x16::implementation_name(), "x86::v4x::i32x16");
            assert_eq!(simd::v4x::u8x64::implementation_name(), "x86::v4x::u8x64");
        }

        // Bare top-level aliases resolve to the native (fastest) x86 types.
        {
            use magetypes::simd::{f32x4, f32x8};
            assert_eq!(f32x4::implementation_name(), "x86::v3::f32x4");
            assert_eq!(f32x8::implementation_name(), "x86::v3::f32x8");
        }
        #[cfg(feature = "w512")]
        {
            use magetypes::simd::f32x16;
            assert_eq!(f32x16::implementation_name(), "polyfill::v3_512::f32x16");
        }
    }

    // ========== aarch64: neon (128-bit native, 256-bit polyfill) ==========
    #[cfg(target_arch = "aarch64")]
    {
        use magetypes::simd;

        // 128-bit, native NEON
        assert_eq!(simd::neon::f32x4::implementation_name(), "arm::neon::f32x4");
        assert_eq!(simd::neon::f64x2::implementation_name(), "arm::neon::f64x2");
        assert_eq!(simd::neon::i8x16::implementation_name(), "arm::neon::i8x16");
        assert_eq!(simd::neon::u8x16::implementation_name(), "arm::neon::u8x16");
        assert_eq!(simd::neon::i16x8::implementation_name(), "arm::neon::i16x8");
        assert_eq!(simd::neon::u16x8::implementation_name(), "arm::neon::u16x8");
        assert_eq!(simd::neon::i32x4::implementation_name(), "arm::neon::i32x4");
        assert_eq!(simd::neon::u32x4::implementation_name(), "arm::neon::u32x4");
        assert_eq!(simd::neon::i64x2::implementation_name(), "arm::neon::i64x2");
        assert_eq!(simd::neon::u64x2::implementation_name(), "arm::neon::u64x2");

        // 256-bit, polyfilled as 2×128-bit NEON
        assert_eq!(
            simd::neon::f32x8::implementation_name(),
            "polyfill::neon::f32x8"
        );
        assert_eq!(
            simd::neon::f64x4::implementation_name(),
            "polyfill::neon::f64x4"
        );
        assert_eq!(
            simd::neon::i8x32::implementation_name(),
            "polyfill::neon::i8x32"
        );
        assert_eq!(
            simd::neon::u8x32::implementation_name(),
            "polyfill::neon::u8x32"
        );
        assert_eq!(
            simd::neon::i16x16::implementation_name(),
            "polyfill::neon::i16x16"
        );
        assert_eq!(
            simd::neon::u16x16::implementation_name(),
            "polyfill::neon::u16x16"
        );
        assert_eq!(
            simd::neon::i32x8::implementation_name(),
            "polyfill::neon::i32x8"
        );
        assert_eq!(
            simd::neon::u32x8::implementation_name(),
            "polyfill::neon::u32x8"
        );
        assert_eq!(
            simd::neon::i64x4::implementation_name(),
            "polyfill::neon::i64x4"
        );
        assert_eq!(
            simd::neon::u64x4::implementation_name(),
            "polyfill::neon::u64x4"
        );

        // Bare top-level aliases resolve to native NEON.
        use magetypes::simd::f32x4;
        assert_eq!(f32x4::implementation_name(), "arm::neon::f32x4");
    }

    // ========== wasm32: wasm128 (128-bit native, 256-bit polyfill) ==========
    #[cfg(target_arch = "wasm32")]
    {
        use magetypes::simd;

        // 128-bit, native SIMD128
        assert_eq!(
            simd::wasm128::f32x4::implementation_name(),
            "wasm::wasm128::f32x4"
        );
        assert_eq!(
            simd::wasm128::f64x2::implementation_name(),
            "wasm::wasm128::f64x2"
        );
        assert_eq!(
            simd::wasm128::i8x16::implementation_name(),
            "wasm::wasm128::i8x16"
        );
        assert_eq!(
            simd::wasm128::u8x16::implementation_name(),
            "wasm::wasm128::u8x16"
        );
        assert_eq!(
            simd::wasm128::i16x8::implementation_name(),
            "wasm::wasm128::i16x8"
        );
        assert_eq!(
            simd::wasm128::u16x8::implementation_name(),
            "wasm::wasm128::u16x8"
        );
        assert_eq!(
            simd::wasm128::i32x4::implementation_name(),
            "wasm::wasm128::i32x4"
        );
        assert_eq!(
            simd::wasm128::u32x4::implementation_name(),
            "wasm::wasm128::u32x4"
        );
        assert_eq!(
            simd::wasm128::i64x2::implementation_name(),
            "wasm::wasm128::i64x2"
        );
        assert_eq!(
            simd::wasm128::u64x2::implementation_name(),
            "wasm::wasm128::u64x2"
        );

        // 256-bit, polyfilled as 2×128-bit SIMD128
        assert_eq!(
            simd::wasm128::f32x8::implementation_name(),
            "polyfill::wasm128::f32x8"
        );
        assert_eq!(
            simd::wasm128::i32x8::implementation_name(),
            "polyfill::wasm128::i32x8"
        );
        assert_eq!(
            simd::wasm128::u64x4::implementation_name(),
            "polyfill::wasm128::u64x4"
        );

        // Bare top-level aliases resolve to native SIMD128.
        use magetypes::simd::f32x4;
        assert_eq!(f32x4::implementation_name(), "wasm::wasm128::f32x4");
    }
}

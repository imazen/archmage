//! Exhaustive tests for #[arcane] macro trait/token name recognition.
//!
//! These tests verify that the proc macro correctly recognizes ALL token types
//! and trait bounds supported by archmage, in all supported forms:
//! - Concrete token: `token: ConcreteType`
//! - impl Trait: `token: impl TraitBound`
//! - Generic inline: `fn foo<T: TraitBound>(token: T, ...)`
//! - Generic where: `fn foo<T>(token: T, ...) where T: TraitBound`
//!
//! If a token/trait exists in the runtime crate but these tests fail to compile,
//! it means the proc macro's recognition maps are out of sync.

#![allow(unused)]

// =============================================================================
// x86_64 tests
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod x86_concrete_tokens {
    //! Test every concrete x86 token type with #[arcane].
    use archmage::{SimdToken, arcane};
    use std::arch::x86_64::*;

    // --- Tier tokens ---

    #[arcane]
    fn with_x64v2(token: archmage::X64V2Token) -> u32 {
        _mm_crc32_u8(0, 0)
    }

    #[arcane]
    fn with_x64v3(token: archmage::X64V3Token) -> f32 {
        let a = _mm256_set1_ps(2.0);
        let r = _mm256_fmadd_ps(a, a, a);
        let lo = _mm256_castps256_ps128(r);
        unsafe { _mm_cvtss_f32(lo) }
    }

    #[arcane]
    fn with_desktop64(token: archmage::Desktop64) -> f32 {
        let a = _mm256_set1_ps(3.0);
        let r = _mm256_fmadd_ps(a, a, a);
        let lo = _mm256_castps256_ps128(r);
        unsafe { _mm_cvtss_f32(lo) }
    }

    #[arcane]
    fn with_avx2fma(token: archmage::Avx2FmaToken) -> f32 {
        let a = _mm256_set1_ps(2.0);
        let b = _mm256_set1_ps(3.0);
        let c = _mm256_set1_ps(1.0);
        let r = _mm256_fmadd_ps(a, b, c);
        let lo = _mm256_castps256_ps128(r);
        unsafe { _mm_cvtss_f32(lo) }
    }

    // --- AVX-512 tokens ---

    #[cfg(feature = "avx512")]
    #[arcane]
    fn with_x64v4(token: archmage::X64V4Token) -> f32 {
        let v = _mm512_set1_ps(4.0);
        unsafe { _mm_cvtss_f32(_mm512_castps512_ps128(v)) }
    }

    #[cfg(feature = "avx512")]
    #[arcane]
    fn with_avx512token(token: archmage::Avx512Token) -> f32 {
        let v = _mm512_set1_ps(5.0);
        unsafe { _mm_cvtss_f32(_mm512_castps512_ps128(v)) }
    }

    #[cfg(feature = "avx512")]
    #[arcane]
    fn with_server64(token: archmage::Server64) -> f32 {
        let v = _mm512_set1_ps(6.0);
        unsafe { _mm_cvtss_f32(_mm512_castps512_ps128(v)) }
    }

    #[cfg(feature = "avx512")]
    #[arcane]
    fn with_x64v4x(token: archmage::X64V4xToken) -> f32 {
        let v = _mm512_set1_ps(7.0);
        unsafe { _mm_cvtss_f32(_mm512_castps512_ps128(v)) }
    }

    #[cfg(feature = "avx512")]
    #[arcane]
    fn with_avx512fp16(token: archmage::Avx512Fp16Token) -> f32 {
        let v = _mm512_set1_ps(8.0);
        unsafe { _mm_cvtss_f32(_mm512_castps512_ps128(v)) }
    }

    // --- Runtime tests for concrete tokens ---

    #[test]
    fn test_concrete_x64v2() {
        if let Some(t) = archmage::X64V2Token::summon() {
            let _ = with_x64v2(t);
        }
    }

    #[test]
    fn test_concrete_x64v3() {
        if let Some(t) = archmage::X64V3Token::summon() {
            assert_eq!(with_x64v3(t), 6.0); // 2*2+2
        }
    }

    #[test]
    fn test_concrete_desktop64() {
        if let Some(t) = archmage::Desktop64::summon() {
            assert_eq!(with_desktop64(t), 12.0); // 3*3+3
        }
    }

    #[test]
    fn test_concrete_avx2fma() {
        if let Some(t) = archmage::Avx2FmaToken::summon() {
            assert_eq!(with_avx2fma(t), 7.0); // 2*3+1
        }
    }

    #[cfg(feature = "avx512")]
    #[test]
    fn test_concrete_avx512_tokens() {
        // These may or may not be available depending on hardware
        if let Some(t) = archmage::X64V4Token::summon() {
            assert_eq!(with_x64v4(t), 4.0);
        }
        if let Some(t) = archmage::Avx512Token::summon() {
            assert_eq!(with_avx512token(t), 5.0);
        }
        if let Some(t) = archmage::Server64::summon() {
            assert_eq!(with_server64(t), 6.0);
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_impl_trait {
    //! Test impl Trait bounds with #[arcane].
    use archmage::{SimdToken, arcane};
    use std::arch::x86_64::*;

    #[arcane]
    fn with_has128(token: impl archmage::Has128BitSimd) -> f32 {
        let v = _mm_set1_ps(1.0);
        unsafe { _mm_cvtss_f32(v) }
    }

    #[arcane]
    fn with_has256(token: impl archmage::Has256BitSimd) -> f32 {
        let v = _mm256_set1_ps(2.0);
        let lo = _mm256_castps256_ps128(v);
        unsafe { _mm_cvtss_f32(lo) }
    }

    #[arcane]
    fn with_has512(token: impl archmage::Has512BitSimd) -> f32 {
        let v = _mm512_set1_ps(3.0);
        unsafe { _mm_cvtss_f32(_mm512_castps512_ps128(v)) }
    }

    #[arcane]
    fn with_hasx64v2(token: impl archmage::HasX64V2) -> u32 {
        _mm_crc32_u8(0, 0)
    }

    #[arcane]
    fn with_hasx64v4(token: impl archmage::HasX64V4) -> f32 {
        let v = _mm512_set1_ps(4.0);
        unsafe { _mm_cvtss_f32(_mm512_castps512_ps128(v)) }
    }

    #[test]
    fn test_impl_trait_has128() {
        // X64V2Token implements Has128BitSimd
        if let Some(t) = archmage::X64V2Token::summon() {
            assert_eq!(with_has128(t), 1.0);
        }
    }

    #[test]
    fn test_impl_trait_has256() {
        // X64V3Token implements Has256BitSimd
        if let Some(t) = archmage::X64V3Token::summon() {
            assert_eq!(with_has256(t), 2.0);
        }
        // Avx2FmaToken (alias for X64V3Token) also has Has256BitSimd
        if let Some(t) = archmage::Avx2FmaToken::summon() {
            assert_eq!(with_has256(t), 2.0);
        }
    }

    #[cfg(feature = "avx512")]
    #[test]
    fn test_impl_trait_has512() {
        if let Some(t) = archmage::X64V4Token::summon() {
            assert_eq!(with_has512(t), 3.0);
        }
    }

    #[test]
    fn test_impl_trait_hasx64v2() {
        if let Some(t) = archmage::X64V2Token::summon() {
            let _ = with_hasx64v2(t);
        }
        // X64V3Token also has HasX64V2
        if let Some(t) = archmage::X64V3Token::summon() {
            let _ = with_hasx64v2(t);
        }
    }

    #[cfg(feature = "avx512")]
    #[test]
    fn test_impl_trait_hasx64v4() {
        if let Some(t) = archmage::X64V4Token::summon() {
            assert_eq!(with_hasx64v4(t), 4.0);
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_generic_inline {
    //! Test generic type parameters with inline bounds.
    use archmage::{SimdToken, arcane};
    use std::arch::x86_64::*;

    #[arcane]
    fn generic_has128<T: archmage::Has128BitSimd>(token: T) -> f32 {
        let v = _mm_set1_ps(10.0);
        unsafe { _mm_cvtss_f32(v) }
    }

    #[arcane]
    fn generic_has256<T: archmage::Has256BitSimd>(token: T) -> f32 {
        let v = _mm256_set1_ps(20.0);
        let lo = _mm256_castps256_ps128(v);
        unsafe { _mm_cvtss_f32(lo) }
    }

    #[arcane]
    fn generic_hasx64v2<T: archmage::HasX64V2>(token: T) -> u32 {
        _mm_crc32_u8(0, 0)
    }

    #[arcane]
    fn generic_hasx64v4<T: archmage::HasX64V4>(token: T) -> f32 {
        let v = _mm512_set1_ps(40.0);
        unsafe { _mm_cvtss_f32(_mm512_castps512_ps128(v)) }
    }

    #[test]
    fn test_generic_inline_has128() {
        if let Some(t) = archmage::X64V2Token::summon() {
            assert_eq!(generic_has128(t), 10.0);
        }
    }

    #[test]
    fn test_generic_inline_has256() {
        if let Some(t) = archmage::X64V3Token::summon() {
            assert_eq!(generic_has256(t), 20.0);
        }
    }

    #[test]
    fn test_generic_inline_hasx64v2() {
        if let Some(t) = archmage::X64V2Token::summon() {
            let _ = generic_hasx64v2(t);
        }
    }

    #[cfg(feature = "avx512")]
    #[test]
    fn test_generic_inline_hasx64v4() {
        if let Some(t) = archmage::X64V4Token::summon() {
            assert_eq!(generic_hasx64v4(t), 40.0);
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_generic_where {
    //! Test generic type parameters with where clauses.
    use archmage::{SimdToken, arcane};
    use std::arch::x86_64::*;

    #[arcane]
    fn where_has128<T>(token: T) -> f32
    where
        T: archmage::Has128BitSimd,
    {
        let v = _mm_set1_ps(100.0);
        unsafe { _mm_cvtss_f32(v) }
    }

    #[arcane]
    fn where_has256<T>(token: T) -> f32
    where
        T: archmage::Has256BitSimd,
    {
        let v = _mm256_set1_ps(200.0);
        let lo = _mm256_castps256_ps128(v);
        unsafe { _mm_cvtss_f32(lo) }
    }

    #[arcane]
    fn where_hasx64v2<T>(token: T) -> u32
    where
        T: archmage::HasX64V2,
    {
        _mm_crc32_u8(0, 0)
    }

    #[arcane]
    fn where_hasx64v4<T>(token: T) -> f32
    where
        T: archmage::HasX64V4,
    {
        let v = _mm512_set1_ps(400.0);
        unsafe { _mm_cvtss_f32(_mm512_castps512_ps128(v)) }
    }

    #[test]
    fn test_where_has128() {
        if let Some(t) = archmage::X64V2Token::summon() {
            assert_eq!(where_has128(t), 100.0);
        }
    }

    #[test]
    fn test_where_has256() {
        if let Some(t) = archmage::Avx2FmaToken::summon() {
            assert_eq!(where_has256(t), 200.0);
        }
    }

    #[test]
    fn test_where_hasx64v2() {
        if let Some(t) = archmage::X64V3Token::summon() {
            let _ = where_hasx64v2(t);
        }
    }

    #[cfg(feature = "avx512")]
    #[test]
    fn test_where_hasx64v4() {
        if let Some(t) = archmage::X64V4Token::summon() {
            assert_eq!(where_hasx64v4(t), 400.0);
        }
    }
}

// =============================================================================
// Cross-token compatibility tests
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod x86_cross_compat {
    //! Verify that higher tokens can be passed to functions expecting lower bounds.
    use archmage::{SimdToken, arcane};
    use std::arch::x86_64::*;

    #[arcane]
    fn needs_has128(token: impl archmage::Has128BitSimd) -> f32 {
        let v = _mm_set1_ps(1.0);
        unsafe { _mm_cvtss_f32(v) }
    }

    #[arcane]
    fn needs_has256(token: impl archmage::Has256BitSimd) -> f32 {
        let v = _mm256_set1_ps(2.0);
        let lo = _mm256_castps256_ps128(v);
        unsafe { _mm_cvtss_f32(lo) }
    }

    #[arcane]
    fn needs_hasx64v2(token: impl archmage::HasX64V2) -> u32 {
        _mm_crc32_u8(0, 0)
    }

    /// X64V3Token implements Has128BitSimd, Has256BitSimd, and HasX64V2
    #[test]
    fn test_x64v3_satisfies_lower_bounds() {
        if let Some(t) = archmage::X64V3Token::summon() {
            assert_eq!(needs_has128(t), 1.0);
            assert_eq!(needs_has256(t), 2.0);
            let _ = needs_hasx64v2(t);
        }
    }

    /// Desktop64 is X64V3Token
    #[test]
    fn test_desktop64_satisfies_lower_bounds() {
        if let Some(t) = archmage::Desktop64::summon() {
            assert_eq!(needs_has128(t), 1.0);
            assert_eq!(needs_has256(t), 2.0);
            let _ = needs_hasx64v2(t);
        }
    }

    /// Avx2FmaToken (alias for X64V3Token) implements Has128BitSimd, Has256BitSimd, and HasX64V2
    #[test]
    fn test_avx2fma_satisfies_lower_bounds() {
        if let Some(t) = archmage::Avx2FmaToken::summon() {
            assert_eq!(needs_has128(t), 1.0);
            assert_eq!(needs_has256(t), 2.0);
            let _ = needs_hasx64v2(t);
        }
    }

    /// X64V4Token implements everything
    #[cfg(feature = "avx512")]
    #[test]
    fn test_x64v4_satisfies_all_bounds() {
        if let Some(t) = archmage::X64V4Token::summon() {
            assert_eq!(needs_has128(t), 1.0);
            assert_eq!(needs_has256(t), 2.0);
            let _ = needs_hasx64v2(t);
        }
    }

    /// Server64 is X64V4Token
    #[cfg(feature = "avx512")]
    #[test]
    fn test_server64_satisfies_all_bounds() {
        if let Some(t) = archmage::Server64::summon() {
            assert_eq!(needs_has128(t), 1.0);
            assert_eq!(needs_has256(t), 2.0);
            let _ = needs_hasx64v2(t);
        }
    }
}

// =============================================================================
// Token alias identity tests
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod x86_alias_identity {
    //! Verify that type aliases are truly identical.
    use archmage::SimdToken;

    #[test]
    fn desktop64_is_x64v3() {
        // Desktop64 is a type alias for X64V3Token
        let _: Option<archmage::Desktop64> = archmage::X64V3Token::summon();
        let _: Option<archmage::X64V3Token> = archmage::Desktop64::summon();
    }

    #[test]
    fn avx2fma_is_x64v3() {
        // Avx2FmaToken is a type alias for X64V3Token
        let _: Option<archmage::Avx2FmaToken> = archmage::X64V3Token::summon();
        let _: Option<archmage::X64V3Token> = archmage::Avx2FmaToken::summon();
    }

    #[cfg(feature = "avx512")]
    #[test]
    fn server64_is_x64v4_is_avx512() {
        // All three are aliases for the same type
        let _: Option<archmage::Server64> = archmage::X64V4Token::summon();
        let _: Option<archmage::X64V4Token> = archmage::Avx512Token::summon();
        let _: Option<archmage::Avx512Token> = archmage::Server64::summon();
    }
}

// =============================================================================
// ARM tests (compile-only on non-ARM, runtime on ARM)
// =============================================================================

#[cfg(target_arch = "aarch64")]
mod arm_concrete_tokens {
    use archmage::{SimdToken, arcane};
    use std::arch::aarch64::*;

    #[arcane]
    fn with_neon(token: archmage::NeonToken) -> f32 {
        let v = vdupq_n_f32(1.0);
        vgetq_lane_f32::<0>(v)
    }

    #[arcane]
    fn with_arm64(token: archmage::Arm64) -> f32 {
        let v = vdupq_n_f32(2.0);
        vgetq_lane_f32::<0>(v)
    }

    #[arcane]
    fn with_neon_aes(token: archmage::NeonAesToken) -> f32 {
        let v = vdupq_n_f32(3.0);
        vgetq_lane_f32::<0>(v)
    }

    #[arcane]
    fn with_neon_sha3(token: archmage::NeonSha3Token) -> f32 {
        let v = vdupq_n_f32(4.0);
        vgetq_lane_f32::<0>(v)
    }

    #[arcane]
    fn with_neon_crc(token: archmage::NeonCrcToken) -> f32 {
        let v = vdupq_n_f32(5.0);
        vgetq_lane_f32::<0>(v)
    }

    // impl Trait forms

    #[arcane]
    fn with_has_neon(token: impl archmage::HasNeon) -> f32 {
        let v = vdupq_n_f32(10.0);
        vgetq_lane_f32::<0>(v)
    }

    #[arcane]
    fn with_has_neon_aes(token: impl archmage::HasNeonAes) -> f32 {
        let v = vdupq_n_f32(20.0);
        vgetq_lane_f32::<0>(v)
    }

    #[arcane]
    fn with_has_neon_sha3(token: impl archmage::HasNeonSha3) -> f32 {
        let v = vdupq_n_f32(30.0);
        vgetq_lane_f32::<0>(v)
    }

    // Generic forms

    #[arcane]
    fn generic_has_neon<T: archmage::HasNeon>(token: T) -> f32 {
        let v = vdupq_n_f32(100.0);
        vgetq_lane_f32::<0>(v)
    }

    #[arcane]
    fn where_has_neon<T>(token: T) -> f32
    where
        T: archmage::HasNeon,
    {
        let v = vdupq_n_f32(200.0);
        vgetq_lane_f32::<0>(v)
    }

    #[test]
    fn test_arm_concrete() {
        if let Some(t) = archmage::NeonToken::summon() {
            assert_eq!(with_neon(t), 1.0);
        }
        if let Some(t) = archmage::Arm64::summon() {
            assert_eq!(with_arm64(t), 2.0);
        }
        if let Some(t) = archmage::NeonCrcToken::summon() {
            assert_eq!(with_neon_crc(t), 5.0);
        }
    }

    #[test]
    fn test_arm_impl_trait() {
        if let Some(t) = archmage::NeonToken::summon() {
            assert_eq!(with_has_neon(t), 10.0);
        }
    }

    #[test]
    fn test_arm_generic() {
        if let Some(t) = archmage::NeonToken::summon() {
            assert_eq!(generic_has_neon(t), 100.0);
            assert_eq!(where_has_neon(t), 200.0);
        }
    }

    #[test]
    fn arm64_is_neon() {
        let _: Option<archmage::Arm64> = archmage::NeonToken::summon();
        let _: Option<archmage::NeonToken> = archmage::Arm64::summon();
    }
}

// =============================================================================
// WASM tests (compile-only, not runnable on x86 or ARM)
// =============================================================================

#[cfg(target_arch = "wasm32")]
mod wasm_tokens {
    use archmage::{SimdToken, arcane};
    use std::arch::wasm32::*;

    #[arcane]
    fn with_simd128(token: archmage::Wasm128Token) -> f32 {
        let v = f32x4_splat(1.0);
        f32x4_extract_lane::<0>(v)
    }

    #[test]
    fn test_wasm_concrete() {
        if let Some(t) = archmage::Wasm128Token::summon() {
            assert_eq!(with_simd128(t), 1.0);
        }
    }
}

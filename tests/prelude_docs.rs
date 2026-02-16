//! Tests backing every claim in the prelude documentation.
//!
//! Each test is named after the doc claim it verifies.

// =============================================================================
// Category 1: Traits are re-exported
// =============================================================================

#[test]
fn simd_token_trait_re_exported() {
    use archmage::prelude::SimdToken;
    // SimdToken provides summon() and compiled_with()
    let _ = archmage::ScalarToken::summon();
    let _ = archmage::ScalarToken::compiled_with();
}

#[test]
fn into_concrete_token_re_exported() {
    use archmage::prelude::IntoConcreteToken;
    let token = archmage::ScalarToken;
    assert!(token.as_scalar().is_some());
}

#[test]
fn tier_traits_re_exported() {
    // These must compile — trait names are accessible
    fn _requires_v2(_: impl archmage::prelude::HasX64V2) {}
    fn _requires_neon(_: impl archmage::prelude::HasNeon) {}
    fn _requires_neon_aes(_: impl archmage::prelude::HasNeonAes) {}
    fn _requires_neon_sha3(_: impl archmage::prelude::HasNeonSha3) {}
}

#[cfg(feature = "avx512")]
#[test]
fn avx512_tier_trait_re_exported() {
    fn _requires_v4(_: impl archmage::prelude::HasX64V4) {}
}

// =============================================================================
// Category 2: Tokens
// =============================================================================

#[test]
fn desktop64_is_x64v3token() {
    use archmage::prelude::*;
    // Desktop64 and X64V3Token are the same type (type alias)
    // Prove they're interchangeable: summon one, use as the other
    let token: Option<Desktop64> = X64V3Token::summon();
    let _: Option<X64V3Token> = token;
    assert_eq!(Desktop64::compiled_with(), X64V3Token::compiled_with());
}

#[cfg(feature = "avx512")]
#[test]
fn server64_is_x64v4token() {
    use archmage::prelude::*;
    let token: Option<Server64> = X64V4Token::summon();
    let _: Option<X64V4Token> = token;
    assert_eq!(Server64::compiled_with(), X64V4Token::compiled_with());
}

#[test]
fn arm64_is_neon_token() {
    use archmage::prelude::*;
    let token: Option<Arm64> = NeonToken::summon();
    let _: Option<NeonToken> = token;
    assert_eq!(Arm64::compiled_with(), NeonToken::compiled_with());
}

#[test]
fn scalar_token_always_available() {
    use archmage::prelude::*;
    assert!(ScalarToken::summon().is_some());
    assert_eq!(ScalarToken::compiled_with(), Some(true));
}

#[test]
fn all_token_types_accessible_from_prelude() {
    use archmage::prelude::*;
    // These must all compile — the types exist
    let _ = ScalarToken::compiled_with();
    let _ = X64V2Token::compiled_with();
    let _ = X64V3Token::compiled_with();
    let _ = Desktop64::compiled_with();
    let _ = NeonToken::compiled_with();
    let _ = Arm64::compiled_with();
    let _ = NeonAesToken::compiled_with();
    let _ = NeonSha3Token::compiled_with();
    let _ = NeonCrcToken::compiled_with();
    let _ = Wasm128Token::compiled_with();
}

#[cfg(feature = "avx512")]
#[test]
fn avx512_tokens_accessible_from_prelude() {
    use archmage::prelude::*;
    let _ = X64V4Token::compiled_with();
    let _ = Avx512Token::compiled_with();
    let _ = X64V4xToken::compiled_with();
    let _ = Avx512Fp16Token::compiled_with();
    let _ = Server64::compiled_with();
}

#[test]
fn stubs_return_none_on_wrong_arch() {
    use archmage::prelude::*;
    // On x86_64, NeonToken::summon() returns None (it's a stub)
    #[cfg(target_arch = "x86_64")]
    {
        assert!(NeonToken::summon().is_none());
        assert!(Wasm128Token::summon().is_none());
    }
    // On aarch64, X64V3Token::summon() returns None
    #[cfg(target_arch = "aarch64")]
    {
        assert!(X64V3Token::summon().is_none());
        assert!(Wasm128Token::summon().is_none());
    }
}

// =============================================================================
// Category 3: Macros
// =============================================================================

#[cfg(all(target_arch = "x86_64", feature = "macros"))]
mod macro_tests {
    use archmage::prelude::*;

    #[arcane]
    fn arcane_via_prelude(_token: Desktop64, x: f32) -> f32 {
        x * 2.0
    }

    #[rite]
    fn rite_via_prelude(_token: Desktop64, x: f32) -> f32 {
        x + 1.0
    }

    #[test]
    fn arcane_works_through_prelude() {
        if let Some(token) = Desktop64::summon() {
            assert_eq!(arcane_via_prelude(token, 21.0), 42.0);
        }
    }

    #[test]
    fn rite_works_through_prelude() {
        if let Some(token) = Desktop64::summon() {
            // rite functions are unsafe to call directly (from non-target_feature context)
            let result = unsafe { rite_via_prelude(token, 41.0) };
            assert_eq!(result, 42.0);
        }
    }
}

// =============================================================================
// Category 4: Platform intrinsics (non-overlapping value intrinsics)
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod platform_intrinsic_tests {
    use archmage::prelude::*;

    #[test]
    fn m256_type_accessible() {
        // __m256 comes from core::arch::x86_64 via prelude
        let _: __m256 = unsafe { _mm256_setzero_ps() };
    }

    #[arcane]
    fn setzero_via_prelude(_token: Desktop64) -> __m256 {
        _mm256_setzero_ps() // Safe inside #[arcane]
    }

    #[test]
    fn value_intrinsics_safe_inside_arcane() {
        if let Some(token) = Desktop64::summon() {
            let v = setzero_via_prelude(token);
            let mut out = [0.0f32; 8];
            unsafe { std::arch::x86_64::_mm256_storeu_ps(out.as_mut_ptr(), v) };
            assert_eq!(out, [0.0; 8]);
        }
    }

    /// Value-only intrinsics (add, mul, fma, shuffle) resolve through the
    /// prelude because they only exist in core::arch — no overlap with
    /// safe_unaligned_simd.
    #[arcane]
    fn value_ops_via_prelude(_token: Desktop64, a: __m256, b: __m256) -> __m256 {
        let sum = _mm256_add_ps(a, b);
        let product = _mm256_mul_ps(sum, sum);
        _mm256_fmadd_ps(product, a, b)
    }

    #[test]
    fn value_intrinsics_resolve_through_prelude() {
        if let Some(token) = Desktop64::summon() {
            let a = unsafe { _mm256_set1_ps(1.0) };
            let b = unsafe { _mm256_set1_ps(2.0) };
            let _result = value_ops_via_prelude(token, a, b);
            // Just verifying it compiles and runs
        }
    }
}

// =============================================================================
// Category 5: Safe memory ops — explicit imports required
// =============================================================================

#[cfg(all(target_arch = "x86_64", feature = "safe_unaligned_simd"))]
mod safe_memory_tests {
    use archmage::prelude::*;
    // Memory ops (load/store) overlap between core::arch and safe_unaligned_simd,
    // so they must be imported explicitly. The prelude gives you everything else.
    use safe_unaligned_simd::x86_64::{_mm256_loadu_ps, _mm256_storeu_ps};

    #[arcane]
    fn safe_load(_token: Desktop64, data: &[f32; 8]) -> __m256 {
        _mm256_loadu_ps(data)
    }

    #[arcane]
    fn safe_store(_token: Desktop64, v: __m256, out: &mut [f32; 8]) {
        _mm256_storeu_ps(out, v)
    }

    #[test]
    fn safe_load_takes_reference() {
        if let Some(token) = Desktop64::summon() {
            let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let v = safe_load(token, &data);
            let mut out = [0.0f32; 8];
            unsafe { std::arch::x86_64::_mm256_storeu_ps(out.as_mut_ptr(), v) };
            assert_eq!(out, data);
        }
    }

    #[test]
    fn safe_store_takes_mut_reference() {
        if let Some(token) = Desktop64::summon() {
            let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let v = safe_load(token, &data);
            let mut out = [0.0f32; 8];
            safe_store(token, v, &mut out);
            assert_eq!(out, data);
        }
    }

    #[test]
    fn safe_load_matches_unsafe_load() {
        if let Some(token) = Desktop64::summon() {
            let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

            // Safe version (via explicit safe_unaligned_simd import)
            let safe_v = safe_load(token, &data);

            // Unsafe version (explicit core::arch)
            let unsafe_v = unsafe { core::arch::x86_64::_mm256_loadu_ps(data.as_ptr()) };

            // Compare bit-for-bit
            let mut safe_out = [0.0f32; 8];
            let mut unsafe_out = [0.0f32; 8];
            unsafe {
                std::arch::x86_64::_mm256_storeu_ps(safe_out.as_mut_ptr(), safe_v);
                std::arch::x86_64::_mm256_storeu_ps(unsafe_out.as_mut_ptr(), unsafe_v);
            }
            assert_eq!(safe_out, unsafe_out);
        }
    }

    #[test]
    fn safe_store_matches_unsafe_store() {
        if let Some(token) = Desktop64::summon() {
            let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let v = safe_load(token, &data);

            // Safe version
            let mut safe_out = [0.0f32; 8];
            safe_store(token, v, &mut safe_out);

            // Unsafe version
            let mut unsafe_out = [0.0f32; 8];
            unsafe { std::arch::x86_64::_mm256_storeu_ps(unsafe_out.as_mut_ptr(), v) };

            assert_eq!(safe_out, unsafe_out);
        }
    }

    /// An #[arcane] function with ZERO unsafe blocks.
    /// Value intrinsics + safe memory ops = fully safe SIMD.
    #[arcane]
    fn multiply_and_add_safe(
        _token: Desktop64,
        a: &[f32; 8],
        b: &[f32; 8],
        c: &[f32; 8],
    ) -> [f32; 8] {
        let va = _mm256_loadu_ps(a);
        let vb = _mm256_loadu_ps(b);
        let vc = _mm256_loadu_ps(c);
        let result = _mm256_fmadd_ps(va, vb, vc);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(&mut out, result);
        out
    }

    #[test]
    fn complete_arcane_function_with_zero_unsafe() {
        if let Some(token) = Desktop64::summon() {
            let a = [1.0f32; 8];
            let b = [2.0f32; 8];
            let c = [3.0f32; 8];
            let result = multiply_and_add_safe(token, &a, &b, &c);
            // a*b + c = 1*2 + 3 = 5
            assert_eq!(result, [5.0f32; 8]);
        }
    }
}

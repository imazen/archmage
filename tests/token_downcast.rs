//! Comprehensive tests for token downcast (extraction) methods and
//! IntoConcreteToken coverage for all token types.
//!
//! Extraction methods (e.g., `.v1()`, `.neon()`) only exist on real tokens,
//! not stubs. So downcast tests are cfg-gated to the correct architecture.
//! IntoConcreteToken and SimdToken trait methods work on stubs too.

#[allow(deprecated)] // forge_token_dangerously
use archmage::*;

// ============================================================================
// x86 extraction methods (downcast: higher → lower)
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod x86_downcast {
    use super::*;

    #[test]
    fn v2_extracts_to_v1() {
        if let Some(token) = X64V2Token::summon() {
            let _v1: X64V1Token = token.v1();
        }
    }

    #[test]
    fn crypto_extracts_to_v2_and_v1() {
        if let Some(token) = X64CryptoToken::summon() {
            let _v2: X64V2Token = token.v2();
            let _v1: X64V1Token = token.v1();
        }
    }

    #[test]
    fn v3_extracts_to_v2_and_v1() {
        if let Some(token) = X64V3Token::summon() {
            let _v2: X64V2Token = token.v2();
            let _v1: X64V1Token = token.v1();
        }
    }

    #[test]
    fn v3crypto_extracts_to_v3_v2_v1() {
        if let Some(token) = X64V3CryptoToken::summon() {
            let _v3: X64V3Token = token.v3();
            let _v2: X64V2Token = token.v2();
            let _v1: X64V1Token = token.v1();
        }
    }

    #[cfg(feature = "avx512")]
    #[test]
    fn v4_extracts_to_v3_v2_v1() {
        if let Some(token) = X64V4Token::summon() {
            let _v3: X64V3Token = token.v3();
            let _v2: X64V2Token = token.v2();
            let _v1: X64V1Token = token.v1();
        }
    }

    #[cfg(feature = "avx512")]
    #[test]
    fn v4x_extracts_to_v4_avx512_v3_v2_v1() {
        if let Some(token) = X64V4xToken::summon() {
            let _v4: X64V4Token = token.v4();
            let _avx512: X64V4Token = token.avx512();
            let _v3: X64V3Token = token.v3();
            let _v2: X64V2Token = token.v2();
            let _v1: X64V1Token = token.v1();
        }
    }

    #[cfg(feature = "avx512")]
    #[test]
    fn fp16_extracts_to_v4_avx512_v3_v2_v1() {
        if let Some(token) = Avx512Fp16Token::summon() {
            let _v4: X64V4Token = token.v4();
            let _avx512: X64V4Token = token.avx512();
            let _v3: X64V3Token = token.v3();
            let _v2: X64V2Token = token.v2();
            let _v1: X64V1Token = token.v1();
        }
    }

    // Extraction chain: verify chained downcasts work
    #[test]
    fn chained_downcast_v3_to_v2_to_v1() {
        if let Some(v3) = X64V3Token::summon() {
            let v2 = v3.v2();
            let _v1 = v2.v1();
        }
    }

    #[cfg(feature = "avx512")]
    #[test]
    fn chained_downcast_v4x_full_chain() {
        if let Some(v4x) = X64V4xToken::summon() {
            let v4 = v4x.v4();
            let v3 = v4.v3();
            let v2 = v3.v2();
            let _v1 = v2.v1();
        }
    }
}

// ============================================================================
// ARM extraction methods (only on aarch64)
// ============================================================================

#[cfg(target_arch = "aarch64")]
mod arm_downcast {
    use super::*;

    #[test]
    fn neon_aes_extracts_to_neon() {
        if let Some(token) = NeonAesToken::summon() {
            let _neon: NeonToken = token.neon();
        }
    }

    #[test]
    fn neon_sha3_extracts_to_neon() {
        if let Some(token) = NeonSha3Token::summon() {
            let _neon: NeonToken = token.neon();
        }
    }

    #[test]
    fn neon_crc_extracts_to_neon() {
        if let Some(token) = NeonCrcToken::summon() {
            let _neon: NeonToken = token.neon();
        }
    }

    #[test]
    fn arm64v2_extracts_to_neon() {
        if let Some(token) = Arm64V2Token::summon() {
            let _neon: NeonToken = token.neon();
        }
    }

    #[test]
    fn arm64v3_extracts_to_arm_v2_and_neon() {
        if let Some(token) = Arm64V3Token::summon() {
            let _v2: Arm64V2Token = token.arm_v2();
            let _neon: NeonToken = token.neon();
        }
    }

    #[test]
    fn arm_chained_downcast_v3_to_v2_to_neon() {
        if let Some(v3) = Arm64V3Token::summon() {
            let v2 = v3.arm_v2();
            let _neon = v2.neon();
        }
    }
}

// ============================================================================
// WASM extraction methods (only on wasm32)
// ============================================================================

#[cfg(target_arch = "wasm32")]
mod wasm_downcast {
    use super::*;

    #[test]
    fn wasm128_relaxed_extracts_to_wasm128() {
        if let Some(token) = Wasm128RelaxedToken::summon() {
            let _wasm128: Wasm128Token = token.wasm128();
        }
    }
}

// ============================================================================
// IntoConcreteToken: comprehensive coverage for all token types
// (Works with stubs — forge tokens for cross-arch testing)
// ============================================================================

#[allow(deprecated)]
mod into_concrete {
    use super::*;

    #[test]
    fn x64v1_into_concrete() {
        let token = if let Some(t) = X64V1Token::summon() {
            t
        } else {
            unsafe { X64V1Token::forge_token_dangerously() }
        };
        assert!(token.as_x64v1().is_some());
        assert!(token.as_x64v2().is_none());
        assert!(token.as_x64v3().is_none());
        assert!(token.as_neon().is_none());
        assert!(token.as_scalar().is_none());
    }

    #[test]
    fn x64_crypto_into_concrete() {
        let token = if let Some(t) = X64CryptoToken::summon() {
            t
        } else {
            unsafe { X64CryptoToken::forge_token_dangerously() }
        };
        assert!(token.as_x64_crypto().is_some());
        assert!(
            token.as_x64v2().is_none(),
            "Crypto is not V2 (different type)"
        );
        assert!(token.as_x64v3().is_none());
        assert!(token.as_neon().is_none());
    }

    #[test]
    fn x64v3_crypto_into_concrete() {
        let token = if let Some(t) = X64V3CryptoToken::summon() {
            t
        } else {
            unsafe { X64V3CryptoToken::forge_token_dangerously() }
        };
        assert!(token.as_x64v3_crypto().is_some());
        assert!(
            token.as_x64v3().is_none(),
            "V3Crypto is not V3 (different type)"
        );
        assert!(token.as_x64_crypto().is_none());
        assert!(token.as_neon().is_none());
    }

    #[test]
    fn arm64v2_into_concrete() {
        let token = if let Some(t) = Arm64V2Token::summon() {
            t
        } else {
            unsafe { Arm64V2Token::forge_token_dangerously() }
        };
        assert!(token.as_arm_v2().is_some());
        assert!(
            token.as_neon().is_none(),
            "Arm64V2 is not Neon (different type)"
        );
        assert!(token.as_arm_v3().is_none());
        assert!(token.as_x64v3().is_none());
    }

    #[test]
    fn arm64v3_into_concrete() {
        let token = if let Some(t) = Arm64V3Token::summon() {
            t
        } else {
            unsafe { Arm64V3Token::forge_token_dangerously() }
        };
        assert!(token.as_arm_v3().is_some());
        assert!(
            token.as_arm_v2().is_none(),
            "Arm64V3 is not Arm64V2 (different type)"
        );
        assert!(token.as_neon().is_none());
        assert!(token.as_x64v3().is_none());
    }

    #[test]
    fn wasm128_relaxed_into_concrete() {
        let token = if let Some(t) = Wasm128RelaxedToken::summon() {
            t
        } else {
            unsafe { Wasm128RelaxedToken::forge_token_dangerously() }
        };
        assert!(token.as_wasm128_relaxed().is_some());
        assert!(
            token.as_wasm128().is_none(),
            "Relaxed is not base Wasm128 (different type)"
        );
        assert!(token.as_neon().is_none());
    }

    #[cfg(feature = "avx512")]
    #[test]
    fn x64v4x_into_concrete() {
        let token = if let Some(t) = X64V4xToken::summon() {
            t
        } else {
            unsafe { X64V4xToken::forge_token_dangerously() }
        };
        assert!(token.as_x64v4x().is_some());
        assert!(token.as_x64v4().is_none(), "V4x is not V4 (different type)");
        assert!(token.as_x64v3().is_none());
        assert!(token.as_neon().is_none());
    }

    // Verify that IntoConcreteToken returns None for all wrong types
    #[test]
    fn neon_rejects_all_x86_types() {
        let token = if let Some(t) = NeonToken::summon() {
            t
        } else {
            unsafe { NeonToken::forge_token_dangerously() }
        };
        assert!(token.as_neon().is_some());
        assert!(token.as_x64v1().is_none());
        assert!(token.as_x64v2().is_none());
        assert!(token.as_x64v3().is_none());
        assert!(token.as_x64_crypto().is_none());
        assert!(token.as_x64v3_crypto().is_none());
        assert!(token.as_wasm128().is_none());
        assert!(token.as_scalar().is_none());
        assert!(token.as_neon_aes().is_none());
        assert!(token.as_neon_sha3().is_none());
        assert!(token.as_neon_crc().is_none());
        assert!(token.as_arm_v2().is_none());
        assert!(token.as_arm_v3().is_none());
        assert!(token.as_wasm128_relaxed().is_none());
    }

    #[test]
    fn scalar_rejects_all_simd_types() {
        let token = ScalarToken::summon().unwrap();
        assert!(token.as_scalar().is_some());
        assert!(token.as_x64v1().is_none());
        assert!(token.as_x64v2().is_none());
        assert!(token.as_x64v3().is_none());
        assert!(token.as_x64_crypto().is_none());
        assert!(token.as_x64v3_crypto().is_none());
        assert!(token.as_neon().is_none());
        assert!(token.as_neon_aes().is_none());
        assert!(token.as_neon_sha3().is_none());
        assert!(token.as_neon_crc().is_none());
        assert!(token.as_arm_v2().is_none());
        assert!(token.as_arm_v3().is_none());
        assert!(token.as_wasm128().is_none());
        assert!(token.as_wasm128_relaxed().is_none());
    }

    #[cfg(feature = "avx512")]
    #[test]
    fn scalar_rejects_avx512_types() {
        let token = ScalarToken::summon().unwrap();
        assert!(token.as_x64v4().is_none());
        assert!(token.as_x64v4x().is_none());
        assert!(token.as_avx512_fp16().is_none());
    }
}

// ============================================================================
// X64V1Token: baseline, always available on x86_64
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod x64v1 {
    use super::*;

    #[test]
    fn v1_always_available_on_x86_64() {
        assert!(X64V1Token::summon().is_some());
    }

    #[test]
    fn v1_compiled_with_always_true_on_x86_64() {
        // SSE2 is the x86_64 baseline — always compile-time guaranteed.
        // With testable_dispatch, compiled_with() returns None to force runtime detection.
        #[cfg(not(feature = "testable_dispatch"))]
        assert_eq!(X64V1Token::compiled_with(), Some(true));
        #[cfg(feature = "testable_dispatch")]
        assert_eq!(X64V1Token::compiled_with(), None);
    }

    #[test]
    fn v1_name_is_nonempty() {
        assert!(!X64V1Token::NAME.is_empty());
    }

    #[test]
    fn v1_features_contain_sse2() {
        assert!(X64V1Token::TARGET_FEATURES.contains("sse2"));
    }

    #[test]
    fn sse2token_is_v1() {
        assert_eq!(Sse2Token::compiled_with(), X64V1Token::compiled_with());
        assert_eq!(
            Sse2Token::summon().is_some(),
            X64V1Token::summon().is_some()
        );
        assert_eq!(Sse2Token::NAME, X64V1Token::NAME);
    }
}

// ============================================================================
// X64CryptoToken / X64V3CryptoToken basics
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod crypto_tokens {
    use super::*;

    #[test]
    fn crypto_features_contain_aes_and_pclmulqdq() {
        let features = X64CryptoToken::TARGET_FEATURES;
        assert!(features.contains("aes"), "Crypto should have aes");
        assert!(
            features.contains("pclmulqdq"),
            "Crypto should have pclmulqdq"
        );
    }

    #[test]
    fn v3crypto_features_contain_vaes_and_vpclmulqdq() {
        let features = X64V3CryptoToken::TARGET_FEATURES;
        assert!(features.contains("vaes"), "V3Crypto should have vaes");
        assert!(
            features.contains("vpclmulqdq"),
            "V3Crypto should have vpclmulqdq"
        );
        // Also contains V3 features
        assert!(features.contains("avx2"), "V3Crypto should have avx2");
        assert!(features.contains("fma"), "V3Crypto should have fma");
    }

    #[test]
    fn crypto_hierarchy_summoning() {
        // If V3Crypto is available, V3 and Crypto should also be
        if X64V3CryptoToken::summon().is_some() {
            assert!(X64V3Token::summon().is_some(), "V3Crypto implies V3");
            assert!(
                X64CryptoToken::summon().is_some(),
                "V3Crypto implies Crypto"
            );
        }
        // If Crypto is available, V2 should also be
        if X64CryptoToken::summon().is_some() {
            assert!(X64V2Token::summon().is_some(), "Crypto implies V2");
        }
    }

    #[test]
    fn crypto_names_are_nonempty() {
        assert!(!X64CryptoToken::NAME.is_empty());
        assert!(!X64V3CryptoToken::NAME.is_empty());
    }
}

// ============================================================================
// ARM token features (verifiable on any platform — reads from stubs too)
// ============================================================================

mod arm_features {
    use super::*;

    #[test]
    fn neon_aes_features_contain_aes() {
        let features = NeonAesToken::TARGET_FEATURES;
        assert!(features.contains("neon"), "NeonAes should have neon");
        assert!(features.contains("aes"), "NeonAes should have aes");
    }

    #[test]
    fn neon_sha3_features_contain_sha3() {
        let features = NeonSha3Token::TARGET_FEATURES;
        assert!(features.contains("neon"), "NeonSha3 should have neon");
        assert!(features.contains("sha3"), "NeonSha3 should have sha3");
    }

    #[test]
    fn neon_crc_features_contain_crc() {
        let features = NeonCrcToken::TARGET_FEATURES;
        assert!(features.contains("neon"), "NeonCrc should have neon");
        assert!(features.contains("crc"), "NeonCrc should have crc");
    }

    #[test]
    fn arm64v2_features_contain_tier2_set() {
        let features = Arm64V2Token::TARGET_FEATURES;
        assert!(features.contains("neon"));
        assert!(features.contains("crc"));
        assert!(features.contains("rdm"));
        assert!(features.contains("aes"));
        assert!(features.contains("sha2"));
    }

    #[test]
    fn arm64v3_features_contain_tier3_set() {
        let features = Arm64V3Token::TARGET_FEATURES;
        // Should have everything from V2 plus V3-specific
        assert!(features.contains("neon"));
        assert!(features.contains("sha3"));
        assert!(features.contains("i8mm"));
    }

    #[test]
    fn arm_token_names_are_nonempty() {
        assert!(!NeonAesToken::NAME.is_empty());
        assert!(!NeonSha3Token::NAME.is_empty());
        assert!(!NeonCrcToken::NAME.is_empty());
        assert!(!Arm64V2Token::NAME.is_empty());
        assert!(!Arm64V3Token::NAME.is_empty());
    }
}

// ============================================================================
// WASM token features (verifiable on any platform — reads from stubs too)
// ============================================================================

mod wasm_features {
    use super::*;

    #[test]
    fn wasm128_relaxed_features_contain_relaxed_simd() {
        let features = Wasm128RelaxedToken::TARGET_FEATURES;
        assert!(features.contains("simd128"), "Relaxed should have simd128");
        assert!(
            features.contains("relaxed-simd"),
            "Relaxed should have relaxed-simd"
        );
    }

    #[test]
    fn wasm128_relaxed_name_is_nonempty() {
        assert!(!Wasm128RelaxedToken::NAME.is_empty());
    }
}

// ============================================================================
// ScalarToken basics
// ============================================================================

mod scalar {
    use super::*;

    #[test]
    fn scalar_always_available() {
        assert!(ScalarToken::summon().is_some());
    }

    #[test]
    fn scalar_name_is_nonempty() {
        assert!(!ScalarToken::NAME.is_empty());
    }

    #[test]
    fn scalar_compiled_with_always_some_true() {
        assert_eq!(ScalarToken::compiled_with(), Some(true));
    }
}

// ============================================================================
// Token hierarchy: if higher token available, lower must be too
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod hierarchy {
    use super::*;

    #[test]
    fn v3_implies_v2_implies_v1() {
        if X64V3Token::summon().is_some() {
            assert!(X64V2Token::summon().is_some(), "V3 implies V2");
        }
        if X64V2Token::summon().is_some() {
            assert!(X64V1Token::summon().is_some(), "V2 implies V1");
        }
    }

    #[cfg(feature = "avx512")]
    #[test]
    fn v4x_implies_v4_implies_v3() {
        if X64V4xToken::summon().is_some() {
            assert!(X64V4Token::summon().is_some(), "V4x implies V4");
        }
        if X64V4Token::summon().is_some() {
            assert!(X64V3Token::summon().is_some(), "V4 implies V3");
        }
    }
}

//! Tests that verify feature detection is consistent with CPU capabilities.
//!
//! These tests run actual SIMD operations to verify that when a token is available,
//! the corresponding instructions actually work. If feature detection lies about
//! capabilities, these tests will crash (SIGILL).
//!
//! Cross-platform: tests for x86_64, aarch64, and wasm32.

use archmage::SimdToken;
use archmage::*;

// ============================================================================
// x86_64 Tests
// ============================================================================
#[cfg(target_arch = "x86_64")]
mod x86_64_tests {
    use super::*;

    /// SSE2 is baseline on x86_64 and always works.
    #[test]
    fn sse2_instructions_work_baseline() {
        // SSE2 is baseline on x86_64, no token needed
        unsafe {
            use core::arch::x86_64::*;
            let a = _mm_setzero_ps();
            let b = _mm_set1_ps(1.0);
            let c = _mm_add_ps(a, b);
            std::hint::black_box(c);
        }
    }

    /// If X64V2Token is available, SSE4.2 instructions must work.
    #[test]
    fn x64v2_instructions_work_when_token_available() {
        if let Some(_token) = X64V2Token::summon() {
            unsafe {
                use core::arch::x86_64::*;
                let result = _mm_crc32_u8(0, 0);
                std::hint::black_box(result);
                let v = _mm_set1_epi32(42);
                let extracted = _mm_extract_epi32::<0>(v);
                std::hint::black_box(extracted);
            }
        }
    }

    /// If X64V3Token is available, AVX2 and FMA instructions must work.
    #[test]
    fn x64v3_instructions_work_when_token_available() {
        if let Some(_token) = X64V3Token::summon() {
            unsafe {
                use core::arch::x86_64::*;
                let a = _mm256_setzero_ps();
                let b = _mm256_set1_ps(1.0);
                let c = _mm256_add_ps(a, b);
                std::hint::black_box(c);
                let d = _mm256_setzero_si256();
                let e = _mm256_set1_epi32(1);
                let f = _mm256_add_epi32(d, e);
                std::hint::black_box(f);
                let g = _mm256_set1_ps(1.0);
                let h = _mm256_set1_ps(2.0);
                let i = _mm256_set1_ps(3.0);
                let j = _mm256_fmadd_ps(g, h, i);
                std::hint::black_box(j);
            }
        }
    }

    /// If X64CryptoToken is available, PCLMULQDQ and AES-NI must work.
    #[test]
    fn x64_crypto_instructions_work_when_token_available() {
        if let Some(_token) = X64CryptoToken::summon() {
            unsafe {
                use core::arch::x86_64::*;
                let a = _mm_set_epi64x(0x0123456789ABCDEFi64, 0x1111111111111111i64);
                let clmul = _mm_clmulepi64_si128::<0x00>(a, a);
                std::hint::black_box(clmul);
                let key = _mm_setzero_si128();
                let enc = _mm_aesenc_si128(a, key);
                std::hint::black_box(enc);
            }
        }
    }

    /// If X64V3CryptoToken is available, VPCLMULQDQ and VAES must work.
    #[test]
    fn x64v3_crypto_instructions_work_when_token_available() {
        if let Some(_token) = X64V3CryptoToken::summon() {
            unsafe {
                use core::arch::x86_64::*;
                let a = _mm256_set1_epi64x(0x0123456789ABCDEFi64);
                let clmul = _mm256_clmulepi64_epi128::<0x00>(a, a);
                std::hint::black_box(clmul);
                let key = _mm256_setzero_si256();
                let enc = _mm256_aesenc_epi128(a, key);
                std::hint::black_box(enc);
            }
        }
    }

    /// If X64V4Token is available, AVX-512 instructions must work.
    #[cfg(feature = "avx512")]
    #[test]
    fn x64v4_instructions_work_when_token_available() {
        if let Some(_token) = X64V4Token::summon() {
            unsafe {
                use core::arch::x86_64::*;
                let a = _mm512_setzero_ps();
                let b = _mm512_set1_ps(1.0);
                let c = _mm512_add_ps(a, b);
                std::hint::black_box(c);
            }
        }
    }

    /// Verify token hierarchy: v4 implies v3 implies v2.
    #[test]
    fn token_hierarchy_is_correct() {
        if X64V3Token::summon().is_some() {
            assert!(
                X64V2Token::summon().is_some(),
                "v3 implies v2 should be available"
            );
            assert!(
                Avx2FmaToken::summon().is_some(),
                "v3 implies Avx2FmaToken (same type)"
            );
        }

        #[cfg(feature = "avx512")]
        if X64V4Token::summon().is_some() {
            assert!(
                X64V3Token::summon().is_some(),
                "v4 implies v3 should be available"
            );
            assert!(
                X64V2Token::summon().is_some(),
                "v4 implies v2 should be available"
            );
        }

        // Crypto hierarchy
        if X64CryptoToken::summon().is_some() {
            assert!(
                X64V2Token::summon().is_some(),
                "Crypto implies v2 should be available"
            );
        }
        if X64V3CryptoToken::summon().is_some() {
            assert!(
                X64V3Token::summon().is_some(),
                "V3Crypto implies v3 should be available"
            );
            assert!(
                X64CryptoToken::summon().is_some(),
                "V3Crypto implies Crypto should be available"
            );
        }
    }

    #[test]
    fn avx2fma_is_x64v3() {
        let v3 = X64V3Token::summon();
        let avx2fma = Avx2FmaToken::summon();
        assert_eq!(v3.is_some(), avx2fma.is_some());
        assert_eq!(Avx2FmaToken::NAME, X64V3Token::NAME);
    }

    #[test]
    fn print_detected_features() {
        println!("Feature detection results (x86_64):");
        println!("  SSE2:         always (baseline on x86_64)");
        println!("  x86-64-v2:    {}", X64V2Token::summon().is_some());
        println!("  x86 Crypto:   {}", X64CryptoToken::summon().is_some());
        println!(
            "  x86-64-v3:    {} (Desktop64/Avx2FmaToken)",
            X64V3Token::summon().is_some()
        );
        println!(
            "  x86-64-v3 Crypto: {}",
            X64V3CryptoToken::summon().is_some()
        );
        #[cfg(feature = "avx512")]
        {
            println!(
                "  x86-64-v4:    {} (Avx512Token/Server64)",
                X64V4Token::summon().is_some()
            );
            println!("  AVX-512 Modern: {}", X64V4xToken::summon().is_some());
            println!("  AVX-512 FP16:   {}", Avx512Fp16Token::summon().is_some());
        }
    }
}

// ============================================================================
// AArch64 Tests
// ============================================================================
#[cfg(target_arch = "aarch64")]
mod aarch64_tests {
    use super::*;

    /// NEON is baseline on aarch64 — NeonToken::summon() should always succeed.
    #[test]
    fn neon_always_available() {
        assert!(
            NeonToken::summon().is_some(),
            "NEON should always be available on aarch64"
        );
    }

    /// If NeonToken is available, basic NEON operations must work.
    #[test]
    fn neon_instructions_work() {
        if let Some(_token) = NeonToken::summon() {
            unsafe {
                use std::arch::aarch64::*;
                let a = vdupq_n_f32(1.0);
                let b = vdupq_n_f32(2.0);
                let c = vaddq_f32(a, b);
                std::hint::black_box(c);
                let d = vdupq_n_s32(1);
                let e = vdupq_n_s32(2);
                let f = vaddq_s32(d, e);
                std::hint::black_box(f);
            }
        }
    }

    /// If Arm64V2Token is available, RDM and DotProd must work.
    #[test]
    fn arm64v2_instructions_work_when_token_available() {
        if let Some(_token) = Arm64V2Token::summon() {
            unsafe {
                use std::arch::aarch64::*;
                // RDM: saturating rounding doubling multiply accumulate
                let a = vdup_n_s16(100);
                let b = vdup_n_s16(200);
                let acc = vdup_n_s16(0);
                let rdm = vqrdmlah_s16(acc, a, b);
                std::hint::black_box(rdm);

                // DotProd: vdot_s32 is nightly-only (stdarch_neon_dotprod)
                // Verified: compiles on nightly but unstable on stable Rust
            }
        }
    }

    /// If NeonAesToken is available, AES round operations must work.
    #[test]
    fn neon_aes_instructions_work_when_token_available() {
        if let Some(_token) = NeonAesToken::summon() {
            unsafe {
                use std::arch::aarch64::*;
                let state = vdupq_n_u8(0x42);
                let key = vdupq_n_u8(0x2b);
                let enc = vaeseq_u8(state, key);
                std::hint::black_box(enc);
                // p64 multiply
                let result = vmull_p64(0x0123456789ABCDEF, 0xFEDCBA9876543210);
                std::hint::black_box(result);
            }
        }
    }

    /// If NeonSha3Token is available, SHA-3 instructions must work.
    #[test]
    fn neon_sha3_instructions_work_when_token_available() {
        if let Some(_token) = NeonSha3Token::summon() {
            unsafe {
                use std::arch::aarch64::*;
                let a = vdupq_n_u8(0x0F);
                let b = vdupq_n_u8(0x33);
                let c = vdupq_n_u8(0x55);
                let bcax = vbcaxq_u8(a, b, c);
                std::hint::black_box(bcax);
                let eor3 = veor3q_u8(a, b, c);
                std::hint::black_box(eor3);
            }
        }
    }

    /// If NeonCrcToken is available, CRC32 instructions must work.
    #[test]
    fn neon_crc_instructions_work_when_token_available() {
        if let Some(_token) = NeonCrcToken::summon() {
            unsafe {
                use std::arch::aarch64::*;
                let crc = __crc32b(0xFFFFFFFF, 0x42);
                std::hint::black_box(crc);
                let crc_c = __crc32cb(0xFFFFFFFF, 0x42);
                std::hint::black_box(crc_c);
            }
        }
    }

    /// Verify token hierarchy: V3 implies V2 implies Neon.
    #[test]
    fn token_hierarchy_is_correct() {
        if Arm64V3Token::summon().is_some() {
            assert!(Arm64V2Token::summon().is_some(), "Arm64V3 implies Arm64V2");
            assert!(NeonToken::summon().is_some(), "Arm64V3 implies Neon");
            assert!(
                NeonSha3Token::summon().is_some(),
                "Arm64V3 implies NeonSha3"
            );
        }

        if Arm64V2Token::summon().is_some() {
            assert!(NeonToken::summon().is_some(), "Arm64V2 implies Neon");
            assert!(NeonAesToken::summon().is_some(), "Arm64V2 implies NeonAes");
            assert!(NeonCrcToken::summon().is_some(), "Arm64V2 implies NeonCrc");
        }
    }

    #[test]
    fn print_detected_features() {
        println!("Feature detection results (aarch64):");
        println!("  NEON:        {}", NeonToken::summon().is_some());
        println!("  NeonAes:     {}", NeonAesToken::summon().is_some());
        println!("  NeonSha3:    {}", NeonSha3Token::summon().is_some());
        println!("  NeonCrc:     {}", NeonCrcToken::summon().is_some());
        println!("  Arm64-v2:    {}", Arm64V2Token::summon().is_some());
        println!("  Arm64-v3:    {}", Arm64V3Token::summon().is_some());
    }
}

// ============================================================================
// WASM Tests
// ============================================================================
#[cfg(target_arch = "wasm32")]
mod wasm32_tests {
    use super::*;

    /// Wasm128Token should be available when compiled with simd128 target feature.
    #[test]
    fn wasm128_available_with_simd128() {
        // When compiled with RUSTFLAGS="-C target-feature=+simd128",
        // Wasm128Token::summon() should return Some.
        #[cfg(target_feature = "simd128")]
        {
            assert!(
                Wasm128Token::summon().is_some(),
                "Wasm128Token should be available when compiled with +simd128"
            );
        }
        #[cfg(not(target_feature = "simd128"))]
        {
            println!("simd128 not enabled - Wasm128Token not available");
        }
    }

    /// If Wasm128Token is available, basic SIMD128 operations must work.
    #[cfg(target_feature = "simd128")]
    #[test]
    fn wasm128_instructions_work() {
        if let Some(_token) = Wasm128Token::summon() {
            use core::arch::wasm32::*;
            let a = f32x4_splat(1.0);
            let b = f32x4_splat(2.0);
            let c = f32x4_add(a, b);
            std::hint::black_box(c);

            let d = i32x4_splat(1);
            let e = i32x4_splat(2);
            let f = i32x4_add(d, e);
            std::hint::black_box(f);
        }
    }

    #[test]
    fn print_detected_features() {
        println!("Feature detection results (wasm32):");
        println!("  Wasm128:  {}", Wasm128Token::summon().is_some());
    }
}

// ============================================================================
// Cross-Platform Tests (always run)
// ============================================================================

/// ScalarToken should always be available on every platform.
#[test]
fn scalar_token_always_available() {
    assert!(
        ScalarToken::summon().is_some(),
        "ScalarToken should always be available"
    );
}

/// Verify that tokens on unsupported architectures return None.
#[test]
fn wrong_arch_tokens_return_none() {
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        assert!(
            X64V2Token::summon().is_none(),
            "x86 tokens should be None on non-x86"
        );
        assert!(X64V3Token::summon().is_none());
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        assert!(
            NeonToken::summon().is_none(),
            "ARM tokens should be None on non-ARM"
        );
        assert!(Arm64V2Token::summon().is_none());
        assert!(Arm64V3Token::summon().is_none());
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        assert!(
            Wasm128Token::summon().is_none(),
            "WASM tokens should be None on non-WASM"
        );
    }
}

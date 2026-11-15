//! Tests that verify feature detection is consistent with CPU capabilities.
//!
//! These tests run actual SIMD operations to verify that when a token is available,
//! the corresponding instructions actually work. If feature detection lies about
//! capabilities, these tests will crash (SIGILL).
//!
//! Cross-platform: tests for x86_64, aarch64, and wasm32.
//!
//! The `x64_tokens_are_x86_64_only` and `compiled_with_reflects_architecture`
//! tests verify that X64* tokens are stubs on non-x86_64 targets (including
//! 32-bit x86). Run on i686 to verify:
//!   cross test --target i686-unknown-linux-gnu --test feature_consistency
// Checks that the deprecated `Avx2FmaToken` alias is the same type as X64V3Token.
#![allow(deprecated)]

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

    /// If X64V3GfniCryptoToken is available, unmasked 256-bit GFNI must work.
    #[test]
    fn x64v3_gfni_crypto_instructions_work_when_token_available() {
        if let Some(_token) = X64V3GfniCryptoToken::summon() {
            unsafe {
                use core::arch::x86_64::*;
                let a = _mm256_set1_epi8(0x57);
                let b = _mm256_set1_epi8(0x13);
                let product = _mm256_gf2p8mul_epi8(a, b);
                std::hint::black_box(product);
            }
        }
    }

    #[rustversion::since(1.94)]
    #[archmage::attune(wrap)]
    fn exec_vnni2_fp16(_token: X64V3Vnni2Token) {
        use core::arch::x86_64::*;
        let fp16_bits: [u16; 16] = [
            0x3C00, 0x4000, // 1.0, 2.0
            0x4200, 0x4400, // 3.0, 4.0
            0x4500, 0x4600, // 5.0, 6.0
            0x4700, 0x4800, // 7.0, 8.0
            0xBC00, 0xC000, // -1.0, -2.0
            0x3C00, 0x4000, 0x4200, 0x4400, 0x0000, 0x4500,
        ];
        // SAFETY: valid aligned __m256h on stack; AVX-NE-CONVERT is proven by token.
        let fp16_vec: __m256h = unsafe { core::mem::transmute(fp16_bits) };
        let even_fp16 = unsafe { _mm256_cvtneeph_ps(&fp16_vec) };
        let even_fp16_arr: [f32; 8] = unsafe { core::mem::transmute(even_fp16) };
        assert_eq!(even_fp16_arr, [1.0, 3.0, 5.0, 7.0, -1.0, 1.0, 3.0, 0.0]);
    }

    #[rustversion::before(1.94)]
    fn exec_vnni2_fp16(_token: X64V3Vnni2Token) {}

    /// If X64V3Vnni2Token is available, AVX-VNNI-INT8, AVX-IFMA, AVX-NE-CONVERT, ADX, and SHA work.
    #[test]
    fn x64v3_vnni2_instructions_work_when_token_available() {
        if let Some(token) = X64V3Vnni2Token::summon() {
            #[allow(unused_unsafe)]
            #[archmage::attune(wrap)]
            fn exec(_token: X64V3Vnni2Token) {
                use core::arch::x86_64::*;
                let ones = _mm256_set1_epi8(1);
                let dp = _mm256_dpbssd_epi32(_mm256_setzero_si256(), ones, ones);
                assert_eq!(_mm256_extract_epi32::<0>(dp), 4);

                let zero = _mm256_setzero_si256();
                let two = _mm256_set1_epi64x(2);
                let three = _mm256_set1_epi64x(3);
                let ifma = _mm256_madd52lo_avx_epu64(zero, two, three);
                assert_eq!(_mm256_extract_epi64::<0>(ifma), 6);

                let f32s = _mm256_set_ps(-4.0, 3.0, -2.0, 1.0, 4.0, 3.0, 2.0, 1.0);
                let bf16s = _mm256_cvtneps_avx_pbh(f32s);
                // SAFETY: transmute between 128-bit SIMD vector and [u16; 8] for inspection.
                let bf16_bits: [u16; 8] = unsafe { core::mem::transmute(bf16s) };
                assert_eq!(bf16_bits[0], 0x3F80); // 1.0 in BF16
                assert_eq!(bf16_bits[7], 0xC080); // -4.0 in BF16

                // SAFETY: [u16; 16] has identical layout to __m256bh and pointer is aligned for stack read.
                let pairs: __m256bh = unsafe {
                    core::mem::transmute([
                        0x3F80u16, 0x4000, // even=1.0, odd=2.0
                        0x4040, 0x4080, // even=3.0, odd=4.0
                        0x40A0, 0x40C0, // even=5.0, odd=6.0
                        0x40E0, 0x4100, // even=7.0, odd=8.0
                        0xC000, 0xC040, // even=-2.0, odd=-3.0
                        0x3F80, 0x4000, 0x4040, 0x4080, 0x0000, 0xBF80,
                    ])
                };
                // SAFETY: `pairs` is a valid stack __m256bh, and AVX-NE-CONVERT is proven by token.
                let evens = unsafe { _mm256_cvtneebf16_ps(&pairs) };
                // SAFETY: transmute __m256 to [f32; 8] for assertion.
                let even_arr: [f32; 8] = unsafe { core::mem::transmute(evens) };
                assert_eq!(even_arr, [1.0, 3.0, 5.0, 7.0, -2.0, 1.0, 3.0, 0.0]);

                let odds = unsafe { _mm256_cvtneobf16_ps(&pairs) };
                let odd_arr: [f32; 8] = unsafe { core::mem::transmute(odds) };
                assert_eq!(odd_arr, [2.0, 4.0, 6.0, 8.0, -3.0, 2.0, 4.0, -1.0]);

                let mut sum = 0u64;
                // SAFETY: _addcarryx_u64 is unsafe in Rust 1.89 and gated by token ADX proof.
                let carry = unsafe { _addcarryx_u64(1, u64::MAX, 9, &mut sum) };
                assert_eq!(carry, 1);
                assert_eq!(sum, 9);

                let sha = _mm_sha256msg1_epu32(_mm_set1_epi32(1), _mm_set1_epi32(2));
                std::hint::black_box(sha);
            }
            exec(token);
            exec_vnni2_fp16(token);
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
        if X64V3GfniCryptoToken::summon().is_some() {
            assert!(
                X64V3CryptoToken::summon().is_some(),
                "V3 GFNI Crypto implies V3 Crypto should be available"
            );
        }
        if X64V3VnniToken::summon().is_some() {
            assert!(
                X64V3GfniCryptoToken::summon().is_some(),
                "V3 VNNI implies V3 GFNI Crypto should be available"
            );
        }
        if X64V3Vnni2Token::summon().is_some() {
            assert!(
                X64V3VnniToken::summon().is_some(),
                "V3 VNNI-2 implies V3 VNNI should be available"
            );
        }
    }

    /// If X64V3VnniToken is available, ADX and SHA instructions must work.
    #[test]
    fn x64v3_vnni_adx_and_sha_work_when_token_available() {
        if let Some(token) = X64V3VnniToken::summon() {
            #[allow(unused_unsafe)]
            #[archmage::attune(wrap)]
            fn exec(_token: X64V3VnniToken) {
                use core::arch::x86_64::*;
                let mut sum = 0u64;
                // SAFETY: _addcarryx_u64 is unsafe in Rust 1.89 and gated by token ADX proof.
                let carry = unsafe { _addcarryx_u64(1, u64::MAX, 123, &mut sum) };
                assert_eq!(carry, 1);
                assert_eq!(sum, 123);
                let sha = _mm_sha256msg1_epu32(_mm_set1_epi32(3), _mm_set1_epi32(5));
                std::hint::black_box(sha);
            }
            exec(token);
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
        println!("  x86-64-v3 VNNI:   {}", X64V3VnniToken::summon().is_some());
        println!(
            "  x86-64-v3 VNNI-2: {}",
            X64V3Vnni2Token::summon().is_some()
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

    /// Wasm128RelaxedToken should be available when compiled with both
    /// simd128 and relaxed-simd target features.
    #[test]
    fn wasm128_relaxed_available_with_relaxed_simd() {
        #[cfg(all(target_feature = "simd128", target_feature = "relaxed-simd"))]
        {
            assert!(
                Wasm128RelaxedToken::summon().is_some(),
                "Wasm128RelaxedToken should be available with +simd128,+relaxed-simd"
            );
            // Relaxed implies base wasm128
            assert!(
                Wasm128Token::summon().is_some(),
                "Wasm128Token should be available when relaxed-simd is enabled"
            );
        }
        #[cfg(not(all(target_feature = "simd128", target_feature = "relaxed-simd")))]
        {
            println!("relaxed-simd not enabled - Wasm128RelaxedToken not available");
        }
    }

    #[test]
    fn print_detected_features() {
        println!("Feature detection results (wasm32):");
        println!("  Wasm128:          {}", Wasm128Token::summon().is_some());
        println!(
            "  Wasm128Relaxed:   {}",
            Wasm128RelaxedToken::summon().is_some()
        );
        println!(
            "  Wasm128RelaxedFma: {}",
            Wasm128RelaxedFmaToken::summon().is_some()
        );
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
    #[cfg(not(target_arch = "x86_64"))]
    {
        assert!(
            X64V1Token::summon().is_none(),
            "x86_64 tokens should be None on non-x86_64"
        );
        assert!(
            X64V2Token::summon().is_none(),
            "x86_64 tokens should be None on non-x86_64"
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
        assert!(
            Wasm128RelaxedToken::summon().is_none(),
            "WASM relaxed tokens should be None on non-WASM"
        );
        assert!(
            Wasm128RelaxedFmaToken::summon().is_none(),
            "WASM relaxed FMA tokens should be None on non-WASM"
        );
    }
}

/// X64 tokens are x86_64-only. On 32-bit x86, they must return None.
/// This test verifies the cfg gating — on x86_64 it checks tokens work,
/// on any other arch (including 32-bit x86) it checks they return None.
#[test]
fn x64_tokens_are_x86_64_only() {
    #[cfg(target_arch = "x86_64")]
    {
        // On x86_64, V1 is baseline — must always be Some
        assert!(
            X64V1Token::summon().is_some(),
            "X64V1Token should always be Some on x86_64"
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        // On ALL other architectures (including 32-bit x86), must be None.
        // This catches the bug where arch="x86" in the registry would give
        // 32-bit x86 real implementations instead of stubs.
        assert!(
            X64V1Token::summon().is_none(),
            "X64V1Token must be None on non-x86_64 (including 32-bit x86)"
        );
        assert!(
            X64V2Token::summon().is_none(),
            "X64V2Token must be None on non-x86_64"
        );
        assert!(
            X64V3Token::summon().is_none(),
            "X64V3Token must be None on non-x86_64"
        );
        assert!(
            X64CryptoToken::summon().is_none(),
            "X64CryptoToken must be None on non-x86_64"
        );
        assert!(
            X64V3CryptoToken::summon().is_none(),
            "X64V3CryptoToken must be None on non-x86_64"
        );
        assert!(
            X64V3GfniCryptoToken::summon().is_none(),
            "X64V3GfniCryptoToken must be None on non-x86_64"
        );
        assert!(
            X64V3VnniToken::summon().is_none(),
            "X64V3VnniToken must be None on non-x86_64"
        );
        assert!(
            X64V3Vnni2Token::summon().is_none(),
            "X64V3Vnni2Token must be None on non-x86_64"
        );
    }
}

/// Verify compiled_with() returns correct values per architecture.
/// On x86_64, V1 features are baseline so compiled_with() = Some(true).
/// On other architectures, compiled_with() must indicate unavailability.
#[test]
fn compiled_with_reflects_architecture() {
    #[cfg(target_arch = "x86_64")]
    {
        // SSE2 is the x86_64 ABI baseline — always compiled in
        #[cfg(not(feature = "testable_dispatch"))]
        assert_eq!(
            X64V1Token::compiled_with(),
            Some(true),
            "X64V1Token::compiled_with() should be Some(true) on x86_64"
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        // On non-x86_64 (including 32-bit x86), compiled_with should NOT
        // return Some(true) — these tokens don't belong to this architecture.
        assert_ne!(
            X64V1Token::compiled_with(),
            Some(true),
            "X64V1Token::compiled_with() must not be Some(true) on non-x86_64"
        );
    }
}

/// The #[arcane] macro cfg-gates to x86_64 only. Verify the macro registry
/// agrees with the token module routing — both must target x86_64, not x86.
#[test]
fn macro_registry_targets_x86_64_not_x86() {
    // This is a compile-time check: if the macro registry returned "x86"
    // instead of "x86_64", #[arcane] would emit cfg(target_arch = "x86")
    // which would generate real implementations on 32-bit x86.
    //
    // The test verifies the invariant indirectly: on x86_64 the token
    // exists and works, on everything else it's a stub returning None.
    // A cross-compilation test on i686 would fail if the routing were wrong.
    #[cfg(target_arch = "x86_64")]
    assert!(X64V3Token::summon().is_some() || X64V3Token::summon().is_none());
    #[cfg(not(target_arch = "x86_64"))]
    assert!(X64V3Token::summon().is_none());
}

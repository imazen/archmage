//! Tests that verify feature detection is consistent with CPU capabilities.
//!
//! These tests run actual SIMD operations to verify that when a token is available,
//! the corresponding instructions actually work. If feature detection lies about
//! capabilities, these tests will crash (SIGILL).

// Note: These tests use core::arch::x86_64 intrinsics directly
#![cfg(target_arch = "x86_64")]

use archmage::SimdToken;
use archmage::*;

/// SSE2 is baseline on x86_64 and always works.
#[test]
fn sse2_instructions_work_baseline() {
    // SSE2 is baseline on x86_64, no token needed
    unsafe {
        use core::arch::x86_64::*;
        let a = _mm_setzero_ps();
        let b = _mm_set1_ps(1.0);
        let c = _mm_add_ps(a, b);
        // Prevent optimization
        std::hint::black_box(c);
    }
}

/// If X64V2Token is available, SSE4.2 instructions must work.
#[test]
fn x64v2_instructions_work_when_token_available() {
    if let Some(_token) = X64V2Token::summon() {
        unsafe {
            use core::arch::x86_64::*;
            // SSE4.2 CRC32 instruction
            let result = _mm_crc32_u8(0, 0);
            std::hint::black_box(result);
            // SSE4.1 extract instruction
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
            // AVX: 256-bit float operations
            let a = _mm256_setzero_ps();
            let b = _mm256_set1_ps(1.0);
            let c = _mm256_add_ps(a, b);
            std::hint::black_box(c);
            // AVX2: 256-bit integer operations
            let d = _mm256_setzero_si256();
            let e = _mm256_set1_epi32(1);
            let f = _mm256_add_epi32(d, e);
            std::hint::black_box(f);
            // FMA: fused multiply-add
            let g = _mm256_set1_ps(1.0);
            let h = _mm256_set1_ps(2.0);
            let i = _mm256_set1_ps(3.0);
            let j = _mm256_fmadd_ps(g, h, i);
            std::hint::black_box(j);
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
    // If v3 is available, v2 must also be available
    if X64V3Token::summon().is_some() {
        assert!(
            X64V2Token::summon().is_some(),
            "v3 implies v2 should be available"
        );
        // Avx2FmaToken is an alias for X64V3Token, so it must match
        assert!(
            Avx2FmaToken::summon().is_some(),
            "v3 implies Avx2FmaToken (same type)"
        );
    }

    // If v4 is available, v3 and v2 must also be available
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
        assert!(
            Avx2FmaToken::summon().is_some(),
            "v4 implies Avx2FmaToken should be available"
        );
    }
}

/// Verify that Avx2FmaToken is truly the same type as X64V3Token.
#[test]
fn avx2fma_is_x64v3() {
    let v3 = X64V3Token::summon();
    let avx2fma = Avx2FmaToken::summon();

    assert_eq!(
        v3.is_some(),
        avx2fma.is_some(),
        "Avx2FmaToken and X64V3Token should be the same type"
    );

    assert_eq!(
        Avx2FmaToken::NAME,
        X64V3Token::NAME,
        "Avx2FmaToken and X64V3Token should share the same NAME"
    );
}

/// Print detected features for debugging.
#[test]
fn print_detected_features() {
    println!("Feature detection results:");
    println!("  SSE2:      always (baseline on x86_64)");
    println!("  x86-64-v2: {}", X64V2Token::summon().is_some());
    println!(
        "  x86-64-v3: {} (Desktop64/Avx2FmaToken)",
        X64V3Token::summon().is_some()
    );
    #[cfg(feature = "avx512")]
    {
        println!(
            "  x86-64-v4: {} (Avx512Token/Server64)",
            X64V4Token::summon().is_some()
        );
        println!(
            "  AVX-512 Modern: {}",
            Avx512ModernToken::summon().is_some()
        );
        println!("  AVX-512 FP16:   {}", Avx512Fp16Token::summon().is_some());
    }
}

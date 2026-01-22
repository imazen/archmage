//! Tests that verify feature detection is consistent with CPU capabilities.
//!
//! These tests run actual SIMD operations to verify that when a token is available,
//! the corresponding instructions actually work. If feature detection lies about
//! capabilities, these tests will crash (SIGILL).

// Note: These tests use core::arch::x86_64 intrinsics directly
#![cfg(target_arch = "x86_64")]

use archmage::SimdToken;
use archmage::tokens::x86::*;

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

/// If AVX token is available, AVX instructions must work.
#[test]
fn avx_instructions_work_when_token_available() {
    if let Some(_token) = AvxToken::try_new() {
        unsafe {
            use core::arch::x86_64::*;
            let a = _mm256_setzero_ps();
            let b = _mm256_set1_ps(1.0);
            let c = _mm256_add_ps(a, b);
            std::hint::black_box(c);
        }
    }
}

/// If AVX2 token is available, AVX2 instructions must work.
#[test]
fn avx2_instructions_work_when_token_available() {
    if let Some(_token) = Avx2Token::try_new() {
        unsafe {
            use core::arch::x86_64::*;
            let a = _mm256_setzero_si256();
            let b = _mm256_set1_epi32(1);
            let c = _mm256_add_epi32(a, b); // AVX2 integer instruction
            std::hint::black_box(c);
        }
    }
}

/// If FMA token is available, FMA instructions must work.
#[test]
fn fma_instructions_work_when_token_available() {
    if let Some(_token) = FmaToken::try_new() {
        unsafe {
            use core::arch::x86_64::*;
            let a = _mm256_set1_ps(1.0);
            let b = _mm256_set1_ps(2.0);
            let c = _mm256_set1_ps(3.0);
            let d = _mm256_fmadd_ps(a, b, c); // FMA instruction
            std::hint::black_box(d);
        }
    }
}

/// If AVX-512F token is available, AVX-512 instructions must work.
#[cfg(feature = "avx512")]
#[test]
fn avx512f_instructions_work_when_token_available() {
    if let Some(_token) = Avx512fToken::try_new() {
        unsafe {
            use core::arch::x86_64::*;
            let a = _mm512_setzero_ps();
            let b = _mm512_set1_ps(1.0);
            let c = _mm512_add_ps(a, b);
            std::hint::black_box(c);
        }
    }
}

/// Verify token hierarchy: AVX2 implies AVX implies SSE4.1 implies SSE2.
#[test]
fn token_hierarchy_is_correct() {
    // If AVX2 is available, all predecessors must also be available
    if Avx2Token::try_new().is_some() {
        assert!(AvxToken::try_new().is_some(), "AVX2 implies AVX");
        assert!(Sse42Token::try_new().is_some(), "AVX2 implies SSE4.2");
        assert!(Sse41Token::try_new().is_some(), "AVX2 implies SSE4.1");
        // Note: SSE2 is baseline on x86_64, always available
    }

    // If AVX-512 (x64-v4) is available, AVX2 and FMA must also be available
    #[cfg(feature = "avx512")]
    if X64V4Token::try_new().is_some() {
        assert!(Avx2Token::try_new().is_some(), "x64-v4 implies AVX2");
        assert!(Avx2FmaToken::try_new().is_some(), "x64-v4 implies AVX2+FMA");
    }

    // Profile token hierarchy
    #[cfg(feature = "avx512")]
    if X64V4Token::try_new().is_some() {
        assert!(X64V3Token::try_new().is_some(), "v4 implies v3");
        assert!(X64V2Token::try_new().is_some(), "v4 implies v2");
    }

    if X64V3Token::try_new().is_some() {
        assert!(X64V2Token::try_new().is_some(), "v3 implies v2");
        assert!(Avx2Token::try_new().is_some(), "v3 implies AVX2");
        assert!(FmaToken::try_new().is_some(), "v3 implies FMA");
    }
}

/// Verify that combined tokens match their components.
#[test]
fn combined_tokens_match_components() {
    // Avx2FmaToken should be available iff both AVX2 and FMA are available
    let avx2 = Avx2Token::try_new().is_some();
    let fma = FmaToken::try_new().is_some();
    let avx2_fma = Avx2FmaToken::try_new().is_some();

    assert_eq!(
        avx2_fma,
        avx2 && fma,
        "Avx2FmaToken should be available iff both AVX2 and FMA are"
    );
}

/// Print detected features for debugging.
#[test]
fn print_detected_features() {
    println!("Feature detection results:");
    println!("  SSE2:      always (baseline on x86_64)");
    println!("  SSE4.1:    {}", Sse41Token::try_new().is_some());
    println!("  SSE4.2:    {}", Sse42Token::try_new().is_some());
    println!("  AVX:       {}", AvxToken::try_new().is_some());
    println!("  AVX2:      {}", Avx2Token::try_new().is_some());
    println!("  AVX2+FMA:  {}", Avx2FmaToken::try_new().is_some());
    println!("  x86-64-v2: {}", X64V2Token::try_new().is_some());
    println!("  x86-64-v3: {}", X64V3Token::try_new().is_some());
    #[cfg(feature = "avx512")]
    println!("  x86-64-v4: {}", X64V4Token::try_new().is_some());
}

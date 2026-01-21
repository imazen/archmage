//! Tests that can run under Miri (no SIMD intrinsics).
//!
//! These tests verify the token system's logic without executing
//! actual SIMD instructions.

// Only run these tests on x86_64 where the tokens are available
#![cfg(target_arch = "x86_64")]

use archmage::SimdToken;
use archmage::tokens::x86::*;

/// Verify all tokens are zero-sized types.
#[test]
fn tokens_are_zst() {
    assert_eq!(core::mem::size_of::<Sse2Token>(), 0);
    assert_eq!(core::mem::size_of::<Avx2Token>(), 0);
    assert_eq!(core::mem::size_of::<FmaToken>(), 0);
    assert_eq!(core::mem::size_of::<Avx2FmaToken>(), 0);
    assert_eq!(core::mem::size_of::<X64V2Token>(), 0);
    assert_eq!(core::mem::size_of::<X64V3Token>(), 0);
    assert_eq!(core::mem::size_of::<X64V4Token>(), 0);
}

/// Verify tokens implement Copy (important for ergonomics).
#[test]
fn tokens_are_copy() {
    fn assert_copy<T: Copy>() {}
    assert_copy::<Sse2Token>();
    assert_copy::<Avx2Token>();
    assert_copy::<FmaToken>();
    assert_copy::<Avx2FmaToken>();
    assert_copy::<X64V2Token>();
    assert_copy::<X64V3Token>();
    assert_copy::<X64V4Token>();
}

/// Verify SSE2 is always available on x86_64 (baseline).
#[test]
fn sse2_always_available() {
    // SSE2 is baseline on x86_64, must always succeed
    assert!(Sse2Token::try_new().is_some());
}

/// Test that token names are set correctly.
#[test]
fn token_names() {
    assert_eq!(Sse2Token::NAME, "SSE2");
    assert_eq!(Avx2Token::NAME, "AVX2");
    assert_eq!(FmaToken::NAME, "FMA");
    assert_eq!(Avx2FmaToken::NAME, "AVX2+FMA");
    assert_eq!(X64V2Token::NAME, "x86-64-v2");
    assert_eq!(X64V3Token::NAME, "x86-64-v3");
    assert_eq!(X64V4Token::NAME, "AVX-512"); // X64V4Token is alias for Avx512Token
}

/// Test token hierarchy - if we have AVX2, we can get SSE2.
#[test]
fn token_hierarchy_avx2_implies_sse2() {
    if let Some(avx2) = Avx2Token::try_new() {
        let sse2: Sse2Token = avx2.sse2();
        // Can't really verify the token is valid without running SIMD,
        // but we can verify it's the right type and ZST
        assert_eq!(core::mem::size_of_val(&sse2), 0);
    }
}

/// Test combined token hierarchy.
#[test]
fn token_hierarchy_avx2fma() {
    if let Some(token) = Avx2FmaToken::try_new() {
        let _avx2: Avx2Token = token.avx2();
        let _fma: FmaToken = token.fma();
        let _avx: AvxToken = token.avx();
        let _sse41: Sse41Token = token.sse41();
        let _sse2: Sse2Token = token.sse2();
    }
}

/// Test profile token hierarchy.
#[test]
fn token_hierarchy_x64v3() {
    if let Some(token) = X64V3Token::try_new() {
        let _avx2: Avx2Token = token.avx2();
        let _fma: FmaToken = token.fma();
        let _avx2_fma: Avx2FmaToken = token.avx2_fma();
    }
}

/// Test that try_new is consistent (calling twice gives same result).
#[test]
fn try_new_is_consistent() {
    let result1 = Avx2Token::try_new().is_some();
    let result2 = Avx2Token::try_new().is_some();
    assert_eq!(result1, result2);

    let result1 = X64V3Token::try_new().is_some();
    let result2 = X64V3Token::try_new().is_some();
    assert_eq!(result1, result2);
}

/// Test marker trait bounds compile correctly.
#[test]
fn marker_traits_compile() {
    use archmage::{Has128BitSimd, Has256BitSimd, HasFma};

    fn requires_128<T: Has128BitSimd>(_: T) {}
    fn requires_256<T: Has256BitSimd>(_: T) {}
    fn requires_fma<T: HasFma>(_: T) {}

    if let Some(token) = Sse2Token::try_new() {
        requires_128(token);
    }
    if let Some(token) = Avx2Token::try_new() {
        requires_128(token);
        requires_256(token);
    }
    if let Some(token) = Avx2FmaToken::try_new() {
        requires_128(token);
        requires_256(token);
        requires_fma(token);
    }
    if let Some(token) = X64V3Token::try_new() {
        requires_128(token);
        requires_256(token);
        requires_fma(token);
    }
}

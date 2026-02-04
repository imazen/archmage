//! Tests that can run under Miri (no SIMD intrinsics).
//!
//! These tests verify the token system's logic without executing
//! actual SIMD instructions.

// Only run these tests on x86_64 where the tokens are available
#![cfg(target_arch = "x86_64")]

use archmage::SimdToken;
use archmage::*;

/// Verify all tokens are zero-sized types.
#[test]
fn tokens_are_zst() {
    assert_eq!(core::mem::size_of::<X64V2Token>(), 0);
    assert_eq!(core::mem::size_of::<X64V3Token>(), 0);
    assert_eq!(core::mem::size_of::<Avx2FmaToken>(), 0); // type alias for X64V3Token
    #[cfg(feature = "avx512")]
    assert_eq!(core::mem::size_of::<X64V4Token>(), 0);
}

/// Verify tokens implement Copy (important for ergonomics).
#[test]
fn tokens_are_copy() {
    fn assert_copy<T: Copy>() {}
    assert_copy::<X64V2Token>();
    assert_copy::<X64V3Token>();
    assert_copy::<Avx2FmaToken>();
    #[cfg(feature = "avx512")]
    assert_copy::<X64V4Token>();
}

/// Test that token names are set correctly.
#[test]
fn token_names() {
    assert_eq!(X64V2Token::NAME, "x86-64-v2");
    assert_eq!(X64V3Token::NAME, "x86-64-v3");
    // Avx2FmaToken is a type alias for X64V3Token
    assert_eq!(Avx2FmaToken::NAME, "x86-64-v3");
    #[cfg(feature = "avx512")]
    assert_eq!(X64V4Token::NAME, "AVX-512"); // X64V4Token is alias for Avx512Token
}

/// Test token hierarchy: X64V3Token can produce X64V2Token.
#[test]
fn token_hierarchy_x64v3() {
    if let Some(token) = X64V3Token::summon() {
        let _v2: X64V2Token = token.v2();
    }
}

/// Test that summon is consistent (calling twice gives same result).
#[test]
fn summon_is_consistent() {
    let result1 = X64V3Token::summon().is_some();
    let result2 = X64V3Token::summon().is_some();
    assert_eq!(result1, result2);

    let result1 = X64V2Token::summon().is_some();
    let result2 = X64V2Token::summon().is_some();
    assert_eq!(result1, result2);
}

/// Test marker trait bounds compile correctly.
#[test]
fn marker_traits_compile() {
    use archmage::{Has128BitSimd, Has256BitSimd};

    fn requires_128<T: Has128BitSimd>(_: T) {}
    fn requires_256<T: Has256BitSimd>(_: T) {}

    if let Some(token) = X64V2Token::summon() {
        requires_128(token);
    }
    if let Some(token) = Avx2FmaToken::summon() {
        requires_128(token);
        requires_256(token);
    }
    if let Some(token) = X64V3Token::summon() {
        requires_128(token);
        requires_256(token);
    }
}

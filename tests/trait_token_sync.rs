//! Sync test: Ensures archmage-macros recognizes all traits and tokens.
//!
//! IMPORTANT: When adding new traits or tokens to archmage, you MUST:
//! 1. Add the trait/token to src/tokens/
//! 2. Update archmage-macros/src/lib.rs trait_to_features() and/or token_to_features()
//! 3. Add a test function here to verify the macro recognizes it
//!
//! If this file doesn't compile, the macro mappings are out of sync.
//! The compiler error will tell you which trait/token is unrecognized.

#![cfg(target_arch = "x86_64")]

use archmage::arcane;
use archmage::tokens::x86::*;
use archmage::SimdToken;
use archmage::{
    Has128BitSimd, Has256BitSimd, Has512BitSimd, HasAvx, HasAvx2, HasAvx2Fma, HasAvx512,
    HasDesktop64, HasModernAvx512, HasServer64, HasSse42, HasX64V3,
};

// =============================================================================
// Trait bound tests - verify trait_to_features() in the macro
// =============================================================================

#[arcane]
fn verify_has_sse42(_token: impl HasSse42) {}

#[arcane]
fn verify_has_avx(_token: impl HasAvx) {}

#[arcane]
fn verify_has_avx2(_token: impl HasAvx2) {}

#[arcane]
fn verify_has_avx2_fma(_token: impl HasAvx2Fma) {}

#[arcane]
fn verify_has_x64v3(_token: impl HasX64V3) {}

#[arcane]
fn verify_has_desktop64(_token: impl HasDesktop64) {}

#[arcane]
fn verify_has_avx512(_token: impl HasAvx512) {}

#[arcane]
fn verify_has_server64(_token: impl HasServer64) {}

#[arcane]
fn verify_has_modern_avx512(_token: impl HasModernAvx512) {}

#[arcane]
fn verify_has_128bit_simd(_token: impl Has128BitSimd) {}

#[arcane]
fn verify_has_256bit_simd(_token: impl Has256BitSimd) {}

#[arcane]
fn verify_has_512bit_simd(_token: impl Has512BitSimd) {}

// =============================================================================
// Token type tests - verify token_to_features() in the macro
// =============================================================================

#[arcane]
fn verify_sse42_token(_token: Sse42Token) {}

#[arcane]
fn verify_avx_token(_token: AvxToken) {}

#[arcane]
fn verify_avx2_token(_token: Avx2Token) {}

#[arcane]
fn verify_avx2_fma_token(_token: Avx2FmaToken) {}

#[arcane]
fn verify_x64v3_token(_token: X64V3Token) {}

#[arcane]
fn verify_avx512_token(_token: Avx512Token) {}

#[arcane]
fn verify_x64v4_token(_token: X64V4Token) {}

#[arcane]
fn verify_avx512_modern_token(_token: Avx512ModernToken) {}

#[arcane]
fn verify_avx512_fp16_token(_token: Avx512Fp16Token) {}

// =============================================================================
// Alias tests - verify aliases work with the macro
// =============================================================================

#[arcane]
fn verify_desktop64_alias(_token: Desktop64) {}

#[arcane]
fn verify_server64_alias(_token: Server64) {}

// =============================================================================
// Runtime test to ensure this file actually gets compiled
// =============================================================================

#[test]
fn trait_token_sync_compiles() {
    // This test exists to ensure the file is compiled.
    // The actual verification happens at compile time - if any trait/token
    // is not recognized by the macro, this file won't compile.
    //
    // If you're adding a new trait or token:
    // 1. Add a verify_xxx function above using the #[arcane] macro
    // 2. If it doesn't compile, update archmage-macros/src/lib.rs:
    //    - trait_to_features() for traits
    //    - token_to_features() for tokens
}

#[test]
fn can_call_verified_functions() {
    // Actually call the functions at runtime to catch any issues
    // with the generated code (not just macro recognition)

    if let Some(token) = Sse42Token::try_new() {
        verify_sse42_token(token);
        verify_has_sse42(token);
        verify_has_128bit_simd(token);
    }

    if let Some(token) = AvxToken::try_new() {
        verify_avx_token(token);
        verify_has_avx(token);
        verify_has_256bit_simd(token);
    }

    if let Some(token) = Avx2Token::try_new() {
        verify_avx2_token(token);
        verify_has_avx2(token);
    }

    if let Some(token) = Avx2FmaToken::try_new() {
        verify_avx2_fma_token(token);
        verify_has_avx2_fma(token);
    }

    if let Some(token) = X64V3Token::try_new() {
        verify_x64v3_token(token);
        verify_has_x64v3(token);
        verify_desktop64_alias(token);
        verify_has_desktop64(token);
    }

    if let Some(token) = Avx512Token::try_new() {
        verify_avx512_token(token);
        verify_has_avx512(token);
        verify_has_512bit_simd(token);
    }

    if let Some(token) = X64V4Token::try_new() {
        verify_x64v4_token(token);
        verify_server64_alias(token);
        verify_has_server64(token);
    }

    if let Some(token) = Avx512ModernToken::try_new() {
        verify_avx512_modern_token(token);
        verify_has_modern_avx512(token);
    }

    if let Some(token) = Avx512Fp16Token::try_new() {
        verify_avx512_fp16_token(token);
    }
}

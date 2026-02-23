//! Comprehensive crypto intrinsic exercise tests for X64CryptoToken and X64V3CryptoToken.
//!
//! X64CryptoToken (SSE4.2 + PCLMULQDQ + AES-NI):
//!   - Carryless multiplication (PCLMULQDQ)
//!   - AES encrypt/decrypt rounds (AES-NI)
//!   - AES inverse mix columns / key generation assist
//!
//! X64V3CryptoToken (AVX2 + FMA + VPCLMULQDQ + VAES):
//!   - 256-bit carryless multiplication (VPCLMULQDQ)
//!   - 256-bit AES encrypt/decrypt rounds (VAES)

#![cfg(target_arch = "x86_64")]
#![allow(unused_imports, unused_variables, dead_code)]
#![allow(clippy::eq_op)]

use archmage::{SimdToken, X64CryptoToken, X64V3CryptoToken, arcane};
use core::arch::x86_64::*;
use core::hint::black_box;

// =============================================================================
// X64CryptoToken - PCLMULQDQ + AES-NI (128-bit)
// =============================================================================

/// Run all X64CryptoToken intrinsic tests.
#[test]
fn test_x64_crypto_intrinsics() {
    if let Some(token) = X64CryptoToken::summon() {
        exercise_pclmulqdq(token);
        exercise_aes_enc_dec(token);
        exercise_aes_key_assist(token);
        println!("All X64CryptoToken intrinsic tests passed!");
    } else {
        println!("X64CryptoToken not available - skipping tests");
    }
}

#[arcane]
fn exercise_pclmulqdq(token: X64CryptoToken) {
    // Carryless multiply: _mm_clmulepi64_si128
    // Multiplies two 64-bit halves of 128-bit inputs without carry propagation.
    // Used in CRC32, GCM, and polynomial arithmetic.
    let a = _mm_set_epi64x(0x0123456789ABCDEFi64, 0xFEDCBA9876543210u64 as i64);
    let b = _mm_set_epi64x(0x1111111111111111i64, 0x2222222222222222i64);

    // IMM8 selects which 64-bit halves to multiply:
    //   0x00: a[63:0]   * b[63:0]
    //   0x01: a[127:64] * b[63:0]
    //   0x10: a[63:0]   * b[127:64]
    //   0x11: a[127:64] * b[127:64]
    let r00 = _mm_clmulepi64_si128::<0x00>(a, b);
    let r01 = _mm_clmulepi64_si128::<0x01>(a, b);
    let r10 = _mm_clmulepi64_si128::<0x10>(a, b);
    let r11 = _mm_clmulepi64_si128::<0x11>(a, b);

    black_box(r00);
    black_box(r01);
    black_box(r10);
    black_box(r11);

    // Self-multiply should be deterministic
    let self_mul = _mm_clmulepi64_si128::<0x00>(a, a);
    let self_mul2 = _mm_clmulepi64_si128::<0x00>(a, a);
    let lo1 = _mm_extract_epi64::<0>(self_mul);
    let lo2 = _mm_extract_epi64::<0>(self_mul2);
    assert_eq!(lo1, lo2, "PCLMULQDQ should be deterministic");
}

#[arcane]
fn exercise_aes_enc_dec(token: X64CryptoToken) {
    // AES round operations operate on 128-bit state blocks.
    // Use 64-bit setters to avoid i8 literal range issues.
    let state = _mm_set_epi64x(0x3243f6a8885a308di64, 0x313198a2e0370734i64);
    let round_key = _mm_set_epi64x(0x2b7e151628aed2a6u64 as i64, 0xabf7158809cf4f3cu64 as i64);

    // Encrypt: SubBytes + ShiftRows + MixColumns + AddRoundKey
    let enc = _mm_aesenc_si128(state, round_key);
    black_box(enc);

    // Encrypt last round: SubBytes + ShiftRows + AddRoundKey (no MixColumns)
    let enc_last = _mm_aesenclast_si128(state, round_key);
    black_box(enc_last);

    // Decrypt: InvSubBytes + InvShiftRows + InvMixColumns + AddRoundKey
    let dec = _mm_aesdec_si128(enc, round_key);
    black_box(dec);

    // Decrypt last round: InvSubBytes + InvShiftRows + AddRoundKey (no InvMixColumns)
    let dec_last = _mm_aesdeclast_si128(enc, round_key);
    black_box(dec_last);

    // Enc and dec should produce different results from input
    let state_lo = _mm_extract_epi64::<0>(state);
    let enc_lo = _mm_extract_epi64::<0>(enc);
    assert_ne!(state_lo, enc_lo, "AES encrypt should transform the state");
}

#[arcane]
fn exercise_aes_key_assist(token: X64CryptoToken) {
    let key = _mm_set_epi64x(0x2b7e151628aed2a6u64 as i64, 0xabf7158809cf4f3cu64 as i64);

    // Inverse mix columns (used in key schedule for decryption)
    let imc = _mm_aesimc_si128(key);
    black_box(imc);

    // Key generation assist with round constant
    let assist = _mm_aeskeygenassist_si128::<0x01>(key);
    black_box(assist);
    let assist2 = _mm_aeskeygenassist_si128::<0x02>(key);
    black_box(assist2);

    // Different RCON values should produce different results
    let a1_lo = _mm_extract_epi64::<0>(assist);
    let a2_lo = _mm_extract_epi64::<0>(assist2);
    assert_ne!(
        a1_lo, a2_lo,
        "Different RCON values should produce different key assists"
    );
}

// =============================================================================
// X64V3CryptoToken - VPCLMULQDQ + VAES (256-bit)
// =============================================================================

/// Run all X64V3CryptoToken intrinsic tests.
#[test]
fn test_x64v3_crypto_intrinsics() {
    if let Some(token) = X64V3CryptoToken::summon() {
        exercise_vpclmulqdq(token);
        exercise_vaes_256(token);
        println!("All X64V3CryptoToken intrinsic tests passed!");
    } else {
        println!("X64V3CryptoToken not available - skipping tests");
    }
}

#[arcane]
fn exercise_vpclmulqdq(token: X64V3CryptoToken) {
    // 256-bit carryless multiply: processes two 128-bit lanes in parallel
    let a = _mm256_set_epi64x(
        0x0123456789ABCDEFi64,
        0xFEDCBA9876543210u64 as i64,
        0x1111111111111111i64,
        0x2222222222222222i64,
    );
    let b = _mm256_set_epi64x(
        0xAAAAAAAAAAAAAAAAu64 as i64,
        0xBBBBBBBBBBBBBBBBu64 as i64,
        0xCCCCCCCCCCCCCCCCu64 as i64,
        0xDDDDDDDDDDDDDDDDu64 as i64,
    );

    // Same IMM8 encoding as 128-bit, applied per 128-bit lane
    let r00 = _mm256_clmulepi64_epi128::<0x00>(a, b);
    let r01 = _mm256_clmulepi64_epi128::<0x01>(a, b);
    let r10 = _mm256_clmulepi64_epi128::<0x10>(a, b);
    let r11 = _mm256_clmulepi64_epi128::<0x11>(a, b);

    black_box(r00);
    black_box(r01);
    black_box(r10);
    black_box(r11);
}

#[arcane]
fn exercise_vaes_256(token: X64V3CryptoToken) {
    // 256-bit VAES: processes two 128-bit AES blocks in parallel
    let state = _mm256_set_epi64x(
        0x3243f6a8885a308di64,
        0x313198a2e0370734i64,
        0x3243f6a8885a308di64,
        0x313198a2e0370734i64,
    );
    let round_key = _mm256_set_epi64x(
        0x2b7e151628aed2a6u64 as i64,
        0xabf7158809cf4f3cu64 as i64,
        0x2b7e151628aed2a6u64 as i64,
        0xabf7158809cf4f3cu64 as i64,
    );

    // 256-bit encrypt: two AES blocks at once
    let enc = _mm256_aesenc_epi128(state, round_key);
    black_box(enc);

    // 256-bit encrypt last round
    let enc_last = _mm256_aesenclast_epi128(state, round_key);
    black_box(enc_last);

    // 256-bit decrypt
    let dec = _mm256_aesdec_epi128(enc, round_key);
    black_box(dec);

    // 256-bit decrypt last round
    let dec_last = _mm256_aesdeclast_epi128(enc, round_key);
    black_box(dec_last);
}

/// Verify that X64V3CryptoToken implies X64CryptoToken and X64V3Token.
#[test]
fn test_crypto_token_hierarchy() {
    if X64V3CryptoToken::summon().is_some() {
        assert!(
            X64CryptoToken::summon().is_some(),
            "V3Crypto implies Crypto"
        );
        assert!(
            archmage::X64V3Token::summon().is_some(),
            "V3Crypto implies V3"
        );
    }
    if X64CryptoToken::summon().is_some() {
        assert!(
            archmage::X64V2Token::summon().is_some(),
            "Crypto implies V2"
        );
    }
}

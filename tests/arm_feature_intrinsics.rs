//! Per-feature intrinsic exercise tests for ARM64 tokens beyond base NEON.
//!
//! Each section exercises intrinsics for a specific CPU feature to verify that
//! the archmage token correctly enables those features via `#[arcane]`.
//!
//! If feature detection lies about capabilities, these tests will crash (SIGILL).
//!
//! **Tokens tested:**
//! - `Arm64V2Token` — RDM, DotProd, CRC, SHA2, AES round ops, p64 multiply
//! - `Arm64V3Token` — SHA3 (vbcax, veor3, vrax1, vxar, vsha512)
//! - `NeonAesToken` — AES round ops + p64 multiply
//! - `NeonSha3Token` — SHA3 intrinsics
//! - `NeonCrcToken` — CRC32 intrinsics

#![cfg(target_arch = "aarch64")]
#![allow(unused_imports, unused_variables, dead_code)]
#![allow(clippy::eq_op, clippy::identity_op)]

use archmage::{
    Arm64V2Token, Arm64V3Token, NeonAesToken, NeonCrcToken, NeonSha3Token, NeonToken, SimdToken,
    arcane,
};
use core::hint::black_box;
use std::arch::aarch64::*;

// =============================================================================
// RDM — Rounding Doubling Multiply Accumulate (Arm64V2Token)
// All 36 stable intrinsics
// =============================================================================

#[test]
fn test_rdm_intrinsics() {
    if let Some(token) = Arm64V2Token::summon() {
        exercise_rdm(token);
        println!("All RDM intrinsic tests passed!");
    } else {
        println!("Arm64V2Token not available - skipping RDM tests");
    }
}

#[arcane]
fn exercise_rdm(token: Arm64V2Token) {
    let s16x4 = vdup_n_s16(100);
    let s16x8 = vdupq_n_s16(100);
    let s32x2 = vdup_n_s32(100);
    let s32x4 = vdupq_n_s32(100);

    // vqrdmlah — Signed saturating rounding doubling multiply accumulate returning high half
    black_box(vqrdmlah_s16(s16x4, s16x4, s16x4));
    black_box(vqrdmlahq_s16(s16x8, s16x8, s16x8));
    black_box(vqrdmlah_s32(s32x2, s32x2, s32x2));
    black_box(vqrdmlahq_s32(s32x4, s32x4, s32x4));

    // vqrdmlah lane variants
    black_box(vqrdmlah_lane_s16::<0>(s16x4, s16x4, s16x4));
    black_box(vqrdmlah_laneq_s16::<0>(s16x4, s16x4, s16x8));
    black_box(vqrdmlahq_lane_s16::<0>(s16x8, s16x8, s16x4));
    black_box(vqrdmlahq_laneq_s16::<0>(s16x8, s16x8, s16x8));
    black_box(vqrdmlah_lane_s32::<0>(s32x2, s32x2, s32x2));
    black_box(vqrdmlah_laneq_s32::<0>(s32x2, s32x2, s32x4));
    black_box(vqrdmlahq_lane_s32::<0>(s32x4, s32x4, s32x2));
    black_box(vqrdmlahq_laneq_s32::<0>(s32x4, s32x4, s32x4));

    // Scalar variants
    black_box(vqrdmlahh_s16(100i16, 100i16, 100i16));
    black_box(vqrdmlahs_s32(100i32, 100i32, 100i32));
    black_box(vqrdmlahh_lane_s16::<0>(100i16, 100i16, s16x4));
    black_box(vqrdmlahh_laneq_s16::<0>(100i16, 100i16, s16x8));
    black_box(vqrdmlahs_lane_s32::<0>(100i32, 100i32, s32x2));
    black_box(vqrdmlahs_laneq_s32::<0>(100i32, 100i32, s32x4));

    // vqrdmlsh — Signed saturating rounding doubling multiply subtract returning high half
    black_box(vqrdmlsh_s16(s16x4, s16x4, s16x4));
    black_box(vqrdmlshq_s16(s16x8, s16x8, s16x8));
    black_box(vqrdmlsh_s32(s32x2, s32x2, s32x2));
    black_box(vqrdmlshq_s32(s32x4, s32x4, s32x4));

    // vqrdmlsh lane variants
    black_box(vqrdmlsh_lane_s16::<0>(s16x4, s16x4, s16x4));
    black_box(vqrdmlsh_laneq_s16::<0>(s16x4, s16x4, s16x8));
    black_box(vqrdmlshq_lane_s16::<0>(s16x8, s16x8, s16x4));
    black_box(vqrdmlshq_laneq_s16::<0>(s16x8, s16x8, s16x8));
    black_box(vqrdmlsh_lane_s32::<0>(s32x2, s32x2, s32x2));
    black_box(vqrdmlsh_laneq_s32::<0>(s32x2, s32x2, s32x4));
    black_box(vqrdmlshq_lane_s32::<0>(s32x4, s32x4, s32x2));
    black_box(vqrdmlshq_laneq_s32::<0>(s32x4, s32x4, s32x4));

    // Scalar variants
    black_box(vqrdmlshh_s16(100i16, 100i16, 100i16));
    black_box(vqrdmlshs_s32(100i32, 100i32, 100i32));
    black_box(vqrdmlshh_lane_s16::<0>(100i16, 100i16, s16x4));
    black_box(vqrdmlshh_laneq_s16::<0>(100i16, 100i16, s16x8));
    black_box(vqrdmlshs_lane_s32::<0>(100i32, 100i32, s32x2));
    black_box(vqrdmlshs_laneq_s32::<0>(100i32, 100i32, s32x4));

    // Verify basic correctness: vqrdmlah(0, a, b) should equal vqrdmulh(a, b)
    let zero4 = vdup_n_s16(0);
    let a4 = vdup_n_s16(1000);
    let b4 = vdup_n_s16(2000);
    let acc_result = vqrdmlah_s16(zero4, a4, b4);
    let mul_result = vqrdmulh_s16(a4, b4);
    let acc_lane0 = vget_lane_s16::<0>(acc_result);
    let mul_lane0 = vget_lane_s16::<0>(mul_result);
    assert_eq!(
        acc_lane0, mul_lane0,
        "vqrdmlah(0, a, b) should equal vqrdmulh(a, b)"
    );
}

// =============================================================================
// DotProd — Dot Product (Arm64V2Token)
// ALL dotprod intrinsics are nightly-only (stdarch_neon_dotprod).
// Verified: vdot_s32, vdotq_s32, vdot_u32, vdotq_u32 + lane variants
// all require #![feature(stdarch_neon_dotprod)] on stable Rust 1.93.
// =============================================================================

// =============================================================================
// AES Round Operations (NeonAesToken / Arm64V2Token)
// vaeseq, vaesdq, vaesmcq, vaesimcq
// =============================================================================

#[test]
fn test_aes_round_intrinsics() {
    if let Some(token) = NeonAesToken::summon() {
        exercise_aes_rounds(token);
        println!("All ARM AES round intrinsic tests passed!");
    } else {
        println!("NeonAesToken not available - skipping AES round tests");
    }
}

#[arcane]
fn exercise_aes_rounds(token: NeonAesToken) {
    let state = vdupq_n_u8(0x42);
    let round_key = vdupq_n_u8(0x2b);

    // AES single round encrypt (SubBytes + ShiftRows)
    let enc = vaeseq_u8(state, round_key);
    black_box(enc);

    // AES single round decrypt (InvSubBytes + InvShiftRows)
    let dec = vaesdq_u8(enc, round_key);
    black_box(dec);

    // AES mix columns
    let mixed = vaesmcq_u8(enc);
    black_box(mixed);

    // AES inverse mix columns
    let inv_mixed = vaesimcq_u8(enc);
    black_box(inv_mixed);
}

// =============================================================================
// P64 Multiply (NeonAesToken — neon + aes feature)
// vmull_p64, vmull_high_p64
// =============================================================================

#[test]
fn test_p64_multiply_intrinsics() {
    if let Some(token) = NeonAesToken::summon() {
        exercise_p64_multiply(token);
        println!("All p64 multiply intrinsic tests passed!");
    } else {
        println!("NeonAesToken not available - skipping p64 multiply tests");
    }
}

#[arcane]
fn exercise_p64_multiply(token: NeonAesToken) {
    // Polynomial multiply long: two poly64_t inputs -> poly128_t
    let a: u64 = 0x0123456789ABCDEF;
    let b: u64 = 0xFEDCBA9876543210;
    // vmull_p64 takes poly64_t values (reinterpretation of u64)
    let result = vmull_p64(a, b);
    black_box(result);

    // vmull_high_p64 takes poly64x2_t (the upper lane of each input)
    let a_q = vdupq_n_p64(0x1111111111111111);
    let b_q = vdupq_n_p64(0x2222222222222222);
    let result_high = vmull_high_p64(a_q, b_q);
    black_box(result_high);
}

// =============================================================================
// SHA2 (Arm64V2Token)
// SHA-1 and SHA-256 acceleration instructions
// =============================================================================

#[test]
fn test_sha2_intrinsics() {
    if let Some(token) = Arm64V2Token::summon() {
        exercise_sha2(token);
        println!("All SHA2 intrinsic tests passed!");
    } else {
        println!("Arm64V2Token not available - skipping SHA2 tests");
    }
}

#[arcane]
fn exercise_sha2(token: Arm64V2Token) {
    // SHA-256
    let hash_abcd = vdupq_n_u32(0x6a09e667);
    let hash_efgh = vdupq_n_u32(0xbb67ae85);
    let wk = vdupq_n_u32(0x428a2f98);
    let msg0 = vdupq_n_u32(0);
    let msg1 = vdupq_n_u32(1);
    let msg2 = vdupq_n_u32(2);
    let msg3 = vdupq_n_u32(3);

    // SHA-256 hash update (2 rounds)
    let h1 = vsha256hq_u32(hash_abcd, hash_efgh, wk);
    black_box(h1);
    let h2 = vsha256h2q_u32(hash_efgh, hash_abcd, wk);
    black_box(h2);

    // SHA-256 schedule update
    let su0 = vsha256su0q_u32(msg0, msg1);
    black_box(su0);
    let su1 = vsha256su1q_u32(msg0, msg2, msg3);
    black_box(su1);

    // SHA-1
    let hash_sha1 = vdupq_n_u32(0x67452301);
    let hash_e = 0xC3D2E1F0u32;

    // SHA-1 hash update operations (choose, majority, parity)
    let sha1c = vsha1cq_u32(hash_sha1, hash_e, wk);
    black_box(sha1c);
    let sha1m = vsha1mq_u32(hash_sha1, hash_e, wk);
    black_box(sha1m);
    let sha1p = vsha1pq_u32(hash_sha1, hash_e, wk);
    black_box(sha1p);

    // SHA-1 fixed rotate
    let sha1h = vsha1h_u32(hash_e);
    black_box(sha1h);

    // SHA-1 schedule update
    let sha1su0 = vsha1su0q_u32(msg0, msg1, msg2);
    black_box(sha1su0);
    let sha1su1 = vsha1su1q_u32(msg0, msg3);
    black_box(sha1su1);
}

// =============================================================================
// CRC32 (NeonCrcToken)
// Polynomial CRC32/CRC32C acceleration
// =============================================================================

#[test]
fn test_crc32_intrinsics() {
    if let Some(token) = NeonCrcToken::summon() {
        exercise_crc32(token);
        println!("All CRC32 intrinsic tests passed!");
    } else {
        println!("NeonCrcToken not available - skipping CRC32 tests");
    }
}

#[arcane]
fn exercise_crc32(token: NeonCrcToken) {
    let crc: u32 = 0xFFFFFFFF;

    // CRC-32 (ISO 3309 polynomial)
    let c8 = __crc32b(crc, 0x42);
    black_box(c8);
    let c16 = __crc32h(crc, 0x4242);
    black_box(c16);
    let c32 = __crc32w(crc, 0x42424242);
    black_box(c32);
    let c64 = __crc32d(crc, 0x4242424242424242);
    black_box(c64);

    // CRC-32C (Castagnoli polynomial — iSCSI, ext4, etc.)
    let cc8 = __crc32cb(crc, 0x42);
    black_box(cc8);
    let cc16 = __crc32ch(crc, 0x4242);
    black_box(cc16);
    let cc32 = __crc32cw(crc, 0x42424242);
    black_box(cc32);
    let cc64 = __crc32cd(crc, 0x4242424242424242);
    black_box(cc64);

    // CRC-32 and CRC-32C should produce different results for the same input
    assert_ne!(c8, cc8, "CRC-32 and CRC-32C should differ");

    // Verify determinism
    let c8_again = __crc32b(crc, 0x42);
    assert_eq!(c8, c8_again, "CRC-32 should be deterministic");
}

// =============================================================================
// SHA3 — SHA-512 and 3-way XOR/BCAX (NeonSha3Token / Arm64V3Token)
// All 22 stable intrinsics
// =============================================================================

#[test]
fn test_sha3_intrinsics() {
    if let Some(token) = NeonSha3Token::summon() {
        exercise_sha3(token);
        println!("All SHA3 intrinsic tests passed!");
    } else {
        println!("NeonSha3Token not available - skipping SHA3 tests");
    }
}

#[arcane]
fn exercise_sha3(token: NeonSha3Token) {
    // === BCAX — Bit Clear and XOR: result = a ^ (b & ~c) ===
    let a_s8 = vdupq_n_s8(0x0F);
    let b_s8 = vdupq_n_s8(0x33);
    let c_s8 = vdupq_n_s8(0x55);
    black_box(vbcaxq_s8(a_s8, b_s8, c_s8));

    let a_s16 = vdupq_n_s16(0x0F0F);
    let b_s16 = vdupq_n_s16(0x3333);
    let c_s16 = vdupq_n_s16(0x5555);
    black_box(vbcaxq_s16(a_s16, b_s16, c_s16));

    let a_s32 = vdupq_n_s32(0x0F0F0F0F);
    let b_s32 = vdupq_n_s32(0x33333333);
    let c_s32 = vdupq_n_s32(0x55555555);
    black_box(vbcaxq_s32(a_s32, b_s32, c_s32));

    let a_s64 = vdupq_n_s64(0x0F0F0F0F0F0F0F0F);
    let b_s64 = vdupq_n_s64(0x3333333333333333);
    let c_s64 = vdupq_n_s64(0x5555555555555555);
    black_box(vbcaxq_s64(a_s64, b_s64, c_s64));

    let a_u8 = vdupq_n_u8(0x0F);
    let b_u8 = vdupq_n_u8(0x33);
    let c_u8 = vdupq_n_u8(0x55);
    black_box(vbcaxq_u8(a_u8, b_u8, c_u8));

    let a_u16 = vdupq_n_u16(0x0F0F);
    let b_u16 = vdupq_n_u16(0x3333);
    let c_u16 = vdupq_n_u16(0x5555);
    black_box(vbcaxq_u16(a_u16, b_u16, c_u16));

    let a_u32 = vdupq_n_u32(0x0F0F0F0F);
    let b_u32 = vdupq_n_u32(0x33333333);
    let c_u32 = vdupq_n_u32(0x55555555);
    black_box(vbcaxq_u32(a_u32, b_u32, c_u32));

    let a_u64 = vdupq_n_u64(0x0F0F0F0F0F0F0F0F);
    let b_u64 = vdupq_n_u64(0x3333333333333333);
    let c_u64 = vdupq_n_u64(0x5555555555555555);
    black_box(vbcaxq_u64(a_u64, b_u64, c_u64));

    // === EOR3 — Three-way Exclusive OR: result = a ^ b ^ c ===
    black_box(veor3q_s8(a_s8, b_s8, c_s8));
    black_box(veor3q_s16(a_s16, b_s16, c_s16));
    black_box(veor3q_s32(a_s32, b_s32, c_s32));
    black_box(veor3q_s64(a_s64, b_s64, c_s64));
    black_box(veor3q_u8(a_u8, b_u8, c_u8));
    black_box(veor3q_u16(a_u16, b_u16, c_u16));
    black_box(veor3q_u32(a_u32, b_u32, c_u32));
    black_box(veor3q_u64(a_u64, b_u64, c_u64));

    // === RAX1 — Rotate and XOR: result = a ^ rotate_left(b, 1) ===
    let x_u64 = vdupq_n_u64(0x0123456789ABCDEF);
    let y_u64 = vdupq_n_u64(0xFEDCBA9876543210);
    black_box(vrax1q_u64(x_u64, y_u64));

    // === XAR — XOR and Rotate: result = rotate_right(a ^ b, imm) ===
    black_box(vxarq_u64::<1>(x_u64, y_u64));
    black_box(vxarq_u64::<32>(x_u64, y_u64));

    // === SHA-512 hash operations ===
    let ab = vdupq_n_u64(0x6a09e667f3bcc908);
    let cd = vdupq_n_u64(0xbb67ae8584caa73b);
    let ef = vdupq_n_u64(0x3c6ef372fe94f82b);
    let gh = vdupq_n_u64(0xa54ff53a5f1d36f1);

    // SHA-512 hash update part 1
    let h1 = vsha512hq_u64(ab, cd, ef);
    black_box(h1);

    // SHA-512 hash update part 2
    let h2 = vsha512h2q_u64(ab, cd, ef);
    black_box(h2);

    // SHA-512 schedule update 0
    let su0 = vsha512su0q_u64(ab, cd);
    black_box(su0);

    // SHA-512 schedule update 1
    let su1 = vsha512su1q_u64(ab, cd, ef);
    black_box(su1);
}

// =============================================================================
// Token Hierarchy Verification
// =============================================================================

#[test]
fn test_arm_token_hierarchy() {
    // Arm64V3 implies Arm64V2 implies Neon
    if Arm64V3Token::summon().is_some() {
        assert!(
            Arm64V2Token::summon().is_some(),
            "Arm64V3 implies Arm64V2"
        );
        assert!(NeonToken::summon().is_some(), "Arm64V3 implies Neon");
        assert!(
            NeonSha3Token::summon().is_some(),
            "Arm64V3 implies NeonSha3"
        );
    }

    if Arm64V2Token::summon().is_some() {
        assert!(NeonToken::summon().is_some(), "Arm64V2 implies Neon");
        assert!(
            NeonAesToken::summon().is_some(),
            "Arm64V2 implies NeonAes"
        );
        assert!(
            NeonCrcToken::summon().is_some(),
            "Arm64V2 implies NeonCrc"
        );
    }
}

// =============================================================================
// Nightly-Only / Missing Features (documented)
// =============================================================================

// FP16 (neon,fp16): ALL 214+ intrinsics are UNSTABLE in Rust stdarch.
// Examples: vdivq_f16, vsqrtq_f16, vmaxnmq_f16, etc.
// Cannot be tested on stable Rust. Document and skip.
//
// FCMA (fcma): 34 intrinsics, ALL UNSTABLE (nightly_arm_intrinsics).
// Complex number multiply-accumulate operations.
// Examples: vcadd_rot90_f32, vcmla_f32, vcmla_rot90_f32, etc.
//
// I8MM (i8mm): 4 intrinsics, ALL UNSTABLE.
// Int8 matrix multiply: vsmmla_s32, vummla_u32, vusmmlaq_s32, etc.
//
// FHM (fhm): ZERO intrinsics in Rust stdarch.
// FP16 fused multiply-add half-precision to single-precision.
// The hardware feature exists but Rust has no intrinsic bindings yet.
//
// BF16 (bf16): ZERO intrinsics in Rust stdarch.
// BFloat16 support. The hardware feature exists but Rust has no bindings yet.
//
// DotProd laneq variants: vdot_laneq_s32, vdotq_laneq_s32, etc.
// These are UNSTABLE (require nightly).

//! AVX-512 FP16 intrinsic exercise tests for Avx512Fp16Token.
//!
//! STATUS: ALL 935 avx512fp16 intrinsics are UNSTABLE in Rust (require nightly
//! `#![feature(avx512fp16)]`). There are ZERO stable avx512fp16 intrinsics as of
//! Rust 1.93. This file tests what we CAN test on stable:
//!   - Token summoning and hierarchy
//!   - Avx512Fp16Token implies X64V4Token, X64V3Token, X64V2Token
//!
//! When avx512fp16 intrinsics stabilize, add comprehensive exercise tests here
//! following the pattern in avx512_intrinsics_exercise.rs.
//!
//! Hardware: Intel Sapphire Rapids (2023+), Emerald Rapids. NOT available on
//! AMD Zen 4 (has AVX-512 but not FP16) or earlier Intel (Skylake-X, Ice Lake).
//!
//! Intrinsic categories (935 total, all nightly-only):
//!   - Arithmetic: add, sub, mul, div, sqrt, rcp, rsqrt, min, max (512/256/128-bit)
//!   - FMA: fmadd, fmsub, fnmadd, fnmsub, fmaddsub, fmsubadd (+ complex variants)
//!   - Comparison: cmp_ph_mask, cmp_sh_mask, comi_sh, ucomi_sh
//!   - Conversion: cvtph_ps, cvtps_ph, cvtph_pd, cvtpd_ph, cvtph_epi16, cvtepi16_ph
//!   - Scalar: add_sh, mul_sh, div_sh, sqrt_sh, rcp_sh, rsqrt_sh
//!   - Masked: mask_add_ph, maskz_add_ph (all arithmetic has mask/maskz variants)
//!   - Set/Get: set_ph, set1_ph, setzero_ph, castph_ps, castph_si512
//!   - Reduction: reduce_add_ph, reduce_mul_ph, reduce_min_ph, reduce_max_ph (via masks)

#![cfg(target_arch = "x86_64")]
#![cfg(feature = "avx512")]

use archmage::SimdToken;

/// Verify Avx512Fp16Token hierarchy: FP16 implies V4, V3, V2.
#[test]
fn fp16_token_hierarchy() {
    if archmage::Avx512Fp16Token::summon().is_some() {
        assert!(
            archmage::X64V4Token::summon().is_some(),
            "Avx512Fp16 implies X64V4"
        );
        assert!(
            archmage::X64V3Token::summon().is_some(),
            "Avx512Fp16 implies X64V3"
        );
        assert!(
            archmage::X64V2Token::summon().is_some(),
            "Avx512Fp16 implies X64V2"
        );
    }
}

/// Print FP16 detection status.
#[test]
fn print_fp16_status() {
    let available = archmage::Avx512Fp16Token::summon().is_some();
    println!("Avx512Fp16Token available: {available}");
    println!("Note: 935 avx512fp16 intrinsics exist but ALL are nightly-only (unstable).");
    println!("No intrinsic exercise tests possible on stable Rust.");
    if !available {
        println!("(This is expected — FP16 requires Sapphire Rapids or newer Intel.)");
    }
}

// =============================================================================
// Nightly-only intrinsic tests
// =============================================================================
// When Rust stabilizes avx512fp16 intrinsics, uncomment and expand the sections
// below. Use the pattern from avx512_intrinsics_exercise.rs:
//   - #[arcane] functions with Avx512Fp16Token
//   - black_box() all results
//   - Value verification where possible
//
// Categories to cover (~60 representative from 935):
//
// ARITHMETIC (512-bit):
//   _mm512_add_ph, _mm512_sub_ph, _mm512_mul_ph, _mm512_div_ph
//   _mm512_sqrt_ph, _mm512_rcp_ph, _mm512_rsqrt_ph
//   _mm512_min_ph, _mm512_max_ph, _mm512_abs_ph
//
// ARITHMETIC (256-bit):
//   _mm256_add_ph, _mm256_sub_ph, _mm256_mul_ph, _mm256_div_ph
//   _mm256_sqrt_ph, _mm256_min_ph, _mm256_max_ph
//
// ARITHMETIC (128-bit):
//   _mm_add_ph, _mm_sub_ph, _mm_mul_ph, _mm_div_ph
//   _mm_sqrt_ph, _mm_min_ph, _mm_max_ph
//
// FMA (512-bit):
//   _mm512_fmadd_ph, _mm512_fmsub_ph, _mm512_fnmadd_ph, _mm512_fmaddsub_ph
//
// COMPARISON:
//   _mm512_cmp_ph_mask, _mm_cmp_sh_mask
//
// CONVERSION:
//   _mm256_cvtph_ps (256->256), _mm512_cvtph_ps (256->512)
//   _mm512_cvtps_ph (512->256), _mm512_cvtph_pd (128->512)
//   _mm256_cvtph_epi16, _mm256_cvtepi16_ph
//
// SCALAR:
//   _mm_add_sh, _mm_mul_sh, _mm_div_sh, _mm_sqrt_sh
//
// MASKED:
//   _mm512_mask_add_ph, _mm512_maskz_add_ph
//   _mm512_mask_mul_ph, _mm512_maskz_mul_ph
//
// REDUCTION (via scalar extraction):
//   _mm512_reduce_add_ph, _mm512_reduce_min_ph, _mm512_reduce_max_ph
//
// SET/INIT:
//   _mm512_set1_ph, _mm512_setzero_ph, _mm512_set_ph
//   _mm256_set1_ph, _mm_set1_ph, _mm_set_sh

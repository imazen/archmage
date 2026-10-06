//! AVX-512 FP16 intrinsic exercise tests for Avx512Fp16Token.
//!
//! STATUS: the avx512fp16 intrinsics on `__m128h`/`__m256h`/`__m512h` are stable
//! since Rust 1.94.0 (`stdarch_x86_avx512fp16`). Those that take or return the
//! scalar `f16` type still need nightly, because `f16` itself is unstable. This
//! file tests:
//!   - Token summoning and hierarchy
//!   - Avx512Fp16Token implies X64V4Token, X64V3Token, X64V2Token
//!   - Arithmetic, FMA, conversion, comparison and masked intrinsics at 512,
//!     256 and 128 bits (`test_fp16_intrinsics`, gated on Rust >= 1.94)
//!
//! Hardware: Intel Sapphire Rapids (2023+), Emerald Rapids. NOT available on
//! AMD Zen 4 (has AVX-512 but not FP16) or earlier Intel (Skylake-X, Ice Lake).
//!
//! Intrinsic categories:
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

use archmage::{Avx512Fp16Token, SimdToken, arcane, rite};
use core::arch::x86_64::*;

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
    if !available {
        println!("(This is expected — FP16 requires Sapphire Rapids or newer Intel.)");
    }
}

// =============================================================================
// Exercise tests: stable since Rust 1.94.0 (`stdarch_x86_avx512fp16`)
// =============================================================================
// Intrinsics that take or return the scalar `f16` type (`_mm512_set1_ph`,
// `_mm512_reduce_add_ph`, the `_sh` scalar forms) still need nightly, because
// `f16` itself is unstable. These tests build vectors from bit patterns and
// read results back through f32 conversions: 0x3C00 = 1.0, 0x3E00 = 1.5,
// 0x4000 = 2.0, 0x4400 = 4.0, 0xC000 = -2.0. Every expected value is exact.
// CI runs them under Intel SDE's Sapphire Rapids model (-spr).

#[rustversion::since(1.94)]
#[test]
fn test_fp16_intrinsics() {
    if let Some(token) = Avx512Fp16Token::summon() {
        exercise_fp16_512(token);
        exercise_fp16_vl(token);
        println!("AVX-512 FP16 intrinsic tests passed!");
    } else {
        println!("Avx512Fp16Token not available - skipping FP16 tests");
    }
}

#[rustversion::since(1.94)]
#[rite]
fn splat512(_t: Avx512Fp16Token, bits: u16) -> __m512h {
    _mm512_castsi512_ph(_mm512_set1_epi16(bits as i16))
}

/// Lanes 0 and 1 of `v`, widened to f32.
#[rustversion::since(1.94)]
#[rite]
fn low_lanes512(_t: Avx512Fp16Token, v: __m512h) -> [f32; 2] {
    let w = _mm512_castps512_ps128(_mm512_cvtxph_ps(_mm512_castph512_ph256(v)));
    [_mm_cvtss_f32(w), f32::from_bits(_mm_extract_ps::<1>(w) as u32)]
}

#[rustversion::since(1.94)]
#[arcane]
fn exercise_fp16_512(token: Avx512Fp16Token) {
    let one = splat512(token, 0x3C00);
    let a = splat512(token, 0x3E00);
    let b = splat512(token, 0x4000);
    let four = splat512(token, 0x4400);
    let neg2 = splat512(token, 0xC000);

    // Arithmetic
    assert_eq!(low_lanes512(token, _mm512_add_ph(a, b))[0], 3.5);
    assert_eq!(low_lanes512(token, _mm512_sub_ph(b, a))[0], 0.5);
    assert_eq!(low_lanes512(token, _mm512_mul_ph(a, b))[0], 3.0);
    assert_eq!(low_lanes512(token, _mm512_div_ph(a, b))[0], 0.75);
    assert_eq!(low_lanes512(token, _mm512_sqrt_ph(four))[0], 2.0);
    assert_eq!(low_lanes512(token, _mm512_min_ph(a, b))[0], 1.5);
    assert_eq!(low_lanes512(token, _mm512_max_ph(a, b))[0], 2.0);
    assert_eq!(low_lanes512(token, _mm512_abs_ph(neg2))[0], 2.0);
    // The reciprocal estimate is within 2^-11 relative error.
    assert!((low_lanes512(token, _mm512_rcp_ph(b))[0] - 0.5).abs() < 0.001);

    // FMA: a * b + c, a * b - c, -(a * b) + c
    assert_eq!(low_lanes512(token, _mm512_fmadd_ph(a, b, one))[0], 4.0);
    assert_eq!(low_lanes512(token, _mm512_fmsub_ph(a, b, one))[0], 2.0);
    assert_eq!(low_lanes512(token, _mm512_fnmadd_ph(a, b, one))[0], -2.0);

    // Conversions
    let narrowed = _mm512_cvtxps_ph(_mm512_set1_ps(3.25));
    assert_eq!(_mm512_cvtss_f32(_mm512_cvtxph_ps(narrowed)), 3.25);
    let from_int = _mm512_cvtepi16_ph(_mm512_set1_epi16(7));
    assert_eq!(low_lanes512(token, from_int)[0], 7.0);
    let back = _mm512_cvtph_epi16(from_int);
    assert_eq!(_mm_extract_epi16::<0>(_mm512_castsi512_si128(back)), 7);

    // Compare: 1.5 < 2.0 in all 32 lanes
    assert_eq!(_mm512_cmp_ph_mask::<_CMP_LT_OQ>(a, b), u32::MAX);

    // Masking: lane 0 adds, lane 1 keeps `src` (mask) or zeroes (maskz)
    let masked = low_lanes512(token, _mm512_mask_add_ph(one, 0b01, a, b));
    assert_eq!(masked, [3.5, 1.0]);
    let zeroed = low_lanes512(token, _mm512_maskz_add_ph(0b10, a, b));
    assert_eq!(zeroed, [0.0, 3.5]);
    assert_eq!(low_lanes512(token, _mm512_setzero_ph())[0], 0.0);
}

/// Lane 0 of a 256-bit half vector, widened to f32.
#[rustversion::since(1.94)]
#[rite]
fn lane0_256(_t: Avx512Fp16Token, v: __m256h) -> f32 {
    _mm256_cvtss_f32(_mm256_cvtxph_ps(_mm256_castph256_ph128(v)))
}

/// Lane 0 of a 128-bit half vector, widened to f32.
#[rustversion::since(1.94)]
#[rite]
fn lane0_128(_t: Avx512Fp16Token, v: __m128h) -> f32 {
    _mm_cvtss_f32(_mm_cvtxph_ps(v))
}

#[rustversion::since(1.94)]
#[arcane]
fn exercise_fp16_vl(token: Avx512Fp16Token) {
    let a256 = _mm256_castsi256_ph(_mm256_set1_epi16(0x3E00));
    let b256 = _mm256_castsi256_ph(_mm256_set1_epi16(0x4000));
    assert_eq!(lane0_256(token, _mm256_add_ph(a256, b256)), 3.5);
    assert_eq!(lane0_256(token, _mm256_sub_ph(b256, a256)), 0.5);
    assert_eq!(lane0_256(token, _mm256_mul_ph(a256, b256)), 3.0);
    assert_eq!(lane0_256(token, _mm256_div_ph(a256, b256)), 0.75);

    let a128 = _mm_castsi128_ph(_mm_set1_epi16(0x3E00));
    let b128 = _mm_castsi128_ph(_mm_set1_epi16(0x4000));
    assert_eq!(lane0_128(token, _mm_add_ph(a128, b128)), 3.5);
    assert_eq!(lane0_128(token, _mm_sub_ph(b128, a128)), 0.5);
    assert_eq!(lane0_128(token, _mm_mul_ph(a128, b128)), 3.0);
    assert_eq!(lane0_128(token, _mm_div_ph(a128, b128)), 0.75);
}

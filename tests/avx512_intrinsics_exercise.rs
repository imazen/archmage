//! Comprehensive AVX-512 intrinsic exercise tests for Avx512ModernToken.
//!
//! This file exercises intrinsics available with Avx512ModernToken (Ice Lake / Zen 4)
//! to verify they compile and execute correctly with archmage tokens.
//!
//! Avx512ModernToken requires: AVX-512F, CD, VL, DQ, BW, VPOPCNTDQ, IFMA, VBMI,
//! VBMI2, BITALG, VNNI, BF16, VPCLMULQDQ, GFNI, VAES + x86-64-v3 baseline

#![cfg(target_arch = "x86_64")]
#![allow(unused_imports, unused_variables, dead_code, unused_mut)]
#![allow(clippy::needless_return, clippy::identity_op, clippy::unnecessary_cast)]
#![allow(clippy::eq_op)]

use archmage::{arcane, SimdToken};
use archmage::tokens::x86::Avx512ModernToken;
#[cfg(feature = "safe_unaligned_simd")]
use archmage::mem::{avx512f, avx512f_vl, avx512bw, avx512bw_vl, avx512vbmi2, avx512vbmi2_vl};
use core::arch::x86_64::*;
use core::hint::black_box;

/// Run all AVX-512 Modern intrinsic tests.
#[test]
fn test_avx512_modern_intrinsics() {
    if let Some(token) = Avx512ModernToken::try_new() {
        exercise_avx512f(token);
        exercise_avx512f_vl(token);
        exercise_avx512bw(token);
        exercise_avx512bw_vl(token);
        exercise_avx512dq(token);
        exercise_avx512dq_vl(token);
        exercise_avx512cd(token);
        exercise_avx512cd_vl(token);
        exercise_avx512vbmi(token);
        exercise_avx512vbmi2(token);
        exercise_avx512vnni(token);
        exercise_avx512bitalg(token);
        exercise_avx512bf16(token);
        exercise_avx512ifma(token);
        exercise_gfni(token);
        exercise_vaes(token);
        exercise_vpclmulqdq(token);
        #[cfg(feature = "safe_unaligned_simd")]
        exercise_safe_mem_ops(token);
        println!("All AVX-512 Modern intrinsic tests passed!");
    } else {
        println!("Avx512ModernToken not available - skipping tests");
    }
}

// =============================================================================
// AVX-512F (Foundation) - 512-bit operations
// =============================================================================

#[arcane]
fn exercise_avx512f(token: Avx512ModernToken) {
    // Initialization
    let zero_ps = _mm512_setzero_ps();
    let zero_pd = _mm512_setzero_pd();
    let zero_i = _mm512_setzero_si512();
    let ones_ps = _mm512_set1_ps(1.0);
    let ones_pd = _mm512_set1_pd(1.0);
    let ones_i32 = _mm512_set1_epi32(1);
    let ones_i64 = _mm512_set1_epi64(1);
    let twos_ps = _mm512_set1_ps(2.0);
    let twos_pd = _mm512_set1_pd(2.0);

    // Masks
    let mask8: __mmask8 = 0xFF;
    let mask16: __mmask16 = 0xFFFF;

    // Arrays for load/store
    let mut arr_f32 = [0.0f32; 16];
    let mut arr_f64 = [0.0f64; 8];
    let mut arr_i32 = [0i32; 16];
    let mut arr_i64 = [0i64; 8];

    // === Arithmetic Operations (Float) ===
    black_box(_mm512_add_ps(ones_ps, twos_ps));
    black_box(_mm512_add_pd(ones_pd, twos_pd));
    black_box(_mm512_sub_ps(twos_ps, ones_ps));
    black_box(_mm512_sub_pd(twos_pd, ones_pd));
    black_box(_mm512_mul_ps(ones_ps, twos_ps));
    black_box(_mm512_mul_pd(ones_pd, twos_pd));
    black_box(_mm512_div_ps(twos_ps, ones_ps));
    black_box(_mm512_div_pd(twos_pd, ones_pd));

    // Min/max
    black_box(_mm512_min_ps(ones_ps, twos_ps));
    black_box(_mm512_min_pd(ones_pd, twos_pd));
    black_box(_mm512_max_ps(ones_ps, twos_ps));
    black_box(_mm512_max_pd(ones_pd, twos_pd));

    // Square root, reciprocal
    black_box(_mm512_sqrt_ps(ones_ps));
    black_box(_mm512_sqrt_pd(ones_pd));
    black_box(_mm512_rsqrt14_ps(ones_ps));
    black_box(_mm512_rsqrt14_pd(ones_pd));
    black_box(_mm512_rcp14_ps(ones_ps));
    black_box(_mm512_rcp14_pd(ones_pd));

    // === FMA Operations ===
    black_box(_mm512_fmadd_ps(ones_ps, twos_ps, ones_ps));
    black_box(_mm512_fmadd_pd(ones_pd, twos_pd, ones_pd));
    black_box(_mm512_fmsub_ps(ones_ps, twos_ps, ones_ps));
    black_box(_mm512_fmsub_pd(ones_pd, twos_pd, ones_pd));
    black_box(_mm512_fnmadd_ps(ones_ps, twos_ps, ones_ps));
    black_box(_mm512_fnmadd_pd(ones_pd, twos_pd, ones_pd));
    black_box(_mm512_fnmsub_ps(ones_ps, twos_ps, ones_ps));
    black_box(_mm512_fnmsub_pd(ones_pd, twos_pd, ones_pd));
    black_box(_mm512_fmaddsub_ps(ones_ps, twos_ps, ones_ps));
    black_box(_mm512_fmaddsub_pd(ones_pd, twos_pd, ones_pd));
    black_box(_mm512_fmsubadd_ps(ones_ps, twos_ps, ones_ps));
    black_box(_mm512_fmsubadd_pd(ones_pd, twos_pd, ones_pd));

    // === Integer Arithmetic ===
    black_box(_mm512_add_epi32(ones_i32, ones_i32));
    black_box(_mm512_add_epi64(ones_i64, ones_i64));
    black_box(_mm512_sub_epi32(ones_i32, ones_i32));
    black_box(_mm512_sub_epi64(ones_i64, ones_i64));
    black_box(_mm512_mullo_epi32(ones_i32, ones_i32));
    black_box(_mm512_mullo_epi64(ones_i64, ones_i64));
    black_box(_mm512_min_epi32(ones_i32, ones_i32));
    black_box(_mm512_min_epi64(ones_i64, ones_i64));
    black_box(_mm512_max_epi32(ones_i32, ones_i32));
    black_box(_mm512_max_epi64(ones_i64, ones_i64));
    black_box(_mm512_min_epu32(ones_i32, ones_i32));
    black_box(_mm512_min_epu64(ones_i64, ones_i64));
    black_box(_mm512_max_epu32(ones_i32, ones_i32));
    black_box(_mm512_max_epu64(ones_i64, ones_i64));
    black_box(_mm512_abs_epi32(ones_i32));
    black_box(_mm512_abs_epi64(ones_i64));

    // === Bitwise Operations ===
    black_box(_mm512_and_si512(ones_i32, ones_i32));
    black_box(_mm512_or_si512(ones_i32, ones_i32));
    black_box(_mm512_xor_si512(ones_i32, ones_i32));
    black_box(_mm512_andnot_si512(ones_i32, ones_i32));

    // === Shifts ===
    black_box(_mm512_slli_epi32::<1>(ones_i32));
    black_box(_mm512_slli_epi64::<1>(ones_i64));
    black_box(_mm512_srli_epi32::<1>(ones_i32));
    black_box(_mm512_srli_epi64::<1>(ones_i64));
    black_box(_mm512_srai_epi32::<1>(ones_i32));
    black_box(_mm512_srai_epi64::<1>(ones_i64));
    black_box(_mm512_sllv_epi32(ones_i32, ones_i32));
    black_box(_mm512_sllv_epi64(ones_i64, ones_i64));
    black_box(_mm512_srlv_epi32(ones_i32, ones_i32));
    black_box(_mm512_srlv_epi64(ones_i64, ones_i64));
    black_box(_mm512_srav_epi32(ones_i32, ones_i32));
    black_box(_mm512_srav_epi64(ones_i64, ones_i64));
    black_box(_mm512_rolv_epi32(ones_i32, ones_i32));
    black_box(_mm512_rolv_epi64(ones_i64, ones_i64));
    black_box(_mm512_rorv_epi32(ones_i32, ones_i32));
    black_box(_mm512_rorv_epi64(ones_i64, ones_i64));

    // === Comparisons ===
    black_box(_mm512_cmpeq_epi32_mask(ones_i32, ones_i32));
    black_box(_mm512_cmpeq_epi64_mask(ones_i64, ones_i64));
    black_box(_mm512_cmplt_epi32_mask(ones_i32, ones_i32));
    black_box(_mm512_cmpgt_epi32_mask(ones_i32, ones_i32));
    black_box(_mm512_cmpeq_ps_mask(ones_ps, ones_ps));
    black_box(_mm512_cmpeq_pd_mask(ones_pd, ones_pd));
    black_box(_mm512_cmplt_ps_mask(ones_ps, twos_ps));
    black_box(_mm512_cmplt_pd_mask(ones_pd, twos_pd));

    // === Conversions ===
    black_box(_mm512_cvtepi32_ps(ones_i32));
    black_box(_mm512_cvtepi32_pd(_mm256_set1_epi32(1)));
    black_box(_mm512_cvtps_epi32(ones_ps));
    black_box(_mm512_cvtpd_epi32(ones_pd));
    black_box(_mm512_cvtps_pd(_mm256_set1_ps(1.0)));
    black_box(_mm512_cvtpd_ps(ones_pd));
    black_box(_mm512_cvtepu32_pd(_mm256_set1_epi32(1)));
    black_box(_mm512_cvtepi32_epi64(_mm256_set1_epi32(1)));
    black_box(_mm512_cvtepu32_epi64(_mm256_set1_epi32(1)));

    // === Shuffles and Permutes ===
    black_box(_mm512_shuffle_ps::<0b00_01_10_11>(ones_ps, twos_ps));
    black_box(_mm512_shuffle_pd::<0b01010101>(ones_pd, twos_pd));
    black_box(_mm512_shuffle_i32x4::<0b00_01_10_11>(ones_i32, ones_i32));
    black_box(_mm512_shuffle_i64x2::<0b00_01_10_11>(ones_i64, ones_i64));
    black_box(_mm512_permutex_epi64::<0b00_01_10_11>(ones_i64));
    black_box(_mm512_permutex_pd::<0b00_01_10_11>(ones_pd));
    black_box(_mm512_permutexvar_epi32(ones_i32, ones_i32));
    black_box(_mm512_permutexvar_epi64(ones_i64, ones_i64));
    black_box(_mm512_permutexvar_ps(ones_i32, ones_ps));
    black_box(_mm512_permutexvar_pd(ones_i64, ones_pd));

    // === Blends ===
    black_box(_mm512_mask_blend_ps(mask16, ones_ps, twos_ps));
    black_box(_mm512_mask_blend_pd(mask8, ones_pd, twos_pd));
    black_box(_mm512_mask_blend_epi32(mask16, ones_i32, ones_i32));
    black_box(_mm512_mask_blend_epi64(mask8, ones_i64, ones_i64));

    // === Unpack ===
    black_box(_mm512_unpackhi_ps(ones_ps, twos_ps));
    black_box(_mm512_unpacklo_ps(ones_ps, twos_ps));
    black_box(_mm512_unpackhi_pd(ones_pd, twos_pd));
    black_box(_mm512_unpacklo_pd(ones_pd, twos_pd));
    black_box(_mm512_unpackhi_epi32(ones_i32, ones_i32));
    black_box(_mm512_unpacklo_epi32(ones_i32, ones_i32));
    black_box(_mm512_unpackhi_epi64(ones_i64, ones_i64));
    black_box(_mm512_unpacklo_epi64(ones_i64, ones_i64));

    // === Broadcasts ===
    black_box(_mm512_broadcastss_ps(_mm_set1_ps(1.0)));
    black_box(_mm512_broadcastsd_pd(_mm_set1_pd(1.0)));
    black_box(_mm512_broadcastd_epi32(_mm_set1_epi32(1)));
    black_box(_mm512_broadcastq_epi64(_mm_set1_epi64x(1)));

    // === Reductions ===
    black_box(_mm512_reduce_add_ps(ones_ps));
    black_box(_mm512_reduce_add_pd(ones_pd));
    black_box(_mm512_reduce_add_epi32(ones_i32));
    black_box(_mm512_reduce_add_epi64(ones_i64));
    black_box(_mm512_reduce_mul_ps(ones_ps));
    black_box(_mm512_reduce_mul_pd(ones_pd));
    black_box(_mm512_reduce_min_ps(ones_ps));
    black_box(_mm512_reduce_min_pd(ones_pd));
    black_box(_mm512_reduce_max_ps(ones_ps));
    black_box(_mm512_reduce_max_pd(ones_pd));

    // === Load/Store (safe with archmage::mem) ===
    #[cfg(feature = "safe_unaligned_simd")]
    {
        let arr_f32_sized: &[f32; 16] = arr_f32.as_slice().try_into().unwrap();
        let arr_f64_sized: &[f64; 8] = arr_f64.as_slice().try_into().unwrap();
        black_box(avx512f::_mm512_loadu_ps(token, arr_f32_sized));
        black_box(avx512f::_mm512_loadu_pd(token, arr_f64_sized));
        avx512f::_mm512_storeu_ps(token, &mut arr_f32, ones_ps);
        avx512f::_mm512_storeu_pd(token, &mut arr_f64, ones_pd);
    }

    // === Masked Operations ===
    black_box(_mm512_mask_add_ps(ones_ps, mask16, ones_ps, twos_ps));
    black_box(_mm512_maskz_add_ps(mask16, ones_ps, twos_ps));
    black_box(_mm512_mask_add_pd(ones_pd, mask8, ones_pd, twos_pd));
    black_box(_mm512_maskz_add_pd(mask8, ones_pd, twos_pd));
    black_box(_mm512_mask_add_epi32(ones_i32, mask16, ones_i32, ones_i32));
    black_box(_mm512_maskz_add_epi32(mask16, ones_i32, ones_i32));
    black_box(_mm512_mask_mov_ps(ones_ps, mask16, twos_ps));
    black_box(_mm512_maskz_mov_ps(mask16, ones_ps));
    black_box(_mm512_mask_mov_epi32(ones_i32, mask16, ones_i32));
    black_box(_mm512_maskz_mov_epi32(mask16, ones_i32));

    // === Ternary Logic ===
    black_box(_mm512_ternarylogic_epi32::<0xF0>(ones_i32, ones_i32, ones_i32));
    black_box(_mm512_ternarylogic_epi64::<0xF0>(ones_i64, ones_i64, ones_i64));

    // === Getexp/Getmant ===
    black_box(_mm512_getexp_ps(ones_ps));
    black_box(_mm512_getexp_pd(ones_pd));
    black_box(_mm512_getmant_ps::<_MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC>(ones_ps));
    black_box(_mm512_getmant_pd::<_MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC>(ones_pd));

    // === Scale ===
    black_box(_mm512_scalef_ps(ones_ps, ones_ps));
    black_box(_mm512_scalef_pd(ones_pd, ones_pd));

    // === Fixup ===
    black_box(_mm512_fixupimm_ps::<0>(ones_ps, ones_ps, ones_i32));
    black_box(_mm512_fixupimm_pd::<0>(ones_pd, ones_pd, ones_i64));
}

// =============================================================================
// AVX-512F+VL - 128/256-bit operations
// =============================================================================

#[arcane]
fn exercise_avx512f_vl(token: Avx512ModernToken) {
    // 256-bit vectors
    let v256_ps = _mm256_set1_ps(1.0);
    let v256_pd = _mm256_set1_pd(1.0);
    let v256_i32 = _mm256_set1_epi32(1);
    let v256_i64 = _mm256_set1_epi64x(1);

    // 128-bit vectors
    let v128_ps = _mm_set1_ps(1.0);
    let v128_pd = _mm_set1_pd(1.0);
    let v128_i32 = _mm_set1_epi32(1);
    let v128_i64 = _mm_set1_epi64x(1);

    let mask4: __mmask8 = 0x0F;
    let mask8: __mmask8 = 0xFF;

    // 256-bit masked operations
    black_box(_mm256_mask_add_ps(v256_ps, mask8, v256_ps, v256_ps));
    black_box(_mm256_maskz_add_ps(mask8, v256_ps, v256_ps));
    black_box(_mm256_mask_add_pd(v256_pd, mask4, v256_pd, v256_pd));
    black_box(_mm256_maskz_add_pd(mask4, v256_pd, v256_pd));
    black_box(_mm256_mask_add_epi32(v256_i32, mask8, v256_i32, v256_i32));
    black_box(_mm256_maskz_add_epi32(mask8, v256_i32, v256_i32));
    black_box(_mm256_mask_add_epi64(v256_i64, mask4, v256_i64, v256_i64));
    black_box(_mm256_maskz_add_epi64(mask4, v256_i64, v256_i64));

    // 256-bit FMA
    black_box(_mm256_mask_fmadd_ps(v256_ps, mask8, v256_ps, v256_ps));
    black_box(_mm256_maskz_fmadd_ps(mask8, v256_ps, v256_ps, v256_ps));
    black_box(_mm256_mask_fmadd_pd(v256_pd, mask4, v256_pd, v256_pd));
    black_box(_mm256_maskz_fmadd_pd(mask4, v256_pd, v256_pd, v256_pd));

    // 256-bit permutes
    black_box(_mm256_permutexvar_epi32(v256_i32, v256_i32));
    black_box(_mm256_permutexvar_ps(v256_i32, v256_ps));

    // 256-bit ternary logic
    black_box(_mm256_ternarylogic_epi32::<0xF0>(v256_i32, v256_i32, v256_i32));
    black_box(_mm256_ternarylogic_epi64::<0xF0>(v256_i64, v256_i64, v256_i64));

    // 128-bit masked operations
    black_box(_mm_mask_add_ps(v128_ps, mask4, v128_ps, v128_ps));
    black_box(_mm_maskz_add_ps(mask4, v128_ps, v128_ps));
    black_box(_mm_mask_add_pd(v128_pd, 0x03, v128_pd, v128_pd));
    black_box(_mm_maskz_add_pd(0x03, v128_pd, v128_pd));
    black_box(_mm_mask_add_epi32(v128_i32, mask4, v128_i32, v128_i32));
    black_box(_mm_maskz_add_epi32(mask4, v128_i32, v128_i32));
    black_box(_mm_mask_add_epi64(v128_i64, 0x03, v128_i64, v128_i64));
    black_box(_mm_maskz_add_epi64(0x03, v128_i64, v128_i64));

    // 128-bit ternary logic
    black_box(_mm_ternarylogic_epi32::<0xF0>(v128_i32, v128_i32, v128_i32));
    black_box(_mm_ternarylogic_epi64::<0xF0>(v128_i64, v128_i64, v128_i64));
}

// =============================================================================
// AVX-512BW - Byte/Word operations
// =============================================================================

#[arcane]
fn exercise_avx512bw(token: Avx512ModernToken) {
    let zero_i = _mm512_setzero_si512();
    let ones_i8 = _mm512_set1_epi8(1);
    let ones_i16 = _mm512_set1_epi16(1);
    let mask32: __mmask32 = 0xFFFFFFFF;
    let mask64: __mmask64 = 0xFFFFFFFFFFFFFFFF;

    // Byte operations
    black_box(_mm512_add_epi8(ones_i8, ones_i8));
    black_box(_mm512_sub_epi8(ones_i8, ones_i8));
    black_box(_mm512_adds_epi8(ones_i8, ones_i8));
    black_box(_mm512_adds_epu8(ones_i8, ones_i8));
    black_box(_mm512_subs_epi8(ones_i8, ones_i8));
    black_box(_mm512_subs_epu8(ones_i8, ones_i8));
    black_box(_mm512_min_epi8(ones_i8, ones_i8));
    black_box(_mm512_max_epi8(ones_i8, ones_i8));
    black_box(_mm512_min_epu8(ones_i8, ones_i8));
    black_box(_mm512_max_epu8(ones_i8, ones_i8));
    black_box(_mm512_abs_epi8(ones_i8));
    black_box(_mm512_avg_epu8(ones_i8, ones_i8));

    // Word operations
    black_box(_mm512_add_epi16(ones_i16, ones_i16));
    black_box(_mm512_sub_epi16(ones_i16, ones_i16));
    black_box(_mm512_adds_epi16(ones_i16, ones_i16));
    black_box(_mm512_adds_epu16(ones_i16, ones_i16));
    black_box(_mm512_subs_epi16(ones_i16, ones_i16));
    black_box(_mm512_subs_epu16(ones_i16, ones_i16));
    black_box(_mm512_min_epi16(ones_i16, ones_i16));
    black_box(_mm512_max_epi16(ones_i16, ones_i16));
    black_box(_mm512_min_epu16(ones_i16, ones_i16));
    black_box(_mm512_max_epu16(ones_i16, ones_i16));
    black_box(_mm512_abs_epi16(ones_i16));
    black_box(_mm512_avg_epu16(ones_i16, ones_i16));
    black_box(_mm512_mulhi_epi16(ones_i16, ones_i16));
    black_box(_mm512_mulhi_epu16(ones_i16, ones_i16));
    black_box(_mm512_mullo_epi16(ones_i16, ones_i16));
    black_box(_mm512_madd_epi16(ones_i16, ones_i16));
    black_box(_mm512_maddubs_epi16(ones_i8, ones_i8));

    // Shifts
    black_box(_mm512_slli_epi16::<1>(ones_i16));
    black_box(_mm512_srli_epi16::<1>(ones_i16));
    black_box(_mm512_srai_epi16::<1>(ones_i16));
    black_box(_mm512_sllv_epi16(ones_i16, ones_i16));
    black_box(_mm512_srlv_epi16(ones_i16, ones_i16));
    black_box(_mm512_srav_epi16(ones_i16, ones_i16));

    // Pack
    black_box(_mm512_packs_epi16(ones_i16, ones_i16));
    black_box(_mm512_packus_epi16(ones_i16, ones_i16));
    black_box(_mm512_packs_epi32(_mm512_set1_epi32(1), _mm512_set1_epi32(1)));
    black_box(_mm512_packus_epi32(_mm512_set1_epi32(1), _mm512_set1_epi32(1)));

    // Unpack
    black_box(_mm512_unpackhi_epi8(ones_i8, ones_i8));
    black_box(_mm512_unpacklo_epi8(ones_i8, ones_i8));
    black_box(_mm512_unpackhi_epi16(ones_i16, ones_i16));
    black_box(_mm512_unpacklo_epi16(ones_i16, ones_i16));

    // Shuffle
    black_box(_mm512_shuffle_epi8(ones_i8, zero_i));

    // Comparisons
    black_box(_mm512_cmpeq_epi8_mask(ones_i8, ones_i8));
    black_box(_mm512_cmpeq_epi16_mask(ones_i16, ones_i16));
    black_box(_mm512_cmplt_epi8_mask(ones_i8, ones_i8));
    black_box(_mm512_cmplt_epi16_mask(ones_i16, ones_i16));

    // Masked operations
    black_box(_mm512_mask_add_epi8(ones_i8, mask64, ones_i8, ones_i8));
    black_box(_mm512_maskz_add_epi8(mask64, ones_i8, ones_i8));
    black_box(_mm512_mask_add_epi16(ones_i16, mask32, ones_i16, ones_i16));
    black_box(_mm512_maskz_add_epi16(mask32, ones_i16, ones_i16));

    // Permute
    black_box(_mm512_permutexvar_epi16(ones_i16, ones_i16));

    // Alignr
    black_box(_mm512_alignr_epi8::<1>(ones_i8, ones_i8));

    // Blend
    black_box(_mm512_mask_blend_epi8(mask64, ones_i8, ones_i8));
    black_box(_mm512_mask_blend_epi16(mask32, ones_i16, ones_i16));

    // Move mask
    black_box(_mm512_movepi8_mask(ones_i8));
    black_box(_mm512_movepi16_mask(ones_i16));
    black_box(_mm512_movm_epi8(mask64));
    black_box(_mm512_movm_epi16(mask32));
}

// =============================================================================
// AVX-512BW+VL - Byte/Word 128/256-bit operations
// =============================================================================

#[arcane]
fn exercise_avx512bw_vl(token: Avx512ModernToken) {
    let v256_i8 = _mm256_set1_epi8(1);
    let v256_i16 = _mm256_set1_epi16(1);
    let v128_i8 = _mm_set1_epi8(1);
    let v128_i16 = _mm_set1_epi16(1);
    let mask16: __mmask16 = 0xFFFF;
    let mask32: __mmask32 = 0xFFFFFFFF;

    // 256-bit masked byte operations
    black_box(_mm256_mask_add_epi8(v256_i8, mask32, v256_i8, v256_i8));
    black_box(_mm256_maskz_add_epi8(mask32, v256_i8, v256_i8));
    black_box(_mm256_mask_add_epi16(v256_i16, mask16, v256_i16, v256_i16));
    black_box(_mm256_maskz_add_epi16(mask16, v256_i16, v256_i16));

    // 256-bit shuffle/permute
    black_box(_mm256_permutexvar_epi16(v256_i16, v256_i16));

    // 128-bit masked byte operations
    black_box(_mm_mask_add_epi8(v128_i8, mask16, v128_i8, v128_i8));
    black_box(_mm_maskz_add_epi8(mask16, v128_i8, v128_i8));
    black_box(_mm_mask_add_epi16(v128_i16, 0xFF, v128_i16, v128_i16));
    black_box(_mm_maskz_add_epi16(0xFF, v128_i16, v128_i16));
}

// =============================================================================
// AVX-512DQ - Double/Quad operations
// =============================================================================

#[arcane]
fn exercise_avx512dq(token: Avx512ModernToken) {
    let ones_ps = _mm512_set1_ps(1.0);
    let ones_pd = _mm512_set1_pd(1.0);
    let ones_i64 = _mm512_set1_epi64(1);
    let mask8: __mmask8 = 0xFF;
    let mask16: __mmask16 = 0xFFFF;

    // Integer multiply
    black_box(_mm512_mullox_epi64(ones_i64, ones_i64));

    // Float <-> int64 conversions
    black_box(_mm512_cvtepi64_ps(ones_i64));
    black_box(_mm512_cvtepi64_pd(ones_i64));
    black_box(_mm512_cvtepu64_ps(ones_i64));
    black_box(_mm512_cvtepu64_pd(ones_i64));
    black_box(_mm512_cvttpd_epi64(ones_pd));
    black_box(_mm512_cvttpd_epu64(ones_pd));
    black_box(_mm512_cvttps_epi64(_mm256_set1_ps(1.0)));
    black_box(_mm512_cvttps_epu64(_mm256_set1_ps(1.0)));

    // Bitwise float operations
    black_box(_mm512_and_ps(ones_ps, ones_ps));
    black_box(_mm512_and_pd(ones_pd, ones_pd));
    black_box(_mm512_andnot_ps(ones_ps, ones_ps));
    black_box(_mm512_andnot_pd(ones_pd, ones_pd));
    black_box(_mm512_or_ps(ones_ps, ones_ps));
    black_box(_mm512_or_pd(ones_pd, ones_pd));
    black_box(_mm512_xor_ps(ones_ps, ones_ps));
    black_box(_mm512_xor_pd(ones_pd, ones_pd));

    // Range
    black_box(_mm512_range_ps::<0>(ones_ps, ones_ps));
    black_box(_mm512_range_pd::<0>(ones_pd, ones_pd));

    // Reduce
    black_box(_mm512_reduce_ps::<0>(ones_ps));
    black_box(_mm512_reduce_pd::<0>(ones_pd));

    // Insert/extract
    black_box(_mm512_insertf32x8::<0>(ones_ps, _mm256_set1_ps(1.0)));
    black_box(_mm512_insertf64x2::<0>(ones_pd, _mm_set1_pd(1.0)));
    black_box(_mm512_inserti32x8::<0>(_mm512_set1_epi32(1), _mm256_set1_epi32(1)));
    black_box(_mm512_inserti64x2::<0>(ones_i64, _mm_set1_epi64x(1)));
    black_box(_mm512_extractf32x8_ps::<0>(ones_ps));
    black_box(_mm512_extractf64x2_pd::<0>(ones_pd));
    black_box(_mm512_extracti32x8_epi32::<0>(_mm512_set1_epi32(1)));
    black_box(_mm512_extracti64x2_epi64::<0>(ones_i64));

    // Broadcast
    black_box(_mm512_broadcast_f32x8(_mm256_set1_ps(1.0)));
    black_box(_mm512_broadcast_f64x2(_mm_set1_pd(1.0)));
    black_box(_mm512_broadcast_i32x8(_mm256_set1_epi32(1)));
    black_box(_mm512_broadcast_i64x2(_mm_set1_epi64x(1)));

    // FP class
    black_box(_mm512_fpclass_ps_mask::<0>(ones_ps));
    black_box(_mm512_fpclass_pd_mask::<0>(ones_pd));

    // Masked operations
    black_box(_mm512_mask_and_ps(ones_ps, mask16, ones_ps, ones_ps));
    black_box(_mm512_maskz_and_ps(mask16, ones_ps, ones_ps));
    black_box(_mm512_mask_and_pd(ones_pd, mask8, ones_pd, ones_pd));
    black_box(_mm512_maskz_and_pd(mask8, ones_pd, ones_pd));
}

// =============================================================================
// AVX-512DQ+VL - Double/Quad 128/256-bit operations
// =============================================================================

#[arcane]
fn exercise_avx512dq_vl(token: Avx512ModernToken) {
    let v256_ps = _mm256_set1_ps(1.0);
    let v256_pd = _mm256_set1_pd(1.0);
    let v256_i64 = _mm256_set1_epi64x(1);
    let v128_ps = _mm_set1_ps(1.0);
    let v128_pd = _mm_set1_pd(1.0);
    let v128_i64 = _mm_set1_epi64x(1);
    let mask4: __mmask8 = 0x0F;
    let mask8: __mmask8 = 0xFF;

    // 256-bit operations
    black_box(_mm256_and_ps(v256_ps, v256_ps)); // SSE
    black_box(_mm256_mask_and_ps(v256_ps, mask8, v256_ps, v256_ps));
    black_box(_mm256_maskz_and_ps(mask8, v256_ps, v256_ps));

    // 256-bit conversions
    black_box(_mm256_cvtepi64_ps(v256_i64));
    black_box(_mm256_cvtepi64_pd(v256_i64));

    // 256-bit fp class
    black_box(_mm256_fpclass_ps_mask::<0>(v256_ps));
    black_box(_mm256_fpclass_pd_mask::<0>(v256_pd));

    // 128-bit operations
    black_box(_mm_mask_and_ps(v128_ps, mask4, v128_ps, v128_ps));
    black_box(_mm_maskz_and_ps(mask4, v128_ps, v128_ps));
    black_box(_mm_cvtepi64_ps(v128_i64));
    black_box(_mm_cvtepi64_pd(v128_i64));
    black_box(_mm_fpclass_ps_mask::<0>(v128_ps));
    black_box(_mm_fpclass_pd_mask::<0>(v128_pd));
}

// =============================================================================
// AVX-512CD - Conflict Detection
// =============================================================================

#[arcane]
fn exercise_avx512cd(token: Avx512ModernToken) {
    let ones_i32 = _mm512_set1_epi32(1);
    let ones_i64 = _mm512_set1_epi64(1);
    let mask8: __mmask8 = 0xFF;
    let mask16: __mmask16 = 0xFFFF;

    // Conflict detection
    black_box(_mm512_conflict_epi32(ones_i32));
    black_box(_mm512_conflict_epi64(ones_i64));

    // Leading zero count
    black_box(_mm512_lzcnt_epi32(ones_i32));
    black_box(_mm512_lzcnt_epi64(ones_i64));

    // Masked conflict detection
    black_box(_mm512_mask_conflict_epi32(ones_i32, mask16, ones_i32));
    black_box(_mm512_maskz_conflict_epi32(mask16, ones_i32));
    black_box(_mm512_mask_conflict_epi64(ones_i64, mask8, ones_i64));
    black_box(_mm512_maskz_conflict_epi64(mask8, ones_i64));

    // Masked lzcnt
    black_box(_mm512_mask_lzcnt_epi32(ones_i32, mask16, ones_i32));
    black_box(_mm512_maskz_lzcnt_epi32(mask16, ones_i32));
    black_box(_mm512_mask_lzcnt_epi64(ones_i64, mask8, ones_i64));
    black_box(_mm512_maskz_lzcnt_epi64(mask8, ones_i64));

    // Broadcast mask
    black_box(_mm512_broadcastmb_epi64(mask8));
    black_box(_mm512_broadcastmw_epi32(mask16));
}

// =============================================================================
// AVX-512CD+VL - Conflict Detection 128/256-bit
// =============================================================================

#[arcane]
fn exercise_avx512cd_vl(token: Avx512ModernToken) {
    let v256_i32 = _mm256_set1_epi32(1);
    let v256_i64 = _mm256_set1_epi64x(1);
    let v128_i32 = _mm_set1_epi32(1);
    let v128_i64 = _mm_set1_epi64x(1);
    let mask4: __mmask8 = 0x0F;
    let mask8: __mmask8 = 0xFF;

    // 256-bit conflict detection
    black_box(_mm256_conflict_epi32(v256_i32));
    black_box(_mm256_conflict_epi64(v256_i64));
    black_box(_mm256_lzcnt_epi32(v256_i32));
    black_box(_mm256_lzcnt_epi64(v256_i64));

    // 256-bit masked
    black_box(_mm256_mask_conflict_epi32(v256_i32, mask8, v256_i32));
    black_box(_mm256_maskz_conflict_epi32(mask8, v256_i32));
    black_box(_mm256_mask_lzcnt_epi32(v256_i32, mask8, v256_i32));
    black_box(_mm256_maskz_lzcnt_epi32(mask8, v256_i32));

    // 128-bit conflict detection
    black_box(_mm_conflict_epi32(v128_i32));
    black_box(_mm_conflict_epi64(v128_i64));
    black_box(_mm_lzcnt_epi32(v128_i32));
    black_box(_mm_lzcnt_epi64(v128_i64));

    // 128-bit masked
    black_box(_mm_mask_conflict_epi32(v128_i32, mask4, v128_i32));
    black_box(_mm_maskz_conflict_epi32(mask4, v128_i32));
}

// =============================================================================
// AVX-512VBMI - Vector Byte Manipulation Instructions
// =============================================================================

#[arcane]
fn exercise_avx512vbmi(token: Avx512ModernToken) {
    let ones_i8 = _mm512_set1_epi8(1);
    let zero_i = _mm512_setzero_si512();
    let mask64: __mmask64 = 0xFFFFFFFFFFFFFFFF;

    // Permute bytes
    black_box(_mm512_permutexvar_epi8(zero_i, ones_i8));
    black_box(_mm512_permutex2var_epi8(ones_i8, zero_i, ones_i8));

    // Masked permute
    black_box(_mm512_mask_permutexvar_epi8(ones_i8, mask64, zero_i, ones_i8));
    black_box(_mm512_maskz_permutexvar_epi8(mask64, zero_i, ones_i8));
    black_box(_mm512_mask_permutex2var_epi8(ones_i8, mask64, zero_i, ones_i8));
    black_box(_mm512_maskz_permutex2var_epi8(mask64, ones_i8, zero_i, ones_i8));

    // Multishift
    black_box(_mm512_multishift_epi64_epi8(ones_i8, ones_i8));
    black_box(_mm512_mask_multishift_epi64_epi8(ones_i8, mask64, ones_i8, ones_i8));
    black_box(_mm512_maskz_multishift_epi64_epi8(mask64, ones_i8, ones_i8));
}

// =============================================================================
// AVX-512VBMI2 - Vector Byte Manipulation Instructions 2
// =============================================================================

#[arcane]
fn exercise_avx512vbmi2(token: Avx512ModernToken) {
    let ones_i8 = _mm512_set1_epi8(1);
    let ones_i16 = _mm512_set1_epi16(1);
    let ones_i32 = _mm512_set1_epi32(1);
    let ones_i64 = _mm512_set1_epi64(1);
    let mask16: __mmask16 = 0xFFFF;
    let mask32: __mmask32 = 0xFFFFFFFF;
    let mask64: __mmask64 = 0xFFFFFFFFFFFFFFFF;

    // Compress/expand
    black_box(_mm512_mask_compress_epi8(ones_i8, mask64, ones_i8));
    black_box(_mm512_maskz_compress_epi8(mask64, ones_i8));
    black_box(_mm512_mask_compress_epi16(ones_i16, mask32, ones_i16));
    black_box(_mm512_maskz_compress_epi16(mask32, ones_i16));
    black_box(_mm512_mask_expand_epi8(ones_i8, mask64, ones_i8));
    black_box(_mm512_maskz_expand_epi8(mask64, ones_i8));
    black_box(_mm512_mask_expand_epi16(ones_i16, mask32, ones_i16));
    black_box(_mm512_maskz_expand_epi16(mask32, ones_i16));

    // Concatenate and shift
    black_box(_mm512_shldi_epi16::<1>(ones_i16, ones_i16));
    black_box(_mm512_shldi_epi32::<1>(ones_i32, ones_i32));
    black_box(_mm512_shldi_epi64::<1>(ones_i64, ones_i64));
    black_box(_mm512_shrdi_epi16::<1>(ones_i16, ones_i16));
    black_box(_mm512_shrdi_epi32::<1>(ones_i32, ones_i32));
    black_box(_mm512_shrdi_epi64::<1>(ones_i64, ones_i64));
    black_box(_mm512_shldv_epi16(ones_i16, ones_i16, ones_i16));
    black_box(_mm512_shldv_epi32(ones_i32, ones_i32, ones_i32));
    black_box(_mm512_shldv_epi64(ones_i64, ones_i64, ones_i64));
    black_box(_mm512_shrdv_epi16(ones_i16, ones_i16, ones_i16));
    black_box(_mm512_shrdv_epi32(ones_i32, ones_i32, ones_i32));
    black_box(_mm512_shrdv_epi64(ones_i64, ones_i64, ones_i64));

    // Masked concatenate shift
    black_box(_mm512_mask_shldi_epi16::<1>(ones_i16, mask32, ones_i16, ones_i16));
    black_box(_mm512_maskz_shldi_epi16::<1>(mask32, ones_i16, ones_i16));
}

// =============================================================================
// AVX-512VNNI - Vector Neural Network Instructions
// =============================================================================

#[arcane]
fn exercise_avx512vnni(token: Avx512ModernToken) {
    let ones_i32 = _mm512_set1_epi32(1);
    let ones_i8_as_i32 = _mm512_set1_epi8(1);
    let mask16: __mmask16 = 0xFFFF;

    // Dot product unsigned/signed byte to int32
    black_box(_mm512_dpbusd_epi32(ones_i32, ones_i8_as_i32, ones_i8_as_i32));
    black_box(_mm512_dpbusds_epi32(ones_i32, ones_i8_as_i32, ones_i8_as_i32));

    // Dot product signed word to int32
    let ones_i16_as_i32 = _mm512_set1_epi16(1);
    black_box(_mm512_dpwssd_epi32(ones_i32, ones_i16_as_i32, ones_i16_as_i32));
    black_box(_mm512_dpwssds_epi32(ones_i32, ones_i16_as_i32, ones_i16_as_i32));

    // Masked versions
    black_box(_mm512_mask_dpbusd_epi32(ones_i32, mask16, ones_i8_as_i32, ones_i8_as_i32));
    black_box(_mm512_maskz_dpbusd_epi32(mask16, ones_i32, ones_i8_as_i32, ones_i8_as_i32));
    black_box(_mm512_mask_dpbusds_epi32(ones_i32, mask16, ones_i8_as_i32, ones_i8_as_i32));
    black_box(_mm512_maskz_dpbusds_epi32(mask16, ones_i32, ones_i8_as_i32, ones_i8_as_i32));
    black_box(_mm512_mask_dpwssd_epi32(ones_i32, mask16, ones_i16_as_i32, ones_i16_as_i32));
    black_box(_mm512_maskz_dpwssd_epi32(mask16, ones_i32, ones_i16_as_i32, ones_i16_as_i32));
    black_box(_mm512_mask_dpwssds_epi32(ones_i32, mask16, ones_i16_as_i32, ones_i16_as_i32));
    black_box(_mm512_maskz_dpwssds_epi32(mask16, ones_i32, ones_i16_as_i32, ones_i16_as_i32));
}

// =============================================================================
// AVX-512BITALG - Bit Algorithms
// =============================================================================

#[arcane]
fn exercise_avx512bitalg(token: Avx512ModernToken) {
    let ones_i8 = _mm512_set1_epi8(1);
    let ones_i16 = _mm512_set1_epi16(1);
    let mask32: __mmask32 = 0xFFFFFFFF;
    let mask64: __mmask64 = 0xFFFFFFFFFFFFFFFF;

    // Population count
    black_box(_mm512_popcnt_epi8(ones_i8));
    black_box(_mm512_popcnt_epi16(ones_i16));

    // Masked popcnt
    black_box(_mm512_mask_popcnt_epi8(ones_i8, mask64, ones_i8));
    black_box(_mm512_maskz_popcnt_epi8(mask64, ones_i8));
    black_box(_mm512_mask_popcnt_epi16(ones_i16, mask32, ones_i16));
    black_box(_mm512_maskz_popcnt_epi16(mask32, ones_i16));

    // Bit gather/scatter in each byte
    black_box(_mm512_bitshuffle_epi64_mask(ones_i8, ones_i8));
    black_box(_mm512_mask_bitshuffle_epi64_mask(mask64, ones_i8, ones_i8));
}

// =============================================================================
// AVX-512BF16 - BFloat16 operations
// =============================================================================

#[arcane]
fn exercise_avx512bf16(token: Avx512ModernToken) {
    let ones_ps = _mm512_set1_ps(1.0);
    let ones_ps_256 = _mm256_set1_ps(1.0);
    let mask16: __mmask16 = 0xFFFF;

    // Convert float to BF16 - returns __m256bh
    let bf16_256: __m256bh = _mm512_cvtneps_pbh(ones_ps);
    black_box(bf16_256);

    let bf16_128: __m128bh = _mm256_cvtneps_pbh(ones_ps_256);
    black_box(bf16_128);

    // Convert BF16 to float - takes __m256bh returns __m512
    black_box(_mm512_cvtpbh_ps(bf16_256));

    // BF16 dot product - creates zero bf16 vectors properly
    let zero_bf16_512: __m512bh = _mm512_cvtne2ps_pbh(ones_ps, ones_ps);
    black_box(_mm512_dpbf16_ps(ones_ps, zero_bf16_512, zero_bf16_512));

    // Masked operations
    black_box(_mm512_mask_cvtneps_pbh(bf16_256, mask16, ones_ps));
    black_box(_mm512_maskz_cvtneps_pbh(mask16, ones_ps));
    black_box(_mm512_mask_dpbf16_ps(ones_ps, mask16, zero_bf16_512, zero_bf16_512));
    black_box(_mm512_maskz_dpbf16_ps(mask16, ones_ps, zero_bf16_512, zero_bf16_512));
}

// =============================================================================
// AVX-512IFMA - Integer FMA
// =============================================================================

#[arcane]
fn exercise_avx512ifma(token: Avx512ModernToken) {
    let ones_i64 = _mm512_set1_epi64(1);
    let mask8: __mmask8 = 0xFF;

    // 52-bit integer FMA (multiply low/high 52 bits, add)
    black_box(_mm512_madd52lo_epu64(ones_i64, ones_i64, ones_i64));
    black_box(_mm512_madd52hi_epu64(ones_i64, ones_i64, ones_i64));

    // Masked versions
    black_box(_mm512_mask_madd52lo_epu64(ones_i64, mask8, ones_i64, ones_i64));
    black_box(_mm512_maskz_madd52lo_epu64(mask8, ones_i64, ones_i64, ones_i64));
    black_box(_mm512_mask_madd52hi_epu64(ones_i64, mask8, ones_i64, ones_i64));
    black_box(_mm512_maskz_madd52hi_epu64(mask8, ones_i64, ones_i64, ones_i64));

    // VL variants
    let v256_i64 = _mm256_set1_epi64x(1);
    let v128_i64 = _mm_set1_epi64x(1);
    let mask4: __mmask8 = 0x0F;
    let mask2: __mmask8 = 0x03;

    black_box(_mm256_madd52lo_epu64(v256_i64, v256_i64, v256_i64));
    black_box(_mm256_madd52hi_epu64(v256_i64, v256_i64, v256_i64));
    black_box(_mm256_mask_madd52lo_epu64(v256_i64, mask4, v256_i64, v256_i64));
    black_box(_mm256_maskz_madd52lo_epu64(mask4, v256_i64, v256_i64, v256_i64));

    black_box(_mm_madd52lo_epu64(v128_i64, v128_i64, v128_i64));
    black_box(_mm_madd52hi_epu64(v128_i64, v128_i64, v128_i64));
    black_box(_mm_mask_madd52lo_epu64(v128_i64, mask2, v128_i64, v128_i64));
    black_box(_mm_maskz_madd52lo_epu64(mask2, v128_i64, v128_i64, v128_i64));
}

// =============================================================================
// GFNI - Galois Field New Instructions
// =============================================================================

#[arcane]
fn exercise_gfni(token: Avx512ModernToken) {
    let ones_i8 = _mm512_set1_epi8(1);
    let v256_i8 = _mm256_set1_epi8(1);
    let v128_i8 = _mm_set1_epi8(1);
    let mask64: __mmask64 = 0xFFFFFFFFFFFFFFFF;
    let mask32: __mmask32 = 0xFFFFFFFF;
    let mask16: __mmask16 = 0xFFFF;

    // 512-bit GFNI
    black_box(_mm512_gf2p8affine_epi64_epi8::<0>(ones_i8, ones_i8));
    black_box(_mm512_gf2p8affineinv_epi64_epi8::<0>(ones_i8, ones_i8));
    black_box(_mm512_gf2p8mul_epi8(ones_i8, ones_i8));

    // 512-bit masked
    black_box(_mm512_mask_gf2p8affine_epi64_epi8::<0>(ones_i8, mask64, ones_i8, ones_i8));
    black_box(_mm512_maskz_gf2p8affine_epi64_epi8::<0>(mask64, ones_i8, ones_i8));
    black_box(_mm512_mask_gf2p8mul_epi8(ones_i8, mask64, ones_i8, ones_i8));
    black_box(_mm512_maskz_gf2p8mul_epi8(mask64, ones_i8, ones_i8));

    // 256-bit GFNI
    black_box(_mm256_gf2p8affine_epi64_epi8::<0>(v256_i8, v256_i8));
    black_box(_mm256_gf2p8affineinv_epi64_epi8::<0>(v256_i8, v256_i8));
    black_box(_mm256_gf2p8mul_epi8(v256_i8, v256_i8));

    // 256-bit masked
    black_box(_mm256_mask_gf2p8affine_epi64_epi8::<0>(v256_i8, mask32, v256_i8, v256_i8));
    black_box(_mm256_maskz_gf2p8affine_epi64_epi8::<0>(mask32, v256_i8, v256_i8));

    // 128-bit GFNI
    black_box(_mm_gf2p8affine_epi64_epi8::<0>(v128_i8, v128_i8));
    black_box(_mm_gf2p8affineinv_epi64_epi8::<0>(v128_i8, v128_i8));
    black_box(_mm_gf2p8mul_epi8(v128_i8, v128_i8));

    // 128-bit masked
    black_box(_mm_mask_gf2p8affine_epi64_epi8::<0>(v128_i8, mask16, v128_i8, v128_i8));
    black_box(_mm_maskz_gf2p8affine_epi64_epi8::<0>(mask16, v128_i8, v128_i8));
}

// =============================================================================
// VAES - Vector AES
// =============================================================================

#[arcane]
fn exercise_vaes(token: Avx512ModernToken) {
    let key_512 = _mm512_setzero_si512();
    let data_512 = _mm512_set1_epi8(0x42);
    let key_256 = _mm256_setzero_si256();
    let data_256 = _mm256_set1_epi8(0x42);

    // 512-bit AES
    black_box(_mm512_aesenc_epi128(data_512, key_512));
    black_box(_mm512_aesenclast_epi128(data_512, key_512));
    black_box(_mm512_aesdec_epi128(data_512, key_512));
    black_box(_mm512_aesdeclast_epi128(data_512, key_512));

    // 256-bit AES
    black_box(_mm256_aesenc_epi128(data_256, key_256));
    black_box(_mm256_aesenclast_epi128(data_256, key_256));
    black_box(_mm256_aesdec_epi128(data_256, key_256));
    black_box(_mm256_aesdeclast_epi128(data_256, key_256));
}

// =============================================================================
// VPCLMULQDQ - Vector Carryless Multiply
// =============================================================================

#[arcane]
fn exercise_vpclmulqdq(token: Avx512ModernToken) {
    let data_512 = _mm512_set1_epi64(0x0123456789ABCDEF);
    let data_256 = _mm256_set1_epi64x(0x0123456789ABCDEF);

    // 512-bit carryless multiply
    black_box(_mm512_clmulepi64_epi128::<0x00>(data_512, data_512));
    black_box(_mm512_clmulepi64_epi128::<0x01>(data_512, data_512));
    black_box(_mm512_clmulepi64_epi128::<0x10>(data_512, data_512));
    black_box(_mm512_clmulepi64_epi128::<0x11>(data_512, data_512));

    // 256-bit carryless multiply
    black_box(_mm256_clmulepi64_epi128::<0x00>(data_256, data_256));
    black_box(_mm256_clmulepi64_epi128::<0x01>(data_256, data_256));
    black_box(_mm256_clmulepi64_epi128::<0x10>(data_256, data_256));
    black_box(_mm256_clmulepi64_epi128::<0x11>(data_256, data_256));
}

// =============================================================================
// Safe Memory Operations (archmage::mem) - NO UNSAFE
// =============================================================================

#[cfg(feature = "safe_unaligned_simd")]
#[arcane]
fn exercise_safe_mem_ops(token: Avx512ModernToken) {
    // === Arrays for load/store ===
    let mut arr_f32_16 = [1.0f32; 16];
    let mut arr_f64_8 = [1.0f64; 8];
    let mut arr_f32_8 = [1.0f32; 8];
    let mut arr_f64_4 = [1.0f64; 4];
    let mut arr_f32_4 = [1.0f32; 4];
    let mut arr_f64_2 = [1.0f64; 2];
    let mut arr_i32_16 = [1i32; 16];
    let mut arr_i64_8 = [1i64; 8];
    let mut arr_i32_8 = [1i32; 8];
    let mut arr_i64_4 = [1i64; 4];
    let mut arr_i32_4 = [1i32; 4];
    let mut arr_i64_2 = [1i64; 2];
    let mut arr_i16_32 = [1i16; 32];
    let mut arr_i16_16 = [1i16; 16];
    let mut arr_i16_8 = [1i16; 8];
    let mut arr_i8_64 = [1i8; 64];
    let mut arr_i8_32 = [1i8; 32];
    let mut arr_i8_16 = [1i8; 16];

    // Vectors for store operations
    let vec_512_ps = _mm512_set1_ps(2.0);
    let vec_512_pd = _mm512_set1_pd(2.0);
    let vec_512_i32 = _mm512_set1_epi32(2);
    let vec_512_i64 = _mm512_set1_epi64(2);
    let vec_512_i16 = _mm512_set1_epi16(2);
    let vec_512_i8 = _mm512_set1_epi8(2);
    let vec_256_ps = _mm256_set1_ps(2.0);
    let vec_256_pd = _mm256_set1_pd(2.0);
    let vec_256_i32 = _mm256_set1_epi32(2);
    let vec_256_i64 = _mm256_set1_epi64x(2);
    let vec_256_i16 = _mm256_set1_epi16(2);
    let vec_256_i8 = _mm256_set1_epi8(2);
    let vec_128_ps = _mm_set1_ps(2.0);
    let vec_128_pd = _mm_set1_pd(2.0);
    let vec_128_i32 = _mm_set1_epi32(2);
    let vec_128_i64 = _mm_set1_epi64x(2);
    let vec_128_i16 = _mm_set1_epi16(2);
    let vec_128_i8 = _mm_set1_epi8(2);

    // Masks
    let mask2: __mmask8 = 0x03;
    let mask4: __mmask8 = 0x0F;
    let mask8: __mmask8 = 0xFF;
    let mask16: __mmask16 = 0xFFFF;
    let mask32: __mmask32 = 0xFFFFFFFF;
    let mask64: __mmask64 = 0xFFFFFFFFFFFFFFFF;

    // =========================================================================
    // AVX-512F 512-bit Load/Store
    // =========================================================================

    // Float loads
    black_box(avx512f::_mm512_loadu_ps(token, &arr_f32_16));
    black_box(avx512f::_mm512_mask_loadu_ps(token, vec_512_ps, mask16, &arr_f32_16));
    black_box(avx512f::_mm512_maskz_loadu_ps(token, mask16, &arr_f32_16));

    // Double loads
    black_box(avx512f::_mm512_loadu_pd(token, &arr_f64_8));
    black_box(avx512f::_mm512_mask_loadu_pd(token, vec_512_pd, mask8, &arr_f64_8));
    black_box(avx512f::_mm512_maskz_loadu_pd(token, mask8, &arr_f64_8));

    // Integer loads (epi32/epi64)
    black_box(avx512f::_mm512_loadu_epi32(token, &arr_i32_16));
    black_box(avx512f::_mm512_mask_loadu_epi32(token, vec_512_i32, mask16, &arr_i32_16));
    black_box(avx512f::_mm512_maskz_loadu_epi32(token, mask16, &arr_i32_16));
    black_box(avx512f::_mm512_loadu_epi64(token, &arr_i64_8));
    black_box(avx512f::_mm512_mask_loadu_epi64(token, vec_512_i64, mask8, &arr_i64_8));
    black_box(avx512f::_mm512_maskz_loadu_epi64(token, mask8, &arr_i64_8));
    black_box(avx512f::_mm512_loadu_si512(token, &arr_i32_16));

    // Float stores
    avx512f::_mm512_storeu_ps(token, &mut arr_f32_16, vec_512_ps);
    avx512f::_mm512_mask_storeu_ps(token, &mut arr_f32_16, mask16, vec_512_ps);

    // Double stores
    avx512f::_mm512_storeu_pd(token, &mut arr_f64_8, vec_512_pd);
    avx512f::_mm512_mask_storeu_pd(token, &mut arr_f64_8, mask8, vec_512_pd);

    // Integer stores
    avx512f::_mm512_storeu_epi32(token, &mut arr_i32_16, vec_512_i32);
    avx512f::_mm512_mask_storeu_epi32(token, &mut arr_i32_16, mask16, vec_512_i32);
    avx512f::_mm512_storeu_epi64(token, &mut arr_i64_8, vec_512_i64);
    avx512f::_mm512_mask_storeu_epi64(token, &mut arr_i64_8, mask8, vec_512_i64);
    avx512f::_mm512_storeu_si512(token, &mut arr_i32_16, vec_512_i32);

    // Expand loads (masked load with expansion)
    black_box(avx512f::_mm512_mask_expandloadu_ps(token, vec_512_ps, mask16, &arr_f32_16));
    black_box(avx512f::_mm512_maskz_expandloadu_ps(token, mask16, &arr_f32_16));
    black_box(avx512f::_mm512_mask_expandloadu_pd(token, vec_512_pd, mask8, &arr_f64_8));
    black_box(avx512f::_mm512_maskz_expandloadu_pd(token, mask8, &arr_f64_8));
    black_box(avx512f::_mm512_mask_expandloadu_epi32(token, vec_512_i32, mask16, &arr_i32_16));
    black_box(avx512f::_mm512_maskz_expandloadu_epi32(token, mask16, &arr_i32_16));
    black_box(avx512f::_mm512_mask_expandloadu_epi64(token, vec_512_i64, mask8, &arr_i64_8));
    black_box(avx512f::_mm512_maskz_expandloadu_epi64(token, mask8, &arr_i64_8));

    // Compress stores (masked store with compression)
    avx512f::_mm512_mask_compressstoreu_ps(token, &mut arr_f32_16, mask16, vec_512_ps);
    avx512f::_mm512_mask_compressstoreu_pd(token, &mut arr_f64_8, mask8, vec_512_pd);
    avx512f::_mm512_mask_compressstoreu_epi32(token, &mut arr_i32_16, mask16, vec_512_i32);
    avx512f::_mm512_mask_compressstoreu_epi64(token, &mut arr_i64_8, mask8, vec_512_i64);

    // Convert and store (truncating stores)
    avx512f::_mm512_mask_cvtepi32_storeu_epi16(token, &mut arr_i16_16, mask16, vec_512_i32);
    avx512f::_mm512_mask_cvtepi32_storeu_epi8(token, &mut arr_i8_16, mask16, vec_512_i32);
    avx512f::_mm512_mask_cvtepi64_storeu_epi32(token, &mut arr_i32_8, mask8, vec_512_i64);
    avx512f::_mm512_mask_cvtepi64_storeu_epi16(token, &mut arr_i16_8, mask8, vec_512_i64);
    let mut arr_i8_8: [i8; 8] = [0; 8];
    avx512f::_mm512_mask_cvtepi64_storeu_epi8(token, &mut arr_i8_8, mask8, vec_512_i64);

    // Saturating convert and store
    avx512f::_mm512_mask_cvtsepi32_storeu_epi16(token, &mut arr_i16_16, mask16, vec_512_i32);
    avx512f::_mm512_mask_cvtsepi32_storeu_epi8(token, &mut arr_i8_16, mask16, vec_512_i32);
    avx512f::_mm512_mask_cvtsepi64_storeu_epi32(token, &mut arr_i32_8, mask8, vec_512_i64);
    avx512f::_mm512_mask_cvtsepi64_storeu_epi16(token, &mut arr_i16_8, mask8, vec_512_i64);
    avx512f::_mm512_mask_cvtsepi64_storeu_epi8(token, &mut arr_i8_8, mask8, vec_512_i64);

    // Unsigned saturating convert and store
    avx512f::_mm512_mask_cvtusepi32_storeu_epi16(token, &mut arr_i16_16, mask16, vec_512_i32);
    avx512f::_mm512_mask_cvtusepi32_storeu_epi8(token, &mut arr_i8_16, mask16, vec_512_i32);
    avx512f::_mm512_mask_cvtusepi64_storeu_epi32(token, &mut arr_i32_8, mask8, vec_512_i64);
    avx512f::_mm512_mask_cvtusepi64_storeu_epi16(token, &mut arr_i16_8, mask8, vec_512_i64);
    avx512f::_mm512_mask_cvtusepi64_storeu_epi8(token, &mut arr_i8_8, mask8, vec_512_i64);

    // =========================================================================
    // AVX-512F+VL 256-bit Load/Store
    // =========================================================================

    black_box(avx512f_vl::_mm256_loadu_epi32(token, &arr_i32_8));
    black_box(avx512f_vl::_mm256_mask_loadu_epi32(token, vec_256_i32, mask8, &arr_i32_8));
    black_box(avx512f_vl::_mm256_maskz_loadu_epi32(token, mask8, &arr_i32_8));
    black_box(avx512f_vl::_mm256_loadu_epi64(token, &arr_i64_4));
    black_box(avx512f_vl::_mm256_mask_loadu_epi64(token, vec_256_i64, mask4, &arr_i64_4));
    black_box(avx512f_vl::_mm256_maskz_loadu_epi64(token, mask4, &arr_i64_4));
    black_box(avx512f_vl::_mm256_mask_loadu_ps(token, vec_256_ps, mask8, &arr_f32_8));
    black_box(avx512f_vl::_mm256_maskz_loadu_ps(token, mask8, &arr_f32_8));
    black_box(avx512f_vl::_mm256_mask_loadu_pd(token, vec_256_pd, mask4, &arr_f64_4));
    black_box(avx512f_vl::_mm256_maskz_loadu_pd(token, mask4, &arr_f64_4));

    avx512f_vl::_mm256_storeu_epi32(token, &mut arr_i32_8, vec_256_i32);
    avx512f_vl::_mm256_mask_storeu_epi32(token, &mut arr_i32_8, mask8, vec_256_i32);
    avx512f_vl::_mm256_storeu_epi64(token, &mut arr_i64_4, vec_256_i64);
    avx512f_vl::_mm256_mask_storeu_epi64(token, &mut arr_i64_4, mask4, vec_256_i64);
    avx512f_vl::_mm256_mask_storeu_ps(token, &mut arr_f32_8, mask8, vec_256_ps);
    avx512f_vl::_mm256_mask_storeu_pd(token, &mut arr_f64_4, mask4, vec_256_pd);

    // 256-bit expand loads
    black_box(avx512f_vl::_mm256_mask_expandloadu_ps(token, vec_256_ps, mask8, &arr_f32_8));
    black_box(avx512f_vl::_mm256_maskz_expandloadu_ps(token, mask8, &arr_f32_8));
    black_box(avx512f_vl::_mm256_mask_expandloadu_pd(token, vec_256_pd, mask4, &arr_f64_4));
    black_box(avx512f_vl::_mm256_maskz_expandloadu_pd(token, mask4, &arr_f64_4));
    black_box(avx512f_vl::_mm256_mask_expandloadu_epi32(token, vec_256_i32, mask8, &arr_i32_8));
    black_box(avx512f_vl::_mm256_maskz_expandloadu_epi32(token, mask8, &arr_i32_8));
    black_box(avx512f_vl::_mm256_mask_expandloadu_epi64(token, vec_256_i64, mask4, &arr_i64_4));
    black_box(avx512f_vl::_mm256_maskz_expandloadu_epi64(token, mask4, &arr_i64_4));

    // 256-bit compress stores
    avx512f_vl::_mm256_mask_compressstoreu_ps(token, &mut arr_f32_8, mask8, vec_256_ps);
    avx512f_vl::_mm256_mask_compressstoreu_pd(token, &mut arr_f64_4, mask4, vec_256_pd);
    avx512f_vl::_mm256_mask_compressstoreu_epi32(token, &mut arr_i32_8, mask8, vec_256_i32);
    avx512f_vl::_mm256_mask_compressstoreu_epi64(token, &mut arr_i64_4, mask4, vec_256_i64);

    // =========================================================================
    // AVX-512F+VL 128-bit Load/Store
    // =========================================================================

    black_box(avx512f_vl::_mm_loadu_epi32(token, &arr_i32_4));
    black_box(avx512f_vl::_mm_mask_loadu_epi32(token, vec_128_i32, mask4, &arr_i32_4));
    black_box(avx512f_vl::_mm_maskz_loadu_epi32(token, mask4, &arr_i32_4));
    black_box(avx512f_vl::_mm_loadu_epi64(token, &arr_i64_2));
    black_box(avx512f_vl::_mm_mask_loadu_epi64(token, vec_128_i64, mask2, &arr_i64_2));
    black_box(avx512f_vl::_mm_maskz_loadu_epi64(token, mask2, &arr_i64_2));
    black_box(avx512f_vl::_mm_mask_loadu_ps(token, vec_128_ps, mask4, &arr_f32_4));
    black_box(avx512f_vl::_mm_maskz_loadu_ps(token, mask4, &arr_f32_4));
    black_box(avx512f_vl::_mm_mask_loadu_pd(token, vec_128_pd, mask2, &arr_f64_2));
    black_box(avx512f_vl::_mm_maskz_loadu_pd(token, mask2, &arr_f64_2));

    avx512f_vl::_mm_storeu_epi32(token, &mut arr_i32_4, vec_128_i32);
    avx512f_vl::_mm_mask_storeu_epi32(token, &mut arr_i32_4, mask4, vec_128_i32);
    avx512f_vl::_mm_storeu_epi64(token, &mut arr_i64_2, vec_128_i64);
    avx512f_vl::_mm_mask_storeu_epi64(token, &mut arr_i64_2, mask2, vec_128_i64);
    avx512f_vl::_mm_mask_storeu_ps(token, &mut arr_f32_4, mask4, vec_128_ps);
    avx512f_vl::_mm_mask_storeu_pd(token, &mut arr_f64_2, mask2, vec_128_pd);

    // 128-bit expand loads
    black_box(avx512f_vl::_mm_mask_expandloadu_ps(token, vec_128_ps, mask4, &arr_f32_4));
    black_box(avx512f_vl::_mm_maskz_expandloadu_ps(token, mask4, &arr_f32_4));
    black_box(avx512f_vl::_mm_mask_expandloadu_pd(token, vec_128_pd, mask2, &arr_f64_2));
    black_box(avx512f_vl::_mm_maskz_expandloadu_pd(token, mask2, &arr_f64_2));
    black_box(avx512f_vl::_mm_mask_expandloadu_epi32(token, vec_128_i32, mask4, &arr_i32_4));
    black_box(avx512f_vl::_mm_maskz_expandloadu_epi32(token, mask4, &arr_i32_4));
    black_box(avx512f_vl::_mm_mask_expandloadu_epi64(token, vec_128_i64, mask2, &arr_i64_2));
    black_box(avx512f_vl::_mm_maskz_expandloadu_epi64(token, mask2, &arr_i64_2));

    // 128-bit compress stores
    avx512f_vl::_mm_mask_compressstoreu_ps(token, &mut arr_f32_4, mask4, vec_128_ps);
    avx512f_vl::_mm_mask_compressstoreu_pd(token, &mut arr_f64_2, mask2, vec_128_pd);
    avx512f_vl::_mm_mask_compressstoreu_epi32(token, &mut arr_i32_4, mask4, vec_128_i32);
    avx512f_vl::_mm_mask_compressstoreu_epi64(token, &mut arr_i64_2, mask2, vec_128_i64);

    // =========================================================================
    // AVX-512BW 512-bit Byte/Word Load/Store
    // =========================================================================

    black_box(avx512bw::_mm512_loadu_epi16(token, &arr_i16_32));
    black_box(avx512bw::_mm512_mask_loadu_epi16(token, vec_512_i16, mask32, &arr_i16_32));
    black_box(avx512bw::_mm512_maskz_loadu_epi16(token, mask32, &arr_i16_32));
    black_box(avx512bw::_mm512_loadu_epi8(token, &arr_i8_64));
    black_box(avx512bw::_mm512_mask_loadu_epi8(token, vec_512_i8, mask64, &arr_i8_64));
    black_box(avx512bw::_mm512_maskz_loadu_epi8(token, mask64, &arr_i8_64));

    avx512bw::_mm512_storeu_epi16(token, &mut arr_i16_32, vec_512_i16);
    avx512bw::_mm512_mask_storeu_epi16(token, &mut arr_i16_32, mask32, vec_512_i16);
    avx512bw::_mm512_storeu_epi8(token, &mut arr_i8_64, vec_512_i8);
    avx512bw::_mm512_mask_storeu_epi8(token, &mut arr_i8_64, mask64, vec_512_i8);

    // Convert and store (epi16 -> epi8)
    avx512bw::_mm512_mask_cvtepi16_storeu_epi8(token, &mut arr_i8_32, mask32, vec_512_i16);
    avx512bw::_mm512_mask_cvtsepi16_storeu_epi8(token, &mut arr_i8_32, mask32, vec_512_i16);
    avx512bw::_mm512_mask_cvtusepi16_storeu_epi8(token, &mut arr_i8_32, mask32, vec_512_i16);

    // =========================================================================
    // AVX-512BW+VL 256-bit Byte/Word Load/Store
    // =========================================================================

    black_box(avx512bw_vl::_mm256_loadu_epi16(token, &arr_i16_16));
    black_box(avx512bw_vl::_mm256_mask_loadu_epi16(token, vec_256_i16, mask16, &arr_i16_16));
    black_box(avx512bw_vl::_mm256_maskz_loadu_epi16(token, mask16, &arr_i16_16));
    black_box(avx512bw_vl::_mm256_loadu_epi8(token, &arr_i8_32));
    black_box(avx512bw_vl::_mm256_mask_loadu_epi8(token, vec_256_i8, mask32, &arr_i8_32));
    black_box(avx512bw_vl::_mm256_maskz_loadu_epi8(token, mask32, &arr_i8_32));

    avx512bw_vl::_mm256_storeu_epi16(token, &mut arr_i16_16, vec_256_i16);
    avx512bw_vl::_mm256_mask_storeu_epi16(token, &mut arr_i16_16, mask16, vec_256_i16);
    avx512bw_vl::_mm256_storeu_epi8(token, &mut arr_i8_32, vec_256_i8);
    avx512bw_vl::_mm256_mask_storeu_epi8(token, &mut arr_i8_32, mask32, vec_256_i8);

    avx512bw_vl::_mm256_mask_cvtepi16_storeu_epi8(token, &mut arr_i8_16, mask16, vec_256_i16);
    avx512bw_vl::_mm256_mask_cvtsepi16_storeu_epi8(token, &mut arr_i8_16, mask16, vec_256_i16);
    avx512bw_vl::_mm256_mask_cvtusepi16_storeu_epi8(token, &mut arr_i8_16, mask16, vec_256_i16);

    // =========================================================================
    // AVX-512BW+VL 128-bit Byte/Word Load/Store
    // =========================================================================

    black_box(avx512bw_vl::_mm_loadu_epi16(token, &arr_i16_8));
    black_box(avx512bw_vl::_mm_mask_loadu_epi16(token, vec_128_i16, mask8, &arr_i16_8));
    black_box(avx512bw_vl::_mm_maskz_loadu_epi16(token, mask8, &arr_i16_8));
    black_box(avx512bw_vl::_mm_loadu_epi8(token, &arr_i8_16));
    black_box(avx512bw_vl::_mm_mask_loadu_epi8(token, vec_128_i8, mask16, &arr_i8_16));
    black_box(avx512bw_vl::_mm_maskz_loadu_epi8(token, mask16, &arr_i8_16));

    avx512bw_vl::_mm_storeu_epi16(token, &mut arr_i16_8, vec_128_i16);
    avx512bw_vl::_mm_mask_storeu_epi16(token, &mut arr_i16_8, mask8, vec_128_i16);
    avx512bw_vl::_mm_storeu_epi8(token, &mut arr_i8_16, vec_128_i8);
    avx512bw_vl::_mm_mask_storeu_epi8(token, &mut arr_i8_16, mask16, vec_128_i8);

    avx512bw_vl::_mm_mask_cvtepi16_storeu_epi8(token, &mut arr_i8_8, mask8, vec_128_i16);
    avx512bw_vl::_mm_mask_cvtsepi16_storeu_epi8(token, &mut arr_i8_8, mask8, vec_128_i16);
    avx512bw_vl::_mm_mask_cvtusepi16_storeu_epi8(token, &mut arr_i8_8, mask8, vec_128_i16);

    // =========================================================================
    // AVX-512VBMI2 512-bit Byte/Word Expand/Compress
    // =========================================================================

    black_box(avx512vbmi2::_mm512_mask_expandloadu_epi16(token, vec_512_i16, mask32, &arr_i16_32));
    black_box(avx512vbmi2::_mm512_maskz_expandloadu_epi16(token, mask32, &arr_i16_32));
    black_box(avx512vbmi2::_mm512_mask_expandloadu_epi8(token, vec_512_i8, mask64, &arr_i8_64));
    black_box(avx512vbmi2::_mm512_maskz_expandloadu_epi8(token, mask64, &arr_i8_64));

    avx512vbmi2::_mm512_mask_compressstoreu_epi16(token, &mut arr_i16_32, mask32, vec_512_i16);
    avx512vbmi2::_mm512_mask_compressstoreu_epi8(token, &mut arr_i8_64, mask64, vec_512_i8);

    // =========================================================================
    // AVX-512VBMI2+VL 256-bit Byte/Word Expand/Compress
    // =========================================================================

    black_box(avx512vbmi2_vl::_mm256_mask_expandloadu_epi16(token, vec_256_i16, mask16, &arr_i16_16));
    black_box(avx512vbmi2_vl::_mm256_maskz_expandloadu_epi16(token, mask16, &arr_i16_16));
    black_box(avx512vbmi2_vl::_mm256_mask_expandloadu_epi8(token, vec_256_i8, mask32, &arr_i8_32));
    black_box(avx512vbmi2_vl::_mm256_maskz_expandloadu_epi8(token, mask32, &arr_i8_32));

    avx512vbmi2_vl::_mm256_mask_compressstoreu_epi16(token, &mut arr_i16_16, mask16, vec_256_i16);
    avx512vbmi2_vl::_mm256_mask_compressstoreu_epi8(token, &mut arr_i8_32, mask32, vec_256_i8);

    // =========================================================================
    // AVX-512VBMI2+VL 128-bit Byte/Word Expand/Compress
    // =========================================================================

    black_box(avx512vbmi2_vl::_mm_mask_expandloadu_epi16(token, vec_128_i16, mask8, &arr_i16_8));
    black_box(avx512vbmi2_vl::_mm_maskz_expandloadu_epi16(token, mask8, &arr_i16_8));
    black_box(avx512vbmi2_vl::_mm_mask_expandloadu_epi8(token, vec_128_i8, mask16, &arr_i8_16));
    black_box(avx512vbmi2_vl::_mm_maskz_expandloadu_epi8(token, mask16, &arr_i8_16));

    avx512vbmi2_vl::_mm_mask_compressstoreu_epi16(token, &mut arr_i16_8, mask8, vec_128_i16);
    avx512vbmi2_vl::_mm_mask_compressstoreu_epi8(token, &mut arr_i8_16, mask16, vec_128_i8);
}

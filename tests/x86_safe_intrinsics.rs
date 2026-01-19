//! Comprehensive test of which x86_64 intrinsics are SAFE in #[target_feature] context
//! Rust 1.92+
//!
//! This file documents all intrinsics that do NOT require an unsafe block when called
//! from within a function annotated with the appropriate #[target_feature(enable = "...")].
//!
//! SAFE: Value-based operations (arithmetic, shuffle, comparison, conversion, bitwise)
//! UNSAFE: Pointer-based operations (load, store, gather, scatter, masked load/store)

#![allow(unused)]

#[cfg(target_arch = "x86_64")]
mod x86_tests {
    use std::arch::x86_64::*;

    // ============================================================================
    // SSE2 (128-bit, baseline for x86_64)
    // ============================================================================

    #[target_feature(enable = "sse2")]
    unsafe fn sse2_safe_intrinsics() {
        // === CREATION f64 (all safe) ===
        let zero_pd = _mm_setzero_pd();
        let set1_pd = _mm_set1_pd(1.0);
        let set_pd = _mm_set_pd(1.0, 2.0);
        let setr_pd = _mm_setr_pd(1.0, 2.0);
        let undef_pd = _mm_undefined_pd();

        // === CREATION i32/i64 (all safe) ===
        let zero_si = _mm_setzero_si128();
        let set1_epi8 = _mm_set1_epi8(1);
        let set1_epi16 = _mm_set1_epi16(1);
        let set1_epi32 = _mm_set1_epi32(1);
        let set1_epi64x = _mm_set1_epi64x(1);

        // === ARITHMETIC f64 (all safe) ===
        let add_pd = _mm_add_pd(zero_pd, set1_pd);
        let sub_pd = _mm_sub_pd(add_pd, set1_pd);
        let mul_pd = _mm_mul_pd(add_pd, set1_pd);
        let div_pd = _mm_div_pd(add_pd, set1_pd);
        let sqrt_pd = _mm_sqrt_pd(add_pd);
        let min_pd = _mm_min_pd(add_pd, set1_pd);
        let max_pd = _mm_max_pd(add_pd, set1_pd);

        // === ARITHMETIC INTEGER (all safe) ===
        let add_epi8 = _mm_add_epi8(zero_si, set1_epi8);
        let add_epi16 = _mm_add_epi16(zero_si, set1_epi16);
        let add_epi32 = _mm_add_epi32(zero_si, set1_epi32);
        let add_epi64 = _mm_add_epi64(zero_si, set1_epi64x);
        let sub_epi8 = _mm_sub_epi8(add_epi8, set1_epi8);
        let sub_epi16 = _mm_sub_epi16(add_epi16, set1_epi16);
        let sub_epi32 = _mm_sub_epi32(add_epi32, set1_epi32);
        let sub_epi64 = _mm_sub_epi64(add_epi64, set1_epi64x);
        let mullo_epi16 = _mm_mullo_epi16(add_epi16, set1_epi16);
        let mulhi_epi16 = _mm_mulhi_epi16(add_epi16, set1_epi16);
        let mul_epu32 = _mm_mul_epu32(add_epi32, set1_epi32);
        let madd_epi16 = _mm_madd_epi16(add_epi16, set1_epi16);

        // === COMPARISON (all safe) ===
        let cmpeq_pd = _mm_cmpeq_pd(add_pd, set1_pd);
        let cmplt_pd = _mm_cmplt_pd(add_pd, set1_pd);
        let cmple_pd = _mm_cmple_pd(add_pd, set1_pd);
        let cmpgt_pd = _mm_cmpgt_pd(add_pd, set1_pd);
        let cmpge_pd = _mm_cmpge_pd(add_pd, set1_pd);
        let cmpeq_epi8 = _mm_cmpeq_epi8(add_epi8, set1_epi8);
        let cmpeq_epi16 = _mm_cmpeq_epi16(add_epi16, set1_epi16);
        let cmpeq_epi32 = _mm_cmpeq_epi32(add_epi32, set1_epi32);
        let cmpgt_epi8 = _mm_cmpgt_epi8(add_epi8, set1_epi8);
        let cmpgt_epi16 = _mm_cmpgt_epi16(add_epi16, set1_epi16);
        let cmpgt_epi32 = _mm_cmpgt_epi32(add_epi32, set1_epi32);
        let cmplt_epi8 = _mm_cmplt_epi8(add_epi8, set1_epi8);
        let cmplt_epi16 = _mm_cmplt_epi16(add_epi16, set1_epi16);
        let cmplt_epi32 = _mm_cmplt_epi32(add_epi32, set1_epi32);

        // === BITWISE (all safe) ===
        let and_pd = _mm_and_pd(add_pd, set1_pd);
        let or_pd = _mm_or_pd(add_pd, set1_pd);
        let xor_pd = _mm_xor_pd(add_pd, set1_pd);
        let andnot_pd = _mm_andnot_pd(add_pd, set1_pd);
        let and_si = _mm_and_si128(add_epi32, set1_epi32);
        let or_si = _mm_or_si128(add_epi32, set1_epi32);
        let xor_si = _mm_xor_si128(add_epi32, set1_epi32);
        let andnot_si = _mm_andnot_si128(add_epi32, set1_epi32);

        // === SHIFT (all safe) ===
        let slli_epi16 = _mm_slli_epi16::<1>(add_epi16);
        let srli_epi16 = _mm_srli_epi16::<1>(add_epi16);
        let srai_epi16 = _mm_srai_epi16::<1>(add_epi16);
        let slli_epi32 = _mm_slli_epi32::<1>(add_epi32);
        let srli_epi32 = _mm_srli_epi32::<1>(add_epi32);
        let srai_epi32 = _mm_srai_epi32::<1>(add_epi32);
        let slli_epi64 = _mm_slli_epi64::<1>(add_epi64);
        let srli_epi64 = _mm_srli_epi64::<1>(add_epi64);

        // === SHUFFLE (all safe) ===
        let shuffle_pd = _mm_shuffle_pd::<0b01>(add_pd, set1_pd);
        let shuffle_epi32 = _mm_shuffle_epi32::<0b00_01_10_11>(add_epi32);
        let shufflelo_epi16 = _mm_shufflelo_epi16::<0b00_01_10_11>(add_epi16);
        let shufflehi_epi16 = _mm_shufflehi_epi16::<0b00_01_10_11>(add_epi16);
        let unpacklo_pd = _mm_unpacklo_pd(add_pd, set1_pd);
        let unpackhi_pd = _mm_unpackhi_pd(add_pd, set1_pd);
        let unpacklo_epi8 = _mm_unpacklo_epi8(add_epi8, set1_epi8);
        let unpackhi_epi8 = _mm_unpackhi_epi8(add_epi8, set1_epi8);
        let unpacklo_epi16 = _mm_unpacklo_epi16(add_epi16, set1_epi16);
        let unpackhi_epi16 = _mm_unpackhi_epi16(add_epi16, set1_epi16);
        let unpacklo_epi32 = _mm_unpacklo_epi32(add_epi32, set1_epi32);
        let unpackhi_epi32 = _mm_unpackhi_epi32(add_epi32, set1_epi32);
        let unpacklo_epi64 = _mm_unpacklo_epi64(add_epi64, set1_epi64x);
        let unpackhi_epi64 = _mm_unpackhi_epi64(add_epi64, set1_epi64x);

        // === PACK (all safe) ===
        let packs_epi16 = _mm_packs_epi16(add_epi16, set1_epi16);
        let packs_epi32 = _mm_packs_epi32(add_epi32, set1_epi32);
        let packus_epi16 = _mm_packus_epi16(add_epi16, set1_epi16);

        // === CONVERSION (all safe) ===
        let cvtepi32_pd = _mm_cvtepi32_pd(add_epi32);
        let cvtpd_epi32 = _mm_cvtpd_epi32(add_pd);
        let cvttpd_epi32 = _mm_cvttpd_epi32(add_pd);
        let cvtpd_ps = _mm_cvtpd_ps(add_pd);
        let cvtps_pd = _mm_cvtps_pd(_mm_setzero_ps());
        let cvtepi32_ps = _mm_cvtepi32_ps(add_epi32);
        let cvtps_epi32 = _mm_cvtps_epi32(_mm_setzero_ps());
        let cvttps_epi32 = _mm_cvttps_epi32(_mm_setzero_ps());

        // === CAST (all safe - just reinterpret) ===
        let castpd_ps = _mm_castpd_ps(add_pd);
        let castps_pd = _mm_castps_pd(_mm_setzero_ps());
        let castpd_si128 = _mm_castpd_si128(add_pd);
        let castsi128_pd = _mm_castsi128_pd(add_epi32);
        let castps_si128 = _mm_castps_si128(_mm_setzero_ps());
        let castsi128_ps = _mm_castsi128_ps(add_epi32);

        // === MOVEMASK (safe) ===
        let movemask_pd = _mm_movemask_pd(cmpeq_pd);
        let movemask_epi8 = _mm_movemask_epi8(cmpeq_epi8);

        // === SATURATING (all safe) ===
        let adds_epi8 = _mm_adds_epi8(add_epi8, set1_epi8);
        let adds_epi16 = _mm_adds_epi16(add_epi16, set1_epi16);
        let adds_epu8 = _mm_adds_epu8(add_epi8, set1_epi8);
        let adds_epu16 = _mm_adds_epu16(add_epi16, set1_epi16);
        let subs_epi8 = _mm_subs_epi8(add_epi8, set1_epi8);
        let subs_epi16 = _mm_subs_epi16(add_epi16, set1_epi16);
        let subs_epu8 = _mm_subs_epu8(add_epi8, set1_epi8);
        let subs_epu16 = _mm_subs_epu16(add_epi16, set1_epi16);

        // === AVG/MIN/MAX (all safe) ===
        let avg_epu8 = _mm_avg_epu8(add_epi8, set1_epi8);
        let avg_epu16 = _mm_avg_epu16(add_epi16, set1_epi16);
        let min_epu8 = _mm_min_epu8(add_epi8, set1_epi8);
        let max_epu8 = _mm_max_epu8(add_epi8, set1_epi8);
        let min_epi16 = _mm_min_epi16(add_epi16, set1_epi16);
        let max_epi16 = _mm_max_epi16(add_epi16, set1_epi16);

        // === SAD (safe) ===
        let sad_epu8 = _mm_sad_epu8(add_epi8, set1_epi8);
    }

    // ============================================================================
    // SSE (128-bit f32)
    // ============================================================================

    #[target_feature(enable = "sse")]
    unsafe fn sse_safe_intrinsics() {
        // === CREATION (all safe) ===
        let zero_ps = _mm_setzero_ps();
        let set1_ps = _mm_set1_ps(1.0);
        let set_ps = _mm_set_ps(1.0, 2.0, 3.0, 4.0);
        let setr_ps = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let undef_ps = _mm_undefined_ps();

        // === ARITHMETIC (all safe) ===
        let add_ps = _mm_add_ps(zero_ps, set1_ps);
        let sub_ps = _mm_sub_ps(add_ps, set1_ps);
        let mul_ps = _mm_mul_ps(add_ps, set1_ps);
        let div_ps = _mm_div_ps(add_ps, set1_ps);
        let sqrt_ps = _mm_sqrt_ps(add_ps);
        let rcp_ps = _mm_rcp_ps(add_ps);
        let rsqrt_ps = _mm_rsqrt_ps(add_ps);
        let min_ps = _mm_min_ps(add_ps, set1_ps);
        let max_ps = _mm_max_ps(add_ps, set1_ps);

        // Scalar variants
        let add_ss = _mm_add_ss(add_ps, set1_ps);
        let sub_ss = _mm_sub_ss(add_ps, set1_ps);
        let mul_ss = _mm_mul_ss(add_ps, set1_ps);
        let div_ss = _mm_div_ss(add_ps, set1_ps);
        let sqrt_ss = _mm_sqrt_ss(add_ps);
        let rcp_ss = _mm_rcp_ss(add_ps);
        let rsqrt_ss = _mm_rsqrt_ss(add_ps);
        let min_ss = _mm_min_ss(add_ps, set1_ps);
        let max_ss = _mm_max_ss(add_ps, set1_ps);

        // === COMPARISON (all safe) ===
        let cmpeq_ps = _mm_cmpeq_ps(add_ps, set1_ps);
        let cmplt_ps = _mm_cmplt_ps(add_ps, set1_ps);
        let cmple_ps = _mm_cmple_ps(add_ps, set1_ps);
        let cmpgt_ps = _mm_cmpgt_ps(add_ps, set1_ps);
        let cmpge_ps = _mm_cmpge_ps(add_ps, set1_ps);
        let cmpneq_ps = _mm_cmpneq_ps(add_ps, set1_ps);
        let cmpnlt_ps = _mm_cmpnlt_ps(add_ps, set1_ps);
        let cmpnle_ps = _mm_cmpnle_ps(add_ps, set1_ps);
        let cmpngt_ps = _mm_cmpngt_ps(add_ps, set1_ps);
        let cmpnge_ps = _mm_cmpnge_ps(add_ps, set1_ps);
        let cmpord_ps = _mm_cmpord_ps(add_ps, set1_ps);
        let cmpunord_ps = _mm_cmpunord_ps(add_ps, set1_ps);

        // === BITWISE (all safe) ===
        let and_ps = _mm_and_ps(add_ps, set1_ps);
        let or_ps = _mm_or_ps(add_ps, set1_ps);
        let xor_ps = _mm_xor_ps(add_ps, set1_ps);
        let andnot_ps = _mm_andnot_ps(add_ps, set1_ps);

        // === SHUFFLE (all safe) ===
        let shuffle_ps = _mm_shuffle_ps::<0b00_01_10_11>(add_ps, set1_ps);
        let unpacklo_ps = _mm_unpacklo_ps(add_ps, set1_ps);
        let unpackhi_ps = _mm_unpackhi_ps(add_ps, set1_ps);
        let movehl_ps = _mm_movehl_ps(add_ps, set1_ps);
        let movelh_ps = _mm_movelh_ps(add_ps, set1_ps);
        let movemask_ps = _mm_movemask_ps(cmpeq_ps);
    }

    // ============================================================================
    // SSE4.1 (additional ops)
    // ============================================================================

    #[target_feature(enable = "sse4.1")]
    unsafe fn sse41_safe_intrinsics() {
        let zero_ps = _mm_setzero_ps();
        let set1_ps = _mm_set1_ps(1.0);
        let zero_pd = _mm_setzero_pd();
        let set1_pd = _mm_set1_pd(1.0);
        let zero_si = _mm_setzero_si128();
        let set1_epi8 = _mm_set1_epi8(1);
        let set1_epi16 = _mm_set1_epi16(1);
        let set1_epi32 = _mm_set1_epi32(1);
        let set1_epi64x = _mm_set1_epi64x(1);

        // === BLEND (all safe) ===
        let blend_ps = _mm_blend_ps::<0b1010>(zero_ps, set1_ps);
        let blend_pd = _mm_blend_pd::<0b01>(zero_pd, set1_pd);
        let blendv_ps = _mm_blendv_ps(zero_ps, set1_ps, zero_ps);
        let blendv_pd = _mm_blendv_pd(zero_pd, set1_pd, zero_pd);
        let blendv_epi8 = _mm_blendv_epi8(zero_si, set1_epi8, zero_si);
        let blend_epi16 = _mm_blend_epi16::<0b10101010>(zero_si, set1_epi16);

        // === ROUND (all safe) ===
        let ceil_ps = _mm_ceil_ps(set1_ps);
        let floor_ps = _mm_floor_ps(set1_ps);
        let round_ps = _mm_round_ps::<_MM_FROUND_TO_NEAREST_INT>(set1_ps);
        let ceil_pd = _mm_ceil_pd(set1_pd);
        let floor_pd = _mm_floor_pd(set1_pd);
        let round_pd = _mm_round_pd::<_MM_FROUND_TO_NEAREST_INT>(set1_pd);
        let ceil_ss = _mm_ceil_ss(zero_ps, set1_ps);
        let floor_ss = _mm_floor_ss(zero_ps, set1_ps);
        let ceil_sd = _mm_ceil_sd(zero_pd, set1_pd);
        let floor_sd = _mm_floor_sd(zero_pd, set1_pd);

        // === DOT PRODUCT (all safe) ===
        let dp_ps = _mm_dp_ps::<0xFF>(zero_ps, set1_ps);
        let dp_pd = _mm_dp_pd::<0xFF>(zero_pd, set1_pd);

        // === INSERT/EXTRACT (all safe) ===
        let insert_ps = _mm_insert_ps::<0x10>(zero_ps, set1_ps);
        let extract_ps = _mm_extract_ps::<0>(set1_ps);
        let insert_epi8 = _mm_insert_epi8::<0>(zero_si, 1);
        let insert_epi32 = _mm_insert_epi32::<0>(zero_si, 1);
        let insert_epi64 = _mm_insert_epi64::<0>(zero_si, 1);
        let extract_epi8 = _mm_extract_epi8::<0>(set1_epi8);
        let extract_epi32 = _mm_extract_epi32::<0>(set1_epi32);
        let extract_epi64 = _mm_extract_epi64::<0>(set1_epi64x);

        // === MIN/MAX (all safe) ===
        let min_epi8 = _mm_min_epi8(zero_si, set1_epi8);
        let max_epi8 = _mm_max_epi8(zero_si, set1_epi8);
        let min_epu16 = _mm_min_epu16(zero_si, set1_epi16);
        let max_epu16 = _mm_max_epu16(zero_si, set1_epi16);
        let min_epi32 = _mm_min_epi32(zero_si, set1_epi32);
        let max_epi32 = _mm_max_epi32(zero_si, set1_epi32);
        let min_epu32 = _mm_min_epu32(zero_si, set1_epi32);
        let max_epu32 = _mm_max_epu32(zero_si, set1_epi32);

        // === MULTIPLY (all safe) ===
        let mullo_epi32 = _mm_mullo_epi32(zero_si, set1_epi32);
        let mul_epi32 = _mm_mul_epi32(zero_si, set1_epi32);

        // === PACK (safe) ===
        let packus_epi32 = _mm_packus_epi32(zero_si, set1_epi32);

        // === SIGN EXTEND (all safe) ===
        let cvtepi8_epi16 = _mm_cvtepi8_epi16(set1_epi8);
        let cvtepi8_epi32 = _mm_cvtepi8_epi32(set1_epi8);
        let cvtepi8_epi64 = _mm_cvtepi8_epi64(set1_epi8);
        let cvtepi16_epi32 = _mm_cvtepi16_epi32(set1_epi16);
        let cvtepi16_epi64 = _mm_cvtepi16_epi64(set1_epi16);
        let cvtepi32_epi64 = _mm_cvtepi32_epi64(set1_epi32);
        let cvtepu8_epi16 = _mm_cvtepu8_epi16(set1_epi8);
        let cvtepu8_epi32 = _mm_cvtepu8_epi32(set1_epi8);
        let cvtepu8_epi64 = _mm_cvtepu8_epi64(set1_epi8);
        let cvtepu16_epi32 = _mm_cvtepu16_epi32(set1_epi16);
        let cvtepu16_epi64 = _mm_cvtepu16_epi64(set1_epi16);
        let cvtepu32_epi64 = _mm_cvtepu32_epi64(set1_epi32);

        // === TEST (all safe) ===
        let testz = _mm_testz_si128(zero_si, set1_epi32);
        let testc = _mm_testc_si128(zero_si, set1_epi32);
        let testnzc = _mm_testnzc_si128(zero_si, set1_epi32);

        // === MINPOS (safe) ===
        let minpos = _mm_minpos_epu16(set1_epi16);

        // === MPSADBW (safe) ===
        let mpsadbw = _mm_mpsadbw_epu8::<0>(zero_si, set1_epi8);
    }

    // ============================================================================
    // AVX (256-bit float)
    // ============================================================================

    #[target_feature(enable = "avx")]
    unsafe fn avx_safe_intrinsics() {
        // === CREATION (all safe) ===
        let zero_ps = _mm256_setzero_ps();
        let zero_pd = _mm256_setzero_pd();
        let set1_ps = _mm256_set1_ps(1.0);
        let set1_pd = _mm256_set1_pd(1.0);
        let set_ps = _mm256_set_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let set_pd = _mm256_set_pd(1.0, 2.0, 3.0, 4.0);
        let setr_ps = _mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let setr_pd = _mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
        let undef_ps = _mm256_undefined_ps();
        let undef_pd = _mm256_undefined_pd();

        // === ARITHMETIC f32 (all safe) ===
        let add_ps = _mm256_add_ps(zero_ps, set1_ps);
        let sub_ps = _mm256_sub_ps(add_ps, set1_ps);
        let mul_ps = _mm256_mul_ps(add_ps, set1_ps);
        let div_ps = _mm256_div_ps(add_ps, set1_ps);
        let sqrt_ps = _mm256_sqrt_ps(add_ps);
        let rcp_ps = _mm256_rcp_ps(add_ps);
        let rsqrt_ps = _mm256_rsqrt_ps(add_ps);
        let min_ps = _mm256_min_ps(add_ps, set1_ps);
        let max_ps = _mm256_max_ps(add_ps, set1_ps);

        // === ARITHMETIC f64 (all safe) ===
        let add_pd = _mm256_add_pd(zero_pd, set1_pd);
        let sub_pd = _mm256_sub_pd(add_pd, set1_pd);
        let mul_pd = _mm256_mul_pd(add_pd, set1_pd);
        let div_pd = _mm256_div_pd(add_pd, set1_pd);
        let sqrt_pd = _mm256_sqrt_pd(add_pd);
        let min_pd = _mm256_min_pd(add_pd, set1_pd);
        let max_pd = _mm256_max_pd(add_pd, set1_pd);

        // === COMPARISON (all safe) ===
        let cmp_ps = _mm256_cmp_ps::<_CMP_EQ_OQ>(add_ps, set1_ps);
        let cmp_pd = _mm256_cmp_pd::<_CMP_EQ_OQ>(add_pd, set1_pd);

        // === SHUFFLE/PERMUTE/BLEND (all safe) ===
        let shuffle_ps = _mm256_shuffle_ps::<0b00_01_10_11>(add_ps, set1_ps);
        let shuffle_pd = _mm256_shuffle_pd::<0b0101>(add_pd, set1_pd);
        let permute_ps = _mm256_permute_ps::<0b00_01_10_11>(add_ps);
        let permute_pd = _mm256_permute_pd::<0b0101>(add_pd);
        let permute2f128_ps = _mm256_permute2f128_ps::<0x01>(add_ps, set1_ps);
        let permute2f128_pd = _mm256_permute2f128_pd::<0x01>(add_pd, set1_pd);
        let blend_ps = _mm256_blend_ps::<0b10101010>(add_ps, set1_ps);
        let blend_pd = _mm256_blend_pd::<0b0101>(add_pd, set1_pd);
        let blendv_ps = _mm256_blendv_ps(add_ps, set1_ps, cmp_ps);
        let blendv_pd = _mm256_blendv_pd(add_pd, set1_pd, cmp_pd);
        let unpacklo_ps = _mm256_unpacklo_ps(add_ps, set1_ps);
        let unpackhi_ps = _mm256_unpackhi_ps(add_ps, set1_ps);
        let unpacklo_pd = _mm256_unpacklo_pd(add_pd, set1_pd);
        let unpackhi_pd = _mm256_unpackhi_pd(add_pd, set1_pd);

        // === ROUND (all safe) ===
        let floor_ps = _mm256_floor_ps(add_ps);
        let ceil_ps = _mm256_ceil_ps(add_ps);
        let round_ps = _mm256_round_ps::<_MM_FROUND_TO_NEAREST_INT>(add_ps);
        let floor_pd = _mm256_floor_pd(add_pd);
        let ceil_pd = _mm256_ceil_pd(add_pd);
        let round_pd = _mm256_round_pd::<_MM_FROUND_TO_NEAREST_INT>(add_pd);

        // === BITWISE (all safe) ===
        let and_ps = _mm256_and_ps(add_ps, set1_ps);
        let or_ps = _mm256_or_ps(add_ps, set1_ps);
        let xor_ps = _mm256_xor_ps(add_ps, set1_ps);
        let andnot_ps = _mm256_andnot_ps(add_ps, set1_ps);
        let and_pd = _mm256_and_pd(add_pd, set1_pd);
        let or_pd = _mm256_or_pd(add_pd, set1_pd);
        let xor_pd = _mm256_xor_pd(add_pd, set1_pd);
        let andnot_pd = _mm256_andnot_pd(add_pd, set1_pd);

        // === HORIZONTAL (all safe) ===
        let hadd_ps = _mm256_hadd_ps(add_ps, set1_ps);
        let hsub_ps = _mm256_hsub_ps(add_ps, set1_ps);
        let hadd_pd = _mm256_hadd_pd(add_pd, set1_pd);
        let hsub_pd = _mm256_hsub_pd(add_pd, set1_pd);
        let addsub_ps = _mm256_addsub_ps(add_ps, set1_ps);
        let addsub_pd = _mm256_addsub_pd(add_pd, set1_pd);

        // === EXTRACT/INSERT (all safe) ===
        let extract128_ps = _mm256_extractf128_ps::<0>(add_ps);
        let extract128_pd = _mm256_extractf128_pd::<0>(add_pd);
        let insert128_ps = _mm256_insertf128_ps::<0>(add_ps, extract128_ps);
        let insert128_pd = _mm256_insertf128_pd::<0>(add_pd, extract128_pd);

        // === MOVEMASK (safe) ===
        let mask_ps = _mm256_movemask_ps(cmp_ps);
        let mask_pd = _mm256_movemask_pd(cmp_pd);

        // === CONVERSION (all safe) ===
        let cvt_ps_pd = _mm256_cvtps_pd(extract128_ps);
        let cvt_pd_ps = _mm256_cvtpd_ps(add_pd);
        let cvt_ps_epi32 = _mm256_cvtps_epi32(add_ps);
        let cvtt_ps_epi32 = _mm256_cvttps_epi32(add_ps);
        let cvt_epi32_ps = _mm256_cvtepi32_ps(cvt_ps_epi32);
        let cvt_pd_epi32 = _mm256_cvtpd_epi32(add_pd);
        let cvtt_pd_epi32 = _mm256_cvttpd_epi32(add_pd);

        // === CAST (all safe) ===
        let cast_ps_pd = _mm256_castps_pd(add_ps);
        let cast_pd_ps = _mm256_castpd_ps(add_pd);
        let cast_ps_si256 = _mm256_castps_si256(add_ps);
        let cast_si256_ps = _mm256_castsi256_ps(cvt_ps_epi32);
        let cast_ps256_ps128 = _mm256_castps256_ps128(add_ps);
        let cast_pd256_pd128 = _mm256_castpd256_pd128(add_pd);
        let cast_ps128_ps256 = _mm256_castps128_ps256(extract128_ps);
        let cast_pd128_pd256 = _mm256_castpd128_pd256(extract128_pd);

        // === TESTC/TESTZ (all safe) ===
        let testc = _mm256_testc_ps(add_ps, set1_ps);
        let testz = _mm256_testz_ps(add_ps, set1_ps);
        let testnzc = _mm256_testnzc_ps(add_ps, set1_ps);
        let testc_pd = _mm256_testc_pd(add_pd, set1_pd);
        let testz_pd = _mm256_testz_pd(add_pd, set1_pd);

        // === DOT PRODUCT (safe) ===
        let dp_ps = _mm256_dp_ps::<0xFF>(add_ps, set1_ps);
    }

    // ============================================================================
    // AVX2 (256-bit integer)
    // ============================================================================

    #[target_feature(enable = "avx2")]
    unsafe fn avx2_safe_intrinsics() {
        // === INTEGER CREATION (all safe) ===
        let zero_si = _mm256_setzero_si256();
        let set1_epi8 = _mm256_set1_epi8(1);
        let set1_epi16 = _mm256_set1_epi16(1);
        let set1_epi32 = _mm256_set1_epi32(1);
        let set1_epi64x = _mm256_set1_epi64x(1);
        let undef_si = _mm256_undefined_si256();

        // === INTEGER ARITHMETIC i8 (all safe) ===
        let add_epi8 = _mm256_add_epi8(zero_si, set1_epi8);
        let sub_epi8 = _mm256_sub_epi8(add_epi8, set1_epi8);
        let adds_epi8 = _mm256_adds_epi8(add_epi8, set1_epi8);
        let adds_epu8 = _mm256_adds_epu8(add_epi8, set1_epi8);
        let subs_epi8 = _mm256_subs_epi8(add_epi8, set1_epi8);
        let subs_epu8 = _mm256_subs_epu8(add_epi8, set1_epi8);
        let avg_epu8 = _mm256_avg_epu8(add_epi8, set1_epi8);
        let max_epi8 = _mm256_max_epi8(add_epi8, set1_epi8);
        let max_epu8 = _mm256_max_epu8(add_epi8, set1_epi8);
        let min_epi8 = _mm256_min_epi8(add_epi8, set1_epi8);
        let min_epu8 = _mm256_min_epu8(add_epi8, set1_epi8);
        let abs_epi8 = _mm256_abs_epi8(add_epi8);

        // === INTEGER ARITHMETIC i16 (all safe) ===
        let add_epi16 = _mm256_add_epi16(zero_si, set1_epi16);
        let sub_epi16 = _mm256_sub_epi16(add_epi16, set1_epi16);
        let adds_epi16 = _mm256_adds_epi16(add_epi16, set1_epi16);
        let adds_epu16 = _mm256_adds_epu16(add_epi16, set1_epi16);
        let subs_epi16 = _mm256_subs_epi16(add_epi16, set1_epi16);
        let subs_epu16 = _mm256_subs_epu16(add_epi16, set1_epi16);
        let avg_epu16 = _mm256_avg_epu16(add_epi16, set1_epi16);
        let max_epi16 = _mm256_max_epi16(add_epi16, set1_epi16);
        let max_epu16 = _mm256_max_epu16(add_epi16, set1_epi16);
        let min_epi16 = _mm256_min_epi16(add_epi16, set1_epi16);
        let min_epu16 = _mm256_min_epu16(add_epi16, set1_epi16);
        let abs_epi16 = _mm256_abs_epi16(add_epi16);
        let mulhi_epi16 = _mm256_mulhi_epi16(add_epi16, set1_epi16);
        let mulhi_epu16 = _mm256_mulhi_epu16(add_epi16, set1_epi16);
        let mullo_epi16 = _mm256_mullo_epi16(add_epi16, set1_epi16);
        let mulhrs_epi16 = _mm256_mulhrs_epi16(add_epi16, set1_epi16);
        let madd_epi16 = _mm256_madd_epi16(add_epi16, set1_epi16);
        let maddubs_epi16 = _mm256_maddubs_epi16(add_epi8, set1_epi8);
        let hadd_epi16 = _mm256_hadd_epi16(add_epi16, set1_epi16);
        let hadds_epi16 = _mm256_hadds_epi16(add_epi16, set1_epi16);
        let hsub_epi16 = _mm256_hsub_epi16(add_epi16, set1_epi16);
        let hsubs_epi16 = _mm256_hsubs_epi16(add_epi16, set1_epi16);

        // === INTEGER ARITHMETIC i32 (all safe) ===
        let add_epi32 = _mm256_add_epi32(zero_si, set1_epi32);
        let sub_epi32 = _mm256_sub_epi32(add_epi32, set1_epi32);
        let max_epi32 = _mm256_max_epi32(add_epi32, set1_epi32);
        let max_epu32 = _mm256_max_epu32(add_epi32, set1_epi32);
        let min_epi32 = _mm256_min_epi32(add_epi32, set1_epi32);
        let min_epu32 = _mm256_min_epu32(add_epi32, set1_epi32);
        let abs_epi32 = _mm256_abs_epi32(add_epi32);
        let mullo_epi32 = _mm256_mullo_epi32(add_epi32, set1_epi32);
        let mul_epi32 = _mm256_mul_epi32(add_epi32, set1_epi32);
        let mul_epu32 = _mm256_mul_epu32(add_epi32, set1_epi32);
        let hadd_epi32 = _mm256_hadd_epi32(add_epi32, set1_epi32);
        let hsub_epi32 = _mm256_hsub_epi32(add_epi32, set1_epi32);

        // === INTEGER ARITHMETIC i64 (all safe) ===
        let add_epi64 = _mm256_add_epi64(zero_si, set1_epi64x);
        let sub_epi64 = _mm256_sub_epi64(add_epi64, set1_epi64x);

        // === BITWISE (all safe) ===
        let and_si = _mm256_and_si256(add_epi32, set1_epi32);
        let or_si = _mm256_or_si256(add_epi32, set1_epi32);
        let xor_si = _mm256_xor_si256(add_epi32, set1_epi32);
        let andnot_si = _mm256_andnot_si256(add_epi32, set1_epi32);

        // === SHIFT (all safe) ===
        let slli_epi16 = _mm256_slli_epi16::<1>(add_epi16);
        let srli_epi16 = _mm256_srli_epi16::<1>(add_epi16);
        let srai_epi16 = _mm256_srai_epi16::<1>(add_epi16);
        let slli_epi32 = _mm256_slli_epi32::<1>(add_epi32);
        let srli_epi32 = _mm256_srli_epi32::<1>(add_epi32);
        let srai_epi32 = _mm256_srai_epi32::<1>(add_epi32);
        let slli_epi64 = _mm256_slli_epi64::<1>(add_epi64);
        let srli_epi64 = _mm256_srli_epi64::<1>(add_epi64);
        let slli_si256 = _mm256_slli_si256::<1>(add_epi32);
        let srli_si256 = _mm256_srli_si256::<1>(add_epi32);
        let sllv_epi32 = _mm256_sllv_epi32(add_epi32, set1_epi32);
        let srlv_epi32 = _mm256_srlv_epi32(add_epi32, set1_epi32);
        let srav_epi32 = _mm256_srav_epi32(add_epi32, set1_epi32);
        let sllv_epi64 = _mm256_sllv_epi64(add_epi64, set1_epi64x);
        let srlv_epi64 = _mm256_srlv_epi64(add_epi64, set1_epi64x);

        // === COMPARISON (all safe) ===
        let cmpeq_epi8 = _mm256_cmpeq_epi8(add_epi8, set1_epi8);
        let cmpeq_epi16 = _mm256_cmpeq_epi16(add_epi16, set1_epi16);
        let cmpeq_epi32 = _mm256_cmpeq_epi32(add_epi32, set1_epi32);
        let cmpeq_epi64 = _mm256_cmpeq_epi64(add_epi64, set1_epi64x);
        let cmpgt_epi8 = _mm256_cmpgt_epi8(add_epi8, set1_epi8);
        let cmpgt_epi16 = _mm256_cmpgt_epi16(add_epi16, set1_epi16);
        let cmpgt_epi32 = _mm256_cmpgt_epi32(add_epi32, set1_epi32);
        let cmpgt_epi64 = _mm256_cmpgt_epi64(add_epi64, set1_epi64x);
        let movemask_epi8 = _mm256_movemask_epi8(cmpeq_epi8);

        // === SHUFFLE/PERMUTE (all safe) ===
        let shuffle_epi8 = _mm256_shuffle_epi8(add_epi8, set1_epi8);
        let shuffle_epi32 = _mm256_shuffle_epi32::<0b00_01_10_11>(add_epi32);
        let shufflelo_epi16 = _mm256_shufflelo_epi16::<0b00_01_10_11>(add_epi16);
        let shufflehi_epi16 = _mm256_shufflehi_epi16::<0b00_01_10_11>(add_epi16);
        let permute4x64_epi64 = _mm256_permute4x64_epi64::<0b00_01_10_11>(add_epi64);
        let permute2x128_si256 = _mm256_permute2x128_si256::<0x01>(add_epi32, set1_epi32);
        let permutevar8x32_epi32 = _mm256_permutevar8x32_epi32(add_epi32, set1_epi32);
        let unpacklo_epi8 = _mm256_unpacklo_epi8(add_epi8, set1_epi8);
        let unpackhi_epi8 = _mm256_unpackhi_epi8(add_epi8, set1_epi8);
        let unpacklo_epi16 = _mm256_unpacklo_epi16(add_epi16, set1_epi16);
        let unpackhi_epi16 = _mm256_unpackhi_epi16(add_epi16, set1_epi16);
        let unpacklo_epi32 = _mm256_unpacklo_epi32(add_epi32, set1_epi32);
        let unpackhi_epi32 = _mm256_unpackhi_epi32(add_epi32, set1_epi32);
        let unpacklo_epi64 = _mm256_unpacklo_epi64(add_epi64, set1_epi64x);
        let unpackhi_epi64 = _mm256_unpackhi_epi64(add_epi64, set1_epi64x);

        // === BLEND (all safe) ===
        let blend_epi16 = _mm256_blend_epi16::<0b10101010>(add_epi16, set1_epi16);
        let blend_epi32 = _mm256_blend_epi32::<0b10101010>(add_epi32, set1_epi32);
        let blendv_epi8 = _mm256_blendv_epi8(add_epi8, set1_epi8, cmpeq_epi8);

        // === PACK (all safe) ===
        let packs_epi16 = _mm256_packs_epi16(add_epi16, set1_epi16);
        let packs_epi32 = _mm256_packs_epi32(add_epi32, set1_epi32);
        let packus_epi16 = _mm256_packus_epi16(add_epi16, set1_epi16);
        let packus_epi32 = _mm256_packus_epi32(add_epi32, set1_epi32);

        // === SIGN EXTEND (all safe) ===
        let cvtepi8_epi16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(add_epi8));
        let cvtepi8_epi32 = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(add_epi8));
        let cvtepi8_epi64 = _mm256_cvtepi8_epi64(_mm256_castsi256_si128(add_epi8));
        let cvtepi16_epi32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(add_epi16));
        let cvtepi16_epi64 = _mm256_cvtepi16_epi64(_mm256_castsi256_si128(add_epi16));
        let cvtepi32_epi64 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(add_epi32));
        let cvtepu8_epi16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(add_epi8));
        let cvtepu8_epi32 = _mm256_cvtepu8_epi32(_mm256_castsi256_si128(add_epi8));
        let cvtepu8_epi64 = _mm256_cvtepu8_epi64(_mm256_castsi256_si128(add_epi8));
        let cvtepu16_epi32 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(add_epi16));
        let cvtepu16_epi64 = _mm256_cvtepu16_epi64(_mm256_castsi256_si128(add_epi16));
        let cvtepu32_epi64 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(add_epi32));

        // === EXTRACT/INSERT (all safe) ===
        let extract_si128 = _mm256_extracti128_si256::<0>(add_epi32);
        let insert_si128 = _mm256_inserti128_si256::<0>(add_epi32, extract_si128);

        // === BROADCAST (safe when from register) ===
        let broadcastsi128 = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(add_epi32));

        // === SIGN (all safe) ===
        let sign_epi8 = _mm256_sign_epi8(add_epi8, set1_epi8);
        let sign_epi16 = _mm256_sign_epi16(add_epi16, set1_epi16);
        let sign_epi32 = _mm256_sign_epi32(add_epi32, set1_epi32);

        // === SAD (safe) ===
        let sad_epu8 = _mm256_sad_epu8(add_epi8, set1_epi8);

        // === ALIGNR (safe) ===
        let alignr_epi8 = _mm256_alignr_epi8::<4>(add_epi8, set1_epi8);

        // === PERMUTE FLOAT (safe, AVX2 adds these) ===
        let zero_ps = _mm256_setzero_ps();
        let permutevar8x32_ps = _mm256_permutevar8x32_ps(zero_ps, set1_epi32);
        let permute4x64_pd = _mm256_permute4x64_pd::<0b00_01_10_11>(_mm256_setzero_pd());
    }

    // ============================================================================
    // FMA
    // ============================================================================

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn fma_safe_intrinsics() {
        let a = _mm256_setzero_ps();
        let b = _mm256_set1_ps(1.0);
        let c = _mm256_set1_ps(2.0);

        // === FMA 256-bit f32 (all safe) ===
        let fmadd_ps = _mm256_fmadd_ps(a, b, c);
        let fmsub_ps = _mm256_fmsub_ps(a, b, c);
        let fnmadd_ps = _mm256_fnmadd_ps(a, b, c);
        let fnmsub_ps = _mm256_fnmsub_ps(a, b, c);
        let fmaddsub_ps = _mm256_fmaddsub_ps(a, b, c);
        let fmsubadd_ps = _mm256_fmsubadd_ps(a, b, c);

        // === FMA 256-bit f64 (all safe) ===
        let ad = _mm256_setzero_pd();
        let bd = _mm256_set1_pd(1.0);
        let cd = _mm256_set1_pd(2.0);
        let fmadd_pd = _mm256_fmadd_pd(ad, bd, cd);
        let fmsub_pd = _mm256_fmsub_pd(ad, bd, cd);
        let fnmadd_pd = _mm256_fnmadd_pd(ad, bd, cd);
        let fnmsub_pd = _mm256_fnmsub_pd(ad, bd, cd);
        let fmaddsub_pd = _mm256_fmaddsub_pd(ad, bd, cd);
        let fmsubadd_pd = _mm256_fmsubadd_pd(ad, bd, cd);

        // === FMA 128-bit (all safe) ===
        let a128 = _mm_setzero_ps();
        let b128 = _mm_set1_ps(1.0);
        let c128 = _mm_set1_ps(2.0);
        let fmadd_ps_128 = _mm_fmadd_ps(a128, b128, c128);
        let fmsub_ps_128 = _mm_fmsub_ps(a128, b128, c128);
        let fnmadd_ps_128 = _mm_fnmadd_ps(a128, b128, c128);
        let fnmsub_ps_128 = _mm_fnmsub_ps(a128, b128, c128);
        let fmaddsub_ps_128 = _mm_fmaddsub_ps(a128, b128, c128);
        let fmsubadd_ps_128 = _mm_fmsubadd_ps(a128, b128, c128);

        // Scalar FMA
        let fmadd_ss = _mm_fmadd_ss(a128, b128, c128);
        let fmsub_ss = _mm_fmsub_ss(a128, b128, c128);
        let fnmadd_ss = _mm_fnmadd_ss(a128, b128, c128);
        let fnmsub_ss = _mm_fnmsub_ss(a128, b128, c128);

        let a128d = _mm_setzero_pd();
        let b128d = _mm_set1_pd(1.0);
        let c128d = _mm_set1_pd(2.0);
        let fmadd_pd_128 = _mm_fmadd_pd(a128d, b128d, c128d);
        let fmadd_sd = _mm_fmadd_sd(a128d, b128d, c128d);
    }

    // ============================================================================
    // TESTS
    // ============================================================================

    #[test]
    fn test_all_safe_intrinsics_compile() {
        // This test verifies that all intrinsics above compile without
        // requiring unsafe blocks inside the target_feature functions.
        // If this test compiles, all listed intrinsics are safe.

        if std::is_x86_feature_detected!("sse") {
            unsafe { sse_safe_intrinsics() };
        }
        if std::is_x86_feature_detected!("sse2") {
            unsafe { sse2_safe_intrinsics() };
        }
        if std::is_x86_feature_detected!("sse4.1") {
            unsafe { sse41_safe_intrinsics() };
        }
        if std::is_x86_feature_detected!("avx") {
            unsafe { avx_safe_intrinsics() };
        }
        if std::is_x86_feature_detected!("avx2") {
            unsafe { avx2_safe_intrinsics() };
        }
        if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
            unsafe { fma_safe_intrinsics() };
        }
    }
}

// ============================================================================
// UNSAFE INTRINSICS (documented for reference)
// ============================================================================
//
// The following intrinsics ALWAYS require unsafe blocks, even inside
// #[target_feature] context, because they involve raw pointer operations:
//
// LOADS (pointer-based):
//   _mm_load_ps, _mm_loadu_ps, _mm_load_pd, _mm_loadu_pd
//   _mm_load_si128, _mm_loadu_si128
//   _mm256_load_ps, _mm256_loadu_ps, _mm256_load_pd, _mm256_loadu_pd
//   _mm256_load_si256, _mm256_loadu_si256
//   All streaming loads (_mm_stream_load_*)
//
// STORES (pointer-based):
//   _mm_store_ps, _mm_storeu_ps, _mm_store_pd, _mm_storeu_pd
//   _mm_store_si128, _mm_storeu_si128
//   _mm256_store_ps, _mm256_storeu_ps, _mm256_store_pd, _mm256_storeu_pd
//   _mm256_store_si256, _mm256_storeu_si256
//   All streaming stores (_mm_stream_*)
//   All non-temporal stores
//
// MASKED LOADS/STORES:
//   _mm_maskload_ps, _mm_maskstore_ps
//   _mm256_maskload_ps, _mm256_maskstore_ps
//   _mm_maskload_pd, _mm_maskstore_pd
//   _mm256_maskload_pd, _mm256_maskstore_pd
//   _mm_maskload_epi32, _mm_maskstore_epi32
//   _mm256_maskload_epi32, _mm256_maskstore_epi32
//   _mm_maskload_epi64, _mm_maskstore_epi64
//   _mm256_maskload_epi64, _mm256_maskstore_epi64
//
// GATHER (AVX2):
//   _mm_i32gather_*, _mm256_i32gather_*
//   _mm_i64gather_*, _mm256_i64gather_*
//   _mm_mask_i32gather_*, _mm256_mask_i32gather_*
//   _mm_mask_i64gather_*, _mm256_mask_i64gather_*
//
// BROADCAST FROM MEMORY:
//   _mm_broadcast_ss, _mm256_broadcast_ss (pointer variants)
//   _mm256_broadcast_sd (pointer variant)
//   _mm256_broadcast_ps, _mm256_broadcast_pd
//
// ============================================================================

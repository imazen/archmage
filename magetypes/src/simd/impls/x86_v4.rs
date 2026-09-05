//! Backend implementations for X64V4Token and X64V4xToken (native AVX-512).
//!
//! Implements the W512 backend traits using native 512-bit AVX-512 intrinsics
//! for both X64V4Token (base AVX-512) and X64V4xToken (+ VPOPCNTDQ, BITALG, etc.).
//!
//! X64V4xToken also gets extension trait impls (popcnt) for Modern-only features.
//!
//! W128 and W256 types use X64V3Token (V4 downcasts to V3 for narrower widths).
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.
//!
//! # Safety
//!
//! `#[arcane]` checks value intrinsics against the receiver token's features.
//! Whole-array loads/stores and bit casts use `crate::simd_storage` helpers,
//! which require POD storage and enforce equal sizes at compile time.
//! Token-bearing wrappers never implement the storage POD trait.

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
use archmage::{X64V4Token, X64V4xToken, arcane};

use crate::simd::backends::*;

// ============================================================================
// X64V4Token — base AVX-512 (F/BW/CD/DQ/VL)
// ============================================================================

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl F32x16Backend for archmage::X64V4Token {
    type Repr = __m512;

    #[arcane(shared, _self = X64V4Token)]
    fn splat(self, v: f32) -> __m512 {
        _mm512_set1_ps(v)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn zero(self) -> __m512 {
        _mm512_setzero_ps()
    }

    #[arcane(shared, _self = X64V4Token)]
    fn load(self, data: &[f32; 16]) -> __m512 {
        crate::simd_storage::copy(data)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn from_array(self, arr: [f32; 16]) -> __m512 {
        crate::simd_storage::cast(arr)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn store(self, repr: __m512, out: &mut [f32; 16]) {
        crate::simd_storage::store(repr, out);
    }

    #[arcane(shared, _self = X64V4Token)]
    fn to_array(self, repr: __m512) -> [f32; 16] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn add(self, a: __m512, b: __m512) -> __m512 {
        _mm512_add_ps(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn sub(self, a: __m512, b: __m512) -> __m512 {
        _mm512_sub_ps(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn mul(self, a: __m512, b: __m512) -> __m512 {
        _mm512_mul_ps(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn div(self, a: __m512, b: __m512) -> __m512 {
        _mm512_div_ps(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn neg(self, a: __m512) -> __m512 {
        _mm512_sub_ps(_mm512_setzero_ps(), a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn min(self, a: __m512, b: __m512) -> __m512 {
        _mm512_min_ps(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn max(self, a: __m512, b: __m512) -> __m512 {
        _mm512_max_ps(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn sqrt(self, a: __m512) -> __m512 {
        _mm512_sqrt_ps(a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn abs(self, a: __m512) -> __m512 {
        let mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFF_FFFFu32 as i32));
        _mm512_and_ps(a, mask)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn floor(self, a: __m512) -> __m512 {
        _mm512_roundscale_ps::<0x01>(a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn ceil(self, a: __m512) -> __m512 {
        _mm512_roundscale_ps::<0x02>(a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn round(self, a: __m512) -> __m512 {
        _mm512_roundscale_ps::<0x00>(a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn mul_add(self, a: __m512, b: __m512, c: __m512) -> __m512 {
        _mm512_fmadd_ps(a, b, c)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn mul_sub(self, a: __m512, b: __m512, c: __m512) -> __m512 {
        _mm512_fmsub_ps(a, b, c)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_eq(self, a: __m512, b: __m512) -> __m512 {
        let mask = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(a, b);
        _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_ne(self, a: __m512, b: __m512) -> __m512 {
        let mask = _mm512_cmp_ps_mask::<_CMP_NEQ_UQ>(a, b);
        _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_lt(self, a: __m512, b: __m512) -> __m512 {
        let mask = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(a, b);
        _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_le(self, a: __m512, b: __m512) -> __m512 {
        let mask = _mm512_cmp_ps_mask::<_CMP_LE_OQ>(a, b);
        _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_gt(self, a: __m512, b: __m512) -> __m512 {
        let mask = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(a, b);
        _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_ge(self, a: __m512, b: __m512) -> __m512 {
        let mask = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(a, b);
        _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn blend(self, mask: __m512, if_true: __m512, if_false: __m512) -> __m512 {
        let mask_i = _mm512_castps_si512(mask);
        let k = _mm512_cmpneq_epi32_mask(mask_i, _mm512_setzero_si512());
        _mm512_mask_blend_ps(k, if_false, if_true)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn reduce_add(self, a: __m512) -> f32 {
        _mm512_reduce_add_ps(a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn reduce_min(self, a: __m512) -> f32 {
        _mm512_reduce_min_ps(a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn reduce_max(self, a: __m512) -> f32 {
        _mm512_reduce_max_ps(a)
    }

    // Raw 14-bit hardware estimate — the cheapest >=12-bit path on AVX-512.
    #[arcane(shared, _self = X64V4Token)]
    fn rcp_approx(self, a: __m512) -> __m512 {
        _mm512_rcp14_ps(a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn rsqrt_approx(self, a: __m512) -> __m512 {
        _mm512_rsqrt14_ps(a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn recip(self, a: __m512) -> __m512 {
        // token nibbles: QNAN->2 SNAN->2 ZERO->6 ONE->0 -INF->7 +INF->8 NEG->0 POS->0
        const RECIP_FIXUP_TBL: i32 = 0x0087_0622u32 as i32;
        {
            let r = _mm512_rcp14_ps(a);
            let e = _mm512_fnmadd_ps(a, r, _mm512_set1_ps(1.0));
            let refined = _mm512_fmadd_ps(r, e, r);
            _mm512_fixupimm_ps::<0>(refined, a, _mm512_set1_epi32(RECIP_FIXUP_TBL))
        }
    }

    #[arcane(shared, _self = X64V4Token)]
    fn rsqrt(self, a: __m512) -> __m512 {
        // token nibbles: QNAN->2 SNAN->2 ZERO->6 ONE->0 -INF->3 +INF->8 NEG->3 POS->0
        const RSQRT_FIXUP_TBL: i32 = 0x0383_0622u32 as i32;
        {
            let y = _mm512_rsqrt14_ps(a);
            let t = _mm512_mul_ps(a, _mm512_mul_ps(y, y));
            let refined = _mm512_mul_ps(
                y,
                _mm512_fnmadd_ps(_mm512_set1_ps(0.5), t, _mm512_set1_ps(1.5)),
            );
            _mm512_fixupimm_ps::<0>(refined, a, _mm512_set1_epi32(RSQRT_FIXUP_TBL))
        }
    }

    #[arcane(shared, _self = X64V4Token)]
    fn not(self, a: __m512) -> __m512 {
        let all_ones = _mm512_castsi512_ps(_mm512_set1_epi32(-1));
        _mm512_xor_ps(a, all_ones)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitand(self, a: __m512, b: __m512) -> __m512 {
        _mm512_and_ps(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitor(self, a: __m512, b: __m512) -> __m512 {
        _mm512_or_ps(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitxor(self, a: __m512, b: __m512) -> __m512 {
        _mm512_xor_ps(a, b)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl F64x8Backend for archmage::X64V4Token {
    type Repr = __m512d;

    #[arcane(shared, _self = X64V4Token)]
    fn splat(self, v: f64) -> __m512d {
        _mm512_set1_pd(v)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn zero(self) -> __m512d {
        _mm512_setzero_pd()
    }

    #[arcane(shared, _self = X64V4Token)]
    fn load(self, data: &[f64; 8]) -> __m512d {
        crate::simd_storage::copy(data)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn from_array(self, arr: [f64; 8]) -> __m512d {
        crate::simd_storage::cast(arr)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn store(self, repr: __m512d, out: &mut [f64; 8]) {
        crate::simd_storage::store(repr, out);
    }

    #[arcane(shared, _self = X64V4Token)]
    fn to_array(self, repr: __m512d) -> [f64; 8] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn add(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_add_pd(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn sub(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_sub_pd(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn mul(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_mul_pd(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn div(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_div_pd(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn neg(self, a: __m512d) -> __m512d {
        _mm512_sub_pd(_mm512_setzero_pd(), a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn min(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_min_pd(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn max(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_max_pd(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn sqrt(self, a: __m512d) -> __m512d {
        _mm512_sqrt_pd(a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn abs(self, a: __m512d) -> __m512d {
        let mask = _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFF_FFFF_FFFF_FFFFu64 as i64));
        _mm512_and_pd(a, mask)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn floor(self, a: __m512d) -> __m512d {
        _mm512_roundscale_pd::<0x01>(a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn ceil(self, a: __m512d) -> __m512d {
        _mm512_roundscale_pd::<0x02>(a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn round(self, a: __m512d) -> __m512d {
        _mm512_roundscale_pd::<0x00>(a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn mul_add(self, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        _mm512_fmadd_pd(a, b, c)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn mul_sub(self, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        _mm512_fmsub_pd(a, b, c)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_eq(self, a: __m512d, b: __m512d) -> __m512d {
        let mask = _mm512_cmp_pd_mask::<_CMP_EQ_OQ>(a, b);
        _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_ne(self, a: __m512d, b: __m512d) -> __m512d {
        let mask = _mm512_cmp_pd_mask::<_CMP_NEQ_UQ>(a, b);
        _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_lt(self, a: __m512d, b: __m512d) -> __m512d {
        let mask = _mm512_cmp_pd_mask::<_CMP_LT_OQ>(a, b);
        _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_le(self, a: __m512d, b: __m512d) -> __m512d {
        let mask = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(a, b);
        _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_gt(self, a: __m512d, b: __m512d) -> __m512d {
        let mask = _mm512_cmp_pd_mask::<_CMP_GT_OQ>(a, b);
        _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_ge(self, a: __m512d, b: __m512d) -> __m512d {
        let mask = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(a, b);
        _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn blend(self, mask: __m512d, if_true: __m512d, if_false: __m512d) -> __m512d {
        let mask_i = _mm512_castpd_si512(mask);
        let k = _mm512_cmpneq_epi64_mask(mask_i, _mm512_setzero_si512());
        _mm512_mask_blend_pd(k, if_false, if_true)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn reduce_add(self, a: __m512d) -> f64 {
        _mm512_reduce_add_pd(a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn reduce_min(self, a: __m512d) -> f64 {
        _mm512_reduce_min_pd(a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn reduce_max(self, a: __m512d) -> f64 {
        _mm512_reduce_max_pd(a)
    }

    // Raw 14-bit hardware estimate — the cheapest >=12-bit path on AVX-512.
    #[arcane(shared, _self = X64V4Token)]
    fn rcp_approx(self, a: __m512d) -> __m512d {
        _mm512_rcp14_pd(a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn rsqrt_approx(self, a: __m512d) -> __m512d {
        _mm512_rsqrt14_pd(a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn recip(self, a: __m512d) -> __m512d {
        _mm512_div_pd(_mm512_set1_pd(1.0), a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn rsqrt(self, a: __m512d) -> __m512d {
        _mm512_div_pd(_mm512_set1_pd(1.0), _mm512_sqrt_pd(a))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn not(self, a: __m512d) -> __m512d {
        let all_ones = _mm512_castsi512_pd(_mm512_set1_epi64(-1));
        _mm512_xor_pd(a, all_ones)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitand(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_and_pd(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitor(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_or_pd(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitxor(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_xor_pd(a, b)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I8x64Backend for archmage::X64V4Token {
    type Repr = __m512i;

    #[arcane(shared, _self = X64V4Token)]
    fn splat(self, v: i8) -> __m512i {
        _mm512_set1_epi8(v as _)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[arcane(shared, _self = X64V4Token)]
    fn load(self, data: &[i8; 64]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn from_array(self, arr: [i8; 64]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn store(self, repr: __m512i, out: &mut [i8; 64]) {
        crate::simd_storage::store(repr, out);
    }

    #[arcane(shared, _self = X64V4Token)]
    fn to_array(self, repr: __m512i) -> [i8; 64] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi8(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi8(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi8(_mm512_setzero_si512(), a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epi8(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epi8(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn abs(self, a: __m512i) -> __m512i {
        _mm512_abs_epi8(a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epi8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epi8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epi8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epi8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epi8_mask(b, a);
        _mm512_maskz_set1_epi8(mask, -1)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epi8_mask(b, a);
        _mm512_maskz_set1_epi8(mask, -1)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi8_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi8(k, if_false, if_true)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn reduce_add(self, a: __m512i) -> i8 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [i8; 64] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0i8, i8::wrapping_add)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi8(-1))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        let count = _mm_cvtsi32_si128(N);
        let shifted = _mm512_sll_epi16(a, count);
        let mask = _mm512_set1_epi8(((0xFFu16 << N) & 0xFF) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        let count = _mm_cvtsi32_si128(N);
        // Sign-extend bytes to 16-bit, shift, mask back to 8-bit
        let lo = _mm512_sra_epi16(_mm512_slli_epi16::<8>(a), count);
        let hi = _mm512_sra_epi16(a, count);
        // Combine: take low byte from lo, high byte from hi
        let mask = _mm512_set1_epi16(0x00FFu16 as i16);
        _mm512_or_si512(
            _mm512_and_si512(_mm512_srli_epi16::<8>(lo), mask),
            _mm512_andnot_si512(mask, hi),
        )
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        let count = _mm_cvtsi32_si128(N);
        let shifted = _mm512_srl_epi16(a, count);
        let mask = _mm512_set1_epi8(((0xFFu16 >> N) & 0xFF) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shl_uniform(self, a: __m512i, count: u32) -> __m512i {
        let shifted = _mm512_sll_epi16(a, _mm_cvtsi32_si128(count as i32));
        let mask = _mm512_set1_epi8(0xFFu8.checked_shl(count).unwrap_or(0) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_logical_uniform(self, a: __m512i, count: u32) -> __m512i {
        let shifted = _mm512_srl_epi16(a, _mm_cvtsi32_si128(count as i32));
        let mask = _mm512_set1_epi8(0xFFu8.checked_shr(count).unwrap_or(0) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_arithmetic_uniform(self, a: __m512i, count: u32) -> __m512i {
        let shifted = _mm512_srl_epi16(a, _mm_cvtsi32_si128(count as i32));
        let byte_mask = _mm512_set1_epi8(0xFFu8.checked_shr(count).unwrap_or(0) as i8);
        let logical = _mm512_and_si512(shifted, byte_mask);
        let sign = _mm512_movm_epi8(_mm512_cmplt_epi8_mask(a, _mm512_setzero_si512()));
        // `count.min(8)` saturates the fill to the whole byte, which
        // is the contracted sign fill for out-of-range counts.
        let fill = _mm512_set1_epi8(((0xFF00u16 >> count.min(8)) & 0xFF) as u8 as i8);
        _mm512_or_si512(logical, _mm512_and_si512(sign, fill))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn saturating_add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_adds_epi8(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn saturating_sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_subs_epi8(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn all_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi8_mask(a, _mm512_setzero_si512());
        mask as u64 == 0xFFFF_FFFF_FFFF_FFFFu64
    }

    #[arcane(shared, _self = X64V4Token)]
    fn any_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi8_mask(a, _mm512_setzero_si512());
        mask as u64 != 0
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitmask(self, a: __m512i) -> u64 {
        // Extract high bit of each lane: compare < 0 for signed interpretation
        let zero = _mm512_setzero_si512();
        _mm512_cmpneq_epi8_mask(_mm512_and_si512(a, _mm512_set1_epi8(1_i8 << (8 - 1))), zero) as u64
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl U8x64Backend for archmage::X64V4Token {
    type Repr = __m512i;

    #[arcane(shared, _self = X64V4Token)]
    fn splat(self, v: u8) -> __m512i {
        _mm512_set1_epi8(v as _)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[arcane(shared, _self = X64V4Token)]
    fn load(self, data: &[u8; 64]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn from_array(self, arr: [u8; 64]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn store(self, repr: __m512i, out: &mut [u8; 64]) {
        crate::simd_storage::store(repr, out);
    }

    #[arcane(shared, _self = X64V4Token)]
    fn to_array(self, repr: __m512i) -> [u8; 64] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi8(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi8(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi8(_mm512_setzero_si512(), a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epu8(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epu8(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epu8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epu8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epu8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epu8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epu8_mask(b, a);
        _mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epu8_mask(b, a);
        _mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi8_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi8(k, if_false, if_true)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn reduce_add(self, a: __m512i) -> u8 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [u8; 64] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0u8, u8::wrapping_add)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi8(-1i8))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        let count = _mm_cvtsi32_si128(N);
        let shifted = _mm512_sll_epi16(a, count);
        let mask = _mm512_set1_epi8(((0xFFu16 << N) & 0xFF) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        let count = _mm_cvtsi32_si128(N);
        let shifted = _mm512_srl_epi16(a, count);
        let byte_mask = _mm512_set1_epi8(((0xFFu16 >> N) & 0xFF) as i8);
        let logical = _mm512_and_si512(shifted, byte_mask);
        // Sign-extend: fill the high N bits of negative lanes. The u16
        // shift keeps the fill mask 0x00 at N == 0 (identity shift).
        let sign = _mm512_movm_epi8(_mm512_cmplt_epi8_mask(a, _mm512_setzero_si512()));
        let fill = _mm512_set1_epi8(((0xFF00u16 >> N) & 0xFF) as i8);
        _mm512_or_si512(logical, _mm512_and_si512(sign, fill))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        let count = _mm_cvtsi32_si128(N);
        let shifted = _mm512_srl_epi16(a, count);
        let mask = _mm512_set1_epi8(((0xFFu16 >> N) & 0xFF) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shl_uniform(self, a: __m512i, count: u32) -> __m512i {
        let shifted = _mm512_sll_epi16(a, _mm_cvtsi32_si128(count as i32));
        let mask = _mm512_set1_epi8(0xFFu8.checked_shl(count).unwrap_or(0) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_logical_uniform(self, a: __m512i, count: u32) -> __m512i {
        let shifted = _mm512_srl_epi16(a, _mm_cvtsi32_si128(count as i32));
        let mask = _mm512_set1_epi8(0xFFu8.checked_shr(count).unwrap_or(0) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_arithmetic_uniform(self, a: __m512i, count: u32) -> __m512i {
        let shifted = _mm512_srl_epi16(a, _mm_cvtsi32_si128(count as i32));
        let mask = _mm512_set1_epi8(0xFFu8.checked_shr(count).unwrap_or(0) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn saturating_add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_adds_epu8(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn saturating_sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_subs_epu8(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn all_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi8_mask(a, _mm512_setzero_si512());
        mask as u64 == 0xFFFF_FFFF_FFFF_FFFFu64
    }

    #[arcane(shared, _self = X64V4Token)]
    fn any_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi8_mask(a, _mm512_setzero_si512());
        mask as u64 != 0
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitmask(self, a: __m512i) -> u64 {
        // Extract high bit of each lane: compare < 0 for signed interpretation
        let zero = _mm512_setzero_si512();
        _mm512_cmpneq_epi8_mask(_mm512_and_si512(a, _mm512_set1_epi8(1_i8 << (8 - 1))), zero) as u64
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I16x32Backend for archmage::X64V4Token {
    type Repr = __m512i;

    #[arcane(shared, _self = X64V4Token)]
    fn splat(self, v: i16) -> __m512i {
        _mm512_set1_epi16(v as _)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[arcane(shared, _self = X64V4Token)]
    fn load(self, data: &[i16; 32]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn from_array(self, arr: [i16; 32]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn store(self, repr: __m512i, out: &mut [i16; 32]) {
        crate::simd_storage::store(repr, out);
    }

    #[arcane(shared, _self = X64V4Token)]
    fn to_array(self, repr: __m512i) -> [i16; 32] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi16(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi16(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn mul(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_mullo_epi16(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi16(_mm512_setzero_si512(), a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epi16(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epi16(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn abs(self, a: __m512i) -> __m512i {
        _mm512_abs_epi16(a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epi16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epi16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epi16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epi16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epi16_mask(b, a);
        _mm512_maskz_set1_epi16(mask, -1)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epi16_mask(b, a);
        _mm512_maskz_set1_epi16(mask, -1)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi16_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi16(k, if_false, if_true)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn reduce_add(self, a: __m512i) -> i16 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [i16; 32] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0i16, i16::wrapping_add)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi16(-1))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sll_epi16(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sra_epi16(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi16(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shl_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_sll_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_logical_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_srl_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_arithmetic_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_sra_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn saturating_add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_adds_epi16(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn saturating_sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_subs_epi16(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn all_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi16_mask(a, _mm512_setzero_si512());
        mask as u64 == 0xFFFF_FFFFu64
    }

    #[arcane(shared, _self = X64V4Token)]
    fn any_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi16_mask(a, _mm512_setzero_si512());
        mask as u64 != 0
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitmask(self, a: __m512i) -> u64 {
        // Extract high bit of each lane: compare < 0 for signed interpretation
        let zero = _mm512_setzero_si512();
        _mm512_cmpneq_epi16_mask(
            _mm512_and_si512(a, _mm512_set1_epi16(1_i16 << (16 - 1))),
            zero,
        ) as u64
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl U16x32Backend for archmage::X64V4Token {
    type Repr = __m512i;

    #[arcane(shared, _self = X64V4Token)]
    fn splat(self, v: u16) -> __m512i {
        _mm512_set1_epi16(v as _)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[arcane(shared, _self = X64V4Token)]
    fn load(self, data: &[u16; 32]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn from_array(self, arr: [u16; 32]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn store(self, repr: __m512i, out: &mut [u16; 32]) {
        crate::simd_storage::store(repr, out);
    }

    #[arcane(shared, _self = X64V4Token)]
    fn to_array(self, repr: __m512i) -> [u16; 32] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi16(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi16(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn mul(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_mullo_epi16(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi16(_mm512_setzero_si512(), a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epu16(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epu16(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epu16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epu16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epu16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epu16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epu16_mask(b, a);
        _mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epu16_mask(b, a);
        _mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi16_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi16(k, if_false, if_true)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn reduce_add(self, a: __m512i) -> u16 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [u16; 32] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0u16, u16::wrapping_add)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi16(-1i16))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sll_epi16(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi16(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi16(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shl_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_sll_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_logical_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_srl_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_arithmetic_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_srl_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn saturating_add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_adds_epu16(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn saturating_sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_subs_epu16(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn all_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi16_mask(a, _mm512_setzero_si512());
        mask as u64 == 0xFFFF_FFFFu64
    }

    #[arcane(shared, _self = X64V4Token)]
    fn any_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi16_mask(a, _mm512_setzero_si512());
        mask as u64 != 0
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitmask(self, a: __m512i) -> u64 {
        // Extract high bit of each lane: compare < 0 for signed interpretation
        let zero = _mm512_setzero_si512();
        _mm512_cmpneq_epi16_mask(
            _mm512_and_si512(a, _mm512_set1_epi16(1_i16 << (16 - 1))),
            zero,
        ) as u64
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I32x16Backend for archmage::X64V4Token {
    type Repr = __m512i;

    #[arcane(shared, _self = X64V4Token)]
    fn splat(self, v: i32) -> __m512i {
        _mm512_set1_epi32(v as _)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[arcane(shared, _self = X64V4Token)]
    fn load(self, data: &[i32; 16]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn from_array(self, arr: [i32; 16]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn store(self, repr: __m512i, out: &mut [i32; 16]) {
        crate::simd_storage::store(repr, out);
    }

    #[arcane(shared, _self = X64V4Token)]
    fn to_array(self, repr: __m512i) -> [i32; 16] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi32(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi32(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn mul(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_mullo_epi32(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi32(_mm512_setzero_si512(), a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epi32(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epi32(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn abs(self, a: __m512i) -> __m512i {
        _mm512_abs_epi32(a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epi32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epi32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epi32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epi32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epi32_mask(b, a);
        _mm512_maskz_set1_epi32(mask, -1)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epi32_mask(b, a);
        _mm512_maskz_set1_epi32(mask, -1)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi32_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi32(k, if_false, if_true)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn reduce_add(self, a: __m512i) -> i32 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [i32; 16] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0i32, i32::wrapping_add)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi32(-1))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sll_epi32(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sra_epi32(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi32(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shl_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_sll_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_logical_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_srl_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_arithmetic_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_sra_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn all_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi32_mask(a, _mm512_setzero_si512());
        mask as u64 == 0xFFFFu64
    }

    #[arcane(shared, _self = X64V4Token)]
    fn any_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi32_mask(a, _mm512_setzero_si512());
        mask as u64 != 0
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitmask(self, a: __m512i) -> u64 {
        // Extract high bit of each lane: compare < 0 for signed interpretation
        let zero = _mm512_setzero_si512();
        _mm512_cmpneq_epi32_mask(
            _mm512_and_si512(a, _mm512_set1_epi32(1_i32 << (32 - 1))),
            zero,
        ) as u64
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl U32x16Backend for archmage::X64V4Token {
    type Repr = __m512i;

    #[arcane(shared, _self = X64V4Token)]
    fn splat(self, v: u32) -> __m512i {
        _mm512_set1_epi32(v as _)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[arcane(shared, _self = X64V4Token)]
    fn load(self, data: &[u32; 16]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn from_array(self, arr: [u32; 16]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn store(self, repr: __m512i, out: &mut [u32; 16]) {
        crate::simd_storage::store(repr, out);
    }

    #[arcane(shared, _self = X64V4Token)]
    fn to_array(self, repr: __m512i) -> [u32; 16] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi32(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi32(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn mul(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_mullo_epi32(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi32(_mm512_setzero_si512(), a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epu32(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epu32(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epu32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1i32)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epu32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1i32)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epu32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1i32)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epu32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1i32)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epu32_mask(b, a);
        _mm512_maskz_set1_epi32(mask, -1i32)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epu32_mask(b, a);
        _mm512_maskz_set1_epi32(mask, -1i32)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi32_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi32(k, if_false, if_true)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn reduce_add(self, a: __m512i) -> u32 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [u32; 16] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0u32, u32::wrapping_add)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi32(-1i32))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sll_epi32(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi32(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi32(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shl_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_sll_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_logical_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_srl_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_arithmetic_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_srl_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn all_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi32_mask(a, _mm512_setzero_si512());
        mask as u64 == 0xFFFFu64
    }

    #[arcane(shared, _self = X64V4Token)]
    fn any_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi32_mask(a, _mm512_setzero_si512());
        mask as u64 != 0
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitmask(self, a: __m512i) -> u64 {
        // Extract high bit of each lane: compare < 0 for signed interpretation
        let zero = _mm512_setzero_si512();
        _mm512_cmpneq_epi32_mask(
            _mm512_and_si512(a, _mm512_set1_epi32(1_i32 << (32 - 1))),
            zero,
        ) as u64
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I64x8Backend for archmage::X64V4Token {
    type Repr = __m512i;

    #[arcane(shared, _self = X64V4Token)]
    fn splat(self, v: i64) -> __m512i {
        _mm512_set1_epi64(v as _)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[arcane(shared, _self = X64V4Token)]
    fn load(self, data: &[i64; 8]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn from_array(self, arr: [i64; 8]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn store(self, repr: __m512i, out: &mut [i64; 8]) {
        crate::simd_storage::store(repr, out);
    }

    #[arcane(shared, _self = X64V4Token)]
    fn to_array(self, repr: __m512i) -> [i64; 8] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi64(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi64(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi64(_mm512_setzero_si512(), a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epi64(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epi64(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn abs(self, a: __m512i) -> __m512i {
        _mm512_abs_epi64(a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epi64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epi64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epi64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epi64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epi64_mask(b, a);
        _mm512_maskz_set1_epi64(mask, -1)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epi64_mask(b, a);
        _mm512_maskz_set1_epi64(mask, -1)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi64_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi64(k, if_false, if_true)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn reduce_add(self, a: __m512i) -> i64 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [i64; 8] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0i64, i64::wrapping_add)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi64(-1))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sll_epi64(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sra_epi64(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi64(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn all_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi64_mask(a, _mm512_setzero_si512());
        mask as u64 == 0xFFu64
    }

    #[arcane(shared, _self = X64V4Token)]
    fn any_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi64_mask(a, _mm512_setzero_si512());
        mask as u64 != 0
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitmask(self, a: __m512i) -> u64 {
        // Extract high bit of each lane: compare < 0 for signed interpretation
        let zero = _mm512_setzero_si512();
        _mm512_cmpneq_epi64_mask(
            _mm512_and_si512(a, _mm512_set1_epi64(1_i64 << (64 - 1))),
            zero,
        ) as u64
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl U64x8Backend for archmage::X64V4Token {
    type Repr = __m512i;

    #[arcane(shared, _self = X64V4Token)]
    fn splat(self, v: u64) -> __m512i {
        _mm512_set1_epi64(v as _)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[arcane(shared, _self = X64V4Token)]
    fn load(self, data: &[u64; 8]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn from_array(self, arr: [u64; 8]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn store(self, repr: __m512i, out: &mut [u64; 8]) {
        crate::simd_storage::store(repr, out);
    }

    #[arcane(shared, _self = X64V4Token)]
    fn to_array(self, repr: __m512i) -> [u64; 8] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi64(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi64(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi64(_mm512_setzero_si512(), a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epu64(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epu64(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epu64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1i64)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epu64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1i64)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epu64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1i64)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epu64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1i64)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epu64_mask(b, a);
        _mm512_maskz_set1_epi64(mask, -1i64)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epu64_mask(b, a);
        _mm512_maskz_set1_epi64(mask, -1i64)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi64_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi64(k, if_false, if_true)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn reduce_add(self, a: __m512i) -> u64 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [u64; 8] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0u64, u64::wrapping_add)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi64(-1i64))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sll_epi64(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi64(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi64(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn all_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi64_mask(a, _mm512_setzero_si512());
        mask as u64 == 0xFFu64
    }

    #[arcane(shared, _self = X64V4Token)]
    fn any_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi64_mask(a, _mm512_setzero_si512());
        mask as u64 != 0
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitmask(self, a: __m512i) -> u64 {
        // Extract high bit of each lane: compare < 0 for signed interpretation
        let zero = _mm512_setzero_si512();
        _mm512_cmpneq_epi64_mask(
            _mm512_and_si512(a, _mm512_set1_epi64(1_i64 << (64 - 1))),
            zero,
        ) as u64
    }
}

// ============================================================================
// X64V4xToken — base AVX-512 (same intrinsics as V4)
// ============================================================================

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl F32x16Backend for archmage::X64V4xToken {
    type Repr = __m512;

    #[arcane(shared, _self = X64V4xToken)]
    fn splat(self, v: f32) -> __m512 {
        _mm512_set1_ps(v)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn zero(self) -> __m512 {
        _mm512_setzero_ps()
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn load(self, data: &[f32; 16]) -> __m512 {
        crate::simd_storage::copy(data)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn from_array(self, arr: [f32; 16]) -> __m512 {
        crate::simd_storage::cast(arr)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn store(self, repr: __m512, out: &mut [f32; 16]) {
        crate::simd_storage::store(repr, out);
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn to_array(self, repr: __m512) -> [f32; 16] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn add(self, a: __m512, b: __m512) -> __m512 {
        _mm512_add_ps(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn sub(self, a: __m512, b: __m512) -> __m512 {
        _mm512_sub_ps(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn mul(self, a: __m512, b: __m512) -> __m512 {
        _mm512_mul_ps(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn div(self, a: __m512, b: __m512) -> __m512 {
        _mm512_div_ps(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn neg(self, a: __m512) -> __m512 {
        _mm512_sub_ps(_mm512_setzero_ps(), a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn min(self, a: __m512, b: __m512) -> __m512 {
        _mm512_min_ps(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn max(self, a: __m512, b: __m512) -> __m512 {
        _mm512_max_ps(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn sqrt(self, a: __m512) -> __m512 {
        _mm512_sqrt_ps(a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn abs(self, a: __m512) -> __m512 {
        let mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFF_FFFFu32 as i32));
        _mm512_and_ps(a, mask)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn floor(self, a: __m512) -> __m512 {
        _mm512_roundscale_ps::<0x01>(a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn ceil(self, a: __m512) -> __m512 {
        _mm512_roundscale_ps::<0x02>(a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn round(self, a: __m512) -> __m512 {
        _mm512_roundscale_ps::<0x00>(a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn mul_add(self, a: __m512, b: __m512, c: __m512) -> __m512 {
        _mm512_fmadd_ps(a, b, c)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn mul_sub(self, a: __m512, b: __m512, c: __m512) -> __m512 {
        _mm512_fmsub_ps(a, b, c)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_eq(self, a: __m512, b: __m512) -> __m512 {
        let mask = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(a, b);
        _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_ne(self, a: __m512, b: __m512) -> __m512 {
        let mask = _mm512_cmp_ps_mask::<_CMP_NEQ_UQ>(a, b);
        _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_lt(self, a: __m512, b: __m512) -> __m512 {
        let mask = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(a, b);
        _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_le(self, a: __m512, b: __m512) -> __m512 {
        let mask = _mm512_cmp_ps_mask::<_CMP_LE_OQ>(a, b);
        _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_gt(self, a: __m512, b: __m512) -> __m512 {
        let mask = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(a, b);
        _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_ge(self, a: __m512, b: __m512) -> __m512 {
        let mask = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(a, b);
        _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn blend(self, mask: __m512, if_true: __m512, if_false: __m512) -> __m512 {
        let mask_i = _mm512_castps_si512(mask);
        let k = _mm512_cmpneq_epi32_mask(mask_i, _mm512_setzero_si512());
        _mm512_mask_blend_ps(k, if_false, if_true)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn reduce_add(self, a: __m512) -> f32 {
        _mm512_reduce_add_ps(a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn reduce_min(self, a: __m512) -> f32 {
        _mm512_reduce_min_ps(a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn reduce_max(self, a: __m512) -> f32 {
        _mm512_reduce_max_ps(a)
    }

    // Raw 14-bit hardware estimate — the cheapest >=12-bit path on AVX-512.
    #[arcane(shared, _self = X64V4xToken)]
    fn rcp_approx(self, a: __m512) -> __m512 {
        _mm512_rcp14_ps(a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn rsqrt_approx(self, a: __m512) -> __m512 {
        _mm512_rsqrt14_ps(a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn recip(self, a: __m512) -> __m512 {
        // token nibbles: QNAN->2 SNAN->2 ZERO->6 ONE->0 -INF->7 +INF->8 NEG->0 POS->0
        const RECIP_FIXUP_TBL: i32 = 0x0087_0622u32 as i32;
        {
            let r = _mm512_rcp14_ps(a);
            let e = _mm512_fnmadd_ps(a, r, _mm512_set1_ps(1.0));
            let refined = _mm512_fmadd_ps(r, e, r);
            _mm512_fixupimm_ps::<0>(refined, a, _mm512_set1_epi32(RECIP_FIXUP_TBL))
        }
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn rsqrt(self, a: __m512) -> __m512 {
        // token nibbles: QNAN->2 SNAN->2 ZERO->6 ONE->0 -INF->3 +INF->8 NEG->3 POS->0
        const RSQRT_FIXUP_TBL: i32 = 0x0383_0622u32 as i32;
        {
            let y = _mm512_rsqrt14_ps(a);
            let t = _mm512_mul_ps(a, _mm512_mul_ps(y, y));
            let refined = _mm512_mul_ps(
                y,
                _mm512_fnmadd_ps(_mm512_set1_ps(0.5), t, _mm512_set1_ps(1.5)),
            );
            _mm512_fixupimm_ps::<0>(refined, a, _mm512_set1_epi32(RSQRT_FIXUP_TBL))
        }
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn not(self, a: __m512) -> __m512 {
        let all_ones = _mm512_castsi512_ps(_mm512_set1_epi32(-1));
        _mm512_xor_ps(a, all_ones)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitand(self, a: __m512, b: __m512) -> __m512 {
        _mm512_and_ps(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitor(self, a: __m512, b: __m512) -> __m512 {
        _mm512_or_ps(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitxor(self, a: __m512, b: __m512) -> __m512 {
        _mm512_xor_ps(a, b)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl F64x8Backend for archmage::X64V4xToken {
    type Repr = __m512d;

    #[arcane(shared, _self = X64V4xToken)]
    fn splat(self, v: f64) -> __m512d {
        _mm512_set1_pd(v)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn zero(self) -> __m512d {
        _mm512_setzero_pd()
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn load(self, data: &[f64; 8]) -> __m512d {
        crate::simd_storage::copy(data)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn from_array(self, arr: [f64; 8]) -> __m512d {
        crate::simd_storage::cast(arr)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn store(self, repr: __m512d, out: &mut [f64; 8]) {
        crate::simd_storage::store(repr, out);
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn to_array(self, repr: __m512d) -> [f64; 8] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn add(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_add_pd(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn sub(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_sub_pd(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn mul(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_mul_pd(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn div(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_div_pd(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn neg(self, a: __m512d) -> __m512d {
        _mm512_sub_pd(_mm512_setzero_pd(), a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn min(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_min_pd(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn max(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_max_pd(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn sqrt(self, a: __m512d) -> __m512d {
        _mm512_sqrt_pd(a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn abs(self, a: __m512d) -> __m512d {
        let mask = _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFF_FFFF_FFFF_FFFFu64 as i64));
        _mm512_and_pd(a, mask)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn floor(self, a: __m512d) -> __m512d {
        _mm512_roundscale_pd::<0x01>(a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn ceil(self, a: __m512d) -> __m512d {
        _mm512_roundscale_pd::<0x02>(a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn round(self, a: __m512d) -> __m512d {
        _mm512_roundscale_pd::<0x00>(a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn mul_add(self, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        _mm512_fmadd_pd(a, b, c)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn mul_sub(self, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        _mm512_fmsub_pd(a, b, c)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_eq(self, a: __m512d, b: __m512d) -> __m512d {
        let mask = _mm512_cmp_pd_mask::<_CMP_EQ_OQ>(a, b);
        _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_ne(self, a: __m512d, b: __m512d) -> __m512d {
        let mask = _mm512_cmp_pd_mask::<_CMP_NEQ_UQ>(a, b);
        _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_lt(self, a: __m512d, b: __m512d) -> __m512d {
        let mask = _mm512_cmp_pd_mask::<_CMP_LT_OQ>(a, b);
        _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_le(self, a: __m512d, b: __m512d) -> __m512d {
        let mask = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(a, b);
        _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_gt(self, a: __m512d, b: __m512d) -> __m512d {
        let mask = _mm512_cmp_pd_mask::<_CMP_GT_OQ>(a, b);
        _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_ge(self, a: __m512d, b: __m512d) -> __m512d {
        let mask = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(a, b);
        _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn blend(self, mask: __m512d, if_true: __m512d, if_false: __m512d) -> __m512d {
        let mask_i = _mm512_castpd_si512(mask);
        let k = _mm512_cmpneq_epi64_mask(mask_i, _mm512_setzero_si512());
        _mm512_mask_blend_pd(k, if_false, if_true)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn reduce_add(self, a: __m512d) -> f64 {
        _mm512_reduce_add_pd(a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn reduce_min(self, a: __m512d) -> f64 {
        _mm512_reduce_min_pd(a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn reduce_max(self, a: __m512d) -> f64 {
        _mm512_reduce_max_pd(a)
    }

    // Raw 14-bit hardware estimate — the cheapest >=12-bit path on AVX-512.
    #[arcane(shared, _self = X64V4xToken)]
    fn rcp_approx(self, a: __m512d) -> __m512d {
        _mm512_rcp14_pd(a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn rsqrt_approx(self, a: __m512d) -> __m512d {
        _mm512_rsqrt14_pd(a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn recip(self, a: __m512d) -> __m512d {
        _mm512_div_pd(_mm512_set1_pd(1.0), a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn rsqrt(self, a: __m512d) -> __m512d {
        _mm512_div_pd(_mm512_set1_pd(1.0), _mm512_sqrt_pd(a))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn not(self, a: __m512d) -> __m512d {
        let all_ones = _mm512_castsi512_pd(_mm512_set1_epi64(-1));
        _mm512_xor_pd(a, all_ones)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitand(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_and_pd(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitor(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_or_pd(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitxor(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_xor_pd(a, b)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I8x64Backend for archmage::X64V4xToken {
    type Repr = __m512i;

    #[arcane(shared, _self = X64V4xToken)]
    fn splat(self, v: i8) -> __m512i {
        _mm512_set1_epi8(v as _)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn load(self, data: &[i8; 64]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn from_array(self, arr: [i8; 64]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn store(self, repr: __m512i, out: &mut [i8; 64]) {
        crate::simd_storage::store(repr, out);
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn to_array(self, repr: __m512i) -> [i8; 64] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi8(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi8(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi8(_mm512_setzero_si512(), a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epi8(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epi8(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn abs(self, a: __m512i) -> __m512i {
        _mm512_abs_epi8(a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epi8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epi8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epi8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epi8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epi8_mask(b, a);
        _mm512_maskz_set1_epi8(mask, -1)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epi8_mask(b, a);
        _mm512_maskz_set1_epi8(mask, -1)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi8_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi8(k, if_false, if_true)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn reduce_add(self, a: __m512i) -> i8 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [i8; 64] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0i8, i8::wrapping_add)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi8(-1))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        let count = _mm_cvtsi32_si128(N);
        let shifted = _mm512_sll_epi16(a, count);
        let mask = _mm512_set1_epi8(((0xFFu16 << N) & 0xFF) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        let count = _mm_cvtsi32_si128(N);
        // Sign-extend bytes to 16-bit, shift, mask back to 8-bit
        let lo = _mm512_sra_epi16(_mm512_slli_epi16::<8>(a), count);
        let hi = _mm512_sra_epi16(a, count);
        // Combine: take low byte from lo, high byte from hi
        let mask = _mm512_set1_epi16(0x00FFu16 as i16);
        _mm512_or_si512(
            _mm512_and_si512(_mm512_srli_epi16::<8>(lo), mask),
            _mm512_andnot_si512(mask, hi),
        )
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        let count = _mm_cvtsi32_si128(N);
        let shifted = _mm512_srl_epi16(a, count);
        let mask = _mm512_set1_epi8(((0xFFu16 >> N) & 0xFF) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shl_uniform(self, a: __m512i, count: u32) -> __m512i {
        let shifted = _mm512_sll_epi16(a, _mm_cvtsi32_si128(count as i32));
        let mask = _mm512_set1_epi8(0xFFu8.checked_shl(count).unwrap_or(0) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_logical_uniform(self, a: __m512i, count: u32) -> __m512i {
        let shifted = _mm512_srl_epi16(a, _mm_cvtsi32_si128(count as i32));
        let mask = _mm512_set1_epi8(0xFFu8.checked_shr(count).unwrap_or(0) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_arithmetic_uniform(self, a: __m512i, count: u32) -> __m512i {
        let shifted = _mm512_srl_epi16(a, _mm_cvtsi32_si128(count as i32));
        let byte_mask = _mm512_set1_epi8(0xFFu8.checked_shr(count).unwrap_or(0) as i8);
        let logical = _mm512_and_si512(shifted, byte_mask);
        let sign = _mm512_movm_epi8(_mm512_cmplt_epi8_mask(a, _mm512_setzero_si512()));
        // `count.min(8)` saturates the fill to the whole byte, which
        // is the contracted sign fill for out-of-range counts.
        let fill = _mm512_set1_epi8(((0xFF00u16 >> count.min(8)) & 0xFF) as u8 as i8);
        _mm512_or_si512(logical, _mm512_and_si512(sign, fill))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn saturating_add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_adds_epi8(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn saturating_sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_subs_epi8(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn all_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi8_mask(a, _mm512_setzero_si512());
        mask as u64 == 0xFFFF_FFFF_FFFF_FFFFu64
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn any_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi8_mask(a, _mm512_setzero_si512());
        mask as u64 != 0
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitmask(self, a: __m512i) -> u64 {
        // Extract high bit of each lane: compare < 0 for signed interpretation
        let zero = _mm512_setzero_si512();
        _mm512_cmpneq_epi8_mask(_mm512_and_si512(a, _mm512_set1_epi8(1_i8 << (8 - 1))), zero) as u64
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl U8x64Backend for archmage::X64V4xToken {
    type Repr = __m512i;

    #[arcane(shared, _self = X64V4xToken)]
    fn splat(self, v: u8) -> __m512i {
        _mm512_set1_epi8(v as _)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn load(self, data: &[u8; 64]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn from_array(self, arr: [u8; 64]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn store(self, repr: __m512i, out: &mut [u8; 64]) {
        crate::simd_storage::store(repr, out);
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn to_array(self, repr: __m512i) -> [u8; 64] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi8(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi8(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi8(_mm512_setzero_si512(), a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epu8(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epu8(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epu8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epu8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epu8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epu8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epu8_mask(b, a);
        _mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epu8_mask(b, a);
        _mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi8_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi8(k, if_false, if_true)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn reduce_add(self, a: __m512i) -> u8 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [u8; 64] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0u8, u8::wrapping_add)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi8(-1i8))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        let count = _mm_cvtsi32_si128(N);
        let shifted = _mm512_sll_epi16(a, count);
        let mask = _mm512_set1_epi8(((0xFFu16 << N) & 0xFF) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        let count = _mm_cvtsi32_si128(N);
        let shifted = _mm512_srl_epi16(a, count);
        let byte_mask = _mm512_set1_epi8(((0xFFu16 >> N) & 0xFF) as i8);
        let logical = _mm512_and_si512(shifted, byte_mask);
        // Sign-extend: fill the high N bits of negative lanes. The u16
        // shift keeps the fill mask 0x00 at N == 0 (identity shift).
        let sign = _mm512_movm_epi8(_mm512_cmplt_epi8_mask(a, _mm512_setzero_si512()));
        let fill = _mm512_set1_epi8(((0xFF00u16 >> N) & 0xFF) as i8);
        _mm512_or_si512(logical, _mm512_and_si512(sign, fill))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        let count = _mm_cvtsi32_si128(N);
        let shifted = _mm512_srl_epi16(a, count);
        let mask = _mm512_set1_epi8(((0xFFu16 >> N) & 0xFF) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shl_uniform(self, a: __m512i, count: u32) -> __m512i {
        let shifted = _mm512_sll_epi16(a, _mm_cvtsi32_si128(count as i32));
        let mask = _mm512_set1_epi8(0xFFu8.checked_shl(count).unwrap_or(0) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_logical_uniform(self, a: __m512i, count: u32) -> __m512i {
        let shifted = _mm512_srl_epi16(a, _mm_cvtsi32_si128(count as i32));
        let mask = _mm512_set1_epi8(0xFFu8.checked_shr(count).unwrap_or(0) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_arithmetic_uniform(self, a: __m512i, count: u32) -> __m512i {
        let shifted = _mm512_srl_epi16(a, _mm_cvtsi32_si128(count as i32));
        let mask = _mm512_set1_epi8(0xFFu8.checked_shr(count).unwrap_or(0) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn saturating_add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_adds_epu8(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn saturating_sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_subs_epu8(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn all_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi8_mask(a, _mm512_setzero_si512());
        mask as u64 == 0xFFFF_FFFF_FFFF_FFFFu64
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn any_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi8_mask(a, _mm512_setzero_si512());
        mask as u64 != 0
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitmask(self, a: __m512i) -> u64 {
        // Extract high bit of each lane: compare < 0 for signed interpretation
        let zero = _mm512_setzero_si512();
        _mm512_cmpneq_epi8_mask(_mm512_and_si512(a, _mm512_set1_epi8(1_i8 << (8 - 1))), zero) as u64
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I16x32Backend for archmage::X64V4xToken {
    type Repr = __m512i;

    #[arcane(shared, _self = X64V4xToken)]
    fn splat(self, v: i16) -> __m512i {
        _mm512_set1_epi16(v as _)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn load(self, data: &[i16; 32]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn from_array(self, arr: [i16; 32]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn store(self, repr: __m512i, out: &mut [i16; 32]) {
        crate::simd_storage::store(repr, out);
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn to_array(self, repr: __m512i) -> [i16; 32] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi16(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi16(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn mul(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_mullo_epi16(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi16(_mm512_setzero_si512(), a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epi16(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epi16(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn abs(self, a: __m512i) -> __m512i {
        _mm512_abs_epi16(a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epi16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epi16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epi16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epi16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epi16_mask(b, a);
        _mm512_maskz_set1_epi16(mask, -1)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epi16_mask(b, a);
        _mm512_maskz_set1_epi16(mask, -1)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi16_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi16(k, if_false, if_true)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn reduce_add(self, a: __m512i) -> i16 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [i16; 32] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0i16, i16::wrapping_add)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi16(-1))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sll_epi16(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sra_epi16(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi16(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shl_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_sll_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_logical_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_srl_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_arithmetic_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_sra_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn saturating_add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_adds_epi16(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn saturating_sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_subs_epi16(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn all_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi16_mask(a, _mm512_setzero_si512());
        mask as u64 == 0xFFFF_FFFFu64
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn any_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi16_mask(a, _mm512_setzero_si512());
        mask as u64 != 0
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitmask(self, a: __m512i) -> u64 {
        // Extract high bit of each lane: compare < 0 for signed interpretation
        let zero = _mm512_setzero_si512();
        _mm512_cmpneq_epi16_mask(
            _mm512_and_si512(a, _mm512_set1_epi16(1_i16 << (16 - 1))),
            zero,
        ) as u64
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl U16x32Backend for archmage::X64V4xToken {
    type Repr = __m512i;

    #[arcane(shared, _self = X64V4xToken)]
    fn splat(self, v: u16) -> __m512i {
        _mm512_set1_epi16(v as _)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn load(self, data: &[u16; 32]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn from_array(self, arr: [u16; 32]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn store(self, repr: __m512i, out: &mut [u16; 32]) {
        crate::simd_storage::store(repr, out);
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn to_array(self, repr: __m512i) -> [u16; 32] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi16(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi16(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn mul(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_mullo_epi16(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi16(_mm512_setzero_si512(), a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epu16(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epu16(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epu16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epu16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epu16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epu16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epu16_mask(b, a);
        _mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epu16_mask(b, a);
        _mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi16_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi16(k, if_false, if_true)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn reduce_add(self, a: __m512i) -> u16 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [u16; 32] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0u16, u16::wrapping_add)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi16(-1i16))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sll_epi16(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi16(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi16(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shl_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_sll_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_logical_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_srl_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_arithmetic_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_srl_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn saturating_add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_adds_epu16(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn saturating_sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_subs_epu16(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn all_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi16_mask(a, _mm512_setzero_si512());
        mask as u64 == 0xFFFF_FFFFu64
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn any_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi16_mask(a, _mm512_setzero_si512());
        mask as u64 != 0
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitmask(self, a: __m512i) -> u64 {
        // Extract high bit of each lane: compare < 0 for signed interpretation
        let zero = _mm512_setzero_si512();
        _mm512_cmpneq_epi16_mask(
            _mm512_and_si512(a, _mm512_set1_epi16(1_i16 << (16 - 1))),
            zero,
        ) as u64
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I32x16Backend for archmage::X64V4xToken {
    type Repr = __m512i;

    #[arcane(shared, _self = X64V4xToken)]
    fn splat(self, v: i32) -> __m512i {
        _mm512_set1_epi32(v as _)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn load(self, data: &[i32; 16]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn from_array(self, arr: [i32; 16]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn store(self, repr: __m512i, out: &mut [i32; 16]) {
        crate::simd_storage::store(repr, out);
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn to_array(self, repr: __m512i) -> [i32; 16] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi32(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi32(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn mul(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_mullo_epi32(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi32(_mm512_setzero_si512(), a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epi32(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epi32(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn abs(self, a: __m512i) -> __m512i {
        _mm512_abs_epi32(a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epi32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epi32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epi32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epi32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epi32_mask(b, a);
        _mm512_maskz_set1_epi32(mask, -1)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epi32_mask(b, a);
        _mm512_maskz_set1_epi32(mask, -1)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi32_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi32(k, if_false, if_true)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn reduce_add(self, a: __m512i) -> i32 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [i32; 16] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0i32, i32::wrapping_add)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi32(-1))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sll_epi32(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sra_epi32(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi32(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shl_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_sll_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_logical_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_srl_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_arithmetic_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_sra_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn all_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi32_mask(a, _mm512_setzero_si512());
        mask as u64 == 0xFFFFu64
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn any_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi32_mask(a, _mm512_setzero_si512());
        mask as u64 != 0
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitmask(self, a: __m512i) -> u64 {
        // Extract high bit of each lane: compare < 0 for signed interpretation
        let zero = _mm512_setzero_si512();
        _mm512_cmpneq_epi32_mask(
            _mm512_and_si512(a, _mm512_set1_epi32(1_i32 << (32 - 1))),
            zero,
        ) as u64
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl U32x16Backend for archmage::X64V4xToken {
    type Repr = __m512i;

    #[arcane(shared, _self = X64V4xToken)]
    fn splat(self, v: u32) -> __m512i {
        _mm512_set1_epi32(v as _)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn load(self, data: &[u32; 16]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn from_array(self, arr: [u32; 16]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn store(self, repr: __m512i, out: &mut [u32; 16]) {
        crate::simd_storage::store(repr, out);
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn to_array(self, repr: __m512i) -> [u32; 16] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi32(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi32(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn mul(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_mullo_epi32(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi32(_mm512_setzero_si512(), a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epu32(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epu32(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epu32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1i32)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epu32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1i32)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epu32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1i32)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epu32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1i32)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epu32_mask(b, a);
        _mm512_maskz_set1_epi32(mask, -1i32)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epu32_mask(b, a);
        _mm512_maskz_set1_epi32(mask, -1i32)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi32_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi32(k, if_false, if_true)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn reduce_add(self, a: __m512i) -> u32 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [u32; 16] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0u32, u32::wrapping_add)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi32(-1i32))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sll_epi32(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi32(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi32(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shl_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_sll_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_logical_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_srl_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_arithmetic_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_srl_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn all_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi32_mask(a, _mm512_setzero_si512());
        mask as u64 == 0xFFFFu64
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn any_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi32_mask(a, _mm512_setzero_si512());
        mask as u64 != 0
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitmask(self, a: __m512i) -> u64 {
        // Extract high bit of each lane: compare < 0 for signed interpretation
        let zero = _mm512_setzero_si512();
        _mm512_cmpneq_epi32_mask(
            _mm512_and_si512(a, _mm512_set1_epi32(1_i32 << (32 - 1))),
            zero,
        ) as u64
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I64x8Backend for archmage::X64V4xToken {
    type Repr = __m512i;

    #[arcane(shared, _self = X64V4xToken)]
    fn splat(self, v: i64) -> __m512i {
        _mm512_set1_epi64(v as _)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn load(self, data: &[i64; 8]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn from_array(self, arr: [i64; 8]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn store(self, repr: __m512i, out: &mut [i64; 8]) {
        crate::simd_storage::store(repr, out);
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn to_array(self, repr: __m512i) -> [i64; 8] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi64(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi64(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi64(_mm512_setzero_si512(), a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epi64(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epi64(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn abs(self, a: __m512i) -> __m512i {
        _mm512_abs_epi64(a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epi64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epi64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epi64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epi64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epi64_mask(b, a);
        _mm512_maskz_set1_epi64(mask, -1)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epi64_mask(b, a);
        _mm512_maskz_set1_epi64(mask, -1)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi64_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi64(k, if_false, if_true)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn reduce_add(self, a: __m512i) -> i64 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [i64; 8] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0i64, i64::wrapping_add)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi64(-1))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sll_epi64(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sra_epi64(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi64(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn all_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi64_mask(a, _mm512_setzero_si512());
        mask as u64 == 0xFFu64
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn any_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi64_mask(a, _mm512_setzero_si512());
        mask as u64 != 0
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitmask(self, a: __m512i) -> u64 {
        // Extract high bit of each lane: compare < 0 for signed interpretation
        let zero = _mm512_setzero_si512();
        _mm512_cmpneq_epi64_mask(
            _mm512_and_si512(a, _mm512_set1_epi64(1_i64 << (64 - 1))),
            zero,
        ) as u64
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl U64x8Backend for archmage::X64V4xToken {
    type Repr = __m512i;

    #[arcane(shared, _self = X64V4xToken)]
    fn splat(self, v: u64) -> __m512i {
        _mm512_set1_epi64(v as _)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn load(self, data: &[u64; 8]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn from_array(self, arr: [u64; 8]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn store(self, repr: __m512i, out: &mut [u64; 8]) {
        crate::simd_storage::store(repr, out);
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn to_array(self, repr: __m512i) -> [u64; 8] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi64(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi64(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi64(_mm512_setzero_si512(), a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epu64(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epu64(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epu64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1i64)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epu64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1i64)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epu64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1i64)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epu64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1i64)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epu64_mask(b, a);
        _mm512_maskz_set1_epi64(mask, -1i64)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epu64_mask(b, a);
        _mm512_maskz_set1_epi64(mask, -1i64)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi64_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi64(k, if_false, if_true)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn reduce_add(self, a: __m512i) -> u64 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [u64; 8] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0u64, u64::wrapping_add)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi64(-1i64))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sll_epi64(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi64(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi64(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn all_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi64_mask(a, _mm512_setzero_si512());
        mask as u64 == 0xFFu64
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn any_true(self, a: __m512i) -> bool {
        let mask = _mm512_cmpneq_epi64_mask(a, _mm512_setzero_si512());
        mask as u64 != 0
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitmask(self, a: __m512i) -> u64 {
        // Extract high bit of each lane: compare < 0 for signed interpretation
        let zero = _mm512_setzero_si512();
        _mm512_cmpneq_epi64_mask(
            _mm512_and_si512(a, _mm512_set1_epi64(1_i64 << (64 - 1))),
            zero,
        ) as u64
    }
}

// ============================================================================
// X64V4xToken — extension: popcnt (VPOPCNTDQ + BITALG)
// ============================================================================

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl i8x64PopcntBackend for archmage::X64V4xToken {
    #[arcane(shared, _self = X64V4xToken)]
    fn popcnt(self, a: __m512i) -> __m512i {
        _mm512_popcnt_epi8(a)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl u8x64PopcntBackend for archmage::X64V4xToken {
    #[arcane(shared, _self = X64V4xToken)]
    fn popcnt(self, a: __m512i) -> __m512i {
        _mm512_popcnt_epi8(a)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl i16x32PopcntBackend for archmage::X64V4xToken {
    #[arcane(shared, _self = X64V4xToken)]
    fn popcnt(self, a: __m512i) -> __m512i {
        _mm512_popcnt_epi16(a)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl u16x32PopcntBackend for archmage::X64V4xToken {
    #[arcane(shared, _self = X64V4xToken)]
    fn popcnt(self, a: __m512i) -> __m512i {
        _mm512_popcnt_epi16(a)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl i32x16PopcntBackend for archmage::X64V4xToken {
    #[arcane(shared, _self = X64V4xToken)]
    fn popcnt(self, a: __m512i) -> __m512i {
        _mm512_popcnt_epi32(a)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl u32x16PopcntBackend for archmage::X64V4xToken {
    #[arcane(shared, _self = X64V4xToken)]
    fn popcnt(self, a: __m512i) -> __m512i {
        _mm512_popcnt_epi32(a)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl i64x8PopcntBackend for archmage::X64V4xToken {
    #[arcane(shared, _self = X64V4xToken)]
    fn popcnt(self, a: __m512i) -> __m512i {
        _mm512_popcnt_epi64(a)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl u64x8PopcntBackend for archmage::X64V4xToken {
    #[arcane(shared, _self = X64V4xToken)]
    fn popcnt(self, a: __m512i) -> __m512i {
        _mm512_popcnt_epi64(a)
    }
}

// ============================================================================
// F32x16Convert — native AVX-512 conversions
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "w512"))]
#[cfg(feature = "w512")]
impl F32x16Convert for archmage::X64V4Token {
    #[arcane(shared, _self = X64V4Token)]
    fn bitcast_f32_to_i32(self, a: __m512) -> __m512i {
        _mm512_castps_si512(a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn bitcast_i32_to_f32(self, a: __m512i) -> __m512 {
        _mm512_castsi512_ps(a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn convert_f32_to_i32(self, a: __m512) -> __m512i {
        _mm512_cvttps_epi32(a)
    }

    // Issue #80 fixup, mask-register form.
    #[arcane(shared, _self = X64V4Token)]
    fn convert_f32_to_i32_saturating(self, a: __m512) -> __m512i {
        let t = _mm512_cvttps_epi32(a);
        let big = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(a, _mm512_set1_ps(2_147_483_648.0));
        let t = _mm512_mask_mov_epi32(t, big, _mm512_set1_epi32(i32::MAX));
        let not_nan = _mm512_cmp_ps_mask::<_CMP_ORD_Q>(a, a);
        _mm512_maskz_mov_epi32(not_nan, t)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn convert_f32_to_i32_round(self, a: __m512) -> __m512i {
        _mm512_cvtps_epi32(a)
    }

    #[arcane(shared, _self = X64V4Token)]
    fn convert_i32_to_f32(self, a: __m512i) -> __m512 {
        _mm512_cvtepi32_ps(a)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "w512"))]
#[cfg(feature = "w512")]
impl F32x16Convert for archmage::X64V4xToken {
    #[arcane(shared, _self = X64V4xToken)]
    fn bitcast_f32_to_i32(self, a: __m512) -> __m512i {
        _mm512_castps_si512(a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn bitcast_i32_to_f32(self, a: __m512i) -> __m512 {
        _mm512_castsi512_ps(a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn convert_f32_to_i32(self, a: __m512) -> __m512i {
        _mm512_cvttps_epi32(a)
    }

    // Issue #80 fixup, mask-register form.
    #[arcane(shared, _self = X64V4xToken)]
    fn convert_f32_to_i32_saturating(self, a: __m512) -> __m512i {
        let t = _mm512_cvttps_epi32(a);
        let big = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(a, _mm512_set1_ps(2_147_483_648.0));
        let t = _mm512_mask_mov_epi32(t, big, _mm512_set1_epi32(i32::MAX));
        let not_nan = _mm512_cmp_ps_mask::<_CMP_ORD_Q>(a, a);
        _mm512_maskz_mov_epi32(not_nan, t)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn convert_f32_to_i32_round(self, a: __m512) -> __m512i {
        _mm512_cvtps_epi32(a)
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn convert_i32_to_f32(self, a: __m512i) -> __m512 {
        _mm512_cvtepi32_ps(a)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl U8x64Widen for archmage::X64V4Token {
    #[arcane(shared, _self = X64V4Token)]
    fn widen_low_u8_to_u16(self, a: __m512i) -> __m512i {
        _mm512_cvtepu8_epi16(_mm512_castsi512_si256(a))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn widen_high_u8_to_u16(self, a: __m512i) -> __m512i {
        _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(a))
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl U16x32Widen for archmage::X64V4Token {
    #[arcane(shared, _self = X64V4Token)]
    fn widen_low_u16_to_u32(self, a: __m512i) -> __m512i {
        _mm512_cvtepu16_epi32(_mm512_castsi512_si256(a))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn widen_high_u16_to_u32(self, a: __m512i) -> __m512i {
        _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64::<1>(a))
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I8x64Widen for archmage::X64V4Token {
    #[arcane(shared, _self = X64V4Token)]
    fn widen_low_i8_to_i16(self, a: __m512i) -> __m512i {
        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(a))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn widen_high_i8_to_i16(self, a: __m512i) -> __m512i {
        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64::<1>(a))
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I16x32Widen for archmage::X64V4Token {
    #[arcane(shared, _self = X64V4Token)]
    fn widen_low_i16_to_i32(self, a: __m512i) -> __m512i {
        _mm512_cvtepi16_epi32(_mm512_castsi512_si256(a))
    }

    #[arcane(shared, _self = X64V4Token)]
    fn widen_high_i16_to_i32(self, a: __m512i) -> __m512i {
        _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(a))
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I16x32Narrow for archmage::X64V4Token {
    #[arcane(shared, _self = X64V4Token)]
    fn narrow_saturating_i16_to_i8(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_inserti64x4::<1>(
            _mm512_castsi256_si512(_mm512_cvtsepi16_epi8(a)),
            _mm512_cvtsepi16_epi8(b),
        )
    }

    #[arcane(shared, _self = X64V4Token)]
    fn narrow_saturating_i16_to_u8(self, a: __m512i, b: __m512i) -> __m512i {
        // Clamp at zero so the unsigned-source instruction sees
        // the same value the signed source held; above the
        // clamp both readings agree.
        let zero = _mm512_setzero_si512();
        _mm512_inserti64x4::<1>(
            _mm512_castsi256_si512(_mm512_cvtusepi16_epi8(_mm512_max_epi16(a, zero))),
            _mm512_cvtusepi16_epi8(_mm512_max_epi16(b, zero)),
        )
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I32x16Narrow for archmage::X64V4Token {
    #[arcane(shared, _self = X64V4Token)]
    fn narrow_saturating_i32_to_i16(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_inserti64x4::<1>(
            _mm512_castsi256_si512(_mm512_cvtsepi32_epi16(a)),
            _mm512_cvtsepi32_epi16(b),
        )
    }

    #[arcane(shared, _self = X64V4Token)]
    fn narrow_saturating_i32_to_u16(self, a: __m512i, b: __m512i) -> __m512i {
        // Clamp at zero so the unsigned-source instruction sees
        // the same value the signed source held; above the
        // clamp both readings agree.
        let zero = _mm512_setzero_si512();
        _mm512_inserti64x4::<1>(
            _mm512_castsi256_si512(_mm512_cvtusepi32_epi16(_mm512_max_epi32(a, zero))),
            _mm512_cvtusepi32_epi16(_mm512_max_epi32(b, zero)),
        )
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl U8x64Widen for archmage::X64V4xToken {
    #[arcane(shared, _self = X64V4xToken)]
    fn widen_low_u8_to_u16(self, a: __m512i) -> __m512i {
        _mm512_cvtepu8_epi16(_mm512_castsi512_si256(a))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn widen_high_u8_to_u16(self, a: __m512i) -> __m512i {
        _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(a))
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl U16x32Widen for archmage::X64V4xToken {
    #[arcane(shared, _self = X64V4xToken)]
    fn widen_low_u16_to_u32(self, a: __m512i) -> __m512i {
        _mm512_cvtepu16_epi32(_mm512_castsi512_si256(a))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn widen_high_u16_to_u32(self, a: __m512i) -> __m512i {
        _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64::<1>(a))
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I8x64Widen for archmage::X64V4xToken {
    #[arcane(shared, _self = X64V4xToken)]
    fn widen_low_i8_to_i16(self, a: __m512i) -> __m512i {
        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(a))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn widen_high_i8_to_i16(self, a: __m512i) -> __m512i {
        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64::<1>(a))
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I16x32Widen for archmage::X64V4xToken {
    #[arcane(shared, _self = X64V4xToken)]
    fn widen_low_i16_to_i32(self, a: __m512i) -> __m512i {
        _mm512_cvtepi16_epi32(_mm512_castsi512_si256(a))
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn widen_high_i16_to_i32(self, a: __m512i) -> __m512i {
        _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(a))
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I16x32Narrow for archmage::X64V4xToken {
    #[arcane(shared, _self = X64V4xToken)]
    fn narrow_saturating_i16_to_i8(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_inserti64x4::<1>(
            _mm512_castsi256_si512(_mm512_cvtsepi16_epi8(a)),
            _mm512_cvtsepi16_epi8(b),
        )
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn narrow_saturating_i16_to_u8(self, a: __m512i, b: __m512i) -> __m512i {
        // Clamp at zero so the unsigned-source instruction sees
        // the same value the signed source held; above the
        // clamp both readings agree.
        let zero = _mm512_setzero_si512();
        _mm512_inserti64x4::<1>(
            _mm512_castsi256_si512(_mm512_cvtusepi16_epi8(_mm512_max_epi16(a, zero))),
            _mm512_cvtusepi16_epi8(_mm512_max_epi16(b, zero)),
        )
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I32x16Narrow for archmage::X64V4xToken {
    #[arcane(shared, _self = X64V4xToken)]
    fn narrow_saturating_i32_to_i16(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_inserti64x4::<1>(
            _mm512_castsi256_si512(_mm512_cvtsepi32_epi16(a)),
            _mm512_cvtsepi32_epi16(b),
        )
    }

    #[arcane(shared, _self = X64V4xToken)]
    fn narrow_saturating_i32_to_u16(self, a: __m512i, b: __m512i) -> __m512i {
        // Clamp at zero so the unsigned-source instruction sees
        // the same value the signed source held; above the
        // clamp both readings agree.
        let zero = _mm512_setzero_si512();
        _mm512_inserti64x4::<1>(
            _mm512_castsi256_si512(_mm512_cvtusepi32_epi16(_mm512_max_epi32(a, zero))),
            _mm512_cvtusepi32_epi16(_mm512_max_epi32(b, zero)),
        )
    }
}

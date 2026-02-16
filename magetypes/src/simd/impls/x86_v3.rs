//! Backend implementations for X64V3Token (x86-64).
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::simd::backends::*;

#[cfg(target_arch = "x86_64")]
impl F32x4Backend for archmage::X64V3Token {
    type Repr = __m128;

    // ====== Construction ======

    #[inline(always)]
    fn splat(v: f32) -> __m128 {
        unsafe { _mm_set1_ps(v) }
    }

    #[inline(always)]
    fn zero() -> __m128 {
        unsafe { _mm_setzero_ps() }
    }

    #[inline(always)]
    fn load(data: &[f32; 4]) -> __m128 {
        unsafe { _mm_loadu_ps(data.as_ptr()) }
    }

    #[inline(always)]
    fn from_array(arr: [f32; 4]) -> __m128 {
        // SAFETY: [f32; 4] and __m128 have identical size and layout.
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m128, out: &mut [f32; 4]) {
        unsafe { _mm_storeu_ps(out.as_mut_ptr(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: __m128) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        unsafe { _mm_storeu_ps(out.as_mut_ptr(), repr) };
        out
    }

    // ====== Arithmetic ======

    #[inline(always)]
    fn add(a: __m128, b: __m128) -> __m128 {
        unsafe { _mm_add_ps(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m128, b: __m128) -> __m128 {
        unsafe { _mm_sub_ps(a, b) }
    }

    #[inline(always)]
    fn mul(a: __m128, b: __m128) -> __m128 {
        unsafe { _mm_mul_ps(a, b) }
    }

    #[inline(always)]
    fn div(a: __m128, b: __m128) -> __m128 {
        unsafe { _mm_div_ps(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m128) -> __m128 {
        unsafe { _mm_sub_ps(_mm_setzero_ps(), a) }
    }

    // ====== Math ======

    #[inline(always)]
    fn min(a: __m128, b: __m128) -> __m128 {
        unsafe { _mm_min_ps(a, b) }
    }

    #[inline(always)]
    fn max(a: __m128, b: __m128) -> __m128 {
        unsafe { _mm_max_ps(a, b) }
    }

    #[inline(always)]
    fn sqrt(a: __m128) -> __m128 {
        unsafe { _mm_sqrt_ps(a) }
    }

    #[inline(always)]
    fn abs(a: __m128) -> __m128 {
        unsafe {
            let mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFF_FFFFu32 as i32));
            _mm_and_ps(a, mask)
        }
    }

    #[inline(always)]
    fn floor(a: __m128) -> __m128 {
        unsafe { _mm_floor_ps(a) }
    }

    #[inline(always)]
    fn ceil(a: __m128) -> __m128 {
        unsafe { _mm_ceil_ps(a) }
    }

    #[inline(always)]
    fn round(a: __m128) -> __m128 {
        unsafe { _mm_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a) }
    }

    #[inline(always)]
    fn mul_add(a: __m128, b: __m128, c: __m128) -> __m128 {
        unsafe { _mm_fmadd_ps(a, b, c) }
    }

    #[inline(always)]
    fn mul_sub(a: __m128, b: __m128, c: __m128) -> __m128 {
        unsafe { _mm_fmsub_ps(a, b, c) }
    }

    // ====== Comparisons ======

    #[inline(always)]
    fn simd_eq(a: __m128, b: __m128) -> __m128 {
        unsafe { _mm_cmp_ps::<_CMP_EQ_OQ>(a, b) }
    }

    #[inline(always)]
    fn simd_ne(a: __m128, b: __m128) -> __m128 {
        unsafe { _mm_cmp_ps::<_CMP_NEQ_OQ>(a, b) }
    }

    #[inline(always)]
    fn simd_lt(a: __m128, b: __m128) -> __m128 {
        unsafe { _mm_cmp_ps::<_CMP_LT_OQ>(a, b) }
    }

    #[inline(always)]
    fn simd_le(a: __m128, b: __m128) -> __m128 {
        unsafe { _mm_cmp_ps::<_CMP_LE_OQ>(a, b) }
    }

    #[inline(always)]
    fn simd_gt(a: __m128, b: __m128) -> __m128 {
        unsafe { _mm_cmp_ps::<_CMP_GT_OQ>(a, b) }
    }

    #[inline(always)]
    fn simd_ge(a: __m128, b: __m128) -> __m128 {
        unsafe { _mm_cmp_ps::<_CMP_GE_OQ>(a, b) }
    }

    #[inline(always)]
    fn blend(mask: __m128, if_true: __m128, if_false: __m128) -> __m128 {
        unsafe { _mm_blendv_ps(if_false, if_true, mask) }
    }

    // ====== Reductions ======

    #[inline(always)]
    fn reduce_add(a: __m128) -> f32 {
        unsafe {
            let h1 = _mm_hadd_ps(a, a);
            let h2 = _mm_hadd_ps(h1, h1);
            _mm_cvtss_f32(h2)
        }
    }

    #[inline(always)]
    fn reduce_min(a: __m128) -> f32 {
        unsafe {
            let shuf = _mm_shuffle_ps::<0b10_11_00_01>(a, a);
            let m1 = _mm_min_ps(a, shuf);
            let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
            let m2 = _mm_min_ps(m1, shuf2);
            _mm_cvtss_f32(m2)
        }
    }

    #[inline(always)]
    fn reduce_max(a: __m128) -> f32 {
        unsafe {
            let shuf = _mm_shuffle_ps::<0b10_11_00_01>(a, a);
            let m1 = _mm_max_ps(a, shuf);
            let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
            let m2 = _mm_max_ps(m1, shuf2);
            _mm_cvtss_f32(m2)
        }
    }

    // ====== Approximations ======

    #[inline(always)]
    fn rcp_approx(a: __m128) -> __m128 {
        unsafe { _mm_rcp_ps(a) }
    }

    #[inline(always)]
    fn rsqrt_approx(a: __m128) -> __m128 {
        unsafe { _mm_rsqrt_ps(a) }
    }

    // ====== Bitwise ======

    #[inline(always)]
    fn not(a: __m128) -> __m128 {
        unsafe {
            let ones = _mm_set1_epi32(-1);
            let as_int = _mm_castps_si128(a);
            _mm_castsi128_ps(_mm_xor_si128(as_int, ones))
        }
    }

    #[inline(always)]
    fn bitand(a: __m128, b: __m128) -> __m128 {
        unsafe { _mm_and_ps(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m128, b: __m128) -> __m128 {
        unsafe { _mm_or_ps(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m128, b: __m128) -> __m128 {
        unsafe { _mm_xor_ps(a, b) }
    }
}

#[cfg(target_arch = "x86_64")]
impl F32x8Backend for archmage::X64V3Token {
    type Repr = __m256;

    // ====== Construction ======

    #[inline(always)]
    fn splat(v: f32) -> __m256 {
        unsafe { _mm256_set1_ps(v) }
    }

    #[inline(always)]
    fn zero() -> __m256 {
        unsafe { _mm256_setzero_ps() }
    }

    #[inline(always)]
    fn load(data: &[f32; 8]) -> __m256 {
        unsafe { _mm256_loadu_ps(data.as_ptr()) }
    }

    #[inline(always)]
    fn from_array(arr: [f32; 8]) -> __m256 {
        // SAFETY: [f32; 8] and __m256 have identical size and layout.
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m256, out: &mut [f32; 8]) {
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: __m256) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), repr) };
        out
    }

    // ====== Arithmetic ======

    #[inline(always)]
    fn add(a: __m256, b: __m256) -> __m256 {
        unsafe { _mm256_add_ps(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m256, b: __m256) -> __m256 {
        unsafe { _mm256_sub_ps(a, b) }
    }

    #[inline(always)]
    fn mul(a: __m256, b: __m256) -> __m256 {
        unsafe { _mm256_mul_ps(a, b) }
    }

    #[inline(always)]
    fn div(a: __m256, b: __m256) -> __m256 {
        unsafe { _mm256_div_ps(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m256) -> __m256 {
        unsafe { _mm256_sub_ps(_mm256_setzero_ps(), a) }
    }

    // ====== Math ======

    #[inline(always)]
    fn min(a: __m256, b: __m256) -> __m256 {
        unsafe { _mm256_min_ps(a, b) }
    }

    #[inline(always)]
    fn max(a: __m256, b: __m256) -> __m256 {
        unsafe { _mm256_max_ps(a, b) }
    }

    #[inline(always)]
    fn sqrt(a: __m256) -> __m256 {
        unsafe { _mm256_sqrt_ps(a) }
    }

    #[inline(always)]
    fn abs(a: __m256) -> __m256 {
        unsafe {
            let mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFFu32 as i32));
            _mm256_and_ps(a, mask)
        }
    }

    #[inline(always)]
    fn floor(a: __m256) -> __m256 {
        unsafe { _mm256_floor_ps(a) }
    }

    #[inline(always)]
    fn ceil(a: __m256) -> __m256 {
        unsafe { _mm256_ceil_ps(a) }
    }

    #[inline(always)]
    fn round(a: __m256) -> __m256 {
        unsafe { _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a) }
    }

    #[inline(always)]
    fn mul_add(a: __m256, b: __m256, c: __m256) -> __m256 {
        unsafe { _mm256_fmadd_ps(a, b, c) }
    }

    #[inline(always)]
    fn mul_sub(a: __m256, b: __m256, c: __m256) -> __m256 {
        unsafe { _mm256_fmsub_ps(a, b, c) }
    }

    // ====== Comparisons ======

    #[inline(always)]
    fn simd_eq(a: __m256, b: __m256) -> __m256 {
        unsafe { _mm256_cmp_ps::<_CMP_EQ_OQ>(a, b) }
    }

    #[inline(always)]
    fn simd_ne(a: __m256, b: __m256) -> __m256 {
        unsafe { _mm256_cmp_ps::<_CMP_NEQ_OQ>(a, b) }
    }

    #[inline(always)]
    fn simd_lt(a: __m256, b: __m256) -> __m256 {
        unsafe { _mm256_cmp_ps::<_CMP_LT_OQ>(a, b) }
    }

    #[inline(always)]
    fn simd_le(a: __m256, b: __m256) -> __m256 {
        unsafe { _mm256_cmp_ps::<_CMP_LE_OQ>(a, b) }
    }

    #[inline(always)]
    fn simd_gt(a: __m256, b: __m256) -> __m256 {
        unsafe { _mm256_cmp_ps::<_CMP_GT_OQ>(a, b) }
    }

    #[inline(always)]
    fn simd_ge(a: __m256, b: __m256) -> __m256 {
        unsafe { _mm256_cmp_ps::<_CMP_GE_OQ>(a, b) }
    }

    #[inline(always)]
    fn blend(mask: __m256, if_true: __m256, if_false: __m256) -> __m256 {
        unsafe { _mm256_blendv_ps(if_false, if_true, mask) }
    }

    // ====== Reductions ======

    #[inline(always)]
    fn reduce_add(a: __m256) -> f32 {
        unsafe {
            let hi = _mm256_extractf128_ps::<1>(a);
            let lo = _mm256_castps256_ps128(a);
            let sum = _mm_add_ps(lo, hi);
            let h1 = _mm_hadd_ps(sum, sum);
            let h2 = _mm_hadd_ps(h1, h1);
            _mm_cvtss_f32(h2)
        }
    }

    #[inline(always)]
    fn reduce_min(a: __m256) -> f32 {
        unsafe {
            let hi = _mm256_extractf128_ps::<1>(a);
            let lo = _mm256_castps256_ps128(a);
            let m = _mm_min_ps(lo, hi);
            let shuf = _mm_shuffle_ps::<0b10_11_00_01>(m, m);
            let m1 = _mm_min_ps(m, shuf);
            let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
            let m2 = _mm_min_ps(m1, shuf2);
            _mm_cvtss_f32(m2)
        }
    }

    #[inline(always)]
    fn reduce_max(a: __m256) -> f32 {
        unsafe {
            let hi = _mm256_extractf128_ps::<1>(a);
            let lo = _mm256_castps256_ps128(a);
            let m = _mm_max_ps(lo, hi);
            let shuf = _mm_shuffle_ps::<0b10_11_00_01>(m, m);
            let m1 = _mm_max_ps(m, shuf);
            let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
            let m2 = _mm_max_ps(m1, shuf2);
            _mm_cvtss_f32(m2)
        }
    }

    // ====== Approximations ======

    #[inline(always)]
    fn rcp_approx(a: __m256) -> __m256 {
        unsafe { _mm256_rcp_ps(a) }
    }

    #[inline(always)]
    fn rsqrt_approx(a: __m256) -> __m256 {
        unsafe { _mm256_rsqrt_ps(a) }
    }

    // ====== Bitwise ======

    #[inline(always)]
    fn not(a: __m256) -> __m256 {
        unsafe {
            let ones = _mm256_set1_epi32(-1);
            let as_int = _mm256_castps_si256(a);
            _mm256_castsi256_ps(_mm256_xor_si256(as_int, ones))
        }
    }

    #[inline(always)]
    fn bitand(a: __m256, b: __m256) -> __m256 {
        unsafe { _mm256_and_ps(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m256, b: __m256) -> __m256 {
        unsafe { _mm256_or_ps(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m256, b: __m256) -> __m256 {
        unsafe { _mm256_xor_ps(a, b) }
    }
}

#[cfg(target_arch = "x86_64")]
impl F64x2Backend for archmage::X64V3Token {
    type Repr = __m128d;

    // ====== Construction ======

    #[inline(always)]
    fn splat(v: f64) -> __m128d {
        unsafe { _mm_set1_pd(v) }
    }

    #[inline(always)]
    fn zero() -> __m128d {
        unsafe { _mm_setzero_pd() }
    }

    #[inline(always)]
    fn load(data: &[f64; 2]) -> __m128d {
        unsafe { _mm_loadu_pd(data.as_ptr()) }
    }

    #[inline(always)]
    fn from_array(arr: [f64; 2]) -> __m128d {
        // SAFETY: [f64; 2] and __m128d have identical size and layout.
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m128d, out: &mut [f64; 2]) {
        unsafe { _mm_storeu_pd(out.as_mut_ptr(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: __m128d) -> [f64; 2] {
        let mut out = [0.0f64; 2];
        unsafe { _mm_storeu_pd(out.as_mut_ptr(), repr) };
        out
    }

    // ====== Arithmetic ======

    #[inline(always)]
    fn add(a: __m128d, b: __m128d) -> __m128d {
        unsafe { _mm_add_pd(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m128d, b: __m128d) -> __m128d {
        unsafe { _mm_sub_pd(a, b) }
    }

    #[inline(always)]
    fn mul(a: __m128d, b: __m128d) -> __m128d {
        unsafe { _mm_mul_pd(a, b) }
    }

    #[inline(always)]
    fn div(a: __m128d, b: __m128d) -> __m128d {
        unsafe { _mm_div_pd(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m128d) -> __m128d {
        unsafe { _mm_sub_pd(_mm_setzero_pd(), a) }
    }

    // ====== Math ======

    #[inline(always)]
    fn min(a: __m128d, b: __m128d) -> __m128d {
        unsafe { _mm_min_pd(a, b) }
    }

    #[inline(always)]
    fn max(a: __m128d, b: __m128d) -> __m128d {
        unsafe { _mm_max_pd(a, b) }
    }

    #[inline(always)]
    fn sqrt(a: __m128d) -> __m128d {
        unsafe { _mm_sqrt_pd(a) }
    }

    #[inline(always)]
    fn abs(a: __m128d) -> __m128d {
        unsafe {
            let mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFF_FFFF_FFFF_FFFFu64 as i64));
            _mm_and_pd(a, mask)
        }
    }

    #[inline(always)]
    fn floor(a: __m128d) -> __m128d {
        unsafe { _mm_floor_pd(a) }
    }

    #[inline(always)]
    fn ceil(a: __m128d) -> __m128d {
        unsafe { _mm_ceil_pd(a) }
    }

    #[inline(always)]
    fn round(a: __m128d) -> __m128d {
        unsafe { _mm_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a) }
    }

    #[inline(always)]
    fn mul_add(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
        unsafe { _mm_fmadd_pd(a, b, c) }
    }

    #[inline(always)]
    fn mul_sub(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
        unsafe { _mm_fmsub_pd(a, b, c) }
    }

    // ====== Comparisons ======

    #[inline(always)]
    fn simd_eq(a: __m128d, b: __m128d) -> __m128d {
        unsafe { _mm_cmp_pd::<_CMP_EQ_OQ>(a, b) }
    }

    #[inline(always)]
    fn simd_ne(a: __m128d, b: __m128d) -> __m128d {
        unsafe { _mm_cmp_pd::<_CMP_NEQ_OQ>(a, b) }
    }

    #[inline(always)]
    fn simd_lt(a: __m128d, b: __m128d) -> __m128d {
        unsafe { _mm_cmp_pd::<_CMP_LT_OQ>(a, b) }
    }

    #[inline(always)]
    fn simd_le(a: __m128d, b: __m128d) -> __m128d {
        unsafe { _mm_cmp_pd::<_CMP_LE_OQ>(a, b) }
    }

    #[inline(always)]
    fn simd_gt(a: __m128d, b: __m128d) -> __m128d {
        unsafe { _mm_cmp_pd::<_CMP_GT_OQ>(a, b) }
    }

    #[inline(always)]
    fn simd_ge(a: __m128d, b: __m128d) -> __m128d {
        unsafe { _mm_cmp_pd::<_CMP_GE_OQ>(a, b) }
    }

    #[inline(always)]
    fn blend(mask: __m128d, if_true: __m128d, if_false: __m128d) -> __m128d {
        unsafe { _mm_blendv_pd(if_false, if_true, mask) }
    }

    // ====== Reductions ======

    #[inline(always)]
    fn reduce_add(a: __m128d) -> f64 {
        unsafe {
            let h = _mm_hadd_pd(a, a);
            _mm_cvtsd_f64(h)
        }
    }

    #[inline(always)]
    fn reduce_min(a: __m128d) -> f64 {
        unsafe {
            let shuf = _mm_shuffle_pd::<0b01>(a, a);
            let m = _mm_min_pd(a, shuf);
            _mm_cvtsd_f64(m)
        }
    }

    #[inline(always)]
    fn reduce_max(a: __m128d) -> f64 {
        unsafe {
            let shuf = _mm_shuffle_pd::<0b01>(a, a);
            let m = _mm_max_pd(a, shuf);
            _mm_cvtsd_f64(m)
        }
    }

    // ====== Approximations ======

    // ====== Bitwise ======

    #[inline(always)]
    fn not(a: __m128d) -> __m128d {
        unsafe {
            let ones = _mm_set1_epi64x(-1);
            let as_int = _mm_castpd_si128(a);
            _mm_castsi128_pd(_mm_xor_si128(as_int, ones))
        }
    }

    #[inline(always)]
    fn bitand(a: __m128d, b: __m128d) -> __m128d {
        unsafe { _mm_and_pd(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m128d, b: __m128d) -> __m128d {
        unsafe { _mm_or_pd(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m128d, b: __m128d) -> __m128d {
        unsafe { _mm_xor_pd(a, b) }
    }
}

#[cfg(target_arch = "x86_64")]
impl F64x4Backend for archmage::X64V3Token {
    type Repr = __m256d;

    // ====== Construction ======

    #[inline(always)]
    fn splat(v: f64) -> __m256d {
        unsafe { _mm256_set1_pd(v) }
    }

    #[inline(always)]
    fn zero() -> __m256d {
        unsafe { _mm256_setzero_pd() }
    }

    #[inline(always)]
    fn load(data: &[f64; 4]) -> __m256d {
        unsafe { _mm256_loadu_pd(data.as_ptr()) }
    }

    #[inline(always)]
    fn from_array(arr: [f64; 4]) -> __m256d {
        // SAFETY: [f64; 4] and __m256d have identical size and layout.
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m256d, out: &mut [f64; 4]) {
        unsafe { _mm256_storeu_pd(out.as_mut_ptr(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: __m256d) -> [f64; 4] {
        let mut out = [0.0f64; 4];
        unsafe { _mm256_storeu_pd(out.as_mut_ptr(), repr) };
        out
    }

    // ====== Arithmetic ======

    #[inline(always)]
    fn add(a: __m256d, b: __m256d) -> __m256d {
        unsafe { _mm256_add_pd(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m256d, b: __m256d) -> __m256d {
        unsafe { _mm256_sub_pd(a, b) }
    }

    #[inline(always)]
    fn mul(a: __m256d, b: __m256d) -> __m256d {
        unsafe { _mm256_mul_pd(a, b) }
    }

    #[inline(always)]
    fn div(a: __m256d, b: __m256d) -> __m256d {
        unsafe { _mm256_div_pd(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m256d) -> __m256d {
        unsafe { _mm256_sub_pd(_mm256_setzero_pd(), a) }
    }

    // ====== Math ======

    #[inline(always)]
    fn min(a: __m256d, b: __m256d) -> __m256d {
        unsafe { _mm256_min_pd(a, b) }
    }

    #[inline(always)]
    fn max(a: __m256d, b: __m256d) -> __m256d {
        unsafe { _mm256_max_pd(a, b) }
    }

    #[inline(always)]
    fn sqrt(a: __m256d) -> __m256d {
        unsafe { _mm256_sqrt_pd(a) }
    }

    #[inline(always)]
    fn abs(a: __m256d) -> __m256d {
        unsafe {
            let mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFF_FFFF_FFFF_FFFFu64 as i64));
            _mm256_and_pd(a, mask)
        }
    }

    #[inline(always)]
    fn floor(a: __m256d) -> __m256d {
        unsafe { _mm256_floor_pd(a) }
    }

    #[inline(always)]
    fn ceil(a: __m256d) -> __m256d {
        unsafe { _mm256_ceil_pd(a) }
    }

    #[inline(always)]
    fn round(a: __m256d) -> __m256d {
        unsafe { _mm256_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a) }
    }

    #[inline(always)]
    fn mul_add(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
        unsafe { _mm256_fmadd_pd(a, b, c) }
    }

    #[inline(always)]
    fn mul_sub(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
        unsafe { _mm256_fmsub_pd(a, b, c) }
    }

    // ====== Comparisons ======

    #[inline(always)]
    fn simd_eq(a: __m256d, b: __m256d) -> __m256d {
        unsafe { _mm256_cmp_pd::<_CMP_EQ_OQ>(a, b) }
    }

    #[inline(always)]
    fn simd_ne(a: __m256d, b: __m256d) -> __m256d {
        unsafe { _mm256_cmp_pd::<_CMP_NEQ_OQ>(a, b) }
    }

    #[inline(always)]
    fn simd_lt(a: __m256d, b: __m256d) -> __m256d {
        unsafe { _mm256_cmp_pd::<_CMP_LT_OQ>(a, b) }
    }

    #[inline(always)]
    fn simd_le(a: __m256d, b: __m256d) -> __m256d {
        unsafe { _mm256_cmp_pd::<_CMP_LE_OQ>(a, b) }
    }

    #[inline(always)]
    fn simd_gt(a: __m256d, b: __m256d) -> __m256d {
        unsafe { _mm256_cmp_pd::<_CMP_GT_OQ>(a, b) }
    }

    #[inline(always)]
    fn simd_ge(a: __m256d, b: __m256d) -> __m256d {
        unsafe { _mm256_cmp_pd::<_CMP_GE_OQ>(a, b) }
    }

    #[inline(always)]
    fn blend(mask: __m256d, if_true: __m256d, if_false: __m256d) -> __m256d {
        unsafe { _mm256_blendv_pd(if_false, if_true, mask) }
    }

    // ====== Reductions ======

    #[inline(always)]
    fn reduce_add(a: __m256d) -> f64 {
        unsafe {
            let hi = _mm256_extractf128_pd::<1>(a);
            let lo = _mm256_castpd256_pd128(a);
            let sum = _mm_add_pd(lo, hi);
            let h = _mm_hadd_pd(sum, sum);
            _mm_cvtsd_f64(h)
        }
    }

    #[inline(always)]
    fn reduce_min(a: __m256d) -> f64 {
        unsafe {
            let hi = _mm256_extractf128_pd::<1>(a);
            let lo = _mm256_castpd256_pd128(a);
            let m = _mm_min_pd(lo, hi);
            let shuf = _mm_shuffle_pd::<0b01>(m, m);
            let m2 = _mm_min_pd(m, shuf);
            _mm_cvtsd_f64(m2)
        }
    }

    #[inline(always)]
    fn reduce_max(a: __m256d) -> f64 {
        unsafe {
            let hi = _mm256_extractf128_pd::<1>(a);
            let lo = _mm256_castpd256_pd128(a);
            let m = _mm_max_pd(lo, hi);
            let shuf = _mm_shuffle_pd::<0b01>(m, m);
            let m2 = _mm_max_pd(m, shuf);
            _mm_cvtsd_f64(m2)
        }
    }

    // ====== Approximations ======

    // ====== Bitwise ======

    #[inline(always)]
    fn not(a: __m256d) -> __m256d {
        unsafe {
            let ones = _mm256_set1_epi64x(-1);
            let as_int = _mm256_castpd_si256(a);
            _mm256_castsi256_pd(_mm256_xor_si256(as_int, ones))
        }
    }

    #[inline(always)]
    fn bitand(a: __m256d, b: __m256d) -> __m256d {
        unsafe { _mm256_and_pd(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m256d, b: __m256d) -> __m256d {
        unsafe { _mm256_or_pd(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m256d, b: __m256d) -> __m256d {
        unsafe { _mm256_xor_pd(a, b) }
    }
}

#[cfg(target_arch = "x86_64")]
impl I32x4Backend for archmage::X64V3Token {
    type Repr = __m128i;

    // ====== Construction ======

    #[inline(always)]
    fn splat(v: i32) -> __m128i {
        unsafe { _mm_set1_epi32(v) }
    }

    #[inline(always)]
    fn zero() -> __m128i {
        unsafe { _mm_setzero_si128() }
    }

    #[inline(always)]
    fn load(data: &[i32; 4]) -> __m128i {
        unsafe { _mm_loadu_si128(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [i32; 4]) -> __m128i {
        // SAFETY: [i32; 4] and __m128i have identical size and layout.
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m128i, out: &mut [i32; 4]) {
        unsafe { _mm_storeu_si128(out.as_mut_ptr().cast(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: __m128i) -> [i32; 4] {
        let mut out = [0i32; 4];
        unsafe { _mm_storeu_si128(out.as_mut_ptr().cast(), repr) };
        out
    }

    // ====== Arithmetic ======

    #[inline(always)]
    fn add(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_add_epi32(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_sub_epi32(a, b) }
    }

    #[inline(always)]
    fn mul(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_mullo_epi32(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m128i) -> __m128i {
        unsafe { _mm_sub_epi32(_mm_setzero_si128(), a) }
    }

    // ====== Math ======

    #[inline(always)]
    fn min(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_min_epi32(a, b) }
    }

    #[inline(always)]
    fn max(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_max_epi32(a, b) }
    }

    #[inline(always)]
    fn abs(a: __m128i) -> __m128i {
        unsafe { _mm_abs_epi32(a) }
    }

    // ====== Comparisons ======

    #[inline(always)]
    fn simd_eq(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_cmpeq_epi32(a, b) }
    }

    #[inline(always)]
    fn simd_ne(a: __m128i, b: __m128i) -> __m128i {
        unsafe {
            let eq = _mm_cmpeq_epi32(a, b);
            _mm_andnot_si128(eq, _mm_set1_epi32(-1))
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_cmpgt_epi32(b, a) }
    }

    #[inline(always)]
    fn simd_le(a: __m128i, b: __m128i) -> __m128i {
        unsafe {
            let gt = _mm_cmpgt_epi32(a, b);
            _mm_andnot_si128(gt, _mm_set1_epi32(-1))
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_cmpgt_epi32(a, b) }
    }

    #[inline(always)]
    fn simd_ge(a: __m128i, b: __m128i) -> __m128i {
        unsafe {
            let lt = _mm_cmpgt_epi32(b, a);
            _mm_andnot_si128(lt, _mm_set1_epi32(-1))
        }
    }

    #[inline(always)]
    fn blend(mask: __m128i, if_true: __m128i, if_false: __m128i) -> __m128i {
        unsafe { _mm_blendv_epi8(if_false, if_true, mask) }
    }

    // ====== Reductions ======

    #[inline(always)]
    fn reduce_add(a: __m128i) -> i32 {
        unsafe {
            let hi = _mm_shuffle_epi32::<0b01_00_11_10>(a);
            let sum = _mm_add_epi32(a, hi);
            let hi2 = _mm_shuffle_epi32::<0b00_00_00_01>(sum);
            let sum2 = _mm_add_epi32(sum, hi2);
            _mm_cvtsi128_si32(sum2)
        }
    }

    // ====== Bitwise ======

    #[inline(always)]
    fn not(a: __m128i) -> __m128i {
        unsafe { _mm_andnot_si128(a, _mm_set1_epi32(-1)) }
    }

    #[inline(always)]
    fn bitand(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_and_si128(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_or_si128(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_xor_si128(a, b) }
    }

    // ====== Shifts ======

    #[inline(always)]
    fn shl_const<const N: i32>(a: __m128i) -> __m128i {
        unsafe { _mm_slli_epi32::<N>(a) }
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: __m128i) -> __m128i {
        unsafe { _mm_srai_epi32::<N>(a) }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: __m128i) -> __m128i {
        unsafe { _mm_srli_epi32::<N>(a) }
    }

    // ====== Boolean ======

    #[inline(always)]
    fn all_true(a: __m128i) -> bool {
        unsafe { _mm_movemask_ps(_mm_castsi128_ps(a)) == 0xF }
    }

    #[inline(always)]
    fn any_true(a: __m128i) -> bool {
        unsafe { _mm_movemask_ps(_mm_castsi128_ps(a)) != 0 }
    }

    #[inline(always)]
    fn bitmask(a: __m128i) -> u32 {
        unsafe { _mm_movemask_ps(_mm_castsi128_ps(a)) as u32 }
    }
}

#[cfg(target_arch = "x86_64")]
impl I32x8Backend for archmage::X64V3Token {
    type Repr = __m256i;

    // ====== Construction ======

    #[inline(always)]
    fn splat(v: i32) -> __m256i {
        unsafe { _mm256_set1_epi32(v) }
    }

    #[inline(always)]
    fn zero() -> __m256i {
        unsafe { _mm256_setzero_si256() }
    }

    #[inline(always)]
    fn load(data: &[i32; 8]) -> __m256i {
        unsafe { _mm256_loadu_si256(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [i32; 8]) -> __m256i {
        // SAFETY: [i32; 8] and __m256i have identical size and layout.
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m256i, out: &mut [i32; 8]) {
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: __m256i) -> [i32; 8] {
        let mut out = [0i32; 8];
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), repr) };
        out
    }

    // ====== Arithmetic ======

    #[inline(always)]
    fn add(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_add_epi32(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_sub_epi32(a, b) }
    }

    #[inline(always)]
    fn mul(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_mullo_epi32(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m256i) -> __m256i {
        unsafe { _mm256_sub_epi32(_mm256_setzero_si256(), a) }
    }

    // ====== Math ======

    #[inline(always)]
    fn min(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_min_epi32(a, b) }
    }

    #[inline(always)]
    fn max(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_max_epi32(a, b) }
    }

    #[inline(always)]
    fn abs(a: __m256i) -> __m256i {
        unsafe { _mm256_abs_epi32(a) }
    }

    // ====== Comparisons ======

    #[inline(always)]
    fn simd_eq(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_cmpeq_epi32(a, b) }
    }

    #[inline(always)]
    fn simd_ne(a: __m256i, b: __m256i) -> __m256i {
        unsafe {
            let eq = _mm256_cmpeq_epi32(a, b);
            _mm256_andnot_si256(eq, _mm256_set1_epi32(-1))
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_cmpgt_epi32(b, a) }
    }

    #[inline(always)]
    fn simd_le(a: __m256i, b: __m256i) -> __m256i {
        unsafe {
            let gt = _mm256_cmpgt_epi32(a, b);
            _mm256_andnot_si256(gt, _mm256_set1_epi32(-1))
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_cmpgt_epi32(a, b) }
    }

    #[inline(always)]
    fn simd_ge(a: __m256i, b: __m256i) -> __m256i {
        unsafe {
            let lt = _mm256_cmpgt_epi32(b, a);
            _mm256_andnot_si256(lt, _mm256_set1_epi32(-1))
        }
    }

    #[inline(always)]
    fn blend(mask: __m256i, if_true: __m256i, if_false: __m256i) -> __m256i {
        unsafe { _mm256_blendv_epi8(if_false, if_true, mask) }
    }

    // ====== Reductions ======

    #[inline(always)]
    fn reduce_add(a: __m256i) -> i32 {
        unsafe {
            let lo = _mm256_castsi256_si128(a);
            let hi = _mm256_extracti128_si256::<1>(a);
            let sum = _mm_add_epi32(lo, hi);
            let hi2 = _mm_shuffle_epi32::<0b01_00_11_10>(sum);
            let sum2 = _mm_add_epi32(sum, hi2);
            let hi3 = _mm_shuffle_epi32::<0b00_00_00_01>(sum2);
            let sum3 = _mm_add_epi32(sum2, hi3);
            _mm_cvtsi128_si32(sum3)
        }
    }

    // ====== Bitwise ======

    #[inline(always)]
    fn not(a: __m256i) -> __m256i {
        unsafe { _mm256_andnot_si256(a, _mm256_set1_epi32(-1)) }
    }

    #[inline(always)]
    fn bitand(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_and_si256(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_or_si256(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_xor_si256(a, b) }
    }

    // ====== Shifts ======

    #[inline(always)]
    fn shl_const<const N: i32>(a: __m256i) -> __m256i {
        unsafe { _mm256_slli_epi32::<N>(a) }
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: __m256i) -> __m256i {
        unsafe { _mm256_srai_epi32::<N>(a) }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: __m256i) -> __m256i {
        unsafe { _mm256_srli_epi32::<N>(a) }
    }

    // ====== Boolean ======

    #[inline(always)]
    fn all_true(a: __m256i) -> bool {
        unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(a)) == 0xFF }
    }

    #[inline(always)]
    fn any_true(a: __m256i) -> bool {
        unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(a)) != 0 }
    }

    #[inline(always)]
    fn bitmask(a: __m256i) -> u32 {
        unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(a)) as u32 }
    }
}

#[cfg(target_arch = "x86_64")]
impl U32x4Backend for archmage::X64V3Token {
    type Repr = __m128i;

    // ====== Construction ======

    #[inline(always)]
    fn splat(v: u32) -> __m128i {
        unsafe { _mm_set1_epi32(v as i32) }
    }

    #[inline(always)]
    fn zero() -> __m128i {
        unsafe { _mm_setzero_si128() }
    }

    #[inline(always)]
    fn load(data: &[u32; 4]) -> __m128i {
        unsafe { _mm_loadu_si128(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [u32; 4]) -> __m128i {
        // SAFETY: [u32; 4] and __m128i have identical size and layout.
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m128i, out: &mut [u32; 4]) {
        unsafe { _mm_storeu_si128(out.as_mut_ptr().cast(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: __m128i) -> [u32; 4] {
        let mut out = [0u32; 4];
        unsafe { _mm_storeu_si128(out.as_mut_ptr().cast(), repr) };
        out
    }

    // ====== Arithmetic ======

    #[inline(always)]
    fn add(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_add_epi32(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_sub_epi32(a, b) }
    }

    #[inline(always)]
    fn mul(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_mullo_epi32(a, b) }
    }

    // ====== Math ======

    #[inline(always)]
    fn min(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_min_epu32(a, b) }
    }

    #[inline(always)]
    fn max(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_max_epu32(a, b) }
    }

    // ====== Comparisons ======

    #[inline(always)]
    fn simd_eq(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_cmpeq_epi32(a, b) }
    }

    #[inline(always)]
    fn simd_ne(a: __m128i, b: __m128i) -> __m128i {
        unsafe {
            let eq = _mm_cmpeq_epi32(a, b);
            _mm_andnot_si128(eq, _mm_set1_epi32(-1))
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m128i, b: __m128i) -> __m128i {
        // Unsigned comparison via bias trick: XOR both with 0x80000000
        // to convert to signed range, then use signed cmpgt.
        unsafe {
            let bias = _mm_set1_epi32(i32::MIN);
            let sa = _mm_xor_si128(a, bias);
            let sb = _mm_xor_si128(b, bias);
            _mm_cmpgt_epi32(sa, sb)
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m128i, b: __m128i) -> __m128i {
        <Self as U32x4Backend>::simd_gt(b, a)
    }

    #[inline(always)]
    fn simd_le(a: __m128i, b: __m128i) -> __m128i {
        unsafe {
            let gt = <Self as U32x4Backend>::simd_gt(a, b);
            _mm_andnot_si128(gt, _mm_set1_epi32(-1))
        }
    }

    #[inline(always)]
    fn simd_ge(a: __m128i, b: __m128i) -> __m128i {
        unsafe {
            let lt = <Self as U32x4Backend>::simd_gt(b, a);
            _mm_andnot_si128(lt, _mm_set1_epi32(-1))
        }
    }

    #[inline(always)]
    fn blend(mask: __m128i, if_true: __m128i, if_false: __m128i) -> __m128i {
        unsafe { _mm_blendv_epi8(if_false, if_true, mask) }
    }

    // ====== Reductions ======

    #[inline(always)]
    fn reduce_add(a: __m128i) -> u32 {
        unsafe {
            let hi = _mm_shuffle_epi32::<0b01_00_11_10>(a);
            let sum = _mm_add_epi32(a, hi);
            let hi2 = _mm_shuffle_epi32::<0b00_00_00_01>(sum);
            let sum2 = _mm_add_epi32(sum, hi2);
            _mm_cvtsi128_si32(sum2) as u32
        }
    }

    // ====== Bitwise ======

    #[inline(always)]
    fn not(a: __m128i) -> __m128i {
        unsafe { _mm_andnot_si128(a, _mm_set1_epi32(-1)) }
    }

    #[inline(always)]
    fn bitand(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_and_si128(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_or_si128(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_xor_si128(a, b) }
    }

    // ====== Shifts ======

    #[inline(always)]
    fn shl_const<const N: i32>(a: __m128i) -> __m128i {
        unsafe { _mm_slli_epi32::<N>(a) }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: __m128i) -> __m128i {
        unsafe { _mm_srli_epi32::<N>(a) }
    }

    // ====== Boolean ======

    #[inline(always)]
    fn all_true(a: __m128i) -> bool {
        unsafe { _mm_movemask_ps(_mm_castsi128_ps(a)) == 0xF }
    }

    #[inline(always)]
    fn any_true(a: __m128i) -> bool {
        unsafe { _mm_movemask_ps(_mm_castsi128_ps(a)) != 0 }
    }

    #[inline(always)]
    fn bitmask(a: __m128i) -> u32 {
        unsafe { _mm_movemask_ps(_mm_castsi128_ps(a)) as u32 }
    }
}

#[cfg(target_arch = "x86_64")]
impl U32x8Backend for archmage::X64V3Token {
    type Repr = __m256i;

    // ====== Construction ======

    #[inline(always)]
    fn splat(v: u32) -> __m256i {
        unsafe { _mm256_set1_epi32(v as i32) }
    }

    #[inline(always)]
    fn zero() -> __m256i {
        unsafe { _mm256_setzero_si256() }
    }

    #[inline(always)]
    fn load(data: &[u32; 8]) -> __m256i {
        unsafe { _mm256_loadu_si256(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [u32; 8]) -> __m256i {
        // SAFETY: [u32; 8] and __m256i have identical size and layout.
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m256i, out: &mut [u32; 8]) {
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: __m256i) -> [u32; 8] {
        let mut out = [0u32; 8];
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), repr) };
        out
    }

    // ====== Arithmetic ======

    #[inline(always)]
    fn add(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_add_epi32(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_sub_epi32(a, b) }
    }

    #[inline(always)]
    fn mul(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_mullo_epi32(a, b) }
    }

    // ====== Math ======

    #[inline(always)]
    fn min(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_min_epu32(a, b) }
    }

    #[inline(always)]
    fn max(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_max_epu32(a, b) }
    }

    // ====== Comparisons ======

    #[inline(always)]
    fn simd_eq(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_cmpeq_epi32(a, b) }
    }

    #[inline(always)]
    fn simd_ne(a: __m256i, b: __m256i) -> __m256i {
        unsafe {
            let eq = _mm256_cmpeq_epi32(a, b);
            _mm256_andnot_si256(eq, _mm256_set1_epi32(-1))
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m256i, b: __m256i) -> __m256i {
        // Unsigned comparison via bias trick: XOR both with 0x80000000
        // to convert to signed range, then use signed cmpgt.
        unsafe {
            let bias = _mm256_set1_epi32(i32::MIN);
            let sa = _mm256_xor_si256(a, bias);
            let sb = _mm256_xor_si256(b, bias);
            _mm256_cmpgt_epi32(sa, sb)
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m256i, b: __m256i) -> __m256i {
        <Self as U32x8Backend>::simd_gt(b, a)
    }

    #[inline(always)]
    fn simd_le(a: __m256i, b: __m256i) -> __m256i {
        unsafe {
            let gt = <Self as U32x8Backend>::simd_gt(a, b);
            _mm256_andnot_si256(gt, _mm256_set1_epi32(-1))
        }
    }

    #[inline(always)]
    fn simd_ge(a: __m256i, b: __m256i) -> __m256i {
        unsafe {
            let lt = <Self as U32x8Backend>::simd_gt(b, a);
            _mm256_andnot_si256(lt, _mm256_set1_epi32(-1))
        }
    }

    #[inline(always)]
    fn blend(mask: __m256i, if_true: __m256i, if_false: __m256i) -> __m256i {
        unsafe { _mm256_blendv_epi8(if_false, if_true, mask) }
    }

    // ====== Reductions ======

    #[inline(always)]
    fn reduce_add(a: __m256i) -> u32 {
        unsafe {
            let lo = _mm256_castsi256_si128(a);
            let hi = _mm256_extracti128_si256::<1>(a);
            let sum = _mm_add_epi32(lo, hi);
            let hi2 = _mm_shuffle_epi32::<0b01_00_11_10>(sum);
            let sum2 = _mm_add_epi32(sum, hi2);
            let hi3 = _mm_shuffle_epi32::<0b00_00_00_01>(sum2);
            let sum3 = _mm_add_epi32(sum2, hi3);
            _mm_cvtsi128_si32(sum3) as u32
        }
    }

    // ====== Bitwise ======

    #[inline(always)]
    fn not(a: __m256i) -> __m256i {
        unsafe { _mm256_andnot_si256(a, _mm256_set1_epi32(-1)) }
    }

    #[inline(always)]
    fn bitand(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_and_si256(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_or_si256(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_xor_si256(a, b) }
    }

    // ====== Shifts ======

    #[inline(always)]
    fn shl_const<const N: i32>(a: __m256i) -> __m256i {
        unsafe { _mm256_slli_epi32::<N>(a) }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: __m256i) -> __m256i {
        unsafe { _mm256_srli_epi32::<N>(a) }
    }

    // ====== Boolean ======

    #[inline(always)]
    fn all_true(a: __m256i) -> bool {
        unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(a)) == 0xFF }
    }

    #[inline(always)]
    fn any_true(a: __m256i) -> bool {
        unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(a)) != 0 }
    }

    #[inline(always)]
    fn bitmask(a: __m256i) -> u32 {
        unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(a)) as u32 }
    }
}

#[cfg(target_arch = "x86_64")]
impl I64x2Backend for archmage::X64V3Token {
    type Repr = __m128i;

    // ====== Construction ======

    #[inline(always)]
    fn splat(v: i64) -> __m128i {
        unsafe { _mm_set1_epi64x(v) }
    }

    #[inline(always)]
    fn zero() -> __m128i {
        unsafe { _mm_setzero_si128() }
    }

    #[inline(always)]
    fn load(data: &[i64; 2]) -> __m128i {
        unsafe { _mm_loadu_si128(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [i64; 2]) -> __m128i {
        // SAFETY: [i64; 2] and __m128i have identical size and layout.
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m128i, out: &mut [i64; 2]) {
        unsafe { _mm_storeu_si128(out.as_mut_ptr().cast(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: __m128i) -> [i64; 2] {
        let mut out = [0i64; 2];
        unsafe { _mm_storeu_si128(out.as_mut_ptr().cast(), repr) };
        out
    }

    // ====== Arithmetic ======

    #[inline(always)]
    fn add(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_add_epi64(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_sub_epi64(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m128i) -> __m128i {
        unsafe { _mm_sub_epi64(_mm_setzero_si128(), a) }
    }

    // ====== Math ======

    #[inline(always)]
    fn min(a: __m128i, b: __m128i) -> __m128i {
        // Polyfill: compare+select (no native i64 min on AVX2)
        unsafe {
            let mask = _mm_cmpgt_epi64(a, b);
            _mm_blendv_epi8(a, b, mask)
        }
    }

    #[inline(always)]
    fn max(a: __m128i, b: __m128i) -> __m128i {
        // Polyfill: compare+select (no native i64 max on AVX2)
        unsafe {
            let mask = _mm_cmpgt_epi64(a, b);
            _mm_blendv_epi8(b, a, mask)
        }
    }

    #[inline(always)]
    fn abs(a: __m128i) -> __m128i {
        // Polyfill: (a ^ sign) - sign (two's complement trick)
        unsafe {
            let zero = _mm_setzero_si128();
            let sign = _mm_cmpgt_epi64(zero, a);
            _mm_sub_epi64(_mm_xor_si128(a, sign), sign)
        }
    }

    // ====== Comparisons ======

    #[inline(always)]
    fn simd_eq(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_cmpeq_epi64(a, b) }
    }

    #[inline(always)]
    fn simd_ne(a: __m128i, b: __m128i) -> __m128i {
        unsafe {
            let eq = _mm_cmpeq_epi64(a, b);
            _mm_xor_si128(eq, _mm_set1_epi64x(-1))
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_cmpgt_epi64(b, a) }
    }

    #[inline(always)]
    fn simd_le(a: __m128i, b: __m128i) -> __m128i {
        unsafe {
            let gt = _mm_cmpgt_epi64(a, b);
            _mm_xor_si128(gt, _mm_set1_epi64x(-1))
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_cmpgt_epi64(a, b) }
    }

    #[inline(always)]
    fn simd_ge(a: __m128i, b: __m128i) -> __m128i {
        unsafe {
            let lt = _mm_cmpgt_epi64(b, a);
            _mm_xor_si128(lt, _mm_set1_epi64x(-1))
        }
    }

    #[inline(always)]
    fn blend(mask: __m128i, if_true: __m128i, if_false: __m128i) -> __m128i {
        unsafe { _mm_blendv_epi8(if_false, if_true, mask) }
    }

    // ====== Reductions ======

    #[inline(always)]
    fn reduce_add(a: __m128i) -> i64 {
        unsafe {
            let hi = _mm_unpackhi_epi64(a, a);
            let sum = _mm_add_epi64(a, hi);
            // Extract low 64-bit lane
            core::mem::transmute::<__m128i, [i64; 2]>(sum)[0]
        }
    }

    // ====== Bitwise ======

    #[inline(always)]
    fn not(a: __m128i) -> __m128i {
        unsafe { _mm_xor_si128(a, _mm_set1_epi64x(-1)) }
    }

    #[inline(always)]
    fn bitand(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_and_si128(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_or_si128(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_xor_si128(a, b) }
    }

    // ====== Shifts ======

    #[inline(always)]
    fn shl_const<const N: i32>(a: __m128i) -> __m128i {
        unsafe { _mm_slli_epi64::<N>(a) }
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: __m128i) -> __m128i {
        // Polyfill: no native _srai_epi64 on AVX2.
        // Use logical shift + sign extension.
        unsafe {
            // Broadcast sign of each 64-bit lane to all bits of that lane
            let sign_ext = _mm_srai_epi32::<31>(a);
            let sign64 = _mm_shuffle_epi32::<0xF5>(sign_ext);
            let logical = _mm_srli_epi64::<N>(a);
            // mask = NOT(srli(-1, N)) = upper N bits set, avoids {64 - N} const expr
            let all_ones = _mm_set1_epi64x(-1);
            let mask = _mm_andnot_si128(_mm_srli_epi64::<N>(all_ones), all_ones);
            _mm_or_si128(logical, _mm_and_si128(sign64, mask))
        }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: __m128i) -> __m128i {
        unsafe { _mm_srli_epi64::<N>(a) }
    }

    // ====== Boolean ======

    #[inline(always)]
    fn all_true(a: __m128i) -> bool {
        unsafe { _mm_movemask_pd(_mm_castsi128_pd(a)) == 0x3 }
    }

    #[inline(always)]
    fn any_true(a: __m128i) -> bool {
        unsafe { _mm_movemask_pd(_mm_castsi128_pd(a)) != 0 }
    }

    #[inline(always)]
    fn bitmask(a: __m128i) -> u32 {
        unsafe { _mm_movemask_pd(_mm_castsi128_pd(a)) as u32 }
    }
}

#[cfg(target_arch = "x86_64")]
impl I64x4Backend for archmage::X64V3Token {
    type Repr = __m256i;

    // ====== Construction ======

    #[inline(always)]
    fn splat(v: i64) -> __m256i {
        unsafe { _mm256_set1_epi64x(v) }
    }

    #[inline(always)]
    fn zero() -> __m256i {
        unsafe { _mm256_setzero_si256() }
    }

    #[inline(always)]
    fn load(data: &[i64; 4]) -> __m256i {
        unsafe { _mm256_loadu_si256(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [i64; 4]) -> __m256i {
        // SAFETY: [i64; 4] and __m256i have identical size and layout.
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m256i, out: &mut [i64; 4]) {
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: __m256i) -> [i64; 4] {
        let mut out = [0i64; 4];
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), repr) };
        out
    }

    // ====== Arithmetic ======

    #[inline(always)]
    fn add(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_add_epi64(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_sub_epi64(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m256i) -> __m256i {
        unsafe { _mm256_sub_epi64(_mm256_setzero_si256(), a) }
    }

    // ====== Math ======

    #[inline(always)]
    fn min(a: __m256i, b: __m256i) -> __m256i {
        // Polyfill: compare+select (no native i64 min on AVX2)
        unsafe {
            let mask = _mm256_cmpgt_epi64(a, b);
            _mm256_blendv_epi8(a, b, mask)
        }
    }

    #[inline(always)]
    fn max(a: __m256i, b: __m256i) -> __m256i {
        // Polyfill: compare+select (no native i64 max on AVX2)
        unsafe {
            let mask = _mm256_cmpgt_epi64(a, b);
            _mm256_blendv_epi8(b, a, mask)
        }
    }

    #[inline(always)]
    fn abs(a: __m256i) -> __m256i {
        // Polyfill: (a ^ sign) - sign (two's complement trick)
        unsafe {
            let zero = _mm256_setzero_si256();
            let sign = _mm256_cmpgt_epi64(zero, a);
            _mm256_sub_epi64(_mm256_xor_si256(a, sign), sign)
        }
    }

    // ====== Comparisons ======

    #[inline(always)]
    fn simd_eq(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_cmpeq_epi64(a, b) }
    }

    #[inline(always)]
    fn simd_ne(a: __m256i, b: __m256i) -> __m256i {
        unsafe {
            let eq = _mm256_cmpeq_epi64(a, b);
            _mm256_xor_si256(eq, _mm256_set1_epi64x(-1))
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_cmpgt_epi64(b, a) }
    }

    #[inline(always)]
    fn simd_le(a: __m256i, b: __m256i) -> __m256i {
        unsafe {
            let gt = _mm256_cmpgt_epi64(a, b);
            _mm256_xor_si256(gt, _mm256_set1_epi64x(-1))
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_cmpgt_epi64(a, b) }
    }

    #[inline(always)]
    fn simd_ge(a: __m256i, b: __m256i) -> __m256i {
        unsafe {
            let lt = _mm256_cmpgt_epi64(b, a);
            _mm256_xor_si256(lt, _mm256_set1_epi64x(-1))
        }
    }

    #[inline(always)]
    fn blend(mask: __m256i, if_true: __m256i, if_false: __m256i) -> __m256i {
        unsafe { _mm256_blendv_epi8(if_false, if_true, mask) }
    }

    // ====== Reductions ======

    #[inline(always)]
    fn reduce_add(a: __m256i) -> i64 {
        unsafe {
            let lo = _mm256_castsi256_si128(a);
            let hi = _mm256_extracti128_si256::<1>(a);
            let sum = _mm_add_epi64(lo, hi);
            let hi2 = _mm_unpackhi_epi64(sum, sum);
            let sum2 = _mm_add_epi64(sum, hi2);
            core::mem::transmute::<__m128i, [i64; 2]>(sum2)[0]
        }
    }

    // ====== Bitwise ======

    #[inline(always)]
    fn not(a: __m256i) -> __m256i {
        unsafe { _mm256_xor_si256(a, _mm256_set1_epi64x(-1)) }
    }

    #[inline(always)]
    fn bitand(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_and_si256(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_or_si256(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_xor_si256(a, b) }
    }

    // ====== Shifts ======

    #[inline(always)]
    fn shl_const<const N: i32>(a: __m256i) -> __m256i {
        unsafe { _mm256_slli_epi64::<N>(a) }
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: __m256i) -> __m256i {
        // Polyfill: no native _srai_epi64 on AVX2.
        // Use logical shift + sign extension.
        unsafe {
            // Broadcast sign of each 64-bit lane to all bits of that lane
            let sign_ext = _mm256_srai_epi32::<31>(a);
            let sign64 = _mm256_shuffle_epi32::<0xF5>(sign_ext);
            let logical = _mm256_srli_epi64::<N>(a);
            // mask = NOT(srli(-1, N)) = upper N bits set, avoids {64 - N} const expr
            let all_ones = _mm256_set1_epi64x(-1);
            let mask = _mm256_andnot_si256(_mm256_srli_epi64::<N>(all_ones), all_ones);
            _mm256_or_si256(logical, _mm256_and_si256(sign64, mask))
        }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: __m256i) -> __m256i {
        unsafe { _mm256_srli_epi64::<N>(a) }
    }

    // ====== Boolean ======

    #[inline(always)]
    fn all_true(a: __m256i) -> bool {
        unsafe { _mm256_movemask_pd(_mm256_castsi256_pd(a)) == 0xF }
    }

    #[inline(always)]
    fn any_true(a: __m256i) -> bool {
        unsafe { _mm256_movemask_pd(_mm256_castsi256_pd(a)) != 0 }
    }

    #[inline(always)]
    fn bitmask(a: __m256i) -> u32 {
        unsafe { _mm256_movemask_pd(_mm256_castsi256_pd(a)) as u32 }
    }
}

#[cfg(target_arch = "x86_64")]
impl F32x4Convert for archmage::X64V3Token {
    #[inline(always)]
    fn bitcast_f32_to_i32(a: __m128) -> __m128i {
        unsafe { _mm_castps_si128(a) }
    }

    #[inline(always)]
    fn bitcast_i32_to_f32(a: __m128i) -> __m128 {
        unsafe { _mm_castsi128_ps(a) }
    }

    #[inline(always)]
    fn convert_f32_to_i32(a: __m128) -> __m128i {
        unsafe { _mm_cvttps_epi32(a) }
    }

    #[inline(always)]
    fn convert_f32_to_i32_round(a: __m128) -> __m128i {
        unsafe { _mm_cvtps_epi32(a) }
    }

    #[inline(always)]
    fn convert_i32_to_f32(a: __m128i) -> __m128 {
        unsafe { _mm_cvtepi32_ps(a) }
    }
}

#[cfg(target_arch = "x86_64")]
impl F32x8Convert for archmage::X64V3Token {
    #[inline(always)]
    fn bitcast_f32_to_i32(a: __m256) -> __m256i {
        unsafe { _mm256_castps_si256(a) }
    }

    #[inline(always)]
    fn bitcast_i32_to_f32(a: __m256i) -> __m256 {
        unsafe { _mm256_castsi256_ps(a) }
    }

    #[inline(always)]
    fn convert_f32_to_i32(a: __m256) -> __m256i {
        unsafe { _mm256_cvttps_epi32(a) }
    }

    #[inline(always)]
    fn convert_f32_to_i32_round(a: __m256) -> __m256i {
        unsafe { _mm256_cvtps_epi32(a) }
    }

    #[inline(always)]
    fn convert_i32_to_f32(a: __m256i) -> __m256 {
        unsafe { _mm256_cvtepi32_ps(a) }
    }
}

#[cfg(target_arch = "x86_64")]
impl U32x4Bitcast for archmage::X64V3Token {
    #[inline(always)]
    fn bitcast_u32_to_i32(a: __m128i) -> __m128i {
        a
    }

    #[inline(always)]
    fn bitcast_i32_to_u32(a: __m128i) -> __m128i {
        a
    }
}

#[cfg(target_arch = "x86_64")]
impl U32x8Bitcast for archmage::X64V3Token {
    #[inline(always)]
    fn bitcast_u32_to_i32(a: __m256i) -> __m256i {
        a
    }

    #[inline(always)]
    fn bitcast_i32_to_u32(a: __m256i) -> __m256i {
        a
    }
}

#[cfg(target_arch = "x86_64")]
impl I64x2Bitcast for archmage::X64V3Token {
    #[inline(always)]
    fn bitcast_i64_to_f64(a: __m128i) -> __m128d {
        unsafe { _mm_castsi128_pd(a) }
    }

    #[inline(always)]
    fn bitcast_f64_to_i64(a: __m128d) -> __m128i {
        unsafe { _mm_castpd_si128(a) }
    }
}

#[cfg(target_arch = "x86_64")]
impl I64x4Bitcast for archmage::X64V3Token {
    #[inline(always)]
    fn bitcast_i64_to_f64(a: __m256i) -> __m256d {
        unsafe { _mm256_castsi256_pd(a) }
    }

    #[inline(always)]
    fn bitcast_f64_to_i64(a: __m256d) -> __m256i {
        unsafe { _mm256_castpd_si256(a) }
    }
}

//! F32x8Backend implementation for X64V3Token (AVX2 + FMA).
//!
//! Repr = `__m256`. All methods use native AVX/AVX2/FMA intrinsics.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::simd::backends::F32x8Backend;

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

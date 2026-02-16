//! Backend implementations for X64V4Token and Avx512ModernToken (native AVX-512).
//!
//! Implements the W512 backend traits using native 512-bit AVX-512 intrinsics
//! for both X64V4Token (base AVX-512) and Avx512ModernToken (+ VPOPCNTDQ, BITALG, etc.).
//!
//! Avx512ModernToken also gets extension trait impls (popcnt) for Modern-only features.
//!
//! W128 and W256 types use X64V3Token (V4 downcasts to V3 for narrower widths).
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::simd::backends::*;

// ============================================================================
// X64V4Token — base AVX-512 (F/BW/CD/DQ/VL)
// ============================================================================

#[cfg(target_arch = "x86_64")]
impl F32x16Backend for archmage::X64V4Token {
    type Repr = __m512;

    #[inline(always)]
    fn splat(v: f32) -> __m512 {
        unsafe { _mm512_set1_ps(v) }
    }

    #[inline(always)]
    fn zero() -> __m512 {
        unsafe { _mm512_setzero_ps() }
    }

    #[inline(always)]
    fn load(data: &[f32; 16]) -> __m512 {
        unsafe { _mm512_loadu_ps(data.as_ptr()) }
    }

    #[inline(always)]
    fn from_array(arr: [f32; 16]) -> __m512 {
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m512, out: &mut [f32; 16]) {
        unsafe { _mm512_storeu_ps(out.as_mut_ptr(), repr) }
    }

    #[inline(always)]
    fn to_array(repr: __m512) -> [f32; 16] {
        unsafe { core::mem::transmute(repr) }
    }

    #[inline(always)]
    fn add(a: __m512, b: __m512) -> __m512 {
        unsafe { _mm512_add_ps(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m512, b: __m512) -> __m512 {
        unsafe { _mm512_sub_ps(a, b) }
    }

    #[inline(always)]
    fn mul(a: __m512, b: __m512) -> __m512 {
        unsafe { _mm512_mul_ps(a, b) }
    }

    #[inline(always)]
    fn div(a: __m512, b: __m512) -> __m512 {
        unsafe { _mm512_div_ps(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m512) -> __m512 {
        unsafe { _mm512_sub_ps(_mm512_setzero_ps(), a) }
    }

    #[inline(always)]
    fn min(a: __m512, b: __m512) -> __m512 {
        unsafe { _mm512_min_ps(a, b) }
    }

    #[inline(always)]
    fn max(a: __m512, b: __m512) -> __m512 {
        unsafe { _mm512_max_ps(a, b) }
    }

    #[inline(always)]
    fn sqrt(a: __m512) -> __m512 {
        unsafe { _mm512_sqrt_ps(a) }
    }

    #[inline(always)]
    fn abs(a: __m512) -> __m512 {
        unsafe {
            let mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFF_FFFFu32 as i32));
            _mm512_and_ps(a, mask)
        }
    }

    #[inline(always)]
    fn floor(a: __m512) -> __m512 {
        unsafe { _mm512_roundscale_ps::<0x01>(a) }
    }

    #[inline(always)]
    fn ceil(a: __m512) -> __m512 {
        unsafe { _mm512_roundscale_ps::<0x02>(a) }
    }

    #[inline(always)]
    fn round(a: __m512) -> __m512 {
        unsafe { _mm512_roundscale_ps::<0x00>(a) }
    }

    #[inline(always)]
    fn mul_add(a: __m512, b: __m512, c: __m512) -> __m512 {
        unsafe { _mm512_fmadd_ps(a, b, c) }
    }

    #[inline(always)]
    fn mul_sub(a: __m512, b: __m512, c: __m512) -> __m512 {
        unsafe { _mm512_fmsub_ps(a, b, c) }
    }

    #[inline(always)]
    fn simd_eq(a: __m512, b: __m512) -> __m512 {
        unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(a, b);
            _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
        }
    }

    #[inline(always)]
    fn simd_ne(a: __m512, b: __m512) -> __m512 {
        unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_NEQ_UQ>(a, b);
            _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m512, b: __m512) -> __m512 {
        unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(a, b);
            _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
        }
    }

    #[inline(always)]
    fn simd_le(a: __m512, b: __m512) -> __m512 {
        unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_LE_OQ>(a, b);
            _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m512, b: __m512) -> __m512 {
        unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(a, b);
            _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
        }
    }

    #[inline(always)]
    fn simd_ge(a: __m512, b: __m512) -> __m512 {
        unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(a, b);
            _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
        }
    }

    #[inline(always)]
    fn blend(mask: __m512, if_true: __m512, if_false: __m512) -> __m512 {
        unsafe {
            let mask_i = _mm512_castps_si512(mask);
            let k = _mm512_cmpneq_epi32_mask(mask_i, _mm512_setzero_si512());
            _mm512_mask_blend_ps(k, if_false, if_true)
        }
    }

    #[inline(always)]
    fn reduce_add(a: __m512) -> f32 {
        unsafe { _mm512_reduce_add_ps(a) }
    }

    #[inline(always)]
    fn reduce_min(a: __m512) -> f32 {
        unsafe { _mm512_reduce_min_ps(a) }
    }

    #[inline(always)]
    fn reduce_max(a: __m512) -> f32 {
        unsafe { _mm512_reduce_max_ps(a) }
    }

    #[inline(always)]
    fn rcp_approx(a: __m512) -> __m512 {
        unsafe {
            let approx = _mm512_rcp14_ps(a);
            // One Newton-Raphson iteration: x' = x * (2 - a*x)
            let two = _mm512_set1_ps(2.0);
            _mm512_mul_ps(approx, _mm512_sub_ps(two, _mm512_mul_ps(a, approx)))
        }
    }

    #[inline(always)]
    fn rsqrt_approx(a: __m512) -> __m512 {
        unsafe {
            let approx = _mm512_rsqrt14_ps(a);
            // One Newton-Raphson iteration: x' = 0.5 * x * (3 - a*x*x)
            let half = _mm512_set1_ps(0.5);
            let three = _mm512_set1_ps(3.0);
            _mm512_mul_ps(
                _mm512_mul_ps(half, approx),
                _mm512_sub_ps(three, _mm512_mul_ps(a, _mm512_mul_ps(approx, approx))),
            )
        }
    }

    #[inline(always)]
    fn not(a: __m512) -> __m512 {
        unsafe {
            let all_ones = _mm512_castsi512_ps(_mm512_set1_epi32(-1));
            _mm512_xor_ps(a, all_ones)
        }
    }

    #[inline(always)]
    fn bitand(a: __m512, b: __m512) -> __m512 {
        unsafe { _mm512_and_ps(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m512, b: __m512) -> __m512 {
        unsafe { _mm512_or_ps(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m512, b: __m512) -> __m512 {
        unsafe { _mm512_xor_ps(a, b) }
    }
}

#[cfg(target_arch = "x86_64")]
impl F64x8Backend for archmage::X64V4Token {
    type Repr = __m512d;

    #[inline(always)]
    fn splat(v: f64) -> __m512d {
        unsafe { _mm512_set1_pd(v) }
    }

    #[inline(always)]
    fn zero() -> __m512d {
        unsafe { _mm512_setzero_pd() }
    }

    #[inline(always)]
    fn load(data: &[f64; 8]) -> __m512d {
        unsafe { _mm512_loadu_pd(data.as_ptr()) }
    }

    #[inline(always)]
    fn from_array(arr: [f64; 8]) -> __m512d {
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m512d, out: &mut [f64; 8]) {
        unsafe { _mm512_storeu_pd(out.as_mut_ptr(), repr) }
    }

    #[inline(always)]
    fn to_array(repr: __m512d) -> [f64; 8] {
        unsafe { core::mem::transmute(repr) }
    }

    #[inline(always)]
    fn add(a: __m512d, b: __m512d) -> __m512d {
        unsafe { _mm512_add_pd(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m512d, b: __m512d) -> __m512d {
        unsafe { _mm512_sub_pd(a, b) }
    }

    #[inline(always)]
    fn mul(a: __m512d, b: __m512d) -> __m512d {
        unsafe { _mm512_mul_pd(a, b) }
    }

    #[inline(always)]
    fn div(a: __m512d, b: __m512d) -> __m512d {
        unsafe { _mm512_div_pd(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m512d) -> __m512d {
        unsafe { _mm512_sub_pd(_mm512_setzero_pd(), a) }
    }

    #[inline(always)]
    fn min(a: __m512d, b: __m512d) -> __m512d {
        unsafe { _mm512_min_pd(a, b) }
    }

    #[inline(always)]
    fn max(a: __m512d, b: __m512d) -> __m512d {
        unsafe { _mm512_max_pd(a, b) }
    }

    #[inline(always)]
    fn sqrt(a: __m512d) -> __m512d {
        unsafe { _mm512_sqrt_pd(a) }
    }

    #[inline(always)]
    fn abs(a: __m512d) -> __m512d {
        unsafe {
            let mask = _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFF_FFFF_FFFF_FFFFu64 as i64));
            _mm512_and_pd(a, mask)
        }
    }

    #[inline(always)]
    fn floor(a: __m512d) -> __m512d {
        unsafe { _mm512_roundscale_pd::<0x01>(a) }
    }

    #[inline(always)]
    fn ceil(a: __m512d) -> __m512d {
        unsafe { _mm512_roundscale_pd::<0x02>(a) }
    }

    #[inline(always)]
    fn round(a: __m512d) -> __m512d {
        unsafe { _mm512_roundscale_pd::<0x00>(a) }
    }

    #[inline(always)]
    fn mul_add(a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        unsafe { _mm512_fmadd_pd(a, b, c) }
    }

    #[inline(always)]
    fn mul_sub(a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        unsafe { _mm512_fmsub_pd(a, b, c) }
    }

    #[inline(always)]
    fn simd_eq(a: __m512d, b: __m512d) -> __m512d {
        unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_EQ_OQ>(a, b);
            _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
        }
    }

    #[inline(always)]
    fn simd_ne(a: __m512d, b: __m512d) -> __m512d {
        unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_NEQ_UQ>(a, b);
            _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m512d, b: __m512d) -> __m512d {
        unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_LT_OQ>(a, b);
            _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
        }
    }

    #[inline(always)]
    fn simd_le(a: __m512d, b: __m512d) -> __m512d {
        unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(a, b);
            _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m512d, b: __m512d) -> __m512d {
        unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_GT_OQ>(a, b);
            _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
        }
    }

    #[inline(always)]
    fn simd_ge(a: __m512d, b: __m512d) -> __m512d {
        unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(a, b);
            _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
        }
    }

    #[inline(always)]
    fn blend(mask: __m512d, if_true: __m512d, if_false: __m512d) -> __m512d {
        unsafe {
            let mask_i = _mm512_castpd_si512(mask);
            let k = _mm512_cmpneq_epi64_mask(mask_i, _mm512_setzero_si512());
            _mm512_mask_blend_pd(k, if_false, if_true)
        }
    }

    #[inline(always)]
    fn reduce_add(a: __m512d) -> f64 {
        unsafe { _mm512_reduce_add_pd(a) }
    }

    #[inline(always)]
    fn reduce_min(a: __m512d) -> f64 {
        unsafe { _mm512_reduce_min_pd(a) }
    }

    #[inline(always)]
    fn reduce_max(a: __m512d) -> f64 {
        unsafe { _mm512_reduce_max_pd(a) }
    }

    #[inline(always)]
    fn rcp_approx(a: __m512d) -> __m512d {
        unsafe {
            let approx = _mm512_rcp14_pd(a);
            // One Newton-Raphson iteration: x' = x * (2 - a*x)
            let two = _mm512_set1_pd(2.0);
            _mm512_mul_pd(approx, _mm512_sub_pd(two, _mm512_mul_pd(a, approx)))
        }
    }

    #[inline(always)]
    fn rsqrt_approx(a: __m512d) -> __m512d {
        unsafe {
            let approx = _mm512_rsqrt14_pd(a);
            // One Newton-Raphson iteration: x' = 0.5 * x * (3 - a*x*x)
            let half = _mm512_set1_pd(0.5);
            let three = _mm512_set1_pd(3.0);
            _mm512_mul_pd(
                _mm512_mul_pd(half, approx),
                _mm512_sub_pd(three, _mm512_mul_pd(a, _mm512_mul_pd(approx, approx))),
            )
        }
    }

    #[inline(always)]
    fn not(a: __m512d) -> __m512d {
        unsafe {
            let all_ones = _mm512_castsi512_pd(_mm512_set1_epi64(-1));
            _mm512_xor_pd(a, all_ones)
        }
    }

    #[inline(always)]
    fn bitand(a: __m512d, b: __m512d) -> __m512d {
        unsafe { _mm512_and_pd(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m512d, b: __m512d) -> __m512d {
        unsafe { _mm512_or_pd(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m512d, b: __m512d) -> __m512d {
        unsafe { _mm512_xor_pd(a, b) }
    }
}

#[cfg(target_arch = "x86_64")]
impl I8x64Backend for archmage::X64V4Token {
    type Repr = __m512i;

    #[inline(always)]
    fn splat(v: i8) -> __m512i {
        unsafe { _mm512_set1_epi8(v as _) }
    }

    #[inline(always)]
    fn zero() -> __m512i {
        unsafe { _mm512_setzero_si512() }
    }

    #[inline(always)]
    fn load(data: &[i8; 64]) -> __m512i {
        unsafe { _mm512_loadu_si512(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [i8; 64]) -> __m512i {
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m512i, out: &mut [i8; 64]) {
        unsafe { _mm512_storeu_si512(out.as_mut_ptr().cast(), repr) }
    }

    #[inline(always)]
    fn to_array(repr: __m512i) -> [i8; 64] {
        unsafe { core::mem::transmute(repr) }
    }

    #[inline(always)]
    fn add(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_add_epi8(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi8(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi8(_mm512_setzero_si512(), a) }
    }

    #[inline(always)]
    fn min(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_min_epi8(a, b) }
    }

    #[inline(always)]
    fn max(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_max_epi8(a, b) }
    }

    #[inline(always)]
    fn abs(a: __m512i) -> __m512i {
        unsafe { _mm512_abs_epi8(a) }
    }

    #[inline(always)]
    fn simd_eq(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpeq_epi8_mask(a, b);
            _mm512_maskz_set1_epi8(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_ne(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpneq_epi8_mask(a, b);
            _mm512_maskz_set1_epi8(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmplt_epi8_mask(a, b);
            _mm512_maskz_set1_epi8(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_le(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmple_epi8_mask(a, b);
            _mm512_maskz_set1_epi8(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GT = LT with swapped args
            let mask = _mm512_cmplt_epi8_mask(b, a);
            _mm512_maskz_set1_epi8(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_ge(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GE = LE with swapped args
            let mask = _mm512_cmple_epi8_mask(b, a);
            _mm512_maskz_set1_epi8(mask, -1)
        }
    }

    #[inline(always)]
    fn blend(mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        unsafe {
            let k = _mm512_cmpneq_epi8_mask(mask, _mm512_setzero_si512());
            _mm512_mask_blend_epi8(k, if_false, if_true)
        }
    }

    #[inline(always)]
    fn reduce_add(a: __m512i) -> i8 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [i8; 64] = unsafe { core::mem::transmute(a) };
        arr.iter().copied().fold(0i8, i8::wrapping_add)
    }

    #[inline(always)]
    fn not(a: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, _mm512_set1_epi8(-1)) }
    }

    #[inline(always)]
    fn bitand(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_and_si512(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe {
            let count = _mm_cvtsi32_si128(N);
            let shifted = _mm512_sll_epi16(a, count);
            let mask = _mm512_set1_epi8(((0xFFu16 << N) & 0xFF) as i8);
            _mm512_and_si512(shifted, mask)
        }
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe {
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
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe {
            let count = _mm_cvtsi32_si128(N);
            let shifted = _mm512_srl_epi16(a, count);
            let mask = _mm512_set1_epi8(((0xFFu16 >> N) & 0xFF) as i8);
            _mm512_and_si512(shifted, mask)
        }
    }

    #[inline(always)]
    fn all_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi8_mask(a, _mm512_setzero_si512());
            mask as u64 == 0xFFFF_FFFF_FFFF_FFFFu64
        }
    }

    #[inline(always)]
    fn any_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi8_mask(a, _mm512_setzero_si512());
            mask as u64 != 0
        }
    }

    #[inline(always)]
    fn bitmask(a: __m512i) -> u64 {
        unsafe {
            // Extract high bit of each lane: compare < 0 for signed interpretation
            let zero = _mm512_setzero_si512();
            _mm512_cmpneq_epi8_mask(
                _mm512_and_si512(a, _mm512_set1_epi8(1 << (8 - 1) as i8)),
                zero,
            ) as u64
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl U8x64Backend for archmage::X64V4Token {
    type Repr = __m512i;

    #[inline(always)]
    fn splat(v: u8) -> __m512i {
        unsafe { _mm512_set1_epi8(v as _) }
    }

    #[inline(always)]
    fn zero() -> __m512i {
        unsafe { _mm512_setzero_si512() }
    }

    #[inline(always)]
    fn load(data: &[u8; 64]) -> __m512i {
        unsafe { _mm512_loadu_si512(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [u8; 64]) -> __m512i {
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m512i, out: &mut [u8; 64]) {
        unsafe { _mm512_storeu_si512(out.as_mut_ptr().cast(), repr) }
    }

    #[inline(always)]
    fn to_array(repr: __m512i) -> [u8; 64] {
        unsafe { core::mem::transmute(repr) }
    }

    #[inline(always)]
    fn add(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_add_epi8(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi8(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi8(_mm512_setzero_si512(), a) }
    }

    #[inline(always)]
    fn min(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_min_epu8(a, b) }
    }

    #[inline(always)]
    fn max(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_max_epu8(a, b) }
    }

    #[inline(always)]
    fn simd_eq(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpeq_epu8_mask(a, b);
            _mm512_maskz_set1_epi8(mask, -1i8)
        }
    }

    #[inline(always)]
    fn simd_ne(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpneq_epu8_mask(a, b);
            _mm512_maskz_set1_epi8(mask, -1i8)
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmplt_epu8_mask(a, b);
            _mm512_maskz_set1_epi8(mask, -1i8)
        }
    }

    #[inline(always)]
    fn simd_le(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmple_epu8_mask(a, b);
            _mm512_maskz_set1_epi8(mask, -1i8)
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GT = LT with swapped args
            let mask = _mm512_cmplt_epu8_mask(b, a);
            _mm512_maskz_set1_epi8(mask, -1i8)
        }
    }

    #[inline(always)]
    fn simd_ge(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GE = LE with swapped args
            let mask = _mm512_cmple_epu8_mask(b, a);
            _mm512_maskz_set1_epi8(mask, -1i8)
        }
    }

    #[inline(always)]
    fn blend(mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        unsafe {
            let k = _mm512_cmpneq_epi8_mask(mask, _mm512_setzero_si512());
            _mm512_mask_blend_epi8(k, if_false, if_true)
        }
    }

    #[inline(always)]
    fn reduce_add(a: __m512i) -> u8 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [u8; 64] = unsafe { core::mem::transmute(a) };
        arr.iter().copied().fold(0u8, u8::wrapping_add)
    }

    #[inline(always)]
    fn not(a: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, _mm512_set1_epi8(-1i8)) }
    }

    #[inline(always)]
    fn bitand(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_and_si512(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe {
            let count = _mm_cvtsi32_si128(N);
            let shifted = _mm512_sll_epi16(a, count);
            let mask = _mm512_set1_epi8(((0xFFu16 << N) & 0xFF) as i8);
            _mm512_and_si512(shifted, mask)
        }
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe {
            let count = _mm_cvtsi32_si128(N);
            let shifted = _mm512_srl_epi16(a, count);
            let mask = _mm512_set1_epi8(((0xFFu16 >> N) & 0xFF) as i8);
            _mm512_and_si512(shifted, mask)
        }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe {
            let count = _mm_cvtsi32_si128(N);
            let shifted = _mm512_srl_epi16(a, count);
            let mask = _mm512_set1_epi8(((0xFFu16 >> N) & 0xFF) as i8);
            _mm512_and_si512(shifted, mask)
        }
    }

    #[inline(always)]
    fn all_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi8_mask(a, _mm512_setzero_si512());
            mask as u64 == 0xFFFF_FFFF_FFFF_FFFFu64
        }
    }

    #[inline(always)]
    fn any_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi8_mask(a, _mm512_setzero_si512());
            mask as u64 != 0
        }
    }

    #[inline(always)]
    fn bitmask(a: __m512i) -> u64 {
        unsafe {
            // Extract high bit of each lane: compare < 0 for signed interpretation
            let zero = _mm512_setzero_si512();
            _mm512_cmpneq_epi8_mask(
                _mm512_and_si512(a, _mm512_set1_epi8(1 << (8 - 1) as i8)),
                zero,
            ) as u64
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl I16x32Backend for archmage::X64V4Token {
    type Repr = __m512i;

    #[inline(always)]
    fn splat(v: i16) -> __m512i {
        unsafe { _mm512_set1_epi16(v as _) }
    }

    #[inline(always)]
    fn zero() -> __m512i {
        unsafe { _mm512_setzero_si512() }
    }

    #[inline(always)]
    fn load(data: &[i16; 32]) -> __m512i {
        unsafe { _mm512_loadu_si512(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [i16; 32]) -> __m512i {
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m512i, out: &mut [i16; 32]) {
        unsafe { _mm512_storeu_si512(out.as_mut_ptr().cast(), repr) }
    }

    #[inline(always)]
    fn to_array(repr: __m512i) -> [i16; 32] {
        unsafe { core::mem::transmute(repr) }
    }

    #[inline(always)]
    fn add(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_add_epi16(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi16(a, b) }
    }

    #[inline(always)]
    fn mul(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_mullo_epi16(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi16(_mm512_setzero_si512(), a) }
    }

    #[inline(always)]
    fn min(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_min_epi16(a, b) }
    }

    #[inline(always)]
    fn max(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_max_epi16(a, b) }
    }

    #[inline(always)]
    fn abs(a: __m512i) -> __m512i {
        unsafe { _mm512_abs_epi16(a) }
    }

    #[inline(always)]
    fn simd_eq(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpeq_epi16_mask(a, b);
            _mm512_maskz_set1_epi16(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_ne(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpneq_epi16_mask(a, b);
            _mm512_maskz_set1_epi16(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmplt_epi16_mask(a, b);
            _mm512_maskz_set1_epi16(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_le(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmple_epi16_mask(a, b);
            _mm512_maskz_set1_epi16(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GT = LT with swapped args
            let mask = _mm512_cmplt_epi16_mask(b, a);
            _mm512_maskz_set1_epi16(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_ge(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GE = LE with swapped args
            let mask = _mm512_cmple_epi16_mask(b, a);
            _mm512_maskz_set1_epi16(mask, -1)
        }
    }

    #[inline(always)]
    fn blend(mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        unsafe {
            let k = _mm512_cmpneq_epi16_mask(mask, _mm512_setzero_si512());
            _mm512_mask_blend_epi16(k, if_false, if_true)
        }
    }

    #[inline(always)]
    fn reduce_add(a: __m512i) -> i16 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [i16; 32] = unsafe { core::mem::transmute(a) };
        arr.iter().copied().fold(0i16, i16::wrapping_add)
    }

    #[inline(always)]
    fn not(a: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, _mm512_set1_epi16(-1)) }
    }

    #[inline(always)]
    fn bitand(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_and_si512(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_sll_epi16(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_sra_epi16(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_srl_epi16(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn all_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi16_mask(a, _mm512_setzero_si512());
            mask as u64 == 0xFFFF_FFFFu64
        }
    }

    #[inline(always)]
    fn any_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi16_mask(a, _mm512_setzero_si512());
            mask as u64 != 0
        }
    }

    #[inline(always)]
    fn bitmask(a: __m512i) -> u64 {
        unsafe {
            // Extract high bit of each lane: compare < 0 for signed interpretation
            let zero = _mm512_setzero_si512();
            _mm512_cmpneq_epi16_mask(
                _mm512_and_si512(a, _mm512_set1_epi16(1 << (16 - 1) as i16)),
                zero,
            ) as u64
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl U16x32Backend for archmage::X64V4Token {
    type Repr = __m512i;

    #[inline(always)]
    fn splat(v: u16) -> __m512i {
        unsafe { _mm512_set1_epi16(v as _) }
    }

    #[inline(always)]
    fn zero() -> __m512i {
        unsafe { _mm512_setzero_si512() }
    }

    #[inline(always)]
    fn load(data: &[u16; 32]) -> __m512i {
        unsafe { _mm512_loadu_si512(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [u16; 32]) -> __m512i {
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m512i, out: &mut [u16; 32]) {
        unsafe { _mm512_storeu_si512(out.as_mut_ptr().cast(), repr) }
    }

    #[inline(always)]
    fn to_array(repr: __m512i) -> [u16; 32] {
        unsafe { core::mem::transmute(repr) }
    }

    #[inline(always)]
    fn add(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_add_epi16(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi16(a, b) }
    }

    #[inline(always)]
    fn mul(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_mullo_epi16(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi16(_mm512_setzero_si512(), a) }
    }

    #[inline(always)]
    fn min(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_min_epu16(a, b) }
    }

    #[inline(always)]
    fn max(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_max_epu16(a, b) }
    }

    #[inline(always)]
    fn simd_eq(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpeq_epu16_mask(a, b);
            _mm512_maskz_set1_epi16(mask, -1i16)
        }
    }

    #[inline(always)]
    fn simd_ne(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpneq_epu16_mask(a, b);
            _mm512_maskz_set1_epi16(mask, -1i16)
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmplt_epu16_mask(a, b);
            _mm512_maskz_set1_epi16(mask, -1i16)
        }
    }

    #[inline(always)]
    fn simd_le(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmple_epu16_mask(a, b);
            _mm512_maskz_set1_epi16(mask, -1i16)
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GT = LT with swapped args
            let mask = _mm512_cmplt_epu16_mask(b, a);
            _mm512_maskz_set1_epi16(mask, -1i16)
        }
    }

    #[inline(always)]
    fn simd_ge(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GE = LE with swapped args
            let mask = _mm512_cmple_epu16_mask(b, a);
            _mm512_maskz_set1_epi16(mask, -1i16)
        }
    }

    #[inline(always)]
    fn blend(mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        unsafe {
            let k = _mm512_cmpneq_epi16_mask(mask, _mm512_setzero_si512());
            _mm512_mask_blend_epi16(k, if_false, if_true)
        }
    }

    #[inline(always)]
    fn reduce_add(a: __m512i) -> u16 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [u16; 32] = unsafe { core::mem::transmute(a) };
        arr.iter().copied().fold(0u16, u16::wrapping_add)
    }

    #[inline(always)]
    fn not(a: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, _mm512_set1_epi16(-1i16)) }
    }

    #[inline(always)]
    fn bitand(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_and_si512(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_sll_epi16(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_srl_epi16(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_srl_epi16(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn all_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi16_mask(a, _mm512_setzero_si512());
            mask as u64 == 0xFFFF_FFFFu64
        }
    }

    #[inline(always)]
    fn any_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi16_mask(a, _mm512_setzero_si512());
            mask as u64 != 0
        }
    }

    #[inline(always)]
    fn bitmask(a: __m512i) -> u64 {
        unsafe {
            // Extract high bit of each lane: compare < 0 for signed interpretation
            let zero = _mm512_setzero_si512();
            _mm512_cmpneq_epi16_mask(
                _mm512_and_si512(a, _mm512_set1_epi16(1 << (16 - 1) as i16)),
                zero,
            ) as u64
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl I32x16Backend for archmage::X64V4Token {
    type Repr = __m512i;

    #[inline(always)]
    fn splat(v: i32) -> __m512i {
        unsafe { _mm512_set1_epi32(v as _) }
    }

    #[inline(always)]
    fn zero() -> __m512i {
        unsafe { _mm512_setzero_si512() }
    }

    #[inline(always)]
    fn load(data: &[i32; 16]) -> __m512i {
        unsafe { _mm512_loadu_si512(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [i32; 16]) -> __m512i {
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m512i, out: &mut [i32; 16]) {
        unsafe { _mm512_storeu_si512(out.as_mut_ptr().cast(), repr) }
    }

    #[inline(always)]
    fn to_array(repr: __m512i) -> [i32; 16] {
        unsafe { core::mem::transmute(repr) }
    }

    #[inline(always)]
    fn add(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_add_epi32(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi32(a, b) }
    }

    #[inline(always)]
    fn mul(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_mullo_epi32(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi32(_mm512_setzero_si512(), a) }
    }

    #[inline(always)]
    fn min(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_min_epi32(a, b) }
    }

    #[inline(always)]
    fn max(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_max_epi32(a, b) }
    }

    #[inline(always)]
    fn abs(a: __m512i) -> __m512i {
        unsafe { _mm512_abs_epi32(a) }
    }

    #[inline(always)]
    fn simd_eq(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpeq_epi32_mask(a, b);
            _mm512_maskz_set1_epi32(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_ne(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpneq_epi32_mask(a, b);
            _mm512_maskz_set1_epi32(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmplt_epi32_mask(a, b);
            _mm512_maskz_set1_epi32(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_le(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmple_epi32_mask(a, b);
            _mm512_maskz_set1_epi32(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GT = LT with swapped args
            let mask = _mm512_cmplt_epi32_mask(b, a);
            _mm512_maskz_set1_epi32(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_ge(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GE = LE with swapped args
            let mask = _mm512_cmple_epi32_mask(b, a);
            _mm512_maskz_set1_epi32(mask, -1)
        }
    }

    #[inline(always)]
    fn blend(mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        unsafe {
            let k = _mm512_cmpneq_epi32_mask(mask, _mm512_setzero_si512());
            _mm512_mask_blend_epi32(k, if_false, if_true)
        }
    }

    #[inline(always)]
    fn reduce_add(a: __m512i) -> i32 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [i32; 16] = unsafe { core::mem::transmute(a) };
        arr.iter().copied().fold(0i32, i32::wrapping_add)
    }

    #[inline(always)]
    fn not(a: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, _mm512_set1_epi32(-1)) }
    }

    #[inline(always)]
    fn bitand(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_and_si512(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_sll_epi32(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_sra_epi32(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_srl_epi32(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn all_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi32_mask(a, _mm512_setzero_si512());
            mask as u64 == 0xFFFFu64
        }
    }

    #[inline(always)]
    fn any_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi32_mask(a, _mm512_setzero_si512());
            mask as u64 != 0
        }
    }

    #[inline(always)]
    fn bitmask(a: __m512i) -> u64 {
        unsafe {
            // Extract high bit of each lane: compare < 0 for signed interpretation
            let zero = _mm512_setzero_si512();
            _mm512_cmpneq_epi32_mask(
                _mm512_and_si512(a, _mm512_set1_epi32(1 << (32 - 1) as i32)),
                zero,
            ) as u64
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl U32x16Backend for archmage::X64V4Token {
    type Repr = __m512i;

    #[inline(always)]
    fn splat(v: u32) -> __m512i {
        unsafe { _mm512_set1_epi32(v as _) }
    }

    #[inline(always)]
    fn zero() -> __m512i {
        unsafe { _mm512_setzero_si512() }
    }

    #[inline(always)]
    fn load(data: &[u32; 16]) -> __m512i {
        unsafe { _mm512_loadu_si512(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [u32; 16]) -> __m512i {
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m512i, out: &mut [u32; 16]) {
        unsafe { _mm512_storeu_si512(out.as_mut_ptr().cast(), repr) }
    }

    #[inline(always)]
    fn to_array(repr: __m512i) -> [u32; 16] {
        unsafe { core::mem::transmute(repr) }
    }

    #[inline(always)]
    fn add(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_add_epi32(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi32(a, b) }
    }

    #[inline(always)]
    fn mul(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_mullo_epi32(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi32(_mm512_setzero_si512(), a) }
    }

    #[inline(always)]
    fn min(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_min_epu32(a, b) }
    }

    #[inline(always)]
    fn max(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_max_epu32(a, b) }
    }

    #[inline(always)]
    fn simd_eq(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpeq_epu32_mask(a, b);
            _mm512_maskz_set1_epi32(mask, -1i32)
        }
    }

    #[inline(always)]
    fn simd_ne(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpneq_epu32_mask(a, b);
            _mm512_maskz_set1_epi32(mask, -1i32)
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmplt_epu32_mask(a, b);
            _mm512_maskz_set1_epi32(mask, -1i32)
        }
    }

    #[inline(always)]
    fn simd_le(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmple_epu32_mask(a, b);
            _mm512_maskz_set1_epi32(mask, -1i32)
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GT = LT with swapped args
            let mask = _mm512_cmplt_epu32_mask(b, a);
            _mm512_maskz_set1_epi32(mask, -1i32)
        }
    }

    #[inline(always)]
    fn simd_ge(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GE = LE with swapped args
            let mask = _mm512_cmple_epu32_mask(b, a);
            _mm512_maskz_set1_epi32(mask, -1i32)
        }
    }

    #[inline(always)]
    fn blend(mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        unsafe {
            let k = _mm512_cmpneq_epi32_mask(mask, _mm512_setzero_si512());
            _mm512_mask_blend_epi32(k, if_false, if_true)
        }
    }

    #[inline(always)]
    fn reduce_add(a: __m512i) -> u32 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [u32; 16] = unsafe { core::mem::transmute(a) };
        arr.iter().copied().fold(0u32, u32::wrapping_add)
    }

    #[inline(always)]
    fn not(a: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, _mm512_set1_epi32(-1i32)) }
    }

    #[inline(always)]
    fn bitand(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_and_si512(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_sll_epi32(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_srl_epi32(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_srl_epi32(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn all_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi32_mask(a, _mm512_setzero_si512());
            mask as u64 == 0xFFFFu64
        }
    }

    #[inline(always)]
    fn any_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi32_mask(a, _mm512_setzero_si512());
            mask as u64 != 0
        }
    }

    #[inline(always)]
    fn bitmask(a: __m512i) -> u64 {
        unsafe {
            // Extract high bit of each lane: compare < 0 for signed interpretation
            let zero = _mm512_setzero_si512();
            _mm512_cmpneq_epi32_mask(
                _mm512_and_si512(a, _mm512_set1_epi32(1 << (32 - 1) as i32)),
                zero,
            ) as u64
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl I64x8Backend for archmage::X64V4Token {
    type Repr = __m512i;

    #[inline(always)]
    fn splat(v: i64) -> __m512i {
        unsafe { _mm512_set1_epi64(v as _) }
    }

    #[inline(always)]
    fn zero() -> __m512i {
        unsafe { _mm512_setzero_si512() }
    }

    #[inline(always)]
    fn load(data: &[i64; 8]) -> __m512i {
        unsafe { _mm512_loadu_si512(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [i64; 8]) -> __m512i {
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m512i, out: &mut [i64; 8]) {
        unsafe { _mm512_storeu_si512(out.as_mut_ptr().cast(), repr) }
    }

    #[inline(always)]
    fn to_array(repr: __m512i) -> [i64; 8] {
        unsafe { core::mem::transmute(repr) }
    }

    #[inline(always)]
    fn add(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_add_epi64(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi64(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi64(_mm512_setzero_si512(), a) }
    }

    #[inline(always)]
    fn min(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_min_epi64(a, b) }
    }

    #[inline(always)]
    fn max(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_max_epi64(a, b) }
    }

    #[inline(always)]
    fn abs(a: __m512i) -> __m512i {
        unsafe { _mm512_abs_epi64(a) }
    }

    #[inline(always)]
    fn simd_eq(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpeq_epi64_mask(a, b);
            _mm512_maskz_set1_epi64(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_ne(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpneq_epi64_mask(a, b);
            _mm512_maskz_set1_epi64(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmplt_epi64_mask(a, b);
            _mm512_maskz_set1_epi64(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_le(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmple_epi64_mask(a, b);
            _mm512_maskz_set1_epi64(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GT = LT with swapped args
            let mask = _mm512_cmplt_epi64_mask(b, a);
            _mm512_maskz_set1_epi64(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_ge(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GE = LE with swapped args
            let mask = _mm512_cmple_epi64_mask(b, a);
            _mm512_maskz_set1_epi64(mask, -1)
        }
    }

    #[inline(always)]
    fn blend(mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        unsafe {
            let k = _mm512_cmpneq_epi64_mask(mask, _mm512_setzero_si512());
            _mm512_mask_blend_epi64(k, if_false, if_true)
        }
    }

    #[inline(always)]
    fn reduce_add(a: __m512i) -> i64 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [i64; 8] = unsafe { core::mem::transmute(a) };
        arr.iter().copied().fold(0i64, i64::wrapping_add)
    }

    #[inline(always)]
    fn not(a: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, _mm512_set1_epi64(-1)) }
    }

    #[inline(always)]
    fn bitand(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_and_si512(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_sll_epi64(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_sra_epi64(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_srl_epi64(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn all_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi64_mask(a, _mm512_setzero_si512());
            mask as u64 == 0xFFu64
        }
    }

    #[inline(always)]
    fn any_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi64_mask(a, _mm512_setzero_si512());
            mask as u64 != 0
        }
    }

    #[inline(always)]
    fn bitmask(a: __m512i) -> u64 {
        unsafe {
            // Extract high bit of each lane: compare < 0 for signed interpretation
            let zero = _mm512_setzero_si512();
            _mm512_cmpneq_epi64_mask(
                _mm512_and_si512(a, _mm512_set1_epi64(1 << (64 - 1) as i64)),
                zero,
            ) as u64
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl U64x8Backend for archmage::X64V4Token {
    type Repr = __m512i;

    #[inline(always)]
    fn splat(v: u64) -> __m512i {
        unsafe { _mm512_set1_epi64(v as _) }
    }

    #[inline(always)]
    fn zero() -> __m512i {
        unsafe { _mm512_setzero_si512() }
    }

    #[inline(always)]
    fn load(data: &[u64; 8]) -> __m512i {
        unsafe { _mm512_loadu_si512(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [u64; 8]) -> __m512i {
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m512i, out: &mut [u64; 8]) {
        unsafe { _mm512_storeu_si512(out.as_mut_ptr().cast(), repr) }
    }

    #[inline(always)]
    fn to_array(repr: __m512i) -> [u64; 8] {
        unsafe { core::mem::transmute(repr) }
    }

    #[inline(always)]
    fn add(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_add_epi64(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi64(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi64(_mm512_setzero_si512(), a) }
    }

    #[inline(always)]
    fn min(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_min_epu64(a, b) }
    }

    #[inline(always)]
    fn max(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_max_epu64(a, b) }
    }

    #[inline(always)]
    fn simd_eq(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpeq_epu64_mask(a, b);
            _mm512_maskz_set1_epi64(mask, -1i64)
        }
    }

    #[inline(always)]
    fn simd_ne(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpneq_epu64_mask(a, b);
            _mm512_maskz_set1_epi64(mask, -1i64)
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmplt_epu64_mask(a, b);
            _mm512_maskz_set1_epi64(mask, -1i64)
        }
    }

    #[inline(always)]
    fn simd_le(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmple_epu64_mask(a, b);
            _mm512_maskz_set1_epi64(mask, -1i64)
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GT = LT with swapped args
            let mask = _mm512_cmplt_epu64_mask(b, a);
            _mm512_maskz_set1_epi64(mask, -1i64)
        }
    }

    #[inline(always)]
    fn simd_ge(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GE = LE with swapped args
            let mask = _mm512_cmple_epu64_mask(b, a);
            _mm512_maskz_set1_epi64(mask, -1i64)
        }
    }

    #[inline(always)]
    fn blend(mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        unsafe {
            let k = _mm512_cmpneq_epi64_mask(mask, _mm512_setzero_si512());
            _mm512_mask_blend_epi64(k, if_false, if_true)
        }
    }

    #[inline(always)]
    fn reduce_add(a: __m512i) -> u64 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [u64; 8] = unsafe { core::mem::transmute(a) };
        arr.iter().copied().fold(0u64, u64::wrapping_add)
    }

    #[inline(always)]
    fn not(a: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, _mm512_set1_epi64(-1i64)) }
    }

    #[inline(always)]
    fn bitand(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_and_si512(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_sll_epi64(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_srl_epi64(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_srl_epi64(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn all_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi64_mask(a, _mm512_setzero_si512());
            mask as u64 == 0xFFu64
        }
    }

    #[inline(always)]
    fn any_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi64_mask(a, _mm512_setzero_si512());
            mask as u64 != 0
        }
    }

    #[inline(always)]
    fn bitmask(a: __m512i) -> u64 {
        unsafe {
            // Extract high bit of each lane: compare < 0 for signed interpretation
            let zero = _mm512_setzero_si512();
            _mm512_cmpneq_epi64_mask(
                _mm512_and_si512(a, _mm512_set1_epi64(1 << (64 - 1) as i64)),
                zero,
            ) as u64
        }
    }
}

// ============================================================================
// Avx512ModernToken — base AVX-512 (same intrinsics as V4)
// ============================================================================

#[cfg(target_arch = "x86_64")]
impl F32x16Backend for archmage::Avx512ModernToken {
    type Repr = __m512;

    #[inline(always)]
    fn splat(v: f32) -> __m512 {
        unsafe { _mm512_set1_ps(v) }
    }

    #[inline(always)]
    fn zero() -> __m512 {
        unsafe { _mm512_setzero_ps() }
    }

    #[inline(always)]
    fn load(data: &[f32; 16]) -> __m512 {
        unsafe { _mm512_loadu_ps(data.as_ptr()) }
    }

    #[inline(always)]
    fn from_array(arr: [f32; 16]) -> __m512 {
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m512, out: &mut [f32; 16]) {
        unsafe { _mm512_storeu_ps(out.as_mut_ptr(), repr) }
    }

    #[inline(always)]
    fn to_array(repr: __m512) -> [f32; 16] {
        unsafe { core::mem::transmute(repr) }
    }

    #[inline(always)]
    fn add(a: __m512, b: __m512) -> __m512 {
        unsafe { _mm512_add_ps(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m512, b: __m512) -> __m512 {
        unsafe { _mm512_sub_ps(a, b) }
    }

    #[inline(always)]
    fn mul(a: __m512, b: __m512) -> __m512 {
        unsafe { _mm512_mul_ps(a, b) }
    }

    #[inline(always)]
    fn div(a: __m512, b: __m512) -> __m512 {
        unsafe { _mm512_div_ps(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m512) -> __m512 {
        unsafe { _mm512_sub_ps(_mm512_setzero_ps(), a) }
    }

    #[inline(always)]
    fn min(a: __m512, b: __m512) -> __m512 {
        unsafe { _mm512_min_ps(a, b) }
    }

    #[inline(always)]
    fn max(a: __m512, b: __m512) -> __m512 {
        unsafe { _mm512_max_ps(a, b) }
    }

    #[inline(always)]
    fn sqrt(a: __m512) -> __m512 {
        unsafe { _mm512_sqrt_ps(a) }
    }

    #[inline(always)]
    fn abs(a: __m512) -> __m512 {
        unsafe {
            let mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFF_FFFFu32 as i32));
            _mm512_and_ps(a, mask)
        }
    }

    #[inline(always)]
    fn floor(a: __m512) -> __m512 {
        unsafe { _mm512_roundscale_ps::<0x01>(a) }
    }

    #[inline(always)]
    fn ceil(a: __m512) -> __m512 {
        unsafe { _mm512_roundscale_ps::<0x02>(a) }
    }

    #[inline(always)]
    fn round(a: __m512) -> __m512 {
        unsafe { _mm512_roundscale_ps::<0x00>(a) }
    }

    #[inline(always)]
    fn mul_add(a: __m512, b: __m512, c: __m512) -> __m512 {
        unsafe { _mm512_fmadd_ps(a, b, c) }
    }

    #[inline(always)]
    fn mul_sub(a: __m512, b: __m512, c: __m512) -> __m512 {
        unsafe { _mm512_fmsub_ps(a, b, c) }
    }

    #[inline(always)]
    fn simd_eq(a: __m512, b: __m512) -> __m512 {
        unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(a, b);
            _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
        }
    }

    #[inline(always)]
    fn simd_ne(a: __m512, b: __m512) -> __m512 {
        unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_NEQ_UQ>(a, b);
            _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m512, b: __m512) -> __m512 {
        unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(a, b);
            _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
        }
    }

    #[inline(always)]
    fn simd_le(a: __m512, b: __m512) -> __m512 {
        unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_LE_OQ>(a, b);
            _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m512, b: __m512) -> __m512 {
        unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(a, b);
            _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
        }
    }

    #[inline(always)]
    fn simd_ge(a: __m512, b: __m512) -> __m512 {
        unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(a, b);
            _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
        }
    }

    #[inline(always)]
    fn blend(mask: __m512, if_true: __m512, if_false: __m512) -> __m512 {
        unsafe {
            let mask_i = _mm512_castps_si512(mask);
            let k = _mm512_cmpneq_epi32_mask(mask_i, _mm512_setzero_si512());
            _mm512_mask_blend_ps(k, if_false, if_true)
        }
    }

    #[inline(always)]
    fn reduce_add(a: __m512) -> f32 {
        unsafe { _mm512_reduce_add_ps(a) }
    }

    #[inline(always)]
    fn reduce_min(a: __m512) -> f32 {
        unsafe { _mm512_reduce_min_ps(a) }
    }

    #[inline(always)]
    fn reduce_max(a: __m512) -> f32 {
        unsafe { _mm512_reduce_max_ps(a) }
    }

    #[inline(always)]
    fn rcp_approx(a: __m512) -> __m512 {
        unsafe {
            let approx = _mm512_rcp14_ps(a);
            // One Newton-Raphson iteration: x' = x * (2 - a*x)
            let two = _mm512_set1_ps(2.0);
            _mm512_mul_ps(approx, _mm512_sub_ps(two, _mm512_mul_ps(a, approx)))
        }
    }

    #[inline(always)]
    fn rsqrt_approx(a: __m512) -> __m512 {
        unsafe {
            let approx = _mm512_rsqrt14_ps(a);
            // One Newton-Raphson iteration: x' = 0.5 * x * (3 - a*x*x)
            let half = _mm512_set1_ps(0.5);
            let three = _mm512_set1_ps(3.0);
            _mm512_mul_ps(
                _mm512_mul_ps(half, approx),
                _mm512_sub_ps(three, _mm512_mul_ps(a, _mm512_mul_ps(approx, approx))),
            )
        }
    }

    #[inline(always)]
    fn not(a: __m512) -> __m512 {
        unsafe {
            let all_ones = _mm512_castsi512_ps(_mm512_set1_epi32(-1));
            _mm512_xor_ps(a, all_ones)
        }
    }

    #[inline(always)]
    fn bitand(a: __m512, b: __m512) -> __m512 {
        unsafe { _mm512_and_ps(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m512, b: __m512) -> __m512 {
        unsafe { _mm512_or_ps(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m512, b: __m512) -> __m512 {
        unsafe { _mm512_xor_ps(a, b) }
    }
}

#[cfg(target_arch = "x86_64")]
impl F64x8Backend for archmage::Avx512ModernToken {
    type Repr = __m512d;

    #[inline(always)]
    fn splat(v: f64) -> __m512d {
        unsafe { _mm512_set1_pd(v) }
    }

    #[inline(always)]
    fn zero() -> __m512d {
        unsafe { _mm512_setzero_pd() }
    }

    #[inline(always)]
    fn load(data: &[f64; 8]) -> __m512d {
        unsafe { _mm512_loadu_pd(data.as_ptr()) }
    }

    #[inline(always)]
    fn from_array(arr: [f64; 8]) -> __m512d {
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m512d, out: &mut [f64; 8]) {
        unsafe { _mm512_storeu_pd(out.as_mut_ptr(), repr) }
    }

    #[inline(always)]
    fn to_array(repr: __m512d) -> [f64; 8] {
        unsafe { core::mem::transmute(repr) }
    }

    #[inline(always)]
    fn add(a: __m512d, b: __m512d) -> __m512d {
        unsafe { _mm512_add_pd(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m512d, b: __m512d) -> __m512d {
        unsafe { _mm512_sub_pd(a, b) }
    }

    #[inline(always)]
    fn mul(a: __m512d, b: __m512d) -> __m512d {
        unsafe { _mm512_mul_pd(a, b) }
    }

    #[inline(always)]
    fn div(a: __m512d, b: __m512d) -> __m512d {
        unsafe { _mm512_div_pd(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m512d) -> __m512d {
        unsafe { _mm512_sub_pd(_mm512_setzero_pd(), a) }
    }

    #[inline(always)]
    fn min(a: __m512d, b: __m512d) -> __m512d {
        unsafe { _mm512_min_pd(a, b) }
    }

    #[inline(always)]
    fn max(a: __m512d, b: __m512d) -> __m512d {
        unsafe { _mm512_max_pd(a, b) }
    }

    #[inline(always)]
    fn sqrt(a: __m512d) -> __m512d {
        unsafe { _mm512_sqrt_pd(a) }
    }

    #[inline(always)]
    fn abs(a: __m512d) -> __m512d {
        unsafe {
            let mask = _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFF_FFFF_FFFF_FFFFu64 as i64));
            _mm512_and_pd(a, mask)
        }
    }

    #[inline(always)]
    fn floor(a: __m512d) -> __m512d {
        unsafe { _mm512_roundscale_pd::<0x01>(a) }
    }

    #[inline(always)]
    fn ceil(a: __m512d) -> __m512d {
        unsafe { _mm512_roundscale_pd::<0x02>(a) }
    }

    #[inline(always)]
    fn round(a: __m512d) -> __m512d {
        unsafe { _mm512_roundscale_pd::<0x00>(a) }
    }

    #[inline(always)]
    fn mul_add(a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        unsafe { _mm512_fmadd_pd(a, b, c) }
    }

    #[inline(always)]
    fn mul_sub(a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        unsafe { _mm512_fmsub_pd(a, b, c) }
    }

    #[inline(always)]
    fn simd_eq(a: __m512d, b: __m512d) -> __m512d {
        unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_EQ_OQ>(a, b);
            _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
        }
    }

    #[inline(always)]
    fn simd_ne(a: __m512d, b: __m512d) -> __m512d {
        unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_NEQ_UQ>(a, b);
            _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m512d, b: __m512d) -> __m512d {
        unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_LT_OQ>(a, b);
            _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
        }
    }

    #[inline(always)]
    fn simd_le(a: __m512d, b: __m512d) -> __m512d {
        unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(a, b);
            _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m512d, b: __m512d) -> __m512d {
        unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_GT_OQ>(a, b);
            _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
        }
    }

    #[inline(always)]
    fn simd_ge(a: __m512d, b: __m512d) -> __m512d {
        unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(a, b);
            _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
        }
    }

    #[inline(always)]
    fn blend(mask: __m512d, if_true: __m512d, if_false: __m512d) -> __m512d {
        unsafe {
            let mask_i = _mm512_castpd_si512(mask);
            let k = _mm512_cmpneq_epi64_mask(mask_i, _mm512_setzero_si512());
            _mm512_mask_blend_pd(k, if_false, if_true)
        }
    }

    #[inline(always)]
    fn reduce_add(a: __m512d) -> f64 {
        unsafe { _mm512_reduce_add_pd(a) }
    }

    #[inline(always)]
    fn reduce_min(a: __m512d) -> f64 {
        unsafe { _mm512_reduce_min_pd(a) }
    }

    #[inline(always)]
    fn reduce_max(a: __m512d) -> f64 {
        unsafe { _mm512_reduce_max_pd(a) }
    }

    #[inline(always)]
    fn rcp_approx(a: __m512d) -> __m512d {
        unsafe {
            let approx = _mm512_rcp14_pd(a);
            // One Newton-Raphson iteration: x' = x * (2 - a*x)
            let two = _mm512_set1_pd(2.0);
            _mm512_mul_pd(approx, _mm512_sub_pd(two, _mm512_mul_pd(a, approx)))
        }
    }

    #[inline(always)]
    fn rsqrt_approx(a: __m512d) -> __m512d {
        unsafe {
            let approx = _mm512_rsqrt14_pd(a);
            // One Newton-Raphson iteration: x' = 0.5 * x * (3 - a*x*x)
            let half = _mm512_set1_pd(0.5);
            let three = _mm512_set1_pd(3.0);
            _mm512_mul_pd(
                _mm512_mul_pd(half, approx),
                _mm512_sub_pd(three, _mm512_mul_pd(a, _mm512_mul_pd(approx, approx))),
            )
        }
    }

    #[inline(always)]
    fn not(a: __m512d) -> __m512d {
        unsafe {
            let all_ones = _mm512_castsi512_pd(_mm512_set1_epi64(-1));
            _mm512_xor_pd(a, all_ones)
        }
    }

    #[inline(always)]
    fn bitand(a: __m512d, b: __m512d) -> __m512d {
        unsafe { _mm512_and_pd(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m512d, b: __m512d) -> __m512d {
        unsafe { _mm512_or_pd(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m512d, b: __m512d) -> __m512d {
        unsafe { _mm512_xor_pd(a, b) }
    }
}

#[cfg(target_arch = "x86_64")]
impl I8x64Backend for archmage::Avx512ModernToken {
    type Repr = __m512i;

    #[inline(always)]
    fn splat(v: i8) -> __m512i {
        unsafe { _mm512_set1_epi8(v as _) }
    }

    #[inline(always)]
    fn zero() -> __m512i {
        unsafe { _mm512_setzero_si512() }
    }

    #[inline(always)]
    fn load(data: &[i8; 64]) -> __m512i {
        unsafe { _mm512_loadu_si512(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [i8; 64]) -> __m512i {
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m512i, out: &mut [i8; 64]) {
        unsafe { _mm512_storeu_si512(out.as_mut_ptr().cast(), repr) }
    }

    #[inline(always)]
    fn to_array(repr: __m512i) -> [i8; 64] {
        unsafe { core::mem::transmute(repr) }
    }

    #[inline(always)]
    fn add(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_add_epi8(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi8(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi8(_mm512_setzero_si512(), a) }
    }

    #[inline(always)]
    fn min(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_min_epi8(a, b) }
    }

    #[inline(always)]
    fn max(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_max_epi8(a, b) }
    }

    #[inline(always)]
    fn abs(a: __m512i) -> __m512i {
        unsafe { _mm512_abs_epi8(a) }
    }

    #[inline(always)]
    fn simd_eq(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpeq_epi8_mask(a, b);
            _mm512_maskz_set1_epi8(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_ne(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpneq_epi8_mask(a, b);
            _mm512_maskz_set1_epi8(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmplt_epi8_mask(a, b);
            _mm512_maskz_set1_epi8(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_le(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmple_epi8_mask(a, b);
            _mm512_maskz_set1_epi8(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GT = LT with swapped args
            let mask = _mm512_cmplt_epi8_mask(b, a);
            _mm512_maskz_set1_epi8(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_ge(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GE = LE with swapped args
            let mask = _mm512_cmple_epi8_mask(b, a);
            _mm512_maskz_set1_epi8(mask, -1)
        }
    }

    #[inline(always)]
    fn blend(mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        unsafe {
            let k = _mm512_cmpneq_epi8_mask(mask, _mm512_setzero_si512());
            _mm512_mask_blend_epi8(k, if_false, if_true)
        }
    }

    #[inline(always)]
    fn reduce_add(a: __m512i) -> i8 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [i8; 64] = unsafe { core::mem::transmute(a) };
        arr.iter().copied().fold(0i8, i8::wrapping_add)
    }

    #[inline(always)]
    fn not(a: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, _mm512_set1_epi8(-1)) }
    }

    #[inline(always)]
    fn bitand(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_and_si512(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe {
            let count = _mm_cvtsi32_si128(N);
            let shifted = _mm512_sll_epi16(a, count);
            let mask = _mm512_set1_epi8(((0xFFu16 << N) & 0xFF) as i8);
            _mm512_and_si512(shifted, mask)
        }
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe {
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
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe {
            let count = _mm_cvtsi32_si128(N);
            let shifted = _mm512_srl_epi16(a, count);
            let mask = _mm512_set1_epi8(((0xFFu16 >> N) & 0xFF) as i8);
            _mm512_and_si512(shifted, mask)
        }
    }

    #[inline(always)]
    fn all_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi8_mask(a, _mm512_setzero_si512());
            mask as u64 == 0xFFFF_FFFF_FFFF_FFFFu64
        }
    }

    #[inline(always)]
    fn any_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi8_mask(a, _mm512_setzero_si512());
            mask as u64 != 0
        }
    }

    #[inline(always)]
    fn bitmask(a: __m512i) -> u64 {
        unsafe {
            // Extract high bit of each lane: compare < 0 for signed interpretation
            let zero = _mm512_setzero_si512();
            _mm512_cmpneq_epi8_mask(
                _mm512_and_si512(a, _mm512_set1_epi8(1 << (8 - 1) as i8)),
                zero,
            ) as u64
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl U8x64Backend for archmage::Avx512ModernToken {
    type Repr = __m512i;

    #[inline(always)]
    fn splat(v: u8) -> __m512i {
        unsafe { _mm512_set1_epi8(v as _) }
    }

    #[inline(always)]
    fn zero() -> __m512i {
        unsafe { _mm512_setzero_si512() }
    }

    #[inline(always)]
    fn load(data: &[u8; 64]) -> __m512i {
        unsafe { _mm512_loadu_si512(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [u8; 64]) -> __m512i {
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m512i, out: &mut [u8; 64]) {
        unsafe { _mm512_storeu_si512(out.as_mut_ptr().cast(), repr) }
    }

    #[inline(always)]
    fn to_array(repr: __m512i) -> [u8; 64] {
        unsafe { core::mem::transmute(repr) }
    }

    #[inline(always)]
    fn add(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_add_epi8(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi8(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi8(_mm512_setzero_si512(), a) }
    }

    #[inline(always)]
    fn min(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_min_epu8(a, b) }
    }

    #[inline(always)]
    fn max(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_max_epu8(a, b) }
    }

    #[inline(always)]
    fn simd_eq(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpeq_epu8_mask(a, b);
            _mm512_maskz_set1_epi8(mask, -1i8)
        }
    }

    #[inline(always)]
    fn simd_ne(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpneq_epu8_mask(a, b);
            _mm512_maskz_set1_epi8(mask, -1i8)
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmplt_epu8_mask(a, b);
            _mm512_maskz_set1_epi8(mask, -1i8)
        }
    }

    #[inline(always)]
    fn simd_le(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmple_epu8_mask(a, b);
            _mm512_maskz_set1_epi8(mask, -1i8)
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GT = LT with swapped args
            let mask = _mm512_cmplt_epu8_mask(b, a);
            _mm512_maskz_set1_epi8(mask, -1i8)
        }
    }

    #[inline(always)]
    fn simd_ge(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GE = LE with swapped args
            let mask = _mm512_cmple_epu8_mask(b, a);
            _mm512_maskz_set1_epi8(mask, -1i8)
        }
    }

    #[inline(always)]
    fn blend(mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        unsafe {
            let k = _mm512_cmpneq_epi8_mask(mask, _mm512_setzero_si512());
            _mm512_mask_blend_epi8(k, if_false, if_true)
        }
    }

    #[inline(always)]
    fn reduce_add(a: __m512i) -> u8 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [u8; 64] = unsafe { core::mem::transmute(a) };
        arr.iter().copied().fold(0u8, u8::wrapping_add)
    }

    #[inline(always)]
    fn not(a: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, _mm512_set1_epi8(-1i8)) }
    }

    #[inline(always)]
    fn bitand(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_and_si512(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe {
            let count = _mm_cvtsi32_si128(N);
            let shifted = _mm512_sll_epi16(a, count);
            let mask = _mm512_set1_epi8(((0xFFu16 << N) & 0xFF) as i8);
            _mm512_and_si512(shifted, mask)
        }
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe {
            let count = _mm_cvtsi32_si128(N);
            let shifted = _mm512_srl_epi16(a, count);
            let mask = _mm512_set1_epi8(((0xFFu16 >> N) & 0xFF) as i8);
            _mm512_and_si512(shifted, mask)
        }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe {
            let count = _mm_cvtsi32_si128(N);
            let shifted = _mm512_srl_epi16(a, count);
            let mask = _mm512_set1_epi8(((0xFFu16 >> N) & 0xFF) as i8);
            _mm512_and_si512(shifted, mask)
        }
    }

    #[inline(always)]
    fn all_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi8_mask(a, _mm512_setzero_si512());
            mask as u64 == 0xFFFF_FFFF_FFFF_FFFFu64
        }
    }

    #[inline(always)]
    fn any_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi8_mask(a, _mm512_setzero_si512());
            mask as u64 != 0
        }
    }

    #[inline(always)]
    fn bitmask(a: __m512i) -> u64 {
        unsafe {
            // Extract high bit of each lane: compare < 0 for signed interpretation
            let zero = _mm512_setzero_si512();
            _mm512_cmpneq_epi8_mask(
                _mm512_and_si512(a, _mm512_set1_epi8(1 << (8 - 1) as i8)),
                zero,
            ) as u64
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl I16x32Backend for archmage::Avx512ModernToken {
    type Repr = __m512i;

    #[inline(always)]
    fn splat(v: i16) -> __m512i {
        unsafe { _mm512_set1_epi16(v as _) }
    }

    #[inline(always)]
    fn zero() -> __m512i {
        unsafe { _mm512_setzero_si512() }
    }

    #[inline(always)]
    fn load(data: &[i16; 32]) -> __m512i {
        unsafe { _mm512_loadu_si512(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [i16; 32]) -> __m512i {
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m512i, out: &mut [i16; 32]) {
        unsafe { _mm512_storeu_si512(out.as_mut_ptr().cast(), repr) }
    }

    #[inline(always)]
    fn to_array(repr: __m512i) -> [i16; 32] {
        unsafe { core::mem::transmute(repr) }
    }

    #[inline(always)]
    fn add(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_add_epi16(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi16(a, b) }
    }

    #[inline(always)]
    fn mul(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_mullo_epi16(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi16(_mm512_setzero_si512(), a) }
    }

    #[inline(always)]
    fn min(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_min_epi16(a, b) }
    }

    #[inline(always)]
    fn max(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_max_epi16(a, b) }
    }

    #[inline(always)]
    fn abs(a: __m512i) -> __m512i {
        unsafe { _mm512_abs_epi16(a) }
    }

    #[inline(always)]
    fn simd_eq(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpeq_epi16_mask(a, b);
            _mm512_maskz_set1_epi16(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_ne(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpneq_epi16_mask(a, b);
            _mm512_maskz_set1_epi16(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmplt_epi16_mask(a, b);
            _mm512_maskz_set1_epi16(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_le(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmple_epi16_mask(a, b);
            _mm512_maskz_set1_epi16(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GT = LT with swapped args
            let mask = _mm512_cmplt_epi16_mask(b, a);
            _mm512_maskz_set1_epi16(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_ge(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GE = LE with swapped args
            let mask = _mm512_cmple_epi16_mask(b, a);
            _mm512_maskz_set1_epi16(mask, -1)
        }
    }

    #[inline(always)]
    fn blend(mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        unsafe {
            let k = _mm512_cmpneq_epi16_mask(mask, _mm512_setzero_si512());
            _mm512_mask_blend_epi16(k, if_false, if_true)
        }
    }

    #[inline(always)]
    fn reduce_add(a: __m512i) -> i16 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [i16; 32] = unsafe { core::mem::transmute(a) };
        arr.iter().copied().fold(0i16, i16::wrapping_add)
    }

    #[inline(always)]
    fn not(a: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, _mm512_set1_epi16(-1)) }
    }

    #[inline(always)]
    fn bitand(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_and_si512(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_sll_epi16(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_sra_epi16(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_srl_epi16(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn all_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi16_mask(a, _mm512_setzero_si512());
            mask as u64 == 0xFFFF_FFFFu64
        }
    }

    #[inline(always)]
    fn any_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi16_mask(a, _mm512_setzero_si512());
            mask as u64 != 0
        }
    }

    #[inline(always)]
    fn bitmask(a: __m512i) -> u64 {
        unsafe {
            // Extract high bit of each lane: compare < 0 for signed interpretation
            let zero = _mm512_setzero_si512();
            _mm512_cmpneq_epi16_mask(
                _mm512_and_si512(a, _mm512_set1_epi16(1 << (16 - 1) as i16)),
                zero,
            ) as u64
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl U16x32Backend for archmage::Avx512ModernToken {
    type Repr = __m512i;

    #[inline(always)]
    fn splat(v: u16) -> __m512i {
        unsafe { _mm512_set1_epi16(v as _) }
    }

    #[inline(always)]
    fn zero() -> __m512i {
        unsafe { _mm512_setzero_si512() }
    }

    #[inline(always)]
    fn load(data: &[u16; 32]) -> __m512i {
        unsafe { _mm512_loadu_si512(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [u16; 32]) -> __m512i {
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m512i, out: &mut [u16; 32]) {
        unsafe { _mm512_storeu_si512(out.as_mut_ptr().cast(), repr) }
    }

    #[inline(always)]
    fn to_array(repr: __m512i) -> [u16; 32] {
        unsafe { core::mem::transmute(repr) }
    }

    #[inline(always)]
    fn add(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_add_epi16(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi16(a, b) }
    }

    #[inline(always)]
    fn mul(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_mullo_epi16(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi16(_mm512_setzero_si512(), a) }
    }

    #[inline(always)]
    fn min(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_min_epu16(a, b) }
    }

    #[inline(always)]
    fn max(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_max_epu16(a, b) }
    }

    #[inline(always)]
    fn simd_eq(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpeq_epu16_mask(a, b);
            _mm512_maskz_set1_epi16(mask, -1i16)
        }
    }

    #[inline(always)]
    fn simd_ne(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpneq_epu16_mask(a, b);
            _mm512_maskz_set1_epi16(mask, -1i16)
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmplt_epu16_mask(a, b);
            _mm512_maskz_set1_epi16(mask, -1i16)
        }
    }

    #[inline(always)]
    fn simd_le(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmple_epu16_mask(a, b);
            _mm512_maskz_set1_epi16(mask, -1i16)
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GT = LT with swapped args
            let mask = _mm512_cmplt_epu16_mask(b, a);
            _mm512_maskz_set1_epi16(mask, -1i16)
        }
    }

    #[inline(always)]
    fn simd_ge(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GE = LE with swapped args
            let mask = _mm512_cmple_epu16_mask(b, a);
            _mm512_maskz_set1_epi16(mask, -1i16)
        }
    }

    #[inline(always)]
    fn blend(mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        unsafe {
            let k = _mm512_cmpneq_epi16_mask(mask, _mm512_setzero_si512());
            _mm512_mask_blend_epi16(k, if_false, if_true)
        }
    }

    #[inline(always)]
    fn reduce_add(a: __m512i) -> u16 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [u16; 32] = unsafe { core::mem::transmute(a) };
        arr.iter().copied().fold(0u16, u16::wrapping_add)
    }

    #[inline(always)]
    fn not(a: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, _mm512_set1_epi16(-1i16)) }
    }

    #[inline(always)]
    fn bitand(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_and_si512(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_sll_epi16(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_srl_epi16(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_srl_epi16(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn all_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi16_mask(a, _mm512_setzero_si512());
            mask as u64 == 0xFFFF_FFFFu64
        }
    }

    #[inline(always)]
    fn any_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi16_mask(a, _mm512_setzero_si512());
            mask as u64 != 0
        }
    }

    #[inline(always)]
    fn bitmask(a: __m512i) -> u64 {
        unsafe {
            // Extract high bit of each lane: compare < 0 for signed interpretation
            let zero = _mm512_setzero_si512();
            _mm512_cmpneq_epi16_mask(
                _mm512_and_si512(a, _mm512_set1_epi16(1 << (16 - 1) as i16)),
                zero,
            ) as u64
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl I32x16Backend for archmage::Avx512ModernToken {
    type Repr = __m512i;

    #[inline(always)]
    fn splat(v: i32) -> __m512i {
        unsafe { _mm512_set1_epi32(v as _) }
    }

    #[inline(always)]
    fn zero() -> __m512i {
        unsafe { _mm512_setzero_si512() }
    }

    #[inline(always)]
    fn load(data: &[i32; 16]) -> __m512i {
        unsafe { _mm512_loadu_si512(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [i32; 16]) -> __m512i {
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m512i, out: &mut [i32; 16]) {
        unsafe { _mm512_storeu_si512(out.as_mut_ptr().cast(), repr) }
    }

    #[inline(always)]
    fn to_array(repr: __m512i) -> [i32; 16] {
        unsafe { core::mem::transmute(repr) }
    }

    #[inline(always)]
    fn add(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_add_epi32(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi32(a, b) }
    }

    #[inline(always)]
    fn mul(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_mullo_epi32(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi32(_mm512_setzero_si512(), a) }
    }

    #[inline(always)]
    fn min(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_min_epi32(a, b) }
    }

    #[inline(always)]
    fn max(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_max_epi32(a, b) }
    }

    #[inline(always)]
    fn abs(a: __m512i) -> __m512i {
        unsafe { _mm512_abs_epi32(a) }
    }

    #[inline(always)]
    fn simd_eq(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpeq_epi32_mask(a, b);
            _mm512_maskz_set1_epi32(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_ne(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpneq_epi32_mask(a, b);
            _mm512_maskz_set1_epi32(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmplt_epi32_mask(a, b);
            _mm512_maskz_set1_epi32(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_le(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmple_epi32_mask(a, b);
            _mm512_maskz_set1_epi32(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GT = LT with swapped args
            let mask = _mm512_cmplt_epi32_mask(b, a);
            _mm512_maskz_set1_epi32(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_ge(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GE = LE with swapped args
            let mask = _mm512_cmple_epi32_mask(b, a);
            _mm512_maskz_set1_epi32(mask, -1)
        }
    }

    #[inline(always)]
    fn blend(mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        unsafe {
            let k = _mm512_cmpneq_epi32_mask(mask, _mm512_setzero_si512());
            _mm512_mask_blend_epi32(k, if_false, if_true)
        }
    }

    #[inline(always)]
    fn reduce_add(a: __m512i) -> i32 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [i32; 16] = unsafe { core::mem::transmute(a) };
        arr.iter().copied().fold(0i32, i32::wrapping_add)
    }

    #[inline(always)]
    fn not(a: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, _mm512_set1_epi32(-1)) }
    }

    #[inline(always)]
    fn bitand(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_and_si512(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_sll_epi32(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_sra_epi32(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_srl_epi32(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn all_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi32_mask(a, _mm512_setzero_si512());
            mask as u64 == 0xFFFFu64
        }
    }

    #[inline(always)]
    fn any_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi32_mask(a, _mm512_setzero_si512());
            mask as u64 != 0
        }
    }

    #[inline(always)]
    fn bitmask(a: __m512i) -> u64 {
        unsafe {
            // Extract high bit of each lane: compare < 0 for signed interpretation
            let zero = _mm512_setzero_si512();
            _mm512_cmpneq_epi32_mask(
                _mm512_and_si512(a, _mm512_set1_epi32(1 << (32 - 1) as i32)),
                zero,
            ) as u64
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl U32x16Backend for archmage::Avx512ModernToken {
    type Repr = __m512i;

    #[inline(always)]
    fn splat(v: u32) -> __m512i {
        unsafe { _mm512_set1_epi32(v as _) }
    }

    #[inline(always)]
    fn zero() -> __m512i {
        unsafe { _mm512_setzero_si512() }
    }

    #[inline(always)]
    fn load(data: &[u32; 16]) -> __m512i {
        unsafe { _mm512_loadu_si512(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [u32; 16]) -> __m512i {
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m512i, out: &mut [u32; 16]) {
        unsafe { _mm512_storeu_si512(out.as_mut_ptr().cast(), repr) }
    }

    #[inline(always)]
    fn to_array(repr: __m512i) -> [u32; 16] {
        unsafe { core::mem::transmute(repr) }
    }

    #[inline(always)]
    fn add(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_add_epi32(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi32(a, b) }
    }

    #[inline(always)]
    fn mul(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_mullo_epi32(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi32(_mm512_setzero_si512(), a) }
    }

    #[inline(always)]
    fn min(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_min_epu32(a, b) }
    }

    #[inline(always)]
    fn max(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_max_epu32(a, b) }
    }

    #[inline(always)]
    fn simd_eq(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpeq_epu32_mask(a, b);
            _mm512_maskz_set1_epi32(mask, -1i32)
        }
    }

    #[inline(always)]
    fn simd_ne(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpneq_epu32_mask(a, b);
            _mm512_maskz_set1_epi32(mask, -1i32)
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmplt_epu32_mask(a, b);
            _mm512_maskz_set1_epi32(mask, -1i32)
        }
    }

    #[inline(always)]
    fn simd_le(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmple_epu32_mask(a, b);
            _mm512_maskz_set1_epi32(mask, -1i32)
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GT = LT with swapped args
            let mask = _mm512_cmplt_epu32_mask(b, a);
            _mm512_maskz_set1_epi32(mask, -1i32)
        }
    }

    #[inline(always)]
    fn simd_ge(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GE = LE with swapped args
            let mask = _mm512_cmple_epu32_mask(b, a);
            _mm512_maskz_set1_epi32(mask, -1i32)
        }
    }

    #[inline(always)]
    fn blend(mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        unsafe {
            let k = _mm512_cmpneq_epi32_mask(mask, _mm512_setzero_si512());
            _mm512_mask_blend_epi32(k, if_false, if_true)
        }
    }

    #[inline(always)]
    fn reduce_add(a: __m512i) -> u32 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [u32; 16] = unsafe { core::mem::transmute(a) };
        arr.iter().copied().fold(0u32, u32::wrapping_add)
    }

    #[inline(always)]
    fn not(a: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, _mm512_set1_epi32(-1i32)) }
    }

    #[inline(always)]
    fn bitand(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_and_si512(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_sll_epi32(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_srl_epi32(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_srl_epi32(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn all_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi32_mask(a, _mm512_setzero_si512());
            mask as u64 == 0xFFFFu64
        }
    }

    #[inline(always)]
    fn any_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi32_mask(a, _mm512_setzero_si512());
            mask as u64 != 0
        }
    }

    #[inline(always)]
    fn bitmask(a: __m512i) -> u64 {
        unsafe {
            // Extract high bit of each lane: compare < 0 for signed interpretation
            let zero = _mm512_setzero_si512();
            _mm512_cmpneq_epi32_mask(
                _mm512_and_si512(a, _mm512_set1_epi32(1 << (32 - 1) as i32)),
                zero,
            ) as u64
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl I64x8Backend for archmage::Avx512ModernToken {
    type Repr = __m512i;

    #[inline(always)]
    fn splat(v: i64) -> __m512i {
        unsafe { _mm512_set1_epi64(v as _) }
    }

    #[inline(always)]
    fn zero() -> __m512i {
        unsafe { _mm512_setzero_si512() }
    }

    #[inline(always)]
    fn load(data: &[i64; 8]) -> __m512i {
        unsafe { _mm512_loadu_si512(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [i64; 8]) -> __m512i {
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m512i, out: &mut [i64; 8]) {
        unsafe { _mm512_storeu_si512(out.as_mut_ptr().cast(), repr) }
    }

    #[inline(always)]
    fn to_array(repr: __m512i) -> [i64; 8] {
        unsafe { core::mem::transmute(repr) }
    }

    #[inline(always)]
    fn add(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_add_epi64(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi64(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi64(_mm512_setzero_si512(), a) }
    }

    #[inline(always)]
    fn min(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_min_epi64(a, b) }
    }

    #[inline(always)]
    fn max(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_max_epi64(a, b) }
    }

    #[inline(always)]
    fn abs(a: __m512i) -> __m512i {
        unsafe { _mm512_abs_epi64(a) }
    }

    #[inline(always)]
    fn simd_eq(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpeq_epi64_mask(a, b);
            _mm512_maskz_set1_epi64(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_ne(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpneq_epi64_mask(a, b);
            _mm512_maskz_set1_epi64(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmplt_epi64_mask(a, b);
            _mm512_maskz_set1_epi64(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_le(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmple_epi64_mask(a, b);
            _mm512_maskz_set1_epi64(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GT = LT with swapped args
            let mask = _mm512_cmplt_epi64_mask(b, a);
            _mm512_maskz_set1_epi64(mask, -1)
        }
    }

    #[inline(always)]
    fn simd_ge(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GE = LE with swapped args
            let mask = _mm512_cmple_epi64_mask(b, a);
            _mm512_maskz_set1_epi64(mask, -1)
        }
    }

    #[inline(always)]
    fn blend(mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        unsafe {
            let k = _mm512_cmpneq_epi64_mask(mask, _mm512_setzero_si512());
            _mm512_mask_blend_epi64(k, if_false, if_true)
        }
    }

    #[inline(always)]
    fn reduce_add(a: __m512i) -> i64 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [i64; 8] = unsafe { core::mem::transmute(a) };
        arr.iter().copied().fold(0i64, i64::wrapping_add)
    }

    #[inline(always)]
    fn not(a: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, _mm512_set1_epi64(-1)) }
    }

    #[inline(always)]
    fn bitand(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_and_si512(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_sll_epi64(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_sra_epi64(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_srl_epi64(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn all_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi64_mask(a, _mm512_setzero_si512());
            mask as u64 == 0xFFu64
        }
    }

    #[inline(always)]
    fn any_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi64_mask(a, _mm512_setzero_si512());
            mask as u64 != 0
        }
    }

    #[inline(always)]
    fn bitmask(a: __m512i) -> u64 {
        unsafe {
            // Extract high bit of each lane: compare < 0 for signed interpretation
            let zero = _mm512_setzero_si512();
            _mm512_cmpneq_epi64_mask(
                _mm512_and_si512(a, _mm512_set1_epi64(1 << (64 - 1) as i64)),
                zero,
            ) as u64
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl U64x8Backend for archmage::Avx512ModernToken {
    type Repr = __m512i;

    #[inline(always)]
    fn splat(v: u64) -> __m512i {
        unsafe { _mm512_set1_epi64(v as _) }
    }

    #[inline(always)]
    fn zero() -> __m512i {
        unsafe { _mm512_setzero_si512() }
    }

    #[inline(always)]
    fn load(data: &[u64; 8]) -> __m512i {
        unsafe { _mm512_loadu_si512(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [u64; 8]) -> __m512i {
        unsafe { core::mem::transmute(arr) }
    }

    #[inline(always)]
    fn store(repr: __m512i, out: &mut [u64; 8]) {
        unsafe { _mm512_storeu_si512(out.as_mut_ptr().cast(), repr) }
    }

    #[inline(always)]
    fn to_array(repr: __m512i) -> [u64; 8] {
        unsafe { core::mem::transmute(repr) }
    }

    #[inline(always)]
    fn add(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_add_epi64(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi64(a, b) }
    }

    #[inline(always)]
    fn neg(a: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi64(_mm512_setzero_si512(), a) }
    }

    #[inline(always)]
    fn min(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_min_epu64(a, b) }
    }

    #[inline(always)]
    fn max(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_max_epu64(a, b) }
    }

    #[inline(always)]
    fn simd_eq(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpeq_epu64_mask(a, b);
            _mm512_maskz_set1_epi64(mask, -1i64)
        }
    }

    #[inline(always)]
    fn simd_ne(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmpneq_epu64_mask(a, b);
            _mm512_maskz_set1_epi64(mask, -1i64)
        }
    }

    #[inline(always)]
    fn simd_lt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmplt_epu64_mask(a, b);
            _mm512_maskz_set1_epi64(mask, -1i64)
        }
    }

    #[inline(always)]
    fn simd_le(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let mask = _mm512_cmple_epu64_mask(a, b);
            _mm512_maskz_set1_epi64(mask, -1i64)
        }
    }

    #[inline(always)]
    fn simd_gt(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GT = LT with swapped args
            let mask = _mm512_cmplt_epu64_mask(b, a);
            _mm512_maskz_set1_epi64(mask, -1i64)
        }
    }

    #[inline(always)]
    fn simd_ge(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            // GE = LE with swapped args
            let mask = _mm512_cmple_epu64_mask(b, a);
            _mm512_maskz_set1_epi64(mask, -1i64)
        }
    }

    #[inline(always)]
    fn blend(mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        unsafe {
            let k = _mm512_cmpneq_epi64_mask(mask, _mm512_setzero_si512());
            _mm512_mask_blend_epi64(k, if_false, if_true)
        }
    }

    #[inline(always)]
    fn reduce_add(a: __m512i) -> u64 {
        // No native integer reduce_add in AVX-512; use transmute to array
        let arr: [u64; 8] = unsafe { core::mem::transmute(a) };
        arr.iter().copied().fold(0u64, u64::wrapping_add)
    }

    #[inline(always)]
    fn not(a: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, _mm512_set1_epi64(-1i64)) }
    }

    #[inline(always)]
    fn bitand(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_and_si512(a, b) }
    }

    #[inline(always)]
    fn bitor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn bitxor(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_sll_epi64(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_srl_epi64(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: __m512i) -> __m512i {
        unsafe { _mm512_srl_epi64(a, _mm_cvtsi32_si128(N)) }
    }

    #[inline(always)]
    fn all_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi64_mask(a, _mm512_setzero_si512());
            mask as u64 == 0xFFu64
        }
    }

    #[inline(always)]
    fn any_true(a: __m512i) -> bool {
        unsafe {
            let mask = _mm512_cmpneq_epi64_mask(a, _mm512_setzero_si512());
            mask as u64 != 0
        }
    }

    #[inline(always)]
    fn bitmask(a: __m512i) -> u64 {
        unsafe {
            // Extract high bit of each lane: compare < 0 for signed interpretation
            let zero = _mm512_setzero_si512();
            _mm512_cmpneq_epi64_mask(
                _mm512_and_si512(a, _mm512_set1_epi64(1 << (64 - 1) as i64)),
                zero,
            ) as u64
        }
    }
}

// ============================================================================
// Avx512ModernToken — extension: popcnt (VPOPCNTDQ + BITALG)
// ============================================================================

#[cfg(target_arch = "x86_64")]
impl i8x64PopcntBackend for archmage::Avx512ModernToken {
    #[inline(always)]
    fn popcnt(a: __m512i) -> __m512i {
        unsafe { _mm512_popcnt_epi8(a) }
    }
}

#[cfg(target_arch = "x86_64")]
impl u8x64PopcntBackend for archmage::Avx512ModernToken {
    #[inline(always)]
    fn popcnt(a: __m512i) -> __m512i {
        unsafe { _mm512_popcnt_epi8(a) }
    }
}

#[cfg(target_arch = "x86_64")]
impl i16x32PopcntBackend for archmage::Avx512ModernToken {
    #[inline(always)]
    fn popcnt(a: __m512i) -> __m512i {
        unsafe { _mm512_popcnt_epi16(a) }
    }
}

#[cfg(target_arch = "x86_64")]
impl u16x32PopcntBackend for archmage::Avx512ModernToken {
    #[inline(always)]
    fn popcnt(a: __m512i) -> __m512i {
        unsafe { _mm512_popcnt_epi16(a) }
    }
}

#[cfg(target_arch = "x86_64")]
impl i32x16PopcntBackend for archmage::Avx512ModernToken {
    #[inline(always)]
    fn popcnt(a: __m512i) -> __m512i {
        unsafe { _mm512_popcnt_epi32(a) }
    }
}

#[cfg(target_arch = "x86_64")]
impl u32x16PopcntBackend for archmage::Avx512ModernToken {
    #[inline(always)]
    fn popcnt(a: __m512i) -> __m512i {
        unsafe { _mm512_popcnt_epi32(a) }
    }
}

#[cfg(target_arch = "x86_64")]
impl i64x8PopcntBackend for archmage::Avx512ModernToken {
    #[inline(always)]
    fn popcnt(a: __m512i) -> __m512i {
        unsafe { _mm512_popcnt_epi64(a) }
    }
}

#[cfg(target_arch = "x86_64")]
impl u64x8PopcntBackend for archmage::Avx512ModernToken {
    #[inline(always)]
    fn popcnt(a: __m512i) -> __m512i {
        unsafe { _mm512_popcnt_epi64(a) }
    }
}

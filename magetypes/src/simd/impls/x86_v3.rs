//! Backend implementations for X64V3Token (x86-64).
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.
//!
//! # Safety (audit contract — checked backend boundaries)
//!
//! `#[arcane]` checks value intrinsics against the receiver token's features.
//! SSE2-only arithmetic uses the checked x86-64 baseline boundary.
//! Whole-array loads/stores and bit casts use `crate::simd_storage` helpers,
//! which require POD storage and enforce equal sizes at compile time.
//! Token-bearing wrappers never implement the storage POD trait.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use archmage::{X64V3Token, arcane};

use crate::simd::backends::*;

macro_rules! sse2_baseline {
    (fn $name:ident(self, $($arg:ident: $ty:ty),* $(,)?) -> $ret:ty $body:block) => {
        #[inline(always)]
        fn $name(self, $($arg: $ty),*) -> $ret {
            #[target_feature(enable = "sse2")]
            #[inline]
            fn inner($($arg: $ty),*) -> $ret $body
            // SAFETY: SSE2 is guaranteed by the x86-64 architecture. The inner
            // body is safe Rust checked with precisely SSE2, not the AVX tier.
            unsafe { inner($($arg),*) }
        }
    };
}

#[cfg(target_arch = "x86_64")]
impl F32x4Backend for archmage::X64V3Token {
    type Repr = __m128;

    // ====== Construction ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn splat(self, v: f32) -> __m128 {
        _mm_set1_ps(v)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn zero(self) -> __m128 {
        _mm_setzero_ps()
    }

    #[inline(always)]
    fn load(self, data: &[f32; 4]) -> __m128 {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [f32; 4]) -> __m128 {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m128, out: &mut [f32; 4]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m128) -> [f32; 4] {
        crate::simd_storage::cast(repr)
    }

    // ====== Arithmetic ======

    sse2_baseline! {
    fn add(self, a: __m128, b: __m128) -> __m128 {
        _mm_add_ps(a, b)
    }
    }

    sse2_baseline! {
    fn sub(self, a: __m128, b: __m128) -> __m128 {
        _mm_sub_ps(a, b)
    }
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn mul(self, a: __m128, b: __m128) -> __m128 {
        _mm_mul_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn div(self, a: __m128, b: __m128) -> __m128 {
        _mm_div_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn neg(self, a: __m128) -> __m128 {
        _mm_sub_ps(_mm_setzero_ps(), a)
    }

    // ====== Math ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn min(self, a: __m128, b: __m128) -> __m128 {
        _mm_min_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn max(self, a: __m128, b: __m128) -> __m128 {
        _mm_max_ps(a, b)
    }

    sse2_baseline! {
    fn sqrt(self, a: __m128) -> __m128 {
        _mm_sqrt_ps(a)
    }
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn abs(self, a: __m128) -> __m128 {
        let mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFF_FFFFi32));
        _mm_and_ps(a, mask)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn floor(self, a: __m128) -> __m128 {
        _mm_floor_ps(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn ceil(self, a: __m128) -> __m128 {
        _mm_ceil_ps(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn round(self, a: __m128) -> __m128 {
        _mm_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn mul_add(self, a: __m128, b: __m128, c: __m128) -> __m128 {
        _mm_fmadd_ps(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn mul_sub(self, a: __m128, b: __m128, c: __m128) -> __m128 {
        _mm_fmsub_ps(a, b, c)
    }

    // ====== Comparisons ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_eq(self, a: __m128, b: __m128) -> __m128 {
        _mm_cmp_ps::<_CMP_EQ_OQ>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ne(self, a: __m128, b: __m128) -> __m128 {
        _mm_cmp_ps::<_CMP_NEQ_OQ>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_lt(self, a: __m128, b: __m128) -> __m128 {
        _mm_cmp_ps::<_CMP_LT_OQ>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_le(self, a: __m128, b: __m128) -> __m128 {
        _mm_cmp_ps::<_CMP_LE_OQ>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_gt(self, a: __m128, b: __m128) -> __m128 {
        _mm_cmp_ps::<_CMP_GT_OQ>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ge(self, a: __m128, b: __m128) -> __m128 {
        _mm_cmp_ps::<_CMP_GE_OQ>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn blend(self, mask: __m128, if_true: __m128, if_false: __m128) -> __m128 {
        _mm_blendv_ps(if_false, if_true, mask)
    }

    // ====== Reductions ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_add(self, a: __m128) -> f32 {
        {
            let shuf = _mm_shuffle_ps::<0b10_11_00_01>(a, a);
            let s1 = _mm_add_ps(a, shuf);
            let s2 = _mm_add_ps(s1, _mm_movehl_ps(s1, s1));
            _mm_cvtss_f32(s2)
        }
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_min(self, a: __m128) -> f32 {
        {
            let shuf = _mm_shuffle_ps::<0b10_11_00_01>(a, a);
            let m1 = _mm_min_ps(a, shuf);
            let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
            let m2 = _mm_min_ps(m1, shuf2);
            _mm_cvtss_f32(m2)
        }
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_max(self, a: __m128) -> f32 {
        {
            let shuf = _mm_shuffle_ps::<0b10_11_00_01>(a, a);
            let m1 = _mm_max_ps(a, shuf);
            let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
            let m2 = _mm_max_ps(m1, shuf2);
            _mm_cvtss_f32(m2)
        }
    }

    // ====== Approximations ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn rcp_approx(self, a: __m128) -> __m128 {
        _mm_rcp_ps(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn rsqrt_approx(self, a: __m128) -> __m128 {
        _mm_rsqrt_ps(a)
    }

    // Estimate + one FMA-Newton step + a branchless rail rescue: the
    // fastest form meeting the bare-name contract (<= 4 ULP AND exact
    // IEEE rails at ±0/±inf/NaN). The Newton error term `e = 1 - a*r`
    // is NaN exactly on rail lanes (inf*0), so an unordered _self-
    // compare selects them and blendv rescues those lanes to the raw
    // estimate — which is rail-EXACT on x86 (rcpps(±0) = ±inf,
    // rcpps(±inf) = ±0, signed; likewise rsqrtps). Measured cost of
    // the rescue vs the plain Newton form on Zen 5: recip +2.6%,
    // rsqrt +20% — vs +21%/+232% for exact division
    // (benchmarks/recip_x86_zen5-9950x3d_2026-09-03.md). Subnormal
    // inputs remain unspecified at this tier (rcpps flushes them);
    // `recip_portable`/`rsqrt_portable` divide: 0 ULP + subnormals.
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn recip(self, a: __m128) -> __m128 {
        let one = { _mm_set1_ps(1.0) };
        let r = <Self as F32x4Backend>::rcp_approx(_self, a);
        {
            let e = _mm_fnmadd_ps(a, r, one);
            let refined = _mm_fmadd_ps(r, e, r);
            let bad = _mm_cmp_ps::<_CMP_UNORD_Q>(e, e);
            _mm_blendv_ps(refined, r, bad)
        }
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn rsqrt(self, a: __m128) -> __m128 {
        let half = { _mm_set1_ps(0.5) };
        let three_halves = { _mm_set1_ps(1.5) };
        let y = <Self as F32x4Backend>::rsqrt_approx(_self, a);
        {
            let t = _mm_mul_ps(a, _mm_mul_ps(y, y));
            let refined = _mm_mul_ps(y, _mm_fnmadd_ps(half, t, three_halves));
            let bad = _mm_cmp_ps::<_CMP_UNORD_Q>(t, t);
            _mm_blendv_ps(refined, y, bad)
        }
    }

    // ====== Bitwise ======

    sse2_baseline! {
    fn not(self, a: __m128) -> __m128 {
        let ones = _mm_set1_epi32(-1);
        let as_int = _mm_castps_si128(a);
        _mm_castsi128_ps(_mm_xor_si128(as_int, ones))
    }
    }

    sse2_baseline! {
    fn bitand(self, a: __m128, b: __m128) -> __m128 {
        _mm_and_ps(a, b)
    }
    }

    sse2_baseline! {
    fn bitor(self, a: __m128, b: __m128) -> __m128 {
        _mm_or_ps(a, b)
    }
    }

    sse2_baseline! {
    fn bitxor(self, a: __m128, b: __m128) -> __m128 {
        _mm_xor_ps(a, b)
    }
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn to_u8_bytes(self, a: __m128) -> [u8; 4] {
        let i32s = _mm_cvtps_epi32(a);
        let i16s = _mm_packs_epi32(i32s, i32s);
        let u8s = _mm_packus_epi16(i16s, i16s);
        (_mm_cvtsi128_si32(u8s) as u32).to_ne_bytes()
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn store_rgba_bytes(self, r: __m128, g: __m128, b: __m128, a: __m128) -> [u8; 16] {
        let rg = _mm_packs_epi32(_mm_cvtps_epi32(r), _mm_cvtps_epi32(g));
        let ba = _mm_packs_epi32(_mm_cvtps_epi32(b), _mm_cvtps_epi32(a));
        // [R0-3,G0-3,B0-3,A0-3] -> interleaved RGBA pixels 0-3.
        let packed = _mm_packus_epi16(rg, ba);
        let shuf = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
        crate::simd_storage::cast(_mm_shuffle_epi8(packed, shuf))
    }
}

#[cfg(target_arch = "x86_64")]
impl F32x8Backend for archmage::X64V3Token {
    type Repr = __m256;

    // ====== Construction ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn splat(self, v: f32) -> __m256 {
        _mm256_set1_ps(v)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn zero(self) -> __m256 {
        _mm256_setzero_ps()
    }

    #[inline(always)]
    fn load(self, data: &[f32; 8]) -> __m256 {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [f32; 8]) -> __m256 {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m256, out: &mut [f32; 8]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m256) -> [f32; 8] {
        crate::simd_storage::cast(repr)
    }

    // ====== Arithmetic ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn add(self, a: __m256, b: __m256) -> __m256 {
        _mm256_add_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn sub(self, a: __m256, b: __m256) -> __m256 {
        _mm256_sub_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn mul(self, a: __m256, b: __m256) -> __m256 {
        _mm256_mul_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn div(self, a: __m256, b: __m256) -> __m256 {
        _mm256_div_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn neg(self, a: __m256) -> __m256 {
        _mm256_sub_ps(_mm256_setzero_ps(), a)
    }

    // ====== Math ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn min(self, a: __m256, b: __m256) -> __m256 {
        _mm256_min_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn max(self, a: __m256, b: __m256) -> __m256 {
        _mm256_max_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn sqrt(self, a: __m256) -> __m256 {
        _mm256_sqrt_ps(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn abs(self, a: __m256) -> __m256 {
        let mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFFi32));
        _mm256_and_ps(a, mask)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn floor(self, a: __m256) -> __m256 {
        _mm256_floor_ps(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn ceil(self, a: __m256) -> __m256 {
        _mm256_ceil_ps(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn round(self, a: __m256) -> __m256 {
        _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn mul_add(self, a: __m256, b: __m256, c: __m256) -> __m256 {
        _mm256_fmadd_ps(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn mul_sub(self, a: __m256, b: __m256, c: __m256) -> __m256 {
        _mm256_fmsub_ps(a, b, c)
    }

    // ====== Comparisons ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_eq(self, a: __m256, b: __m256) -> __m256 {
        _mm256_cmp_ps::<_CMP_EQ_OQ>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ne(self, a: __m256, b: __m256) -> __m256 {
        _mm256_cmp_ps::<_CMP_NEQ_OQ>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_lt(self, a: __m256, b: __m256) -> __m256 {
        _mm256_cmp_ps::<_CMP_LT_OQ>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_le(self, a: __m256, b: __m256) -> __m256 {
        _mm256_cmp_ps::<_CMP_LE_OQ>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_gt(self, a: __m256, b: __m256) -> __m256 {
        _mm256_cmp_ps::<_CMP_GT_OQ>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ge(self, a: __m256, b: __m256) -> __m256 {
        _mm256_cmp_ps::<_CMP_GE_OQ>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn blend(self, mask: __m256, if_true: __m256, if_false: __m256) -> __m256 {
        _mm256_blendv_ps(if_false, if_true, mask)
    }

    // ====== Reductions ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_add(self, a: __m256) -> f32 {
        {
            let hi = _mm256_extractf128_ps::<1>(a);
            let lo = _mm256_castps256_ps128(a);
            let sum = _mm_add_ps(lo, hi);
            let shuf = _mm_shuffle_ps::<0b10_11_00_01>(sum, sum);
            let s1 = _mm_add_ps(sum, shuf);
            let s2 = _mm_add_ps(s1, _mm_movehl_ps(s1, s1));
            _mm_cvtss_f32(s2)
        }
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_min(self, a: __m256) -> f32 {
        {
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

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_max(self, a: __m256) -> f32 {
        {
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

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn rcp_approx(self, a: __m256) -> __m256 {
        _mm256_rcp_ps(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn rsqrt_approx(self, a: __m256) -> __m256 {
        _mm256_rsqrt_ps(a)
    }

    // Estimate + one FMA-Newton step + a branchless rail rescue: the
    // fastest form meeting the bare-name contract (<= 4 ULP AND exact
    // IEEE rails at ±0/±inf/NaN). The Newton error term `e = 1 - a*r`
    // is NaN exactly on rail lanes (inf*0), so an unordered _self-
    // compare selects them and blendv rescues those lanes to the raw
    // estimate — which is rail-EXACT on x86 (rcpps(±0) = ±inf,
    // rcpps(±inf) = ±0, signed; likewise rsqrtps). Measured cost of
    // the rescue vs the plain Newton form on Zen 5: recip +2.6%,
    // rsqrt +20% — vs +21%/+232% for exact division
    // (benchmarks/recip_x86_zen5-9950x3d_2026-09-03.md). Subnormal
    // inputs remain unspecified at this tier (rcpps flushes them);
    // `recip_portable`/`rsqrt_portable` divide: 0 ULP + subnormals.
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn recip(self, a: __m256) -> __m256 {
        let one = { _mm256_set1_ps(1.0) };
        let r = <Self as F32x8Backend>::rcp_approx(_self, a);
        {
            let e = _mm256_fnmadd_ps(a, r, one);
            let refined = _mm256_fmadd_ps(r, e, r);
            let bad = _mm256_cmp_ps::<_CMP_UNORD_Q>(e, e);
            _mm256_blendv_ps(refined, r, bad)
        }
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn rsqrt(self, a: __m256) -> __m256 {
        let half = { _mm256_set1_ps(0.5) };
        let three_halves = { _mm256_set1_ps(1.5) };
        let y = <Self as F32x8Backend>::rsqrt_approx(_self, a);
        {
            let t = _mm256_mul_ps(a, _mm256_mul_ps(y, y));
            let refined = _mm256_mul_ps(y, _mm256_fnmadd_ps(half, t, three_halves));
            let bad = _mm256_cmp_ps::<_CMP_UNORD_Q>(t, t);
            _mm256_blendv_ps(refined, y, bad)
        }
    }

    // ====== Bitwise ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn not(self, a: __m256) -> __m256 {
        let ones = _mm256_set1_epi32(-1);
        let as_int = _mm256_castps_si256(a);
        _mm256_castsi256_ps(_mm256_xor_si256(as_int, ones))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitand(self, a: __m256, b: __m256) -> __m256 {
        _mm256_and_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitor(self, a: __m256, b: __m256) -> __m256 {
        _mm256_or_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitxor(self, a: __m256, b: __m256) -> __m256 {
        _mm256_xor_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn to_u8_bytes(self, a: __m256) -> [u8; 8] {
        let i32s = _mm256_cvtps_epi32(a);
        let lo = _mm256_castsi256_si128(i32s);
        let hi = _mm256_extracti128_si256::<1>(i32s);
        let i16s = _mm_packs_epi32(lo, hi);
        let u8s = _mm_packus_epi16(i16s, i16s);
        (_mm_cvtsi128_si64(u8s) as u64).to_ne_bytes()
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn store_rgba_bytes(self, r: __m256, g: __m256, b: __m256, a: __m256) -> [u8; 32] {
        // AVX2 packs are lane-wise: lane0 holds pixels 0-3, lane1 4-7.
        let rg = _mm256_packs_epi32(_mm256_cvtps_epi32(r), _mm256_cvtps_epi32(g));
        let ba = _mm256_packs_epi32(_mm256_cvtps_epi32(b), _mm256_cvtps_epi32(a));
        let packed = _mm256_packus_epi16(rg, ba);
        let shuf = _mm256_setr_epi8(
            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6,
            10, 14, 3, 7, 11, 15,
        );
        crate::simd_storage::cast(_mm256_shuffle_epi8(packed, shuf))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn transpose_8x8_repr(self, rows: [__m256; 8]) -> [__m256; 8] {
        let t0 = _mm256_unpacklo_ps(rows[0], rows[1]);
        let t1 = _mm256_unpackhi_ps(rows[0], rows[1]);
        let t2 = _mm256_unpacklo_ps(rows[2], rows[3]);
        let t3 = _mm256_unpackhi_ps(rows[2], rows[3]);
        let t4 = _mm256_unpacklo_ps(rows[4], rows[5]);
        let t5 = _mm256_unpackhi_ps(rows[4], rows[5]);
        let t6 = _mm256_unpacklo_ps(rows[6], rows[7]);
        let t7 = _mm256_unpackhi_ps(rows[6], rows[7]);
        let s0 = _mm256_shuffle_ps::<0x44>(t0, t2);
        let s1 = _mm256_shuffle_ps::<0xEE>(t0, t2);
        let s2 = _mm256_shuffle_ps::<0x44>(t1, t3);
        let s3 = _mm256_shuffle_ps::<0xEE>(t1, t3);
        let s4 = _mm256_shuffle_ps::<0x44>(t4, t6);
        let s5 = _mm256_shuffle_ps::<0xEE>(t4, t6);
        let s6 = _mm256_shuffle_ps::<0x44>(t5, t7);
        let s7 = _mm256_shuffle_ps::<0xEE>(t5, t7);
        [
            _mm256_permute2f128_ps::<0x20>(s0, s4),
            _mm256_permute2f128_ps::<0x20>(s1, s5),
            _mm256_permute2f128_ps::<0x20>(s2, s6),
            _mm256_permute2f128_ps::<0x20>(s3, s7),
            _mm256_permute2f128_ps::<0x31>(s0, s4),
            _mm256_permute2f128_ps::<0x31>(s1, s5),
            _mm256_permute2f128_ps::<0x31>(s2, s6),
            _mm256_permute2f128_ps::<0x31>(s3, s7),
        ]
    }
}

#[cfg(target_arch = "x86_64")]
impl F64x2Backend for archmage::X64V3Token {
    type Repr = __m128d;

    // ====== Construction ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn splat(self, v: f64) -> __m128d {
        _mm_set1_pd(v)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn zero(self) -> __m128d {
        _mm_setzero_pd()
    }

    #[inline(always)]
    fn load(self, data: &[f64; 2]) -> __m128d {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [f64; 2]) -> __m128d {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m128d, out: &mut [f64; 2]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m128d) -> [f64; 2] {
        crate::simd_storage::cast(repr)
    }

    // ====== Arithmetic ======

    sse2_baseline! {
    fn add(self, a: __m128d, b: __m128d) -> __m128d {
        _mm_add_pd(a, b)
    }
    }

    sse2_baseline! {
    fn sub(self, a: __m128d, b: __m128d) -> __m128d {
        _mm_sub_pd(a, b)
    }
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn mul(self, a: __m128d, b: __m128d) -> __m128d {
        _mm_mul_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn div(self, a: __m128d, b: __m128d) -> __m128d {
        _mm_div_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn neg(self, a: __m128d) -> __m128d {
        _mm_sub_pd(_mm_setzero_pd(), a)
    }

    // ====== Math ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn min(self, a: __m128d, b: __m128d) -> __m128d {
        _mm_min_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn max(self, a: __m128d, b: __m128d) -> __m128d {
        _mm_max_pd(a, b)
    }

    sse2_baseline! {
    fn sqrt(self, a: __m128d) -> __m128d {
        _mm_sqrt_pd(a)
    }
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn abs(self, a: __m128d) -> __m128d {
        let mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFF_FFFF_FFFF_FFFFi64));
        _mm_and_pd(a, mask)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn floor(self, a: __m128d) -> __m128d {
        _mm_floor_pd(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn ceil(self, a: __m128d) -> __m128d {
        _mm_ceil_pd(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn round(self, a: __m128d) -> __m128d {
        _mm_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn mul_add(self, a: __m128d, b: __m128d, c: __m128d) -> __m128d {
        _mm_fmadd_pd(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn mul_sub(self, a: __m128d, b: __m128d, c: __m128d) -> __m128d {
        _mm_fmsub_pd(a, b, c)
    }

    // ====== Comparisons ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_eq(self, a: __m128d, b: __m128d) -> __m128d {
        _mm_cmp_pd::<_CMP_EQ_OQ>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ne(self, a: __m128d, b: __m128d) -> __m128d {
        _mm_cmp_pd::<_CMP_NEQ_OQ>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_lt(self, a: __m128d, b: __m128d) -> __m128d {
        _mm_cmp_pd::<_CMP_LT_OQ>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_le(self, a: __m128d, b: __m128d) -> __m128d {
        _mm_cmp_pd::<_CMP_LE_OQ>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_gt(self, a: __m128d, b: __m128d) -> __m128d {
        _mm_cmp_pd::<_CMP_GT_OQ>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ge(self, a: __m128d, b: __m128d) -> __m128d {
        _mm_cmp_pd::<_CMP_GE_OQ>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn blend(self, mask: __m128d, if_true: __m128d, if_false: __m128d) -> __m128d {
        _mm_blendv_pd(if_false, if_true, mask)
    }

    // ====== Reductions ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_add(self, a: __m128d) -> f64 {
        {
            let h = _mm_hadd_pd(a, a);
            _mm_cvtsd_f64(h)
        }
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_min(self, a: __m128d) -> f64 {
        {
            let shuf = _mm_shuffle_pd::<0b01>(a, a);
            let m = _mm_min_pd(a, shuf);
            _mm_cvtsd_f64(m)
        }
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_max(self, a: __m128d) -> f64 {
        {
            let shuf = _mm_shuffle_pd::<0b01>(a, a);
            let m = _mm_max_pd(a, shuf);
            _mm_cvtsd_f64(m)
        }
    }

    // ====== Approximations ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn rcp_approx(self, a: __m128d) -> __m128d {
        let one = { _mm_set1_pd(1.0) };
        <Self as F64x2Backend>::div(_self, one, a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn rsqrt_approx(self, a: __m128d) -> __m128d {
        let one = { _mm_set1_pd(1.0) };
        <Self as F64x2Backend>::div(_self, one, <Self as F64x2Backend>::sqrt(_self, a))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn recip(self, a: __m128d) -> __m128d {
        let one = { _mm_set1_pd(1.0) };
        <Self as F64x2Backend>::div(_self, one, a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn rsqrt(self, a: __m128d) -> __m128d {
        let one = { _mm_set1_pd(1.0) };
        <Self as F64x2Backend>::div(_self, one, <Self as F64x2Backend>::sqrt(_self, a))
    }

    // ====== Bitwise ======

    sse2_baseline! {
    fn not(self, a: __m128d) -> __m128d {
        let ones = _mm_set1_epi64x(-1);
        let as_int = _mm_castpd_si128(a);
        _mm_castsi128_pd(_mm_xor_si128(as_int, ones))
    }
    }

    sse2_baseline! {
    fn bitand(self, a: __m128d, b: __m128d) -> __m128d {
        _mm_and_pd(a, b)
    }
    }

    sse2_baseline! {
    fn bitor(self, a: __m128d, b: __m128d) -> __m128d {
        _mm_or_pd(a, b)
    }
    }

    sse2_baseline! {
    fn bitxor(self, a: __m128d, b: __m128d) -> __m128d {
        _mm_xor_pd(a, b)
    }
    }
}

#[cfg(target_arch = "x86_64")]
impl F64x4Backend for archmage::X64V3Token {
    type Repr = __m256d;

    // ====== Construction ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn splat(self, v: f64) -> __m256d {
        _mm256_set1_pd(v)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn zero(self) -> __m256d {
        _mm256_setzero_pd()
    }

    #[inline(always)]
    fn load(self, data: &[f64; 4]) -> __m256d {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [f64; 4]) -> __m256d {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m256d, out: &mut [f64; 4]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m256d) -> [f64; 4] {
        crate::simd_storage::cast(repr)
    }

    // ====== Arithmetic ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn add(self, a: __m256d, b: __m256d) -> __m256d {
        _mm256_add_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn sub(self, a: __m256d, b: __m256d) -> __m256d {
        _mm256_sub_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn mul(self, a: __m256d, b: __m256d) -> __m256d {
        _mm256_mul_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn div(self, a: __m256d, b: __m256d) -> __m256d {
        _mm256_div_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn neg(self, a: __m256d) -> __m256d {
        _mm256_sub_pd(_mm256_setzero_pd(), a)
    }

    // ====== Math ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn min(self, a: __m256d, b: __m256d) -> __m256d {
        _mm256_min_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn max(self, a: __m256d, b: __m256d) -> __m256d {
        _mm256_max_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn sqrt(self, a: __m256d) -> __m256d {
        _mm256_sqrt_pd(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn abs(self, a: __m256d) -> __m256d {
        let mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFF_FFFF_FFFF_FFFFi64));
        _mm256_and_pd(a, mask)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn floor(self, a: __m256d) -> __m256d {
        _mm256_floor_pd(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn ceil(self, a: __m256d) -> __m256d {
        _mm256_ceil_pd(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn round(self, a: __m256d) -> __m256d {
        _mm256_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn mul_add(self, a: __m256d, b: __m256d, c: __m256d) -> __m256d {
        _mm256_fmadd_pd(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn mul_sub(self, a: __m256d, b: __m256d, c: __m256d) -> __m256d {
        _mm256_fmsub_pd(a, b, c)
    }

    // ====== Comparisons ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_eq(self, a: __m256d, b: __m256d) -> __m256d {
        _mm256_cmp_pd::<_CMP_EQ_OQ>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ne(self, a: __m256d, b: __m256d) -> __m256d {
        _mm256_cmp_pd::<_CMP_NEQ_OQ>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_lt(self, a: __m256d, b: __m256d) -> __m256d {
        _mm256_cmp_pd::<_CMP_LT_OQ>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_le(self, a: __m256d, b: __m256d) -> __m256d {
        _mm256_cmp_pd::<_CMP_LE_OQ>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_gt(self, a: __m256d, b: __m256d) -> __m256d {
        _mm256_cmp_pd::<_CMP_GT_OQ>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ge(self, a: __m256d, b: __m256d) -> __m256d {
        _mm256_cmp_pd::<_CMP_GE_OQ>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn blend(self, mask: __m256d, if_true: __m256d, if_false: __m256d) -> __m256d {
        _mm256_blendv_pd(if_false, if_true, mask)
    }

    // ====== Reductions ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_add(self, a: __m256d) -> f64 {
        {
            let hi = _mm256_extractf128_pd::<1>(a);
            let lo = _mm256_castpd256_pd128(a);
            let sum = _mm_add_pd(lo, hi);
            let h = _mm_hadd_pd(sum, sum);
            _mm_cvtsd_f64(h)
        }
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_min(self, a: __m256d) -> f64 {
        {
            let hi = _mm256_extractf128_pd::<1>(a);
            let lo = _mm256_castpd256_pd128(a);
            let m = _mm_min_pd(lo, hi);
            let shuf = _mm_shuffle_pd::<0b01>(m, m);
            let m2 = _mm_min_pd(m, shuf);
            _mm_cvtsd_f64(m2)
        }
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_max(self, a: __m256d) -> f64 {
        {
            let hi = _mm256_extractf128_pd::<1>(a);
            let lo = _mm256_castpd256_pd128(a);
            let m = _mm_max_pd(lo, hi);
            let shuf = _mm_shuffle_pd::<0b01>(m, m);
            let m2 = _mm_max_pd(m, shuf);
            _mm_cvtsd_f64(m2)
        }
    }

    // ====== Approximations ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn rcp_approx(self, a: __m256d) -> __m256d {
        let one = { _mm256_set1_pd(1.0) };
        <Self as F64x4Backend>::div(_self, one, a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn rsqrt_approx(self, a: __m256d) -> __m256d {
        let one = { _mm256_set1_pd(1.0) };
        <Self as F64x4Backend>::div(_self, one, <Self as F64x4Backend>::sqrt(_self, a))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn recip(self, a: __m256d) -> __m256d {
        let one = { _mm256_set1_pd(1.0) };
        <Self as F64x4Backend>::div(_self, one, a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn rsqrt(self, a: __m256d) -> __m256d {
        let one = { _mm256_set1_pd(1.0) };
        <Self as F64x4Backend>::div(_self, one, <Self as F64x4Backend>::sqrt(_self, a))
    }

    // ====== Bitwise ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn not(self, a: __m256d) -> __m256d {
        let ones = _mm256_set1_epi64x(-1);
        let as_int = _mm256_castpd_si256(a);
        _mm256_castsi256_pd(_mm256_xor_si256(as_int, ones))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitand(self, a: __m256d, b: __m256d) -> __m256d {
        _mm256_and_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitor(self, a: __m256d, b: __m256d) -> __m256d {
        _mm256_or_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitxor(self, a: __m256d, b: __m256d) -> __m256d {
        _mm256_xor_pd(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
impl I32x4Backend for archmage::X64V3Token {
    type Repr = __m128i;

    // ====== Construction ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn splat(self, v: i32) -> __m128i {
        _mm_set1_epi32(v)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn zero(self) -> __m128i {
        _mm_setzero_si128()
    }

    #[inline(always)]
    fn load(self, data: &[i32; 4]) -> __m128i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i32; 4]) -> __m128i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m128i, out: &mut [i32; 4]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m128i) -> [i32; 4] {
        crate::simd_storage::cast(repr)
    }

    // ====== Arithmetic ======

    sse2_baseline! {
    fn add(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_add_epi32(a, b)
    }
    }

    sse2_baseline! {
    fn sub(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_sub_epi32(a, b)
    }
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn mul(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_mullo_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn neg(self, a: __m128i) -> __m128i {
        _mm_sub_epi32(_mm_setzero_si128(), a)
    }

    // ====== Math ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn min(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_min_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn max(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_max_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn abs(self, a: __m128i) -> __m128i {
        _mm_abs_epi32(a)
    }

    // ====== Comparisons ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_eq(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_cmpeq_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ne(self, a: __m128i, b: __m128i) -> __m128i {
        let eq = _mm_cmpeq_epi32(a, b);
        _mm_andnot_si128(eq, _mm_set1_epi32(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_lt(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_cmpgt_epi32(b, a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_le(self, a: __m128i, b: __m128i) -> __m128i {
        let gt = _mm_cmpgt_epi32(a, b);
        _mm_andnot_si128(gt, _mm_set1_epi32(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_gt(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_cmpgt_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ge(self, a: __m128i, b: __m128i) -> __m128i {
        let lt = _mm_cmpgt_epi32(b, a);
        _mm_andnot_si128(lt, _mm_set1_epi32(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn blend(self, mask: __m128i, if_true: __m128i, if_false: __m128i) -> __m128i {
        _mm_blendv_epi8(if_false, if_true, mask)
    }

    // ====== Reductions ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_add(self, a: __m128i) -> i32 {
        {
            let hi = _mm_shuffle_epi32::<0b01_00_11_10>(a);
            let sum = _mm_add_epi32(a, hi);
            let hi2 = _mm_shuffle_epi32::<0b00_00_00_01>(sum);
            let sum2 = _mm_add_epi32(sum, hi2);
            _mm_cvtsi128_si32(sum2)
        }
    }

    // ====== Bitwise ======

    sse2_baseline! {
    fn not(self, a: __m128i) -> __m128i {
        _mm_andnot_si128(a, _mm_set1_epi32(-1))
    }
    }

    sse2_baseline! {
    fn bitand(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_and_si128(a, b)
    }
    }

    sse2_baseline! {
    fn bitor(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_or_si128(a, b)
    }
    }

    sse2_baseline! {
    fn bitxor(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_xor_si128(a, b)
    }
    }

    // ====== Shifts ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_const<const N: i32>(self, a: __m128i) -> __m128i {
        _mm_slli_epi32::<N>(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m128i) -> __m128i {
        _mm_srai_epi32::<N>(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_const<const N: i32>(self, a: __m128i) -> __m128i {
        _mm_srli_epi32::<N>(a)
    }

    // ====== Uniform variable shifts ======
    // PSLLD/PSRLD give 0 once the count reaches 32 and PSRAD clamps to
    // a sign fill — the portable contract is the hardware behaviour.
    // _mm_cvtsi32_si128 zero-extends, so u32::MAX is just a large
    // out-of-range count, not a negative one.

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_uniform(self, a: __m128i, count: u32) -> __m128i {
        _mm_sll_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_uniform(self, a: __m128i, count: u32) -> __m128i {
        _mm_srl_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_arithmetic_uniform(self, a: __m128i, count: u32) -> __m128i {
        _mm_sra_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    // ====== Boolean ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn all_true(self, a: __m128i) -> bool {
        _mm_movemask_ps(_mm_castsi128_ps(a)) == 0xF
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn any_true(self, a: __m128i) -> bool {
        _mm_movemask_ps(_mm_castsi128_ps(a)) != 0
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitmask(self, a: __m128i) -> u32 {
        _mm_movemask_ps(_mm_castsi128_ps(a)) as u32
    }
}

#[cfg(target_arch = "x86_64")]
impl I32x8Backend for archmage::X64V3Token {
    type Repr = __m256i;

    // ====== Construction ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn splat(self, v: i32) -> __m256i {
        _mm256_set1_epi32(v)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn zero(self) -> __m256i {
        _mm256_setzero_si256()
    }

    #[inline(always)]
    fn load(self, data: &[i32; 8]) -> __m256i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i32; 8]) -> __m256i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m256i, out: &mut [i32; 8]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m256i) -> [i32; 8] {
        crate::simd_storage::cast(repr)
    }

    // ====== Arithmetic ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn add(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_add_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn sub(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_sub_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn mul(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_mullo_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn neg(self, a: __m256i) -> __m256i {
        _mm256_sub_epi32(_mm256_setzero_si256(), a)
    }

    // ====== Math ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn min(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_min_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn max(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_max_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn abs(self, a: __m256i) -> __m256i {
        _mm256_abs_epi32(a)
    }

    // ====== Comparisons ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_eq(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpeq_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ne(self, a: __m256i, b: __m256i) -> __m256i {
        let eq = _mm256_cmpeq_epi32(a, b);
        _mm256_andnot_si256(eq, _mm256_set1_epi32(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_lt(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpgt_epi32(b, a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_le(self, a: __m256i, b: __m256i) -> __m256i {
        let gt = _mm256_cmpgt_epi32(a, b);
        _mm256_andnot_si256(gt, _mm256_set1_epi32(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_gt(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpgt_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ge(self, a: __m256i, b: __m256i) -> __m256i {
        let lt = _mm256_cmpgt_epi32(b, a);
        _mm256_andnot_si256(lt, _mm256_set1_epi32(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn blend(self, mask: __m256i, if_true: __m256i, if_false: __m256i) -> __m256i {
        _mm256_blendv_epi8(if_false, if_true, mask)
    }

    // ====== Reductions ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_add(self, a: __m256i) -> i32 {
        {
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

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn not(self, a: __m256i) -> __m256i {
        _mm256_andnot_si256(a, _mm256_set1_epi32(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitand(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_and_si256(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitor(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_or_si256(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitxor(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_xor_si256(a, b)
    }

    // ====== Shifts ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_const<const N: i32>(self, a: __m256i) -> __m256i {
        _mm256_slli_epi32::<N>(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m256i) -> __m256i {
        _mm256_srai_epi32::<N>(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_const<const N: i32>(self, a: __m256i) -> __m256i {
        _mm256_srli_epi32::<N>(a)
    }

    // ====== Uniform variable shifts ======
    // PSLLD/PSRLD give 0 once the count reaches 32 and PSRAD clamps to
    // a sign fill — the portable contract is the hardware behaviour.
    // _mm_cvtsi32_si128 zero-extends, so u32::MAX is just a large
    // out-of-range count, not a negative one.

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_uniform(self, a: __m256i, count: u32) -> __m256i {
        _mm256_sll_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_uniform(self, a: __m256i, count: u32) -> __m256i {
        _mm256_srl_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_arithmetic_uniform(self, a: __m256i, count: u32) -> __m256i {
        _mm256_sra_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    // ====== Boolean ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn all_true(self, a: __m256i) -> bool {
        _mm256_movemask_ps(_mm256_castsi256_ps(a)) == 0xFF
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn any_true(self, a: __m256i) -> bool {
        _mm256_movemask_ps(_mm256_castsi256_ps(a)) != 0
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitmask(self, a: __m256i) -> u32 {
        _mm256_movemask_ps(_mm256_castsi256_ps(a)) as u32
    }
}

#[cfg(target_arch = "x86_64")]
impl U32x4Backend for archmage::X64V3Token {
    type Repr = __m128i;

    // ====== Construction ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn splat(self, v: u32) -> __m128i {
        _mm_set1_epi32(v as i32)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn zero(self) -> __m128i {
        _mm_setzero_si128()
    }

    #[inline(always)]
    fn load(self, data: &[u32; 4]) -> __m128i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u32; 4]) -> __m128i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m128i, out: &mut [u32; 4]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m128i) -> [u32; 4] {
        crate::simd_storage::cast(repr)
    }

    // ====== Arithmetic ======

    sse2_baseline! {
    fn add(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_add_epi32(a, b)
    }
    }

    sse2_baseline! {
    fn sub(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_sub_epi32(a, b)
    }
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn mul(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_mullo_epi32(a, b)
    }

    // ====== Math ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn min(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_min_epu32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn max(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_max_epu32(a, b)
    }

    // ====== Comparisons ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_eq(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_cmpeq_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ne(self, a: __m128i, b: __m128i) -> __m128i {
        let eq = _mm_cmpeq_epi32(a, b);
        _mm_andnot_si128(eq, _mm_set1_epi32(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_gt(self, a: __m128i, b: __m128i) -> __m128i {
        // Unsigned comparison via bias trick: XOR both with 0x80000000
        // to convert to signed range, then use signed cmpgt.
        {
            let bias = _mm_set1_epi32(i32::MIN);
            let sa = _mm_xor_si128(a, bias);
            let sb = _mm_xor_si128(b, bias);
            _mm_cmpgt_epi32(sa, sb)
        }
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_lt(self, a: __m128i, b: __m128i) -> __m128i {
        <Self as U32x4Backend>::simd_gt(_self, b, a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_le(self, a: __m128i, b: __m128i) -> __m128i {
        let gt = <Self as U32x4Backend>::simd_gt(_self, a, b);
        _mm_andnot_si128(gt, _mm_set1_epi32(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ge(self, a: __m128i, b: __m128i) -> __m128i {
        let lt = <Self as U32x4Backend>::simd_gt(_self, b, a);
        _mm_andnot_si128(lt, _mm_set1_epi32(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn blend(self, mask: __m128i, if_true: __m128i, if_false: __m128i) -> __m128i {
        _mm_blendv_epi8(if_false, if_true, mask)
    }

    // ====== Reductions ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_add(self, a: __m128i) -> u32 {
        {
            let hi = _mm_shuffle_epi32::<0b01_00_11_10>(a);
            let sum = _mm_add_epi32(a, hi);
            let hi2 = _mm_shuffle_epi32::<0b00_00_00_01>(sum);
            let sum2 = _mm_add_epi32(sum, hi2);
            _mm_cvtsi128_si32(sum2) as u32
        }
    }

    // ====== Bitwise ======

    sse2_baseline! {
    fn not(self, a: __m128i) -> __m128i {
        _mm_andnot_si128(a, _mm_set1_epi32(-1))
    }
    }

    sse2_baseline! {
    fn bitand(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_and_si128(a, b)
    }
    }

    sse2_baseline! {
    fn bitor(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_or_si128(a, b)
    }
    }

    sse2_baseline! {
    fn bitxor(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_xor_si128(a, b)
    }
    }

    // ====== Shifts ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_const<const N: i32>(self, a: __m128i) -> __m128i {
        _mm_slli_epi32::<N>(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_const<const N: i32>(self, a: __m128i) -> __m128i {
        _mm_srli_epi32::<N>(a)
    }

    // ====== Uniform variable shifts ======
    // PSLLD/PSRLD give 0 once the count reaches 32 — the portable
    // contract is the hardware behaviour. _mm_cvtsi32_si128
    // zero-extends, so u32::MAX is just a large out-of-range count.

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_uniform(self, a: __m128i, count: u32) -> __m128i {
        _mm_sll_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_uniform(self, a: __m128i, count: u32) -> __m128i {
        _mm_srl_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    // ====== Boolean ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn all_true(self, a: __m128i) -> bool {
        _mm_movemask_ps(_mm_castsi128_ps(a)) == 0xF
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn any_true(self, a: __m128i) -> bool {
        _mm_movemask_ps(_mm_castsi128_ps(a)) != 0
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitmask(self, a: __m128i) -> u32 {
        _mm_movemask_ps(_mm_castsi128_ps(a)) as u32
    }
}

#[cfg(target_arch = "x86_64")]
impl U32x8Backend for archmage::X64V3Token {
    type Repr = __m256i;

    // ====== Construction ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn splat(self, v: u32) -> __m256i {
        _mm256_set1_epi32(v as i32)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn zero(self) -> __m256i {
        _mm256_setzero_si256()
    }

    #[inline(always)]
    fn load(self, data: &[u32; 8]) -> __m256i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u32; 8]) -> __m256i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m256i, out: &mut [u32; 8]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m256i) -> [u32; 8] {
        crate::simd_storage::cast(repr)
    }

    // ====== Arithmetic ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn add(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_add_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn sub(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_sub_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn mul(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_mullo_epi32(a, b)
    }

    // ====== Math ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn min(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_min_epu32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn max(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_max_epu32(a, b)
    }

    // ====== Comparisons ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_eq(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpeq_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ne(self, a: __m256i, b: __m256i) -> __m256i {
        let eq = _mm256_cmpeq_epi32(a, b);
        _mm256_andnot_si256(eq, _mm256_set1_epi32(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_gt(self, a: __m256i, b: __m256i) -> __m256i {
        // Unsigned comparison via bias trick: XOR both with 0x80000000
        // to convert to signed range, then use signed cmpgt.
        {
            let bias = _mm256_set1_epi32(i32::MIN);
            let sa = _mm256_xor_si256(a, bias);
            let sb = _mm256_xor_si256(b, bias);
            _mm256_cmpgt_epi32(sa, sb)
        }
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_lt(self, a: __m256i, b: __m256i) -> __m256i {
        <Self as U32x8Backend>::simd_gt(_self, b, a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_le(self, a: __m256i, b: __m256i) -> __m256i {
        let gt = <Self as U32x8Backend>::simd_gt(_self, a, b);
        _mm256_andnot_si256(gt, _mm256_set1_epi32(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ge(self, a: __m256i, b: __m256i) -> __m256i {
        let lt = <Self as U32x8Backend>::simd_gt(_self, b, a);
        _mm256_andnot_si256(lt, _mm256_set1_epi32(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn blend(self, mask: __m256i, if_true: __m256i, if_false: __m256i) -> __m256i {
        _mm256_blendv_epi8(if_false, if_true, mask)
    }

    // ====== Reductions ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_add(self, a: __m256i) -> u32 {
        {
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

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn not(self, a: __m256i) -> __m256i {
        _mm256_andnot_si256(a, _mm256_set1_epi32(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitand(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_and_si256(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitor(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_or_si256(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitxor(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_xor_si256(a, b)
    }

    // ====== Shifts ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_const<const N: i32>(self, a: __m256i) -> __m256i {
        _mm256_slli_epi32::<N>(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_const<const N: i32>(self, a: __m256i) -> __m256i {
        _mm256_srli_epi32::<N>(a)
    }

    // ====== Uniform variable shifts ======
    // PSLLD/PSRLD give 0 once the count reaches 32 — the portable
    // contract is the hardware behaviour. _mm_cvtsi32_si128
    // zero-extends, so u32::MAX is just a large out-of-range count.

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_uniform(self, a: __m256i, count: u32) -> __m256i {
        _mm256_sll_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_uniform(self, a: __m256i, count: u32) -> __m256i {
        _mm256_srl_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    // ====== Boolean ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn all_true(self, a: __m256i) -> bool {
        _mm256_movemask_ps(_mm256_castsi256_ps(a)) == 0xFF
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn any_true(self, a: __m256i) -> bool {
        _mm256_movemask_ps(_mm256_castsi256_ps(a)) != 0
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitmask(self, a: __m256i) -> u32 {
        _mm256_movemask_ps(_mm256_castsi256_ps(a)) as u32
    }
}

#[cfg(target_arch = "x86_64")]
impl I64x2Backend for archmage::X64V3Token {
    type Repr = __m128i;

    // ====== Construction ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn splat(self, v: i64) -> __m128i {
        _mm_set1_epi64x(v)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn zero(self) -> __m128i {
        _mm_setzero_si128()
    }

    #[inline(always)]
    fn load(self, data: &[i64; 2]) -> __m128i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i64; 2]) -> __m128i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m128i, out: &mut [i64; 2]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m128i) -> [i64; 2] {
        crate::simd_storage::cast(repr)
    }

    // ====== Arithmetic ======

    sse2_baseline! {
    fn add(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_add_epi64(a, b)
    }
    }

    sse2_baseline! {
    fn sub(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_sub_epi64(a, b)
    }
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn neg(self, a: __m128i) -> __m128i {
        _mm_sub_epi64(_mm_setzero_si128(), a)
    }

    // ====== Math ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn min(self, a: __m128i, b: __m128i) -> __m128i {
        // Polyfill: compare+select (no native i64 min on AVX2)
        {
            let mask = _mm_cmpgt_epi64(a, b);
            _mm_blendv_epi8(a, b, mask)
        }
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn max(self, a: __m128i, b: __m128i) -> __m128i {
        // Polyfill: compare+select (no native i64 max on AVX2)
        {
            let mask = _mm_cmpgt_epi64(a, b);
            _mm_blendv_epi8(b, a, mask)
        }
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn abs(self, a: __m128i) -> __m128i {
        // Polyfill: (a ^ sign) - sign (two's complement trick)
        {
            let zero = _mm_setzero_si128();
            let sign = _mm_cmpgt_epi64(zero, a);
            _mm_sub_epi64(_mm_xor_si128(a, sign), sign)
        }
    }

    // ====== Comparisons ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_eq(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_cmpeq_epi64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ne(self, a: __m128i, b: __m128i) -> __m128i {
        let eq = _mm_cmpeq_epi64(a, b);
        _mm_xor_si128(eq, _mm_set1_epi64x(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_lt(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_cmpgt_epi64(b, a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_le(self, a: __m128i, b: __m128i) -> __m128i {
        let gt = _mm_cmpgt_epi64(a, b);
        _mm_xor_si128(gt, _mm_set1_epi64x(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_gt(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_cmpgt_epi64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ge(self, a: __m128i, b: __m128i) -> __m128i {
        let lt = _mm_cmpgt_epi64(b, a);
        _mm_xor_si128(lt, _mm_set1_epi64x(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn blend(self, mask: __m128i, if_true: __m128i, if_false: __m128i) -> __m128i {
        _mm_blendv_epi8(if_false, if_true, mask)
    }

    // ====== Reductions ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_add(self, a: __m128i) -> i64 {
        {
            let hi = _mm_unpackhi_epi64(a, a);
            let sum = _mm_add_epi64(a, hi);
            // Extract low 64-bit lane
            crate::simd_storage::cast::<__m128i, [i64; 2]>(sum)[0]
        }
    }

    // ====== Bitwise ======

    sse2_baseline! {
    fn not(self, a: __m128i) -> __m128i {
        _mm_xor_si128(a, _mm_set1_epi64x(-1))
    }
    }

    sse2_baseline! {
    fn bitand(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_and_si128(a, b)
    }
    }

    sse2_baseline! {
    fn bitor(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_or_si128(a, b)
    }
    }

    sse2_baseline! {
    fn bitxor(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_xor_si128(a, b)
    }
    }

    // ====== Shifts ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_const<const N: i32>(self, a: __m128i) -> __m128i {
        _mm_slli_epi64::<N>(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m128i) -> __m128i {
        // Polyfill: no native _srai_epi64 on AVX2.
        // Use logical shift + sign extension.
        {
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

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_const<const N: i32>(self, a: __m128i) -> __m128i {
        _mm_srli_epi64::<N>(a)
    }

    // ====== Boolean ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn all_true(self, a: __m128i) -> bool {
        _mm_movemask_pd(_mm_castsi128_pd(a)) == 0x3
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn any_true(self, a: __m128i) -> bool {
        _mm_movemask_pd(_mm_castsi128_pd(a)) != 0
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitmask(self, a: __m128i) -> u32 {
        _mm_movemask_pd(_mm_castsi128_pd(a)) as u32
    }
}

#[cfg(target_arch = "x86_64")]
impl I64x4Backend for archmage::X64V3Token {
    type Repr = __m256i;

    // ====== Construction ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn splat(self, v: i64) -> __m256i {
        _mm256_set1_epi64x(v)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn zero(self) -> __m256i {
        _mm256_setzero_si256()
    }

    #[inline(always)]
    fn load(self, data: &[i64; 4]) -> __m256i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i64; 4]) -> __m256i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m256i, out: &mut [i64; 4]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m256i) -> [i64; 4] {
        crate::simd_storage::cast(repr)
    }

    // ====== Arithmetic ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn add(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_add_epi64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn sub(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_sub_epi64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn neg(self, a: __m256i) -> __m256i {
        _mm256_sub_epi64(_mm256_setzero_si256(), a)
    }

    // ====== Math ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn min(self, a: __m256i, b: __m256i) -> __m256i {
        // Polyfill: compare+select (no native i64 min on AVX2)
        {
            let mask = _mm256_cmpgt_epi64(a, b);
            _mm256_blendv_epi8(a, b, mask)
        }
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn max(self, a: __m256i, b: __m256i) -> __m256i {
        // Polyfill: compare+select (no native i64 max on AVX2)
        {
            let mask = _mm256_cmpgt_epi64(a, b);
            _mm256_blendv_epi8(b, a, mask)
        }
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn abs(self, a: __m256i) -> __m256i {
        // Polyfill: (a ^ sign) - sign (two's complement trick)
        {
            let zero = _mm256_setzero_si256();
            let sign = _mm256_cmpgt_epi64(zero, a);
            _mm256_sub_epi64(_mm256_xor_si256(a, sign), sign)
        }
    }

    // ====== Comparisons ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_eq(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpeq_epi64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ne(self, a: __m256i, b: __m256i) -> __m256i {
        let eq = _mm256_cmpeq_epi64(a, b);
        _mm256_xor_si256(eq, _mm256_set1_epi64x(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_lt(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpgt_epi64(b, a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_le(self, a: __m256i, b: __m256i) -> __m256i {
        let gt = _mm256_cmpgt_epi64(a, b);
        _mm256_xor_si256(gt, _mm256_set1_epi64x(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_gt(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpgt_epi64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ge(self, a: __m256i, b: __m256i) -> __m256i {
        let lt = _mm256_cmpgt_epi64(b, a);
        _mm256_xor_si256(lt, _mm256_set1_epi64x(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn blend(self, mask: __m256i, if_true: __m256i, if_false: __m256i) -> __m256i {
        _mm256_blendv_epi8(if_false, if_true, mask)
    }

    // ====== Reductions ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_add(self, a: __m256i) -> i64 {
        {
            let lo = _mm256_castsi256_si128(a);
            let hi = _mm256_extracti128_si256::<1>(a);
            let sum = _mm_add_epi64(lo, hi);
            let hi2 = _mm_unpackhi_epi64(sum, sum);
            let sum2 = _mm_add_epi64(sum, hi2);
            crate::simd_storage::cast::<__m128i, [i64; 2]>(sum2)[0]
        }
    }

    // ====== Bitwise ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn not(self, a: __m256i) -> __m256i {
        _mm256_xor_si256(a, _mm256_set1_epi64x(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitand(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_and_si256(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitor(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_or_si256(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitxor(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_xor_si256(a, b)
    }

    // ====== Shifts ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_const<const N: i32>(self, a: __m256i) -> __m256i {
        _mm256_slli_epi64::<N>(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m256i) -> __m256i {
        // Polyfill: no native _srai_epi64 on AVX2.
        // Use logical shift + sign extension.
        {
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

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_const<const N: i32>(self, a: __m256i) -> __m256i {
        _mm256_srli_epi64::<N>(a)
    }

    // ====== Boolean ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn all_true(self, a: __m256i) -> bool {
        _mm256_movemask_pd(_mm256_castsi256_pd(a)) == 0xF
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn any_true(self, a: __m256i) -> bool {
        _mm256_movemask_pd(_mm256_castsi256_pd(a)) != 0
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitmask(self, a: __m256i) -> u32 {
        _mm256_movemask_pd(_mm256_castsi256_pd(a)) as u32
    }
}

#[cfg(target_arch = "x86_64")]
impl I8x16Backend for archmage::X64V3Token {
    type Repr = __m128i;

    // ====== Construction ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn splat(self, v: i8) -> __m128i {
        _mm_set1_epi8(v)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn zero(self) -> __m128i {
        _mm_setzero_si128()
    }

    #[inline(always)]
    fn load(self, data: &[i8; 16]) -> __m128i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i8; 16]) -> __m128i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m128i, out: &mut [i8; 16]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m128i) -> [i8; 16] {
        crate::simd_storage::cast(repr)
    }

    // ====== Arithmetic ======

    sse2_baseline! {
    fn add(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_add_epi8(a, b)
    }
    }

    sse2_baseline! {
    fn sub(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_sub_epi8(a, b)
    }
    }
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn neg(self, a: __m128i) -> __m128i {
        _mm_sub_epi8(_mm_setzero_si128(), a)
    }

    // ====== Math ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn min(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_min_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn max(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_max_epi8(a, b)
    }
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn abs(self, a: __m128i) -> __m128i {
        _mm_abs_epi8(a)
    }

    // ====== Comparisons ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_eq(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_cmpeq_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ne(self, a: __m128i, b: __m128i) -> __m128i {
        let eq = _mm_cmpeq_epi8(a, b);
        _mm_andnot_si128(eq, _mm_set1_epi8(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_lt(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_cmpgt_epi8(b, a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_le(self, a: __m128i, b: __m128i) -> __m128i {
        let gt = _mm_cmpgt_epi8(a, b);
        _mm_andnot_si128(gt, _mm_set1_epi8(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_gt(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_cmpgt_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ge(self, a: __m128i, b: __m128i) -> __m128i {
        let lt = _mm_cmpgt_epi8(b, a);
        _mm_andnot_si128(lt, _mm_set1_epi8(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn blend(self, mask: __m128i, if_true: __m128i, if_false: __m128i) -> __m128i {
        _mm_blendv_epi8(if_false, if_true, mask)
    }

    // ====== Reductions ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_add(self, a: __m128i) -> i8 {
        let arr = <Self as I8x16Backend>::to_array(_self, a);
        arr.iter().copied().fold(0i8, i8::wrapping_add)
    }

    // ====== Bitwise ======

    sse2_baseline! {
    fn not(self, a: __m128i) -> __m128i {
        _mm_andnot_si128(a, _mm_set1_epi8(-1))
    }
    }

    sse2_baseline! {
    fn bitand(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_and_si128(a, b)
    }
    }

    sse2_baseline! {
    fn bitor(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_or_si128(a, b)
    }
    }

    sse2_baseline! {
    fn bitxor(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_xor_si128(a, b)
    }
    }

    // ====== Shifts (polyfill via 16-bit) ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_const<const N: i32>(self, a: __m128i) -> __m128i {
        let shifted = _mm_slli_epi16::<N>(a);
        let mask = _mm_set1_epi8((0xFFu8.wrapping_shl(N as u32)) as i8);
        _mm_and_si128(shifted, mask)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_const<const N: i32>(self, a: __m128i) -> __m128i {
        let shifted = _mm_srli_epi16::<N>(a);
        let mask = _mm_set1_epi8((0xFFu8.wrapping_shr(N as u32)) as i8);
        _mm_and_si128(shifted, mask)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m128i) -> __m128i {
        let shifted = _mm_srli_epi16::<N>(a);
        let byte_mask = _mm_set1_epi8((0xFFu8.wrapping_shr(N as u32)) as i8);
        let logical = _mm_and_si128(shifted, byte_mask);
        let zero = _mm_setzero_si128();
        let sign = _mm_cmpgt_epi8(zero, a);
        // High-N-bits fill mask via u16 shift: 0x00 at N == 0 (identity
        // shift needs no sign fill). A u8 `0xFF << (8 - N)` would wrap
        // the shift amount at N == 0 and corrupt negative lanes.
        let fill = _mm_set1_epi8(((0xFF00u16 >> N) & 0xFF) as i8);
        _mm_or_si128(logical, _mm_and_si128(sign, fill))
    }

    // ====== Uniform variable shifts (8-bit: polyfill via 16-bit) ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_uniform(self, a: __m128i, count: u32) -> __m128i {
        let shifted = _mm_sll_epi16(a, _mm_cvtsi32_si128(count as i32));
        // `checked_shl` yields None (-> mask 0) once count >= 8,
        // which is the all-zero result the contract requires.
        let mask = _mm_set1_epi8(0xFFu8.checked_shl(count).unwrap_or(0) as i8);
        _mm_and_si128(shifted, mask)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_uniform(self, a: __m128i, count: u32) -> __m128i {
        let shifted = _mm_srl_epi16(a, _mm_cvtsi32_si128(count as i32));
        let mask = _mm_set1_epi8(0xFFu8.checked_shr(count).unwrap_or(0) as i8);
        _mm_and_si128(shifted, mask)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_arithmetic_uniform(self, a: __m128i, count: u32) -> __m128i {
        let shifted = _mm_srl_epi16(a, _mm_cvtsi32_si128(count as i32));
        let byte_mask = _mm_set1_epi8(0xFFu8.checked_shr(count).unwrap_or(0) as i8);
        let logical = _mm_and_si128(shifted, byte_mask);
        let zero = _mm_setzero_si128();
        let sign = _mm_cmpgt_epi8(zero, a);
        // High-`count`-bits fill mask. `count.min(8)` saturates the
        // fill to the whole byte, which is the sign fill the
        // contract requires for out-of-range counts; a plain
        // `>> count` would be a u16 overflow at count >= 16.
        let fill = _mm_set1_epi8(((0xFF00u16 >> count.min(8)) & 0xFF) as u8 as i8);
        _mm_or_si128(logical, _mm_and_si128(sign, fill))
    }

    // ====== Saturating arithmetic ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn saturating_add(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_adds_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn saturating_sub(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_subs_epi8(a, b)
    }

    // ====== Boolean ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn all_true(self, a: __m128i) -> bool {
        _mm_movemask_epi8(a) == 0xFFFF_u32 as i32
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn any_true(self, a: __m128i) -> bool {
        _mm_movemask_epi8(a) != 0
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitmask(self, a: __m128i) -> u32 {
        _mm_movemask_epi8(a) as u32
    }
}

#[cfg(target_arch = "x86_64")]
impl I8x32Backend for archmage::X64V3Token {
    type Repr = __m256i;

    // ====== Construction ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn splat(self, v: i8) -> __m256i {
        _mm256_set1_epi8(v)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn zero(self) -> __m256i {
        _mm256_setzero_si256()
    }

    #[inline(always)]
    fn load(self, data: &[i8; 32]) -> __m256i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i8; 32]) -> __m256i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m256i, out: &mut [i8; 32]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m256i) -> [i8; 32] {
        crate::simd_storage::cast(repr)
    }

    // ====== Arithmetic ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn add(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_add_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn sub(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_sub_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn neg(self, a: __m256i) -> __m256i {
        _mm256_sub_epi8(_mm256_setzero_si256(), a)
    }

    // ====== Math ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn min(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_min_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn max(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_max_epi8(a, b)
    }
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn abs(self, a: __m256i) -> __m256i {
        _mm256_abs_epi8(a)
    }

    // ====== Comparisons ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_eq(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpeq_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ne(self, a: __m256i, b: __m256i) -> __m256i {
        let eq = _mm256_cmpeq_epi8(a, b);
        _mm256_andnot_si256(eq, _mm256_set1_epi8(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_lt(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpgt_epi8(b, a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_le(self, a: __m256i, b: __m256i) -> __m256i {
        let gt = _mm256_cmpgt_epi8(a, b);
        _mm256_andnot_si256(gt, _mm256_set1_epi8(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_gt(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpgt_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ge(self, a: __m256i, b: __m256i) -> __m256i {
        let lt = _mm256_cmpgt_epi8(b, a);
        _mm256_andnot_si256(lt, _mm256_set1_epi8(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn blend(self, mask: __m256i, if_true: __m256i, if_false: __m256i) -> __m256i {
        _mm256_blendv_epi8(if_false, if_true, mask)
    }

    // ====== Reductions ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_add(self, a: __m256i) -> i8 {
        let arr = <Self as I8x32Backend>::to_array(_self, a);
        arr.iter().copied().fold(0i8, i8::wrapping_add)
    }

    // ====== Bitwise ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn not(self, a: __m256i) -> __m256i {
        _mm256_andnot_si256(a, _mm256_set1_epi8(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitand(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_and_si256(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitor(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_or_si256(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitxor(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_xor_si256(a, b)
    }

    // ====== Shifts (polyfill via 16-bit) ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_const<const N: i32>(self, a: __m256i) -> __m256i {
        let shifted = _mm256_slli_epi16::<N>(a);
        let mask = _mm256_set1_epi8((0xFFu8.wrapping_shl(N as u32)) as i8);
        _mm256_and_si256(shifted, mask)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_const<const N: i32>(self, a: __m256i) -> __m256i {
        let shifted = _mm256_srli_epi16::<N>(a);
        let mask = _mm256_set1_epi8((0xFFu8.wrapping_shr(N as u32)) as i8);
        _mm256_and_si256(shifted, mask)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m256i) -> __m256i {
        let shifted = _mm256_srli_epi16::<N>(a);
        let byte_mask = _mm256_set1_epi8((0xFFu8.wrapping_shr(N as u32)) as i8);
        let logical = _mm256_and_si256(shifted, byte_mask);
        let zero = _mm256_setzero_si256();
        let sign = _mm256_cmpgt_epi8(zero, a);
        // High-N-bits fill mask via u16 shift: 0x00 at N == 0 (identity
        // shift needs no sign fill). A u8 `0xFF << (8 - N)` would wrap
        // the shift amount at N == 0 and corrupt negative lanes.
        let fill = _mm256_set1_epi8(((0xFF00u16 >> N) & 0xFF) as i8);
        _mm256_or_si256(logical, _mm256_and_si256(sign, fill))
    }

    // ====== Uniform variable shifts (8-bit: polyfill via 16-bit) ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_uniform(self, a: __m256i, count: u32) -> __m256i {
        let shifted = _mm256_sll_epi16(a, _mm_cvtsi32_si128(count as i32));
        // `checked_shl` yields None (-> mask 0) once count >= 8,
        // which is the all-zero result the contract requires.
        let mask = _mm256_set1_epi8(0xFFu8.checked_shl(count).unwrap_or(0) as i8);
        _mm256_and_si256(shifted, mask)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_uniform(self, a: __m256i, count: u32) -> __m256i {
        let shifted = _mm256_srl_epi16(a, _mm_cvtsi32_si128(count as i32));
        let mask = _mm256_set1_epi8(0xFFu8.checked_shr(count).unwrap_or(0) as i8);
        _mm256_and_si256(shifted, mask)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_arithmetic_uniform(self, a: __m256i, count: u32) -> __m256i {
        let shifted = _mm256_srl_epi16(a, _mm_cvtsi32_si128(count as i32));
        let byte_mask = _mm256_set1_epi8(0xFFu8.checked_shr(count).unwrap_or(0) as i8);
        let logical = _mm256_and_si256(shifted, byte_mask);
        let zero = _mm256_setzero_si256();
        let sign = _mm256_cmpgt_epi8(zero, a);
        // High-`count`-bits fill mask. `count.min(8)` saturates the
        // fill to the whole byte, which is the sign fill the
        // contract requires for out-of-range counts; a plain
        // `>> count` would be a u16 overflow at count >= 16.
        let fill = _mm256_set1_epi8(((0xFF00u16 >> count.min(8)) & 0xFF) as u8 as i8);
        _mm256_or_si256(logical, _mm256_and_si256(sign, fill))
    }

    // ====== Saturating arithmetic ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn saturating_add(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_adds_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn saturating_sub(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_subs_epi8(a, b)
    }

    // ====== Boolean ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn all_true(self, a: __m256i) -> bool {
        _mm256_movemask_epi8(a) == -1_i32
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn any_true(self, a: __m256i) -> bool {
        _mm256_movemask_epi8(a) != 0
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitmask(self, a: __m256i) -> u32 {
        _mm256_movemask_epi8(a) as u32
    }
}

#[cfg(target_arch = "x86_64")]
impl U8x16Backend for archmage::X64V3Token {
    type Repr = __m128i;

    // ====== Construction ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn splat(self, v: u8) -> __m128i {
        _mm_set1_epi8(v as i8)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn zero(self) -> __m128i {
        _mm_setzero_si128()
    }

    #[inline(always)]
    fn load(self, data: &[u8; 16]) -> __m128i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u8; 16]) -> __m128i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m128i, out: &mut [u8; 16]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m128i) -> [u8; 16] {
        crate::simd_storage::cast(repr)
    }

    // ====== Arithmetic ======

    sse2_baseline! {
    fn add(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_add_epi8(a, b)
    }
    }

    sse2_baseline! {
    fn sub(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_sub_epi8(a, b)
    }
    }

    // ====== Math ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn min(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_min_epu8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn max(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_max_epu8(a, b)
    }

    // ====== Comparisons ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_eq(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_cmpeq_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ne(self, a: __m128i, b: __m128i) -> __m128i {
        let eq = _mm_cmpeq_epi8(a, b);
        _mm_andnot_si128(eq, _mm_set1_epi8(-1_i8))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_gt(self, a: __m128i, b: __m128i) -> __m128i {
        let bias = _mm_set1_epi8(i8::MIN);
        let sa = _mm_xor_si128(a, bias);
        let sb = _mm_xor_si128(b, bias);
        _mm_cmpgt_epi8(sa, sb)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_lt(self, a: __m128i, b: __m128i) -> __m128i {
        <Self as U8x16Backend>::simd_gt(_self, b, a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_le(self, a: __m128i, b: __m128i) -> __m128i {
        let gt = <Self as U8x16Backend>::simd_gt(_self, a, b);
        _mm_andnot_si128(gt, _mm_set1_epi8(-1_i8))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ge(self, a: __m128i, b: __m128i) -> __m128i {
        let lt = <Self as U8x16Backend>::simd_gt(_self, b, a);
        _mm_andnot_si128(lt, _mm_set1_epi8(-1_i8))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn blend(self, mask: __m128i, if_true: __m128i, if_false: __m128i) -> __m128i {
        _mm_blendv_epi8(if_false, if_true, mask)
    }

    // ====== Reductions ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_add(self, a: __m128i) -> u8 {
        let arr = <Self as U8x16Backend>::to_array(_self, a);
        arr.iter().copied().fold(0u8, u8::wrapping_add)
    }

    // ====== Bitwise ======

    sse2_baseline! {
    fn not(self, a: __m128i) -> __m128i {
        _mm_andnot_si128(a, _mm_set1_epi8(-1_i8))
    }
    }

    sse2_baseline! {
    fn bitand(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_and_si128(a, b)
    }
    }

    sse2_baseline! {
    fn bitor(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_or_si128(a, b)
    }
    }

    sse2_baseline! {
    fn bitxor(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_xor_si128(a, b)
    }
    }

    // ====== Shifts (polyfill via 16-bit) ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_const<const N: i32>(self, a: __m128i) -> __m128i {
        let shifted = _mm_slli_epi16::<N>(a);
        let mask = _mm_set1_epi8((0xFFu8.wrapping_shl(N as u32)) as i8);
        _mm_and_si128(shifted, mask)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_const<const N: i32>(self, a: __m128i) -> __m128i {
        let shifted = _mm_srli_epi16::<N>(a);
        let mask = _mm_set1_epi8((0xFFu8.wrapping_shr(N as u32)) as i8);
        _mm_and_si128(shifted, mask)
    }

    // ====== Uniform variable shifts (8-bit: polyfill via 16-bit) ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_uniform(self, a: __m128i, count: u32) -> __m128i {
        let shifted = _mm_sll_epi16(a, _mm_cvtsi32_si128(count as i32));
        // `checked_shl` yields None (-> mask 0) once count >= 8,
        // which is the all-zero result the contract requires.
        let mask = _mm_set1_epi8(0xFFu8.checked_shl(count).unwrap_or(0) as i8);
        _mm_and_si128(shifted, mask)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_uniform(self, a: __m128i, count: u32) -> __m128i {
        let shifted = _mm_srl_epi16(a, _mm_cvtsi32_si128(count as i32));
        let mask = _mm_set1_epi8(0xFFu8.checked_shr(count).unwrap_or(0) as i8);
        _mm_and_si128(shifted, mask)
    }

    // ====== Saturating arithmetic ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn saturating_add(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_adds_epu8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn saturating_sub(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_subs_epu8(a, b)
    }

    // ====== Boolean ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn all_true(self, a: __m128i) -> bool {
        _mm_movemask_epi8(a) == 0xFFFF_u32 as i32
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn any_true(self, a: __m128i) -> bool {
        _mm_movemask_epi8(a) != 0
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitmask(self, a: __m128i) -> u32 {
        _mm_movemask_epi8(a) as u32
    }
}

#[cfg(target_arch = "x86_64")]
impl U8x32Backend for archmage::X64V3Token {
    type Repr = __m256i;

    // ====== Construction ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn splat(self, v: u8) -> __m256i {
        _mm256_set1_epi8(v as i8)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn zero(self) -> __m256i {
        _mm256_setzero_si256()
    }

    #[inline(always)]
    fn load(self, data: &[u8; 32]) -> __m256i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u8; 32]) -> __m256i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m256i, out: &mut [u8; 32]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m256i) -> [u8; 32] {
        crate::simd_storage::cast(repr)
    }

    // ====== Arithmetic ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn add(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_add_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn sub(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_sub_epi8(a, b)
    }

    // ====== Math ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn min(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_min_epu8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn max(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_max_epu8(a, b)
    }

    // ====== Comparisons ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_eq(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpeq_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ne(self, a: __m256i, b: __m256i) -> __m256i {
        let eq = _mm256_cmpeq_epi8(a, b);
        _mm256_andnot_si256(eq, _mm256_set1_epi8(-1_i8))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_gt(self, a: __m256i, b: __m256i) -> __m256i {
        let bias = _mm256_set1_epi8(i8::MIN);
        let sa = _mm256_xor_si256(a, bias);
        let sb = _mm256_xor_si256(b, bias);
        _mm256_cmpgt_epi8(sa, sb)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_lt(self, a: __m256i, b: __m256i) -> __m256i {
        <Self as U8x32Backend>::simd_gt(_self, b, a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_le(self, a: __m256i, b: __m256i) -> __m256i {
        let gt = <Self as U8x32Backend>::simd_gt(_self, a, b);
        _mm256_andnot_si256(gt, _mm256_set1_epi8(-1_i8))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ge(self, a: __m256i, b: __m256i) -> __m256i {
        let lt = <Self as U8x32Backend>::simd_gt(_self, b, a);
        _mm256_andnot_si256(lt, _mm256_set1_epi8(-1_i8))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn blend(self, mask: __m256i, if_true: __m256i, if_false: __m256i) -> __m256i {
        _mm256_blendv_epi8(if_false, if_true, mask)
    }

    // ====== Reductions ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_add(self, a: __m256i) -> u8 {
        let arr = <Self as U8x32Backend>::to_array(_self, a);
        arr.iter().copied().fold(0u8, u8::wrapping_add)
    }

    // ====== Bitwise ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn not(self, a: __m256i) -> __m256i {
        _mm256_andnot_si256(a, _mm256_set1_epi8(-1_i8))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitand(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_and_si256(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitor(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_or_si256(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitxor(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_xor_si256(a, b)
    }

    // ====== Shifts (polyfill via 16-bit) ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_const<const N: i32>(self, a: __m256i) -> __m256i {
        let shifted = _mm256_slli_epi16::<N>(a);
        let mask = _mm256_set1_epi8((0xFFu8.wrapping_shl(N as u32)) as i8);
        _mm256_and_si256(shifted, mask)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_const<const N: i32>(self, a: __m256i) -> __m256i {
        let shifted = _mm256_srli_epi16::<N>(a);
        let mask = _mm256_set1_epi8((0xFFu8.wrapping_shr(N as u32)) as i8);
        _mm256_and_si256(shifted, mask)
    }

    // ====== Uniform variable shifts (8-bit: polyfill via 16-bit) ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_uniform(self, a: __m256i, count: u32) -> __m256i {
        let shifted = _mm256_sll_epi16(a, _mm_cvtsi32_si128(count as i32));
        // `checked_shl` yields None (-> mask 0) once count >= 8,
        // which is the all-zero result the contract requires.
        let mask = _mm256_set1_epi8(0xFFu8.checked_shl(count).unwrap_or(0) as i8);
        _mm256_and_si256(shifted, mask)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_uniform(self, a: __m256i, count: u32) -> __m256i {
        let shifted = _mm256_srl_epi16(a, _mm_cvtsi32_si128(count as i32));
        let mask = _mm256_set1_epi8(0xFFu8.checked_shr(count).unwrap_or(0) as i8);
        _mm256_and_si256(shifted, mask)
    }

    // ====== Saturating arithmetic ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn saturating_add(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_adds_epu8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn saturating_sub(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_subs_epu8(a, b)
    }

    // ====== Boolean ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn all_true(self, a: __m256i) -> bool {
        _mm256_movemask_epi8(a) == -1_i32
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn any_true(self, a: __m256i) -> bool {
        _mm256_movemask_epi8(a) != 0
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitmask(self, a: __m256i) -> u32 {
        _mm256_movemask_epi8(a) as u32
    }
}

#[cfg(target_arch = "x86_64")]
impl I16x8Backend for archmage::X64V3Token {
    type Repr = __m128i;

    // ====== Construction ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn splat(self, v: i16) -> __m128i {
        _mm_set1_epi16(v)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn zero(self) -> __m128i {
        _mm_setzero_si128()
    }

    #[inline(always)]
    fn load(self, data: &[i16; 8]) -> __m128i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i16; 8]) -> __m128i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m128i, out: &mut [i16; 8]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m128i) -> [i16; 8] {
        crate::simd_storage::cast(repr)
    }

    // ====== Arithmetic ======

    sse2_baseline! {
    fn add(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_add_epi16(a, b)
    }
    }

    sse2_baseline! {
    fn sub(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_sub_epi16(a, b)
    }
    }
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn mul(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_mullo_epi16(a, b)
    }
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn neg(self, a: __m128i) -> __m128i {
        _mm_sub_epi16(_mm_setzero_si128(), a)
    }

    // ====== Math ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn min(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_min_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn max(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_max_epi16(a, b)
    }
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn abs(self, a: __m128i) -> __m128i {
        _mm_abs_epi16(a)
    }

    // ====== Comparisons ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_eq(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_cmpeq_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ne(self, a: __m128i, b: __m128i) -> __m128i {
        let eq = _mm_cmpeq_epi16(a, b);
        _mm_andnot_si128(eq, _mm_set1_epi16(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_lt(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_cmpgt_epi16(b, a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_le(self, a: __m128i, b: __m128i) -> __m128i {
        let gt = _mm_cmpgt_epi16(a, b);
        _mm_andnot_si128(gt, _mm_set1_epi16(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_gt(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_cmpgt_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ge(self, a: __m128i, b: __m128i) -> __m128i {
        let lt = _mm_cmpgt_epi16(b, a);
        _mm_andnot_si128(lt, _mm_set1_epi16(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn blend(self, mask: __m128i, if_true: __m128i, if_false: __m128i) -> __m128i {
        _mm_blendv_epi8(if_false, if_true, mask)
    }

    // ====== Reductions ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_add(self, a: __m128i) -> i16 {
        let arr = <Self as I16x8Backend>::to_array(_self, a);
        arr.iter().copied().fold(0i16, i16::wrapping_add)
    }

    // ====== Bitwise ======

    sse2_baseline! {
    fn not(self, a: __m128i) -> __m128i {
        _mm_andnot_si128(a, _mm_set1_epi16(-1))
    }
    }

    sse2_baseline! {
    fn bitand(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_and_si128(a, b)
    }
    }

    sse2_baseline! {
    fn bitor(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_or_si128(a, b)
    }
    }

    sse2_baseline! {
    fn bitxor(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_xor_si128(a, b)
    }
    }

    // ====== Shifts ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_const<const N: i32>(self, a: __m128i) -> __m128i {
        _mm_slli_epi16::<N>(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_const<const N: i32>(self, a: __m128i) -> __m128i {
        _mm_srli_epi16::<N>(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m128i) -> __m128i {
        _mm_srai_epi16::<N>(a)
    }

    // ====== Uniform variable shifts ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_uniform(self, a: __m128i, count: u32) -> __m128i {
        _mm_sll_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_uniform(self, a: __m128i, count: u32) -> __m128i {
        _mm_srl_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_arithmetic_uniform(self, a: __m128i, count: u32) -> __m128i {
        _mm_sra_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    // ====== Saturating arithmetic ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn saturating_add(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_adds_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn saturating_sub(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_subs_epi16(a, b)
    }

    // ====== Boolean ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn all_true(self, a: __m128i) -> bool {
        _mm_movemask_epi8(a) == 0xFFFF_u32 as i32
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn any_true(self, a: __m128i) -> bool {
        _mm_movemask_epi8(a) != 0
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitmask(self, a: __m128i) -> u32 {
        let shifted = _mm_srai_epi16::<15>(a);
        let packed = _mm_packs_epi16(shifted, shifted);
        (_mm_movemask_epi8(packed) & 0xFF) as u32
    }
}

#[cfg(target_arch = "x86_64")]
impl I16x16Backend for archmage::X64V3Token {
    type Repr = __m256i;

    // ====== Construction ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn splat(self, v: i16) -> __m256i {
        _mm256_set1_epi16(v)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn zero(self) -> __m256i {
        _mm256_setzero_si256()
    }

    #[inline(always)]
    fn load(self, data: &[i16; 16]) -> __m256i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i16; 16]) -> __m256i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m256i, out: &mut [i16; 16]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m256i) -> [i16; 16] {
        crate::simd_storage::cast(repr)
    }

    // ====== Arithmetic ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn add(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_add_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn sub(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_sub_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn mul(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_mullo_epi16(a, b)
    }
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn neg(self, a: __m256i) -> __m256i {
        _mm256_sub_epi16(_mm256_setzero_si256(), a)
    }

    // ====== Math ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn min(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_min_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn max(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_max_epi16(a, b)
    }
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn abs(self, a: __m256i) -> __m256i {
        _mm256_abs_epi16(a)
    }

    // ====== Comparisons ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_eq(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpeq_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ne(self, a: __m256i, b: __m256i) -> __m256i {
        let eq = _mm256_cmpeq_epi16(a, b);
        _mm256_andnot_si256(eq, _mm256_set1_epi16(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_lt(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpgt_epi16(b, a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_le(self, a: __m256i, b: __m256i) -> __m256i {
        let gt = _mm256_cmpgt_epi16(a, b);
        _mm256_andnot_si256(gt, _mm256_set1_epi16(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_gt(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpgt_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ge(self, a: __m256i, b: __m256i) -> __m256i {
        let lt = _mm256_cmpgt_epi16(b, a);
        _mm256_andnot_si256(lt, _mm256_set1_epi16(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn blend(self, mask: __m256i, if_true: __m256i, if_false: __m256i) -> __m256i {
        _mm256_blendv_epi8(if_false, if_true, mask)
    }

    // ====== Reductions ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_add(self, a: __m256i) -> i16 {
        let arr = <Self as I16x16Backend>::to_array(_self, a);
        arr.iter().copied().fold(0i16, i16::wrapping_add)
    }

    // ====== Bitwise ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn not(self, a: __m256i) -> __m256i {
        _mm256_andnot_si256(a, _mm256_set1_epi16(-1))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitand(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_and_si256(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitor(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_or_si256(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitxor(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_xor_si256(a, b)
    }

    // ====== Shifts ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_const<const N: i32>(self, a: __m256i) -> __m256i {
        _mm256_slli_epi16::<N>(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_const<const N: i32>(self, a: __m256i) -> __m256i {
        _mm256_srli_epi16::<N>(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m256i) -> __m256i {
        _mm256_srai_epi16::<N>(a)
    }

    // ====== Uniform variable shifts ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_uniform(self, a: __m256i, count: u32) -> __m256i {
        _mm256_sll_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_uniform(self, a: __m256i, count: u32) -> __m256i {
        _mm256_srl_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_arithmetic_uniform(self, a: __m256i, count: u32) -> __m256i {
        _mm256_sra_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    // ====== Saturating arithmetic ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn saturating_add(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_adds_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn saturating_sub(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_subs_epi16(a, b)
    }

    // ====== Boolean ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn all_true(self, a: __m256i) -> bool {
        _mm256_movemask_epi8(a) == -1_i32
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn any_true(self, a: __m256i) -> bool {
        _mm256_movemask_epi8(a) != 0
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitmask(self, a: __m256i) -> u32 {
        let shifted = _mm256_srai_epi16::<15>(a);
        let lo = _mm256_castsi256_si128(shifted);
        let hi = _mm256_extracti128_si256::<1>(shifted);
        let packed = _mm_packs_epi16(lo, hi);
        (_mm_movemask_epi8(packed) as u32) & 0xFFFF
    }
}

#[cfg(target_arch = "x86_64")]
impl U16x8Backend for archmage::X64V3Token {
    type Repr = __m128i;

    // ====== Construction ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn splat(self, v: u16) -> __m128i {
        _mm_set1_epi16(v as i16)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn zero(self) -> __m128i {
        _mm_setzero_si128()
    }

    #[inline(always)]
    fn load(self, data: &[u16; 8]) -> __m128i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u16; 8]) -> __m128i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m128i, out: &mut [u16; 8]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m128i) -> [u16; 8] {
        crate::simd_storage::cast(repr)
    }

    // ====== Arithmetic ======

    sse2_baseline! {
    fn add(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_add_epi16(a, b)
    }
    }

    sse2_baseline! {
    fn sub(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_sub_epi16(a, b)
    }
    }
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn mul(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_mullo_epi16(a, b)
    }

    // ====== Math ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn min(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_min_epu16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn max(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_max_epu16(a, b)
    }

    // ====== Comparisons ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_eq(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_cmpeq_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ne(self, a: __m128i, b: __m128i) -> __m128i {
        let eq = _mm_cmpeq_epi16(a, b);
        _mm_andnot_si128(eq, _mm_set1_epi16(-1_i16))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_gt(self, a: __m128i, b: __m128i) -> __m128i {
        let bias = _mm_set1_epi16(i16::MIN);
        let sa = _mm_xor_si128(a, bias);
        let sb = _mm_xor_si128(b, bias);
        _mm_cmpgt_epi16(sa, sb)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_lt(self, a: __m128i, b: __m128i) -> __m128i {
        <Self as U16x8Backend>::simd_gt(_self, b, a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_le(self, a: __m128i, b: __m128i) -> __m128i {
        let gt = <Self as U16x8Backend>::simd_gt(_self, a, b);
        _mm_andnot_si128(gt, _mm_set1_epi16(-1_i16))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ge(self, a: __m128i, b: __m128i) -> __m128i {
        let lt = <Self as U16x8Backend>::simd_gt(_self, b, a);
        _mm_andnot_si128(lt, _mm_set1_epi16(-1_i16))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn blend(self, mask: __m128i, if_true: __m128i, if_false: __m128i) -> __m128i {
        _mm_blendv_epi8(if_false, if_true, mask)
    }

    // ====== Reductions ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_add(self, a: __m128i) -> u16 {
        let arr = <Self as U16x8Backend>::to_array(_self, a);
        arr.iter().copied().fold(0u16, u16::wrapping_add)
    }

    // ====== Bitwise ======

    sse2_baseline! {
    fn not(self, a: __m128i) -> __m128i {
        _mm_andnot_si128(a, _mm_set1_epi16(-1_i16))
    }
    }

    sse2_baseline! {
    fn bitand(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_and_si128(a, b)
    }
    }

    sse2_baseline! {
    fn bitor(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_or_si128(a, b)
    }
    }

    sse2_baseline! {
    fn bitxor(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_xor_si128(a, b)
    }
    }

    // ====== Shifts ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_const<const N: i32>(self, a: __m128i) -> __m128i {
        _mm_slli_epi16::<N>(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_const<const N: i32>(self, a: __m128i) -> __m128i {
        _mm_srli_epi16::<N>(a)
    }

    // ====== Uniform variable shifts ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_uniform(self, a: __m128i, count: u32) -> __m128i {
        _mm_sll_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_uniform(self, a: __m128i, count: u32) -> __m128i {
        _mm_srl_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    // ====== Saturating arithmetic ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn saturating_add(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_adds_epu16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn saturating_sub(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_subs_epu16(a, b)
    }

    // ====== Boolean ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn all_true(self, a: __m128i) -> bool {
        _mm_movemask_epi8(a) == 0xFFFF_u32 as i32
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn any_true(self, a: __m128i) -> bool {
        _mm_movemask_epi8(a) != 0
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitmask(self, a: __m128i) -> u32 {
        let shifted = _mm_srai_epi16::<15>(a);
        let packed = _mm_packs_epi16(shifted, shifted);
        (_mm_movemask_epi8(packed) & 0xFF) as u32
    }
}

#[cfg(target_arch = "x86_64")]
impl U16x16Backend for archmage::X64V3Token {
    type Repr = __m256i;

    // ====== Construction ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn splat(self, v: u16) -> __m256i {
        _mm256_set1_epi16(v as i16)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn zero(self) -> __m256i {
        _mm256_setzero_si256()
    }

    #[inline(always)]
    fn load(self, data: &[u16; 16]) -> __m256i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u16; 16]) -> __m256i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m256i, out: &mut [u16; 16]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m256i) -> [u16; 16] {
        crate::simd_storage::cast(repr)
    }

    // ====== Arithmetic ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn add(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_add_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn sub(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_sub_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn mul(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_mullo_epi16(a, b)
    }

    // ====== Math ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn min(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_min_epu16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn max(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_max_epu16(a, b)
    }

    // ====== Comparisons ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_eq(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpeq_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ne(self, a: __m256i, b: __m256i) -> __m256i {
        let eq = _mm256_cmpeq_epi16(a, b);
        _mm256_andnot_si256(eq, _mm256_set1_epi16(-1_i16))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_gt(self, a: __m256i, b: __m256i) -> __m256i {
        let bias = _mm256_set1_epi16(i16::MIN);
        let sa = _mm256_xor_si256(a, bias);
        let sb = _mm256_xor_si256(b, bias);
        _mm256_cmpgt_epi16(sa, sb)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_lt(self, a: __m256i, b: __m256i) -> __m256i {
        <Self as U16x16Backend>::simd_gt(_self, b, a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_le(self, a: __m256i, b: __m256i) -> __m256i {
        let gt = <Self as U16x16Backend>::simd_gt(_self, a, b);
        _mm256_andnot_si256(gt, _mm256_set1_epi16(-1_i16))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ge(self, a: __m256i, b: __m256i) -> __m256i {
        let lt = <Self as U16x16Backend>::simd_gt(_self, b, a);
        _mm256_andnot_si256(lt, _mm256_set1_epi16(-1_i16))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn blend(self, mask: __m256i, if_true: __m256i, if_false: __m256i) -> __m256i {
        _mm256_blendv_epi8(if_false, if_true, mask)
    }

    // ====== Reductions ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_add(self, a: __m256i) -> u16 {
        let arr = <Self as U16x16Backend>::to_array(_self, a);
        arr.iter().copied().fold(0u16, u16::wrapping_add)
    }

    // ====== Bitwise ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn not(self, a: __m256i) -> __m256i {
        _mm256_andnot_si256(a, _mm256_set1_epi16(-1_i16))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitand(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_and_si256(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitor(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_or_si256(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitxor(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_xor_si256(a, b)
    }

    // ====== Shifts ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_const<const N: i32>(self, a: __m256i) -> __m256i {
        _mm256_slli_epi16::<N>(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_const<const N: i32>(self, a: __m256i) -> __m256i {
        _mm256_srli_epi16::<N>(a)
    }

    // ====== Uniform variable shifts ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_uniform(self, a: __m256i, count: u32) -> __m256i {
        _mm256_sll_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_uniform(self, a: __m256i, count: u32) -> __m256i {
        _mm256_srl_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    // ====== Saturating arithmetic ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn saturating_add(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_adds_epu16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn saturating_sub(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_subs_epu16(a, b)
    }

    // ====== Boolean ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn all_true(self, a: __m256i) -> bool {
        _mm256_movemask_epi8(a) == -1_i32
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn any_true(self, a: __m256i) -> bool {
        _mm256_movemask_epi8(a) != 0
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitmask(self, a: __m256i) -> u32 {
        let shifted = _mm256_srai_epi16::<15>(a);
        let lo = _mm256_castsi256_si128(shifted);
        let hi = _mm256_extracti128_si256::<1>(shifted);
        let packed = _mm_packs_epi16(lo, hi);
        (_mm_movemask_epi8(packed) as u32) & 0xFFFF
    }
}

#[cfg(target_arch = "x86_64")]
impl U64x2Backend for archmage::X64V3Token {
    type Repr = __m128i;

    // ====== Construction ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn splat(self, v: u64) -> __m128i {
        _mm_set1_epi64x(v as i64)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn zero(self) -> __m128i {
        _mm_setzero_si128()
    }

    #[inline(always)]
    fn load(self, data: &[u64; 2]) -> __m128i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u64; 2]) -> __m128i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m128i, out: &mut [u64; 2]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m128i) -> [u64; 2] {
        crate::simd_storage::cast(repr)
    }

    // ====== Arithmetic ======

    sse2_baseline! {
    fn add(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_add_epi64(a, b)
    }
    }

    sse2_baseline! {
    fn sub(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_sub_epi64(a, b)
    }
    }

    // ====== Math ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn min(self, a: __m128i, b: __m128i) -> __m128i {
        let bias = _mm_set1_epi64x(i64::MIN);
        let a_biased = _mm_xor_si128(a, bias);
        let b_biased = _mm_xor_si128(b, bias);
        let mask = _mm_cmpgt_epi64(a_biased, b_biased);
        _mm_blendv_epi8(a, b, mask)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn max(self, a: __m128i, b: __m128i) -> __m128i {
        let bias = _mm_set1_epi64x(i64::MIN);
        let a_biased = _mm_xor_si128(a, bias);
        let b_biased = _mm_xor_si128(b, bias);
        let mask = _mm_cmpgt_epi64(a_biased, b_biased);
        _mm_blendv_epi8(b, a, mask)
    }

    // ====== Comparisons ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_eq(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_cmpeq_epi64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ne(self, a: __m128i, b: __m128i) -> __m128i {
        let eq = _mm_cmpeq_epi64(a, b);
        _mm_andnot_si128(eq, _mm_set1_epi64x(-1_i64))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_gt(self, a: __m128i, b: __m128i) -> __m128i {
        let bias = _mm_set1_epi64x(i64::MIN);
        let sa = _mm_xor_si128(a, bias);
        let sb = _mm_xor_si128(b, bias);
        _mm_cmpgt_epi64(sa, sb)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_lt(self, a: __m128i, b: __m128i) -> __m128i {
        <Self as U64x2Backend>::simd_gt(_self, b, a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_le(self, a: __m128i, b: __m128i) -> __m128i {
        let gt = <Self as U64x2Backend>::simd_gt(_self, a, b);
        _mm_andnot_si128(gt, _mm_set1_epi64x(-1_i64))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ge(self, a: __m128i, b: __m128i) -> __m128i {
        let lt = <Self as U64x2Backend>::simd_gt(_self, b, a);
        _mm_andnot_si128(lt, _mm_set1_epi64x(-1_i64))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn blend(self, mask: __m128i, if_true: __m128i, if_false: __m128i) -> __m128i {
        _mm_blendv_epi8(if_false, if_true, mask)
    }

    // ====== Reductions ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_add(self, a: __m128i) -> u64 {
        let arr = <Self as U64x2Backend>::to_array(_self, a);
        arr.iter().copied().fold(0u64, u64::wrapping_add)
    }

    // ====== Bitwise ======

    sse2_baseline! {
    fn not(self, a: __m128i) -> __m128i {
        _mm_andnot_si128(a, _mm_set1_epi64x(-1_i64))
    }
    }

    sse2_baseline! {
    fn bitand(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_and_si128(a, b)
    }
    }

    sse2_baseline! {
    fn bitor(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_or_si128(a, b)
    }
    }

    sse2_baseline! {
    fn bitxor(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_xor_si128(a, b)
    }
    }

    // ====== Shifts ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_const<const N: i32>(self, a: __m128i) -> __m128i {
        _mm_slli_epi64::<N>(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_const<const N: i32>(self, a: __m128i) -> __m128i {
        _mm_srli_epi64::<N>(a)
    }

    // ====== Boolean ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn all_true(self, a: __m128i) -> bool {
        _mm_movemask_pd(_mm_castsi128_pd(a)) == 0x3
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn any_true(self, a: __m128i) -> bool {
        _mm_movemask_pd(_mm_castsi128_pd(a)) != 0
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitmask(self, a: __m128i) -> u32 {
        _mm_movemask_pd(_mm_castsi128_pd(a)) as u32
    }
}

#[cfg(target_arch = "x86_64")]
impl U64x4Backend for archmage::X64V3Token {
    type Repr = __m256i;

    // ====== Construction ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn splat(self, v: u64) -> __m256i {
        _mm256_set1_epi64x(v as i64)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn zero(self) -> __m256i {
        _mm256_setzero_si256()
    }

    #[inline(always)]
    fn load(self, data: &[u64; 4]) -> __m256i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u64; 4]) -> __m256i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m256i, out: &mut [u64; 4]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m256i) -> [u64; 4] {
        crate::simd_storage::cast(repr)
    }

    // ====== Arithmetic ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn add(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_add_epi64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn sub(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_sub_epi64(a, b)
    }

    // ====== Math ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn min(self, a: __m256i, b: __m256i) -> __m256i {
        let bias = _mm256_set1_epi64x(i64::MIN);
        let a_biased = _mm256_xor_si256(a, bias);
        let b_biased = _mm256_xor_si256(b, bias);
        let mask = _mm256_cmpgt_epi64(a_biased, b_biased);
        _mm256_blendv_epi8(a, b, mask)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn max(self, a: __m256i, b: __m256i) -> __m256i {
        let bias = _mm256_set1_epi64x(i64::MIN);
        let a_biased = _mm256_xor_si256(a, bias);
        let b_biased = _mm256_xor_si256(b, bias);
        let mask = _mm256_cmpgt_epi64(a_biased, b_biased);
        _mm256_blendv_epi8(b, a, mask)
    }

    // ====== Comparisons ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_eq(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpeq_epi64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ne(self, a: __m256i, b: __m256i) -> __m256i {
        let eq = _mm256_cmpeq_epi64(a, b);
        _mm256_andnot_si256(eq, _mm256_set1_epi64x(-1_i64))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_gt(self, a: __m256i, b: __m256i) -> __m256i {
        let bias = _mm256_set1_epi64x(i64::MIN);
        let sa = _mm256_xor_si256(a, bias);
        let sb = _mm256_xor_si256(b, bias);
        _mm256_cmpgt_epi64(sa, sb)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_lt(self, a: __m256i, b: __m256i) -> __m256i {
        <Self as U64x4Backend>::simd_gt(_self, b, a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_le(self, a: __m256i, b: __m256i) -> __m256i {
        let gt = <Self as U64x4Backend>::simd_gt(_self, a, b);
        _mm256_andnot_si256(gt, _mm256_set1_epi64x(-1_i64))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn simd_ge(self, a: __m256i, b: __m256i) -> __m256i {
        let lt = <Self as U64x4Backend>::simd_gt(_self, b, a);
        _mm256_andnot_si256(lt, _mm256_set1_epi64x(-1_i64))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn blend(self, mask: __m256i, if_true: __m256i, if_false: __m256i) -> __m256i {
        _mm256_blendv_epi8(if_false, if_true, mask)
    }

    // ====== Reductions ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn reduce_add(self, a: __m256i) -> u64 {
        let arr = <Self as U64x4Backend>::to_array(_self, a);
        arr.iter().copied().fold(0u64, u64::wrapping_add)
    }

    // ====== Bitwise ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn not(self, a: __m256i) -> __m256i {
        _mm256_andnot_si256(a, _mm256_set1_epi64x(-1_i64))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitand(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_and_si256(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitor(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_or_si256(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitxor(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_xor_si256(a, b)
    }

    // ====== Shifts ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shl_const<const N: i32>(self, a: __m256i) -> __m256i {
        _mm256_slli_epi64::<N>(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn shr_logical_const<const N: i32>(self, a: __m256i) -> __m256i {
        _mm256_srli_epi64::<N>(a)
    }

    // ====== Boolean ======

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn all_true(self, a: __m256i) -> bool {
        _mm256_movemask_pd(_mm256_castsi256_pd(a)) == 0xF
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn any_true(self, a: __m256i) -> bool {
        _mm256_movemask_pd(_mm256_castsi256_pd(a)) != 0
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitmask(self, a: __m256i) -> u32 {
        _mm256_movemask_pd(_mm256_castsi256_pd(a)) as u32
    }
}

#[cfg(target_arch = "x86_64")]
impl F32x4Convert for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_f32_to_i32(self, a: __m128) -> __m128i {
        _mm_castps_si128(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_i32_to_f32(self, a: __m128i) -> __m128 {
        _mm_castsi128_ps(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn convert_f32_to_i32(self, a: __m128) -> __m128i {
        _mm_cvttps_epi32(a)
    }

    // Issue #80 fixup: cvttps yields the i32::MIN sentinel for +overflow
    // and NaN; patch those two classes (-overflow already saturates to MIN).
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn convert_f32_to_i32_saturating(self, a: __m128) -> __m128i {
        let t = _mm_cvttps_epi32(a);
        let big = _mm_cmp_ps::<_CMP_GE_OQ>(a, _mm_set1_ps(2_147_483_648.0));
        let t = _mm_blendv_epi8(t, _mm_set1_epi32(i32::MAX), _mm_castps_si128(big));
        let nan = _mm_cmp_ps::<_CMP_UNORD_Q>(a, a);
        _mm_andnot_si128(_mm_castps_si128(nan), t)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn convert_f32_to_i32_round(self, a: __m128) -> __m128i {
        _mm_cvtps_epi32(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn convert_i32_to_f32(self, a: __m128i) -> __m128 {
        _mm_cvtepi32_ps(a)
    }
}

#[cfg(target_arch = "x86_64")]
impl F32x8Convert for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_f32_to_i32(self, a: __m256) -> __m256i {
        _mm256_castps_si256(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_i32_to_f32(self, a: __m256i) -> __m256 {
        _mm256_castsi256_ps(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn convert_f32_to_i32(self, a: __m256) -> __m256i {
        _mm256_cvttps_epi32(a)
    }

    // Issue #80 fixup — see the 128-bit impl.
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn convert_f32_to_i32_saturating(self, a: __m256) -> __m256i {
        let t = _mm256_cvttps_epi32(a);
        let big = _mm256_cmp_ps::<_CMP_GE_OQ>(a, _mm256_set1_ps(2_147_483_648.0));
        let t = _mm256_blendv_epi8(t, _mm256_set1_epi32(i32::MAX), _mm256_castps_si256(big));
        let nan = _mm256_cmp_ps::<_CMP_UNORD_Q>(a, a);
        _mm256_andnot_si256(_mm256_castps_si256(nan), t)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn convert_f32_to_i32_round(self, a: __m256) -> __m256i {
        _mm256_cvtps_epi32(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn convert_i32_to_f32(self, a: __m256i) -> __m256 {
        _mm256_cvtepi32_ps(a)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "w512"))]
impl F32x16Convert for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_f32_to_i32(self, a: [__m256; 2]) -> [__m256i; 2] {
        [_mm256_castps_si256(a[0]), _mm256_castps_si256(a[1])]
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_i32_to_f32(self, a: [__m256i; 2]) -> [__m256; 2] {
        [_mm256_castsi256_ps(a[0]), _mm256_castsi256_ps(a[1])]
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn convert_f32_to_i32(self, a: [__m256; 2]) -> [__m256i; 2] {
        [_mm256_cvttps_epi32(a[0]), _mm256_cvttps_epi32(a[1])]
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn convert_f32_to_i32_saturating(self, a: [__m256; 2]) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            let t = _mm256_cvttps_epi32(a[i]);
            let big = _mm256_cmp_ps::<_CMP_GE_OQ>(a[i], _mm256_set1_ps(2_147_483_648.0));
            let t = _mm256_blendv_epi8(t, _mm256_set1_epi32(i32::MAX), _mm256_castps_si256(big));
            let nan = _mm256_cmp_ps::<_CMP_UNORD_Q>(a[i], a[i]);
            _mm256_andnot_si256(_mm256_castps_si256(nan), t)
        })
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn convert_f32_to_i32_round(self, a: [__m256; 2]) -> [__m256i; 2] {
        [_mm256_cvtps_epi32(a[0]), _mm256_cvtps_epi32(a[1])]
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn convert_i32_to_f32(self, a: [__m256i; 2]) -> [__m256; 2] {
        [_mm256_cvtepi32_ps(a[0]), _mm256_cvtepi32_ps(a[1])]
    }
}

#[cfg(target_arch = "x86_64")]
impl U32x4Bitcast for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_u32_to_i32(self, a: __m128i) -> __m128i {
        a
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_i32_to_u32(self, a: __m128i) -> __m128i {
        a
    }
}

#[cfg(target_arch = "x86_64")]
impl U32x8Bitcast for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_u32_to_i32(self, a: __m256i) -> __m256i {
        a
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_i32_to_u32(self, a: __m256i) -> __m256i {
        a
    }
}

#[cfg(target_arch = "x86_64")]
impl I64x2Bitcast for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_i64_to_f64(self, a: __m128i) -> __m128d {
        _mm_castsi128_pd(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_f64_to_i64(self, a: __m128d) -> __m128i {
        _mm_castpd_si128(a)
    }
}

#[cfg(target_arch = "x86_64")]
impl I64x4Bitcast for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_i64_to_f64(self, a: __m256i) -> __m256d {
        _mm256_castsi256_pd(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_f64_to_i64(self, a: __m256d) -> __m256i {
        _mm256_castpd_si256(a)
    }
}

#[cfg(target_arch = "x86_64")]
impl I8x16Bitcast for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_i8_to_u8(self, a: __m128i) -> __m128i {
        a
    }
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_u8_to_i8(self, a: __m128i) -> __m128i {
        a
    }
}

#[cfg(target_arch = "x86_64")]
impl I8x32Bitcast for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_i8_to_u8(self, a: __m256i) -> __m256i {
        a
    }
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_u8_to_i8(self, a: __m256i) -> __m256i {
        a
    }
}

#[cfg(target_arch = "x86_64")]
impl I16x8Bitcast for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_i16_to_u16(self, a: __m128i) -> __m128i {
        a
    }
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_u16_to_i16(self, a: __m128i) -> __m128i {
        a
    }
}

#[cfg(target_arch = "x86_64")]
impl I16x16Bitcast for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_i16_to_u16(self, a: __m256i) -> __m256i {
        a
    }
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_u16_to_i16(self, a: __m256i) -> __m256i {
        a
    }
}

#[cfg(target_arch = "x86_64")]
impl U64x2Bitcast for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_u64_to_i64(self, a: __m128i) -> __m128i {
        a
    }
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_i64_to_u64(self, a: __m128i) -> __m128i {
        a
    }
}

#[cfg(target_arch = "x86_64")]
impl U64x4Bitcast for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_u64_to_i64(self, a: __m256i) -> __m256i {
        a
    }
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn bitcast_i64_to_u64(self, a: __m256i) -> __m256i {
        a
    }
}

#[cfg(target_arch = "x86_64")]
impl U8x16Widen for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn widen_low_u8_to_u16(self, a: __m128i) -> __m128i {
        _mm_cvtepu8_epi16(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn widen_high_u8_to_u16(self, a: __m128i) -> __m128i {
        _mm_cvtepu8_epi16(_mm_srli_si128::<8>(a))
    }
}

#[cfg(target_arch = "x86_64")]
impl U16x8Widen for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn widen_low_u16_to_u32(self, a: __m128i) -> __m128i {
        _mm_cvtepu16_epi32(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn widen_high_u16_to_u32(self, a: __m128i) -> __m128i {
        _mm_cvtepu16_epi32(_mm_srli_si128::<8>(a))
    }
}

#[cfg(target_arch = "x86_64")]
impl I8x16Widen for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn widen_low_i8_to_i16(self, a: __m128i) -> __m128i {
        _mm_cvtepi8_epi16(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn widen_high_i8_to_i16(self, a: __m128i) -> __m128i {
        _mm_cvtepi8_epi16(_mm_srli_si128::<8>(a))
    }
}

#[cfg(target_arch = "x86_64")]
impl I16x8Widen for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn widen_low_i16_to_i32(self, a: __m128i) -> __m128i {
        _mm_cvtepi16_epi32(a)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn widen_high_i16_to_i32(self, a: __m128i) -> __m128i {
        _mm_cvtepi16_epi32(_mm_srli_si128::<8>(a))
    }
}

#[cfg(target_arch = "x86_64")]
impl U8x32Widen for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn widen_low_u8_to_u16(self, a: __m256i) -> __m256i {
        _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn widen_high_u8_to_u16(self, a: __m256i) -> __m256i {
        _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(a))
    }
}

#[cfg(target_arch = "x86_64")]
impl U16x16Widen for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn widen_low_u16_to_u32(self, a: __m256i) -> __m256i {
        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(a))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn widen_high_u16_to_u32(self, a: __m256i) -> __m256i {
        _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(a))
    }
}

#[cfg(target_arch = "x86_64")]
impl I8x32Widen for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn widen_low_i8_to_i16(self, a: __m256i) -> __m256i {
        _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn widen_high_i8_to_i16(self, a: __m256i) -> __m256i {
        _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(a))
    }
}

#[cfg(target_arch = "x86_64")]
impl I16x16Widen for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn widen_low_i16_to_i32(self, a: __m256i) -> __m256i {
        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn widen_high_i16_to_i32(self, a: __m256i) -> __m256i {
        _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(a))
    }
}

#[cfg(target_arch = "x86_64")]
impl I16x8Narrow for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn narrow_saturating_i16_to_i8(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_packs_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn narrow_saturating_i16_to_u8(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_packus_epi16(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
impl I32x4Narrow for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn narrow_saturating_i32_to_i16(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_packs_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn narrow_saturating_i32_to_u16(self, a: __m128i, b: __m128i) -> __m128i {
        _mm_packus_epi32(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
impl I16x16Narrow for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn narrow_saturating_i16_to_i8(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi16(a, b))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn narrow_saturating_i16_to_u8(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(a, b))
    }
}

#[cfg(target_arch = "x86_64")]
impl I32x8Narrow for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn narrow_saturating_i32_to_i16(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(a, b))
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn narrow_saturating_i32_to_u16(self, a: __m256i, b: __m256i) -> __m256i {
        _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(a, b))
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl U8x64Widen for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn widen_low_u8_to_u16(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a[0])),
            _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(a[0])),
        ]
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn widen_high_u8_to_u16(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a[1])),
            _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(a[1])),
        ]
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl U16x32Widen for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn widen_low_u16_to_u32(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            _mm256_cvtepu16_epi32(_mm256_castsi256_si128(a[0])),
            _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(a[0])),
        ]
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn widen_high_u16_to_u32(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            _mm256_cvtepu16_epi32(_mm256_castsi256_si128(a[1])),
            _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(a[1])),
        ]
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I8x64Widen for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn widen_low_i8_to_i16(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a[0])),
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(a[0])),
        ]
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn widen_high_i8_to_i16(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a[1])),
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(a[1])),
        ]
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I16x32Widen for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn widen_low_i16_to_i32(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a[0])),
            _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(a[0])),
        ]
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn widen_high_i16_to_i32(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a[1])),
            _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(a[1])),
        ]
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I16x32Narrow for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn narrow_saturating_i16_to_i8(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi16(a[0], a[1])),
            _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi16(b[0], b[1])),
        ]
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn narrow_saturating_i16_to_u8(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(a[0], a[1])),
            _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(b[0], b[1])),
        ]
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I32x16Narrow for archmage::X64V3Token {
    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn narrow_saturating_i32_to_i16(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(a[0], a[1])),
            _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(b[0], b[1])),
        ]
    }

    #[arcane(suppress_const_test, _self = X64V3Token)]
    fn narrow_saturating_i32_to_u16(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(a[0], a[1])),
            _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(b[0], b[1])),
        ]
    }
}
#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl F32x16Backend for archmage::X64V3Token {
    type Repr = [__m256; 2];

    #[inline(always)]
    fn splat(self, v: f32) -> [__m256; 2] {
        let h = <archmage::X64V3Token as F32x8Backend>::splat(self, v);
        [h, h]
    }

    #[inline(always)]
    fn zero(self) -> [__m256; 2] {
        let h = <archmage::X64V3Token as F32x8Backend>::zero(self);
        [h, h]
    }

    #[inline(always)]
    fn load(self, data: &[f32; 16]) -> [__m256; 2] {
        let (lo, hi) = data.split_at(8);
        [
            <archmage::X64V3Token as F32x8Backend>::load(self, lo.try_into().unwrap()),
            <archmage::X64V3Token as F32x8Backend>::load(self, hi.try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(self, arr: [f32; 16]) -> [__m256; 2] {
        let mut lo = [0.0f32; 8];
        let mut hi = [0.0f32; 8];
        lo.copy_from_slice(&arr[..8]);
        hi.copy_from_slice(&arr[8..]);
        [
            <archmage::X64V3Token as F32x8Backend>::from_array(self, lo),
            <archmage::X64V3Token as F32x8Backend>::from_array(self, hi),
        ]
    }

    #[inline(always)]
    fn store(self, repr: [__m256; 2], out: &mut [f32; 16]) {
        let (lo, hi) = out.split_at_mut(8);
        <archmage::X64V3Token as F32x8Backend>::store(self, repr[0], lo.try_into().unwrap());
        <archmage::X64V3Token as F32x8Backend>::store(self, repr[1], hi.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(self, repr: [__m256; 2]) -> [f32; 16] {
        let lo = <archmage::X64V3Token as F32x8Backend>::to_array(self, repr[0]);
        let hi = <archmage::X64V3Token as F32x8Backend>::to_array(self, repr[1]);
        let mut out = [0.0f32; 16];
        out[..8].copy_from_slice(&lo);
        out[8..].copy_from_slice(&hi);
        out
    }

    #[inline(always)]
    fn add(self, a: [__m256; 2], b: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::add(self, a[0], b[0]),
            <archmage::X64V3Token as F32x8Backend>::add(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn sub(self, a: [__m256; 2], b: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::sub(self, a[0], b[0]),
            <archmage::X64V3Token as F32x8Backend>::sub(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn mul(self, a: [__m256; 2], b: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::mul(self, a[0], b[0]),
            <archmage::X64V3Token as F32x8Backend>::mul(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn div(self, a: [__m256; 2], b: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::div(self, a[0], b[0]),
            <archmage::X64V3Token as F32x8Backend>::div(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn neg(self, a: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::neg(self, a[0]),
            <archmage::X64V3Token as F32x8Backend>::neg(self, a[1]),
        ]
    }

    #[inline(always)]
    fn min(self, a: [__m256; 2], b: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::min(self, a[0], b[0]),
            <archmage::X64V3Token as F32x8Backend>::min(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn max(self, a: [__m256; 2], b: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::max(self, a[0], b[0]),
            <archmage::X64V3Token as F32x8Backend>::max(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn sqrt(self, a: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::sqrt(self, a[0]),
            <archmage::X64V3Token as F32x8Backend>::sqrt(self, a[1]),
        ]
    }

    #[inline(always)]
    fn abs(self, a: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::abs(self, a[0]),
            <archmage::X64V3Token as F32x8Backend>::abs(self, a[1]),
        ]
    }

    #[inline(always)]
    fn floor(self, a: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::floor(self, a[0]),
            <archmage::X64V3Token as F32x8Backend>::floor(self, a[1]),
        ]
    }

    #[inline(always)]
    fn ceil(self, a: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::ceil(self, a[0]),
            <archmage::X64V3Token as F32x8Backend>::ceil(self, a[1]),
        ]
    }

    #[inline(always)]
    fn round(self, a: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::round(self, a[0]),
            <archmage::X64V3Token as F32x8Backend>::round(self, a[1]),
        ]
    }

    #[inline(always)]
    fn mul_add(self, a: [__m256; 2], b: [__m256; 2], c: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::mul_add(self, a[0], b[0], c[0]),
            <archmage::X64V3Token as F32x8Backend>::mul_add(self, a[1], b[1], c[1]),
        ]
    }

    #[inline(always)]
    fn mul_sub(self, a: [__m256; 2], b: [__m256; 2], c: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::mul_sub(self, a[0], b[0], c[0]),
            <archmage::X64V3Token as F32x8Backend>::mul_sub(self, a[1], b[1], c[1]),
        ]
    }

    #[inline(always)]
    fn reduce_add(self, a: [__m256; 2]) -> f32 {
        <archmage::X64V3Token as F32x8Backend>::reduce_add(self, a[0])
            + <archmage::X64V3Token as F32x8Backend>::reduce_add(self, a[1])
    }

    #[inline(always)]
    fn reduce_min(self, a: [__m256; 2]) -> f32 {
        let lo = <archmage::X64V3Token as F32x8Backend>::reduce_min(self, a[0]);
        let hi = <archmage::X64V3Token as F32x8Backend>::reduce_min(self, a[1]);
        if lo < hi { lo } else { hi }
    }

    #[inline(always)]
    fn reduce_max(self, a: [__m256; 2]) -> f32 {
        let lo = <archmage::X64V3Token as F32x8Backend>::reduce_max(self, a[0]);
        let hi = <archmage::X64V3Token as F32x8Backend>::reduce_max(self, a[1]);
        if lo > hi { lo } else { hi }
    }

    #[inline(always)]
    fn rcp_approx(self, a: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::rcp_approx(self, a[0]),
            <archmage::X64V3Token as F32x8Backend>::rcp_approx(self, a[1]),
        ]
    }

    #[inline(always)]
    fn rsqrt_approx(self, a: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::rsqrt_approx(self, a[0]),
            <archmage::X64V3Token as F32x8Backend>::rsqrt_approx(self, a[1]),
        ]
    }

    #[inline(always)]
    fn recip(self, a: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::recip(self, a[0]),
            <archmage::X64V3Token as F32x8Backend>::recip(self, a[1]),
        ]
    }

    #[inline(always)]
    fn rsqrt(self, a: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::rsqrt(self, a[0]),
            <archmage::X64V3Token as F32x8Backend>::rsqrt(self, a[1]),
        ]
    }

    #[inline(always)]
    fn simd_eq(self, a: [__m256; 2], b: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::simd_eq(self, a[0], b[0]),
            <archmage::X64V3Token as F32x8Backend>::simd_eq(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_ne(self, a: [__m256; 2], b: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::simd_ne(self, a[0], b[0]),
            <archmage::X64V3Token as F32x8Backend>::simd_ne(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_lt(self, a: [__m256; 2], b: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::simd_lt(self, a[0], b[0]),
            <archmage::X64V3Token as F32x8Backend>::simd_lt(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_le(self, a: [__m256; 2], b: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::simd_le(self, a[0], b[0]),
            <archmage::X64V3Token as F32x8Backend>::simd_le(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_gt(self, a: [__m256; 2], b: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::simd_gt(self, a[0], b[0]),
            <archmage::X64V3Token as F32x8Backend>::simd_gt(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_ge(self, a: [__m256; 2], b: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::simd_ge(self, a[0], b[0]),
            <archmage::X64V3Token as F32x8Backend>::simd_ge(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn blend(self, mask: [__m256; 2], if_true: [__m256; 2], if_false: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::blend(self, mask[0], if_true[0], if_false[0]),
            <archmage::X64V3Token as F32x8Backend>::blend(self, mask[1], if_true[1], if_false[1]),
        ]
    }

    #[inline(always)]
    fn not(self, a: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::not(self, a[0]),
            <archmage::X64V3Token as F32x8Backend>::not(self, a[1]),
        ]
    }

    #[inline(always)]
    fn bitand(self, a: [__m256; 2], b: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::bitand(self, a[0], b[0]),
            <archmage::X64V3Token as F32x8Backend>::bitand(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn bitor(self, a: [__m256; 2], b: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::bitor(self, a[0], b[0]),
            <archmage::X64V3Token as F32x8Backend>::bitor(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn bitxor(self, a: [__m256; 2], b: [__m256; 2]) -> [__m256; 2] {
        [
            <archmage::X64V3Token as F32x8Backend>::bitxor(self, a[0], b[0]),
            <archmage::X64V3Token as F32x8Backend>::bitxor(self, a[1], b[1]),
        ]
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl F64x8Backend for archmage::X64V3Token {
    type Repr = [__m256d; 2];

    #[inline(always)]
    fn splat(self, v: f64) -> [__m256d; 2] {
        let h = <archmage::X64V3Token as F64x4Backend>::splat(self, v);
        [h, h]
    }

    #[inline(always)]
    fn zero(self) -> [__m256d; 2] {
        let h = <archmage::X64V3Token as F64x4Backend>::zero(self);
        [h, h]
    }

    #[inline(always)]
    fn load(self, data: &[f64; 8]) -> [__m256d; 2] {
        let (lo, hi) = data.split_at(4);
        [
            <archmage::X64V3Token as F64x4Backend>::load(self, lo.try_into().unwrap()),
            <archmage::X64V3Token as F64x4Backend>::load(self, hi.try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(self, arr: [f64; 8]) -> [__m256d; 2] {
        let mut lo = [0.0f64; 4];
        let mut hi = [0.0f64; 4];
        lo.copy_from_slice(&arr[..4]);
        hi.copy_from_slice(&arr[4..]);
        [
            <archmage::X64V3Token as F64x4Backend>::from_array(self, lo),
            <archmage::X64V3Token as F64x4Backend>::from_array(self, hi),
        ]
    }

    #[inline(always)]
    fn store(self, repr: [__m256d; 2], out: &mut [f64; 8]) {
        let (lo, hi) = out.split_at_mut(4);
        <archmage::X64V3Token as F64x4Backend>::store(self, repr[0], lo.try_into().unwrap());
        <archmage::X64V3Token as F64x4Backend>::store(self, repr[1], hi.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(self, repr: [__m256d; 2]) -> [f64; 8] {
        let lo = <archmage::X64V3Token as F64x4Backend>::to_array(self, repr[0]);
        let hi = <archmage::X64V3Token as F64x4Backend>::to_array(self, repr[1]);
        let mut out = [0.0f64; 8];
        out[..4].copy_from_slice(&lo);
        out[4..].copy_from_slice(&hi);
        out
    }

    #[inline(always)]
    fn add(self, a: [__m256d; 2], b: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::add(self, a[0], b[0]),
            <archmage::X64V3Token as F64x4Backend>::add(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn sub(self, a: [__m256d; 2], b: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::sub(self, a[0], b[0]),
            <archmage::X64V3Token as F64x4Backend>::sub(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn mul(self, a: [__m256d; 2], b: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::mul(self, a[0], b[0]),
            <archmage::X64V3Token as F64x4Backend>::mul(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn div(self, a: [__m256d; 2], b: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::div(self, a[0], b[0]),
            <archmage::X64V3Token as F64x4Backend>::div(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn neg(self, a: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::neg(self, a[0]),
            <archmage::X64V3Token as F64x4Backend>::neg(self, a[1]),
        ]
    }

    #[inline(always)]
    fn min(self, a: [__m256d; 2], b: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::min(self, a[0], b[0]),
            <archmage::X64V3Token as F64x4Backend>::min(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn max(self, a: [__m256d; 2], b: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::max(self, a[0], b[0]),
            <archmage::X64V3Token as F64x4Backend>::max(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn sqrt(self, a: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::sqrt(self, a[0]),
            <archmage::X64V3Token as F64x4Backend>::sqrt(self, a[1]),
        ]
    }

    #[inline(always)]
    fn abs(self, a: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::abs(self, a[0]),
            <archmage::X64V3Token as F64x4Backend>::abs(self, a[1]),
        ]
    }

    #[inline(always)]
    fn floor(self, a: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::floor(self, a[0]),
            <archmage::X64V3Token as F64x4Backend>::floor(self, a[1]),
        ]
    }

    #[inline(always)]
    fn ceil(self, a: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::ceil(self, a[0]),
            <archmage::X64V3Token as F64x4Backend>::ceil(self, a[1]),
        ]
    }

    #[inline(always)]
    fn round(self, a: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::round(self, a[0]),
            <archmage::X64V3Token as F64x4Backend>::round(self, a[1]),
        ]
    }

    #[inline(always)]
    fn mul_add(self, a: [__m256d; 2], b: [__m256d; 2], c: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::mul_add(self, a[0], b[0], c[0]),
            <archmage::X64V3Token as F64x4Backend>::mul_add(self, a[1], b[1], c[1]),
        ]
    }

    #[inline(always)]
    fn mul_sub(self, a: [__m256d; 2], b: [__m256d; 2], c: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::mul_sub(self, a[0], b[0], c[0]),
            <archmage::X64V3Token as F64x4Backend>::mul_sub(self, a[1], b[1], c[1]),
        ]
    }

    #[inline(always)]
    fn reduce_add(self, a: [__m256d; 2]) -> f64 {
        <archmage::X64V3Token as F64x4Backend>::reduce_add(self, a[0])
            + <archmage::X64V3Token as F64x4Backend>::reduce_add(self, a[1])
    }

    #[inline(always)]
    fn reduce_min(self, a: [__m256d; 2]) -> f64 {
        let lo = <archmage::X64V3Token as F64x4Backend>::reduce_min(self, a[0]);
        let hi = <archmage::X64V3Token as F64x4Backend>::reduce_min(self, a[1]);
        if lo < hi { lo } else { hi }
    }

    #[inline(always)]
    fn reduce_max(self, a: [__m256d; 2]) -> f64 {
        let lo = <archmage::X64V3Token as F64x4Backend>::reduce_max(self, a[0]);
        let hi = <archmage::X64V3Token as F64x4Backend>::reduce_max(self, a[1]);
        if lo > hi { lo } else { hi }
    }

    #[inline(always)]
    fn rcp_approx(self, a: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::rcp_approx(self, a[0]),
            <archmage::X64V3Token as F64x4Backend>::rcp_approx(self, a[1]),
        ]
    }

    #[inline(always)]
    fn rsqrt_approx(self, a: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::rsqrt_approx(self, a[0]),
            <archmage::X64V3Token as F64x4Backend>::rsqrt_approx(self, a[1]),
        ]
    }

    #[inline(always)]
    fn recip(self, a: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::recip(self, a[0]),
            <archmage::X64V3Token as F64x4Backend>::recip(self, a[1]),
        ]
    }

    #[inline(always)]
    fn rsqrt(self, a: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::rsqrt(self, a[0]),
            <archmage::X64V3Token as F64x4Backend>::rsqrt(self, a[1]),
        ]
    }

    #[inline(always)]
    fn simd_eq(self, a: [__m256d; 2], b: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::simd_eq(self, a[0], b[0]),
            <archmage::X64V3Token as F64x4Backend>::simd_eq(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_ne(self, a: [__m256d; 2], b: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::simd_ne(self, a[0], b[0]),
            <archmage::X64V3Token as F64x4Backend>::simd_ne(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_lt(self, a: [__m256d; 2], b: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::simd_lt(self, a[0], b[0]),
            <archmage::X64V3Token as F64x4Backend>::simd_lt(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_le(self, a: [__m256d; 2], b: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::simd_le(self, a[0], b[0]),
            <archmage::X64V3Token as F64x4Backend>::simd_le(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_gt(self, a: [__m256d; 2], b: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::simd_gt(self, a[0], b[0]),
            <archmage::X64V3Token as F64x4Backend>::simd_gt(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_ge(self, a: [__m256d; 2], b: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::simd_ge(self, a[0], b[0]),
            <archmage::X64V3Token as F64x4Backend>::simd_ge(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn blend(
        self,
        mask: [__m256d; 2],
        if_true: [__m256d; 2],
        if_false: [__m256d; 2],
    ) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::blend(self, mask[0], if_true[0], if_false[0]),
            <archmage::X64V3Token as F64x4Backend>::blend(self, mask[1], if_true[1], if_false[1]),
        ]
    }

    #[inline(always)]
    fn not(self, a: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::not(self, a[0]),
            <archmage::X64V3Token as F64x4Backend>::not(self, a[1]),
        ]
    }

    #[inline(always)]
    fn bitand(self, a: [__m256d; 2], b: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::bitand(self, a[0], b[0]),
            <archmage::X64V3Token as F64x4Backend>::bitand(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn bitor(self, a: [__m256d; 2], b: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::bitor(self, a[0], b[0]),
            <archmage::X64V3Token as F64x4Backend>::bitor(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn bitxor(self, a: [__m256d; 2], b: [__m256d; 2]) -> [__m256d; 2] {
        [
            <archmage::X64V3Token as F64x4Backend>::bitxor(self, a[0], b[0]),
            <archmage::X64V3Token as F64x4Backend>::bitxor(self, a[1], b[1]),
        ]
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I8x64Backend for archmage::X64V3Token {
    type Repr = [__m256i; 2];

    #[inline(always)]
    fn splat(self, v: i8) -> [__m256i; 2] {
        let h = <archmage::X64V3Token as I8x32Backend>::splat(self, v);
        [h, h]
    }

    #[inline(always)]
    fn zero(self) -> [__m256i; 2] {
        let h = <archmage::X64V3Token as I8x32Backend>::zero(self);
        [h, h]
    }

    #[inline(always)]
    fn load(self, data: &[i8; 64]) -> [__m256i; 2] {
        let (lo, hi) = data.split_at(32);
        [
            <archmage::X64V3Token as I8x32Backend>::load(self, lo.try_into().unwrap()),
            <archmage::X64V3Token as I8x32Backend>::load(self, hi.try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(self, arr: [i8; 64]) -> [__m256i; 2] {
        let mut lo = [0; 32];
        let mut hi = [0; 32];
        lo.copy_from_slice(&arr[..32]);
        hi.copy_from_slice(&arr[32..]);
        [
            <archmage::X64V3Token as I8x32Backend>::from_array(self, lo),
            <archmage::X64V3Token as I8x32Backend>::from_array(self, hi),
        ]
    }

    #[inline(always)]
    fn store(self, repr: [__m256i; 2], out: &mut [i8; 64]) {
        let (lo, hi) = out.split_at_mut(32);
        <archmage::X64V3Token as I8x32Backend>::store(self, repr[0], lo.try_into().unwrap());
        <archmage::X64V3Token as I8x32Backend>::store(self, repr[1], hi.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(self, repr: [__m256i; 2]) -> [i8; 64] {
        let lo = <archmage::X64V3Token as I8x32Backend>::to_array(self, repr[0]);
        let hi = <archmage::X64V3Token as I8x32Backend>::to_array(self, repr[1]);
        let mut out = [0; 64];
        out[..32].copy_from_slice(&lo);
        out[32..].copy_from_slice(&hi);
        out
    }

    #[inline(always)]
    fn add(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I8x32Backend>::add(self, a[0], b[0]),
            <archmage::X64V3Token as I8x32Backend>::add(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn sub(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I8x32Backend>::sub(self, a[0], b[0]),
            <archmage::X64V3Token as I8x32Backend>::sub(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn neg(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I8x32Backend>::neg(self, a[0]),
            <archmage::X64V3Token as I8x32Backend>::neg(self, a[1]),
        ]
    }

    #[inline(always)]
    fn min(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I8x32Backend>::min(self, a[0], b[0]),
            <archmage::X64V3Token as I8x32Backend>::min(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn max(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I8x32Backend>::max(self, a[0], b[0]),
            <archmage::X64V3Token as I8x32Backend>::max(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn abs(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I8x32Backend>::abs(self, a[0]),
            <archmage::X64V3Token as I8x32Backend>::abs(self, a[1]),
        ]
    }

    #[inline(always)]
    fn reduce_add(self, a: [__m256i; 2]) -> i8 {
        <archmage::X64V3Token as I8x32Backend>::reduce_add(self, a[0]).wrapping_add(
            <archmage::X64V3Token as I8x32Backend>::reduce_add(self, a[1]),
        )
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I8x32Backend>::shl_const::<N>(self, a[0]),
            <archmage::X64V3Token as I8x32Backend>::shl_const::<N>(self, a[1]),
        ]
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I8x32Backend>::shr_arithmetic_const::<N>(self, a[0]),
            <archmage::X64V3Token as I8x32Backend>::shr_arithmetic_const::<N>(self, a[1]),
        ]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I8x32Backend>::shr_logical_const::<N>(self, a[0]),
            <archmage::X64V3Token as I8x32Backend>::shr_logical_const::<N>(self, a[1]),
        ]
    }

    #[inline(always)]
    fn shl_uniform(self, a: [__m256i; 2], count: u32) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as I8x32Backend>::shl_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn shr_logical_uniform(self, a: [__m256i; 2], count: u32) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as I8x32Backend>::shr_logical_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: [__m256i; 2], count: u32) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as I8x32Backend>::shr_arithmetic_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn saturating_add(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as I8x32Backend>::saturating_add(self, a[i], b[i])
        })
    }

    #[inline(always)]
    fn saturating_sub(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as I8x32Backend>::saturating_sub(self, a[i], b[i])
        })
    }

    // AND/OR the halves first, then one reduction — one vector op +
    // one movemask instead of two movemasks + a scalar short-circuit
    // (fearless_simd's polyfill lesson; branchless, fewer ops).
    #[inline(always)]
    fn all_true(self, a: [__m256i; 2]) -> bool {
        let combined = <archmage::X64V3Token as I8x32Backend>::bitand(self, a[0], a[1]);
        <archmage::X64V3Token as I8x32Backend>::all_true(self, combined)
    }

    #[inline(always)]
    fn any_true(self, a: [__m256i; 2]) -> bool {
        let combined = <archmage::X64V3Token as I8x32Backend>::bitor(self, a[0], a[1]);
        <archmage::X64V3Token as I8x32Backend>::any_true(self, combined)
    }

    #[inline(always)]
    fn bitmask(self, a: [__m256i; 2]) -> u64 {
        let lo = <archmage::X64V3Token as I8x32Backend>::bitmask(self, a[0]) as u64;
        let hi = <archmage::X64V3Token as I8x32Backend>::bitmask(self, a[1]) as u64;
        lo | (hi << 32)
    }

    #[inline(always)]
    fn simd_eq(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I8x32Backend>::simd_eq(self, a[0], b[0]),
            <archmage::X64V3Token as I8x32Backend>::simd_eq(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_ne(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I8x32Backend>::simd_ne(self, a[0], b[0]),
            <archmage::X64V3Token as I8x32Backend>::simd_ne(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_lt(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I8x32Backend>::simd_lt(self, a[0], b[0]),
            <archmage::X64V3Token as I8x32Backend>::simd_lt(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_le(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I8x32Backend>::simd_le(self, a[0], b[0]),
            <archmage::X64V3Token as I8x32Backend>::simd_le(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_gt(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I8x32Backend>::simd_gt(self, a[0], b[0]),
            <archmage::X64V3Token as I8x32Backend>::simd_gt(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_ge(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I8x32Backend>::simd_ge(self, a[0], b[0]),
            <archmage::X64V3Token as I8x32Backend>::simd_ge(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn blend(
        self,
        mask: [__m256i; 2],
        if_true: [__m256i; 2],
        if_false: [__m256i; 2],
    ) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I8x32Backend>::blend(self, mask[0], if_true[0], if_false[0]),
            <archmage::X64V3Token as I8x32Backend>::blend(self, mask[1], if_true[1], if_false[1]),
        ]
    }

    #[inline(always)]
    fn not(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I8x32Backend>::not(self, a[0]),
            <archmage::X64V3Token as I8x32Backend>::not(self, a[1]),
        ]
    }

    #[inline(always)]
    fn bitand(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I8x32Backend>::bitand(self, a[0], b[0]),
            <archmage::X64V3Token as I8x32Backend>::bitand(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn bitor(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I8x32Backend>::bitor(self, a[0], b[0]),
            <archmage::X64V3Token as I8x32Backend>::bitor(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn bitxor(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I8x32Backend>::bitxor(self, a[0], b[0]),
            <archmage::X64V3Token as I8x32Backend>::bitxor(self, a[1], b[1]),
        ]
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl U8x64Backend for archmage::X64V3Token {
    type Repr = [__m256i; 2];

    #[inline(always)]
    fn splat(self, v: u8) -> [__m256i; 2] {
        let h = <archmage::X64V3Token as U8x32Backend>::splat(self, v);
        [h, h]
    }

    #[inline(always)]
    fn zero(self) -> [__m256i; 2] {
        let h = <archmage::X64V3Token as U8x32Backend>::zero(self);
        [h, h]
    }

    #[inline(always)]
    fn load(self, data: &[u8; 64]) -> [__m256i; 2] {
        let (lo, hi) = data.split_at(32);
        [
            <archmage::X64V3Token as U8x32Backend>::load(self, lo.try_into().unwrap()),
            <archmage::X64V3Token as U8x32Backend>::load(self, hi.try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(self, arr: [u8; 64]) -> [__m256i; 2] {
        let mut lo = [0; 32];
        let mut hi = [0; 32];
        lo.copy_from_slice(&arr[..32]);
        hi.copy_from_slice(&arr[32..]);
        [
            <archmage::X64V3Token as U8x32Backend>::from_array(self, lo),
            <archmage::X64V3Token as U8x32Backend>::from_array(self, hi),
        ]
    }

    #[inline(always)]
    fn store(self, repr: [__m256i; 2], out: &mut [u8; 64]) {
        let (lo, hi) = out.split_at_mut(32);
        <archmage::X64V3Token as U8x32Backend>::store(self, repr[0], lo.try_into().unwrap());
        <archmage::X64V3Token as U8x32Backend>::store(self, repr[1], hi.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(self, repr: [__m256i; 2]) -> [u8; 64] {
        let lo = <archmage::X64V3Token as U8x32Backend>::to_array(self, repr[0]);
        let hi = <archmage::X64V3Token as U8x32Backend>::to_array(self, repr[1]);
        let mut out = [0; 64];
        out[..32].copy_from_slice(&lo);
        out[32..].copy_from_slice(&hi);
        out
    }

    #[inline(always)]
    fn add(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U8x32Backend>::add(self, a[0], b[0]),
            <archmage::X64V3Token as U8x32Backend>::add(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn sub(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U8x32Backend>::sub(self, a[0], b[0]),
            <archmage::X64V3Token as U8x32Backend>::sub(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn neg(self, a: [__m256i; 2]) -> [__m256i; 2] {
        let z = <archmage::X64V3Token as U8x32Backend>::zero(self);
        [
            <archmage::X64V3Token as U8x32Backend>::sub(self, z, a[0]),
            <archmage::X64V3Token as U8x32Backend>::sub(self, z, a[1]),
        ]
    }

    #[inline(always)]
    fn min(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U8x32Backend>::min(self, a[0], b[0]),
            <archmage::X64V3Token as U8x32Backend>::min(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn max(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U8x32Backend>::max(self, a[0], b[0]),
            <archmage::X64V3Token as U8x32Backend>::max(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn reduce_add(self, a: [__m256i; 2]) -> u8 {
        <archmage::X64V3Token as U8x32Backend>::reduce_add(self, a[0]).wrapping_add(
            <archmage::X64V3Token as U8x32Backend>::reduce_add(self, a[1]),
        )
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U8x32Backend>::shl_const::<N>(self, a[0]),
            <archmage::X64V3Token as U8x32Backend>::shl_const::<N>(self, a[1]),
        ]
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U8x32Backend>::shr_logical_const::<N>(self, a[0]),
            <archmage::X64V3Token as U8x32Backend>::shr_logical_const::<N>(self, a[1]),
        ]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U8x32Backend>::shr_logical_const::<N>(self, a[0]),
            <archmage::X64V3Token as U8x32Backend>::shr_logical_const::<N>(self, a[1]),
        ]
    }

    #[inline(always)]
    fn shl_uniform(self, a: [__m256i; 2], count: u32) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as U8x32Backend>::shl_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn shr_logical_uniform(self, a: [__m256i; 2], count: u32) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as U8x32Backend>::shr_logical_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: [__m256i; 2], count: u32) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as U8x32Backend>::shr_logical_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn saturating_add(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as U8x32Backend>::saturating_add(self, a[i], b[i])
        })
    }

    #[inline(always)]
    fn saturating_sub(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as U8x32Backend>::saturating_sub(self, a[i], b[i])
        })
    }

    // AND/OR the halves first, then one reduction — one vector op +
    // one movemask instead of two movemasks + a scalar short-circuit
    // (fearless_simd's polyfill lesson; branchless, fewer ops).
    #[inline(always)]
    fn all_true(self, a: [__m256i; 2]) -> bool {
        let combined = <archmage::X64V3Token as U8x32Backend>::bitand(self, a[0], a[1]);
        <archmage::X64V3Token as U8x32Backend>::all_true(self, combined)
    }

    #[inline(always)]
    fn any_true(self, a: [__m256i; 2]) -> bool {
        let combined = <archmage::X64V3Token as U8x32Backend>::bitor(self, a[0], a[1]);
        <archmage::X64V3Token as U8x32Backend>::any_true(self, combined)
    }

    #[inline(always)]
    fn bitmask(self, a: [__m256i; 2]) -> u64 {
        let lo = <archmage::X64V3Token as U8x32Backend>::bitmask(self, a[0]) as u64;
        let hi = <archmage::X64V3Token as U8x32Backend>::bitmask(self, a[1]) as u64;
        lo | (hi << 32)
    }

    #[inline(always)]
    fn simd_eq(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U8x32Backend>::simd_eq(self, a[0], b[0]),
            <archmage::X64V3Token as U8x32Backend>::simd_eq(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_ne(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U8x32Backend>::simd_ne(self, a[0], b[0]),
            <archmage::X64V3Token as U8x32Backend>::simd_ne(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_lt(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U8x32Backend>::simd_lt(self, a[0], b[0]),
            <archmage::X64V3Token as U8x32Backend>::simd_lt(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_le(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U8x32Backend>::simd_le(self, a[0], b[0]),
            <archmage::X64V3Token as U8x32Backend>::simd_le(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_gt(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U8x32Backend>::simd_gt(self, a[0], b[0]),
            <archmage::X64V3Token as U8x32Backend>::simd_gt(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_ge(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U8x32Backend>::simd_ge(self, a[0], b[0]),
            <archmage::X64V3Token as U8x32Backend>::simd_ge(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn blend(
        self,
        mask: [__m256i; 2],
        if_true: [__m256i; 2],
        if_false: [__m256i; 2],
    ) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U8x32Backend>::blend(self, mask[0], if_true[0], if_false[0]),
            <archmage::X64V3Token as U8x32Backend>::blend(self, mask[1], if_true[1], if_false[1]),
        ]
    }

    #[inline(always)]
    fn not(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U8x32Backend>::not(self, a[0]),
            <archmage::X64V3Token as U8x32Backend>::not(self, a[1]),
        ]
    }

    #[inline(always)]
    fn bitand(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U8x32Backend>::bitand(self, a[0], b[0]),
            <archmage::X64V3Token as U8x32Backend>::bitand(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn bitor(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U8x32Backend>::bitor(self, a[0], b[0]),
            <archmage::X64V3Token as U8x32Backend>::bitor(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn bitxor(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U8x32Backend>::bitxor(self, a[0], b[0]),
            <archmage::X64V3Token as U8x32Backend>::bitxor(self, a[1], b[1]),
        ]
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I16x32Backend for archmage::X64V3Token {
    type Repr = [__m256i; 2];

    #[inline(always)]
    fn splat(self, v: i16) -> [__m256i; 2] {
        let h = <archmage::X64V3Token as I16x16Backend>::splat(self, v);
        [h, h]
    }

    #[inline(always)]
    fn zero(self) -> [__m256i; 2] {
        let h = <archmage::X64V3Token as I16x16Backend>::zero(self);
        [h, h]
    }

    #[inline(always)]
    fn load(self, data: &[i16; 32]) -> [__m256i; 2] {
        let (lo, hi) = data.split_at(16);
        [
            <archmage::X64V3Token as I16x16Backend>::load(self, lo.try_into().unwrap()),
            <archmage::X64V3Token as I16x16Backend>::load(self, hi.try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(self, arr: [i16; 32]) -> [__m256i; 2] {
        let mut lo = [0; 16];
        let mut hi = [0; 16];
        lo.copy_from_slice(&arr[..16]);
        hi.copy_from_slice(&arr[16..]);
        [
            <archmage::X64V3Token as I16x16Backend>::from_array(self, lo),
            <archmage::X64V3Token as I16x16Backend>::from_array(self, hi),
        ]
    }

    #[inline(always)]
    fn store(self, repr: [__m256i; 2], out: &mut [i16; 32]) {
        let (lo, hi) = out.split_at_mut(16);
        <archmage::X64V3Token as I16x16Backend>::store(self, repr[0], lo.try_into().unwrap());
        <archmage::X64V3Token as I16x16Backend>::store(self, repr[1], hi.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(self, repr: [__m256i; 2]) -> [i16; 32] {
        let lo = <archmage::X64V3Token as I16x16Backend>::to_array(self, repr[0]);
        let hi = <archmage::X64V3Token as I16x16Backend>::to_array(self, repr[1]);
        let mut out = [0; 32];
        out[..16].copy_from_slice(&lo);
        out[16..].copy_from_slice(&hi);
        out
    }

    #[inline(always)]
    fn add(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I16x16Backend>::add(self, a[0], b[0]),
            <archmage::X64V3Token as I16x16Backend>::add(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn sub(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I16x16Backend>::sub(self, a[0], b[0]),
            <archmage::X64V3Token as I16x16Backend>::sub(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn mul(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I16x16Backend>::mul(self, a[0], b[0]),
            <archmage::X64V3Token as I16x16Backend>::mul(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn neg(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I16x16Backend>::neg(self, a[0]),
            <archmage::X64V3Token as I16x16Backend>::neg(self, a[1]),
        ]
    }

    #[inline(always)]
    fn min(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I16x16Backend>::min(self, a[0], b[0]),
            <archmage::X64V3Token as I16x16Backend>::min(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn max(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I16x16Backend>::max(self, a[0], b[0]),
            <archmage::X64V3Token as I16x16Backend>::max(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn abs(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I16x16Backend>::abs(self, a[0]),
            <archmage::X64V3Token as I16x16Backend>::abs(self, a[1]),
        ]
    }

    #[inline(always)]
    fn reduce_add(self, a: [__m256i; 2]) -> i16 {
        <archmage::X64V3Token as I16x16Backend>::reduce_add(self, a[0]).wrapping_add(
            <archmage::X64V3Token as I16x16Backend>::reduce_add(self, a[1]),
        )
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I16x16Backend>::shl_const::<N>(self, a[0]),
            <archmage::X64V3Token as I16x16Backend>::shl_const::<N>(self, a[1]),
        ]
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I16x16Backend>::shr_arithmetic_const::<N>(self, a[0]),
            <archmage::X64V3Token as I16x16Backend>::shr_arithmetic_const::<N>(self, a[1]),
        ]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I16x16Backend>::shr_logical_const::<N>(self, a[0]),
            <archmage::X64V3Token as I16x16Backend>::shr_logical_const::<N>(self, a[1]),
        ]
    }

    #[inline(always)]
    fn shl_uniform(self, a: [__m256i; 2], count: u32) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as I16x16Backend>::shl_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn shr_logical_uniform(self, a: [__m256i; 2], count: u32) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as I16x16Backend>::shr_logical_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: [__m256i; 2], count: u32) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as I16x16Backend>::shr_arithmetic_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn saturating_add(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as I16x16Backend>::saturating_add(self, a[i], b[i])
        })
    }

    #[inline(always)]
    fn saturating_sub(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as I16x16Backend>::saturating_sub(self, a[i], b[i])
        })
    }

    // AND/OR the halves first, then one reduction — one vector op +
    // one movemask instead of two movemasks + a scalar short-circuit
    // (fearless_simd's polyfill lesson; branchless, fewer ops).
    #[inline(always)]
    fn all_true(self, a: [__m256i; 2]) -> bool {
        let combined = <archmage::X64V3Token as I16x16Backend>::bitand(self, a[0], a[1]);
        <archmage::X64V3Token as I16x16Backend>::all_true(self, combined)
    }

    #[inline(always)]
    fn any_true(self, a: [__m256i; 2]) -> bool {
        let combined = <archmage::X64V3Token as I16x16Backend>::bitor(self, a[0], a[1]);
        <archmage::X64V3Token as I16x16Backend>::any_true(self, combined)
    }

    #[inline(always)]
    fn bitmask(self, a: [__m256i; 2]) -> u64 {
        let lo = <archmage::X64V3Token as I16x16Backend>::bitmask(self, a[0]) as u64;
        let hi = <archmage::X64V3Token as I16x16Backend>::bitmask(self, a[1]) as u64;
        lo | (hi << 16)
    }

    #[inline(always)]
    fn simd_eq(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I16x16Backend>::simd_eq(self, a[0], b[0]),
            <archmage::X64V3Token as I16x16Backend>::simd_eq(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_ne(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I16x16Backend>::simd_ne(self, a[0], b[0]),
            <archmage::X64V3Token as I16x16Backend>::simd_ne(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_lt(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I16x16Backend>::simd_lt(self, a[0], b[0]),
            <archmage::X64V3Token as I16x16Backend>::simd_lt(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_le(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I16x16Backend>::simd_le(self, a[0], b[0]),
            <archmage::X64V3Token as I16x16Backend>::simd_le(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_gt(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I16x16Backend>::simd_gt(self, a[0], b[0]),
            <archmage::X64V3Token as I16x16Backend>::simd_gt(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_ge(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I16x16Backend>::simd_ge(self, a[0], b[0]),
            <archmage::X64V3Token as I16x16Backend>::simd_ge(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn blend(
        self,
        mask: [__m256i; 2],
        if_true: [__m256i; 2],
        if_false: [__m256i; 2],
    ) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I16x16Backend>::blend(self, mask[0], if_true[0], if_false[0]),
            <archmage::X64V3Token as I16x16Backend>::blend(self, mask[1], if_true[1], if_false[1]),
        ]
    }

    #[inline(always)]
    fn not(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I16x16Backend>::not(self, a[0]),
            <archmage::X64V3Token as I16x16Backend>::not(self, a[1]),
        ]
    }

    #[inline(always)]
    fn bitand(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I16x16Backend>::bitand(self, a[0], b[0]),
            <archmage::X64V3Token as I16x16Backend>::bitand(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn bitor(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I16x16Backend>::bitor(self, a[0], b[0]),
            <archmage::X64V3Token as I16x16Backend>::bitor(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn bitxor(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I16x16Backend>::bitxor(self, a[0], b[0]),
            <archmage::X64V3Token as I16x16Backend>::bitxor(self, a[1], b[1]),
        ]
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl U16x32Backend for archmage::X64V3Token {
    type Repr = [__m256i; 2];

    #[inline(always)]
    fn splat(self, v: u16) -> [__m256i; 2] {
        let h = <archmage::X64V3Token as U16x16Backend>::splat(self, v);
        [h, h]
    }

    #[inline(always)]
    fn zero(self) -> [__m256i; 2] {
        let h = <archmage::X64V3Token as U16x16Backend>::zero(self);
        [h, h]
    }

    #[inline(always)]
    fn load(self, data: &[u16; 32]) -> [__m256i; 2] {
        let (lo, hi) = data.split_at(16);
        [
            <archmage::X64V3Token as U16x16Backend>::load(self, lo.try_into().unwrap()),
            <archmage::X64V3Token as U16x16Backend>::load(self, hi.try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(self, arr: [u16; 32]) -> [__m256i; 2] {
        let mut lo = [0; 16];
        let mut hi = [0; 16];
        lo.copy_from_slice(&arr[..16]);
        hi.copy_from_slice(&arr[16..]);
        [
            <archmage::X64V3Token as U16x16Backend>::from_array(self, lo),
            <archmage::X64V3Token as U16x16Backend>::from_array(self, hi),
        ]
    }

    #[inline(always)]
    fn store(self, repr: [__m256i; 2], out: &mut [u16; 32]) {
        let (lo, hi) = out.split_at_mut(16);
        <archmage::X64V3Token as U16x16Backend>::store(self, repr[0], lo.try_into().unwrap());
        <archmage::X64V3Token as U16x16Backend>::store(self, repr[1], hi.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(self, repr: [__m256i; 2]) -> [u16; 32] {
        let lo = <archmage::X64V3Token as U16x16Backend>::to_array(self, repr[0]);
        let hi = <archmage::X64V3Token as U16x16Backend>::to_array(self, repr[1]);
        let mut out = [0; 32];
        out[..16].copy_from_slice(&lo);
        out[16..].copy_from_slice(&hi);
        out
    }

    #[inline(always)]
    fn add(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U16x16Backend>::add(self, a[0], b[0]),
            <archmage::X64V3Token as U16x16Backend>::add(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn sub(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U16x16Backend>::sub(self, a[0], b[0]),
            <archmage::X64V3Token as U16x16Backend>::sub(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn mul(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U16x16Backend>::mul(self, a[0], b[0]),
            <archmage::X64V3Token as U16x16Backend>::mul(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn neg(self, a: [__m256i; 2]) -> [__m256i; 2] {
        let z = <archmage::X64V3Token as U16x16Backend>::zero(self);
        [
            <archmage::X64V3Token as U16x16Backend>::sub(self, z, a[0]),
            <archmage::X64V3Token as U16x16Backend>::sub(self, z, a[1]),
        ]
    }

    #[inline(always)]
    fn min(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U16x16Backend>::min(self, a[0], b[0]),
            <archmage::X64V3Token as U16x16Backend>::min(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn max(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U16x16Backend>::max(self, a[0], b[0]),
            <archmage::X64V3Token as U16x16Backend>::max(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn reduce_add(self, a: [__m256i; 2]) -> u16 {
        <archmage::X64V3Token as U16x16Backend>::reduce_add(self, a[0]).wrapping_add(
            <archmage::X64V3Token as U16x16Backend>::reduce_add(self, a[1]),
        )
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U16x16Backend>::shl_const::<N>(self, a[0]),
            <archmage::X64V3Token as U16x16Backend>::shl_const::<N>(self, a[1]),
        ]
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U16x16Backend>::shr_logical_const::<N>(self, a[0]),
            <archmage::X64V3Token as U16x16Backend>::shr_logical_const::<N>(self, a[1]),
        ]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U16x16Backend>::shr_logical_const::<N>(self, a[0]),
            <archmage::X64V3Token as U16x16Backend>::shr_logical_const::<N>(self, a[1]),
        ]
    }

    #[inline(always)]
    fn shl_uniform(self, a: [__m256i; 2], count: u32) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as U16x16Backend>::shl_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn shr_logical_uniform(self, a: [__m256i; 2], count: u32) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as U16x16Backend>::shr_logical_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: [__m256i; 2], count: u32) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as U16x16Backend>::shr_logical_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn saturating_add(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as U16x16Backend>::saturating_add(self, a[i], b[i])
        })
    }

    #[inline(always)]
    fn saturating_sub(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as U16x16Backend>::saturating_sub(self, a[i], b[i])
        })
    }

    // AND/OR the halves first, then one reduction — one vector op +
    // one movemask instead of two movemasks + a scalar short-circuit
    // (fearless_simd's polyfill lesson; branchless, fewer ops).
    #[inline(always)]
    fn all_true(self, a: [__m256i; 2]) -> bool {
        let combined = <archmage::X64V3Token as U16x16Backend>::bitand(self, a[0], a[1]);
        <archmage::X64V3Token as U16x16Backend>::all_true(self, combined)
    }

    #[inline(always)]
    fn any_true(self, a: [__m256i; 2]) -> bool {
        let combined = <archmage::X64V3Token as U16x16Backend>::bitor(self, a[0], a[1]);
        <archmage::X64V3Token as U16x16Backend>::any_true(self, combined)
    }

    #[inline(always)]
    fn bitmask(self, a: [__m256i; 2]) -> u64 {
        let lo = <archmage::X64V3Token as U16x16Backend>::bitmask(self, a[0]) as u64;
        let hi = <archmage::X64V3Token as U16x16Backend>::bitmask(self, a[1]) as u64;
        lo | (hi << 16)
    }

    #[inline(always)]
    fn simd_eq(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U16x16Backend>::simd_eq(self, a[0], b[0]),
            <archmage::X64V3Token as U16x16Backend>::simd_eq(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_ne(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U16x16Backend>::simd_ne(self, a[0], b[0]),
            <archmage::X64V3Token as U16x16Backend>::simd_ne(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_lt(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U16x16Backend>::simd_lt(self, a[0], b[0]),
            <archmage::X64V3Token as U16x16Backend>::simd_lt(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_le(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U16x16Backend>::simd_le(self, a[0], b[0]),
            <archmage::X64V3Token as U16x16Backend>::simd_le(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_gt(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U16x16Backend>::simd_gt(self, a[0], b[0]),
            <archmage::X64V3Token as U16x16Backend>::simd_gt(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_ge(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U16x16Backend>::simd_ge(self, a[0], b[0]),
            <archmage::X64V3Token as U16x16Backend>::simd_ge(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn blend(
        self,
        mask: [__m256i; 2],
        if_true: [__m256i; 2],
        if_false: [__m256i; 2],
    ) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U16x16Backend>::blend(self, mask[0], if_true[0], if_false[0]),
            <archmage::X64V3Token as U16x16Backend>::blend(self, mask[1], if_true[1], if_false[1]),
        ]
    }

    #[inline(always)]
    fn not(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U16x16Backend>::not(self, a[0]),
            <archmage::X64V3Token as U16x16Backend>::not(self, a[1]),
        ]
    }

    #[inline(always)]
    fn bitand(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U16x16Backend>::bitand(self, a[0], b[0]),
            <archmage::X64V3Token as U16x16Backend>::bitand(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn bitor(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U16x16Backend>::bitor(self, a[0], b[0]),
            <archmage::X64V3Token as U16x16Backend>::bitor(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn bitxor(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U16x16Backend>::bitxor(self, a[0], b[0]),
            <archmage::X64V3Token as U16x16Backend>::bitxor(self, a[1], b[1]),
        ]
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I32x16Backend for archmage::X64V3Token {
    type Repr = [__m256i; 2];

    #[inline(always)]
    fn splat(self, v: i32) -> [__m256i; 2] {
        let h = <archmage::X64V3Token as I32x8Backend>::splat(self, v);
        [h, h]
    }

    #[inline(always)]
    fn zero(self) -> [__m256i; 2] {
        let h = <archmage::X64V3Token as I32x8Backend>::zero(self);
        [h, h]
    }

    #[inline(always)]
    fn load(self, data: &[i32; 16]) -> [__m256i; 2] {
        let (lo, hi) = data.split_at(8);
        [
            <archmage::X64V3Token as I32x8Backend>::load(self, lo.try_into().unwrap()),
            <archmage::X64V3Token as I32x8Backend>::load(self, hi.try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(self, arr: [i32; 16]) -> [__m256i; 2] {
        let mut lo = [0; 8];
        let mut hi = [0; 8];
        lo.copy_from_slice(&arr[..8]);
        hi.copy_from_slice(&arr[8..]);
        [
            <archmage::X64V3Token as I32x8Backend>::from_array(self, lo),
            <archmage::X64V3Token as I32x8Backend>::from_array(self, hi),
        ]
    }

    #[inline(always)]
    fn store(self, repr: [__m256i; 2], out: &mut [i32; 16]) {
        let (lo, hi) = out.split_at_mut(8);
        <archmage::X64V3Token as I32x8Backend>::store(self, repr[0], lo.try_into().unwrap());
        <archmage::X64V3Token as I32x8Backend>::store(self, repr[1], hi.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(self, repr: [__m256i; 2]) -> [i32; 16] {
        let lo = <archmage::X64V3Token as I32x8Backend>::to_array(self, repr[0]);
        let hi = <archmage::X64V3Token as I32x8Backend>::to_array(self, repr[1]);
        let mut out = [0; 16];
        out[..8].copy_from_slice(&lo);
        out[8..].copy_from_slice(&hi);
        out
    }

    #[inline(always)]
    fn add(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I32x8Backend>::add(self, a[0], b[0]),
            <archmage::X64V3Token as I32x8Backend>::add(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn sub(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I32x8Backend>::sub(self, a[0], b[0]),
            <archmage::X64V3Token as I32x8Backend>::sub(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn mul(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I32x8Backend>::mul(self, a[0], b[0]),
            <archmage::X64V3Token as I32x8Backend>::mul(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn neg(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I32x8Backend>::neg(self, a[0]),
            <archmage::X64V3Token as I32x8Backend>::neg(self, a[1]),
        ]
    }

    #[inline(always)]
    fn min(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I32x8Backend>::min(self, a[0], b[0]),
            <archmage::X64V3Token as I32x8Backend>::min(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn max(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I32x8Backend>::max(self, a[0], b[0]),
            <archmage::X64V3Token as I32x8Backend>::max(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn abs(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I32x8Backend>::abs(self, a[0]),
            <archmage::X64V3Token as I32x8Backend>::abs(self, a[1]),
        ]
    }

    #[inline(always)]
    fn reduce_add(self, a: [__m256i; 2]) -> i32 {
        <archmage::X64V3Token as I32x8Backend>::reduce_add(self, a[0]).wrapping_add(
            <archmage::X64V3Token as I32x8Backend>::reduce_add(self, a[1]),
        )
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I32x8Backend>::shl_const::<N>(self, a[0]),
            <archmage::X64V3Token as I32x8Backend>::shl_const::<N>(self, a[1]),
        ]
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I32x8Backend>::shr_arithmetic_const::<N>(self, a[0]),
            <archmage::X64V3Token as I32x8Backend>::shr_arithmetic_const::<N>(self, a[1]),
        ]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I32x8Backend>::shr_logical_const::<N>(self, a[0]),
            <archmage::X64V3Token as I32x8Backend>::shr_logical_const::<N>(self, a[1]),
        ]
    }

    #[inline(always)]
    fn shl_uniform(self, a: [__m256i; 2], count: u32) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as I32x8Backend>::shl_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn shr_logical_uniform(self, a: [__m256i; 2], count: u32) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as I32x8Backend>::shr_logical_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: [__m256i; 2], count: u32) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as I32x8Backend>::shr_arithmetic_uniform(self, a[i], count)
        })
    }

    // AND/OR the halves first, then one reduction — one vector op +
    // one movemask instead of two movemasks + a scalar short-circuit
    // (fearless_simd's polyfill lesson; branchless, fewer ops).
    #[inline(always)]
    fn all_true(self, a: [__m256i; 2]) -> bool {
        let combined = <archmage::X64V3Token as I32x8Backend>::bitand(self, a[0], a[1]);
        <archmage::X64V3Token as I32x8Backend>::all_true(self, combined)
    }

    #[inline(always)]
    fn any_true(self, a: [__m256i; 2]) -> bool {
        let combined = <archmage::X64V3Token as I32x8Backend>::bitor(self, a[0], a[1]);
        <archmage::X64V3Token as I32x8Backend>::any_true(self, combined)
    }

    #[inline(always)]
    fn bitmask(self, a: [__m256i; 2]) -> u64 {
        let lo = <archmage::X64V3Token as I32x8Backend>::bitmask(self, a[0]) as u64;
        let hi = <archmage::X64V3Token as I32x8Backend>::bitmask(self, a[1]) as u64;
        lo | (hi << 8)
    }

    #[inline(always)]
    fn simd_eq(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I32x8Backend>::simd_eq(self, a[0], b[0]),
            <archmage::X64V3Token as I32x8Backend>::simd_eq(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_ne(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I32x8Backend>::simd_ne(self, a[0], b[0]),
            <archmage::X64V3Token as I32x8Backend>::simd_ne(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_lt(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I32x8Backend>::simd_lt(self, a[0], b[0]),
            <archmage::X64V3Token as I32x8Backend>::simd_lt(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_le(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I32x8Backend>::simd_le(self, a[0], b[0]),
            <archmage::X64V3Token as I32x8Backend>::simd_le(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_gt(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I32x8Backend>::simd_gt(self, a[0], b[0]),
            <archmage::X64V3Token as I32x8Backend>::simd_gt(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_ge(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I32x8Backend>::simd_ge(self, a[0], b[0]),
            <archmage::X64V3Token as I32x8Backend>::simd_ge(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn blend(
        self,
        mask: [__m256i; 2],
        if_true: [__m256i; 2],
        if_false: [__m256i; 2],
    ) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I32x8Backend>::blend(self, mask[0], if_true[0], if_false[0]),
            <archmage::X64V3Token as I32x8Backend>::blend(self, mask[1], if_true[1], if_false[1]),
        ]
    }

    #[inline(always)]
    fn not(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I32x8Backend>::not(self, a[0]),
            <archmage::X64V3Token as I32x8Backend>::not(self, a[1]),
        ]
    }

    #[inline(always)]
    fn bitand(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I32x8Backend>::bitand(self, a[0], b[0]),
            <archmage::X64V3Token as I32x8Backend>::bitand(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn bitor(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I32x8Backend>::bitor(self, a[0], b[0]),
            <archmage::X64V3Token as I32x8Backend>::bitor(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn bitxor(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I32x8Backend>::bitxor(self, a[0], b[0]),
            <archmage::X64V3Token as I32x8Backend>::bitxor(self, a[1], b[1]),
        ]
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl U32x16Backend for archmage::X64V3Token {
    type Repr = [__m256i; 2];

    #[inline(always)]
    fn splat(self, v: u32) -> [__m256i; 2] {
        let h = <archmage::X64V3Token as U32x8Backend>::splat(self, v);
        [h, h]
    }

    #[inline(always)]
    fn zero(self) -> [__m256i; 2] {
        let h = <archmage::X64V3Token as U32x8Backend>::zero(self);
        [h, h]
    }

    #[inline(always)]
    fn load(self, data: &[u32; 16]) -> [__m256i; 2] {
        let (lo, hi) = data.split_at(8);
        [
            <archmage::X64V3Token as U32x8Backend>::load(self, lo.try_into().unwrap()),
            <archmage::X64V3Token as U32x8Backend>::load(self, hi.try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(self, arr: [u32; 16]) -> [__m256i; 2] {
        let mut lo = [0; 8];
        let mut hi = [0; 8];
        lo.copy_from_slice(&arr[..8]);
        hi.copy_from_slice(&arr[8..]);
        [
            <archmage::X64V3Token as U32x8Backend>::from_array(self, lo),
            <archmage::X64V3Token as U32x8Backend>::from_array(self, hi),
        ]
    }

    #[inline(always)]
    fn store(self, repr: [__m256i; 2], out: &mut [u32; 16]) {
        let (lo, hi) = out.split_at_mut(8);
        <archmage::X64V3Token as U32x8Backend>::store(self, repr[0], lo.try_into().unwrap());
        <archmage::X64V3Token as U32x8Backend>::store(self, repr[1], hi.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(self, repr: [__m256i; 2]) -> [u32; 16] {
        let lo = <archmage::X64V3Token as U32x8Backend>::to_array(self, repr[0]);
        let hi = <archmage::X64V3Token as U32x8Backend>::to_array(self, repr[1]);
        let mut out = [0; 16];
        out[..8].copy_from_slice(&lo);
        out[8..].copy_from_slice(&hi);
        out
    }

    #[inline(always)]
    fn add(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U32x8Backend>::add(self, a[0], b[0]),
            <archmage::X64V3Token as U32x8Backend>::add(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn sub(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U32x8Backend>::sub(self, a[0], b[0]),
            <archmage::X64V3Token as U32x8Backend>::sub(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn mul(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U32x8Backend>::mul(self, a[0], b[0]),
            <archmage::X64V3Token as U32x8Backend>::mul(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn neg(self, a: [__m256i; 2]) -> [__m256i; 2] {
        let z = <archmage::X64V3Token as U32x8Backend>::zero(self);
        [
            <archmage::X64V3Token as U32x8Backend>::sub(self, z, a[0]),
            <archmage::X64V3Token as U32x8Backend>::sub(self, z, a[1]),
        ]
    }

    #[inline(always)]
    fn min(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U32x8Backend>::min(self, a[0], b[0]),
            <archmage::X64V3Token as U32x8Backend>::min(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn max(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U32x8Backend>::max(self, a[0], b[0]),
            <archmage::X64V3Token as U32x8Backend>::max(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn reduce_add(self, a: [__m256i; 2]) -> u32 {
        <archmage::X64V3Token as U32x8Backend>::reduce_add(self, a[0]).wrapping_add(
            <archmage::X64V3Token as U32x8Backend>::reduce_add(self, a[1]),
        )
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U32x8Backend>::shl_const::<N>(self, a[0]),
            <archmage::X64V3Token as U32x8Backend>::shl_const::<N>(self, a[1]),
        ]
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U32x8Backend>::shr_logical_const::<N>(self, a[0]),
            <archmage::X64V3Token as U32x8Backend>::shr_logical_const::<N>(self, a[1]),
        ]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U32x8Backend>::shr_logical_const::<N>(self, a[0]),
            <archmage::X64V3Token as U32x8Backend>::shr_logical_const::<N>(self, a[1]),
        ]
    }

    #[inline(always)]
    fn shl_uniform(self, a: [__m256i; 2], count: u32) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as U32x8Backend>::shl_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn shr_logical_uniform(self, a: [__m256i; 2], count: u32) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as U32x8Backend>::shr_logical_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: [__m256i; 2], count: u32) -> [__m256i; 2] {
        core::array::from_fn(|i| {
            <archmage::X64V3Token as U32x8Backend>::shr_logical_uniform(self, a[i], count)
        })
    }

    // AND/OR the halves first, then one reduction — one vector op +
    // one movemask instead of two movemasks + a scalar short-circuit
    // (fearless_simd's polyfill lesson; branchless, fewer ops).
    #[inline(always)]
    fn all_true(self, a: [__m256i; 2]) -> bool {
        let combined = <archmage::X64V3Token as U32x8Backend>::bitand(self, a[0], a[1]);
        <archmage::X64V3Token as U32x8Backend>::all_true(self, combined)
    }

    #[inline(always)]
    fn any_true(self, a: [__m256i; 2]) -> bool {
        let combined = <archmage::X64V3Token as U32x8Backend>::bitor(self, a[0], a[1]);
        <archmage::X64V3Token as U32x8Backend>::any_true(self, combined)
    }

    #[inline(always)]
    fn bitmask(self, a: [__m256i; 2]) -> u64 {
        let lo = <archmage::X64V3Token as U32x8Backend>::bitmask(self, a[0]) as u64;
        let hi = <archmage::X64V3Token as U32x8Backend>::bitmask(self, a[1]) as u64;
        lo | (hi << 8)
    }

    #[inline(always)]
    fn simd_eq(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U32x8Backend>::simd_eq(self, a[0], b[0]),
            <archmage::X64V3Token as U32x8Backend>::simd_eq(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_ne(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U32x8Backend>::simd_ne(self, a[0], b[0]),
            <archmage::X64V3Token as U32x8Backend>::simd_ne(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_lt(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U32x8Backend>::simd_lt(self, a[0], b[0]),
            <archmage::X64V3Token as U32x8Backend>::simd_lt(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_le(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U32x8Backend>::simd_le(self, a[0], b[0]),
            <archmage::X64V3Token as U32x8Backend>::simd_le(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_gt(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U32x8Backend>::simd_gt(self, a[0], b[0]),
            <archmage::X64V3Token as U32x8Backend>::simd_gt(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_ge(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U32x8Backend>::simd_ge(self, a[0], b[0]),
            <archmage::X64V3Token as U32x8Backend>::simd_ge(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn blend(
        self,
        mask: [__m256i; 2],
        if_true: [__m256i; 2],
        if_false: [__m256i; 2],
    ) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U32x8Backend>::blend(self, mask[0], if_true[0], if_false[0]),
            <archmage::X64V3Token as U32x8Backend>::blend(self, mask[1], if_true[1], if_false[1]),
        ]
    }

    #[inline(always)]
    fn not(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U32x8Backend>::not(self, a[0]),
            <archmage::X64V3Token as U32x8Backend>::not(self, a[1]),
        ]
    }

    #[inline(always)]
    fn bitand(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U32x8Backend>::bitand(self, a[0], b[0]),
            <archmage::X64V3Token as U32x8Backend>::bitand(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn bitor(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U32x8Backend>::bitor(self, a[0], b[0]),
            <archmage::X64V3Token as U32x8Backend>::bitor(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn bitxor(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U32x8Backend>::bitxor(self, a[0], b[0]),
            <archmage::X64V3Token as U32x8Backend>::bitxor(self, a[1], b[1]),
        ]
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I64x8Backend for archmage::X64V3Token {
    type Repr = [__m256i; 2];

    #[inline(always)]
    fn splat(self, v: i64) -> [__m256i; 2] {
        let h = <archmage::X64V3Token as I64x4Backend>::splat(self, v);
        [h, h]
    }

    #[inline(always)]
    fn zero(self) -> [__m256i; 2] {
        let h = <archmage::X64V3Token as I64x4Backend>::zero(self);
        [h, h]
    }

    #[inline(always)]
    fn load(self, data: &[i64; 8]) -> [__m256i; 2] {
        let (lo, hi) = data.split_at(4);
        [
            <archmage::X64V3Token as I64x4Backend>::load(self, lo.try_into().unwrap()),
            <archmage::X64V3Token as I64x4Backend>::load(self, hi.try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(self, arr: [i64; 8]) -> [__m256i; 2] {
        let mut lo = [0; 4];
        let mut hi = [0; 4];
        lo.copy_from_slice(&arr[..4]);
        hi.copy_from_slice(&arr[4..]);
        [
            <archmage::X64V3Token as I64x4Backend>::from_array(self, lo),
            <archmage::X64V3Token as I64x4Backend>::from_array(self, hi),
        ]
    }

    #[inline(always)]
    fn store(self, repr: [__m256i; 2], out: &mut [i64; 8]) {
        let (lo, hi) = out.split_at_mut(4);
        <archmage::X64V3Token as I64x4Backend>::store(self, repr[0], lo.try_into().unwrap());
        <archmage::X64V3Token as I64x4Backend>::store(self, repr[1], hi.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(self, repr: [__m256i; 2]) -> [i64; 8] {
        let lo = <archmage::X64V3Token as I64x4Backend>::to_array(self, repr[0]);
        let hi = <archmage::X64V3Token as I64x4Backend>::to_array(self, repr[1]);
        let mut out = [0; 8];
        out[..4].copy_from_slice(&lo);
        out[4..].copy_from_slice(&hi);
        out
    }

    #[inline(always)]
    fn add(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I64x4Backend>::add(self, a[0], b[0]),
            <archmage::X64V3Token as I64x4Backend>::add(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn sub(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I64x4Backend>::sub(self, a[0], b[0]),
            <archmage::X64V3Token as I64x4Backend>::sub(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn neg(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I64x4Backend>::neg(self, a[0]),
            <archmage::X64V3Token as I64x4Backend>::neg(self, a[1]),
        ]
    }

    #[inline(always)]
    fn min(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I64x4Backend>::min(self, a[0], b[0]),
            <archmage::X64V3Token as I64x4Backend>::min(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn max(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I64x4Backend>::max(self, a[0], b[0]),
            <archmage::X64V3Token as I64x4Backend>::max(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn abs(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I64x4Backend>::abs(self, a[0]),
            <archmage::X64V3Token as I64x4Backend>::abs(self, a[1]),
        ]
    }

    #[inline(always)]
    fn reduce_add(self, a: [__m256i; 2]) -> i64 {
        <archmage::X64V3Token as I64x4Backend>::reduce_add(self, a[0]).wrapping_add(
            <archmage::X64V3Token as I64x4Backend>::reduce_add(self, a[1]),
        )
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I64x4Backend>::shl_const::<N>(self, a[0]),
            <archmage::X64V3Token as I64x4Backend>::shl_const::<N>(self, a[1]),
        ]
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I64x4Backend>::shr_arithmetic_const::<N>(self, a[0]),
            <archmage::X64V3Token as I64x4Backend>::shr_arithmetic_const::<N>(self, a[1]),
        ]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I64x4Backend>::shr_logical_const::<N>(self, a[0]),
            <archmage::X64V3Token as I64x4Backend>::shr_logical_const::<N>(self, a[1]),
        ]
    }

    // AND/OR the halves first, then one reduction — one vector op +
    // one movemask instead of two movemasks + a scalar short-circuit
    // (fearless_simd's polyfill lesson; branchless, fewer ops).
    #[inline(always)]
    fn all_true(self, a: [__m256i; 2]) -> bool {
        let combined = <archmage::X64V3Token as I64x4Backend>::bitand(self, a[0], a[1]);
        <archmage::X64V3Token as I64x4Backend>::all_true(self, combined)
    }

    #[inline(always)]
    fn any_true(self, a: [__m256i; 2]) -> bool {
        let combined = <archmage::X64V3Token as I64x4Backend>::bitor(self, a[0], a[1]);
        <archmage::X64V3Token as I64x4Backend>::any_true(self, combined)
    }

    #[inline(always)]
    fn bitmask(self, a: [__m256i; 2]) -> u64 {
        let lo = <archmage::X64V3Token as I64x4Backend>::bitmask(self, a[0]) as u64;
        let hi = <archmage::X64V3Token as I64x4Backend>::bitmask(self, a[1]) as u64;
        lo | (hi << 4)
    }

    #[inline(always)]
    fn simd_eq(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I64x4Backend>::simd_eq(self, a[0], b[0]),
            <archmage::X64V3Token as I64x4Backend>::simd_eq(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_ne(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I64x4Backend>::simd_ne(self, a[0], b[0]),
            <archmage::X64V3Token as I64x4Backend>::simd_ne(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_lt(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I64x4Backend>::simd_lt(self, a[0], b[0]),
            <archmage::X64V3Token as I64x4Backend>::simd_lt(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_le(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I64x4Backend>::simd_le(self, a[0], b[0]),
            <archmage::X64V3Token as I64x4Backend>::simd_le(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_gt(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I64x4Backend>::simd_gt(self, a[0], b[0]),
            <archmage::X64V3Token as I64x4Backend>::simd_gt(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_ge(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I64x4Backend>::simd_ge(self, a[0], b[0]),
            <archmage::X64V3Token as I64x4Backend>::simd_ge(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn blend(
        self,
        mask: [__m256i; 2],
        if_true: [__m256i; 2],
        if_false: [__m256i; 2],
    ) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I64x4Backend>::blend(self, mask[0], if_true[0], if_false[0]),
            <archmage::X64V3Token as I64x4Backend>::blend(self, mask[1], if_true[1], if_false[1]),
        ]
    }

    #[inline(always)]
    fn not(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I64x4Backend>::not(self, a[0]),
            <archmage::X64V3Token as I64x4Backend>::not(self, a[1]),
        ]
    }

    #[inline(always)]
    fn bitand(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I64x4Backend>::bitand(self, a[0], b[0]),
            <archmage::X64V3Token as I64x4Backend>::bitand(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn bitor(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I64x4Backend>::bitor(self, a[0], b[0]),
            <archmage::X64V3Token as I64x4Backend>::bitor(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn bitxor(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as I64x4Backend>::bitxor(self, a[0], b[0]),
            <archmage::X64V3Token as I64x4Backend>::bitxor(self, a[1], b[1]),
        ]
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl U64x8Backend for archmage::X64V3Token {
    type Repr = [__m256i; 2];

    #[inline(always)]
    fn splat(self, v: u64) -> [__m256i; 2] {
        let h = <archmage::X64V3Token as U64x4Backend>::splat(self, v);
        [h, h]
    }

    #[inline(always)]
    fn zero(self) -> [__m256i; 2] {
        let h = <archmage::X64V3Token as U64x4Backend>::zero(self);
        [h, h]
    }

    #[inline(always)]
    fn load(self, data: &[u64; 8]) -> [__m256i; 2] {
        let (lo, hi) = data.split_at(4);
        [
            <archmage::X64V3Token as U64x4Backend>::load(self, lo.try_into().unwrap()),
            <archmage::X64V3Token as U64x4Backend>::load(self, hi.try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(self, arr: [u64; 8]) -> [__m256i; 2] {
        let mut lo = [0; 4];
        let mut hi = [0; 4];
        lo.copy_from_slice(&arr[..4]);
        hi.copy_from_slice(&arr[4..]);
        [
            <archmage::X64V3Token as U64x4Backend>::from_array(self, lo),
            <archmage::X64V3Token as U64x4Backend>::from_array(self, hi),
        ]
    }

    #[inline(always)]
    fn store(self, repr: [__m256i; 2], out: &mut [u64; 8]) {
        let (lo, hi) = out.split_at_mut(4);
        <archmage::X64V3Token as U64x4Backend>::store(self, repr[0], lo.try_into().unwrap());
        <archmage::X64V3Token as U64x4Backend>::store(self, repr[1], hi.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(self, repr: [__m256i; 2]) -> [u64; 8] {
        let lo = <archmage::X64V3Token as U64x4Backend>::to_array(self, repr[0]);
        let hi = <archmage::X64V3Token as U64x4Backend>::to_array(self, repr[1]);
        let mut out = [0; 8];
        out[..4].copy_from_slice(&lo);
        out[4..].copy_from_slice(&hi);
        out
    }

    #[inline(always)]
    fn add(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U64x4Backend>::add(self, a[0], b[0]),
            <archmage::X64V3Token as U64x4Backend>::add(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn sub(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U64x4Backend>::sub(self, a[0], b[0]),
            <archmage::X64V3Token as U64x4Backend>::sub(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn neg(self, a: [__m256i; 2]) -> [__m256i; 2] {
        let z = <archmage::X64V3Token as U64x4Backend>::zero(self);
        [
            <archmage::X64V3Token as U64x4Backend>::sub(self, z, a[0]),
            <archmage::X64V3Token as U64x4Backend>::sub(self, z, a[1]),
        ]
    }

    #[inline(always)]
    fn min(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U64x4Backend>::min(self, a[0], b[0]),
            <archmage::X64V3Token as U64x4Backend>::min(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn max(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U64x4Backend>::max(self, a[0], b[0]),
            <archmage::X64V3Token as U64x4Backend>::max(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn reduce_add(self, a: [__m256i; 2]) -> u64 {
        <archmage::X64V3Token as U64x4Backend>::reduce_add(self, a[0]).wrapping_add(
            <archmage::X64V3Token as U64x4Backend>::reduce_add(self, a[1]),
        )
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U64x4Backend>::shl_const::<N>(self, a[0]),
            <archmage::X64V3Token as U64x4Backend>::shl_const::<N>(self, a[1]),
        ]
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U64x4Backend>::shr_logical_const::<N>(self, a[0]),
            <archmage::X64V3Token as U64x4Backend>::shr_logical_const::<N>(self, a[1]),
        ]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U64x4Backend>::shr_logical_const::<N>(self, a[0]),
            <archmage::X64V3Token as U64x4Backend>::shr_logical_const::<N>(self, a[1]),
        ]
    }

    // AND/OR the halves first, then one reduction — one vector op +
    // one movemask instead of two movemasks + a scalar short-circuit
    // (fearless_simd's polyfill lesson; branchless, fewer ops).
    #[inline(always)]
    fn all_true(self, a: [__m256i; 2]) -> bool {
        let combined = <archmage::X64V3Token as U64x4Backend>::bitand(self, a[0], a[1]);
        <archmage::X64V3Token as U64x4Backend>::all_true(self, combined)
    }

    #[inline(always)]
    fn any_true(self, a: [__m256i; 2]) -> bool {
        let combined = <archmage::X64V3Token as U64x4Backend>::bitor(self, a[0], a[1]);
        <archmage::X64V3Token as U64x4Backend>::any_true(self, combined)
    }

    #[inline(always)]
    fn bitmask(self, a: [__m256i; 2]) -> u64 {
        let lo = <archmage::X64V3Token as U64x4Backend>::bitmask(self, a[0]) as u64;
        let hi = <archmage::X64V3Token as U64x4Backend>::bitmask(self, a[1]) as u64;
        lo | (hi << 4)
    }

    #[inline(always)]
    fn simd_eq(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U64x4Backend>::simd_eq(self, a[0], b[0]),
            <archmage::X64V3Token as U64x4Backend>::simd_eq(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_ne(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U64x4Backend>::simd_ne(self, a[0], b[0]),
            <archmage::X64V3Token as U64x4Backend>::simd_ne(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_lt(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U64x4Backend>::simd_lt(self, a[0], b[0]),
            <archmage::X64V3Token as U64x4Backend>::simd_lt(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_le(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U64x4Backend>::simd_le(self, a[0], b[0]),
            <archmage::X64V3Token as U64x4Backend>::simd_le(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_gt(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U64x4Backend>::simd_gt(self, a[0], b[0]),
            <archmage::X64V3Token as U64x4Backend>::simd_gt(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_ge(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U64x4Backend>::simd_ge(self, a[0], b[0]),
            <archmage::X64V3Token as U64x4Backend>::simd_ge(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn blend(
        self,
        mask: [__m256i; 2],
        if_true: [__m256i; 2],
        if_false: [__m256i; 2],
    ) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U64x4Backend>::blend(self, mask[0], if_true[0], if_false[0]),
            <archmage::X64V3Token as U64x4Backend>::blend(self, mask[1], if_true[1], if_false[1]),
        ]
    }

    #[inline(always)]
    fn not(self, a: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U64x4Backend>::not(self, a[0]),
            <archmage::X64V3Token as U64x4Backend>::not(self, a[1]),
        ]
    }

    #[inline(always)]
    fn bitand(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U64x4Backend>::bitand(self, a[0], b[0]),
            <archmage::X64V3Token as U64x4Backend>::bitand(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn bitor(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U64x4Backend>::bitor(self, a[0], b[0]),
            <archmage::X64V3Token as U64x4Backend>::bitor(self, a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn bitxor(self, a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        [
            <archmage::X64V3Token as U64x4Backend>::bitxor(self, a[0], b[0]),
            <archmage::X64V3Token as U64x4Backend>::bitxor(self, a[1], b[1]),
        ]
    }
}

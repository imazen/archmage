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
//! # Safety (audit contract — checked backend boundaries)
//!
//! `#[arcane]` checks value intrinsics against the receiver token's features.
//! SSE2-only arithmetic runs in `#[arcane]` regions for `X64V1Token`, the
//! x86-64 baseline, so it inlines into callers without AVX.
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
impl F32x16Backend for X64V4Token {
    type Repr = __m512;

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn splat(self, v: f32) -> __m512 {
        _mm512_set1_ps(v)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn zero(self) -> __m512 {
        _mm512_setzero_ps()
    }

    #[inline(always)]
    fn load(self, data: &[f32; 16]) -> __m512 {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [f32; 16]) -> __m512 {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m512, out: &mut [f32; 16]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m512) -> [f32; 16] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn add(self, a: __m512, b: __m512) -> __m512 {
        _mm512_add_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn sub(self, a: __m512, b: __m512) -> __m512 {
        _mm512_sub_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn mul(self, a: __m512, b: __m512) -> __m512 {
        _mm512_mul_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn div(self, a: __m512, b: __m512) -> __m512 {
        _mm512_div_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn neg(self, a: __m512) -> __m512 {
        _mm512_xor_ps(a, _mm512_set1_ps(-0.0))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn min(self, a: __m512, b: __m512) -> __m512 {
        _mm512_range_ps::<0x04>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn max(self, a: __m512, b: __m512) -> __m512 {
        _mm512_range_ps::<0x05>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn minimum(self, a: __m512, b: __m512) -> __m512 {
        _mm512_mask_blend_ps(
            _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(a, b),
            _mm512_or_ps(_mm512_min_ps(a, b), _mm512_min_ps(b, a)),
            _mm512_set1_ps(f32::NAN),
        )
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn maximum(self, a: __m512, b: __m512) -> __m512 {
        _mm512_mask_blend_ps(
            _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(a, b),
            _mm512_and_ps(_mm512_max_ps(a, b), _mm512_max_ps(b, a)),
            _mm512_set1_ps(f32::NAN),
        )
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn min_fast(self, a: __m512, b: __m512) -> __m512 {
        _mm512_min_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn max_fast(self, a: __m512, b: __m512) -> __m512 {
        _mm512_max_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn sqrt(self, a: __m512) -> __m512 {
        _mm512_sqrt_ps(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn abs(self, a: __m512) -> __m512 {
        let mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFF_FFFFu32 as i32));
        _mm512_and_ps(a, mask)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn trunc(self, a: __m512) -> __m512 {
        _mm512_roundscale_ps::<0x03>(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn mul_neg_add_relaxed(self, a: __m512, b: __m512, c: __m512) -> __m512 {
        _mm512_fnmadd_ps(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn mul_neg_add_fused(self, a: __m512, b: __m512, c: __m512) -> __m512 {
        _mm512_fnmadd_ps(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn floor(self, a: __m512) -> __m512 {
        _mm512_roundscale_ps::<0x01>(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn ceil(self, a: __m512) -> __m512 {
        _mm512_roundscale_ps::<0x02>(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn round(self, a: __m512) -> __m512 {
        _mm512_roundscale_ps::<0x00>(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn mul_add(self, a: __m512, b: __m512, c: __m512) -> __m512 {
        _mm512_fmadd_ps(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn mul_sub(self, a: __m512, b: __m512, c: __m512) -> __m512 {
        _mm512_fmsub_ps(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn mul_add_portable(self, a: __m512, b: __m512, c: __m512) -> __m512 {
        _mm512_fmadd_ps(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn mul_sub_portable(self, a: __m512, b: __m512, c: __m512) -> __m512 {
        _mm512_fmsub_ps(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_eq(self, a: __m512, b: __m512) -> __m512 {
        let mask = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(a, b);
        _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_ne(self, a: __m512, b: __m512) -> __m512 {
        let mask = _mm512_cmp_ps_mask::<_CMP_NEQ_UQ>(a, b);
        _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_lt(self, a: __m512, b: __m512) -> __m512 {
        let mask = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(a, b);
        _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_le(self, a: __m512, b: __m512) -> __m512 {
        let mask = _mm512_cmp_ps_mask::<_CMP_LE_OQ>(a, b);
        _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_gt(self, a: __m512, b: __m512) -> __m512 {
        let mask = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(a, b);
        _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_ge(self, a: __m512, b: __m512) -> __m512 {
        let mask = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(a, b);
        _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn blend(self, mask: __m512, if_true: __m512, if_false: __m512) -> __m512 {
        let mask_i = _mm512_castps_si512(mask);
        let k = _mm512_cmpneq_epi32_mask(mask_i, _mm512_setzero_si512());
        _mm512_mask_blend_ps(k, if_false, if_true)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn reduce_add(self, a: __m512) -> f32 {
        _mm512_reduce_add_ps(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn reduce_min(self, a: __m512) -> f32 {
        _mm512_reduce_min_ps(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn reduce_max(self, a: __m512) -> f32 {
        _mm512_reduce_max_ps(a)
    }

    // Raw 14-bit hardware estimate — the cheapest >=12-bit path on AVX-512.
    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn rcp_approx(self, a: __m512) -> __m512 {
        _mm512_rcp14_ps(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn rsqrt_approx(self, a: __m512) -> __m512 {
        _mm512_rsqrt14_ps(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn from_u16_lanes(self, lanes: &[u16; 16]) -> __m512 {
        _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(crate::simd_storage::cast(*lanes)))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn to_u16_lanes<const CFG: u32>(self, value: __m512) -> [u16; 16] {
        let opts = const { crate::numerics::IntConvert::decode(CFG) };
        let value = if opts.saturates_all() {
            _mm512_min_ps(_mm512_set1_ps(65535.0), value)
        } else {
            value
        };
        if opts.nonnegative() {
            let i = if opts.rounds() {
                _mm512_cvtps_epu32(value)
            } else {
                _mm512_cvttps_epu32(value)
            };
            return crate::simd_storage::cast(_mm512_cvtusepi32_epi16(i));
        }

        let i = if opts.rounds() {
            _mm512_cvtps_epi32(value)
        } else {
            _mm512_cvttps_epi32(value)
        };
        crate::simd_storage::cast(_mm512_cvtusepi32_epi16(_mm512_max_epi32(
            i,
            _mm512_setzero_si512(),
        )))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn to_u8_lanes<const CFG: u32>(self, value: __m512) -> [u8; 16] {
        let opts = const { crate::numerics::IntConvert::decode(CFG) };
        let value = if opts.saturates_all() {
            _mm512_min_ps(_mm512_set1_ps(255.0), value)
        } else {
            value
        };
        if opts.nonnegative() {
            let i = if opts.rounds() {
                _mm512_cvtps_epu32(value)
            } else {
                _mm512_cvttps_epu32(value)
            };
            return crate::simd_storage::cast(_mm512_cvtusepi32_epi8(i));
        }
        let i = if opts.rounds() {
            _mm512_cvtps_epi32(value)
        } else {
            _mm512_cvttps_epi32(value)
        };
        crate::simd_storage::cast(_mm512_cvtusepi32_epi8(_mm512_max_epi32(
            i,
            _mm512_setzero_si512(),
        )))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn to_u32_lanes<const CFG: u32>(self, value: __m512) -> __m512 {
        let opts = const { crate::numerics::IntConvert::decode(CFG) };
        let clamped = if opts.nonnegative() {
            value
        } else {
            _mm512_max_ps(value, _mm512_setzero_ps())
        };
        let i = if opts.rounds() {
            _mm512_cvtps_epu32(clamped)
        } else {
            _mm512_cvttps_epu32(clamped)
        };
        _mm512_castsi512_ps(i)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn from_u32_lanes(self, value: __m512) -> __m512 {
        _mm512_cvtepu32_ps(_mm512_castps_si512(value))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn from_f16_lanes<const CFG: u32>(self, lanes: &[u16; 16]) -> __m512 {
        let opts = const { crate::numerics::HalfConvert::decode(CFG) };
        let h = crate::simd_storage::cast(*lanes);
        let h = if opts.flushes() {
            let mag = _mm256_and_si256(h, _mm256_set1_epi16(0x7fff));
            let small = _mm256_cmpgt_epi16(_mm256_set1_epi16(0x400), mag);
            _mm256_andnot_si256(_mm256_and_si256(small, _mm256_set1_epi16(0x7fff)), h)
        } else {
            h
        };
        _mm512_cvtph_ps(h)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn to_f16_lanes<const CFG: u32>(self, value: __m512) -> [u16; 16] {
        let opts = const { crate::numerics::HalfConvert::decode(CFG) };
        let h = if opts.rounds() {
            _mm512_cvtps_ph::<0>(value)
        } else {
            _mm512_cvtps_ph::<3>(value)
        };
        let h = if opts.flushes() {
            let mag = _mm256_and_si256(h, _mm256_set1_epi16(0x7fff));
            let small = _mm256_cmpgt_epi16(_mm256_set1_epi16(0x400), mag);
            _mm256_andnot_si256(_mm256_and_si256(small, _mm256_set1_epi16(0x7fff)), h)
        } else {
            h
        };
        crate::simd_storage::cast(h)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn from_bf16_lanes(self, lanes: &[u16; 16]) -> __m512 {
        _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(
            crate::simd_storage::cast(*lanes),
        )))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn to_bf16_lanes<const CFG: u32>(self, value: __m512) -> [u16; 16] {
        let opts = const { crate::numerics::HalfConvert::decode(CFG) };
        let raw = _mm512_castps_si512(value);
        let rounded = if opts.rounds() {
            let lsb = _mm512_and_si512(_mm512_srli_epi32::<16>(raw), _mm512_set1_epi32(1));
            _mm512_add_epi32(raw, _mm512_add_epi32(lsb, _mm512_set1_epi32(0x7fff)))
        } else {
            raw
        };
        let i = _mm512_srli_epi32::<16>(rounded);
        let out: [u16; 16] = crate::simd_storage::cast(_mm512_cvtusepi32_epi16(_mm512_max_epi32(
            i,
            _mm512_setzero_si512(),
        )));
        if opts.is_finite() && !opts.flushes() {
            out
        } else {
            // Special and signed-subnormal handling is shared with scalar.
            let original: [f32; 16] = crate::simd_storage::cast(value);
            core::array::from_fn(|i| {
                if original[i].is_nan() || (opts.flushes() && out[i] & 0x7f80 == 0) {
                    crate::scalar::to_bf16_with::<CFG>(original[i])
                } else {
                    out[i]
                }
            })
        }
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn from_u8_lanes_wide(self, lanes: &[u8; 16]) -> __m512 {
        _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(crate::simd_storage::cast(*lanes)))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn deinterleave_rgba_wide(self, rgba: [__m512; 4]) -> [__m512; 4] {
        let [a, b, c, d] = rgba;
        let rg = _mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29);
        let ba = _mm512_add_epi32(rg, _mm512_set1_epi32(2));
        let low = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
        let high = _mm512_add_epi32(low, _mm512_set1_epi32(8));
        let ab_rg = _mm512_permutex2var_ps(a, rg, b);
        let cd_rg = _mm512_permutex2var_ps(c, rg, d);
        let ab_ba = _mm512_permutex2var_ps(a, ba, b);
        let cd_ba = _mm512_permutex2var_ps(c, ba, d);
        [
            _mm512_permutex2var_ps(ab_rg, low, cd_rg),
            _mm512_permutex2var_ps(ab_rg, high, cd_rg),
            _mm512_permutex2var_ps(ab_ba, low, cd_ba),
            _mm512_permutex2var_ps(ab_ba, high, cd_ba),
        ]
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn interleave_rgba_wide(self, channels: [__m512; 4]) -> [__m512; 4] {
        let [r, g, b, a] = channels;
        let low = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
        let high = _mm512_add_epi32(low, _mm512_set1_epi32(8));
        let first = _mm512_setr_epi32(0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23);
        let second = _mm512_add_epi32(first, _mm512_set1_epi32(8));
        let rg0 = _mm512_permutex2var_ps(r, low, g);
        let ba0 = _mm512_permutex2var_ps(b, low, a);
        let rg1 = _mm512_permutex2var_ps(r, high, g);
        let ba1 = _mm512_permutex2var_ps(b, high, a);
        [
            _mm512_permutex2var_ps(rg0, first, ba0),
            _mm512_permutex2var_ps(rg0, second, ba0),
            _mm512_permutex2var_ps(rg1, first, ba1),
            _mm512_permutex2var_ps(rg1, second, ba1),
        ]
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn broadcast_rgba_alpha_repr(self, value: __m512) -> __m512 {
        _mm512_permute_ps::<255>(value)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn with_rgba_alpha_repr(self, value: __m512, src: __m512) -> __m512 {
        _mm512_mask_blend_ps(0x8888, value, src)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
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

    #[arcane(suppress_const_test, _self = X64V4Token)]
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

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn not(self, a: __m512) -> __m512 {
        let all_ones = _mm512_castsi512_ps(_mm512_set1_epi32(-1));
        _mm512_xor_ps(a, all_ones)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitand(self, a: __m512, b: __m512) -> __m512 {
        _mm512_and_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitor(self, a: __m512, b: __m512) -> __m512 {
        _mm512_or_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitxor(self, a: __m512, b: __m512) -> __m512 {
        _mm512_xor_ps(a, b)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl F64x8Backend for X64V4Token {
    type Repr = __m512d;

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn splat(self, v: f64) -> __m512d {
        _mm512_set1_pd(v)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn zero(self) -> __m512d {
        _mm512_setzero_pd()
    }

    #[inline(always)]
    fn load(self, data: &[f64; 8]) -> __m512d {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [f64; 8]) -> __m512d {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m512d, out: &mut [f64; 8]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m512d) -> [f64; 8] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn add(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_add_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn sub(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_sub_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn mul(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_mul_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn div(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_div_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn neg(self, a: __m512d) -> __m512d {
        _mm512_xor_pd(a, _mm512_set1_pd(-0.0))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn min(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_mask_blend_pd(
            _mm512_cmp_pd_mask::<_CMP_UNORD_Q>(b, b),
            _mm512_min_pd(a, b),
            a,
        )
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn max(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_mask_blend_pd(
            _mm512_cmp_pd_mask::<_CMP_UNORD_Q>(b, b),
            _mm512_max_pd(a, b),
            a,
        )
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn minimum(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_mask_blend_pd(
            _mm512_cmp_pd_mask::<_CMP_UNORD_Q>(a, b),
            _mm512_or_pd(_mm512_min_pd(a, b), _mm512_min_pd(b, a)),
            _mm512_set1_pd(f64::NAN),
        )
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn maximum(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_mask_blend_pd(
            _mm512_cmp_pd_mask::<_CMP_UNORD_Q>(a, b),
            _mm512_and_pd(_mm512_max_pd(a, b), _mm512_max_pd(b, a)),
            _mm512_set1_pd(f64::NAN),
        )
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn min_fast(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_min_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn max_fast(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_max_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn sqrt(self, a: __m512d) -> __m512d {
        _mm512_sqrt_pd(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn abs(self, a: __m512d) -> __m512d {
        let mask = _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFF_FFFF_FFFF_FFFFu64 as i64));
        _mm512_and_pd(a, mask)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn trunc(self, a: __m512d) -> __m512d {
        _mm512_roundscale_pd::<0x03>(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn mul_neg_add_relaxed(self, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        _mm512_fnmadd_pd(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn mul_neg_add_fused(self, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        _mm512_fnmadd_pd(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn floor(self, a: __m512d) -> __m512d {
        _mm512_roundscale_pd::<0x01>(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn ceil(self, a: __m512d) -> __m512d {
        _mm512_roundscale_pd::<0x02>(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn round(self, a: __m512d) -> __m512d {
        _mm512_roundscale_pd::<0x00>(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn mul_add(self, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        _mm512_fmadd_pd(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn mul_sub(self, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        _mm512_fmsub_pd(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn mul_add_portable(self, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        _mm512_fmadd_pd(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn mul_sub_portable(self, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        _mm512_fmsub_pd(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_eq(self, a: __m512d, b: __m512d) -> __m512d {
        let mask = _mm512_cmp_pd_mask::<_CMP_EQ_OQ>(a, b);
        _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_ne(self, a: __m512d, b: __m512d) -> __m512d {
        let mask = _mm512_cmp_pd_mask::<_CMP_NEQ_UQ>(a, b);
        _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_lt(self, a: __m512d, b: __m512d) -> __m512d {
        let mask = _mm512_cmp_pd_mask::<_CMP_LT_OQ>(a, b);
        _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_le(self, a: __m512d, b: __m512d) -> __m512d {
        let mask = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(a, b);
        _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_gt(self, a: __m512d, b: __m512d) -> __m512d {
        let mask = _mm512_cmp_pd_mask::<_CMP_GT_OQ>(a, b);
        _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_ge(self, a: __m512d, b: __m512d) -> __m512d {
        let mask = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(a, b);
        _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn blend(self, mask: __m512d, if_true: __m512d, if_false: __m512d) -> __m512d {
        let mask_i = _mm512_castpd_si512(mask);
        let k = _mm512_cmpneq_epi64_mask(mask_i, _mm512_setzero_si512());
        _mm512_mask_blend_pd(k, if_false, if_true)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn reduce_add(self, a: __m512d) -> f64 {
        _mm512_reduce_add_pd(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn reduce_min(self, a: __m512d) -> f64 {
        _mm512_reduce_min_pd(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn reduce_max(self, a: __m512d) -> f64 {
        _mm512_reduce_max_pd(a)
    }

    // Raw 14-bit hardware estimate — the cheapest >=12-bit path on AVX-512.
    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn rcp_approx(self, a: __m512d) -> __m512d {
        _mm512_rcp14_pd(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn rsqrt_approx(self, a: __m512d) -> __m512d {
        _mm512_rsqrt14_pd(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn recip(self, a: __m512d) -> __m512d {
        _mm512_div_pd(_mm512_set1_pd(1.0), a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn rsqrt(self, a: __m512d) -> __m512d {
        _mm512_div_pd(_mm512_set1_pd(1.0), _mm512_sqrt_pd(a))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn not(self, a: __m512d) -> __m512d {
        let all_ones = _mm512_castsi512_pd(_mm512_set1_epi64(-1));
        _mm512_xor_pd(a, all_ones)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitand(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_and_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitor(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_or_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitxor(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_xor_pd(a, b)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I8x64Backend for X64V4Token {
    type Repr = __m512i;

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn splat(self, v: i8) -> __m512i {
        _mm512_set1_epi8(v)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    fn load(self, data: &[i8; 64]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i8; 64]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m512i, out: &mut [i8; 64]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m512i) -> [i8; 64] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi8(_mm512_setzero_si512(), a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn abs(self, a: __m512i) -> __m512i {
        _mm512_abs_epi8(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epi8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epi8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epi8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epi8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epi8_mask(b, a);
        _mm512_maskz_set1_epi8(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epi8_mask(b, a);
        _mm512_maskz_set1_epi8(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi8_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi8(k, if_false, if_true)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn reduce_add(self, a: __m512i) -> i8 {
        // AVX-512 has no single-instruction integer reduction; fold the lanes
        let arr: [i8; 64] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0i8, i8::wrapping_add)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi8(-1))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        let count = _mm_cvtsi32_si128(N);
        let shifted = _mm512_sll_epi16(a, count);
        let mask = _mm512_set1_epi8(((0xFFu16 << N) & 0xFF) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
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

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        let count = _mm_cvtsi32_si128(N);
        let shifted = _mm512_srl_epi16(a, count);
        let mask = _mm512_set1_epi8(((0xFFu16 >> N) & 0xFF) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn saturating_add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_adds_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn saturating_sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_subs_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn all_true(self, a: __m512i) -> bool {
        // Sign-bit contract: `movepi*_mask` reads the sign bits
        // directly. The `cmpneq` form this replaces answered a
        // different question (lane nonzero) and disagreed with the
        // V3 movemask backends.
        _mm512_movepi8_mask(a) == 0xFFFF_FFFF_FFFF_FFFFu64
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn any_true(self, a: __m512i) -> bool {
        _mm512_movepi8_mask(a) != 0
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitmask(self, a: __m512i) -> u64 {
        _mm512_movepi8_mask(a)
    }
    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn widen_low_i8_to_i16(self, a: __m512i) -> __m512i {
        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(a))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn widen_high_i8_to_i16(self, a: __m512i) -> __m512i {
        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64::<1>(a))
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl U8x64Backend for X64V4Token {
    type Repr = __m512i;

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn splat(self, v: u8) -> __m512i {
        _mm512_set1_epi8(v as _)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    fn load(self, data: &[u8; 64]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u8; 64]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m512i, out: &mut [u8; 64]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m512i) -> [u8; 64] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi8(_mm512_setzero_si512(), a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epu8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epu8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epu8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epu8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epu8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epu8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epu8_mask(b, a);
        _mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epu8_mask(b, a);
        _mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi8_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi8(k, if_false, if_true)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn reduce_add(self, a: __m512i) -> u8 {
        // AVX-512 has no single-instruction integer reduction; fold the lanes
        let arr: [u8; 64] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0u8, u8::wrapping_add)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi8(-1i8))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        let count = _mm_cvtsi32_si128(N);
        let shifted = _mm512_sll_epi16(a, count);
        let mask = _mm512_set1_epi8(((0xFFu16 << N) & 0xFF) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
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

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        let count = _mm_cvtsi32_si128(N);
        let shifted = _mm512_srl_epi16(a, count);
        let mask = _mm512_set1_epi8(((0xFFu16 >> N) & 0xFF) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn saturating_add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_adds_epu8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn saturating_sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_subs_epu8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn all_true(self, a: __m512i) -> bool {
        // Sign-bit contract: `movepi*_mask` reads the sign bits
        // directly. The `cmpneq` form this replaces answered a
        // different question (lane nonzero) and disagreed with the
        // V3 movemask backends.
        _mm512_movepi8_mask(a) == 0xFFFF_FFFF_FFFF_FFFFu64
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn any_true(self, a: __m512i) -> bool {
        _mm512_movepi8_mask(a) != 0
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitmask(self, a: __m512i) -> u64 {
        _mm512_movepi8_mask(a)
    }
    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn widen_low_u8_to_u16(self, a: __m512i) -> __m512i {
        _mm512_cvtepu8_epi16(_mm512_castsi512_si256(a))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn widen_high_u8_to_u16(self, a: __m512i) -> __m512i {
        _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(a))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn abs_diff(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi8(_mm512_max_epu8(a, b), _mm512_min_epu8(a, b))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn reduce_add_u32(self, a: __m512i) -> u32 {
        {
            let s = _mm512_sad_epu8(a, _mm512_setzero_si512());
            _mm512_reduce_add_epi64(s) as u32
        }
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn sum_abs_diff(self, a: __m512i, b: __m512i) -> u32 {
        {
            let s = _mm512_sad_epu8(a, b);
            _mm512_reduce_add_epi64(s) as u32
        }
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn pairwise_widen_add(self, a: __m512i) -> __m512i {
        _mm512_add_epi16(
            _mm512_and_si512(a, _mm512_set1_epi16(255)),
            _mm512_srli_epi16::<8>(a),
        )
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I16x32Backend for X64V4Token {
    type Repr = __m512i;

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn splat(self, v: i16) -> __m512i {
        _mm512_set1_epi16(v)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    fn load(self, data: &[i16; 32]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i16; 32]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m512i, out: &mut [i16; 32]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m512i) -> [i16; 32] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn mul(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_mullo_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi16(_mm512_setzero_si512(), a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn abs(self, a: __m512i) -> __m512i {
        _mm512_abs_epi16(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epi16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epi16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epi16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epi16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epi16_mask(b, a);
        _mm512_maskz_set1_epi16(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epi16_mask(b, a);
        _mm512_maskz_set1_epi16(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi16_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi16(k, if_false, if_true)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn reduce_add(self, a: __m512i) -> i16 {
        // AVX-512 has no single-instruction integer reduction; fold the lanes
        let arr: [i16; 32] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0i16, i16::wrapping_add)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi16(-1))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sll_epi16(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sra_epi16(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi16(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shl_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_sll_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shr_logical_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_srl_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shr_arithmetic_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_sra_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn saturating_add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_adds_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn saturating_sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_subs_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn all_true(self, a: __m512i) -> bool {
        // Sign-bit contract: `movepi*_mask` reads the sign bits
        // directly. The `cmpneq` form this replaces answered a
        // different question (lane nonzero) and disagreed with the
        // V3 movemask backends.
        _mm512_movepi16_mask(a) as u64 == 0xFFFF_FFFFu64
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn any_true(self, a: __m512i) -> bool {
        _mm512_movepi16_mask(a) as u64 != 0
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitmask(self, a: __m512i) -> u64 {
        _mm512_movepi16_mask(a) as u64
    }
    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn widen_low_i16_to_i32(self, a: __m512i) -> __m512i {
        _mm512_cvtepi16_epi32(_mm512_castsi512_si256(a))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn widen_high_i16_to_i32(self, a: __m512i) -> __m512i {
        _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(a))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn narrow_saturating_i16_to_i8(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_inserti64x4::<1>(
            _mm512_castsi256_si512(_mm512_cvtsepi16_epi8(a)),
            _mm512_cvtsepi16_epi8(b),
        )
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
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

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn madd_adjacent(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_madd_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn abs_diff(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi16(_mm512_max_epi16(a, b), _mm512_min_epi16(a, b))
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl U16x32Backend for X64V4Token {
    type Repr = __m512i;

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn splat(self, v: u16) -> __m512i {
        _mm512_set1_epi16(v as _)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    fn load(self, data: &[u16; 32]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u16; 32]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m512i, out: &mut [u16; 32]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m512i) -> [u16; 32] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn mul(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_mullo_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi16(_mm512_setzero_si512(), a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epu16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epu16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epu16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epu16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epu16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epu16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epu16_mask(b, a);
        _mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epu16_mask(b, a);
        _mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi16_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi16(k, if_false, if_true)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn reduce_add(self, a: __m512i) -> u16 {
        // AVX-512 has no single-instruction integer reduction; fold the lanes
        let arr: [u16; 32] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0u16, u16::wrapping_add)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi16(-1i16))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sll_epi16(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi16(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi16(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shl_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_sll_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shr_logical_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_srl_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shr_arithmetic_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_srl_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn saturating_add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_adds_epu16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn saturating_sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_subs_epu16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn all_true(self, a: __m512i) -> bool {
        // Sign-bit contract: `movepi*_mask` reads the sign bits
        // directly. The `cmpneq` form this replaces answered a
        // different question (lane nonzero) and disagreed with the
        // V3 movemask backends.
        _mm512_movepi16_mask(a) as u64 == 0xFFFF_FFFFu64
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn any_true(self, a: __m512i) -> bool {
        _mm512_movepi16_mask(a) as u64 != 0
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitmask(self, a: __m512i) -> u64 {
        _mm512_movepi16_mask(a) as u64
    }
    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn widen_low_u16_to_u32(self, a: __m512i) -> __m512i {
        _mm512_cvtepu16_epi32(_mm512_castsi512_si256(a))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn widen_high_u16_to_u32(self, a: __m512i) -> __m512i {
        _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64::<1>(a))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn pairwise_widen_add(self, a: __m512i) -> __m512i {
        _mm512_add_epi32(
            _mm512_and_si512(a, _mm512_set1_epi32(65535)),
            _mm512_srli_epi32::<16>(a),
        )
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I32x16Backend for X64V4Token {
    type Repr = __m512i;

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn splat(self, v: i32) -> __m512i {
        _mm512_set1_epi32(v)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    fn load(self, data: &[i32; 16]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i32; 16]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m512i, out: &mut [i32; 16]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m512i) -> [i32; 16] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn mul(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_mullo_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi32(_mm512_setzero_si512(), a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn abs(self, a: __m512i) -> __m512i {
        _mm512_abs_epi32(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epi32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epi32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epi32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epi32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epi32_mask(b, a);
        _mm512_maskz_set1_epi32(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epi32_mask(b, a);
        _mm512_maskz_set1_epi32(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi32_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi32(k, if_false, if_true)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn reduce_add(self, a: __m512i) -> i32 {
        // AVX-512 has no single-instruction integer reduction; fold the lanes
        let arr: [i32; 16] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0i32, i32::wrapping_add)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi32(-1))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        match N {
            0 => _mm512_slli_epi32::<0>(a),
            1 => _mm512_slli_epi32::<1>(a),
            2 => _mm512_slli_epi32::<2>(a),
            3 => _mm512_slli_epi32::<3>(a),
            4 => _mm512_slli_epi32::<4>(a),
            5 => _mm512_slli_epi32::<5>(a),
            6 => _mm512_slli_epi32::<6>(a),
            7 => _mm512_slli_epi32::<7>(a),
            8 => _mm512_slli_epi32::<8>(a),
            9 => _mm512_slli_epi32::<9>(a),
            10 => _mm512_slli_epi32::<10>(a),
            11 => _mm512_slli_epi32::<11>(a),
            12 => _mm512_slli_epi32::<12>(a),
            13 => _mm512_slli_epi32::<13>(a),
            14 => _mm512_slli_epi32::<14>(a),
            15 => _mm512_slli_epi32::<15>(a),
            16 => _mm512_slli_epi32::<16>(a),
            17 => _mm512_slli_epi32::<17>(a),
            18 => _mm512_slli_epi32::<18>(a),
            19 => _mm512_slli_epi32::<19>(a),
            20 => _mm512_slli_epi32::<20>(a),
            21 => _mm512_slli_epi32::<21>(a),
            22 => _mm512_slli_epi32::<22>(a),
            23 => _mm512_slli_epi32::<23>(a),
            24 => _mm512_slli_epi32::<24>(a),
            25 => _mm512_slli_epi32::<25>(a),
            26 => _mm512_slli_epi32::<26>(a),
            27 => _mm512_slli_epi32::<27>(a),
            28 => _mm512_slli_epi32::<28>(a),
            29 => _mm512_slli_epi32::<29>(a),
            30 => _mm512_slli_epi32::<30>(a),
            31 => _mm512_slli_epi32::<31>(a),
            _ => unreachable!("shift count checked by the public API"),
        }
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sra_epi32(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi32(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shl_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_sll_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shr_logical_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_srl_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shr_arithmetic_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_sra_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn all_true(self, a: __m512i) -> bool {
        // Sign-bit contract: `movepi*_mask` reads the sign bits
        // directly. The `cmpneq` form this replaces answered a
        // different question (lane nonzero) and disagreed with the
        // V3 movemask backends.
        _mm512_movepi32_mask(a) as u64 == 0xFFFFu64
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn any_true(self, a: __m512i) -> bool {
        _mm512_movepi32_mask(a) as u64 != 0
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitmask(self, a: __m512i) -> u64 {
        _mm512_movepi32_mask(a) as u64
    }
    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn narrow_saturating_i32_to_i16(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_inserti64x4::<1>(
            _mm512_castsi256_si512(_mm512_cvtsepi32_epi16(a)),
            _mm512_cvtsepi32_epi16(b),
        )
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
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
impl U32x16Backend for X64V4Token {
    type Repr = __m512i;

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn splat(self, v: u32) -> __m512i {
        _mm512_set1_epi32(v as _)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    fn load(self, data: &[u32; 16]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u32; 16]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m512i, out: &mut [u32; 16]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m512i) -> [u32; 16] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn mul(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_mullo_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi32(_mm512_setzero_si512(), a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epu32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epu32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epu32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1i32)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epu32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1i32)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epu32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1i32)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epu32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1i32)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epu32_mask(b, a);
        _mm512_maskz_set1_epi32(mask, -1i32)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epu32_mask(b, a);
        _mm512_maskz_set1_epi32(mask, -1i32)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi32_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi32(k, if_false, if_true)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn reduce_add(self, a: __m512i) -> u32 {
        // AVX-512 has no single-instruction integer reduction; fold the lanes
        let arr: [u32; 16] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0u32, u32::wrapping_add)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi32(-1i32))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        match N {
            0 => _mm512_slli_epi32::<0>(a),
            1 => _mm512_slli_epi32::<1>(a),
            2 => _mm512_slli_epi32::<2>(a),
            3 => _mm512_slli_epi32::<3>(a),
            4 => _mm512_slli_epi32::<4>(a),
            5 => _mm512_slli_epi32::<5>(a),
            6 => _mm512_slli_epi32::<6>(a),
            7 => _mm512_slli_epi32::<7>(a),
            8 => _mm512_slli_epi32::<8>(a),
            9 => _mm512_slli_epi32::<9>(a),
            10 => _mm512_slli_epi32::<10>(a),
            11 => _mm512_slli_epi32::<11>(a),
            12 => _mm512_slli_epi32::<12>(a),
            13 => _mm512_slli_epi32::<13>(a),
            14 => _mm512_slli_epi32::<14>(a),
            15 => _mm512_slli_epi32::<15>(a),
            16 => _mm512_slli_epi32::<16>(a),
            17 => _mm512_slli_epi32::<17>(a),
            18 => _mm512_slli_epi32::<18>(a),
            19 => _mm512_slli_epi32::<19>(a),
            20 => _mm512_slli_epi32::<20>(a),
            21 => _mm512_slli_epi32::<21>(a),
            22 => _mm512_slli_epi32::<22>(a),
            23 => _mm512_slli_epi32::<23>(a),
            24 => _mm512_slli_epi32::<24>(a),
            25 => _mm512_slli_epi32::<25>(a),
            26 => _mm512_slli_epi32::<26>(a),
            27 => _mm512_slli_epi32::<27>(a),
            28 => _mm512_slli_epi32::<28>(a),
            29 => _mm512_slli_epi32::<29>(a),
            30 => _mm512_slli_epi32::<30>(a),
            31 => _mm512_slli_epi32::<31>(a),
            _ => unreachable!("shift count checked by the public API"),
        }
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi32(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi32(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shl_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_sll_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shr_logical_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_srl_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shr_arithmetic_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_srl_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn all_true(self, a: __m512i) -> bool {
        // Sign-bit contract: `movepi*_mask` reads the sign bits
        // directly. The `cmpneq` form this replaces answered a
        // different question (lane nonzero) and disagreed with the
        // V3 movemask backends.
        _mm512_movepi32_mask(a) as u64 == 0xFFFFu64
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn any_true(self, a: __m512i) -> bool {
        _mm512_movepi32_mask(a) as u64 != 0
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitmask(self, a: __m512i) -> u64 {
        _mm512_movepi32_mask(a) as u64
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I64x8Backend for X64V4Token {
    type Repr = __m512i;

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn splat(self, v: i64) -> __m512i {
        _mm512_set1_epi64(v)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    fn load(self, data: &[i64; 8]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i64; 8]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m512i, out: &mut [i64; 8]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m512i) -> [i64; 8] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi64(_mm512_setzero_si512(), a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epi64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epi64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn abs(self, a: __m512i) -> __m512i {
        _mm512_abs_epi64(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epi64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epi64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epi64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epi64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epi64_mask(b, a);
        _mm512_maskz_set1_epi64(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epi64_mask(b, a);
        _mm512_maskz_set1_epi64(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi64_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi64(k, if_false, if_true)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn reduce_add(self, a: __m512i) -> i64 {
        // AVX-512 has no single-instruction integer reduction; fold the lanes
        let arr: [i64; 8] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0i64, i64::wrapping_add)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi64(-1))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sll_epi64(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sra_epi64(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi64(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn all_true(self, a: __m512i) -> bool {
        // Sign-bit contract: `movepi*_mask` reads the sign bits
        // directly. The `cmpneq` form this replaces answered a
        // different question (lane nonzero) and disagreed with the
        // V3 movemask backends.
        _mm512_movepi64_mask(a) as u64 == 0xFFu64
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn any_true(self, a: __m512i) -> bool {
        _mm512_movepi64_mask(a) as u64 != 0
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitmask(self, a: __m512i) -> u64 {
        _mm512_movepi64_mask(a) as u64
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl U64x8Backend for X64V4Token {
    type Repr = __m512i;

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn splat(self, v: u64) -> __m512i {
        _mm512_set1_epi64(v as _)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    fn load(self, data: &[u64; 8]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u64; 8]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m512i, out: &mut [u64; 8]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m512i) -> [u64; 8] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi64(_mm512_setzero_si512(), a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epu64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epu64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epu64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1i64)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epu64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1i64)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epu64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1i64)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epu64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1i64)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epu64_mask(b, a);
        _mm512_maskz_set1_epi64(mask, -1i64)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epu64_mask(b, a);
        _mm512_maskz_set1_epi64(mask, -1i64)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi64_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi64(k, if_false, if_true)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn reduce_add(self, a: __m512i) -> u64 {
        // AVX-512 has no single-instruction integer reduction; fold the lanes
        let arr: [u64; 8] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0u64, u64::wrapping_add)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi64(-1i64))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sll_epi64(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi64(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi64(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn all_true(self, a: __m512i) -> bool {
        // Sign-bit contract: `movepi*_mask` reads the sign bits
        // directly. The `cmpneq` form this replaces answered a
        // different question (lane nonzero) and disagreed with the
        // V3 movemask backends.
        _mm512_movepi64_mask(a) as u64 == 0xFFu64
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn any_true(self, a: __m512i) -> bool {
        _mm512_movepi64_mask(a) as u64 != 0
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitmask(self, a: __m512i) -> u64 {
        _mm512_movepi64_mask(a) as u64
    }
}

// ============================================================================
// X64V4xToken — base AVX-512 (same intrinsics as V4)
// ============================================================================

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl F32x16Backend for X64V4xToken {
    type Repr = __m512;

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn splat(self, v: f32) -> __m512 {
        _mm512_set1_ps(v)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn zero(self) -> __m512 {
        _mm512_setzero_ps()
    }

    #[inline(always)]
    fn load(self, data: &[f32; 16]) -> __m512 {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [f32; 16]) -> __m512 {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m512, out: &mut [f32; 16]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m512) -> [f32; 16] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn add(self, a: __m512, b: __m512) -> __m512 {
        _mm512_add_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn sub(self, a: __m512, b: __m512) -> __m512 {
        _mm512_sub_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn mul(self, a: __m512, b: __m512) -> __m512 {
        _mm512_mul_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn div(self, a: __m512, b: __m512) -> __m512 {
        _mm512_div_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn neg(self, a: __m512) -> __m512 {
        _mm512_xor_ps(a, _mm512_set1_ps(-0.0))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn min(self, a: __m512, b: __m512) -> __m512 {
        _mm512_range_ps::<0x04>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn max(self, a: __m512, b: __m512) -> __m512 {
        _mm512_range_ps::<0x05>(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn minimum(self, a: __m512, b: __m512) -> __m512 {
        _mm512_mask_blend_ps(
            _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(a, b),
            _mm512_or_ps(_mm512_min_ps(a, b), _mm512_min_ps(b, a)),
            _mm512_set1_ps(f32::NAN),
        )
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn maximum(self, a: __m512, b: __m512) -> __m512 {
        _mm512_mask_blend_ps(
            _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(a, b),
            _mm512_and_ps(_mm512_max_ps(a, b), _mm512_max_ps(b, a)),
            _mm512_set1_ps(f32::NAN),
        )
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn min_fast(self, a: __m512, b: __m512) -> __m512 {
        _mm512_min_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn max_fast(self, a: __m512, b: __m512) -> __m512 {
        _mm512_max_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn sqrt(self, a: __m512) -> __m512 {
        _mm512_sqrt_ps(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn abs(self, a: __m512) -> __m512 {
        let mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFF_FFFFu32 as i32));
        _mm512_and_ps(a, mask)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn trunc(self, a: __m512) -> __m512 {
        _mm512_roundscale_ps::<0x03>(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn mul_neg_add_relaxed(self, a: __m512, b: __m512, c: __m512) -> __m512 {
        _mm512_fnmadd_ps(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn mul_neg_add_fused(self, a: __m512, b: __m512, c: __m512) -> __m512 {
        _mm512_fnmadd_ps(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn floor(self, a: __m512) -> __m512 {
        _mm512_roundscale_ps::<0x01>(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn ceil(self, a: __m512) -> __m512 {
        _mm512_roundscale_ps::<0x02>(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn round(self, a: __m512) -> __m512 {
        _mm512_roundscale_ps::<0x00>(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn mul_add(self, a: __m512, b: __m512, c: __m512) -> __m512 {
        _mm512_fmadd_ps(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn mul_sub(self, a: __m512, b: __m512, c: __m512) -> __m512 {
        _mm512_fmsub_ps(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn mul_add_portable(self, a: __m512, b: __m512, c: __m512) -> __m512 {
        _mm512_fmadd_ps(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn mul_sub_portable(self, a: __m512, b: __m512, c: __m512) -> __m512 {
        _mm512_fmsub_ps(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_eq(self, a: __m512, b: __m512) -> __m512 {
        let mask = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(a, b);
        _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_ne(self, a: __m512, b: __m512) -> __m512 {
        let mask = _mm512_cmp_ps_mask::<_CMP_NEQ_UQ>(a, b);
        _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_lt(self, a: __m512, b: __m512) -> __m512 {
        let mask = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(a, b);
        _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_le(self, a: __m512, b: __m512) -> __m512 {
        let mask = _mm512_cmp_ps_mask::<_CMP_LE_OQ>(a, b);
        _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_gt(self, a: __m512, b: __m512) -> __m512 {
        let mask = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(a, b);
        _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_ge(self, a: __m512, b: __m512) -> __m512 {
        let mask = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(a, b);
        _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn blend(self, mask: __m512, if_true: __m512, if_false: __m512) -> __m512 {
        let mask_i = _mm512_castps_si512(mask);
        let k = _mm512_cmpneq_epi32_mask(mask_i, _mm512_setzero_si512());
        _mm512_mask_blend_ps(k, if_false, if_true)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn reduce_add(self, a: __m512) -> f32 {
        _mm512_reduce_add_ps(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn reduce_min(self, a: __m512) -> f32 {
        _mm512_reduce_min_ps(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn reduce_max(self, a: __m512) -> f32 {
        _mm512_reduce_max_ps(a)
    }

    // Raw 14-bit hardware estimate — the cheapest >=12-bit path on AVX-512.
    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn rcp_approx(self, a: __m512) -> __m512 {
        _mm512_rcp14_ps(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn rsqrt_approx(self, a: __m512) -> __m512 {
        _mm512_rsqrt14_ps(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn from_u16_lanes(self, lanes: &[u16; 16]) -> __m512 {
        _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(crate::simd_storage::cast(*lanes)))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn to_u16_lanes<const CFG: u32>(self, value: __m512) -> [u16; 16] {
        let opts = const { crate::numerics::IntConvert::decode(CFG) };
        let value = if opts.saturates_all() {
            _mm512_min_ps(_mm512_set1_ps(65535.0), value)
        } else {
            value
        };
        if opts.nonnegative() {
            let i = if opts.rounds() {
                _mm512_cvtps_epu32(value)
            } else {
                _mm512_cvttps_epu32(value)
            };
            return crate::simd_storage::cast(_mm512_cvtusepi32_epi16(i));
        }

        let i = if opts.rounds() {
            _mm512_cvtps_epi32(value)
        } else {
            _mm512_cvttps_epi32(value)
        };
        crate::simd_storage::cast(_mm512_cvtusepi32_epi16(_mm512_max_epi32(
            i,
            _mm512_setzero_si512(),
        )))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn to_u8_lanes<const CFG: u32>(self, value: __m512) -> [u8; 16] {
        let opts = const { crate::numerics::IntConvert::decode(CFG) };
        let value = if opts.saturates_all() {
            _mm512_min_ps(_mm512_set1_ps(255.0), value)
        } else {
            value
        };
        if opts.nonnegative() {
            let i = if opts.rounds() {
                _mm512_cvtps_epu32(value)
            } else {
                _mm512_cvttps_epu32(value)
            };
            return crate::simd_storage::cast(_mm512_cvtusepi32_epi8(i));
        }
        let i = if opts.rounds() {
            _mm512_cvtps_epi32(value)
        } else {
            _mm512_cvttps_epi32(value)
        };
        crate::simd_storage::cast(_mm512_cvtusepi32_epi8(_mm512_max_epi32(
            i,
            _mm512_setzero_si512(),
        )))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn to_u32_lanes<const CFG: u32>(self, value: __m512) -> __m512 {
        let opts = const { crate::numerics::IntConvert::decode(CFG) };
        let clamped = if opts.nonnegative() {
            value
        } else {
            _mm512_max_ps(value, _mm512_setzero_ps())
        };
        let i = if opts.rounds() {
            _mm512_cvtps_epu32(clamped)
        } else {
            _mm512_cvttps_epu32(clamped)
        };
        _mm512_castsi512_ps(i)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn from_u32_lanes(self, value: __m512) -> __m512 {
        _mm512_cvtepu32_ps(_mm512_castps_si512(value))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn from_f16_lanes<const CFG: u32>(self, lanes: &[u16; 16]) -> __m512 {
        let opts = const { crate::numerics::HalfConvert::decode(CFG) };
        let h = crate::simd_storage::cast(*lanes);
        let h = if opts.flushes() {
            let mag = _mm256_and_si256(h, _mm256_set1_epi16(0x7fff));
            let small = _mm256_cmpgt_epi16(_mm256_set1_epi16(0x400), mag);
            _mm256_andnot_si256(_mm256_and_si256(small, _mm256_set1_epi16(0x7fff)), h)
        } else {
            h
        };
        _mm512_cvtph_ps(h)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn to_f16_lanes<const CFG: u32>(self, value: __m512) -> [u16; 16] {
        let opts = const { crate::numerics::HalfConvert::decode(CFG) };
        let h = if opts.rounds() {
            _mm512_cvtps_ph::<0>(value)
        } else {
            _mm512_cvtps_ph::<3>(value)
        };
        let h = if opts.flushes() {
            let mag = _mm256_and_si256(h, _mm256_set1_epi16(0x7fff));
            let small = _mm256_cmpgt_epi16(_mm256_set1_epi16(0x400), mag);
            _mm256_andnot_si256(_mm256_and_si256(small, _mm256_set1_epi16(0x7fff)), h)
        } else {
            h
        };
        crate::simd_storage::cast(h)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn from_bf16_lanes(self, lanes: &[u16; 16]) -> __m512 {
        _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(
            crate::simd_storage::cast(*lanes),
        )))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn to_bf16_lanes<const CFG: u32>(self, value: __m512) -> [u16; 16] {
        let opts = const { crate::numerics::HalfConvert::decode(CFG) };
        let raw = _mm512_castps_si512(value);
        let rounded = if opts.rounds() {
            let lsb = _mm512_and_si512(_mm512_srli_epi32::<16>(raw), _mm512_set1_epi32(1));
            _mm512_add_epi32(raw, _mm512_add_epi32(lsb, _mm512_set1_epi32(0x7fff)))
        } else {
            raw
        };
        let i = _mm512_srli_epi32::<16>(rounded);
        let out: [u16; 16] = crate::simd_storage::cast(_mm512_cvtusepi32_epi16(_mm512_max_epi32(
            i,
            _mm512_setzero_si512(),
        )));
        if opts.is_finite() && !opts.flushes() {
            out
        } else {
            // Special and signed-subnormal handling is shared with scalar.
            let original: [f32; 16] = crate::simd_storage::cast(value);
            core::array::from_fn(|i| {
                if original[i].is_nan() || (opts.flushes() && out[i] & 0x7f80 == 0) {
                    crate::scalar::to_bf16_with::<CFG>(original[i])
                } else {
                    out[i]
                }
            })
        }
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn from_u8_lanes_wide(self, lanes: &[u8; 16]) -> __m512 {
        _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(crate::simd_storage::cast(*lanes)))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn deinterleave_rgba_wide(self, rgba: [__m512; 4]) -> [__m512; 4] {
        let [a, b, c, d] = rgba;
        let rg = _mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29);
        let ba = _mm512_add_epi32(rg, _mm512_set1_epi32(2));
        let low = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
        let high = _mm512_add_epi32(low, _mm512_set1_epi32(8));
        let ab_rg = _mm512_permutex2var_ps(a, rg, b);
        let cd_rg = _mm512_permutex2var_ps(c, rg, d);
        let ab_ba = _mm512_permutex2var_ps(a, ba, b);
        let cd_ba = _mm512_permutex2var_ps(c, ba, d);
        [
            _mm512_permutex2var_ps(ab_rg, low, cd_rg),
            _mm512_permutex2var_ps(ab_rg, high, cd_rg),
            _mm512_permutex2var_ps(ab_ba, low, cd_ba),
            _mm512_permutex2var_ps(ab_ba, high, cd_ba),
        ]
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn interleave_rgba_wide(self, channels: [__m512; 4]) -> [__m512; 4] {
        let [r, g, b, a] = channels;
        let low = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
        let high = _mm512_add_epi32(low, _mm512_set1_epi32(8));
        let first = _mm512_setr_epi32(0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23);
        let second = _mm512_add_epi32(first, _mm512_set1_epi32(8));
        let rg0 = _mm512_permutex2var_ps(r, low, g);
        let ba0 = _mm512_permutex2var_ps(b, low, a);
        let rg1 = _mm512_permutex2var_ps(r, high, g);
        let ba1 = _mm512_permutex2var_ps(b, high, a);
        [
            _mm512_permutex2var_ps(rg0, first, ba0),
            _mm512_permutex2var_ps(rg0, second, ba0),
            _mm512_permutex2var_ps(rg1, first, ba1),
            _mm512_permutex2var_ps(rg1, second, ba1),
        ]
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn broadcast_rgba_alpha_repr(self, value: __m512) -> __m512 {
        _mm512_permute_ps::<255>(value)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn with_rgba_alpha_repr(self, value: __m512, src: __m512) -> __m512 {
        _mm512_mask_blend_ps(0x8888, value, src)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
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

    #[arcane(suppress_const_test, _self = X64V4xToken)]
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

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn not(self, a: __m512) -> __m512 {
        let all_ones = _mm512_castsi512_ps(_mm512_set1_epi32(-1));
        _mm512_xor_ps(a, all_ones)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitand(self, a: __m512, b: __m512) -> __m512 {
        _mm512_and_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitor(self, a: __m512, b: __m512) -> __m512 {
        _mm512_or_ps(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitxor(self, a: __m512, b: __m512) -> __m512 {
        _mm512_xor_ps(a, b)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl F64x8Backend for X64V4xToken {
    type Repr = __m512d;

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn splat(self, v: f64) -> __m512d {
        _mm512_set1_pd(v)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn zero(self) -> __m512d {
        _mm512_setzero_pd()
    }

    #[inline(always)]
    fn load(self, data: &[f64; 8]) -> __m512d {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [f64; 8]) -> __m512d {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m512d, out: &mut [f64; 8]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m512d) -> [f64; 8] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn add(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_add_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn sub(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_sub_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn mul(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_mul_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn div(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_div_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn neg(self, a: __m512d) -> __m512d {
        _mm512_xor_pd(a, _mm512_set1_pd(-0.0))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn min(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_mask_blend_pd(
            _mm512_cmp_pd_mask::<_CMP_UNORD_Q>(b, b),
            _mm512_min_pd(a, b),
            a,
        )
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn max(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_mask_blend_pd(
            _mm512_cmp_pd_mask::<_CMP_UNORD_Q>(b, b),
            _mm512_max_pd(a, b),
            a,
        )
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn minimum(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_mask_blend_pd(
            _mm512_cmp_pd_mask::<_CMP_UNORD_Q>(a, b),
            _mm512_or_pd(_mm512_min_pd(a, b), _mm512_min_pd(b, a)),
            _mm512_set1_pd(f64::NAN),
        )
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn maximum(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_mask_blend_pd(
            _mm512_cmp_pd_mask::<_CMP_UNORD_Q>(a, b),
            _mm512_and_pd(_mm512_max_pd(a, b), _mm512_max_pd(b, a)),
            _mm512_set1_pd(f64::NAN),
        )
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn min_fast(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_min_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn max_fast(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_max_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn sqrt(self, a: __m512d) -> __m512d {
        _mm512_sqrt_pd(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn abs(self, a: __m512d) -> __m512d {
        let mask = _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFF_FFFF_FFFF_FFFFu64 as i64));
        _mm512_and_pd(a, mask)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn trunc(self, a: __m512d) -> __m512d {
        _mm512_roundscale_pd::<0x03>(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn mul_neg_add_relaxed(self, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        _mm512_fnmadd_pd(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn mul_neg_add_fused(self, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        _mm512_fnmadd_pd(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn floor(self, a: __m512d) -> __m512d {
        _mm512_roundscale_pd::<0x01>(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn ceil(self, a: __m512d) -> __m512d {
        _mm512_roundscale_pd::<0x02>(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn round(self, a: __m512d) -> __m512d {
        _mm512_roundscale_pd::<0x00>(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn mul_add(self, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        _mm512_fmadd_pd(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn mul_sub(self, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        _mm512_fmsub_pd(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn mul_add_portable(self, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        _mm512_fmadd_pd(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn mul_sub_portable(self, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        _mm512_fmsub_pd(a, b, c)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_eq(self, a: __m512d, b: __m512d) -> __m512d {
        let mask = _mm512_cmp_pd_mask::<_CMP_EQ_OQ>(a, b);
        _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_ne(self, a: __m512d, b: __m512d) -> __m512d {
        let mask = _mm512_cmp_pd_mask::<_CMP_NEQ_UQ>(a, b);
        _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_lt(self, a: __m512d, b: __m512d) -> __m512d {
        let mask = _mm512_cmp_pd_mask::<_CMP_LT_OQ>(a, b);
        _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_le(self, a: __m512d, b: __m512d) -> __m512d {
        let mask = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(a, b);
        _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_gt(self, a: __m512d, b: __m512d) -> __m512d {
        let mask = _mm512_cmp_pd_mask::<_CMP_GT_OQ>(a, b);
        _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_ge(self, a: __m512d, b: __m512d) -> __m512d {
        let mask = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(a, b);
        _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn blend(self, mask: __m512d, if_true: __m512d, if_false: __m512d) -> __m512d {
        let mask_i = _mm512_castpd_si512(mask);
        let k = _mm512_cmpneq_epi64_mask(mask_i, _mm512_setzero_si512());
        _mm512_mask_blend_pd(k, if_false, if_true)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn reduce_add(self, a: __m512d) -> f64 {
        _mm512_reduce_add_pd(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn reduce_min(self, a: __m512d) -> f64 {
        _mm512_reduce_min_pd(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn reduce_max(self, a: __m512d) -> f64 {
        _mm512_reduce_max_pd(a)
    }

    // Raw 14-bit hardware estimate — the cheapest >=12-bit path on AVX-512.
    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn rcp_approx(self, a: __m512d) -> __m512d {
        _mm512_rcp14_pd(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn rsqrt_approx(self, a: __m512d) -> __m512d {
        _mm512_rsqrt14_pd(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn recip(self, a: __m512d) -> __m512d {
        _mm512_div_pd(_mm512_set1_pd(1.0), a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn rsqrt(self, a: __m512d) -> __m512d {
        _mm512_div_pd(_mm512_set1_pd(1.0), _mm512_sqrt_pd(a))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn not(self, a: __m512d) -> __m512d {
        let all_ones = _mm512_castsi512_pd(_mm512_set1_epi64(-1));
        _mm512_xor_pd(a, all_ones)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitand(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_and_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitor(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_or_pd(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitxor(self, a: __m512d, b: __m512d) -> __m512d {
        _mm512_xor_pd(a, b)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I8x64Backend for X64V4xToken {
    type Repr = __m512i;

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn splat(self, v: i8) -> __m512i {
        _mm512_set1_epi8(v)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    fn load(self, data: &[i8; 64]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i8; 64]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m512i, out: &mut [i8; 64]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m512i) -> [i8; 64] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi8(_mm512_setzero_si512(), a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn abs(self, a: __m512i) -> __m512i {
        _mm512_abs_epi8(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epi8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epi8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epi8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epi8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epi8_mask(b, a);
        _mm512_maskz_set1_epi8(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epi8_mask(b, a);
        _mm512_maskz_set1_epi8(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi8_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi8(k, if_false, if_true)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn reduce_add(self, a: __m512i) -> i8 {
        // AVX-512 has no single-instruction integer reduction; fold the lanes
        let arr: [i8; 64] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0i8, i8::wrapping_add)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi8(-1))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        let count = _mm_cvtsi32_si128(N);
        let shifted = _mm512_sll_epi16(a, count);
        let mask = _mm512_set1_epi8(((0xFFu16 << N) & 0xFF) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
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

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        let count = _mm_cvtsi32_si128(N);
        let shifted = _mm512_srl_epi16(a, count);
        let mask = _mm512_set1_epi8(((0xFFu16 >> N) & 0xFF) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn saturating_add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_adds_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn saturating_sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_subs_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn all_true(self, a: __m512i) -> bool {
        // Sign-bit contract: `movepi*_mask` reads the sign bits
        // directly. The `cmpneq` form this replaces answered a
        // different question (lane nonzero) and disagreed with the
        // V3 movemask backends.
        _mm512_movepi8_mask(a) == 0xFFFF_FFFF_FFFF_FFFFu64
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn any_true(self, a: __m512i) -> bool {
        _mm512_movepi8_mask(a) != 0
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitmask(self, a: __m512i) -> u64 {
        _mm512_movepi8_mask(a)
    }
    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn widen_low_i8_to_i16(self, a: __m512i) -> __m512i {
        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(a))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn widen_high_i8_to_i16(self, a: __m512i) -> __m512i {
        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64::<1>(a))
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl U8x64Backend for X64V4xToken {
    type Repr = __m512i;

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn splat(self, v: u8) -> __m512i {
        _mm512_set1_epi8(v as _)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    fn load(self, data: &[u8; 64]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u8; 64]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m512i, out: &mut [u8; 64]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m512i) -> [u8; 64] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi8(_mm512_setzero_si512(), a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epu8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epu8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epu8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epu8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epu8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epu8_mask(a, b);
        _mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epu8_mask(b, a);
        _mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epu8_mask(b, a);
        _mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi8_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi8(k, if_false, if_true)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn reduce_add(self, a: __m512i) -> u8 {
        // AVX-512 has no single-instruction integer reduction; fold the lanes
        let arr: [u8; 64] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0u8, u8::wrapping_add)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi8(-1i8))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        let count = _mm_cvtsi32_si128(N);
        let shifted = _mm512_sll_epi16(a, count);
        let mask = _mm512_set1_epi8(((0xFFu16 << N) & 0xFF) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
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

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        let count = _mm_cvtsi32_si128(N);
        let shifted = _mm512_srl_epi16(a, count);
        let mask = _mm512_set1_epi8(((0xFFu16 >> N) & 0xFF) as i8);
        _mm512_and_si512(shifted, mask)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn saturating_add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_adds_epu8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn saturating_sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_subs_epu8(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn all_true(self, a: __m512i) -> bool {
        // Sign-bit contract: `movepi*_mask` reads the sign bits
        // directly. The `cmpneq` form this replaces answered a
        // different question (lane nonzero) and disagreed with the
        // V3 movemask backends.
        _mm512_movepi8_mask(a) == 0xFFFF_FFFF_FFFF_FFFFu64
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn any_true(self, a: __m512i) -> bool {
        _mm512_movepi8_mask(a) != 0
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitmask(self, a: __m512i) -> u64 {
        _mm512_movepi8_mask(a)
    }
    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn widen_low_u8_to_u16(self, a: __m512i) -> __m512i {
        _mm512_cvtepu8_epi16(_mm512_castsi512_si256(a))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn widen_high_u8_to_u16(self, a: __m512i) -> __m512i {
        _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(a))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn abs_diff(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi8(_mm512_max_epu8(a, b), _mm512_min_epu8(a, b))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn reduce_add_u32(self, a: __m512i) -> u32 {
        {
            let s = _mm512_sad_epu8(a, _mm512_setzero_si512());
            _mm512_reduce_add_epi64(s) as u32
        }
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn sum_abs_diff(self, a: __m512i, b: __m512i) -> u32 {
        {
            let s = _mm512_sad_epu8(a, b);
            _mm512_reduce_add_epi64(s) as u32
        }
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn pairwise_widen_add(self, a: __m512i) -> __m512i {
        _mm512_add_epi16(
            _mm512_and_si512(a, _mm512_set1_epi16(255)),
            _mm512_srli_epi16::<8>(a),
        )
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I16x32Backend for X64V4xToken {
    type Repr = __m512i;

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn splat(self, v: i16) -> __m512i {
        _mm512_set1_epi16(v)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    fn load(self, data: &[i16; 32]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i16; 32]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m512i, out: &mut [i16; 32]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m512i) -> [i16; 32] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn mul(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_mullo_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi16(_mm512_setzero_si512(), a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn abs(self, a: __m512i) -> __m512i {
        _mm512_abs_epi16(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epi16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epi16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epi16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epi16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epi16_mask(b, a);
        _mm512_maskz_set1_epi16(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epi16_mask(b, a);
        _mm512_maskz_set1_epi16(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi16_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi16(k, if_false, if_true)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn reduce_add(self, a: __m512i) -> i16 {
        // AVX-512 has no single-instruction integer reduction; fold the lanes
        let arr: [i16; 32] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0i16, i16::wrapping_add)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi16(-1))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sll_epi16(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sra_epi16(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi16(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shl_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_sll_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shr_logical_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_srl_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shr_arithmetic_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_sra_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn saturating_add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_adds_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn saturating_sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_subs_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn all_true(self, a: __m512i) -> bool {
        // Sign-bit contract: `movepi*_mask` reads the sign bits
        // directly. The `cmpneq` form this replaces answered a
        // different question (lane nonzero) and disagreed with the
        // V3 movemask backends.
        _mm512_movepi16_mask(a) as u64 == 0xFFFF_FFFFu64
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn any_true(self, a: __m512i) -> bool {
        _mm512_movepi16_mask(a) as u64 != 0
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitmask(self, a: __m512i) -> u64 {
        _mm512_movepi16_mask(a) as u64
    }
    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn widen_low_i16_to_i32(self, a: __m512i) -> __m512i {
        _mm512_cvtepi16_epi32(_mm512_castsi512_si256(a))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn widen_high_i16_to_i32(self, a: __m512i) -> __m512i {
        _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(a))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn narrow_saturating_i16_to_i8(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_inserti64x4::<1>(
            _mm512_castsi256_si512(_mm512_cvtsepi16_epi8(a)),
            _mm512_cvtsepi16_epi8(b),
        )
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
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

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn madd_adjacent(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_madd_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn abs_diff(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi16(_mm512_max_epi16(a, b), _mm512_min_epi16(a, b))
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl U16x32Backend for X64V4xToken {
    type Repr = __m512i;

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn splat(self, v: u16) -> __m512i {
        _mm512_set1_epi16(v as _)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    fn load(self, data: &[u16; 32]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u16; 32]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m512i, out: &mut [u16; 32]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m512i) -> [u16; 32] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn mul(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_mullo_epi16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi16(_mm512_setzero_si512(), a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epu16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epu16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epu16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epu16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epu16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epu16_mask(a, b);
        _mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epu16_mask(b, a);
        _mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epu16_mask(b, a);
        _mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi16_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi16(k, if_false, if_true)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn reduce_add(self, a: __m512i) -> u16 {
        // AVX-512 has no single-instruction integer reduction; fold the lanes
        let arr: [u16; 32] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0u16, u16::wrapping_add)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi16(-1i16))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sll_epi16(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi16(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi16(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shl_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_sll_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shr_logical_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_srl_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shr_arithmetic_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_srl_epi16(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn saturating_add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_adds_epu16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn saturating_sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_subs_epu16(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn all_true(self, a: __m512i) -> bool {
        // Sign-bit contract: `movepi*_mask` reads the sign bits
        // directly. The `cmpneq` form this replaces answered a
        // different question (lane nonzero) and disagreed with the
        // V3 movemask backends.
        _mm512_movepi16_mask(a) as u64 == 0xFFFF_FFFFu64
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn any_true(self, a: __m512i) -> bool {
        _mm512_movepi16_mask(a) as u64 != 0
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitmask(self, a: __m512i) -> u64 {
        _mm512_movepi16_mask(a) as u64
    }
    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn widen_low_u16_to_u32(self, a: __m512i) -> __m512i {
        _mm512_cvtepu16_epi32(_mm512_castsi512_si256(a))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn widen_high_u16_to_u32(self, a: __m512i) -> __m512i {
        _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64::<1>(a))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn pairwise_widen_add(self, a: __m512i) -> __m512i {
        _mm512_add_epi32(
            _mm512_and_si512(a, _mm512_set1_epi32(65535)),
            _mm512_srli_epi32::<16>(a),
        )
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I32x16Backend for X64V4xToken {
    type Repr = __m512i;

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn splat(self, v: i32) -> __m512i {
        _mm512_set1_epi32(v)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    fn load(self, data: &[i32; 16]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i32; 16]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m512i, out: &mut [i32; 16]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m512i) -> [i32; 16] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn mul(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_mullo_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi32(_mm512_setzero_si512(), a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn abs(self, a: __m512i) -> __m512i {
        _mm512_abs_epi32(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epi32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epi32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epi32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epi32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epi32_mask(b, a);
        _mm512_maskz_set1_epi32(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epi32_mask(b, a);
        _mm512_maskz_set1_epi32(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi32_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi32(k, if_false, if_true)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn reduce_add(self, a: __m512i) -> i32 {
        // AVX-512 has no single-instruction integer reduction; fold the lanes
        let arr: [i32; 16] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0i32, i32::wrapping_add)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi32(-1))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        match N {
            0 => _mm512_slli_epi32::<0>(a),
            1 => _mm512_slli_epi32::<1>(a),
            2 => _mm512_slli_epi32::<2>(a),
            3 => _mm512_slli_epi32::<3>(a),
            4 => _mm512_slli_epi32::<4>(a),
            5 => _mm512_slli_epi32::<5>(a),
            6 => _mm512_slli_epi32::<6>(a),
            7 => _mm512_slli_epi32::<7>(a),
            8 => _mm512_slli_epi32::<8>(a),
            9 => _mm512_slli_epi32::<9>(a),
            10 => _mm512_slli_epi32::<10>(a),
            11 => _mm512_slli_epi32::<11>(a),
            12 => _mm512_slli_epi32::<12>(a),
            13 => _mm512_slli_epi32::<13>(a),
            14 => _mm512_slli_epi32::<14>(a),
            15 => _mm512_slli_epi32::<15>(a),
            16 => _mm512_slli_epi32::<16>(a),
            17 => _mm512_slli_epi32::<17>(a),
            18 => _mm512_slli_epi32::<18>(a),
            19 => _mm512_slli_epi32::<19>(a),
            20 => _mm512_slli_epi32::<20>(a),
            21 => _mm512_slli_epi32::<21>(a),
            22 => _mm512_slli_epi32::<22>(a),
            23 => _mm512_slli_epi32::<23>(a),
            24 => _mm512_slli_epi32::<24>(a),
            25 => _mm512_slli_epi32::<25>(a),
            26 => _mm512_slli_epi32::<26>(a),
            27 => _mm512_slli_epi32::<27>(a),
            28 => _mm512_slli_epi32::<28>(a),
            29 => _mm512_slli_epi32::<29>(a),
            30 => _mm512_slli_epi32::<30>(a),
            31 => _mm512_slli_epi32::<31>(a),
            _ => unreachable!("shift count checked by the public API"),
        }
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sra_epi32(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi32(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shl_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_sll_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shr_logical_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_srl_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shr_arithmetic_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_sra_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn all_true(self, a: __m512i) -> bool {
        // Sign-bit contract: `movepi*_mask` reads the sign bits
        // directly. The `cmpneq` form this replaces answered a
        // different question (lane nonzero) and disagreed with the
        // V3 movemask backends.
        _mm512_movepi32_mask(a) as u64 == 0xFFFFu64
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn any_true(self, a: __m512i) -> bool {
        _mm512_movepi32_mask(a) as u64 != 0
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitmask(self, a: __m512i) -> u64 {
        _mm512_movepi32_mask(a) as u64
    }
    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn narrow_saturating_i32_to_i16(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_inserti64x4::<1>(
            _mm512_castsi256_si512(_mm512_cvtsepi32_epi16(a)),
            _mm512_cvtsepi32_epi16(b),
        )
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
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
impl U32x16Backend for X64V4xToken {
    type Repr = __m512i;

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn splat(self, v: u32) -> __m512i {
        _mm512_set1_epi32(v as _)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    fn load(self, data: &[u32; 16]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u32; 16]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m512i, out: &mut [u32; 16]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m512i) -> [u32; 16] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn mul(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_mullo_epi32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi32(_mm512_setzero_si512(), a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epu32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epu32(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epu32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1i32)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epu32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1i32)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epu32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1i32)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epu32_mask(a, b);
        _mm512_maskz_set1_epi32(mask, -1i32)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epu32_mask(b, a);
        _mm512_maskz_set1_epi32(mask, -1i32)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epu32_mask(b, a);
        _mm512_maskz_set1_epi32(mask, -1i32)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi32_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi32(k, if_false, if_true)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn reduce_add(self, a: __m512i) -> u32 {
        // AVX-512 has no single-instruction integer reduction; fold the lanes
        let arr: [u32; 16] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0u32, u32::wrapping_add)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi32(-1i32))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        match N {
            0 => _mm512_slli_epi32::<0>(a),
            1 => _mm512_slli_epi32::<1>(a),
            2 => _mm512_slli_epi32::<2>(a),
            3 => _mm512_slli_epi32::<3>(a),
            4 => _mm512_slli_epi32::<4>(a),
            5 => _mm512_slli_epi32::<5>(a),
            6 => _mm512_slli_epi32::<6>(a),
            7 => _mm512_slli_epi32::<7>(a),
            8 => _mm512_slli_epi32::<8>(a),
            9 => _mm512_slli_epi32::<9>(a),
            10 => _mm512_slli_epi32::<10>(a),
            11 => _mm512_slli_epi32::<11>(a),
            12 => _mm512_slli_epi32::<12>(a),
            13 => _mm512_slli_epi32::<13>(a),
            14 => _mm512_slli_epi32::<14>(a),
            15 => _mm512_slli_epi32::<15>(a),
            16 => _mm512_slli_epi32::<16>(a),
            17 => _mm512_slli_epi32::<17>(a),
            18 => _mm512_slli_epi32::<18>(a),
            19 => _mm512_slli_epi32::<19>(a),
            20 => _mm512_slli_epi32::<20>(a),
            21 => _mm512_slli_epi32::<21>(a),
            22 => _mm512_slli_epi32::<22>(a),
            23 => _mm512_slli_epi32::<23>(a),
            24 => _mm512_slli_epi32::<24>(a),
            25 => _mm512_slli_epi32::<25>(a),
            26 => _mm512_slli_epi32::<26>(a),
            27 => _mm512_slli_epi32::<27>(a),
            28 => _mm512_slli_epi32::<28>(a),
            29 => _mm512_slli_epi32::<29>(a),
            30 => _mm512_slli_epi32::<30>(a),
            31 => _mm512_slli_epi32::<31>(a),
            _ => unreachable!("shift count checked by the public API"),
        }
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi32(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi32(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shl_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_sll_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shr_logical_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_srl_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shr_arithmetic_uniform(self, a: __m512i, count: u32) -> __m512i {
        _mm512_srl_epi32(a, _mm_cvtsi32_si128(count as i32))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn all_true(self, a: __m512i) -> bool {
        // Sign-bit contract: `movepi*_mask` reads the sign bits
        // directly. The `cmpneq` form this replaces answered a
        // different question (lane nonzero) and disagreed with the
        // V3 movemask backends.
        _mm512_movepi32_mask(a) as u64 == 0xFFFFu64
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn any_true(self, a: __m512i) -> bool {
        _mm512_movepi32_mask(a) as u64 != 0
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitmask(self, a: __m512i) -> u64 {
        _mm512_movepi32_mask(a) as u64
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl I64x8Backend for X64V4xToken {
    type Repr = __m512i;

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn splat(self, v: i64) -> __m512i {
        _mm512_set1_epi64(v)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    fn load(self, data: &[i64; 8]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i64; 8]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m512i, out: &mut [i64; 8]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m512i) -> [i64; 8] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi64(_mm512_setzero_si512(), a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epi64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epi64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn abs(self, a: __m512i) -> __m512i {
        _mm512_abs_epi64(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epi64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epi64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epi64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epi64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epi64_mask(b, a);
        _mm512_maskz_set1_epi64(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epi64_mask(b, a);
        _mm512_maskz_set1_epi64(mask, -1)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi64_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi64(k, if_false, if_true)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn reduce_add(self, a: __m512i) -> i64 {
        // AVX-512 has no single-instruction integer reduction; fold the lanes
        let arr: [i64; 8] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0i64, i64::wrapping_add)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi64(-1))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sll_epi64(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sra_epi64(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi64(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn all_true(self, a: __m512i) -> bool {
        // Sign-bit contract: `movepi*_mask` reads the sign bits
        // directly. The `cmpneq` form this replaces answered a
        // different question (lane nonzero) and disagreed with the
        // V3 movemask backends.
        _mm512_movepi64_mask(a) as u64 == 0xFFu64
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn any_true(self, a: __m512i) -> bool {
        _mm512_movepi64_mask(a) as u64 != 0
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitmask(self, a: __m512i) -> u64 {
        _mm512_movepi64_mask(a) as u64
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl U64x8Backend for X64V4xToken {
    type Repr = __m512i;

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn splat(self, v: u64) -> __m512i {
        _mm512_set1_epi64(v as _)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn zero(self) -> __m512i {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    fn load(self, data: &[u64; 8]) -> __m512i {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u64; 8]) -> __m512i {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: __m512i, out: &mut [u64; 8]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: __m512i) -> [u64; 8] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn neg(self, a: __m512i) -> __m512i {
        _mm512_sub_epi64(_mm512_setzero_si512(), a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epu64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epu64(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpeq_epu64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1i64)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmpneq_epu64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1i64)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmplt_epu64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1i64)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {
        let mask = _mm512_cmple_epu64_mask(a, b);
        _mm512_maskz_set1_epi64(mask, -1i64)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {
        // GT = LT with swapped args
        let mask = _mm512_cmplt_epu64_mask(b, a);
        _mm512_maskz_set1_epi64(mask, -1i64)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {
        // GE = LE with swapped args
        let mask = _mm512_cmple_epu64_mask(b, a);
        _mm512_maskz_set1_epi64(mask, -1i64)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {
        let k = _mm512_cmpneq_epi64_mask(mask, _mm512_setzero_si512());
        _mm512_mask_blend_epi64(k, if_false, if_true)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn reduce_add(self, a: __m512i) -> u64 {
        // AVX-512 has no single-instruction integer reduction; fold the lanes
        let arr: [u64; 8] = crate::simd_storage::cast(a);
        arr.iter().copied().fold(0u64, u64::wrapping_add)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn not(self, a: __m512i) -> __m512i {
        _mm512_xor_si512(a, _mm512_set1_epi64(-1i64))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitand(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_sll_epi64(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi64(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {
        _mm512_srl_epi64(a, _mm_cvtsi32_si128(N))
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn all_true(self, a: __m512i) -> bool {
        // Sign-bit contract: `movepi*_mask` reads the sign bits
        // directly. The `cmpneq` form this replaces answered a
        // different question (lane nonzero) and disagreed with the
        // V3 movemask backends.
        _mm512_movepi64_mask(a) as u64 == 0xFFu64
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn any_true(self, a: __m512i) -> bool {
        _mm512_movepi64_mask(a) as u64 != 0
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitmask(self, a: __m512i) -> u64 {
        _mm512_movepi64_mask(a) as u64
    }
}

// ============================================================================
// X64V4xToken — extension: popcnt (VPOPCNTDQ + BITALG)
// ============================================================================

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl i8x64PopcntBackend for X64V4xToken {
    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn popcnt(self, a: __m512i) -> __m512i {
        _mm512_popcnt_epi8(a)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl u8x64PopcntBackend for X64V4xToken {
    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn popcnt(self, a: __m512i) -> __m512i {
        _mm512_popcnt_epi8(a)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl i16x32PopcntBackend for X64V4xToken {
    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn popcnt(self, a: __m512i) -> __m512i {
        _mm512_popcnt_epi16(a)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl u16x32PopcntBackend for X64V4xToken {
    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn popcnt(self, a: __m512i) -> __m512i {
        _mm512_popcnt_epi16(a)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl i32x16PopcntBackend for X64V4xToken {
    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn popcnt(self, a: __m512i) -> __m512i {
        _mm512_popcnt_epi32(a)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl u32x16PopcntBackend for X64V4xToken {
    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn popcnt(self, a: __m512i) -> __m512i {
        _mm512_popcnt_epi32(a)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl i64x8PopcntBackend for X64V4xToken {
    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn popcnt(self, a: __m512i) -> __m512i {
        _mm512_popcnt_epi64(a)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl u64x8PopcntBackend for X64V4xToken {
    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn popcnt(self, a: __m512i) -> __m512i {
        _mm512_popcnt_epi64(a)
    }
}

// ============================================================================
// F32x16Convert — native AVX-512 conversions
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "w512"))]
#[cfg(feature = "w512")]
impl F32x16Convert for X64V4Token {
    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitcast_f32_to_i32(self, a: __m512) -> __m512i {
        _mm512_castps_si512(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn bitcast_i32_to_f32(self, a: __m512i) -> __m512 {
        _mm512_castsi512_ps(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn convert_f32_to_i32(self, a: __m512) -> __m512i {
        _mm512_cvttps_epi32(a)
    }

    // Issue #80 fixup, mask-register form.
    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn convert_f32_to_i32_saturating(self, a: __m512) -> __m512i {
        let t = _mm512_cvttps_epi32(a);
        let big = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(a, _mm512_set1_ps(2_147_483_648.0));
        let t = _mm512_mask_mov_epi32(t, big, _mm512_set1_epi32(i32::MAX));
        let not_nan = _mm512_cmp_ps_mask::<_CMP_ORD_Q>(a, a);
        _mm512_maskz_mov_epi32(not_nan, t)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn convert_f32_to_i32_round(self, a: __m512) -> __m512i {
        _mm512_cvtps_epi32(a)
    }

    #[arcane(suppress_const_test, _self = X64V4Token)]
    fn convert_i32_to_f32(self, a: __m512i) -> __m512 {
        _mm512_cvtepi32_ps(a)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "w512"))]
#[cfg(feature = "w512")]
impl F32x16Convert for X64V4xToken {
    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitcast_f32_to_i32(self, a: __m512) -> __m512i {
        _mm512_castps_si512(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn bitcast_i32_to_f32(self, a: __m512i) -> __m512 {
        _mm512_castsi512_ps(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn convert_f32_to_i32(self, a: __m512) -> __m512i {
        _mm512_cvttps_epi32(a)
    }

    // Issue #80 fixup, mask-register form.
    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn convert_f32_to_i32_saturating(self, a: __m512) -> __m512i {
        let t = _mm512_cvttps_epi32(a);
        let big = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(a, _mm512_set1_ps(2_147_483_648.0));
        let t = _mm512_mask_mov_epi32(t, big, _mm512_set1_epi32(i32::MAX));
        let not_nan = _mm512_cmp_ps_mask::<_CMP_ORD_Q>(a, a);
        _mm512_maskz_mov_epi32(not_nan, t)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn convert_f32_to_i32_round(self, a: __m512) -> __m512i {
        _mm512_cvtps_epi32(a)
    }

    #[arcane(suppress_const_test, _self = X64V4xToken)]
    fn convert_i32_to_f32(self, a: __m512i) -> __m512 {
        _mm512_cvtepi32_ps(a)
    }
}

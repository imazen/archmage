//! Direct AVX-512VL implementations for the narrow vector backends.
//!
//! `to_u8` (both widths) and `f32x4::{store_4_rgba_u8, store_4_rgb_u8}` use
//! AVX-512VL
//! instead of the V3 bodies, because the narrowing instruction makes them
//! cheaper with identical bytes. `max(x, 0)` then `cvtps` leaves +inf and
//! values at or above 2^31 as `0x8000_0000`, which unsigned saturation
//! (`vpmovusdb`) turns into 255; NaN and negatives are already 0. That is
//! the V3 contract (round half to even, clamp to 0..=255, NaN to 0), with
//! one float operation per plane instead of the V3 form's upper clamp and
//! saturating packs. Measured on Zen 5 in
//! `benchmarks/pixel_pack_zen5-9950x3d_2026-10-03.md`; `f32x8`'s RGBA store
//! stays delegated, because its only faster form needs 512-bit registers.
//! These helpers stay at 128/256 bits and take a concrete `X64V4Token`,
//! so the soundness scanner verifies their AVX-512VL intrinsics; the
//! generated impls pass `self` (V4) or `self.v4()` (V4x, FP16).

#![cfg(all(target_arch = "x86_64", feature = "avx512"))]

use crate::IntConvert;
use core::arch::x86_64::{__m128, __m128d, __m256, __m256d};

/// `f32x4::to_u8`: `max(x, 0)`, round-to-nearest-even `cvtps`, `vpmovusdb`.
#[archmage::arcane(import_intrinsics)]
pub(super) fn to_u8_bytes_x4_v4(_token: archmage::X64V4Token, a: __m128) -> [u8; 4] {
    let bytes = _mm_cvtusepi32_epi8(_mm_cvtps_epi32(_mm_max_ps(a, _mm_setzero_ps())));
    (_mm_cvtsi128_si32(bytes) as u32).to_ne_bytes()
}

/// `f32x8::to_u8`: the same sequence at 256 bits, narrowing to 8 bytes.
#[archmage::arcane(import_intrinsics)]
pub(super) fn to_u8_bytes_x8_v4(_token: archmage::X64V4Token, a: __m256) -> [u8; 8] {
    let bytes = _mm256_cvtusepi32_epi8(_mm256_cvtps_epi32(_mm256_max_ps(a, _mm256_setzero_ps())));
    (_mm_cvtsi128_si64(bytes) as u64).to_ne_bytes()
}

/// `f32x4::store_4_rgba_u8`: two planes per 256-bit register, so `max`,
/// `cvtps` and `vpmovusdb` each cover two planes; one `pshufb` interleaves.
#[archmage::arcane(import_intrinsics)]
pub(super) fn store_rgba_bytes_x4_v4(
    _token: archmage::X64V4Token,
    r: __m128,
    g: __m128,
    b: __m128,
    a: __m128,
) -> [u8; 16] {
    let zero = _mm256_setzero_ps();
    let rg = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(r), g);
    let ba = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(b), a);
    let rg = _mm256_cvtusepi32_epi8(_mm256_cvtps_epi32(_mm256_max_ps(rg, zero)));
    let ba = _mm256_cvtusepi32_epi8(_mm256_cvtps_epi32(_mm256_max_ps(ba, zero)));
    // [R0-3, G0-3, B0-3, A0-3] -> interleaved RGBA pixels 0-3.
    let planes = _mm_unpacklo_epi64(rg, ba);
    let shuf = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    crate::simd_storage::cast(_mm_shuffle_epi8(planes, shuf))
}

/// `f32x4::store_4_rgb_u8`: pack R+G in one 256-bit `vpmovusdb` and B in one
/// 128-bit `vpmovusdb`, interleave 12 RGB bytes with one `pshufb`, and write 8+4B.
#[archmage::arcane(import_intrinsics)]
pub(super) fn store_rgb_bytes_x4_v4(
    _token: archmage::X64V4Token,
    r: __m128,
    g: __m128,
    b: __m128,
) -> [u8; 12] {
    let rg = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(r), g);
    let rg8 = _mm256_cvtusepi32_epi8(_mm256_cvtps_epi32(_mm256_max_ps(rg, _mm256_setzero_ps())));
    let b8 = _mm_cvtusepi32_epi8(_mm_cvtps_epi32(_mm_max_ps(b, _mm_setzero_ps())));
    let planes = _mm_unpacklo_epi64(rg8, b8);
    let shuf = _mm_setr_epi8(0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, -1, -1, -1, -1);
    let packed = _mm_shuffle_epi8(planes, shuf);
    let mut out = [0u8; 12];
    crate::simd_storage::store(
        _mm_cvtsi128_si64(packed) as u64,
        out.first_chunk_mut::<8>().unwrap(),
    );
    crate::simd_storage::store(
        _mm_extract_epi32::<2>(packed) as u32,
        out.last_chunk_mut::<4>().unwrap(),
    );
    out
}

// 0x04 and 0x05 preserve the sign of the selected operand.
#[archmage::arcane(import_intrinsics)]
pub(super) fn min_x4_v4(_token: archmage::X64V4Token, a: __m128, b: __m128) -> __m128 {
    _mm_range_ps::<0x04>(a, b)
}

#[archmage::arcane(import_intrinsics)]
pub(super) fn max_x4_v4(_token: archmage::X64V4Token, a: __m128, b: __m128) -> __m128 {
    _mm_range_ps::<0x05>(a, b)
}

#[archmage::arcane(import_intrinsics)]
pub(super) fn to_u16_lanes_x4_v4<const CFG: u32>(
    _token: archmage::X64V4Token,
    value: __m128,
) -> [u16; 4] {
    let opts = const { IntConvert::decode(CFG) };
    let x = if opts.nonnegative() {
        value
    } else {
        _mm_max_ps(value, _mm_setzero_ps())
    };
    let i = if opts.rounds() {
        _mm_cvtps_epu32(x)
    } else {
        _mm_cvttps_epu32(x)
    };
    let packed = _mm_cvtusepi32_epi16(i);
    crate::simd_storage::cast(_mm_cvtsi128_si64(packed) as u64)
}

#[archmage::arcane(import_intrinsics)]
pub(super) fn to_u8_lanes_x4_v4<const CFG: u32>(
    _token: archmage::X64V4Token,
    value: __m128,
) -> [u8; 4] {
    let opts = const { IntConvert::decode(CFG) };
    let x = if opts.nonnegative() {
        value
    } else {
        _mm_max_ps(value, _mm_setzero_ps())
    };
    let i = if opts.rounds() {
        _mm_cvtps_epu32(x)
    } else {
        _mm_cvttps_epu32(x)
    };
    let packed = _mm_cvtusepi32_epi8(i);
    (_mm_cvtsi128_si32(packed) as u32).to_ne_bytes()
}

#[archmage::arcane(import_intrinsics)]
pub(super) fn to_u32_lanes_x4_v4<const CFG: u32>(
    _token: archmage::X64V4Token,
    value: __m128,
) -> __m128 {
    let opts = const { IntConvert::decode(CFG) };
    let x = if opts.nonnegative() {
        value
    } else {
        _mm_max_ps(value, _mm_setzero_ps())
    };
    let i = if opts.rounds() {
        _mm_cvtps_epu32(x)
    } else {
        _mm_cvttps_epu32(x)
    };
    _mm_castsi128_ps(i)
}

#[archmage::arcane(import_intrinsics)]
pub(super) fn from_u32_lanes_x4_v4(_token: archmage::X64V4Token, value: __m128) -> __m128 {
    _mm512_castps512_ps128(_mm512_cvtepu32_ps(_mm512_castsi128_si512(
        _mm_castps_si128(value),
    )))
}

// 0x04 and 0x05 preserve the sign of the selected operand.
#[archmage::arcane(import_intrinsics)]
pub(super) fn min_x8_v4(_token: archmage::X64V4Token, a: __m256, b: __m256) -> __m256 {
    _mm256_range_ps::<0x04>(a, b)
}

#[archmage::arcane(import_intrinsics)]
pub(super) fn max_x8_v4(_token: archmage::X64V4Token, a: __m256, b: __m256) -> __m256 {
    _mm256_range_ps::<0x05>(a, b)
}

#[archmage::arcane(import_intrinsics)]
pub(super) fn to_u16_lanes_x8_v4<const CFG: u32>(
    _token: archmage::X64V4Token,
    value: __m256,
) -> [u16; 8] {
    let opts = const { IntConvert::decode(CFG) };
    let x = if opts.nonnegative() {
        value
    } else {
        _mm256_max_ps(value, _mm256_setzero_ps())
    };
    let i = if opts.rounds() {
        _mm256_cvtps_epu32(x)
    } else {
        _mm256_cvttps_epu32(x)
    };
    let packed = _mm256_cvtusepi32_epi16(i);
    crate::simd_storage::cast(packed)
}

#[archmage::arcane(import_intrinsics)]
pub(super) fn to_u8_lanes_x8_v4<const CFG: u32>(
    _token: archmage::X64V4Token,
    value: __m256,
) -> [u8; 8] {
    let opts = const { IntConvert::decode(CFG) };
    let x = if opts.nonnegative() {
        value
    } else {
        _mm256_max_ps(value, _mm256_setzero_ps())
    };
    let i = if opts.rounds() {
        _mm256_cvtps_epu32(x)
    } else {
        _mm256_cvttps_epu32(x)
    };
    let packed = _mm256_cvtusepi32_epi8(i);
    (_mm_cvtsi128_si64(packed) as u64).to_ne_bytes()
}

#[archmage::arcane(import_intrinsics)]
pub(super) fn to_u32_lanes_x8_v4<const CFG: u32>(
    _token: archmage::X64V4Token,
    value: __m256,
) -> __m256 {
    let opts = const { IntConvert::decode(CFG) };
    let x = if opts.nonnegative() {
        value
    } else {
        _mm256_max_ps(value, _mm256_setzero_ps())
    };
    let i = if opts.rounds() {
        _mm256_cvtps_epu32(x)
    } else {
        _mm256_cvttps_epu32(x)
    };
    _mm256_castsi256_ps(i)
}

#[archmage::arcane(import_intrinsics)]
pub(super) fn from_u32_lanes_x8_v4(_token: archmage::X64V4Token, value: __m256) -> __m256 {
    _mm512_castps512_ps256(_mm512_cvtepu32_ps(_mm512_castsi256_si512(
        _mm256_castps_si256(value),
    )))
}

#[archmage::arcane(import_intrinsics)]
pub(super) fn rcp_approx_f64x2_v4(_token: archmage::X64V4Token, value: __m128d) -> __m128d {
    _mm_rcp14_pd(value)
}

#[archmage::arcane(import_intrinsics)]
pub(super) fn rsqrt_approx_f64x2_v4(_token: archmage::X64V4Token, value: __m128d) -> __m128d {
    _mm_rsqrt14_pd(value)
}

#[archmage::arcane(import_intrinsics)]
pub(super) fn rcp_approx_f64x4_v4(_token: archmage::X64V4Token, value: __m256d) -> __m256d {
    _mm256_rcp14_pd(value)
}

#[archmage::arcane(import_intrinsics)]
pub(super) fn rsqrt_approx_f64x4_v4(_token: archmage::X64V4Token, value: __m256d) -> __m256d {
    _mm256_rsqrt14_pd(value)
}

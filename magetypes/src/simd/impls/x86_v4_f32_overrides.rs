//! The three `F32x4Backend` / `F32x8Backend` methods that the AVX-512 tokens
//! do **not** delegate to `X64V3Token` (see the generated
//! `x86_v4_f32_delegated.rs`, which forwards everything else).
//!
//! `to_u8` (both widths) and `f32x4::store_4_rgba_u8` use AVX-512VL
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

use core::arch::x86_64::{__m128, __m256};

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

//! Combined `core::arch` + `safe_unaligned_simd` intrinsics for `x86`.
//!
//! **Auto-generated** by `cargo xtask generate` — do not edit manually.
//!
//! This module glob-imports all of `core::arch::x86` (types, value intrinsics,
//! and unsafe memory ops), then explicitly re-exports the safe reference-based
//! memory operations from `safe_unaligned_simd`. Rust's name resolution rules
//! make explicit imports shadow glob imports, so `_mm256_loadu_ps` etc. resolve
//! to the safe versions automatically.

#[allow(unused_imports)]
pub use core::arch::x86::*;

#[allow(unused_imports)]
pub use safe_unaligned_simd::x86::{
    _mm_broadcast_ss, _mm_load_pd1, _mm_load_ps1, _mm_load_sd, _mm_load_ss, _mm_load1_pd,
    _mm_load1_ps, _mm_loadh_pd, _mm_loadl_epi64, _mm_loadl_pd, _mm_loadu_pd, _mm_loadu_ps,
    _mm_loadu_si16, _mm_loadu_si32, _mm_loadu_si64, _mm_loadu_si128, _mm_store_sd, _mm_store_ss,
    _mm_storeh_pd, _mm_storel_epi64, _mm_storel_pd, _mm_storeu_pd, _mm_storeu_ps, _mm_storeu_si16,
    _mm_storeu_si32, _mm_storeu_si64, _mm_storeu_si128, _mm256_broadcast_pd, _mm256_broadcast_ps,
    _mm256_broadcast_sd, _mm256_broadcast_ss, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_loadu_si256,
    _mm256_loadu2_m128, _mm256_loadu2_m128d, _mm256_loadu2_m128i, _mm256_storeu_pd,
    _mm256_storeu_ps, _mm256_storeu_si256, _mm256_storeu2_m128, _mm256_storeu2_m128d,
    _mm256_storeu2_m128i,
};

#[cfg(feature = "avx512")]
#[allow(unused_imports)]
pub use safe_unaligned_simd::x86::{
    _mm_loadu_epi8, _mm_loadu_epi16, _mm_loadu_epi32, _mm_loadu_epi64,
    _mm_mask_compressstoreu_epi8, _mm_mask_compressstoreu_epi16, _mm_mask_compressstoreu_epi32,
    _mm_mask_compressstoreu_epi64, _mm_mask_compressstoreu_pd, _mm_mask_compressstoreu_ps,
    _mm_mask_cvtepi16_storeu_epi8, _mm_mask_cvtepi32_storeu_epi8, _mm_mask_cvtepi32_storeu_epi16,
    _mm_mask_cvtepi64_storeu_epi8, _mm_mask_cvtepi64_storeu_epi16, _mm_mask_cvtepi64_storeu_epi32,
    _mm_mask_cvtsepi16_storeu_epi8, _mm_mask_cvtsepi32_storeu_epi8,
    _mm_mask_cvtsepi32_storeu_epi16, _mm_mask_cvtsepi64_storeu_epi8,
    _mm_mask_cvtsepi64_storeu_epi16, _mm_mask_cvtsepi64_storeu_epi32,
    _mm_mask_cvtusepi16_storeu_epi8, _mm_mask_cvtusepi32_storeu_epi8,
    _mm_mask_cvtusepi32_storeu_epi16, _mm_mask_cvtusepi64_storeu_epi8,
    _mm_mask_cvtusepi64_storeu_epi16, _mm_mask_cvtusepi64_storeu_epi32, _mm_mask_expandloadu_epi8,
    _mm_mask_expandloadu_epi16, _mm_mask_expandloadu_epi32, _mm_mask_expandloadu_epi64,
    _mm_mask_expandloadu_pd, _mm_mask_expandloadu_ps, _mm_mask_loadu_epi8, _mm_mask_loadu_epi16,
    _mm_mask_loadu_epi32, _mm_mask_loadu_epi64, _mm_mask_loadu_pd, _mm_mask_loadu_ps,
    _mm_mask_storeu_epi8, _mm_mask_storeu_epi16, _mm_mask_storeu_epi32, _mm_mask_storeu_epi64,
    _mm_mask_storeu_pd, _mm_mask_storeu_ps, _mm_maskz_expandloadu_epi8,
    _mm_maskz_expandloadu_epi16, _mm_maskz_expandloadu_epi32, _mm_maskz_expandloadu_epi64,
    _mm_maskz_expandloadu_pd, _mm_maskz_expandloadu_ps, _mm_maskz_loadu_epi8,
    _mm_maskz_loadu_epi16, _mm_maskz_loadu_epi32, _mm_maskz_loadu_epi64, _mm_maskz_loadu_pd,
    _mm_maskz_loadu_ps, _mm_storeu_epi8, _mm_storeu_epi16, _mm_storeu_epi32, _mm_storeu_epi64,
    _mm256_loadu_epi8, _mm256_loadu_epi16, _mm256_loadu_epi32, _mm256_loadu_epi64,
    _mm256_mask_compressstoreu_epi8, _mm256_mask_compressstoreu_epi16,
    _mm256_mask_compressstoreu_epi32, _mm256_mask_compressstoreu_epi64,
    _mm256_mask_compressstoreu_pd, _mm256_mask_compressstoreu_ps, _mm256_mask_cvtepi16_storeu_epi8,
    _mm256_mask_cvtepi32_storeu_epi8, _mm256_mask_cvtepi32_storeu_epi16,
    _mm256_mask_cvtepi64_storeu_epi8, _mm256_mask_cvtepi64_storeu_epi16,
    _mm256_mask_cvtepi64_storeu_epi32, _mm256_mask_cvtsepi16_storeu_epi8,
    _mm256_mask_cvtsepi32_storeu_epi8, _mm256_mask_cvtsepi32_storeu_epi16,
    _mm256_mask_cvtsepi64_storeu_epi8, _mm256_mask_cvtsepi64_storeu_epi16,
    _mm256_mask_cvtsepi64_storeu_epi32, _mm256_mask_cvtusepi16_storeu_epi8,
    _mm256_mask_cvtusepi32_storeu_epi8, _mm256_mask_cvtusepi32_storeu_epi16,
    _mm256_mask_cvtusepi64_storeu_epi8, _mm256_mask_cvtusepi64_storeu_epi16,
    _mm256_mask_cvtusepi64_storeu_epi32, _mm256_mask_expandloadu_epi8,
    _mm256_mask_expandloadu_epi16, _mm256_mask_expandloadu_epi32, _mm256_mask_expandloadu_epi64,
    _mm256_mask_expandloadu_pd, _mm256_mask_expandloadu_ps, _mm256_mask_loadu_epi8,
    _mm256_mask_loadu_epi16, _mm256_mask_loadu_epi32, _mm256_mask_loadu_epi64,
    _mm256_mask_loadu_pd, _mm256_mask_loadu_ps, _mm256_mask_storeu_epi8, _mm256_mask_storeu_epi16,
    _mm256_mask_storeu_epi32, _mm256_mask_storeu_epi64, _mm256_mask_storeu_pd,
    _mm256_mask_storeu_ps, _mm256_maskz_expandloadu_epi8, _mm256_maskz_expandloadu_epi16,
    _mm256_maskz_expandloadu_epi32, _mm256_maskz_expandloadu_epi64, _mm256_maskz_expandloadu_pd,
    _mm256_maskz_expandloadu_ps, _mm256_maskz_loadu_epi8, _mm256_maskz_loadu_epi16,
    _mm256_maskz_loadu_epi32, _mm256_maskz_loadu_epi64, _mm256_maskz_loadu_pd,
    _mm256_maskz_loadu_ps, _mm256_storeu_epi8, _mm256_storeu_epi16, _mm256_storeu_epi32,
    _mm256_storeu_epi64, _mm512_loadu_epi8, _mm512_loadu_epi16, _mm512_loadu_epi32,
    _mm512_loadu_epi64, _mm512_loadu_pd, _mm512_loadu_ps, _mm512_loadu_si512,
    _mm512_mask_compressstoreu_epi8, _mm512_mask_compressstoreu_epi16,
    _mm512_mask_compressstoreu_epi32, _mm512_mask_compressstoreu_epi64,
    _mm512_mask_compressstoreu_pd, _mm512_mask_compressstoreu_ps, _mm512_mask_cvtepi16_storeu_epi8,
    _mm512_mask_cvtepi32_storeu_epi8, _mm512_mask_cvtepi32_storeu_epi16,
    _mm512_mask_cvtepi64_storeu_epi8, _mm512_mask_cvtepi64_storeu_epi16,
    _mm512_mask_cvtepi64_storeu_epi32, _mm512_mask_cvtsepi16_storeu_epi8,
    _mm512_mask_cvtsepi32_storeu_epi8, _mm512_mask_cvtsepi32_storeu_epi16,
    _mm512_mask_cvtsepi64_storeu_epi8, _mm512_mask_cvtsepi64_storeu_epi16,
    _mm512_mask_cvtsepi64_storeu_epi32, _mm512_mask_cvtusepi16_storeu_epi8,
    _mm512_mask_cvtusepi32_storeu_epi8, _mm512_mask_cvtusepi32_storeu_epi16,
    _mm512_mask_cvtusepi64_storeu_epi8, _mm512_mask_cvtusepi64_storeu_epi16,
    _mm512_mask_cvtusepi64_storeu_epi32, _mm512_mask_expandloadu_epi8,
    _mm512_mask_expandloadu_epi16, _mm512_mask_expandloadu_epi32, _mm512_mask_expandloadu_epi64,
    _mm512_mask_expandloadu_pd, _mm512_mask_expandloadu_ps, _mm512_mask_loadu_epi8,
    _mm512_mask_loadu_epi16, _mm512_mask_loadu_epi32, _mm512_mask_loadu_epi64,
    _mm512_mask_loadu_pd, _mm512_mask_loadu_ps, _mm512_mask_storeu_epi8, _mm512_mask_storeu_epi16,
    _mm512_mask_storeu_epi32, _mm512_mask_storeu_epi64, _mm512_mask_storeu_pd,
    _mm512_mask_storeu_ps, _mm512_maskz_expandloadu_epi8, _mm512_maskz_expandloadu_epi16,
    _mm512_maskz_expandloadu_epi32, _mm512_maskz_expandloadu_epi64, _mm512_maskz_expandloadu_pd,
    _mm512_maskz_expandloadu_ps, _mm512_maskz_loadu_epi8, _mm512_maskz_loadu_epi16,
    _mm512_maskz_loadu_epi32, _mm512_maskz_loadu_epi64, _mm512_maskz_loadu_pd,
    _mm512_maskz_loadu_ps, _mm512_storeu_epi8, _mm512_storeu_epi16, _mm512_storeu_epi32,
    _mm512_storeu_epi64, _mm512_storeu_pd, _mm512_storeu_ps, _mm512_storeu_si512,
};

# Archmage Intrinsic Reference

This document lists all safe SIMD intrinsics available through `archmage::mem`.

**Auto-generated** by `cargo xtask generate` - do not edit manually.

## Overview

| Architecture | Module | Functions | Required Feature |
|-------------|--------|-----------|------------------|
| x86_64 | `avx` | 17 | - |
| x86_64 | `modern` | 6 | `avx512` |
| x86_64 | `modern_vl` | 12 | `avx512` |
| x86_64 | `v4` | 49 | `avx512` |
| x86_64 | `v4_bw` | 13 | `avx512` |
| x86_64 | `v4_bw_vl` | 26 | `avx512` |
| x86_64 | `v4_vl` | 86 | `avx512` |
| aarch64 | `neon` | 240 | - |

## x86_64 Intrinsics

### `archmage::mem::avx`

Token: `Has256BitSimd`

| Function | Category | Intel Docs |
|----------|----------|------------|
| `_mm256_broadcast_pd` | Other | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_broadcast_pd) |
| `_mm256_broadcast_ps` | Other | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_broadcast_ps) |
| `_mm256_broadcast_sd` | Other | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_broadcast_sd) |
| `_mm_broadcast_ss` | Other | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_broadcast_ss) |
| `_mm256_broadcast_ss` | Other | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_broadcast_ss) |
| `_mm256_loadu_pd` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_loadu_pd) |
| `_mm256_loadu_ps` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_loadu_ps) |
| `_mm256_loadu_si256` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_loadu_si256) |
| `_mm256_loadu2_m128` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_loadu2_m128) |
| `_mm256_loadu2_m128d` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_loadu2_m128d) |
| `_mm256_loadu2_m128i` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_loadu2_m128i) |
| `_mm256_storeu_pd` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_storeu_pd) |
| `_mm256_storeu_ps` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_storeu_ps) |
| `_mm256_storeu_si256` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_storeu_si256) |
| `_mm256_storeu2_m128` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_storeu2_m128) |
| `_mm256_storeu2_m128d` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_storeu2_m128d) |
| `_mm256_storeu2_m128i` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_storeu2_m128i) |

### `archmage::mem::modern` (requires `avx512` feature)

Token: `Avx512ModernToken`

| Function | Category | Intel Docs |
|----------|----------|------------|
| `_mm512_mask_expandloadu_epi16` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_expandloadu_epi16) |
| `_mm512_maskz_expandloadu_epi16` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_maskz_expandloadu_epi16) |
| `_mm512_mask_expandloadu_epi8` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_expandloadu_epi8) |
| `_mm512_maskz_expandloadu_epi8` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_maskz_expandloadu_epi8) |
| `_mm512_mask_compressstoreu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_compressstoreu_epi16) |
| `_mm512_mask_compressstoreu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_compressstoreu_epi8) |

### `archmage::mem::modern_vl` (requires `avx512` feature)

Token: `Avx512ModernToken`

| Function | Category | Intel Docs |
|----------|----------|------------|
| `_mm_mask_expandloadu_epi16` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_expandloadu_epi16) |
| `_mm_maskz_expandloadu_epi16` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_maskz_expandloadu_epi16) |
| `_mm256_mask_expandloadu_epi16` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_expandloadu_epi16) |
| `_mm256_maskz_expandloadu_epi16` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_maskz_expandloadu_epi16) |
| `_mm_mask_expandloadu_epi8` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_expandloadu_epi8) |
| `_mm_maskz_expandloadu_epi8` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_maskz_expandloadu_epi8) |
| `_mm256_mask_expandloadu_epi8` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_expandloadu_epi8) |
| `_mm256_maskz_expandloadu_epi8` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_maskz_expandloadu_epi8) |
| `_mm_mask_compressstoreu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_compressstoreu_epi16) |
| `_mm256_mask_compressstoreu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_compressstoreu_epi16) |
| `_mm_mask_compressstoreu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_compressstoreu_epi8) |
| `_mm256_mask_compressstoreu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_compressstoreu_epi8) |

### `archmage::mem::v4` (requires `avx512` feature)

Token: `HasX64V4`

| Function | Category | Intel Docs |
|----------|----------|------------|
| `_mm512_mask_expandloadu_epi32` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_expandloadu_epi32) |
| `_mm512_maskz_expandloadu_epi32` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_maskz_expandloadu_epi32) |
| `_mm512_mask_expandloadu_epi64` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_expandloadu_epi64) |
| `_mm512_maskz_expandloadu_epi64` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_maskz_expandloadu_epi64) |
| `_mm512_mask_expandloadu_pd` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_expandloadu_pd) |
| `_mm512_maskz_expandloadu_pd` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_maskz_expandloadu_pd) |
| `_mm512_mask_expandloadu_ps` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_expandloadu_ps) |
| `_mm512_maskz_expandloadu_ps` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_maskz_expandloadu_ps) |
| `_mm512_loadu_epi32` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_loadu_epi32) |
| `_mm512_mask_loadu_epi32` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_loadu_epi32) |
| `_mm512_maskz_loadu_epi32` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_maskz_loadu_epi32) |
| `_mm512_loadu_epi64` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_loadu_epi64) |
| `_mm512_mask_loadu_epi64` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_loadu_epi64) |
| `_mm512_maskz_loadu_epi64` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_maskz_loadu_epi64) |
| `_mm512_loadu_pd` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_loadu_pd) |
| `_mm512_mask_loadu_pd` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_loadu_pd) |
| `_mm512_maskz_loadu_pd` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_maskz_loadu_pd) |
| `_mm512_loadu_ps` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_loadu_ps) |
| `_mm512_mask_loadu_ps` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_loadu_ps) |
| `_mm512_maskz_loadu_ps` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_maskz_loadu_ps) |
| `_mm512_loadu_si512` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_loadu_si512) |
| `_mm512_mask_compressstoreu_epi32` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_compressstoreu_epi32) |
| `_mm512_mask_compressstoreu_epi64` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_compressstoreu_epi64) |
| `_mm512_mask_compressstoreu_pd` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_compressstoreu_pd) |
| `_mm512_mask_compressstoreu_ps` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_compressstoreu_ps) |
| `_mm512_mask_cvtepi32_storeu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_cvtepi32_storeu_epi16) |
| `_mm512_mask_cvtepi32_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_cvtepi32_storeu_epi8) |
| `_mm512_mask_cvtepi64_storeu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_cvtepi64_storeu_epi16) |
| `_mm512_mask_cvtepi64_storeu_epi32` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_cvtepi64_storeu_epi32) |
| `_mm512_mask_cvtepi64_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_cvtepi64_storeu_epi8) |
| `_mm512_mask_cvtsepi32_storeu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_cvtsepi32_storeu_epi16) |
| `_mm512_mask_cvtsepi32_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_cvtsepi32_storeu_epi8) |
| `_mm512_mask_cvtsepi64_storeu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_cvtsepi64_storeu_epi16) |
| `_mm512_mask_cvtsepi64_storeu_epi32` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_cvtsepi64_storeu_epi32) |
| `_mm512_mask_cvtsepi64_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_cvtsepi64_storeu_epi8) |
| `_mm512_mask_cvtusepi32_storeu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_cvtusepi32_storeu_epi16) |
| `_mm512_mask_cvtusepi32_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_cvtusepi32_storeu_epi8) |
| `_mm512_mask_cvtusepi64_storeu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_cvtusepi64_storeu_epi16) |
| `_mm512_mask_cvtusepi64_storeu_epi32` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_cvtusepi64_storeu_epi32) |
| `_mm512_mask_cvtusepi64_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_cvtusepi64_storeu_epi8) |
| `_mm512_mask_storeu_epi32` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_storeu_epi32) |
| `_mm512_storeu_epi32` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_storeu_epi32) |
| `_mm512_mask_storeu_epi64` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_storeu_epi64) |
| `_mm512_storeu_epi64` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_storeu_epi64) |
| `_mm512_mask_storeu_pd` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_storeu_pd) |
| `_mm512_storeu_pd` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_storeu_pd) |
| `_mm512_mask_storeu_ps` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_storeu_ps) |
| `_mm512_storeu_ps` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_storeu_ps) |
| `_mm512_storeu_si512` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_storeu_si512) |

### `archmage::mem::v4_bw` (requires `avx512` feature)

Token: `HasX64V4`

| Function | Category | Intel Docs |
|----------|----------|------------|
| `_mm512_loadu_epi16` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_loadu_epi16) |
| `_mm512_mask_loadu_epi16` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_loadu_epi16) |
| `_mm512_maskz_loadu_epi16` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_maskz_loadu_epi16) |
| `_mm512_loadu_epi8` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_loadu_epi8) |
| `_mm512_mask_loadu_epi8` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_loadu_epi8) |
| `_mm512_maskz_loadu_epi8` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_maskz_loadu_epi8) |
| `_mm512_mask_cvtepi16_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_cvtepi16_storeu_epi8) |
| `_mm512_mask_cvtsepi16_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_cvtsepi16_storeu_epi8) |
| `_mm512_mask_cvtusepi16_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_cvtusepi16_storeu_epi8) |
| `_mm512_mask_storeu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_storeu_epi16) |
| `_mm512_storeu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_storeu_epi16) |
| `_mm512_mask_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_mask_storeu_epi8) |
| `_mm512_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_storeu_epi8) |

### `archmage::mem::v4_bw_vl` (requires `avx512` feature)

Token: `HasX64V4`

| Function | Category | Intel Docs |
|----------|----------|------------|
| `_mm_loadu_epi16` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_loadu_epi16) |
| `_mm_mask_loadu_epi16` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_loadu_epi16) |
| `_mm_maskz_loadu_epi16` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_maskz_loadu_epi16) |
| `_mm256_loadu_epi16` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_loadu_epi16) |
| `_mm256_mask_loadu_epi16` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_loadu_epi16) |
| `_mm256_maskz_loadu_epi16` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_maskz_loadu_epi16) |
| `_mm_loadu_epi8` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_loadu_epi8) |
| `_mm_mask_loadu_epi8` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_loadu_epi8) |
| `_mm_maskz_loadu_epi8` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_maskz_loadu_epi8) |
| `_mm256_loadu_epi8` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_loadu_epi8) |
| `_mm256_mask_loadu_epi8` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_loadu_epi8) |
| `_mm256_maskz_loadu_epi8` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_maskz_loadu_epi8) |
| `_mm_mask_cvtepi16_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_cvtepi16_storeu_epi8) |
| `_mm256_mask_cvtepi16_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_cvtepi16_storeu_epi8) |
| `_mm_mask_cvtsepi16_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_cvtsepi16_storeu_epi8) |
| `_mm256_mask_cvtsepi16_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_cvtsepi16_storeu_epi8) |
| `_mm_mask_cvtusepi16_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_cvtusepi16_storeu_epi8) |
| `_mm256_mask_cvtusepi16_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_cvtusepi16_storeu_epi8) |
| `_mm_mask_storeu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_storeu_epi16) |
| `_mm_storeu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_storeu_epi16) |
| `_mm256_mask_storeu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_storeu_epi16) |
| `_mm256_storeu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_storeu_epi16) |
| `_mm_mask_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_storeu_epi8) |
| `_mm_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_storeu_epi8) |
| `_mm256_mask_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_storeu_epi8) |
| `_mm256_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_storeu_epi8) |

### `archmage::mem::v4_vl` (requires `avx512` feature)

Token: `HasX64V4`

| Function | Category | Intel Docs |
|----------|----------|------------|
| `_mm_mask_expandloadu_epi32` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_expandloadu_epi32) |
| `_mm_maskz_expandloadu_epi32` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_maskz_expandloadu_epi32) |
| `_mm256_mask_expandloadu_epi32` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_expandloadu_epi32) |
| `_mm256_maskz_expandloadu_epi32` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_maskz_expandloadu_epi32) |
| `_mm_mask_expandloadu_epi64` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_expandloadu_epi64) |
| `_mm_maskz_expandloadu_epi64` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_maskz_expandloadu_epi64) |
| `_mm256_mask_expandloadu_epi64` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_expandloadu_epi64) |
| `_mm256_maskz_expandloadu_epi64` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_maskz_expandloadu_epi64) |
| `_mm_mask_expandloadu_pd` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_expandloadu_pd) |
| `_mm_maskz_expandloadu_pd` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_maskz_expandloadu_pd) |
| `_mm256_mask_expandloadu_pd` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_expandloadu_pd) |
| `_mm256_maskz_expandloadu_pd` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_maskz_expandloadu_pd) |
| `_mm_mask_expandloadu_ps` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_expandloadu_ps) |
| `_mm_maskz_expandloadu_ps` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_maskz_expandloadu_ps) |
| `_mm256_mask_expandloadu_ps` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_expandloadu_ps) |
| `_mm256_maskz_expandloadu_ps` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_maskz_expandloadu_ps) |
| `_mm_loadu_epi32` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_loadu_epi32) |
| `_mm_mask_loadu_epi32` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_loadu_epi32) |
| `_mm_maskz_loadu_epi32` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_maskz_loadu_epi32) |
| `_mm256_loadu_epi32` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_loadu_epi32) |
| `_mm256_mask_loadu_epi32` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_loadu_epi32) |
| `_mm256_maskz_loadu_epi32` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_maskz_loadu_epi32) |
| `_mm_loadu_epi64` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_loadu_epi64) |
| `_mm_mask_loadu_epi64` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_loadu_epi64) |
| `_mm_maskz_loadu_epi64` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_maskz_loadu_epi64) |
| `_mm256_loadu_epi64` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_loadu_epi64) |
| `_mm256_mask_loadu_epi64` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_loadu_epi64) |
| `_mm256_maskz_loadu_epi64` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_maskz_loadu_epi64) |
| `_mm_mask_loadu_pd` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_loadu_pd) |
| `_mm_maskz_loadu_pd` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_maskz_loadu_pd) |
| `_mm256_mask_loadu_pd` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_loadu_pd) |
| `_mm256_maskz_loadu_pd` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_maskz_loadu_pd) |
| `_mm_mask_loadu_ps` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_loadu_ps) |
| `_mm_maskz_loadu_ps` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_maskz_loadu_ps) |
| `_mm256_mask_loadu_ps` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_loadu_ps) |
| `_mm256_maskz_loadu_ps` | Load | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_maskz_loadu_ps) |
| `_mm_mask_compressstoreu_epi32` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_compressstoreu_epi32) |
| `_mm256_mask_compressstoreu_epi32` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_compressstoreu_epi32) |
| `_mm_mask_compressstoreu_epi64` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_compressstoreu_epi64) |
| `_mm256_mask_compressstoreu_epi64` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_compressstoreu_epi64) |
| `_mm_mask_compressstoreu_pd` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_compressstoreu_pd) |
| `_mm256_mask_compressstoreu_pd` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_compressstoreu_pd) |
| `_mm_mask_compressstoreu_ps` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_compressstoreu_ps) |
| `_mm256_mask_compressstoreu_ps` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_compressstoreu_ps) |
| `_mm_mask_cvtepi32_storeu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_cvtepi32_storeu_epi16) |
| `_mm256_mask_cvtepi32_storeu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_cvtepi32_storeu_epi16) |
| `_mm_mask_cvtepi32_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_cvtepi32_storeu_epi8) |
| `_mm256_mask_cvtepi32_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_cvtepi32_storeu_epi8) |
| `_mm_mask_cvtepi64_storeu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_cvtepi64_storeu_epi16) |
| `_mm256_mask_cvtepi64_storeu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_cvtepi64_storeu_epi16) |
| `_mm_mask_cvtepi64_storeu_epi32` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_cvtepi64_storeu_epi32) |
| `_mm256_mask_cvtepi64_storeu_epi32` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_cvtepi64_storeu_epi32) |
| `_mm_mask_cvtepi64_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_cvtepi64_storeu_epi8) |
| `_mm256_mask_cvtepi64_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_cvtepi64_storeu_epi8) |
| `_mm_mask_cvtsepi32_storeu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_cvtsepi32_storeu_epi16) |
| `_mm256_mask_cvtsepi32_storeu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_cvtsepi32_storeu_epi16) |
| `_mm_mask_cvtsepi32_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_cvtsepi32_storeu_epi8) |
| `_mm256_mask_cvtsepi32_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_cvtsepi32_storeu_epi8) |
| `_mm_mask_cvtsepi64_storeu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_cvtsepi64_storeu_epi16) |
| `_mm256_mask_cvtsepi64_storeu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_cvtsepi64_storeu_epi16) |
| `_mm_mask_cvtsepi64_storeu_epi32` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_cvtsepi64_storeu_epi32) |
| `_mm256_mask_cvtsepi64_storeu_epi32` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_cvtsepi64_storeu_epi32) |
| `_mm_mask_cvtsepi64_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_cvtsepi64_storeu_epi8) |
| `_mm256_mask_cvtsepi64_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_cvtsepi64_storeu_epi8) |
| `_mm_mask_cvtusepi32_storeu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_cvtusepi32_storeu_epi16) |
| `_mm256_mask_cvtusepi32_storeu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_cvtusepi32_storeu_epi16) |
| `_mm_mask_cvtusepi32_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_cvtusepi32_storeu_epi8) |
| `_mm256_mask_cvtusepi32_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_cvtusepi32_storeu_epi8) |
| `_mm_mask_cvtusepi64_storeu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_cvtusepi64_storeu_epi16) |
| `_mm256_mask_cvtusepi64_storeu_epi16` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_cvtusepi64_storeu_epi16) |
| `_mm_mask_cvtusepi64_storeu_epi32` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_cvtusepi64_storeu_epi32) |
| `_mm256_mask_cvtusepi64_storeu_epi32` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_cvtusepi64_storeu_epi32) |
| `_mm_mask_cvtusepi64_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_cvtusepi64_storeu_epi8) |
| `_mm256_mask_cvtusepi64_storeu_epi8` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_cvtusepi64_storeu_epi8) |
| `_mm_mask_storeu_epi32` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_storeu_epi32) |
| `_mm_storeu_epi32` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_storeu_epi32) |
| `_mm256_mask_storeu_epi32` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_storeu_epi32) |
| `_mm256_storeu_epi32` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_storeu_epi32) |
| `_mm_mask_storeu_epi64` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_storeu_epi64) |
| `_mm_storeu_epi64` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_storeu_epi64) |
| `_mm256_mask_storeu_epi64` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_storeu_epi64) |
| `_mm256_storeu_epi64` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_storeu_epi64) |
| `_mm_mask_storeu_pd` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_storeu_pd) |
| `_mm256_mask_storeu_pd` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_storeu_pd) |
| `_mm_mask_storeu_ps` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm_mask_storeu_ps) |
| `_mm256_mask_storeu_ps` | Store | [Intel Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm256_mask_storeu_ps) |

## AArch64 Intrinsics

### `archmage::mem::neon`

Token: `HasNeon`

| Function | Category | ARM Docs |
|----------|----------|----------|
| `vld1_u8` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_u8) |
| `vld1_s8` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_s8) |
| `vld1_u16` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_u16) |
| `vld1_s16` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_s16) |
| `vld1_u32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_u32) |
| `vld1_s32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_s32) |
| `vld1_f32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_f32) |
| `vld1_u64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_u64) |
| `vld1_s64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_s64) |
| `vld1_f64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_f64) |
| `vld1_u8_x2` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_u8_x2) |
| `vld1_s8_x2` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_s8_x2) |
| `vld1_u16_x2` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_u16_x2) |
| `vld1_s16_x2` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_s16_x2) |
| `vld1_u32_x2` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_u32_x2) |
| `vld1_s32_x2` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_s32_x2) |
| `vld1_f32_x2` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_f32_x2) |
| `vld1_u64_x2` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_u64_x2) |
| `vld1_s64_x2` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_s64_x2) |
| `vld1_f64_x2` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_f64_x2) |
| `vld1_u8_x3` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_u8_x3) |
| `vld1_s8_x3` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_s8_x3) |
| `vld1_u16_x3` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_u16_x3) |
| `vld1_s16_x3` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_s16_x3) |
| `vld1_u32_x3` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_u32_x3) |
| `vld1_s32_x3` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_s32_x3) |
| `vld1_f32_x3` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_f32_x3) |
| `vld1_u64_x3` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_u64_x3) |
| `vld1_s64_x3` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_s64_x3) |
| `vld1_f64_x3` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_f64_x3) |
| `vld1_u8_x4` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_u8_x4) |
| `vld1_s8_x4` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_s8_x4) |
| `vld1_u16_x4` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_u16_x4) |
| `vld1_s16_x4` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_s16_x4) |
| `vld1_u32_x4` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_u32_x4) |
| `vld1_s32_x4` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_s32_x4) |
| `vld1_f32_x4` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_f32_x4) |
| `vld1_u64_x4` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_u64_x4) |
| `vld1_s64_x4` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_s64_x4) |
| `vld1_f64_x4` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_f64_x4) |
| `vld1q_u8` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_u8) |
| `vld1q_s8` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_s8) |
| `vld1q_u16` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_u16) |
| `vld1q_s16` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_s16) |
| `vld1q_u32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_u32) |
| `vld1q_s32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_s32) |
| `vld1q_f32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_f32) |
| `vld1q_u64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_u64) |
| `vld1q_s64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_s64) |
| `vld1q_f64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_f64) |
| `vld1q_u8_x2` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_u8_x2) |
| `vld1q_s8_x2` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_s8_x2) |
| `vld1q_u16_x2` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_u16_x2) |
| `vld1q_s16_x2` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_s16_x2) |
| `vld1q_u32_x2` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_u32_x2) |
| `vld1q_s32_x2` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_s32_x2) |
| `vld1q_f32_x2` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_f32_x2) |
| `vld1q_u64_x2` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_u64_x2) |
| `vld1q_s64_x2` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_s64_x2) |
| `vld1q_f64_x2` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_f64_x2) |
| `vld1q_u8_x3` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_u8_x3) |
| `vld1q_s8_x3` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_s8_x3) |
| `vld1q_u16_x3` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_u16_x3) |
| `vld1q_s16_x3` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_s16_x3) |
| `vld1q_u32_x3` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_u32_x3) |
| `vld1q_s32_x3` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_s32_x3) |
| `vld1q_f32_x3` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_f32_x3) |
| `vld1q_u64_x3` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_u64_x3) |
| `vld1q_s64_x3` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_s64_x3) |
| `vld1q_f64_x3` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_f64_x3) |
| `vld1q_u8_x4` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_u8_x4) |
| `vld1q_s8_x4` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_s8_x4) |
| `vld1q_u16_x4` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_u16_x4) |
| `vld1q_s16_x4` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_s16_x4) |
| `vld1q_u32_x4` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_u32_x4) |
| `vld1q_s32_x4` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_s32_x4) |
| `vld1q_f32_x4` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_f32_x4) |
| `vld1q_u64_x4` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_u64_x4) |
| `vld1q_s64_x4` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_s64_x4) |
| `vld1q_f64_x4` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_f64_x4) |
| `vst1_u8` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_u8) |
| `vst1_s8` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_s8) |
| `vst1_u16` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_u16) |
| `vst1_s16` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_s16) |
| `vst1_u32` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_u32) |
| `vst1_s32` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_s32) |
| `vst1_f32` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_f32) |
| `vst1_u64` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_u64) |
| `vst1_s64` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_s64) |
| `vst1_f64` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_f64) |
| `vst1_u8_x2` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_u8_x2) |
| `vst1_s8_x2` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_s8_x2) |
| `vst1_u16_x2` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_u16_x2) |
| `vst1_s16_x2` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_s16_x2) |
| `vst1_u32_x2` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_u32_x2) |
| `vst1_s32_x2` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_s32_x2) |
| `vst1_f32_x2` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_f32_x2) |
| `vst1_u64_x2` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_u64_x2) |
| `vst1_s64_x2` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_s64_x2) |
| `vst1_f64_x2` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_f64_x2) |
| `vst1_u8_x3` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_u8_x3) |
| `vst1_s8_x3` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_s8_x3) |
| `vst1_u16_x3` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_u16_x3) |
| `vst1_s16_x3` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_s16_x3) |
| `vst1_u32_x3` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_u32_x3) |
| `vst1_s32_x3` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_s32_x3) |
| `vst1_f32_x3` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_f32_x3) |
| `vst1_u64_x3` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_u64_x3) |
| `vst1_s64_x3` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_s64_x3) |
| `vst1_f64_x3` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_f64_x3) |
| `vst1_u8_x4` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_u8_x4) |
| `vst1_s8_x4` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_s8_x4) |
| `vst1_u16_x4` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_u16_x4) |
| `vst1_s16_x4` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_s16_x4) |
| `vst1_u32_x4` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_u32_x4) |
| `vst1_s32_x4` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_s32_x4) |
| `vst1_f32_x4` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_f32_x4) |
| `vst1_u64_x4` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_u64_x4) |
| `vst1_s64_x4` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_s64_x4) |
| `vst1_f64_x4` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1_f64_x4) |
| `vst1q_u8` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_u8) |
| `vst1q_s8` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_s8) |
| `vst1q_u16` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_u16) |
| `vst1q_s16` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_s16) |
| `vst1q_u32` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_u32) |
| `vst1q_s32` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_s32) |
| `vst1q_f32` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_f32) |
| `vst1q_u64` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_u64) |
| `vst1q_s64` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_s64) |
| `vst1q_f64` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_f64) |
| `vst1q_u8_x2` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_u8_x2) |
| `vst1q_s8_x2` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_s8_x2) |
| `vst1q_u16_x2` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_u16_x2) |
| `vst1q_s16_x2` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_s16_x2) |
| `vst1q_u32_x2` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_u32_x2) |
| `vst1q_s32_x2` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_s32_x2) |
| `vst1q_f32_x2` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_f32_x2) |
| `vst1q_u64_x2` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_u64_x2) |
| `vst1q_s64_x2` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_s64_x2) |
| `vst1q_f64_x2` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_f64_x2) |
| `vst1q_u8_x3` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_u8_x3) |
| `vst1q_s8_x3` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_s8_x3) |
| `vst1q_u16_x3` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_u16_x3) |
| `vst1q_s16_x3` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_s16_x3) |
| `vst1q_u32_x3` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_u32_x3) |
| `vst1q_s32_x3` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_s32_x3) |
| `vst1q_f32_x3` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_f32_x3) |
| `vst1q_u64_x3` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_u64_x3) |
| `vst1q_s64_x3` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_s64_x3) |
| `vst1q_f64_x3` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_f64_x3) |
| `vst1q_u8_x4` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_u8_x4) |
| `vst1q_s8_x4` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_s8_x4) |
| `vst1q_u16_x4` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_u16_x4) |
| `vst1q_s16_x4` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_s16_x4) |
| `vst1q_u32_x4` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_u32_x4) |
| `vst1q_s32_x4` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_s32_x4) |
| `vst1q_f32_x4` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_f32_x4) |
| `vst1q_u64_x4` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_u64_x4) |
| `vst1q_s64_x4` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_s64_x4) |
| `vst1q_f64_x4` | Store | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vst1q_f64_x4) |
| `vld1_dup_s8` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_dup_s8) |
| `vld2_dup_s8` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld2_dup_s8) |
| `vld3_dup_s8` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld3_dup_s8) |
| `vld4_dup_s8` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld4_dup_s8) |
| `vld1_dup_u8` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_dup_u8) |
| `vld2_dup_u8` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld2_dup_u8) |
| `vld3_dup_u8` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld3_dup_u8) |
| `vld4_dup_u8` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld4_dup_u8) |
| `vld1_dup_s16` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_dup_s16) |
| `vld2_dup_s16` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld2_dup_s16) |
| `vld3_dup_s16` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld3_dup_s16) |
| `vld4_dup_s16` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld4_dup_s16) |
| `vld1_dup_u16` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_dup_u16) |
| `vld2_dup_u16` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld2_dup_u16) |
| `vld3_dup_u16` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld3_dup_u16) |
| `vld4_dup_u16` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld4_dup_u16) |
| `vld1_dup_s32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_dup_s32) |
| `vld2_dup_s32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld2_dup_s32) |
| `vld3_dup_s32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld3_dup_s32) |
| `vld4_dup_s32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld4_dup_s32) |
| `vld1_dup_u32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_dup_u32) |
| `vld2_dup_u32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld2_dup_u32) |
| `vld3_dup_u32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld3_dup_u32) |
| `vld4_dup_u32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld4_dup_u32) |
| `vld1_dup_f32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_dup_f32) |
| `vld2_dup_f32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld2_dup_f32) |
| `vld3_dup_f32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld3_dup_f32) |
| `vld4_dup_f32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld4_dup_f32) |
| `vld1_dup_s64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_dup_s64) |
| `vld2_dup_s64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld2_dup_s64) |
| `vld3_dup_s64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld3_dup_s64) |
| `vld4_dup_s64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld4_dup_s64) |
| `vld1_dup_u64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_dup_u64) |
| `vld2_dup_u64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld2_dup_u64) |
| `vld3_dup_u64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld3_dup_u64) |
| `vld4_dup_u64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld4_dup_u64) |
| `vld1_dup_f64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1_dup_f64) |
| `vld2_dup_f64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld2_dup_f64) |
| `vld3_dup_f64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld3_dup_f64) |
| `vld4_dup_f64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld4_dup_f64) |
| `vld1q_dup_s8` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_dup_s8) |
| `vld2q_dup_s8` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld2q_dup_s8) |
| `vld3q_dup_s8` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld3q_dup_s8) |
| `vld4q_dup_s8` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld4q_dup_s8) |
| `vld1q_dup_u8` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_dup_u8) |
| `vld2q_dup_u8` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld2q_dup_u8) |
| `vld3q_dup_u8` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld3q_dup_u8) |
| `vld4q_dup_u8` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld4q_dup_u8) |
| `vld1q_dup_s16` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_dup_s16) |
| `vld2q_dup_s16` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld2q_dup_s16) |
| `vld3q_dup_s16` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld3q_dup_s16) |
| `vld4q_dup_s16` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld4q_dup_s16) |
| `vld1q_dup_u16` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_dup_u16) |
| `vld2q_dup_u16` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld2q_dup_u16) |
| `vld3q_dup_u16` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld3q_dup_u16) |
| `vld4q_dup_u16` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld4q_dup_u16) |
| `vld1q_dup_s32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_dup_s32) |
| `vld2q_dup_s32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld2q_dup_s32) |
| `vld3q_dup_s32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld3q_dup_s32) |
| `vld4q_dup_s32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld4q_dup_s32) |
| `vld1q_dup_u32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_dup_u32) |
| `vld2q_dup_u32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld2q_dup_u32) |
| `vld3q_dup_u32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld3q_dup_u32) |
| `vld4q_dup_u32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld4q_dup_u32) |
| `vld1q_dup_f32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_dup_f32) |
| `vld2q_dup_f32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld2q_dup_f32) |
| `vld3q_dup_f32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld3q_dup_f32) |
| `vld4q_dup_f32` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld4q_dup_f32) |
| `vld1q_dup_s64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_dup_s64) |
| `vld2q_dup_s64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld2q_dup_s64) |
| `vld3q_dup_s64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld3q_dup_s64) |
| `vld4q_dup_s64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld4q_dup_s64) |
| `vld1q_dup_u64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_dup_u64) |
| `vld2q_dup_u64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld2q_dup_u64) |
| `vld3q_dup_u64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld3q_dup_u64) |
| `vld4q_dup_u64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld4q_dup_u64) |
| `vld1q_dup_f64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld1q_dup_f64) |
| `vld2q_dup_f64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld2q_dup_f64) |
| `vld3q_dup_f64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld3q_dup_f64) |
| `vld4q_dup_f64` | Load | [ARM Docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vld4q_dup_f64) |

## Usage Examples

### AVX Load/Store

```rust
use archmage::{Avx2Token, SimdToken};
use archmage::mem::avx;

fn process_f32(data: &mut [f32; 8]) {
    if let Some(token) = Avx2Token::try_new() {
        let v = avx::_mm256_loadu_ps(token, data);
        // Process v...
        avx::_mm256_storeu_ps(token, data, v);
    }
}
```

### NEON Load/Store

```rust
use archmage::{NeonToken, SimdToken};
use archmage::mem::neon;

fn process_f32(data: &mut [f32; 4]) {
    if let Some(token) = NeonToken::try_new() {
        let v = neon::vld1q_f32(token, data);
        // Process v...
        neon::vst1q_f32(token, data, v);
    }
}
```

### AVX-512 Load/Store

```rust
#[cfg(feature = "avx512")]
use archmage::{X64V4Token, SimdToken};
#[cfg(feature = "avx512")]
use archmage::mem::v4;

#[cfg(feature = "avx512")]
fn process_f32_512(data: &mut [f32; 16]) {
    if let Some(token) = X64V4Token::try_new() {
        let v = v4::_mm512_loadu_ps(token, data);
        // Process v...
        v4::_mm512_storeu_ps(token, data, v);
    }
}
```

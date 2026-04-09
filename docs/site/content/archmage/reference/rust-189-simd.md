+++
title = "What Rust 1.89 stabilized for SIMD"
description = "Complete inventory of the x86 SIMD intrinsics and target features stabilized in Rust 1.89"
weight = 50
+++

Rust 1.89 (August 2025) stabilized two feature gates that together brought ~867 intrinsic functions, 25 target features, mask types, and comparison/permutation constants from nightly-only to stable Rust. This page is the detailed inventory behind archmage's MSRV requirement.

## Feature gate: `avx512_target_feature`

**Tracking issue:** [rust-lang/rust#111137](https://github.com/rust-lang/rust/issues/111137)
**Target feature PR:** [rust-lang/rust#138940](https://github.com/rust-lang/rust/pull/138940)
**Intrinsics PR:** [rust-lang/stdarch#1819](https://github.com/rust-lang/stdarch/pull/1819)

### Target features (22)

All moved from `Unstable(sym::avx512_target_feature)` to `Stable` in `rustc_target/src/target_features.rs`:

**AVX-512 core (14):**

| Feature | Implies |
|---------|---------|
| `avx512f` | `avx2`, `fma`, `f16c` |
| `avx512bw` | `avx512f` |
| `avx512cd` | `avx512f` |
| `avx512dq` | `avx512f` |
| `avx512vl` | `avx512f` |
| `avx512bf16` | `avx512bw` |
| `avx512bitalg` | `avx512bw` |
| `avx512fp16` | `avx512bw` |
| `avx512ifma` | `avx512f` |
| `avx512vbmi` | `avx512bw` |
| `avx512vbmi2` | `avx512bw` |
| `avx512vnni` | `avx512f` |
| `avx512vp2intersect` | `avx512f` |
| `avx512vpopcntdq` | `avx512f` |

**VEX variants (5):**

| Feature | Implies |
|---------|---------|
| `avxifma` | `avx2` |
| `avxneconvert` | `avx2` |
| `avxvnni` | `avx2` |
| `avxvnniint8` | `avx2` |
| `avxvnniint16` | `avx2` |

**Crypto / GF (3):**

| Feature | Implies |
|---------|---------|
| `gfni` | `sse2` |
| `vaes` | `avx2`, `aes` |
| `vpclmulqdq` | `avx`, `pclmulqdq` |

### Intrinsics

857 unique intrinsic functions stabilized under `feature = "stdarch_x86_avx512"`, distributed across modules:

| Module | Stable annotations | Description |
|--------|-------------------|-------------|
| `avx512f` | 2874 | Foundation: arithmetic, compare, convert, shuffle, mask ops on 512-bit vectors |
| `avx512bw` | 826 | Byte/word (8/16-bit) operations |
| `avx512dq` | 399 | Doubleword/quadword operations, FP conversions |
| `avx512vbmi2` | 150 | Compress/expand, shift concatenation |
| `avx512vnni` | 68 | Vector neural network instructions (int8/int16 dot products) |
| `avx512cd` | 42 | Conflict detection (lzcnt, broadcast mask) |
| `avx512bf16` | 36 | BFloat16 conversions and dot products |
| `gfni` | 30 | Galois field byte operations |
| `avx512vbmi` | 30 | Variable byte-granularity permute |
| `avx512bitalg` | 24 | Bit manipulation (popcnt on bytes/words, mask shift) |
| `avx512ifma` | 22 | 52-bit integer fused multiply-add |
| `avx512vpopcntdq` | 18 | Vector popcnt on 32/64-bit elements |
| `vaes` | 8 | 256/512-bit AES rounds |
| `avxneconvert` | 6 | AVX non-exception FP16/BF16 conversions |
| `vpclmulqdq` | 2 | 256/512-bit carry-less multiply |
| x86_64-only (`avx512f`, `avx512bw`) | 32 | 64-bit-only variants (cvt, extract, etc.) |

The "stable annotations" count is larger than the unique function count because each function gets annotations on both the definition and the `pub use` re-export.

### Types and constants

Also stabilized in the same PR:

- **Mask types:** `__mmask8`, `__mmask16`, `__mmask32`, `__mmask64`
- **Mask conversion functions:** `_cvtmask{8,16,32,64}_u32`/`_u64`, `_cvtu32_mask{8,16,32}`, `_cvtu64_mask64`
- **Mask operations:** `_kadd_mask{8,16,32,64}`, `_kand_mask*`, `_kandn_mask*`, `_knot_mask*`, `_kor_mask*`, `_kxor_mask*`, `_kxnor_mask*`, etc.
- **Comparison enum:** `_MM_CMPINT_ENUM` with 8 variants (`_MM_CMPINT_EQ`, `_LT`, `_LE`, `_FALSE`, `_NE`, `_NLT`, `_NLE`, `_TRUE`)
- **Mantissa enums:** `_MM_MANTISSA_NORM_ENUM`, `_MM_MANTISSA_SIGN_ENUM` with variants
- **Permutation constants:** 256 `_MM_PERM_*` constants (`_MM_PERM_AAAA` through `_MM_PERM_DDDD`) plus `_MM_PERM_ENUM`

## Feature gate: `sha512_sm_x86`

**Tracking issue:** [rust-lang/rust#126624](https://github.com/rust-lang/rust/issues/126624)
**Target feature PR:** [rust-lang/rust#140767](https://github.com/rust-lang/rust/pull/140767)
**Intrinsics PR:** [rust-lang/stdarch#1796](https://github.com/rust-lang/stdarch/pull/1796)

### Target features (3)

| Feature | Implies |
|---------|---------|
| `sha512` | `avx2` |
| `sm3` | `avx` |
| `sm4` | `avx2` |

### Intrinsics (10)

**SHA-512:**
- `_mm256_sha512msg1_epi64`
- `_mm256_sha512msg2_epi64`
- `_mm256_sha512rnds2_epi64`

**SM3 (Chinese hash):**
- `_mm_sm3msg1_epi32`
- `_mm_sm3msg2_epi32`
- `_mm_sm3rnds2_epi32`

**SM4 (Chinese block cipher):**
- `_mm256_sm4key4_epi32`
- `_mm256_sm4rnds4_epi32`
- `_mm_sm4key4_epi32`
- `_mm_sm4rnds4_epi32`

## What's still unstable

These x86 target features remain behind feature gates as of 1.89:

| Feature | Gate | Notes |
|---------|------|-------|
| `kl`, `widekl` | `keylocker_x86` | KeyLocker AES acceleration |
| `avx10.1`, `avx10.2` | `avx10_target_feature` | AVX10 convergence features |
| `lahfsahf` | `lahfsahf_target_feature` | Legacy flags load/store |
| `rtm` | `rtm_target_feature` | Restricted transactional memory |
| `sse4a` | `sse4a_target_feature` | AMD SSE4a |
| `tbm` | `tbm_target_feature` | AMD trailing bit manipulation |
| `xop` | `xop_target_feature` | AMD XOP |
| `x87` | `x87_target_feature` | x87 FPU (ABI-fixed, unlikely to stabilize) |

## Why this matters for archmage

Archmage maps capability tokens to `#[target_feature]` attributes. Before 1.89, writing `#[target_feature(enable = "avx512f")]` on stable Rust was a compiler error. Every AVX-512 token (`X64V4Token`, `X64V4xToken`, `Avx512Fp16Token`) and its generated `#[arcane]` wrapper would fail to compile without `#![feature(avx512_target_feature)]`.

With 1.89, the full token hierarchy compiles on stable:

| Token | Features used |
|-------|--------------|
| `X64V4Token` | `avx512f`, `avx512bw`, `avx512cd`, `avx512dq`, `avx512vl` |
| `X64V4xToken` | Above + `avx512vbmi`, `avx512vbmi2`, `avx512bitalg`, `avx512vnni`, `avx512vpopcntdq`, `avx512ifma`, `avx512bf16`, `gfni`, `vaes`, `vpclmulqdq` |
| `Avx512Fp16Token` | Above + `avx512fp16` |

This also means `is_x86_feature_detected!("avx512f")` and friends work on stable for runtime detection via `summon()`.

Combined with the earlier stabilizations (1.86: safe `#[target_feature]` calls, 1.87: safe value-based intrinsics, 1.88: `as_chunks`), 1.89 was the first release where you could write AVX-512 code on stable Rust with zero `unsafe`.

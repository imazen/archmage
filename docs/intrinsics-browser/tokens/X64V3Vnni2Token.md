# X64V3Vnni2Token — x86-64-v3 VNNI-2

Proof that x86-64-v3 VNNI plus cross-vendor AVX-VNNI-2 extensions are available.

**Architecture:** x86_64 | **Features:** sse, sse2, sse3, ssse3, sse4.1, sse4.2, popcnt, cmpxchg16b, avx, avx2, fma, bmi1, bmi2, f16c, lzcnt, movbe, pclmulqdq, aes, vpclmulqdq, vaes, gfni, avxvnni, adx, sha, avxvnniint8, avxneconvert, avxifma
**Total intrinsics:** 30 (18 safe, 12 unsafe, 26 stable, 4 unstable/unknown)

## Usage

```rust
use archmage::prelude::*;

if let Some(token) = X64V3Vnni2Token::summon() {
    process(token, &mut data);
}

#[arcane(import_intrinsics)]  // Entry point only
fn process(token: X64V3Vnni2Token, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(8) {
        process_chunk(token, chunk.try_into().unwrap());
    }
}

#[rite(import_intrinsics)]  // All inner helpers
fn process_chunk(_: X64V3Vnni2Token, chunk: &mut [f32; 8]) {
    let v = _mm256_loadu_ps(chunk);  // safe!
    let doubled = _mm256_add_ps(v, v);    // value intrinsic (safe inside #[rite])
    _mm256_storeu_ps(chunk, doubled);  // safe!
}
// No unsafe anywhere. Use #![forbid(unsafe_code)] in your crate.
```



## All Intrinsics

### Stable, Safe (18 intrinsics)

| Name | Description | Instruction | Timing (H/Z4) |
|------|-------------|-------------|---------------|
| `_mm256_cvtneps_avx_pbh` | Convert packed single precision (32-bit) floating-point elem... | vcvtneps2bf16 | 4/1, 3/1 |
| `_mm256_dpbssd_epi32` | Multiply groups of 4 adjacent pairs of signed 8-bit integers... | vpdpbssd | — |
| `_mm256_dpbssds_epi32` | Multiply groups of 4 adjacent pairs of signed 8-bit integers... | vpdpbssds | — |
| `_mm256_dpbsud_epi32` | Multiply groups of 4 adjacent pairs of signed 8-bit integers... | vpdpbsud | — |
| `_mm256_dpbsuds_epi32` | Multiply groups of 4 adjacent pairs of signed 8-bit integers... | vpdpbsuds | — |
| `_mm256_dpbuud_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbuud | — |
| `_mm256_dpbuuds_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbuuds | — |
| `_mm256_madd52hi_avx_epu64` | Multiply packed unsigned 52-bit integers in each 64-bit elem... | vpmadd52huq | — |
| `_mm256_madd52lo_avx_epu64` | Multiply packed unsigned 52-bit integers in each 64-bit elem... | vpmadd52luq | — |
| `_mm_cvtneps_avx_pbh` | Convert packed single precision (32-bit) floating-point elem... | vcvtneps2bf16 | 4/1, 3/1 |
| `_mm_dpbssd_epi32` | Multiply groups of 4 adjacent pairs of signed 8-bit integers... | vpdpbssd | — |
| `_mm_dpbssds_epi32` | Multiply groups of 4 adjacent pairs of signed 8-bit integers... | vpdpbssds | — |
| `_mm_dpbsud_epi32` | Multiply groups of 4 adjacent pairs of signed 8-bit integers... | vpdpbsud | — |
| `_mm_dpbsuds_epi32` | Multiply groups of 4 adjacent pairs of signed 8-bit integers... | vpdpbsuds | — |
| `_mm_dpbuud_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbuud | — |
| `_mm_dpbuuds_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbuuds | — |
| `_mm_madd52hi_avx_epu64` | Multiply packed unsigned 52-bit integers in each 64-bit elem... | vpmadd52huq | — |
| `_mm_madd52lo_avx_epu64` | Multiply packed unsigned 52-bit integers in each 64-bit elem... | vpmadd52luq | — |

### Stable, Unsafe (8 intrinsics) — use import_intrinsics for safe versions

| Name | Description | Safe Variant |
|------|-------------|--------------|
| `_mm256_cvtneebf16_ps` | Convert packed BF16 (16-bit) floating-point even-indexed ele... | — |
| `_mm256_cvtneeph_ps` | Convert packed half-precision (16-bit) floating-point even-i... | — |
| `_mm256_cvtneobf16_ps` | Convert packed BF16 (16-bit) floating-point odd-indexed elem... | — |
| `_mm256_cvtneoph_ps` | Convert packed half-precision (16-bit) floating-point odd-in... | — |
| `_mm_cvtneebf16_ps` | Convert packed BF16 (16-bit) floating-point even-indexed ele... | — |
| `_mm_cvtneeph_ps` | Convert packed half-precision (16-bit) floating-point even-i... | — |
| `_mm_cvtneobf16_ps` | Convert packed BF16 (16-bit) floating-point odd-indexed elem... | — |
| `_mm_cvtneoph_ps` | Convert packed half-precision (16-bit) floating-point odd-in... | — |

### Unstable/Nightly (4 intrinsics)

| Name | Description | Instruction |
|------|-------------|-------------|
| `_mm256_bcstnebf16_ps` | Convert scalar BF16 (16-bit) floating point element stored a... | vbcstnebf162ps |
| `_mm256_bcstnesh_ps` | Convert scalar half-precision (16-bit) floating-point elemen... | vbcstnesh2ps |
| `_mm_bcstnebf16_ps` | Convert scalar BF16 (16-bit) floating point element stored a... | vbcstnebf162ps |
| `_mm_bcstnesh_ps` | Convert scalar half-precision (16-bit) floating-point elemen... | vbcstnesh2ps |



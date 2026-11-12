# X64V3VnniToken — x86-64-v3 VNNI

Proof that VEX-encoded AVX-VNNI (`avxvnni`) is available on top of `X64V3GfniCryptoToken`.

**Architecture:** x86_64 | **Features:** sse, sse2, sse3, ssse3, sse4.1, sse4.2, popcnt, cmpxchg16b, avx, avx2, fma, bmi1, bmi2, f16c, lzcnt, movbe, pclmulqdq, aes, vpclmulqdq, vaes, gfni, avxvnni
**Total intrinsics:** 8 (8 safe, 0 unsafe, 8 stable, 0 unstable/unknown)

## Usage

```rust
use archmage::prelude::*;

if let Some(token) = X64V3VnniToken::summon() {
    process(token, &mut data);
}

#[arcane(import_intrinsics)]  // Entry point only
fn process(token: X64V3VnniToken, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(8) {
        process_chunk(token, chunk.try_into().unwrap());
    }
}

#[rite(import_intrinsics)]  // All inner helpers
fn process_chunk(_: X64V3VnniToken, chunk: &mut [f32; 8]) {
    let v = _mm256_loadu_ps(chunk);  // safe!
    let doubled = _mm256_add_ps(v, v);    // value intrinsic (safe inside #[rite])
    _mm256_storeu_ps(chunk, doubled);  // safe!
}
// No unsafe anywhere. Use #![forbid(unsafe_code)] in your crate.
```



## All Intrinsics

### Stable, Safe (8 intrinsics)

| Name | Description | Instruction | Timing (H/Z4) |
|------|-------------|-------------|---------------|
| `_mm256_dpbusd_avx_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusd | — |
| `_mm256_dpbusds_avx_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusds | — |
| `_mm256_dpwssd_avx_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssd | — |
| `_mm256_dpwssds_avx_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssds | — |
| `_mm_dpbusd_avx_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusd | — |
| `_mm_dpbusds_avx_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusds | — |
| `_mm_dpwssd_avx_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssd | — |
| `_mm_dpwssds_avx_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssds | — |



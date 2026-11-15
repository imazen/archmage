# X64V3VnniToken — x86-64-v3 VNNI

Proof that VEX-encoded AVX-VNNI (`avxvnni`), `adx`, and `sha` are available on top of `X64V3GfniCryptoToken`.

**Architecture:** x86_64 | **Features:** sse, sse2, sse3, ssse3, sse4.1, sse4.2, popcnt, cmpxchg16b, avx, avx2, fma, bmi1, bmi2, f16c, lzcnt, movbe, pclmulqdq, aes, vpclmulqdq, vaes, gfni, avxvnni, adx, sha
**Total intrinsics:** 17 (17 safe, 0 unsafe, 17 stable, 0 unstable/unknown)

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

### Stable, Safe (17 intrinsics)

| Name | Description | Instruction | Timing (H/Z4) |
|------|-------------|-------------|---------------|
| `_addcarryx_u32` | Adds unsigned 32-bit integers `a` and `b` with unsigned 8-bi... | adc | — |
| `_addcarryx_u64` | Adds unsigned 64-bit integers `a` and `b` with unsigned 8-bi... | adc | — |
| `_mm256_dpbusd_avx_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusd | — |
| `_mm256_dpbusds_avx_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusds | — |
| `_mm256_dpwssd_avx_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssd | — |
| `_mm256_dpwssds_avx_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssds | — |
| `_mm_dpbusd_avx_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusd | — |
| `_mm_dpbusds_avx_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusds | — |
| `_mm_dpwssd_avx_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssd | — |
| `_mm_dpwssds_avx_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssds | — |
| `_mm_sha1msg1_epu32` | Performs an intermediate calculation for the next four SHA1 ... | sha1msg1 | — |
| `_mm_sha1msg2_epu32` | Performs the final calculation for the next four SHA1 messag... | sha1msg2 | — |
| `_mm_sha1nexte_epu32` | Calculate SHA1 state variable E after four rounds of operati... | sha1nexte | — |
| `_mm_sha1rnds4_epu32` | Performs four rounds of SHA1 operation using an initial SHA1... | sha1rnds4 | — |
| `_mm_sha256msg1_epu32` | Performs an intermediate calculation for the next four SHA25... | sha256msg1 | — |
| `_mm_sha256msg2_epu32` | Performs the final calculation for the next four SHA256 mess... | sha256msg2 | — |
| `_mm_sha256rnds2_epu32` | Performs 2 rounds of SHA256 operation using an initial SHA25... | sha256rnds2 | — |



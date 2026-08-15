# X64V3GfniCryptoToken — x86-64-v3 GFNI Crypto

Proof that AVX2 + GFNI + VPCLMULQDQ + VAES are available.

**Architecture:** x86_64 | **Features:** sse, sse2, sse3, ssse3, sse4.1, sse4.2, popcnt, cmpxchg16b, avx, avx2, fma, bmi1, bmi2, f16c, lzcnt, movbe, pclmulqdq, aes, vpclmulqdq, vaes, gfni
**Total intrinsics:** 6 (6 safe, 0 unsafe, 6 stable, 0 unstable/unknown)

## Usage

```rust
use archmage::prelude::*;

if let Some(token) = X64V3GfniCryptoToken::summon() {
    process(token, &mut data);
}

#[arcane(import_intrinsics)]  // Entry point only
fn process(token: X64V3GfniCryptoToken, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(8) {
        process_chunk(token, chunk.try_into().unwrap());
    }
}

#[rite(import_intrinsics)]  // All inner helpers
fn process_chunk(_: X64V3GfniCryptoToken, chunk: &mut [f32; 8]) {
    let v = _mm256_loadu_ps(chunk);  // safe!
    let doubled = _mm256_add_ps(v, v);    // value intrinsic (safe inside #[rite])
    _mm256_storeu_ps(chunk, doubled);  // safe!
}
// No unsafe anywhere. Use #![forbid(unsafe_code)] in your crate.
```



## All Intrinsics

### Stable, Safe (6 intrinsics)

| Name | Description | Instruction | Timing (H/Z4) |
|------|-------------|-------------|---------------|
| `_mm256_gf2p8affine_epi64_epi8` | Performs an affine transformation on the packed bytes in x. ... | vgf2p8affineqb | — |
| `_mm256_gf2p8affineinv_epi64_epi8` | Performs an affine transformation on the inverted packed byt... | vgf2p8affineinvqb | — |
| `_mm256_gf2p8mul_epi8` | Performs a multiplication in GF(2^8) on the packed bytes. Th... | vgf2p8mulb | — |
| `_mm_gf2p8affine_epi64_epi8` | Performs an affine transformation on the packed bytes in x. ... | gf2p8affineqb | — |
| `_mm_gf2p8affineinv_epi64_epi8` | Performs an affine transformation on the inverted packed byt... | gf2p8affineinvqb | — |
| `_mm_gf2p8mul_epi8` | Performs a multiplication in GF(2^8) on the packed bytes. Th... | gf2p8mulb | — |



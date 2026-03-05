# X64V3CryptoToken — x86-64-v3 Crypto

Proof that AVX2 + VPCLMULQDQ + VAES are available.

**Architecture:** x86_64 | **Features:** sse, sse2, sse3, ssse3, sse4.1, sse4.2, popcnt, cmpxchg16b, avx, avx2, fma, bmi1, bmi2, f16c, lzcnt, movbe, pclmulqdq, aes, vpclmulqdq, vaes
**Total intrinsics:** 5 (5 safe, 0 unsafe, 5 stable, 0 unstable/unknown)

## Usage

```rust
use archmage::{X64V3CryptoToken, SimdToken};

if let Some(token) = X64V3CryptoToken::summon() {
    process(token, &mut data);
}

#[arcane(import_intrinsics)]  // Entry point only
fn process(token: X64V3CryptoToken, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(8) {
        process_chunk(token, chunk.try_into().unwrap());
    }
}

#[rite(import_intrinsics)]  // All inner helpers
fn process_chunk(_: X64V3CryptoToken, chunk: &mut [f32; 8]) {
    let v = _mm256_loadu_ps(chunk);  // safe!
    let doubled = _mm256_add_ps(v, v);    // value intrinsic (safe inside #[rite])
    _mm256_storeu_ps(chunk, doubled);  // safe!
}
// No unsafe anywhere. Use #![forbid(unsafe_code)] in your crate.
```



## All Intrinsics

### Stable, Safe (5 intrinsics)

| Name | Description | Instruction | Timing (H/Z4) |
|------|-------------|-------------|---------------|
| `_mm256_aesdec_epi128` | Performs one round of an AES decryption flow on each 128-bit... | vaesdec | —/4/1 |
| `_mm256_aesdeclast_epi128` | Performs the last round of an AES decryption flow on each 12... | vaesdeclast | —/4/1 |
| `_mm256_aesenc_epi128` | Performs one round of an AES encryption flow on each 128-bit... | vaesenc | —/4/1 |
| `_mm256_aesenclast_epi128` | Performs the last round of an AES encryption flow on each 12... | vaesenclast | —/4/1 |
| `_mm256_clmulepi64_epi128` | Performs a carry-less multiplication of two 64-bit polynomia... | vpclmul | 7/2, 4/1 |



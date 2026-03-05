# X64CryptoToken — x86-64 Crypto

Proof that PCLMULQDQ + AES-NI are available (on top of x86-64-v2).

**Architecture:** x86_64 | **Features:** sse, sse2, sse3, ssse3, sse4.1, sse4.2, popcnt, cmpxchg16b, pclmulqdq, aes
**Total intrinsics:** 7 (7 safe, 0 unsafe, 7 stable, 0 unstable/unknown)

## Usage

```rust
use archmage::{X64CryptoToken, SimdToken};

if let Some(token) = X64CryptoToken::summon() {
    process(token, &mut data);
}

#[arcane(import_intrinsics)]  // Entry point only
fn process(token: X64CryptoToken, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(4) {
        process_chunk(token, chunk.try_into().unwrap());
    }
}

#[rite(import_intrinsics)]  // All inner helpers
fn process_chunk(_: X64CryptoToken, chunk: &mut [f32; 4]) {
    let v = _mm_loadu_ps(chunk);  // safe!
    let doubled = _mm_add_ps(v, v);  // value intrinsic (safe inside #[rite])
    _mm_storeu_ps(chunk, doubled);  // safe!
}
// No unsafe anywhere. Use #![forbid(unsafe_code)] in your crate.
```



## All Intrinsics

### Stable, Safe (7 intrinsics)

| Name | Description | Instruction | Timing (H/Z4) |
|------|-------------|-------------|---------------|
| `_mm_aesdec_si128` | Performs one round of an AES decryption flow on data (state)... | aesdec | 7/1, 4/1 |
| `_mm_aesdeclast_si128` | Performs the last round of an AES decryption flow on data (s... | aesdeclast | 7/1, 4/1 |
| `_mm_aesenc_si128` | Performs one round of an AES encryption flow on data (state)... | aesenc | 7/1, 4/1 |
| `_mm_aesenclast_si128` | Performs the last round of an AES encryption flow on data (s... | aesenclast | 7/1, 4/1 |
| `_mm_aesimc_si128` | Performs the `InvMixColumns` transformation on `a` | aesimc | 7/1, 4/1 |
| `_mm_aeskeygenassist_si128` | Assist in expanding the AES cipher key.  Assist in expanding... | aeskeygenassist | 7/1, 4/1 |
| `_mm_clmulepi64_si128` | Performs a carry-less multiplication of two 64-bit polynomia... | pclmul | 7/2, 4/1 |



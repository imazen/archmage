# X64V4Token (Avx512Token, Server64) — AVX-512

Proof that AVX-512 (F + CD + VL + DQ + BW) is available.

**Architecture:** x86_64 | **Features:** sse, sse2, sse3, ssse3, sse4.1, sse4.2, popcnt, cmpxchg16b, avx, avx2, fma, bmi1, bmi2, f16c, lzcnt, movbe, pclmulqdq, aes, avx512f, avx512bw, avx512cd, avx512dq, avx512vl
**Total intrinsics:** 3902 (3560 safe, 342 unsafe, 3902 stable, 0 unstable/unknown)

## Usage

```rust
use archmage::prelude::*;

if let Some(token) = X64V4Token::summon() {
    process(token, &mut data);
}

#[arcane]  // Entry point only
fn process(token: X64V4Token, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(16) {
        process_chunk(token, chunk.try_into().unwrap());
    }
}

#[rite]  // All inner helpers
fn process_chunk(_: X64V4Token, chunk: &mut [f32; 16]) {
    let v = _mm512_loadu_ps(chunk.as_ptr());  // safe inside #[rite]
    let doubled = _mm512_add_ps(v, v);
    _mm512_storeu_ps(chunk.as_mut_ptr(), doubled);
}
// Use #![forbid(unsafe_code)] with safe_unaligned_simd for memory ops.
```

## Safe Memory Operations (safe_unaligned_simd)

| Function | Safe Signature |
|----------|---------------|
| `_mm256_loadu_epi16` | `fn _mm256_loadu_epi16<T: Is256BitsUnaligned>(mem_addr: &T) -> __m256i` |
| `_mm256_loadu_epi32` | `fn _mm256_loadu_epi32<T: Is256BitsUnaligned>(mem_addr: &T) -> __m256i` |
| `_mm256_loadu_epi64` | `fn _mm256_loadu_epi64<T: Is256BitsUnaligned>(mem_addr: &T) -> __m256i` |
| `_mm256_loadu_epi8` | `fn _mm256_loadu_epi8<T: Is256BitsUnaligned>(mem_addr: &T) -> __m256i` |
| `_mm256_mask_compressstoreu_pd` | `fn _mm256_mask_compressstoreu_pd(base_addr: &mut [f64; 4], k: __mmask8, a: __m256d) -> ()` |
| `_mm256_mask_compressstoreu_ps` | `fn _mm256_mask_compressstoreu_ps(base_addr: &mut [f32; 8], k: __mmask8, a: __m256) -> ()` |
| `_mm256_mask_expandloadu_pd` | `fn _mm256_mask_expandloadu_pd(src: __m256d, k: __mmask8, mem_addr: &[f64; 4]) -> __m256d` |
| `_mm256_mask_expandloadu_ps` | `fn _mm256_mask_expandloadu_ps(src: __m256, k: __mmask8, mem_addr: &[f32; 8]) -> __m256` |
| `_mm256_mask_loadu_pd` | `fn _mm256_mask_loadu_pd(src: __m256d, k: __mmask8, mem_addr: &[f64; 4]) -> __m256d` |
| `_mm256_mask_loadu_ps` | `fn _mm256_mask_loadu_ps(src: __m256, k: __mmask8, mem_addr: &[f32; 8]) -> __m256` |
| `_mm256_mask_storeu_epi16` | `fn _mm256_mask_storeu_epi16<T: Is256BitsUnaligned>(mem_addr: &mut T, k: __mmask16, a: __m256i) -> ()` |
| `_mm256_mask_storeu_epi32` | `fn _mm256_mask_storeu_epi32<T: Is256BitsUnaligned>(mem_addr: &mut T, k: __mmask8, a: __m256i) -> ()` |
| `_mm256_mask_storeu_epi64` | `fn _mm256_mask_storeu_epi64<T: Is256BitsUnaligned>(mem_addr: &mut T, k: __mmask8, a: __m256i) -> ()` |
| `_mm256_mask_storeu_epi8` | `fn _mm256_mask_storeu_epi8<T: Is256BitsUnaligned>(mem_addr: &mut T, k: __mmask32, a: __m256i) -> ()` |
| `_mm256_mask_storeu_pd` | `fn _mm256_mask_storeu_pd(mem_addr: &mut [f64; 4], k: __mmask8, a: __m256d) -> ()` |
| `_mm256_mask_storeu_ps` | `fn _mm256_mask_storeu_ps(mem_addr: &mut [f32; 8], k: __mmask8, a: __m256) -> ()` |
| `_mm256_maskz_expandloadu_epi32` | `fn _mm256_maskz_expandloadu_epi32<T: Is256BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m256i` |
| `_mm256_maskz_expandloadu_epi64` | `fn _mm256_maskz_expandloadu_epi64<T: Is256BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m256i` |
| `_mm256_maskz_expandloadu_pd` | `fn _mm256_maskz_expandloadu_pd(k: __mmask8, mem_addr: &[f64; 4]) -> __m256d` |
| `_mm256_maskz_expandloadu_ps` | `fn _mm256_maskz_expandloadu_ps(k: __mmask8, mem_addr: &[f32; 8]) -> __m256` |
| `_mm256_maskz_loadu_epi16` | `fn _mm256_maskz_loadu_epi16<T: Is256BitsUnaligned>(k: __mmask16, mem_addr: &T) -> __m256i` |
| `_mm256_maskz_loadu_epi32` | `fn _mm256_maskz_loadu_epi32<T: Is256BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m256i` |
| `_mm256_maskz_loadu_epi64` | `fn _mm256_maskz_loadu_epi64<T: Is256BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m256i` |
| `_mm256_maskz_loadu_epi8` | `fn _mm256_maskz_loadu_epi8<T: Is256BitsUnaligned>(k: __mmask32, mem_addr: &T) -> __m256i` |
| `_mm256_maskz_loadu_pd` | `fn _mm256_maskz_loadu_pd(k: __mmask8, mem_addr: &[f64; 4]) -> __m256d` |
| `_mm256_maskz_loadu_ps` | `fn _mm256_maskz_loadu_ps(k: __mmask8, mem_addr: &[f32; 8]) -> __m256` |
| `_mm256_storeu_epi16` | `fn _mm256_storeu_epi16<T: Is256BitsUnaligned>(mem_addr: &mut T, a: __m256i) -> ()` |
| `_mm256_storeu_epi32` | `fn _mm256_storeu_epi32<T: Is256BitsUnaligned>(mem_addr: &mut T, a: __m256i) -> ()` |
| `_mm256_storeu_epi64` | `fn _mm256_storeu_epi64<T: Is256BitsUnaligned>(mem_addr: &mut T, a: __m256i) -> ()` |
| `_mm256_storeu_epi8` | `fn _mm256_storeu_epi8<T: Is256BitsUnaligned>(mem_addr: &mut T, a: __m256i) -> ()` |
| `_mm512_loadu_epi16` | `fn _mm512_loadu_epi16<T: Is512BitsUnaligned>(mem_addr: &T) -> __m512i` |
| `_mm512_loadu_epi32` | `fn _mm512_loadu_epi32<T: Is512BitsUnaligned>(mem_addr: &T) -> __m512i` |
| `_mm512_loadu_epi64` | `fn _mm512_loadu_epi64<T: Is512BitsUnaligned>(mem_addr: &T) -> __m512i` |
| `_mm512_loadu_epi8` | `fn _mm512_loadu_epi8<T: Is512BitsUnaligned>(mem_addr: &T) -> __m512i` |
| `_mm512_loadu_pd` | `fn _mm512_loadu_pd(mem_addr: &[f64; 8]) -> __m512d` |
| `_mm512_loadu_ps` | `fn _mm512_loadu_ps(mem_addr: &[f32; 16]) -> __m512` |
| `_mm512_loadu_si512` | `fn _mm512_loadu_si512<T: Is512BitsUnaligned>(mem_addr: &T) -> __m512i` |
| `_mm512_mask_compressstoreu_pd` | `fn _mm512_mask_compressstoreu_pd(base_addr: &mut [f64; 8], k: __mmask8, a: __m512d) -> ()` |
| `_mm512_mask_compressstoreu_ps` | `fn _mm512_mask_compressstoreu_ps(base_addr: &mut [f32; 16], k: __mmask16, a: __m512) -> ()` |
| `_mm512_mask_expandloadu_pd` | `fn _mm512_mask_expandloadu_pd(src: __m512d, k: __mmask8, mem_addr: &[f64; 8]) -> __m512d` |
| `_mm512_mask_expandloadu_ps` | `fn _mm512_mask_expandloadu_ps(src: __m512, k: __mmask16, mem_addr: &[f32; 16]) -> __m512` |
| `_mm512_mask_loadu_pd` | `fn _mm512_mask_loadu_pd(src: __m512d, k: __mmask8, mem_addr: &[f64; 8]) -> __m512d` |
| `_mm512_mask_loadu_ps` | `fn _mm512_mask_loadu_ps(src: __m512, k: __mmask16, mem_addr: &[f32; 16]) -> __m512` |
| `_mm512_mask_storeu_epi16` | `fn _mm512_mask_storeu_epi16<T: Is512BitsUnaligned>(mem_addr: &mut T, k: __mmask32, a: __m512i) -> ()` |
| `_mm512_mask_storeu_epi32` | `fn _mm512_mask_storeu_epi32<T: Is512BitsUnaligned>(mem_addr: &mut T, k: __mmask16, a: __m512i) -> ()` |
| `_mm512_mask_storeu_epi64` | `fn _mm512_mask_storeu_epi64<T: Is512BitsUnaligned>(mem_addr: &mut T, k: __mmask8, a: __m512i) -> ()` |
| `_mm512_mask_storeu_epi8` | `fn _mm512_mask_storeu_epi8<T: Is512BitsUnaligned>(mem_addr: &mut T, k: __mmask64, a: __m512i) -> ()` |
| `_mm512_mask_storeu_pd` | `fn _mm512_mask_storeu_pd(mem_addr: &mut [f64; 8], k: __mmask8, a: __m512d) -> ()` |
| `_mm512_mask_storeu_ps` | `fn _mm512_mask_storeu_ps(mem_addr: &mut [f32; 16], k: __mmask16, a: __m512) -> ()` |
| `_mm512_maskz_expandloadu_epi64` | `fn _mm512_maskz_expandloadu_epi64<T: Is512BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m512i` |
| `_mm512_maskz_expandloadu_pd` | `fn _mm512_maskz_expandloadu_pd(k: __mmask8, mem_addr: &[f64; 8]) -> __m512d` |
| `_mm512_maskz_expandloadu_ps` | `fn _mm512_maskz_expandloadu_ps(k: __mmask16, mem_addr: &[f32; 16]) -> __m512` |
| `_mm512_maskz_loadu_epi16` | `fn _mm512_maskz_loadu_epi16<T: Is512BitsUnaligned>(k: __mmask32, mem_addr: &T) -> __m512i` |
| `_mm512_maskz_loadu_epi32` | `fn _mm512_maskz_loadu_epi32<T: Is512BitsUnaligned>(k: __mmask16, mem_addr: &T) -> __m512i` |
| `_mm512_maskz_loadu_epi64` | `fn _mm512_maskz_loadu_epi64<T: Is512BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m512i` |
| `_mm512_maskz_loadu_epi8` | `fn _mm512_maskz_loadu_epi8<T: Is512BitsUnaligned>(k: __mmask64, mem_addr: &T) -> __m512i` |
| `_mm512_maskz_loadu_pd` | `fn _mm512_maskz_loadu_pd(k: __mmask8, mem_addr: &[f64; 8]) -> __m512d` |
| `_mm512_maskz_loadu_ps` | `fn _mm512_maskz_loadu_ps(k: __mmask16, mem_addr: &[f32; 16]) -> __m512` |
| `_mm512_storeu_epi16` | `fn _mm512_storeu_epi16<T: Is512BitsUnaligned>(mem_addr: &mut T, a: __m512i) -> ()` |
| `_mm512_storeu_epi32` | `fn _mm512_storeu_epi32<T: Is512BitsUnaligned>(mem_addr: &mut T, a: __m512i) -> ()` |
| `_mm512_storeu_epi64` | `fn _mm512_storeu_epi64<T: Is512BitsUnaligned>(mem_addr: &mut T, a: __m512i) -> ()` |
| `_mm512_storeu_epi8` | `fn _mm512_storeu_epi8<T: Is512BitsUnaligned>(mem_addr: &mut T, a: __m512i) -> ()` |
| `_mm512_storeu_pd` | `fn _mm512_storeu_pd(mem_addr: &mut [f64; 8], a: __m512d) -> ()` |
| `_mm512_storeu_ps` | `fn _mm512_storeu_ps(mem_addr: &mut [f32; 16], a: __m512) -> ()` |
| `_mm512_storeu_si512` | `fn _mm512_storeu_si512<T: Is512BitsUnaligned>(mem_addr: &mut T, a: __m512i) -> ()` |
| `_mm_loadu_epi16` | `fn _mm_loadu_epi16<T: Is128BitsUnaligned>(mem_addr: &T) -> __m128i` |
| `_mm_loadu_epi32` | `fn _mm_loadu_epi32<T: Is128BitsUnaligned>(mem_addr: &T) -> __m128i` |
| `_mm_loadu_epi64` | `fn _mm_loadu_epi64<T: Is128BitsUnaligned>(mem_addr: &T) -> __m128i` |
| `_mm_loadu_epi8` | `fn _mm_loadu_epi8<T: Is128BitsUnaligned>(mem_addr: &T) -> __m128i` |
| `_mm_mask_compressstoreu_pd` | `fn _mm_mask_compressstoreu_pd(base_addr: &mut [f64; 2], k: __mmask8, a: __m128d) -> ()` |
| `_mm_mask_compressstoreu_ps` | `fn _mm_mask_compressstoreu_ps(base_addr: &mut [f32; 4], k: __mmask8, a: __m128) -> ()` |
| `_mm_mask_expandloadu_pd` | `fn _mm_mask_expandloadu_pd(src: __m128d, k: __mmask8, mem_addr: &[f64; 2]) -> __m128d` |
| `_mm_mask_expandloadu_ps` | `fn _mm_mask_expandloadu_ps(src: __m128, k: __mmask8, mem_addr: &[f32; 4]) -> __m128` |
| `_mm_mask_loadu_pd` | `fn _mm_mask_loadu_pd(src: __m128d, k: __mmask8, mem_addr: &[f64; 2]) -> __m128d` |
| `_mm_mask_loadu_ps` | `fn _mm_mask_loadu_ps(src: __m128, k: __mmask8, mem_addr: &[f32; 4]) -> __m128` |
| `_mm_mask_storeu_epi16` | `fn _mm_mask_storeu_epi16<T: Is128BitsUnaligned>(mem_addr: &mut T, k: __mmask8, a: __m128i) -> ()` |
| `_mm_mask_storeu_epi32` | `fn _mm_mask_storeu_epi32<T: Is128BitsUnaligned>(mem_addr: &mut T, k: __mmask8, a: __m128i) -> ()` |
| `_mm_mask_storeu_epi64` | `fn _mm_mask_storeu_epi64<T: Is128BitsUnaligned>(mem_addr: &mut T, k: __mmask8, a: __m128i) -> ()` |
| `_mm_mask_storeu_epi8` | `fn _mm_mask_storeu_epi8<T: Is128BitsUnaligned>(mem_addr: &mut T, k: __mmask16, a: __m128i) -> ()` |
| `_mm_mask_storeu_pd` | `fn _mm_mask_storeu_pd(mem_addr: &mut [f64; 2], k: __mmask8, a: __m128d) -> ()` |
| `_mm_mask_storeu_ps` | `fn _mm_mask_storeu_ps(mem_addr: &mut [f32; 4], k: __mmask8, a: __m128) -> ()` |
| `_mm_maskz_expandloadu_epi32` | `fn _mm_maskz_expandloadu_epi32<T: Is128BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m128i` |
| `_mm_maskz_expandloadu_epi64` | `fn _mm_maskz_expandloadu_epi64<T: Is128BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m128i` |
| `_mm_maskz_expandloadu_pd` | `fn _mm_maskz_expandloadu_pd(k: __mmask8, mem_addr: &[f64; 2]) -> __m128d` |
| `_mm_maskz_expandloadu_ps` | `fn _mm_maskz_expandloadu_ps(k: __mmask8, mem_addr: &[f32; 4]) -> __m128` |
| `_mm_maskz_loadu_epi16` | `fn _mm_maskz_loadu_epi16<T: Is128BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m128i` |
| `_mm_maskz_loadu_epi32` | `fn _mm_maskz_loadu_epi32<T: Is128BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m128i` |
| `_mm_maskz_loadu_epi64` | `fn _mm_maskz_loadu_epi64<T: Is128BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m128i` |
| `_mm_maskz_loadu_epi8` | `fn _mm_maskz_loadu_epi8<T: Is128BitsUnaligned>(k: __mmask16, mem_addr: &T) -> __m128i` |
| `_mm_maskz_loadu_pd` | `fn _mm_maskz_loadu_pd(k: __mmask8, mem_addr: &[f64; 2]) -> __m128d` |
| `_mm_maskz_loadu_ps` | `fn _mm_maskz_loadu_ps(k: __mmask8, mem_addr: &[f32; 4]) -> __m128` |
| `_mm_storeu_epi16` | `fn _mm_storeu_epi16<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) -> ()` |
| `_mm_storeu_epi32` | `fn _mm_storeu_epi32<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) -> ()` |
| `_mm_storeu_epi64` | `fn _mm_storeu_epi64<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) -> ()` |
| `_mm_storeu_epi8` | `fn _mm_storeu_epi8<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) -> ()` |


## All Intrinsics

### Stable, Safe (3560 intrinsics)

| Name | Description | Instruction | Timing (H/Z4) |
|------|-------------|-------------|---------------|
| `_cvtmask16_u32` | Convert 16-bit mask a into an integer value, and store the r... |  | — |
| `_cvtmask32_u32` | Convert 32-bit mask a into an integer value, and store the r... |  | — |
| `_cvtmask64_u64` | Convert 64-bit mask a into an integer value, and store the r... |  | — |
| `_cvtmask8_u32` | Convert 8-bit mask a to a 32-bit integer value and store the... |  | — |
| `_cvtu32_mask16` | Convert 32-bit integer value a to an 16-bit mask and store t... |  | — |
| `_cvtu32_mask32` | Convert integer value a into an 32-bit mask, and store the r... |  | — |
| `_cvtu32_mask8` | Convert 32-bit integer value a to an 8-bit mask and store th... |  | — |
| `_cvtu64_mask64` | Convert integer value a into an 64-bit mask, and store the r... |  | — |
| `_kadd_mask16` | Add 16-bit masks a and b, and store the result in dst |  | — |
| `_kadd_mask32` | Add 32-bit masks in a and b, and store the result in k |  | — |
| `_kadd_mask64` | Add 64-bit masks in a and b, and store the result in k |  | — |
| `_kadd_mask8` | Add 8-bit masks a and b, and store the result in dst |  | — |
| `_kand_mask16` | Compute the bitwise AND of 16-bit masks a and b, and store t... | and | — |
| `_kand_mask32` | Compute the bitwise AND of 32-bit masks a and b, and store t... |  | — |
| `_kand_mask64` | Compute the bitwise AND of 64-bit masks a and b, and store t... |  | — |
| `_kand_mask8` | Bitwise AND of 8-bit masks a and b, and store the result in ... |  | — |
| `_kandn_mask16` | Compute the bitwise NOT of 16-bit masks a and then AND with ... | not | — |
| `_kandn_mask32` | Compute the bitwise NOT of 32-bit masks a and then AND with ... |  | — |
| `_kandn_mask64` | Compute the bitwise NOT of 64-bit masks a and then AND with ... |  | — |
| `_kandn_mask8` | Bitwise AND NOT of 8-bit masks a and b, and store the result... |  | — |
| `_knot_mask16` | Compute the bitwise NOT of 16-bit mask a, and store the resu... |  | — |
| `_knot_mask32` | Compute the bitwise NOT of 32-bit mask a, and store the resu... |  | — |
| `_knot_mask64` | Compute the bitwise NOT of 64-bit mask a, and store the resu... |  | — |
| `_knot_mask8` | Bitwise NOT of 8-bit mask a, and store the result in dst |  | — |
| `_kor_mask16` | Compute the bitwise OR of 16-bit masks a and b, and store th... | or | — |
| `_kor_mask32` | Compute the bitwise OR of 32-bit masks a and b, and store th... |  | — |
| `_kor_mask64` | Compute the bitwise OR of 64-bit masks a and b, and store th... |  | — |
| `_kor_mask8` | Bitwise OR of 8-bit masks a and b, and store the result in d... |  | — |
| `_kortestc_mask16_u8` | Compute the bitwise OR of 16-bit masks a and b. If the resul... |  | — |
| `_kortestc_mask32_u8` | Compute the bitwise OR of 32-bit masks a and b. If the resul... |  | — |
| `_kortestc_mask64_u8` | Compute the bitwise OR of 64-bit masks a and b. If the resul... |  | — |
| `_kortestc_mask8_u8` | Compute the bitwise OR of 8-bit masks a and b. If the result... |  | — |
| `_kortestz_mask16_u8` | Compute the bitwise OR of 16-bit masks a and b. If the resul... |  | — |
| `_kortestz_mask32_u8` | Compute the bitwise OR of 32-bit masks a and b. If the resul... |  | — |
| `_kortestz_mask64_u8` | Compute the bitwise OR of 64-bit masks a and b. If the resul... |  | — |
| `_kortestz_mask8_u8` | Compute the bitwise OR of 8-bit masks a and b. If the result... |  | — |
| `_kshiftli_mask16` | Shift 16-bit mask a left by count bits while shifting in zer... |  | — |
| `_kshiftli_mask32` | Shift the bits of 32-bit mask a left by count while shifting... |  | — |
| `_kshiftli_mask64` | Shift the bits of 64-bit mask a left by count while shifting... |  | — |
| `_kshiftli_mask8` | Shift 8-bit mask a left by count bits while shifting in zero... |  | — |
| `_kshiftri_mask16` | Shift 16-bit mask a right by count bits while shifting in ze... |  | — |
| `_kshiftri_mask32` | Shift the bits of 32-bit mask a right by count while shiftin... |  | — |
| `_kshiftri_mask64` | Shift the bits of 64-bit mask a right by count while shiftin... |  | — |
| `_kshiftri_mask8` | Shift 8-bit mask a right by count bits while shifting in zer... |  | — |
| `_ktestc_mask16_u8` | Compute the bitwise NOT of 16-bit mask a and then AND with 1... |  | — |
| `_ktestc_mask32_u8` | Compute the bitwise NOT of 32-bit mask a and then AND with 1... |  | — |
| `_ktestc_mask64_u8` | Compute the bitwise NOT of 64-bit mask a and then AND with 8... |  | — |
| `_ktestc_mask8_u8` | Compute the bitwise NOT of 8-bit mask a and then AND with 8-... |  | — |
| `_ktestz_mask16_u8` | Compute the bitwise AND of 16-bit masks a and  b, if the res... |  | — |
| `_ktestz_mask32_u8` | Compute the bitwise AND of 32-bit masks a and  b, if the res... |  | — |
| `_ktestz_mask64_u8` | Compute the bitwise AND of 64-bit masks a and  b, if the res... |  | — |
| `_ktestz_mask8_u8` | Compute the bitwise AND of 8-bit masks a and  b, if the resu... |  | — |
| `_kxnor_mask16` | Compute the bitwise XNOR of 16-bit masks a and b, and store ... | xor | — |
| `_kxnor_mask32` | Compute the bitwise XNOR of 32-bit masks a and b, and store ... |  | — |
| `_kxnor_mask64` | Compute the bitwise XNOR of 64-bit masks a and b, and store ... |  | — |
| `_kxnor_mask8` | Bitwise XNOR of 8-bit masks a and b, and store the result in... |  | — |
| `_kxor_mask16` | Compute the bitwise XOR of 16-bit masks a and b, and store t... | xor | — |
| `_kxor_mask32` | Compute the bitwise XOR of 32-bit masks a and b, and store t... |  | — |
| `_kxor_mask64` | Compute the bitwise XOR of 64-bit masks a and b, and store t... |  | — |
| `_kxor_mask8` | Bitwise XOR of 8-bit masks a and b, and store the result in ... |  | — |
| `_mm256_abs_epi64` | Compute the absolute value of packed signed 64-bit integers ... | vpabsq | — |
| `_mm256_alignr_epi32` | Concatenate a and b into a 64-byte immediate result, shift t... | valignd | — |
| `_mm256_alignr_epi64` | Concatenate a and b into a 64-byte immediate result, shift t... | valignq | — |
| `_mm256_broadcast_f32x2` | Broadcasts the lower 2 packed single-precision (32-bit) floa... |  | — |
| `_mm256_broadcast_f32x4` | Broadcast the 4 packed single-precision (32-bit) floating-po... |  | — |
| `_mm256_broadcast_f64x2` | Broadcasts the 2 packed double-precision (64-bit) floating-p... |  | — |
| `_mm256_broadcast_i32x2` | Broadcasts the lower 2 packed 32-bit integers from a to all ... |  | — |
| `_mm256_broadcast_i32x4` | Broadcast the 4 packed 32-bit integers from a to all element... |  | — |
| `_mm256_broadcast_i64x2` | Broadcasts the 2 packed 64-bit integers from a to all elemen... |  | — |
| `_mm256_broadcastmb_epi64` | Broadcast the low 8-bits from input mask k to all 64-bit ele... | vpbroadcast | — |
| `_mm256_broadcastmw_epi32` | Broadcast the low 16-bits from input mask k to all 32-bit el... | vpbroadcast | — |
| `_mm256_cmp_epi16_mask` | Compare packed signed 16-bit integers in a and b based on th... | vpcmp | — |
| `_mm256_cmp_epi32_mask` | Compare packed signed 32-bit integers in a and b based on th... | vpcmp | — |
| `_mm256_cmp_epi64_mask` | Compare packed signed 64-bit integers in a and b based on th... | vpcmp | — |
| `_mm256_cmp_epi8_mask` | Compare packed signed 8-bit integers in a and b based on the... | vpcmp | — |
| `_mm256_cmp_epu16_mask` | Compare packed unsigned 16-bit integers in a and b based on ... | vpcmp | — |
| `_mm256_cmp_epu32_mask` | Compare packed unsigned 32-bit integers in a and b based on ... | vpcmp | — |
| `_mm256_cmp_epu64_mask` | Compare packed unsigned 64-bit integers in a and b based on ... | vpcmp | — |
| `_mm256_cmp_epu8_mask` | Compare packed unsigned 8-bit integers in a and b based on t... | vpcmp | — |
| `_mm256_cmp_pd_mask` | Compare packed double-precision (64-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm256_cmp_ps_mask` | Compare packed single-precision (32-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm256_cmpeq_epi16_mask` | Compare packed signed 16-bit integers in a and b for equalit... | vpcmp | 1/1, 1/1 |
| `_mm256_cmpeq_epi32_mask` | Compare packed 32-bit integers in a and b for equality, and ... | vpcmp | 1/1, 1/1 |
| `_mm256_cmpeq_epi64_mask` | Compare packed 64-bit integers in a and b for equality, and ... | vpcmp | 1/1, 1/1 |
| `_mm256_cmpeq_epi8_mask` | Compare packed signed 8-bit integers in a and b for equality... | vpcmp | 1/1, 1/1 |
| `_mm256_cmpeq_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for equal... | vpcmp | — |
| `_mm256_cmpeq_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for equal... | vpcmp | — |
| `_mm256_cmpeq_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for equal... | vpcmp | — |
| `_mm256_cmpeq_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for equali... | vpcmp | — |
| `_mm256_cmpge_epi16_mask` | Compare packed signed 16-bit integers in a and b for greater... | vpcmp | — |
| `_mm256_cmpge_epi32_mask` | Compare packed signed 32-bit integers in a and b for greater... | vpcmp | — |
| `_mm256_cmpge_epi64_mask` | Compare packed signed 64-bit integers in a and b for greater... | vpcmp | — |
| `_mm256_cmpge_epi8_mask` | Compare packed signed 8-bit integers in a and b for greater-... | vpcmp | — |
| `_mm256_cmpge_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for great... | vpcmp | — |
| `_mm256_cmpge_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for great... | vpcmp | — |
| `_mm256_cmpge_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for great... | vpcmp | — |
| `_mm256_cmpge_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for greate... | vpcmp | — |
| `_mm256_cmpgt_epi16_mask` | Compare packed signed 16-bit integers in a and b for greater... | vpcmp | 1/1, 1/1 |
| `_mm256_cmpgt_epi32_mask` | Compare packed signed 32-bit integers in a and b for greater... | vpcmp | 1/1, 1/1 |
| `_mm256_cmpgt_epi64_mask` | Compare packed signed 64-bit integers in a and b for greater... | vpcmp | 1/1, 1/1 |
| `_mm256_cmpgt_epi8_mask` | Compare packed signed 8-bit integers in a and b for greater-... | vpcmp | 1/1, 1/1 |
| `_mm256_cmpgt_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for great... | vpcmp | — |
| `_mm256_cmpgt_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for great... | vpcmp | — |
| `_mm256_cmpgt_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for great... | vpcmp | — |
| `_mm256_cmpgt_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for greate... | vpcmp | — |
| `_mm256_cmple_epi16_mask` | Compare packed signed 16-bit integers in a and b for less-th... | vpcmp | — |
| `_mm256_cmple_epi32_mask` | Compare packed signed 32-bit integers in a and b for less-th... | vpcmp | — |
| `_mm256_cmple_epi64_mask` | Compare packed signed 64-bit integers in a and b for less-th... | vpcmp | — |
| `_mm256_cmple_epi8_mask` | Compare packed signed 8-bit integers in a and b for less-tha... | vpcmp | — |
| `_mm256_cmple_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for less-... | vpcmp | — |
| `_mm256_cmple_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for less-... | vpcmp | — |
| `_mm256_cmple_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for less-... | vpcmp | — |
| `_mm256_cmple_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for less-t... | vpcmp | — |
| `_mm256_cmplt_epi16_mask` | Compare packed signed 16-bit integers in a and b for less-th... | vpcmp | 1/1, 1/1 |
| `_mm256_cmplt_epi32_mask` | Compare packed signed 32-bit integers in a and b for less-th... | vpcmp | 1/1, 1/1 |
| `_mm256_cmplt_epi64_mask` | Compare packed signed 64-bit integers in a and b for less-th... | vpcmp | 1/1, 1/1 |
| `_mm256_cmplt_epi8_mask` | Compare packed signed 8-bit integers in a and b for less-tha... | vpcmp | 1/1, 1/1 |
| `_mm256_cmplt_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for less-... | vpcmp | — |
| `_mm256_cmplt_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for less-... | vpcmp | — |
| `_mm256_cmplt_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for less-... | vpcmp | — |
| `_mm256_cmplt_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for less-t... | vpcmp | — |
| `_mm256_cmpneq_epi16_mask` | Compare packed signed 16-bit integers in a and b for not-equ... | vpcmp | — |
| `_mm256_cmpneq_epi32_mask` | Compare packed 32-bit integers in a and b for not-equal, and... | vpcmp | — |
| `_mm256_cmpneq_epi64_mask` | Compare packed signed 64-bit integers in a and b for not-equ... | vpcmp | — |
| `_mm256_cmpneq_epi8_mask` | Compare packed signed 8-bit integers in a and b for not-equa... | vpcmp | — |
| `_mm256_cmpneq_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for not-e... | vpcmp | — |
| `_mm256_cmpneq_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for not-e... | vpcmp | — |
| `_mm256_cmpneq_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for not-e... | vpcmp | — |
| `_mm256_cmpneq_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for not-eq... | vpcmp | — |
| `_mm256_conflict_epi32` | Test each 32-bit element of a for equality with all other el... | vpconflictd | — |
| `_mm256_conflict_epi64` | Test each 64-bit element of a for equality with all other el... | vpconflictq | — |
| `_mm256_cvtepi16_epi8` | Convert packed 16-bit integers in a to packed 8-bit integers... | vpmovwb | 4/1, 3/1 |
| `_mm256_cvtepi32_epi16` | Convert packed 32-bit integers in a to packed 16-bit integer... | vpmovdw | 4/1, 3/1 |
| `_mm256_cvtepi32_epi8` | Convert packed 32-bit integers in a to packed 8-bit integers... | vpmovdb | 4/1, 3/1 |
| `_mm256_cvtepi64_epi16` | Convert packed 64-bit integers in a to packed 16-bit integer... | vpmovqw | 4/1, 3/1 |
| `_mm256_cvtepi64_epi32` | Convert packed 64-bit integers in a to packed 32-bit integer... | vpmovqd | 4/1, 3/1 |
| `_mm256_cvtepi64_epi8` | Convert packed 64-bit integers in a to packed 8-bit integers... | vpmovqb | 4/1, 3/1 |
| `_mm256_cvtepi64_pd` | Convert packed signed 64-bit integers in a to packed double-... | vcvtqq2pd | 4/1, 3/1 |
| `_mm256_cvtepi64_ps` | Convert packed signed 64-bit integers in a to packed single-... | vcvtqq2ps | 4/1, 3/1 |
| `_mm256_cvtepu32_pd` | Convert packed unsigned 32-bit integers in a to packed doubl... | vcvtudq2pd | 4/1, 3/1 |
| `_mm256_cvtepu64_pd` | Convert packed unsigned 64-bit integers in a to packed doubl... | vcvtuqq2pd | 4/1, 3/1 |
| `_mm256_cvtepu64_ps` | Convert packed unsigned 64-bit integers in a to packed singl... | vcvtuqq2ps | 4/1, 3/1 |
| `_mm256_cvtpd_epi64` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2qq | 4/1, 3/1 |
| `_mm256_cvtpd_epu32` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2udq | 4/1, 3/1 |
| `_mm256_cvtpd_epu64` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2uqq | 4/1, 3/1 |
| `_mm256_cvtps_epi64` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2qq | 4/1, 3/1 |
| `_mm256_cvtps_epu32` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2udq | 4/1, 3/1 |
| `_mm256_cvtps_epu64` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2uqq | 4/1, 3/1 |
| `_mm256_cvtsepi16_epi8` | Convert packed signed 16-bit integers in a to packed 8-bit i... | vpmovswb | 4/1, 3/1 |
| `_mm256_cvtsepi32_epi16` | Convert packed signed 32-bit integers in a to packed 16-bit ... | vpmovsdw | 4/1, 3/1 |
| `_mm256_cvtsepi32_epi8` | Convert packed signed 32-bit integers in a to packed 8-bit i... | vpmovsdb | 4/1, 3/1 |
| `_mm256_cvtsepi64_epi16` | Convert packed signed 64-bit integers in a to packed 16-bit ... | vpmovsqw | 4/1, 3/1 |
| `_mm256_cvtsepi64_epi32` | Convert packed signed 64-bit integers in a to packed 32-bit ... | vpmovsqd | 4/1, 3/1 |
| `_mm256_cvtsepi64_epi8` | Convert packed signed 64-bit integers in a to packed 8-bit i... | vpmovsqb | 4/1, 3/1 |
| `_mm256_cvttpd_epi64` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2qq | 4/1, 3/1 |
| `_mm256_cvttpd_epu32` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2udq | 4/1, 3/1 |
| `_mm256_cvttpd_epu64` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2uqq | 4/1, 3/1 |
| `_mm256_cvttps_epi64` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2qq | 4/1, 3/1 |
| `_mm256_cvttps_epu32` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2udq | 4/1, 3/1 |
| `_mm256_cvttps_epu64` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2uqq | 4/1, 3/1 |
| `_mm256_cvtusepi16_epi8` | Convert packed unsigned 16-bit integers in a to packed unsig... | vpmovuswb | 4/1, 3/1 |
| `_mm256_cvtusepi32_epi16` | Convert packed unsigned 32-bit integers in a to packed unsig... | vpmovusdw | 4/1, 3/1 |
| `_mm256_cvtusepi32_epi8` | Convert packed unsigned 32-bit integers in a to packed unsig... | vpmovusdb | 4/1, 3/1 |
| `_mm256_cvtusepi64_epi16` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqw | 4/1, 3/1 |
| `_mm256_cvtusepi64_epi32` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqd | 4/1, 3/1 |
| `_mm256_cvtusepi64_epi8` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqb | 4/1, 3/1 |
| `_mm256_dbsad_epu8` | Compute the sum of absolute differences (SADs) of quadruplet... | vdbpsadbw | — |
| `_mm256_extractf32x4_ps` | Extract 128 bits (composed of 4 packed single-precision (32-... | vextract | — |
| `_mm256_extractf64x2_pd` | Extracts 128 bits (composed of 2 packed double-precision (64... |  | — |
| `_mm256_extracti32x4_epi32` | Extract 128 bits (composed of 4 packed 32-bit integers) from... | vextract | — |
| `_mm256_extracti64x2_epi64` | Extracts 128 bits (composed of 2 packed 64-bit integers) fro... |  | — |
| `_mm256_fixupimm_pd` | Fix up packed double-precision (64-bit) floating-point eleme... | vfixupimmpd | — |
| `_mm256_fixupimm_ps` | Fix up packed single-precision (32-bit) floating-point eleme... | vfixupimmps | — |
| `_mm256_fpclass_pd_mask` | Test packed double-precision (64-bit) floating-point element... | vfpclasspd | — |
| `_mm256_fpclass_ps_mask` | Test packed single-precision (32-bit) floating-point element... | vfpclassps | — |
| `_mm256_getexp_pd` | Convert the exponent of each packed double-precision (64-bit... | vgetexppd | — |
| `_mm256_getexp_ps` | Convert the exponent of each packed single-precision (32-bit... | vgetexpps | — |
| `_mm256_getmant_pd` | Normalize the mantissas of packed double-precision (64-bit) ... | vgetmantpd | — |
| `_mm256_getmant_ps` | Normalize the mantissas of packed single-precision (32-bit) ... | vgetmantps | — |
| `_mm256_insertf32x4` | Copy a to dst, then insert 128 bits (composed of 4 packed si... | vinsert | — |
| `_mm256_insertf64x2` | Copy a to dst, then insert 128 bits (composed of 2 packed do... |  | — |
| `_mm256_inserti32x4` | Copy a to dst, then insert 128 bits (composed of 4 packed 32... | vinsert | — |
| `_mm256_inserti64x2` | Copy a to dst, then insert 128 bits (composed of 2 packed 64... |  | — |
| `_mm256_lzcnt_epi32` | Counts the number of leading zero bits in each packed 32-bit... | vplzcntd | — |
| `_mm256_lzcnt_epi64` | Counts the number of leading zero bits in each packed 64-bit... | vplzcntq | — |
| `_mm256_mask2_permutex2var_epi16` | Shuffle 16-bit integers in a and b across lanes using the co... | vpermi2w | 3/1, 2/1 |
| `_mm256_mask2_permutex2var_epi32` | Shuffle 32-bit integers in a and b across lanes using the co... | vpermi2d | 3/1, 2/1 |
| `_mm256_mask2_permutex2var_epi64` | Shuffle 64-bit integers in a and b across lanes using the co... | vpermi2q | 3/1, 2/1 |
| `_mm256_mask2_permutex2var_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vperm | 3/1, 2/1 |
| `_mm256_mask2_permutex2var_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vperm | 3/1, 2/1 |
| `_mm256_mask3_fmadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmadd | 5/1, 4/1 |
| `_mm256_mask3_fmadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmadd | 5/1, 4/1 |
| `_mm256_mask3_fmaddsub_pd` | Multiply packed single-precision (32-bit) floating-point ele... | vfmaddsub | 5/1, 4/1 |
| `_mm256_mask3_fmaddsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmaddsub | 5/1, 4/1 |
| `_mm256_mask3_fmsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsub | 5/1, 4/1 |
| `_mm256_mask3_fmsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsub | 5/1, 4/1 |
| `_mm256_mask3_fmsubadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsubadd | 5/1, 4/1 |
| `_mm256_mask3_fmsubadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsubadd | 5/1, 4/1 |
| `_mm256_mask3_fnmadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmadd | 5/1, 4/1 |
| `_mm256_mask3_fnmadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmadd | 5/1, 4/1 |
| `_mm256_mask3_fnmsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmsub | 5/1, 4/1 |
| `_mm256_mask3_fnmsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmsub | 5/1, 4/1 |
| `_mm256_mask_abs_epi16` | Compute the absolute value of packed signed 16-bit integers ... | vpabsw | — |
| `_mm256_mask_abs_epi32` | Compute the absolute value of packed signed 32-bit integers ... | vpabsd | — |
| `_mm256_mask_abs_epi64` | Compute the absolute value of packed signed 64-bit integers ... | vpabsq | — |
| `_mm256_mask_abs_epi8` | Compute the absolute value of packed signed 8-bit integers i... | vpabsb | — |
| `_mm256_mask_add_epi16` | Add packed 16-bit integers in a and b, and store the results... | vpaddw | 1/1, 1/1 |
| `_mm256_mask_add_epi32` | Add packed 32-bit integers in a and b, and store the results... | vpaddd | 1/1, 1/1 |
| `_mm256_mask_add_epi64` | Add packed 64-bit integers in a and b, and store the results... | vpaddq | 1/1, 1/1 |
| `_mm256_mask_add_epi8` | Add packed 8-bit integers in a and b, and store the results ... | vpaddb | 1/1, 1/1 |
| `_mm256_mask_add_pd` | Add packed double-precision (64-bit) floating-point elements... | vaddpd | 3/1, 3/1 |
| `_mm256_mask_add_ps` | Add packed single-precision (32-bit) floating-point elements... | vaddps | 3/1, 3/1 |
| `_mm256_mask_adds_epi16` | Add packed signed 16-bit integers in a and b using saturatio... | vpaddsw | 1/1, 1/1 |
| `_mm256_mask_adds_epi8` | Add packed signed 8-bit integers in a and b using saturation... | vpaddsb | 1/1, 1/1 |
| `_mm256_mask_adds_epu16` | Add packed unsigned 16-bit integers in a and b using saturat... | vpaddusw | — |
| `_mm256_mask_adds_epu8` | Add packed unsigned 8-bit integers in a and b using saturati... | vpaddusb | — |
| `_mm256_mask_alignr_epi32` | Concatenate a and b into a 64-byte immediate result, shift t... | valignd | — |
| `_mm256_mask_alignr_epi64` | Concatenate a and b into a 64-byte immediate result, shift t... | valignq | — |
| `_mm256_mask_alignr_epi8` | Concatenate pairs of 16-byte blocks in a and b into a 32-byt... | vpalignr | — |
| `_mm256_mask_and_epi32` | Performs element-by-element bitwise AND between packed 32-bi... | vpandd | 1/1, 1/1 |
| `_mm256_mask_and_epi64` | Compute the bitwise AND of packed 64-bit integers in a and b... | vpandq | 1/1, 1/1 |
| `_mm256_mask_and_pd` | Compute the bitwise AND of packed double-precision (64-bit) ... | vandpd | 1/1, 1/1 |
| `_mm256_mask_and_ps` | Compute the bitwise AND of packed single-precision (32-bit) ... | vandps | 1/1, 1/1 |
| `_mm256_mask_andnot_epi32` | Compute the bitwise NOT of packed 32-bit integers in a and t... | vpandnd | 1/1, 1/1 |
| `_mm256_mask_andnot_epi64` | Compute the bitwise NOT of packed 64-bit integers in a and t... | vpandnq | 1/1, 1/1 |
| `_mm256_mask_andnot_pd` | Compute the bitwise NOT of packed double-precision (64-bit) ... | vandnpd | 1/1, 1/1 |
| `_mm256_mask_andnot_ps` | Compute the bitwise NOT of packed single-precision (32-bit) ... | vandnps | 1/1, 1/1 |
| `_mm256_mask_avg_epu16` | Average packed unsigned 16-bit integers in a and b, and stor... | vpavgw | — |
| `_mm256_mask_avg_epu8` | Average packed unsigned 8-bit integers in a and b, and store... | vpavgb | — |
| `_mm256_mask_blend_epi16` | Blend packed 16-bit integers from a and b using control mask... | vmovdqu16 | 1/1, 1/1 |
| `_mm256_mask_blend_epi32` | Blend packed 32-bit integers from a and b using control mask... | vmovdqa32 | 1/1, 1/1 |
| `_mm256_mask_blend_epi64` | Blend packed 64-bit integers from a and b using control mask... | vmovdqa64 | 1/1, 1/1 |
| `_mm256_mask_blend_epi8` | Blend packed 8-bit integers from a and b using control mask ... | vmovdqu8 | 1/1, 1/1 |
| `_mm256_mask_blend_pd` | Blend packed double-precision (64-bit) floating-point elemen... | vmovapd | 1/1, 1/1 |
| `_mm256_mask_blend_ps` | Blend packed single-precision (32-bit) floating-point elemen... | vmovaps | 1/1, 1/1 |
| `_mm256_mask_broadcast_f32x2` | Broadcasts the lower 2 packed single-precision (32-bit) floa... | vbroadcastf32x2 | — |
| `_mm256_mask_broadcast_f32x4` | Broadcast the 4 packed single-precision (32-bit) floating-po... |  | — |
| `_mm256_mask_broadcast_f64x2` | Broadcasts the 2 packed double-precision (64-bit) floating-p... |  | — |
| `_mm256_mask_broadcast_i32x2` | Broadcasts the lower 2 packed 32-bit integers from a to all ... | vbroadcasti32x2 | — |
| `_mm256_mask_broadcast_i32x4` | Broadcast the 4 packed 32-bit integers from a to all element... |  | — |
| `_mm256_mask_broadcast_i64x2` | Broadcasts the 2 packed 64-bit integers from a to all elemen... |  | — |
| `_mm256_mask_broadcastb_epi8` | Broadcast the low packed 8-bit integer from a to all element... | vpbroadcastb | — |
| `_mm256_mask_broadcastd_epi32` | Broadcast the low packed 32-bit integer from a to all elemen... | vpbroadcast | — |
| `_mm256_mask_broadcastq_epi64` | Broadcast the low packed 64-bit integer from a to all elemen... | vpbroadcast | — |
| `_mm256_mask_broadcastsd_pd` | Broadcast the low double-precision (64-bit) floating-point e... | vbroadcastsd | — |
| `_mm256_mask_broadcastss_ps` | Broadcast the low single-precision (32-bit) floating-point e... | vbroadcastss | — |
| `_mm256_mask_broadcastw_epi16` | Broadcast the low packed 16-bit integer from a to all elemen... | vpbroadcastw | — |
| `_mm256_mask_cmp_epi16_mask` | Compare packed signed 16-bit integers in a and b based on th... | vpcmp | — |
| `_mm256_mask_cmp_epi32_mask` | Compare packed signed 32-bit integers in a and b based on th... | vpcmp | — |
| `_mm256_mask_cmp_epi64_mask` | Compare packed signed 64-bit integers in a and b based on th... | vpcmp | — |
| `_mm256_mask_cmp_epi8_mask` | Compare packed signed 8-bit integers in a and b based on the... | vpcmp | — |
| `_mm256_mask_cmp_epu16_mask` | Compare packed unsigned 16-bit integers in a and b based on ... | vpcmp | — |
| `_mm256_mask_cmp_epu32_mask` | Compare packed unsigned 32-bit integers in a and b based on ... | vpcmp | — |
| `_mm256_mask_cmp_epu64_mask` | Compare packed unsigned 64-bit integers in a and b based on ... | vpcmp | — |
| `_mm256_mask_cmp_epu8_mask` | Compare packed unsigned 8-bit integers in a and b based on t... | vpcmp | — |
| `_mm256_mask_cmp_pd_mask` | Compare packed double-precision (64-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm256_mask_cmp_ps_mask` | Compare packed single-precision (32-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm256_mask_cmpeq_epi16_mask` | Compare packed signed 16-bit integers in a and b for equalit... | vpcmp | 1/1, 1/1 |
| `_mm256_mask_cmpeq_epi32_mask` | Compare packed 32-bit integers in a and b for equality, and ... | vpcmp | 1/1, 1/1 |
| `_mm256_mask_cmpeq_epi64_mask` | Compare packed 64-bit integers in a and b for equality, and ... | vpcmp | 1/1, 1/1 |
| `_mm256_mask_cmpeq_epi8_mask` | Compare packed signed 8-bit integers in a and b for equality... | vpcmp | 1/1, 1/1 |
| `_mm256_mask_cmpeq_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for equal... | vpcmp | — |
| `_mm256_mask_cmpeq_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for equal... | vpcmp | — |
| `_mm256_mask_cmpeq_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for equal... | vpcmp | — |
| `_mm256_mask_cmpeq_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for equali... | vpcmp | — |
| `_mm256_mask_cmpge_epi16_mask` | Compare packed signed 16-bit integers in a and b for greater... | vpcmp | — |
| `_mm256_mask_cmpge_epi32_mask` | Compare packed signed 32-bit integers in a and b for greater... | vpcmp | — |
| `_mm256_mask_cmpge_epi64_mask` | Compare packed signed 64-bit integers in a and b for greater... | vpcmp | — |
| `_mm256_mask_cmpge_epi8_mask` | Compare packed signed 8-bit integers in a and b for greater-... | vpcmp | — |
| `_mm256_mask_cmpge_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for great... | vpcmp | — |
| `_mm256_mask_cmpge_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for great... | vpcmp | — |
| `_mm256_mask_cmpge_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for great... | vpcmp | — |
| `_mm256_mask_cmpge_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for greate... | vpcmp | — |
| `_mm256_mask_cmpgt_epi16_mask` | Compare packed signed 16-bit integers in a and b for greater... | vpcmp | 1/1, 1/1 |
| `_mm256_mask_cmpgt_epi32_mask` | Compare packed signed 32-bit integers in a and b for greater... | vpcmp | 1/1, 1/1 |
| `_mm256_mask_cmpgt_epi64_mask` | Compare packed signed 64-bit integers in a and b for greater... | vpcmp | 1/1, 1/1 |
| `_mm256_mask_cmpgt_epi8_mask` | Compare packed signed 8-bit integers in a and b for greater-... | vpcmp | 1/1, 1/1 |
| `_mm256_mask_cmpgt_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for great... | vpcmp | — |
| `_mm256_mask_cmpgt_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for great... | vpcmp | — |
| `_mm256_mask_cmpgt_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for great... | vpcmp | — |
| `_mm256_mask_cmpgt_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for greate... | vpcmp | — |
| `_mm256_mask_cmple_epi16_mask` | Compare packed signed 16-bit integers in a and b for less-th... | vpcmp | — |
| `_mm256_mask_cmple_epi32_mask` | Compare packed signed 32-bit integers in a and b for less-th... | vpcmp | — |
| `_mm256_mask_cmple_epi64_mask` | Compare packed signed 64-bit integers in a and b for less-th... | vpcmp | — |
| `_mm256_mask_cmple_epi8_mask` | Compare packed signed 8-bit integers in a and b for less-tha... | vpcmp | — |
| `_mm256_mask_cmple_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for less-... | vpcmp | — |
| `_mm256_mask_cmple_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for less-... | vpcmp | — |
| `_mm256_mask_cmple_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for less-... | vpcmp | — |
| `_mm256_mask_cmple_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for less-t... | vpcmp | — |
| `_mm256_mask_cmplt_epi16_mask` | Compare packed signed 16-bit integers in a and b for less-th... | vpcmp | 1/1, 1/1 |
| `_mm256_mask_cmplt_epi32_mask` | Compare packed signed 32-bit integers in a and b for less-th... | vpcmp | 1/1, 1/1 |
| `_mm256_mask_cmplt_epi64_mask` | Compare packed signed 64-bit integers in a and b for less-th... | vpcmp | 1/1, 1/1 |
| `_mm256_mask_cmplt_epi8_mask` | Compare packed signed 8-bit integers in a and b for less-tha... | vpcmp | 1/1, 1/1 |
| `_mm256_mask_cmplt_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for less-... | vpcmp | — |
| `_mm256_mask_cmplt_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for less-... | vpcmp | — |
| `_mm256_mask_cmplt_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for less-... | vpcmp | — |
| `_mm256_mask_cmplt_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for less-t... | vpcmp | — |
| `_mm256_mask_cmpneq_epi16_mask` | Compare packed signed 16-bit integers in a and b for not-equ... | vpcmp | — |
| `_mm256_mask_cmpneq_epi32_mask` | Compare packed 32-bit integers in a and b for not-equal, and... | vpcmp | — |
| `_mm256_mask_cmpneq_epi64_mask` | Compare packed signed 64-bit integers in a and b for not-equ... | vpcmp | — |
| `_mm256_mask_cmpneq_epi8_mask` | Compare packed signed 8-bit integers in a and b for not-equa... | vpcmp | — |
| `_mm256_mask_cmpneq_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for not-e... | vpcmp | — |
| `_mm256_mask_cmpneq_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for not-e... | vpcmp | — |
| `_mm256_mask_cmpneq_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for not-e... | vpcmp | — |
| `_mm256_mask_cmpneq_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for not-eq... | vpcmp | — |
| `_mm256_mask_compress_epi32` | Contiguously store the active 32-bit integers in a (those wi... | vpcompressd | — |
| `_mm256_mask_compress_epi64` | Contiguously store the active 64-bit integers in a (those wi... | vpcompressq | — |
| `_mm256_mask_compress_pd` | Contiguously store the active double-precision (64-bit) floa... | vcompresspd | — |
| `_mm256_mask_compress_ps` | Contiguously store the active single-precision (32-bit) floa... | vcompressps | — |
| `_mm256_mask_conflict_epi32` | Test each 32-bit element of a for equality with all other el... | vpconflictd | — |
| `_mm256_mask_conflict_epi64` | Test each 64-bit element of a for equality with all other el... | vpconflictq | — |
| `_mm256_mask_cvt_roundps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2ph | 4/1, 3/1 |
| `_mm256_mask_cvtepi16_epi32` | Sign extend packed 16-bit integers in a to packed 32-bit int... | vpmovsxwd | 4/1, 3/1 |
| `_mm256_mask_cvtepi16_epi64` | Sign extend packed 16-bit integers in a to packed 64-bit int... | vpmovsxwq | 4/1, 3/1 |
| `_mm256_mask_cvtepi16_epi8` | Convert packed 16-bit integers in a to packed 8-bit integers... | vpmovwb | 4/1, 3/1 |
| `_mm256_mask_cvtepi32_epi16` | Convert packed 32-bit integers in a to packed 16-bit integer... | vpmovdw | 4/1, 3/1 |
| `_mm256_mask_cvtepi32_epi64` | Sign extend packed 32-bit integers in a to packed 64-bit int... | vpmovsxdq | 4/1, 3/1 |
| `_mm256_mask_cvtepi32_epi8` | Convert packed 32-bit integers in a to packed 8-bit integers... | vpmovdb | 4/1, 3/1 |
| `_mm256_mask_cvtepi32_pd` | Convert packed signed 32-bit integers in a to packed double-... | vcvtdq2pd | 4/1, 3/1 |
| `_mm256_mask_cvtepi32_ps` | Convert packed signed 32-bit integers in a to packed single-... | vcvtdq2ps | 4/1, 3/1 |
| `_mm256_mask_cvtepi64_epi16` | Convert packed 64-bit integers in a to packed 16-bit integer... | vpmovqw | 4/1, 3/1 |
| `_mm256_mask_cvtepi64_epi32` | Convert packed 64-bit integers in a to packed 32-bit integer... | vpmovqd | 4/1, 3/1 |
| `_mm256_mask_cvtepi64_epi8` | Convert packed 64-bit integers in a to packed 8-bit integers... | vpmovqb | 4/1, 3/1 |
| `_mm256_mask_cvtepi64_pd` | Convert packed signed 64-bit integers in a to packed double-... | vcvtqq2pd | 4/1, 3/1 |
| `_mm256_mask_cvtepi64_ps` | Convert packed signed 64-bit integers in a to packed single-... | vcvtqq2ps | 4/1, 3/1 |
| `_mm256_mask_cvtepi8_epi16` | Sign extend packed 8-bit integers in a to packed 16-bit inte... | vpmovsxbw | 4/1, 3/1 |
| `_mm256_mask_cvtepi8_epi32` | Sign extend packed 8-bit integers in a to packed 32-bit inte... | vpmovsxbd | 4/1, 3/1 |
| `_mm256_mask_cvtepi8_epi64` | Sign extend packed 8-bit integers in the low 4 bytes of a to... | vpmovsxbq | 4/1, 3/1 |
| `_mm256_mask_cvtepu16_epi32` | Zero extend packed unsigned 16-bit integers in a to packed 3... | vpmovzxwd | 4/1, 3/1 |
| `_mm256_mask_cvtepu16_epi64` | Zero extend packed unsigned 16-bit integers in the low 8 byt... | vpmovzxwq | 4/1, 3/1 |
| `_mm256_mask_cvtepu32_epi64` | Zero extend packed unsigned 32-bit integers in a to packed 6... | vpmovzxdq | 4/1, 3/1 |
| `_mm256_mask_cvtepu32_pd` | Convert packed unsigned 32-bit integers in a to packed doubl... | vcvtudq2pd | 4/1, 3/1 |
| `_mm256_mask_cvtepu64_pd` | Convert packed unsigned 64-bit integers in a to packed doubl... | vcvtuqq2pd | 4/1, 3/1 |
| `_mm256_mask_cvtepu64_ps` | Convert packed unsigned 64-bit integers in a to packed singl... | vcvtuqq2ps | 4/1, 3/1 |
| `_mm256_mask_cvtepu8_epi16` | Zero extend packed unsigned 8-bit integers in a to packed 16... | vpmovzxbw | 4/1, 3/1 |
| `_mm256_mask_cvtepu8_epi32` | Zero extend packed unsigned 8-bit integers in the low 8 byte... | vpmovzxbd | 4/1, 3/1 |
| `_mm256_mask_cvtepu8_epi64` | Zero extend packed unsigned 8-bit integers in the low 4 byte... | vpmovzxbq | 4/1, 3/1 |
| `_mm256_mask_cvtpd_epi32` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2dq | 4/1, 3/1 |
| `_mm256_mask_cvtpd_epi64` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2qq | 4/1, 3/1 |
| `_mm256_mask_cvtpd_epu32` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2udq | 4/1, 3/1 |
| `_mm256_mask_cvtpd_epu64` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2uqq | 4/1, 3/1 |
| `_mm256_mask_cvtpd_ps` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2ps | 4/1, 3/1 |
| `_mm256_mask_cvtph_ps` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2ps | 4/1, 3/1 |
| `_mm256_mask_cvtps_epi32` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2dq | 4/1, 3/1 |
| `_mm256_mask_cvtps_epi64` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2qq | 4/1, 3/1 |
| `_mm256_mask_cvtps_epu32` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2udq | 4/1, 3/1 |
| `_mm256_mask_cvtps_epu64` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2uqq | 4/1, 3/1 |
| `_mm256_mask_cvtps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2ph | 4/1, 3/1 |
| `_mm256_mask_cvtsepi16_epi8` | Convert packed signed 16-bit integers in a to packed 8-bit i... | vpmovswb | 4/1, 3/1 |
| `_mm256_mask_cvtsepi32_epi16` | Convert packed signed 32-bit integers in a to packed 16-bit ... | vpmovsdw | 4/1, 3/1 |
| `_mm256_mask_cvtsepi32_epi8` | Convert packed signed 32-bit integers in a to packed 8-bit i... | vpmovsdb | 4/1, 3/1 |
| `_mm256_mask_cvtsepi64_epi16` | Convert packed signed 64-bit integers in a to packed 16-bit ... | vpmovsqw | 4/1, 3/1 |
| `_mm256_mask_cvtsepi64_epi32` | Convert packed signed 64-bit integers in a to packed 32-bit ... | vpmovsqd | 4/1, 3/1 |
| `_mm256_mask_cvtsepi64_epi8` | Convert packed signed 64-bit integers in a to packed 8-bit i... | vpmovsqb | 4/1, 3/1 |
| `_mm256_mask_cvttpd_epi32` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2dq | 4/1, 3/1 |
| `_mm256_mask_cvttpd_epi64` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2qq | 4/1, 3/1 |
| `_mm256_mask_cvttpd_epu32` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2udq | 4/1, 3/1 |
| `_mm256_mask_cvttpd_epu64` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2uqq | 4/1, 3/1 |
| `_mm256_mask_cvttps_epi32` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2dq | 4/1, 3/1 |
| `_mm256_mask_cvttps_epi64` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2qq | 4/1, 3/1 |
| `_mm256_mask_cvttps_epu32` | Convert packed double-precision (32-bit) floating-point elem... | vcvttps2udq | 4/1, 3/1 |
| `_mm256_mask_cvttps_epu64` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2uqq | 4/1, 3/1 |
| `_mm256_mask_cvtusepi16_epi8` | Convert packed unsigned 16-bit integers in a to packed unsig... | vpmovuswb | 4/1, 3/1 |
| `_mm256_mask_cvtusepi32_epi16` | Convert packed unsigned 32-bit integers in a to packed unsig... | vpmovusdw | 4/1, 3/1 |
| `_mm256_mask_cvtusepi32_epi8` | Convert packed unsigned 32-bit integers in a to packed unsig... | vpmovusdb | 4/1, 3/1 |
| `_mm256_mask_cvtusepi64_epi16` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqw | 4/1, 3/1 |
| `_mm256_mask_cvtusepi64_epi32` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqd | 4/1, 3/1 |
| `_mm256_mask_cvtusepi64_epi8` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqb | 4/1, 3/1 |
| `_mm256_mask_dbsad_epu8` | Compute the sum of absolute differences (SADs) of quadruplet... | vdbpsadbw | — |
| `_mm256_mask_div_pd` | Divide packed double-precision (64-bit) floating-point eleme... | vdivpd | 35/28, 13/10 |
| `_mm256_mask_div_ps` | Divide packed single-precision (32-bit) floating-point eleme... | vdivps | 21/14, 10/7 |
| `_mm256_mask_expand_epi32` | Load contiguous active 32-bit integers from a (those with th... | vpexpandd | — |
| `_mm256_mask_expand_epi64` | Load contiguous active 64-bit integers from a (those with th... | vpexpandq | — |
| `_mm256_mask_expand_pd` | Load contiguous active double-precision (64-bit) floating-po... | vexpandpd | — |
| `_mm256_mask_expand_ps` | Load contiguous active single-precision (32-bit) floating-po... | vexpandps | — |
| `_mm256_mask_extractf32x4_ps` | Extract 128 bits (composed of 4 packed single-precision (32-... | vextractf32x4 | — |
| `_mm256_mask_extractf64x2_pd` | Extracts 128 bits (composed of 2 packed double-precision (64... | vextractf64x2 | — |
| `_mm256_mask_extracti32x4_epi32` | Extract 128 bits (composed of 4 packed 32-bit integers) from... | vextracti32x4 | — |
| `_mm256_mask_extracti64x2_epi64` | Extracts 128 bits (composed of 2 packed 64-bit integers) fro... | vextracti64x2 | — |
| `_mm256_mask_fixupimm_pd` | Fix up packed double-precision (64-bit) floating-point eleme... | vfixupimmpd | — |
| `_mm256_mask_fixupimm_ps` | Fix up packed single-precision (32-bit) floating-point eleme... | vfixupimmps | — |
| `_mm256_mask_fmadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmadd | 5/1, 4/1 |
| `_mm256_mask_fmadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmadd | 5/1, 4/1 |
| `_mm256_mask_fmaddsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmaddsub | 5/1, 4/1 |
| `_mm256_mask_fmaddsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmaddsub | 5/1, 4/1 |
| `_mm256_mask_fmsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsub | 5/1, 4/1 |
| `_mm256_mask_fmsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsub | 5/1, 4/1 |
| `_mm256_mask_fmsubadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsubadd | 5/1, 4/1 |
| `_mm256_mask_fmsubadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsubadd | 5/1, 4/1 |
| `_mm256_mask_fnmadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmadd | 5/1, 4/1 |
| `_mm256_mask_fnmadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmadd | 5/1, 4/1 |
| `_mm256_mask_fnmsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmsub | 5/1, 4/1 |
| `_mm256_mask_fnmsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmsub | 5/1, 4/1 |
| `_mm256_mask_fpclass_pd_mask` | Test packed double-precision (64-bit) floating-point element... | vfpclasspd | — |
| `_mm256_mask_fpclass_ps_mask` | Test packed single-precision (32-bit) floating-point element... | vfpclassps | — |
| `_mm256_mask_getexp_pd` | Convert the exponent of each packed double-precision (64-bit... | vgetexppd | — |
| `_mm256_mask_getexp_ps` | Convert the exponent of each packed single-precision (32-bit... | vgetexpps | — |
| `_mm256_mask_getmant_pd` | Normalize the mantissas of packed double-precision (64-bit) ... | vgetmantpd | — |
| `_mm256_mask_getmant_ps` | Normalize the mantissas of packed single-precision (32-bit) ... | vgetmantps | — |
| `_mm256_mask_insertf32x4` | Copy a to tmp, then insert 128 bits (composed of 4 packed si... | vinsertf32x4 | — |
| `_mm256_mask_insertf64x2` | Copy a to tmp, then insert 128 bits (composed of 2 packed do... | vinsertf64x2 | — |
| `_mm256_mask_inserti32x4` | Copy a to tmp, then insert 128 bits (composed of 4 packed 32... | vinserti32x4 | — |
| `_mm256_mask_inserti64x2` | Copy a to tmp, then insert 128 bits (composed of 2 packed 64... | vinserti64x2 | — |
| `_mm256_mask_lzcnt_epi32` | Counts the number of leading zero bits in each packed 32-bit... | vplzcntd | — |
| `_mm256_mask_lzcnt_epi64` | Counts the number of leading zero bits in each packed 64-bit... | vplzcntq | — |
| `_mm256_mask_madd_epi16` | Multiply packed signed 16-bit integers in a and b, producing... | vpmaddwd | — |
| `_mm256_mask_maddubs_epi16` | Multiply packed unsigned 8-bit integers in a by packed signe... | vpmaddubsw | — |
| `_mm256_mask_max_epi16` | Compare packed signed 16-bit integers in a and b, and store ... | vpmaxsw | — |
| `_mm256_mask_max_epi32` | Compare packed signed 32-bit integers in a and b, and store ... | vpmaxsd | — |
| `_mm256_mask_max_epi64` | Compare packed signed 64-bit integers in a and b, and store ... | vpmaxsq | — |
| `_mm256_mask_max_epi8` | Compare packed signed 8-bit integers in a and b, and store p... | vpmaxsb | — |
| `_mm256_mask_max_epu16` | Compare packed unsigned 16-bit integers in a and b, and stor... | vpmaxuw | — |
| `_mm256_mask_max_epu32` | Compare packed unsigned 32-bit integers in a and b, and stor... | vpmaxud | — |
| `_mm256_mask_max_epu64` | Compare packed unsigned 64-bit integers in a and b, and stor... | vpmaxuq | — |
| `_mm256_mask_max_epu8` | Compare packed unsigned 8-bit integers in a and b, and store... | vpmaxub | — |
| `_mm256_mask_max_pd` | Compare packed double-precision (64-bit) floating-point elem... | vmaxpd | — |
| `_mm256_mask_max_ps` | Compare packed single-precision (32-bit) floating-point elem... | vmaxps | — |
| `_mm256_mask_min_epi16` | Compare packed signed 16-bit integers in a and b, and store ... | vpminsw | — |
| `_mm256_mask_min_epi32` | Compare packed signed 32-bit integers in a and b, and store ... | vpminsd | — |
| `_mm256_mask_min_epi64` | Compare packed signed 64-bit integers in a and b, and store ... | vpminsq | — |
| `_mm256_mask_min_epi8` | Compare packed signed 8-bit integers in a and b, and store p... | vpminsb | — |
| `_mm256_mask_min_epu16` | Compare packed unsigned 16-bit integers in a and b, and stor... | vpminuw | — |
| `_mm256_mask_min_epu32` | Compare packed unsigned 32-bit integers in a and b, and stor... | vpminud | — |
| `_mm256_mask_min_epu64` | Compare packed unsigned 64-bit integers in a and b, and stor... | vpminuq | — |
| `_mm256_mask_min_epu8` | Compare packed unsigned 8-bit integers in a and b, and store... | vpminub | — |
| `_mm256_mask_min_pd` | Compare packed double-precision (64-bit) floating-point elem... | vminpd | — |
| `_mm256_mask_min_ps` | Compare packed single-precision (32-bit) floating-point elem... | vminps | — |
| `_mm256_mask_mov_epi16` | Move packed 16-bit integers from a into dst using writemask ... | vmovdqu16 | — |
| `_mm256_mask_mov_epi32` | Move packed 32-bit integers from a to dst using writemask k ... | vmovdqa32 | — |
| `_mm256_mask_mov_epi64` | Move packed 64-bit integers from a to dst using writemask k ... | vmovdqa64 | — |
| `_mm256_mask_mov_epi8` | Move packed 8-bit integers from a into dst using writemask k... | vmovdqu8 | — |
| `_mm256_mask_mov_pd` | Move packed double-precision (64-bit) floating-point element... | vmovapd | — |
| `_mm256_mask_mov_ps` | Move packed single-precision (32-bit) floating-point element... | vmovaps | — |
| `_mm256_mask_movedup_pd` | Duplicate even-indexed double-precision (64-bit) floating-po... | vmovddup | — |
| `_mm256_mask_movehdup_ps` | Duplicate odd-indexed single-precision (32-bit) floating-poi... | vmovshdup | — |
| `_mm256_mask_moveldup_ps` | Duplicate even-indexed single-precision (32-bit) floating-po... | vmovsldup | — |
| `_mm256_mask_mul_epi32` | Multiply the low signed 32-bit integers from each packed 64-... | vpmuldq | 10/2, 4/1 |
| `_mm256_mask_mul_epu32` | Multiply the low unsigned 32-bit integers from each packed 6... | vpmuludq | 10/2, 4/1 |
| `_mm256_mask_mul_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vmulpd | 5/1, 3/1 |
| `_mm256_mask_mul_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vmulps | 5/1, 3/1 |
| `_mm256_mask_mulhi_epi16` | Multiply the packed signed 16-bit integers in a and b, produ... | vpmulhw | 5/1, 3/1 |
| `_mm256_mask_mulhi_epu16` | Multiply the packed unsigned 16-bit integers in a and b, pro... | vpmulhuw | 5/1, 3/1 |
| `_mm256_mask_mulhrs_epi16` | Multiply packed signed 16-bit integers in a and b, producing... | vpmulhrsw | — |
| `_mm256_mask_mullo_epi16` | Multiply the packed 16-bit integers in a and b, producing in... | vpmullw | 5/1, 3/1 |
| `_mm256_mask_mullo_epi32` | Multiply the packed 32-bit integers in a and b, producing in... | vpmulld | 10/2, 4/1 |
| `_mm256_mask_mullo_epi64` | Multiply packed 64-bit integers in `a` and `b`, producing in... | vpmullq | — |
| `_mm256_mask_or_epi32` | Compute the bitwise OR of packed 32-bit integers in a and b,... | vpord | 1/1, 1/1 |
| `_mm256_mask_or_epi64` | Compute the bitwise OR of packed 64-bit integers in a and b,... | vporq | 1/1, 1/1 |
| `_mm256_mask_or_pd` | Compute the bitwise OR of packed double-precision (64-bit) f... | vorpd | 1/1, 1/1 |
| `_mm256_mask_or_ps` | Compute the bitwise OR of packed single-precision (32-bit) f... | vorps | 1/1, 1/1 |
| `_mm256_mask_packs_epi16` | Convert packed signed 16-bit integers from a and b to packed... | vpacksswb | 1/1, 1/1 |
| `_mm256_mask_packs_epi32` | Convert packed signed 32-bit integers from a and b to packed... | vpackssdw | 1/1, 1/1 |
| `_mm256_mask_packus_epi16` | Convert packed signed 16-bit integers from a and b to packed... | vpackuswb | 1/1, 1/1 |
| `_mm256_mask_packus_epi32` | Convert packed signed 32-bit integers from a and b to packed... | vpackusdw | 1/1, 1/1 |
| `_mm256_mask_permute_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vshufpd | 3/1, 2/1 |
| `_mm256_mask_permute_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vshufps | 3/1, 2/1 |
| `_mm256_mask_permutevar_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vpermilpd | 3/1, 2/1 |
| `_mm256_mask_permutevar_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vpermilps | 3/1, 2/1 |
| `_mm256_mask_permutex2var_epi16` | Shuffle 16-bit integers in a and b across lanes using the co... | vpermt2w | 3/1, 2/1 |
| `_mm256_mask_permutex2var_epi32` | Shuffle 32-bit integers in a and b across lanes using the co... | vpermt2d | 3/1, 2/1 |
| `_mm256_mask_permutex2var_epi64` | Shuffle 64-bit integers in a and b across lanes using the co... | vpermt2q | 3/1, 2/1 |
| `_mm256_mask_permutex2var_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vpermt2pd | 3/1, 2/1 |
| `_mm256_mask_permutex2var_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vpermt2ps | 3/1, 2/1 |
| `_mm256_mask_permutex_epi64` | Shuffle 64-bit integers in a within 256-bit lanes using the ... | vperm | 3/1, 2/1 |
| `_mm256_mask_permutex_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vperm | 3/1, 2/1 |
| `_mm256_mask_permutexvar_epi16` | Shuffle 16-bit integers in a across lanes using the correspo... | vpermw | 3/1, 2/1 |
| `_mm256_mask_permutexvar_epi32` | Shuffle 32-bit integers in a across lanes using the correspo... | vpermd | 3/1, 2/1 |
| `_mm256_mask_permutexvar_epi64` | Shuffle 64-bit integers in a across lanes using the correspo... | vpermq | 3/1, 2/1 |
| `_mm256_mask_permutexvar_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vpermpd | 3/1, 2/1 |
| `_mm256_mask_permutexvar_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vpermps | 3/1, 2/1 |
| `_mm256_mask_range_pd` | Calculate the max, min, absolute max, or absolute min (depen... | vrangepd | — |
| `_mm256_mask_range_ps` | Calculate the max, min, absolute max, or absolute min (depen... | vrangeps | — |
| `_mm256_mask_rcp14_pd` | Compute the approximate reciprocal of packed double-precisio... | vrcp14pd | — |
| `_mm256_mask_rcp14_ps` | Compute the approximate reciprocal of packed single-precisio... | vrcp14ps | — |
| `_mm256_mask_reduce_add_epi16` | Reduce the packed 16-bit integers in a by addition using mas... |  | 1/1, 1/1 |
| `_mm256_mask_reduce_add_epi8` | Reduce the packed 8-bit integers in a by addition using mask... |  | 1/1, 1/1 |
| `_mm256_mask_reduce_and_epi16` | Reduce the packed 16-bit integers in a by bitwise AND using ... |  | 1/1, 1/1 |
| `_mm256_mask_reduce_and_epi8` | Reduce the packed 8-bit integers in a by bitwise AND using m... |  | 1/1, 1/1 |
| `_mm256_mask_reduce_max_epi16` | Reduce the packed 16-bit integers in a by maximum using mask... |  | — |
| `_mm256_mask_reduce_max_epi8` | Reduce the packed 8-bit integers in a by maximum using mask ... |  | — |
| `_mm256_mask_reduce_max_epu16` | Reduce the packed unsigned 16-bit integers in a by maximum u... |  | — |
| `_mm256_mask_reduce_max_epu8` | Reduce the packed unsigned 8-bit integers in a by maximum us... |  | — |
| `_mm256_mask_reduce_min_epi16` | Reduce the packed 16-bit integers in a by minimum using mask... |  | — |
| `_mm256_mask_reduce_min_epi8` | Reduce the packed 8-bit integers in a by minimum using mask ... |  | — |
| `_mm256_mask_reduce_min_epu16` | Reduce the packed unsigned 16-bit integers in a by minimum u... |  | — |
| `_mm256_mask_reduce_min_epu8` | Reduce the packed unsigned 8-bit integers in a by minimum us... |  | — |
| `_mm256_mask_reduce_mul_epi16` | Reduce the packed 16-bit integers in a by multiplication usi... |  | — |
| `_mm256_mask_reduce_mul_epi8` | Reduce the packed 8-bit integers in a by multiplication usin... |  | — |
| `_mm256_mask_reduce_or_epi16` | Reduce the packed 16-bit integers in a by bitwise OR using m... |  | 1/1, 1/1 |
| `_mm256_mask_reduce_or_epi8` | Reduce the packed 8-bit integers in a by bitwise OR using ma... |  | 1/1, 1/1 |
| `_mm256_mask_reduce_pd` | Extract the reduced argument of packed double-precision (64-... | vreducepd | — |
| `_mm256_mask_reduce_ps` | Extract the reduced argument of packed single-precision (32-... | vreduceps | — |
| `_mm256_mask_rol_epi32` | Rotate the bits in each packed 32-bit integer in a to the le... | vprold | — |
| `_mm256_mask_rol_epi64` | Rotate the bits in each packed 64-bit integer in a to the le... | vprolq | — |
| `_mm256_mask_rolv_epi32` | Rotate the bits in each packed 32-bit integer in a to the le... | vprolvd | — |
| `_mm256_mask_rolv_epi64` | Rotate the bits in each packed 64-bit integer in a to the le... | vprolvq | — |
| `_mm256_mask_ror_epi32` | Rotate the bits in each packed 32-bit integer in a to the ri... | vprold | — |
| `_mm256_mask_ror_epi64` | Rotate the bits in each packed 64-bit integer in a to the ri... | vprolq | — |
| `_mm256_mask_rorv_epi32` | Rotate the bits in each packed 32-bit integer in a to the ri... | vprorvd | — |
| `_mm256_mask_rorv_epi64` | Rotate the bits in each packed 64-bit integer in a to the ri... | vprorvq | — |
| `_mm256_mask_roundscale_pd` | Round packed double-precision (64-bit) floating-point elemen... | vrndscalepd | — |
| `_mm256_mask_roundscale_ps` | Round packed single-precision (32-bit) floating-point elemen... | vrndscaleps | — |
| `_mm256_mask_rsqrt14_pd` | Compute the approximate reciprocal square root of packed dou... | vrsqrt14pd | — |
| `_mm256_mask_rsqrt14_ps` | Compute the approximate reciprocal square root of packed sin... | vrsqrt14ps | — |
| `_mm256_mask_scalef_pd` | Scale the packed double-precision (64-bit) floating-point el... | vscalefpd | — |
| `_mm256_mask_scalef_ps` | Scale the packed single-precision (32-bit) floating-point el... | vscalefps | — |
| `_mm256_mask_set1_epi16` | Broadcast 16-bit integer a to all elements of dst using writ... | vpbroadcastw | — |
| `_mm256_mask_set1_epi32` | Broadcast 32-bit integer a to all elements of dst using writ... | vpbroadcastd | — |
| `_mm256_mask_set1_epi64` | Broadcast 64-bit integer a to all elements of dst using writ... | vpbroadcastq | — |
| `_mm256_mask_set1_epi8` | Broadcast 8-bit integer a to all elements of dst using write... | vpbroadcast | — |
| `_mm256_mask_shuffle_epi32` | Shuffle 32-bit integers in a within 128-bit lanes using the ... | vpshufd | 1/1, 1/1 |
| `_mm256_mask_shuffle_epi8` | Shuffle 8-bit integers in a within 128-bit lanes using the c... | vpshufb | 1/1, 1/1 |
| `_mm256_mask_shuffle_f32x4` | Shuffle 128-bits (composed of 4 single-precision (32-bit) fl... | vshuff32x4 | 1/1, 1/1 |
| `_mm256_mask_shuffle_f64x2` | Shuffle 128-bits (composed of 2 double-precision (64-bit) fl... | vshuff64x2 | 1/1, 1/1 |
| `_mm256_mask_shuffle_i32x4` | Shuffle 128-bits (composed of 4 32-bit integers) selected by... | vshufi32x4 | 1/1, 1/1 |
| `_mm256_mask_shuffle_i64x2` | Shuffle 128-bits (composed of 2 64-bit integers) selected by... | vshufi64x2 | 1/1, 1/1 |
| `_mm256_mask_shuffle_pd` | Shuffle double-precision (64-bit) floating-point elements wi... | vshufpd | 1/1, 1/1 |
| `_mm256_mask_shuffle_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vshufps | 1/1, 1/1 |
| `_mm256_mask_shufflehi_epi16` | Shuffle 16-bit integers in the high 64 bits of 128-bit lanes... | vpshufhw | 1/1, 1/1 |
| `_mm256_mask_shufflelo_epi16` | Shuffle 16-bit integers in the low 64 bits of 128-bit lanes ... | vpshuflw | 1/1, 1/1 |
| `_mm256_mask_sll_epi16` | Shift packed 16-bit integers in a left by count while shifti... | vpsllw | 1/1, 1/1 |
| `_mm256_mask_sll_epi32` | Shift packed 32-bit integers in a left by count while shifti... | vpslld | 1/1, 1/1 |
| `_mm256_mask_sll_epi64` | Shift packed 64-bit integers in a left by count while shifti... | vpsllq | 1/1, 1/1 |
| `_mm256_mask_slli_epi16` | Shift packed 16-bit integers in a left by imm8 while shiftin... | vpsllw | 1/1, 1/1 |
| `_mm256_mask_slli_epi32` | Shift packed 32-bit integers in a left by imm8 while shiftin... | vpslld | 1/1, 1/1 |
| `_mm256_mask_slli_epi64` | Shift packed 64-bit integers in a left by imm8 while shiftin... | vpsllq | 1/1, 1/1 |
| `_mm256_mask_sllv_epi16` | Shift packed 16-bit integers in a left by the amount specifi... | vpsllvw | 1/1, 1/1 |
| `_mm256_mask_sllv_epi32` | Shift packed 32-bit integers in a left by the amount specifi... | vpsllvd | 1/1, 1/1 |
| `_mm256_mask_sllv_epi64` | Shift packed 64-bit integers in a left by the amount specifi... | vpsllvq | 1/1, 1/1 |
| `_mm256_mask_sqrt_pd` | Compute the square root of packed double-precision (64-bit) ... | vsqrtpd | 28/28, 20/13 |
| `_mm256_mask_sqrt_ps` | Compute the square root of packed single-precision (32-bit) ... | vsqrtps | 19/14, 14/7 |
| `_mm256_mask_sra_epi16` | Shift packed 16-bit integers in a right by count while shift... | vpsraw | 1/1, 1/1 |
| `_mm256_mask_sra_epi32` | Shift packed 32-bit integers in a right by count while shift... | vpsrad | 1/1, 1/1 |
| `_mm256_mask_sra_epi64` | Shift packed 64-bit integers in a right by count while shift... | vpsraq | 1/1, 1/1 |
| `_mm256_mask_srai_epi16` | Shift packed 16-bit integers in a right by imm8 while shifti... | vpsraw | 1/1, 1/1 |
| `_mm256_mask_srai_epi32` | Shift packed 32-bit integers in a right by imm8 while shifti... | vpsrad | 1/1, 1/1 |
| `_mm256_mask_srai_epi64` | Shift packed 64-bit integers in a right by imm8 while shifti... | vpsraq | 1/1, 1/1 |
| `_mm256_mask_srav_epi16` | Shift packed 16-bit integers in a right by the amount specif... | vpsravw | 1/1, 1/1 |
| `_mm256_mask_srav_epi32` | Shift packed 32-bit integers in a right by the amount specif... | vpsravd | 1/1, 1/1 |
| `_mm256_mask_srav_epi64` | Shift packed 64-bit integers in a right by the amount specif... | vpsravq | 1/1, 1/1 |
| `_mm256_mask_srl_epi16` | Shift packed 16-bit integers in a right by count while shift... | vpsrlw | 1/1, 1/1 |
| `_mm256_mask_srl_epi32` | Shift packed 32-bit integers in a right by count while shift... | vpsrld | 1/1, 1/1 |
| `_mm256_mask_srl_epi64` | Shift packed 64-bit integers in a right by count while shift... | vpsrlq | 1/1, 1/1 |
| `_mm256_mask_srli_epi16` | Shift packed 16-bit integers in a right by imm8 while shifti... | vpsrlw | 1/1, 1/1 |
| `_mm256_mask_srli_epi32` | Shift packed 32-bit integers in a right by imm8 while shifti... | vpsrld | 1/1, 1/1 |
| `_mm256_mask_srli_epi64` | Shift packed 64-bit integers in a right by imm8 while shifti... | vpsrlq | 1/1, 1/1 |
| `_mm256_mask_srlv_epi16` | Shift packed 16-bit integers in a right by the amount specif... | vpsrlvw | 1/1, 1/1 |
| `_mm256_mask_srlv_epi32` | Shift packed 32-bit integers in a right by the amount specif... | vpsrlvd | 1/1, 1/1 |
| `_mm256_mask_srlv_epi64` | Shift packed 64-bit integers in a right by the amount specif... | vpsrlvq | 1/1, 1/1 |
| `_mm256_mask_sub_epi16` | Subtract packed 16-bit integers in b from packed 16-bit inte... | vpsubw | 1/1, 1/1 |
| `_mm256_mask_sub_epi32` | Subtract packed 32-bit integers in b from packed 32-bit inte... | vpsubd | 1/1, 1/1 |
| `_mm256_mask_sub_epi64` | Subtract packed 64-bit integers in b from packed 64-bit inte... | vpsubq | 1/1, 1/1 |
| `_mm256_mask_sub_epi8` | Subtract packed 8-bit integers in b from packed 8-bit intege... | vpsubb | 1/1, 1/1 |
| `_mm256_mask_sub_pd` | Subtract packed double-precision (64-bit) floating-point ele... | vsubpd | — |
| `_mm256_mask_sub_ps` | Subtract packed single-precision (32-bit) floating-point ele... | vsubps | — |
| `_mm256_mask_subs_epi16` | Subtract packed signed 16-bit integers in b from packed 16-b... | vpsubsw | 1/1, 1/1 |
| `_mm256_mask_subs_epi8` | Subtract packed signed 8-bit integers in b from packed 8-bit... | vpsubsb | 1/1, 1/1 |
| `_mm256_mask_subs_epu16` | Subtract packed unsigned 16-bit integers in b from packed un... | vpsubusw | — |
| `_mm256_mask_subs_epu8` | Subtract packed unsigned 8-bit integers in b from packed uns... | vpsubusb | — |
| `_mm256_mask_ternarylogic_epi32` | Bitwise ternary logic that provides the capability to implem... | vpternlogd | — |
| `_mm256_mask_ternarylogic_epi64` | Bitwise ternary logic that provides the capability to implem... | vpternlogq | — |
| `_mm256_mask_test_epi16_mask` | Compute the bitwise AND of packed 16-bit integers in a and b... | vptestmw | — |
| `_mm256_mask_test_epi32_mask` | Compute the bitwise AND of packed 32-bit integers in a and b... | vptestmd | — |
| `_mm256_mask_test_epi64_mask` | Compute the bitwise AND of packed 64-bit integers in a and b... | vptestmq | — |
| `_mm256_mask_test_epi8_mask` | Compute the bitwise AND of packed 8-bit integers in a and b,... | vptestmb | — |
| `_mm256_mask_testn_epi16_mask` | Compute the bitwise NAND of packed 16-bit integers in a and ... | vptestnmw | — |
| `_mm256_mask_testn_epi32_mask` | Compute the bitwise NAND of packed 32-bit integers in a and ... | vptestnmd | — |
| `_mm256_mask_testn_epi64_mask` | Compute the bitwise NAND of packed 64-bit integers in a and ... | vptestnmq | — |
| `_mm256_mask_testn_epi8_mask` | Compute the bitwise NAND of packed 8-bit integers in a and b... | vptestnmb | — |
| `_mm256_mask_unpackhi_epi16` | Unpack and interleave 16-bit integers from the high half of ... | vpunpckhwd | 1/1, 1/1 |
| `_mm256_mask_unpackhi_epi32` | Unpack and interleave 32-bit integers from the high half of ... | vpunpckhdq | 1/1, 1/1 |
| `_mm256_mask_unpackhi_epi64` | Unpack and interleave 64-bit integers from the high half of ... | vpunpckhqdq | 1/1, 1/1 |
| `_mm256_mask_unpackhi_epi8` | Unpack and interleave 8-bit integers from the high half of e... | vpunpckhbw | 1/1, 1/1 |
| `_mm256_mask_unpackhi_pd` | Unpack and interleave double-precision (64-bit) floating-poi... | vunpckhpd | 1/1, 1/1 |
| `_mm256_mask_unpackhi_ps` | Unpack and interleave single-precision (32-bit) floating-poi... | vunpckhps | 1/1, 1/1 |
| `_mm256_mask_unpacklo_epi16` | Unpack and interleave 16-bit integers from the low half of e... | vpunpcklwd | 1/1, 1/1 |
| `_mm256_mask_unpacklo_epi32` | Unpack and interleave 32-bit integers from the low half of e... | vpunpckldq | 1/1, 1/1 |
| `_mm256_mask_unpacklo_epi64` | Unpack and interleave 64-bit integers from the low half of e... | vpunpcklqdq | 1/1, 1/1 |
| `_mm256_mask_unpacklo_epi8` | Unpack and interleave 8-bit integers from the low half of ea... | vpunpcklbw | 1/1, 1/1 |
| `_mm256_mask_unpacklo_pd` | Unpack and interleave double-precision (64-bit) floating-poi... | vunpcklpd | 1/1, 1/1 |
| `_mm256_mask_unpacklo_ps` | Unpack and interleave single-precision (32-bit) floating-poi... | vunpcklps | 1/1, 1/1 |
| `_mm256_mask_xor_epi32` | Compute the bitwise XOR of packed 32-bit integers in a and b... | vpxord | 1/1, 1/1 |
| `_mm256_mask_xor_epi64` | Compute the bitwise XOR of packed 64-bit integers in a and b... | vpxorq | 1/1, 1/1 |
| `_mm256_mask_xor_pd` | Compute the bitwise XOR of packed double-precision (64-bit) ... | vxorpd | 1/1, 1/1 |
| `_mm256_mask_xor_ps` | Compute the bitwise XOR of packed single-precision (32-bit) ... | vxorps | 1/1, 1/1 |
| `_mm256_maskz_abs_epi16` | Compute the absolute value of packed signed 16-bit integers ... | vpabsw | — |
| `_mm256_maskz_abs_epi32` | Compute the absolute value of packed signed 32-bit integers ... | vpabsd | — |
| `_mm256_maskz_abs_epi64` | Compute the absolute value of packed signed 64-bit integers ... | vpabsq | — |
| `_mm256_maskz_abs_epi8` | Compute the absolute value of packed signed 8-bit integers i... | vpabsb | — |
| `_mm256_maskz_add_epi16` | Add packed 16-bit integers in a and b, and store the results... | vpaddw | 1/1, 1/1 |
| `_mm256_maskz_add_epi32` | Add packed 32-bit integers in a and b, and store the results... | vpaddd | 1/1, 1/1 |
| `_mm256_maskz_add_epi64` | Add packed 64-bit integers in a and b, and store the results... | vpaddq | 1/1, 1/1 |
| `_mm256_maskz_add_epi8` | Add packed 8-bit integers in a and b, and store the results ... | vpaddb | 1/1, 1/1 |
| `_mm256_maskz_add_pd` | Add packed double-precision (64-bit) floating-point elements... | vaddpd | 3/1, 3/1 |
| `_mm256_maskz_add_ps` | Add packed single-precision (32-bit) floating-point elements... | vaddps | 3/1, 3/1 |
| `_mm256_maskz_adds_epi16` | Add packed signed 16-bit integers in a and b using saturatio... | vpaddsw | 1/1, 1/1 |
| `_mm256_maskz_adds_epi8` | Add packed signed 8-bit integers in a and b using saturation... | vpaddsb | 1/1, 1/1 |
| `_mm256_maskz_adds_epu16` | Add packed unsigned 16-bit integers in a and b using saturat... | vpaddusw | — |
| `_mm256_maskz_adds_epu8` | Add packed unsigned 8-bit integers in a and b using saturati... | vpaddusb | — |
| `_mm256_maskz_alignr_epi32` | Concatenate a and b into a 64-byte immediate result, shift t... | valignd | — |
| `_mm256_maskz_alignr_epi64` | Concatenate a and b into a 64-byte immediate result, shift t... | valignq | — |
| `_mm256_maskz_alignr_epi8` | Concatenate pairs of 16-byte blocks in a and b into a 32-byt... | vpalignr | — |
| `_mm256_maskz_and_epi32` | Compute the bitwise AND of packed 32-bit integers in a and b... | vpandd | 1/1, 1/1 |
| `_mm256_maskz_and_epi64` | Compute the bitwise AND of packed 64-bit integers in a and b... | vpandq | 1/1, 1/1 |
| `_mm256_maskz_and_pd` | Compute the bitwise AND of packed double-precision (64-bit) ... | vandpd | 1/1, 1/1 |
| `_mm256_maskz_and_ps` | Compute the bitwise AND of packed single-precision (32-bit) ... | vandps | 1/1, 1/1 |
| `_mm256_maskz_andnot_epi32` | Compute the bitwise NOT of packed 32-bit integers in a and t... | vpandnd | 1/1, 1/1 |
| `_mm256_maskz_andnot_epi64` | Compute the bitwise NOT of packed 64-bit integers in a and t... | vpandnq | 1/1, 1/1 |
| `_mm256_maskz_andnot_pd` | Compute the bitwise NOT of packed double-precision (64-bit) ... | vandnpd | 1/1, 1/1 |
| `_mm256_maskz_andnot_ps` | Compute the bitwise NOT of packed single-precision (32-bit) ... | vandnps | 1/1, 1/1 |
| `_mm256_maskz_avg_epu16` | Average packed unsigned 16-bit integers in a and b, and stor... | vpavgw | — |
| `_mm256_maskz_avg_epu8` | Average packed unsigned 8-bit integers in a and b, and store... | vpavgb | — |
| `_mm256_maskz_broadcast_f32x2` | Broadcasts the lower 2 packed single-precision (32-bit) floa... | vbroadcastf32x2 | — |
| `_mm256_maskz_broadcast_f32x4` | Broadcast the 4 packed single-precision (32-bit) floating-po... |  | — |
| `_mm256_maskz_broadcast_f64x2` | Broadcasts the 2 packed double-precision (64-bit) floating-p... |  | — |
| `_mm256_maskz_broadcast_i32x2` | Broadcasts the lower 2 packed 32-bit integers from a to all ... | vbroadcasti32x2 | — |
| `_mm256_maskz_broadcast_i32x4` | Broadcast the 4 packed 32-bit integers from a to all element... |  | — |
| `_mm256_maskz_broadcast_i64x2` | Broadcasts the 2 packed 64-bit integers from a to all elemen... |  | — |
| `_mm256_maskz_broadcastb_epi8` | Broadcast the low packed 8-bit integer from a to all element... | vpbroadcastb | — |
| `_mm256_maskz_broadcastd_epi32` | Broadcast the low packed 32-bit integer from a to all elemen... | vpbroadcast | — |
| `_mm256_maskz_broadcastq_epi64` | Broadcast the low packed 64-bit integer from a to all elemen... | vpbroadcast | — |
| `_mm256_maskz_broadcastsd_pd` | Broadcast the low double-precision (64-bit) floating-point e... | vbroadcastsd | — |
| `_mm256_maskz_broadcastss_ps` | Broadcast the low single-precision (32-bit) floating-point e... | vbroadcastss | — |
| `_mm256_maskz_broadcastw_epi16` | Broadcast the low packed 16-bit integer from a to all elemen... | vpbroadcastw | — |
| `_mm256_maskz_compress_epi32` | Contiguously store the active 32-bit integers in a (those wi... | vpcompressd | — |
| `_mm256_maskz_compress_epi64` | Contiguously store the active 64-bit integers in a (those wi... | vpcompressq | — |
| `_mm256_maskz_compress_pd` | Contiguously store the active double-precision (64-bit) floa... | vcompresspd | — |
| `_mm256_maskz_compress_ps` | Contiguously store the active single-precision (32-bit) floa... | vcompressps | — |
| `_mm256_maskz_conflict_epi32` | Test each 32-bit element of a for equality with all other el... | vpconflictd | — |
| `_mm256_maskz_conflict_epi64` | Test each 64-bit element of a for equality with all other el... | vpconflictq | — |
| `_mm256_maskz_cvt_roundps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2ph | 4/1, 3/1 |
| `_mm256_maskz_cvtepi16_epi32` | Sign extend packed 16-bit integers in a to packed 32-bit int... | vpmovsxwd | 4/1, 3/1 |
| `_mm256_maskz_cvtepi16_epi64` | Sign extend packed 16-bit integers in a to packed 64-bit int... | vpmovsxwq | 4/1, 3/1 |
| `_mm256_maskz_cvtepi16_epi8` | Convert packed 16-bit integers in a to packed 8-bit integers... | vpmovwb | 4/1, 3/1 |
| `_mm256_maskz_cvtepi32_epi16` | Convert packed 32-bit integers in a to packed 16-bit integer... | vpmovdw | 4/1, 3/1 |
| `_mm256_maskz_cvtepi32_epi64` | Sign extend packed 32-bit integers in a to packed 64-bit int... | vpmovsxdq | 4/1, 3/1 |
| `_mm256_maskz_cvtepi32_epi8` | Convert packed 32-bit integers in a to packed 8-bit integers... | vpmovdb | 4/1, 3/1 |
| `_mm256_maskz_cvtepi32_pd` | Convert packed signed 32-bit integers in a to packed double-... | vcvtdq2pd | 4/1, 3/1 |
| `_mm256_maskz_cvtepi32_ps` | Convert packed signed 32-bit integers in a to packed single-... | vcvtdq2ps | 4/1, 3/1 |
| `_mm256_maskz_cvtepi64_epi16` | Convert packed 64-bit integers in a to packed 16-bit integer... | vpmovqw | 4/1, 3/1 |
| `_mm256_maskz_cvtepi64_epi32` | Convert packed 64-bit integers in a to packed 32-bit integer... | vpmovqd | 4/1, 3/1 |
| `_mm256_maskz_cvtepi64_epi8` | Convert packed 64-bit integers in a to packed 8-bit integers... | vpmovqb | 4/1, 3/1 |
| `_mm256_maskz_cvtepi64_pd` | Convert packed signed 64-bit integers in a to packed double-... | vcvtqq2pd | 4/1, 3/1 |
| `_mm256_maskz_cvtepi64_ps` | Convert packed signed 64-bit integers in a to packed single-... | vcvtqq2ps | 4/1, 3/1 |
| `_mm256_maskz_cvtepi8_epi16` | Sign extend packed 8-bit integers in a to packed 16-bit inte... | vpmovsxbw | 4/1, 3/1 |
| `_mm256_maskz_cvtepi8_epi32` | Sign extend packed 8-bit integers in a to packed 32-bit inte... | vpmovsxbd | 4/1, 3/1 |
| `_mm256_maskz_cvtepi8_epi64` | Sign extend packed 8-bit integers in the low 4 bytes of a to... | vpmovsxbq | 4/1, 3/1 |
| `_mm256_maskz_cvtepu16_epi32` | Zero extend packed unsigned 16-bit integers in a to packed 3... | vpmovzxwd | 4/1, 3/1 |
| `_mm256_maskz_cvtepu16_epi64` | Zero extend packed unsigned 16-bit integers in the low 8 byt... | vpmovzxwq | 4/1, 3/1 |
| `_mm256_maskz_cvtepu32_epi64` | Zero extend packed unsigned 32-bit integers in a to packed 6... | vpmovzxdq | 4/1, 3/1 |
| `_mm256_maskz_cvtepu32_pd` | Convert packed unsigned 32-bit integers in a to packed doubl... | vcvtudq2pd | 4/1, 3/1 |
| `_mm256_maskz_cvtepu64_pd` | Convert packed unsigned 64-bit integers in a to packed doubl... | vcvtuqq2pd | 4/1, 3/1 |
| `_mm256_maskz_cvtepu64_ps` | Convert packed unsigned 64-bit integers in a to packed singl... | vcvtuqq2ps | 4/1, 3/1 |
| `_mm256_maskz_cvtepu8_epi16` | Zero extend packed unsigned 8-bit integers in a to packed 16... | vpmovzxbw | 4/1, 3/1 |
| `_mm256_maskz_cvtepu8_epi32` | Zero extend packed unsigned 8-bit integers in the low 8 byte... | vpmovzxbd | 4/1, 3/1 |
| `_mm256_maskz_cvtepu8_epi64` | Zero extend packed unsigned 8-bit integers in the low 4 byte... | vpmovzxbq | 4/1, 3/1 |
| `_mm256_maskz_cvtpd_epi32` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2dq | 4/1, 3/1 |
| `_mm256_maskz_cvtpd_epi64` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2qq | 4/1, 3/1 |
| `_mm256_maskz_cvtpd_epu32` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2udq | 4/1, 3/1 |
| `_mm256_maskz_cvtpd_epu64` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2uqq | 4/1, 3/1 |
| `_mm256_maskz_cvtpd_ps` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2ps | 4/1, 3/1 |
| `_mm256_maskz_cvtph_ps` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2ps | 4/1, 3/1 |
| `_mm256_maskz_cvtps_epi32` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2dq | 4/1, 3/1 |
| `_mm256_maskz_cvtps_epi64` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2qq | 4/1, 3/1 |
| `_mm256_maskz_cvtps_epu32` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2udq | 4/1, 3/1 |
| `_mm256_maskz_cvtps_epu64` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2uqq | 4/1, 3/1 |
| `_mm256_maskz_cvtps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2ph | 4/1, 3/1 |
| `_mm256_maskz_cvtsepi16_epi8` | Convert packed signed 16-bit integers in a to packed 8-bit i... | vpmovswb | 4/1, 3/1 |
| `_mm256_maskz_cvtsepi32_epi16` | Convert packed signed 32-bit integers in a to packed 16-bit ... | vpmovsdw | 4/1, 3/1 |
| `_mm256_maskz_cvtsepi32_epi8` | Convert packed signed 32-bit integers in a to packed 8-bit i... | vpmovsdb | 4/1, 3/1 |
| `_mm256_maskz_cvtsepi64_epi16` | Convert packed signed 64-bit integers in a to packed 16-bit ... | vpmovsqw | 4/1, 3/1 |
| `_mm256_maskz_cvtsepi64_epi32` | Convert packed signed 64-bit integers in a to packed 32-bit ... | vpmovsqd | 4/1, 3/1 |
| `_mm256_maskz_cvtsepi64_epi8` | Convert packed signed 64-bit integers in a to packed 8-bit i... | vpmovsqb | 4/1, 3/1 |
| `_mm256_maskz_cvttpd_epi32` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2dq | 4/1, 3/1 |
| `_mm256_maskz_cvttpd_epi64` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2qq | 4/1, 3/1 |
| `_mm256_maskz_cvttpd_epu32` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2udq | 4/1, 3/1 |
| `_mm256_maskz_cvttpd_epu64` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2uqq | 4/1, 3/1 |
| `_mm256_maskz_cvttps_epi32` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2dq | 4/1, 3/1 |
| `_mm256_maskz_cvttps_epi64` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2qq | 4/1, 3/1 |
| `_mm256_maskz_cvttps_epu32` | Convert packed double-precision (32-bit) floating-point elem... | vcvttps2udq | 4/1, 3/1 |
| `_mm256_maskz_cvttps_epu64` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2uqq | 4/1, 3/1 |
| `_mm256_maskz_cvtusepi16_epi8` | Convert packed unsigned 16-bit integers in a to packed unsig... | vpmovuswb | 4/1, 3/1 |
| `_mm256_maskz_cvtusepi32_epi16` | Convert packed unsigned 32-bit integers in a to packed unsig... | vpmovusdw | 4/1, 3/1 |
| `_mm256_maskz_cvtusepi32_epi8` | Convert packed unsigned 32-bit integers in a to packed unsig... | vpmovusdb | 4/1, 3/1 |
| `_mm256_maskz_cvtusepi64_epi16` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqw | 4/1, 3/1 |
| `_mm256_maskz_cvtusepi64_epi32` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqd | 4/1, 3/1 |
| `_mm256_maskz_cvtusepi64_epi8` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqb | 4/1, 3/1 |
| `_mm256_maskz_dbsad_epu8` | Compute the sum of absolute differences (SADs) of quadruplet... | vdbpsadbw | — |
| `_mm256_maskz_div_pd` | Divide packed double-precision (64-bit) floating-point eleme... | vdivpd | 35/28, 13/10 |
| `_mm256_maskz_div_ps` | Divide packed single-precision (32-bit) floating-point eleme... | vdivps | 21/14, 10/7 |
| `_mm256_maskz_expand_epi32` | Load contiguous active 32-bit integers from a (those with th... | vpexpandd | — |
| `_mm256_maskz_expand_epi64` | Load contiguous active 64-bit integers from a (those with th... | vpexpandq | — |
| `_mm256_maskz_expand_pd` | Load contiguous active double-precision (64-bit) floating-po... | vexpandpd | — |
| `_mm256_maskz_expand_ps` | Load contiguous active single-precision (32-bit) floating-po... | vexpandps | — |
| `_mm256_maskz_extractf32x4_ps` | Extract 128 bits (composed of 4 packed single-precision (32-... | vextractf32x4 | — |
| `_mm256_maskz_extractf64x2_pd` | Extracts 128 bits (composed of 2 packed double-precision (64... | vextractf64x2 | — |
| `_mm256_maskz_extracti32x4_epi32` | Extract 128 bits (composed of 4 packed 32-bit integers) from... | vextracti32x4 | — |
| `_mm256_maskz_extracti64x2_epi64` | Extracts 128 bits (composed of 2 packed 64-bit integers) fro... | vextracti64x2 | — |
| `_mm256_maskz_fixupimm_pd` | Fix up packed double-precision (64-bit) floating-point eleme... | vfixupimmpd | — |
| `_mm256_maskz_fixupimm_ps` | Fix up packed single-precision (32-bit) floating-point eleme... | vfixupimmps | — |
| `_mm256_maskz_fmadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmadd | 5/1, 4/1 |
| `_mm256_maskz_fmadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmadd | 5/1, 4/1 |
| `_mm256_maskz_fmaddsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmaddsub | 5/1, 4/1 |
| `_mm256_maskz_fmaddsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmaddsub | 5/1, 4/1 |
| `_mm256_maskz_fmsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsub | 5/1, 4/1 |
| `_mm256_maskz_fmsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsub | 5/1, 4/1 |
| `_mm256_maskz_fmsubadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsubadd | 5/1, 4/1 |
| `_mm256_maskz_fmsubadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsubadd | 5/1, 4/1 |
| `_mm256_maskz_fnmadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmadd | 5/1, 4/1 |
| `_mm256_maskz_fnmadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmadd | 5/1, 4/1 |
| `_mm256_maskz_fnmsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmsub | 5/1, 4/1 |
| `_mm256_maskz_fnmsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmsub | 5/1, 4/1 |
| `_mm256_maskz_getexp_pd` | Convert the exponent of each packed double-precision (64-bit... | vgetexppd | — |
| `_mm256_maskz_getexp_ps` | Convert the exponent of each packed single-precision (32-bit... | vgetexpps | — |
| `_mm256_maskz_getmant_pd` | Normalize the mantissas of packed double-precision (64-bit) ... | vgetmantpd | — |
| `_mm256_maskz_getmant_ps` | Normalize the mantissas of packed single-precision (32-bit) ... | vgetmantps | — |
| `_mm256_maskz_insertf32x4` | Copy a to tmp, then insert 128 bits (composed of 4 packed si... | vinsertf32x4 | — |
| `_mm256_maskz_insertf64x2` | Copy a to tmp, then insert 128 bits (composed of 2 packed do... | vinsertf64x2 | — |
| `_mm256_maskz_inserti32x4` | Copy a to tmp, then insert 128 bits (composed of 4 packed 32... | vinserti32x4 | — |
| `_mm256_maskz_inserti64x2` | Copy a to tmp, then insert 128 bits (composed of 2 packed 64... | vinserti64x2 | — |
| `_mm256_maskz_lzcnt_epi32` | Counts the number of leading zero bits in each packed 32-bit... | vplzcntd | — |
| `_mm256_maskz_lzcnt_epi64` | Counts the number of leading zero bits in each packed 64-bit... | vplzcntq | — |
| `_mm256_maskz_madd_epi16` | Multiply packed signed 16-bit integers in a and b, producing... | vpmaddwd | — |
| `_mm256_maskz_maddubs_epi16` | Multiply packed unsigned 8-bit integers in a by packed signe... | vpmaddubsw | — |
| `_mm256_maskz_max_epi16` | Compare packed signed 16-bit integers in a and b, and store ... | vpmaxsw | — |
| `_mm256_maskz_max_epi32` | Compare packed signed 32-bit integers in a and b, and store ... | vpmaxsd | — |
| `_mm256_maskz_max_epi64` | Compare packed signed 64-bit integers in a and b, and store ... | vpmaxsq | — |
| `_mm256_maskz_max_epi8` | Compare packed signed 8-bit integers in a and b, and store p... | vpmaxsb | — |
| `_mm256_maskz_max_epu16` | Compare packed unsigned 16-bit integers in a and b, and stor... | vpmaxuw | — |
| `_mm256_maskz_max_epu32` | Compare packed unsigned 32-bit integers in a and b, and stor... | vpmaxud | — |
| `_mm256_maskz_max_epu64` | Compare packed unsigned 64-bit integers in a and b, and stor... | vpmaxuq | — |
| `_mm256_maskz_max_epu8` | Compare packed unsigned 8-bit integers in a and b, and store... | vpmaxub | — |
| `_mm256_maskz_max_pd` | Compare packed double-precision (64-bit) floating-point elem... | vmaxpd | — |
| `_mm256_maskz_max_ps` | Compare packed single-precision (32-bit) floating-point elem... | vmaxps | — |
| `_mm256_maskz_min_epi16` | Compare packed signed 16-bit integers in a and b, and store ... | vpminsw | — |
| `_mm256_maskz_min_epi32` | Compare packed signed 32-bit integers in a and b, and store ... | vpminsd | — |
| `_mm256_maskz_min_epi64` | Compare packed signed 64-bit integers in a and b, and store ... | vpminsq | — |
| `_mm256_maskz_min_epi8` | Compare packed signed 8-bit integers in a and b, and store p... | vpminsb | — |
| `_mm256_maskz_min_epu16` | Compare packed unsigned 16-bit integers in a and b, and stor... | vpminuw | — |
| `_mm256_maskz_min_epu32` | Compare packed unsigned 32-bit integers in a and b, and stor... | vpminud | — |
| `_mm256_maskz_min_epu64` | Compare packed unsigned 64-bit integers in a and b, and stor... | vpminuq | — |
| `_mm256_maskz_min_epu8` | Compare packed unsigned 8-bit integers in a and b, and store... | vpminub | — |
| `_mm256_maskz_min_pd` | Compare packed double-precision (64-bit) floating-point elem... | vminpd | — |
| `_mm256_maskz_min_ps` | Compare packed single-precision (32-bit) floating-point elem... | vminps | — |
| `_mm256_maskz_mov_epi16` | Move packed 16-bit integers from a into dst using zeromask k... | vmovdqu16 | — |
| `_mm256_maskz_mov_epi32` | Move packed 32-bit integers from a into dst using zeromask k... | vmovdqa32 | — |
| `_mm256_maskz_mov_epi64` | Move packed 64-bit integers from a into dst using zeromask k... | vmovdqa64 | — |
| `_mm256_maskz_mov_epi8` | Move packed 8-bit integers from a into dst using zeromask k ... | vmovdqu8 | — |
| `_mm256_maskz_mov_pd` | Move packed double-precision (64-bit) floating-point element... | vmovapd | — |
| `_mm256_maskz_mov_ps` | Move packed single-precision (32-bit) floating-point element... | vmovaps | — |
| `_mm256_maskz_movedup_pd` | Duplicate even-indexed double-precision (64-bit) floating-po... | vmovddup | — |
| `_mm256_maskz_movehdup_ps` | Duplicate odd-indexed single-precision (32-bit) floating-poi... | vmovshdup | — |
| `_mm256_maskz_moveldup_ps` | Duplicate even-indexed single-precision (32-bit) floating-po... | vmovsldup | — |
| `_mm256_maskz_mul_epi32` | Multiply the low signed 32-bit integers from each packed 64-... | vpmuldq | 10/2, 4/1 |
| `_mm256_maskz_mul_epu32` | Multiply the low unsigned 32-bit integers from each packed 6... | vpmuludq | 10/2, 4/1 |
| `_mm256_maskz_mul_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vmulpd | 5/1, 3/1 |
| `_mm256_maskz_mul_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vmulps | 5/1, 3/1 |
| `_mm256_maskz_mulhi_epi16` | Multiply the packed signed 16-bit integers in a and b, produ... | vpmulhw | 5/1, 3/1 |
| `_mm256_maskz_mulhi_epu16` | Multiply the packed unsigned 16-bit integers in a and b, pro... | vpmulhuw | 5/1, 3/1 |
| `_mm256_maskz_mulhrs_epi16` | Multiply packed signed 16-bit integers in a and b, producing... | vpmulhrsw | — |
| `_mm256_maskz_mullo_epi16` | Multiply the packed 16-bit integers in a and b, producing in... | vpmullw | 5/1, 3/1 |
| `_mm256_maskz_mullo_epi32` | Multiply the packed 32-bit integers in a and b, producing in... | vpmulld | 10/2, 4/1 |
| `_mm256_maskz_mullo_epi64` | Multiply packed 64-bit integers in `a` and `b`, producing in... | vpmullq | — |
| `_mm256_maskz_or_epi32` | Compute the bitwise OR of packed 32-bit integers in a and b,... | vpord | 1/1, 1/1 |
| `_mm256_maskz_or_epi64` | Compute the bitwise OR of packed 64-bit integers in a and b,... | vporq | 1/1, 1/1 |
| `_mm256_maskz_or_pd` | Compute the bitwise OR of packed double-precision (64-bit) f... | vorpd | 1/1, 1/1 |
| `_mm256_maskz_or_ps` | Compute the bitwise OR of packed single-precision (32-bit) f... | vorps | 1/1, 1/1 |
| `_mm256_maskz_packs_epi16` | Convert packed signed 16-bit integers from a and b to packed... | vpacksswb | 1/1, 1/1 |
| `_mm256_maskz_packs_epi32` | Convert packed signed 32-bit integers from a and b to packed... | vpackssdw | 1/1, 1/1 |
| `_mm256_maskz_packus_epi16` | Convert packed signed 16-bit integers from a and b to packed... | vpackuswb | 1/1, 1/1 |
| `_mm256_maskz_packus_epi32` | Convert packed signed 32-bit integers from a and b to packed... | vpackusdw | 1/1, 1/1 |
| `_mm256_maskz_permute_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vshufpd | 3/1, 2/1 |
| `_mm256_maskz_permute_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vshufps | 3/1, 2/1 |
| `_mm256_maskz_permutevar_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vpermilpd | 3/1, 2/1 |
| `_mm256_maskz_permutevar_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vpermilps | 3/1, 2/1 |
| `_mm256_maskz_permutex2var_epi16` | Shuffle 16-bit integers in a and b across lanes using the co... | vperm | 3/1, 2/1 |
| `_mm256_maskz_permutex2var_epi32` | Shuffle 32-bit integers in a and b across lanes using the co... | vperm | 3/1, 2/1 |
| `_mm256_maskz_permutex2var_epi64` | Shuffle 64-bit integers in a and b across lanes using the co... | vperm | 3/1, 2/1 |
| `_mm256_maskz_permutex2var_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vperm | 3/1, 2/1 |
| `_mm256_maskz_permutex2var_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vperm | 3/1, 2/1 |
| `_mm256_maskz_permutex_epi64` | Shuffle 64-bit integers in a within 256-bit lanes using the ... | vperm | 3/1, 2/1 |
| `_mm256_maskz_permutex_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vperm | 3/1, 2/1 |
| `_mm256_maskz_permutexvar_epi16` | Shuffle 16-bit integers in a across lanes using the correspo... | vpermw | 3/1, 2/1 |
| `_mm256_maskz_permutexvar_epi32` | Shuffle 32-bit integers in a across lanes using the correspo... | vpermd | 3/1, 2/1 |
| `_mm256_maskz_permutexvar_epi64` | Shuffle 64-bit integers in a across lanes using the correspo... | vpermq | 3/1, 2/1 |
| `_mm256_maskz_permutexvar_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vpermpd | 3/1, 2/1 |
| `_mm256_maskz_permutexvar_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vpermps | 3/1, 2/1 |
| `_mm256_maskz_range_pd` | Calculate the max, min, absolute max, or absolute min (depen... | vrangepd | — |
| `_mm256_maskz_range_ps` | Calculate the max, min, absolute max, or absolute min (depen... | vrangeps | — |
| `_mm256_maskz_rcp14_pd` | Compute the approximate reciprocal of packed double-precisio... | vrcp14pd | — |
| `_mm256_maskz_rcp14_ps` | Compute the approximate reciprocal of packed single-precisio... | vrcp14ps | — |
| `_mm256_maskz_reduce_pd` | Extract the reduced argument of packed double-precision (64-... | vreducepd | — |
| `_mm256_maskz_reduce_ps` | Extract the reduced argument of packed single-precision (32-... | vreduceps | — |
| `_mm256_maskz_rol_epi32` | Rotate the bits in each packed 32-bit integer in a to the le... | vprold | — |
| `_mm256_maskz_rol_epi64` | Rotate the bits in each packed 64-bit integer in a to the le... | vprolq | — |
| `_mm256_maskz_rolv_epi32` | Rotate the bits in each packed 32-bit integer in a to the le... | vprolvd | — |
| `_mm256_maskz_rolv_epi64` | Rotate the bits in each packed 64-bit integer in a to the le... | vprolvq | — |
| `_mm256_maskz_ror_epi32` | Rotate the bits in each packed 32-bit integer in a to the ri... | vprold | — |
| `_mm256_maskz_ror_epi64` | Rotate the bits in each packed 64-bit integer in a to the ri... | vprolq | — |
| `_mm256_maskz_rorv_epi32` | Rotate the bits in each packed 32-bit integer in a to the ri... | vprorvd | — |
| `_mm256_maskz_rorv_epi64` | Rotate the bits in each packed 64-bit integer in a to the ri... | vprorvq | — |
| `_mm256_maskz_roundscale_pd` | Round packed double-precision (64-bit) floating-point elemen... | vrndscalepd | — |
| `_mm256_maskz_roundscale_ps` | Round packed single-precision (32-bit) floating-point elemen... | vrndscaleps | — |
| `_mm256_maskz_rsqrt14_pd` | Compute the approximate reciprocal square root of packed dou... | vrsqrt14pd | — |
| `_mm256_maskz_rsqrt14_ps` | Compute the approximate reciprocal square root of packed sin... | vrsqrt14ps | — |
| `_mm256_maskz_scalef_pd` | Scale the packed double-precision (64-bit) floating-point el... | vscalefpd | — |
| `_mm256_maskz_scalef_ps` | Scale the packed single-precision (32-bit) floating-point el... | vscalefps | — |
| `_mm256_maskz_set1_epi16` | Broadcast the low packed 16-bit integer from a to all elemen... | vpbroadcastw | — |
| `_mm256_maskz_set1_epi32` | Broadcast 32-bit integer a to all elements of dst using zero... | vpbroadcastd | — |
| `_mm256_maskz_set1_epi64` | Broadcast 64-bit integer a to all elements of dst using zero... | vpbroadcastq | — |
| `_mm256_maskz_set1_epi8` | Broadcast 8-bit integer a to all elements of dst using zerom... | vpbroadcast | — |
| `_mm256_maskz_shuffle_epi32` | Shuffle 32-bit integers in a within 128-bit lanes using the ... | vpshufd | 1/1, 1/1 |
| `_mm256_maskz_shuffle_epi8` | Shuffle packed 8-bit integers in a according to shuffle cont... | vpshufb | 1/1, 1/1 |
| `_mm256_maskz_shuffle_f32x4` | Shuffle 128-bits (composed of 4 single-precision (32-bit) fl... | vshuff32x4 | 1/1, 1/1 |
| `_mm256_maskz_shuffle_f64x2` | Shuffle 128-bits (composed of 2 double-precision (64-bit) fl... | vshuff64x2 | 1/1, 1/1 |
| `_mm256_maskz_shuffle_i32x4` | Shuffle 128-bits (composed of 4 32-bit integers) selected by... | vshufi32x4 | 1/1, 1/1 |
| `_mm256_maskz_shuffle_i64x2` | Shuffle 128-bits (composed of 2 64-bit integers) selected by... | vshufi64x2 | 1/1, 1/1 |
| `_mm256_maskz_shuffle_pd` | Shuffle double-precision (64-bit) floating-point elements wi... | vshufpd | 1/1, 1/1 |
| `_mm256_maskz_shuffle_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vshufps | 1/1, 1/1 |
| `_mm256_maskz_shufflehi_epi16` | Shuffle 16-bit integers in the high 64 bits of 128-bit lanes... | vpshufhw | 1/1, 1/1 |
| `_mm256_maskz_shufflelo_epi16` | Shuffle 16-bit integers in the low 64 bits of 128-bit lanes ... | vpshuflw | 1/1, 1/1 |
| `_mm256_maskz_sll_epi16` | Shift packed 16-bit integers in a left by count while shifti... | vpsllw | 1/1, 1/1 |
| `_mm256_maskz_sll_epi32` | Shift packed 32-bit integers in a left by count while shifti... | vpslld | 1/1, 1/1 |
| `_mm256_maskz_sll_epi64` | Shift packed 64-bit integers in a left by count while shifti... | vpsllq | 1/1, 1/1 |
| `_mm256_maskz_slli_epi16` | Shift packed 16-bit integers in a left by imm8 while shiftin... | vpsllw | 1/1, 1/1 |
| `_mm256_maskz_slli_epi32` | Shift packed 32-bit integers in a left by imm8 while shiftin... | vpslld | 1/1, 1/1 |
| `_mm256_maskz_slli_epi64` | Shift packed 64-bit integers in a left by imm8 while shiftin... | vpsllq | 1/1, 1/1 |
| `_mm256_maskz_sllv_epi16` | Shift packed 16-bit integers in a left by the amount specifi... | vpsllvw | 1/1, 1/1 |
| `_mm256_maskz_sllv_epi32` | Shift packed 32-bit integers in a left by the amount specifi... | vpsllvd | 1/1, 1/1 |
| `_mm256_maskz_sllv_epi64` | Shift packed 64-bit integers in a left by the amount specifi... | vpsllvq | 1/1, 1/1 |
| `_mm256_maskz_sqrt_pd` | Compute the square root of packed double-precision (64-bit) ... | vsqrtpd | 28/28, 20/13 |
| `_mm256_maskz_sqrt_ps` | Compute the square root of packed single-precision (32-bit) ... | vsqrtps | 19/14, 14/7 |
| `_mm256_maskz_sra_epi16` | Shift packed 16-bit integers in a right by count while shift... | vpsraw | 1/1, 1/1 |
| `_mm256_maskz_sra_epi32` | Shift packed 32-bit integers in a right by count while shift... | vpsrad | 1/1, 1/1 |
| `_mm256_maskz_sra_epi64` | Shift packed 64-bit integers in a right by count while shift... | vpsraq | 1/1, 1/1 |
| `_mm256_maskz_srai_epi16` | Shift packed 16-bit integers in a right by imm8 while shifti... | vpsraw | 1/1, 1/1 |
| `_mm256_maskz_srai_epi32` | Shift packed 32-bit integers in a right by imm8 while shifti... | vpsrad | 1/1, 1/1 |
| `_mm256_maskz_srai_epi64` | Shift packed 64-bit integers in a right by imm8 while shifti... | vpsraq | 1/1, 1/1 |
| `_mm256_maskz_srav_epi16` | Shift packed 16-bit integers in a right by the amount specif... | vpsravw | 1/1, 1/1 |
| `_mm256_maskz_srav_epi32` | Shift packed 32-bit integers in a right by the amount specif... | vpsravd | 1/1, 1/1 |
| `_mm256_maskz_srav_epi64` | Shift packed 64-bit integers in a right by the amount specif... | vpsravq | 1/1, 1/1 |
| `_mm256_maskz_srl_epi16` | Shift packed 16-bit integers in a right by count while shift... | vpsrlw | 1/1, 1/1 |
| `_mm256_maskz_srl_epi32` | Shift packed 32-bit integers in a right by count while shift... | vpsrld | 1/1, 1/1 |
| `_mm256_maskz_srl_epi64` | Shift packed 64-bit integers in a right by count while shift... | vpsrlq | 1/1, 1/1 |
| `_mm256_maskz_srli_epi16` | Shift packed 16-bit integers in a right by imm8 while shifti... | vpsrlw | 1/1, 1/1 |
| `_mm256_maskz_srli_epi32` | Shift packed 32-bit integers in a right by imm8 while shifti... | vpsrld | 1/1, 1/1 |
| `_mm256_maskz_srli_epi64` | Shift packed 64-bit integers in a right by imm8 while shifti... | vpsrlq | 1/1, 1/1 |
| `_mm256_maskz_srlv_epi16` | Shift packed 16-bit integers in a right by the amount specif... | vpsrlvw | 1/1, 1/1 |
| `_mm256_maskz_srlv_epi32` | Shift packed 32-bit integers in a right by the amount specif... | vpsrlvd | 1/1, 1/1 |
| `_mm256_maskz_srlv_epi64` | Shift packed 64-bit integers in a right by the amount specif... | vpsrlvq | 1/1, 1/1 |
| `_mm256_maskz_sub_epi16` | Subtract packed 16-bit integers in b from packed 16-bit inte... | vpsubw | 1/1, 1/1 |
| `_mm256_maskz_sub_epi32` | Subtract packed 32-bit integers in b from packed 32-bit inte... | vpsubd | 1/1, 1/1 |
| `_mm256_maskz_sub_epi64` | Subtract packed 64-bit integers in b from packed 64-bit inte... | vpsubq | 1/1, 1/1 |
| `_mm256_maskz_sub_epi8` | Subtract packed 8-bit integers in b from packed 8-bit intege... | vpsubb | 1/1, 1/1 |
| `_mm256_maskz_sub_pd` | Subtract packed double-precision (64-bit) floating-point ele... | vsubpd | — |
| `_mm256_maskz_sub_ps` | Subtract packed single-precision (32-bit) floating-point ele... | vsubps | — |
| `_mm256_maskz_subs_epi16` | Subtract packed signed 16-bit integers in b from packed 16-b... | vpsubsw | 1/1, 1/1 |
| `_mm256_maskz_subs_epi8` | Subtract packed signed 8-bit integers in b from packed 8-bit... | vpsubsb | 1/1, 1/1 |
| `_mm256_maskz_subs_epu16` | Subtract packed unsigned 16-bit integers in b from packed un... | vpsubusw | — |
| `_mm256_maskz_subs_epu8` | Subtract packed unsigned 8-bit integers in b from packed uns... | vpsubusb | — |
| `_mm256_maskz_ternarylogic_epi32` | Bitwise ternary logic that provides the capability to implem... | vpternlogd | — |
| `_mm256_maskz_ternarylogic_epi64` | Bitwise ternary logic that provides the capability to implem... | vpternlogq | — |
| `_mm256_maskz_unpackhi_epi16` | Unpack and interleave 16-bit integers from the high half of ... | vpunpckhwd | 1/1, 1/1 |
| `_mm256_maskz_unpackhi_epi32` | Unpack and interleave 32-bit integers from the high half of ... | vpunpckhdq | 1/1, 1/1 |
| `_mm256_maskz_unpackhi_epi64` | Unpack and interleave 64-bit integers from the high half of ... | vpunpckhqdq | 1/1, 1/1 |
| `_mm256_maskz_unpackhi_epi8` | Unpack and interleave 8-bit integers from the high half of e... | vpunpckhbw | 1/1, 1/1 |
| `_mm256_maskz_unpackhi_pd` | Unpack and interleave double-precision (64-bit) floating-poi... | vunpckhpd | 1/1, 1/1 |
| `_mm256_maskz_unpackhi_ps` | Unpack and interleave single-precision (32-bit) floating-poi... | vunpckhps | 1/1, 1/1 |
| `_mm256_maskz_unpacklo_epi16` | Unpack and interleave 16-bit integers from the low half of e... | vpunpcklwd | 1/1, 1/1 |
| `_mm256_maskz_unpacklo_epi32` | Unpack and interleave 32-bit integers from the low half of e... | vpunpckldq | 1/1, 1/1 |
| `_mm256_maskz_unpacklo_epi64` | Unpack and interleave 64-bit integers from the low half of e... | vpunpcklqdq | 1/1, 1/1 |
| `_mm256_maskz_unpacklo_epi8` | Unpack and interleave 8-bit integers from the low half of ea... | vpunpcklbw | 1/1, 1/1 |
| `_mm256_maskz_unpacklo_pd` | Unpack and interleave double-precision (64-bit) floating-poi... | vunpcklpd | 1/1, 1/1 |
| `_mm256_maskz_unpacklo_ps` | Unpack and interleave single-precision (32-bit) floating-poi... | vunpcklps | 1/1, 1/1 |
| `_mm256_maskz_xor_epi32` | Compute the bitwise XOR of packed 32-bit integers in a and b... | vpxord | 1/1, 1/1 |
| `_mm256_maskz_xor_epi64` | Compute the bitwise XOR of packed 64-bit integers in a and b... | vpxorq | 1/1, 1/1 |
| `_mm256_maskz_xor_pd` | Compute the bitwise XOR of packed double-precision (64-bit) ... | vxorpd | 1/1, 1/1 |
| `_mm256_maskz_xor_ps` | Compute the bitwise XOR of packed single-precision (32-bit) ... | vxorps | 1/1, 1/1 |
| `_mm256_max_epi64` | Compare packed signed 64-bit integers in a and b, and store ... | vpmaxsq | — |
| `_mm256_max_epu64` | Compare packed unsigned 64-bit integers in a and b, and stor... | vpmaxuq | — |
| `_mm256_min_epi64` | Compare packed signed 64-bit integers in a and b, and store ... | vpminsq | — |
| `_mm256_min_epu64` | Compare packed unsigned 64-bit integers in a and b, and stor... | vpminuq | — |
| `_mm256_movepi16_mask` | Set each bit of mask register k based on the most significan... | vpmovw2m | — |
| `_mm256_movepi32_mask` | Set each bit of mask register k based on the most significan... |  | — |
| `_mm256_movepi64_mask` | Set each bit of mask register k based on the most significan... |  | — |
| `_mm256_movepi8_mask` | Set each bit of mask register k based on the most significan... | vpmovmskb | — |
| `_mm256_movm_epi16` | Set each packed 16-bit integer in dst to all ones or all zer... | vpmovm2w | — |
| `_mm256_movm_epi32` | Set each packed 32-bit integer in dst to all ones or all zer... | vpmovm2d | — |
| `_mm256_movm_epi64` | Set each packed 64-bit integer in dst to all ones or all zer... | vpmovm2q | — |
| `_mm256_movm_epi8` | Set each packed 8-bit integer in dst to all ones or all zero... | vpmovm2b | — |
| `_mm256_mullo_epi64` | Multiply packed 64-bit integers in `a` and `b`, producing in... | vpmullq | — |
| `_mm256_or_epi32` | Compute the bitwise OR of packed 32-bit integers in a and b,... | vor | 1/1, 1/1 |
| `_mm256_or_epi64` | Compute the bitwise OR of packed 64-bit integers in a and b,... | vor | 1/1, 1/1 |
| `_mm256_permutex2var_epi16` | Shuffle 16-bit integers in a and b across lanes using the co... | vperm | 3/1, 2/1 |
| `_mm256_permutex2var_epi32` | Shuffle 32-bit integers in a and b across lanes using the co... | vperm | 3/1, 2/1 |
| `_mm256_permutex2var_epi64` | Shuffle 64-bit integers in a and b across lanes using the co... | vperm | 3/1, 2/1 |
| `_mm256_permutex2var_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vperm | 3/1, 2/1 |
| `_mm256_permutex2var_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vperm | 3/1, 2/1 |
| `_mm256_permutex_epi64` | Shuffle 64-bit integers in a within 256-bit lanes using the ... | vperm | 3/1, 2/1 |
| `_mm256_permutex_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vperm | 3/1, 2/1 |
| `_mm256_permutexvar_epi16` | Shuffle 16-bit integers in a across lanes using the correspo... | vpermw | 3/1, 2/1 |
| `_mm256_permutexvar_epi32` | Shuffle 32-bit integers in a across lanes using the correspo... | vperm | 3/1, 2/1 |
| `_mm256_permutexvar_epi64` | Shuffle 64-bit integers in a across lanes using the correspo... | vperm | 3/1, 2/1 |
| `_mm256_permutexvar_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vpermpd | 3/1, 2/1 |
| `_mm256_permutexvar_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vpermps | 3/1, 2/1 |
| `_mm256_range_pd` | Calculate the max, min, absolute max, or absolute min (depen... | vrangepd | — |
| `_mm256_range_ps` | Calculate the max, min, absolute max, or absolute min (depen... | vrangeps | — |
| `_mm256_rcp14_pd` | Compute the approximate reciprocal of packed double-precisio... | vrcp14pd | — |
| `_mm256_rcp14_ps` | Compute the approximate reciprocal of packed single-precisio... | vrcp14ps | — |
| `_mm256_reduce_add_epi16` | Reduce the packed 16-bit integers in a by addition. Returns ... |  | 1/1, 1/1 |
| `_mm256_reduce_add_epi8` | Reduce the packed 8-bit integers in a by addition. Returns t... |  | 1/1, 1/1 |
| `_mm256_reduce_and_epi16` | Reduce the packed 16-bit integers in a by bitwise AND. Retur... |  | 1/1, 1/1 |
| `_mm256_reduce_and_epi8` | Reduce the packed 8-bit integers in a by bitwise AND. Return... |  | 1/1, 1/1 |
| `_mm256_reduce_max_epi16` | Reduce the packed 16-bit integers in a by maximum. Returns t... |  | — |
| `_mm256_reduce_max_epi8` | Reduce the packed 8-bit integers in a by maximum. Returns th... |  | — |
| `_mm256_reduce_max_epu16` | Reduce the packed unsigned 16-bit integers in a by maximum. ... |  | — |
| `_mm256_reduce_max_epu8` | Reduce the packed unsigned 8-bit integers in a by maximum. R... |  | — |
| `_mm256_reduce_min_epi16` | Reduce the packed 16-bit integers in a by minimum. Returns t... |  | — |
| `_mm256_reduce_min_epi8` | Reduce the packed 8-bit integers in a by minimum. Returns th... |  | — |
| `_mm256_reduce_min_epu16` | Reduce the packed unsigned 16-bit integers in a by minimum. ... |  | — |
| `_mm256_reduce_min_epu8` | Reduce the packed unsigned 8-bit integers in a by minimum. R... |  | — |
| `_mm256_reduce_mul_epi16` | Reduce the packed 16-bit integers in a by multiplication. Re... |  | — |
| `_mm256_reduce_mul_epi8` | Reduce the packed 8-bit integers in a by multiplication. Ret... |  | — |
| `_mm256_reduce_or_epi16` | Reduce the packed 16-bit integers in a by bitwise OR. Return... |  | 1/1, 1/1 |
| `_mm256_reduce_or_epi8` | Reduce the packed 8-bit integers in a by bitwise OR. Returns... |  | 1/1, 1/1 |
| `_mm256_reduce_pd` | Extract the reduced argument of packed double-precision (64-... | vreducepd | — |
| `_mm256_reduce_ps` | Extract the reduced argument of packed single-precision (32-... | vreduceps | — |
| `_mm256_rol_epi32` | Rotate the bits in each packed 32-bit integer in a to the le... | vprold | — |
| `_mm256_rol_epi64` | Rotate the bits in each packed 64-bit integer in a to the le... | vprolq | — |
| `_mm256_rolv_epi32` | Rotate the bits in each packed 32-bit integer in a to the le... | vprolvd | — |
| `_mm256_rolv_epi64` | Rotate the bits in each packed 64-bit integer in a to the le... | vprolvq | — |
| `_mm256_ror_epi32` | Rotate the bits in each packed 32-bit integer in a to the ri... | vprold | — |
| `_mm256_ror_epi64` | Rotate the bits in each packed 64-bit integer in a to the ri... | vprolq | — |
| `_mm256_rorv_epi32` | Rotate the bits in each packed 32-bit integer in a to the ri... | vprorvd | — |
| `_mm256_rorv_epi64` | Rotate the bits in each packed 64-bit integer in a to the ri... | vprorvq | — |
| `_mm256_roundscale_pd` | Round packed double-precision (64-bit) floating-point elemen... | vrndscalepd | — |
| `_mm256_roundscale_ps` | Round packed single-precision (32-bit) floating-point elemen... | vrndscaleps | — |
| `_mm256_rsqrt14_pd` | Compute the approximate reciprocal square root of packed dou... | vrsqrt14pd | — |
| `_mm256_rsqrt14_ps` | Compute the approximate reciprocal square root of packed sin... | vrsqrt14ps | — |
| `_mm256_scalef_pd` | Scale the packed double-precision (64-bit) floating-point el... | vscalefpd | — |
| `_mm256_scalef_ps` | Scale the packed single-precision (32-bit) floating-point el... | vscalefps | — |
| `_mm256_shuffle_f32x4` | Shuffle 128-bits (composed of 4 single-precision (32-bit) fl... | vperm | 1/1, 1/1 |
| `_mm256_shuffle_f64x2` | Shuffle 128-bits (composed of 2 double-precision (64-bit) fl... | vperm | 1/1, 1/1 |
| `_mm256_shuffle_i32x4` | Shuffle 128-bits (composed of 4 32-bit integers) selected by... | vperm | 1/1, 1/1 |
| `_mm256_shuffle_i64x2` | Shuffle 128-bits (composed of 2 64-bit integers) selected by... | vperm | 1/1, 1/1 |
| `_mm256_sllv_epi16` | Shift packed 16-bit integers in a left by the amount specifi... | vpsllvw | 1/1, 1/1 |
| `_mm256_sra_epi64` | Shift packed 64-bit integers in a right by count while shift... | vpsraq | 1/1, 1/1 |
| `_mm256_srai_epi64` | Shift packed 64-bit integers in a right by imm8 while shifti... | vpsraq | 1/1, 1/1 |
| `_mm256_srav_epi16` | Shift packed 16-bit integers in a right by the amount specif... | vpsravw | 1/1, 1/1 |
| `_mm256_srav_epi64` | Shift packed 64-bit integers in a right by the amount specif... | vpsravq | 1/1, 1/1 |
| `_mm256_srlv_epi16` | Shift packed 16-bit integers in a right by the amount specif... | vpsrlvw | 1/1, 1/1 |
| `_mm256_ternarylogic_epi32` | Bitwise ternary logic that provides the capability to implem... | vpternlogd | — |
| `_mm256_ternarylogic_epi64` | Bitwise ternary logic that provides the capability to implem... | vpternlogq | — |
| `_mm256_test_epi16_mask` | Compute the bitwise AND of packed 16-bit integers in a and b... | vptestmw | — |
| `_mm256_test_epi32_mask` | Compute the bitwise AND of packed 32-bit integers in a and b... | vptestmd | — |
| `_mm256_test_epi64_mask` | Compute the bitwise AND of packed 64-bit integers in a and b... | vptestmq | — |
| `_mm256_test_epi8_mask` | Compute the bitwise AND of packed 8-bit integers in a and b,... | vptestmb | — |
| `_mm256_testn_epi16_mask` | Compute the bitwise NAND of packed 16-bit integers in a and ... | vptestnmw | — |
| `_mm256_testn_epi32_mask` | Compute the bitwise NAND of packed 32-bit integers in a and ... | vptestnmd | — |
| `_mm256_testn_epi64_mask` | Compute the bitwise NAND of packed 64-bit integers in a and ... | vptestnmq | — |
| `_mm256_testn_epi8_mask` | Compute the bitwise NAND of packed 8-bit integers in a and b... | vptestnmb | — |
| `_mm256_xor_epi32` | Compute the bitwise XOR of packed 32-bit integers in a and b... | vxor | 1/1, 1/1 |
| `_mm256_xor_epi64` | Compute the bitwise XOR of packed 64-bit integers in a and b... | vxor | 1/1, 1/1 |
| `_mm512_abs_epi16` | Compute the absolute value of packed signed 16-bit integers ... | vpabsw | — |
| `_mm512_abs_epi32` | Computes the absolute values of packed 32-bit integers in `a... | vpabsd | — |
| `_mm512_abs_epi64` | Compute the absolute value of packed signed 64-bit integers ... | vpabsq | — |
| `_mm512_abs_epi8` | Compute the absolute value of packed signed 8-bit integers i... | vpabsb | — |
| `_mm512_abs_pd` | Finds the absolute value of each packed double-precision (64... | vpandq | — |
| `_mm512_abs_ps` | Finds the absolute value of each packed single-precision (32... | vpandd | — |
| `_mm512_add_epi16` | Add packed 16-bit integers in a and b, and store the results... | vpaddw | —/1/1 |
| `_mm512_add_epi32` | Add packed 32-bit integers in a and b, and store the results... | vpaddd | —/1/1 |
| `_mm512_add_epi64` | Add packed 64-bit integers in a and b, and store the results... | vpaddq | —/1/1 |
| `_mm512_add_epi8` | Add packed 8-bit integers in a and b, and store the results ... | vpaddb | —/1/1 |
| `_mm512_add_pd` | Add packed double-precision (64-bit) floating-point elements... | vaddpd | —/3/1 |
| `_mm512_add_ps` | Add packed single-precision (32-bit) floating-point elements... | vaddps | —/3/1 |
| `_mm512_add_round_pd` | Add packed double-precision (64-bit) floating-point elements... | vaddpd | — |
| `_mm512_add_round_ps` | Add packed single-precision (32-bit) floating-point elements... | vaddps | — |
| `_mm512_adds_epi16` | Add packed signed 16-bit integers in a and b using saturatio... | vpaddsw | —/1/1 |
| `_mm512_adds_epi8` | Add packed signed 8-bit integers in a and b using saturation... | vpaddsb | —/1/1 |
| `_mm512_adds_epu16` | Add packed unsigned 16-bit integers in a and b using saturat... | vpaddusw | — |
| `_mm512_adds_epu8` | Add packed unsigned 8-bit integers in a and b using saturati... | vpaddusb | — |
| `_mm512_alignr_epi32` | Concatenate a and b into a 128-byte immediate result, shift ... | valignd | — |
| `_mm512_alignr_epi64` | Concatenate a and b into a 128-byte immediate result, shift ... | valignq | — |
| `_mm512_alignr_epi8` | Concatenate pairs of 16-byte blocks in a and b into a 32-byt... | vpalignr | — |
| `_mm512_and_epi32` | Compute the bitwise AND of packed 32-bit integers in a and b... | vpandq | —/1/1 |
| `_mm512_and_epi64` | Compute the bitwise AND of 512 bits (composed of packed 64-b... | vpandq | —/1/1 |
| `_mm512_and_pd` | Compute the bitwise AND of packed double-precision (64-bit) ... | vandp | —/1/1 |
| `_mm512_and_ps` | Compute the bitwise AND of packed single-precision (32-bit) ... | vandps | —/1/1 |
| `_mm512_and_si512` | Compute the bitwise AND of 512 bits (representing integer da... | vpandq | —/1/1 |
| `_mm512_andnot_epi32` | Compute the bitwise NOT of packed 32-bit integers in a and t... | vpandnq | —/1/1 |
| `_mm512_andnot_epi64` | Compute the bitwise NOT of 512 bits (composed of packed 64-b... | vpandnq | —/1/1 |
| `_mm512_andnot_pd` | Compute the bitwise NOT of packed double-precision (64-bit) ... | vandnp | —/1/1 |
| `_mm512_andnot_ps` | Compute the bitwise NOT of packed single-precision (32-bit) ... | vandnps | —/1/1 |
| `_mm512_andnot_si512` | Compute the bitwise NOT of 512 bits (representing integer da... | vpandnq | —/1/1 |
| `_mm512_avg_epu16` | Average packed unsigned 16-bit integers in a and b, and stor... | vpavgw | — |
| `_mm512_avg_epu8` | Average packed unsigned 8-bit integers in a and b, and store... | vpavgb | — |
| `_mm512_broadcast_f32x2` | Broadcasts the lower 2 packed single-precision (32-bit) floa... |  | — |
| `_mm512_broadcast_f32x4` | Broadcast the 4 packed single-precision (32-bit) floating-po... |  | — |
| `_mm512_broadcast_f32x8` | Broadcasts the 8 packed single-precision (32-bit) floating-p... |  | — |
| `_mm512_broadcast_f64x2` | Broadcasts the 2 packed double-precision (64-bit) floating-p... |  | — |
| `_mm512_broadcast_f64x4` | Broadcast the 4 packed double-precision (64-bit) floating-po... |  | — |
| `_mm512_broadcast_i32x2` | Broadcasts the lower 2 packed 32-bit integers from a to all ... |  | — |
| `_mm512_broadcast_i32x4` | Broadcast the 4 packed 32-bit integers from a to all element... |  | — |
| `_mm512_broadcast_i32x8` | Broadcasts the 8 packed 32-bit integers from a to all elemen... |  | — |
| `_mm512_broadcast_i64x2` | Broadcasts the 2 packed 64-bit integers from a to all elemen... |  | — |
| `_mm512_broadcast_i64x4` | Broadcast the 4 packed 64-bit integers from a to all element... |  | — |
| `_mm512_broadcastb_epi8` | Broadcast the low packed 8-bit integer from a to all element... | vpbroadcastb | — |
| `_mm512_broadcastd_epi32` | Broadcast the low packed 32-bit integer from a to all elemen... | vbroadcast | — |
| `_mm512_broadcastmb_epi64` | Broadcast the low 8-bits from input mask k to all 64-bit ele... | vpbroadcast | — |
| `_mm512_broadcastmw_epi32` | Broadcast the low 16-bits from input mask k to all 32-bit el... | vpbroadcast | — |
| `_mm512_broadcastq_epi64` | Broadcast the low packed 64-bit integer from a to all elemen... | vbroadcast | — |
| `_mm512_broadcastsd_pd` | Broadcast the low double-precision (64-bit) floating-point e... | vbroadcastsd | — |
| `_mm512_broadcastss_ps` | Broadcast the low single-precision (32-bit) floating-point e... | vbroadcastss | — |
| `_mm512_broadcastw_epi16` | Broadcast the low packed 16-bit integer from a to all elemen... | vpbroadcastw | — |
| `_mm512_bslli_epi128` | Shift 128-bit lanes in a left by imm8 bytes while shifting i... | vpslldq | — |
| `_mm512_bsrli_epi128` | Shift 128-bit lanes in a right by imm8 bytes while shifting ... | vpsrldq | — |
| `_mm512_castpd128_pd512` | Cast vector of type __m128d to type __m512d; the upper 384 b... |  | — |
| `_mm512_castpd256_pd512` | Cast vector of type __m256d to type __m512d; the upper 256 b... |  | — |
| `_mm512_castpd512_pd128` | Cast vector of type __m512d to type __m128d. This intrinsic ... |  | — |
| `_mm512_castpd512_pd256` | Cast vector of type __m512d to type __m256d. This intrinsic ... |  | — |
| `_mm512_castpd_ps` | Cast vector of type __m512d to type __m512. This intrinsic i... |  | — |
| `_mm512_castpd_si512` | Cast vector of type __m512d to type __m512i. This intrinsic ... |  | — |
| `_mm512_castps128_ps512` | Cast vector of type __m128 to type __m512; the upper 384 bit... |  | — |
| `_mm512_castps256_ps512` | Cast vector of type __m256 to type __m512; the upper 256 bit... |  | — |
| `_mm512_castps512_ps128` | Cast vector of type __m512 to type __m128. This intrinsic is... |  | — |
| `_mm512_castps512_ps256` | Cast vector of type __m512 to type __m256. This intrinsic is... |  | — |
| `_mm512_castps_pd` | Cast vector of type __m512 to type __m512d. This intrinsic i... |  | — |
| `_mm512_castps_si512` | Cast vector of type __m512 to type __m512i. This intrinsic i... |  | — |
| `_mm512_castsi128_si512` | Cast vector of type __m128i to type __m512i; the upper 384 b... |  | — |
| `_mm512_castsi256_si512` | Cast vector of type __m256i to type __m512i; the upper 256 b... |  | — |
| `_mm512_castsi512_pd` | Cast vector of type __m512i to type __m512d. This intrinsic ... |  | — |
| `_mm512_castsi512_ps` | Cast vector of type __m512i to type __m512. This intrinsic i... |  | — |
| `_mm512_castsi512_si128` | Cast vector of type __m512i to type __m128i. This intrinsic ... |  | — |
| `_mm512_castsi512_si256` | Cast vector of type __m512i to type __m256i. This intrinsic ... |  | — |
| `_mm512_cmp_epi16_mask` | Compare packed signed 16-bit integers in a and b based on th... | vpcmp | — |
| `_mm512_cmp_epi32_mask` | Compare packed signed 32-bit integers in a and b based on th... | vpcmp | — |
| `_mm512_cmp_epi64_mask` | Compare packed signed 64-bit integers in a and b based on th... | vpcmp | — |
| `_mm512_cmp_epi8_mask` | Compare packed signed 8-bit integers in a and b based on the... | vpcmp | — |
| `_mm512_cmp_epu16_mask` | Compare packed unsigned 16-bit integers in a and b based on ... | vpcmp | — |
| `_mm512_cmp_epu32_mask` | Compare packed unsigned 32-bit integers in a and b based on ... | vpcmp | — |
| `_mm512_cmp_epu64_mask` | Compare packed unsigned 64-bit integers in a and b based on ... | vpcmp | — |
| `_mm512_cmp_epu8_mask` | Compare packed unsigned 8-bit integers in a and b based on t... | vpcmp | — |
| `_mm512_cmp_pd_mask` | Compare packed double-precision (64-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm512_cmp_ps_mask` | Compare packed single-precision (32-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm512_cmp_round_pd_mask` | Compare packed double-precision (64-bit) floating-point elem... | vcmp | — |
| `_mm512_cmp_round_ps_mask` | Compare packed single-precision (32-bit) floating-point elem... | vcmp | — |
| `_mm512_cmpeq_epi16_mask` | Compare packed signed 16-bit integers in a and b for equalit... | vpcmp | 1/1, 1/1 |
| `_mm512_cmpeq_epi32_mask` | Compare packed 32-bit integers in a and b for equality, and ... | vpcmp | 1/1, 1/1 |
| `_mm512_cmpeq_epi64_mask` | Compare packed 64-bit integers in a and b for equality, and ... | vpcmp | 1/1, 1/1 |
| `_mm512_cmpeq_epi8_mask` | Compare packed signed 8-bit integers in a and b for equality... | vpcmp | 1/1, 1/1 |
| `_mm512_cmpeq_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for equal... | vpcmp | — |
| `_mm512_cmpeq_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for equal... | vpcmp | — |
| `_mm512_cmpeq_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for equal... | vpcmp | — |
| `_mm512_cmpeq_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for equali... | vpcmp | — |
| `_mm512_cmpeq_pd_mask` | Compare packed double-precision (64-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm512_cmpeq_ps_mask` | Compare packed single-precision (32-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm512_cmpge_epi16_mask` | Compare packed signed 16-bit integers in a and b for greater... | vpcmp | — |
| `_mm512_cmpge_epi32_mask` | Compare packed signed 32-bit integers in a and b for greater... | vpcmp | — |
| `_mm512_cmpge_epi64_mask` | Compare packed signed 64-bit integers in a and b for greater... | vpcmp | — |
| `_mm512_cmpge_epi8_mask` | Compare packed signed 8-bit integers in a and b for greater-... | vpcmp | — |
| `_mm512_cmpge_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for great... | vpcmp | — |
| `_mm512_cmpge_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for great... | vpcmp | — |
| `_mm512_cmpge_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for great... | vpcmp | — |
| `_mm512_cmpge_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for greate... | vpcmp | — |
| `_mm512_cmpgt_epi16_mask` | Compare packed signed 16-bit integers in a and b for greater... | vpcmp | 1/1, 1/1 |
| `_mm512_cmpgt_epi32_mask` | Compare packed signed 32-bit integers in a and b for greater... | vpcmp | 1/1, 1/1 |
| `_mm512_cmpgt_epi64_mask` | Compare packed signed 64-bit integers in a and b for greater... | vpcmp | 1/1, 1/1 |
| `_mm512_cmpgt_epi8_mask` | Compare packed signed 8-bit integers in a and b for greater-... | vpcmp | 1/1, 1/1 |
| `_mm512_cmpgt_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for great... | vpcmp | — |
| `_mm512_cmpgt_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for great... | vpcmp | — |
| `_mm512_cmpgt_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for great... | vpcmp | — |
| `_mm512_cmpgt_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for greate... | vpcmp | — |
| `_mm512_cmple_epi16_mask` | Compare packed signed 16-bit integers in a and b for less-th... | vpcmp | — |
| `_mm512_cmple_epi32_mask` | Compare packed signed 32-bit integers in a and b for less-th... | vpcmp | — |
| `_mm512_cmple_epi64_mask` | Compare packed signed 64-bit integers in a and b for less-th... | vpcmp | — |
| `_mm512_cmple_epi8_mask` | Compare packed signed 8-bit integers in a and b for less-tha... | vpcmp | — |
| `_mm512_cmple_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for less-... | vpcmp | — |
| `_mm512_cmple_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for less-... | vpcmp | — |
| `_mm512_cmple_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for less-... | vpcmp | — |
| `_mm512_cmple_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for less-t... | vpcmp | — |
| `_mm512_cmple_pd_mask` | Compare packed double-precision (64-bit) floating-point elem... | vcmp | — |
| `_mm512_cmple_ps_mask` | Compare packed single-precision (32-bit) floating-point elem... | vcmp | — |
| `_mm512_cmplt_epi16_mask` | Compare packed signed 16-bit integers in a and b for less-th... | vpcmp | 1/1, 1/1 |
| `_mm512_cmplt_epi32_mask` | Compare packed signed 32-bit integers in a and b for less-th... | vpcmp | 1/1, 1/1 |
| `_mm512_cmplt_epi64_mask` | Compare packed signed 64-bit integers in a and b for less-th... | vpcmp | 1/1, 1/1 |
| `_mm512_cmplt_epi8_mask` | Compare packed signed 8-bit integers in a and b for less-tha... | vpcmp | 1/1, 1/1 |
| `_mm512_cmplt_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for less-... | vpcmp | — |
| `_mm512_cmplt_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for less-... | vpcmp | — |
| `_mm512_cmplt_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for less-... | vpcmp | — |
| `_mm512_cmplt_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for less-t... | vpcmp | — |
| `_mm512_cmplt_pd_mask` | Compare packed double-precision (64-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm512_cmplt_ps_mask` | Compare packed single-precision (32-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm512_cmpneq_epi16_mask` | Compare packed signed 16-bit integers in a and b for not-equ... | vpcmp | — |
| `_mm512_cmpneq_epi32_mask` | Compare packed 32-bit integers in a and b for not-equal, and... | vpcmp | — |
| `_mm512_cmpneq_epi64_mask` | Compare packed signed 64-bit integers in a and b for not-equ... | vpcmp | — |
| `_mm512_cmpneq_epi8_mask` | Compare packed signed 8-bit integers in a and b for not-equa... | vpcmp | — |
| `_mm512_cmpneq_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for not-e... | vpcmp | — |
| `_mm512_cmpneq_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for not-e... | vpcmp | — |
| `_mm512_cmpneq_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for not-e... | vpcmp | — |
| `_mm512_cmpneq_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for not-eq... | vpcmp | — |
| `_mm512_cmpneq_pd_mask` | Compare packed double-precision (64-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm512_cmpneq_ps_mask` | Compare packed single-precision (32-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm512_cmpnle_pd_mask` | Compare packed double-precision (64-bit) floating-point elem... | vcmp | — |
| `_mm512_cmpnle_ps_mask` | Compare packed single-precision (32-bit) floating-point elem... | vcmp | — |
| `_mm512_cmpnlt_pd_mask` | Compare packed double-precision (64-bit) floating-point elem... | vcmp | — |
| `_mm512_cmpnlt_ps_mask` | Compare packed single-precision (32-bit) floating-point elem... | vcmp | — |
| `_mm512_cmpord_pd_mask` | Compare packed double-precision (64-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm512_cmpord_ps_mask` | Compare packed single-precision (32-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm512_cmpunord_pd_mask` | Compare packed double-precision (64-bit) floating-point elem... | vcmp | — |
| `_mm512_cmpunord_ps_mask` | Compare packed single-precision (32-bit) floating-point elem... | vcmp | — |
| `_mm512_conflict_epi32` | Test each 32-bit element of a for equality with all other el... | vpconflictd | — |
| `_mm512_conflict_epi64` | Test each 64-bit element of a for equality with all other el... | vpconflictq | — |
| `_mm512_cvt_roundepi32_ps` | Convert packed signed 32-bit integers in a to packed single-... | vcvtdq2ps | 4/1, 3/1 |
| `_mm512_cvt_roundepi64_pd` | Convert packed signed 64-bit integers in a to packed double-... | vcvtqq2pd | 4/1, 3/1 |
| `_mm512_cvt_roundepi64_ps` | Convert packed signed 64-bit integers in a to packed single-... | vcvtqq2ps | 4/1, 3/1 |
| `_mm512_cvt_roundepu32_ps` | Convert packed unsigned 32-bit integers in a to packed singl... | vcvtudq2ps | 4/1, 3/1 |
| `_mm512_cvt_roundepu64_pd` | Convert packed unsigned 64-bit integers in a to packed doubl... | vcvtuqq2pd | 4/1, 3/1 |
| `_mm512_cvt_roundepu64_ps` | Convert packed unsigned 64-bit integers in a to packed singl... | vcvtuqq2ps | 4/1, 3/1 |
| `_mm512_cvt_roundpd_epi32` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2dq | 4/1, 3/1 |
| `_mm512_cvt_roundpd_epi64` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2qq | 4/1, 3/1 |
| `_mm512_cvt_roundpd_epu32` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2udq | 4/1, 3/1 |
| `_mm512_cvt_roundpd_epu64` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2uqq | 4/1, 3/1 |
| `_mm512_cvt_roundpd_ps` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2ps | 4/1, 3/1 |
| `_mm512_cvt_roundph_ps` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2ps | 4/1, 3/1 |
| `_mm512_cvt_roundps_epi32` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2dq | 4/1, 3/1 |
| `_mm512_cvt_roundps_epi64` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2qq | 4/1, 3/1 |
| `_mm512_cvt_roundps_epu32` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2udq | 4/1, 3/1 |
| `_mm512_cvt_roundps_epu64` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2uqq | 4/1, 3/1 |
| `_mm512_cvt_roundps_pd` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2pd | 4/1, 3/1 |
| `_mm512_cvt_roundps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2ph | 4/1, 3/1 |
| `_mm512_cvtepi16_epi32` | Sign extend packed 16-bit integers in a to packed 32-bit int... | vpmovsxwd | 4/1, 3/1 |
| `_mm512_cvtepi16_epi64` | Sign extend packed 16-bit integers in a to packed 64-bit int... | vpmovsxwq | 4/1, 3/1 |
| `_mm512_cvtepi16_epi8` | Convert packed 16-bit integers in a to packed 8-bit integers... | vpmovwb | 4/1, 3/1 |
| `_mm512_cvtepi32_epi16` | Convert packed 32-bit integers in a to packed 16-bit integer... | vpmovdw | 4/1, 3/1 |
| `_mm512_cvtepi32_epi64` | Sign extend packed 32-bit integers in a to packed 64-bit int... | vpmovsxdq | 4/1, 3/1 |
| `_mm512_cvtepi32_epi8` | Convert packed 32-bit integers in a to packed 8-bit integers... | vpmovdb | 4/1, 3/1 |
| `_mm512_cvtepi32_pd` | Convert packed signed 32-bit integers in a to packed double-... | vcvtdq2pd | 4/1, 3/1 |
| `_mm512_cvtepi32_ps` | Convert packed signed 32-bit integers in a to packed single-... | vcvtdq2ps | 4/1, 3/1 |
| `_mm512_cvtepi32lo_pd` | Performs element-by-element conversion of the lower half of ... | vcvtdq2pd | 4/1, 3/1 |
| `_mm512_cvtepi64_epi16` | Convert packed 64-bit integers in a to packed 16-bit integer... | vpmovqw | 4/1, 3/1 |
| `_mm512_cvtepi64_epi32` | Convert packed 64-bit integers in a to packed 32-bit integer... | vpmovqd | 4/1, 3/1 |
| `_mm512_cvtepi64_epi8` | Convert packed 64-bit integers in a to packed 8-bit integers... | vpmovqb | 4/1, 3/1 |
| `_mm512_cvtepi64_pd` | Convert packed signed 64-bit integers in a to packed double-... | vcvtqq2pd | 4/1, 3/1 |
| `_mm512_cvtepi64_ps` | Convert packed signed 64-bit integers in a to packed single-... | vcvtqq2ps | 4/1, 3/1 |
| `_mm512_cvtepi8_epi16` | Sign extend packed 8-bit integers in a to packed 16-bit inte... | vpmovsxbw | 4/1, 3/1 |
| `_mm512_cvtepi8_epi32` | Sign extend packed 8-bit integers in a to packed 32-bit inte... | vpmovsxbd | 4/1, 3/1 |
| `_mm512_cvtepi8_epi64` | Sign extend packed 8-bit integers in the low 8 bytes of a to... | vpmovsxbq | 4/1, 3/1 |
| `_mm512_cvtepu16_epi32` | Zero extend packed unsigned 16-bit integers in a to packed 3... | vpmovzxwd | 4/1, 3/1 |
| `_mm512_cvtepu16_epi64` | Zero extend packed unsigned 16-bit integers in a to packed 6... | vpmovzxwq | 4/1, 3/1 |
| `_mm512_cvtepu32_epi64` | Zero extend packed unsigned 32-bit integers in a to packed 6... | vpmovzxdq | 4/1, 3/1 |
| `_mm512_cvtepu32_pd` | Convert packed unsigned 32-bit integers in a to packed doubl... | vcvtudq2pd | 4/1, 3/1 |
| `_mm512_cvtepu32_ps` | Convert packed unsigned 32-bit integers in a to packed singl... | vcvtudq2ps | 4/1, 3/1 |
| `_mm512_cvtepu32lo_pd` | Performs element-by-element conversion of the lower half of ... | vcvtudq2pd | 4/1, 3/1 |
| `_mm512_cvtepu64_pd` | Convert packed unsigned 64-bit integers in a to packed doubl... | vcvtuqq2pd | 4/1, 3/1 |
| `_mm512_cvtepu64_ps` | Convert packed unsigned 64-bit integers in a to packed singl... | vcvtuqq2ps | 4/1, 3/1 |
| `_mm512_cvtepu8_epi16` | Zero extend packed unsigned 8-bit integers in a to packed 16... | vpmovzxbw | 4/1, 3/1 |
| `_mm512_cvtepu8_epi32` | Zero extend packed unsigned 8-bit integers in a to packed 32... | vpmovzxbd | 4/1, 3/1 |
| `_mm512_cvtepu8_epi64` | Zero extend packed unsigned 8-bit integers in the low 8 byte... | vpmovzxbq | 4/1, 3/1 |
| `_mm512_cvtpd_epi32` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2dq | 4/1, 3/1 |
| `_mm512_cvtpd_epi64` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2qq | 4/1, 3/1 |
| `_mm512_cvtpd_epu32` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2udq | 4/1, 3/1 |
| `_mm512_cvtpd_epu64` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2uqq | 4/1, 3/1 |
| `_mm512_cvtpd_ps` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2ps | 4/1, 3/1 |
| `_mm512_cvtpd_pslo` | Performs an element-by-element conversion of packed double-p... | vcvtpd2ps | 4/1, 3/1 |
| `_mm512_cvtph_ps` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2ps | 4/1, 3/1 |
| `_mm512_cvtps_epi32` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2dq | 4/1, 3/1 |
| `_mm512_cvtps_epi64` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2qq | 4/1, 3/1 |
| `_mm512_cvtps_epu32` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2udq | 4/1, 3/1 |
| `_mm512_cvtps_epu64` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2uqq | 4/1, 3/1 |
| `_mm512_cvtps_pd` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2pd | 4/1, 3/1 |
| `_mm512_cvtps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2ph | 4/1, 3/1 |
| `_mm512_cvtpslo_pd` | Performs element-by-element conversion of the lower half of ... | vcvtps2pd | 4/1, 3/1 |
| `_mm512_cvtsd_f64` | Copy the lower double-precision (64-bit) floating-point elem... |  | 4/1, 3/1 |
| `_mm512_cvtsepi16_epi8` | Convert packed signed 16-bit integers in a to packed 8-bit i... | vpmovswb | 4/1, 3/1 |
| `_mm512_cvtsepi32_epi16` | Convert packed signed 32-bit integers in a to packed 16-bit ... | vpmovsdw | 4/1, 3/1 |
| `_mm512_cvtsepi32_epi8` | Convert packed signed 32-bit integers in a to packed 8-bit i... | vpmovsdb | 4/1, 3/1 |
| `_mm512_cvtsepi64_epi16` | Convert packed signed 64-bit integers in a to packed 16-bit ... | vpmovsqw | 4/1, 3/1 |
| `_mm512_cvtsepi64_epi32` | Convert packed signed 64-bit integers in a to packed 32-bit ... | vpmovsqd | 4/1, 3/1 |
| `_mm512_cvtsepi64_epi8` | Convert packed signed 64-bit integers in a to packed 8-bit i... | vpmovsqb | 4/1, 3/1 |
| `_mm512_cvtsi512_si32` | Copy the lower 32-bit integer in a to dst | vmovd | 4/1, 3/1 |
| `_mm512_cvtss_f32` | Copy the lower single-precision (32-bit) floating-point elem... |  | 4/1, 3/1 |
| `_mm512_cvtt_roundpd_epi32` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2dq | 4/1, 3/1 |
| `_mm512_cvtt_roundpd_epi64` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2qq | 4/1, 3/1 |
| `_mm512_cvtt_roundpd_epu32` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2udq | 4/1, 3/1 |
| `_mm512_cvtt_roundpd_epu64` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2uqq | 4/1, 3/1 |
| `_mm512_cvtt_roundps_epi32` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2dq | 4/1, 3/1 |
| `_mm512_cvtt_roundps_epi64` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2qq | 4/1, 3/1 |
| `_mm512_cvtt_roundps_epu32` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2udq | 4/1, 3/1 |
| `_mm512_cvtt_roundps_epu64` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2uqq | 4/1, 3/1 |
| `_mm512_cvttpd_epi32` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2dq | 4/1, 3/1 |
| `_mm512_cvttpd_epi64` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2qq | 4/1, 3/1 |
| `_mm512_cvttpd_epu32` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2udq | 4/1, 3/1 |
| `_mm512_cvttpd_epu64` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2uqq | 4/1, 3/1 |
| `_mm512_cvttps_epi32` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2dq | 4/1, 3/1 |
| `_mm512_cvttps_epi64` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2qq | 4/1, 3/1 |
| `_mm512_cvttps_epu32` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2udq | 4/1, 3/1 |
| `_mm512_cvttps_epu64` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2uqq | 4/1, 3/1 |
| `_mm512_cvtusepi16_epi8` | Convert packed unsigned 16-bit integers in a to packed unsig... | vpmovuswb | 4/1, 3/1 |
| `_mm512_cvtusepi32_epi16` | Convert packed unsigned 32-bit integers in a to packed unsig... | vpmovusdw | 4/1, 3/1 |
| `_mm512_cvtusepi32_epi8` | Convert packed unsigned 32-bit integers in a to packed unsig... | vpmovusdb | 4/1, 3/1 |
| `_mm512_cvtusepi64_epi16` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqw | 4/1, 3/1 |
| `_mm512_cvtusepi64_epi32` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqd | 4/1, 3/1 |
| `_mm512_cvtusepi64_epi8` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqb | 4/1, 3/1 |
| `_mm512_dbsad_epu8` | Compute the sum of absolute differences (SADs) of quadruplet... | vdbpsadbw | — |
| `_mm512_div_pd` | Divide packed double-precision (64-bit) floating-point eleme... | vdivpd | 20/14, 13/7 |
| `_mm512_div_ps` | Divide packed single-precision (32-bit) floating-point eleme... | vdivps | 13/7, 10/4 |
| `_mm512_div_round_pd` | Divide packed double-precision (64-bit) floating-point eleme... | vdivpd | — |
| `_mm512_div_round_ps` | Divide packed single-precision (32-bit) floating-point eleme... | vdivps | — |
| `_mm512_extractf32x4_ps` | Extract 128 bits (composed of 4 packed single-precision (32-... | vextractf32x4 | — |
| `_mm512_extractf32x8_ps` | Extracts 256 bits (composed of 8 packed single-precision (32... |  | — |
| `_mm512_extractf64x2_pd` | Extracts 128 bits (composed of 2 packed double-precision (64... |  | — |
| `_mm512_extractf64x4_pd` | Extract 256 bits (composed of 4 packed double-precision (64-... | vextractf64x4 | — |
| `_mm512_extracti32x4_epi32` | Extract 128 bits (composed of 4 packed 32-bit integers) from... | vextractf32x4 | — |
| `_mm512_extracti32x8_epi32` | Extracts 256 bits (composed of 8 packed 32-bit integers) fro... |  | — |
| `_mm512_extracti64x2_epi64` | Extracts 128 bits (composed of 2 packed 64-bit integers) fro... |  | — |
| `_mm512_extracti64x4_epi64` | Extract 256 bits (composed of 4 packed 64-bit integers) from... | vextractf64x4 | — |
| `_mm512_fixupimm_pd` | Fix up packed double-precision (64-bit) floating-point eleme... | vfixupimmpd | — |
| `_mm512_fixupimm_ps` | Fix up packed single-precision (32-bit) floating-point eleme... | vfixupimmps | — |
| `_mm512_fixupimm_round_pd` | Fix up packed double-precision (64-bit) floating-point eleme... | vfixupimmpd | — |
| `_mm512_fixupimm_round_ps` | Fix up packed single-precision (32-bit) floating-point eleme... | vfixupimmps | — |
| `_mm512_fmadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmadd | —/4/1 |
| `_mm512_fmadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmadd | —/4/1 |
| `_mm512_fmadd_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmadd | —/4/1 |
| `_mm512_fmadd_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmadd | —/4/1 |
| `_mm512_fmaddsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmaddsub | —/4/1 |
| `_mm512_fmaddsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmaddsub | —/4/1 |
| `_mm512_fmaddsub_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmaddsub | —/4/1 |
| `_mm512_fmaddsub_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmaddsub | —/4/1 |
| `_mm512_fmsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsub | —/4/1 |
| `_mm512_fmsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsub | —/4/1 |
| `_mm512_fmsub_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsub | —/4/1 |
| `_mm512_fmsub_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsub | —/4/1 |
| `_mm512_fmsubadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsubadd | —/4/1 |
| `_mm512_fmsubadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsubadd | —/4/1 |
| `_mm512_fmsubadd_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsubadd | —/4/1 |
| `_mm512_fmsubadd_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsubadd | —/4/1 |
| `_mm512_fnmadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmadd | —/4/1 |
| `_mm512_fnmadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmadd | —/4/1 |
| `_mm512_fnmadd_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmadd | —/4/1 |
| `_mm512_fnmadd_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmadd | —/4/1 |
| `_mm512_fnmsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmsub | —/4/1 |
| `_mm512_fnmsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmsub | —/4/1 |
| `_mm512_fnmsub_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmsub | —/4/1 |
| `_mm512_fnmsub_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmsub | —/4/1 |
| `_mm512_fpclass_pd_mask` | Test packed double-precision (64-bit) floating-point element... | vfpclasspd | — |
| `_mm512_fpclass_ps_mask` | Test packed single-precision (32-bit) floating-point element... | vfpclassps | — |
| `_mm512_getexp_pd` | Convert the exponent of each packed double-precision (64-bit... | vgetexppd | — |
| `_mm512_getexp_ps` | Convert the exponent of each packed single-precision (32-bit... | vgetexpps | — |
| `_mm512_getexp_round_pd` | Convert the exponent of each packed double-precision (64-bit... | vgetexppd | — |
| `_mm512_getexp_round_ps` | Convert the exponent of each packed single-precision (32-bit... | vgetexpps | — |
| `_mm512_getmant_pd` | Normalize the mantissas of packed double-precision (64-bit) ... | vgetmantpd | — |
| `_mm512_getmant_ps` | Normalize the mantissas of packed single-precision (32-bit) ... | vgetmantps | — |
| `_mm512_getmant_round_pd` | Normalize the mantissas of packed double-precision (64-bit) ... | vgetmantpd | — |
| `_mm512_getmant_round_ps` | Normalize the mantissas of packed single-precision (32-bit) ... | vgetmantps | — |
| `_mm512_insertf32x4` | Copy a to dst, then insert 128 bits (composed of 4 packed si... | vinsertf32x4 | — |
| `_mm512_insertf32x8` | Copy a to dst, then insert 256 bits (composed of 8 packed si... |  | — |
| `_mm512_insertf64x2` | Copy a to dst, then insert 128 bits (composed of 2 packed do... |  | — |
| `_mm512_insertf64x4` | Copy a to dst, then insert 256 bits (composed of 4 packed do... | vinsertf64x4 | — |
| `_mm512_inserti32x4` | Copy a to dst, then insert 128 bits (composed of 4 packed 32... | vinsertf32x4 | — |
| `_mm512_inserti32x8` | Copy a to dst, then insert 256 bits (composed of 8 packed 32... |  | — |
| `_mm512_inserti64x2` | Copy a to dst, then insert 128 bits (composed of 2 packed 64... |  | — |
| `_mm512_inserti64x4` | Copy a to dst, then insert 256 bits (composed of 4 packed 64... | vinsertf64x4 | — |
| `_mm512_int2mask` | Converts integer mask into bitmask, storing the result in ds... |  | — |
| `_mm512_kand` | Compute the bitwise AND of 16-bit masks a and b, and store t... | and | — |
| `_mm512_kandn` | Compute the bitwise NOT of 16-bit masks a and then AND with ... | not | — |
| `_mm512_kmov` | Copy 16-bit mask a to k | mov | — |
| `_mm512_knot` | Compute the bitwise NOT of 16-bit mask a, and store the resu... |  | — |
| `_mm512_kor` | Compute the bitwise OR of 16-bit masks a and b, and store th... | or | — |
| `_mm512_kortestc` | Performs bitwise OR between k1 and k2, storing the result in... | cmp | — |
| `_mm512_kortestz` | Performs bitwise OR between k1 and k2, storing the result in... | xor | — |
| `_mm512_kunpackb` | Unpack and interleave 8 bits from masks a and b, and store t... | mov | — |
| `_mm512_kunpackd` | Unpack and interleave 32 bits from masks a and b, and store ... | mov | — |
| `_mm512_kunpackw` | Unpack and interleave 16 bits from masks a and b, and store ... | mov | — |
| `_mm512_kxnor` | Compute the bitwise XNOR of 16-bit masks a and b, and store ... | xor | — |
| `_mm512_kxor` | Compute the bitwise XOR of 16-bit masks a and b, and store t... | xor | — |
| `_mm512_lzcnt_epi32` | Counts the number of leading zero bits in each packed 32-bit... | vplzcntd | — |
| `_mm512_lzcnt_epi64` | Counts the number of leading zero bits in each packed 64-bit... | vplzcntq | — |
| `_mm512_madd_epi16` | Multiply packed signed 16-bit integers in a and b, producing... | vpmaddwd | — |
| `_mm512_maddubs_epi16` | Vertically multiply each unsigned 8-bit integer from a with ... | vpmaddubsw | — |
| `_mm512_mask2_permutex2var_epi16` | Shuffle 16-bit integers in a and b across lanes using the co... | vpermi2w | —/2/1 |
| `_mm512_mask2_permutex2var_epi32` | Shuffle 32-bit integers in a and b across lanes using the co... | vpermi2d | —/2/1 |
| `_mm512_mask2_permutex2var_epi64` | Shuffle 64-bit integers in a and b across lanes using the co... | vpermi2q | —/2/1 |
| `_mm512_mask2_permutex2var_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vperm | —/2/1 |
| `_mm512_mask2_permutex2var_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vperm | —/2/1 |
| `_mm512_mask2int` | Converts bit mask k1 into an integer value, storing the resu... | mov | — |
| `_mm512_mask3_fmadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmadd | —/4/1 |
| `_mm512_mask3_fmadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmadd | —/4/1 |
| `_mm512_mask3_fmadd_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmadd | —/4/1 |
| `_mm512_mask3_fmadd_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmadd | —/4/1 |
| `_mm512_mask3_fmaddsub_pd` | Multiply packed single-precision (32-bit) floating-point ele... | vfmaddsub | —/4/1 |
| `_mm512_mask3_fmaddsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmaddsub | —/4/1 |
| `_mm512_mask3_fmaddsub_round_pd` | Multiply packed single-precision (32-bit) floating-point ele... | vfmaddsub | —/4/1 |
| `_mm512_mask3_fmaddsub_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmaddsub | —/4/1 |
| `_mm512_mask3_fmsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsub | —/4/1 |
| `_mm512_mask3_fmsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsub | —/4/1 |
| `_mm512_mask3_fmsub_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsub | —/4/1 |
| `_mm512_mask3_fmsub_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsub | —/4/1 |
| `_mm512_mask3_fmsubadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsubadd | —/4/1 |
| `_mm512_mask3_fmsubadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsubadd | —/4/1 |
| `_mm512_mask3_fmsubadd_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsubadd | —/4/1 |
| `_mm512_mask3_fmsubadd_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsubadd | —/4/1 |
| `_mm512_mask3_fnmadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmadd | —/4/1 |
| `_mm512_mask3_fnmadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmadd | —/4/1 |
| `_mm512_mask3_fnmadd_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmadd | —/4/1 |
| `_mm512_mask3_fnmadd_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmadd | —/4/1 |
| `_mm512_mask3_fnmsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmsub | —/4/1 |
| `_mm512_mask3_fnmsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmsub | —/4/1 |
| `_mm512_mask3_fnmsub_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmsub | —/4/1 |
| `_mm512_mask3_fnmsub_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmsub | —/4/1 |
| `_mm512_mask_abs_epi16` | Compute the absolute value of packed signed 16-bit integers ... | vpabsw | — |
| `_mm512_mask_abs_epi32` | Computes the absolute value of packed 32-bit integers in `a`... | vpabsd | — |
| `_mm512_mask_abs_epi64` | Compute the absolute value of packed signed 64-bit integers ... | vpabsq | — |
| `_mm512_mask_abs_epi8` | Compute the absolute value of packed signed 8-bit integers i... | vpabsb | — |
| `_mm512_mask_abs_pd` | Finds the absolute value of each packed double-precision (64... | vpandq | — |
| `_mm512_mask_abs_ps` | Finds the absolute value of each packed single-precision (32... | vpandd | — |
| `_mm512_mask_add_epi16` | Add packed 16-bit integers in a and b, and store the results... | vpaddw | —/1/1 |
| `_mm512_mask_add_epi32` | Add packed 32-bit integers in a and b, and store the results... | vpaddd | —/1/1 |
| `_mm512_mask_add_epi64` | Add packed 64-bit integers in a and b, and store the results... | vpaddq | —/1/1 |
| `_mm512_mask_add_epi8` | Add packed 8-bit integers in a and b, and store the results ... | vpaddb | —/1/1 |
| `_mm512_mask_add_pd` | Add packed double-precision (64-bit) floating-point elements... | vaddpd | —/3/1 |
| `_mm512_mask_add_ps` | Add packed single-precision (32-bit) floating-point elements... | vaddps | —/3/1 |
| `_mm512_mask_add_round_pd` | Add packed double-precision (64-bit) floating-point elements... | vaddpd | — |
| `_mm512_mask_add_round_ps` | Add packed single-precision (32-bit) floating-point elements... | vaddps | — |
| `_mm512_mask_adds_epi16` | Add packed signed 16-bit integers in a and b using saturatio... | vpaddsw | —/1/1 |
| `_mm512_mask_adds_epi8` | Add packed signed 8-bit integers in a and b using saturation... | vpaddsb | —/1/1 |
| `_mm512_mask_adds_epu16` | Add packed unsigned 16-bit integers in a and b using saturat... | vpaddusw | — |
| `_mm512_mask_adds_epu8` | Add packed unsigned 8-bit integers in a and b using saturati... | vpaddusb | — |
| `_mm512_mask_alignr_epi32` | Concatenate a and b into a 128-byte immediate result, shift ... | valignd | — |
| `_mm512_mask_alignr_epi64` | Concatenate a and b into a 128-byte immediate result, shift ... | valignq | — |
| `_mm512_mask_alignr_epi8` | Concatenate pairs of 16-byte blocks in a and b into a 32-byt... | vpalignr | — |
| `_mm512_mask_and_epi32` | Performs element-by-element bitwise AND between packed 32-bi... | vpandd | —/1/1 |
| `_mm512_mask_and_epi64` | Compute the bitwise AND of packed 64-bit integers in a and b... | vpandq | —/1/1 |
| `_mm512_mask_and_pd` | Compute the bitwise AND of packed double-precision (64-bit) ... | vandpd | —/1/1 |
| `_mm512_mask_and_ps` | Compute the bitwise AND of packed single-precision (32-bit) ... | vandps | —/1/1 |
| `_mm512_mask_andnot_epi32` | Compute the bitwise NOT of packed 32-bit integers in a and t... | vpandnd | —/1/1 |
| `_mm512_mask_andnot_epi64` | Compute the bitwise NOT of packed 64-bit integers in a and t... | vpandnq | —/1/1 |
| `_mm512_mask_andnot_pd` | Compute the bitwise NOT of packed double-precision (64-bit) ... | vandnpd | —/1/1 |
| `_mm512_mask_andnot_ps` | Compute the bitwise NOT of packed single-precision (32-bit) ... | vandnps | —/1/1 |
| `_mm512_mask_avg_epu16` | Average packed unsigned 16-bit integers in a and b, and stor... | vpavgw | — |
| `_mm512_mask_avg_epu8` | Average packed unsigned 8-bit integers in a and b, and store... | vpavgb | — |
| `_mm512_mask_blend_epi16` | Blend packed 16-bit integers from a and b using control mask... | vmovdqu16 | 1/1, 1/1 |
| `_mm512_mask_blend_epi32` | Blend packed 32-bit integers from a and b using control mask... | vmovdqa32 | 1/1, 1/1 |
| `_mm512_mask_blend_epi64` | Blend packed 64-bit integers from a and b using control mask... | vmovdqa64 | 1/1, 1/1 |
| `_mm512_mask_blend_epi8` | Blend packed 8-bit integers from a and b using control mask ... | vmovdqu8 | 1/1, 1/1 |
| `_mm512_mask_blend_pd` | Blend packed double-precision (64-bit) floating-point elemen... | vmovapd | 1/1, 1/1 |
| `_mm512_mask_blend_ps` | Blend packed single-precision (32-bit) floating-point elemen... | vmovaps | 1/1, 1/1 |
| `_mm512_mask_broadcast_f32x2` | Broadcasts the lower 2 packed single-precision (32-bit) floa... | vbroadcastf32x2 | — |
| `_mm512_mask_broadcast_f32x4` | Broadcast the 4 packed single-precision (32-bit) floating-po... |  | — |
| `_mm512_mask_broadcast_f32x8` | Broadcasts the 8 packed single-precision (32-bit) floating-p... |  | — |
| `_mm512_mask_broadcast_f64x2` | Broadcasts the 2 packed double-precision (64-bit) floating-p... |  | — |
| `_mm512_mask_broadcast_f64x4` | Broadcast the 4 packed double-precision (64-bit) floating-po... |  | — |
| `_mm512_mask_broadcast_i32x2` | Broadcasts the lower 2 packed 32-bit integers from a to all ... | vbroadcasti32x2 | — |
| `_mm512_mask_broadcast_i32x4` | Broadcast the 4 packed 32-bit integers from a to all element... |  | — |
| `_mm512_mask_broadcast_i32x8` | Broadcasts the 8 packed 32-bit integers from a to all elemen... |  | — |
| `_mm512_mask_broadcast_i64x2` | Broadcasts the 2 packed 64-bit integers from a to all elemen... |  | — |
| `_mm512_mask_broadcast_i64x4` | Broadcast the 4 packed 64-bit integers from a to all element... |  | — |
| `_mm512_mask_broadcastb_epi8` | Broadcast the low packed 8-bit integer from a to all element... | vpbroadcastb | — |
| `_mm512_mask_broadcastd_epi32` | Broadcast the low packed 32-bit integer from a to all elemen... | vpbroadcast | — |
| `_mm512_mask_broadcastq_epi64` | Broadcast the low packed 64-bit integer from a to all elemen... | vpbroadcast | — |
| `_mm512_mask_broadcastsd_pd` | Broadcast the low double-precision (64-bit) floating-point e... | vbroadcastsd | — |
| `_mm512_mask_broadcastss_ps` | Broadcast the low single-precision (32-bit) floating-point e... | vbroadcastss | — |
| `_mm512_mask_broadcastw_epi16` | Broadcast the low packed 16-bit integer from a to all elemen... | vpbroadcastw | — |
| `_mm512_mask_cmp_epi16_mask` | Compare packed signed 16-bit integers in a and b based on th... | vpcmp | — |
| `_mm512_mask_cmp_epi32_mask` | Compare packed signed 32-bit integers in a and b based on th... | vpcmp | — |
| `_mm512_mask_cmp_epi64_mask` | Compare packed signed 64-bit integers in a and b based on th... | vpcmp | — |
| `_mm512_mask_cmp_epi8_mask` | Compare packed signed 8-bit integers in a and b based on the... | vpcmp | — |
| `_mm512_mask_cmp_epu16_mask` | Compare packed unsigned 16-bit integers in a and b based on ... | vpcmp | — |
| `_mm512_mask_cmp_epu32_mask` | Compare packed unsigned 32-bit integers in a and b based on ... | vpcmp | — |
| `_mm512_mask_cmp_epu64_mask` | Compare packed unsigned 64-bit integers in a and b based on ... | vpcmp | — |
| `_mm512_mask_cmp_epu8_mask` | Compare packed unsigned 8-bit integers in a and b based on t... | vpcmp | — |
| `_mm512_mask_cmp_pd_mask` | Compare packed double-precision (64-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm512_mask_cmp_ps_mask` | Compare packed single-precision (32-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm512_mask_cmp_round_pd_mask` | Compare packed double-precision (64-bit) floating-point elem... | vcmp | — |
| `_mm512_mask_cmp_round_ps_mask` | Compare packed single-precision (32-bit) floating-point elem... | vcmp | — |
| `_mm512_mask_cmpeq_epi16_mask` | Compare packed signed 16-bit integers in a and b for equalit... | vpcmp | 1/1, 1/1 |
| `_mm512_mask_cmpeq_epi32_mask` | Compare packed 32-bit integers in a and b for equality, and ... | vpcmp | 1/1, 1/1 |
| `_mm512_mask_cmpeq_epi64_mask` | Compare packed 64-bit integers in a and b for equality, and ... | vpcmp | 1/1, 1/1 |
| `_mm512_mask_cmpeq_epi8_mask` | Compare packed signed 8-bit integers in a and b for equality... | vpcmp | 1/1, 1/1 |
| `_mm512_mask_cmpeq_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for equal... | vpcmp | — |
| `_mm512_mask_cmpeq_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for equal... | vpcmp | — |
| `_mm512_mask_cmpeq_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for equal... | vpcmp | — |
| `_mm512_mask_cmpeq_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for equali... | vpcmp | — |
| `_mm512_mask_cmpeq_pd_mask` | Compare packed double-precision (64-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm512_mask_cmpeq_ps_mask` | Compare packed single-precision (32-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm512_mask_cmpge_epi16_mask` | Compare packed signed 16-bit integers in a and b for greater... | vpcmp | — |
| `_mm512_mask_cmpge_epi32_mask` | Compare packed signed 32-bit integers in a and b for greater... | vpcmp | — |
| `_mm512_mask_cmpge_epi64_mask` | Compare packed signed 64-bit integers in a and b for greater... | vpcmp | — |
| `_mm512_mask_cmpge_epi8_mask` | Compare packed signed 8-bit integers in a and b for greater-... | vpcmp | — |
| `_mm512_mask_cmpge_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for great... | vpcmp | — |
| `_mm512_mask_cmpge_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for great... | vpcmp | — |
| `_mm512_mask_cmpge_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for great... | vpcmp | — |
| `_mm512_mask_cmpge_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for greate... | vpcmp | — |
| `_mm512_mask_cmpgt_epi16_mask` | Compare packed signed 16-bit integers in a and b for greater... | vpcmp | 1/1, 1/1 |
| `_mm512_mask_cmpgt_epi32_mask` | Compare packed signed 32-bit integers in a and b for greater... | vpcmp | 1/1, 1/1 |
| `_mm512_mask_cmpgt_epi64_mask` | Compare packed signed 64-bit integers in a and b for greater... | vpcmp | 1/1, 1/1 |
| `_mm512_mask_cmpgt_epi8_mask` | Compare packed signed 8-bit integers in a and b for greater-... | vpcmp | 1/1, 1/1 |
| `_mm512_mask_cmpgt_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for great... | vpcmp | — |
| `_mm512_mask_cmpgt_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for great... | vpcmp | — |
| `_mm512_mask_cmpgt_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for great... | vpcmp | — |
| `_mm512_mask_cmpgt_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for greate... | vpcmp | — |
| `_mm512_mask_cmple_epi16_mask` | Compare packed signed 16-bit integers in a and b for less-th... | vpcmp | — |
| `_mm512_mask_cmple_epi32_mask` | Compare packed signed 32-bit integers in a and b for less-th... | vpcmp | — |
| `_mm512_mask_cmple_epi64_mask` | Compare packed signed 64-bit integers in a and b for less-th... | vpcmp | — |
| `_mm512_mask_cmple_epi8_mask` | Compare packed signed 8-bit integers in a and b for less-tha... | vpcmp | — |
| `_mm512_mask_cmple_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for less-... | vpcmp | — |
| `_mm512_mask_cmple_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for less-... | vpcmp | — |
| `_mm512_mask_cmple_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for less-... | vpcmp | — |
| `_mm512_mask_cmple_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for less-t... | vpcmp | — |
| `_mm512_mask_cmple_pd_mask` | Compare packed double-precision (64-bit) floating-point elem... | vcmp | — |
| `_mm512_mask_cmple_ps_mask` | Compare packed single-precision (32-bit) floating-point elem... | vcmp | — |
| `_mm512_mask_cmplt_epi16_mask` | Compare packed signed 16-bit integers in a and b for less-th... | vpcmp | 1/1, 1/1 |
| `_mm512_mask_cmplt_epi32_mask` | Compare packed signed 32-bit integers in a and b for less-th... | vpcmp | 1/1, 1/1 |
| `_mm512_mask_cmplt_epi64_mask` | Compare packed signed 64-bit integers in a and b for less-th... | vpcmp | 1/1, 1/1 |
| `_mm512_mask_cmplt_epi8_mask` | Compare packed signed 8-bit integers in a and b for less-tha... | vpcmp | 1/1, 1/1 |
| `_mm512_mask_cmplt_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for less-... | vpcmp | — |
| `_mm512_mask_cmplt_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for less-... | vpcmp | — |
| `_mm512_mask_cmplt_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for less-... | vpcmp | — |
| `_mm512_mask_cmplt_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for less-t... | vpcmp | — |
| `_mm512_mask_cmplt_pd_mask` | Compare packed double-precision (64-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm512_mask_cmplt_ps_mask` | Compare packed single-precision (32-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm512_mask_cmpneq_epi16_mask` | Compare packed signed 16-bit integers in a and b for not-equ... | vpcmp | — |
| `_mm512_mask_cmpneq_epi32_mask` | Compare packed 32-bit integers in a and b for not-equal, and... | vpcmp | — |
| `_mm512_mask_cmpneq_epi64_mask` | Compare packed signed 64-bit integers in a and b for not-equ... | vpcmp | — |
| `_mm512_mask_cmpneq_epi8_mask` | Compare packed signed 8-bit integers in a and b for not-equa... | vpcmp | — |
| `_mm512_mask_cmpneq_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for not-e... | vpcmp | — |
| `_mm512_mask_cmpneq_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for not-e... | vpcmp | — |
| `_mm512_mask_cmpneq_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for not-e... | vpcmp | — |
| `_mm512_mask_cmpneq_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for not-eq... | vpcmp | — |
| `_mm512_mask_cmpneq_pd_mask` | Compare packed double-precision (64-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm512_mask_cmpneq_ps_mask` | Compare packed single-precision (32-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm512_mask_cmpnle_pd_mask` | Compare packed double-precision (64-bit) floating-point elem... | vcmp | — |
| `_mm512_mask_cmpnle_ps_mask` | Compare packed single-precision (32-bit) floating-point elem... | vcmp | — |
| `_mm512_mask_cmpnlt_pd_mask` | Compare packed double-precision (64-bit) floating-point elem... | vcmp | — |
| `_mm512_mask_cmpnlt_ps_mask` | Compare packed single-precision (32-bit) floating-point elem... | vcmp | — |
| `_mm512_mask_cmpord_pd_mask` | Compare packed double-precision (64-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm512_mask_cmpord_ps_mask` | Compare packed single-precision (32-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm512_mask_cmpunord_pd_mask` | Compare packed double-precision (64-bit) floating-point elem... | vcmp | — |
| `_mm512_mask_cmpunord_ps_mask` | Compare packed single-precision (32-bit) floating-point elem... | vcmp | — |
| `_mm512_mask_compress_epi32` | Contiguously store the active 32-bit integers in a (those wi... | vpcompressd | — |
| `_mm512_mask_compress_epi64` | Contiguously store the active 64-bit integers in a (those wi... | vpcompressq | — |
| `_mm512_mask_compress_pd` | Contiguously store the active double-precision (64-bit) floa... | vcompresspd | — |
| `_mm512_mask_compress_ps` | Contiguously store the active single-precision (32-bit) floa... | vcompressps | — |
| `_mm512_mask_conflict_epi32` | Test each 32-bit element of a for equality with all other el... | vpconflictd | — |
| `_mm512_mask_conflict_epi64` | Test each 64-bit element of a for equality with all other el... | vpconflictq | — |
| `_mm512_mask_cvt_roundepi32_ps` | Convert packed signed 32-bit integers in a to packed single-... | vcvtdq2ps | 4/1, 3/1 |
| `_mm512_mask_cvt_roundepi64_pd` | Convert packed signed 64-bit integers in a to packed double-... | vcvtqq2pd | 4/1, 3/1 |
| `_mm512_mask_cvt_roundepi64_ps` | Convert packed signed 64-bit integers in a to packed single-... | vcvtqq2ps | 4/1, 3/1 |
| `_mm512_mask_cvt_roundepu32_ps` | Convert packed unsigned 32-bit integers in a to packed singl... | vcvtudq2ps | 4/1, 3/1 |
| `_mm512_mask_cvt_roundepu64_pd` | Convert packed unsigned 64-bit integers in a to packed doubl... | vcvtuqq2pd | 4/1, 3/1 |
| `_mm512_mask_cvt_roundepu64_ps` | Convert packed unsigned 64-bit integers in a to packed singl... | vcvtuqq2ps | 4/1, 3/1 |
| `_mm512_mask_cvt_roundpd_epi32` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2dq | 4/1, 3/1 |
| `_mm512_mask_cvt_roundpd_epi64` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2qq | 4/1, 3/1 |
| `_mm512_mask_cvt_roundpd_epu32` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2udq | 4/1, 3/1 |
| `_mm512_mask_cvt_roundpd_epu64` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2uqq | 4/1, 3/1 |
| `_mm512_mask_cvt_roundpd_ps` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2ps | 4/1, 3/1 |
| `_mm512_mask_cvt_roundph_ps` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2ps | 4/1, 3/1 |
| `_mm512_mask_cvt_roundps_epi32` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2dq | 4/1, 3/1 |
| `_mm512_mask_cvt_roundps_epi64` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2qq | 4/1, 3/1 |
| `_mm512_mask_cvt_roundps_epu32` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2udq | 4/1, 3/1 |
| `_mm512_mask_cvt_roundps_epu64` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2uqq | 4/1, 3/1 |
| `_mm512_mask_cvt_roundps_pd` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2pd | 4/1, 3/1 |
| `_mm512_mask_cvt_roundps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2ph | 4/1, 3/1 |
| `_mm512_mask_cvtepi16_epi32` | Sign extend packed 16-bit integers in a to packed 32-bit int... | vpmovsxwd | 4/1, 3/1 |
| `_mm512_mask_cvtepi16_epi64` | Sign extend packed 16-bit integers in a to packed 64-bit int... | vpmovsxwq | 4/1, 3/1 |
| `_mm512_mask_cvtepi16_epi8` | Convert packed 16-bit integers in a to packed 8-bit integers... | vpmovwb | 4/1, 3/1 |
| `_mm512_mask_cvtepi32_epi16` | Convert packed 32-bit integers in a to packed 16-bit integer... | vpmovdw | 4/1, 3/1 |
| `_mm512_mask_cvtepi32_epi64` | Sign extend packed 32-bit integers in a to packed 64-bit int... | vpmovsxdq | 4/1, 3/1 |
| `_mm512_mask_cvtepi32_epi8` | Convert packed 32-bit integers in a to packed 8-bit integers... | vpmovdb | 4/1, 3/1 |
| `_mm512_mask_cvtepi32_pd` | Convert packed signed 32-bit integers in a to packed double-... | vcvtdq2pd | 4/1, 3/1 |
| `_mm512_mask_cvtepi32_ps` | Convert packed signed 32-bit integers in a to packed single-... | vcvtdq2ps | 4/1, 3/1 |
| `_mm512_mask_cvtepi32lo_pd` | Performs element-by-element conversion of the lower half of ... | vcvtdq2pd | 4/1, 3/1 |
| `_mm512_mask_cvtepi64_epi16` | Convert packed 64-bit integers in a to packed 16-bit integer... | vpmovqw | 4/1, 3/1 |
| `_mm512_mask_cvtepi64_epi32` | Convert packed 64-bit integers in a to packed 32-bit integer... | vpmovqd | 4/1, 3/1 |
| `_mm512_mask_cvtepi64_epi8` | Convert packed 64-bit integers in a to packed 8-bit integers... | vpmovqb | 4/1, 3/1 |
| `_mm512_mask_cvtepi64_pd` | Convert packed signed 64-bit integers in a to packed double-... | vcvtqq2pd | 4/1, 3/1 |
| `_mm512_mask_cvtepi64_ps` | Convert packed signed 64-bit integers in a to packed single-... | vcvtqq2ps | 4/1, 3/1 |
| `_mm512_mask_cvtepi8_epi16` | Sign extend packed 8-bit integers in a to packed 16-bit inte... | vpmovsxbw | 4/1, 3/1 |
| `_mm512_mask_cvtepi8_epi32` | Sign extend packed 8-bit integers in a to packed 32-bit inte... | vpmovsxbd | 4/1, 3/1 |
| `_mm512_mask_cvtepi8_epi64` | Sign extend packed 8-bit integers in the low 8 bytes of a to... | vpmovsxbq | 4/1, 3/1 |
| `_mm512_mask_cvtepu16_epi32` | Zero extend packed unsigned 16-bit integers in a to packed 3... | vpmovzxwd | 4/1, 3/1 |
| `_mm512_mask_cvtepu16_epi64` | Zero extend packed unsigned 16-bit integers in a to packed 6... | vpmovzxwq | 4/1, 3/1 |
| `_mm512_mask_cvtepu32_epi64` | Zero extend packed unsigned 32-bit integers in a to packed 6... | vpmovzxdq | 4/1, 3/1 |
| `_mm512_mask_cvtepu32_pd` | Convert packed unsigned 32-bit integers in a to packed doubl... | vcvtudq2pd | 4/1, 3/1 |
| `_mm512_mask_cvtepu32_ps` | Convert packed unsigned 32-bit integers in a to packed singl... | vcvtudq2ps | 4/1, 3/1 |
| `_mm512_mask_cvtepu32lo_pd` | Performs element-by-element conversion of the lower half of ... | vcvtudq2pd | 4/1, 3/1 |
| `_mm512_mask_cvtepu64_pd` | Convert packed unsigned 64-bit integers in a to packed doubl... | vcvtuqq2pd | 4/1, 3/1 |
| `_mm512_mask_cvtepu64_ps` | Convert packed unsigned 64-bit integers in a to packed singl... | vcvtuqq2ps | 4/1, 3/1 |
| `_mm512_mask_cvtepu8_epi16` | Zero extend packed unsigned 8-bit integers in a to packed 16... | vpmovzxbw | 4/1, 3/1 |
| `_mm512_mask_cvtepu8_epi32` | Zero extend packed unsigned 8-bit integers in a to packed 32... | vpmovzxbd | 4/1, 3/1 |
| `_mm512_mask_cvtepu8_epi64` | Zero extend packed unsigned 8-bit integers in the low 8 byte... | vpmovzxbq | 4/1, 3/1 |
| `_mm512_mask_cvtpd_epi32` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2dq | 4/1, 3/1 |
| `_mm512_mask_cvtpd_epi64` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2qq | 4/1, 3/1 |
| `_mm512_mask_cvtpd_epu32` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2udq | 4/1, 3/1 |
| `_mm512_mask_cvtpd_epu64` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2uqq | 4/1, 3/1 |
| `_mm512_mask_cvtpd_ps` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2ps | 4/1, 3/1 |
| `_mm512_mask_cvtpd_pslo` | Performs an element-by-element conversion of packed double-p... | vcvtpd2ps | 4/1, 3/1 |
| `_mm512_mask_cvtph_ps` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2ps | 4/1, 3/1 |
| `_mm512_mask_cvtps_epi32` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2dq | 4/1, 3/1 |
| `_mm512_mask_cvtps_epi64` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2qq | 4/1, 3/1 |
| `_mm512_mask_cvtps_epu32` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2udq | 4/1, 3/1 |
| `_mm512_mask_cvtps_epu64` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2uqq | 4/1, 3/1 |
| `_mm512_mask_cvtps_pd` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2pd | 4/1, 3/1 |
| `_mm512_mask_cvtps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2ph | 4/1, 3/1 |
| `_mm512_mask_cvtpslo_pd` | Performs element-by-element conversion of the lower half of ... | vcvtps2pd | 4/1, 3/1 |
| `_mm512_mask_cvtsepi16_epi8` | Convert packed signed 16-bit integers in a to packed 8-bit i... | vpmovswb | 4/1, 3/1 |
| `_mm512_mask_cvtsepi32_epi16` | Convert packed signed 32-bit integers in a to packed 16-bit ... | vpmovsdw | 4/1, 3/1 |
| `_mm512_mask_cvtsepi32_epi8` | Convert packed signed 32-bit integers in a to packed 8-bit i... | vpmovsdb | 4/1, 3/1 |
| `_mm512_mask_cvtsepi64_epi16` | Convert packed signed 64-bit integers in a to packed 16-bit ... | vpmovsqw | 4/1, 3/1 |
| `_mm512_mask_cvtsepi64_epi32` | Convert packed signed 64-bit integers in a to packed 32-bit ... | vpmovsqd | 4/1, 3/1 |
| `_mm512_mask_cvtsepi64_epi8` | Convert packed signed 64-bit integers in a to packed 8-bit i... | vpmovsqb | 4/1, 3/1 |
| `_mm512_mask_cvtt_roundpd_epi32` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2dq | 4/1, 3/1 |
| `_mm512_mask_cvtt_roundpd_epi64` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2qq | 4/1, 3/1 |
| `_mm512_mask_cvtt_roundpd_epu32` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2udq | 4/1, 3/1 |
| `_mm512_mask_cvtt_roundpd_epu64` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2uqq | 4/1, 3/1 |
| `_mm512_mask_cvtt_roundps_epi32` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2dq | 4/1, 3/1 |
| `_mm512_mask_cvtt_roundps_epi64` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2qq | 4/1, 3/1 |
| `_mm512_mask_cvtt_roundps_epu32` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2udq | 4/1, 3/1 |
| `_mm512_mask_cvtt_roundps_epu64` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2uqq | 4/1, 3/1 |
| `_mm512_mask_cvttpd_epi32` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2dq | 4/1, 3/1 |
| `_mm512_mask_cvttpd_epi64` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2qq | 4/1, 3/1 |
| `_mm512_mask_cvttpd_epu32` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2udq | 4/1, 3/1 |
| `_mm512_mask_cvttpd_epu64` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2uqq | 4/1, 3/1 |
| `_mm512_mask_cvttps_epi32` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2dq | 4/1, 3/1 |
| `_mm512_mask_cvttps_epi64` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2qq | 4/1, 3/1 |
| `_mm512_mask_cvttps_epu32` | Convert packed double-precision (32-bit) floating-point elem... | vcvttps2udq | 4/1, 3/1 |
| `_mm512_mask_cvttps_epu64` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2uqq | 4/1, 3/1 |
| `_mm512_mask_cvtusepi16_epi8` | Convert packed unsigned 16-bit integers in a to packed unsig... | vpmovuswb | 4/1, 3/1 |
| `_mm512_mask_cvtusepi32_epi16` | Convert packed unsigned 32-bit integers in a to packed unsig... | vpmovusdw | 4/1, 3/1 |
| `_mm512_mask_cvtusepi32_epi8` | Convert packed unsigned 32-bit integers in a to packed unsig... | vpmovusdb | 4/1, 3/1 |
| `_mm512_mask_cvtusepi64_epi16` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqw | 4/1, 3/1 |
| `_mm512_mask_cvtusepi64_epi32` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqd | 4/1, 3/1 |
| `_mm512_mask_cvtusepi64_epi8` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqb | 4/1, 3/1 |
| `_mm512_mask_dbsad_epu8` | Compute the sum of absolute differences (SADs) of quadruplet... | vdbpsadbw | — |
| `_mm512_mask_div_pd` | Divide packed double-precision (64-bit) floating-point eleme... | vdivpd | 20/14, 13/7 |
| `_mm512_mask_div_ps` | Divide packed single-precision (32-bit) floating-point eleme... | vdivps | 13/7, 10/4 |
| `_mm512_mask_div_round_pd` | Divide packed double-precision (64-bit) floating-point eleme... | vdivpd | — |
| `_mm512_mask_div_round_ps` | Divide packed single-precision (32-bit) floating-point eleme... | vdivps | — |
| `_mm512_mask_expand_epi32` | Load contiguous active 32-bit integers from a (those with th... | vpexpandd | — |
| `_mm512_mask_expand_epi64` | Load contiguous active 64-bit integers from a (those with th... | vpexpandq | — |
| `_mm512_mask_expand_pd` | Load contiguous active double-precision (64-bit) floating-po... | vexpandpd | — |
| `_mm512_mask_expand_ps` | Load contiguous active single-precision (32-bit) floating-po... | vexpandps | — |
| `_mm512_mask_extractf32x4_ps` | Extract 128 bits (composed of 4 packed single-precision (32-... | vextractf32x4 | — |
| `_mm512_mask_extractf32x8_ps` | Extracts 256 bits (composed of 8 packed single-precision (32... | vextractf32x8 | — |
| `_mm512_mask_extractf64x2_pd` | Extracts 128 bits (composed of 2 packed double-precision (64... | vextractf64x2 | — |
| `_mm512_mask_extractf64x4_pd` | Extract 256 bits (composed of 4 packed double-precision (64-... | vextractf64x4 | — |
| `_mm512_mask_extracti32x4_epi32` | Extract 128 bits (composed of 4 packed 32-bit integers) from... | vextracti32x4 | — |
| `_mm512_mask_extracti32x8_epi32` | Extracts 256 bits (composed of 8 packed 32-bit integers) fro... | vextracti32x8 | — |
| `_mm512_mask_extracti64x2_epi64` | Extracts 128 bits (composed of 2 packed 64-bit integers) fro... | vextracti64x2 | — |
| `_mm512_mask_extracti64x4_epi64` | Extract 256 bits (composed of 4 packed 64-bit integers) from... | vextracti64x4 | — |
| `_mm512_mask_fixupimm_pd` | Fix up packed double-precision (64-bit) floating-point eleme... | vfixupimmpd | — |
| `_mm512_mask_fixupimm_ps` | Fix up packed single-precision (32-bit) floating-point eleme... | vfixupimmps | — |
| `_mm512_mask_fixupimm_round_pd` | Fix up packed double-precision (64-bit) floating-point eleme... | vfixupimmpd | — |
| `_mm512_mask_fixupimm_round_ps` | Fix up packed single-precision (32-bit) floating-point eleme... | vfixupimmps | — |
| `_mm512_mask_fmadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmadd | —/4/1 |
| `_mm512_mask_fmadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmadd | —/4/1 |
| `_mm512_mask_fmadd_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmadd | —/4/1 |
| `_mm512_mask_fmadd_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmadd | —/4/1 |
| `_mm512_mask_fmaddsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmaddsub | —/4/1 |
| `_mm512_mask_fmaddsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmaddsub | —/4/1 |
| `_mm512_mask_fmaddsub_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmaddsub | —/4/1 |
| `_mm512_mask_fmaddsub_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmaddsub | —/4/1 |
| `_mm512_mask_fmsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsub | —/4/1 |
| `_mm512_mask_fmsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsub | —/4/1 |
| `_mm512_mask_fmsub_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsub | —/4/1 |
| `_mm512_mask_fmsub_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsub | —/4/1 |
| `_mm512_mask_fmsubadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsubadd | —/4/1 |
| `_mm512_mask_fmsubadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsubadd | —/4/1 |
| `_mm512_mask_fmsubadd_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsubadd | —/4/1 |
| `_mm512_mask_fmsubadd_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsubadd | —/4/1 |
| `_mm512_mask_fnmadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmadd | —/4/1 |
| `_mm512_mask_fnmadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmadd | —/4/1 |
| `_mm512_mask_fnmadd_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmadd | —/4/1 |
| `_mm512_mask_fnmadd_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmadd | —/4/1 |
| `_mm512_mask_fnmsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmsub | —/4/1 |
| `_mm512_mask_fnmsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmsub | —/4/1 |
| `_mm512_mask_fnmsub_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmsub | —/4/1 |
| `_mm512_mask_fnmsub_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmsub | —/4/1 |
| `_mm512_mask_fpclass_pd_mask` | Test packed double-precision (64-bit) floating-point element... | vfpclasspd | — |
| `_mm512_mask_fpclass_ps_mask` | Test packed single-precision (32-bit) floating-point element... | vfpclassps | — |
| `_mm512_mask_getexp_pd` | Convert the exponent of each packed double-precision (64-bit... | vgetexppd | — |
| `_mm512_mask_getexp_ps` | Convert the exponent of each packed single-precision (32-bit... | vgetexpps | — |
| `_mm512_mask_getexp_round_pd` | Convert the exponent of each packed double-precision (64-bit... | vgetexppd | — |
| `_mm512_mask_getexp_round_ps` | Convert the exponent of each packed single-precision (32-bit... | vgetexpps | — |
| `_mm512_mask_getmant_pd` | Normalize the mantissas of packed double-precision (64-bit) ... | vgetmantpd | — |
| `_mm512_mask_getmant_ps` | Normalize the mantissas of packed single-precision (32-bit) ... | vgetmantps | — |
| `_mm512_mask_getmant_round_pd` | Normalize the mantissas of packed double-precision (64-bit) ... | vgetmantpd | — |
| `_mm512_mask_getmant_round_ps` | Normalize the mantissas of packed single-precision (32-bit) ... | vgetmantps | — |
| `_mm512_mask_insertf32x4` | Copy a to tmp, then insert 128 bits (composed of 4 packed si... | vinsertf32x4 | — |
| `_mm512_mask_insertf32x8` | Copy a to tmp, then insert 256 bits (composed of 8 packed si... | vinsertf32x8 | — |
| `_mm512_mask_insertf64x2` | Copy a to tmp, then insert 128 bits (composed of 2 packed do... | vinsertf64x2 | — |
| `_mm512_mask_insertf64x4` | Copy a to tmp, then insert 256 bits (composed of 4 packed do... | vinsertf64x4 | — |
| `_mm512_mask_inserti32x4` | Copy a to tmp, then insert 128 bits (composed of 4 packed 32... | vinserti32x4 | — |
| `_mm512_mask_inserti32x8` | Copy a to tmp, then insert 256 bits (composed of 8 packed 32... | vinserti32x8 | — |
| `_mm512_mask_inserti64x2` | Copy a to tmp, then insert 128 bits (composed of 2 packed 64... | vinserti64x2 | — |
| `_mm512_mask_inserti64x4` | Copy a to tmp, then insert 256 bits (composed of 4 packed 64... | vinserti64x4 | — |
| `_mm512_mask_lzcnt_epi32` | Counts the number of leading zero bits in each packed 32-bit... | vplzcntd | — |
| `_mm512_mask_lzcnt_epi64` | Counts the number of leading zero bits in each packed 64-bit... | vplzcntq | — |
| `_mm512_mask_madd_epi16` | Multiply packed signed 16-bit integers in a and b, producing... | vpmaddwd | — |
| `_mm512_mask_maddubs_epi16` | Multiply packed unsigned 8-bit integers in a by packed signe... | vpmaddubsw | — |
| `_mm512_mask_max_epi16` | Compare packed signed 16-bit integers in a and b, and store ... | vpmaxsw | — |
| `_mm512_mask_max_epi32` | Compare packed signed 32-bit integers in a and b, and store ... | vpmaxsd | — |
| `_mm512_mask_max_epi64` | Compare packed signed 64-bit integers in a and b, and store ... | vpmaxsq | — |
| `_mm512_mask_max_epi8` | Compare packed signed 8-bit integers in a and b, and store p... | vpmaxsb | — |
| `_mm512_mask_max_epu16` | Compare packed unsigned 16-bit integers in a and b, and stor... | vpmaxuw | — |
| `_mm512_mask_max_epu32` | Compare packed unsigned 32-bit integers in a and b, and stor... | vpmaxud | — |
| `_mm512_mask_max_epu64` | Compare packed unsigned 64-bit integers in a and b, and stor... | vpmaxuq | — |
| `_mm512_mask_max_epu8` | Compare packed unsigned 8-bit integers in a and b, and store... | vpmaxub | — |
| `_mm512_mask_max_pd` | Compare packed double-precision (64-bit) floating-point elem... | vmaxpd | — |
| `_mm512_mask_max_ps` | Compare packed single-precision (32-bit) floating-point elem... | vmaxps | — |
| `_mm512_mask_max_round_pd` | Compare packed double-precision (64-bit) floating-point elem... | vmaxpd | — |
| `_mm512_mask_max_round_ps` | Compare packed single-precision (32-bit) floating-point elem... | vmaxps | — |
| `_mm512_mask_min_epi16` | Compare packed signed 16-bit integers in a and b, and store ... | vpminsw | — |
| `_mm512_mask_min_epi32` | Compare packed signed 32-bit integers in a and b, and store ... | vpminsd | — |
| `_mm512_mask_min_epi64` | Compare packed signed 64-bit integers in a and b, and store ... | vpminsq | — |
| `_mm512_mask_min_epi8` | Compare packed signed 8-bit integers in a and b, and store p... | vpminsb | — |
| `_mm512_mask_min_epu16` | Compare packed unsigned 16-bit integers in a and b, and stor... | vpminuw | — |
| `_mm512_mask_min_epu32` | Compare packed unsigned 32-bit integers in a and b, and stor... | vpminud | — |
| `_mm512_mask_min_epu64` | Compare packed unsigned 64-bit integers in a and b, and stor... | vpminuq | — |
| `_mm512_mask_min_epu8` | Compare packed unsigned 8-bit integers in a and b, and store... | vpminub | — |
| `_mm512_mask_min_pd` | Compare packed double-precision (64-bit) floating-point elem... | vminpd | — |
| `_mm512_mask_min_ps` | Compare packed single-precision (32-bit) floating-point elem... | vminps | — |
| `_mm512_mask_min_round_pd` | Compare packed double-precision (64-bit) floating-point elem... | vminpd | — |
| `_mm512_mask_min_round_ps` | Compare packed single-precision (32-bit) floating-point elem... | vminps | — |
| `_mm512_mask_mov_epi16` | Move packed 16-bit integers from a into dst using writemask ... | vmovdqu16 | — |
| `_mm512_mask_mov_epi32` | Move packed 32-bit integers from a to dst using writemask k ... | vmovdqa32 | — |
| `_mm512_mask_mov_epi64` | Move packed 64-bit integers from a to dst using writemask k ... | vmovdqa64 | — |
| `_mm512_mask_mov_epi8` | Move packed 8-bit integers from a into dst using writemask k... | vmovdqu8 | — |
| `_mm512_mask_mov_pd` | Move packed double-precision (64-bit) floating-point element... | vmovapd | — |
| `_mm512_mask_mov_ps` | Move packed single-precision (32-bit) floating-point element... | vmovaps | — |
| `_mm512_mask_movedup_pd` | Duplicate even-indexed double-precision (64-bit) floating-po... | vmovddup | — |
| `_mm512_mask_movehdup_ps` | Duplicate odd-indexed single-precision (32-bit) floating-poi... | vmovshdup | — |
| `_mm512_mask_moveldup_ps` | Duplicate even-indexed single-precision (32-bit) floating-po... | vmovsldup | — |
| `_mm512_mask_mul_epi32` | Multiply the low signed 32-bit integers from each packed 64-... | vpmuldq | 10/2, 4/1 |
| `_mm512_mask_mul_epu32` | Multiply the low unsigned 32-bit integers from each packed 6... | vpmuludq | 10/2, 4/1 |
| `_mm512_mask_mul_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vmulpd | —/3/1 |
| `_mm512_mask_mul_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vmulps | —/3/1 |
| `_mm512_mask_mul_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vmulpd | — |
| `_mm512_mask_mul_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vmulps | — |
| `_mm512_mask_mulhi_epi16` | Multiply the packed signed 16-bit integers in a and b, produ... | vpmulhw | 5/1, 3/1 |
| `_mm512_mask_mulhi_epu16` | Multiply the packed unsigned 16-bit integers in a and b, pro... | vpmulhuw | 5/1, 3/1 |
| `_mm512_mask_mulhrs_epi16` | Multiply packed signed 16-bit integers in a and b, producing... | vpmulhrsw | — |
| `_mm512_mask_mullo_epi16` | Multiply the packed 16-bit integers in a and b, producing in... | vpmullw | 5/1, 3/1 |
| `_mm512_mask_mullo_epi32` | Multiply the packed 32-bit integers in a and b, producing in... | vpmulld | 10/2, 4/1 |
| `_mm512_mask_mullo_epi64` | Multiply packed 64-bit integers in `a` and `b`, producing in... | vpmullq | — |
| `_mm512_mask_mullox_epi64` | Multiplies elements in packed 64-bit integer vectors a and b... |  | — |
| `_mm512_mask_or_epi32` | Compute the bitwise OR of packed 32-bit integers in a and b,... | vpord | —/1/1 |
| `_mm512_mask_or_epi64` | Compute the bitwise OR of packed 64-bit integers in a and b,... | vporq | —/1/1 |
| `_mm512_mask_or_pd` | Compute the bitwise OR of packed double-precision (64-bit) f... | vorpd | —/1/1 |
| `_mm512_mask_or_ps` | Compute the bitwise OR of packed single-precision (32-bit) f... | vorps | —/1/1 |
| `_mm512_mask_packs_epi16` | Convert packed signed 16-bit integers from a and b to packed... | vpacksswb | 1/1, 1/1 |
| `_mm512_mask_packs_epi32` | Convert packed signed 32-bit integers from a and b to packed... | vpackssdw | 1/1, 1/1 |
| `_mm512_mask_packus_epi16` | Convert packed signed 16-bit integers from a and b to packed... | vpackuswb | 1/1, 1/1 |
| `_mm512_mask_packus_epi32` | Convert packed signed 32-bit integers from a and b to packed... | vpackusdw | 1/1, 1/1 |
| `_mm512_mask_permute_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vshufpd | —/2/1 |
| `_mm512_mask_permute_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vshufps | —/2/1 |
| `_mm512_mask_permutevar_epi32` | Shuffle 32-bit integers in a across lanes using the correspo... | vpermd | —/2/1 |
| `_mm512_mask_permutevar_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vpermilpd | —/2/1 |
| `_mm512_mask_permutevar_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vpermilps | —/2/1 |
| `_mm512_mask_permutex2var_epi16` | Shuffle 16-bit integers in a and b across lanes using the co... | vpermt2w | —/2/1 |
| `_mm512_mask_permutex2var_epi32` | Shuffle 32-bit integers in a and b across lanes using the co... | vpermt2d | —/2/1 |
| `_mm512_mask_permutex2var_epi64` | Shuffle 64-bit integers in a and b across lanes using the co... | vpermt2q | —/2/1 |
| `_mm512_mask_permutex2var_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vpermt2pd | —/2/1 |
| `_mm512_mask_permutex2var_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vpermt2ps | —/2/1 |
| `_mm512_mask_permutex_epi64` | Shuffle 64-bit integers in a within 256-bit lanes using the ... | vperm | —/2/1 |
| `_mm512_mask_permutex_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vperm | —/2/1 |
| `_mm512_mask_permutexvar_epi16` | Shuffle 16-bit integers in a across lanes using the correspo... | vpermw | —/2/1 |
| `_mm512_mask_permutexvar_epi32` | Shuffle 32-bit integers in a across lanes using the correspo... | vpermd | —/2/1 |
| `_mm512_mask_permutexvar_epi64` | Shuffle 64-bit integers in a across lanes using the correspo... | vpermq | —/2/1 |
| `_mm512_mask_permutexvar_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vpermpd | —/2/1 |
| `_mm512_mask_permutexvar_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vpermps | —/2/1 |
| `_mm512_mask_range_pd` | Calculate the max, min, absolute max, or absolute min (depen... | vrangepd | — |
| `_mm512_mask_range_ps` | Calculate the max, min, absolute max, or absolute min (depen... | vrangeps | — |
| `_mm512_mask_range_round_pd` | Calculate the max, min, absolute max, or absolute min (depen... | vrangepd | — |
| `_mm512_mask_range_round_ps` | Calculate the max, min, absolute max, or absolute min (depen... | vrangeps | — |
| `_mm512_mask_rcp14_pd` | Compute the approximate reciprocal of packed double-precisio... | vrcp14pd | — |
| `_mm512_mask_rcp14_ps` | Compute the approximate reciprocal of packed single-precisio... | vrcp14ps | — |
| `_mm512_mask_reduce_add_epi32` | Reduce the packed 32-bit integers in a by addition using mas... |  | —/1/1 |
| `_mm512_mask_reduce_add_epi64` | Reduce the packed 64-bit integers in a by addition using mas... |  | —/1/1 |
| `_mm512_mask_reduce_add_pd` | Reduce the packed double-precision (64-bit) floating-point e... |  | —/3/1 |
| `_mm512_mask_reduce_add_ps` | Reduce the packed single-precision (32-bit) floating-point e... |  | —/3/1 |
| `_mm512_mask_reduce_and_epi32` | Reduce the packed 32-bit integers in a by bitwise AND using ... |  | —/1/1 |
| `_mm512_mask_reduce_and_epi64` | Reduce the packed 64-bit integers in a by addition using mas... |  | —/1/1 |
| `_mm512_mask_reduce_max_epi32` | Reduce the packed signed 32-bit integers in a by maximum usi... |  | — |
| `_mm512_mask_reduce_max_epi64` | Reduce the packed signed 64-bit integers in a by maximum usi... |  | — |
| `_mm512_mask_reduce_max_epu32` | Reduce the packed unsigned 32-bit integers in a by maximum u... |  | — |
| `_mm512_mask_reduce_max_epu64` | Reduce the packed unsigned 64-bit integers in a by maximum u... |  | — |
| `_mm512_mask_reduce_max_pd` | Reduce the packed double-precision (64-bit) floating-point e... |  | — |
| `_mm512_mask_reduce_max_ps` | Reduce the packed single-precision (32-bit) floating-point e... |  | — |
| `_mm512_mask_reduce_min_epi32` | Reduce the packed signed 32-bit integers in a by maximum usi... |  | — |
| `_mm512_mask_reduce_min_epi64` | Reduce the packed signed 64-bit integers in a by maximum usi... |  | — |
| `_mm512_mask_reduce_min_epu32` | Reduce the packed unsigned 32-bit integers in a by maximum u... |  | — |
| `_mm512_mask_reduce_min_epu64` | Reduce the packed signed 64-bit integers in a by maximum usi... |  | — |
| `_mm512_mask_reduce_min_pd` | Reduce the packed double-precision (64-bit) floating-point e... |  | — |
| `_mm512_mask_reduce_min_ps` | Reduce the packed single-precision (32-bit) floating-point e... |  | — |
| `_mm512_mask_reduce_mul_epi32` | Reduce the packed 32-bit integers in a by multiplication usi... |  | 10/2, 4/1 |
| `_mm512_mask_reduce_mul_epi64` | Reduce the packed 64-bit integers in a by multiplication usi... |  | — |
| `_mm512_mask_reduce_mul_pd` | Reduce the packed double-precision (64-bit) floating-point e... |  | —/3/1 |
| `_mm512_mask_reduce_mul_ps` | Reduce the packed single-precision (32-bit) floating-point e... |  | —/3/1 |
| `_mm512_mask_reduce_or_epi32` | Reduce the packed 32-bit integers in a by bitwise OR using m... |  | —/1/1 |
| `_mm512_mask_reduce_or_epi64` | Reduce the packed 64-bit integers in a by bitwise OR using m... |  | —/1/1 |
| `_mm512_mask_reduce_pd` | Extract the reduced argument of packed double-precision (64-... | vreducepd | — |
| `_mm512_mask_reduce_ps` | Extract the reduced argument of packed single-precision (32-... | vreduceps | — |
| `_mm512_mask_reduce_round_pd` | Extract the reduced argument of packed double-precision (64-... | vreducepd | — |
| `_mm512_mask_reduce_round_ps` | Extract the reduced argument of packed single-precision (32-... | vreduceps | — |
| `_mm512_mask_rol_epi32` | Rotate the bits in each packed 32-bit integer in a to the le... | vprold | — |
| `_mm512_mask_rol_epi64` | Rotate the bits in each packed 64-bit integer in a to the le... | vprolq | — |
| `_mm512_mask_rolv_epi32` | Rotate the bits in each packed 32-bit integer in a to the le... | vprolvd | — |
| `_mm512_mask_rolv_epi64` | Rotate the bits in each packed 64-bit integer in a to the le... | vprolvq | — |
| `_mm512_mask_ror_epi32` | Rotate the bits in each packed 32-bit integer in a to the ri... | vprold | — |
| `_mm512_mask_ror_epi64` | Rotate the bits in each packed 64-bit integer in a to the ri... | vprolq | — |
| `_mm512_mask_rorv_epi32` | Rotate the bits in each packed 32-bit integer in a to the ri... | vprorvd | — |
| `_mm512_mask_rorv_epi64` | Rotate the bits in each packed 64-bit integer in a to the ri... | vprorvq | — |
| `_mm512_mask_roundscale_pd` | Round packed double-precision (64-bit) floating-point elemen... | vrndscalepd | — |
| `_mm512_mask_roundscale_ps` | Round packed single-precision (32-bit) floating-point elemen... | vrndscaleps | — |
| `_mm512_mask_roundscale_round_pd` | Round packed double-precision (64-bit) floating-point elemen... | vrndscalepd | — |
| `_mm512_mask_roundscale_round_ps` | Round packed single-precision (32-bit) floating-point elemen... | vrndscaleps | — |
| `_mm512_mask_rsqrt14_pd` | Compute the approximate reciprocal square root of packed dou... | vrsqrt14pd | — |
| `_mm512_mask_rsqrt14_ps` | Compute the approximate reciprocal square root of packed sin... | vrsqrt14ps | — |
| `_mm512_mask_scalef_pd` | Scale the packed double-precision (64-bit) floating-point el... | vscalefpd | — |
| `_mm512_mask_scalef_ps` | Scale the packed single-precision (32-bit) floating-point el... | vscalefps | — |
| `_mm512_mask_scalef_round_pd` | Scale the packed double-precision (64-bit) floating-point el... | vscalefpd | — |
| `_mm512_mask_scalef_round_ps` | Scale the packed single-precision (32-bit) floating-point el... | vscalefps | — |
| `_mm512_mask_set1_epi16` | Broadcast 16-bit integer a to all elements of dst using writ... | vpbroadcastw | — |
| `_mm512_mask_set1_epi32` | Broadcast 32-bit integer a to all elements of dst using writ... | vpbroadcastd | — |
| `_mm512_mask_set1_epi64` | Broadcast 64-bit integer a to all elements of dst using writ... | vpbroadcastq | — |
| `_mm512_mask_set1_epi8` | Broadcast 8-bit integer a to all elements of dst using write... | vpbroadcast | — |
| `_mm512_mask_shuffle_epi32` | Shuffle 32-bit integers in a within 128-bit lanes using the ... | vpshufd | —/1/1 |
| `_mm512_mask_shuffle_epi8` | Shuffle 8-bit integers in a within 128-bit lanes using the c... | vpshufb | —/1/1 |
| `_mm512_mask_shuffle_f32x4` | Shuffle 128-bits (composed of 4 single-precision (32-bit) fl... | vshuff32x4 | —/1/1 |
| `_mm512_mask_shuffle_f64x2` | Shuffle 128-bits (composed of 2 double-precision (64-bit) fl... | vshuff64x2 | —/1/1 |
| `_mm512_mask_shuffle_i32x4` | Shuffle 128-bits (composed of 4 32-bit integers) selected by... | vshufi32x4 | —/1/1 |
| `_mm512_mask_shuffle_i64x2` | Shuffle 128-bits (composed of 2 64-bit integers) selected by... | vshufi64x2 | —/1/1 |
| `_mm512_mask_shuffle_pd` | Shuffle double-precision (64-bit) floating-point elements wi... | vshufpd | —/1/1 |
| `_mm512_mask_shuffle_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vshufps | —/1/1 |
| `_mm512_mask_shufflehi_epi16` | Shuffle 16-bit integers in the high 64 bits of 128-bit lanes... | vpshufhw | —/1/1 |
| `_mm512_mask_shufflelo_epi16` | Shuffle 16-bit integers in the low 64 bits of 128-bit lanes ... | vpshuflw | —/1/1 |
| `_mm512_mask_sll_epi16` | Shift packed 16-bit integers in a left by count while shifti... | vpsllw | —/1/1 |
| `_mm512_mask_sll_epi32` | Shift packed 32-bit integers in a left by count while shifti... | vpslld | —/1/1 |
| `_mm512_mask_sll_epi64` | Shift packed 64-bit integers in a left by count while shifti... | vpsllq | —/1/1 |
| `_mm512_mask_slli_epi16` | Shift packed 16-bit integers in a left by imm8 while shiftin... | vpsllw | —/1/1 |
| `_mm512_mask_slli_epi32` | Shift packed 32-bit integers in a left by imm8 while shiftin... | vpslld | —/1/1 |
| `_mm512_mask_slli_epi64` | Shift packed 64-bit integers in a left by imm8 while shiftin... | vpsllq | —/1/1 |
| `_mm512_mask_sllv_epi16` | Shift packed 16-bit integers in a left by the amount specifi... | vpsllvw | —/1/1 |
| `_mm512_mask_sllv_epi32` | Shift packed 32-bit integers in a left by the amount specifi... | vpsllvd | —/1/1 |
| `_mm512_mask_sllv_epi64` | Shift packed 64-bit integers in a left by the amount specifi... | vpsllvq | —/1/1 |
| `_mm512_mask_sqrt_pd` | Compute the square root of packed double-precision (64-bit) ... | vsqrtpd | 16/14, 20/9 |
| `_mm512_mask_sqrt_ps` | Compute the square root of packed single-precision (32-bit) ... | vsqrtps | 11/7, 14/5 |
| `_mm512_mask_sqrt_round_pd` | Compute the square root of packed double-precision (64-bit) ... | vsqrtpd | — |
| `_mm512_mask_sqrt_round_ps` | Compute the square root of packed single-precision (32-bit) ... | vsqrtps | — |
| `_mm512_mask_sra_epi16` | Shift packed 16-bit integers in a right by count while shift... | vpsraw | —/1/1 |
| `_mm512_mask_sra_epi32` | Shift packed 32-bit integers in a right by count while shift... | vpsrad | —/1/1 |
| `_mm512_mask_sra_epi64` | Shift packed 64-bit integers in a right by count while shift... | vpsraq | —/1/1 |
| `_mm512_mask_srai_epi16` | Shift packed 16-bit integers in a right by imm8 while shifti... | vpsraw | —/1/1 |
| `_mm512_mask_srai_epi32` | Shift packed 32-bit integers in a right by imm8 while shifti... | vpsrad | —/1/1 |
| `_mm512_mask_srai_epi64` | Shift packed 64-bit integers in a right by imm8 while shifti... | vpsraq | —/1/1 |
| `_mm512_mask_srav_epi16` | Shift packed 16-bit integers in a right by the amount specif... | vpsravw | —/1/1 |
| `_mm512_mask_srav_epi32` | Shift packed 32-bit integers in a right by the amount specif... | vpsravd | —/1/1 |
| `_mm512_mask_srav_epi64` | Shift packed 64-bit integers in a right by the amount specif... | vpsravq | —/1/1 |
| `_mm512_mask_srl_epi16` | Shift packed 16-bit integers in a right by count while shift... | vpsrlw | —/1/1 |
| `_mm512_mask_srl_epi32` | Shift packed 32-bit integers in a right by count while shift... | vpsrld | —/1/1 |
| `_mm512_mask_srl_epi64` | Shift packed 64-bit integers in a right by count while shift... | vpsrlq | —/1/1 |
| `_mm512_mask_srli_epi16` | Shift packed 16-bit integers in a right by imm8 while shifti... | vpsrlw | —/1/1 |
| `_mm512_mask_srli_epi32` | Shift packed 32-bit integers in a right by imm8 while shifti... | vpsrld | —/1/1 |
| `_mm512_mask_srli_epi64` | Shift packed 64-bit integers in a right by imm8 while shifti... | vpsrlq | —/1/1 |
| `_mm512_mask_srlv_epi16` | Shift packed 16-bit integers in a right by the amount specif... | vpsrlvw | —/1/1 |
| `_mm512_mask_srlv_epi32` | Shift packed 32-bit integers in a right by the amount specif... | vpsrlvd | —/1/1 |
| `_mm512_mask_srlv_epi64` | Shift packed 64-bit integers in a right by the amount specif... | vpsrlvq | —/1/1 |
| `_mm512_mask_sub_epi16` | Subtract packed 16-bit integers in b from packed 16-bit inte... | vpsubw | —/1/1 |
| `_mm512_mask_sub_epi32` | Subtract packed 32-bit integers in b from packed 32-bit inte... | vpsubd | —/1/1 |
| `_mm512_mask_sub_epi64` | Subtract packed 64-bit integers in b from packed 64-bit inte... | vpsubq | —/1/1 |
| `_mm512_mask_sub_epi8` | Subtract packed 8-bit integers in b from packed 8-bit intege... | vpsubb | —/1/1 |
| `_mm512_mask_sub_pd` | Subtract packed double-precision (64-bit) floating-point ele... | vsubpd | — |
| `_mm512_mask_sub_ps` | Subtract packed single-precision (32-bit) floating-point ele... | vsubps | — |
| `_mm512_mask_sub_round_pd` | Subtract packed double-precision (64-bit) floating-point ele... | vsubpd | — |
| `_mm512_mask_sub_round_ps` | Subtract packed single-precision (32-bit) floating-point ele... | vsubps | — |
| `_mm512_mask_subs_epi16` | Subtract packed signed 16-bit integers in b from packed 16-b... | vpsubsw | —/1/1 |
| `_mm512_mask_subs_epi8` | Subtract packed signed 8-bit integers in b from packed 8-bit... | vpsubsb | —/1/1 |
| `_mm512_mask_subs_epu16` | Subtract packed unsigned 16-bit integers in b from packed un... | vpsubusw | — |
| `_mm512_mask_subs_epu8` | Subtract packed unsigned 8-bit integers in b from packed uns... | vpsubusb | — |
| `_mm512_mask_ternarylogic_epi32` | Bitwise ternary logic that provides the capability to implem... | vpternlogd | — |
| `_mm512_mask_ternarylogic_epi64` | Bitwise ternary logic that provides the capability to implem... | vpternlogq | — |
| `_mm512_mask_test_epi16_mask` | Compute the bitwise AND of packed 16-bit integers in a and b... | vptestmw | — |
| `_mm512_mask_test_epi32_mask` | Compute the bitwise AND of packed 32-bit integers in a and b... | vptestmd | — |
| `_mm512_mask_test_epi64_mask` | Compute the bitwise AND of packed 64-bit integers in a and b... | vptestmq | — |
| `_mm512_mask_test_epi8_mask` | Compute the bitwise AND of packed 8-bit integers in a and b,... | vptestmb | — |
| `_mm512_mask_testn_epi16_mask` | Compute the bitwise NAND of packed 16-bit integers in a and ... | vptestnmw | — |
| `_mm512_mask_testn_epi32_mask` | Compute the bitwise NAND of packed 32-bit integers in a and ... | vptestnmd | — |
| `_mm512_mask_testn_epi64_mask` | Compute the bitwise NAND of packed 64-bit integers in a and ... | vptestnmq | — |
| `_mm512_mask_testn_epi8_mask` | Compute the bitwise NAND of packed 8-bit integers in a and b... | vptestnmb | — |
| `_mm512_mask_unpackhi_epi16` | Unpack and interleave 16-bit integers from the high half of ... | vpunpckhwd | 1/1, 1/1 |
| `_mm512_mask_unpackhi_epi32` | Unpack and interleave 32-bit integers from the high half of ... | vpunpckhdq | 1/1, 1/1 |
| `_mm512_mask_unpackhi_epi64` | Unpack and interleave 64-bit integers from the high half of ... | vpunpckhqdq | 1/1, 1/1 |
| `_mm512_mask_unpackhi_epi8` | Unpack and interleave 8-bit integers from the high half of e... | vpunpckhbw | 1/1, 1/1 |
| `_mm512_mask_unpackhi_pd` | Unpack and interleave double-precision (64-bit) floating-poi... | vunpckhpd | 1/1, 1/1 |
| `_mm512_mask_unpackhi_ps` | Unpack and interleave single-precision (32-bit) floating-poi... | vunpckhps | 1/1, 1/1 |
| `_mm512_mask_unpacklo_epi16` | Unpack and interleave 16-bit integers from the low half of e... | vpunpcklwd | 1/1, 1/1 |
| `_mm512_mask_unpacklo_epi32` | Unpack and interleave 32-bit integers from the low half of e... | vpunpckldq | 1/1, 1/1 |
| `_mm512_mask_unpacklo_epi64` | Unpack and interleave 64-bit integers from the low half of e... | vpunpcklqdq | 1/1, 1/1 |
| `_mm512_mask_unpacklo_epi8` | Unpack and interleave 8-bit integers from the low half of ea... | vpunpcklbw | 1/1, 1/1 |
| `_mm512_mask_unpacklo_pd` | Unpack and interleave double-precision (64-bit) floating-poi... | vunpcklpd | 1/1, 1/1 |
| `_mm512_mask_unpacklo_ps` | Unpack and interleave single-precision (32-bit) floating-poi... | vunpcklps | 1/1, 1/1 |
| `_mm512_mask_xor_epi32` | Compute the bitwise XOR of packed 32-bit integers in a and b... | vpxord | —/1/1 |
| `_mm512_mask_xor_epi64` | Compute the bitwise XOR of packed 64-bit integers in a and b... | vpxorq | —/1/1 |
| `_mm512_mask_xor_pd` | Compute the bitwise XOR of packed double-precision (64-bit) ... | vxorpd | —/1/1 |
| `_mm512_mask_xor_ps` | Compute the bitwise XOR of packed single-precision (32-bit) ... | vxorps | —/1/1 |
| `_mm512_maskz_abs_epi16` | Compute the absolute value of packed signed 16-bit integers ... | vpabsw | — |
| `_mm512_maskz_abs_epi32` | Computes the absolute value of packed 32-bit integers in `a`... | vpabsd | — |
| `_mm512_maskz_abs_epi64` | Compute the absolute value of packed signed 64-bit integers ... | vpabsq | — |
| `_mm512_maskz_abs_epi8` | Compute the absolute value of packed signed 8-bit integers i... | vpabsb | — |
| `_mm512_maskz_add_epi16` | Add packed 16-bit integers in a and b, and store the results... | vpaddw | —/1/1 |
| `_mm512_maskz_add_epi32` | Add packed 32-bit integers in a and b, and store the results... | vpaddd | —/1/1 |
| `_mm512_maskz_add_epi64` | Add packed 64-bit integers in a and b, and store the results... | vpaddq | —/1/1 |
| `_mm512_maskz_add_epi8` | Add packed 8-bit integers in a and b, and store the results ... | vpaddb | —/1/1 |
| `_mm512_maskz_add_pd` | Add packed double-precision (64-bit) floating-point elements... | vaddpd | —/3/1 |
| `_mm512_maskz_add_ps` | Add packed single-precision (32-bit) floating-point elements... | vaddps | —/3/1 |
| `_mm512_maskz_add_round_pd` | Add packed double-precision (64-bit) floating-point elements... | vaddpd | — |
| `_mm512_maskz_add_round_ps` | Add packed single-precision (32-bit) floating-point elements... | vaddps | — |
| `_mm512_maskz_adds_epi16` | Add packed signed 16-bit integers in a and b using saturatio... | vpaddsw | —/1/1 |
| `_mm512_maskz_adds_epi8` | Add packed signed 8-bit integers in a and b using saturation... | vpaddsb | —/1/1 |
| `_mm512_maskz_adds_epu16` | Add packed unsigned 16-bit integers in a and b using saturat... | vpaddusw | — |
| `_mm512_maskz_adds_epu8` | Add packed unsigned 8-bit integers in a and b using saturati... | vpaddusb | — |
| `_mm512_maskz_alignr_epi32` | Concatenate a and b into a 128-byte immediate result, shift ... | valignd | — |
| `_mm512_maskz_alignr_epi64` | Concatenate a and b into a 128-byte immediate result, shift ... | valignq | — |
| `_mm512_maskz_alignr_epi8` | Concatenate pairs of 16-byte blocks in a and b into a 32-byt... | vpalignr | — |
| `_mm512_maskz_and_epi32` | Compute the bitwise AND of packed 32-bit integers in a and b... | vpandd | —/1/1 |
| `_mm512_maskz_and_epi64` | Compute the bitwise AND of packed 64-bit integers in a and b... | vpandq | —/1/1 |
| `_mm512_maskz_and_pd` | Compute the bitwise AND of packed double-precision (64-bit) ... | vandpd | —/1/1 |
| `_mm512_maskz_and_ps` | Compute the bitwise AND of packed single-precision (32-bit) ... | vandps | —/1/1 |
| `_mm512_maskz_andnot_epi32` | Compute the bitwise NOT of packed 32-bit integers in a and t... | vpandnd | —/1/1 |
| `_mm512_maskz_andnot_epi64` | Compute the bitwise NOT of packed 64-bit integers in a and t... | vpandnq | —/1/1 |
| `_mm512_maskz_andnot_pd` | Compute the bitwise NOT of packed double-precision (64-bit) ... | vandnpd | —/1/1 |
| `_mm512_maskz_andnot_ps` | Compute the bitwise NOT of packed single-precision (32-bit) ... | vandnps | —/1/1 |
| `_mm512_maskz_avg_epu16` | Average packed unsigned 16-bit integers in a and b, and stor... | vpavgw | — |
| `_mm512_maskz_avg_epu8` | Average packed unsigned 8-bit integers in a and b, and store... | vpavgb | — |
| `_mm512_maskz_broadcast_f32x2` | Broadcasts the lower 2 packed single-precision (32-bit) floa... | vbroadcastf32x2 | — |
| `_mm512_maskz_broadcast_f32x4` | Broadcast the 4 packed single-precision (32-bit) floating-po... |  | — |
| `_mm512_maskz_broadcast_f32x8` | Broadcasts the 8 packed single-precision (32-bit) floating-p... |  | — |
| `_mm512_maskz_broadcast_f64x2` | Broadcasts the 2 packed double-precision (64-bit) floating-p... |  | — |
| `_mm512_maskz_broadcast_f64x4` | Broadcast the 4 packed double-precision (64-bit) floating-po... |  | — |
| `_mm512_maskz_broadcast_i32x2` | Broadcasts the lower 2 packed 32-bit integers from a to all ... | vbroadcasti32x2 | — |
| `_mm512_maskz_broadcast_i32x4` | Broadcast the 4 packed 32-bit integers from a to all element... |  | — |
| `_mm512_maskz_broadcast_i32x8` | Broadcasts the 8 packed 32-bit integers from a to all elemen... |  | — |
| `_mm512_maskz_broadcast_i64x2` | Broadcasts the 2 packed 64-bit integers from a to all elemen... |  | — |
| `_mm512_maskz_broadcast_i64x4` | Broadcast the 4 packed 64-bit integers from a to all element... |  | — |
| `_mm512_maskz_broadcastb_epi8` | Broadcast the low packed 8-bit integer from a to all element... | vpbroadcastb | — |
| `_mm512_maskz_broadcastd_epi32` | Broadcast the low packed 32-bit integer from a to all elemen... | vpbroadcast | — |
| `_mm512_maskz_broadcastq_epi64` | Broadcast the low packed 64-bit integer from a to all elemen... | vpbroadcast | — |
| `_mm512_maskz_broadcastsd_pd` | Broadcast the low double-precision (64-bit) floating-point e... | vbroadcastsd | — |
| `_mm512_maskz_broadcastss_ps` | Broadcast the low single-precision (32-bit) floating-point e... | vbroadcastss | — |
| `_mm512_maskz_broadcastw_epi16` | Broadcast the low packed 16-bit integer from a to all elemen... | vpbroadcastw | — |
| `_mm512_maskz_compress_epi32` | Contiguously store the active 32-bit integers in a (those wi... | vpcompressd | — |
| `_mm512_maskz_compress_epi64` | Contiguously store the active 64-bit integers in a (those wi... | vpcompressq | — |
| `_mm512_maskz_compress_pd` | Contiguously store the active double-precision (64-bit) floa... | vcompresspd | — |
| `_mm512_maskz_compress_ps` | Contiguously store the active single-precision (32-bit) floa... | vcompressps | — |
| `_mm512_maskz_conflict_epi32` | Test each 32-bit element of a for equality with all other el... | vpconflictd | — |
| `_mm512_maskz_conflict_epi64` | Test each 64-bit element of a for equality with all other el... | vpconflictq | — |
| `_mm512_maskz_cvt_roundepi32_ps` | Convert packed signed 32-bit integers in a to packed single-... | vcvtdq2ps | 4/1, 3/1 |
| `_mm512_maskz_cvt_roundepi64_pd` | Convert packed signed 64-bit integers in a to packed double-... | vcvtqq2pd | 4/1, 3/1 |
| `_mm512_maskz_cvt_roundepi64_ps` | Convert packed signed 64-bit integers in a to packed single-... | vcvtqq2ps | 4/1, 3/1 |
| `_mm512_maskz_cvt_roundepu32_ps` | Convert packed unsigned 32-bit integers in a to packed singl... | vcvtudq2ps | 4/1, 3/1 |
| `_mm512_maskz_cvt_roundepu64_pd` | Convert packed unsigned 64-bit integers in a to packed doubl... | vcvtuqq2pd | 4/1, 3/1 |
| `_mm512_maskz_cvt_roundepu64_ps` | Convert packed unsigned 64-bit integers in a to packed singl... | vcvtuqq2ps | 4/1, 3/1 |
| `_mm512_maskz_cvt_roundpd_epi32` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2dq | 4/1, 3/1 |
| `_mm512_maskz_cvt_roundpd_epi64` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2qq | 4/1, 3/1 |
| `_mm512_maskz_cvt_roundpd_epu32` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2udq | 4/1, 3/1 |
| `_mm512_maskz_cvt_roundpd_epu64` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2uqq | 4/1, 3/1 |
| `_mm512_maskz_cvt_roundpd_ps` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2ps | 4/1, 3/1 |
| `_mm512_maskz_cvt_roundph_ps` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2ps | 4/1, 3/1 |
| `_mm512_maskz_cvt_roundps_epi32` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2dq | 4/1, 3/1 |
| `_mm512_maskz_cvt_roundps_epi64` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2qq | 4/1, 3/1 |
| `_mm512_maskz_cvt_roundps_epu32` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2udq | 4/1, 3/1 |
| `_mm512_maskz_cvt_roundps_epu64` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2uqq | 4/1, 3/1 |
| `_mm512_maskz_cvt_roundps_pd` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2pd | 4/1, 3/1 |
| `_mm512_maskz_cvt_roundps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2ph | 4/1, 3/1 |
| `_mm512_maskz_cvtepi16_epi32` | Sign extend packed 16-bit integers in a to packed 32-bit int... | vpmovsxwd | 4/1, 3/1 |
| `_mm512_maskz_cvtepi16_epi64` | Sign extend packed 16-bit integers in a to packed 64-bit int... | vpmovsxwq | 4/1, 3/1 |
| `_mm512_maskz_cvtepi16_epi8` | Convert packed 16-bit integers in a to packed 8-bit integers... | vpmovwb | 4/1, 3/1 |
| `_mm512_maskz_cvtepi32_epi16` | Convert packed 32-bit integers in a to packed 16-bit integer... | vpmovdw | 4/1, 3/1 |
| `_mm512_maskz_cvtepi32_epi64` | Sign extend packed 32-bit integers in a to packed 64-bit int... | vpmovsxdq | 4/1, 3/1 |
| `_mm512_maskz_cvtepi32_epi8` | Convert packed 32-bit integers in a to packed 8-bit integers... | vpmovdb | 4/1, 3/1 |
| `_mm512_maskz_cvtepi32_pd` | Convert packed signed 32-bit integers in a to packed double-... | vcvtdq2pd | 4/1, 3/1 |
| `_mm512_maskz_cvtepi32_ps` | Convert packed signed 32-bit integers in a to packed single-... | vcvtdq2ps | 4/1, 3/1 |
| `_mm512_maskz_cvtepi64_epi16` | Convert packed 64-bit integers in a to packed 16-bit integer... | vpmovqw | 4/1, 3/1 |
| `_mm512_maskz_cvtepi64_epi32` | Convert packed 64-bit integers in a to packed 32-bit integer... | vpmovqd | 4/1, 3/1 |
| `_mm512_maskz_cvtepi64_epi8` | Convert packed 64-bit integers in a to packed 8-bit integers... | vpmovqb | 4/1, 3/1 |
| `_mm512_maskz_cvtepi64_pd` | Convert packed signed 64-bit integers in a to packed double-... | vcvtqq2pd | 4/1, 3/1 |
| `_mm512_maskz_cvtepi64_ps` | Convert packed signed 64-bit integers in a to packed single-... | vcvtqq2ps | 4/1, 3/1 |
| `_mm512_maskz_cvtepi8_epi16` | Sign extend packed 8-bit integers in a to packed 16-bit inte... | vpmovsxbw | 4/1, 3/1 |
| `_mm512_maskz_cvtepi8_epi32` | Sign extend packed 8-bit integers in a to packed 32-bit inte... | vpmovsxbd | 4/1, 3/1 |
| `_mm512_maskz_cvtepi8_epi64` | Sign extend packed 8-bit integers in the low 8 bytes of a to... | vpmovsxbq | 4/1, 3/1 |
| `_mm512_maskz_cvtepu16_epi32` | Zero extend packed unsigned 16-bit integers in a to packed 3... | vpmovzxwd | 4/1, 3/1 |
| `_mm512_maskz_cvtepu16_epi64` | Zero extend packed unsigned 16-bit integers in a to packed 6... | vpmovzxwq | 4/1, 3/1 |
| `_mm512_maskz_cvtepu32_epi64` | Zero extend packed unsigned 32-bit integers in a to packed 6... | vpmovzxdq | 4/1, 3/1 |
| `_mm512_maskz_cvtepu32_pd` | Convert packed unsigned 32-bit integers in a to packed doubl... | vcvtudq2pd | 4/1, 3/1 |
| `_mm512_maskz_cvtepu32_ps` | Convert packed unsigned 32-bit integers in a to packed singl... | vcvtudq2ps | 4/1, 3/1 |
| `_mm512_maskz_cvtepu64_pd` | Convert packed unsigned 64-bit integers in a to packed doubl... | vcvtuqq2pd | 4/1, 3/1 |
| `_mm512_maskz_cvtepu64_ps` | Convert packed unsigned 64-bit integers in a to packed singl... | vcvtuqq2ps | 4/1, 3/1 |
| `_mm512_maskz_cvtepu8_epi16` | Zero extend packed unsigned 8-bit integers in a to packed 16... | vpmovzxbw | 4/1, 3/1 |
| `_mm512_maskz_cvtepu8_epi32` | Zero extend packed unsigned 8-bit integers in a to packed 32... | vpmovzxbd | 4/1, 3/1 |
| `_mm512_maskz_cvtepu8_epi64` | Zero extend packed unsigned 8-bit integers in the low 8 byte... | vpmovzxbq | 4/1, 3/1 |
| `_mm512_maskz_cvtpd_epi32` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2dq | 4/1, 3/1 |
| `_mm512_maskz_cvtpd_epi64` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2qq | 4/1, 3/1 |
| `_mm512_maskz_cvtpd_epu32` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2udq | 4/1, 3/1 |
| `_mm512_maskz_cvtpd_epu64` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2uqq | 4/1, 3/1 |
| `_mm512_maskz_cvtpd_ps` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2ps | 4/1, 3/1 |
| `_mm512_maskz_cvtph_ps` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2ps | 4/1, 3/1 |
| `_mm512_maskz_cvtps_epi32` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2dq | 4/1, 3/1 |
| `_mm512_maskz_cvtps_epi64` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2qq | 4/1, 3/1 |
| `_mm512_maskz_cvtps_epu32` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2udq | 4/1, 3/1 |
| `_mm512_maskz_cvtps_epu64` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2uqq | 4/1, 3/1 |
| `_mm512_maskz_cvtps_pd` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2pd | 4/1, 3/1 |
| `_mm512_maskz_cvtps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2ph | 4/1, 3/1 |
| `_mm512_maskz_cvtsepi16_epi8` | Convert packed signed 16-bit integers in a to packed 8-bit i... | vpmovswb | 4/1, 3/1 |
| `_mm512_maskz_cvtsepi32_epi16` | Convert packed signed 32-bit integers in a to packed 16-bit ... | vpmovsdw | 4/1, 3/1 |
| `_mm512_maskz_cvtsepi32_epi8` | Convert packed signed 32-bit integers in a to packed 8-bit i... | vpmovsdb | 4/1, 3/1 |
| `_mm512_maskz_cvtsepi64_epi16` | Convert packed signed 64-bit integers in a to packed 16-bit ... | vpmovsqw | 4/1, 3/1 |
| `_mm512_maskz_cvtsepi64_epi32` | Convert packed signed 64-bit integers in a to packed 32-bit ... | vpmovsqd | 4/1, 3/1 |
| `_mm512_maskz_cvtsepi64_epi8` | Convert packed signed 64-bit integers in a to packed 8-bit i... | vpmovsqb | 4/1, 3/1 |
| `_mm512_maskz_cvtt_roundpd_epi32` | Convert packed single-precision (32-bit) floating-point elem... | vcvttpd2dq | 4/1, 3/1 |
| `_mm512_maskz_cvtt_roundpd_epi64` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2qq | 4/1, 3/1 |
| `_mm512_maskz_cvtt_roundpd_epu32` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2udq | 4/1, 3/1 |
| `_mm512_maskz_cvtt_roundpd_epu64` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2uqq | 4/1, 3/1 |
| `_mm512_maskz_cvtt_roundps_epi32` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2dq | 4/1, 3/1 |
| `_mm512_maskz_cvtt_roundps_epi64` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2qq | 4/1, 3/1 |
| `_mm512_maskz_cvtt_roundps_epu32` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2udq | 4/1, 3/1 |
| `_mm512_maskz_cvtt_roundps_epu64` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2uqq | 4/1, 3/1 |
| `_mm512_maskz_cvttpd_epi32` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2dq | 4/1, 3/1 |
| `_mm512_maskz_cvttpd_epi64` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2qq | 4/1, 3/1 |
| `_mm512_maskz_cvttpd_epu32` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2udq | 4/1, 3/1 |
| `_mm512_maskz_cvttpd_epu64` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2uqq | 4/1, 3/1 |
| `_mm512_maskz_cvttps_epi32` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2dq | 4/1, 3/1 |
| `_mm512_maskz_cvttps_epi64` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2qq | 4/1, 3/1 |
| `_mm512_maskz_cvttps_epu32` | Convert packed double-precision (32-bit) floating-point elem... | vcvttps2udq | 4/1, 3/1 |
| `_mm512_maskz_cvttps_epu64` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2uqq | 4/1, 3/1 |
| `_mm512_maskz_cvtusepi16_epi8` | Convert packed unsigned 16-bit integers in a to packed unsig... | vpmovuswb | 4/1, 3/1 |
| `_mm512_maskz_cvtusepi32_epi16` | Convert packed unsigned 32-bit integers in a to packed unsig... | vpmovusdw | 4/1, 3/1 |
| `_mm512_maskz_cvtusepi32_epi8` | Convert packed unsigned 32-bit integers in a to packed unsig... | vpmovusdb | 4/1, 3/1 |
| `_mm512_maskz_cvtusepi64_epi16` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqw | 4/1, 3/1 |
| `_mm512_maskz_cvtusepi64_epi32` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqd | 4/1, 3/1 |
| `_mm512_maskz_cvtusepi64_epi8` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqb | 4/1, 3/1 |
| `_mm512_maskz_dbsad_epu8` | Compute the sum of absolute differences (SADs) of quadruplet... | vdbpsadbw | — |
| `_mm512_maskz_div_pd` | Divide packed double-precision (64-bit) floating-point eleme... | vdivpd | 20/14, 13/7 |
| `_mm512_maskz_div_ps` | Divide packed single-precision (32-bit) floating-point eleme... | vdivps | 13/7, 10/4 |
| `_mm512_maskz_div_round_pd` | Divide packed double-precision (64-bit) floating-point eleme... | vdivpd | — |
| `_mm512_maskz_div_round_ps` | Divide packed single-precision (32-bit) floating-point eleme... | vdivps | — |
| `_mm512_maskz_expand_epi32` | Load contiguous active 32-bit integers from a (those with th... | vpexpandd | — |
| `_mm512_maskz_expand_epi64` | Load contiguous active 64-bit integers from a (those with th... | vpexpandq | — |
| `_mm512_maskz_expand_pd` | Load contiguous active double-precision (64-bit) floating-po... | vexpandpd | — |
| `_mm512_maskz_expand_ps` | Load contiguous active single-precision (32-bit) floating-po... | vexpandps | — |
| `_mm512_maskz_extractf32x4_ps` | Extract 128 bits (composed of 4 packed single-precision (32-... | vextractf32x4 | — |
| `_mm512_maskz_extractf32x8_ps` | Extracts 256 bits (composed of 8 packed single-precision (32... | vextractf32x8 | — |
| `_mm512_maskz_extractf64x2_pd` | Extracts 128 bits (composed of 2 packed double-precision (64... | vextractf64x2 | — |
| `_mm512_maskz_extractf64x4_pd` | Extract 256 bits (composed of 4 packed double-precision (64-... | vextractf64x4 | — |
| `_mm512_maskz_extracti32x4_epi32` | Extract 128 bits (composed of 4 packed 32-bit integers) from... | vextracti32x4 | — |
| `_mm512_maskz_extracti32x8_epi32` | Extracts 256 bits (composed of 8 packed 32-bit integers) fro... | vextracti32x8 | — |
| `_mm512_maskz_extracti64x2_epi64` | Extracts 128 bits (composed of 2 packed 64-bit integers) fro... | vextracti64x2 | — |
| `_mm512_maskz_extracti64x4_epi64` | Extract 256 bits (composed of 4 packed 64-bit integers) from... | vextracti64x4 | — |
| `_mm512_maskz_fixupimm_pd` | Fix up packed double-precision (64-bit) floating-point eleme... | vfixupimmpd | — |
| `_mm512_maskz_fixupimm_ps` | Fix up packed single-precision (32-bit) floating-point eleme... | vfixupimmps | — |
| `_mm512_maskz_fixupimm_round_pd` | Fix up packed double-precision (64-bit) floating-point eleme... | vfixupimmpd | — |
| `_mm512_maskz_fixupimm_round_ps` | Fix up packed single-precision (32-bit) floating-point eleme... | vfixupimmps | — |
| `_mm512_maskz_fmadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmadd | —/4/1 |
| `_mm512_maskz_fmadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmadd | —/4/1 |
| `_mm512_maskz_fmadd_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmadd | —/4/1 |
| `_mm512_maskz_fmadd_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmadd | —/4/1 |
| `_mm512_maskz_fmaddsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmaddsub | —/4/1 |
| `_mm512_maskz_fmaddsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmaddsub | —/4/1 |
| `_mm512_maskz_fmaddsub_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmaddsub | —/4/1 |
| `_mm512_maskz_fmaddsub_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmaddsub | —/4/1 |
| `_mm512_maskz_fmsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsub | —/4/1 |
| `_mm512_maskz_fmsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsub | —/4/1 |
| `_mm512_maskz_fmsub_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsub | —/4/1 |
| `_mm512_maskz_fmsub_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsub | —/4/1 |
| `_mm512_maskz_fmsubadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsubadd | —/4/1 |
| `_mm512_maskz_fmsubadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsubadd | —/4/1 |
| `_mm512_maskz_fmsubadd_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsubadd | —/4/1 |
| `_mm512_maskz_fmsubadd_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsubadd | —/4/1 |
| `_mm512_maskz_fnmadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmadd | —/4/1 |
| `_mm512_maskz_fnmadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmadd | —/4/1 |
| `_mm512_maskz_fnmadd_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmadd | —/4/1 |
| `_mm512_maskz_fnmadd_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmadd | —/4/1 |
| `_mm512_maskz_fnmsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmsub | —/4/1 |
| `_mm512_maskz_fnmsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmsub | —/4/1 |
| `_mm512_maskz_fnmsub_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmsub | —/4/1 |
| `_mm512_maskz_fnmsub_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmsub | —/4/1 |
| `_mm512_maskz_getexp_pd` | Convert the exponent of each packed double-precision (64-bit... | vgetexppd | — |
| `_mm512_maskz_getexp_ps` | Convert the exponent of each packed single-precision (32-bit... | vgetexpps | — |
| `_mm512_maskz_getexp_round_pd` | Convert the exponent of each packed double-precision (64-bit... | vgetexppd | — |
| `_mm512_maskz_getexp_round_ps` | Convert the exponent of each packed single-precision (32-bit... | vgetexpps | — |
| `_mm512_maskz_getmant_pd` | Normalize the mantissas of packed double-precision (64-bit) ... | vgetmantpd | — |
| `_mm512_maskz_getmant_ps` | Normalize the mantissas of packed single-precision (32-bit) ... | vgetmantps | — |
| `_mm512_maskz_getmant_round_pd` | Normalize the mantissas of packed double-precision (64-bit) ... | vgetmantpd | — |
| `_mm512_maskz_getmant_round_ps` | Normalize the mantissas of packed single-precision (32-bit) ... | vgetmantps | — |
| `_mm512_maskz_insertf32x4` | Copy a to tmp, then insert 128 bits (composed of 4 packed si... | vinsertf32x4 | — |
| `_mm512_maskz_insertf32x8` | Copy a to tmp, then insert 256 bits (composed of 8 packed si... | vinsertf32x8 | — |
| `_mm512_maskz_insertf64x2` | Copy a to tmp, then insert 128 bits (composed of 2 packed do... | vinsertf64x2 | — |
| `_mm512_maskz_insertf64x4` | Copy a to tmp, then insert 256 bits (composed of 4 packed do... | vinsertf64x4 | — |
| `_mm512_maskz_inserti32x4` | Copy a to tmp, then insert 128 bits (composed of 4 packed 32... | vinserti32x4 | — |
| `_mm512_maskz_inserti32x8` | Copy a to tmp, then insert 256 bits (composed of 8 packed 32... | vinserti32x8 | — |
| `_mm512_maskz_inserti64x2` | Copy a to tmp, then insert 128 bits (composed of 2 packed 64... | vinserti64x2 | — |
| `_mm512_maskz_inserti64x4` | Copy a to tmp, then insert 256 bits (composed of 4 packed 64... | vinserti64x4 | — |
| `_mm512_maskz_lzcnt_epi32` | Counts the number of leading zero bits in each packed 32-bit... | vplzcntd | — |
| `_mm512_maskz_lzcnt_epi64` | Counts the number of leading zero bits in each packed 64-bit... | vplzcntq | — |
| `_mm512_maskz_madd_epi16` | Multiply packed signed 16-bit integers in a and b, producing... | vpmaddwd | — |
| `_mm512_maskz_maddubs_epi16` | Multiply packed unsigned 8-bit integers in a by packed signe... | vpmaddubsw | — |
| `_mm512_maskz_max_epi16` | Compare packed signed 16-bit integers in a and b, and store ... | vpmaxsw | — |
| `_mm512_maskz_max_epi32` | Compare packed signed 32-bit integers in a and b, and store ... | vpmaxsd | — |
| `_mm512_maskz_max_epi64` | Compare packed signed 64-bit integers in a and b, and store ... | vpmaxsq | — |
| `_mm512_maskz_max_epi8` | Compare packed signed 8-bit integers in a and b, and store p... | vpmaxsb | — |
| `_mm512_maskz_max_epu16` | Compare packed unsigned 16-bit integers in a and b, and stor... | vpmaxuw | — |
| `_mm512_maskz_max_epu32` | Compare packed unsigned 32-bit integers in a and b, and stor... | vpmaxud | — |
| `_mm512_maskz_max_epu64` | Compare packed unsigned 64-bit integers in a and b, and stor... | vpmaxuq | — |
| `_mm512_maskz_max_epu8` | Compare packed unsigned 8-bit integers in a and b, and store... | vpmaxub | — |
| `_mm512_maskz_max_pd` | Compare packed double-precision (64-bit) floating-point elem... | vmaxpd | — |
| `_mm512_maskz_max_ps` | Compare packed single-precision (32-bit) floating-point elem... | vmaxps | — |
| `_mm512_maskz_max_round_pd` | Compare packed double-precision (64-bit) floating-point elem... | vmaxpd | — |
| `_mm512_maskz_max_round_ps` | Compare packed single-precision (32-bit) floating-point elem... | vmaxps | — |
| `_mm512_maskz_min_epi16` | Compare packed signed 16-bit integers in a and b, and store ... | vpminsw | — |
| `_mm512_maskz_min_epi32` | Compare packed signed 32-bit integers in a and b, and store ... | vpminsd | — |
| `_mm512_maskz_min_epi64` | Compare packed signed 64-bit integers in a and b, and store ... | vpminsq | — |
| `_mm512_maskz_min_epi8` | Compare packed signed 8-bit integers in a and b, and store p... | vpminsb | — |
| `_mm512_maskz_min_epu16` | Compare packed unsigned 16-bit integers in a and b, and stor... | vpminuw | — |
| `_mm512_maskz_min_epu32` | Compare packed unsigned 32-bit integers in a and b, and stor... | vpminud | — |
| `_mm512_maskz_min_epu64` | Compare packed unsigned 64-bit integers in a and b, and stor... | vpminuq | — |
| `_mm512_maskz_min_epu8` | Compare packed unsigned 8-bit integers in a and b, and store... | vpminub | — |
| `_mm512_maskz_min_pd` | Compare packed double-precision (64-bit) floating-point elem... | vminpd | — |
| `_mm512_maskz_min_ps` | Compare packed single-precision (32-bit) floating-point elem... | vminps | — |
| `_mm512_maskz_min_round_pd` | Compare packed double-precision (64-bit) floating-point elem... | vminpd | — |
| `_mm512_maskz_min_round_ps` | Compare packed single-precision (32-bit) floating-point elem... | vminps | — |
| `_mm512_maskz_mov_epi16` | Move packed 16-bit integers from a into dst using zeromask k... | vmovdqu16 | — |
| `_mm512_maskz_mov_epi32` | Move packed 32-bit integers from a into dst using zeromask k... | vmovdqa32 | — |
| `_mm512_maskz_mov_epi64` | Move packed 64-bit integers from a into dst using zeromask k... | vmovdqa64 | — |
| `_mm512_maskz_mov_epi8` | Move packed 8-bit integers from a into dst using zeromask k ... | vmovdqu8 | — |
| `_mm512_maskz_mov_pd` | Move packed double-precision (64-bit) floating-point element... | vmovapd | — |
| `_mm512_maskz_mov_ps` | Move packed single-precision (32-bit) floating-point element... | vmovaps | — |
| `_mm512_maskz_movedup_pd` | Duplicate even-indexed double-precision (64-bit) floating-po... | vmovddup | — |
| `_mm512_maskz_movehdup_ps` | Duplicate odd-indexed single-precision (32-bit) floating-poi... | vmovshdup | — |
| `_mm512_maskz_moveldup_ps` | Duplicate even-indexed single-precision (32-bit) floating-po... | vmovsldup | — |
| `_mm512_maskz_mul_epi32` | Multiply the low signed 32-bit integers from each packed 64-... | vpmuldq | 10/2, 4/1 |
| `_mm512_maskz_mul_epu32` | Multiply the low unsigned 32-bit integers from each packed 6... | vpmuludq | 10/2, 4/1 |
| `_mm512_maskz_mul_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vmulpd | —/3/1 |
| `_mm512_maskz_mul_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vmulps | —/3/1 |
| `_mm512_maskz_mul_round_pd` | Multiply packed single-precision (32-bit) floating-point ele... | vmulpd | — |
| `_mm512_maskz_mul_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vmulps | — |
| `_mm512_maskz_mulhi_epi16` | Multiply the packed signed 16-bit integers in a and b, produ... | vpmulhw | 5/1, 3/1 |
| `_mm512_maskz_mulhi_epu16` | Multiply the packed unsigned 16-bit integers in a and b, pro... | vpmulhuw | 5/1, 3/1 |
| `_mm512_maskz_mulhrs_epi16` | Multiply packed signed 16-bit integers in a and b, producing... | vpmulhrsw | — |
| `_mm512_maskz_mullo_epi16` | Multiply the packed 16-bit integers in a and b, producing in... | vpmullw | 5/1, 3/1 |
| `_mm512_maskz_mullo_epi32` | Multiply the packed 32-bit integers in a and b, producing in... | vpmulld | 10/2, 4/1 |
| `_mm512_maskz_mullo_epi64` | Multiply packed 64-bit integers in `a` and `b`, producing in... | vpmullq | — |
| `_mm512_maskz_or_epi32` | Compute the bitwise OR of packed 32-bit integers in a and b,... | vpord | —/1/1 |
| `_mm512_maskz_or_epi64` | Compute the bitwise OR of packed 64-bit integers in a and b,... | vporq | —/1/1 |
| `_mm512_maskz_or_pd` | Compute the bitwise OR of packed double-precision (64-bit) f... | vorpd | —/1/1 |
| `_mm512_maskz_or_ps` | Compute the bitwise OR of packed single-precision (32-bit) f... | vorps | —/1/1 |
| `_mm512_maskz_packs_epi16` | Convert packed signed 16-bit integers from a and b to packed... | vpacksswb | 1/1, 1/1 |
| `_mm512_maskz_packs_epi32` | Convert packed signed 32-bit integers from a and b to packed... | vpackssdw | 1/1, 1/1 |
| `_mm512_maskz_packus_epi16` | Convert packed signed 16-bit integers from a and b to packed... | vpackuswb | 1/1, 1/1 |
| `_mm512_maskz_packus_epi32` | Convert packed signed 32-bit integers from a and b to packed... | vpackusdw | 1/1, 1/1 |
| `_mm512_maskz_permute_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vshufpd | —/2/1 |
| `_mm512_maskz_permute_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vshufps | —/2/1 |
| `_mm512_maskz_permutevar_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vpermilpd | —/2/1 |
| `_mm512_maskz_permutevar_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vpermilps | —/2/1 |
| `_mm512_maskz_permutex2var_epi16` | Shuffle 16-bit integers in a and b across lanes using the co... | vperm | —/2/1 |
| `_mm512_maskz_permutex2var_epi32` | Shuffle 32-bit integers in a and b across lanes using the co... | vperm | —/2/1 |
| `_mm512_maskz_permutex2var_epi64` | Shuffle 64-bit integers in a and b across lanes using the co... | vperm | —/2/1 |
| `_mm512_maskz_permutex2var_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vperm | —/2/1 |
| `_mm512_maskz_permutex2var_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vperm | —/2/1 |
| `_mm512_maskz_permutex_epi64` | Shuffle 64-bit integers in a within 256-bit lanes using the ... | vperm | —/2/1 |
| `_mm512_maskz_permutex_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vperm | —/2/1 |
| `_mm512_maskz_permutexvar_epi16` | Shuffle 16-bit integers in a across lanes using the correspo... | vpermw | —/2/1 |
| `_mm512_maskz_permutexvar_epi32` | Shuffle 32-bit integers in a across lanes using the correspo... | vpermd | —/2/1 |
| `_mm512_maskz_permutexvar_epi64` | Shuffle 64-bit integers in a across lanes using the correspo... | vpermq | —/2/1 |
| `_mm512_maskz_permutexvar_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vpermpd | —/2/1 |
| `_mm512_maskz_permutexvar_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vpermps | —/2/1 |
| `_mm512_maskz_range_pd` | Calculate the max, min, absolute max, or absolute min (depen... | vrangepd | — |
| `_mm512_maskz_range_ps` | Calculate the max, min, absolute max, or absolute min (depen... | vrangeps | — |
| `_mm512_maskz_range_round_pd` | Calculate the max, min, absolute max, or absolute min (depen... | vrangepd | — |
| `_mm512_maskz_range_round_ps` | Calculate the max, min, absolute max, or absolute min (depen... | vrangeps | — |
| `_mm512_maskz_rcp14_pd` | Compute the approximate reciprocal of packed double-precisio... | vrcp14pd | — |
| `_mm512_maskz_rcp14_ps` | Compute the approximate reciprocal of packed single-precisio... | vrcp14ps | — |
| `_mm512_maskz_reduce_pd` | Extract the reduced argument of packed double-precision (64-... | vreducepd | — |
| `_mm512_maskz_reduce_ps` | Extract the reduced argument of packed single-precision (32-... | vreduceps | — |
| `_mm512_maskz_reduce_round_pd` | Extract the reduced argument of packed double-precision (64-... | vreducepd | — |
| `_mm512_maskz_reduce_round_ps` | Extract the reduced argument of packed single-precision (32-... | vreduceps | — |
| `_mm512_maskz_rol_epi32` | Rotate the bits in each packed 32-bit integer in a to the le... | vprold | — |
| `_mm512_maskz_rol_epi64` | Rotate the bits in each packed 64-bit integer in a to the le... | vprolq | — |
| `_mm512_maskz_rolv_epi32` | Rotate the bits in each packed 32-bit integer in a to the le... | vprolvd | — |
| `_mm512_maskz_rolv_epi64` | Rotate the bits in each packed 64-bit integer in a to the le... | vprolvq | — |
| `_mm512_maskz_ror_epi32` | Rotate the bits in each packed 32-bit integer in a to the ri... | vprold | — |
| `_mm512_maskz_ror_epi64` | Rotate the bits in each packed 64-bit integer in a to the ri... | vprolq | — |
| `_mm512_maskz_rorv_epi32` | Rotate the bits in each packed 32-bit integer in a to the ri... | vprorvd | — |
| `_mm512_maskz_rorv_epi64` | Rotate the bits in each packed 64-bit integer in a to the ri... | vprorvq | — |
| `_mm512_maskz_roundscale_pd` | Round packed double-precision (64-bit) floating-point elemen... | vrndscalepd | — |
| `_mm512_maskz_roundscale_ps` | Round packed single-precision (32-bit) floating-point elemen... | vrndscaleps | — |
| `_mm512_maskz_roundscale_round_pd` | Round packed double-precision (64-bit) floating-point elemen... | vrndscalepd | — |
| `_mm512_maskz_roundscale_round_ps` | Round packed single-precision (32-bit) floating-point elemen... | vrndscaleps | — |
| `_mm512_maskz_rsqrt14_pd` | Compute the approximate reciprocal square root of packed dou... | vrsqrt14pd | — |
| `_mm512_maskz_rsqrt14_ps` | Compute the approximate reciprocal square root of packed sin... | vrsqrt14ps | — |
| `_mm512_maskz_scalef_pd` | Scale the packed double-precision (64-bit) floating-point el... | vscalefpd | — |
| `_mm512_maskz_scalef_ps` | Scale the packed single-precision (32-bit) floating-point el... | vscalefps | — |
| `_mm512_maskz_scalef_round_pd` | Scale the packed double-precision (64-bit) floating-point el... | vscalefpd | — |
| `_mm512_maskz_scalef_round_ps` | Scale the packed single-precision (32-bit) floating-point el... | vscalefps | — |
| `_mm512_maskz_set1_epi16` | Broadcast the low packed 16-bit integer from a to all elemen... | vpbroadcastw | — |
| `_mm512_maskz_set1_epi32` | Broadcast 32-bit integer a to all elements of dst using zero... | vpbroadcastd | — |
| `_mm512_maskz_set1_epi64` | Broadcast 64-bit integer a to all elements of dst using zero... | vpbroadcastq | — |
| `_mm512_maskz_set1_epi8` | Broadcast 8-bit integer a to all elements of dst using zerom... | vpbroadcast | — |
| `_mm512_maskz_shuffle_epi32` | Shuffle 32-bit integers in a within 128-bit lanes using the ... | vpshufd | —/1/1 |
| `_mm512_maskz_shuffle_epi8` | Shuffle packed 8-bit integers in a according to shuffle cont... | vpshufb | —/1/1 |
| `_mm512_maskz_shuffle_f32x4` | Shuffle 128-bits (composed of 4 single-precision (32-bit) fl... | vshuff32x4 | —/1/1 |
| `_mm512_maskz_shuffle_f64x2` | Shuffle 128-bits (composed of 2 double-precision (64-bit) fl... | vshuff64x2 | —/1/1 |
| `_mm512_maskz_shuffle_i32x4` | Shuffle 128-bits (composed of 4 32-bit integers) selected by... | vshufi32x4 | —/1/1 |
| `_mm512_maskz_shuffle_i64x2` | Shuffle 128-bits (composed of 2 64-bit integers) selected by... | vshufi64x2 | —/1/1 |
| `_mm512_maskz_shuffle_pd` | Shuffle double-precision (64-bit) floating-point elements wi... | vshufpd | —/1/1 |
| `_mm512_maskz_shuffle_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vshufps | —/1/1 |
| `_mm512_maskz_shufflehi_epi16` | Shuffle 16-bit integers in the high 64 bits of 128-bit lanes... | vpshufhw | —/1/1 |
| `_mm512_maskz_shufflelo_epi16` | Shuffle 16-bit integers in the low 64 bits of 128-bit lanes ... | vpshuflw | —/1/1 |
| `_mm512_maskz_sll_epi16` | Shift packed 16-bit integers in a left by count while shifti... | vpsllw | —/1/1 |
| `_mm512_maskz_sll_epi32` | Shift packed 32-bit integers in a left by count while shifti... | vpslld | —/1/1 |
| `_mm512_maskz_sll_epi64` | Shift packed 64-bit integers in a left by count while shifti... | vpsllq | —/1/1 |
| `_mm512_maskz_slli_epi16` | Shift packed 16-bit integers in a left by imm8 while shiftin... | vpsllw | —/1/1 |
| `_mm512_maskz_slli_epi32` | Shift packed 32-bit integers in a left by imm8 while shiftin... | vpslld | —/1/1 |
| `_mm512_maskz_slli_epi64` | Shift packed 64-bit integers in a left by imm8 while shiftin... | vpsllq | —/1/1 |
| `_mm512_maskz_sllv_epi16` | Shift packed 16-bit integers in a left by the amount specifi... | vpsllvw | —/1/1 |
| `_mm512_maskz_sllv_epi32` | Shift packed 32-bit integers in a left by the amount specifi... | vpsllvd | —/1/1 |
| `_mm512_maskz_sllv_epi64` | Shift packed 64-bit integers in a left by the amount specifi... | vpsllvq | —/1/1 |
| `_mm512_maskz_sqrt_pd` | Compute the square root of packed double-precision (64-bit) ... | vsqrtpd | 16/14, 20/9 |
| `_mm512_maskz_sqrt_ps` | Compute the square root of packed single-precision (32-bit) ... | vsqrtps | 11/7, 14/5 |
| `_mm512_maskz_sqrt_round_pd` | Compute the square root of packed double-precision (64-bit) ... | vsqrtpd | — |
| `_mm512_maskz_sqrt_round_ps` | Compute the square root of packed single-precision (32-bit) ... | vsqrtps | — |
| `_mm512_maskz_sra_epi16` | Shift packed 16-bit integers in a right by count while shift... | vpsraw | —/1/1 |
| `_mm512_maskz_sra_epi32` | Shift packed 32-bit integers in a right by count while shift... | vpsrad | —/1/1 |
| `_mm512_maskz_sra_epi64` | Shift packed 64-bit integers in a right by count while shift... | vpsraq | —/1/1 |
| `_mm512_maskz_srai_epi16` | Shift packed 16-bit integers in a right by imm8 while shifti... | vpsraw | —/1/1 |
| `_mm512_maskz_srai_epi32` | Shift packed 32-bit integers in a right by imm8 while shifti... | vpsrad | —/1/1 |
| `_mm512_maskz_srai_epi64` | Shift packed 64-bit integers in a right by imm8 while shifti... | vpsraq | —/1/1 |
| `_mm512_maskz_srav_epi16` | Shift packed 16-bit integers in a right by the amount specif... | vpsravw | —/1/1 |
| `_mm512_maskz_srav_epi32` | Shift packed 32-bit integers in a right by the amount specif... | vpsravd | —/1/1 |
| `_mm512_maskz_srav_epi64` | Shift packed 64-bit integers in a right by the amount specif... | vpsravq | —/1/1 |
| `_mm512_maskz_srl_epi16` | Shift packed 16-bit integers in a right by count while shift... | vpsrlw | —/1/1 |
| `_mm512_maskz_srl_epi32` | Shift packed 32-bit integers in a right by count while shift... | vpsrld | —/1/1 |
| `_mm512_maskz_srl_epi64` | Shift packed 64-bit integers in a right by count while shift... | vpsrlq | —/1/1 |
| `_mm512_maskz_srli_epi16` | Shift packed 16-bit integers in a right by imm8 while shifti... | vpsrlw | —/1/1 |
| `_mm512_maskz_srli_epi32` | Shift packed 32-bit integers in a right by imm8 while shifti... | vpsrld | —/1/1 |
| `_mm512_maskz_srli_epi64` | Shift packed 64-bit integers in a right by imm8 while shifti... | vpsrlq | —/1/1 |
| `_mm512_maskz_srlv_epi16` | Shift packed 16-bit integers in a right by the amount specif... | vpsrlvw | —/1/1 |
| `_mm512_maskz_srlv_epi32` | Shift packed 32-bit integers in a right by the amount specif... | vpsrlvd | —/1/1 |
| `_mm512_maskz_srlv_epi64` | Shift packed 64-bit integers in a right by the amount specif... | vpsrlvq | —/1/1 |
| `_mm512_maskz_sub_epi16` | Subtract packed 16-bit integers in b from packed 16-bit inte... | vpsubw | —/1/1 |
| `_mm512_maskz_sub_epi32` | Subtract packed 32-bit integers in b from packed 32-bit inte... | vpsubd | —/1/1 |
| `_mm512_maskz_sub_epi64` | Subtract packed 64-bit integers in b from packed 64-bit inte... | vpsubq | —/1/1 |
| `_mm512_maskz_sub_epi8` | Subtract packed 8-bit integers in b from packed 8-bit intege... | vpsubb | —/1/1 |
| `_mm512_maskz_sub_pd` | Subtract packed double-precision (64-bit) floating-point ele... | vsubpd | — |
| `_mm512_maskz_sub_ps` | Subtract packed single-precision (32-bit) floating-point ele... | vsubps | — |
| `_mm512_maskz_sub_round_pd` | Subtract packed double-precision (64-bit) floating-point ele... | vsubpd | — |
| `_mm512_maskz_sub_round_ps` | Subtract packed single-precision (32-bit) floating-point ele... | vsubps | — |
| `_mm512_maskz_subs_epi16` | Subtract packed signed 16-bit integers in b from packed 16-b... | vpsubsw | —/1/1 |
| `_mm512_maskz_subs_epi8` | Subtract packed signed 8-bit integers in b from packed 8-bit... | vpsubsb | —/1/1 |
| `_mm512_maskz_subs_epu16` | Subtract packed unsigned 16-bit integers in b from packed un... | vpsubusw | — |
| `_mm512_maskz_subs_epu8` | Subtract packed unsigned 8-bit integers in b from packed uns... | vpsubusb | — |
| `_mm512_maskz_ternarylogic_epi32` | Bitwise ternary logic that provides the capability to implem... | vpternlogd | — |
| `_mm512_maskz_ternarylogic_epi64` | Bitwise ternary logic that provides the capability to implem... | vpternlogq | — |
| `_mm512_maskz_unpackhi_epi16` | Unpack and interleave 16-bit integers from the high half of ... | vpunpckhwd | 1/1, 1/1 |
| `_mm512_maskz_unpackhi_epi32` | Unpack and interleave 32-bit integers from the high half of ... | vpunpckhdq | 1/1, 1/1 |
| `_mm512_maskz_unpackhi_epi64` | Unpack and interleave 64-bit integers from the high half of ... | vpunpckhqdq | 1/1, 1/1 |
| `_mm512_maskz_unpackhi_epi8` | Unpack and interleave 8-bit integers from the high half of e... | vpunpckhbw | 1/1, 1/1 |
| `_mm512_maskz_unpackhi_pd` | Unpack and interleave double-precision (64-bit) floating-poi... | vunpckhpd | 1/1, 1/1 |
| `_mm512_maskz_unpackhi_ps` | Unpack and interleave single-precision (32-bit) floating-poi... | vunpckhps | 1/1, 1/1 |
| `_mm512_maskz_unpacklo_epi16` | Unpack and interleave 16-bit integers from the low half of e... | vpunpcklwd | 1/1, 1/1 |
| `_mm512_maskz_unpacklo_epi32` | Unpack and interleave 32-bit integers from the low half of e... | vpunpckldq | 1/1, 1/1 |
| `_mm512_maskz_unpacklo_epi64` | Unpack and interleave 64-bit integers from the low half of e... | vpunpcklqdq | 1/1, 1/1 |
| `_mm512_maskz_unpacklo_epi8` | Unpack and interleave 8-bit integers from the low half of ea... | vpunpcklbw | 1/1, 1/1 |
| `_mm512_maskz_unpacklo_pd` | Unpack and interleave double-precision (64-bit) floating-poi... | vunpcklpd | 1/1, 1/1 |
| `_mm512_maskz_unpacklo_ps` | Unpack and interleave single-precision (32-bit) floating-poi... | vunpcklps | 1/1, 1/1 |
| `_mm512_maskz_xor_epi32` | Compute the bitwise XOR of packed 32-bit integers in a and b... | vpxord | —/1/1 |
| `_mm512_maskz_xor_epi64` | Compute the bitwise XOR of packed 64-bit integers in a and b... | vpxorq | —/1/1 |
| `_mm512_maskz_xor_pd` | Compute the bitwise XOR of packed double-precision (64-bit) ... | vxorpd | —/1/1 |
| `_mm512_maskz_xor_ps` | Compute the bitwise XOR of packed single-precision (32-bit) ... | vxorps | —/1/1 |
| `_mm512_max_epi16` | Compare packed signed 16-bit integers in a and b, and store ... | vpmaxsw | — |
| `_mm512_max_epi32` | Compare packed signed 32-bit integers in a and b, and store ... | vpmaxsd | — |
| `_mm512_max_epi64` | Compare packed signed 64-bit integers in a and b, and store ... | vpmaxsq | — |
| `_mm512_max_epi8` | Compare packed signed 8-bit integers in a and b, and store p... | vpmaxsb | — |
| `_mm512_max_epu16` | Compare packed unsigned 16-bit integers in a and b, and stor... | vpmaxuw | — |
| `_mm512_max_epu32` | Compare packed unsigned 32-bit integers in a and b, and stor... | vpmaxud | — |
| `_mm512_max_epu64` | Compare packed unsigned 64-bit integers in a and b, and stor... | vpmaxuq | — |
| `_mm512_max_epu8` | Compare packed unsigned 8-bit integers in a and b, and store... | vpmaxub | — |
| `_mm512_max_pd` | Compare packed double-precision (64-bit) floating-point elem... | vmaxpd | — |
| `_mm512_max_ps` | Compare packed single-precision (32-bit) floating-point elem... | vmaxps | — |
| `_mm512_max_round_pd` | Compare packed double-precision (64-bit) floating-point elem... | vmaxpd | — |
| `_mm512_max_round_ps` | Compare packed single-precision (32-bit) floating-point elem... | vmaxps | — |
| `_mm512_min_epi16` | Compare packed signed 16-bit integers in a and b, and store ... | vpminsw | — |
| `_mm512_min_epi32` | Compare packed signed 32-bit integers in a and b, and store ... | vpminsd | — |
| `_mm512_min_epi64` | Compare packed signed 64-bit integers in a and b, and store ... | vpminsq | — |
| `_mm512_min_epi8` | Compare packed signed 8-bit integers in a and b, and store p... | vpminsb | — |
| `_mm512_min_epu16` | Compare packed unsigned 16-bit integers in a and b, and stor... | vpminuw | — |
| `_mm512_min_epu32` | Compare packed unsigned 32-bit integers in a and b, and stor... | vpminud | — |
| `_mm512_min_epu64` | Compare packed unsigned 64-bit integers in a and b, and stor... | vpminuq | — |
| `_mm512_min_epu8` | Compare packed unsigned 8-bit integers in a and b, and store... | vpminub | — |
| `_mm512_min_pd` | Compare packed double-precision (64-bit) floating-point elem... | vminpd | — |
| `_mm512_min_ps` | Compare packed single-precision (32-bit) floating-point elem... | vminps | — |
| `_mm512_min_round_pd` | Compare packed double-precision (64-bit) floating-point elem... | vminpd | — |
| `_mm512_min_round_ps` | Compare packed single-precision (32-bit) floating-point elem... | vminps | — |
| `_mm512_movedup_pd` | Duplicate even-indexed double-precision (64-bit) floating-po... | vmovddup | — |
| `_mm512_movehdup_ps` | Duplicate odd-indexed single-precision (32-bit) floating-poi... | vmovshdup | — |
| `_mm512_moveldup_ps` | Duplicate even-indexed single-precision (32-bit) floating-po... | vmovsldup | — |
| `_mm512_movepi16_mask` | Set each bit of mask register k based on the most significan... | vpmovw2m | — |
| `_mm512_movepi32_mask` | Set each bit of mask register k based on the most significan... |  | — |
| `_mm512_movepi64_mask` | Set each bit of mask register k based on the most significan... |  | — |
| `_mm512_movepi8_mask` | Set each bit of mask register k based on the most significan... | vpmovb2m | — |
| `_mm512_movm_epi16` | Set each packed 16-bit integer in dst to all ones or all zer... | vpmovm2w | — |
| `_mm512_movm_epi32` | Set each packed 32-bit integer in dst to all ones or all zer... | vpmovm2d | — |
| `_mm512_movm_epi64` | Set each packed 64-bit integer in dst to all ones or all zer... | vpmovm2q | — |
| `_mm512_movm_epi8` | Set each packed 8-bit integer in dst to all ones or all zero... | vpmovm2b | — |
| `_mm512_mul_epi32` | Multiply the low signed 32-bit integers from each packed 64-... | vpmuldq | 10/2, 4/1 |
| `_mm512_mul_epu32` | Multiply the low unsigned 32-bit integers from each packed 6... | vpmuludq | 10/2, 4/1 |
| `_mm512_mul_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vmulpd | —/3/1 |
| `_mm512_mul_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vmulps | —/3/1 |
| `_mm512_mul_round_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vmulpd | — |
| `_mm512_mul_round_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vmulps | — |
| `_mm512_mulhi_epi16` | Multiply the packed signed 16-bit integers in a and b, produ... | vpmulhw | 5/1, 3/1 |
| `_mm512_mulhi_epu16` | Multiply the packed unsigned 16-bit integers in a and b, pro... | vpmulhuw | 5/1, 3/1 |
| `_mm512_mulhrs_epi16` | Multiply packed signed 16-bit integers in a and b, producing... | vpmulhrsw | — |
| `_mm512_mullo_epi16` | Multiply the packed 16-bit integers in a and b, producing in... | vpmullw | 5/1, 3/1 |
| `_mm512_mullo_epi32` | Multiply the packed 32-bit integers in a and b, producing in... | vpmulld | 10/2, 4/1 |
| `_mm512_mullo_epi64` | Multiply packed 64-bit integers in `a` and `b`, producing in... | vpmullq | — |
| `_mm512_mullox_epi64` | Multiplies elements in packed 64-bit integer vectors a and b... |  | — |
| `_mm512_or_epi32` | Compute the bitwise OR of packed 32-bit integers in a and b,... | vporq | —/1/1 |
| `_mm512_or_epi64` | Compute the bitwise OR of packed 64-bit integers in a and b,... | vporq | —/1/1 |
| `_mm512_or_pd` | Compute the bitwise OR of packed double-precision (64-bit) f... | vorp | —/1/1 |
| `_mm512_or_ps` | Compute the bitwise OR of packed single-precision (32-bit) f... | vorps | —/1/1 |
| `_mm512_or_si512` | Compute the bitwise OR of 512 bits (representing integer dat... | vporq | —/1/1 |
| `_mm512_packs_epi16` | Convert packed signed 16-bit integers from a and b to packed... | vpacksswb | 1/1, 1/1 |
| `_mm512_packs_epi32` | Convert packed signed 32-bit integers from a and b to packed... | vpackssdw | 1/1, 1/1 |
| `_mm512_packus_epi16` | Convert packed signed 16-bit integers from a and b to packed... | vpackuswb | 1/1, 1/1 |
| `_mm512_packus_epi32` | Convert packed signed 32-bit integers from a and b to packed... | vpackusdw | 1/1, 1/1 |
| `_mm512_permute_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vshufpd | —/2/1 |
| `_mm512_permute_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vshufps | —/2/1 |
| `_mm512_permutevar_epi32` | Shuffle 32-bit integers in a across lanes using the correspo... | vperm | —/2/1 |
| `_mm512_permutevar_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vpermilpd | —/2/1 |
| `_mm512_permutevar_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vpermilps | —/2/1 |
| `_mm512_permutex2var_epi16` | Shuffle 16-bit integers in a and b across lanes using the co... | vperm | —/2/1 |
| `_mm512_permutex2var_epi32` | Shuffle 32-bit integers in a and b across lanes using the co... | vperm | —/2/1 |
| `_mm512_permutex2var_epi64` | Shuffle 64-bit integers in a and b across lanes using the co... | vperm | —/2/1 |
| `_mm512_permutex2var_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vperm | —/2/1 |
| `_mm512_permutex2var_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vperm | —/2/1 |
| `_mm512_permutex_epi64` | Shuffle 64-bit integers in a within 256-bit lanes using the ... | vperm | —/2/1 |
| `_mm512_permutex_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vperm | —/2/1 |
| `_mm512_permutexvar_epi16` | Shuffle 16-bit integers in a across lanes using the correspo... | vpermw | —/2/1 |
| `_mm512_permutexvar_epi32` | Shuffle 32-bit integers in a across lanes using the correspo... | vperm | —/2/1 |
| `_mm512_permutexvar_epi64` | Shuffle 64-bit integers in a across lanes using the correspo... | vperm | —/2/1 |
| `_mm512_permutexvar_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vpermpd | —/2/1 |
| `_mm512_permutexvar_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vpermps | —/2/1 |
| `_mm512_range_pd` | Calculate the max, min, absolute max, or absolute min (depen... | vrangepd | — |
| `_mm512_range_ps` | Calculate the max, min, absolute max, or absolute min (depen... | vrangeps | — |
| `_mm512_range_round_pd` | Calculate the max, min, absolute max, or absolute min (depen... | vrangepd | — |
| `_mm512_range_round_ps` | Calculate the max, min, absolute max, or absolute min (depen... | vrangeps | — |
| `_mm512_rcp14_pd` | Compute the approximate reciprocal of packed double-precisio... | vrcp14pd | — |
| `_mm512_rcp14_ps` | Compute the approximate reciprocal of packed single-precisio... | vrcp14ps | — |
| `_mm512_reduce_add_epi32` | Reduce the packed 32-bit integers in a by addition. Returns ... |  | —/1/1 |
| `_mm512_reduce_add_epi64` | Reduce the packed 64-bit integers in a by addition. Returns ... |  | —/1/1 |
| `_mm512_reduce_add_pd` | Reduce the packed double-precision (64-bit) floating-point e... |  | —/3/1 |
| `_mm512_reduce_add_ps` | Reduce the packed single-precision (32-bit) floating-point e... |  | —/3/1 |
| `_mm512_reduce_and_epi32` | Reduce the packed 32-bit integers in a by bitwise AND. Retur... |  | —/1/1 |
| `_mm512_reduce_and_epi64` | Reduce the packed 64-bit integers in a by bitwise AND. Retur... |  | —/1/1 |
| `_mm512_reduce_max_epi32` | Reduce the packed signed 32-bit integers in a by maximum. Re... |  | — |
| `_mm512_reduce_max_epi64` | Reduce the packed signed 64-bit integers in a by maximum. Re... |  | — |
| `_mm512_reduce_max_epu32` | Reduce the packed unsigned 32-bit integers in a by maximum. ... |  | — |
| `_mm512_reduce_max_epu64` | Reduce the packed unsigned 64-bit integers in a by maximum. ... |  | — |
| `_mm512_reduce_max_pd` | Reduce the packed double-precision (64-bit) floating-point e... |  | — |
| `_mm512_reduce_max_ps` | Reduce the packed single-precision (32-bit) floating-point e... |  | — |
| `_mm512_reduce_min_epi32` | Reduce the packed signed 32-bit integers in a by minimum. Re... |  | — |
| `_mm512_reduce_min_epi64` | Reduce the packed signed 64-bit integers in a by minimum. Re... |  | — |
| `_mm512_reduce_min_epu32` | Reduce the packed unsigned 32-bit integers in a by minimum. ... |  | — |
| `_mm512_reduce_min_epu64` | Reduce the packed unsigned 64-bit integers in a by minimum. ... |  | — |
| `_mm512_reduce_min_pd` | Reduce the packed double-precision (64-bit) floating-point e... |  | — |
| `_mm512_reduce_min_ps` | Reduce the packed single-precision (32-bit) floating-point e... |  | — |
| `_mm512_reduce_mul_epi32` | Reduce the packed 32-bit integers in a by multiplication. Re... |  | 10/2, 4/1 |
| `_mm512_reduce_mul_epi64` | Reduce the packed 64-bit integers in a by multiplication. Re... |  | — |
| `_mm512_reduce_mul_pd` | Reduce the packed double-precision (64-bit) floating-point e... |  | —/3/1 |
| `_mm512_reduce_mul_ps` | Reduce the packed single-precision (32-bit) floating-point e... |  | —/3/1 |
| `_mm512_reduce_or_epi32` | Reduce the packed 32-bit integers in a by bitwise OR. Return... |  | —/1/1 |
| `_mm512_reduce_or_epi64` | Reduce the packed 64-bit integers in a by bitwise OR. Return... |  | —/1/1 |
| `_mm512_reduce_pd` | Extract the reduced argument of packed double-precision (64-... | vreducepd | — |
| `_mm512_reduce_ps` | Extract the reduced argument of packed single-precision (32-... | vreduceps | — |
| `_mm512_reduce_round_pd` | Extract the reduced argument of packed double-precision (64-... | vreducepd | — |
| `_mm512_reduce_round_ps` | Extract the reduced argument of packed single-precision (32-... | vreduceps | — |
| `_mm512_rol_epi32` | Rotate the bits in each packed 32-bit integer in a to the le... | vprold | — |
| `_mm512_rol_epi64` | Rotate the bits in each packed 64-bit integer in a to the le... | vprolq | — |
| `_mm512_rolv_epi32` | Rotate the bits in each packed 32-bit integer in a to the le... | vprolvd | — |
| `_mm512_rolv_epi64` | Rotate the bits in each packed 64-bit integer in a to the le... | vprolvq | — |
| `_mm512_ror_epi32` | Rotate the bits in each packed 32-bit integer in a to the ri... | vprold | — |
| `_mm512_ror_epi64` | Rotate the bits in each packed 64-bit integer in a to the ri... | vprolq | — |
| `_mm512_rorv_epi32` | Rotate the bits in each packed 32-bit integer in a to the ri... | vprorvd | — |
| `_mm512_rorv_epi64` | Rotate the bits in each packed 64-bit integer in a to the ri... | vprorvq | — |
| `_mm512_roundscale_pd` | Round packed double-precision (64-bit) floating-point elemen... | vrndscalepd | — |
| `_mm512_roundscale_ps` | Round packed single-precision (32-bit) floating-point elemen... | vrndscaleps | — |
| `_mm512_roundscale_round_pd` | Round packed double-precision (64-bit) floating-point elemen... | vrndscalepd | — |
| `_mm512_roundscale_round_ps` | Round packed single-precision (32-bit) floating-point elemen... | vrndscaleps | — |
| `_mm512_rsqrt14_pd` | Compute the approximate reciprocal square root of packed dou... | vrsqrt14pd | — |
| `_mm512_rsqrt14_ps` | Compute the approximate reciprocal square root of packed sin... | vrsqrt14ps | — |
| `_mm512_sad_epu8` | Compute the absolute differences of packed unsigned 8-bit in... | vpsadbw | — |
| `_mm512_scalef_pd` | Scale the packed double-precision (64-bit) floating-point el... | vscalefpd | — |
| `_mm512_scalef_ps` | Scale the packed single-precision (32-bit) floating-point el... | vscalefps | — |
| `_mm512_scalef_round_pd` | Scale the packed double-precision (64-bit) floating-point el... | vscalefpd | — |
| `_mm512_scalef_round_ps` | Scale the packed single-precision (32-bit) floating-point el... | vscalefps | — |
| `_mm512_set1_epi16` | Broadcast the low packed 16-bit integer from a to all elemen... |  | — |
| `_mm512_set1_epi32` | Broadcast 32-bit integer `a` to all elements of `dst` |  | — |
| `_mm512_set1_epi64` | Broadcast 64-bit integer `a` to all elements of `dst` |  | — |
| `_mm512_set1_epi8` | Broadcast 8-bit integer a to all elements of dst |  | — |
| `_mm512_set1_pd` | Broadcast 64-bit float `a` to all elements of `dst` |  | — |
| `_mm512_set1_ps` | Broadcast 32-bit float `a` to all elements of `dst` |  | — |
| `_mm512_set4_epi32` | Set packed 32-bit integers in dst with the repeated 4 elemen... |  | — |
| `_mm512_set4_epi64` | Set packed 64-bit integers in dst with the repeated 4 elemen... |  | — |
| `_mm512_set4_pd` | Set packed double-precision (64-bit) floating-point elements... |  | — |
| `_mm512_set4_ps` | Set packed single-precision (32-bit) floating-point elements... |  | — |
| `_mm512_set_epi16` | Set packed 16-bit integers in dst with the supplied values |  | — |
| `_mm512_set_epi32` | Sets packed 32-bit integers in `dst` with the supplied value... |  | — |
| `_mm512_set_epi64` | Set packed 64-bit integers in dst with the supplied values |  | — |
| `_mm512_set_epi8` | Set packed 8-bit integers in dst with the supplied values |  | — |
| `_mm512_set_pd` | Set packed double-precision (64-bit) floating-point elements... |  | — |
| `_mm512_set_ps` | Sets packed 32-bit integers in `dst` with the supplied value... |  | — |
| `_mm512_setr4_epi32` | Set packed 32-bit integers in dst with the repeated 4 elemen... |  | — |
| `_mm512_setr4_epi64` | Set packed 64-bit integers in dst with the repeated 4 elemen... |  | — |
| `_mm512_setr4_pd` | Set packed double-precision (64-bit) floating-point elements... |  | — |
| `_mm512_setr4_ps` | Set packed single-precision (32-bit) floating-point elements... |  | — |
| `_mm512_setr_epi32` | Sets packed 32-bit integers in `dst` with the supplied value... |  | — |
| `_mm512_setr_epi64` | Set packed 64-bit integers in dst with the supplied values i... |  | — |
| `_mm512_setr_pd` | Set packed double-precision (64-bit) floating-point elements... |  | — |
| `_mm512_setr_ps` | Sets packed 32-bit integers in `dst` with the supplied value... |  | — |
| `_mm512_setzero` | Return vector of type `__m512` with all elements set to zero | vxorps | — |
| `_mm512_setzero_epi32` | Return vector of type `__m512i` with all elements set to zer... | vxorps | — |
| `_mm512_setzero_pd` | Returns vector of type `__m512d` with all elements set to ze... | vxorps | — |
| `_mm512_setzero_ps` | Returns vector of type `__m512` with all elements set to zer... | vxorps | — |
| `_mm512_setzero_si512` | Returns vector of type `__m512i` with all elements set to ze... | vxorps | — |
| `_mm512_shuffle_epi32` | Shuffle single-precision (32-bit) floating-point elements in... | vshufps | —/1/1 |
| `_mm512_shuffle_epi8` | Shuffle packed 8-bit integers in a according to shuffle cont... | vpshufb | —/1/1 |
| `_mm512_shuffle_f32x4` | Shuffle 128-bits (composed of 4 single-precision (32-bit) fl... | vshuff64x2 | —/1/1 |
| `_mm512_shuffle_f64x2` | Shuffle 128-bits (composed of 2 double-precision (64-bit) fl... | vshuff64x2 | —/1/1 |
| `_mm512_shuffle_i32x4` | Shuffle 128-bits (composed of 4 32-bit integers) selected by... | vshufi64x2 | —/1/1 |
| `_mm512_shuffle_i64x2` | Shuffle 128-bits (composed of 2 64-bit integers) selected by... | vshufi64x2 | —/1/1 |
| `_mm512_shuffle_pd` | Shuffle double-precision (64-bit) floating-point elements wi... | vshufpd | —/1/1 |
| `_mm512_shuffle_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vshufps | —/1/1 |
| `_mm512_shufflehi_epi16` | Shuffle 16-bit integers in the high 64 bits of 128-bit lanes... | vpshufhw | —/1/1 |
| `_mm512_shufflelo_epi16` | Shuffle 16-bit integers in the low 64 bits of 128-bit lanes ... | vpshuflw | —/1/1 |
| `_mm512_sll_epi16` | Shift packed 16-bit integers in a left by count while shifti... | vpsllw | —/1/1 |
| `_mm512_sll_epi32` | Shift packed 32-bit integers in a left by count while shifti... | vpslld | —/1/1 |
| `_mm512_sll_epi64` | Shift packed 64-bit integers in a left by count while shifti... | vpsllq | —/1/1 |
| `_mm512_slli_epi16` | Shift packed 16-bit integers in a left by imm8 while shiftin... | vpsllw | —/1/1 |
| `_mm512_slli_epi32` | Shift packed 32-bit integers in a left by imm8 while shiftin... | vpslld | —/1/1 |
| `_mm512_slli_epi64` | Shift packed 64-bit integers in a left by imm8 while shiftin... | vpsllq | —/1/1 |
| `_mm512_sllv_epi16` | Shift packed 16-bit integers in a left by the amount specifi... | vpsllvw | —/1/1 |
| `_mm512_sllv_epi32` | Shift packed 32-bit integers in a left by the amount specifi... | vpsllvd | —/1/1 |
| `_mm512_sllv_epi64` | Shift packed 64-bit integers in a left by the amount specifi... | vpsllvq | —/1/1 |
| `_mm512_sqrt_pd` | Compute the square root of packed double-precision (64-bit) ... | vsqrtpd | 16/14, 20/9 |
| `_mm512_sqrt_ps` | Compute the square root of packed single-precision (32-bit) ... | vsqrtps | 11/7, 14/5 |
| `_mm512_sqrt_round_pd` | Compute the square root of packed double-precision (64-bit) ... | vsqrtpd | — |
| `_mm512_sqrt_round_ps` | Compute the square root of packed single-precision (32-bit) ... | vsqrtps | — |
| `_mm512_sra_epi16` | Shift packed 16-bit integers in a right by count while shift... | vpsraw | —/1/1 |
| `_mm512_sra_epi32` | Shift packed 32-bit integers in a right by count while shift... | vpsrad | —/1/1 |
| `_mm512_sra_epi64` | Shift packed 64-bit integers in a right by count while shift... | vpsraq | —/1/1 |
| `_mm512_srai_epi16` | Shift packed 16-bit integers in a right by imm8 while shifti... | vpsraw | —/1/1 |
| `_mm512_srai_epi32` | Shift packed 32-bit integers in a right by imm8 while shifti... | vpsrad | —/1/1 |
| `_mm512_srai_epi64` | Shift packed 64-bit integers in a right by imm8 while shifti... | vpsraq | —/1/1 |
| `_mm512_srav_epi16` | Shift packed 16-bit integers in a right by the amount specif... | vpsravw | —/1/1 |
| `_mm512_srav_epi32` | Shift packed 32-bit integers in a right by the amount specif... | vpsravd | —/1/1 |
| `_mm512_srav_epi64` | Shift packed 64-bit integers in a right by the amount specif... | vpsravq | —/1/1 |
| `_mm512_srl_epi16` | Shift packed 16-bit integers in a right by count while shift... | vpsrlw | —/1/1 |
| `_mm512_srl_epi32` | Shift packed 32-bit integers in a right by count while shift... | vpsrld | —/1/1 |
| `_mm512_srl_epi64` | Shift packed 64-bit integers in a right by count while shift... | vpsrlq | —/1/1 |
| `_mm512_srli_epi16` | Shift packed 16-bit integers in a right by imm8 while shifti... | vpsrlw | —/1/1 |
| `_mm512_srli_epi32` | Shift packed 32-bit integers in a right by imm8 while shifti... | vpsrld | —/1/1 |
| `_mm512_srli_epi64` | Shift packed 64-bit integers in a right by imm8 while shifti... | vpsrlq | —/1/1 |
| `_mm512_srlv_epi16` | Shift packed 16-bit integers in a right by the amount specif... | vpsrlvw | —/1/1 |
| `_mm512_srlv_epi32` | Shift packed 32-bit integers in a right by the amount specif... | vpsrlvd | —/1/1 |
| `_mm512_srlv_epi64` | Shift packed 64-bit integers in a right by the amount specif... | vpsrlvq | —/1/1 |
| `_mm512_sub_epi16` | Subtract packed 16-bit integers in b from packed 16-bit inte... | vpsubw | —/1/1 |
| `_mm512_sub_epi32` | Subtract packed 32-bit integers in b from packed 32-bit inte... | vpsubd | —/1/1 |
| `_mm512_sub_epi64` | Subtract packed 64-bit integers in b from packed 64-bit inte... | vpsubq | —/1/1 |
| `_mm512_sub_epi8` | Subtract packed 8-bit integers in b from packed 8-bit intege... | vpsubb | —/1/1 |
| `_mm512_sub_pd` | Subtract packed double-precision (64-bit) floating-point ele... | vsubpd | — |
| `_mm512_sub_ps` | Subtract packed single-precision (32-bit) floating-point ele... | vsubps | — |
| `_mm512_sub_round_pd` | Subtract packed double-precision (64-bit) floating-point ele... | vsubpd | — |
| `_mm512_sub_round_ps` | Subtract packed single-precision (32-bit) floating-point ele... | vsubps | — |
| `_mm512_subs_epi16` | Subtract packed signed 16-bit integers in b from packed 16-b... | vpsubsw | —/1/1 |
| `_mm512_subs_epi8` | Subtract packed signed 8-bit integers in b from packed 8-bit... | vpsubsb | —/1/1 |
| `_mm512_subs_epu16` | Subtract packed unsigned 16-bit integers in b from packed un... | vpsubusw | — |
| `_mm512_subs_epu8` | Subtract packed unsigned 8-bit integers in b from packed uns... | vpsubusb | — |
| `_mm512_ternarylogic_epi32` | Bitwise ternary logic that provides the capability to implem... | vpternlogd | — |
| `_mm512_ternarylogic_epi64` | Bitwise ternary logic that provides the capability to implem... | vpternlogq | — |
| `_mm512_test_epi16_mask` | Compute the bitwise AND of packed 16-bit integers in a and b... | vptestmw | — |
| `_mm512_test_epi32_mask` | Compute the bitwise AND of packed 32-bit integers in a and b... | vptestmd | — |
| `_mm512_test_epi64_mask` | Compute the bitwise AND of packed 64-bit integers in a and b... | vptestmq | — |
| `_mm512_test_epi8_mask` | Compute the bitwise AND of packed 8-bit integers in a and b,... | vptestmb | — |
| `_mm512_testn_epi16_mask` | Compute the bitwise NAND of packed 16-bit integers in a and ... | vptestnmw | — |
| `_mm512_testn_epi32_mask` | Compute the bitwise NAND of packed 32-bit integers in a and ... | vptestnmd | — |
| `_mm512_testn_epi64_mask` | Compute the bitwise NAND of packed 64-bit integers in a and ... | vptestnmq | — |
| `_mm512_testn_epi8_mask` | Compute the bitwise NAND of packed 8-bit integers in a and b... | vptestnmb | — |
| `_mm512_undefined` | Return vector of type __m512 with indeterminate elements. De... |  | — |
| `_mm512_undefined_epi32` | Return vector of type __m512i with indeterminate elements. D... |  | — |
| `_mm512_undefined_pd` | Returns vector of type `__m512d` with indeterminate elements... |  | — |
| `_mm512_undefined_ps` | Returns vector of type `__m512` with indeterminate elements.... |  | — |
| `_mm512_unpackhi_epi16` | Unpack and interleave 16-bit integers from the high half of ... | vpunpckhwd | 1/1, 1/1 |
| `_mm512_unpackhi_epi32` | Unpack and interleave 32-bit integers from the high half of ... | vunpckhps | 1/1, 1/1 |
| `_mm512_unpackhi_epi64` | Unpack and interleave 64-bit integers from the high half of ... | vunpckhpd | 1/1, 1/1 |
| `_mm512_unpackhi_epi8` | Unpack and interleave 8-bit integers from the high half of e... | vpunpckhbw | 1/1, 1/1 |
| `_mm512_unpackhi_pd` | Unpack and interleave double-precision (64-bit) floating-poi... | vunpckhpd | 1/1, 1/1 |
| `_mm512_unpackhi_ps` | Unpack and interleave single-precision (32-bit) floating-poi... | vunpckhps | 1/1, 1/1 |
| `_mm512_unpacklo_epi16` | Unpack and interleave 16-bit integers from the low half of e... | vpunpcklwd | 1/1, 1/1 |
| `_mm512_unpacklo_epi32` | Unpack and interleave 32-bit integers from the low half of e... | vunpcklps | 1/1, 1/1 |
| `_mm512_unpacklo_epi64` | Unpack and interleave 64-bit integers from the low half of e... | vunpcklpd | 1/1, 1/1 |
| `_mm512_unpacklo_epi8` | Unpack and interleave 8-bit integers from the low half of ea... | vpunpcklbw | 1/1, 1/1 |
| `_mm512_unpacklo_pd` | Unpack and interleave double-precision (64-bit) floating-poi... | vunpcklpd | 1/1, 1/1 |
| `_mm512_unpacklo_ps` | Unpack and interleave single-precision (32-bit) floating-poi... | vunpcklps | 1/1, 1/1 |
| `_mm512_xor_epi32` | Compute the bitwise XOR of packed 32-bit integers in a and b... | vpxorq | —/1/1 |
| `_mm512_xor_epi64` | Compute the bitwise XOR of packed 64-bit integers in a and b... | vpxorq | —/1/1 |
| `_mm512_xor_pd` | Compute the bitwise XOR of packed double-precision (64-bit) ... | vxorp | —/1/1 |
| `_mm512_xor_ps` | Compute the bitwise XOR of packed single-precision (32-bit) ... | vxorps | —/1/1 |
| `_mm512_xor_si512` | Compute the bitwise XOR of 512 bits (representing integer da... | vpxorq | —/1/1 |
| `_mm512_zextpd128_pd512` | Cast vector of type __m128d to type __m512d; the upper 384 b... |  | — |
| `_mm512_zextpd256_pd512` | Cast vector of type __m256d to type __m512d; the upper 256 b... |  | — |
| `_mm512_zextps128_ps512` | Cast vector of type __m128 to type __m512; the upper 384 bit... |  | — |
| `_mm512_zextps256_ps512` | Cast vector of type __m256 to type __m512; the upper 256 bit... |  | — |
| `_mm512_zextsi128_si512` | Cast vector of type __m128i to type __m512i; the upper 384 b... |  | — |
| `_mm512_zextsi256_si512` | Cast vector of type __m256i to type __m512i; the upper 256 b... |  | — |
| `_mm_abs_epi64` | Compute the absolute value of packed signed 64-bit integers ... | vpabsq | — |
| `_mm_add_round_sd` | Add the lower double-precision (64-bit) floating-point eleme... | vaddsd | — |
| `_mm_add_round_ss` | Add the lower single-precision (32-bit) floating-point eleme... | vaddss | — |
| `_mm_alignr_epi32` | Concatenate a and b into a 32-byte immediate result, shift t... | vpalignr | — |
| `_mm_alignr_epi64` | Concatenate a and b into a 32-byte immediate result, shift t... | vpalignr | — |
| `_mm_broadcast_i32x2` | Broadcasts the lower 2 packed 32-bit integers from a to all ... |  | — |
| `_mm_broadcastmb_epi64` | Broadcast the low 8-bits from input mask k to all 64-bit ele... | vpbroadcast | — |
| `_mm_broadcastmw_epi32` | Broadcast the low 16-bits from input mask k to all 32-bit el... | vpbroadcast | — |
| `_mm_cmp_epi16_mask` | Compare packed signed 16-bit integers in a and b based on th... | vpcmp | — |
| `_mm_cmp_epi32_mask` | Compare packed signed 32-bit integers in a and b based on th... | vpcmp | — |
| `_mm_cmp_epi64_mask` | Compare packed signed 64-bit integers in a and b based on th... | vpcmp | — |
| `_mm_cmp_epi8_mask` | Compare packed signed 8-bit integers in a and b based on the... | vpcmp | — |
| `_mm_cmp_epu16_mask` | Compare packed unsigned 16-bit integers in a and b based on ... | vpcmp | — |
| `_mm_cmp_epu32_mask` | Compare packed unsigned 32-bit integers in a and b based on ... | vpcmp | — |
| `_mm_cmp_epu64_mask` | Compare packed unsigned 64-bit integers in a and b based on ... | vpcmp | — |
| `_mm_cmp_epu8_mask` | Compare packed unsigned 8-bit integers in a and b based on t... | vpcmp | — |
| `_mm_cmp_pd_mask` | Compare packed double-precision (64-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm_cmp_ps_mask` | Compare packed single-precision (32-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm_cmp_round_sd_mask` | Compare the lower double-precision (64-bit) floating-point e... | vcmp | — |
| `_mm_cmp_round_ss_mask` | Compare the lower single-precision (32-bit) floating-point e... | vcmp | — |
| `_mm_cmp_sd_mask` | Compare the lower double-precision (64-bit) floating-point e... | vcmp | — |
| `_mm_cmp_ss_mask` | Compare the lower single-precision (32-bit) floating-point e... | vcmp | — |
| `_mm_cmpeq_epi16_mask` | Compare packed signed 16-bit integers in a and b for equalit... | vpcmp | 1/1, 1/1 |
| `_mm_cmpeq_epi32_mask` | Compare packed 32-bit integers in a and b for equality, and ... | vpcmp | 1/1, 1/1 |
| `_mm_cmpeq_epi64_mask` | Compare packed 64-bit integers in a and b for equality, and ... | vpcmp | 1/1, 1/1 |
| `_mm_cmpeq_epi8_mask` | Compare packed signed 8-bit integers in a and b for equality... | vpcmp | 1/1, 1/1 |
| `_mm_cmpeq_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for equal... | vpcmp | — |
| `_mm_cmpeq_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for equal... | vpcmp | — |
| `_mm_cmpeq_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for equal... | vpcmp | — |
| `_mm_cmpeq_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for equali... | vpcmp | — |
| `_mm_cmpge_epi16_mask` | Compare packed signed 16-bit integers in a and b for greater... | vpcmp | — |
| `_mm_cmpge_epi32_mask` | Compare packed signed 32-bit integers in a and b for greater... | vpcmp | — |
| `_mm_cmpge_epi64_mask` | Compare packed signed 64-bit integers in a and b for greater... | vpcmp | — |
| `_mm_cmpge_epi8_mask` | Compare packed signed 8-bit integers in a and b for greater-... | vpcmp | — |
| `_mm_cmpge_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for great... | vpcmp | — |
| `_mm_cmpge_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for great... | vpcmp | — |
| `_mm_cmpge_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for great... | vpcmp | — |
| `_mm_cmpge_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for greate... | vpcmp | — |
| `_mm_cmpgt_epi16_mask` | Compare packed signed 16-bit integers in a and b for greater... | vpcmp | 1/1, 1/1 |
| `_mm_cmpgt_epi32_mask` | Compare packed signed 32-bit integers in a and b for greater... | vpcmp | 1/1, 1/1 |
| `_mm_cmpgt_epi64_mask` | Compare packed signed 64-bit integers in a and b for greater... | vpcmp | 1/1, 1/1 |
| `_mm_cmpgt_epi8_mask` | Compare packed signed 8-bit integers in a and b for greater-... | vpcmp | 1/1, 1/1 |
| `_mm_cmpgt_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for great... | vpcmp | — |
| `_mm_cmpgt_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for great... | vpcmp | — |
| `_mm_cmpgt_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for great... | vpcmp | — |
| `_mm_cmpgt_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for greate... | vpcmp | — |
| `_mm_cmple_epi16_mask` | Compare packed signed 16-bit integers in a and b for less-th... | vpcmp | — |
| `_mm_cmple_epi32_mask` | Compare packed signed 32-bit integers in a and b for less-th... | vpcmp | — |
| `_mm_cmple_epi64_mask` | Compare packed signed 64-bit integers in a and b for less-th... | vpcmp | — |
| `_mm_cmple_epi8_mask` | Compare packed signed 8-bit integers in a and b for less-tha... | vpcmp | — |
| `_mm_cmple_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for less-... | vpcmp | — |
| `_mm_cmple_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for less-... | vpcmp | — |
| `_mm_cmple_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for less-... | vpcmp | — |
| `_mm_cmple_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for less-t... | vpcmp | — |
| `_mm_cmplt_epi16_mask` | Compare packed signed 16-bit integers in a and b for less-th... | vpcmp | 1/1, 1/1 |
| `_mm_cmplt_epi32_mask` | Compare packed signed 32-bit integers in a and b for less-th... | vpcmp | 1/1, 1/1 |
| `_mm_cmplt_epi64_mask` | Compare packed signed 64-bit integers in a and b for less-th... | vpcmp | 1/1, 1/1 |
| `_mm_cmplt_epi8_mask` | Compare packed signed 8-bit integers in a and b for less-tha... | vpcmp | 1/1, 1/1 |
| `_mm_cmplt_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for less-... | vpcmp | — |
| `_mm_cmplt_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for less-... | vpcmp | — |
| `_mm_cmplt_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for less-... | vpcmp | — |
| `_mm_cmplt_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for less-t... | vpcmp | — |
| `_mm_cmpneq_epi16_mask` | Compare packed signed 16-bit integers in a and b for not-equ... | vpcmp | — |
| `_mm_cmpneq_epi32_mask` | Compare packed 32-bit integers in a and b for not-equal, and... | vpcmp | — |
| `_mm_cmpneq_epi64_mask` | Compare packed signed 64-bit integers in a and b for not-equ... | vpcmp | — |
| `_mm_cmpneq_epi8_mask` | Compare packed signed 8-bit integers in a and b for not-equa... | vpcmp | — |
| `_mm_cmpneq_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for not-e... | vpcmp | — |
| `_mm_cmpneq_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for not-e... | vpcmp | — |
| `_mm_cmpneq_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for not-e... | vpcmp | — |
| `_mm_cmpneq_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for not-eq... | vpcmp | — |
| `_mm_comi_round_sd` | Compare the lower double-precision (64-bit) floating-point e... | vcmp | — |
| `_mm_comi_round_ss` | Compare the lower single-precision (32-bit) floating-point e... | vcmp | — |
| `_mm_conflict_epi32` | Test each 32-bit element of a for equality with all other el... | vpconflictd | — |
| `_mm_conflict_epi64` | Test each 64-bit element of a for equality with all other el... | vpconflictq | — |
| `_mm_cvt_roundi32_ss` | Convert the signed 32-bit integer b to a single-precision (3... | vcvtsi2ss | 4/1, 3/1 |
| `_mm_cvt_roundi64_sd` | Convert the signed 64-bit integer b to a double-precision (6... | vcvtsi2sd | 4/1, 3/1 |
| `_mm_cvt_roundi64_ss` | Convert the signed 64-bit integer b to a single-precision (3... | vcvtsi2ss | 4/1, 3/1 |
| `_mm_cvt_roundsd_i32` | Convert the lower single-precision (32-bit) floating-point e... | vcvtsd2si | 4/1, 3/1 |
| `_mm_cvt_roundsd_i64` | Convert the lower double-precision (64-bit) floating-point e... | vcvtsd2si | 4/1, 3/1 |
| `_mm_cvt_roundsd_si32` | Convert the lower double-precision (64-bit) floating-point e... | vcvtsd2si | 4/1, 3/1 |
| `_mm_cvt_roundsd_si64` | Convert the lower double-precision (64-bit) floating-point e... | vcvtsd2si | 4/1, 3/1 |
| `_mm_cvt_roundsd_ss` | Convert the lower double-precision (64-bit) floating-point e... | vcvtsd2ss | 4/1, 3/1 |
| `_mm_cvt_roundsd_u32` | Convert the lower double-precision (64-bit) floating-point e... | vcvtsd2usi | 4/1, 3/1 |
| `_mm_cvt_roundsd_u64` | Convert the lower double-precision (64-bit) floating-point e... | vcvtsd2usi | 4/1, 3/1 |
| `_mm_cvt_roundsi32_ss` | Convert the signed 32-bit integer b to a single-precision (3... | vcvtsi2ss | 4/1, 3/1 |
| `_mm_cvt_roundsi64_sd` | Convert the signed 64-bit integer b to a double-precision (6... | vcvtsi2sd | 4/1, 3/1 |
| `_mm_cvt_roundsi64_ss` | Convert the signed 64-bit integer b to a single-precision (3... | vcvtsi2ss | 4/1, 3/1 |
| `_mm_cvt_roundss_i32` | Convert the lower single-precision (32-bit) floating-point e... | vcvtss2si | 4/1, 3/1 |
| `_mm_cvt_roundss_i64` | Convert the lower single-precision (32-bit) floating-point e... | vcvtss2si | 4/1, 3/1 |
| `_mm_cvt_roundss_sd` | Convert the lower single-precision (32-bit) floating-point e... | vcvtss2sd | 4/1, 3/1 |
| `_mm_cvt_roundss_si32` | Convert the lower single-precision (32-bit) floating-point e... | vcvtss2si | 4/1, 3/1 |
| `_mm_cvt_roundss_si64` | Convert the lower single-precision (32-bit) floating-point e... | vcvtss2si | 4/1, 3/1 |
| `_mm_cvt_roundss_u32` | Convert the lower single-precision (32-bit) floating-point e... | vcvtss2usi | 4/1, 3/1 |
| `_mm_cvt_roundss_u64` | Convert the lower single-precision (32-bit) floating-point e... | vcvtss2usi | 4/1, 3/1 |
| `_mm_cvt_roundu32_ss` | Convert the unsigned 32-bit integer b to a single-precision ... | vcvtusi2ss | 4/1, 3/1 |
| `_mm_cvt_roundu64_sd` | Convert the unsigned 64-bit integer b to a double-precision ... | vcvtusi2sd | 4/1, 3/1 |
| `_mm_cvt_roundu64_ss` | Convert the unsigned 64-bit integer b to a single-precision ... | vcvtusi2ss | 4/1, 3/1 |
| `_mm_cvtepi16_epi8` | Convert packed 16-bit integers in a to packed 8-bit integers... | vpmovwb | 4/1, 3/1 |
| `_mm_cvtepi32_epi16` | Convert packed 32-bit integers in a to packed 16-bit integer... | vpmovdw | 4/1, 3/1 |
| `_mm_cvtepi32_epi8` | Convert packed 32-bit integers in a to packed 8-bit integers... | vpmovdb | 4/1, 3/1 |
| `_mm_cvtepi64_epi16` | Convert packed 64-bit integers in a to packed 16-bit integer... | vpmovqw | 4/1, 3/1 |
| `_mm_cvtepi64_epi32` | Convert packed 64-bit integers in a to packed 32-bit integer... | vpmovqd | 4/1, 3/1 |
| `_mm_cvtepi64_epi8` | Convert packed 64-bit integers in a to packed 8-bit integers... | vpmovqb | 4/1, 3/1 |
| `_mm_cvtepi64_pd` | Convert packed signed 64-bit integers in a to packed double-... | vcvtqq2pd | 4/1, 3/1 |
| `_mm_cvtepi64_ps` | Convert packed signed 64-bit integers in a to packed single-... | vcvtqq2ps | 4/1, 3/1 |
| `_mm_cvtepu32_pd` | Convert packed unsigned 32-bit integers in a to packed doubl... | vcvtudq2pd | 4/1, 3/1 |
| `_mm_cvtepu64_pd` | Convert packed unsigned 64-bit integers in a to packed doubl... | vcvtuqq2pd | 4/1, 3/1 |
| `_mm_cvtepu64_ps` | Convert packed unsigned 64-bit integers in a to packed singl... | vcvtuqq2ps | 4/1, 3/1 |
| `_mm_cvti32_sd` | Convert the signed 32-bit integer b to a double-precision (6... | vcvtsi2sd | 4/1, 3/1 |
| `_mm_cvti32_ss` | Convert the signed 32-bit integer b to a single-precision (3... | vcvtsi2ss | 4/1, 3/1 |
| `_mm_cvti64_sd` | Convert the signed 64-bit integer b to a double-precision (6... | vcvtsi2sd | 4/1, 3/1 |
| `_mm_cvti64_ss` | Convert the signed 64-bit integer b to a single-precision (3... | vcvtsi2ss | 4/1, 3/1 |
| `_mm_cvtpd_epi64` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2qq | 4/1, 3/1 |
| `_mm_cvtpd_epu32` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2udq | 4/1, 3/1 |
| `_mm_cvtpd_epu64` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2uqq | 4/1, 3/1 |
| `_mm_cvtps_epi64` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2qq | 4/1, 3/1 |
| `_mm_cvtps_epu32` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2udq | 4/1, 3/1 |
| `_mm_cvtps_epu64` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2uqq | 4/1, 3/1 |
| `_mm_cvtsd_i32` | Convert the lower double-precision (64-bit) floating-point e... | vcvtsd2si | 4/1, 3/1 |
| `_mm_cvtsd_i64` | Convert the lower double-precision (64-bit) floating-point e... | vcvtsd2si | 4/1, 3/1 |
| `_mm_cvtsd_u32` | Convert the lower double-precision (64-bit) floating-point e... | vcvtsd2usi | 4/1, 3/1 |
| `_mm_cvtsd_u64` | Convert the lower double-precision (64-bit) floating-point e... | vcvtsd2usi | 4/1, 3/1 |
| `_mm_cvtsepi16_epi8` | Convert packed signed 16-bit integers in a to packed 8-bit i... | vpmovswb | 4/1, 3/1 |
| `_mm_cvtsepi32_epi16` | Convert packed signed 32-bit integers in a to packed 16-bit ... | vpmovsdw | 4/1, 3/1 |
| `_mm_cvtsepi32_epi8` | Convert packed signed 32-bit integers in a to packed 8-bit i... | vpmovsdb | 4/1, 3/1 |
| `_mm_cvtsepi64_epi16` | Convert packed signed 64-bit integers in a to packed 16-bit ... | vpmovsqw | 4/1, 3/1 |
| `_mm_cvtsepi64_epi32` | Convert packed signed 64-bit integers in a to packed 32-bit ... | vpmovsqd | 4/1, 3/1 |
| `_mm_cvtsepi64_epi8` | Convert packed signed 64-bit integers in a to packed 8-bit i... | vpmovsqb | 4/1, 3/1 |
| `_mm_cvtss_i32` | Convert the lower single-precision (32-bit) floating-point e... | vcvtss2si | 4/1, 3/1 |
| `_mm_cvtss_i64` | Convert the lower single-precision (32-bit) floating-point e... | vcvtss2si | 4/1, 3/1 |
| `_mm_cvtss_u32` | Convert the lower single-precision (32-bit) floating-point e... | vcvtss2usi | 4/1, 3/1 |
| `_mm_cvtss_u64` | Convert the lower single-precision (32-bit) floating-point e... | vcvtss2usi | 4/1, 3/1 |
| `_mm_cvtt_roundsd_i32` | Convert the lower double-precision (64-bit) floating-point e... | vcvttsd2si | 4/1, 3/1 |
| `_mm_cvtt_roundsd_i64` | Convert the lower double-precision (64-bit) floating-point e... | vcvttsd2si | 4/1, 3/1 |
| `_mm_cvtt_roundsd_si32` | Convert the lower double-precision (64-bit) floating-point e... | vcvttsd2si | 4/1, 3/1 |
| `_mm_cvtt_roundsd_si64` | Convert the lower double-precision (64-bit) floating-point e... | vcvttsd2si | 4/1, 3/1 |
| `_mm_cvtt_roundsd_u32` | Convert the lower double-precision (64-bit) floating-point e... | vcvttsd2usi | 4/1, 3/1 |
| `_mm_cvtt_roundsd_u64` | Convert the lower double-precision (64-bit) floating-point e... | vcvttsd2usi | 4/1, 3/1 |
| `_mm_cvtt_roundss_i32` | Convert the lower single-precision (32-bit) floating-point e... | vcvttss2si | 4/1, 3/1 |
| `_mm_cvtt_roundss_i64` | Convert the lower single-precision (32-bit) floating-point e... | vcvttss2si | 4/1, 3/1 |
| `_mm_cvtt_roundss_si32` | Convert the lower single-precision (32-bit) floating-point e... | vcvttss2si | 4/1, 3/1 |
| `_mm_cvtt_roundss_si64` | Convert the lower single-precision (32-bit) floating-point e... | vcvttss2si | 4/1, 3/1 |
| `_mm_cvtt_roundss_u32` | Convert the lower single-precision (32-bit) floating-point e... | vcvttss2usi | 4/1, 3/1 |
| `_mm_cvtt_roundss_u64` | Convert the lower single-precision (32-bit) floating-point e... | vcvttss2usi | 4/1, 3/1 |
| `_mm_cvttpd_epi64` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2qq | 4/1, 3/1 |
| `_mm_cvttpd_epu32` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2udq | 4/1, 3/1 |
| `_mm_cvttpd_epu64` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2uqq | 4/1, 3/1 |
| `_mm_cvttps_epi64` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2qq | 4/1, 3/1 |
| `_mm_cvttps_epu32` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2udq | 4/1, 3/1 |
| `_mm_cvttps_epu64` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2uqq | 4/1, 3/1 |
| `_mm_cvttsd_i32` | Convert the lower double-precision (64-bit) floating-point e... | vcvttsd2si | 4/1, 3/1 |
| `_mm_cvttsd_i64` | Convert the lower double-precision (64-bit) floating-point e... | vcvttsd2si | 4/1, 3/1 |
| `_mm_cvttsd_u32` | Convert the lower double-precision (64-bit) floating-point e... | vcvttsd2usi | 4/1, 3/1 |
| `_mm_cvttsd_u64` | Convert the lower double-precision (64-bit) floating-point e... | vcvttsd2usi | 4/1, 3/1 |
| `_mm_cvttss_i32` | Convert the lower single-precision (32-bit) floating-point e... | vcvttss2si | 4/1, 3/1 |
| `_mm_cvttss_i64` | Convert the lower single-precision (32-bit) floating-point e... | vcvttss2si | 4/1, 3/1 |
| `_mm_cvttss_u32` | Convert the lower single-precision (32-bit) floating-point e... | vcvttss2usi | 4/1, 3/1 |
| `_mm_cvttss_u64` | Convert the lower single-precision (32-bit) floating-point e... | vcvttss2usi | 4/1, 3/1 |
| `_mm_cvtu32_sd` | Convert the unsigned 32-bit integer b to a double-precision ... | vcvtusi2sd | 4/1, 3/1 |
| `_mm_cvtu32_ss` | Convert the unsigned 32-bit integer b to a single-precision ... | vcvtusi2ss | 4/1, 3/1 |
| `_mm_cvtu64_sd` | Convert the unsigned 64-bit integer b to a double-precision ... | vcvtusi2sd | 4/1, 3/1 |
| `_mm_cvtu64_ss` | Convert the unsigned 64-bit integer b to a single-precision ... | vcvtusi2ss | 4/1, 3/1 |
| `_mm_cvtusepi16_epi8` | Convert packed unsigned 16-bit integers in a to packed unsig... | vpmovuswb | 4/1, 3/1 |
| `_mm_cvtusepi32_epi16` | Convert packed unsigned 32-bit integers in a to packed unsig... | vpmovusdw | 4/1, 3/1 |
| `_mm_cvtusepi32_epi8` | Convert packed unsigned 32-bit integers in a to packed unsig... | vpmovusdb | 4/1, 3/1 |
| `_mm_cvtusepi64_epi16` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqw | 4/1, 3/1 |
| `_mm_cvtusepi64_epi32` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqd | 4/1, 3/1 |
| `_mm_cvtusepi64_epi8` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqb | 4/1, 3/1 |
| `_mm_dbsad_epu8` | Compute the sum of absolute differences (SADs) of quadruplet... | vdbpsadbw | — |
| `_mm_div_round_sd` | Divide the lower double-precision (64-bit) floating-point el... | vdivsd | — |
| `_mm_div_round_ss` | Divide the lower single-precision (32-bit) floating-point el... | vdivss | — |
| `_mm_fixupimm_pd` | Fix up packed double-precision (64-bit) floating-point eleme... | vfixupimmpd | — |
| `_mm_fixupimm_ps` | Fix up packed single-precision (32-bit) floating-point eleme... | vfixupimmps | — |
| `_mm_fixupimm_round_sd` | Fix up the lower double-precision (64-bit) floating-point el... | vfixupimmsd | — |
| `_mm_fixupimm_round_ss` | Fix up the lower single-precision (32-bit) floating-point el... | vfixupimmss | — |
| `_mm_fixupimm_sd` | Fix up the lower double-precision (64-bit) floating-point el... | vfixupimmsd | — |
| `_mm_fixupimm_ss` | Fix up the lower single-precision (32-bit) floating-point el... | vfixupimmss | — |
| `_mm_fmadd_round_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfmadd | 5/1, 4/1 |
| `_mm_fmadd_round_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfmadd | 5/1, 4/1 |
| `_mm_fmsub_round_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfmsub | 5/1, 4/1 |
| `_mm_fmsub_round_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfmsub | 5/1, 4/1 |
| `_mm_fnmadd_round_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfnmadd | 5/1, 4/1 |
| `_mm_fnmadd_round_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfnmadd | 5/1, 4/1 |
| `_mm_fnmsub_round_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfnmsub | 5/1, 4/1 |
| `_mm_fnmsub_round_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfnmsub | 5/1, 4/1 |
| `_mm_fpclass_pd_mask` | Test packed double-precision (64-bit) floating-point element... | vfpclasspd | — |
| `_mm_fpclass_ps_mask` | Test packed single-precision (32-bit) floating-point element... | vfpclassps | — |
| `_mm_fpclass_sd_mask` | Test the lower double-precision (64-bit) floating-point elem... | vfpclasssd | — |
| `_mm_fpclass_ss_mask` | Test the lower single-precision (32-bit) floating-point elem... | vfpclassss | — |
| `_mm_getexp_pd` | Convert the exponent of each packed double-precision (64-bit... | vgetexppd | — |
| `_mm_getexp_ps` | Convert the exponent of each packed single-precision (32-bit... | vgetexpps | — |
| `_mm_getexp_round_sd` | Convert the exponent of the lower double-precision (64-bit) ... | vgetexpsd | — |
| `_mm_getexp_round_ss` | Convert the exponent of the lower single-precision (32-bit) ... | vgetexpss | — |
| `_mm_getexp_sd` | Convert the exponent of the lower double-precision (64-bit) ... | vgetexpsd | — |
| `_mm_getexp_ss` | Convert the exponent of the lower single-precision (32-bit) ... | vgetexpss | — |
| `_mm_getmant_pd` | Normalize the mantissas of packed double-precision (64-bit) ... | vgetmantpd | — |
| `_mm_getmant_ps` | Normalize the mantissas of packed single-precision (32-bit) ... | vgetmantps | — |
| `_mm_getmant_round_sd` | Normalize the mantissas of the lower double-precision (64-bi... | vgetmantsd | — |
| `_mm_getmant_round_ss` | Normalize the mantissas of the lower single-precision (32-bi... | vgetmantss | — |
| `_mm_getmant_sd` | Normalize the mantissas of the lower double-precision (64-bi... | vgetmantsd | — |
| `_mm_getmant_ss` | Normalize the mantissas of the lower single-precision (32-bi... | vgetmantss | — |
| `_mm_lzcnt_epi32` | Counts the number of leading zero bits in each packed 32-bit... | vplzcntd | — |
| `_mm_lzcnt_epi64` | Counts the number of leading zero bits in each packed 64-bit... | vplzcntq | — |
| `_mm_mask2_permutex2var_epi16` | Shuffle 16-bit integers in a and b across lanes using the co... | vpermi2w | 1/1, 1/1 |
| `_mm_mask2_permutex2var_epi32` | Shuffle 32-bit integers in a and b across lanes using the co... | vpermi2d | 1/1, 1/1 |
| `_mm_mask2_permutex2var_epi64` | Shuffle 64-bit integers in a and b across lanes using the co... | vpermi2q | 1/1, 1/1 |
| `_mm_mask2_permutex2var_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vperm | 1/1, 1/1 |
| `_mm_mask2_permutex2var_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vperm | 1/1, 1/1 |
| `_mm_mask3_fmadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmadd | 5/1, 4/1 |
| `_mm_mask3_fmadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmadd | 5/1, 4/1 |
| `_mm_mask3_fmadd_round_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfmadd | 5/1, 4/1 |
| `_mm_mask3_fmadd_round_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfmadd | 5/1, 4/1 |
| `_mm_mask3_fmadd_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfmadd | 5/1, 4/1 |
| `_mm_mask3_fmadd_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfmadd | 5/1, 4/1 |
| `_mm_mask3_fmaddsub_pd` | Multiply packed single-precision (32-bit) floating-point ele... | vfmaddsub | 5/1, 4/1 |
| `_mm_mask3_fmaddsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmaddsub | 5/1, 4/1 |
| `_mm_mask3_fmsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsub | 5/1, 4/1 |
| `_mm_mask3_fmsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsub | 5/1, 4/1 |
| `_mm_mask3_fmsub_round_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfmsub | 5/1, 4/1 |
| `_mm_mask3_fmsub_round_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfmsub | 5/1, 4/1 |
| `_mm_mask3_fmsub_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfmsub | 5/1, 4/1 |
| `_mm_mask3_fmsub_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfmsub | 5/1, 4/1 |
| `_mm_mask3_fmsubadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsubadd | 5/1, 4/1 |
| `_mm_mask3_fmsubadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsubadd | 5/1, 4/1 |
| `_mm_mask3_fnmadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmadd | 5/1, 4/1 |
| `_mm_mask3_fnmadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmadd | 5/1, 4/1 |
| `_mm_mask3_fnmadd_round_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfnmadd | 5/1, 4/1 |
| `_mm_mask3_fnmadd_round_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfnmadd | 5/1, 4/1 |
| `_mm_mask3_fnmadd_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfnmadd | 5/1, 4/1 |
| `_mm_mask3_fnmadd_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfnmadd | 5/1, 4/1 |
| `_mm_mask3_fnmsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmsub | 5/1, 4/1 |
| `_mm_mask3_fnmsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmsub | 5/1, 4/1 |
| `_mm_mask3_fnmsub_round_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfnmsub | 5/1, 4/1 |
| `_mm_mask3_fnmsub_round_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfnmsub | 5/1, 4/1 |
| `_mm_mask3_fnmsub_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfnmsub | 5/1, 4/1 |
| `_mm_mask3_fnmsub_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfnmsub | 5/1, 4/1 |
| `_mm_mask_abs_epi16` | Compute the absolute value of packed signed 16-bit integers ... | vpabsw | — |
| `_mm_mask_abs_epi32` | Compute the absolute value of packed signed 32-bit integers ... | vpabsd | — |
| `_mm_mask_abs_epi64` | Compute the absolute value of packed signed 64-bit integers ... | vpabsq | — |
| `_mm_mask_abs_epi8` | Compute the absolute value of packed signed 8-bit integers i... | vpabsb | — |
| `_mm_mask_add_epi16` | Add packed 16-bit integers in a and b, and store the results... | vpaddw | 1/1, 1/1 |
| `_mm_mask_add_epi32` | Add packed 32-bit integers in a and b, and store the results... | vpaddd | 1/1, 1/1 |
| `_mm_mask_add_epi64` | Add packed 64-bit integers in a and b, and store the results... | vpaddq | 1/1, 1/1 |
| `_mm_mask_add_epi8` | Add packed 8-bit integers in a and b, and store the results ... | vpaddb | 1/1, 1/1 |
| `_mm_mask_add_pd` | Add packed double-precision (64-bit) floating-point elements... | vaddpd | 3/1, 3/1 |
| `_mm_mask_add_ps` | Add packed single-precision (32-bit) floating-point elements... | vaddps | 3/1, 3/1 |
| `_mm_mask_add_round_sd` | Add the lower double-precision (64-bit) floating-point eleme... | vaddsd | — |
| `_mm_mask_add_round_ss` | Add the lower single-precision (32-bit) floating-point eleme... | vaddss | — |
| `_mm_mask_add_sd` | Add the lower double-precision (64-bit) floating-point eleme... | vaddsd | 3/1, 3/1 |
| `_mm_mask_add_ss` | Add the lower single-precision (32-bit) floating-point eleme... | vaddss | 3/1, 3/1 |
| `_mm_mask_adds_epi16` | Add packed signed 16-bit integers in a and b using saturatio... | vpaddsw | 1/1, 1/1 |
| `_mm_mask_adds_epi8` | Add packed signed 8-bit integers in a and b using saturation... | vpaddsb | 1/1, 1/1 |
| `_mm_mask_adds_epu16` | Add packed unsigned 16-bit integers in a and b using saturat... | vpaddusw | — |
| `_mm_mask_adds_epu8` | Add packed unsigned 8-bit integers in a and b using saturati... | vpaddusb | — |
| `_mm_mask_alignr_epi32` | Concatenate a and b into a 32-byte immediate result, shift t... | valignd | — |
| `_mm_mask_alignr_epi64` | Concatenate a and b into a 32-byte immediate result, shift t... | valignq | — |
| `_mm_mask_alignr_epi8` | Concatenate pairs of 16-byte blocks in a and b into a 32-byt... | vpalignr | — |
| `_mm_mask_and_epi32` | Performs element-by-element bitwise AND between packed 32-bi... | vpandd | 1/1, 1/1 |
| `_mm_mask_and_epi64` | Compute the bitwise AND of packed 64-bit integers in a and b... | vpandq | 1/1, 1/1 |
| `_mm_mask_and_pd` | Compute the bitwise AND of packed double-precision (64-bit) ... | vandpd | 1/1, 1/1 |
| `_mm_mask_and_ps` | Compute the bitwise AND of packed single-precision (32-bit) ... | vandps | 1/1, 1/1 |
| `_mm_mask_andnot_epi32` | Compute the bitwise NOT of packed 32-bit integers in a and t... | vpandnd | 1/1, 1/1 |
| `_mm_mask_andnot_epi64` | Compute the bitwise NOT of packed 64-bit integers in a and t... | vpandnq | 1/1, 1/1 |
| `_mm_mask_andnot_pd` | Compute the bitwise NOT of packed double-precision (64-bit) ... | vandnpd | 1/1, 1/1 |
| `_mm_mask_andnot_ps` | Compute the bitwise NOT of packed single-precision (32-bit) ... | vandnps | 1/1, 1/1 |
| `_mm_mask_avg_epu16` | Average packed unsigned 16-bit integers in a and b, and stor... | vpavgw | — |
| `_mm_mask_avg_epu8` | Average packed unsigned 8-bit integers in a and b, and store... | vpavgb | — |
| `_mm_mask_blend_epi16` | Blend packed 16-bit integers from a and b using control mask... | vmovdqu16 | 1/1, 1/1 |
| `_mm_mask_blend_epi32` | Blend packed 32-bit integers from a and b using control mask... | vmovdqa32 | 1/1, 1/1 |
| `_mm_mask_blend_epi64` | Blend packed 64-bit integers from a and b using control mask... | vmovdqa64 | 1/1, 1/1 |
| `_mm_mask_blend_epi8` | Blend packed 8-bit integers from a and b using control mask ... | vmovdqu8 | 1/1, 1/1 |
| `_mm_mask_blend_pd` | Blend packed double-precision (64-bit) floating-point elemen... | vmovapd | 1/1, 1/1 |
| `_mm_mask_blend_ps` | Blend packed single-precision (32-bit) floating-point elemen... | vmovaps | 1/1, 1/1 |
| `_mm_mask_broadcast_i32x2` | Broadcasts the lower 2 packed 32-bit integers from a to all ... | vbroadcasti32x2 | — |
| `_mm_mask_broadcastb_epi8` | Broadcast the low packed 8-bit integer from a to all element... | vpbroadcastb | — |
| `_mm_mask_broadcastd_epi32` | Broadcast the low packed 32-bit integer from a to all elemen... | vpbroadcast | — |
| `_mm_mask_broadcastq_epi64` | Broadcast the low packed 64-bit integer from a to all elemen... | vpbroadcast | — |
| `_mm_mask_broadcastss_ps` | Broadcast the low single-precision (32-bit) floating-point e... | vbroadcastss | — |
| `_mm_mask_broadcastw_epi16` | Broadcast the low packed 16-bit integer from a to all elemen... | vpbroadcastw | — |
| `_mm_mask_cmp_epi16_mask` | Compare packed signed 16-bit integers in a and b based on th... | vpcmp | — |
| `_mm_mask_cmp_epi32_mask` | Compare packed signed 32-bit integers in a and b based on th... | vpcmp | — |
| `_mm_mask_cmp_epi64_mask` | Compare packed signed 64-bit integers in a and b based on th... | vpcmp | — |
| `_mm_mask_cmp_epi8_mask` | Compare packed signed 8-bit integers in a and b based on the... | vpcmp | — |
| `_mm_mask_cmp_epu16_mask` | Compare packed unsigned 16-bit integers in a and b based on ... | vpcmp | — |
| `_mm_mask_cmp_epu32_mask` | Compare packed unsigned 32-bit integers in a and b based on ... | vpcmp | — |
| `_mm_mask_cmp_epu64_mask` | Compare packed unsigned 64-bit integers in a and b based on ... | vpcmp | — |
| `_mm_mask_cmp_epu8_mask` | Compare packed unsigned 8-bit integers in a and b based on t... | vpcmp | — |
| `_mm_mask_cmp_pd_mask` | Compare packed double-precision (64-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm_mask_cmp_ps_mask` | Compare packed single-precision (32-bit) floating-point elem... | vcmp | 3/1, 3/1 |
| `_mm_mask_cmp_round_sd_mask` | Compare the lower double-precision (64-bit) floating-point e... | vcmp | — |
| `_mm_mask_cmp_round_ss_mask` | Compare the lower single-precision (32-bit) floating-point e... | vcmp | — |
| `_mm_mask_cmp_sd_mask` | Compare the lower double-precision (64-bit) floating-point e... | vcmp | — |
| `_mm_mask_cmp_ss_mask` | Compare the lower single-precision (32-bit) floating-point e... | vcmp | — |
| `_mm_mask_cmpeq_epi16_mask` | Compare packed signed 16-bit integers in a and b for equalit... | vpcmp | 1/1, 1/1 |
| `_mm_mask_cmpeq_epi32_mask` | Compare packed 32-bit integers in a and b for equality, and ... | vpcmp | 1/1, 1/1 |
| `_mm_mask_cmpeq_epi64_mask` | Compare packed 64-bit integers in a and b for equality, and ... | vpcmp | 1/1, 1/1 |
| `_mm_mask_cmpeq_epi8_mask` | Compare packed signed 8-bit integers in a and b for equality... | vpcmp | 1/1, 1/1 |
| `_mm_mask_cmpeq_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for equal... | vpcmp | — |
| `_mm_mask_cmpeq_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for equal... | vpcmp | — |
| `_mm_mask_cmpeq_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for equal... | vpcmp | — |
| `_mm_mask_cmpeq_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for equali... | vpcmp | — |
| `_mm_mask_cmpge_epi16_mask` | Compare packed signed 16-bit integers in a and b for greater... | vpcmp | — |
| `_mm_mask_cmpge_epi32_mask` | Compare packed signed 32-bit integers in a and b for greater... | vpcmp | — |
| `_mm_mask_cmpge_epi64_mask` | Compare packed signed 64-bit integers in a and b for greater... | vpcmp | — |
| `_mm_mask_cmpge_epi8_mask` | Compare packed signed 8-bit integers in a and b for greater-... | vpcmp | — |
| `_mm_mask_cmpge_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for great... | vpcmp | — |
| `_mm_mask_cmpge_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for great... | vpcmp | — |
| `_mm_mask_cmpge_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for great... | vpcmp | — |
| `_mm_mask_cmpge_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for greate... | vpcmp | — |
| `_mm_mask_cmpgt_epi16_mask` | Compare packed signed 16-bit integers in a and b for greater... | vpcmp | 1/1, 1/1 |
| `_mm_mask_cmpgt_epi32_mask` | Compare packed signed 32-bit integers in a and b for greater... | vpcmp | 1/1, 1/1 |
| `_mm_mask_cmpgt_epi64_mask` | Compare packed signed 64-bit integers in a and b for greater... | vpcmp | 1/1, 1/1 |
| `_mm_mask_cmpgt_epi8_mask` | Compare packed signed 8-bit integers in a and b for greater-... | vpcmp | 1/1, 1/1 |
| `_mm_mask_cmpgt_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for great... | vpcmp | — |
| `_mm_mask_cmpgt_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for great... | vpcmp | — |
| `_mm_mask_cmpgt_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for great... | vpcmp | — |
| `_mm_mask_cmpgt_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for greate... | vpcmp | — |
| `_mm_mask_cmple_epi16_mask` | Compare packed signed 16-bit integers in a and b for less-th... | vpcmp | — |
| `_mm_mask_cmple_epi32_mask` | Compare packed signed 32-bit integers in a and b for less-th... | vpcmp | — |
| `_mm_mask_cmple_epi64_mask` | Compare packed signed 64-bit integers in a and b for less-th... | vpcmp | — |
| `_mm_mask_cmple_epi8_mask` | Compare packed signed 8-bit integers in a and b for less-tha... | vpcmp | — |
| `_mm_mask_cmple_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for less-... | vpcmp | — |
| `_mm_mask_cmple_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for less-... | vpcmp | — |
| `_mm_mask_cmple_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for less-... | vpcmp | — |
| `_mm_mask_cmple_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for less-t... | vpcmp | — |
| `_mm_mask_cmplt_epi16_mask` | Compare packed signed 16-bit integers in a and b for less-th... | vpcmp | 1/1, 1/1 |
| `_mm_mask_cmplt_epi32_mask` | Compare packed signed 32-bit integers in a and b for less-th... | vpcmp | 1/1, 1/1 |
| `_mm_mask_cmplt_epi64_mask` | Compare packed signed 64-bit integers in a and b for less-th... | vpcmp | 1/1, 1/1 |
| `_mm_mask_cmplt_epi8_mask` | Compare packed signed 8-bit integers in a and b for less-tha... | vpcmp | 1/1, 1/1 |
| `_mm_mask_cmplt_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for less-... | vpcmp | — |
| `_mm_mask_cmplt_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for less-... | vpcmp | — |
| `_mm_mask_cmplt_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for less-... | vpcmp | — |
| `_mm_mask_cmplt_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for less-t... | vpcmp | — |
| `_mm_mask_cmpneq_epi16_mask` | Compare packed signed 16-bit integers in a and b for not-equ... | vpcmp | — |
| `_mm_mask_cmpneq_epi32_mask` | Compare packed 32-bit integers in a and b for not-equal, and... | vpcmp | — |
| `_mm_mask_cmpneq_epi64_mask` | Compare packed signed 64-bit integers in a and b for not-equ... | vpcmp | — |
| `_mm_mask_cmpneq_epi8_mask` | Compare packed signed 8-bit integers in a and b for not-equa... | vpcmp | — |
| `_mm_mask_cmpneq_epu16_mask` | Compare packed unsigned 16-bit integers in a and b for not-e... | vpcmp | — |
| `_mm_mask_cmpneq_epu32_mask` | Compare packed unsigned 32-bit integers in a and b for not-e... | vpcmp | — |
| `_mm_mask_cmpneq_epu64_mask` | Compare packed unsigned 64-bit integers in a and b for not-e... | vpcmp | — |
| `_mm_mask_cmpneq_epu8_mask` | Compare packed unsigned 8-bit integers in a and b for not-eq... | vpcmp | — |
| `_mm_mask_compress_epi32` | Contiguously store the active 32-bit integers in a (those wi... | vpcompressd | — |
| `_mm_mask_compress_epi64` | Contiguously store the active 64-bit integers in a (those wi... | vpcompressq | — |
| `_mm_mask_compress_pd` | Contiguously store the active double-precision (64-bit) floa... | vcompresspd | — |
| `_mm_mask_compress_ps` | Contiguously store the active single-precision (32-bit) floa... | vcompressps | — |
| `_mm_mask_conflict_epi32` | Test each 32-bit element of a for equality with all other el... | vpconflictd | — |
| `_mm_mask_conflict_epi64` | Test each 64-bit element of a for equality with all other el... | vpconflictq | — |
| `_mm_mask_cvt_roundps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2ph | 4/1, 3/1 |
| `_mm_mask_cvt_roundsd_ss` | Convert the lower double-precision (64-bit) floating-point e... | vcvtsd2ss | 4/1, 3/1 |
| `_mm_mask_cvt_roundss_sd` | Convert the lower single-precision (32-bit) floating-point e... | vcvtss2sd | 4/1, 3/1 |
| `_mm_mask_cvtepi16_epi32` | Sign extend packed 16-bit integers in a to packed 32-bit int... | vpmovsxwd | 4/1, 3/1 |
| `_mm_mask_cvtepi16_epi64` | Sign extend packed 16-bit integers in a to packed 64-bit int... | vpmovsxwq | 4/1, 3/1 |
| `_mm_mask_cvtepi16_epi8` | Convert packed 16-bit integers in a to packed 8-bit integers... | vpmovwb | 4/1, 3/1 |
| `_mm_mask_cvtepi32_epi16` | Convert packed 32-bit integers in a to packed 16-bit integer... | vpmovdw | 4/1, 3/1 |
| `_mm_mask_cvtepi32_epi64` | Sign extend packed 32-bit integers in a to packed 64-bit int... | vpmovsxdq | 4/1, 3/1 |
| `_mm_mask_cvtepi32_epi8` | Convert packed 32-bit integers in a to packed 8-bit integers... | vpmovdb | 4/1, 3/1 |
| `_mm_mask_cvtepi32_pd` | Convert packed signed 32-bit integers in a to packed double-... | vcvtdq2pd | 4/1, 3/1 |
| `_mm_mask_cvtepi32_ps` | Convert packed signed 32-bit integers in a to packed single-... | vcvtdq2ps | 4/1, 3/1 |
| `_mm_mask_cvtepi64_epi16` | Convert packed 64-bit integers in a to packed 16-bit integer... | vpmovqw | 4/1, 3/1 |
| `_mm_mask_cvtepi64_epi32` | Convert packed 64-bit integers in a to packed 32-bit integer... | vpmovqd | 4/1, 3/1 |
| `_mm_mask_cvtepi64_epi8` | Convert packed 64-bit integers in a to packed 8-bit integers... | vpmovqb | 4/1, 3/1 |
| `_mm_mask_cvtepi64_pd` | Convert packed signed 64-bit integers in a to packed double-... | vcvtqq2pd | 4/1, 3/1 |
| `_mm_mask_cvtepi64_ps` | Convert packed signed 64-bit integers in a to packed single-... | vcvtqq2ps | 4/1, 3/1 |
| `_mm_mask_cvtepi8_epi16` | Sign extend packed 8-bit integers in a to packed 16-bit inte... | vpmovsxbw | 4/1, 3/1 |
| `_mm_mask_cvtepi8_epi32` | Sign extend packed 8-bit integers in a to packed 32-bit inte... | vpmovsxbd | 4/1, 3/1 |
| `_mm_mask_cvtepi8_epi64` | Sign extend packed 8-bit integers in the low 2 bytes of a to... | vpmovsxbq | 4/1, 3/1 |
| `_mm_mask_cvtepu16_epi32` | Zero extend packed unsigned 16-bit integers in a to packed 3... | vpmovzxwd | 4/1, 3/1 |
| `_mm_mask_cvtepu16_epi64` | Zero extend packed unsigned 16-bit integers in the low 4 byt... | vpmovzxwq | 4/1, 3/1 |
| `_mm_mask_cvtepu32_epi64` | Zero extend packed unsigned 32-bit integers in a to packed 6... | vpmovzxdq | 4/1, 3/1 |
| `_mm_mask_cvtepu32_pd` | Convert packed unsigned 32-bit integers in a to packed doubl... | vcvtudq2pd | 4/1, 3/1 |
| `_mm_mask_cvtepu64_pd` | Convert packed unsigned 64-bit integers in a to packed doubl... | vcvtuqq2pd | 4/1, 3/1 |
| `_mm_mask_cvtepu64_ps` | Convert packed unsigned 64-bit integers in a to packed singl... | vcvtuqq2ps | 4/1, 3/1 |
| `_mm_mask_cvtepu8_epi16` | Zero extend packed unsigned 8-bit integers in a to packed 16... | vpmovzxbw | 4/1, 3/1 |
| `_mm_mask_cvtepu8_epi32` | Zero extend packed unsigned 8-bit integers in the low 4 byte... | vpmovzxbd | 4/1, 3/1 |
| `_mm_mask_cvtepu8_epi64` | Zero extend packed unsigned 8-bit integers in the low 2 byte... | vpmovzxbq | 4/1, 3/1 |
| `_mm_mask_cvtpd_epi32` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2dq | 4/1, 3/1 |
| `_mm_mask_cvtpd_epi64` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2qq | 4/1, 3/1 |
| `_mm_mask_cvtpd_epu32` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2udq | 4/1, 3/1 |
| `_mm_mask_cvtpd_epu64` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2uqq | 4/1, 3/1 |
| `_mm_mask_cvtpd_ps` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2ps | 4/1, 3/1 |
| `_mm_mask_cvtph_ps` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2ps | 4/1, 3/1 |
| `_mm_mask_cvtps_epi32` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2dq | 4/1, 3/1 |
| `_mm_mask_cvtps_epi64` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2qq | 4/1, 3/1 |
| `_mm_mask_cvtps_epu32` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2udq | 4/1, 3/1 |
| `_mm_mask_cvtps_epu64` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2uqq | 4/1, 3/1 |
| `_mm_mask_cvtps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2ph | 4/1, 3/1 |
| `_mm_mask_cvtsd_ss` | Convert the lower double-precision (64-bit) floating-point e... | vcvtsd2ss | 4/1, 3/1 |
| `_mm_mask_cvtsepi16_epi8` | Convert packed signed 16-bit integers in a to packed 8-bit i... | vpmovswb | 4/1, 3/1 |
| `_mm_mask_cvtsepi32_epi16` | Convert packed signed 32-bit integers in a to packed 16-bit ... | vpmovsdw | 4/1, 3/1 |
| `_mm_mask_cvtsepi32_epi8` | Convert packed signed 32-bit integers in a to packed 8-bit i... | vpmovsdb | 4/1, 3/1 |
| `_mm_mask_cvtsepi64_epi16` | Convert packed signed 64-bit integers in a to packed 16-bit ... | vpmovsqw | 4/1, 3/1 |
| `_mm_mask_cvtsepi64_epi32` | Convert packed signed 64-bit integers in a to packed 32-bit ... | vpmovsqd | 4/1, 3/1 |
| `_mm_mask_cvtsepi64_epi8` | Convert packed signed 64-bit integers in a to packed 8-bit i... | vpmovsqb | 4/1, 3/1 |
| `_mm_mask_cvtss_sd` | Convert the lower single-precision (32-bit) floating-point e... | vcvtss2sd | 4/1, 3/1 |
| `_mm_mask_cvttpd_epi32` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2dq | 4/1, 3/1 |
| `_mm_mask_cvttpd_epi64` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2qq | 4/1, 3/1 |
| `_mm_mask_cvttpd_epu32` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2udq | 4/1, 3/1 |
| `_mm_mask_cvttpd_epu64` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2uqq | 4/1, 3/1 |
| `_mm_mask_cvttps_epi32` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2dq | 4/1, 3/1 |
| `_mm_mask_cvttps_epi64` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2qq | 4/1, 3/1 |
| `_mm_mask_cvttps_epu32` | Convert packed double-precision (32-bit) floating-point elem... | vcvttps2udq | 4/1, 3/1 |
| `_mm_mask_cvttps_epu64` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2uqq | 4/1, 3/1 |
| `_mm_mask_cvtusepi16_epi8` | Convert packed unsigned 16-bit integers in a to packed unsig... | vpmovuswb | 4/1, 3/1 |
| `_mm_mask_cvtusepi32_epi16` | Convert packed unsigned 32-bit integers in a to packed unsig... | vpmovusdw | 4/1, 3/1 |
| `_mm_mask_cvtusepi32_epi8` | Convert packed unsigned 32-bit integers in a to packed unsig... | vpmovusdb | 4/1, 3/1 |
| `_mm_mask_cvtusepi64_epi16` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqw | 4/1, 3/1 |
| `_mm_mask_cvtusepi64_epi32` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqd | 4/1, 3/1 |
| `_mm_mask_cvtusepi64_epi8` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqb | 4/1, 3/1 |
| `_mm_mask_dbsad_epu8` | Compute the sum of absolute differences (SADs) of quadruplet... | vdbpsadbw | — |
| `_mm_mask_div_pd` | Divide packed double-precision (64-bit) floating-point eleme... | vdivpd | 20/14, 13/7 |
| `_mm_mask_div_ps` | Divide packed single-precision (32-bit) floating-point eleme... | vdivps | 13/7, 10/4 |
| `_mm_mask_div_round_sd` | Divide the lower double-precision (64-bit) floating-point el... | vdivsd | — |
| `_mm_mask_div_round_ss` | Divide the lower single-precision (32-bit) floating-point el... | vdivss | — |
| `_mm_mask_div_sd` | Divide the lower double-precision (64-bit) floating-point el... | vdivsd | 20/14, 13/7 |
| `_mm_mask_div_ss` | Divide the lower single-precision (32-bit) floating-point el... | vdivss | 13/7, 10/4 |
| `_mm_mask_expand_epi32` | Load contiguous active 32-bit integers from a (those with th... | vpexpandd | — |
| `_mm_mask_expand_epi64` | Load contiguous active 64-bit integers from a (those with th... | vpexpandq | — |
| `_mm_mask_expand_pd` | Load contiguous active double-precision (64-bit) floating-po... | vexpandpd | — |
| `_mm_mask_expand_ps` | Load contiguous active single-precision (32-bit) floating-po... | vexpandps | — |
| `_mm_mask_fixupimm_pd` | Fix up packed double-precision (64-bit) floating-point eleme... | vfixupimmpd | — |
| `_mm_mask_fixupimm_ps` | Fix up packed single-precision (32-bit) floating-point eleme... | vfixupimmps | — |
| `_mm_mask_fixupimm_round_sd` | Fix up the lower double-precision (64-bit) floating-point el... | vfixupimmsd | — |
| `_mm_mask_fixupimm_round_ss` | Fix up the lower single-precision (32-bit) floating-point el... | vfixupimmss | — |
| `_mm_mask_fixupimm_sd` | Fix up the lower double-precision (64-bit) floating-point el... | vfixupimmsd | — |
| `_mm_mask_fixupimm_ss` | Fix up the lower single-precision (32-bit) floating-point el... | vfixupimmss | — |
| `_mm_mask_fmadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmadd | 5/1, 4/1 |
| `_mm_mask_fmadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmadd | 5/1, 4/1 |
| `_mm_mask_fmadd_round_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfmadd | 5/1, 4/1 |
| `_mm_mask_fmadd_round_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfmadd | 5/1, 4/1 |
| `_mm_mask_fmadd_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfmadd | 5/1, 4/1 |
| `_mm_mask_fmadd_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfmadd | 5/1, 4/1 |
| `_mm_mask_fmaddsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmaddsub | 5/1, 4/1 |
| `_mm_mask_fmaddsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmaddsub | 5/1, 4/1 |
| `_mm_mask_fmsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsub | 5/1, 4/1 |
| `_mm_mask_fmsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsub | 5/1, 4/1 |
| `_mm_mask_fmsub_round_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfmsub | 5/1, 4/1 |
| `_mm_mask_fmsub_round_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfmsub | 5/1, 4/1 |
| `_mm_mask_fmsub_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfmsub | 5/1, 4/1 |
| `_mm_mask_fmsub_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfmsub | 5/1, 4/1 |
| `_mm_mask_fmsubadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsubadd | 5/1, 4/1 |
| `_mm_mask_fmsubadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsubadd | 5/1, 4/1 |
| `_mm_mask_fnmadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmadd | 5/1, 4/1 |
| `_mm_mask_fnmadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmadd | 5/1, 4/1 |
| `_mm_mask_fnmadd_round_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfnmadd | 5/1, 4/1 |
| `_mm_mask_fnmadd_round_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfnmadd | 5/1, 4/1 |
| `_mm_mask_fnmadd_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfnmadd | 5/1, 4/1 |
| `_mm_mask_fnmadd_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfnmadd | 5/1, 4/1 |
| `_mm_mask_fnmsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmsub | 5/1, 4/1 |
| `_mm_mask_fnmsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmsub | 5/1, 4/1 |
| `_mm_mask_fnmsub_round_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfnmsub | 5/1, 4/1 |
| `_mm_mask_fnmsub_round_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfnmsub | 5/1, 4/1 |
| `_mm_mask_fnmsub_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfnmsub | 5/1, 4/1 |
| `_mm_mask_fnmsub_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfnmsub | 5/1, 4/1 |
| `_mm_mask_fpclass_pd_mask` | Test packed double-precision (64-bit) floating-point element... | vfpclasspd | — |
| `_mm_mask_fpclass_ps_mask` | Test packed single-precision (32-bit) floating-point element... | vfpclassps | — |
| `_mm_mask_fpclass_sd_mask` | Test the lower double-precision (64-bit) floating-point elem... | vfpclasssd | — |
| `_mm_mask_fpclass_ss_mask` | Test the lower single-precision (32-bit) floating-point elem... | vfpclassss | — |
| `_mm_mask_getexp_pd` | Convert the exponent of each packed double-precision (64-bit... | vgetexppd | — |
| `_mm_mask_getexp_ps` | Convert the exponent of each packed single-precision (32-bit... | vgetexpps | — |
| `_mm_mask_getexp_round_sd` | Convert the exponent of the lower double-precision (64-bit) ... | vgetexpsd | — |
| `_mm_mask_getexp_round_ss` | Convert the exponent of the lower single-precision (32-bit) ... | vgetexpss | — |
| `_mm_mask_getexp_sd` | Convert the exponent of the lower double-precision (64-bit) ... | vgetexpsd | — |
| `_mm_mask_getexp_ss` | Convert the exponent of the lower single-precision (32-bit) ... | vgetexpss | — |
| `_mm_mask_getmant_pd` | Normalize the mantissas of packed double-precision (64-bit) ... | vgetmantpd | — |
| `_mm_mask_getmant_ps` | Normalize the mantissas of packed single-precision (32-bit) ... | vgetmantps | — |
| `_mm_mask_getmant_round_sd` | Normalize the mantissas of the lower double-precision (64-bi... | vgetmantsd | — |
| `_mm_mask_getmant_round_ss` | Normalize the mantissas of the lower single-precision (32-bi... | vgetmantss | — |
| `_mm_mask_getmant_sd` | Normalize the mantissas of the lower double-precision (64-bi... | vgetmantsd | — |
| `_mm_mask_getmant_ss` | Normalize the mantissas of the lower single-precision (32-bi... | vgetmantss | — |
| `_mm_mask_lzcnt_epi32` | Counts the number of leading zero bits in each packed 32-bit... | vplzcntd | — |
| `_mm_mask_lzcnt_epi64` | Counts the number of leading zero bits in each packed 64-bit... | vplzcntq | — |
| `_mm_mask_madd_epi16` | Multiply packed signed 16-bit integers in a and b, producing... | vpmaddwd | — |
| `_mm_mask_maddubs_epi16` | Multiply packed unsigned 8-bit integers in a by packed signe... | vpmaddubsw | — |
| `_mm_mask_max_epi16` | Compare packed signed 16-bit integers in a and b, and store ... | vpmaxsw | — |
| `_mm_mask_max_epi32` | Compare packed signed 32-bit integers in a and b, and store ... | vpmaxsd | — |
| `_mm_mask_max_epi64` | Compare packed signed 64-bit integers in a and b, and store ... | vpmaxsq | — |
| `_mm_mask_max_epi8` | Compare packed signed 8-bit integers in a and b, and store p... | vpmaxsb | — |
| `_mm_mask_max_epu16` | Compare packed unsigned 16-bit integers in a and b, and stor... | vpmaxuw | — |
| `_mm_mask_max_epu32` | Compare packed unsigned 32-bit integers in a and b, and stor... | vpmaxud | — |
| `_mm_mask_max_epu64` | Compare packed unsigned 64-bit integers in a and b, and stor... | vpmaxuq | — |
| `_mm_mask_max_epu8` | Compare packed unsigned 8-bit integers in a and b, and store... | vpmaxub | — |
| `_mm_mask_max_pd` | Compare packed double-precision (64-bit) floating-point elem... | vmaxpd | — |
| `_mm_mask_max_ps` | Compare packed single-precision (32-bit) floating-point elem... | vmaxps | — |
| `_mm_mask_max_round_sd` | Compare the lower double-precision (64-bit) floating-point e... | vmaxsd | — |
| `_mm_mask_max_round_ss` | Compare the lower single-precision (32-bit) floating-point e... | vmaxss | — |
| `_mm_mask_max_sd` | Compare the lower double-precision (64-bit) floating-point e... | vmaxsd | — |
| `_mm_mask_max_ss` | Compare the lower single-precision (32-bit) floating-point e... | vmaxss | — |
| `_mm_mask_min_epi16` | Compare packed signed 16-bit integers in a and b, and store ... | vpminsw | — |
| `_mm_mask_min_epi32` | Compare packed signed 32-bit integers in a and b, and store ... | vpminsd | — |
| `_mm_mask_min_epi64` | Compare packed signed 64-bit integers in a and b, and store ... | vpminsq | — |
| `_mm_mask_min_epi8` | Compare packed signed 8-bit integers in a and b, and store p... | vpminsb | — |
| `_mm_mask_min_epu16` | Compare packed unsigned 16-bit integers in a and b, and stor... | vpminuw | — |
| `_mm_mask_min_epu32` | Compare packed unsigned 32-bit integers in a and b, and stor... | vpminud | — |
| `_mm_mask_min_epu64` | Compare packed unsigned 64-bit integers in a and b, and stor... | vpminuq | — |
| `_mm_mask_min_epu8` | Compare packed unsigned 8-bit integers in a and b, and store... | vpminub | — |
| `_mm_mask_min_pd` | Compare packed double-precision (64-bit) floating-point elem... | vminpd | — |
| `_mm_mask_min_ps` | Compare packed single-precision (32-bit) floating-point elem... | vminps | — |
| `_mm_mask_min_round_sd` | Compare the lower double-precision (64-bit) floating-point e... | vminsd | — |
| `_mm_mask_min_round_ss` | Compare the lower single-precision (32-bit) floating-point e... | vminss | — |
| `_mm_mask_min_sd` | Compare the lower double-precision (64-bit) floating-point e... | vminsd | — |
| `_mm_mask_min_ss` | Compare the lower single-precision (32-bit) floating-point e... | vminss | — |
| `_mm_mask_mov_epi16` | Move packed 16-bit integers from a into dst using writemask ... | vmovdqu16 | — |
| `_mm_mask_mov_epi32` | Move packed 32-bit integers from a to dst using writemask k ... | vmovdqa32 | — |
| `_mm_mask_mov_epi64` | Move packed 64-bit integers from a to dst using writemask k ... | vmovdqa64 | — |
| `_mm_mask_mov_epi8` | Move packed 8-bit integers from a into dst using writemask k... | vmovdqu8 | — |
| `_mm_mask_mov_pd` | Move packed double-precision (64-bit) floating-point element... | vmovapd | — |
| `_mm_mask_mov_ps` | Move packed single-precision (32-bit) floating-point element... | vmovaps | — |
| `_mm_mask_move_sd` | Move the lower double-precision (64-bit) floating-point elem... | vmovsd | — |
| `_mm_mask_move_ss` | Move the lower single-precision (32-bit) floating-point elem... | vmovss | — |
| `_mm_mask_movedup_pd` | Duplicate even-indexed double-precision (64-bit) floating-po... | vmovddup | — |
| `_mm_mask_movehdup_ps` | Duplicate odd-indexed single-precision (32-bit) floating-poi... | vmovshdup | — |
| `_mm_mask_moveldup_ps` | Duplicate even-indexed single-precision (32-bit) floating-po... | vmovsldup | — |
| `_mm_mask_mul_epi32` | Multiply the low signed 32-bit integers from each packed 64-... | vpmuldq | 10/2, 4/1 |
| `_mm_mask_mul_epu32` | Multiply the low unsigned 32-bit integers from each packed 6... | vpmuludq | 10/2, 4/1 |
| `_mm_mask_mul_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vmulpd | 5/1, 3/1 |
| `_mm_mask_mul_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vmulps | 5/1, 3/1 |
| `_mm_mask_mul_round_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vmulsd | — |
| `_mm_mask_mul_round_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vmulss | — |
| `_mm_mask_mul_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vmulsd | 5/1, 3/1 |
| `_mm_mask_mul_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vmulss | 5/1, 3/1 |
| `_mm_mask_mulhi_epi16` | Multiply the packed signed 16-bit integers in a and b, produ... | vpmulhw | 5/1, 3/1 |
| `_mm_mask_mulhi_epu16` | Multiply the packed unsigned 16-bit integers in a and b, pro... | vpmulhuw | 5/1, 3/1 |
| `_mm_mask_mulhrs_epi16` | Multiply packed signed 16-bit integers in a and b, producing... | vpmulhrsw | — |
| `_mm_mask_mullo_epi16` | Multiply the packed 16-bit integers in a and b, producing in... | vpmullw | 5/1, 3/1 |
| `_mm_mask_mullo_epi32` | Multiply the packed 32-bit integers in a and b, producing in... | vpmulld | 10/2, 4/1 |
| `_mm_mask_mullo_epi64` | Multiply packed 64-bit integers in `a` and `b`, producing in... | vpmullq | — |
| `_mm_mask_or_epi32` | Compute the bitwise OR of packed 32-bit integers in a and b,... | vpord | 1/1, 1/1 |
| `_mm_mask_or_epi64` | Compute the bitwise OR of packed 64-bit integers in a and b,... | vporq | 1/1, 1/1 |
| `_mm_mask_or_pd` | Compute the bitwise OR of packed double-precision (64-bit) f... | vorpd | 1/1, 1/1 |
| `_mm_mask_or_ps` | Compute the bitwise OR of packed single-precision (32-bit) f... | vorps | 1/1, 1/1 |
| `_mm_mask_packs_epi16` | Convert packed signed 16-bit integers from a and b to packed... | vpacksswb | 1/1, 1/1 |
| `_mm_mask_packs_epi32` | Convert packed signed 32-bit integers from a and b to packed... | vpackssdw | 1/1, 1/1 |
| `_mm_mask_packus_epi16` | Convert packed signed 16-bit integers from a and b to packed... | vpackuswb | 1/1, 1/1 |
| `_mm_mask_packus_epi32` | Convert packed signed 32-bit integers from a and b to packed... | vpackusdw | 1/1, 1/1 |
| `_mm_mask_permute_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vshufpd | 1/1, 1/1 |
| `_mm_mask_permute_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vshufps | 1/1, 1/1 |
| `_mm_mask_permutevar_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vpermilpd | 1/1, 1/1 |
| `_mm_mask_permutevar_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vpermilps | 1/1, 1/1 |
| `_mm_mask_permutex2var_epi16` | Shuffle 16-bit integers in a and b across lanes using the co... | vpermt2w | 1/1, 1/1 |
| `_mm_mask_permutex2var_epi32` | Shuffle 32-bit integers in a and b across lanes using the co... | vpermt2d | 1/1, 1/1 |
| `_mm_mask_permutex2var_epi64` | Shuffle 64-bit integers in a and b across lanes using the co... | vpermt2q | 1/1, 1/1 |
| `_mm_mask_permutex2var_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vpermt2pd | 1/1, 1/1 |
| `_mm_mask_permutex2var_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vpermt2ps | 1/1, 1/1 |
| `_mm_mask_permutexvar_epi16` | Shuffle 16-bit integers in a across lanes using the correspo... | vpermw | 1/1, 1/1 |
| `_mm_mask_range_pd` | Calculate the max, min, absolute max, or absolute min (depen... | vrangepd | — |
| `_mm_mask_range_ps` | Calculate the max, min, absolute max, or absolute min (depen... | vrangeps | — |
| `_mm_mask_range_round_sd` | Calculate the max, min, absolute max, or absolute min (depen... | vrangesd | — |
| `_mm_mask_range_round_ss` | Calculate the max, min, absolute max, or absolute min (depen... | vrangess | — |
| `_mm_mask_range_sd` | Calculate the max, min, absolute max, or absolute min (depen... | vrangesd | — |
| `_mm_mask_range_ss` | Calculate the max, min, absolute max, or absolute min (depen... | vrangess | — |
| `_mm_mask_rcp14_pd` | Compute the approximate reciprocal of packed double-precisio... | vrcp14pd | — |
| `_mm_mask_rcp14_ps` | Compute the approximate reciprocal of packed single-precisio... | vrcp14ps | — |
| `_mm_mask_rcp14_sd` | Compute the approximate reciprocal of the lower double-preci... | vrcp14sd | — |
| `_mm_mask_rcp14_ss` | Compute the approximate reciprocal of the lower single-preci... | vrcp14ss | — |
| `_mm_mask_reduce_add_epi16` | Reduce the packed 16-bit integers in a by addition using mas... |  | 1/1, 1/1 |
| `_mm_mask_reduce_add_epi8` | Reduce the packed 8-bit integers in a by addition using mask... |  | 1/1, 1/1 |
| `_mm_mask_reduce_and_epi16` | Reduce the packed 16-bit integers in a by bitwise AND using ... |  | 1/1, 1/1 |
| `_mm_mask_reduce_and_epi8` | Reduce the packed 8-bit integers in a by bitwise AND using m... |  | 1/1, 1/1 |
| `_mm_mask_reduce_max_epi16` | Reduce the packed 16-bit integers in a by maximum using mask... |  | — |
| `_mm_mask_reduce_max_epi8` | Reduce the packed 8-bit integers in a by maximum using mask ... |  | — |
| `_mm_mask_reduce_max_epu16` | Reduce the packed unsigned 16-bit integers in a by maximum u... |  | — |
| `_mm_mask_reduce_max_epu8` | Reduce the packed unsigned 8-bit integers in a by maximum us... |  | — |
| `_mm_mask_reduce_min_epi16` | Reduce the packed 16-bit integers in a by minimum using mask... |  | — |
| `_mm_mask_reduce_min_epi8` | Reduce the packed 8-bit integers in a by minimum using mask ... |  | — |
| `_mm_mask_reduce_min_epu16` | Reduce the packed unsigned 16-bit integers in a by minimum u... |  | — |
| `_mm_mask_reduce_min_epu8` | Reduce the packed unsigned 8-bit integers in a by minimum us... |  | — |
| `_mm_mask_reduce_mul_epi16` | Reduce the packed 16-bit integers in a by multiplication usi... |  | — |
| `_mm_mask_reduce_mul_epi8` | Reduce the packed 8-bit integers in a by multiplication usin... |  | — |
| `_mm_mask_reduce_or_epi16` | Reduce the packed 16-bit integers in a by bitwise OR using m... |  | 1/1, 1/1 |
| `_mm_mask_reduce_or_epi8` | Reduce the packed 8-bit integers in a by bitwise OR using ma... |  | 1/1, 1/1 |
| `_mm_mask_reduce_pd` | Extract the reduced argument of packed double-precision (64-... | vreducepd | — |
| `_mm_mask_reduce_ps` | Extract the reduced argument of packed single-precision (32-... | vreduceps | — |
| `_mm_mask_reduce_round_sd` | Extract the reduced argument of the lower double-precision (... | vreducesd | — |
| `_mm_mask_reduce_round_ss` | Extract the reduced argument of the lower single-precision (... | vreducess | — |
| `_mm_mask_reduce_sd` | Extract the reduced argument of the lower double-precision (... | vreducesd | — |
| `_mm_mask_reduce_ss` | Extract the reduced argument of the lower single-precision (... | vreducess | — |
| `_mm_mask_rol_epi32` | Rotate the bits in each packed 32-bit integer in a to the le... | vprold | — |
| `_mm_mask_rol_epi64` | Rotate the bits in each packed 64-bit integer in a to the le... | vprolq | — |
| `_mm_mask_rolv_epi32` | Rotate the bits in each packed 32-bit integer in a to the le... | vprolvd | — |
| `_mm_mask_rolv_epi64` | Rotate the bits in each packed 64-bit integer in a to the le... | vprolvq | — |
| `_mm_mask_ror_epi32` | Rotate the bits in each packed 32-bit integer in a to the ri... | vprold | — |
| `_mm_mask_ror_epi64` | Rotate the bits in each packed 64-bit integer in a to the ri... | vprolq | — |
| `_mm_mask_rorv_epi32` | Rotate the bits in each packed 32-bit integer in a to the ri... | vprorvd | — |
| `_mm_mask_rorv_epi64` | Rotate the bits in each packed 64-bit integer in a to the ri... | vprorvq | — |
| `_mm_mask_roundscale_pd` | Round packed double-precision (64-bit) floating-point elemen... | vrndscalepd | — |
| `_mm_mask_roundscale_ps` | Round packed single-precision (32-bit) floating-point elemen... | vrndscaleps | — |
| `_mm_mask_roundscale_round_sd` | Round the lower double-precision (64-bit) floating-point ele... | vrndscalesd | — |
| `_mm_mask_roundscale_round_ss` | Round the lower single-precision (32-bit) floating-point ele... | vrndscaless | — |
| `_mm_mask_roundscale_sd` | Round the lower double-precision (64-bit) floating-point ele... | vrndscalesd | — |
| `_mm_mask_roundscale_ss` | Round the lower single-precision (32-bit) floating-point ele... | vrndscaless | — |
| `_mm_mask_rsqrt14_pd` | Compute the approximate reciprocal square root of packed dou... | vrsqrt14pd | — |
| `_mm_mask_rsqrt14_ps` | Compute the approximate reciprocal square root of packed sin... | vrsqrt14ps | — |
| `_mm_mask_rsqrt14_sd` | Compute the approximate reciprocal square root of the lower ... | vrsqrt14sd | — |
| `_mm_mask_rsqrt14_ss` | Compute the approximate reciprocal square root of the lower ... | vrsqrt14ss | — |
| `_mm_mask_scalef_pd` | Scale the packed double-precision (64-bit) floating-point el... | vscalefpd | — |
| `_mm_mask_scalef_ps` | Scale the packed single-precision (32-bit) floating-point el... | vscalefps | — |
| `_mm_mask_scalef_round_sd` | Scale the packed double-precision (64-bit) floating-point el... | vscalefsd | — |
| `_mm_mask_scalef_round_ss` | Scale the packed single-precision (32-bit) floating-point el... | vscalefss | — |
| `_mm_mask_scalef_sd` | Scale the packed double-precision (64-bit) floating-point el... | vscalefsd | — |
| `_mm_mask_scalef_ss` | Scale the packed single-precision (32-bit) floating-point el... | vscalefss | — |
| `_mm_mask_set1_epi16` | Broadcast 16-bit integer a to all elements of dst using writ... | vpbroadcastw | — |
| `_mm_mask_set1_epi32` | Broadcast 32-bit integer a to all elements of dst using writ... | vpbroadcastd | — |
| `_mm_mask_set1_epi64` | Broadcast 64-bit integer a to all elements of dst using writ... | vpbroadcastq | — |
| `_mm_mask_set1_epi8` | Broadcast 8-bit integer a to all elements of dst using write... | vpbroadcast | — |
| `_mm_mask_shuffle_epi32` | Shuffle 32-bit integers in a within 128-bit lanes using the ... | vpshufd | 1/1, 1/1 |
| `_mm_mask_shuffle_epi8` | Shuffle 8-bit integers in a within 128-bit lanes using the c... | vpshufb | 1/1, 1/1 |
| `_mm_mask_shuffle_pd` | Shuffle double-precision (64-bit) floating-point elements wi... | vshufpd | 1/1, 1/1 |
| `_mm_mask_shuffle_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vshufps | 1/1, 1/1 |
| `_mm_mask_shufflehi_epi16` | Shuffle 16-bit integers in the high 64 bits of 128-bit lanes... | vpshufhw | 1/1, 1/1 |
| `_mm_mask_shufflelo_epi16` | Shuffle 16-bit integers in the low 64 bits of 128-bit lanes ... | vpshuflw | 1/1, 1/1 |
| `_mm_mask_sll_epi16` | Shift packed 16-bit integers in a left by count while shifti... | vpsllw | 1/1, 1/1 |
| `_mm_mask_sll_epi32` | Shift packed 32-bit integers in a left by count while shifti... | vpslld | 1/1, 1/1 |
| `_mm_mask_sll_epi64` | Shift packed 64-bit integers in a left by count while shifti... | vpsllq | 1/1, 1/1 |
| `_mm_mask_slli_epi16` | Shift packed 16-bit integers in a left by imm8 while shiftin... | vpsllw | 1/1, 1/1 |
| `_mm_mask_slli_epi32` | Shift packed 32-bit integers in a left by imm8 while shiftin... | vpslld | 1/1, 1/1 |
| `_mm_mask_slli_epi64` | Shift packed 64-bit integers in a left by imm8 while shiftin... | vpsllq | 1/1, 1/1 |
| `_mm_mask_sllv_epi16` | Shift packed 16-bit integers in a left by the amount specifi... | vpsllvw | 1/1, 1/1 |
| `_mm_mask_sllv_epi32` | Shift packed 32-bit integers in a left by the amount specifi... | vpsllvd | 1/1, 1/1 |
| `_mm_mask_sllv_epi64` | Shift packed 64-bit integers in a left by the amount specifi... | vpsllvq | 1/1, 1/1 |
| `_mm_mask_sqrt_pd` | Compute the square root of packed double-precision (64-bit) ... | vsqrtpd | 16/14, 20/9 |
| `_mm_mask_sqrt_ps` | Compute the square root of packed single-precision (32-bit) ... | vsqrtps | 11/7, 14/5 |
| `_mm_mask_sqrt_round_sd` | Compute the square root of the lower double-precision (64-bi... | vsqrtsd | — |
| `_mm_mask_sqrt_round_ss` | Compute the square root of the lower single-precision (32-bi... | vsqrtss | — |
| `_mm_mask_sqrt_sd` | Compute the square root of the lower double-precision (64-bi... | vsqrtsd | 16/14, 20/9 |
| `_mm_mask_sqrt_ss` | Compute the square root of the lower single-precision (32-bi... | vsqrtss | 11/7, 14/5 |
| `_mm_mask_sra_epi16` | Shift packed 16-bit integers in a right by count while shift... | vpsraw | 1/1, 1/1 |
| `_mm_mask_sra_epi32` | Shift packed 32-bit integers in a right by count while shift... | vpsrad | 1/1, 1/1 |
| `_mm_mask_sra_epi64` | Shift packed 64-bit integers in a right by count while shift... | vpsraq | 1/1, 1/1 |
| `_mm_mask_srai_epi16` | Shift packed 16-bit integers in a right by imm8 while shifti... | vpsraw | 1/1, 1/1 |
| `_mm_mask_srai_epi32` | Shift packed 32-bit integers in a right by imm8 while shifti... | vpsrad | 1/1, 1/1 |
| `_mm_mask_srai_epi64` | Shift packed 64-bit integers in a right by imm8 while shifti... | vpsraq | 1/1, 1/1 |
| `_mm_mask_srav_epi16` | Shift packed 16-bit integers in a right by the amount specif... | vpsravw | 1/1, 1/1 |
| `_mm_mask_srav_epi32` | Shift packed 32-bit integers in a right by the amount specif... | vpsravd | 1/1, 1/1 |
| `_mm_mask_srav_epi64` | Shift packed 64-bit integers in a right by the amount specif... | vpsravq | 1/1, 1/1 |
| `_mm_mask_srl_epi16` | Shift packed 16-bit integers in a right by count while shift... | vpsrlw | 1/1, 1/1 |
| `_mm_mask_srl_epi32` | Shift packed 32-bit integers in a right by count while shift... | vpsrld | 1/1, 1/1 |
| `_mm_mask_srl_epi64` | Shift packed 64-bit integers in a right by count while shift... | vpsrlq | 1/1, 1/1 |
| `_mm_mask_srli_epi16` | Shift packed 16-bit integers in a right by imm8 while shifti... | vpsrlw | 1/1, 1/1 |
| `_mm_mask_srli_epi32` | Shift packed 32-bit integers in a right by imm8 while shifti... | vpsrld | 1/1, 1/1 |
| `_mm_mask_srli_epi64` | Shift packed 64-bit integers in a right by imm8 while shifti... | vpsrlq | 1/1, 1/1 |
| `_mm_mask_srlv_epi16` | Shift packed 16-bit integers in a right by the amount specif... | vpsrlvw | 1/1, 1/1 |
| `_mm_mask_srlv_epi32` | Shift packed 32-bit integers in a right by the amount specif... | vpsrlvd | 1/1, 1/1 |
| `_mm_mask_srlv_epi64` | Shift packed 64-bit integers in a right by the amount specif... | vpsrlvq | 1/1, 1/1 |
| `_mm_mask_sub_epi16` | Subtract packed 16-bit integers in b from packed 16-bit inte... | vpsubw | 1/1, 1/1 |
| `_mm_mask_sub_epi32` | Subtract packed 32-bit integers in b from packed 32-bit inte... | vpsubd | 1/1, 1/1 |
| `_mm_mask_sub_epi64` | Subtract packed 64-bit integers in b from packed 64-bit inte... | vpsubq | 1/1, 1/1 |
| `_mm_mask_sub_epi8` | Subtract packed 8-bit integers in b from packed 8-bit intege... | vpsubb | 1/1, 1/1 |
| `_mm_mask_sub_pd` | Subtract packed double-precision (64-bit) floating-point ele... | vsubpd | — |
| `_mm_mask_sub_ps` | Subtract packed single-precision (32-bit) floating-point ele... | vsubps | — |
| `_mm_mask_sub_round_sd` | Subtract the lower double-precision (64-bit) floating-point ... | vsubsd | — |
| `_mm_mask_sub_round_ss` | Subtract the lower single-precision (32-bit) floating-point ... | vsubss | — |
| `_mm_mask_sub_sd` | Subtract the lower double-precision (64-bit) floating-point ... | vsubsd | — |
| `_mm_mask_sub_ss` | Subtract the lower single-precision (32-bit) floating-point ... | vsubss | — |
| `_mm_mask_subs_epi16` | Subtract packed signed 16-bit integers in b from packed 16-b... | vpsubsw | 1/1, 1/1 |
| `_mm_mask_subs_epi8` | Subtract packed signed 8-bit integers in b from packed 8-bit... | vpsubsb | 1/1, 1/1 |
| `_mm_mask_subs_epu16` | Subtract packed unsigned 16-bit integers in b from packed un... | vpsubusw | — |
| `_mm_mask_subs_epu8` | Subtract packed unsigned 8-bit integers in b from packed uns... | vpsubusb | — |
| `_mm_mask_ternarylogic_epi32` | Bitwise ternary logic that provides the capability to implem... | vpternlogd | — |
| `_mm_mask_ternarylogic_epi64` | Bitwise ternary logic that provides the capability to implem... | vpternlogq | — |
| `_mm_mask_test_epi16_mask` | Compute the bitwise AND of packed 16-bit integers in a and b... | vptestmw | — |
| `_mm_mask_test_epi32_mask` | Compute the bitwise AND of packed 32-bit integers in a and b... | vptestmd | — |
| `_mm_mask_test_epi64_mask` | Compute the bitwise AND of packed 64-bit integers in a and b... | vptestmq | — |
| `_mm_mask_test_epi8_mask` | Compute the bitwise AND of packed 8-bit integers in a and b,... | vptestmb | — |
| `_mm_mask_testn_epi16_mask` | Compute the bitwise NAND of packed 16-bit integers in a and ... | vptestnmw | — |
| `_mm_mask_testn_epi32_mask` | Compute the bitwise NAND of packed 32-bit integers in a and ... | vptestnmd | — |
| `_mm_mask_testn_epi64_mask` | Compute the bitwise NAND of packed 64-bit integers in a and ... | vptestnmq | — |
| `_mm_mask_testn_epi8_mask` | Compute the bitwise NAND of packed 8-bit integers in a and b... | vptestnmb | — |
| `_mm_mask_unpackhi_epi16` | Unpack and interleave 16-bit integers from the high half of ... | vpunpckhwd | 1/1, 1/1 |
| `_mm_mask_unpackhi_epi32` | Unpack and interleave 32-bit integers from the high half of ... | vpunpckhdq | 1/1, 1/1 |
| `_mm_mask_unpackhi_epi64` | Unpack and interleave 64-bit integers from the high half of ... | vpunpckhqdq | 1/1, 1/1 |
| `_mm_mask_unpackhi_epi8` | Unpack and interleave 8-bit integers from the high half of e... | vpunpckhbw | 1/1, 1/1 |
| `_mm_mask_unpackhi_pd` | Unpack and interleave double-precision (64-bit) floating-poi... | vunpckhpd | 1/1, 1/1 |
| `_mm_mask_unpackhi_ps` | Unpack and interleave single-precision (32-bit) floating-poi... | vunpckhps | 1/1, 1/1 |
| `_mm_mask_unpacklo_epi16` | Unpack and interleave 16-bit integers from the low half of e... | vpunpcklwd | 1/1, 1/1 |
| `_mm_mask_unpacklo_epi32` | Unpack and interleave 32-bit integers from the low half of e... | vpunpckldq | 1/1, 1/1 |
| `_mm_mask_unpacklo_epi64` | Unpack and interleave 64-bit integers from the low half of e... | vpunpcklqdq | 1/1, 1/1 |
| `_mm_mask_unpacklo_epi8` | Unpack and interleave 8-bit integers from the low half of ea... | vpunpcklbw | 1/1, 1/1 |
| `_mm_mask_unpacklo_pd` | Unpack and interleave double-precision (64-bit) floating-poi... | vunpcklpd | 1/1, 1/1 |
| `_mm_mask_unpacklo_ps` | Unpack and interleave single-precision (32-bit) floating-poi... | vunpcklps | 1/1, 1/1 |
| `_mm_mask_xor_epi32` | Compute the bitwise XOR of packed 32-bit integers in a and b... | vpxord | 1/1, 1/1 |
| `_mm_mask_xor_epi64` | Compute the bitwise XOR of packed 64-bit integers in a and b... | vpxorq | 1/1, 1/1 |
| `_mm_mask_xor_pd` | Compute the bitwise XOR of packed double-precision (64-bit) ... | vxorpd | 1/1, 1/1 |
| `_mm_mask_xor_ps` | Compute the bitwise XOR of packed single-precision (32-bit) ... | vxorps | 1/1, 1/1 |
| `_mm_maskz_abs_epi16` | Compute the absolute value of packed signed 16-bit integers ... | vpabsw | — |
| `_mm_maskz_abs_epi32` | Compute the absolute value of packed signed 32-bit integers ... | vpabsd | — |
| `_mm_maskz_abs_epi64` | Compute the absolute value of packed signed 64-bit integers ... | vpabsq | — |
| `_mm_maskz_abs_epi8` | Compute the absolute value of packed signed 8-bit integers i... | vpabsb | — |
| `_mm_maskz_add_epi16` | Add packed 16-bit integers in a and b, and store the results... | vpaddw | 1/1, 1/1 |
| `_mm_maskz_add_epi32` | Add packed 32-bit integers in a and b, and store the results... | vpaddd | 1/1, 1/1 |
| `_mm_maskz_add_epi64` | Add packed 64-bit integers in a and b, and store the results... | vpaddq | 1/1, 1/1 |
| `_mm_maskz_add_epi8` | Add packed 8-bit integers in a and b, and store the results ... | vpaddb | 1/1, 1/1 |
| `_mm_maskz_add_pd` | Add packed double-precision (64-bit) floating-point elements... | vaddpd | 3/1, 3/1 |
| `_mm_maskz_add_ps` | Add packed single-precision (32-bit) floating-point elements... | vaddps | 3/1, 3/1 |
| `_mm_maskz_add_round_sd` | Add the lower double-precision (64-bit) floating-point eleme... | vaddsd | — |
| `_mm_maskz_add_round_ss` | Add the lower single-precision (32-bit) floating-point eleme... | vaddss | — |
| `_mm_maskz_add_sd` | Add the lower double-precision (64-bit) floating-point eleme... | vaddsd | 3/1, 3/1 |
| `_mm_maskz_add_ss` | Add the lower single-precision (32-bit) floating-point eleme... | vaddss | 3/1, 3/1 |
| `_mm_maskz_adds_epi16` | Add packed signed 16-bit integers in a and b using saturatio... | vpaddsw | 1/1, 1/1 |
| `_mm_maskz_adds_epi8` | Add packed signed 8-bit integers in a and b using saturation... | vpaddsb | 1/1, 1/1 |
| `_mm_maskz_adds_epu16` | Add packed unsigned 16-bit integers in a and b using saturat... | vpaddusw | — |
| `_mm_maskz_adds_epu8` | Add packed unsigned 8-bit integers in a and b using saturati... | vpaddusb | — |
| `_mm_maskz_alignr_epi32` | Concatenate a and b into a 32-byte immediate result, shift t... | valignd | — |
| `_mm_maskz_alignr_epi64` | Concatenate a and b into a 32-byte immediate result, shift t... | valignq | — |
| `_mm_maskz_alignr_epi8` | Concatenate pairs of 16-byte blocks in a and b into a 32-byt... | vpalignr | — |
| `_mm_maskz_and_epi32` | Compute the bitwise AND of packed 32-bit integers in a and b... | vpandd | 1/1, 1/1 |
| `_mm_maskz_and_epi64` | Compute the bitwise AND of packed 64-bit integers in a and b... | vpandq | 1/1, 1/1 |
| `_mm_maskz_and_pd` | Compute the bitwise AND of packed double-precision (64-bit) ... | vandpd | 1/1, 1/1 |
| `_mm_maskz_and_ps` | Compute the bitwise AND of packed single-precision (32-bit) ... | vandps | 1/1, 1/1 |
| `_mm_maskz_andnot_epi32` | Compute the bitwise NOT of packed 32-bit integers in a and t... | vpandnd | 1/1, 1/1 |
| `_mm_maskz_andnot_epi64` | Compute the bitwise NOT of packed 64-bit integers in a and t... | vpandnq | 1/1, 1/1 |
| `_mm_maskz_andnot_pd` | Compute the bitwise NOT of packed double-precision (64-bit) ... | vandnpd | 1/1, 1/1 |
| `_mm_maskz_andnot_ps` | Compute the bitwise NOT of packed single-precision (32-bit) ... | vandnps | 1/1, 1/1 |
| `_mm_maskz_avg_epu16` | Average packed unsigned 16-bit integers in a and b, and stor... | vpavgw | — |
| `_mm_maskz_avg_epu8` | Average packed unsigned 8-bit integers in a and b, and store... | vpavgb | — |
| `_mm_maskz_broadcast_i32x2` | Broadcasts the lower 2 packed 32-bit integers from a to all ... | vbroadcasti32x2 | — |
| `_mm_maskz_broadcastb_epi8` | Broadcast the low packed 8-bit integer from a to all element... | vpbroadcastb | — |
| `_mm_maskz_broadcastd_epi32` | Broadcast the low packed 32-bit integer from a to all elemen... | vpbroadcast | — |
| `_mm_maskz_broadcastq_epi64` | Broadcast the low packed 64-bit integer from a to all elemen... | vpbroadcast | — |
| `_mm_maskz_broadcastss_ps` | Broadcast the low single-precision (32-bit) floating-point e... | vbroadcastss | — |
| `_mm_maskz_broadcastw_epi16` | Broadcast the low packed 16-bit integer from a to all elemen... | vpbroadcastw | — |
| `_mm_maskz_compress_epi32` | Contiguously store the active 32-bit integers in a (those wi... | vpcompressd | — |
| `_mm_maskz_compress_epi64` | Contiguously store the active 64-bit integers in a (those wi... | vpcompressq | — |
| `_mm_maskz_compress_pd` | Contiguously store the active double-precision (64-bit) floa... | vcompresspd | — |
| `_mm_maskz_compress_ps` | Contiguously store the active single-precision (32-bit) floa... | vcompressps | — |
| `_mm_maskz_conflict_epi32` | Test each 32-bit element of a for equality with all other el... | vpconflictd | — |
| `_mm_maskz_conflict_epi64` | Test each 64-bit element of a for equality with all other el... | vpconflictq | — |
| `_mm_maskz_cvt_roundps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2ph | 4/1, 3/1 |
| `_mm_maskz_cvt_roundsd_ss` | Convert the lower double-precision (64-bit) floating-point e... | vcvtsd2ss | 4/1, 3/1 |
| `_mm_maskz_cvt_roundss_sd` | Convert the lower single-precision (32-bit) floating-point e... | vcvtss2sd | 4/1, 3/1 |
| `_mm_maskz_cvtepi16_epi32` | Sign extend packed 16-bit integers in a to packed 32-bit int... | vpmovsxwd | 4/1, 3/1 |
| `_mm_maskz_cvtepi16_epi64` | Sign extend packed 16-bit integers in a to packed 64-bit int... | vpmovsxwq | 4/1, 3/1 |
| `_mm_maskz_cvtepi16_epi8` | Convert packed 16-bit integers in a to packed 8-bit integers... | vpmovwb | 4/1, 3/1 |
| `_mm_maskz_cvtepi32_epi16` | Convert packed 32-bit integers in a to packed 16-bit integer... | vpmovdw | 4/1, 3/1 |
| `_mm_maskz_cvtepi32_epi64` | Sign extend packed 32-bit integers in a to packed 64-bit int... | vpmovsxdq | 4/1, 3/1 |
| `_mm_maskz_cvtepi32_epi8` | Convert packed 32-bit integers in a to packed 8-bit integers... | vpmovdb | 4/1, 3/1 |
| `_mm_maskz_cvtepi32_pd` | Convert packed signed 32-bit integers in a to packed double-... | vcvtdq2pd | 4/1, 3/1 |
| `_mm_maskz_cvtepi32_ps` | Convert packed signed 32-bit integers in a to packed single-... | vcvtdq2ps | 4/1, 3/1 |
| `_mm_maskz_cvtepi64_epi16` | Convert packed 64-bit integers in a to packed 16-bit integer... | vpmovqw | 4/1, 3/1 |
| `_mm_maskz_cvtepi64_epi32` | Convert packed 64-bit integers in a to packed 32-bit integer... | vpmovqd | 4/1, 3/1 |
| `_mm_maskz_cvtepi64_epi8` | Convert packed 64-bit integers in a to packed 8-bit integers... | vpmovqb | 4/1, 3/1 |
| `_mm_maskz_cvtepi64_pd` | Convert packed signed 64-bit integers in a to packed double-... | vcvtqq2pd | 4/1, 3/1 |
| `_mm_maskz_cvtepi64_ps` | Convert packed signed 64-bit integers in a to packed single-... | vcvtqq2ps | 4/1, 3/1 |
| `_mm_maskz_cvtepi8_epi16` | Sign extend packed 8-bit integers in a to packed 16-bit inte... | vpmovsxbw | 4/1, 3/1 |
| `_mm_maskz_cvtepi8_epi32` | Sign extend packed 8-bit integers in a to packed 32-bit inte... | vpmovsxbd | 4/1, 3/1 |
| `_mm_maskz_cvtepi8_epi64` | Sign extend packed 8-bit integers in the low 2 bytes of a to... | vpmovsxbq | 4/1, 3/1 |
| `_mm_maskz_cvtepu16_epi32` | Zero extend packed unsigned 16-bit integers in a to packed 3... | vpmovzxwd | 4/1, 3/1 |
| `_mm_maskz_cvtepu16_epi64` | Zero extend packed unsigned 16-bit integers in the low 4 byt... | vpmovzxwq | 4/1, 3/1 |
| `_mm_maskz_cvtepu32_epi64` | Zero extend packed unsigned 32-bit integers in a to packed 6... | vpmovzxdq | 4/1, 3/1 |
| `_mm_maskz_cvtepu32_pd` | Convert packed unsigned 32-bit integers in a to packed doubl... | vcvtudq2pd | 4/1, 3/1 |
| `_mm_maskz_cvtepu64_pd` | Convert packed unsigned 64-bit integers in a to packed doubl... | vcvtuqq2pd | 4/1, 3/1 |
| `_mm_maskz_cvtepu64_ps` | Convert packed unsigned 64-bit integers in a to packed singl... | vcvtuqq2ps | 4/1, 3/1 |
| `_mm_maskz_cvtepu8_epi16` | Zero extend packed unsigned 8-bit integers in a to packed 16... | vpmovzxbw | 4/1, 3/1 |
| `_mm_maskz_cvtepu8_epi32` | Zero extend packed unsigned 8-bit integers in th elow 4 byte... | vpmovzxbd | 4/1, 3/1 |
| `_mm_maskz_cvtepu8_epi64` | Zero extend packed unsigned 8-bit integers in the low 2 byte... | vpmovzxbq | 4/1, 3/1 |
| `_mm_maskz_cvtpd_epi32` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2dq | 4/1, 3/1 |
| `_mm_maskz_cvtpd_epi64` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2qq | 4/1, 3/1 |
| `_mm_maskz_cvtpd_epu32` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2udq | 4/1, 3/1 |
| `_mm_maskz_cvtpd_epu64` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2uqq | 4/1, 3/1 |
| `_mm_maskz_cvtpd_ps` | Convert packed double-precision (64-bit) floating-point elem... | vcvtpd2ps | 4/1, 3/1 |
| `_mm_maskz_cvtph_ps` | Convert packed half-precision (16-bit) floating-point elemen... | vcvtph2ps | 4/1, 3/1 |
| `_mm_maskz_cvtps_epi32` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2dq | 4/1, 3/1 |
| `_mm_maskz_cvtps_epi64` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2qq | 4/1, 3/1 |
| `_mm_maskz_cvtps_epu32` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2udq | 4/1, 3/1 |
| `_mm_maskz_cvtps_epu64` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2uqq | 4/1, 3/1 |
| `_mm_maskz_cvtps_ph` | Convert packed single-precision (32-bit) floating-point elem... | vcvtps2ph | 4/1, 3/1 |
| `_mm_maskz_cvtsd_ss` | Convert the lower double-precision (64-bit) floating-point e... | vcvtsd2ss | 4/1, 3/1 |
| `_mm_maskz_cvtsepi16_epi8` | Convert packed signed 16-bit integers in a to packed 8-bit i... | vpmovswb | 4/1, 3/1 |
| `_mm_maskz_cvtsepi32_epi16` | Convert packed signed 32-bit integers in a to packed 16-bit ... | vpmovsdw | 4/1, 3/1 |
| `_mm_maskz_cvtsepi32_epi8` | Convert packed signed 32-bit integers in a to packed 8-bit i... | vpmovsdb | 4/1, 3/1 |
| `_mm_maskz_cvtsepi64_epi16` | Convert packed signed 64-bit integers in a to packed 16-bit ... | vpmovsqw | 4/1, 3/1 |
| `_mm_maskz_cvtsepi64_epi32` | Convert packed signed 64-bit integers in a to packed 32-bit ... | vpmovsqd | 4/1, 3/1 |
| `_mm_maskz_cvtsepi64_epi8` | Convert packed signed 64-bit integers in a to packed 8-bit i... | vpmovsqb | 4/1, 3/1 |
| `_mm_maskz_cvtss_sd` | Convert the lower single-precision (32-bit) floating-point e... | vcvtss2sd | 4/1, 3/1 |
| `_mm_maskz_cvttpd_epi32` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2dq | 4/1, 3/1 |
| `_mm_maskz_cvttpd_epi64` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2qq | 4/1, 3/1 |
| `_mm_maskz_cvttpd_epu32` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2udq | 4/1, 3/1 |
| `_mm_maskz_cvttpd_epu64` | Convert packed double-precision (64-bit) floating-point elem... | vcvttpd2uqq | 4/1, 3/1 |
| `_mm_maskz_cvttps_epi32` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2dq | 4/1, 3/1 |
| `_mm_maskz_cvttps_epi64` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2qq | 4/1, 3/1 |
| `_mm_maskz_cvttps_epu32` | Convert packed double-precision (32-bit) floating-point elem... | vcvttps2udq | 4/1, 3/1 |
| `_mm_maskz_cvttps_epu64` | Convert packed single-precision (32-bit) floating-point elem... | vcvttps2uqq | 4/1, 3/1 |
| `_mm_maskz_cvtusepi16_epi8` | Convert packed unsigned 16-bit integers in a to packed unsig... | vpmovuswb | 4/1, 3/1 |
| `_mm_maskz_cvtusepi32_epi16` | Convert packed unsigned 32-bit integers in a to packed unsig... | vpmovusdw | 4/1, 3/1 |
| `_mm_maskz_cvtusepi32_epi8` | Convert packed unsigned 32-bit integers in a to packed unsig... | vpmovusdb | 4/1, 3/1 |
| `_mm_maskz_cvtusepi64_epi16` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqw | 4/1, 3/1 |
| `_mm_maskz_cvtusepi64_epi32` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqd | 4/1, 3/1 |
| `_mm_maskz_cvtusepi64_epi8` | Convert packed unsigned 64-bit integers in a to packed unsig... | vpmovusqb | 4/1, 3/1 |
| `_mm_maskz_dbsad_epu8` | Compute the sum of absolute differences (SADs) of quadruplet... | vdbpsadbw | — |
| `_mm_maskz_div_pd` | Divide packed double-precision (64-bit) floating-point eleme... | vdivpd | 20/14, 13/7 |
| `_mm_maskz_div_ps` | Divide packed single-precision (32-bit) floating-point eleme... | vdivps | 13/7, 10/4 |
| `_mm_maskz_div_round_sd` | Divide the lower double-precision (64-bit) floating-point el... | vdivsd | — |
| `_mm_maskz_div_round_ss` | Divide the lower single-precision (32-bit) floating-point el... | vdivss | — |
| `_mm_maskz_div_sd` | Divide the lower double-precision (64-bit) floating-point el... | vdivsd | 20/14, 13/7 |
| `_mm_maskz_div_ss` | Divide the lower single-precision (32-bit) floating-point el... | vdivss | 13/7, 10/4 |
| `_mm_maskz_expand_epi32` | Load contiguous active 32-bit integers from a (those with th... | vpexpandd | — |
| `_mm_maskz_expand_epi64` | Load contiguous active 64-bit integers from a (those with th... | vpexpandq | — |
| `_mm_maskz_expand_pd` | Load contiguous active double-precision (64-bit) floating-po... | vexpandpd | — |
| `_mm_maskz_expand_ps` | Load contiguous active single-precision (32-bit) floating-po... | vexpandps | — |
| `_mm_maskz_fixupimm_pd` | Fix up packed double-precision (64-bit) floating-point eleme... | vfixupimmpd | — |
| `_mm_maskz_fixupimm_ps` | Fix up packed single-precision (32-bit) floating-point eleme... | vfixupimmps | — |
| `_mm_maskz_fixupimm_round_sd` | Fix up the lower double-precision (64-bit) floating-point el... | vfixupimmsd | — |
| `_mm_maskz_fixupimm_round_ss` | Fix up the lower single-precision (32-bit) floating-point el... | vfixupimmss | — |
| `_mm_maskz_fixupimm_sd` | Fix up the lower double-precision (64-bit) floating-point el... | vfixupimmsd | — |
| `_mm_maskz_fixupimm_ss` | Fix up the lower single-precision (32-bit) floating-point el... | vfixupimmss | — |
| `_mm_maskz_fmadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmadd | 5/1, 4/1 |
| `_mm_maskz_fmadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmadd | 5/1, 4/1 |
| `_mm_maskz_fmadd_round_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfmadd | 5/1, 4/1 |
| `_mm_maskz_fmadd_round_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfmadd | 5/1, 4/1 |
| `_mm_maskz_fmadd_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfmadd | 5/1, 4/1 |
| `_mm_maskz_fmadd_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfmadd | 5/1, 4/1 |
| `_mm_maskz_fmaddsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmaddsub | 5/1, 4/1 |
| `_mm_maskz_fmaddsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmaddsub | 5/1, 4/1 |
| `_mm_maskz_fmsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsub | 5/1, 4/1 |
| `_mm_maskz_fmsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsub | 5/1, 4/1 |
| `_mm_maskz_fmsub_round_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfmsub | 5/1, 4/1 |
| `_mm_maskz_fmsub_round_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfmsub | 5/1, 4/1 |
| `_mm_maskz_fmsub_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfmsub | 5/1, 4/1 |
| `_mm_maskz_fmsub_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfmsub | 5/1, 4/1 |
| `_mm_maskz_fmsubadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfmsubadd | 5/1, 4/1 |
| `_mm_maskz_fmsubadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfmsubadd | 5/1, 4/1 |
| `_mm_maskz_fnmadd_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmadd | 5/1, 4/1 |
| `_mm_maskz_fnmadd_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmadd | 5/1, 4/1 |
| `_mm_maskz_fnmadd_round_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfnmadd | 5/1, 4/1 |
| `_mm_maskz_fnmadd_round_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfnmadd | 5/1, 4/1 |
| `_mm_maskz_fnmadd_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfnmadd | 5/1, 4/1 |
| `_mm_maskz_fnmadd_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfnmadd | 5/1, 4/1 |
| `_mm_maskz_fnmsub_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vfnmsub | 5/1, 4/1 |
| `_mm_maskz_fnmsub_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vfnmsub | 5/1, 4/1 |
| `_mm_maskz_fnmsub_round_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfnmsub | 5/1, 4/1 |
| `_mm_maskz_fnmsub_round_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfnmsub | 5/1, 4/1 |
| `_mm_maskz_fnmsub_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vfnmsub | 5/1, 4/1 |
| `_mm_maskz_fnmsub_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vfnmsub | 5/1, 4/1 |
| `_mm_maskz_getexp_pd` | Convert the exponent of each packed double-precision (64-bit... | vgetexppd | — |
| `_mm_maskz_getexp_ps` | Convert the exponent of each packed single-precision (32-bit... | vgetexpps | — |
| `_mm_maskz_getexp_round_sd` | Convert the exponent of the lower double-precision (64-bit) ... | vgetexpsd | — |
| `_mm_maskz_getexp_round_ss` | Convert the exponent of the lower single-precision (32-bit) ... | vgetexpss | — |
| `_mm_maskz_getexp_sd` | Convert the exponent of the lower double-precision (64-bit) ... | vgetexpsd | — |
| `_mm_maskz_getexp_ss` | Convert the exponent of the lower single-precision (32-bit) ... | vgetexpss | — |
| `_mm_maskz_getmant_pd` | Normalize the mantissas of packed double-precision (64-bit) ... | vgetmantpd | — |
| `_mm_maskz_getmant_ps` | Normalize the mantissas of packed single-precision (32-bit) ... | vgetmantps | — |
| `_mm_maskz_getmant_round_sd` | Normalize the mantissas of the lower double-precision (64-bi... | vgetmantsd | — |
| `_mm_maskz_getmant_round_ss` | Normalize the mantissas of the lower single-precision (32-bi... | vgetmantss | — |
| `_mm_maskz_getmant_sd` | Normalize the mantissas of the lower double-precision (64-bi... | vgetmantsd | — |
| `_mm_maskz_getmant_ss` | Normalize the mantissas of the lower single-precision (32-bi... | vgetmantss | — |
| `_mm_maskz_lzcnt_epi32` | Counts the number of leading zero bits in each packed 32-bit... | vplzcntd | — |
| `_mm_maskz_lzcnt_epi64` | Counts the number of leading zero bits in each packed 64-bit... | vplzcntq | — |
| `_mm_maskz_madd_epi16` | Multiply packed signed 16-bit integers in a and b, producing... | vpmaddwd | — |
| `_mm_maskz_maddubs_epi16` | Multiply packed unsigned 8-bit integers in a by packed signe... | vpmaddubsw | — |
| `_mm_maskz_max_epi16` | Compare packed signed 16-bit integers in a and b, and store ... | vpmaxsw | — |
| `_mm_maskz_max_epi32` | Compare packed signed 32-bit integers in a and b, and store ... | vpmaxsd | — |
| `_mm_maskz_max_epi64` | Compare packed signed 64-bit integers in a and b, and store ... | vpmaxsq | — |
| `_mm_maskz_max_epi8` | Compare packed signed 8-bit integers in a and b, and store p... | vpmaxsb | — |
| `_mm_maskz_max_epu16` | Compare packed unsigned 16-bit integers in a and b, and stor... | vpmaxuw | — |
| `_mm_maskz_max_epu32` | Compare packed unsigned 32-bit integers in a and b, and stor... | vpmaxud | — |
| `_mm_maskz_max_epu64` | Compare packed unsigned 64-bit integers in a and b, and stor... | vpmaxuq | — |
| `_mm_maskz_max_epu8` | Compare packed unsigned 8-bit integers in a and b, and store... | vpmaxub | — |
| `_mm_maskz_max_pd` | Compare packed double-precision (64-bit) floating-point elem... | vmaxpd | — |
| `_mm_maskz_max_ps` | Compare packed single-precision (32-bit) floating-point elem... | vmaxps | — |
| `_mm_maskz_max_round_sd` | Compare the lower double-precision (64-bit) floating-point e... | vmaxsd | — |
| `_mm_maskz_max_round_ss` | Compare the lower single-precision (32-bit) floating-point e... | vmaxss | — |
| `_mm_maskz_max_sd` | Compare the lower double-precision (64-bit) floating-point e... | vmaxsd | — |
| `_mm_maskz_max_ss` | Compare the lower single-precision (32-bit) floating-point e... | vmaxss | — |
| `_mm_maskz_min_epi16` | Compare packed signed 16-bit integers in a and b, and store ... | vpminsw | — |
| `_mm_maskz_min_epi32` | Compare packed signed 32-bit integers in a and b, and store ... | vpminsd | — |
| `_mm_maskz_min_epi64` | Compare packed signed 64-bit integers in a and b, and store ... | vpminsq | — |
| `_mm_maskz_min_epi8` | Compare packed signed 8-bit integers in a and b, and store p... | vpminsb | — |
| `_mm_maskz_min_epu16` | Compare packed unsigned 16-bit integers in a and b, and stor... | vpminuw | — |
| `_mm_maskz_min_epu32` | Compare packed unsigned 32-bit integers in a and b, and stor... | vpminud | — |
| `_mm_maskz_min_epu64` | Compare packed unsigned 64-bit integers in a and b, and stor... | vpminuq | — |
| `_mm_maskz_min_epu8` | Compare packed unsigned 8-bit integers in a and b, and store... | vpminub | — |
| `_mm_maskz_min_pd` | Compare packed double-precision (64-bit) floating-point elem... | vminpd | — |
| `_mm_maskz_min_ps` | Compare packed single-precision (32-bit) floating-point elem... | vminps | — |
| `_mm_maskz_min_round_sd` | Compare the lower double-precision (64-bit) floating-point e... | vminsd | — |
| `_mm_maskz_min_round_ss` | Compare the lower single-precision (32-bit) floating-point e... | vminss | — |
| `_mm_maskz_min_sd` | Compare the lower double-precision (64-bit) floating-point e... | vminsd | — |
| `_mm_maskz_min_ss` | Compare the lower single-precision (32-bit) floating-point e... | vminss | — |
| `_mm_maskz_mov_epi16` | Move packed 16-bit integers from a into dst using zeromask k... | vmovdqu16 | — |
| `_mm_maskz_mov_epi32` | Move packed 32-bit integers from a into dst using zeromask k... | vmovdqa32 | — |
| `_mm_maskz_mov_epi64` | Move packed 64-bit integers from a into dst using zeromask k... | vmovdqa64 | — |
| `_mm_maskz_mov_epi8` | Move packed 8-bit integers from a into dst using zeromask k ... | vmovdqu8 | — |
| `_mm_maskz_mov_pd` | Move packed double-precision (64-bit) floating-point element... | vmovapd | — |
| `_mm_maskz_mov_ps` | Move packed single-precision (32-bit) floating-point element... | vmovaps | — |
| `_mm_maskz_move_sd` | Move the lower double-precision (64-bit) floating-point elem... | vmovsd | — |
| `_mm_maskz_move_ss` | Move the lower single-precision (32-bit) floating-point elem... | vmovss | — |
| `_mm_maskz_movedup_pd` | Duplicate even-indexed double-precision (64-bit) floating-po... | vmovddup | — |
| `_mm_maskz_movehdup_ps` | Duplicate odd-indexed single-precision (32-bit) floating-poi... | vmovshdup | — |
| `_mm_maskz_moveldup_ps` | Duplicate even-indexed single-precision (32-bit) floating-po... | vmovsldup | — |
| `_mm_maskz_mul_epi32` | Multiply the low signed 32-bit integers from each packed 64-... | vpmuldq | 10/2, 4/1 |
| `_mm_maskz_mul_epu32` | Multiply the low unsigned 32-bit integers from each packed 6... | vpmuludq | 10/2, 4/1 |
| `_mm_maskz_mul_pd` | Multiply packed double-precision (64-bit) floating-point ele... | vmulpd | 5/1, 3/1 |
| `_mm_maskz_mul_ps` | Multiply packed single-precision (32-bit) floating-point ele... | vmulps | 5/1, 3/1 |
| `_mm_maskz_mul_round_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vmulsd | — |
| `_mm_maskz_mul_round_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vmulss | — |
| `_mm_maskz_mul_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vmulsd | 5/1, 3/1 |
| `_mm_maskz_mul_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vmulss | 5/1, 3/1 |
| `_mm_maskz_mulhi_epi16` | Multiply the packed signed 16-bit integers in a and b, produ... | vpmulhw | 5/1, 3/1 |
| `_mm_maskz_mulhi_epu16` | Multiply the packed unsigned 16-bit integers in a and b, pro... | vpmulhuw | 5/1, 3/1 |
| `_mm_maskz_mulhrs_epi16` | Multiply packed signed 16-bit integers in a and b, producing... | vpmulhrsw | — |
| `_mm_maskz_mullo_epi16` | Multiply the packed 16-bit integers in a and b, producing in... | vpmullw | 5/1, 3/1 |
| `_mm_maskz_mullo_epi32` | Multiply the packed 32-bit integers in a and b, producing in... | vpmulld | 10/2, 4/1 |
| `_mm_maskz_mullo_epi64` | Multiply packed 64-bit integers in `a` and `b`, producing in... | vpmullq | — |
| `_mm_maskz_or_epi32` | Compute the bitwise OR of packed 32-bit integers in a and b,... | vpord | 1/1, 1/1 |
| `_mm_maskz_or_epi64` | Compute the bitwise OR of packed 64-bit integers in a and b,... | vporq | 1/1, 1/1 |
| `_mm_maskz_or_pd` | Compute the bitwise OR of packed double-precision (64-bit) f... | vorpd | 1/1, 1/1 |
| `_mm_maskz_or_ps` | Compute the bitwise OR of packed single-precision (32-bit) f... | vorps | 1/1, 1/1 |
| `_mm_maskz_packs_epi16` | Convert packed signed 16-bit integers from a and b to packed... | vpacksswb | 1/1, 1/1 |
| `_mm_maskz_packs_epi32` | Convert packed signed 32-bit integers from a and b to packed... | vpackssdw | 1/1, 1/1 |
| `_mm_maskz_packus_epi16` | Convert packed signed 16-bit integers from a and b to packed... | vpackuswb | 1/1, 1/1 |
| `_mm_maskz_packus_epi32` | Convert packed signed 32-bit integers from a and b to packed... | vpackusdw | 1/1, 1/1 |
| `_mm_maskz_permute_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vshufpd | 1/1, 1/1 |
| `_mm_maskz_permute_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vshufps | 1/1, 1/1 |
| `_mm_maskz_permutevar_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vpermilpd | 1/1, 1/1 |
| `_mm_maskz_permutevar_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vpermilps | 1/1, 1/1 |
| `_mm_maskz_permutex2var_epi16` | Shuffle 16-bit integers in a and b across lanes using the co... | vperm | 1/1, 1/1 |
| `_mm_maskz_permutex2var_epi32` | Shuffle 32-bit integers in a and b across lanes using the co... | vperm | 1/1, 1/1 |
| `_mm_maskz_permutex2var_epi64` | Shuffle 64-bit integers in a and b across lanes using the co... | vperm | 1/1, 1/1 |
| `_mm_maskz_permutex2var_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vperm | 1/1, 1/1 |
| `_mm_maskz_permutex2var_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vperm | 1/1, 1/1 |
| `_mm_maskz_permutexvar_epi16` | Shuffle 16-bit integers in a across lanes using the correspo... | vpermw | 1/1, 1/1 |
| `_mm_maskz_range_pd` | Calculate the max, min, absolute max, or absolute min (depen... | vrangepd | — |
| `_mm_maskz_range_ps` | Calculate the max, min, absolute max, or absolute min (depen... | vrangeps | — |
| `_mm_maskz_range_round_sd` | Calculate the max, min, absolute max, or absolute min (depen... | vrangesd | — |
| `_mm_maskz_range_round_ss` | Calculate the max, min, absolute max, or absolute min (depen... | vrangess | — |
| `_mm_maskz_range_sd` | Calculate the max, min, absolute max, or absolute min (depen... | vrangesd | — |
| `_mm_maskz_range_ss` | Calculate the max, min, absolute max, or absolute min (depen... | vrangess | — |
| `_mm_maskz_rcp14_pd` | Compute the approximate reciprocal of packed double-precisio... | vrcp14pd | — |
| `_mm_maskz_rcp14_ps` | Compute the approximate reciprocal of packed single-precisio... | vrcp14ps | — |
| `_mm_maskz_rcp14_sd` | Compute the approximate reciprocal of the lower double-preci... | vrcp14sd | — |
| `_mm_maskz_rcp14_ss` | Compute the approximate reciprocal of the lower single-preci... | vrcp14ss | — |
| `_mm_maskz_reduce_pd` | Extract the reduced argument of packed double-precision (64-... | vreducepd | — |
| `_mm_maskz_reduce_ps` | Extract the reduced argument of packed single-precision (32-... | vreduceps | — |
| `_mm_maskz_reduce_round_sd` | Extract the reduced argument of the lower double-precision (... | vreducesd | — |
| `_mm_maskz_reduce_round_ss` | Extract the reduced argument of the lower single-precision (... | vreducess | — |
| `_mm_maskz_reduce_sd` | Extract the reduced argument of the lower double-precision (... | vreducesd | — |
| `_mm_maskz_reduce_ss` | Extract the reduced argument of the lower single-precision (... | vreducess | — |
| `_mm_maskz_rol_epi32` | Rotate the bits in each packed 32-bit integer in a to the le... | vprold | — |
| `_mm_maskz_rol_epi64` | Rotate the bits in each packed 64-bit integer in a to the le... | vprolq | — |
| `_mm_maskz_rolv_epi32` | Rotate the bits in each packed 32-bit integer in a to the le... | vprolvd | — |
| `_mm_maskz_rolv_epi64` | Rotate the bits in each packed 64-bit integer in a to the le... | vprolvq | — |
| `_mm_maskz_ror_epi32` | Rotate the bits in each packed 32-bit integer in a to the ri... | vprold | — |
| `_mm_maskz_ror_epi64` | Rotate the bits in each packed 64-bit integer in a to the ri... | vprolq | — |
| `_mm_maskz_rorv_epi32` | Rotate the bits in each packed 32-bit integer in a to the ri... | vprorvd | — |
| `_mm_maskz_rorv_epi64` | Rotate the bits in each packed 64-bit integer in a to the ri... | vprorvq | — |
| `_mm_maskz_roundscale_pd` | Round packed double-precision (64-bit) floating-point elemen... | vrndscalepd | — |
| `_mm_maskz_roundscale_ps` | Round packed single-precision (32-bit) floating-point elemen... | vrndscaleps | — |
| `_mm_maskz_roundscale_round_sd` | Round the lower double-precision (64-bit) floating-point ele... | vrndscalesd | — |
| `_mm_maskz_roundscale_round_ss` | Round the lower single-precision (32-bit) floating-point ele... | vrndscaless | — |
| `_mm_maskz_roundscale_sd` | Round the lower double-precision (64-bit) floating-point ele... | vrndscalesd | — |
| `_mm_maskz_roundscale_ss` | Round the lower single-precision (32-bit) floating-point ele... | vrndscaless | — |
| `_mm_maskz_rsqrt14_pd` | Compute the approximate reciprocal square root of packed dou... | vrsqrt14pd | — |
| `_mm_maskz_rsqrt14_ps` | Compute the approximate reciprocal square root of packed sin... | vrsqrt14ps | — |
| `_mm_maskz_rsqrt14_sd` | Compute the approximate reciprocal square root of the lower ... | vrsqrt14sd | — |
| `_mm_maskz_rsqrt14_ss` | Compute the approximate reciprocal square root of the lower ... | vrsqrt14ss | — |
| `_mm_maskz_scalef_pd` | Scale the packed double-precision (64-bit) floating-point el... | vscalefpd | — |
| `_mm_maskz_scalef_ps` | Scale the packed single-precision (32-bit) floating-point el... | vscalefps | — |
| `_mm_maskz_scalef_round_sd` | Scale the packed double-precision (64-bit) floating-point el... | vscalefsd | — |
| `_mm_maskz_scalef_round_ss` | Scale the packed single-precision (32-bit) floating-point el... | vscalefss | — |
| `_mm_maskz_scalef_sd` | Scale the packed double-precision (64-bit) floating-point el... | vscalefsd | — |
| `_mm_maskz_scalef_ss` | Scale the packed single-precision (32-bit) floating-point el... | vscalefss | — |
| `_mm_maskz_set1_epi16` | Broadcast the low packed 16-bit integer from a to all elemen... | vpbroadcastw | — |
| `_mm_maskz_set1_epi32` | Broadcast 32-bit integer a to all elements of dst using zero... | vpbroadcastd | — |
| `_mm_maskz_set1_epi64` | Broadcast 64-bit integer a to all elements of dst using zero... | vpbroadcastq | — |
| `_mm_maskz_set1_epi8` | Broadcast 8-bit integer a to all elements of dst using zerom... | vpbroadcast | — |
| `_mm_maskz_shuffle_epi32` | Shuffle 32-bit integers in a within 128-bit lanes using the ... | vpshufd | 1/1, 1/1 |
| `_mm_maskz_shuffle_epi8` | Shuffle packed 8-bit integers in a according to shuffle cont... | vpshufb | 1/1, 1/1 |
| `_mm_maskz_shuffle_pd` | Shuffle double-precision (64-bit) floating-point elements wi... | vshufpd | 1/1, 1/1 |
| `_mm_maskz_shuffle_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vshufps | 1/1, 1/1 |
| `_mm_maskz_shufflehi_epi16` | Shuffle 16-bit integers in the high 64 bits of 128-bit lanes... | vpshufhw | 1/1, 1/1 |
| `_mm_maskz_shufflelo_epi16` | Shuffle 16-bit integers in the low 64 bits of 128-bit lanes ... | vpshuflw | 1/1, 1/1 |
| `_mm_maskz_sll_epi16` | Shift packed 16-bit integers in a left by count while shifti... | vpsllw | 1/1, 1/1 |
| `_mm_maskz_sll_epi32` | Shift packed 32-bit integers in a left by count while shifti... | vpslld | 1/1, 1/1 |
| `_mm_maskz_sll_epi64` | Shift packed 64-bit integers in a left by count while shifti... | vpsllq | 1/1, 1/1 |
| `_mm_maskz_slli_epi16` | Shift packed 16-bit integers in a left by imm8 while shiftin... | vpsllw | 1/1, 1/1 |
| `_mm_maskz_slli_epi32` | Shift packed 32-bit integers in a left by imm8 while shiftin... | vpslld | 1/1, 1/1 |
| `_mm_maskz_slli_epi64` | Shift packed 64-bit integers in a left by imm8 while shiftin... | vpsllq | 1/1, 1/1 |
| `_mm_maskz_sllv_epi16` | Shift packed 16-bit integers in a left by the amount specifi... | vpsllvw | 1/1, 1/1 |
| `_mm_maskz_sllv_epi32` | Shift packed 32-bit integers in a left by the amount specifi... | vpsllvd | 1/1, 1/1 |
| `_mm_maskz_sllv_epi64` | Shift packed 64-bit integers in a left by the amount specifi... | vpsllvq | 1/1, 1/1 |
| `_mm_maskz_sqrt_pd` | Compute the square root of packed double-precision (64-bit) ... | vsqrtpd | 16/14, 20/9 |
| `_mm_maskz_sqrt_ps` | Compute the square root of packed single-precision (32-bit) ... | vsqrtps | 11/7, 14/5 |
| `_mm_maskz_sqrt_round_sd` | Compute the square root of the lower double-precision (64-bi... | vsqrtsd | — |
| `_mm_maskz_sqrt_round_ss` | Compute the square root of the lower single-precision (32-bi... | vsqrtss | — |
| `_mm_maskz_sqrt_sd` | Compute the square root of the lower double-precision (64-bi... | vsqrtsd | 16/14, 20/9 |
| `_mm_maskz_sqrt_ss` | Compute the square root of the lower single-precision (32-bi... | vsqrtss | 11/7, 14/5 |
| `_mm_maskz_sra_epi16` | Shift packed 16-bit integers in a right by count while shift... | vpsraw | 1/1, 1/1 |
| `_mm_maskz_sra_epi32` | Shift packed 32-bit integers in a right by count while shift... | vpsrad | 1/1, 1/1 |
| `_mm_maskz_sra_epi64` | Shift packed 64-bit integers in a right by count while shift... | vpsraq | 1/1, 1/1 |
| `_mm_maskz_srai_epi16` | Shift packed 16-bit integers in a right by imm8 while shifti... | vpsraw | 1/1, 1/1 |
| `_mm_maskz_srai_epi32` | Shift packed 32-bit integers in a right by imm8 while shifti... | vpsrad | 1/1, 1/1 |
| `_mm_maskz_srai_epi64` | Shift packed 64-bit integers in a right by imm8 while shifti... | vpsraq | 1/1, 1/1 |
| `_mm_maskz_srav_epi16` | Shift packed 16-bit integers in a right by the amount specif... | vpsravw | 1/1, 1/1 |
| `_mm_maskz_srav_epi32` | Shift packed 32-bit integers in a right by the amount specif... | vpsravd | 1/1, 1/1 |
| `_mm_maskz_srav_epi64` | Shift packed 64-bit integers in a right by the amount specif... | vpsravq | 1/1, 1/1 |
| `_mm_maskz_srl_epi16` | Shift packed 16-bit integers in a right by count while shift... | vpsrlw | 1/1, 1/1 |
| `_mm_maskz_srl_epi32` | Shift packed 32-bit integers in a right by count while shift... | vpsrld | 1/1, 1/1 |
| `_mm_maskz_srl_epi64` | Shift packed 64-bit integers in a right by count while shift... | vpsrlq | 1/1, 1/1 |
| `_mm_maskz_srli_epi16` | Shift packed 16-bit integers in a right by imm8 while shifti... | vpsrlw | 1/1, 1/1 |
| `_mm_maskz_srli_epi32` | Shift packed 32-bit integers in a right by imm8 while shifti... | vpsrld | 1/1, 1/1 |
| `_mm_maskz_srli_epi64` | Shift packed 64-bit integers in a right by imm8 while shifti... | vpsrlq | 1/1, 1/1 |
| `_mm_maskz_srlv_epi16` | Shift packed 16-bit integers in a right by the amount specif... | vpsrlvw | 1/1, 1/1 |
| `_mm_maskz_srlv_epi32` | Shift packed 32-bit integers in a right by the amount specif... | vpsrlvd | 1/1, 1/1 |
| `_mm_maskz_srlv_epi64` | Shift packed 64-bit integers in a right by the amount specif... | vpsrlvq | 1/1, 1/1 |
| `_mm_maskz_sub_epi16` | Subtract packed 16-bit integers in b from packed 16-bit inte... | vpsubw | 1/1, 1/1 |
| `_mm_maskz_sub_epi32` | Subtract packed 32-bit integers in b from packed 32-bit inte... | vpsubd | 1/1, 1/1 |
| `_mm_maskz_sub_epi64` | Subtract packed 64-bit integers in b from packed 64-bit inte... | vpsubq | 1/1, 1/1 |
| `_mm_maskz_sub_epi8` | Subtract packed 8-bit integers in b from packed 8-bit intege... | vpsubb | 1/1, 1/1 |
| `_mm_maskz_sub_pd` | Subtract packed double-precision (64-bit) floating-point ele... | vsubpd | — |
| `_mm_maskz_sub_ps` | Subtract packed single-precision (32-bit) floating-point ele... | vsubps | — |
| `_mm_maskz_sub_round_sd` | Subtract the lower double-precision (64-bit) floating-point ... | vsubsd | — |
| `_mm_maskz_sub_round_ss` | Subtract the lower single-precision (32-bit) floating-point ... | vsubss | — |
| `_mm_maskz_sub_sd` | Subtract the lower double-precision (64-bit) floating-point ... | vsubsd | — |
| `_mm_maskz_sub_ss` | Subtract the lower single-precision (32-bit) floating-point ... | vsubss | — |
| `_mm_maskz_subs_epi16` | Subtract packed signed 16-bit integers in b from packed 16-b... | vpsubsw | 1/1, 1/1 |
| `_mm_maskz_subs_epi8` | Subtract packed signed 8-bit integers in b from packed 8-bit... | vpsubsb | 1/1, 1/1 |
| `_mm_maskz_subs_epu16` | Subtract packed unsigned 16-bit integers in b from packed un... | vpsubusw | — |
| `_mm_maskz_subs_epu8` | Subtract packed unsigned 8-bit integers in b from packed uns... | vpsubusb | — |
| `_mm_maskz_ternarylogic_epi32` | Bitwise ternary logic that provides the capability to implem... | vpternlogd | — |
| `_mm_maskz_ternarylogic_epi64` | Bitwise ternary logic that provides the capability to implem... | vpternlogq | — |
| `_mm_maskz_unpackhi_epi16` | Unpack and interleave 16-bit integers from the high half of ... | vpunpckhwd | 1/1, 1/1 |
| `_mm_maskz_unpackhi_epi32` | Unpack and interleave 32-bit integers from the high half of ... | vpunpckhdq | 1/1, 1/1 |
| `_mm_maskz_unpackhi_epi64` | Unpack and interleave 64-bit integers from the high half of ... | vpunpckhqdq | 1/1, 1/1 |
| `_mm_maskz_unpackhi_epi8` | Unpack and interleave 8-bit integers from the high half of e... | vpunpckhbw | 1/1, 1/1 |
| `_mm_maskz_unpackhi_pd` | Unpack and interleave double-precision (64-bit) floating-poi... | vunpckhpd | 1/1, 1/1 |
| `_mm_maskz_unpackhi_ps` | Unpack and interleave single-precision (32-bit) floating-poi... | vunpckhps | 1/1, 1/1 |
| `_mm_maskz_unpacklo_epi16` | Unpack and interleave 16-bit integers from the low half of e... | vpunpcklwd | 1/1, 1/1 |
| `_mm_maskz_unpacklo_epi32` | Unpack and interleave 32-bit integers from the low half of e... | vpunpckldq | 1/1, 1/1 |
| `_mm_maskz_unpacklo_epi64` | Unpack and interleave 64-bit integers from the low half of e... | vpunpcklqdq | 1/1, 1/1 |
| `_mm_maskz_unpacklo_epi8` | Unpack and interleave 8-bit integers from the low half of ea... | vpunpcklbw | 1/1, 1/1 |
| `_mm_maskz_unpacklo_pd` | Unpack and interleave double-precision (64-bit) floating-poi... | vunpcklpd | 1/1, 1/1 |
| `_mm_maskz_unpacklo_ps` | Unpack and interleave single-precision (32-bit) floating-poi... | vunpcklps | 1/1, 1/1 |
| `_mm_maskz_xor_epi32` | Compute the bitwise XOR of packed 32-bit integers in a and b... | vpxord | 1/1, 1/1 |
| `_mm_maskz_xor_epi64` | Compute the bitwise XOR of packed 64-bit integers in a and b... | vpxorq | 1/1, 1/1 |
| `_mm_maskz_xor_pd` | Compute the bitwise XOR of packed double-precision (64-bit) ... | vxorpd | 1/1, 1/1 |
| `_mm_maskz_xor_ps` | Compute the bitwise XOR of packed single-precision (32-bit) ... | vxorps | 1/1, 1/1 |
| `_mm_max_epi64` | Compare packed signed 64-bit integers in a and b, and store ... | vpmaxsq | — |
| `_mm_max_epu64` | Compare packed unsigned 64-bit integers in a and b, and stor... | vpmaxuq | — |
| `_mm_max_round_sd` | Compare the lower double-precision (64-bit) floating-point e... | vmaxsd | — |
| `_mm_max_round_ss` | Compare the lower single-precision (32-bit) floating-point e... | vmaxss | — |
| `_mm_min_epi64` | Compare packed signed 64-bit integers in a and b, and store ... | vpminsq | — |
| `_mm_min_epu64` | Compare packed unsigned 64-bit integers in a and b, and stor... | vpminuq | — |
| `_mm_min_round_sd` | Compare the lower double-precision (64-bit) floating-point e... | vminsd | — |
| `_mm_min_round_ss` | Compare the lower single-precision (32-bit) floating-point e... | vminss | — |
| `_mm_movepi16_mask` | Set each bit of mask register k based on the most significan... | vpmovw2m | — |
| `_mm_movepi32_mask` | Set each bit of mask register k based on the most significan... |  | — |
| `_mm_movepi64_mask` | Set each bit of mask register k based on the most significan... |  | — |
| `_mm_movepi8_mask` | Set each bit of mask register k based on the most significan... | vpmovmskb | — |
| `_mm_movm_epi16` | Set each packed 16-bit integer in dst to all ones or all zer... | vpmovm2w | — |
| `_mm_movm_epi32` | Set each packed 32-bit integer in dst to all ones or all zer... | vpmovm2d | — |
| `_mm_movm_epi64` | Set each packed 64-bit integer in dst to all ones or all zer... | vpmovm2q | — |
| `_mm_movm_epi8` | Set each packed 8-bit integer in dst to all ones or all zero... | vpmovm2b | — |
| `_mm_mul_round_sd` | Multiply the lower double-precision (64-bit) floating-point ... | vmulsd | — |
| `_mm_mul_round_ss` | Multiply the lower single-precision (32-bit) floating-point ... | vmulss | — |
| `_mm_mullo_epi64` | Multiply packed 64-bit integers in `a` and `b`, producing in... | vpmullq | — |
| `_mm_or_epi32` | Compute the bitwise OR of packed 32-bit integers in a and b,... | vor | 1/1, 1/1 |
| `_mm_or_epi64` | Compute the bitwise OR of packed 64-bit integers in a and b,... | vor | 1/1, 1/1 |
| `_mm_permutex2var_epi16` | Shuffle 16-bit integers in a and b across lanes using the co... | vperm | 1/1, 1/1 |
| `_mm_permutex2var_epi32` | Shuffle 32-bit integers in a and b across lanes using the co... | vperm | 1/1, 1/1 |
| `_mm_permutex2var_epi64` | Shuffle 64-bit integers in a and b across lanes using the co... | vperm | 1/1, 1/1 |
| `_mm_permutex2var_pd` | Shuffle double-precision (64-bit) floating-point elements in... | vperm | 1/1, 1/1 |
| `_mm_permutex2var_ps` | Shuffle single-precision (32-bit) floating-point elements in... | vperm | 1/1, 1/1 |
| `_mm_permutexvar_epi16` | Shuffle 16-bit integers in a across lanes using the correspo... | vpermw | 1/1, 1/1 |
| `_mm_range_pd` | Calculate the max, min, absolute max, or absolute min (depen... | vrangepd | — |
| `_mm_range_ps` | Calculate the max, min, absolute max, or absolute min (depen... | vrangeps | — |
| `_mm_range_round_sd` | Calculate the max, min, absolute max, or absolute min (depen... | vrangesd | — |
| `_mm_range_round_ss` | Calculate the max, min, absolute max, or absolute min (depen... | vrangess | — |
| `_mm_rcp14_pd` | Compute the approximate reciprocal of packed double-precisio... | vrcp14pd | — |
| `_mm_rcp14_ps` | Compute the approximate reciprocal of packed single-precisio... | vrcp14ps | — |
| `_mm_rcp14_sd` | Compute the approximate reciprocal of the lower double-preci... | vrcp14sd | — |
| `_mm_rcp14_ss` | Compute the approximate reciprocal of the lower single-preci... | vrcp14ss | — |
| `_mm_reduce_add_epi16` | Reduce the packed 16-bit integers in a by addition. Returns ... |  | 1/1, 1/1 |
| `_mm_reduce_add_epi8` | Reduce the packed 8-bit integers in a by addition. Returns t... |  | 1/1, 1/1 |
| `_mm_reduce_and_epi16` | Reduce the packed 16-bit integers in a by bitwise AND. Retur... |  | 1/1, 1/1 |
| `_mm_reduce_and_epi8` | Reduce the packed 8-bit integers in a by bitwise AND. Return... |  | 1/1, 1/1 |
| `_mm_reduce_max_epi16` | Reduce the packed 16-bit integers in a by maximum. Returns t... |  | — |
| `_mm_reduce_max_epi8` | Reduce the packed 8-bit integers in a by maximum. Returns th... |  | — |
| `_mm_reduce_max_epu16` | Reduce the packed unsigned 16-bit integers in a by maximum. ... |  | — |
| `_mm_reduce_max_epu8` | Reduce the packed unsigned 8-bit integers in a by maximum. R... |  | — |
| `_mm_reduce_min_epi16` | Reduce the packed 16-bit integers in a by minimum. Returns t... |  | — |
| `_mm_reduce_min_epi8` | Reduce the packed 8-bit integers in a by minimum. Returns th... |  | — |
| `_mm_reduce_min_epu16` | Reduce the packed unsigned 16-bit integers in a by minimum. ... |  | — |
| `_mm_reduce_min_epu8` | Reduce the packed unsigned 8-bit integers in a by minimum. R... |  | — |
| `_mm_reduce_mul_epi16` | Reduce the packed 16-bit integers in a by multiplication. Re... |  | — |
| `_mm_reduce_mul_epi8` | Reduce the packed 8-bit integers in a by multiplication. Ret... |  | — |
| `_mm_reduce_or_epi16` | Reduce the packed 16-bit integers in a by bitwise OR. Return... |  | 1/1, 1/1 |
| `_mm_reduce_or_epi8` | Reduce the packed 8-bit integers in a by bitwise OR. Returns... |  | 1/1, 1/1 |
| `_mm_reduce_pd` | Extract the reduced argument of packed double-precision (64-... | vreducepd | — |
| `_mm_reduce_ps` | Extract the reduced argument of packed single-precision (32-... | vreduceps | — |
| `_mm_reduce_round_sd` | Extract the reduced argument of the lower double-precision (... | vreducesd | — |
| `_mm_reduce_round_ss` | Extract the reduced argument of the lower single-precision (... | vreducess | — |
| `_mm_reduce_sd` | Extract the reduced argument of the lower double-precision (... | vreducesd | — |
| `_mm_reduce_ss` | Extract the reduced argument of the lower single-precision (... | vreducess | — |
| `_mm_rol_epi32` | Rotate the bits in each packed 32-bit integer in a to the le... | vprold | — |
| `_mm_rol_epi64` | Rotate the bits in each packed 64-bit integer in a to the le... | vprolq | — |
| `_mm_rolv_epi32` | Rotate the bits in each packed 32-bit integer in a to the le... | vprolvd | — |
| `_mm_rolv_epi64` | Rotate the bits in each packed 64-bit integer in a to the le... | vprolvq | — |
| `_mm_ror_epi32` | Rotate the bits in each packed 32-bit integer in a to the ri... | vprold | — |
| `_mm_ror_epi64` | Rotate the bits in each packed 64-bit integer in a to the ri... | vprolq | — |
| `_mm_rorv_epi32` | Rotate the bits in each packed 32-bit integer in a to the ri... | vprorvd | — |
| `_mm_rorv_epi64` | Rotate the bits in each packed 64-bit integer in a to the ri... | vprorvq | — |
| `_mm_roundscale_pd` | Round packed double-precision (64-bit) floating-point elemen... | vrndscalepd | — |
| `_mm_roundscale_ps` | Round packed single-precision (32-bit) floating-point elemen... | vrndscaleps | — |
| `_mm_roundscale_round_sd` | Round the lower double-precision (64-bit) floating-point ele... | vrndscalesd | — |
| `_mm_roundscale_round_ss` | Round the lower single-precision (32-bit) floating-point ele... | vrndscaless | — |
| `_mm_roundscale_sd` | Round the lower double-precision (64-bit) floating-point ele... | vrndscalesd | — |
| `_mm_roundscale_ss` | Round the lower single-precision (32-bit) floating-point ele... | vrndscaless | — |
| `_mm_rsqrt14_pd` | Compute the approximate reciprocal square root of packed dou... | vrsqrt14pd | — |
| `_mm_rsqrt14_ps` | Compute the approximate reciprocal square root of packed sin... | vrsqrt14ps | — |
| `_mm_rsqrt14_sd` | Compute the approximate reciprocal square root of the lower ... | vrsqrt14sd | — |
| `_mm_rsqrt14_ss` | Compute the approximate reciprocal square root of the lower ... | vrsqrt14ss | — |
| `_mm_scalef_pd` | Scale the packed double-precision (64-bit) floating-point el... | vscalefpd | — |
| `_mm_scalef_ps` | Scale the packed single-precision (32-bit) floating-point el... | vscalefps | — |
| `_mm_scalef_round_sd` | Scale the packed double-precision (64-bit) floating-point el... | vscalefsd | — |
| `_mm_scalef_round_ss` | Scale the packed single-precision (32-bit) floating-point el... | vscalefss | — |
| `_mm_scalef_sd` | Scale the packed double-precision (64-bit) floating-point el... | vscalefsd | — |
| `_mm_scalef_ss` | Scale the packed single-precision (32-bit) floating-point el... | vscalefss | — |
| `_mm_sllv_epi16` | Shift packed 16-bit integers in a left by the amount specifi... | vpsllvw | 1/1, 1/1 |
| `_mm_sqrt_round_sd` | Compute the square root of the lower double-precision (64-bi... | vsqrtsd | — |
| `_mm_sqrt_round_ss` | Compute the square root of the lower single-precision (32-bi... | vsqrtss | — |
| `_mm_sra_epi64` | Shift packed 64-bit integers in a right by count while shift... | vpsraq | 1/1, 1/1 |
| `_mm_srai_epi64` | Shift packed 64-bit integers in a right by imm8 while shifti... | vpsraq | 1/1, 1/1 |
| `_mm_srav_epi16` | Shift packed 16-bit integers in a right by the amount specif... | vpsravw | 1/1, 1/1 |
| `_mm_srav_epi64` | Shift packed 64-bit integers in a right by the amount specif... | vpsravq | 1/1, 1/1 |
| `_mm_srlv_epi16` | Shift packed 16-bit integers in a right by the amount specif... | vpsrlvw | 1/1, 1/1 |
| `_mm_sub_round_sd` | Subtract the lower double-precision (64-bit) floating-point ... | vsubsd | — |
| `_mm_sub_round_ss` | Subtract the lower single-precision (32-bit) floating-point ... | vsubss | — |
| `_mm_ternarylogic_epi32` | Bitwise ternary logic that provides the capability to implem... | vpternlogd | — |
| `_mm_ternarylogic_epi64` | Bitwise ternary logic that provides the capability to implem... | vpternlogq | — |
| `_mm_test_epi16_mask` | Compute the bitwise AND of packed 16-bit integers in a and b... | vptestmw | — |
| `_mm_test_epi32_mask` | Compute the bitwise AND of packed 32-bit integers in a and b... | vptestmd | — |
| `_mm_test_epi64_mask` | Compute the bitwise AND of packed 64-bit integers in a and b... | vptestmq | — |
| `_mm_test_epi8_mask` | Compute the bitwise AND of packed 8-bit integers in a and b,... | vptestmb | — |
| `_mm_testn_epi16_mask` | Compute the bitwise NAND of packed 16-bit integers in a and ... | vptestnmw | — |
| `_mm_testn_epi32_mask` | Compute the bitwise NAND of packed 32-bit integers in a and ... | vptestnmd | — |
| `_mm_testn_epi64_mask` | Compute the bitwise NAND of packed 64-bit integers in a and ... | vptestnmq | — |
| `_mm_testn_epi8_mask` | Compute the bitwise NAND of packed 8-bit integers in a and b... | vptestnmb | — |
| `_mm_xor_epi32` | Compute the bitwise XOR of packed 32-bit integers in a and b... | vxor | 1/1, 1/1 |
| `_mm_xor_epi64` | Compute the bitwise XOR of packed 64-bit integers in a and b... | vxor | 1/1, 1/1 |

### Stable, Unsafe (342 intrinsics) — use safe_unaligned_simd

| Name | Description | Safe Variant |
|------|-------------|--------------|
| `_kortest_mask16_u8` | Compute the bitwise OR of 16-bit masks a and b. If the resul... | — |
| `_kortest_mask32_u8` | Compute the bitwise OR of 32-bit masks a and b. If the resul... | — |
| `_kortest_mask64_u8` | Compute the bitwise OR of 64-bit masks a and b. If the resul... | — |
| `_kortest_mask8_u8` | Compute the bitwise OR of 8-bit masks a and b. If the result... | — |
| `_ktest_mask16_u8` | Compute the bitwise AND of 16-bit masks a and b, and if the ... | — |
| `_ktest_mask32_u8` | Compute the bitwise AND of 32-bit masks a and b, and if the ... | — |
| `_ktest_mask64_u8` | Compute the bitwise AND of 64-bit masks a and b, and if the ... | — |
| `_ktest_mask8_u8` | Compute the bitwise AND of 8-bit masks a and b, and if the r... | — |
| `_load_mask16` | Load 16-bit mask from memory | — |
| `_load_mask32` | Load 32-bit mask from memory into k | — |
| `_load_mask64` | Load 64-bit mask from memory into k | — |
| `_load_mask8` | Load 8-bit mask from memory | — |
| `_mm256_i32scatter_epi32` | Stores 8 32-bit integer elements from a to memory starting a... | — |
| `_mm256_i32scatter_epi64` | Scatter 64-bit integers from a into memory using 32-bit indi... | — |
| `_mm256_i32scatter_pd` | Stores 4 double-precision (64-bit) floating-point elements f... | — |
| `_mm256_i32scatter_ps` | Stores 8 single-precision (32-bit) floating-point elements f... | — |
| `_mm256_i64scatter_epi32` | Stores 4 32-bit integer elements from a to memory starting a... | — |
| `_mm256_i64scatter_epi64` | Stores 4 64-bit integer elements from a to memory starting a... | — |
| `_mm256_i64scatter_pd` | Stores 4 double-precision (64-bit) floating-point elements f... | — |
| `_mm256_i64scatter_ps` | Stores 4 single-precision (32-bit) floating-point elements f... | — |
| `_mm256_load_epi32` | Load 256-bits (composed of 8 packed 32-bit integers) from me... | — |
| `_mm256_load_epi64` | Load 256-bits (composed of 4 packed 64-bit integers) from me... | — |
| `_mm256_loadu_epi16` | Load 256-bits (composed of 16 packed 16-bit integers) from m... | safe_unaligned_simd::`_mm256_loadu_epi16` |
| `_mm256_loadu_epi32` | Load 256-bits (composed of 8 packed 32-bit integers) from me... | safe_unaligned_simd::`_mm256_loadu_epi32` |
| `_mm256_loadu_epi64` | Load 256-bits (composed of 4 packed 64-bit integers) from me... | safe_unaligned_simd::`_mm256_loadu_epi64` |
| `_mm256_loadu_epi8` | Load 256-bits (composed of 32 packed 8-bit integers) from me... | safe_unaligned_simd::`_mm256_loadu_epi8` |
| `_mm256_mask_compressstoreu_epi32` | Contiguously store the active 32-bit integers in a (those wi... | — |
| `_mm256_mask_compressstoreu_epi64` | Contiguously store the active 64-bit integers in a (those wi... | — |
| `_mm256_mask_compressstoreu_pd` | Contiguously store the active double-precision (64-bit) floa... | safe_unaligned_simd::`_mm256_mask_compressstoreu_pd` |
| `_mm256_mask_compressstoreu_ps` | Contiguously store the active single-precision (32-bit) floa... | safe_unaligned_simd::`_mm256_mask_compressstoreu_ps` |
| `_mm256_mask_cvtepi16_storeu_epi8` | Convert packed 16-bit integers in a to packed 8-bit integers... | — |
| `_mm256_mask_cvtepi32_storeu_epi16` | Convert packed 32-bit integers in a to packed 16-bit integer... | — |
| `_mm256_mask_cvtepi32_storeu_epi8` | Convert packed 32-bit integers in a to packed 8-bit integers... | — |
| `_mm256_mask_cvtepi64_storeu_epi16` | Convert packed 64-bit integers in a to packed 16-bit integer... | — |
| `_mm256_mask_cvtepi64_storeu_epi32` | Convert packed 64-bit integers in a to packed 32-bit integer... | — |
| `_mm256_mask_cvtepi64_storeu_epi8` | Convert packed 64-bit integers in a to packed 8-bit integers... | — |
| `_mm256_mask_cvtsepi16_storeu_epi8` | Convert packed signed 16-bit integers in a to packed 8-bit i... | — |
| `_mm256_mask_cvtsepi32_storeu_epi16` | Convert packed signed 32-bit integers in a to packed 16-bit ... | — |
| `_mm256_mask_cvtsepi32_storeu_epi8` | Convert packed signed 32-bit integers in a to packed 8-bit i... | — |
| `_mm256_mask_cvtsepi64_storeu_epi16` | Convert packed signed 64-bit integers in a to packed 16-bit ... | — |
| `_mm256_mask_cvtsepi64_storeu_epi32` | Convert packed signed 64-bit integers in a to packed 32-bit ... | — |
| `_mm256_mask_cvtsepi64_storeu_epi8` | Convert packed signed 64-bit integers in a to packed 8-bit i... | — |
| `_mm256_mask_cvtusepi16_storeu_epi8` | Convert packed unsigned 16-bit integers in a to packed unsig... | — |
| `_mm256_mask_cvtusepi32_storeu_epi16` | Convert packed unsigned 32-bit integers in a to packed unsig... | — |
| `_mm256_mask_cvtusepi32_storeu_epi8` | Convert packed unsigned 32-bit integers in a to packed 8-bit... | — |
| `_mm256_mask_cvtusepi64_storeu_epi16` | Convert packed unsigned 64-bit integers in a to packed 16-bi... | — |
| `_mm256_mask_cvtusepi64_storeu_epi32` | Convert packed unsigned 64-bit integers in a to packed 32-bi... | — |
| `_mm256_mask_cvtusepi64_storeu_epi8` | Convert packed unsigned 64-bit integers in a to packed 8-bit... | — |
| `_mm256_mask_expandloadu_epi32` | Load contiguous active 32-bit integers from unaligned memory... | — |
| `_mm256_mask_expandloadu_epi64` | Load contiguous active 64-bit integers from unaligned memory... | — |
| `_mm256_mask_expandloadu_pd` | Load contiguous active double-precision (64-bit) floating-po... | safe_unaligned_simd::`_mm256_mask_expandloadu_pd` |
| `_mm256_mask_expandloadu_ps` | Load contiguous active single-precision (32-bit) floating-po... | safe_unaligned_simd::`_mm256_mask_expandloadu_ps` |
| `_mm256_mask_i32scatter_epi32` | Stores 8 32-bit integer elements from a to memory starting a... | — |
| `_mm256_mask_i32scatter_epi64` | Stores 4 64-bit integer elements from a to memory starting a... | — |
| `_mm256_mask_i32scatter_pd` | Stores 4 double-precision (64-bit) floating-point elements f... | — |
| `_mm256_mask_i32scatter_ps` | Stores 8 single-precision (32-bit) floating-point elements f... | — |
| `_mm256_mask_i64scatter_epi32` | Stores 4 32-bit integer elements from a to memory starting a... | — |
| `_mm256_mask_i64scatter_epi64` | Stores 4 64-bit integer elements from a to memory starting a... | — |
| `_mm256_mask_i64scatter_pd` | Stores 4 double-precision (64-bit) floating-point elements f... | — |
| `_mm256_mask_i64scatter_ps` | Stores 4 single-precision (32-bit) floating-point elements f... | — |
| `_mm256_mask_load_epi32` | Load packed 32-bit integers from memory into dst using write... | — |
| `_mm256_mask_load_epi64` | Load packed 64-bit integers from memory into dst using write... | — |
| `_mm256_mask_load_pd` | Load packed double-precision (64-bit) floating-point element... | — |
| `_mm256_mask_load_ps` | Load packed single-precision (32-bit) floating-point element... | — |
| `_mm256_mask_loadu_epi16` | Load packed 16-bit integers from memory into dst using write... | — |
| `_mm256_mask_loadu_epi32` | Load packed 32-bit integers from memory into dst using write... | — |
| `_mm256_mask_loadu_epi64` | Load packed 64-bit integers from memory into dst using write... | — |
| `_mm256_mask_loadu_epi8` | Load packed 8-bit integers from memory into dst using writem... | — |
| `_mm256_mask_loadu_pd` | Load packed double-precision (64-bit) floating-point element... | safe_unaligned_simd::`_mm256_mask_loadu_pd` |
| `_mm256_mask_loadu_ps` | Load packed single-precision (32-bit) floating-point element... | safe_unaligned_simd::`_mm256_mask_loadu_ps` |
| `_mm256_mask_store_epi32` | Store packed 32-bit integers from a into memory using writem... | — |
| `_mm256_mask_store_epi64` | Store packed 64-bit integers from a into memory using writem... | — |
| `_mm256_mask_store_pd` | Store packed double-precision (64-bit) floating-point elemen... | — |
| `_mm256_mask_store_ps` | Store packed single-precision (32-bit) floating-point elemen... | — |
| `_mm256_mask_storeu_epi16` | Store packed 16-bit integers from a into memory using writem... | safe_unaligned_simd::`_mm256_mask_storeu_epi16` |
| `_mm256_mask_storeu_epi32` | Store packed 32-bit integers from a into memory using writem... | safe_unaligned_simd::`_mm256_mask_storeu_epi32` |
| `_mm256_mask_storeu_epi64` | Store packed 64-bit integers from a into memory using writem... | safe_unaligned_simd::`_mm256_mask_storeu_epi64` |
| `_mm256_mask_storeu_epi8` | Store packed 8-bit integers from a into memory using writema... | safe_unaligned_simd::`_mm256_mask_storeu_epi8` |
| `_mm256_mask_storeu_pd` | Store packed double-precision (64-bit) floating-point elemen... | safe_unaligned_simd::`_mm256_mask_storeu_pd` |
| `_mm256_mask_storeu_ps` | Store packed single-precision (32-bit) floating-point elemen... | safe_unaligned_simd::`_mm256_mask_storeu_ps` |
| `_mm256_maskz_expandloadu_epi32` | Load contiguous active 32-bit integers from unaligned memory... | safe_unaligned_simd::`_mm256_maskz_expandloadu_epi32` |
| `_mm256_maskz_expandloadu_epi64` | Load contiguous active 64-bit integers from unaligned memory... | safe_unaligned_simd::`_mm256_maskz_expandloadu_epi64` |
| `_mm256_maskz_expandloadu_pd` | Load contiguous active double-precision (64-bit) floating-po... | safe_unaligned_simd::`_mm256_maskz_expandloadu_pd` |
| `_mm256_maskz_expandloadu_ps` | Load contiguous active single-precision (32-bit) floating-po... | safe_unaligned_simd::`_mm256_maskz_expandloadu_ps` |
| `_mm256_maskz_load_epi32` | Load packed 32-bit integers from memory into dst using zerom... | — |
| `_mm256_maskz_load_epi64` | Load packed 64-bit integers from memory into dst using zerom... | — |
| `_mm256_maskz_load_pd` | Load packed double-precision (64-bit) floating-point element... | — |
| `_mm256_maskz_load_ps` | Load packed single-precision (32-bit) floating-point element... | — |
| `_mm256_maskz_loadu_epi16` | Load packed 16-bit integers from memory into dst using zerom... | safe_unaligned_simd::`_mm256_maskz_loadu_epi16` |
| `_mm256_maskz_loadu_epi32` | Load packed 32-bit integers from memory into dst using zerom... | safe_unaligned_simd::`_mm256_maskz_loadu_epi32` |
| `_mm256_maskz_loadu_epi64` | Load packed 64-bit integers from memory into dst using zerom... | safe_unaligned_simd::`_mm256_maskz_loadu_epi64` |
| `_mm256_maskz_loadu_epi8` | Load packed 8-bit integers from memory into dst using zeroma... | safe_unaligned_simd::`_mm256_maskz_loadu_epi8` |
| `_mm256_maskz_loadu_pd` | Load packed double-precision (64-bit) floating-point element... | safe_unaligned_simd::`_mm256_maskz_loadu_pd` |
| `_mm256_maskz_loadu_ps` | Load packed single-precision (32-bit) floating-point element... | safe_unaligned_simd::`_mm256_maskz_loadu_ps` |
| `_mm256_mmask_i32gather_epi32` | Loads 8 32-bit integer elements from memory starting at loca... | — |
| `_mm256_mmask_i32gather_epi64` | Loads 4 64-bit integer elements from memory starting at loca... | — |
| `_mm256_mmask_i32gather_pd` | Loads 4 double-precision (64-bit) floating-point elements fr... | — |
| `_mm256_mmask_i32gather_ps` | Loads 8 single-precision (32-bit) floating-point elements fr... | — |
| `_mm256_mmask_i64gather_epi32` | Loads 4 32-bit integer elements from memory starting at loca... | — |
| `_mm256_mmask_i64gather_epi64` | Loads 4 64-bit integer elements from memory starting at loca... | — |
| `_mm256_mmask_i64gather_pd` | Loads 4 double-precision (64-bit) floating-point elements fr... | — |
| `_mm256_mmask_i64gather_ps` | Loads 4 single-precision (32-bit) floating-point elements fr... | — |
| `_mm256_store_epi32` | Store 256-bits (composed of 8 packed 32-bit integers) from a... | — |
| `_mm256_store_epi64` | Store 256-bits (composed of 4 packed 64-bit integers) from a... | — |
| `_mm256_storeu_epi16` | Store 256-bits (composed of 16 packed 16-bit integers) from ... | safe_unaligned_simd::`_mm256_storeu_epi16` |
| `_mm256_storeu_epi32` | Store 256-bits (composed of 8 packed 32-bit integers) from a... | safe_unaligned_simd::`_mm256_storeu_epi32` |
| `_mm256_storeu_epi64` | Store 256-bits (composed of 4 packed 64-bit integers) from a... | safe_unaligned_simd::`_mm256_storeu_epi64` |
| `_mm256_storeu_epi8` | Store 256-bits (composed of 32 packed 8-bit integers) from a... | safe_unaligned_simd::`_mm256_storeu_epi8` |
| `_mm512_i32gather_epi32` | Gather 32-bit integers from memory using 32-bit indices. 32-... | — |
| `_mm512_i32gather_epi64` | Gather 64-bit integers from memory using 32-bit indices. 64-... | — |
| `_mm512_i32gather_pd` | Gather double-precision (64-bit) floating-point elements fro... | — |
| `_mm512_i32gather_ps` | Gather single-precision (32-bit) floating-point elements fro... | — |
| `_mm512_i32logather_epi64` | Loads 8 64-bit integer elements from memory starting at loca... | — |
| `_mm512_i32logather_pd` | Loads 8 double-precision (64-bit) floating-point elements fr... | — |
| `_mm512_i32loscatter_epi64` | Stores 8 64-bit integer elements from a to memory starting a... | — |
| `_mm512_i32loscatter_pd` | Stores 8 double-precision (64-bit) floating-point elements f... | — |
| `_mm512_i32scatter_epi32` | Scatter 32-bit integers from a into memory using 32-bit indi... | — |
| `_mm512_i32scatter_epi64` | Scatter 64-bit integers from a into memory using 32-bit indi... | — |
| `_mm512_i32scatter_pd` | Scatter double-precision (64-bit) floating-point elements fr... | — |
| `_mm512_i32scatter_ps` | Scatter single-precision (32-bit) floating-point elements fr... | — |
| `_mm512_i64gather_epi32` | Gather 32-bit integers from memory using 64-bit indices. 32-... | — |
| `_mm512_i64gather_epi64` | Gather 64-bit integers from memory using 64-bit indices. 64-... | — |
| `_mm512_i64gather_pd` | Gather double-precision (64-bit) floating-point elements fro... | — |
| `_mm512_i64gather_ps` | Gather single-precision (32-bit) floating-point elements fro... | — |
| `_mm512_i64scatter_epi32` | Scatter 32-bit integers from a into memory using 64-bit indi... | — |
| `_mm512_i64scatter_epi64` | Scatter 64-bit integers from a into memory using 64-bit indi... | — |
| `_mm512_i64scatter_pd` | Scatter double-precision (64-bit) floating-point elements fr... | — |
| `_mm512_i64scatter_ps` | Scatter single-precision (32-bit) floating-point elements fr... | — |
| `_mm512_load_epi32` | Load 512-bits (composed of 16 packed 32-bit integers) from m... | — |
| `_mm512_load_epi64` | Load 512-bits (composed of 8 packed 64-bit integers) from me... | — |
| `_mm512_load_pd` | Load 512-bits (composed of 8 packed double-precision (64-bit... | — |
| `_mm512_load_ps` | Load 512-bits (composed of 16 packed single-precision (32-bi... | — |
| `_mm512_load_si512` | Load 512-bits of integer data from memory into dst. mem_addr... | — |
| `_mm512_loadu_epi16` | Load 512-bits (composed of 32 packed 16-bit integers) from m... | safe_unaligned_simd::`_mm512_loadu_epi16` |
| `_mm512_loadu_epi32` | Load 512-bits (composed of 16 packed 32-bit integers) from m... | safe_unaligned_simd::`_mm512_loadu_epi32` |
| `_mm512_loadu_epi64` | Load 512-bits (composed of 8 packed 64-bit integers) from me... | safe_unaligned_simd::`_mm512_loadu_epi64` |
| `_mm512_loadu_epi8` | Load 512-bits (composed of 64 packed 8-bit integers) from me... | safe_unaligned_simd::`_mm512_loadu_epi8` |
| `_mm512_loadu_pd` | Loads 512-bits (composed of 8 packed double-precision (64-bi... | safe_unaligned_simd::`_mm512_loadu_pd` |
| `_mm512_loadu_ps` | Loads 512-bits (composed of 16 packed single-precision (32-b... | safe_unaligned_simd::`_mm512_loadu_ps` |
| `_mm512_loadu_si512` | Load 512-bits of integer data from memory into dst. mem_addr... | safe_unaligned_simd::`_mm512_loadu_si512` |
| `_mm512_mask_compressstoreu_epi32` | Contiguously store the active 32-bit integers in a (those wi... | — |
| `_mm512_mask_compressstoreu_epi64` | Contiguously store the active 64-bit integers in a (those wi... | — |
| `_mm512_mask_compressstoreu_pd` | Contiguously store the active double-precision (64-bit) floa... | safe_unaligned_simd::`_mm512_mask_compressstoreu_pd` |
| `_mm512_mask_compressstoreu_ps` | Contiguously store the active single-precision (32-bit) floa... | safe_unaligned_simd::`_mm512_mask_compressstoreu_ps` |
| `_mm512_mask_cvtepi16_storeu_epi8` | Convert packed 16-bit integers in a to packed 8-bit integers... | — |
| `_mm512_mask_cvtepi32_storeu_epi16` | Convert packed 32-bit integers in a to packed 16-bit integer... | — |
| `_mm512_mask_cvtepi32_storeu_epi8` | Convert packed 32-bit integers in a to packed 8-bit integers... | — |
| `_mm512_mask_cvtepi64_storeu_epi16` | Convert packed 64-bit integers in a to packed 16-bit integer... | — |
| `_mm512_mask_cvtepi64_storeu_epi32` | Convert packed 64-bit integers in a to packed 32-bit integer... | — |
| `_mm512_mask_cvtepi64_storeu_epi8` | Convert packed 64-bit integers in a to packed 8-bit integers... | — |
| `_mm512_mask_cvtsepi16_storeu_epi8` | Convert packed signed 16-bit integers in a to packed 8-bit i... | — |
| `_mm512_mask_cvtsepi32_storeu_epi16` | Convert packed signed 32-bit integers in a to packed 16-bit ... | — |
| `_mm512_mask_cvtsepi32_storeu_epi8` | Convert packed signed 32-bit integers in a to packed 8-bit i... | — |
| `_mm512_mask_cvtsepi64_storeu_epi16` | Convert packed signed 64-bit integers in a to packed 16-bit ... | — |
| `_mm512_mask_cvtsepi64_storeu_epi32` | Convert packed signed 64-bit integers in a to packed 32-bit ... | — |
| `_mm512_mask_cvtsepi64_storeu_epi8` | Convert packed signed 64-bit integers in a to packed 8-bit i... | — |
| `_mm512_mask_cvtusepi16_storeu_epi8` | Convert packed unsigned 16-bit integers in a to packed unsig... | — |
| `_mm512_mask_cvtusepi32_storeu_epi16` | Convert packed unsigned 32-bit integers in a to packed 16-bi... | — |
| `_mm512_mask_cvtusepi32_storeu_epi8` | Convert packed unsigned 32-bit integers in a to packed 8-bit... | — |
| `_mm512_mask_cvtusepi64_storeu_epi16` | Convert packed unsigned 64-bit integers in a to packed 16-bi... | — |
| `_mm512_mask_cvtusepi64_storeu_epi32` | Convert packed unsigned 64-bit integers in a to packed 32-bi... | — |
| `_mm512_mask_cvtusepi64_storeu_epi8` | Convert packed unsigned 64-bit integers in a to packed 8-bit... | — |
| `_mm512_mask_expandloadu_epi32` | Load contiguous active 32-bit integers from unaligned memory... | — |
| `_mm512_mask_expandloadu_epi64` | Load contiguous active 64-bit integers from unaligned memory... | — |
| `_mm512_mask_expandloadu_pd` | Load contiguous active double-precision (64-bit) floating-po... | safe_unaligned_simd::`_mm512_mask_expandloadu_pd` |
| `_mm512_mask_expandloadu_ps` | Load contiguous active single-precision (32-bit) floating-po... | safe_unaligned_simd::`_mm512_mask_expandloadu_ps` |
| `_mm512_mask_i32gather_epi32` | Gather 32-bit integers from memory using 32-bit indices. 32-... | — |
| `_mm512_mask_i32gather_epi64` | Gather 64-bit integers from memory using 32-bit indices. 64-... | — |
| `_mm512_mask_i32gather_pd` | Gather double-precision (64-bit) floating-point elements fro... | — |
| `_mm512_mask_i32gather_ps` | Gather single-precision (32-bit) floating-point elements fro... | — |
| `_mm512_mask_i32logather_epi64` | Loads 8 64-bit integer elements from memory starting at loca... | — |
| `_mm512_mask_i32logather_pd` | Loads 8 double-precision (64-bit) floating-point elements fr... | — |
| `_mm512_mask_i32loscatter_epi64` | Stores 8 64-bit integer elements from a to memory starting a... | — |
| `_mm512_mask_i32loscatter_pd` | Stores 8 double-precision (64-bit) floating-point elements f... | — |
| `_mm512_mask_i32scatter_epi32` | Scatter 32-bit integers from a into memory using 32-bit indi... | — |
| `_mm512_mask_i32scatter_epi64` | Scatter 64-bit integers from a into memory using 32-bit indi... | — |
| `_mm512_mask_i32scatter_pd` | Scatter double-precision (64-bit) floating-point elements fr... | — |
| `_mm512_mask_i32scatter_ps` | Scatter single-precision (32-bit) floating-point elements fr... | — |
| `_mm512_mask_i64gather_epi32` | Gather 32-bit integers from memory using 64-bit indices. 32-... | — |
| `_mm512_mask_i64gather_epi64` | Gather 64-bit integers from memory using 64-bit indices. 64-... | — |
| `_mm512_mask_i64gather_pd` | Gather double-precision (64-bit) floating-point elements fro... | — |
| `_mm512_mask_i64gather_ps` | Gather single-precision (32-bit) floating-point elements fro... | — |
| `_mm512_mask_i64scatter_epi32` | Scatter 32-bit integers from a into memory using 64-bit indi... | — |
| `_mm512_mask_i64scatter_epi64` | Scatter 64-bit integers from a into memory using 64-bit indi... | — |
| `_mm512_mask_i64scatter_pd` | Scatter double-precision (64-bit) floating-point elements fr... | — |
| `_mm512_mask_i64scatter_ps` | Scatter single-precision (32-bit) floating-point elements fr... | — |
| `_mm512_mask_load_epi32` | Load packed 32-bit integers from memory into dst using write... | — |
| `_mm512_mask_load_epi64` | Load packed 64-bit integers from memory into dst using write... | — |
| `_mm512_mask_load_pd` | Load packed double-precision (64-bit) floating-point element... | — |
| `_mm512_mask_load_ps` | Load packed single-precision (32-bit) floating-point element... | — |
| `_mm512_mask_loadu_epi16` | Load packed 16-bit integers from memory into dst using write... | — |
| `_mm512_mask_loadu_epi32` | Load packed 32-bit integers from memory into dst using write... | — |
| `_mm512_mask_loadu_epi64` | Load packed 64-bit integers from memory into dst using write... | — |
| `_mm512_mask_loadu_epi8` | Load packed 8-bit integers from memory into dst using writem... | — |
| `_mm512_mask_loadu_pd` | Load packed double-precision (64-bit) floating-point element... | safe_unaligned_simd::`_mm512_mask_loadu_pd` |
| `_mm512_mask_loadu_ps` | Load packed single-precision (32-bit) floating-point element... | safe_unaligned_simd::`_mm512_mask_loadu_ps` |
| `_mm512_mask_store_epi32` | Store packed 32-bit integers from a into memory using writem... | — |
| `_mm512_mask_store_epi64` | Store packed 64-bit integers from a into memory using writem... | — |
| `_mm512_mask_store_pd` | Store packed double-precision (64-bit) floating-point elemen... | — |
| `_mm512_mask_store_ps` | Store packed single-precision (32-bit) floating-point elemen... | — |
| `_mm512_mask_storeu_epi16` | Store packed 16-bit integers from a into memory using writem... | safe_unaligned_simd::`_mm512_mask_storeu_epi16` |
| `_mm512_mask_storeu_epi32` | Store packed 32-bit integers from a into memory using writem... | safe_unaligned_simd::`_mm512_mask_storeu_epi32` |
| `_mm512_mask_storeu_epi64` | Store packed 64-bit integers from a into memory using writem... | safe_unaligned_simd::`_mm512_mask_storeu_epi64` |
| `_mm512_mask_storeu_epi8` | Store packed 8-bit integers from a into memory using writema... | safe_unaligned_simd::`_mm512_mask_storeu_epi8` |
| `_mm512_mask_storeu_pd` | Store packed double-precision (64-bit) floating-point elemen... | safe_unaligned_simd::`_mm512_mask_storeu_pd` |
| `_mm512_mask_storeu_ps` | Store packed single-precision (32-bit) floating-point elemen... | safe_unaligned_simd::`_mm512_mask_storeu_ps` |
| `_mm512_maskz_expandloadu_epi32` | Load contiguous active 32-bit integers from unaligned memory... | — |
| `_mm512_maskz_expandloadu_epi64` | Load contiguous active 64-bit integers from unaligned memory... | safe_unaligned_simd::`_mm512_maskz_expandloadu_epi64` |
| `_mm512_maskz_expandloadu_pd` | Load contiguous active double-precision (64-bit) floating-po... | safe_unaligned_simd::`_mm512_maskz_expandloadu_pd` |
| `_mm512_maskz_expandloadu_ps` | Load contiguous active single-precision (32-bit) floating-po... | safe_unaligned_simd::`_mm512_maskz_expandloadu_ps` |
| `_mm512_maskz_load_epi32` | Load packed 32-bit integers from memory into dst using zerom... | — |
| `_mm512_maskz_load_epi64` | Load packed 64-bit integers from memory into dst using zerom... | — |
| `_mm512_maskz_load_pd` | Load packed double-precision (64-bit) floating-point element... | — |
| `_mm512_maskz_load_ps` | Load packed single-precision (32-bit) floating-point element... | — |
| `_mm512_maskz_loadu_epi16` | Load packed 16-bit integers from memory into dst using zerom... | safe_unaligned_simd::`_mm512_maskz_loadu_epi16` |
| `_mm512_maskz_loadu_epi32` | Load packed 32-bit integers from memory into dst using zerom... | safe_unaligned_simd::`_mm512_maskz_loadu_epi32` |
| `_mm512_maskz_loadu_epi64` | Load packed 64-bit integers from memory into dst using zerom... | safe_unaligned_simd::`_mm512_maskz_loadu_epi64` |
| `_mm512_maskz_loadu_epi8` | Load packed 8-bit integers from memory into dst using zeroma... | safe_unaligned_simd::`_mm512_maskz_loadu_epi8` |
| `_mm512_maskz_loadu_pd` | Load packed double-precision (64-bit) floating-point element... | safe_unaligned_simd::`_mm512_maskz_loadu_pd` |
| `_mm512_maskz_loadu_ps` | Load packed single-precision (32-bit) floating-point element... | safe_unaligned_simd::`_mm512_maskz_loadu_ps` |
| `_mm512_store_epi32` | Store 512-bits (composed of 16 packed 32-bit integers) from ... | — |
| `_mm512_store_epi64` | Store 512-bits (composed of 8 packed 64-bit integers) from a... | — |
| `_mm512_store_pd` | Store 512-bits (composed of 8 packed double-precision (64-bi... | — |
| `_mm512_store_ps` | Store 512-bits of integer data from a into memory. mem_addr ... | — |
| `_mm512_store_si512` | Store 512-bits of integer data from a into memory. mem_addr ... | — |
| `_mm512_storeu_epi16` | Store 512-bits (composed of 32 packed 16-bit integers) from ... | safe_unaligned_simd::`_mm512_storeu_epi16` |
| `_mm512_storeu_epi32` | Store 512-bits (composed of 16 packed 32-bit integers) from ... | safe_unaligned_simd::`_mm512_storeu_epi32` |
| `_mm512_storeu_epi64` | Store 512-bits (composed of 8 packed 64-bit integers) from a... | safe_unaligned_simd::`_mm512_storeu_epi64` |
| `_mm512_storeu_epi8` | Store 512-bits (composed of 64 packed 8-bit integers) from a... | safe_unaligned_simd::`_mm512_storeu_epi8` |
| `_mm512_storeu_pd` | Stores 512-bits (composed of 8 packed double-precision (64-b... | safe_unaligned_simd::`_mm512_storeu_pd` |
| `_mm512_storeu_ps` | Stores 512-bits (composed of 16 packed single-precision (32-... | safe_unaligned_simd::`_mm512_storeu_ps` |
| `_mm512_storeu_si512` | Store 512-bits of integer data from a into memory. mem_addr ... | safe_unaligned_simd::`_mm512_storeu_si512` |
| `_mm512_stream_load_si512` | Load 512-bits of integer data from memory into dst using a n... | — |
| `_mm512_stream_pd` | Store 512-bits (composed of 8 packed double-precision (64-bi... | — |
| `_mm512_stream_ps` | Store 512-bits (composed of 16 packed single-precision (32-b... | — |
| `_mm512_stream_si512` | Store 512-bits of integer data from a into memory using a no... | — |
| `_mm_i32scatter_epi32` | Stores 4 32-bit integer elements from a to memory starting a... | — |
| `_mm_i32scatter_epi64` | Stores 2 64-bit integer elements from a to memory starting a... | — |
| `_mm_i32scatter_pd` | Stores 2 double-precision (64-bit) floating-point elements f... | — |
| `_mm_i32scatter_ps` | Stores 4 single-precision (32-bit) floating-point elements f... | — |
| `_mm_i64scatter_epi32` | Stores 2 32-bit integer elements from a to memory starting a... | — |
| `_mm_i64scatter_epi64` | Stores 2 64-bit integer elements from a to memory starting a... | — |
| `_mm_i64scatter_pd` | Stores 2 double-precision (64-bit) floating-point elements f... | — |
| `_mm_i64scatter_ps` | Stores 2 single-precision (32-bit) floating-point elements f... | — |
| `_mm_load_epi32` | Load 128-bits (composed of 4 packed 32-bit integers) from me... | — |
| `_mm_load_epi64` | Load 128-bits (composed of 2 packed 64-bit integers) from me... | — |
| `_mm_loadu_epi16` | Load 128-bits (composed of 8 packed 16-bit integers) from me... | safe_unaligned_simd::`_mm_loadu_epi16` |
| `_mm_loadu_epi32` | Load 128-bits (composed of 4 packed 32-bit integers) from me... | safe_unaligned_simd::`_mm_loadu_epi32` |
| `_mm_loadu_epi64` | Load 128-bits (composed of 2 packed 64-bit integers) from me... | safe_unaligned_simd::`_mm_loadu_epi64` |
| `_mm_loadu_epi8` | Load 128-bits (composed of 16 packed 8-bit integers) from me... | safe_unaligned_simd::`_mm_loadu_epi8` |
| `_mm_mask_compressstoreu_epi32` | Contiguously store the active 32-bit integers in a (those wi... | — |
| `_mm_mask_compressstoreu_epi64` | Contiguously store the active 64-bit integers in a (those wi... | — |
| `_mm_mask_compressstoreu_pd` | Contiguously store the active double-precision (64-bit) floa... | safe_unaligned_simd::`_mm_mask_compressstoreu_pd` |
| `_mm_mask_compressstoreu_ps` | Contiguously store the active single-precision (32-bit) floa... | safe_unaligned_simd::`_mm_mask_compressstoreu_ps` |
| `_mm_mask_cvtepi16_storeu_epi8` | Convert packed 16-bit integers in a to packed 8-bit integers... | — |
| `_mm_mask_cvtepi32_storeu_epi16` | Convert packed 32-bit integers in a to packed 16-bit integer... | — |
| `_mm_mask_cvtepi32_storeu_epi8` | Convert packed 32-bit integers in a to packed 8-bit integers... | — |
| `_mm_mask_cvtepi64_storeu_epi16` | Convert packed 64-bit integers in a to packed 16-bit integer... | — |
| `_mm_mask_cvtepi64_storeu_epi32` | Convert packed 64-bit integers in a to packed 32-bit integer... | — |
| `_mm_mask_cvtepi64_storeu_epi8` | Convert packed 64-bit integers in a to packed 8-bit integers... | — |
| `_mm_mask_cvtsepi16_storeu_epi8` | Convert packed signed 16-bit integers in a to packed 8-bit i... | — |
| `_mm_mask_cvtsepi32_storeu_epi16` | Convert packed signed 32-bit integers in a to packed 16-bit ... | — |
| `_mm_mask_cvtsepi32_storeu_epi8` | Convert packed signed 32-bit integers in a to packed 8-bit i... | — |
| `_mm_mask_cvtsepi64_storeu_epi16` | Convert packed signed 64-bit integers in a to packed 16-bit ... | — |
| `_mm_mask_cvtsepi64_storeu_epi32` | Convert packed signed 64-bit integers in a to packed 32-bit ... | — |
| `_mm_mask_cvtsepi64_storeu_epi8` | Convert packed signed 64-bit integers in a to packed 8-bit i... | — |
| `_mm_mask_cvtusepi16_storeu_epi8` | Convert packed unsigned 16-bit integers in a to packed unsig... | — |
| `_mm_mask_cvtusepi32_storeu_epi16` | Convert packed unsigned 32-bit integers in a to packed unsig... | — |
| `_mm_mask_cvtusepi32_storeu_epi8` | Convert packed unsigned 32-bit integers in a to packed 8-bit... | — |
| `_mm_mask_cvtusepi64_storeu_epi16` | Convert packed unsigned 64-bit integers in a to packed 16-bi... | — |
| `_mm_mask_cvtusepi64_storeu_epi32` | Convert packed unsigned 64-bit integers in a to packed 32-bi... | — |
| `_mm_mask_cvtusepi64_storeu_epi8` | Convert packed unsigned 64-bit integers in a to packed 8-bit... | — |
| `_mm_mask_expandloadu_epi32` | Load contiguous active 32-bit integers from unaligned memory... | — |
| `_mm_mask_expandloadu_epi64` | Load contiguous active 64-bit integers from unaligned memory... | — |
| `_mm_mask_expandloadu_pd` | Load contiguous active double-precision (64-bit) floating-po... | safe_unaligned_simd::`_mm_mask_expandloadu_pd` |
| `_mm_mask_expandloadu_ps` | Load contiguous active single-precision (32-bit) floating-po... | safe_unaligned_simd::`_mm_mask_expandloadu_ps` |
| `_mm_mask_i32scatter_epi32` | Stores 4 32-bit integer elements from a to memory starting a... | — |
| `_mm_mask_i32scatter_epi64` | Stores 2 64-bit integer elements from a to memory starting a... | — |
| `_mm_mask_i32scatter_pd` | Stores 2 double-precision (64-bit) floating-point elements f... | — |
| `_mm_mask_i32scatter_ps` | Stores 4 single-precision (32-bit) floating-point elements f... | — |
| `_mm_mask_i64scatter_epi32` | Stores 2 32-bit integer elements from a to memory starting a... | — |
| `_mm_mask_i64scatter_epi64` | Stores 2 64-bit integer elements from a to memory starting a... | — |
| `_mm_mask_i64scatter_pd` | Stores 2 double-precision (64-bit) floating-point elements f... | — |
| `_mm_mask_i64scatter_ps` | Stores 2 single-precision (32-bit) floating-point elements f... | — |
| `_mm_mask_load_epi32` | Load packed 32-bit integers from memory into dst using write... | — |
| `_mm_mask_load_epi64` | Load packed 64-bit integers from memory into dst using write... | — |
| `_mm_mask_load_pd` | Load packed double-precision (64-bit) floating-point element... | — |
| `_mm_mask_load_ps` | Load packed single-precision (32-bit) floating-point element... | — |
| `_mm_mask_load_sd` | Load a double-precision (64-bit) floating-point element from... | — |
| `_mm_mask_load_ss` | Load a single-precision (32-bit) floating-point element from... | — |
| `_mm_mask_loadu_epi16` | Load packed 16-bit integers from memory into dst using write... | — |
| `_mm_mask_loadu_epi32` | Load packed 32-bit integers from memory into dst using write... | — |
| `_mm_mask_loadu_epi64` | Load packed 64-bit integers from memory into dst using write... | — |
| `_mm_mask_loadu_epi8` | Load packed 8-bit integers from memory into dst using writem... | — |
| `_mm_mask_loadu_pd` | Load packed double-precision (64-bit) floating-point element... | safe_unaligned_simd::`_mm_mask_loadu_pd` |
| `_mm_mask_loadu_ps` | Load packed single-precision (32-bit) floating-point element... | safe_unaligned_simd::`_mm_mask_loadu_ps` |
| `_mm_mask_store_epi32` | Store packed 32-bit integers from a into memory using writem... | — |
| `_mm_mask_store_epi64` | Store packed 64-bit integers from a into memory using writem... | — |
| `_mm_mask_store_pd` | Store packed double-precision (64-bit) floating-point elemen... | — |
| `_mm_mask_store_ps` | Store packed single-precision (32-bit) floating-point elemen... | — |
| `_mm_mask_store_sd` | Store a double-precision (64-bit) floating-point element fro... | — |
| `_mm_mask_store_ss` | Store a single-precision (32-bit) floating-point element fro... | — |
| `_mm_mask_storeu_epi16` | Store packed 16-bit integers from a into memory using writem... | safe_unaligned_simd::`_mm_mask_storeu_epi16` |
| `_mm_mask_storeu_epi32` | Store packed 32-bit integers from a into memory using writem... | safe_unaligned_simd::`_mm_mask_storeu_epi32` |
| `_mm_mask_storeu_epi64` | Store packed 64-bit integers from a into memory using writem... | safe_unaligned_simd::`_mm_mask_storeu_epi64` |
| `_mm_mask_storeu_epi8` | Store packed 8-bit integers from a into memory using writema... | safe_unaligned_simd::`_mm_mask_storeu_epi8` |
| `_mm_mask_storeu_pd` | Store packed double-precision (64-bit) floating-point elemen... | safe_unaligned_simd::`_mm_mask_storeu_pd` |
| `_mm_mask_storeu_ps` | Store packed single-precision (32-bit) floating-point elemen... | safe_unaligned_simd::`_mm_mask_storeu_ps` |
| `_mm_maskz_expandloadu_epi32` | Load contiguous active 32-bit integers from unaligned memory... | safe_unaligned_simd::`_mm_maskz_expandloadu_epi32` |
| `_mm_maskz_expandloadu_epi64` | Load contiguous active 64-bit integers from unaligned memory... | safe_unaligned_simd::`_mm_maskz_expandloadu_epi64` |
| `_mm_maskz_expandloadu_pd` | Load contiguous active double-precision (64-bit) floating-po... | safe_unaligned_simd::`_mm_maskz_expandloadu_pd` |
| `_mm_maskz_expandloadu_ps` | Load contiguous active single-precision (32-bit) floating-po... | safe_unaligned_simd::`_mm_maskz_expandloadu_ps` |
| `_mm_maskz_load_epi32` | Load packed 32-bit integers from memory into dst using zerom... | — |
| `_mm_maskz_load_epi64` | Load packed 64-bit integers from memory into dst using zerom... | — |
| `_mm_maskz_load_pd` | Load packed double-precision (64-bit) floating-point element... | — |
| `_mm_maskz_load_ps` | Load packed single-precision (32-bit) floating-point element... | — |
| `_mm_maskz_load_sd` | Load a double-precision (64-bit) floating-point element from... | — |
| `_mm_maskz_load_ss` | Load a single-precision (32-bit) floating-point element from... | — |
| `_mm_maskz_loadu_epi16` | Load packed 16-bit integers from memory into dst using zerom... | safe_unaligned_simd::`_mm_maskz_loadu_epi16` |
| `_mm_maskz_loadu_epi32` | Load packed 32-bit integers from memory into dst using zerom... | safe_unaligned_simd::`_mm_maskz_loadu_epi32` |
| `_mm_maskz_loadu_epi64` | Load packed 64-bit integers from memory into dst using zerom... | safe_unaligned_simd::`_mm_maskz_loadu_epi64` |
| `_mm_maskz_loadu_epi8` | Load packed 8-bit integers from memory into dst using zeroma... | safe_unaligned_simd::`_mm_maskz_loadu_epi8` |
| `_mm_maskz_loadu_pd` | Load packed double-precision (64-bit) floating-point element... | safe_unaligned_simd::`_mm_maskz_loadu_pd` |
| `_mm_maskz_loadu_ps` | Load packed single-precision (32-bit) floating-point element... | safe_unaligned_simd::`_mm_maskz_loadu_ps` |
| `_mm_mmask_i32gather_epi32` | Loads 4 32-bit integer elements from memory starting at loca... | — |
| `_mm_mmask_i32gather_epi64` | Loads 2 64-bit integer elements from memory starting at loca... | — |
| `_mm_mmask_i32gather_pd` | Loads 2 double-precision (64-bit) floating-point elements fr... | — |
| `_mm_mmask_i32gather_ps` | Loads 4 single-precision (32-bit) floating-point elements fr... | — |
| `_mm_mmask_i64gather_epi32` | Loads 2 32-bit integer elements from memory starting at loca... | — |
| `_mm_mmask_i64gather_epi64` | Loads 2 64-bit integer elements from memory starting at loca... | — |
| `_mm_mmask_i64gather_pd` | Loads 2 double-precision (64-bit) floating-point elements fr... | — |
| `_mm_mmask_i64gather_ps` | Loads 2 single-precision (32-bit) floating-point elements fr... | — |
| `_mm_store_epi32` | Store 128-bits (composed of 4 packed 32-bit integers) from a... | — |
| `_mm_store_epi64` | Store 128-bits (composed of 2 packed 64-bit integers) from a... | — |
| `_mm_storeu_epi16` | Store 128-bits (composed of 8 packed 16-bit integers) from a... | safe_unaligned_simd::`_mm_storeu_epi16` |
| `_mm_storeu_epi32` | Store 128-bits (composed of 4 packed 32-bit integers) from a... | safe_unaligned_simd::`_mm_storeu_epi32` |
| `_mm_storeu_epi64` | Store 128-bits (composed of 2 packed 64-bit integers) from a... | safe_unaligned_simd::`_mm_storeu_epi64` |
| `_mm_storeu_epi8` | Store 128-bits (composed of 16 packed 8-bit integers) from a... | safe_unaligned_simd::`_mm_storeu_epi8` |
| `_store_mask16` | Store 16-bit mask to memory | — |
| `_store_mask32` | Store 32-bit mask from a into memory | — |
| `_store_mask64` | Store 64-bit mask from a into memory | — |
| `_store_mask8` | Store 8-bit mask to memory | — |



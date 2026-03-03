# X64V4xToken (Avx512ModernToken) — x86-64-v4x

Proof that extended AVX-512 features are available (x86-64-v4x = Ice Lake / Zen 4 level).

**Architecture:** x86_64 | **Features:** sse, sse2, sse3, ssse3, sse4.1, sse4.2, popcnt, cmpxchg16b, avx, avx2, fma, bmi1, bmi2, f16c, lzcnt, movbe, pclmulqdq, aes, avx512f, avx512bw, avx512cd, avx512dq, avx512vl, avx512vpopcntdq, avx512ifma, avx512vbmi, avx512vbmi2, avx512bitalg, avx512vnni, vpclmulqdq, gfni, vaes
**Total intrinsics:** 308 (290 safe, 18 unsafe, 308 stable, 0 unstable/unknown)

## Usage

```rust
use archmage::prelude::*;

if let Some(token) = X64V4xToken::summon() {
    process(token, &mut data);
}

#[arcane]  // Entry point only
fn process(token: X64V4xToken, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(16) {
        process_chunk(token, chunk.try_into().unwrap());
    }
}

#[rite]  // All inner helpers
fn process_chunk(_: X64V4xToken, chunk: &mut [f32; 16]) {
    let v = _mm512_loadu_ps(chunk.as_ptr());  // safe inside #[rite]
    let doubled = _mm512_add_ps(v, v);
    _mm512_storeu_ps(chunk.as_mut_ptr(), doubled);
}
// Use #![forbid(unsafe_code)] with safe_unaligned_simd for memory ops.
```

## Safe Memory Operations (safe_unaligned_simd)

| Function | Safe Signature |
|----------|---------------|
| `_mm256_maskz_expandloadu_epi8` | `fn _mm256_maskz_expandloadu_epi8<T: Is256BitsUnaligned>(k: __mmask32, mem_addr: &T) -> __m256i` |
| `_mm512_maskz_expandloadu_epi8` | `fn _mm512_maskz_expandloadu_epi8<T: Is512BitsUnaligned>(k: __mmask64, mem_addr: &T) -> __m512i` |
| `_mm_maskz_expandloadu_epi16` | `fn _mm_maskz_expandloadu_epi16<T: Is128BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m128i` |
| `_mm_maskz_expandloadu_epi8` | `fn _mm_maskz_expandloadu_epi8<T: Is128BitsUnaligned>(k: __mmask16, mem_addr: &T) -> __m128i` |


## All Intrinsics

### Stable, Safe (290 intrinsics)

| Name | Description | Instruction | Timing (H/Z4) |
|------|-------------|-------------|---------------|
| `_mm256_bitshuffle_epi64_mask` | Considers the input `b` as packed 64-bit integers and `c` as... | vpshufbitqmb | — |
| `_mm256_dpbusd_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusd | — |
| `_mm256_dpbusds_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusds | — |
| `_mm256_dpwssd_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssd | — |
| `_mm256_dpwssds_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssds | — |
| `_mm256_gf2p8affine_epi64_epi8` | Performs an affine transformation on the packed bytes in x. ... | vgf2p8affineqb | — |
| `_mm256_gf2p8affineinv_epi64_epi8` | Performs an affine transformation on the inverted packed byt... | vgf2p8affineinvqb | — |
| `_mm256_gf2p8mul_epi8` | Performs a multiplication in GF(2^8) on the packed bytes. Th... | vgf2p8mulb | — |
| `_mm256_madd52hi_epu64` | Multiply packed unsigned 52-bit integers in each 64-bit elem... | vpmadd52huq | — |
| `_mm256_madd52lo_epu64` | Multiply packed unsigned 52-bit integers in each 64-bit elem... | vpmadd52luq | — |
| `_mm256_mask2_permutex2var_epi8` | Shuffle 8-bit integers in a and b across lanes using the cor... | vpermi2b | 3/1, 2/1 |
| `_mm256_mask_bitshuffle_epi64_mask` | Considers the input `b` as packed 64-bit integers and `c` as... | vpshufbitqmb | — |
| `_mm256_mask_compress_epi16` | Contiguously store the active 16-bit integers in a (those wi... | vpcompressw | — |
| `_mm256_mask_compress_epi8` | Contiguously store the active 8-bit integers in a (those wit... | vpcompressb | — |
| `_mm256_mask_dpbusd_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusd | — |
| `_mm256_mask_dpbusds_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusds | — |
| `_mm256_mask_dpwssd_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssd | — |
| `_mm256_mask_dpwssds_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssds | — |
| `_mm256_mask_expand_epi16` | Load contiguous active 16-bit integers from a (those with th... | vpexpandw | — |
| `_mm256_mask_expand_epi8` | Load contiguous active 8-bit integers from a (those with the... | vpexpandb | — |
| `_mm256_mask_gf2p8affine_epi64_epi8` | Performs an affine transformation on the packed bytes in x. ... | vgf2p8affineqb | — |
| `_mm256_mask_gf2p8affineinv_epi64_epi8` | Performs an affine transformation on the inverted packed byt... | vgf2p8affineinvqb | — |
| `_mm256_mask_gf2p8mul_epi8` | Performs a multiplication in GF(2^8) on the packed bytes. Th... | vgf2p8mulb | — |
| `_mm256_mask_madd52hi_epu64` | Multiply packed unsigned 52-bit integers in each 64-bit elem... | vpmadd52huq | — |
| `_mm256_mask_madd52lo_epu64` | Multiply packed unsigned 52-bit integers in each 64-bit elem... | vpmadd52luq | — |
| `_mm256_mask_multishift_epi64_epi8` | For each 64-bit element in b, select 8 unaligned bytes using... | vpmultishiftqb | — |
| `_mm256_mask_permutex2var_epi8` | Shuffle 8-bit integers in a and b across lanes using the cor... | vpermt2b | 3/1, 2/1 |
| `_mm256_mask_permutexvar_epi8` | Shuffle 8-bit integers in a across lanes using the correspon... | vpermb | 3/1, 2/1 |
| `_mm256_mask_popcnt_epi16` | For each packed 16-bit integer maps the value to the number ... | vpopcntw | —/2/1 |
| `_mm256_mask_popcnt_epi32` | For each packed 32-bit integer maps the value to the number ... | vpopcntd | —/2/1 |
| `_mm256_mask_popcnt_epi64` | For each packed 64-bit integer maps the value to the number ... | vpopcntq | —/2/1 |
| `_mm256_mask_popcnt_epi8` | For each packed 8-bit integer maps the value to the number o... | vpopcntb | —/2/1 |
| `_mm256_mask_shldi_epi16` | Concatenate packed 16-bit integers in a and b producing an i... | vpshldw | — |
| `_mm256_mask_shldi_epi32` | Concatenate packed 32-bit integers in a and b producing an i... | vpshldd | — |
| `_mm256_mask_shldi_epi64` | Concatenate packed 64-bit integers in a and b producing an i... | vpshldq | — |
| `_mm256_mask_shldv_epi16` | Concatenate packed 16-bit integers in a and b producing an i... | vpshldvw | — |
| `_mm256_mask_shldv_epi32` | Concatenate packed 32-bit integers in a and b producing an i... | vpshldvd | — |
| `_mm256_mask_shldv_epi64` | Concatenate packed 64-bit integers in a and b producing an i... | vpshldvq | — |
| `_mm256_mask_shrdi_epi16` | Concatenate packed 16-bit integers in b and a producing an i... | vpshldw | — |
| `_mm256_mask_shrdi_epi32` | Concatenate packed 32-bit integers in b and a producing an i... | vpshldd | — |
| `_mm256_mask_shrdi_epi64` | Concatenate packed 64-bit integers in b and a producing an i... | vpshldq | — |
| `_mm256_mask_shrdv_epi16` | Concatenate packed 16-bit integers in b and a producing an i... | vpshrdvw | — |
| `_mm256_mask_shrdv_epi32` | Concatenate packed 32-bit integers in b and a producing an i... | vpshrdvd | — |
| `_mm256_mask_shrdv_epi64` | Concatenate packed 64-bit integers in b and a producing an i... | vpshrdvq | — |
| `_mm256_maskz_compress_epi16` | Contiguously store the active 16-bit integers in a (those wi... | vpcompressw | — |
| `_mm256_maskz_compress_epi8` | Contiguously store the active 8-bit integers in a (those wit... | vpcompressb | — |
| `_mm256_maskz_dpbusd_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusd | — |
| `_mm256_maskz_dpbusds_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusds | — |
| `_mm256_maskz_dpwssd_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssd | — |
| `_mm256_maskz_dpwssds_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssds | — |
| `_mm256_maskz_expand_epi16` | Load contiguous active 16-bit integers from a (those with th... | vpexpandw | — |
| `_mm256_maskz_expand_epi8` | Load contiguous active 8-bit integers from a (those with the... | vpexpandb | — |
| `_mm256_maskz_gf2p8affine_epi64_epi8` | Performs an affine transformation on the packed bytes in x. ... | vgf2p8affineqb | — |
| `_mm256_maskz_gf2p8affineinv_epi64_epi8` | Performs an affine transformation on the inverted packed byt... | vgf2p8affineinvqb | — |
| `_mm256_maskz_gf2p8mul_epi8` | Performs a multiplication in GF(2^8) on the packed bytes. Th... | vgf2p8mulb | — |
| `_mm256_maskz_madd52hi_epu64` | Multiply packed unsigned 52-bit integers in each 64-bit elem... | vpmadd52huq | — |
| `_mm256_maskz_madd52lo_epu64` | Multiply packed unsigned 52-bit integers in each 64-bit elem... | vpmadd52luq | — |
| `_mm256_maskz_multishift_epi64_epi8` | For each 64-bit element in b, select 8 unaligned bytes using... | vpmultishiftqb | — |
| `_mm256_maskz_permutex2var_epi8` | Shuffle 8-bit integers in a and b across lanes using the cor... | vperm | 3/1, 2/1 |
| `_mm256_maskz_permutexvar_epi8` | Shuffle 8-bit integers in a across lanes using the correspon... | vpermb | 3/1, 2/1 |
| `_mm256_maskz_popcnt_epi16` | For each packed 16-bit integer maps the value to the number ... | vpopcntw | —/2/1 |
| `_mm256_maskz_popcnt_epi32` | For each packed 32-bit integer maps the value to the number ... | vpopcntd | —/2/1 |
| `_mm256_maskz_popcnt_epi64` | For each packed 64-bit integer maps the value to the number ... | vpopcntq | —/2/1 |
| `_mm256_maskz_popcnt_epi8` | For each packed 8-bit integer maps the value to the number o... | vpopcntb | —/2/1 |
| `_mm256_maskz_shldi_epi16` | Concatenate packed 16-bit integers in a and b producing an i... | vpshldw | — |
| `_mm256_maskz_shldi_epi32` | Concatenate packed 32-bit integers in a and b producing an i... | vpshldd | — |
| `_mm256_maskz_shldi_epi64` | Concatenate packed 64-bit integers in a and b producing an i... | vpshldq | — |
| `_mm256_maskz_shldv_epi16` | Concatenate packed 16-bit integers in a and b producing an i... | vpshldvw | — |
| `_mm256_maskz_shldv_epi32` | Concatenate packed 32-bit integers in a and b producing an i... | vpshldvd | — |
| `_mm256_maskz_shldv_epi64` | Concatenate packed 64-bit integers in a and b producing an i... | vpshldvq | — |
| `_mm256_maskz_shrdi_epi16` | Concatenate packed 16-bit integers in b and a producing an i... | vpshldw | — |
| `_mm256_maskz_shrdi_epi32` | Concatenate packed 32-bit integers in b and a producing an i... | vpshldd | — |
| `_mm256_maskz_shrdi_epi64` | Concatenate packed 64-bit integers in b and a producing an i... | vpshldq | — |
| `_mm256_maskz_shrdv_epi16` | Concatenate packed 16-bit integers in b and a producing an i... | vpshrdvw | — |
| `_mm256_maskz_shrdv_epi32` | Concatenate packed 32-bit integers in b and a producing an i... | vpshrdvd | — |
| `_mm256_maskz_shrdv_epi64` | Concatenate packed 64-bit integers in b and a producing an i... | vpshrdvq | — |
| `_mm256_multishift_epi64_epi8` | For each 64-bit element in b, select 8 unaligned bytes using... | vpmultishiftqb | — |
| `_mm256_permutex2var_epi8` | Shuffle 8-bit integers in a and b across lanes using the cor... | vperm | 3/1, 2/1 |
| `_mm256_permutexvar_epi8` | Shuffle 8-bit integers in a across lanes using the correspon... | vpermb | 3/1, 2/1 |
| `_mm256_popcnt_epi16` | For each packed 16-bit integer maps the value to the number ... | vpopcntw | —/2/1 |
| `_mm256_popcnt_epi32` | For each packed 32-bit integer maps the value to the number ... | vpopcntd | —/2/1 |
| `_mm256_popcnt_epi64` | For each packed 64-bit integer maps the value to the number ... | vpopcntq | —/2/1 |
| `_mm256_popcnt_epi8` | For each packed 8-bit integer maps the value to the number o... | vpopcntb | —/2/1 |
| `_mm256_shldi_epi16` | Concatenate packed 16-bit integers in a and b producing an i... | vpshldw | — |
| `_mm256_shldi_epi32` | Concatenate packed 32-bit integers in a and b producing an i... | vpshldd | — |
| `_mm256_shldi_epi64` | Concatenate packed 64-bit integers in a and b producing an i... | vpshldq | — |
| `_mm256_shldv_epi16` | Concatenate packed 16-bit integers in a and b producing an i... | vpshldvw | — |
| `_mm256_shldv_epi32` | Concatenate packed 32-bit integers in a and b producing an i... | vpshldvd | — |
| `_mm256_shldv_epi64` | Concatenate packed 64-bit integers in a and b producing an i... | vpshldvq | — |
| `_mm256_shrdi_epi16` | Concatenate packed 16-bit integers in b and a producing an i... | vpshldw | — |
| `_mm256_shrdi_epi32` | Concatenate packed 32-bit integers in b and a producing an i... | vpshldd | — |
| `_mm256_shrdi_epi64` | Concatenate packed 64-bit integers in b and a producing an i... | vpshldq | — |
| `_mm256_shrdv_epi16` | Concatenate packed 16-bit integers in b and a producing an i... | vpshrdvw | — |
| `_mm256_shrdv_epi32` | Concatenate packed 32-bit integers in b and a producing an i... | vpshrdvd | — |
| `_mm256_shrdv_epi64` | Concatenate packed 64-bit integers in b and a producing an i... | vpshrdvq | — |
| `_mm512_aesdec_epi128` | Performs one round of an AES decryption flow on each 128-bit... | vaesdec | 7/1, 4/1 |
| `_mm512_aesdeclast_epi128` | Performs the last round of an AES decryption flow on each 12... | vaesdeclast | 7/1, 4/1 |
| `_mm512_aesenc_epi128` | Performs one round of an AES encryption flow on each 128-bit... | vaesenc | 7/1, 4/1 |
| `_mm512_aesenclast_epi128` | Performs the last round of an AES encryption flow on each 12... | vaesenclast | 7/1, 4/1 |
| `_mm512_bitshuffle_epi64_mask` | Considers the input `b` as packed 64-bit integers and `c` as... | vpshufbitqmb | — |
| `_mm512_clmulepi64_epi128` | Performs a carry-less multiplication of two 64-bit polynomia... | vpclmul | 7/2, 4/1 |
| `_mm512_dpbusd_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusd | — |
| `_mm512_dpbusds_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusds | — |
| `_mm512_dpwssd_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssd | — |
| `_mm512_dpwssds_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssds | — |
| `_mm512_gf2p8affine_epi64_epi8` | Performs an affine transformation on the packed bytes in x. ... | vgf2p8affineqb | — |
| `_mm512_gf2p8affineinv_epi64_epi8` | Performs an affine transformation on the inverted packed byt... | vgf2p8affineinvqb | — |
| `_mm512_gf2p8mul_epi8` | Performs a multiplication in GF(2^8) on the packed bytes. Th... | vgf2p8mulb | — |
| `_mm512_madd52hi_epu64` | Multiply packed unsigned 52-bit integers in each 64-bit elem... | vpmadd52huq | — |
| `_mm512_madd52lo_epu64` | Multiply packed unsigned 52-bit integers in each 64-bit elem... | vpmadd52luq | — |
| `_mm512_mask2_permutex2var_epi8` | Shuffle 8-bit integers in a and b across lanes using the cor... | vpermi2b | —/2/1 |
| `_mm512_mask_bitshuffle_epi64_mask` | Considers the input `b` as packed 64-bit integers and `c` as... | vpshufbitqmb | — |
| `_mm512_mask_compress_epi16` | Contiguously store the active 16-bit integers in a (those wi... | vpcompressw | — |
| `_mm512_mask_compress_epi8` | Contiguously store the active 8-bit integers in a (those wit... | vpcompressb | — |
| `_mm512_mask_dpbusd_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusd | — |
| `_mm512_mask_dpbusds_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusds | — |
| `_mm512_mask_dpwssd_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssd | — |
| `_mm512_mask_dpwssds_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssds | — |
| `_mm512_mask_expand_epi16` | Load contiguous active 16-bit integers from a (those with th... | vpexpandw | — |
| `_mm512_mask_expand_epi8` | Load contiguous active 8-bit integers from a (those with the... | vpexpandb | — |
| `_mm512_mask_gf2p8affine_epi64_epi8` | Performs an affine transformation on the packed bytes in x. ... | vgf2p8affineqb | — |
| `_mm512_mask_gf2p8affineinv_epi64_epi8` | Performs an affine transformation on the inverted packed byt... | vgf2p8affineinvqb | — |
| `_mm512_mask_gf2p8mul_epi8` | Performs a multiplication in GF(2^8) on the packed bytes. Th... | vgf2p8mulb | — |
| `_mm512_mask_madd52hi_epu64` | Multiply packed unsigned 52-bit integers in each 64-bit elem... | vpmadd52huq | — |
| `_mm512_mask_madd52lo_epu64` | Multiply packed unsigned 52-bit integers in each 64-bit elem... | vpmadd52luq | — |
| `_mm512_mask_multishift_epi64_epi8` | For each 64-bit element in b, select 8 unaligned bytes using... | vpmultishiftqb | — |
| `_mm512_mask_permutex2var_epi8` | Shuffle 8-bit integers in a and b across lanes using the cor... | vpermt2b | —/2/1 |
| `_mm512_mask_permutexvar_epi8` | Shuffle 8-bit integers in a across lanes using the correspon... | vpermb | —/2/1 |
| `_mm512_mask_popcnt_epi16` | For each packed 16-bit integer maps the value to the number ... | vpopcntw | —/2/1 |
| `_mm512_mask_popcnt_epi32` | For each packed 32-bit integer maps the value to the number ... | vpopcntd | —/2/1 |
| `_mm512_mask_popcnt_epi64` | For each packed 64-bit integer maps the value to the number ... | vpopcntq | —/2/1 |
| `_mm512_mask_popcnt_epi8` | For each packed 8-bit integer maps the value to the number o... | vpopcntb | —/2/1 |
| `_mm512_mask_shldi_epi16` | Concatenate packed 16-bit integers in a and b producing an i... | vpshldw | — |
| `_mm512_mask_shldi_epi32` | Concatenate packed 32-bit integers in a and b producing an i... | vpshldd | — |
| `_mm512_mask_shldi_epi64` | Concatenate packed 64-bit integers in a and b producing an i... | vpshldq | — |
| `_mm512_mask_shldv_epi16` | Concatenate packed 16-bit integers in a and b producing an i... | vpshldvw | — |
| `_mm512_mask_shldv_epi32` | Concatenate packed 32-bit integers in a and b producing an i... | vpshldvd | — |
| `_mm512_mask_shldv_epi64` | Concatenate packed 64-bit integers in a and b producing an i... | vpshldvq | — |
| `_mm512_mask_shrdi_epi16` | Concatenate packed 16-bit integers in b and a producing an i... | vpshldw | — |
| `_mm512_mask_shrdi_epi32` | Concatenate packed 32-bit integers in b and a producing an i... | vpshldd | — |
| `_mm512_mask_shrdi_epi64` | Concatenate packed 64-bit integers in b and a producing an i... | vpshldq | — |
| `_mm512_mask_shrdv_epi16` | Concatenate packed 16-bit integers in b and a producing an i... | vpshrdvw | — |
| `_mm512_mask_shrdv_epi32` | Concatenate packed 32-bit integers in b and a producing an i... | vpshrdvd | — |
| `_mm512_mask_shrdv_epi64` | Concatenate packed 64-bit integers in b and a producing an i... | vpshrdvq | — |
| `_mm512_maskz_compress_epi16` | Contiguously store the active 16-bit integers in a (those wi... | vpcompressw | — |
| `_mm512_maskz_compress_epi8` | Contiguously store the active 8-bit integers in a (those wit... | vpcompressb | — |
| `_mm512_maskz_dpbusd_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusd | — |
| `_mm512_maskz_dpbusds_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusds | — |
| `_mm512_maskz_dpwssd_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssd | — |
| `_mm512_maskz_dpwssds_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssds | — |
| `_mm512_maskz_expand_epi16` | Load contiguous active 16-bit integers from a (those with th... | vpexpandw | — |
| `_mm512_maskz_expand_epi8` | Load contiguous active 8-bit integers from a (those with the... | vpexpandb | — |
| `_mm512_maskz_gf2p8affine_epi64_epi8` | Performs an affine transformation on the packed bytes in x. ... | vgf2p8affineqb | — |
| `_mm512_maskz_gf2p8affineinv_epi64_epi8` | Performs an affine transformation on the inverted packed byt... | vgf2p8affineinvqb | — |
| `_mm512_maskz_gf2p8mul_epi8` | Performs a multiplication in GF(2^8) on the packed bytes. Th... | vgf2p8mulb | — |
| `_mm512_maskz_madd52hi_epu64` | Multiply packed unsigned 52-bit integers in each 64-bit elem... | vpmadd52huq | — |
| `_mm512_maskz_madd52lo_epu64` | Multiply packed unsigned 52-bit integers in each 64-bit elem... | vpmadd52luq | — |
| `_mm512_maskz_multishift_epi64_epi8` | For each 64-bit element in b, select 8 unaligned bytes using... | vpmultishiftqb | — |
| `_mm512_maskz_permutex2var_epi8` | Shuffle 8-bit integers in a and b across lanes using the cor... | vperm | —/2/1 |
| `_mm512_maskz_permutexvar_epi8` | Shuffle 8-bit integers in a across lanes using the correspon... | vpermb | —/2/1 |
| `_mm512_maskz_popcnt_epi16` | For each packed 16-bit integer maps the value to the number ... | vpopcntw | —/2/1 |
| `_mm512_maskz_popcnt_epi32` | For each packed 32-bit integer maps the value to the number ... | vpopcntd | —/2/1 |
| `_mm512_maskz_popcnt_epi64` | For each packed 64-bit integer maps the value to the number ... | vpopcntq | —/2/1 |
| `_mm512_maskz_popcnt_epi8` | For each packed 8-bit integer maps the value to the number o... | vpopcntb | —/2/1 |
| `_mm512_maskz_shldi_epi16` | Concatenate packed 16-bit integers in a and b producing an i... | vpshldw | — |
| `_mm512_maskz_shldi_epi32` | Concatenate packed 32-bit integers in a and b producing an i... | vpshldd | — |
| `_mm512_maskz_shldi_epi64` | Concatenate packed 64-bit integers in a and b producing an i... | vpshldq | — |
| `_mm512_maskz_shldv_epi16` | Concatenate packed 16-bit integers in a and b producing an i... | vpshldvw | — |
| `_mm512_maskz_shldv_epi32` | Concatenate packed 32-bit integers in a and b producing an i... | vpshldvd | — |
| `_mm512_maskz_shldv_epi64` | Concatenate packed 64-bit integers in a and b producing an i... | vpshldvq | — |
| `_mm512_maskz_shrdi_epi16` | Concatenate packed 16-bit integers in b and a producing an i... | vpshldw | — |
| `_mm512_maskz_shrdi_epi32` | Concatenate packed 32-bit integers in b and a producing an i... | vpshldd | — |
| `_mm512_maskz_shrdi_epi64` | Concatenate packed 64-bit integers in b and a producing an i... | vpshldq | — |
| `_mm512_maskz_shrdv_epi16` | Concatenate packed 16-bit integers in b and a producing an i... | vpshrdvw | — |
| `_mm512_maskz_shrdv_epi32` | Concatenate packed 32-bit integers in b and a producing an i... | vpshrdvd | — |
| `_mm512_maskz_shrdv_epi64` | Concatenate packed 64-bit integers in b and a producing an i... | vpshrdvq | — |
| `_mm512_multishift_epi64_epi8` | For each 64-bit element in b, select 8 unaligned bytes using... | vpmultishiftqb | — |
| `_mm512_permutex2var_epi8` | Shuffle 8-bit integers in a and b across lanes using the cor... | vperm | —/2/1 |
| `_mm512_permutexvar_epi8` | Shuffle 8-bit integers in a across lanes using the correspon... | vpermb | —/2/1 |
| `_mm512_popcnt_epi16` | For each packed 16-bit integer maps the value to the number ... | vpopcntw | —/2/1 |
| `_mm512_popcnt_epi32` | For each packed 32-bit integer maps the value to the number ... | vpopcntd | —/2/1 |
| `_mm512_popcnt_epi64` | For each packed 64-bit integer maps the value to the number ... | vpopcntq | —/2/1 |
| `_mm512_popcnt_epi8` | For each packed 8-bit integer maps the value to the number o... | vpopcntb | —/2/1 |
| `_mm512_shldi_epi16` | Concatenate packed 16-bit integers in a and b producing an i... | vpshldw | — |
| `_mm512_shldi_epi32` | Concatenate packed 32-bit integers in a and b producing an i... | vpshldd | — |
| `_mm512_shldi_epi64` | Concatenate packed 64-bit integers in a and b producing an i... | vpshldq | — |
| `_mm512_shldv_epi16` | Concatenate packed 16-bit integers in a and b producing an i... | vpshldvw | — |
| `_mm512_shldv_epi32` | Concatenate packed 32-bit integers in a and b producing an i... | vpshldvd | — |
| `_mm512_shldv_epi64` | Concatenate packed 64-bit integers in a and b producing an i... | vpshldvq | — |
| `_mm512_shrdi_epi16` | Concatenate packed 16-bit integers in b and a producing an i... | vpshldw | — |
| `_mm512_shrdi_epi32` | Concatenate packed 32-bit integers in b and a producing an i... | vpshldd | — |
| `_mm512_shrdi_epi64` | Concatenate packed 64-bit integers in b and a producing an i... | vpshldq | — |
| `_mm512_shrdv_epi16` | Concatenate packed 16-bit integers in b and a producing an i... | vpshrdvw | — |
| `_mm512_shrdv_epi32` | Concatenate packed 32-bit integers in b and a producing an i... | vpshrdvd | — |
| `_mm512_shrdv_epi64` | Concatenate packed 64-bit integers in b and a producing an i... | vpshrdvq | — |
| `_mm_bitshuffle_epi64_mask` | Considers the input `b` as packed 64-bit integers and `c` as... | vpshufbitqmb | — |
| `_mm_dpbusd_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusd | — |
| `_mm_dpbusds_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusds | — |
| `_mm_dpwssd_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssd | — |
| `_mm_dpwssds_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssds | — |
| `_mm_gf2p8affine_epi64_epi8` | Performs an affine transformation on the packed bytes in x. ... | gf2p8affineqb | — |
| `_mm_gf2p8affineinv_epi64_epi8` | Performs an affine transformation on the inverted packed byt... | gf2p8affineinvqb | — |
| `_mm_gf2p8mul_epi8` | Performs a multiplication in GF(2^8) on the packed bytes. Th... | gf2p8mulb | — |
| `_mm_madd52hi_epu64` | Multiply packed unsigned 52-bit integers in each 64-bit elem... | vpmadd52huq | — |
| `_mm_madd52lo_epu64` | Multiply packed unsigned 52-bit integers in each 64-bit elem... | vpmadd52luq | — |
| `_mm_mask2_permutex2var_epi8` | Shuffle 8-bit integers in a and b across lanes using the cor... | vpermi2b | 1/1, 1/1 |
| `_mm_mask_bitshuffle_epi64_mask` | Considers the input `b` as packed 64-bit integers and `c` as... | vpshufbitqmb | — |
| `_mm_mask_compress_epi16` | Contiguously store the active 16-bit integers in a (those wi... | vpcompressw | — |
| `_mm_mask_compress_epi8` | Contiguously store the active 8-bit integers in a (those wit... | vpcompressb | — |
| `_mm_mask_dpbusd_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusd | — |
| `_mm_mask_dpbusds_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusds | — |
| `_mm_mask_dpwssd_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssd | — |
| `_mm_mask_dpwssds_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssds | — |
| `_mm_mask_expand_epi16` | Load contiguous active 16-bit integers from a (those with th... | vpexpandw | — |
| `_mm_mask_expand_epi8` | Load contiguous active 8-bit integers from a (those with the... | vpexpandb | — |
| `_mm_mask_gf2p8affine_epi64_epi8` | Performs an affine transformation on the packed bytes in x. ... | vgf2p8affineqb | — |
| `_mm_mask_gf2p8affineinv_epi64_epi8` | Performs an affine transformation on the inverted packed byt... | vgf2p8affineinvqb | — |
| `_mm_mask_gf2p8mul_epi8` | Performs a multiplication in GF(2^8) on the packed bytes. Th... | vgf2p8mulb | — |
| `_mm_mask_madd52hi_epu64` | Multiply packed unsigned 52-bit integers in each 64-bit elem... | vpmadd52huq | — |
| `_mm_mask_madd52lo_epu64` | Multiply packed unsigned 52-bit integers in each 64-bit elem... | vpmadd52luq | — |
| `_mm_mask_multishift_epi64_epi8` | For each 64-bit element in b, select 8 unaligned bytes using... | vpmultishiftqb | — |
| `_mm_mask_permutex2var_epi8` | Shuffle 8-bit integers in a and b across lanes using the cor... | vpermt2b | 1/1, 1/1 |
| `_mm_mask_permutexvar_epi8` | Shuffle 8-bit integers in a across lanes using the correspon... | vpermb | 1/1, 1/1 |
| `_mm_mask_popcnt_epi16` | For each packed 16-bit integer maps the value to the number ... | vpopcntw | —/2/1 |
| `_mm_mask_popcnt_epi32` | For each packed 32-bit integer maps the value to the number ... | vpopcntd | —/2/1 |
| `_mm_mask_popcnt_epi64` | For each packed 64-bit integer maps the value to the number ... | vpopcntq | —/2/1 |
| `_mm_mask_popcnt_epi8` | For each packed 8-bit integer maps the value to the number o... | vpopcntb | —/2/1 |
| `_mm_mask_shldi_epi16` | Concatenate packed 16-bit integers in a and b producing an i... | vpshldw | — |
| `_mm_mask_shldi_epi32` | Concatenate packed 32-bit integers in a and b producing an i... | vpshldd | — |
| `_mm_mask_shldi_epi64` | Concatenate packed 64-bit integers in a and b producing an i... | vpshldq | — |
| `_mm_mask_shldv_epi16` | Concatenate packed 16-bit integers in a and b producing an i... | vpshldvw | — |
| `_mm_mask_shldv_epi32` | Concatenate packed 32-bit integers in a and b producing an i... | vpshldvd | — |
| `_mm_mask_shldv_epi64` | Concatenate packed 64-bit integers in a and b producing an i... | vpshldvq | — |
| `_mm_mask_shrdi_epi16` | Concatenate packed 16-bit integers in b and a producing an i... | vpshldw | — |
| `_mm_mask_shrdi_epi32` | Concatenate packed 32-bit integers in b and a producing an i... | vpshldd | — |
| `_mm_mask_shrdi_epi64` | Concatenate packed 64-bit integers in b and a producing an i... | vpshldq | — |
| `_mm_mask_shrdv_epi16` | Concatenate packed 16-bit integers in b and a producing an i... | vpshrdvw | — |
| `_mm_mask_shrdv_epi32` | Concatenate packed 32-bit integers in b and a producing an i... | vpshrdvd | — |
| `_mm_mask_shrdv_epi64` | Concatenate packed 64-bit integers in b and a producing an i... | vpshrdvq | — |
| `_mm_maskz_compress_epi16` | Contiguously store the active 16-bit integers in a (those wi... | vpcompressw | — |
| `_mm_maskz_compress_epi8` | Contiguously store the active 8-bit integers in a (those wit... | vpcompressb | — |
| `_mm_maskz_dpbusd_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusd | — |
| `_mm_maskz_dpbusds_epi32` | Multiply groups of 4 adjacent pairs of unsigned 8-bit intege... | vpdpbusds | — |
| `_mm_maskz_dpwssd_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssd | — |
| `_mm_maskz_dpwssds_epi32` | Multiply groups of 2 adjacent pairs of signed 16-bit integer... | vpdpwssds | — |
| `_mm_maskz_expand_epi16` | Load contiguous active 16-bit integers from a (those with th... | vpexpandw | — |
| `_mm_maskz_expand_epi8` | Load contiguous active 8-bit integers from a (those with the... | vpexpandb | — |
| `_mm_maskz_gf2p8affine_epi64_epi8` | Performs an affine transformation on the packed bytes in x. ... | vgf2p8affineqb | — |
| `_mm_maskz_gf2p8affineinv_epi64_epi8` | Performs an affine transformation on the inverted packed byt... | vgf2p8affineinvqb | — |
| `_mm_maskz_gf2p8mul_epi8` | Performs a multiplication in GF(2^8) on the packed bytes. Th... | vgf2p8mulb | — |
| `_mm_maskz_madd52hi_epu64` | Multiply packed unsigned 52-bit integers in each 64-bit elem... | vpmadd52huq | — |
| `_mm_maskz_madd52lo_epu64` | Multiply packed unsigned 52-bit integers in each 64-bit elem... | vpmadd52luq | — |
| `_mm_maskz_multishift_epi64_epi8` | For each 64-bit element in b, select 8 unaligned bytes using... | vpmultishiftqb | — |
| `_mm_maskz_permutex2var_epi8` | Shuffle 8-bit integers in a and b across lanes using the cor... | vperm | 1/1, 1/1 |
| `_mm_maskz_permutexvar_epi8` | Shuffle 8-bit integers in a across lanes using the correspon... | vpermb | 1/1, 1/1 |
| `_mm_maskz_popcnt_epi16` | For each packed 16-bit integer maps the value to the number ... | vpopcntw | —/2/1 |
| `_mm_maskz_popcnt_epi32` | For each packed 32-bit integer maps the value to the number ... | vpopcntd | —/2/1 |
| `_mm_maskz_popcnt_epi64` | For each packed 64-bit integer maps the value to the number ... | vpopcntq | —/2/1 |
| `_mm_maskz_popcnt_epi8` | For each packed 8-bit integer maps the value to the number o... | vpopcntb | —/2/1 |
| `_mm_maskz_shldi_epi16` | Concatenate packed 16-bit integers in a and b producing an i... | vpshldw | — |
| `_mm_maskz_shldi_epi32` | Concatenate packed 32-bit integers in a and b producing an i... | vpshldd | — |
| `_mm_maskz_shldi_epi64` | Concatenate packed 64-bit integers in a and b producing an i... | vpshldq | — |
| `_mm_maskz_shldv_epi16` | Concatenate packed 16-bit integers in a and b producing an i... | vpshldvw | — |
| `_mm_maskz_shldv_epi32` | Concatenate packed 32-bit integers in a and b producing an i... | vpshldvd | — |
| `_mm_maskz_shldv_epi64` | Concatenate packed 64-bit integers in a and b producing an i... | vpshldvq | — |
| `_mm_maskz_shrdi_epi16` | Concatenate packed 16-bit integers in b and a producing an i... | vpshldw | — |
| `_mm_maskz_shrdi_epi32` | Concatenate packed 32-bit integers in b and a producing an i... | vpshldd | — |
| `_mm_maskz_shrdi_epi64` | Concatenate packed 64-bit integers in b and a producing an i... | vpshldq | — |
| `_mm_maskz_shrdv_epi16` | Concatenate packed 16-bit integers in b and a producing an i... | vpshrdvw | — |
| `_mm_maskz_shrdv_epi32` | Concatenate packed 32-bit integers in b and a producing an i... | vpshrdvd | — |
| `_mm_maskz_shrdv_epi64` | Concatenate packed 64-bit integers in b and a producing an i... | vpshrdvq | — |
| `_mm_multishift_epi64_epi8` | For each 64-bit element in b, select 8 unaligned bytes using... | vpmultishiftqb | — |
| `_mm_permutex2var_epi8` | Shuffle 8-bit integers in a and b across lanes using the cor... | vperm | 1/1, 1/1 |
| `_mm_permutexvar_epi8` | Shuffle 8-bit integers in a across lanes using the correspon... | vpermb | 1/1, 1/1 |
| `_mm_popcnt_epi16` | For each packed 16-bit integer maps the value to the number ... | vpopcntw | —/2/1 |
| `_mm_popcnt_epi32` | For each packed 32-bit integer maps the value to the number ... | vpopcntd | —/2/1 |
| `_mm_popcnt_epi64` | For each packed 64-bit integer maps the value to the number ... | vpopcntq | —/2/1 |
| `_mm_popcnt_epi8` | For each packed 8-bit integer maps the value to the number o... | vpopcntb | —/2/1 |
| `_mm_shldi_epi16` | Concatenate packed 16-bit integers in a and b producing an i... | vpshldw | — |
| `_mm_shldi_epi32` | Concatenate packed 32-bit integers in a and b producing an i... | vpshldd | — |
| `_mm_shldi_epi64` | Concatenate packed 64-bit integers in a and b producing an i... | vpshldq | — |
| `_mm_shldv_epi16` | Concatenate packed 16-bit integers in a and b producing an i... | vpshldvw | — |
| `_mm_shldv_epi32` | Concatenate packed 32-bit integers in a and b producing an i... | vpshldvd | — |
| `_mm_shldv_epi64` | Concatenate packed 64-bit integers in a and b producing an i... | vpshldvq | — |
| `_mm_shrdi_epi16` | Concatenate packed 16-bit integers in b and a producing an i... | vpshldw | — |
| `_mm_shrdi_epi32` | Concatenate packed 32-bit integers in b and a producing an i... | vpshldd | — |
| `_mm_shrdi_epi64` | Concatenate packed 64-bit integers in b and a producing an i... | vpshldq | — |
| `_mm_shrdv_epi16` | Concatenate packed 16-bit integers in b and a producing an i... | vpshrdvw | — |
| `_mm_shrdv_epi32` | Concatenate packed 32-bit integers in b and a producing an i... | vpshrdvd | — |
| `_mm_shrdv_epi64` | Concatenate packed 64-bit integers in b and a producing an i... | vpshrdvq | — |

### Stable, Unsafe (18 intrinsics) — use safe_unaligned_simd

| Name | Description | Safe Variant |
|------|-------------|--------------|
| `_mm256_mask_compressstoreu_epi16` | Contiguously store the active 16-bit integers in a (those wi... | — |
| `_mm256_mask_compressstoreu_epi8` | Contiguously store the active 8-bit integers in a (those wit... | — |
| `_mm256_mask_expandloadu_epi16` | Load contiguous active 16-bit integers from unaligned memory... | — |
| `_mm256_mask_expandloadu_epi8` | Load contiguous active 8-bit integers from unaligned memory ... | — |
| `_mm256_maskz_expandloadu_epi16` | Load contiguous active 16-bit integers from unaligned memory... | — |
| `_mm256_maskz_expandloadu_epi8` | Load contiguous active 8-bit integers from unaligned memory ... | safe_unaligned_simd::`_mm256_maskz_expandloadu_epi8` |
| `_mm512_mask_compressstoreu_epi16` | Contiguously store the active 16-bit integers in a (those wi... | — |
| `_mm512_mask_compressstoreu_epi8` | Contiguously store the active 8-bit integers in a (those wit... | — |
| `_mm512_mask_expandloadu_epi16` | Load contiguous active 16-bit integers from unaligned memory... | — |
| `_mm512_mask_expandloadu_epi8` | Load contiguous active 8-bit integers from unaligned memory ... | — |
| `_mm512_maskz_expandloadu_epi16` | Load contiguous active 16-bit integers from unaligned memory... | — |
| `_mm512_maskz_expandloadu_epi8` | Load contiguous active 8-bit integers from unaligned memory ... | safe_unaligned_simd::`_mm512_maskz_expandloadu_epi8` |
| `_mm_mask_compressstoreu_epi16` | Contiguously store the active 16-bit integers in a (those wi... | — |
| `_mm_mask_compressstoreu_epi8` | Contiguously store the active 8-bit integers in a (those wit... | — |
| `_mm_mask_expandloadu_epi16` | Load contiguous active 16-bit integers from unaligned memory... | — |
| `_mm_mask_expandloadu_epi8` | Load contiguous active 8-bit integers from unaligned memory ... | — |
| `_mm_maskz_expandloadu_epi16` | Load contiguous active 16-bit integers from unaligned memory... | safe_unaligned_simd::`_mm_maskz_expandloadu_epi16` |
| `_mm_maskz_expandloadu_epi8` | Load contiguous active 8-bit integers from unaligned memory ... | safe_unaligned_simd::`_mm_maskz_expandloadu_epi8` |



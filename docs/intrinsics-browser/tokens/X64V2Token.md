# X64V2Token — x86-64-v2

Proof that SSE4.2 + POPCNT are available (x86-64-v2 level).

**Architecture:** x86_64 | **Features:** sse, sse2, sse3, ssse3, sse4.1, sse4.2, popcnt, cmpxchg16b
**Total intrinsics:** 110 (106 safe, 4 unsafe, 110 stable, 0 unstable/unknown)

## Usage

```rust
use archmage::{X64V2Token, SimdToken};

if let Some(token) = X64V2Token::summon() {
    process(token, &mut data);
}

#[arcane(import_intrinsics)]  // Entry point only
fn process(token: X64V2Token, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(4) {
        process_chunk(token, chunk.try_into().unwrap());
    }
}

#[rite(import_intrinsics)]  // All inner helpers
fn process_chunk(_: X64V2Token, chunk: &mut [f32; 4]) {
    let v = _mm_loadu_ps(chunk);  // safe!
    let doubled = _mm_add_ps(v, v);  // value intrinsic (safe inside #[rite])
    _mm_storeu_ps(chunk, doubled);  // safe!
}
// No unsafe anywhere. Use #![forbid(unsafe_code)] in your crate.
```



## All Intrinsics

### Stable, Safe (106 intrinsics)

| Name | Description | Instruction | Timing (H/Z4) |
|------|-------------|-------------|---------------|
| `_mm_abs_epi16` | Computes the absolute value of each of the packed 16-bit sig... | pabsw | — |
| `_mm_abs_epi32` | Computes the absolute value of each of the packed 32-bit sig... | pabsd | — |
| `_mm_abs_epi8` | Computes the absolute value of packed 8-bit signed integers ... | pabsb | — |
| `_mm_addsub_pd` | Alternatively add and subtract packed double-precision (64-b... | addsubpd | — |
| `_mm_addsub_ps` | Alternatively add and subtract packed single-precision (32-b... | addsubps | — |
| `_mm_alignr_epi8` | Concatenate 16-byte blocks in `a` and `b` into a 32-byte tem... | palignr | — |
| `_mm_blend_epi16` | Blend packed 16-bit integers from `a` and `b` using the mask... | pblendw | 1/1, 1/1 |
| `_mm_blend_pd` | Blend packed double-precision (64-bit) floating-point elemen... | blendpd | 1/1, 1/1 |
| `_mm_blend_ps` | Blend packed single-precision (32-bit) floating-point elemen... | blendps | 1/1, 1/1 |
| `_mm_blendv_epi8` | Blend packed 8-bit integers from `a` and `b` using `mask`  T... | pblendvb | 1/1, 1/1 |
| `_mm_blendv_pd` | Blend packed double-precision (64-bit) floating-point elemen... | blendvpd | 1/1, 1/1 |
| `_mm_blendv_ps` | Blend packed single-precision (32-bit) floating-point elemen... | blendvps | 1/1, 1/1 |
| `_mm_ceil_pd` | Round the packed double-precision (64-bit) floating-point el... | roundpd | — |
| `_mm_ceil_ps` | Round the packed single-precision (32-bit) floating-point el... | roundps | — |
| `_mm_ceil_sd` | Round the lower double-precision (64-bit) floating-point ele... | roundsd | — |
| `_mm_ceil_ss` | Round the lower single-precision (32-bit) floating-point ele... | roundss | — |
| `_mm_cmpeq_epi64` | Compares packed 64-bit integers in `a` and `b` for equality | pcmpeqq | 1/1, 1/1 |
| `_mm_cmpestra` | Compares packed strings in `a` and `b` with lengths `la` and... | pcmpestri | — |
| `_mm_cmpestrc` | Compares packed strings in `a` and `b` with lengths `la` and... | pcmpestri | — |
| `_mm_cmpestri` | Compares packed strings `a` and `b` with lengths `la` and `l... | pcmpestri | — |
| `_mm_cmpestrm` | Compares packed strings in `a` and `b` with lengths `la` and... | pcmpestrm | — |
| `_mm_cmpestro` | Compares packed strings in `a` and `b` with lengths `la` and... | pcmpestri | — |
| `_mm_cmpestrs` | Compares packed strings in `a` and `b` with lengths `la` and... | pcmpestri | — |
| `_mm_cmpestrz` | Compares packed strings in `a` and `b` with lengths `la` and... | pcmpestri | — |
| `_mm_cmpgt_epi64` | Compares packed 64-bit integers in `a` and `b` for greater-t... | pcmpgtq | 1/1, 1/1 |
| `_mm_cmpistra` | Compares packed strings with implicit lengths in `a` and `b`... | pcmpistri | — |
| `_mm_cmpistrc` | Compares packed strings with implicit lengths in `a` and `b`... | pcmpistri | — |
| `_mm_cmpistri` | Compares packed strings with implicit lengths in `a` and `b`... | pcmpistri | — |
| `_mm_cmpistrm` | Compares packed strings with implicit lengths in `a` and `b`... | pcmpistrm | — |
| `_mm_cmpistro` | Compares packed strings with implicit lengths in `a` and `b`... | pcmpistri | — |
| `_mm_cmpistrs` | Compares packed strings with implicit lengths in `a` and `b`... | pcmpistri | — |
| `_mm_cmpistrz` | Compares packed strings with implicit lengths in `a` and `b`... | pcmpistri | — |
| `_mm_crc32_u16` | Starting with the initial value in `crc`, return the accumul... | crc32 | — |
| `_mm_crc32_u32` | Starting with the initial value in `crc`, return the accumul... | crc32 | — |
| `_mm_crc32_u64` | Starting with the initial value in `crc`, return the accumul... | crc32 | — |
| `_mm_crc32_u8` | Starting with the initial value in `crc`, return the accumul... | crc32 | — |
| `_mm_cvtepi16_epi32` | Sign extend packed 16-bit integers in `a` to packed 32-bit i... | pmovsxwd | 4/1, 3/1 |
| `_mm_cvtepi16_epi64` | Sign extend packed 16-bit integers in `a` to packed 64-bit i... | pmovsxwq | 4/1, 3/1 |
| `_mm_cvtepi32_epi64` | Sign extend packed 32-bit integers in `a` to packed 64-bit i... | pmovsxdq | 4/1, 3/1 |
| `_mm_cvtepi8_epi16` | Sign extend packed 8-bit integers in `a` to packed 16-bit in... | pmovsxbw | 4/1, 3/1 |
| `_mm_cvtepi8_epi32` | Sign extend packed 8-bit integers in `a` to packed 32-bit in... | pmovsxbd | 4/1, 3/1 |
| `_mm_cvtepi8_epi64` | Sign extend packed 8-bit integers in the low 8 bytes of `a` ... | pmovsxbq | 4/1, 3/1 |
| `_mm_cvtepu16_epi32` | Zeroes extend packed unsigned 16-bit integers in `a` to pack... | pmovzxwd | 4/1, 3/1 |
| `_mm_cvtepu16_epi64` | Zeroes extend packed unsigned 16-bit integers in `a` to pack... | pmovzxwq | 4/1, 3/1 |
| `_mm_cvtepu32_epi64` | Zeroes extend packed unsigned 32-bit integers in `a` to pack... | pmovzxdq | 4/1, 3/1 |
| `_mm_cvtepu8_epi16` | Zeroes extend packed unsigned 8-bit integers in `a` to packe... | pmovzxbw | 4/1, 3/1 |
| `_mm_cvtepu8_epi32` | Zeroes extend packed unsigned 8-bit integers in `a` to packe... | pmovzxbd | 4/1, 3/1 |
| `_mm_cvtepu8_epi64` | Zeroes extend packed unsigned 8-bit integers in `a` to packe... | pmovzxbq | 4/1, 3/1 |
| `_mm_dp_pd` | Returns the dot product of two __m128d vectors.  `IMM8 | dppd | — |
| `_mm_dp_ps` | Returns the dot product of two __m128 vectors.  `IMM8 | dpps | — |
| `_mm_extract_epi32` | Extracts an 32-bit integer from `a` selected with `IMM8` | extractps | — |
| `_mm_extract_epi64` | Extracts an 64-bit integer from `a` selected with `IMM1` | pextrq | — |
| `_mm_extract_epi8` | Extracts an 8-bit integer from `a`, selected with `IMM8`. Re... | pextrb | — |
| `_mm_extract_ps` | Extracts a single-precision (32-bit) floating-point element ... | extractps | — |
| `_mm_floor_pd` | Round the packed double-precision (64-bit) floating-point el... | roundpd | — |
| `_mm_floor_ps` | Round the packed single-precision (32-bit) floating-point el... | roundps | — |
| `_mm_floor_sd` | Round the lower double-precision (64-bit) floating-point ele... | roundsd | — |
| `_mm_floor_ss` | Round the lower single-precision (32-bit) floating-point ele... | roundss | — |
| `_mm_hadd_epi16` | Horizontally adds the adjacent pairs of values contained in ... | phaddw | 5/2, 4/2 |
| `_mm_hadd_epi32` | Horizontally adds the adjacent pairs of values contained in ... | phaddd | 5/2, 4/2 |
| `_mm_hadd_pd` | Horizontally adds adjacent pairs of double-precision (64-bit... | haddpd | 5/2, 4/2 |
| `_mm_hadd_ps` | Horizontally adds adjacent pairs of single-precision (32-bit... | haddps | 5/2, 4/2 |
| `_mm_hadds_epi16` | Horizontally adds the adjacent pairs of values contained in ... | phaddsw | — |
| `_mm_hsub_epi16` | Horizontally subtract the adjacent pairs of values contained... | phsubw | 5/2, 4/2 |
| `_mm_hsub_epi32` | Horizontally subtract the adjacent pairs of values contained... | phsubd | 5/2, 4/2 |
| `_mm_hsub_pd` | Horizontally subtract adjacent pairs of double-precision (64... | hsubpd | 5/2, 4/2 |
| `_mm_hsub_ps` | Horizontally adds adjacent pairs of single-precision (32-bit... | hsubps | 5/2, 4/2 |
| `_mm_hsubs_epi16` | Horizontally subtract the adjacent pairs of values contained... | phsubsw | — |
| `_mm_insert_epi32` | Returns a copy of `a` with the 32-bit integer from `i` inser... | pinsrd | — |
| `_mm_insert_epi64` | Returns a copy of `a` with the 64-bit integer from `i` inser... | pinsrq | — |
| `_mm_insert_epi8` | Returns a copy of `a` with the 8-bit integer from `i` insert... | pinsrb | — |
| `_mm_insert_ps` | Select a single value in `b` to store at some position in `a... | insertps | — |
| `_mm_maddubs_epi16` | Multiplies corresponding pairs of packed 8-bit unsigned inte... | pmaddubsw | — |
| `_mm_max_epi32` | Compares packed 32-bit integers in `a` and `b`, and returns ... | pmaxsd | — |
| `_mm_max_epi8` | Compares packed 8-bit integers in `a` and `b` and returns pa... | pmaxsb | — |
| `_mm_max_epu16` | Compares packed unsigned 16-bit integers in `a` and `b`, and... | pmaxuw | — |
| `_mm_max_epu32` | Compares packed unsigned 32-bit integers in `a` and `b`, and... | pmaxud | — |
| `_mm_min_epi32` | Compares packed 32-bit integers in `a` and `b`, and returns ... | pminsd | — |
| `_mm_min_epi8` | Compares packed 8-bit integers in `a` and `b` and returns pa... | pminsb | — |
| `_mm_min_epu16` | Compares packed unsigned 16-bit integers in `a` and `b`, and... | pminuw | — |
| `_mm_min_epu32` | Compares packed unsigned 32-bit integers in `a` and `b`, and... | pminud | — |
| `_mm_minpos_epu16` | Finds the minimum unsigned 16-bit element in the 128-bit __m... | phminposuw | — |
| `_mm_movedup_pd` | Duplicate the low double-precision (64-bit) floating-point e... | movddup | — |
| `_mm_movehdup_ps` | Duplicate odd-indexed single-precision (32-bit) floating-poi... | movshdup | — |
| `_mm_moveldup_ps` | Duplicate even-indexed single-precision (32-bit) floating-po... | movsldup | — |
| `_mm_mpsadbw_epu8` | Subtracts 8-bit unsigned integer values and computes the abs... | mpsadbw | — |
| `_mm_mul_epi32` | Multiplies the low 32-bit integers from each packed 64-bit e... | pmuldq | 10/2, 4/1 |
| `_mm_mulhrs_epi16` | Multiplies packed 16-bit signed integer values, truncate the... | pmulhrsw | — |
| `_mm_mullo_epi32` | Multiplies the packed 32-bit integers in `a` and `b`, produc... | pmulld | 10/2, 4/1 |
| `_mm_packus_epi32` | Converts packed 32-bit integers from `a` and `b` to packed 1... | packusdw | 1/1, 1/1 |
| `_mm_round_pd` | Round the packed double-precision (64-bit) floating-point el... | roundpd | — |
| `_mm_round_ps` | Round the packed single-precision (32-bit) floating-point el... | roundps | — |
| `_mm_round_sd` | Round the lower double-precision (64-bit) floating-point ele... | roundsd | — |
| `_mm_round_ss` | Round the lower single-precision (32-bit) floating-point ele... | roundss | — |
| `_mm_shuffle_epi8` | Shuffles bytes from `a` according to the content of `b`.  Th... | pshufb | 1/1, 1/1 |
| `_mm_sign_epi16` | Negates packed 16-bit integers in `a` when the corresponding... | psignw | — |
| `_mm_sign_epi32` | Negates packed 32-bit integers in `a` when the corresponding... | psignd | — |
| `_mm_sign_epi8` | Negates packed 8-bit integers in `a` when the corresponding ... | psignb | — |
| `_mm_test_all_ones` | Tests whether the specified bits in `a` 128-bit integer vect... | pcmpeqd | — |
| `_mm_test_all_zeros` | Tests whether the specified bits in a 128-bit integer vector... | ptest | — |
| `_mm_test_mix_ones_zeros` | Tests whether the specified bits in a 128-bit integer vector... | ptest | — |
| `_mm_testc_si128` | Tests whether the specified bits in a 128-bit integer vector... | ptest | — |
| `_mm_testnzc_si128` | Tests whether the specified bits in a 128-bit integer vector... | ptest | — |
| `_mm_testz_si128` | Tests whether the specified bits in a 128-bit integer vector... | ptest | — |
| `_popcnt32` | Counts the bits that are set | popcnt | — |
| `_popcnt64` | Counts the bits that are set | popcnt | — |

### Stable, Unsafe (4 intrinsics) — use import_intrinsics for safe versions

| Name | Description | Safe Variant |
|------|-------------|--------------|
| `_mm_lddqu_si128` | Loads 128-bits of integer data from unaligned memory. This i... | — |
| `_mm_loaddup_pd` | Loads a double-precision (64-bit) floating-point element fro... | — |
| `_mm_stream_load_si128` | Load 128-bits of integer data from memory into dst. mem_addr... | — |
| `cmpxchg16b` | Compares and exchange 16 bytes (128 bits) of data atomically... | — |



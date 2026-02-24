# X64V3Token (Desktop64, Avx2FmaToken) — x86-64-v3

Proof that AVX2 + FMA + BMI1/2 + F16C + LZCNT are available (x86-64-v3 level).

**Architecture:** x86_64 | **Features:** sse, sse2, sse3, ssse3, sse4.1, sse4.2, popcnt, cmpxchg16b, avx, avx2, fma, bmi1, bmi2, f16c, lzcnt, movbe
**Total intrinsics:** 445 (374 safe, 71 unsafe, 445 stable, 0 unstable/unknown)

## Usage

```rust
use archmage::prelude::*;

if let Some(token) = Desktop64::summon() {
    process(token, &mut data);
}

#[arcane]  // Entry point only
fn process(token: Desktop64, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(8) {
        process_chunk(token, chunk.try_into().unwrap());
    }
}

#[rite]  // All inner helpers
fn process_chunk(_: Desktop64, chunk: &mut [f32; 8]) {
    let v = safe_unaligned_simd::x86_64::_mm256_loadu_ps(chunk);  // safe!
    let doubled = _mm256_add_ps(v, v);    // value intrinsic (safe inside #[rite])
    safe_unaligned_simd::x86_64::_mm256_storeu_ps(chunk, doubled);  // safe!
}
// No unsafe anywhere. Use #![forbid(unsafe_code)] in your crate.
```

## Safe Memory Operations (safe_unaligned_simd)

| Function | Safe Signature |
|----------|---------------|
| `_mm256_broadcast_pd` | `fn _mm256_broadcast_pd(mem_addr: &__m128d) -> __m256d` |
| `_mm256_broadcast_ps` | `fn _mm256_broadcast_ps(mem_addr: &__m128) -> __m256` |
| `_mm256_broadcast_sd` | `fn _mm256_broadcast_sd(mem_addr: &f64) -> __m256d` |
| `_mm256_broadcast_ss` | `fn _mm256_broadcast_ss(mem_addr: &f32) -> __m256` |
| `_mm256_loadu2_m128` | `fn _mm256_loadu2_m128(hiaddr: &[f32; 4], loaddr: &[f32; 4]) -> __m256` |
| `_mm256_loadu2_m128d` | `fn _mm256_loadu2_m128d(hiaddr: &[f64; 2], loaddr: &[f64; 2]) -> __m256d` |
| `_mm256_loadu2_m128i` | `fn _mm256_loadu2_m128i<T: Is128BitsUnaligned>(hiaddr: &T, loaddr: &T) -> __m256i` |
| `_mm256_loadu_pd` | `fn _mm256_loadu_pd(mem_addr: &[f64; 4]) -> __m256d` |
| `_mm256_loadu_ps` | `fn _mm256_loadu_ps(mem_addr: &[f32; 8]) -> __m256` |
| `_mm256_loadu_si256` | `fn _mm256_loadu_si256<T: Is256BitsUnaligned>(mem_addr: &T) -> __m256i` |
| `_mm256_storeu2_m128` | `fn _mm256_storeu2_m128(hiaddr: &mut [f32; 4], loaddr: &mut [f32; 4], a: __m256) -> ()` |
| `_mm256_storeu2_m128d` | `fn _mm256_storeu2_m128d(hiaddr: &mut [f64; 2], loaddr: &mut [f64; 2], a: __m256d) -> ()` |
| `_mm256_storeu2_m128i` | `fn _mm256_storeu2_m128i<T: Is128BitsUnaligned>(hiaddr: &mut T, loaddr: &mut T, a: __m256i) -> ()` |
| `_mm256_storeu_pd` | `fn _mm256_storeu_pd(mem_addr: &mut [f64; 4], a: __m256d) -> ()` |
| `_mm256_storeu_ps` | `fn _mm256_storeu_ps(mem_addr: &mut [f32; 8], a: __m256) -> ()` |
| `_mm256_storeu_si256` | `fn _mm256_storeu_si256<T: Is256BitsUnaligned>(mem_addr: &mut T, a: __m256i) -> ()` |
| `_mm_broadcast_ss` | `fn _mm_broadcast_ss(mem_addr: &f32) -> __m128` |


## All Intrinsics

### Stable, Safe (374 intrinsics)

| Name | Description | Instruction | Timing (H/Z4) |
|------|-------------|-------------|---------------|
| `_andn_u32` | Bitwise logical `AND` of inverted `a` with `b` | andn | — |
| `_andn_u64` | Bitwise logical `AND` of inverted `a` with `b` | andn | — |
| `_bextr2_u32` | Extracts bits of `a` specified by `control` into the least s... | bextr | — |
| `_bextr2_u64` | Extracts bits of `a` specified by `control` into the least s... | bextr | — |
| `_bextr_u32` | Extracts bits in range | bextr | — |
| `_bextr_u64` | Extracts bits in range | bextr | — |
| `_blsi_u32` | Extracts lowest set isolated bit | blsi | — |
| `_blsi_u64` | Extracts lowest set isolated bit | blsi | — |
| `_blsmsk_u32` | Gets mask up to lowest set bit | blsmsk | — |
| `_blsmsk_u64` | Gets mask up to lowest set bit | blsmsk | — |
| `_blsr_u32` | Resets the lowest set bit of `x`.  If `x` is sets CF | blsr | — |
| `_blsr_u64` | Resets the lowest set bit of `x`.  If `x` is sets CF | blsr | — |
| `_bzhi_u32` | Zeroes higher bits of `a` >= `index` | bzhi | — |
| `_bzhi_u64` | Zeroes higher bits of `a` >= `index` | bzhi | — |
| `_lzcnt_u32` | Counts the leading most significant zero bits.  When the ope... | lzcnt | — |
| `_lzcnt_u64` | Counts the leading most significant zero bits.  When the ope... | lzcnt | — |
| `_mm256_abs_epi16` | Computes the absolute values of packed 16-bit integers in `a... | vpabsw | — |
| `_mm256_abs_epi32` | Computes the absolute values of packed 32-bit integers in `a... | vpabsd | — |
| `_mm256_abs_epi8` | Computes the absolute values of packed 8-bit integers in `a` | vpabsb | — |
| `_mm256_add_epi16` | Adds packed 16-bit integers in `a` and `b` | vpaddw | 1/1, 1/1 |
| `_mm256_add_epi32` | Adds packed 32-bit integers in `a` and `b` | vpaddd | 1/1, 1/1 |
| `_mm256_add_epi64` | Adds packed 64-bit integers in `a` and `b` | vpaddq | 1/1, 1/1 |
| `_mm256_add_epi8` | Adds packed 8-bit integers in `a` and `b` | vpaddb | 1/1, 1/1 |
| `_mm256_add_pd` | Adds packed double-precision (64-bit) floating-point element... | vaddpd | 3/1, 3/1 |
| `_mm256_add_ps` | Adds packed single-precision (32-bit) floating-point element... | vaddps | 3/1, 3/1 |
| `_mm256_adds_epi16` | Adds packed 16-bit integers in `a` and `b` using saturation | vpaddsw | 1/1, 1/1 |
| `_mm256_adds_epi8` | Adds packed 8-bit integers in `a` and `b` using saturation | vpaddsb | 1/1, 1/1 |
| `_mm256_adds_epu16` | Adds packed unsigned 16-bit integers in `a` and `b` using sa... | vpaddusw | — |
| `_mm256_adds_epu8` | Adds packed unsigned 8-bit integers in `a` and `b` using sat... | vpaddusb | — |
| `_mm256_addsub_pd` | Alternatively adds and subtracts packed double-precision (64... | vaddsubpd | — |
| `_mm256_addsub_ps` | Alternatively adds and subtracts packed single-precision (32... | vaddsubps | — |
| `_mm256_alignr_epi8` | Concatenates pairs of 16-byte blocks in `a` and `b` into a 3... | vpalignr | — |
| `_mm256_and_pd` | Computes the bitwise AND of a packed double-precision (64-bi... | vandp | 1/1, 1/1 |
| `_mm256_and_ps` | Computes the bitwise AND of packed single-precision (32-bit)... | vandps | 1/1, 1/1 |
| `_mm256_and_si256` | Computes the bitwise AND of 256 bits (representing integer d... | vandps | 1/1, 1/1 |
| `_mm256_andnot_pd` | Computes the bitwise NOT of packed double-precision (64-bit)... | vandnp | 1/1, 1/1 |
| `_mm256_andnot_ps` | Computes the bitwise NOT of packed single-precision (32-bit)... | vandnps | 1/1, 1/1 |
| `_mm256_andnot_si256` | Computes the bitwise NOT of 256 bits (representing integer d... | vandnps | 1/1, 1/1 |
| `_mm256_avg_epu16` | Averages packed unsigned 16-bit integers in `a` and `b` | vpavgw | — |
| `_mm256_avg_epu8` | Averages packed unsigned 8-bit integers in `a` and `b` | vpavgb | — |
| `_mm256_blend_epi16` | Blends packed 16-bit integers from `a` and `b` using control... | vpblendw | 1/1, 1/1 |
| `_mm256_blend_epi32` | Blends packed 32-bit integers from `a` and `b` using control... | vblendps | 1/1, 1/1 |
| `_mm256_blend_pd` | Blends packed double-precision (64-bit) floating-point eleme... | vblendpd | 1/1, 1/1 |
| `_mm256_blend_ps` | Blends packed single-precision (32-bit) floating-point eleme... | vblendps | 1/1, 1/1 |
| `_mm256_blendv_epi8` | Blends packed 8-bit integers from `a` and `b` using `mask` | vpblendvb | 1/1, 1/1 |
| `_mm256_blendv_pd` | Blends packed double-precision (64-bit) floating-point eleme... | vblendvpd | 1/1, 1/1 |
| `_mm256_blendv_ps` | Blends packed single-precision (32-bit) floating-point eleme... | vblendvps | 1/1, 1/1 |
| `_mm256_broadcast_pd` | Broadcasts 128 bits from memory (composed of 2 packed double... | vbroadcastf128 | — |
| `_mm256_broadcast_ps` | Broadcasts 128 bits from memory (composed of 4 packed single... | vbroadcastf128 | — |
| `_mm256_broadcast_sd` | Broadcasts a double-precision (64-bit) floating-point elemen... | vbroadcastsd | — |
| `_mm256_broadcast_ss` | Broadcasts a single-precision (32-bit) floating-point elemen... | vbroadcastss | — |
| `_mm256_broadcastb_epi8` | Broadcasts the low packed 8-bit integer from `a` to all elem... | vpbroadcastb | — |
| `_mm256_broadcastd_epi32` | Broadcasts the low packed 32-bit integer from `a` to all ele... | vbroadcastss | — |
| `_mm256_broadcastq_epi64` | Broadcasts the low packed 64-bit integer from `a` to all ele... | vbroadcastsd | — |
| `_mm256_broadcastsd_pd` | Broadcasts the low double-precision (64-bit) floating-point ... | vbroadcastsd | — |
| `_mm256_broadcastsi128_si256` | Broadcasts 128 bits of integer data from a to all 128-bit la... |  | — |
| `_mm256_broadcastss_ps` | Broadcasts the low single-precision (32-bit) floating-point ... | vbroadcastss | — |
| `_mm256_broadcastw_epi16` | Broadcasts the low packed 16-bit integer from a to all eleme... | vpbroadcastw | — |
| `_mm256_bslli_epi128` | Shifts 128-bit lanes in `a` left by `imm8` bytes while shift... | vpslldq | — |
| `_mm256_bsrli_epi128` | Shifts 128-bit lanes in `a` right by `imm8` bytes while shif... | vpsrldq | — |
| `_mm256_castpd128_pd256` | Casts vector of type __m128d to type __m256d; the upper 128 ... |  | — |
| `_mm256_castpd256_pd128` | Casts vector of type __m256d to type __m128d |  | — |
| `_mm256_castpd_ps` | Cast vector of type __m256d to type __m256 |  | — |
| `_mm256_castpd_si256` | Casts vector of type __m256d to type __m256i |  | — |
| `_mm256_castps128_ps256` | Casts vector of type __m128 to type __m256; the upper 128 bi... |  | — |
| `_mm256_castps256_ps128` | Casts vector of type __m256 to type __m128 |  | — |
| `_mm256_castps_pd` | Cast vector of type __m256 to type __m256d |  | — |
| `_mm256_castps_si256` | Casts vector of type __m256 to type __m256i |  | — |
| `_mm256_castsi128_si256` | Casts vector of type __m128i to type __m256i; the upper 128 ... |  | — |
| `_mm256_castsi256_pd` | Casts vector of type __m256i to type __m256d |  | — |
| `_mm256_castsi256_ps` | Casts vector of type __m256i to type __m256 |  | — |
| `_mm256_castsi256_si128` | Casts vector of type __m256i to type __m128i |  | — |
| `_mm256_ceil_pd` | Rounds packed double-precision (64-bit) floating point eleme... | vroundpd | — |
| `_mm256_ceil_ps` | Rounds packed single-precision (32-bit) floating point eleme... | vroundps | — |
| `_mm256_cmp_pd` | Compares packed double-precision (64-bit) floating-point ele... | vcmpeqpd | 3/1, 3/1 |
| `_mm256_cmp_ps` | Compares packed single-precision (32-bit) floating-point ele... | vcmpeqps | 3/1, 3/1 |
| `_mm256_cmpeq_epi16` | Compares packed 16-bit integers in `a` and `b` for equality | vpcmpeqw | 1/1, 1/1 |
| `_mm256_cmpeq_epi32` | Compares packed 32-bit integers in `a` and `b` for equality | vpcmpeqd | 1/1, 1/1 |
| `_mm256_cmpeq_epi64` | Compares packed 64-bit integers in `a` and `b` for equality | vpcmpeqq | 1/1, 1/1 |
| `_mm256_cmpeq_epi8` | Compares packed 8-bit integers in `a` and `b` for equality | vpcmpeqb | 1/1, 1/1 |
| `_mm256_cmpgt_epi16` | Compares packed 16-bit integers in `a` and `b` for greater-t... | vpcmpgtw | 1/1, 1/1 |
| `_mm256_cmpgt_epi32` | Compares packed 32-bit integers in `a` and `b` for greater-t... | vpcmpgtd | 1/1, 1/1 |
| `_mm256_cmpgt_epi64` | Compares packed 64-bit integers in `a` and `b` for greater-t... | vpcmpgtq | 1/1, 1/1 |
| `_mm256_cmpgt_epi8` | Compares packed 8-bit integers in `a` and `b` for greater-th... | vpcmpgtb | 1/1, 1/1 |
| `_mm256_cvtepi16_epi32` | Sign-extend 16-bit integers to 32-bit integers | vpmovsxwd | 4/1, 3/1 |
| `_mm256_cvtepi16_epi64` | Sign-extend 16-bit integers to 64-bit integers | vpmovsxwq | 4/1, 3/1 |
| `_mm256_cvtepi32_epi64` | Sign-extend 32-bit integers to 64-bit integers | vpmovsxdq | 4/1, 3/1 |
| `_mm256_cvtepi32_pd` | Converts packed 32-bit integers in `a` to packed double-prec... | vcvtdq2pd | 4/1, 3/1 |
| `_mm256_cvtepi32_ps` | Converts packed 32-bit integers in `a` to packed single-prec... | vcvtdq2ps | 4/1, 3/1 |
| `_mm256_cvtepi8_epi16` | Sign-extend 8-bit integers to 16-bit integers | vpmovsxbw | 4/1, 3/1 |
| `_mm256_cvtepi8_epi32` | Sign-extend 8-bit integers to 32-bit integers | vpmovsxbd | 4/1, 3/1 |
| `_mm256_cvtepi8_epi64` | Sign-extend 8-bit integers to 64-bit integers | vpmovsxbq | 4/1, 3/1 |
| `_mm256_cvtepu16_epi32` | Zeroes extend packed unsigned 16-bit integers in `a` to pack... | vpmovzxwd | 4/1, 3/1 |
| `_mm256_cvtepu16_epi64` | Zero-extend the lower four unsigned 16-bit integers in `a` t... | vpmovzxwq | 4/1, 3/1 |
| `_mm256_cvtepu32_epi64` | Zero-extend unsigned 32-bit integers in `a` to 64-bit intege... | vpmovzxdq | 4/1, 3/1 |
| `_mm256_cvtepu8_epi16` | Zero-extend unsigned 8-bit integers in `a` to 16-bit integer... | vpmovzxbw | 4/1, 3/1 |
| `_mm256_cvtepu8_epi32` | Zero-extend the lower eight unsigned 8-bit integers in `a` t... | vpmovzxbd | 4/1, 3/1 |
| `_mm256_cvtepu8_epi64` | Zero-extend the lower four unsigned 8-bit integers in `a` to... | vpmovzxbq | 4/1, 3/1 |
| `_mm256_cvtpd_epi32` | Converts packed double-precision (64-bit) floating-point ele... | vcvtpd2dq | 4/1, 3/1 |
| `_mm256_cvtpd_ps` | Converts packed double-precision (64-bit) floating-point ele... | vcvtpd2ps | 4/1, 3/1 |
| `_mm256_cvtph_ps` | Converts the 8 x 16-bit half-precision float values in the 1... | "vcvtph2ps" | 4/1, 3/1 |
| `_mm256_cvtps_epi32` | Converts packed single-precision (32-bit) floating-point ele... | vcvtps2dq | 4/1, 3/1 |
| `_mm256_cvtps_pd` | Converts packed single-precision (32-bit) floating-point ele... | vcvtps2pd | 4/1, 3/1 |
| `_mm256_cvtps_ph` | Converts the 8 x 32-bit float values in the 256-bit vector `... | "vcvtps2ph" | 4/1, 3/1 |
| `_mm256_cvtsd_f64` | Returns the first element of the input vector of ` | movsd | 4/1, 3/1 |
| `_mm256_cvtsi256_si32` | Returns the first element of the input vector of ` |  | 4/1, 3/1 |
| `_mm256_cvtss_f32` | Returns the first element of the input vector of ` | movss | 4/1, 3/1 |
| `_mm256_cvttpd_epi32` | Converts packed double-precision (64-bit) floating-point ele... | vcvttpd2dq | 4/1, 3/1 |
| `_mm256_cvttps_epi32` | Converts packed single-precision (32-bit) floating-point ele... | vcvttps2dq | 4/1, 3/1 |
| `_mm256_div_pd` | Computes the division of each of the 4 packed 64-bit floatin... | vdivpd | 35/28, 13/10 |
| `_mm256_div_ps` | Computes the division of each of the 8 packed 32-bit floatin... | vdivps | 21/14, 10/7 |
| `_mm256_dp_ps` | Conditionally multiplies the packed single-precision (32-bit... | vdpps | — |
| `_mm256_extract_epi16` | Extracts a 16-bit integer from `a`, selected with `INDEX`. R... |  | — |
| `_mm256_extract_epi32` | Extracts a 32-bit integer from `a`, selected with `INDEX` |  | — |
| `_mm256_extract_epi64` | Extracts a 64-bit integer from `a`, selected with `INDEX` |  | — |
| `_mm256_extract_epi8` | Extracts an 8-bit integer from `a`, selected with `INDEX`. R... |  | — |
| `_mm256_extractf128_pd` | Extracts 128 bits (composed of 2 packed double-precision (64... | vextractf128 | — |
| `_mm256_extractf128_ps` | Extracts 128 bits (composed of 4 packed single-precision (32... | vextractf128 | — |
| `_mm256_extractf128_si256` | Extracts 128 bits (composed of integer data) from `a`, selec... | vextractf128 | — |
| `_mm256_extracti128_si256` | Extracts 128 bits (of integer data) from `a` selected with `... | vextractf128 | — |
| `_mm256_floor_pd` | Rounds packed double-precision (64-bit) floating point eleme... | vroundpd | — |
| `_mm256_floor_ps` | Rounds packed single-precision (32-bit) floating point eleme... | vroundps | — |
| `_mm256_fmadd_pd` | Multiplies packed double-precision (64-bit) floating-point e... | vfmadd | 5/1, 4/1 |
| `_mm256_fmadd_ps` | Multiplies packed single-precision (32-bit) floating-point e... | vfmadd | 5/1, 4/1 |
| `_mm256_fmaddsub_pd` | Multiplies packed double-precision (64-bit) floating-point e... | vfmaddsub | 5/1, 4/1 |
| `_mm256_fmaddsub_ps` | Multiplies packed single-precision (32-bit) floating-point e... | vfmaddsub | 5/1, 4/1 |
| `_mm256_fmsub_pd` | Multiplies packed double-precision (64-bit) floating-point e... | vfmsub | 5/1, 4/1 |
| `_mm256_fmsub_ps` | Multiplies packed single-precision (32-bit) floating-point e... | vfmsub213ps | 5/1, 4/1 |
| `_mm256_fmsubadd_pd` | Multiplies packed double-precision (64-bit) floating-point e... | vfmsubadd | 5/1, 4/1 |
| `_mm256_fmsubadd_ps` | Multiplies packed single-precision (32-bit) floating-point e... | vfmsubadd | 5/1, 4/1 |
| `_mm256_fnmadd_pd` | Multiplies packed double-precision (64-bit) floating-point e... | vfnmadd | 5/1, 4/1 |
| `_mm256_fnmadd_ps` | Multiplies packed single-precision (32-bit) floating-point e... | vfnmadd | 5/1, 4/1 |
| `_mm256_fnmsub_pd` | Multiplies packed double-precision (64-bit) floating-point e... | vfnmsub | 5/1, 4/1 |
| `_mm256_fnmsub_ps` | Multiplies packed single-precision (32-bit) floating-point e... | vfnmsub | 5/1, 4/1 |
| `_mm256_hadd_epi16` | Horizontally adds adjacent pairs of 16-bit integers in `a` a... | vphaddw | 5/2, 4/2 |
| `_mm256_hadd_epi32` | Horizontally adds adjacent pairs of 32-bit integers in `a` a... | vphaddd | 5/2, 4/2 |
| `_mm256_hadd_pd` | Horizontal addition of adjacent pairs in the two packed vect... | vhaddpd | 5/2, 4/2 |
| `_mm256_hadd_ps` | Horizontal addition of adjacent pairs in the two packed vect... | vhaddps | 5/2, 4/2 |
| `_mm256_hadds_epi16` | Horizontally adds adjacent pairs of 16-bit integers in `a` a... | vphaddsw | — |
| `_mm256_hsub_epi16` | Horizontally subtract adjacent pairs of 16-bit integers in `... | vphsubw | 5/2, 4/2 |
| `_mm256_hsub_epi32` | Horizontally subtract adjacent pairs of 32-bit integers in `... | vphsubd | 5/2, 4/2 |
| `_mm256_hsub_pd` | Horizontal subtraction of adjacent pairs in the two packed v... | vhsubpd | 5/2, 4/2 |
| `_mm256_hsub_ps` | Horizontal subtraction of adjacent pairs in the two packed v... | vhsubps | 5/2, 4/2 |
| `_mm256_hsubs_epi16` | Horizontally subtract adjacent pairs of 16-bit integers in `... | vphsubsw | — |
| `_mm256_insert_epi16` | Copies `a` to result, and inserts the 16-bit integer `i` int... |  | — |
| `_mm256_insert_epi32` | Copies `a` to result, and inserts the 32-bit integer `i` int... |  | — |
| `_mm256_insert_epi64` | Copies `a` to result, and insert the 64-bit integer `i` into... |  | — |
| `_mm256_insert_epi8` | Copies `a` to result, and inserts the 8-bit integer `i` into... |  | — |
| `_mm256_insertf128_pd` | Copies `a` to result, then inserts 128 bits (composed of 2 p... | vinsertf128 | — |
| `_mm256_insertf128_ps` | Copies `a` to result, then inserts 128 bits (composed of 4 p... | vinsertf128 | — |
| `_mm256_insertf128_si256` | Copies `a` to result, then inserts 128 bits from `b` into re... | vinsertf128 | — |
| `_mm256_inserti128_si256` | Copies `a` to `dst`, then insert 128 bits (of integer data) ... | vinsertf128 | — |
| `_mm256_madd_epi16` | Multiplies packed signed 16-bit integers in `a` and `b`, pro... | vpmaddwd | — |
| `_mm256_maddubs_epi16` | Vertically multiplies each unsigned 8-bit integer from `a` w... | vpmaddubsw | — |
| `_mm256_max_epi16` | Compares packed 16-bit integers in `a` and `b`, and returns ... | vpmaxsw | — |
| `_mm256_max_epi32` | Compares packed 32-bit integers in `a` and `b`, and returns ... | vpmaxsd | — |
| `_mm256_max_epi8` | Compares packed 8-bit integers in `a` and `b`, and returns t... | vpmaxsb | — |
| `_mm256_max_epu16` | Compares packed unsigned 16-bit integers in `a` and `b`, and... | vpmaxuw | — |
| `_mm256_max_epu32` | Compares packed unsigned 32-bit integers in `a` and `b`, and... | vpmaxud | — |
| `_mm256_max_epu8` | Compares packed unsigned 8-bit integers in `a` and `b`, and ... | vpmaxub | — |
| `_mm256_max_pd` | Compares packed double-precision (64-bit) floating-point ele... | vmaxpd | — |
| `_mm256_max_ps` | Compares packed single-precision (32-bit) floating-point ele... | vmaxps | — |
| `_mm256_min_epi16` | Compares packed 16-bit integers in `a` and `b`, and returns ... | vpminsw | — |
| `_mm256_min_epi32` | Compares packed 32-bit integers in `a` and `b`, and returns ... | vpminsd | — |
| `_mm256_min_epi8` | Compares packed 8-bit integers in `a` and `b`, and returns t... | vpminsb | — |
| `_mm256_min_epu16` | Compares packed unsigned 16-bit integers in `a` and `b`, and... | vpminuw | — |
| `_mm256_min_epu32` | Compares packed unsigned 32-bit integers in `a` and `b`, and... | vpminud | — |
| `_mm256_min_epu8` | Compares packed unsigned 8-bit integers in `a` and `b`, and ... | vpminub | — |
| `_mm256_min_pd` | Compares packed double-precision (64-bit) floating-point ele... | vminpd | — |
| `_mm256_min_ps` | Compares packed single-precision (32-bit) floating-point ele... | vminps | — |
| `_mm256_movedup_pd` | Duplicate even-indexed double-precision (64-bit) floating-po... | vmovddup | — |
| `_mm256_movehdup_ps` | Duplicate odd-indexed single-precision (32-bit) floating-poi... | vmovshdup | — |
| `_mm256_moveldup_ps` | Duplicate even-indexed single-precision (32-bit) floating-po... | vmovsldup | — |
| `_mm256_movemask_epi8` | Creates mask from the most significant bit of each 8-bit ele... | vpmovmskb | — |
| `_mm256_movemask_pd` | Sets each bit of the returned mask based on the most signifi... | vmovmskpd | — |
| `_mm256_movemask_ps` | Sets each bit of the returned mask based on the most signifi... | vmovmskps | — |
| `_mm256_mpsadbw_epu8` | Computes the sum of absolute differences (SADs) of quadruple... | vmpsadbw | — |
| `_mm256_mul_epi32` | Multiplies the low 32-bit integers from each packed 64-bit e... | vpmuldq | 10/2, 4/1 |
| `_mm256_mul_epu32` | Multiplies the low unsigned 32-bit integers from each packed... | vpmuludq | 10/2, 4/1 |
| `_mm256_mul_pd` | Multiplies packed double-precision (64-bit) floating-point e... | vmulpd | 5/1, 3/1 |
| `_mm256_mul_ps` | Multiplies packed single-precision (32-bit) floating-point e... | vmulps | 5/1, 3/1 |
| `_mm256_mulhi_epi16` | Multiplies the packed 16-bit integers in `a` and `b`, produc... | vpmulhw | 5/1, 3/1 |
| `_mm256_mulhi_epu16` | Multiplies the packed unsigned 16-bit integers in `a` and `b... | vpmulhuw | 5/1, 3/1 |
| `_mm256_mulhrs_epi16` | Multiplies packed 16-bit integers in `a` and `b`, producing ... | vpmulhrsw | — |
| `_mm256_mullo_epi16` | Multiplies the packed 16-bit integers in `a` and `b`, produc... | vpmullw | 5/1, 3/1 |
| `_mm256_mullo_epi32` | Multiplies the packed 32-bit integers in `a` and `b`, produc... | vpmulld | 10/2, 4/1 |
| `_mm256_or_pd` | Computes the bitwise OR packed double-precision (64-bit) flo... | vorp | 1/1, 1/1 |
| `_mm256_or_ps` | Computes the bitwise OR packed single-precision (32-bit) flo... | vorps | 1/1, 1/1 |
| `_mm256_or_si256` | Computes the bitwise OR of 256 bits (representing integer da... | vorps | 1/1, 1/1 |
| `_mm256_packs_epi16` | Converts packed 16-bit integers from `a` and `b` to packed 8... | vpacksswb | 1/1, 1/1 |
| `_mm256_packs_epi32` | Converts packed 32-bit integers from `a` and `b` to packed 1... | vpackssdw | 1/1, 1/1 |
| `_mm256_packus_epi16` | Converts packed 16-bit integers from `a` and `b` to packed 8... | vpackuswb | 1/1, 1/1 |
| `_mm256_packus_epi32` | Converts packed 32-bit integers from `a` and `b` to packed 1... | vpackusdw | 1/1, 1/1 |
| `_mm256_permute2f128_pd` | Shuffles 256 bits (composed of 4 packed double-precision (64... | vperm2f128 | 3/1, 2/1 |
| `_mm256_permute2f128_ps` | Shuffles 256 bits (composed of 8 packed single-precision (32... | vperm2f128 | 3/1, 2/1 |
| `_mm256_permute2f128_si256` | Shuffles 128-bits (composed of integer data) selected by `im... | vperm2f128 | 3/1, 2/1 |
| `_mm256_permute2x128_si256` | Shuffles 128-bits of integer data selected by `imm8` from `a... | vperm2f128 | 3/1, 2/1 |
| `_mm256_permute4x64_epi64` | Permutes 64-bit integers from `a` using control mask `imm8` | vpermpd | 3/1, 2/1 |
| `_mm256_permute4x64_pd` | Shuffles 64-bit floating-point elements in `a` across lanes ... | vpermpd | 3/1, 2/1 |
| `_mm256_permute_pd` | Shuffles double-precision (64-bit) floating-point elements i... | vshufpd | 3/1, 2/1 |
| `_mm256_permute_ps` | Shuffles single-precision (32-bit) floating-point elements i... | vshufps | 3/1, 2/1 |
| `_mm256_permutevar8x32_epi32` | Permutes packed 32-bit integers from `a` according to the co... | vpermps | 3/1, 2/1 |
| `_mm256_permutevar8x32_ps` | Shuffles eight 32-bit floating-point elements in `a` across ... | vpermps | 3/1, 2/1 |
| `_mm256_permutevar_pd` | Shuffles double-precision (64-bit) floating-point elements i... | vpermilpd | 3/1, 2/1 |
| `_mm256_permutevar_ps` | Shuffles single-precision (32-bit) floating-point elements i... | vpermilps | 3/1, 2/1 |
| `_mm256_rcp_ps` | Computes the approximate reciprocal of packed single-precisi... | vrcpps | 7/1, 4/1 |
| `_mm256_round_pd` | Rounds packed double-precision (64-bit) floating point eleme... | vroundpd | — |
| `_mm256_round_ps` | Rounds packed single-precision (32-bit) floating point eleme... | vroundps | — |
| `_mm256_rsqrt_ps` | Computes the approximate reciprocal square root of packed si... | vrsqrtps | 7/1, 4/1 |
| `_mm256_sad_epu8` | Computes the absolute differences of packed unsigned 8-bit i... | vpsadbw | — |
| `_mm256_set1_epi16` | Broadcasts 16-bit integer `a` to all elements of returned ve... | vpshufb | — |
| `_mm256_set1_epi32` | Broadcasts 32-bit integer `a` to all elements of returned ve... |  | — |
| `_mm256_set1_epi64x` | Broadcasts 64-bit integer `a` to all elements of returned ve... | vinsertf128 | — |
| `_mm256_set1_epi8` | Broadcasts 8-bit integer `a` to all elements of returned vec... |  | — |
| `_mm256_set1_pd` | Broadcasts double-precision (64-bit) floating-point value `a... |  | — |
| `_mm256_set1_ps` | Broadcasts single-precision (32-bit) floating-point value `a... |  | — |
| `_mm256_set_epi16` | Sets packed 16-bit integers in returned vector with the supp... |  | — |
| `_mm256_set_epi32` | Sets packed 32-bit integers in returned vector with the supp... |  | — |
| `_mm256_set_epi64x` | Sets packed 64-bit integers in returned vector with the supp... |  | — |
| `_mm256_set_epi8` | Sets packed 8-bit integers in returned vector with the suppl... |  | — |
| `_mm256_set_m128` | Sets packed __m256 returned vector with the supplied values | vinsertf128 | — |
| `_mm256_set_m128d` | Sets packed __m256d returned vector with the supplied values | vinsertf128 | — |
| `_mm256_set_m128i` | Sets packed __m256i returned vector with the supplied values | vinsertf128 | — |
| `_mm256_set_pd` | Sets packed double-precision (64-bit) floating-point element... | vinsertf128 | — |
| `_mm256_set_ps` | Sets packed single-precision (32-bit) floating-point element... |  | — |
| `_mm256_setr_epi16` | Sets packed 16-bit integers in returned vector with the supp... |  | — |
| `_mm256_setr_epi32` | Sets packed 32-bit integers in returned vector with the supp... |  | — |
| `_mm256_setr_epi64x` | Sets packed 64-bit integers in returned vector with the supp... |  | — |
| `_mm256_setr_epi8` | Sets packed 8-bit integers in returned vector with the suppl... |  | — |
| `_mm256_setr_m128` | Sets packed __m256 returned vector with the supplied values | vinsertf128 | — |
| `_mm256_setr_m128d` | Sets packed __m256d returned vector with the supplied values | vinsertf128 | — |
| `_mm256_setr_m128i` | Sets packed __m256i returned vector with the supplied values | vinsertf128 | — |
| `_mm256_setr_pd` | Sets packed double-precision (64-bit) floating-point element... |  | — |
| `_mm256_setr_ps` | Sets packed single-precision (32-bit) floating-point element... |  | — |
| `_mm256_setzero_pd` | Returns vector of type __m256d with all elements set to zero | vxorp | — |
| `_mm256_setzero_ps` | Returns vector of type __m256 with all elements set to zero | vxorps | — |
| `_mm256_setzero_si256` | Returns vector of type __m256i with all elements set to zero | vxor | — |
| `_mm256_shuffle_epi32` | Shuffles 32-bit integers in 128-bit lanes of `a` using the c... | vshufps | 1/1, 1/1 |
| `_mm256_shuffle_epi8` | Shuffles bytes from `a` according to the content of `b`.  Fo... | vpshufb | 1/1, 1/1 |
| `_mm256_shuffle_pd` | Shuffles double-precision (64-bit) floating-point elements w... | vshufpd | 1/1, 1/1 |
| `_mm256_shuffle_ps` | Shuffles single-precision (32-bit) floating-point elements i... | vshufps | 1/1, 1/1 |
| `_mm256_shufflehi_epi16` | Shuffles 16-bit integers in the high 64 bits of 128-bit lane... | vpshufhw | 1/1, 1/1 |
| `_mm256_shufflelo_epi16` | Shuffles 16-bit integers in the low 64 bits of 128-bit lanes... | vpshuflw | 1/1, 1/1 |
| `_mm256_sign_epi16` | Negates packed 16-bit integers in `a` when the corresponding... | vpsignw | — |
| `_mm256_sign_epi32` | Negates packed 32-bit integers in `a` when the corresponding... | vpsignd | — |
| `_mm256_sign_epi8` | Negates packed 8-bit integers in `a` when the corresponding ... | vpsignb | — |
| `_mm256_sll_epi16` | Shifts packed 16-bit integers in `a` left by `count` while s... | vpsllw | 1/1, 1/1 |
| `_mm256_sll_epi32` | Shifts packed 32-bit integers in `a` left by `count` while s... | vpslld | 1/1, 1/1 |
| `_mm256_sll_epi64` | Shifts packed 64-bit integers in `a` left by `count` while s... | vpsllq | 1/1, 1/1 |
| `_mm256_slli_epi16` | Shifts packed 16-bit integers in `a` left by `IMM8` while sh... | vpsllw | 1/1, 1/1 |
| `_mm256_slli_epi32` | Shifts packed 32-bit integers in `a` left by `IMM8` while sh... | vpslld | 1/1, 1/1 |
| `_mm256_slli_epi64` | Shifts packed 64-bit integers in `a` left by `IMM8` while sh... | vpsllq | 1/1, 1/1 |
| `_mm256_slli_si256` | Shifts 128-bit lanes in `a` left by `imm8` bytes while shift... | vpslldq | 1/1, 1/1 |
| `_mm256_sllv_epi32` | Shifts packed 32-bit integers in `a` left by the amount spec... | vpsllvd | 1/1, 1/1 |
| `_mm256_sllv_epi64` | Shifts packed 64-bit integers in `a` left by the amount spec... | vpsllvq | 1/1, 1/1 |
| `_mm256_sqrt_pd` | Returns the square root of packed double-precision (64-bit) ... | vsqrtpd | 28/28, 20/13 |
| `_mm256_sqrt_ps` | Returns the square root of packed single-precision (32-bit) ... | vsqrtps | 19/14, 14/7 |
| `_mm256_sra_epi16` | Shifts packed 16-bit integers in `a` right by `count` while ... | vpsraw | 1/1, 1/1 |
| `_mm256_sra_epi32` | Shifts packed 32-bit integers in `a` right by `count` while ... | vpsrad | 1/1, 1/1 |
| `_mm256_srai_epi16` | Shifts packed 16-bit integers in `a` right by `IMM8` while s... | vpsraw | 1/1, 1/1 |
| `_mm256_srai_epi32` | Shifts packed 32-bit integers in `a` right by `IMM8` while s... | vpsrad | 1/1, 1/1 |
| `_mm256_srav_epi32` | Shifts packed 32-bit integers in `a` right by the amount spe... | vpsravd | 1/1, 1/1 |
| `_mm256_srl_epi16` | Shifts packed 16-bit integers in `a` right by `count` while ... | vpsrlw | 1/1, 1/1 |
| `_mm256_srl_epi32` | Shifts packed 32-bit integers in `a` right by `count` while ... | vpsrld | 1/1, 1/1 |
| `_mm256_srl_epi64` | Shifts packed 64-bit integers in `a` right by `count` while ... | vpsrlq | 1/1, 1/1 |
| `_mm256_srli_epi16` | Shifts packed 16-bit integers in `a` right by `IMM8` while s... | vpsrlw | 1/1, 1/1 |
| `_mm256_srli_epi32` | Shifts packed 32-bit integers in `a` right by `IMM8` while s... | vpsrld | 1/1, 1/1 |
| `_mm256_srli_epi64` | Shifts packed 64-bit integers in `a` right by `IMM8` while s... | vpsrlq | 1/1, 1/1 |
| `_mm256_srli_si256` | Shifts 128-bit lanes in `a` right by `imm8` bytes while shif... | vpsrldq | 1/1, 1/1 |
| `_mm256_srlv_epi32` | Shifts packed 32-bit integers in `a` right by the amount spe... | vpsrlvd | 1/1, 1/1 |
| `_mm256_srlv_epi64` | Shifts packed 64-bit integers in `a` right by the amount spe... | vpsrlvq | 1/1, 1/1 |
| `_mm256_sub_epi16` | Subtract packed 16-bit integers in `b` from packed 16-bit in... | vpsubw | 1/1, 1/1 |
| `_mm256_sub_epi32` | Subtract packed 32-bit integers in `b` from packed 32-bit in... | vpsubd | 1/1, 1/1 |
| `_mm256_sub_epi64` | Subtract packed 64-bit integers in `b` from packed 64-bit in... | vpsubq | 1/1, 1/1 |
| `_mm256_sub_epi8` | Subtract packed 8-bit integers in `b` from packed 8-bit inte... | vpsubb | 1/1, 1/1 |
| `_mm256_sub_pd` | Subtracts packed double-precision (64-bit) floating-point el... | vsubpd | — |
| `_mm256_sub_ps` | Subtracts packed single-precision (32-bit) floating-point el... | vsubps | — |
| `_mm256_subs_epi16` | Subtract packed 16-bit integers in `b` from packed 16-bit in... | vpsubsw | 1/1, 1/1 |
| `_mm256_subs_epi8` | Subtract packed 8-bit integers in `b` from packed 8-bit inte... | vpsubsb | 1/1, 1/1 |
| `_mm256_subs_epu16` | Subtract packed unsigned 16-bit integers in `b` from packed ... | vpsubusw | — |
| `_mm256_subs_epu8` | Subtract packed unsigned 8-bit integers in `b` from packed 8... | vpsubusb | — |
| `_mm256_testc_pd` | Computes the bitwise AND of 256 bits (representing double-pr... | vtestpd | — |
| `_mm256_testc_ps` | Computes the bitwise AND of 256 bits (representing single-pr... | vtestps | — |
| `_mm256_testc_si256` | Computes the bitwise AND of 256 bits (representing integer d... | vptest | — |
| `_mm256_testnzc_pd` | Computes the bitwise AND of 256 bits (representing double-pr... | vtestpd | — |
| `_mm256_testnzc_ps` | Computes the bitwise AND of 256 bits (representing single-pr... | vtestps | — |
| `_mm256_testnzc_si256` | Computes the bitwise AND of 256 bits (representing integer d... | vptest | — |
| `_mm256_testz_pd` | Computes the bitwise AND of 256 bits (representing double-pr... | vtestpd | — |
| `_mm256_testz_ps` | Computes the bitwise AND of 256 bits (representing single-pr... | vtestps | — |
| `_mm256_testz_si256` | Computes the bitwise AND of 256 bits (representing integer d... | vptest | — |
| `_mm256_undefined_pd` | Returns vector of type `__m256d` with indeterminate elements... |  | — |
| `_mm256_undefined_ps` | Returns vector of type `__m256` with indeterminate elements.... |  | — |
| `_mm256_undefined_si256` | Returns vector of type __m256i with with indeterminate eleme... |  | — |
| `_mm256_unpackhi_epi16` | Unpacks and interleave 16-bit integers from the high half of... | vpunpckhwd | 1/1, 1/1 |
| `_mm256_unpackhi_epi32` | Unpacks and interleave 32-bit integers from the high half of... | vunpckhps | 1/1, 1/1 |
| `_mm256_unpackhi_epi64` | Unpacks and interleave 64-bit integers from the high half of... | vunpckhpd | 1/1, 1/1 |
| `_mm256_unpackhi_epi8` | Unpacks and interleave 8-bit integers from the high half of ... | vpunpckhbw | 1/1, 1/1 |
| `_mm256_unpackhi_pd` | Unpacks and interleave double-precision (64-bit) floating-po... | vunpckhpd | 1/1, 1/1 |
| `_mm256_unpackhi_ps` | Unpacks and interleave single-precision (32-bit) floating-po... | vunpckhps | 1/1, 1/1 |
| `_mm256_unpacklo_epi16` | Unpacks and interleave 16-bit integers from the low half of ... | vpunpcklwd | 1/1, 1/1 |
| `_mm256_unpacklo_epi32` | Unpacks and interleave 32-bit integers from the low half of ... | vunpcklps | 1/1, 1/1 |
| `_mm256_unpacklo_epi64` | Unpacks and interleave 64-bit integers from the low half of ... | vunpcklpd | 1/1, 1/1 |
| `_mm256_unpacklo_epi8` | Unpacks and interleave 8-bit integers from the low half of e... | vpunpcklbw | 1/1, 1/1 |
| `_mm256_unpacklo_pd` | Unpacks and interleave double-precision (64-bit) floating-po... | vunpcklpd | 1/1, 1/1 |
| `_mm256_unpacklo_ps` | Unpacks and interleave single-precision (32-bit) floating-po... | vunpcklps | 1/1, 1/1 |
| `_mm256_xor_pd` | Computes the bitwise XOR of packed double-precision (64-bit)... | vxorp | 1/1, 1/1 |
| `_mm256_xor_ps` | Computes the bitwise XOR of packed single-precision (32-bit)... | vxorps | 1/1, 1/1 |
| `_mm256_xor_si256` | Computes the bitwise XOR of 256 bits (representing integer d... | vxorps | 1/1, 1/1 |
| `_mm256_zeroall` | Zeroes the contents of all XMM or YMM registers | vzeroall | — |
| `_mm256_zeroupper` | Zeroes the upper 128 bits of all YMM registers; the lower 12... | vzeroupper | — |
| `_mm256_zextpd128_pd256` | Constructs a 256-bit floating-point vector of ` |  | — |
| `_mm256_zextps128_ps256` | Constructs a 256-bit floating-point vector of ` |  | — |
| `_mm256_zextsi128_si256` | Constructs a 256-bit integer vector from a 128-bit integer v... |  | — |
| `_mm_blend_epi32` | Blends packed 32-bit integers from `a` and `b` using control... | vblendps | 1/1, 1/1 |
| `_mm_broadcast_ss` | Broadcasts a single-precision (32-bit) floating-point elemen... | vbroadcastss | — |
| `_mm_broadcastb_epi8` | Broadcasts the low packed 8-bit integer from `a` to all elem... | vpbroadcastb | — |
| `_mm_broadcastd_epi32` | Broadcasts the low packed 32-bit integer from `a` to all ele... | vbroadcastss | — |
| `_mm_broadcastq_epi64` | Broadcasts the low packed 64-bit integer from `a` to all ele... | vmovddup | — |
| `_mm_broadcastsd_pd` | Broadcasts the low double-precision (64-bit) floating-point ... | vmovddup | — |
| `_mm_broadcastsi128_si256` | Broadcasts 128 bits of integer data from a to all 128-bit la... |  | — |
| `_mm_broadcastss_ps` | Broadcasts the low single-precision (32-bit) floating-point ... | vbroadcastss | — |
| `_mm_broadcastw_epi16` | Broadcasts the low packed 16-bit integer from a to all eleme... | vpbroadcastw | — |
| `_mm_cmp_pd` | Compares packed double-precision (64-bit) floating-point ele... | vcmpeqpd | 3/1, 3/1 |
| `_mm_cmp_ps` | Compares packed single-precision (32-bit) floating-point ele... | vcmpeqps | 3/1, 3/1 |
| `_mm_cmp_sd` | Compares the lower double-precision (64-bit) floating-point ... | vcmpeqsd | — |
| `_mm_cmp_ss` | Compares the lower single-precision (32-bit) floating-point ... | vcmpeqss | — |
| `_mm_cvtph_ps` | Converts the 4 x 16-bit half-precision float values in the l... | "vcvtph2ps" | 4/1, 3/1 |
| `_mm_cvtps_ph` | Converts the 4 x 32-bit float values in the 128-bit vector `... | "vcvtps2ph" | 4/1, 3/1 |
| `_mm_fmadd_pd` | Multiplies packed double-precision (64-bit) floating-point e... | vfmadd | 5/1, 4/1 |
| `_mm_fmadd_ps` | Multiplies packed single-precision (32-bit) floating-point e... | vfmadd | 5/1, 4/1 |
| `_mm_fmadd_sd` | Multiplies the lower double-precision (64-bit) floating-poin... | vfmadd | 5/1, 4/1 |
| `_mm_fmadd_ss` | Multiplies the lower single-precision (32-bit) floating-poin... | vfmadd | 5/1, 4/1 |
| `_mm_fmaddsub_pd` | Multiplies packed double-precision (64-bit) floating-point e... | vfmaddsub | 5/1, 4/1 |
| `_mm_fmaddsub_ps` | Multiplies packed single-precision (32-bit) floating-point e... | vfmaddsub | 5/1, 4/1 |
| `_mm_fmsub_pd` | Multiplies packed double-precision (64-bit) floating-point e... | vfmsub | 5/1, 4/1 |
| `_mm_fmsub_ps` | Multiplies packed single-precision (32-bit) floating-point e... | vfmsub213ps | 5/1, 4/1 |
| `_mm_fmsub_sd` | Multiplies the lower double-precision (64-bit) floating-poin... | vfmsub | 5/1, 4/1 |
| `_mm_fmsub_ss` | Multiplies the lower single-precision (32-bit) floating-poin... | vfmsub | 5/1, 4/1 |
| `_mm_fmsubadd_pd` | Multiplies packed double-precision (64-bit) floating-point e... | vfmsubadd | 5/1, 4/1 |
| `_mm_fmsubadd_ps` | Multiplies packed single-precision (32-bit) floating-point e... | vfmsubadd | 5/1, 4/1 |
| `_mm_fnmadd_pd` | Multiplies packed double-precision (64-bit) floating-point e... | vfnmadd | 5/1, 4/1 |
| `_mm_fnmadd_ps` | Multiplies packed single-precision (32-bit) floating-point e... | vfnmadd | 5/1, 4/1 |
| `_mm_fnmadd_sd` | Multiplies the lower double-precision (64-bit) floating-poin... | vfnmadd | 5/1, 4/1 |
| `_mm_fnmadd_ss` | Multiplies the lower single-precision (32-bit) floating-poin... | vfnmadd | 5/1, 4/1 |
| `_mm_fnmsub_pd` | Multiplies packed double-precision (64-bit) floating-point e... | vfnmsub | 5/1, 4/1 |
| `_mm_fnmsub_ps` | Multiplies packed single-precision (32-bit) floating-point e... | vfnmsub | 5/1, 4/1 |
| `_mm_fnmsub_sd` | Multiplies the lower double-precision (64-bit) floating-poin... | vfnmsub | 5/1, 4/1 |
| `_mm_fnmsub_ss` | Multiplies the lower single-precision (32-bit) floating-poin... | vfnmsub | 5/1, 4/1 |
| `_mm_permute_pd` | Shuffles double-precision (64-bit) floating-point elements i... | vshufpd | 1/1, 1/1 |
| `_mm_permute_ps` | Shuffles single-precision (32-bit) floating-point elements i... | vshufps | 1/1, 1/1 |
| `_mm_permutevar_pd` | Shuffles double-precision (64-bit) floating-point elements i... | vpermilpd | 1/1, 1/1 |
| `_mm_permutevar_ps` | Shuffles single-precision (32-bit) floating-point elements i... | vpermilps | 1/1, 1/1 |
| `_mm_sllv_epi32` | Shifts packed 32-bit integers in `a` left by the amount spec... | vpsllvd | 1/1, 1/1 |
| `_mm_sllv_epi64` | Shifts packed 64-bit integers in `a` left by the amount spec... | vpsllvq | 1/1, 1/1 |
| `_mm_srav_epi32` | Shifts packed 32-bit integers in `a` right by the amount spe... | vpsravd | 1/1, 1/1 |
| `_mm_srlv_epi32` | Shifts packed 32-bit integers in `a` right by the amount spe... | vpsrlvd | 1/1, 1/1 |
| `_mm_srlv_epi64` | Shifts packed 64-bit integers in `a` right by the amount spe... | vpsrlvq | 1/1, 1/1 |
| `_mm_testc_pd` | Computes the bitwise AND of 128 bits (representing double-pr... | vtestpd | — |
| `_mm_testc_ps` | Computes the bitwise AND of 128 bits (representing single-pr... | vtestps | — |
| `_mm_testnzc_pd` | Computes the bitwise AND of 128 bits (representing double-pr... | vtestpd | — |
| `_mm_testnzc_ps` | Computes the bitwise AND of 128 bits (representing single-pr... | vtestps | — |
| `_mm_testz_pd` | Computes the bitwise AND of 128 bits (representing double-pr... | vtestpd | — |
| `_mm_testz_ps` | Computes the bitwise AND of 128 bits (representing single-pr... | vtestps | — |
| `_mm_tzcnt_32` | Counts the number of trailing least significant zero bits.  ... | tzcnt | — |
| `_mm_tzcnt_64` | Counts the number of trailing least significant zero bits.  ... | tzcnt | — |
| `_mulx_u32` |  |  | — |
| `_mulx_u64` | Unsigned multiply without affecting flags.  Unsigned multipl... |  | — |
| `_pdep_u32` | Scatter contiguous low order bits of `a` to the result at th... | pdep | — |
| `_pdep_u64` | Scatter contiguous low order bits of `a` to the result at th... | pdep | — |
| `_pext_u32` | Gathers the bits of `x` specified by the `mask` into the con... | pext | — |
| `_pext_u64` | Gathers the bits of `x` specified by the `mask` into the con... | pext | — |
| `_tzcnt_u16` | Counts the number of trailing least significant zero bits.  ... | tzcnt | — |
| `_tzcnt_u32` | Counts the number of trailing least significant zero bits.  ... | tzcnt | — |
| `_tzcnt_u64` | Counts the number of trailing least significant zero bits.  ... | tzcnt | — |

### Stable, Unsafe (71 intrinsics) — use safe_unaligned_simd

| Name | Description | Safe Variant |
|------|-------------|--------------|
| `_mm256_i32gather_epi32` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm256_i32gather_epi64` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm256_i32gather_pd` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm256_i32gather_ps` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm256_i64gather_epi32` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm256_i64gather_epi64` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm256_i64gather_pd` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm256_i64gather_ps` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm256_lddqu_si256` | Loads 256-bits of integer data from unaligned memory into re... | — |
| `_mm256_load_pd` | Loads 256-bits (composed of 4 packed double-precision (64-bi... | — |
| `_mm256_load_ps` | Loads 256-bits (composed of 8 packed single-precision (32-bi... | — |
| `_mm256_load_si256` | Loads 256-bits of integer data from memory into result. `mem... | — |
| `_mm256_loadu2_m128` | Loads two 128-bit values (composed of 4 packed single-precis... | safe_unaligned_simd::`_mm256_loadu2_m128` |
| `_mm256_loadu2_m128d` | Loads two 128-bit values (composed of 2 packed double-precis... | safe_unaligned_simd::`_mm256_loadu2_m128d` |
| `_mm256_loadu2_m128i` | Loads two 128-bit values (composed of integer data) from mem... | safe_unaligned_simd::`_mm256_loadu2_m128i` |
| `_mm256_loadu_pd` | Loads 256-bits (composed of 4 packed double-precision (64-bi... | safe_unaligned_simd::`_mm256_loadu_pd` |
| `_mm256_loadu_ps` | Loads 256-bits (composed of 8 packed single-precision (32-bi... | safe_unaligned_simd::`_mm256_loadu_ps` |
| `_mm256_loadu_si256` | Loads 256-bits of integer data from memory into result. `mem... | safe_unaligned_simd::`_mm256_loadu_si256` |
| `_mm256_mask_i32gather_epi32` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm256_mask_i32gather_epi64` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm256_mask_i32gather_pd` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm256_mask_i32gather_ps` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm256_mask_i64gather_epi32` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm256_mask_i64gather_epi64` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm256_mask_i64gather_pd` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm256_mask_i64gather_ps` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm256_maskload_epi32` | Loads packed 32-bit integers from memory pointed by `mem_add... | — |
| `_mm256_maskload_epi64` | Loads packed 64-bit integers from memory pointed by `mem_add... | — |
| `_mm256_maskload_pd` | Loads packed double-precision (64-bit) floating-point elemen... | — |
| `_mm256_maskload_ps` | Loads packed single-precision (32-bit) floating-point elemen... | — |
| `_mm256_maskstore_epi32` | Stores packed 32-bit integers from `a` into memory pointed b... | — |
| `_mm256_maskstore_epi64` | Stores packed 64-bit integers from `a` into memory pointed b... | — |
| `_mm256_maskstore_pd` | Stores packed double-precision (64-bit) floating-point eleme... | — |
| `_mm256_maskstore_ps` | Stores packed single-precision (32-bit) floating-point eleme... | — |
| `_mm256_store_pd` | Stores 256-bits (composed of 4 packed double-precision (64-b... | — |
| `_mm256_store_ps` | Stores 256-bits (composed of 8 packed single-precision (32-b... | — |
| `_mm256_store_si256` | Stores 256-bits of integer data from `a` into memory. `mem_a... | — |
| `_mm256_storeu2_m128` | Stores the high and low 128-bit halves (each composed of 4 p... | safe_unaligned_simd::`_mm256_storeu2_m128` |
| `_mm256_storeu2_m128d` | Stores the high and low 128-bit halves (each composed of 2 p... | safe_unaligned_simd::`_mm256_storeu2_m128d` |
| `_mm256_storeu2_m128i` | Stores the high and low 128-bit halves (each composed of int... | safe_unaligned_simd::`_mm256_storeu2_m128i` |
| `_mm256_storeu_pd` | Stores 256-bits (composed of 4 packed double-precision (64-b... | safe_unaligned_simd::`_mm256_storeu_pd` |
| `_mm256_storeu_ps` | Stores 256-bits (composed of 8 packed single-precision (32-b... | safe_unaligned_simd::`_mm256_storeu_ps` |
| `_mm256_storeu_si256` | Stores 256-bits of integer data from `a` into memory. `mem_a... | safe_unaligned_simd::`_mm256_storeu_si256` |
| `_mm256_stream_load_si256` | Load 256-bits of integer data from memory into dst using a n... | — |
| `_mm256_stream_pd` | Moves double-precision values from a 256-bit vector of ` | — |
| `_mm256_stream_ps` | Moves single-precision floating point values from a 256-bit ... | — |
| `_mm256_stream_si256` | Moves integer data from a 256-bit integer vector to a 32-byt... | — |
| `_mm_i32gather_epi32` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm_i32gather_epi64` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm_i32gather_pd` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm_i32gather_ps` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm_i64gather_epi32` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm_i64gather_epi64` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm_i64gather_pd` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm_i64gather_ps` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm_mask_i32gather_epi32` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm_mask_i32gather_epi64` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm_mask_i32gather_pd` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm_mask_i32gather_ps` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm_mask_i64gather_epi32` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm_mask_i64gather_epi64` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm_mask_i64gather_pd` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm_mask_i64gather_ps` | Returns values from `slice` at offsets determined by `offset... | — |
| `_mm_maskload_epi32` | Loads packed 32-bit integers from memory pointed by `mem_add... | — |
| `_mm_maskload_epi64` | Loads packed 64-bit integers from memory pointed by `mem_add... | — |
| `_mm_maskload_pd` | Loads packed double-precision (64-bit) floating-point elemen... | — |
| `_mm_maskload_ps` | Loads packed single-precision (32-bit) floating-point elemen... | — |
| `_mm_maskstore_epi32` | Stores packed 32-bit integers from `a` into memory pointed b... | — |
| `_mm_maskstore_epi64` | Stores packed 64-bit integers from `a` into memory pointed b... | — |
| `_mm_maskstore_pd` | Stores packed double-precision (64-bit) floating-point eleme... | — |
| `_mm_maskstore_ps` | Stores packed single-precision (32-bit) floating-point eleme... | — |



# X64V1Token (Sse2Token) — x86-64-v1

Proof that SSE + SSE2 are available (x86-64-v1 baseline level).

**Architecture:** x86_64 | **Features:** sse, sse2
**Total intrinsics:** 334 (277 safe, 57 unsafe, 334 stable, 0 unstable/unknown)

## Usage

```rust
use archmage::prelude::*;

if let Some(token) = X64V1Token::summon() {
    process(token, &mut data);
}

#[arcane(import_intrinsics)]  // Entry point only
fn process(token: X64V1Token, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(4) {
        process_chunk(token, chunk.try_into().unwrap());
    }
}

#[rite(import_intrinsics)]  // All inner helpers
fn process_chunk(_: X64V1Token, chunk: &mut [f32; 4]) {
    let v = _mm_loadu_ps(chunk);  // safe!
    let doubled = _mm_add_ps(v, v);  // value intrinsic (safe inside #[rite])
    _mm_storeu_ps(chunk, doubled);  // safe!
}
// No unsafe anywhere. Use #![forbid(unsafe_code)] in your crate.
```

## Safe Memory Operations (via import_intrinsics)

| Function | Safe Signature |
|----------|---------------|
| `_mm_load1_pd` | `fn _mm_load1_pd(mem_addr: &f64) -> __m128d` |
| `_mm_load1_ps` | `fn _mm_load1_ps(mem_addr: &f32) -> __m128` |
| `_mm_load_pd1` | `fn _mm_load_pd1(mem_addr: &f64) -> __m128d` |
| `_mm_load_ps1` | `fn _mm_load_ps1(mem_addr: &f32) -> __m128` |
| `_mm_load_sd` | `fn _mm_load_sd(mem_addr: &f64) -> __m128d` |
| `_mm_load_ss` | `fn _mm_load_ss(mem_addr: &f32) -> __m128` |
| `_mm_loadh_pd` | `fn _mm_loadh_pd(a: __m128d, mem_addr: &f64) -> __m128d` |
| `_mm_loadl_epi64` | `fn _mm_loadl_epi64<T: Is128BitsUnaligned>(mem_addr: &T) -> __m128i` |
| `_mm_loadl_pd` | `fn _mm_loadl_pd(a: __m128d, mem_addr: &f64) -> __m128d` |
| `_mm_loadu_pd` | `fn _mm_loadu_pd(mem_addr: &[f64; 2]) -> __m128d` |
| `_mm_loadu_ps` | `fn _mm_loadu_ps(mem_addr: &[f32; 4]) -> __m128` |
| `_mm_loadu_si128` | `fn _mm_loadu_si128<T: Is128BitsUnaligned>(mem_addr: &T) -> __m128i` |
| `_mm_loadu_si16` | `fn _mm_loadu_si16<T: Is16BitsUnaligned>(mem_addr: &T) -> __m128i` |
| `_mm_loadu_si32` | `fn _mm_loadu_si32<T: Is32BitsUnaligned>(mem_addr: &T) -> __m128i` |
| `_mm_loadu_si64` | `fn _mm_loadu_si64<T: Is64BitsUnaligned>(mem_addr: &T) -> __m128i` |
| `_mm_store_sd` | `fn _mm_store_sd(mem_addr: &mut f64, a: __m128d) -> ()` |
| `_mm_store_ss` | `fn _mm_store_ss(mem_addr: &mut f32, a: __m128) -> ()` |
| `_mm_storeh_pd` | `fn _mm_storeh_pd(mem_addr: &mut f64, a: __m128d) -> ()` |
| `_mm_storel_epi64` | `fn _mm_storel_epi64<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) -> ()` |
| `_mm_storel_pd` | `fn _mm_storel_pd(mem_addr: &mut f64, a: __m128d) -> ()` |
| `_mm_storeu_pd` | `fn _mm_storeu_pd(mem_addr: &mut [f64; 2], a: __m128d) -> ()` |
| `_mm_storeu_ps` | `fn _mm_storeu_ps(mem_addr: &mut [f32; 4], a: __m128) -> ()` |
| `_mm_storeu_si128` | `fn _mm_storeu_si128<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) -> ()` |
| `_mm_storeu_si16` | `fn _mm_storeu_si16<T: Is16BitsUnaligned>(mem_addr: &mut T, a: __m128i) -> ()` |
| `_mm_storeu_si32` | `fn _mm_storeu_si32<T: Is32BitsUnaligned>(mem_addr: &mut T, a: __m128i) -> ()` |
| `_mm_storeu_si64` | `fn _mm_storeu_si64<T: Is64BitsUnaligned>(mem_addr: &mut T, a: __m128i) -> ()` |


## All Intrinsics

### Stable, Safe (277 intrinsics)

| Name | Description | Instruction | Timing (H/Z4) |
|------|-------------|-------------|---------------|
| `_MM_TRANSPOSE4_PS` | Transpose the 4x4 matrix formed by 4 rows of __m128 in place |  | — |
| `_mm_add_epi16` | Adds packed 16-bit integers in `a` and `b` | paddw | 1/1, 1/1 |
| `_mm_add_epi32` | Adds packed 32-bit integers in `a` and `b` | paddd | 1/1, 1/1 |
| `_mm_add_epi64` | Adds packed 64-bit integers in `a` and `b` | paddq | 1/1, 1/1 |
| `_mm_add_epi8` | Adds packed 8-bit integers in `a` and `b` | paddb | 1/1, 1/1 |
| `_mm_add_pd` | Adds packed double-precision (64-bit) floating-point element... | addpd | 3/1, 3/1 |
| `_mm_add_ps` | Adds packed single-precision (32-bit) floating-point element... | addps | 3/1, 3/1 |
| `_mm_add_sd` | Returns a new vector with the low element of `a` replaced by... | addsd | 3/1, 3/1 |
| `_mm_add_ss` | Adds the first component of `a` and `b`, the other component... | addss | 3/1, 3/1 |
| `_mm_adds_epi16` | Adds packed 16-bit integers in `a` and `b` using saturation | paddsw | 1/1, 1/1 |
| `_mm_adds_epi8` | Adds packed 8-bit integers in `a` and `b` using saturation | paddsb | 1/1, 1/1 |
| `_mm_adds_epu16` | Adds packed unsigned 16-bit integers in `a` and `b` using sa... | paddusw | — |
| `_mm_adds_epu8` | Adds packed unsigned 8-bit integers in `a` and `b` using sat... | paddusb | — |
| `_mm_and_pd` | Computes the bitwise AND of packed double-precision (64-bit)... | andps | 1/1, 1/1 |
| `_mm_and_ps` | Bitwise AND of packed single-precision (32-bit) floating-poi... | andps | 1/1, 1/1 |
| `_mm_and_si128` | Computes the bitwise AND of 128 bits (representing integer d... | andps | 1/1, 1/1 |
| `_mm_andnot_pd` | Computes the bitwise NOT of `a` and then AND with `b` | andnps | 1/1, 1/1 |
| `_mm_andnot_ps` | Bitwise AND-NOT of packed single-precision (32-bit) floating... | andnps | 1/1, 1/1 |
| `_mm_andnot_si128` | Computes the bitwise NOT of 128 bits (representing integer d... | andnps | 1/1, 1/1 |
| `_mm_avg_epu16` | Averages packed unsigned 16-bit integers in `a` and `b` | pavgw | — |
| `_mm_avg_epu8` | Averages packed unsigned 8-bit integers in `a` and `b` | pavgb | — |
| `_mm_bslli_si128` | Shifts `a` left by `IMM8` bytes while shifting in zeros | pslldq | — |
| `_mm_bsrli_si128` | Shifts `a` right by `IMM8` bytes while shifting in zeros | psrldq | — |
| `_mm_castpd_ps` | Casts a 128-bit floating-point vector of ` |  | — |
| `_mm_castpd_si128` | Casts a 128-bit floating-point vector of ` |  | — |
| `_mm_castps_pd` | Casts a 128-bit floating-point vector of ` |  | — |
| `_mm_castps_si128` | Casts a 128-bit floating-point vector of ` |  | — |
| `_mm_castsi128_pd` | Casts a 128-bit integer vector into a 128-bit floating-point... |  | — |
| `_mm_castsi128_ps` | Casts a 128-bit integer vector into a 128-bit floating-point... |  | — |
| `_mm_cmpeq_epi16` | Compares packed 16-bit integers in `a` and `b` for equality | pcmpeqw | 1/1, 1/1 |
| `_mm_cmpeq_epi32` | Compares packed 32-bit integers in `a` and `b` for equality | pcmpeqd | 1/1, 1/1 |
| `_mm_cmpeq_epi8` | Compares packed 8-bit integers in `a` and `b` for equality | pcmpeqb | 1/1, 1/1 |
| `_mm_cmpeq_pd` | Compares corresponding elements in `a` and `b` for equality | cmpeqpd | 3/1, 3/1 |
| `_mm_cmpeq_ps` | Compares each of the four floats in `a` to the corresponding... | cmpeqps | 3/1, 3/1 |
| `_mm_cmpeq_sd` | Returns a new vector with the low element of `a` replaced by... | cmpeqsd | — |
| `_mm_cmpeq_ss` | Compares the lowest `f32` of both inputs for equality. The l... | cmpeqss | — |
| `_mm_cmpge_pd` | Compares corresponding elements in `a` and `b` for greater-t... | cmplepd | — |
| `_mm_cmpge_ps` | Compares each of the four floats in `a` to the corresponding... | cmpleps | — |
| `_mm_cmpge_sd` | Returns a new vector with the low element of `a` replaced by... | cmplesd | — |
| `_mm_cmpge_ss` | Compares the lowest `f32` of both inputs for greater than or... | cmpless | — |
| `_mm_cmpgt_epi16` | Compares packed 16-bit integers in `a` and `b` for greater-t... | pcmpgtw | 1/1, 1/1 |
| `_mm_cmpgt_epi32` | Compares packed 32-bit integers in `a` and `b` for greater-t... | pcmpgtd | 1/1, 1/1 |
| `_mm_cmpgt_epi8` | Compares packed 8-bit integers in `a` and `b` for greater-th... | pcmpgtb | 1/1, 1/1 |
| `_mm_cmpgt_pd` | Compares corresponding elements in `a` and `b` for greater-t... | cmpltpd | 3/1, 3/1 |
| `_mm_cmpgt_ps` | Compares each of the four floats in `a` to the corresponding... | cmpltps | 3/1, 3/1 |
| `_mm_cmpgt_sd` | Returns a new vector with the low element of `a` replaced by... | cmpltsd | — |
| `_mm_cmpgt_ss` | Compares the lowest `f32` of both inputs for greater than. T... | cmpltss | — |
| `_mm_cmple_pd` | Compares corresponding elements in `a` and `b` for less-than... | cmplepd | — |
| `_mm_cmple_ps` | Compares each of the four floats in `a` to the corresponding... | cmpleps | — |
| `_mm_cmple_sd` | Returns a new vector with the low element of `a` replaced by... | cmplesd | — |
| `_mm_cmple_ss` | Compares the lowest `f32` of both inputs for less than or eq... | cmpless | — |
| `_mm_cmplt_epi16` | Compares packed 16-bit integers in `a` and `b` for less-than | pcmpgtw | 1/1, 1/1 |
| `_mm_cmplt_epi32` | Compares packed 32-bit integers in `a` and `b` for less-than | pcmpgtd | 1/1, 1/1 |
| `_mm_cmplt_epi8` | Compares packed 8-bit integers in `a` and `b` for less-than | pcmpgtb | 1/1, 1/1 |
| `_mm_cmplt_pd` | Compares corresponding elements in `a` and `b` for less-than | cmpltpd | 3/1, 3/1 |
| `_mm_cmplt_ps` | Compares each of the four floats in `a` to the corresponding... | cmpltps | 3/1, 3/1 |
| `_mm_cmplt_sd` | Returns a new vector with the low element of `a` replaced by... | cmpltsd | — |
| `_mm_cmplt_ss` | Compares the lowest `f32` of both inputs for less than. The ... | cmpltss | — |
| `_mm_cmpneq_pd` | Compares corresponding elements in `a` and `b` for not-equal | cmpneqpd | 3/1, 3/1 |
| `_mm_cmpneq_ps` | Compares each of the four floats in `a` to the corresponding... | cmpneqps | 3/1, 3/1 |
| `_mm_cmpneq_sd` | Returns a new vector with the low element of `a` replaced by... | cmpneqsd | — |
| `_mm_cmpneq_ss` | Compares the lowest `f32` of both inputs for inequality. The... | cmpneqss | — |
| `_mm_cmpnge_pd` | Compares corresponding elements in `a` and `b` for not-great... | cmpnlepd | — |
| `_mm_cmpnge_ps` | Compares each of the four floats in `a` to the corresponding... | cmpnleps | — |
| `_mm_cmpnge_sd` | Returns a new vector with the low element of `a` replaced by... | cmpnlesd | — |
| `_mm_cmpnge_ss` | Compares the lowest `f32` of both inputs for not-greater-tha... | cmpnless | — |
| `_mm_cmpngt_pd` | Compares corresponding elements in `a` and `b` for not-great... | cmpnltpd | — |
| `_mm_cmpngt_ps` | Compares each of the four floats in `a` to the corresponding... | cmpnltps | — |
| `_mm_cmpngt_sd` | Returns a new vector with the low element of `a` replaced by... | cmpnltsd | — |
| `_mm_cmpngt_ss` | Compares the lowest `f32` of both inputs for not-greater-tha... | cmpnltss | — |
| `_mm_cmpnle_pd` | Compares corresponding elements in `a` and `b` for not-less-... | cmpnlepd | — |
| `_mm_cmpnle_ps` | Compares each of the four floats in `a` to the corresponding... | cmpnleps | — |
| `_mm_cmpnle_sd` | Returns a new vector with the low element of `a` replaced by... | cmpnlesd | — |
| `_mm_cmpnle_ss` | Compares the lowest `f32` of both inputs for not-less-than-o... | cmpnless | — |
| `_mm_cmpnlt_pd` | Compares corresponding elements in `a` and `b` for not-less-... | cmpnltpd | — |
| `_mm_cmpnlt_ps` | Compares each of the four floats in `a` to the corresponding... | cmpnltps | — |
| `_mm_cmpnlt_sd` | Returns a new vector with the low element of `a` replaced by... | cmpnltsd | — |
| `_mm_cmpnlt_ss` | Compares the lowest `f32` of both inputs for not-less-than. ... | cmpnltss | — |
| `_mm_cmpord_pd` | Compares corresponding elements in `a` and `b` to see if nei... | cmpordpd | 3/1, 3/1 |
| `_mm_cmpord_ps` | Compares each of the four floats in `a` to the corresponding... | cmpordps | 3/1, 3/1 |
| `_mm_cmpord_sd` | Returns a new vector with the low element of `a` replaced by... | cmpordsd | — |
| `_mm_cmpord_ss` | Checks if the lowest `f32` of both inputs are ordered. The l... | cmpordss | — |
| `_mm_cmpunord_pd` | Compares corresponding elements in `a` and `b` to see if eit... | cmpunordpd | — |
| `_mm_cmpunord_ps` | Compares each of the four floats in `a` to the corresponding... | cmpunordps | — |
| `_mm_cmpunord_sd` | Returns a new vector with the low element of `a` replaced by... | cmpunordsd | — |
| `_mm_cmpunord_ss` | Checks if the lowest `f32` of both inputs are unordered. The... | cmpunordss | — |
| `_mm_comieq_sd` | Compares the lower element of `a` and `b` for equality | comisd | — |
| `_mm_comieq_ss` | Compares two 32-bit floats from the low-order bits of `a` an... | comiss | — |
| `_mm_comige_sd` | Compares the lower element of `a` and `b` for greater-than-o... | comisd | — |
| `_mm_comige_ss` | Compares two 32-bit floats from the low-order bits of `a` an... | comiss | — |
| `_mm_comigt_sd` | Compares the lower element of `a` and `b` for greater-than | comisd | — |
| `_mm_comigt_ss` | Compares two 32-bit floats from the low-order bits of `a` an... | comiss | — |
| `_mm_comile_sd` | Compares the lower element of `a` and `b` for less-than-or-e... | comisd | — |
| `_mm_comile_ss` | Compares two 32-bit floats from the low-order bits of `a` an... | comiss | — |
| `_mm_comilt_sd` | Compares the lower element of `a` and `b` for less-than | comisd | — |
| `_mm_comilt_ss` | Compares two 32-bit floats from the low-order bits of `a` an... | comiss | — |
| `_mm_comineq_sd` | Compares the lower element of `a` and `b` for not-equal | comisd | — |
| `_mm_comineq_ss` | Compares two 32-bit floats from the low-order bits of `a` an... | comiss | — |
| `_mm_cvt_si2ss` | Alias for | cvtsi2ss | 4/1, 3/1 |
| `_mm_cvt_ss2si` | Alias for | cvtss2si | 4/1, 3/1 |
| `_mm_cvtepi32_pd` | Converts the lower two packed 32-bit integers in `a` to pack... | cvtdq2pd | 4/1, 3/1 |
| `_mm_cvtepi32_ps` | Converts packed 32-bit integers in `a` to packed single-prec... | cvtdq2ps | 4/1, 3/1 |
| `_mm_cvtpd_epi32` | Converts packed double-precision (64-bit) floating-point ele... | cvtpd2dq | 4/1, 3/1 |
| `_mm_cvtpd_ps` | Converts packed double-precision (64-bit) floating-point ele... | cvtpd2ps | 4/1, 3/1 |
| `_mm_cvtps_epi32` | Converts packed single-precision (32-bit) floating-point ele... | cvtps2dq | 4/1, 3/1 |
| `_mm_cvtps_pd` | Converts packed single-precision (32-bit) floating-point ele... | cvtps2pd | 4/1, 3/1 |
| `_mm_cvtsd_f64` | Returns the lower double-precision (64-bit) floating-point e... |  | 4/1, 3/1 |
| `_mm_cvtsd_si32` | Converts the lower double-precision (64-bit) floating-point ... | cvtsd2si | 4/1, 3/1 |
| `_mm_cvtsd_si64` | Converts the lower double-precision (64-bit) floating-point ... | cvtsd2si | 4/1, 3/1 |
| `_mm_cvtsd_si64x` | Alias for `_mm_cvtsd_si64` | cvtsd2si | 4/1, 3/1 |
| `_mm_cvtsd_ss` | Converts the lower double-precision (64-bit) floating-point ... | cvtsd2ss | 4/1, 3/1 |
| `_mm_cvtsi128_si32` | Returns the lowest element of `a` |  | 4/1, 3/1 |
| `_mm_cvtsi128_si64` | Returns the lowest element of `a` | movq | 4/1, 3/1 |
| `_mm_cvtsi128_si64x` | Returns the lowest element of `a` | movq | 4/1, 3/1 |
| `_mm_cvtsi32_sd` | Returns `a` with its lower element replaced by `b` after con... | cvtsi2sd | 4/1, 3/1 |
| `_mm_cvtsi32_si128` | Returns a vector whose lowest element is `a` and all higher ... |  | 4/1, 3/1 |
| `_mm_cvtsi32_ss` | Converts a 32 bit integer to a 32 bit float. The result vect... | cvtsi2ss | 4/1, 3/1 |
| `_mm_cvtsi64_sd` | Returns `a` with its lower element replaced by `b` after con... | cvtsi2sd | 4/1, 3/1 |
| `_mm_cvtsi64_si128` | Returns a vector whose lowest element is `a` and all higher ... | movq | 4/1, 3/1 |
| `_mm_cvtsi64_ss` | Converts a 64 bit integer to a 32 bit float. The result vect... | cvtsi2ss | 4/1, 3/1 |
| `_mm_cvtsi64x_sd` | Returns `a` with its lower element replaced by `b` after con... | cvtsi2sd | 4/1, 3/1 |
| `_mm_cvtsi64x_si128` | Returns a vector whose lowest element is `a` and all higher ... | movq | 4/1, 3/1 |
| `_mm_cvtss_f32` | Extracts the lowest 32 bit float from the input vector |  | 4/1, 3/1 |
| `_mm_cvtss_sd` | Converts the lower single-precision (32-bit) floating-point ... | cvtss2sd | 4/1, 3/1 |
| `_mm_cvtss_si32` | Converts the lowest 32 bit float in the input vector to a 32... | cvtss2si | 4/1, 3/1 |
| `_mm_cvtss_si64` | Converts the lowest 32 bit float in the input vector to a 64... | cvtss2si | 4/1, 3/1 |
| `_mm_cvtt_ss2si` | Alias for | cvttss2si | 4/1, 3/1 |
| `_mm_cvttpd_epi32` | Converts packed double-precision (64-bit) floating-point ele... | cvttpd2dq | 4/1, 3/1 |
| `_mm_cvttps_epi32` | Converts packed single-precision (32-bit) floating-point ele... | cvttps2dq | 4/1, 3/1 |
| `_mm_cvttsd_si32` | Converts the lower double-precision (64-bit) floating-point ... | cvttsd2si | 4/1, 3/1 |
| `_mm_cvttsd_si64` | Converts the lower double-precision (64-bit) floating-point ... | cvttsd2si | 4/1, 3/1 |
| `_mm_cvttsd_si64x` | Alias for `_mm_cvttsd_si64` | cvttsd2si | 4/1, 3/1 |
| `_mm_cvttss_si32` | Converts the lowest 32 bit float in the input vector to a 32... | cvttss2si | 4/1, 3/1 |
| `_mm_cvttss_si64` | Converts the lowest 32 bit float in the input vector to a 64... | cvttss2si | 4/1, 3/1 |
| `_mm_div_pd` | Divide packed double-precision (64-bit) floating-point eleme... | divpd | 20/14, 13/7 |
| `_mm_div_ps` | Divides packed single-precision (32-bit) floating-point elem... | divps | 13/7, 10/4 |
| `_mm_div_sd` | Returns a new vector with the low element of `a` replaced by... | divsd | 20/14, 13/7 |
| `_mm_div_ss` | Divides the first component of `b` by `a`, the other compone... | divss | 13/7, 10/4 |
| `_mm_extract_epi16` | Returns the `imm8` element of `a` | pextrw | — |
| `_mm_insert_epi16` | Returns a new vector where the `imm8` element of `a` is repl... | pinsrw | — |
| `_mm_lfence` | Performs a serializing operation on all load-from-memory ins... | lfence | — |
| `_mm_madd_epi16` | Multiplies and then horizontally add signed 16 bit integers ... | pmaddwd | — |
| `_mm_max_epi16` | Compares packed 16-bit integers in `a` and `b`, and returns ... | pmaxsw | — |
| `_mm_max_epu8` | Compares packed unsigned 8-bit integers in `a` and `b`, and ... | pmaxub | — |
| `_mm_max_pd` | Returns a new vector with the maximum values from correspond... | maxpd | — |
| `_mm_max_ps` | Compares packed single-precision (32-bit) floating-point ele... | maxps | — |
| `_mm_max_sd` | Returns a new vector with the low element of `a` replaced by... | maxsd | — |
| `_mm_max_ss` | Compares the first single-precision (32-bit) floating-point ... | maxss | — |
| `_mm_mfence` | Performs a serializing operation on all load-from-memory and... | mfence | — |
| `_mm_min_epi16` | Compares packed 16-bit integers in `a` and `b`, and returns ... | pminsw | — |
| `_mm_min_epu8` | Compares packed unsigned 8-bit integers in `a` and `b`, and ... | pminub | — |
| `_mm_min_pd` | Returns a new vector with the minimum values from correspond... | minpd | — |
| `_mm_min_ps` | Compares packed single-precision (32-bit) floating-point ele... | minps | — |
| `_mm_min_sd` | Returns a new vector with the low element of `a` replaced by... | minsd | — |
| `_mm_min_ss` | Compares the first single-precision (32-bit) floating-point ... | minss | — |
| `_mm_move_epi64` | Returns a vector where the low element is extracted from `a`... | movq | — |
| `_mm_move_sd` | Constructs a 128-bit floating-point vector of ` | movsd | — |
| `_mm_move_ss` | Returns a `__m128` with the first component from `b` and the... | movss | — |
| `_mm_movehl_ps` | Combine higher half of `a` and `b`. The higher half of `b` o... | movhlps | — |
| `_mm_movelh_ps` | Combine lower half of `a` and `b`. The lower half of `b` occ... | movlhps | — |
| `_mm_movemask_epi8` | Returns a mask of the most significant bit of each element i... | pmovmskb | — |
| `_mm_movemask_pd` | Returns a mask of the most significant bit of each element i... | movmskpd | — |
| `_mm_movemask_ps` | Returns a mask of the most significant bit of each element i... | movmskps | — |
| `_mm_mul_epu32` | Multiplies the low unsigned 32-bit integers from each packed... | pmuludq | 10/2, 4/1 |
| `_mm_mul_pd` | Multiplies packed double-precision (64-bit) floating-point e... | mulpd | 5/1, 3/1 |
| `_mm_mul_ps` | Multiplies packed single-precision (32-bit) floating-point e... | mulps | 5/1, 3/1 |
| `_mm_mul_sd` | Returns a new vector with the low element of `a` replaced by... | mulsd | 5/1, 3/1 |
| `_mm_mul_ss` | Multiplies the first component of `a` and `b`, the other com... | mulss | 5/1, 3/1 |
| `_mm_mulhi_epi16` | Multiplies the packed 16-bit integers in `a` and `b`.  The m... | pmulhw | 5/1, 3/1 |
| `_mm_mulhi_epu16` | Multiplies the packed unsigned 16-bit integers in `a` and `b... | pmulhuw | 5/1, 3/1 |
| `_mm_mullo_epi16` | Multiplies the packed 16-bit integers in `a` and `b`.  The m... | pmullw | 5/1, 3/1 |
| `_mm_or_pd` | Computes the bitwise OR of `a` and `b` | orps | 1/1, 1/1 |
| `_mm_or_ps` | Bitwise OR of packed single-precision (32-bit) floating-poin... | orps | 1/1, 1/1 |
| `_mm_or_si128` | Computes the bitwise OR of 128 bits (representing integer da... | orps | 1/1, 1/1 |
| `_mm_packs_epi16` | Converts packed 16-bit integers from `a` and `b` to packed 8... | packsswb | 1/1, 1/1 |
| `_mm_packs_epi32` | Converts packed 32-bit integers from `a` and `b` to packed 1... | packssdw | 1/1, 1/1 |
| `_mm_packus_epi16` | Converts packed 16-bit integers from `a` and `b` to packed 8... | packuswb | 1/1, 1/1 |
| `_mm_prefetch` | Fetch the cache line that contains address `p` using the giv... | prefetcht0 | — |
| `_mm_rcp_ps` | Returns the approximate reciprocal of packed single-precisio... | rcpps | 5/1, 4/1 |
| `_mm_rcp_ss` | Returns the approximate reciprocal of the first single-preci... | rcpss | 5/1, 4/1 |
| `_mm_rsqrt_ps` | Returns the approximate reciprocal square root of packed sin... | rsqrtps | 5/1, 4/1 |
| `_mm_rsqrt_ss` | Returns the approximate reciprocal square root of the first ... | rsqrtss | 5/1, 4/1 |
| `_mm_sad_epu8` | Sum the absolute differences of packed unsigned 8-bit intege... | psadbw | — |
| `_mm_set1_epi16` | Broadcasts 16-bit integer `a` to all elements |  | — |
| `_mm_set1_epi32` | Broadcasts 32-bit integer `a` to all elements |  | — |
| `_mm_set1_epi64x` | Broadcasts 64-bit integer `a` to all elements |  | — |
| `_mm_set1_epi8` | Broadcasts 8-bit integer `a` to all elements |  | — |
| `_mm_set1_pd` | Broadcasts double-precision (64-bit) floating-point value a ... |  | — |
| `_mm_set1_ps` | Construct a `__m128` with all element set to `a` | shufps | — |
| `_mm_set_epi16` | Sets packed 16-bit integers with the supplied values |  | — |
| `_mm_set_epi32` | Sets packed 32-bit integers with the supplied values |  | — |
| `_mm_set_epi64x` | Sets packed 64-bit integers with the supplied values, from h... |  | — |
| `_mm_set_epi8` | Sets packed 8-bit integers with the supplied values |  | — |
| `_mm_set_pd` | Sets packed double-precision (64-bit) floating-point element... |  | — |
| `_mm_set_pd1` | Broadcasts double-precision (64-bit) floating-point value a ... |  | — |
| `_mm_set_ps` | Construct a `__m128` from four floating point values highest... | unpcklps | — |
| `_mm_set_ps1` | Alias for | shufps | — |
| `_mm_set_sd` | Copies double-precision (64-bit) floating-point element `a` ... |  | — |
| `_mm_set_ss` | Construct a `__m128` with the lowest element set to `a` and ... | movss | — |
| `_mm_setr_epi16` | Sets packed 16-bit integers with the supplied values in reve... |  | — |
| `_mm_setr_epi32` | Sets packed 32-bit integers with the supplied values in reve... |  | — |
| `_mm_setr_epi8` | Sets packed 8-bit integers with the supplied values in rever... |  | — |
| `_mm_setr_pd` | Sets packed double-precision (64-bit) floating-point element... |  | — |
| `_mm_setr_ps` | Construct a `__m128` from four floating point values lowest ... | unpcklps | — |
| `_mm_setzero_pd` | Returns packed double-precision (64-bit) floating-point elem... | xorp | — |
| `_mm_setzero_ps` | Construct a `__m128` with all elements initialized to zero | xorps | — |
| `_mm_setzero_si128` | Returns a vector with all elements set to zero | xorps | — |
| `_mm_sfence` |  | sfence | — |
| `_mm_shuffle_epi32` | Shuffles 32-bit integers in `a` using the control in `IMM8` | pshufd | 1/1, 1/1 |
| `_mm_shuffle_pd` | Constructs a 128-bit floating-point vector of ` | shufps | 1/1, 1/1 |
| `_mm_shuffle_ps` | Shuffles packed single-precision (32-bit) floating-point ele... | shufps | 1/1, 1/1 |
| `_mm_shufflehi_epi16` | Shuffles 16-bit integers in the high 64 bits of `a` using th... | pshufhw | 1/1, 1/1 |
| `_mm_shufflelo_epi16` | Shuffles 16-bit integers in the low 64 bits of `a` using the... | pshuflw | 1/1, 1/1 |
| `_mm_sll_epi16` | Shifts packed 16-bit integers in `a` left by `count` while s... | psllw | 1/1, 1/1 |
| `_mm_sll_epi32` | Shifts packed 32-bit integers in `a` left by `count` while s... | pslld | 1/1, 1/1 |
| `_mm_sll_epi64` | Shifts packed 64-bit integers in `a` left by `count` while s... | psllq | 1/1, 1/1 |
| `_mm_slli_epi16` | Shifts packed 16-bit integers in `a` left by `IMM8` while sh... | psllw | 1/1, 1/1 |
| `_mm_slli_epi32` | Shifts packed 32-bit integers in `a` left by `IMM8` while sh... | pslld | 1/1, 1/1 |
| `_mm_slli_epi64` | Shifts packed 64-bit integers in `a` left by `IMM8` while sh... | psllq | 1/1, 1/1 |
| `_mm_slli_si128` | Shifts `a` left by `IMM8` bytes while shifting in zeros | pslldq | 1/1, 1/1 |
| `_mm_sqrt_pd` | Returns a new vector with the square root of each of the val... | sqrtpd | 16/14, 20/9 |
| `_mm_sqrt_ps` | Returns the square root of packed single-precision (32-bit) ... | sqrtps | 11/7, 14/5 |
| `_mm_sqrt_sd` | Returns a new vector with the low element of `a` replaced by... | sqrtsd | 16/14, 20/9 |
| `_mm_sqrt_ss` | Returns the square root of the first single-precision (32-bi... | sqrtss | 11/7, 14/5 |
| `_mm_sra_epi16` | Shifts packed 16-bit integers in `a` right by `count` while ... | psraw | 1/1, 1/1 |
| `_mm_sra_epi32` | Shifts packed 32-bit integers in `a` right by `count` while ... | psrad | 1/1, 1/1 |
| `_mm_srai_epi16` | Shifts packed 16-bit integers in `a` right by `IMM8` while s... | psraw | 1/1, 1/1 |
| `_mm_srai_epi32` | Shifts packed 32-bit integers in `a` right by `IMM8` while s... | psrad | 1/1, 1/1 |
| `_mm_srl_epi16` | Shifts packed 16-bit integers in `a` right by `count` while ... | psrlw | 1/1, 1/1 |
| `_mm_srl_epi32` | Shifts packed 32-bit integers in `a` right by `count` while ... | psrld | 1/1, 1/1 |
| `_mm_srl_epi64` | Shifts packed 64-bit integers in `a` right by `count` while ... | psrlq | 1/1, 1/1 |
| `_mm_srli_epi16` | Shifts packed 16-bit integers in `a` right by `IMM8` while s... | psrlw | 1/1, 1/1 |
| `_mm_srli_epi32` | Shifts packed 32-bit integers in `a` right by `IMM8` while s... | psrld | 1/1, 1/1 |
| `_mm_srli_epi64` | Shifts packed 64-bit integers in `a` right by `IMM8` while s... | psrlq | 1/1, 1/1 |
| `_mm_srli_si128` | Shifts `a` right by `IMM8` bytes while shifting in zeros | psrldq | 1/1, 1/1 |
| `_mm_sub_epi16` | Subtracts packed 16-bit integers in `b` from packed 16-bit i... | psubw | 1/1, 1/1 |
| `_mm_sub_epi32` | Subtract packed 32-bit integers in `b` from packed 32-bit in... | psubd | 1/1, 1/1 |
| `_mm_sub_epi64` | Subtract packed 64-bit integers in `b` from packed 64-bit in... | psubq | 1/1, 1/1 |
| `_mm_sub_epi8` | Subtracts packed 8-bit integers in `b` from packed 8-bit int... | psubb | 1/1, 1/1 |
| `_mm_sub_pd` | Subtract packed double-precision (64-bit) floating-point ele... | subpd | — |
| `_mm_sub_ps` | Subtracts packed single-precision (32-bit) floating-point el... | subps | — |
| `_mm_sub_sd` | Returns a new vector with the low element of `a` replaced by... | subsd | — |
| `_mm_sub_ss` | Subtracts the first component of `b` from `a`, the other com... | subss | — |
| `_mm_subs_epi16` | Subtract packed 16-bit integers in `b` from packed 16-bit in... | psubsw | 1/1, 1/1 |
| `_mm_subs_epi8` | Subtract packed 8-bit integers in `b` from packed 8-bit inte... | psubsb | 1/1, 1/1 |
| `_mm_subs_epu16` | Subtract packed unsigned 16-bit integers in `b` from packed ... | psubusw | — |
| `_mm_subs_epu8` | Subtract packed unsigned 8-bit integers in `b` from packed u... | psubusb | — |
| `_mm_ucomieq_sd` | Compares the lower element of `a` and `b` for equality | ucomisd | — |
| `_mm_ucomieq_ss` | Compares two 32-bit floats from the low-order bits of `a` an... | ucomiss | — |
| `_mm_ucomige_sd` | Compares the lower element of `a` and `b` for greater-than-o... | ucomisd | — |
| `_mm_ucomige_ss` | Compares two 32-bit floats from the low-order bits of `a` an... | ucomiss | — |
| `_mm_ucomigt_sd` | Compares the lower element of `a` and `b` for greater-than | ucomisd | — |
| `_mm_ucomigt_ss` | Compares two 32-bit floats from the low-order bits of `a` an... | ucomiss | — |
| `_mm_ucomile_sd` | Compares the lower element of `a` and `b` for less-than-or-e... | ucomisd | — |
| `_mm_ucomile_ss` | Compares two 32-bit floats from the low-order bits of `a` an... | ucomiss | — |
| `_mm_ucomilt_sd` | Compares the lower element of `a` and `b` for less-than | ucomisd | — |
| `_mm_ucomilt_ss` | Compares two 32-bit floats from the low-order bits of `a` an... | ucomiss | — |
| `_mm_ucomineq_sd` | Compares the lower element of `a` and `b` for not-equal | ucomisd | — |
| `_mm_ucomineq_ss` | Compares two 32-bit floats from the low-order bits of `a` an... | ucomiss | — |
| `_mm_undefined_pd` | Returns vector of type __m128d with indeterminate elements.w... |  | — |
| `_mm_undefined_ps` | Returns vector of type __m128 with indeterminate elements.wi... |  | — |
| `_mm_undefined_si128` | Returns vector of type __m128i with indeterminate elements.w... |  | — |
| `_mm_unpackhi_epi16` | Unpacks and interleave 16-bit integers from the high half of... | punpckhwd | 1/1, 1/1 |
| `_mm_unpackhi_epi32` | Unpacks and interleave 32-bit integers from the high half of... | unpckhps | 1/1, 1/1 |
| `_mm_unpackhi_epi64` | Unpacks and interleave 64-bit integers from the high half of... | unpckhpd | 1/1, 1/1 |
| `_mm_unpackhi_epi8` | Unpacks and interleave 8-bit integers from the high half of ... | punpckhbw | 1/1, 1/1 |
| `_mm_unpackhi_pd` | The resulting `__m128d` element is composed by the low-order... | unpckhpd | 1/1, 1/1 |
| `_mm_unpackhi_ps` | Unpacks and interleave single-precision (32-bit) floating-po... | unpckhps | 1/1, 1/1 |
| `_mm_unpacklo_epi16` | Unpacks and interleave 16-bit integers from the low half of ... | punpcklwd | 1/1, 1/1 |
| `_mm_unpacklo_epi32` | Unpacks and interleave 32-bit integers from the low half of ... | unpcklps | 1/1, 1/1 |
| `_mm_unpacklo_epi64` | Unpacks and interleave 64-bit integers from the low half of ... | movlhps | 1/1, 1/1 |
| `_mm_unpacklo_epi8` | Unpacks and interleave 8-bit integers from the low half of `... | punpcklbw | 1/1, 1/1 |
| `_mm_unpacklo_pd` | The resulting `__m128d` element is composed by the high-orde... | movlhps | 1/1, 1/1 |
| `_mm_unpacklo_ps` | Unpacks and interleave single-precision (32-bit) floating-po... | unpcklps | 1/1, 1/1 |
| `_mm_xor_pd` | Computes the bitwise XOR of `a` and `b` | xorps | 1/1, 1/1 |
| `_mm_xor_ps` | Bitwise exclusive OR of packed single-precision (32-bit) flo... | xorps | 1/1, 1/1 |
| `_mm_xor_si128` | Computes the bitwise XOR of 128 bits (representing integer d... | xorps | 1/1, 1/1 |

### Stable, Unsafe (57 intrinsics) — use import_intrinsics for safe versions

| Name | Description | Safe Variant |
|------|-------------|--------------|
| `_MM_GET_EXCEPTION_MASK` | See | — |
| `_MM_GET_EXCEPTION_STATE` | See | — |
| `_MM_GET_FLUSH_ZERO_MODE` | See | — |
| `_MM_GET_ROUNDING_MODE` | See | — |
| `_MM_SET_EXCEPTION_MASK` | See | — |
| `_MM_SET_EXCEPTION_STATE` | See | — |
| `_MM_SET_FLUSH_ZERO_MODE` | See | — |
| `_MM_SET_ROUNDING_MODE` | See | — |
| `_mm_clflush` | Invalidates and flushes the cache line that contains `p` fro... | — |
| `_mm_getcsr` | Gets the unsigned 32-bit value of the MXCSR control and stat... | — |
| `_mm_load1_pd` | Loads a double-precision (64-bit) floating-point element fro... | `_mm_load1_pd` (safe via import_intrinsics) |
| `_mm_load1_ps` | Construct a `__m128` by duplicating the value read from `p` ... | `_mm_load1_ps` (safe via import_intrinsics) |
| `_mm_load_pd` | Loads 128-bits (composed of 2 packed double-precision (64-bi... | — |
| `_mm_load_pd1` | Loads a double-precision (64-bit) floating-point element fro... | `_mm_load_pd1` (safe via import_intrinsics) |
| `_mm_load_ps` | Loads four `f32` values from *aligned* memory into a `__m128... | — |
| `_mm_load_ps1` | Alias for | `_mm_load_ps1` (safe via import_intrinsics) |
| `_mm_load_sd` | Loads a 64-bit double-precision value to the low element of ... | `_mm_load_sd` (safe via import_intrinsics) |
| `_mm_load_si128` | Loads 128-bits of integer data from memory into a new vector... | — |
| `_mm_load_ss` | Construct a `__m128` with the lowest element read from `p` a... | `_mm_load_ss` (safe via import_intrinsics) |
| `_mm_loadh_pd` | Loads a double-precision value into the high-order bits of a... | `_mm_loadh_pd` (safe via import_intrinsics) |
| `_mm_loadl_epi64` | Loads 64-bit integer from memory into first element of retur... | `_mm_loadl_epi64` (safe via import_intrinsics) |
| `_mm_loadl_pd` | Loads a double-precision value into the low-order bits of a ... | `_mm_loadl_pd` (safe via import_intrinsics) |
| `_mm_loadr_pd` | Loads 2 double-precision (64-bit) floating-point elements fr... | — |
| `_mm_loadr_ps` | Loads four `f32` values from aligned memory into a `__m128` ... | — |
| `_mm_loadu_pd` | Loads 128-bits (composed of 2 packed double-precision (64-bi... | `_mm_loadu_pd` (safe via import_intrinsics) |
| `_mm_loadu_ps` | Loads four `f32` values from memory into a `__m128`. There a... | `_mm_loadu_ps` (safe via import_intrinsics) |
| `_mm_loadu_si128` | Loads 128-bits of integer data from memory into a new vector... | `_mm_loadu_si128` (safe via import_intrinsics) |
| `_mm_loadu_si16` | Loads unaligned 16-bits of integer data from memory into new... | `_mm_loadu_si16` (safe via import_intrinsics) |
| `_mm_loadu_si32` | Loads unaligned 32-bits of integer data from memory into new... | `_mm_loadu_si32` (safe via import_intrinsics) |
| `_mm_loadu_si64` | Loads unaligned 64-bits of integer data from memory into new... | `_mm_loadu_si64` (safe via import_intrinsics) |
| `_mm_maskmoveu_si128` | Conditionally store 8-bit integer elements from `a` into mem... | — |
| `_mm_setcsr` | Sets the MXCSR register with the 32-bit unsigned integer val... | — |
| `_mm_store1_pd` | Stores the lower double-precision (64-bit) floating-point el... | — |
| `_mm_store1_ps` | Stores the lowest 32 bit float of `a` repeated four times in... | — |
| `_mm_store_pd` | Stores 128-bits (composed of 2 packed double-precision (64-b... | — |
| `_mm_store_pd1` | Stores the lower double-precision (64-bit) floating-point el... | — |
| `_mm_store_ps` | Stores four 32-bit floats into *aligned* memory.  If the poi... | — |
| `_mm_store_ps1` | Alias for | — |
| `_mm_store_sd` | Stores the lower 64 bits of a 128-bit vector of ` | `_mm_store_sd` (safe via import_intrinsics) |
| `_mm_store_si128` | Stores 128-bits of integer data from `a` into memory.  `mem_... | — |
| `_mm_store_ss` | Stores the lowest 32 bit float of `a` into memory.  This int... | `_mm_store_ss` (safe via import_intrinsics) |
| `_mm_storeh_pd` | Stores the upper 64 bits of a 128-bit vector of ` | `_mm_storeh_pd` (safe via import_intrinsics) |
| `_mm_storel_epi64` | Stores the lower 64-bit integer `a` to a memory location.  `... | `_mm_storel_epi64` (safe via import_intrinsics) |
| `_mm_storel_pd` | Stores the lower 64 bits of a 128-bit vector of ` | `_mm_storel_pd` (safe via import_intrinsics) |
| `_mm_storer_pd` | Stores 2 double-precision (64-bit) floating-point elements f... | — |
| `_mm_storer_ps` | Stores four 32-bit floats into *aligned* memory in reverse o... | — |
| `_mm_storeu_pd` | Stores 128-bits (composed of 2 packed double-precision (64-b... | `_mm_storeu_pd` (safe via import_intrinsics) |
| `_mm_storeu_ps` | Stores four 32-bit floats into memory. There are no restrict... | `_mm_storeu_ps` (safe via import_intrinsics) |
| `_mm_storeu_si128` | Stores 128-bits of integer data from `a` into memory.  `mem_... | `_mm_storeu_si128` (safe via import_intrinsics) |
| `_mm_storeu_si16` | Store 16-bit integer from the first element of a into memory... | `_mm_storeu_si16` (safe via import_intrinsics) |
| `_mm_storeu_si32` | Store 32-bit integer from the first element of a into memory... | `_mm_storeu_si32` (safe via import_intrinsics) |
| `_mm_storeu_si64` | Store 64-bit integer from the first element of a into memory... | `_mm_storeu_si64` (safe via import_intrinsics) |
| `_mm_stream_pd` | Stores a 128-bit floating point vector of ` | — |
| `_mm_stream_ps` | Stores `a` into the memory at `mem_addr` using a non-tempora... | — |
| `_mm_stream_si128` | Stores a 128-bit integer vector to a 128-bit aligned memory ... | — |
| `_mm_stream_si32` | Stores a 32-bit integer value in the specified memory locati... | — |
| `_mm_stream_si64` | Stores a 64-bit integer value in the specified memory locati... | — |



# Safe Memory Operations Reference

Safe unaligned load/store operations from `safe_unaligned_simd` v0.2.3,
organized by the archmage token required to use them inside `#[arcane]` functions.

Regenerate: `cargo xtask generate`

## Avx512ModernToken (4 functions)

### `_mm256_maskz_expandloadu_epi8`

```rust
fn _mm256_maskz_expandloadu_epi8<T: Is256BitsUnaligned>(k: __mmask32, mem_addr: &T) -> __m256i
```

Features: `avx512vbmi2,avx512vl`

### `_mm512_maskz_expandloadu_epi8`

```rust
fn _mm512_maskz_expandloadu_epi8<T: Is512BitsUnaligned>(k: __mmask64, mem_addr: &T) -> __m512i
```

Features: `avx512vbmi2`

### `_mm_maskz_expandloadu_epi16`

```rust
fn _mm_maskz_expandloadu_epi16<T: Is128BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m128i
```

Features: `avx512vbmi2,avx512vl`

### `_mm_maskz_expandloadu_epi8`

```rust
fn _mm_maskz_expandloadu_epi8<T: Is128BitsUnaligned>(k: __mmask16, mem_addr: &T) -> __m128i
```

Features: `avx512vbmi2,avx512vl`

## Baseline (always available) (26 functions)

### `_mm_load1_pd`

```rust
fn _mm_load1_pd(mem_addr: &f64) -> __m128d
```

Features: `sse2`

### `_mm_load1_ps`

```rust
fn _mm_load1_ps(mem_addr: &f32) -> __m128
```

Features: `sse`

### `_mm_load_pd1`

```rust
fn _mm_load_pd1(mem_addr: &f64) -> __m128d
```

Features: `sse2`

### `_mm_load_ps1`

```rust
fn _mm_load_ps1(mem_addr: &f32) -> __m128
```

Features: `sse`

### `_mm_load_sd`

```rust
fn _mm_load_sd(mem_addr: &f64) -> __m128d
```

Features: `sse2`

### `_mm_load_ss`

```rust
fn _mm_load_ss(mem_addr: &f32) -> __m128
```

Features: `sse`

### `_mm_loadh_pd`

```rust
fn _mm_loadh_pd(a: __m128d, mem_addr: &f64) -> __m128d
```

Features: `sse2`

### `_mm_loadl_epi64`

```rust
fn _mm_loadl_epi64<T: Is128BitsUnaligned>(mem_addr: &T) -> __m128i
```

Features: `sse2`

### `_mm_loadl_pd`

```rust
fn _mm_loadl_pd(a: __m128d, mem_addr: &f64) -> __m128d
```

Features: `sse2`

### `_mm_loadu_pd`

```rust
fn _mm_loadu_pd(mem_addr: &[f64; 2]) -> __m128d
```

Features: `sse2`

### `_mm_loadu_ps`

```rust
fn _mm_loadu_ps(mem_addr: &[f32; 4]) -> __m128
```

Features: `sse`

### `_mm_loadu_si128`

```rust
fn _mm_loadu_si128<T: Is128BitsUnaligned>(mem_addr: &T) -> __m128i
```

Features: `sse2`

### `_mm_loadu_si16`

```rust
fn _mm_loadu_si16<T: Is16BitsUnaligned>(mem_addr: &T) -> __m128i
```

Features: `sse2`

### `_mm_loadu_si32`

```rust
fn _mm_loadu_si32<T: Is32BitsUnaligned>(mem_addr: &T) -> __m128i
```

Features: `sse2`

### `_mm_loadu_si64`

```rust
fn _mm_loadu_si64<T: Is64BitsUnaligned>(mem_addr: &T) -> __m128i
```

Features: `sse2`

### `_mm_store_sd`

```rust
fn _mm_store_sd(mem_addr: &mut f64, a: __m128d) -> ()
```

Features: `sse2`

### `_mm_store_ss`

```rust
fn _mm_store_ss(mem_addr: &mut f32, a: __m128) -> ()
```

Features: `sse`

### `_mm_storeh_pd`

```rust
fn _mm_storeh_pd(mem_addr: &mut f64, a: __m128d) -> ()
```

Features: `sse2`

### `_mm_storel_epi64`

```rust
fn _mm_storel_epi64<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) -> ()
```

Features: `sse2`

### `_mm_storel_pd`

```rust
fn _mm_storel_pd(mem_addr: &mut f64, a: __m128d) -> ()
```

Features: `sse2`

### `_mm_storeu_pd`

```rust
fn _mm_storeu_pd(mem_addr: &mut [f64; 2], a: __m128d) -> ()
```

Features: `sse2`

### `_mm_storeu_ps`

```rust
fn _mm_storeu_ps(mem_addr: &mut [f32; 4], a: __m128) -> ()
```

Features: `sse`

### `_mm_storeu_si128`

```rust
fn _mm_storeu_si128<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) -> ()
```

Features: `sse2`

### `_mm_storeu_si16`

```rust
fn _mm_storeu_si16<T: Is16BitsUnaligned>(mem_addr: &mut T, a: __m128i) -> ()
```

Features: `sse2`

### `_mm_storeu_si32`

```rust
fn _mm_storeu_si32<T: Is32BitsUnaligned>(mem_addr: &mut T, a: __m128i) -> ()
```

Features: `sse2`

### `_mm_storeu_si64`

```rust
fn _mm_storeu_si64<T: Is64BitsUnaligned>(mem_addr: &mut T, a: __m128i) -> ()
```

Features: `sse2`

## Has256BitSimd / X64V3Token (17 functions)

### `_mm256_broadcast_pd`

```rust
fn _mm256_broadcast_pd(mem_addr: &__m128d) -> __m256d
```

Features: `avx`

### `_mm256_broadcast_ps`

```rust
fn _mm256_broadcast_ps(mem_addr: &__m128) -> __m256
```

Features: `avx`

### `_mm256_broadcast_sd`

```rust
fn _mm256_broadcast_sd(mem_addr: &f64) -> __m256d
```

Features: `avx`

### `_mm256_broadcast_ss`

```rust
fn _mm256_broadcast_ss(mem_addr: &f32) -> __m256
```

Features: `avx`

### `_mm256_loadu2_m128`

```rust
fn _mm256_loadu2_m128(hiaddr: &[f32; 4], loaddr: &[f32; 4]) -> __m256
```

Features: `avx`

### `_mm256_loadu2_m128d`

```rust
fn _mm256_loadu2_m128d(hiaddr: &[f64; 2], loaddr: &[f64; 2]) -> __m256d
```

Features: `avx`

### `_mm256_loadu2_m128i`

```rust
fn _mm256_loadu2_m128i<T: Is128BitsUnaligned>(hiaddr: &T, loaddr: &T) -> __m256i
```

Features: `avx`

### `_mm256_loadu_pd`

```rust
fn _mm256_loadu_pd(mem_addr: &[f64; 4]) -> __m256d
```

Features: `avx`

### `_mm256_loadu_ps`

```rust
fn _mm256_loadu_ps(mem_addr: &[f32; 8]) -> __m256
```

Features: `avx`

### `_mm256_loadu_si256`

```rust
fn _mm256_loadu_si256<T: Is256BitsUnaligned>(mem_addr: &T) -> __m256i
```

Features: `avx`

### `_mm256_storeu2_m128`

```rust
fn _mm256_storeu2_m128(hiaddr: &mut [f32; 4], loaddr: &mut [f32; 4], a: __m256) -> ()
```

Features: `avx`

### `_mm256_storeu2_m128d`

```rust
fn _mm256_storeu2_m128d(hiaddr: &mut [f64; 2], loaddr: &mut [f64; 2], a: __m256d) -> ()
```

Features: `avx`

### `_mm256_storeu2_m128i`

```rust
fn _mm256_storeu2_m128i<T: Is128BitsUnaligned>(hiaddr: &mut T, loaddr: &mut T, a: __m256i) -> ()
```

Features: `avx`

### `_mm256_storeu_pd`

```rust
fn _mm256_storeu_pd(mem_addr: &mut [f64; 4], a: __m256d) -> ()
```

Features: `avx`

### `_mm256_storeu_ps`

```rust
fn _mm256_storeu_ps(mem_addr: &mut [f32; 8], a: __m256) -> ()
```

Features: `avx`

### `_mm256_storeu_si256`

```rust
fn _mm256_storeu_si256<T: Is256BitsUnaligned>(mem_addr: &mut T, a: __m256i) -> ()
```

Features: `avx`

### `_mm_broadcast_ss`

```rust
fn _mm_broadcast_ss(mem_addr: &f32) -> __m128
```

Features: `avx`

## Wasm128Token (17 functions)

### `i16x8_load_extend_i8x8`

Loads eight 8-bit integers and sign extends each one to a 16-bit lane.

```rust
fn i16x8_load_extend_i8x8<T: Is8BytesUnaligned>(t: &T) -> v128
```

Features: `simd128`

### `i16x8_load_extend_u8x8`

Loads eight 8-bit integers and zero extends each one to a 16-bit lane.

```rust
fn i16x8_load_extend_u8x8<T: Is8BytesUnaligned>(t: &T) -> v128
```

Features: `simd128`

### `i32x4_load_extend_i16x4`

Loads four 16-bit integers and sign extends each one to a 32-bit lane.

```rust
fn i32x4_load_extend_i16x4<T: Is8BytesUnaligned>(t: &T) -> v128
```

Features: `simd128`

### `i32x4_load_extend_u16x4`

Loads four 16-bit integers and zero extends each one to a 32-bit lane.

```rust
fn i32x4_load_extend_u16x4<T: Is8BytesUnaligned>(t: &T) -> v128
```

Features: `simd128`

### `i64x2_load_extend_i32x2`

Loads two 32-bit integers and sign extends each one to a 64-bit lane.

```rust
fn i64x2_load_extend_i32x2<T: Is8BytesUnaligned>(t: &T) -> v128
```

Features: `simd128`

### `i64x2_load_extend_u32x2`

Loads two 32-bit integers and zero extends each one to a 64-bit lane.

```rust
fn i64x2_load_extend_u32x2<T: Is8BytesUnaligned>(t: &T) -> v128
```

Features: `simd128`

### `u16x8_load_extend_u8x8`

Loads eight 8-bit integers and zero extends each one to a 16-bit lane.

```rust
fn u16x8_load_extend_u8x8<T: Is8BytesUnaligned>(t: &T) -> v128
```

Features: `simd128`

### `u32x4_load_extend_u16x4`

Loads four 16-bit integers and zero extends each one to a 32-bit lane.

```rust
fn u32x4_load_extend_u16x4<T: Is8BytesUnaligned>(t: &T) -> v128
```

Features: `simd128`

### `u64x2_load_extend_u32x2`

Loads two 32-bit integers and zero extends each one to a 64-bit lane.

```rust
fn u64x2_load_extend_u32x2<T: Is8BytesUnaligned>(t: &T) -> v128
```

Features: `simd128`

### `v128_load`

Loads a `v128` vector from the given heap address.

```rust
fn v128_load<T: Is16BytesUnaligned>(t: &T) -> v128
```

Features: `simd128`

### `v128_load16_splat`

Loads a single element and splats to all lanes of a `v128` vector.

```rust
fn v128_load16_splat<T: Is2BytesUnaligned>(t: &T) -> v128
```

Features: `simd128`

### `v128_load32_splat`

Loads a single element and splats to all lanes of a `v128` vector.

```rust
fn v128_load32_splat<T: Is4BytesUnaligned>(t: &T) -> v128
```

Features: `simd128`

### `v128_load32_zero`

Loads a 32-bit element into the low bits of the vector and sets all other bits to zero.

```rust
fn v128_load32_zero<T: Is4BytesUnaligned>(t: &T) -> v128
```

Features: `simd128`

### `v128_load64_splat`

Loads a single element and splats to all lanes of a `v128` vector.

```rust
fn v128_load64_splat<T: Is8BytesUnaligned>(t: &T) -> v128
```

Features: `simd128`

### `v128_load64_zero`

Loads a 64-bit element into the low bits of the vector and sets all other bits to zero.

```rust
fn v128_load64_zero<T: Is8BytesUnaligned>(t: &T) -> v128
```

Features: `simd128`

### `v128_load8_splat`

Loads a single element and splats to all lanes of a `v128` vector.

```rust
fn v128_load8_splat<T: Is1ByteUnaligned>(t: &T) -> v128
```

Features: `simd128`

### `v128_store`

Stores a `v128` vector to the given heap address.

```rust
fn v128_store<T: Is16BytesUnaligned>(t: &mut T, v: v128) -> ()
```

Features: `simd128`

## X64V4Token (95 functions)

### `_mm256_loadu_epi16`

```rust
fn _mm256_loadu_epi16<T: Is256BitsUnaligned>(mem_addr: &T) -> __m256i
```

Features: `avx512bw,avx512vl`

### `_mm256_loadu_epi32`

```rust
fn _mm256_loadu_epi32<T: Is256BitsUnaligned>(mem_addr: &T) -> __m256i
```

Features: `avx512f,avx512vl`

### `_mm256_loadu_epi64`

```rust
fn _mm256_loadu_epi64<T: Is256BitsUnaligned>(mem_addr: &T) -> __m256i
```

Features: `avx512f,avx512vl`

### `_mm256_loadu_epi8`

```rust
fn _mm256_loadu_epi8<T: Is256BitsUnaligned>(mem_addr: &T) -> __m256i
```

Features: `avx512bw,avx512vl`

### `_mm256_mask_compressstoreu_pd`

```rust
fn _mm256_mask_compressstoreu_pd(base_addr: &mut [f64; 4], k: __mmask8, a: __m256d) -> ()
```

Features: `avx512f,avx512vl`

### `_mm256_mask_compressstoreu_ps`

```rust
fn _mm256_mask_compressstoreu_ps(base_addr: &mut [f32; 8], k: __mmask8, a: __m256) -> ()
```

Features: `avx512f,avx512vl`

### `_mm256_mask_expandloadu_pd`

```rust
fn _mm256_mask_expandloadu_pd(src: __m256d, k: __mmask8, mem_addr: &[f64; 4]) -> __m256d
```

Features: `avx512f,avx512vl`

### `_mm256_mask_expandloadu_ps`

```rust
fn _mm256_mask_expandloadu_ps(src: __m256, k: __mmask8, mem_addr: &[f32; 8]) -> __m256
```

Features: `avx512f,avx512vl`

### `_mm256_mask_loadu_pd`

```rust
fn _mm256_mask_loadu_pd(src: __m256d, k: __mmask8, mem_addr: &[f64; 4]) -> __m256d
```

Features: `avx512f,avx512vl`

### `_mm256_mask_loadu_ps`

```rust
fn _mm256_mask_loadu_ps(src: __m256, k: __mmask8, mem_addr: &[f32; 8]) -> __m256
```

Features: `avx512f,avx512vl`

### `_mm256_mask_storeu_epi16`

```rust
fn _mm256_mask_storeu_epi16<T: Is256BitsUnaligned>(mem_addr: &mut T, k: __mmask16, a: __m256i) -> ()
```

Features: `avx512bw,avx512vl`

### `_mm256_mask_storeu_epi32`

```rust
fn _mm256_mask_storeu_epi32<T: Is256BitsUnaligned>(mem_addr: &mut T, k: __mmask8, a: __m256i) -> ()
```

Features: `avx512f,avx512vl`

### `_mm256_mask_storeu_epi64`

```rust
fn _mm256_mask_storeu_epi64<T: Is256BitsUnaligned>(mem_addr: &mut T, k: __mmask8, a: __m256i) -> ()
```

Features: `avx512f,avx512vl`

### `_mm256_mask_storeu_epi8`

```rust
fn _mm256_mask_storeu_epi8<T: Is256BitsUnaligned>(mem_addr: &mut T, k: __mmask32, a: __m256i) -> ()
```

Features: `avx512bw,avx512vl`

### `_mm256_mask_storeu_pd`

```rust
fn _mm256_mask_storeu_pd(mem_addr: &mut [f64; 4], k: __mmask8, a: __m256d) -> ()
```

Features: `avx512f,avx512vl`

### `_mm256_mask_storeu_ps`

```rust
fn _mm256_mask_storeu_ps(mem_addr: &mut [f32; 8], k: __mmask8, a: __m256) -> ()
```

Features: `avx512f,avx512vl`

### `_mm256_maskz_expandloadu_epi32`

```rust
fn _mm256_maskz_expandloadu_epi32<T: Is256BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m256i
```

Features: `avx512f,avx512vl`

### `_mm256_maskz_expandloadu_epi64`

```rust
fn _mm256_maskz_expandloadu_epi64<T: Is256BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m256i
```

Features: `avx512f,avx512vl`

### `_mm256_maskz_expandloadu_pd`

```rust
fn _mm256_maskz_expandloadu_pd(k: __mmask8, mem_addr: &[f64; 4]) -> __m256d
```

Features: `avx512f,avx512vl`

### `_mm256_maskz_expandloadu_ps`

```rust
fn _mm256_maskz_expandloadu_ps(k: __mmask8, mem_addr: &[f32; 8]) -> __m256
```

Features: `avx512f,avx512vl`

### `_mm256_maskz_loadu_epi16`

```rust
fn _mm256_maskz_loadu_epi16<T: Is256BitsUnaligned>(k: __mmask16, mem_addr: &T) -> __m256i
```

Features: `avx512bw,avx512vl`

### `_mm256_maskz_loadu_epi32`

```rust
fn _mm256_maskz_loadu_epi32<T: Is256BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m256i
```

Features: `avx512f,avx512vl`

### `_mm256_maskz_loadu_epi64`

```rust
fn _mm256_maskz_loadu_epi64<T: Is256BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m256i
```

Features: `avx512f,avx512vl`

### `_mm256_maskz_loadu_epi8`

```rust
fn _mm256_maskz_loadu_epi8<T: Is256BitsUnaligned>(k: __mmask32, mem_addr: &T) -> __m256i
```

Features: `avx512bw,avx512vl`

### `_mm256_maskz_loadu_pd`

```rust
fn _mm256_maskz_loadu_pd(k: __mmask8, mem_addr: &[f64; 4]) -> __m256d
```

Features: `avx512f,avx512vl`

### `_mm256_maskz_loadu_ps`

```rust
fn _mm256_maskz_loadu_ps(k: __mmask8, mem_addr: &[f32; 8]) -> __m256
```

Features: `avx512f,avx512vl`

### `_mm256_storeu_epi16`

```rust
fn _mm256_storeu_epi16<T: Is256BitsUnaligned>(mem_addr: &mut T, a: __m256i) -> ()
```

Features: `avx512bw,avx512vl`

### `_mm256_storeu_epi32`

```rust
fn _mm256_storeu_epi32<T: Is256BitsUnaligned>(mem_addr: &mut T, a: __m256i) -> ()
```

Features: `avx512f,avx512vl`

### `_mm256_storeu_epi64`

```rust
fn _mm256_storeu_epi64<T: Is256BitsUnaligned>(mem_addr: &mut T, a: __m256i) -> ()
```

Features: `avx512f,avx512vl`

### `_mm256_storeu_epi8`

```rust
fn _mm256_storeu_epi8<T: Is256BitsUnaligned>(mem_addr: &mut T, a: __m256i) -> ()
```

Features: `avx512bw,avx512vl`

### `_mm512_loadu_epi16`

```rust
fn _mm512_loadu_epi16<T: Is512BitsUnaligned>(mem_addr: &T) -> __m512i
```

Features: `avx512bw`

### `_mm512_loadu_epi32`

```rust
fn _mm512_loadu_epi32<T: Is512BitsUnaligned>(mem_addr: &T) -> __m512i
```

Features: `avx512f`

### `_mm512_loadu_epi64`

```rust
fn _mm512_loadu_epi64<T: Is512BitsUnaligned>(mem_addr: &T) -> __m512i
```

Features: `avx512f`

### `_mm512_loadu_epi8`

```rust
fn _mm512_loadu_epi8<T: Is512BitsUnaligned>(mem_addr: &T) -> __m512i
```

Features: `avx512bw`

### `_mm512_loadu_pd`

```rust
fn _mm512_loadu_pd(mem_addr: &[f64; 8]) -> __m512d
```

Features: `avx512f`

### `_mm512_loadu_ps`

```rust
fn _mm512_loadu_ps(mem_addr: &[f32; 16]) -> __m512
```

Features: `avx512f`

### `_mm512_loadu_si512`

```rust
fn _mm512_loadu_si512<T: Is512BitsUnaligned>(mem_addr: &T) -> __m512i
```

Features: `avx512f`

### `_mm512_mask_compressstoreu_pd`

```rust
fn _mm512_mask_compressstoreu_pd(base_addr: &mut [f64; 8], k: __mmask8, a: __m512d) -> ()
```

Features: `avx512f`

### `_mm512_mask_compressstoreu_ps`

```rust
fn _mm512_mask_compressstoreu_ps(base_addr: &mut [f32; 16], k: __mmask16, a: __m512) -> ()
```

Features: `avx512f`

### `_mm512_mask_expandloadu_pd`

```rust
fn _mm512_mask_expandloadu_pd(src: __m512d, k: __mmask8, mem_addr: &[f64; 8]) -> __m512d
```

Features: `avx512f`

### `_mm512_mask_expandloadu_ps`

```rust
fn _mm512_mask_expandloadu_ps(src: __m512, k: __mmask16, mem_addr: &[f32; 16]) -> __m512
```

Features: `avx512f`

### `_mm512_mask_loadu_pd`

```rust
fn _mm512_mask_loadu_pd(src: __m512d, k: __mmask8, mem_addr: &[f64; 8]) -> __m512d
```

Features: `avx512f`

### `_mm512_mask_loadu_ps`

```rust
fn _mm512_mask_loadu_ps(src: __m512, k: __mmask16, mem_addr: &[f32; 16]) -> __m512
```

Features: `avx512f`

### `_mm512_mask_storeu_epi16`

```rust
fn _mm512_mask_storeu_epi16<T: Is512BitsUnaligned>(mem_addr: &mut T, k: __mmask32, a: __m512i) -> ()
```

Features: `avx512bw`

### `_mm512_mask_storeu_epi32`

```rust
fn _mm512_mask_storeu_epi32<T: Is512BitsUnaligned>(mem_addr: &mut T, k: __mmask16, a: __m512i) -> ()
```

Features: `avx512f`

### `_mm512_mask_storeu_epi64`

```rust
fn _mm512_mask_storeu_epi64<T: Is512BitsUnaligned>(mem_addr: &mut T, k: __mmask8, a: __m512i) -> ()
```

Features: `avx512f`

### `_mm512_mask_storeu_epi8`

```rust
fn _mm512_mask_storeu_epi8<T: Is512BitsUnaligned>(mem_addr: &mut T, k: __mmask64, a: __m512i) -> ()
```

Features: `avx512bw`

### `_mm512_mask_storeu_pd`

```rust
fn _mm512_mask_storeu_pd(mem_addr: &mut [f64; 8], k: __mmask8, a: __m512d) -> ()
```

Features: `avx512f`

### `_mm512_mask_storeu_ps`

```rust
fn _mm512_mask_storeu_ps(mem_addr: &mut [f32; 16], k: __mmask16, a: __m512) -> ()
```

Features: `avx512f`

### `_mm512_maskz_expandloadu_epi64`

```rust
fn _mm512_maskz_expandloadu_epi64<T: Is512BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m512i
```

Features: `avx512f`

### `_mm512_maskz_expandloadu_pd`

```rust
fn _mm512_maskz_expandloadu_pd(k: __mmask8, mem_addr: &[f64; 8]) -> __m512d
```

Features: `avx512f`

### `_mm512_maskz_expandloadu_ps`

```rust
fn _mm512_maskz_expandloadu_ps(k: __mmask16, mem_addr: &[f32; 16]) -> __m512
```

Features: `avx512f`

### `_mm512_maskz_loadu_epi16`

```rust
fn _mm512_maskz_loadu_epi16<T: Is512BitsUnaligned>(k: __mmask32, mem_addr: &T) -> __m512i
```

Features: `avx512bw`

### `_mm512_maskz_loadu_epi32`

```rust
fn _mm512_maskz_loadu_epi32<T: Is512BitsUnaligned>(k: __mmask16, mem_addr: &T) -> __m512i
```

Features: `avx512f`

### `_mm512_maskz_loadu_epi64`

```rust
fn _mm512_maskz_loadu_epi64<T: Is512BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m512i
```

Features: `avx512f`

### `_mm512_maskz_loadu_epi8`

```rust
fn _mm512_maskz_loadu_epi8<T: Is512BitsUnaligned>(k: __mmask64, mem_addr: &T) -> __m512i
```

Features: `avx512bw`

### `_mm512_maskz_loadu_pd`

```rust
fn _mm512_maskz_loadu_pd(k: __mmask8, mem_addr: &[f64; 8]) -> __m512d
```

Features: `avx512f`

### `_mm512_maskz_loadu_ps`

```rust
fn _mm512_maskz_loadu_ps(k: __mmask16, mem_addr: &[f32; 16]) -> __m512
```

Features: `avx512f`

### `_mm512_storeu_epi16`

```rust
fn _mm512_storeu_epi16<T: Is512BitsUnaligned>(mem_addr: &mut T, a: __m512i) -> ()
```

Features: `avx512bw`

### `_mm512_storeu_epi32`

```rust
fn _mm512_storeu_epi32<T: Is512BitsUnaligned>(mem_addr: &mut T, a: __m512i) -> ()
```

Features: `avx512f`

### `_mm512_storeu_epi64`

```rust
fn _mm512_storeu_epi64<T: Is512BitsUnaligned>(mem_addr: &mut T, a: __m512i) -> ()
```

Features: `avx512f`

### `_mm512_storeu_epi8`

```rust
fn _mm512_storeu_epi8<T: Is512BitsUnaligned>(mem_addr: &mut T, a: __m512i) -> ()
```

Features: `avx512bw`

### `_mm512_storeu_pd`

```rust
fn _mm512_storeu_pd(mem_addr: &mut [f64; 8], a: __m512d) -> ()
```

Features: `avx512f`

### `_mm512_storeu_ps`

```rust
fn _mm512_storeu_ps(mem_addr: &mut [f32; 16], a: __m512) -> ()
```

Features: `avx512f`

### `_mm512_storeu_si512`

```rust
fn _mm512_storeu_si512<T: Is512BitsUnaligned>(mem_addr: &mut T, a: __m512i) -> ()
```

Features: `avx512f`

### `_mm_loadu_epi16`

```rust
fn _mm_loadu_epi16<T: Is128BitsUnaligned>(mem_addr: &T) -> __m128i
```

Features: `avx512bw,avx512vl`

### `_mm_loadu_epi32`

```rust
fn _mm_loadu_epi32<T: Is128BitsUnaligned>(mem_addr: &T) -> __m128i
```

Features: `avx512f,avx512vl`

### `_mm_loadu_epi64`

```rust
fn _mm_loadu_epi64<T: Is128BitsUnaligned>(mem_addr: &T) -> __m128i
```

Features: `avx512f,avx512vl`

### `_mm_loadu_epi8`

```rust
fn _mm_loadu_epi8<T: Is128BitsUnaligned>(mem_addr: &T) -> __m128i
```

Features: `avx512bw,avx512vl`

### `_mm_mask_compressstoreu_pd`

```rust
fn _mm_mask_compressstoreu_pd(base_addr: &mut [f64; 2], k: __mmask8, a: __m128d) -> ()
```

Features: `avx512f,avx512vl`

### `_mm_mask_compressstoreu_ps`

```rust
fn _mm_mask_compressstoreu_ps(base_addr: &mut [f32; 4], k: __mmask8, a: __m128) -> ()
```

Features: `avx512f,avx512vl`

### `_mm_mask_expandloadu_pd`

```rust
fn _mm_mask_expandloadu_pd(src: __m128d, k: __mmask8, mem_addr: &[f64; 2]) -> __m128d
```

Features: `avx512f,avx512vl`

### `_mm_mask_expandloadu_ps`

```rust
fn _mm_mask_expandloadu_ps(src: __m128, k: __mmask8, mem_addr: &[f32; 4]) -> __m128
```

Features: `avx512f,avx512vl`

### `_mm_mask_loadu_pd`

```rust
fn _mm_mask_loadu_pd(src: __m128d, k: __mmask8, mem_addr: &[f64; 2]) -> __m128d
```

Features: `avx512f,avx512vl`

### `_mm_mask_loadu_ps`

```rust
fn _mm_mask_loadu_ps(src: __m128, k: __mmask8, mem_addr: &[f32; 4]) -> __m128
```

Features: `avx512f,avx512vl`

### `_mm_mask_storeu_epi16`

```rust
fn _mm_mask_storeu_epi16<T: Is128BitsUnaligned>(mem_addr: &mut T, k: __mmask8, a: __m128i) -> ()
```

Features: `avx512bw,avx512vl`

### `_mm_mask_storeu_epi32`

```rust
fn _mm_mask_storeu_epi32<T: Is128BitsUnaligned>(mem_addr: &mut T, k: __mmask8, a: __m128i) -> ()
```

Features: `avx512f,avx512vl`

### `_mm_mask_storeu_epi64`

```rust
fn _mm_mask_storeu_epi64<T: Is128BitsUnaligned>(mem_addr: &mut T, k: __mmask8, a: __m128i) -> ()
```

Features: `avx512f,avx512vl`

### `_mm_mask_storeu_epi8`

```rust
fn _mm_mask_storeu_epi8<T: Is128BitsUnaligned>(mem_addr: &mut T, k: __mmask16, a: __m128i) -> ()
```

Features: `avx512bw,avx512vl`

### `_mm_mask_storeu_pd`

```rust
fn _mm_mask_storeu_pd(mem_addr: &mut [f64; 2], k: __mmask8, a: __m128d) -> ()
```

Features: `avx512f,avx512vl`

### `_mm_mask_storeu_ps`

```rust
fn _mm_mask_storeu_ps(mem_addr: &mut [f32; 4], k: __mmask8, a: __m128) -> ()
```

Features: `avx512f,avx512vl`

### `_mm_maskz_expandloadu_epi32`

```rust
fn _mm_maskz_expandloadu_epi32<T: Is128BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m128i
```

Features: `avx512f,avx512vl`

### `_mm_maskz_expandloadu_epi64`

```rust
fn _mm_maskz_expandloadu_epi64<T: Is128BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m128i
```

Features: `avx512f,avx512vl`

### `_mm_maskz_expandloadu_pd`

```rust
fn _mm_maskz_expandloadu_pd(k: __mmask8, mem_addr: &[f64; 2]) -> __m128d
```

Features: `avx512f,avx512vl`

### `_mm_maskz_expandloadu_ps`

```rust
fn _mm_maskz_expandloadu_ps(k: __mmask8, mem_addr: &[f32; 4]) -> __m128
```

Features: `avx512f,avx512vl`

### `_mm_maskz_loadu_epi16`

```rust
fn _mm_maskz_loadu_epi16<T: Is128BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m128i
```

Features: `avx512bw,avx512vl`

### `_mm_maskz_loadu_epi32`

```rust
fn _mm_maskz_loadu_epi32<T: Is128BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m128i
```

Features: `avx512f,avx512vl`

### `_mm_maskz_loadu_epi64`

```rust
fn _mm_maskz_loadu_epi64<T: Is128BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m128i
```

Features: `avx512f,avx512vl`

### `_mm_maskz_loadu_epi8`

```rust
fn _mm_maskz_loadu_epi8<T: Is128BitsUnaligned>(k: __mmask16, mem_addr: &T) -> __m128i
```

Features: `avx512bw,avx512vl`

### `_mm_maskz_loadu_pd`

```rust
fn _mm_maskz_loadu_pd(k: __mmask8, mem_addr: &[f64; 2]) -> __m128d
```

Features: `avx512f,avx512vl`

### `_mm_maskz_loadu_ps`

```rust
fn _mm_maskz_loadu_ps(k: __mmask8, mem_addr: &[f32; 4]) -> __m128
```

Features: `avx512f,avx512vl`

### `_mm_storeu_epi16`

```rust
fn _mm_storeu_epi16<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) -> ()
```

Features: `avx512bw,avx512vl`

### `_mm_storeu_epi32`

```rust
fn _mm_storeu_epi32<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) -> ()
```

Features: `avx512f,avx512vl`

### `_mm_storeu_epi64`

```rust
fn _mm_storeu_epi64<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) -> ()
```

Features: `avx512f,avx512vl`

### `_mm_storeu_epi8`

```rust
fn _mm_storeu_epi8<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) -> ()
```

Features: `avx512bw,avx512vl`


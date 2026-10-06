+++
title = "Gather & Scatter"
weight = 2
+++

A lookup-table kernel can gather through checked Rust indexing while keeping
its arithmetic in a generated SIMD context:

```rust
use archmage::prelude::*;

#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn lookup_impl(token: Token, table: &[f32], indices: &[usize; 8], gain: f32) -> [f32; 8] {
    let values = core::array::from_fn(|lane| table[indices[lane]]);
    let v = f32x8::from_array_t(token, values);
    (v * f32x8::splat_t(token, gain)).to_array()
}

pub fn lookup(table: &[f32], indices: &[usize; 8], gain: f32) -> [f32; 8] {
    incant!(lookup_impl(table, indices, gain), [v3, neon, wasm128, scalar])
}

assert_eq!(lookup(&[2.0, 3.0], &[0, 1, 0, 1, 0, 1, 0, 1], 2.0),
           [4.0, 6.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.0]);
```

Every index is bounds-checked unless the compiler can prove it valid. An invalid
index panics; it cannot read outside the slice. LLVM may use scalar loads or a
hardware gather. This example does not promise a gather instruction or zero
bounds-check cost.

A scatter can likewise use checked indexing. Define duplicate-index behavior
explicitly (for example, the last lane wins) before selecting an implementation.
Native scatter instructions need not share scalar lane-order semantics.

## AVX-512 hardware gather and scatter

With an `X64V4Token`, `u32x16`, `i32x16` and `f32x16` have three bounds-safe
methods. Indices come in a `u32x16`. An `X64V4xToken` converts with `.v4()`.

| Method | Lane `i` | Index out of range |
|---|---|---|
| `T::gather_wrapping(&table, idx)` | `table[idx[i] & (N - 1)]` | wraps: `table` is `&[E; N]`, `N` a power of two |
| `T::gather_or(&table, idx, or)` | `table[idx[i]]` | lane keeps `or[i]` |
| `v.scatter_select(&mut dst, enable, idx)` | `dst[idx[i]] = v[i]` if bit `i` of `enable` is set | write skipped |

None of them panics or touches memory outside the slice. Indices are unsigned,
and indices at or above 2^31 count as out of range because the instructions
take signed 32-bit offsets. Scatter writes lanes in order, so when lanes share
an index the highest one wins.

```rust
fn main() {
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        use archmage::{SimdToken, X64V4Token};
        use magetypes::simd::generic::{f32x16, u32x16};

        if let Some(token) = X64V4Token::summon() {
            // An 8-bit code to f32 table: 256 entries, a power of two.
            let lut: [f32; 256] = core::array::from_fn(|i| i as f32 / 255.0);
            let codes = u32x16::from_array_t(token, core::array::from_fn(|i| i as u32 * 17));
            let v = f32x16::gather_wrapping(&lut, codes);
            assert_eq!(v.to_array()[15], 1.0);

            // A slice gather: lanes past the end keep the fallback.
            let short = [10.0f32, 20.0, 30.0];
            let lanes = u32x16::from_array_t(token, core::array::from_fn(|i| i as u32));
            let g = f32x16::gather_or(&short, lanes, f32x16::splat_t(token, -1.0));
            assert_eq!(g.to_array()[..4], [10.0, 20.0, 30.0, -1.0]);

            // A scatter: only enabled lanes with in-range indices write.
            let mut out = [0.0f32; 2];
            g.scatter_select(&mut out, 0b11, lanes);
            assert_eq!(out, [10.0, 20.0]);
        }
    }
}
```

There is no version for other widths or backends. A hardware gather is not
reliably faster than per-lane scalar loads, and which one wins changes from one
CPU generation to the next, so a portable wrapper would hide that trade-off.
Elsewhere, use checked indexing as shown above. On AVX-512, measure both forms.

### The intrinsics behind them

Each method bounds its indices first, then makes one gather or scatter at
scale 4, the element size:

| Method | Bounds step | Memory access |
|---|---|---|
| `gather_wrapping` | `_mm512_and_si512` with `N - 1` | `_mm512_i32gather_epi32` (`u32x16`, `i32x16`) or `_mm512_i32gather_ps` (`f32x16`) |
| `gather_or` | `_mm512_cmplt_epu32_mask` against `min(len, 2^31)` | `_mm512_mask_i32gather_epi32` or `_mm512_mask_i32gather_ps` |
| `scatter_select` | `enable` AND `_mm512_cmplt_epu32_mask` against `min(len, 2^31)` | `_mm512_mask_i32scatter_epi32` or `_mm512_mask_i32scatter_ps` |

`_mm512_set1_epi32` broadcasts each bound. The entries below quote Intel's
Intrinsics Guide, data version 3.6.9 (2024-07-12), the copy Rust's stdarch
vendors. Intel's `MEM` is addressed in bits, so the `* 8` in the `addr` lines
converts a byte offset: lane `j` accesses the 4 bytes at
`base_addr + SignExtend64(vindex[j]) * scale`. In the masked forms, memory is
touched only inside `IF k[j]`, and the loops run `j` upward, so the highest
scatter lane wins on a shared index. Rust's `core::arch` signatures take
`scale` as a const parameter and rename `base_addr` to `slice`, `vindex` to
`offsets`, `k` to `mask` and a scatter's `a` to `src`.

#### `_mm512_set1_epi32`

[Intel's entry](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_set1_epi32) · `__m512i _mm512_set1_epi32(int a)` · `VPBROADCASTD zmm, r32` · AVX512F

> Broadcast 32-bit integer "a" to all elements of "dst".

```text
FOR j := 0 to 15
    i := j*32
    dst[i+31:i] := a[31:0]
ENDFOR
dst[MAX:512] := 0
```

#### `_mm512_and_si512`

[Intel's entry](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_and_si512) · `__m512i _mm512_and_si512(__m512i a, __m512i b)` · `VPANDD zmm, zmm, zmm` · AVX512F

> Compute the bitwise AND of 512 bits (representing integer data) in "a" and
> "b", and store the result in "dst".

```text
dst[511:0] := (a[511:0] AND b[511:0])
dst[MAX:512] := 0
```

#### `_mm512_cmplt_epu32_mask`

[Intel's entry](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmplt_epu32_mask) · `__mmask16 _mm512_cmplt_epu32_mask(__m512i a, __m512i b)` · `VPCMPUD k, zmm, zmm, imm8` · AVX512F

> Compare packed unsigned 32-bit integers in "a" and "b" for less-than, and
> store the results in mask vector "k".

```text
FOR j := 0 to 15
    i := j*32
    k[j] := ( a[i+31:i] < b[i+31:i] ) ? 1 : 0
ENDFOR
k[MAX:16] := 0
```

#### `_mm512_i32gather_epi32`

[Intel's entry](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_i32gather_epi32) · `__m512i _mm512_i32gather_epi32(__m512i vindex, void const* base_addr, int scale)` · `VPGATHERDD zmm, vm32z` · AVX512F

> Gather 32-bit integers from memory using 32-bit indices. 32-bit elements are
> loaded from addresses starting at "base_addr" and offset by each 32-bit
> element in "vindex" (each index is scaled by the factor in "scale"). Gathered
> elements are merged into "dst". "scale" should be 1, 2, 4 or 8.

```text
FOR j := 0 to 15
    i := j*32
    m := j*32
    addr := base_addr + SignExtend64(vindex[m+31:m]) * ZeroExtend64(scale) * 8
    dst[i+31:i] := MEM[addr+31:addr]
ENDFOR
dst[MAX:512] := 0
```

#### `_mm512_i32gather_ps`

[Intel's entry](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_i32gather_ps) · `__m512 _mm512_i32gather_ps(__m512i vindex, void const* base_addr, int scale)` · `VGATHERDPS zmm, vm32z` · AVX512F

> Gather single-precision (32-bit) floating-point elements from memory using
> 32-bit indices. 32-bit elements are loaded from addresses starting at
> "base_addr" and offset by each 32-bit element in "vindex" (each index is
> scaled by the factor in "scale"). Gathered elements are merged into "dst".
> "scale" should be 1, 2, 4 or 8.

Operation: the same as `_mm512_i32gather_epi32`.

#### `_mm512_mask_i32gather_epi32`

[Intel's entry](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_i32gather_epi32) · `__m512i _mm512_mask_i32gather_epi32(__m512i src, __mmask16 k, __m512i vindex, void const* base_addr, int scale)` · `VPGATHERDD zmm {k}, vm32z` · AVX512F

> Gather 32-bit integers from memory using 32-bit indices. 32-bit elements are
> loaded from addresses starting at "base_addr" and offset by each 32-bit
> element in "vindex" (each index is scaled by the factor in "scale"). Gathered
> elements are merged into "dst" using writemask "k" (elements are copied from
> "src" when the corresponding mask bit is not set). "scale" should be 1, 2, 4
> or 8.

```text
FOR j := 0 to 15
    i := j*32
    m := j*32
    IF k[j]
        addr := base_addr + SignExtend64(vindex[m+31:m]) * ZeroExtend64(scale) * 8
        dst[i+31:i] := MEM[addr+31:addr]
    ELSE
        dst[i+31:i] := src[i+31:i]
    FI
ENDFOR
dst[MAX:512] := 0
```

#### `_mm512_mask_i32gather_ps`

[Intel's entry](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_i32gather_ps) · `__m512 _mm512_mask_i32gather_ps(__m512 src, __mmask16 k, __m512i vindex, void const* base_addr, int scale)` · `VGATHERDPS zmm {k}, vm32z` · AVX512F

> Gather single-precision (32-bit) floating-point elements from memory using
> 32-bit indices. 32-bit elements are loaded from addresses starting at
> "base_addr" and offset by each 32-bit element in "vindex" (each index is
> scaled by the factor in "scale"). Gathered elements are merged into "dst"
> using writemask "k" (elements are copied from "src" when the corresponding
> mask bit is not set). "scale" should be 1, 2, 4 or 8.

Operation: the same as `_mm512_mask_i32gather_epi32`.

#### `_mm512_mask_i32scatter_epi32`

[Intel's entry](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_i32scatter_epi32) · `void _mm512_mask_i32scatter_epi32(void* base_addr, __mmask16 k, __m512i vindex, __m512i a, int scale)` · `VPSCATTERDD vm32z {k}, zmm` · AVX512F

> Scatter 32-bit integers from "a" into memory using 32-bit indices. 32-bit
> elements are stored at addresses starting at "base_addr" and offset by each
> 32-bit element in "vindex" (each index is scaled by the factor in "scale")
> subject to mask "k" (elements are not stored when the corresponding mask bit
> is not set). "scale" should be 1, 2, 4 or 8.

```text
FOR j := 0 to 15
    i := j*32
    m := j*32
    IF k[j]
        addr := base_addr + SignExtend64(vindex[m+31:m]) * ZeroExtend64(scale) * 8
        MEM[addr+31:addr] := a[i+31:i]
    FI
ENDFOR
```

#### `_mm512_mask_i32scatter_ps`

[Intel's entry](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_i32scatter_ps) · `void _mm512_mask_i32scatter_ps(void* base_addr, __mmask16 k, __m512i vindex, __m512 a, int scale)` · `VSCATTERDPS vm32z {k}, zmm` · AVX512F

> Scatter single-precision (32-bit) floating-point elements from "a" into memory
> using 32-bit indices. 32-bit elements are stored at addresses starting at
> "base_addr" and offset by each 32-bit element in "vindex" (each index is
> scaled by the factor in "scale") subject to mask "k" (elements are not stored
> when the corresponding mask bit is not set). "scale" should be 1, 2, 4 or 8.

Operation: the same as `_mm512_mask_i32scatter_epi32`.

## Prefetch and layout

The generic API has no prefetch method. Prefetch distance is CPU- and
workload-dependent; fixed cycle estimates are not a portable tuning rule.
Start by comparing the actual indexed kernel with a contiguous or transposed
layout. A future safe prefetch API should accept a reference or slice position
and document architecture-specific hint handling.

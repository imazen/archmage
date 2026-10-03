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

## Prefetch and layout

The generic API has no prefetch method. Prefetch distance is CPU- and
workload-dependent; fixed cycle estimates are not a portable tuning rule.
Start by comparing the actual indexed kernel with a contiguous or transposed
layout. A future safe prefetch API should accept a reference or slice position
and document architecture-specific hint handling.

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
    let v = f32x8::from_array(token, values);
    (v * f32x8::splat(token, gain)).to_array()
}

pub fn lookup(table: &[f32], indices: &[usize; 8], gain: f32) -> [f32; 8] {
    incant!(lookup_impl(table, indices, gain), [v3, neon, wasm128, scalar])
}
```

Every index is bounds-checked unless the compiler can prove it valid. An invalid
index panics; it cannot read outside the slice. LLVM may use scalar loads or a
hardware gather. This example does not promise a gather instruction or zero
bounds-check cost.

A scatter can likewise use checked indexing. Define duplicate-index behavior
explicitly (for example, the last lane wins) before selecting an implementation.
Native scatter instructions need not share scalar lane-order semantics.

## Remaining API gap

Magetypes has no portable checked gather/scatter method. The installed
`safe_unaligned_simd` memory reexports do not supply the AVX2 gather used in the
old example either. That example accepted unchecked indices behind a safe
signature, so it has been removed. A future gather should accept a slice and
checked indices, and specify signed offsets, masks, scale, and out-of-range
behavior. A reusable validated index object may amortize checks over repeated
lookups; justify it with a real consumer and codegen measurements first.

## Prefetch and layout

The generic API has no prefetch method. Prefetch distance is CPU- and
workload-dependent; fixed cycle estimates are not a portable tuning rule.
Start by comparing the actual indexed kernel with a contiguous or transposed
layout. A future safe prefetch API should accept a reference or slice position
and document architecture-specific hint handling.

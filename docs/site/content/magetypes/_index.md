+++
title = "Magetypes"
description = "SIMD vector types with natural Rust operators"
sort_by = "weight"
weight = 2

[extra]
sidebar = true
+++

magetypes provides SIMD vector types with ordinary Rust operators. You write a
kernel once, and it compiles for AVX2, AVX-512, NEON, WASM SIMD128 and scalar.

This kernel multiplies a buffer by a gain, using AVX2, NEON or WASM SIMD where
available:

```rust
#![forbid(unsafe_code)]
use archmage::prelude::*;

#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn gain_impl(token: Token, plane: &mut [f32], gain: f32) {
    let factor = f32x8::splat_t(token, gain);
    let (chunks, tail) = f32x8::partition_slice_mut_t(token, plane);
    for chunk in chunks {
        (f32x8::load_t(token, chunk) * factor).store(chunk);
    }
    for value in tail {
        *value *= gain;
    }
}

pub fn apply_gain(plane: &mut [f32], gain: f32) {
    incant!(gain_impl(plane, gain), [v3, neon, wasm128, scalar])
}

let mut data = [2.0; 11];
apply_gain(&mut data, 0.5);
assert_eq!(data, [1.0; 11]);
```

`#[magetypes]` compiles `gain_impl` once per tier in its list, and `incant!`
runs the best one the CPU supports. Call it around your loop, as here, not
inside it. The kernel is adapted from `zenfilters`:
[Reusable generic kernels](@/magetypes/examples/generic-kernels.md) links the
production source.

## Continue with the production patterns

1. [Type and const generics](@/magetypes/dispatch/types-and-dispatch.md): pixel types, mode specialization, and turbofish dispatch.
2. [Reusable backend helpers](@/magetypes/examples/pixel-blending.md): the complete zenblend SrcOver chain.
3. [Memory, bounds, and tails](@/magetypes/memory/load-store.md): array references and row/strip contracts.
4. [Direct intrinsic helpers](@/archmage/concepts/rite.md): a feature-enabled entry and matched helper.
5. [ISA quirks and fixups](@/magetypes/isa-quirks.md): the numerical contract and its costs.
6. [Production coverage](@/magetypes/examples/coverage.md): which patterns the zen crates actually exercise.

Archmage provides tokens, feature contexts, and dispatch. Magetypes provides
vectors such as [`f32x4<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x4.html) and [`f32x8<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html). Generics and `define(...)` compose with
`#[magetypes]`; neither is a substitute for a feature-enabled entry. Inspect
optimized code and benchmark the complete loop before claiming zero overhead.

[Installation](@/archmage/getting-started/installation.md) ·
[Intrinsics browser](https://imazen.github.io/archmage/intrinsics/) ·
[API reference](https://docs.rs/magetypes/latest/magetypes/)

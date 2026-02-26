+++
title = "Magetypes"
description = "Experimental SIMD vector types with natural Rust operators"
sort_by = "weight"
weight = 2

[extra]
sidebar = true
+++

Magetypes provides SIMD vector types — `f32x8`, `i32x4`, `u8x16`, and friends — with natural Rust operators. Instead of writing raw intrinsics, you write `a + b`, `v * v`, `x.reduce_add()`.

**Status: Experimental.** The API is usable and tested across x86-64, AArch64, and WASM, but it may change between minor versions. Pin your dependency version if stability matters.

## Relationship to Archmage

Magetypes depends on [archmage](@/archmage/_index.md) for capability tokens. You cannot construct a magetypes vector without first proving that the CPU supports the required features — this is what "token-gated construction" means.

The types are generic over a backend token `T`. Write one function, get a correct implementation on every architecture:

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Backend,
};

fn dot_product<T: F32x8Backend>(token: T, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let va = f32x8::<T>::load(token, a);
    let vb = f32x8::<T>::load(token, b);

    // Natural operators — no intrinsics, no unsafe
    let product = va * vb;
    product.reduce_add()
}
```

To call this from concrete code, summon a token and pass it in:

```rust
use archmage::{Desktop64, SimdToken};

fn main() {
    // Prove CPU supports AVX2+FMA — returns None on unsupported hardware
    if let Some(token) = Desktop64::summon() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];
        let b = [2.0f32; 8];
        let result = dot_product(token, &a, &b);
        println!("dot: {}", result);  // 72.0
    }
}
```

Every constructor (`from_array`, `splat`, `zero`, `load`) takes a token as its first argument. If you have the vector, the CPU can run the operations on it.

## Cross-Platform Polyfills

Types wider than the hardware's native register width work everywhere via polyfills. An `f32x8` on AArch64 (which has 128-bit NEON registers) is implemented internally as two `f32x4` operations. The API is identical — you pick the size that fits your algorithm, and magetypes handles the rest. See [Polyfills](@/magetypes/cross-platform/polyfills.md) for details.

## What's Here

- [Getting Started](@/magetypes/getting-started/_index.md) — Installation and your first types
- [Types](@/magetypes/types/_index.md) — Available types per platform, properties, feature flags
- [Operations](@/magetypes/operations/_index.md) — Construction, arithmetic, reductions, bitwise
- [Conversions](@/magetypes/conversions/_index.md) — Float/int, width, bitcast, slice casting
- [Math](@/magetypes/math/_index.md) — Transcendentals, precision levels, approximations
- [Memory](@/magetypes/memory/_index.md) — Load/store, gather/scatter, interleaved data, chunked processing
- [Cross-Platform](@/magetypes/cross-platform/_index.md) — Polyfill strategy, known behavioral differences
- [Dispatch](@/magetypes/dispatch/_index.md) — Using magetypes with `incant!` and `#[magetypes]`

+++
title = "Your First SIMD Function"
weight = 2
+++

Start with a real image-plane kernel adapted from `zenfilters`. The
[production call chain and source revision](@/magetypes/examples/generic-kernels.md)
are documented alongside the standalone version. Add both `archmage` and
`magetypes`; their default features support this example.

```rust
#![forbid(unsafe_code)]
use archmage::prelude::*;

#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn gain_impl(token: Token, plane: &mut [f32], gain: f32) {
    let factor = f32x8::splat(token, gain);
    let (chunks, tail) = f32x8::partition_slice_mut(token, plane);
    for chunk in chunks {
        (f32x8::load(token, chunk) * factor).store(chunk);
    }
    for value in tail {
        *value *= gain;
    }
}

pub fn apply_gain(plane: &mut [f32], gain: f32) {
    incant!(gain_impl(plane, gain), [v3, neon, wasm128, scalar])
}

```

The full chain is `apply_gain` → `incant!` → a generated `gain_impl_<tier>` →
array-chunk loads, multiplication, and stores → scalar tail. The macro's
`define(f32x8)` binds a generic vector to each selected token, enabling the right
instructions inside the loop. Dispatch occurs outside the loop.

For reuse across kernels, see the complete
[zenblend generic-helper chain](@/magetypes/examples/pixel-blending.md).
`#[inline(always)]` can help that helper inline, but it cannot replace the
`#[magetypes]` entry context. For direct intrinsics, use `#[arcane(import_intrinsics)]`
at the boundary and `#[rite(import_intrinsics)]` for matched helpers; the
[intrinsics browser](https://imazen.github.io/archmage/intrinsics/) lists the
reference-based memory operations.

A token is proof of the required features, not a runtime allocation. A fixed
lane count is a logical shape: NEON/WASM split `f32x8` into two native vectors.
Normal Rust array alignment is sufficient. [ISA quirks and fixups](@/magetypes/isa-quirks.md)
explain the contracts that are portable and the floating-point differences.

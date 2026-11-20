+++
title = "Your First SIMD Function"
weight = 2
aliases = ["magetypes/getting-started/first-types/"]
+++

Start with a real image-plane kernel adapted from `zenfilters`. The
[production call chain and source revision](@/magetypes/examples/generic-kernels.md)
are documented alongside the standalone version. Add both `archmage` and
`magetypes`; their default features support this example.

```rust
#![forbid(unsafe_code)]
use archmage::prelude::*;

#[attune(_*_t, define(f32x8))]
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
    attuned!(gain_impl(plane, gain))
}

let mut data = [2.0; 11];
apply_gain(&mut data, 0.5);
assert_eq!(data, [1.0; 11]);
```

The full chain is `apply_gain` → `attuned!` → a generated `gain_impl_<tier>_t` →
array-chunk loads, multiplication, and stores → scalar tail. `define(f32x8)`
binds a generic vector to each selected token (`v3`, `neon`, `wasm128`,
`scalar`), enabling the right instructions inside the loop. Dispatch occurs
outside the loop. When you do not need separate `_t` entry functions,
`#[attune(dispatch, define(f32x8))]` with `let token = Token::from_context();`
generates the public dispatcher directly.

For reuse across kernels, see the complete
[zenblend generic-helper chain](@/magetypes/examples/pixel-blending.md).
`#[inline(always)]` can help that helper inline, but it cannot replace the
`#[attune]` entry context. For direct intrinsics, use `#[attune(wrap, import_intrinsics)]`
at the boundary and `#[attune(v3, import_intrinsics)]` for matched helpers; the
[intrinsics browser](https://imazen.github.io/archmage/intrinsics/) lists the
reference-based memory operations.

A token is proof of the required features, not a runtime allocation. A fixed
lane count is a logical shape: NEON/WASM split [`f32x8`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html) into two native vectors.
Normal Rust array alignment is sufficient. [ISA quirks and fixups](@/magetypes/isa-quirks.md)
explain the contracts that are portable and the floating-point differences.

+++
title = "Construction & Extraction"
weight = 1
+++

Construct vectors inside a generated feature context. `define(f32x8)` makes the
body-local spelling concise; explicit [`f32x8::<Token>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html) works equally well.

This API exercise uses the load/multiply/store operations in the
[zenfilters gain chain](@/magetypes/examples/generic-kernels.md), reduced to one
array so the construction and extraction calls are visible.

```rust
use archmage::prelude::*;
#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn double_impl(token: Token, input: &[f32; 8]) -> [f32; 8] {
    let v = f32x8::load_t(token, input);
    (v * f32x8::splat_t(token, 2.0)).to_array()
}
pub fn double(input: &[f32; 8]) -> [f32; 8] {
    incant!(double_impl(input), [v3, neon, wasm128, scalar])
}
assert_eq!(double(&[3.0; 8]), [6.0; 8]);
```

| Call | Contract |
|---|---|
| `f32x8::zero_t(token)` | All lanes zero |
| `f32x8::splat_t(token, value)` | Repeat a scalar |
| `f32x8::from_array_t(token, values)` | Consume a fixed-size array |
| `f32x8::load_t(token, &values)` | Load a fixed-size array reference |
| `f32x8::from_slice_t(token, values)` | Load the first vector; requires enough elements |
| `v.to_array()` | Return all scalar lanes by value |
| `v.store(&mut values)` | Store to a fixed-size array reference |
| `v[index]` | Scalar lane access with Rust bounds checking |

Type inference often supplies `T`; turbofish is needed only when inference is
insufficient. Fixed arrays carry the vector length in the type. Slice lengths
and dynamic lane indices still need checks unless optimization proves them.
See [memory and bounds](@/magetypes/memory/load-store.md).

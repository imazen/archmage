+++
title = "Load & Store"
weight = 1
+++

Process ordinary array references inside a generated target-feature context.
The image-plane example handles every length, including the scalar tail:

```rust
use archmage::prelude::*;

#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn square_impl(token: Token, plane: &mut [f32]) {
    let (chunks, tail) = f32x8::partition_slice_mut(token, plane);
    for chunk in chunks {
        let v = f32x8::load(token, chunk);
        (v * v).store(chunk);
    }
    for value in tail { *value *= *value; }
}

pub fn square(plane: &mut [f32]) {
    incant!(square_impl(plane), [v3, neon, wasm128, scalar])
}

let mut values = [3.0; 11];
square(&mut values);
assert_eq!(values, [9.0; 11]);
```

## Exact calls and checks

| Call | Contract | Runtime work |
|---|---|---|
| `f32x8::<T>::load(token, array_ref)` | `&[f32; 8]` guarantees eight initialized elements | Unaligned vector load or the backend's smaller loads; no slice-length check |
| `f32x8::<T>::from_array(token, array)` | Owned `[f32; 8]` | Value construction; copies can fold into the caller |
| `f32x8::<T>::from_slice(token, slice)` | At least eight elements | Length check unless proved by the compiler |
| `v.store(array_mut_ref)` | `&mut [f32; 8]` guarantees extent and exclusive access | Unaligned store or smaller stores |
| `v.to_array()` | Owned `[f32; 8]` result | Value extraction; stores can fold into the caller |
| `f32x8::<T>::partition_slice_mut(token, slice)` | Array chunks and a scalar tail | Computes chunk/tail extents once; no vector-alignment requirement |

The generic API does not provide `load_aligned`, `store_aligned`, or `stream`
methods. Earlier versions of this page incorrectly advertised them. Do not
substitute a raw pointer cast for these reference-based calls.

For reusable `T: F32x8Backend` helpers, see
[generic kernels](@/magetypes/examples/generic-kernels.md). Such helpers inline
into the `#[magetypes]` context; an inline attribute alone does not enable ISA
features.

## Alignment and non-temporal stores

The reference APIs require normal Rust element alignment, not 32-byte SIMD
alignment. Cache-line crossings and the working set can still affect speed;
measure the complete loop before changing storage layout.

There is no portable non-temporal-store abstraction here. A useful future API
would need to encode address alignment, valid writable extent, and the ordering
or completion fence before subsequent access. Merely wrapping a streaming
intrinsic in a safe function would not establish those obligations.

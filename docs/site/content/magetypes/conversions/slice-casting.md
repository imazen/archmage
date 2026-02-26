+++
title = "Slice Casting"
weight = 4
+++

Magetypes provides safe, token-gated slice casting as an alternative to `bytemuck`. These methods reinterpret scalar slices as SIMD vector slices (and vice versa) without copying data.

## Cast Scalar Slices to Vector Slices

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Backend,
};

#[inline(always)]
fn cast_examples<T: F32x8Backend>(token: T) {
    // View &[f32] as &[f32x8] (zero-copy)
    let data: &[f32] = &[1.0; 64];
    if let Some(chunks) = f32x8::<T>::cast_slice(token, data) {
        // chunks: &[f32x8<T>] with 8 elements (64 / 8 = 8)
        for chunk in chunks {
            let sum = chunk.reduce_add();
        }
    }

    // Mutable version
    let data: &mut [f32] = &mut [0.0; 64];
    if let Some(chunks) = f32x8::<T>::cast_slice_mut(token, data) {
        // chunks: &mut [f32x8<T>]
    }
}
```

`cast_slice` returns `None` if the slice length isn't a multiple of the vector width or if alignment is wrong. No UB possible.

## Byte-Level Access

View a vector's raw bytes. These don't need a token — you already have the vector, which proves CPU support:

```rust
// given token: T where T: F32x8Backend
let v = f32x8::<T>::splat(token, 1.0);

// Immutable byte view (zero-cost)
let bytes: &[u8; 32] = v.as_bytes();

// Mutable byte view
let mut v = f32x8::<T>::splat(token, 0.0);
let bytes: &mut [u8; 32] = v.as_bytes_mut();
```

## Create from Bytes

Construct a vector from raw bytes (token-gated):

```rust
// given token: T where T: F32x8Backend
let bytes = [0u8; 32];
let v = f32x8::<T>::from_bytes(token, &bytes);

// Owned version
let v = f32x8::<T>::from_bytes_owned(token, bytes);
```

## Why Not bytemuck?

Implementing bytemuck's `Pod` and `Zeroable` traits would bypass token-gated construction:

```rust
// bytemuck would allow this — no token, no CPU check:
let v: f32x8 = bytemuck::Zeroable::zeroed();  // Bad: no proof of CPU support

// magetypes requires the token:
// given token: T where T: F32x8Backend
let v = f32x8::<T>::zero(token);  // Good: token proves the CPU can handle it
```

The token-gated `cast_slice` and `from_bytes` methods provide the same functionality without compromising the safety model. `cast_slice` returns `None` on alignment or length mismatch, so you get runtime safety checks without `unsafe`.

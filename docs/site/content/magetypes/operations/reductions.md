+++
title = "Reductions"
weight = 3
+++

Reductions collapse all lanes of a vector into a single scalar value.

## Available Reductions

```rust
use magetypes::simd::{generic::f32x8, backends::F32x8Backend};

#[inline(always)]
fn reduction_example<T: F32x8Backend>(token: T) {
    let v = f32x8::<T>::from_array(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let sum = v.reduce_add();  // 36.0 — sum of all lanes
    let max = v.reduce_max();  // 8.0  — maximum lane value
    let min = v.reduce_min();  // 1.0  — minimum lane value
}
```

These work on float types (`f32x4<T>`, `f32x8<T>`, `f64x2<T>`, etc.) and integer types (`i32x4<T>`, `i32x8<T>`, `u32x4<T>`, etc.).

## Boolean Reductions

For integer mask types, check whether any or all lanes are set:

```rust
let mask = a.simd_lt(b);

let any = mask.any_true();  // true if any lane comparison was true
let all = mask.all_true();  // true if every lane comparison was true
let bits = mask.bitmask();  // Bit pattern of which lanes are true
```

`bitmask()` returns an integer where bit N corresponds to lane N. On an 8-lane vector, `bitmask()` returns a `u8`.

## Example: Find Maximum Value in a Large Array

```rust
use archmage::{X64V3Token, SimdToken, arcane};
use magetypes::simd::{generic::f32x8, backends::F32x8Backend};

#[arcane]
fn find_max(token: X64V3Token, data: &[f32]) -> f32 {
    let chunks = data.chunks_exact(8);
    let remainder = chunks.remainder();

    // SIMD reduction over full chunks
    let mut max_v = f32x8::<X64V3Token>::splat(token, f32::NEG_INFINITY);
    for chunk in chunks {
        let v = f32x8::<X64V3Token>::from_slice(token, chunk);
        max_v = max_v.max(v);
    }

    // Reduce vector to scalar
    let mut result = max_v.reduce_max();

    // Handle remainder
    for &x in remainder {
        if x > result {
            result = x;
        }
    }

    result
}
```

This processes 8 floats per iteration in the SIMD loop, then handles any leftover elements with scalar code.

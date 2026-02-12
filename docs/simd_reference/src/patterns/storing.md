# Storing Data

Getting data from SIMD registers back into slices and arrays.

## Quick Reference

| Pattern | Output | Works in `#[arcane]`? | Notes |
|---------|--------|----------------------|-------|
| `v.to_array()` | `[f32; N]` | Yes | Returns owned array |
| `v.store(token, &mut array)` | writes to `&mut [f32; N]` | Yes | Preferred for magetypes |
| `_mm256_storeu_ps(&mut arr, v)` | writes to `&mut [f32; 8]` | Yes | `safe_unaligned_simd` |
| `_mm256_storeu_ps(ptr, v)` | writes to `*mut f32` | unsafe | Raw stdarch |

## Magetypes

```rust
use archmage::{Desktop64, arcane};
use magetypes::simd::f32x8;

#[arcane]
fn double(token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
    let v = f32x8::load(token, data);
    let doubled = v + v;
    doubled.to_array()
}
```

### Store into a mutable reference

```rust
#[arcane]
fn double_in_place(token: Desktop64, data: &mut [f32; 8]) {
    let v = f32x8::load(token, data);
    let doubled = v + v;
    doubled.store(token, data);
}
```

## safe_unaligned_simd

```rust
use std::arch::x86_64::*;

#[arcane]
fn square(_token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
    let v = safe_unaligned_simd::x86_64::_mm256_loadu_ps(data);
    let squared = _mm256_mul_ps(v, v);
    let mut out = [0.0f32; 8];
    safe_unaligned_simd::x86_64::_mm256_storeu_ps(&mut out, squared);
    out
}
```

## Storing to a slice

When you need to write into a slice (not an array), convert the subslice first:

```rust
#[arcane]
fn process_slice(token: Desktop64, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(8) {
        let v = f32x8::from_slice(token, chunk);
        let result = v * v;
        let arr: &mut [f32; 8] = chunk.try_into().unwrap();
        result.store(token, arr);
    }
}
```

## Extracting individual lanes

When you need a single element (not the whole vector):

```rust
#[arcane]
fn first_element(token: Desktop64, data: &[f32; 8]) -> f32 {
    let v = f32x8::load(token, data);
    v.to_array()[0]  // Extract first lane
}
```

For reductions (sum, min, max), use the dedicated methods:

```rust
let sum: f32 = v.reduce_add();
let max: f32 = v.reduce_max();
let min: f32 = v.reduce_min();
```

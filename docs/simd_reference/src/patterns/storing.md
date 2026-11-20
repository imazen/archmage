# Storing Data

Getting data from SIMD registers back into slices and arrays.

## Quick Reference

| Pattern | Output | Works in `#[attune]`? | Notes |
|---------|--------|-----------------------|-------|
| `v.to_array()` | `[f32; N]` | Yes | Returns owned array |
| `v.store(&mut array)` | writes to `&mut [f32; N]` | Yes | Preferred for magetypes (receiver method, no token argument) |
| `_mm256_storeu_ps(&mut arr, v)` | writes to `&mut [f32; 8]` | Yes | Safe via `import_intrinsics` |
| `_mm256_storeu_ps(ptr, v)` | writes to `*mut f32` | unsafe | Raw stdarch |

## Magetypes

```rust
use archmage::{X64V3Token, attune};
use magetypes::simd::generic::f32x8;

#[attune(wrap)]
fn double(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let v = f32x8::load_t(token, data);
    let doubled = v + v;
    doubled.to_array()
}
```

### Store into a mutable reference

```rust
use archmage::{X64V3Token, attune};
use magetypes::simd::generic::f32x8;

#[attune(wrap)]
fn double_in_place(token: X64V3Token, data: &mut [f32; 8]) {
    let v = f32x8::load_t(token, data);
    let doubled = v + v;
    doubled.store(data);
}
```

## Safe memory ops (via `import_intrinsics`)

```rust
use archmage::{X64V3Token, attune};

#[attune(wrap, import_intrinsics)]
fn square(_token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let v = _mm256_loadu_ps(data);
    let squared = _mm256_mul_ps(v, v);
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, squared);
    out
}
```

## Storing to a slice

When you need to write into a slice (not an array), convert the subslice first:

```rust
use archmage::{X64V3Token, attune};
use magetypes::simd::generic::f32x8;

#[attune(wrap)]
fn process_slice(token: X64V3Token, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(8) {
        let v = f32x8::from_slice_t(token, chunk);
        let result = v * v;
        let arr: &mut [f32; 8] = chunk.try_into().unwrap();
        result.store(arr);
    }
}
```

## Extracting individual lanes

When you need a single element (not the whole vector):

```rust
use archmage::{X64V3Token, attune};
use magetypes::simd::generic::f32x8;

#[attune(wrap)]
fn first_element(token: X64V3Token, data: &[f32; 8]) -> f32 {
    let v = f32x8::load_t(token, data);
    v.to_array()[0]  // Extract first lane
}
```

For reductions (sum, min, max), use the dedicated methods:

```rust
let sum: f32 = v.reduce_add();
let max: f32 = v.reduce_max();
let min: f32 = v.reduce_min();
```

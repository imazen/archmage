+++
title = "Approximations"
weight = 3
+++

Fast approximations for reciprocal and reciprocal square root. These are single-instruction operations on most hardware, with ~12 bits of precision out of the box.

## Reciprocal

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Backend,
};

#[inline(always)]
fn example<T: F32x8Backend>(token: T) {
    let v = f32x8::<T>::splat(token, 4.0);
    let rcp = v.rcp_approx();  // ~ [0.25; 8]
}
```

On x86-64, `rcp_approx()` maps to `vrcpps` (one instruction, ~12-bit precision). On ARM, it uses `vrecpeq_f32`. On WASM, it uses division.

For full precision, use `recip()` which applies Newton-Raphson refinement automatically:

```rust
// given token: T where T: F32x8Backend
// let v = f32x8::<T>::splat(token, 4.0);
let precise_rcp = v.recip();  // Full-precision 1/x
```

## Reciprocal Square Root

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Backend,
};

#[inline(always)]
fn example<T: F32x8Backend>(token: T) {
    let v = f32x8::<T>::splat(token, 4.0);
    let rsqrt = v.rsqrt_approx();  // ~ [0.5; 8]
}
```

On x86-64: `vrsqrtps`. On ARM: `vrsqrteq_f32`. Both are single-instruction approximations.

## Newton-Raphson Refinement

The hardware approximations give ~12 bits of precision. If you need more, add one or two Newton-Raphson iterations:

### Refining rsqrt

One iteration roughly doubles the precision (~23 bits — nearly full f32):

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Backend,
};

#[inline(always)]
fn refined_rsqrt<T: F32x8Backend>(token: T, v: f32x8<T>) -> f32x8<T> {
    let approx = v.rsqrt_approx();
    let half = f32x8::<T>::splat(token, 0.5);
    let three_halves = f32x8::<T>::splat(token, 1.5);

    // Newton-Raphson: x' = x * (1.5 - 0.5 * v * x * x)
    approx * (three_halves - half * v * approx * approx)
}
```

This is a classic pattern in SIMD code. The approximate instruction plus one NR iteration is often faster than `sqrt()` followed by division, especially when you need `1/sqrt(x)` directly.

### Refining rcp_approx

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Backend,
};

#[inline(always)]
fn refined_rcp<T: F32x8Backend>(token: T, v: f32x8<T>) -> f32x8<T> {
    let approx = v.rcp_approx();
    let two = f32x8::<T>::splat(token, 2.0);

    // Newton-Raphson: x' = x * (2 - v * x)
    approx * (two - v * approx)
}
```

Or skip the manual refinement and use `recip()`, which does this for you.

## When to Use Approximations

**Use `rcp_approx()` / `rsqrt_approx()` when:**
- You need speed over precision (graphics, physics, audio)
- You'll do Newton-Raphson refinement anyway
- The result is used in a context where ~12 bits is sufficient (e.g., normalization where you'll renormalize later)

**Use `recip()` or `v.sqrt()` when:**
- You need full precision
- The operation is not in a hot inner loop
- Correctness for edge cases (zero, negative, denormals) matters

`sqrt()` is an exact hardware instruction — not an approximation. It's slower than `rsqrt_approx()` + NR but produces correctly rounded results.

## Example: Fast Normalization

Normalize a vector of 3D positions using `rsqrt_approx` + one NR step:

```rust
use archmage::rite;
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Backend,
};

#[rite(import_intrinsics)]
fn fast_normalize<T: F32x8Backend>(
    token: T,
    x: &mut [f32; 8],
    y: &mut [f32; 8],
    z: &mut [f32; 8],
) {
    let vx = f32x8::<T>::from_array(token, *x);
    let vy = f32x8::<T>::from_array(token, *y);
    let vz = f32x8::<T>::from_array(token, *z);

    // length^2
    let len_sq = vx.mul_add(vx, vy.mul_add(vy, vz * vz));

    // 1/length via rsqrt_approx + Newton-Raphson
    let approx = len_sq.rsqrt_approx();
    let half = f32x8::<T>::splat(token, 0.5);
    let three_halves = f32x8::<T>::splat(token, 1.5);
    let inv_len = approx * (three_halves - half * len_sq * approx * approx);

    *x = (vx * inv_len).to_array();
    *y = (vy * inv_len).to_array();
    *z = (vz * inv_len).to_array();
}
```

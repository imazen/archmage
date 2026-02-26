+++
title = "Arithmetic & Comparisons"
weight = 2
+++

Magetypes vectors support standard Rust operators for element-wise arithmetic.

## Arithmetic Operators

```rust
use magetypes::simd::{generic::f32x8, backends::F32x8Backend};

fn arithmetic_example<T: F32x8Backend>(token: T) {
    let a = f32x8::<T>::splat(token, 2.0);
    let b = f32x8::<T>::splat(token, 3.0);

    let sum  = a + b;   // [5.0; 8]
    let diff = a - b;   // [-1.0; 8]
    let prod = a * b;   // [6.0; 8]
    let quot = a / b;   // [0.666...; 8]
    let neg  = -a;      // [-2.0; 8]
}
```

### Compound Assignment

```rust
fn compound_example<T: F32x8Backend>(token: T) {
    let mut v = f32x8::<T>::splat(token, 1.0);
    v += f32x8::<T>::splat(token, 2.0);  // v = [3.0; 8]
    v *= f32x8::<T>::splat(token, 2.0);  // v = [6.0; 8]
}
```

## Fused Multiply-Add

FMA computes `a * b + c` in a single instruction — faster and more precise than separate multiply and add (no intermediate rounding):

```rust
// a * b + c
let result = a.mul_add(b, c);

// a * b - c
let result = a.mul_sub(b, c);
```

On x86-64 with `Desktop64` (AVX2+FMA), these map to single `vfmadd`/`vfmsub` instructions. On ARM NEON, they use `vfmaq_f32`.

## Comparisons

Comparisons return mask types with one boolean per lane:

```rust
use magetypes::simd::{generic::f32x8, backends::F32x8Backend};

fn comparison_example<T: F32x8Backend>(token: T) {
    let a = f32x8::<T>::from_array(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = f32x8::<T>::splat(token, 4.0);

    let lt = a.simd_lt(b);   // [true, true, true, false, false, false, false, false]
    let eq = a.simd_eq(b);   // [false, false, false, true, false, false, false, false]
    let ge = a.simd_ge(b);   // [false, false, false, true, true, true, true, true]
}
```

Available comparisons: `simd_eq`, `simd_ne`, `simd_lt`, `simd_le`, `simd_gt`, `simd_ge`.

### Blend with Mask

Use a comparison result to select between two vectors:

```rust
let mask = a.simd_lt(b);
let result = mask.blend(true_values, false_values);
// Where mask is true: take from true_values
// Where mask is false: take from false_values
```

## Min / Max

```rust
let min = a.min(b);  // Element-wise minimum
let max = a.max(b);  // Element-wise maximum
```

Clamp to a range:

```rust
fn clamp_example<T: F32x8Backend>(token: T, v: f32x8<T>) -> f32x8<T> {
    v.max(f32x8::<T>::splat(token, 0.0))
     .min(f32x8::<T>::splat(token, 1.0))
}
```

## Absolute Value

```rust
let abs = v.abs();  // |v| for each lane
```

## Example: Dot Product

```rust
use archmage::{Desktop64, SimdToken, arcane};
use magetypes::simd::{generic::f32x8, backends::F32x8Backend};

#[arcane]
fn dot_product(token: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let va = f32x8::<Desktop64>::from_array(token, *a);
    let vb = f32x8::<Desktop64>::from_array(token, *b);
    (va * vb).reduce_add()
}
```

## Example: Vector Normalization

```rust
use archmage::{Desktop64, SimdToken, arcane};
use magetypes::simd::{generic::f32x8, backends::F32x8Backend};

#[arcane]
fn normalize(token: Desktop64, v: &mut [f32; 8]) {
    let vec = f32x8::<Desktop64>::from_array(token, *v);
    let len_sq = (vec * vec).reduce_add();
    let len = len_sq.sqrt();

    if len > 0.0 {
        let inv_len = f32x8::<Desktop64>::splat(token, 1.0 / len);
        let normalized = vec * inv_len;
        *v = normalized.to_array();
    }
}
```

+++
title = "Transcendentals"
weight = 1
+++

Magetypes provides SIMD implementations of common math functions. These are polynomial approximations tuned per platform — faster than calling scalar `f32::exp()` in a loop.

## Exponential Functions

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Convert,
};

#[inline(always)]
fn exponentials<T: F32x8Convert>(token: T) {
    let v = f32x8::<T>::splat(token, 3.0);

    // Base-2 exponential: 2^x
    let result = v.exp2_midp();   // [8.0; 8]

    // Natural exponential: e^x
    let result = v.exp_midp();    // [e^3; 8] ~ [20.09; 8]
}
```

## Logarithms

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Convert,
};

#[inline(always)]
fn logarithms<T: F32x8Convert>(token: T) {
    let v = f32x8::<T>::splat(token, 8.0);

    // Base-2 logarithm: log2(x)
    let result = v.log2_midp();   // [3.0; 8]

    // Natural logarithm: ln(x)
    let result = v.ln_midp();     // [ln(8); 8] ~ [2.08; 8]

    // Base-10 logarithm: log10(x)
    let result = v.log10_midp();  // [log10(8); 8] ~ [0.90; 8]
}
```

## Power

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Convert,
};

#[inline(always)]
fn power<T: F32x8Convert>(token: T) {
    let base = f32x8::<T>::splat(token, 2.0);

    // x^n (computed as exp2(n * log2(x)))
    let result = base.pow_midp(3.0);  // [8.0; 8]
}
```

Note: `pow` takes a scalar exponent, not a vector.

## Roots

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Convert,
};

#[inline(always)]
fn roots<T: F32x8Convert>(token: T) {
    let v = f32x8::<T>::splat(token, 9.0);

    // Square root (hardware instruction on all platforms)
    let result = v.sqrt();            // [3.0; 8]

    // Cube root (polynomial approximation)
    let result = v.cbrt_midp();       // [cbrt(9); 8] ~ [2.08; 8]

    // Reciprocal square root: 1/sqrt(x)
    let result = v.rsqrt();           // [1/3; 8] ~ [0.33; 8]
}
```

`sqrt()` maps to a single hardware instruction (`vsqrtps` on x86, `fsqrt` on ARM). It's exact, not an approximation.

## Precision Variants

Most transcendentals come in multiple precision levels. See [Precision Levels](@/magetypes/math/precision.md) for the full breakdown.

```rust
// given token: T where T: F32x8Convert
let v = f32x8::<T>::splat(token, 2.0);

let fast     = v.exp2_lowp();   // ~12-bit precision, fastest
let balanced = v.exp2_midp();   // ~20-bit precision
```

There are no unsuffixed "full precision" transcendentals. `_midp` is the highest precision level for polynomial approximations. For exact results, use `sqrt()` (which is a hardware instruction, not an approximation).

## Domain Errors

Invalid inputs produce NaN or infinity, matching IEEE 754 behavior:

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Convert,
};

#[inline(always)]
fn domain_errors<T: F32x8Convert>(token: T) {
    let v = f32x8::<T>::from_array(token, [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // sqrt of negative -> NaN
    let sqrt = v.sqrt();   // [NaN, 0.0, 1.0, 1.41, 1.73, 2.0, 2.24, 2.45]

    // log of non-positive -> NaN or -inf
    let log = v.ln_midp(); // [NaN, -inf, 0.0, 0.69, 1.10, 1.39, 1.61, 1.79]
}
```

## Example: Gaussian Function

```rust
use archmage::{arcane, SimdToken};
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Convert,
};

#[arcane(import_intrinsics)]
fn gaussian<T: F32x8Convert>(token: T, x: &[f32; 8], sigma: f32) -> [f32; 8] {
    let v = f32x8::<T>::from_array(token, *x);
    let sigma_v = f32x8::<T>::splat(token, sigma);
    let two = f32x8::<T>::splat(token, 2.0);

    // exp(-x^2 / (2 * sigma^2))
    let x_sq = v * v;
    let two_sigma_sq = two * sigma_v * sigma_v;
    let exponent = -(x_sq / two_sigma_sq);
    let result = exponent.exp_midp();  // Good precision, fast

    result.to_array()
}
```

## Example: Softmax

```rust
use archmage::{arcane, SimdToken};
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Convert,
};

#[arcane(import_intrinsics)]
fn softmax<T: F32x8Convert>(token: T, logits: &[f32; 8]) -> [f32; 8] {
    let v = f32x8::<T>::from_array(token, *logits);

    // Subtract max for numerical stability
    let max = v.reduce_max();
    let shifted = v - f32x8::<T>::splat(token, max);

    // exp(x - max)
    let exp = shifted.exp_midp();

    // Normalize
    let sum = exp.reduce_add();
    let result = exp / f32x8::<T>::splat(token, sum);

    result.to_array()
}
```

## Platform Coverage

- **x86-64**: All functions available on `f32x4`, `f32x8`, `f64x2`, `f64x4`
- **AArch64**: Full support via NEON polynomial approximations
- **WASM**: Most functions available; some use scalar fallback internally

The implementations use platform-tuned polynomial coefficients for best accuracy per instruction count.

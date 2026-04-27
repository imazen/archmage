+++
title = "Float / Integer"
weight = 1
+++

Convert between floating-point and integer vector types.

## Float to Integer

### Truncate (toward zero)

Behaves like `as i32` in Rust — drops the fractional part:

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Convert,
};

#[inline(always)]
fn truncate<T: F32x8Convert>(token: T) {
    let floats = f32x8::<T>::from_array(token, [1.5, 2.7, -3.2, 4.0, 5.9, 6.1, 7.0, 8.5]);

    let ints = floats.to_i32x8();
    // [1, 2, -3, 4, 5, 6, 7, 8]
}
```

### Round to nearest

Rounds to the nearest integer (banker's rounding — ties go to even):

```rust
// given token: T where T: F32x8Convert
let rounded = floats.to_i32x8_round();
// [2, 3, -3, 4, 6, 6, 7, 8]
```

## Integer to Float

```rust
use magetypes::simd::{
    generic::i32x8,
    backends::I32x8Convert,
};

#[inline(always)]
fn int_to_float<T: I32x8Convert>(token: T) {
    let ints = i32x8::<T>::from_array(token, [1, 2, 3, 4, 5, 6, 7, 8]);
    let floats = ints.to_f32x8();
    // [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
}
```

For values that don't fit exactly in f32 (integers above 2^24), the result is the nearest representable float.

## 128-bit Variants

The same methods exist on 128-bit types:

```rust
use magetypes::simd::{
    generic::f32x4,
    backends::F32x4Convert,
};

#[inline(always)]
fn conversions_128<T: F32x4Convert>(token: T) {
    let floats = f32x4::<T>::from_array(token, [1.5, 2.7, -3.2, 4.0]);
    let ints = floats.to_i32x4();          // Truncate
    let rounded = floats.to_i32x4_round(); // Round

    let back = ints.to_f32x4();
}
```

## Convert trait coverage

`F32x4Convert` and `F32x8Convert` are implemented for `X64V3Token`, `NeonToken`, `Wasm128Token`, and `ScalarToken` only. They are **not** implemented for `X64V4Token`, `X64V4xToken`, or `Avx512Fp16Token` — generic kernels parameterized on `T: F32x4Convert` / `T: F32x8Convert` reject V4-family tokens at compile time.

`F32x16Convert` is implemented on **every** token — V3 / V4 / V4x / NEON / WASM / scalar (only `Avx512Fp16Token` is missing). On AVX-512 silicon `f32x16<T>` runs at native 512-bit width; on every other platform the same code path runs through polyfills (two `f32x8` ops on V3, four `f32x4` ops on NEON / WASM, scalar lanes on `ScalarToken`). Widening a kernel to `f32x16` is the cleanest workaround for the missing `F32x4Convert` / `F32x8Convert` impls on V4 — a single kernel covers every platform.

Polyfill cost depends on the kernel shape: pure per-element compute (add / mul / fma / polynomial bodies) is effectively free; reductions pay ~1.5-2× over a hand-tuned native version; heavy cross-lane shuffles can be expensive; and on V3 a transcendental with many live temporaries can quadruple register pressure and start spilling. See [transcendentals → polyfill overhead]({{< ref "/magetypes/math/transcendentals" >}}#f32x16-polyfill-overhead-by-operation) for the full breakdown.

This affects any function bounded on `F32x4Convert` / `F32x8Convert` — including the [transcendentals]({{< ref "/magetypes/math/transcendentals" >}}) (`pow_*`, `log2_*`, `exp2_*`, `ln_*`, `exp_*`, `log10_*`) and `to_i32x4` / `to_i32x8` conversions on `f32x4` / `f32x8`. See [issue #45](https://github.com/imazen/archmage/issues/45) for the full audit and the build-time tradeoffs of the proposed delegation fix.

## Lane Access

Vectors implement `Index<usize>` and `IndexMut<usize>` for single-lane access:

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Backend,
};

#[inline(always)]
fn lane_access<T: F32x8Backend>(token: T) {
    let v = f32x8::<T>::from_array(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let third = v[2];   // 3.0

    let mut v = v;
    v[2] = 99.0;        // [1.0, 2.0, 99.0, 4.0, 5.0, 6.0, 7.0, 8.0]
}
```

Lane indices are runtime values with bounds checking — out-of-bounds panics.

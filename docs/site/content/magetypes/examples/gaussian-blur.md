+++
title = "Gaussian Blur"
description = "Separable Gaussian using trait-bounded helpers that call each other"
weight = 5
+++

When SIMD functions need to call other SIMD functions, `#[magetypes]` gets awkward — the inner call needs an explicit suffixed name (`blur_inner_neon`), and the macro doesn't rename calls in the body. The trait-bound pattern handles this naturally: generic functions call generic functions, and the compiler monomorphizes the whole tree.

This example from `zenfilters` shows a separable Gaussian blur where the dispatch function calls a FIR blur or a stack blur depending on the kernel size, and both of those call inner helpers — all generic.

## Architecture

```
gaussian_blur_dispatch<T>       ← #[magetypes] generates entry points
  ├── gaussian_blur_fir<T>      ← trait-bound generic (composable)
  │     └── uses f32x8 load/store/mul_add
  └── stackblur_plane<T>        ← trait-bound generic (composable)
        └── uses f32x8 load/store/add/sub
```

The `#[magetypes]` macro generates concrete `_neon` and `_wasm128` entry points for dispatch. Inside, it calls trait-bounded helpers that the compiler inlines and monomorphizes.

## Dispatch entry point

```rust
use archmage::prelude::*;
use magetypes::simd::backends::{F32x8Backend, F32x8Convert};
use magetypes::simd::generic::f32x8 as GenericF32x8;

#[magetypes(neon, wasm128)]
pub fn gaussian_blur_dispatch(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
    width: u32,
    height: u32,
    kernel: &GaussianKernel,
) {
    let sigma = kernel_sigma(kernel);
    if should_use_stackblur(sigma) {
        let radius = sigma_to_stackblur_radius(sigma);
        stackblur_plane_generic(token, src, dst, width, height, radius);
        return;
    }
    gaussian_blur_fir_generic(token, src, dst, width, height, kernel);
}
```

The entry point is `#[magetypes]`-annotated, generating `_neon` and `_wasm128` variants. It calls trait-bounded helpers, passing the token through. Because the helpers are `#[inline]`, they get inlined into the `#[arcane]` region and benefit from the target-feature context.

## FIR Gaussian — trait-bounded generic

The FIR blur does horizontal then vertical passes. Each pass uses `partition_slice_mut` for SIMD chunks and a scalar tail.

```rust
/// Type alias shorthand inside generic functions.
/// V<T> = GenericF32x8<T>, resolving to the correct SIMD type per backend.
type V<T> = GenericF32x8<T>;

#[inline]
fn gaussian_blur_fir_generic<T: F32x8Backend + F32x8Convert + Copy>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
    width: u32,
    height: u32,
    kernel: &GaussianKernel,
) {
    let w = width as usize;
    let h = height as usize;
    let radius = kernel.radius;

    let mut h_buf = vec![0.0f32; w * h];  // horizontal pass output
    let mut padded = vec![0.0f32; w + 2 * radius]; // edge-replicated row

    // Horizontal pass
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];

        // Edge replication: mirror boundary pixels
        padded.clear();
        padded.extend(core::iter::repeat_n(row[0], radius));
        padded.extend_from_slice(row);
        padded.extend(core::iter::repeat_n(row[w - 1], radius));

        let out_row = &mut h_buf[y * w..(y + 1) * w];
        let (out_chunks, out_tail) = V::<T>::partition_slice_mut(token, out_row);

        // SIMD: convolve 8 output pixels at a time
        for (ci, out_chunk) in out_chunks.iter_mut().enumerate() {
            let x = ci * 8;
            let mut acc = V::<T>::zero(token);
            for (k, &weight) in kernel.weights().iter().enumerate() {
                let wv = V::<T>::splat(token, weight);
                let src_chunk: &[f32; 8] =
                    padded[x + k..x + k + 8].try_into().unwrap();
                let vals = V::<T>::load(token, src_chunk);
                acc = vals.mul_add(wv, acc);
            }
            acc.store(out_chunk);
        }

        // Scalar tail
        let x_start = out_chunks.len() * 8;
        for (xi, v) in out_tail.iter_mut().enumerate() {
            let x = x_start + xi;
            let mut sum = 0.0f32;
            for (k, &weight) in kernel.weights().iter().enumerate() {
                sum += padded[x + k] * weight;
            }
            *v = sum;
        }
    }

    // Vertical pass — same pattern, reading from h_buf columns
    for y in 0..h {
        let out_row = &mut dst[y * w..(y + 1) * w];
        let (out_chunks, out_tail) = V::<T>::partition_slice_mut(token, out_row);

        for (ci, out_chunk) in out_chunks.iter_mut().enumerate() {
            let x = ci * 8;
            let mut acc = V::<T>::zero(token);
            for (k, &weight) in kernel.weights().iter().enumerate() {
                let sy = (y + k).saturating_sub(radius).min(h - 1);
                let wv = V::<T>::splat(token, weight);
                let src_chunk: &[f32; 8] =
                    h_buf[sy * w + x..sy * w + x + 8].try_into().unwrap();
                acc = V::<T>::load(token, src_chunk).mul_add(wv, acc);
            }
            acc.store(out_chunk);
        }

        let x_start = out_chunks.len() * 8;
        for (xi, v) in out_tail.iter_mut().enumerate() {
            let x = x_start + xi;
            let mut sum = 0.0f32;
            for (k, &weight) in kernel.weights().iter().enumerate() {
                let sy = (y + k).saturating_sub(radius).min(h - 1);
                sum += h_buf[sy * w + x] * weight;
            }
            *v = sum;
        }
    }
}
```

## Why this works for composability

The critical line is the trait bound:

```rust
fn gaussian_blur_fir_generic<T: F32x8Backend + F32x8Convert + Copy>
```

This function can be called from *any* context that has a token implementing `F32x8Backend`. It doesn't know or care whether it's running on AVX2, NEON, or scalar — the token carries that information. When the `#[magetypes]` entry point calls it, `T` resolves to the concrete token type, and the compiler generates the right instructions.

If you tried to do this with `#[magetypes]` on the inner function, you'd get `gaussian_blur_fir_generic_neon`, `gaussian_blur_fir_generic_wasm128`, etc. — and the dispatch function would need to call the right suffix explicitly. With trait bounds, the call is just `gaussian_blur_fir_generic(token, ...)` and Rust's type system handles the rest.

## Extra trait bounds

Note `F32x8Convert` in the bound. Some operations (like `to_i32_round()`, type conversions) live on separate traits from the basic arithmetic. Add them to the bound when your function needs them. Common combinations:

- `F32x8Backend` — arithmetic, comparisons, reductions
- `F32x8Backend + F32x8Convert` — plus float-to-int conversions
- `F32x8Backend + Copy` — needed when you pass the token to sub-functions

The `Copy` bound is always satisfied by tokens (they're zero-size types), but Rust needs it stated explicitly when you pass `token` to multiple callees.

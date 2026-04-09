+++
title = "Pixel Blending"
description = "SrcOver alpha blending on RGBA pixels using f32x4 generics"
weight = 2
+++

When your data has a natural 4-element structure — RGBA pixels, quaternions, 3D coordinates with padding — `f32x4` maps one logical unit to one SIMD register. This example from `zenblend` shows Porter-Duff SrcOver compositing.

## SrcOver row blend

One RGBA pixel per `f32x4`. The function is generic over `T: F32x4Backend`, which means it works with any token that supports 128-bit float SIMD.

```rust
use magetypes::simd::backends::F32x4Backend;
use magetypes::simd::generic::f32x4;

/// Blend foreground over background using premultiplied alpha.
/// fg is modified in-place.
#[inline]
pub fn blend_src_over_row<T: F32x4Backend>(token: T, fg: &mut [f32], bg: &[f32]) {
    let (fg_chunks, _) = f32x4::<T>::partition_slice_mut(token, fg);
    let (bg_chunks, _) = f32x4::<T>::partition_slice(token, bg);

    for (fg_chunk, bg_chunk) in fg_chunks.iter_mut().zip(bg_chunks.iter()) {
        let fg_pixel = f32x4::load(token, fg_chunk);
        let bg_pixel = f32x4::load(token, bg_chunk);
        // Alpha is the 4th element. fg[3] = alpha of fg pixel.
        let inv_alpha = f32x4::splat(token, 1.0 - fg_chunk[3]);
        let result = fg_pixel + bg_pixel * inv_alpha;
        result.store(fg_chunk);
    }
}
```

**Why `f32x4` instead of `f32x8`?** Each RGBA pixel is 4 floats. With `f32x8` you'd process two pixels per iteration, but the alpha value differs per pixel — you'd need to extract individual lanes for the `inv_alpha` splat. Using `f32x4`, each pixel's alpha naturally broadcasts to all four channels.

## Mask multiply (RGB only, alpha preserved)

Sometimes you need to apply a per-pixel opacity mask to the RGB channels but leave alpha untouched. `from_array` lets you construct a vector with different values per lane:

```rust
#[inline]
pub fn mask_row_rgb<T: F32x4Backend>(token: T, fg: &mut [f32], mask: &[f32]) {
    let (fg_chunks, _) = f32x4::<T>::partition_slice_mut(token, fg);

    for (fg_chunk, &m) in fg_chunks.iter_mut().zip(mask.iter()) {
        let fg_pixel = f32x4::load(token, fg_chunk);
        // Mask RGB, keep alpha at 1.0
        let mask_vec = f32x4::from_array(token, [m, m, m, 1.0]);
        let result = fg_pixel * mask_vec;
        result.store(fg_chunk);
    }
}
```

## Linear interpolation between two rows

LERP with a per-pixel `t` value. Uses the pattern `a + (b - a) * t` which compiles to FMA on platforms that support it.

```rust
#[inline]
pub fn lerp_row<T: F32x4Backend>(
    token: T,
    a: &[f32],
    b: &[f32],
    t: &[f32],
    out: &mut [f32],
) {
    let (a_chunks, _) = f32x4::<T>::partition_slice(token, a);
    let (b_chunks, _) = f32x4::<T>::partition_slice(token, b);
    let (out_chunks, _) = f32x4::<T>::partition_slice_mut(token, out);

    for ((a_chunk, b_chunk), (&tv, out_chunk)) in a_chunks
        .iter()
        .zip(b_chunks.iter())
        .zip(t.iter().zip(out_chunks.iter_mut()))
    {
        let a_vec = f32x4::load(token, a_chunk);
        let b_vec = f32x4::load(token, b_chunk);
        let t_vec = f32x4::splat(token, tv);
        let result = a_vec + (b_vec - a_vec) * t_vec;
        result.store(out_chunk);
    }
}
```

## Dispatch

These functions use the trait-bound pattern (`T: F32x4Backend`) rather than `#[magetypes]`. The dispatch happens at the call site — typically in a module that knows the concrete token:

```rust
use archmage::prelude::*;

#[arcane(import_intrinsics)]
fn blend_row_v3(token: X64V3Token, fg: &mut [f32], bg: &[f32]) {
    blend_src_over_row(token, fg, bg);
}

#[arcane(import_intrinsics)]
fn blend_row_neon(token: NeonToken, fg: &mut [f32], bg: &[f32]) {
    blend_src_over_row(token, fg, bg);
}

fn blend_row_scalar(token: ScalarToken, fg: &mut [f32], bg: &[f32]) {
    blend_src_over_row(token, fg, bg);
}

pub fn blend_row(fg: &mut [f32], bg: &[f32]) {
    incant!(blend_row(fg, bg), [v3, neon, wasm128, scalar]);
}
```

Or use `#[magetypes]` on a thin wrapper if you prefer the generated approach.

## Trait-Bound Generics vs `#[magetypes]` Macro

Both patterns achieve the same result. The trade-offs:

| | Trait-bound (`T: Backend`) | `#[magetypes]` macro |
|---|---|---|
| Composability | Functions can call other generic functions | Inner calls need explicit `_neon`/`_v3` suffixes |
| Dispatch | Manual `#[arcane]` wrappers + `incant!` | Auto-generated variants |
| Flexibility | Can add extra trait bounds (`+ F32x8Convert`) | Fixed to listed tiers |
| Readability | Standard Rust generics | Macro substitution can be surprising |

Use `#[magetypes]` when the function is self-contained. Use trait bounds when you want to compose generic SIMD functions (see [Gaussian Blur](@/magetypes/examples/gaussian-blur.md)).

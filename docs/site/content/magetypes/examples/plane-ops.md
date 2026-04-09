+++
title = "Plane Operations"
description = "Scale, offset, and transform a float slice with partition_slice and scalar tail"
weight = 1
+++

The simplest magetypes generic pattern: process a `&mut [f32]` in 8-wide SIMD chunks, then handle the leftover elements scalar. This is the bread and butter of image processing — every plane operation (brightness, contrast, exposure) follows this shape.

These examples come from `zenfilters`, which applies perceptual adjustments in Oklab color space.

## Scale a plane

Multiply every element by a constant. The `#[magetypes]` macro generates one function per listed tier, substituting `Token` with the concrete token type.

```rust
use archmage::prelude::*;
use magetypes::simd::generic::f32x8 as GenericF32x8;

#[magetypes(neon, wasm128)]
pub fn scale_plane(token: Token, plane: &mut [f32], factor: f32) {
    // Type alias inside the function — Token gets replaced per tier
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    let factor_v = f32x8::splat(token, factor);

    // partition_slice_mut splits the slice into aligned 8-element chunks
    // and a remainder tail. The chunks are &mut [f32; 8] references.
    let (chunks, tail) = f32x8::partition_slice_mut(token, plane);

    for chunk in chunks {
        let v = f32x8::load(token, chunk);
        (v * factor_v).store(chunk);
    }

    // Scalar tail — always needed for slices not divisible by 8
    for v in tail {
        *v *= factor;
    }
}
```

**Key pattern:** `partition_slice_mut` handles alignment and remainder for you. The SIMD loop processes `&mut [f32; 8]` chunks — the array reference guarantees 8 elements exist without bounds checks inside the loop. The scalar tail handles 0-7 leftover elements.

## Power contrast

A more complex per-element operation: raise to a power and scale. Uses `pow_lowp_unchecked` — a low-precision SIMD power function available on all backends through magetypes' transcendental support.

```rust
#[magetypes(neon, wasm128)]
pub fn power_contrast_plane(token: Token, plane: &mut [f32], exp: f32, scale: f32) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    let scale_v = f32x8::splat(token, scale);
    let zero_v = f32x8::zero(token);

    let (chunks, tail) = f32x8::partition_slice_mut(token, plane);
    for chunk in chunks {
        let v = f32x8::load(token, &*chunk);
        let powered = v.max(zero_v).pow_lowp_unchecked(exp);
        (powered * scale_v).store(chunk);
    }
    for v in tail {
        if *v > 0.0 {
            *v = fast_powf(*v, exp) * scale;
        }
    }
}
```

## Sigmoid tone mapping

Demonstrates comparisons (`simd_le`, `simd_ge`), `blend` for conditional selection, and `recip` — all available on every backend.

```rust
#[magetypes(neon, wasm128)]
pub fn sigmoid_tone_map_plane(
    token: Token,
    plane: &mut [f32],
    contrast: f32,
    bias_a: f32,
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    let one_v = f32x8::splat(token, 1.0);
    let zero_v = f32x8::zero(token);
    let bias_a_v = f32x8::splat(token, bias_a);
    let has_bias = bias_a.abs() > 1e-6;

    let (chunks, tail) = f32x8::partition_slice_mut(token, plane);
    for chunk in chunks {
        let mut x = f32x8::load(token, &*chunk).max(zero_v).min(one_v);

        if has_bias {
            let denom = (one_v - x).mul_add(bias_a_v, one_v);
            x *= denom.recip();
            x = x.max(zero_v).min(one_v);
        }

        let x_safe = x.max(f32x8::splat(token, 1e-7));
        let ratio = (one_v - x_safe) * x_safe.recip();
        let powered = ratio.pow_lowp_unchecked(contrast);
        let result = (one_v + powered).recip();

        // Conditional selection: where x <= 0, use 0; where x >= 1, use 1
        let is_zero = x.simd_le(zero_v);
        let is_one = x.simd_ge(one_v);
        let r = f32x8::blend(is_zero, zero_v, result);
        let r = f32x8::blend(is_one, one_v, r);
        r.store(chunk);
    }

    // Scalar tail mirrors the SIMD logic
    for v in tail {
        let mut x = v.clamp(0.0, 1.0);
        if has_bias {
            x = x / (bias_a * (1.0 - x) + 1.0);
        }
        *v = if x <= 0.0 {
            0.0
        } else if x >= 1.0 {
            1.0
        } else {
            1.0 / (1.0 + fast_powf((1.0 - x) / x, contrast))
        };
    }
}
```

## Dispatch

These `#[magetypes(neon, wasm128)]` functions generate `_neon` and `_wasm128` suffixed variants. Wire them up with `incant!`:

```rust
pub fn scale_plane(plane: &mut [f32], factor: f32) {
    incant!(scale_plane(plane, factor));
}
```

On x86-64, if you also want an AVX2 path, use `#[magetypes(v3, neon, wasm128, scalar)]` — the `v3` tier generates an `_v3` variant that uses native 256-bit `f32x8` instead of the 2x128-bit polyfill.

## Why This Pattern Works

- **One function body.** All platforms execute the same algorithm. The compiler monomorphizes per backend, so AVX2 gets native `_mm256` ops, NEON gets pairs of `float32x4_t` ops, WASM gets pairs of `v128` ops, and scalar gets an array loop.
- **No unsafe.** The token proves CPU support. `partition_slice_mut` proves the chunk size. Bounds checks are eliminated by the `[f32; 8]` array reference.
- **Scalar tail is explicit.** Don't forget it. Production image data is rarely 8-aligned in width. The tail is cheap — it runs at most 7 iterations.

+++
title = "Convolution Kernel"
description = "Horizontal image filter with f32x4 accumulator and channel-count specialization"
weight = 3
+++

Image resizing is a weighted sum (convolution) of neighboring pixels. This example from `zenresize` shows two patterns: a 4-channel specialization that uses `f32x4` for natural RGBA accumulation, and a dispatch wrapper that selects the right path.

## 4-channel horizontal convolution

Each output pixel is the weighted sum of `max_taps` input pixels. With 4-channel RGBA data, one pixel is exactly one `f32x4` register. The accumulator stays in SIMD the entire time — no lane extraction needed.

```rust
use magetypes::simd::backends::F32x4Backend;
use magetypes::simd::generic::f32x4;

/// 4-channel horizontal f32 convolution using f32x4.
///
/// Uses * + += instead of mul_add because on WASM (no FMA),
/// mul_add goes through a branch to (a * b) + c anyway.
/// Direct * + += gives LLVM a cleaner expression to schedule.
#[inline(always)]
fn filter_h_4ch<T: F32x4Backend>(
    token: T,
    input: &[f32],
    output: &mut [f32],
    weights: &F32WeightTable,
) {
    let (in_pixels, _) = input.as_chunks::<4>();
    let (out_pixels, _) = output.as_chunks_mut::<4>();

    for out_x in 0..weights.len() {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let mut acc = f32x4::zero(token);

        for (t, &weight) in w.iter().enumerate() {
            // Each input pixel is a [f32; 4] — loads directly into f32x4
            acc += f32x4::from_array(token, in_pixels[left + t])
                * f32x4::splat(token, weight);
        }

        out_pixels[out_x] = acc.to_array();
    }
}
```

**Key insight:** `as_chunks::<4>()` gives `&[[f32; 4]]` — the pixel boundaries are in the type system. `from_array` loads a `[f32; 4]` directly into a SIMD register with zero overhead (it's a transmute on aligned data). The accumulator `acc` stays as `f32x4` for all taps, only converting back to an array at the store.

## Channel-count dispatch

The SIMD benefit only applies to 4-channel data (or any channel count matching the SIMD width). For 3 channels, the data isn't aligned to any SIMD boundary — scalar code auto-vectorizes well enough. The `#[magetypes]` macro generates the dispatch function, and the inner `match` selects the right path:

```rust
#[magetypes(neon, wasm128)]
#[inline(always)]
pub fn filter_h_row_f32(
    token: Token,
    input: &[f32],
    output: &mut [f32],
    weights: &F32WeightTable,
    channels: usize,
) {
    match channels {
        4 => filter_h_4ch(token, input, output, weights),
        3 => filter_h_3ch(input, output, weights),     // scalar
        _ => filter_h_generic(input, output, weights, channels), // scalar
    }
}
```

The 3-channel and generic paths don't use SIMD at all — they're plain scalar loops that LLVM auto-vectorizes. The SIMD effort goes where it matters: the 4-channel path that handles ~90% of real-world image data.

## Why `f32x4` and not `f32x8`?

For convolution, the accumulation happens across taps (weights), not across pixels. Each tap multiplies one pixel (4 values) by one scalar weight. With `f32x4`, the weight broadcasts naturally to all 4 channels via `splat`.

With `f32x8`, you'd pack two pixels into one register, but then you'd need the *same* weight for lanes 0-3 and a *different* weight for lanes 4-7. That means either loading two weights and shuffling, or accepting that both pixels use the same weight (which only works for very specific kernel structures). For general-purpose convolution, `f32x4` per pixel is simpler and equally fast.

## The weight table

The `F32WeightTable` stores pre-computed filter weights with zero-padding to `max_taps`:

```rust
struct F32WeightTable {
    left: Vec<u32>,        // left-most input pixel for each output pixel
    data: Vec<f32>,        // flat: max_taps entries per output pixel
    max_taps: usize,
}

impl F32WeightTable {
    fn weights(&self, out_x: usize) -> &[f32] {
        let start = out_x * self.max_taps;
        &self.data[start..start + self.max_taps]
    }
}
```

Zero-padding to `max_taps` means the inner loop always iterates the same number of times — no branch on the actual tap count. The zero weights produce zero contributions via multiply, which the CPU handles with no branch prediction overhead.

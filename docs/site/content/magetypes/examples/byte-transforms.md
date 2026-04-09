+++
title = "Byte-Level Transforms"
description = "WebP lossless inverse transforms using u8x16 generics for pixel prediction"
weight = 7
+++

Not all SIMD is floating-point. Image codecs often manipulate raw bytes — pixel prediction, delta decoding, channel extraction. This example from `zenwebp` shows `u8x16` generics for WebP's lossless inverse transforms: adding a green channel to red and blue, and applying left-pixel prediction.

## Green channel addition

WebP lossless stores images with a "subtract green" transform: R and B are stored relative to G. The inverse adds G back to both channels. With RGBA pixels packed as `[R, G, B, A, R, G, B, A, ...]`, we need to extract every G byte, then add it to its neighboring R and B bytes.

```rust
use magetypes::simd::backends::U8x16Backend;
use magetypes::simd::generic::u8x16;

/// Add green to red and blue channels.
/// Processes 4 RGBA pixels (16 bytes) per iteration.
#[inline(always)]
fn add_green_portable<T: U8x16Backend>(
    token: T,
    data: &mut [u8],
    range: core::ops::Range<usize>,
) {
    // Shuffle mask: broadcast G to R and B positions within each pixel.
    // RGBA layout: [R0, G0, B0, A0, R1, G1, B1, A1, ...]
    // We want:     [G0, 0,  G0, 0,  G1, 0,  G1, 0,  ...]
    let green_mask = u8x16::from_array(token, [
        1, 0, 1, 0,   // pixel 0: G to R pos, 0 to G pos, G to B pos, 0 to A pos
        5, 0, 5, 0,   // pixel 1
        9, 0, 9, 0,   // pixel 2
        13, 0, 13, 0, // pixel 3
    ]);

    let zero = u8x16::zero(token);

    let mut offset = range.start;
    while offset + 16 <= range.end {
        let chunk: &mut [u8; 16] = data[offset..offset + 16]
            .try_into().unwrap();
        let pixels = u8x16::load(token, chunk);

        // Extract green channels via shuffle, zero out G and A positions
        let greens = pixels.shuffle(green_mask);

        // Wrapping byte addition: R += G, B += G, G += 0, A += 0
        let result = pixels.wrapping_add(greens);
        result.store(chunk);

        offset += 16;
    }

    // Scalar tail: handle remaining pixels
    while offset + 4 <= range.end {
        let g = data[offset + 1];
        data[offset] = data[offset].wrapping_add(g);     // R += G
        data[offset + 2] = data[offset + 2].wrapping_add(g); // B += G
        offset += 4;
    }
}
```

## Left-pixel prediction (predictor add)

WebP's predictor 1 stores each pixel as a delta from the pixel to its left. The inverse accumulates left-to-right: `pixel[x] += pixel[x-1]`. This is inherently serial per row (each pixel depends on the previous), but we can process 4 pixels at a time using a parallel prefix sum within the SIMD register.

```rust
/// Apply left-pixel prediction (predictor 1) to 4 RGBA pixels.
/// Each pixel is the sum of itself and all pixels to its left.
#[inline(always)]
fn predictor_add_left<T: U8x16Backend>(
    token: T,
    data: &mut [u8],
    start: usize,
    prev_pixel: [u8; 4],
) {
    let chunk: &mut [u8; 16] = data[start..start + 16]
        .try_into().unwrap();
    let deltas = u8x16::load(token, chunk);

    // Parallel prefix sum across 4 pixels:
    // Step 1: Add pixel 0 to pixel 1 (shift by 4 bytes and add)
    let shifted1 = deltas.shift_left_bytes::<4>();
    let sum1 = deltas.wrapping_add(shifted1);

    // Step 2: Add pixels 0-1 to pixels 2-3 (shift by 8 bytes and add)
    let shifted2 = sum1.shift_left_bytes::<8>();
    let sum2 = sum1.wrapping_add(shifted2);

    // Step 3: Add the previous pixel (from the end of the prior chunk)
    let prev = u8x16::from_array(token, [
        prev_pixel[0], prev_pixel[1], prev_pixel[2], prev_pixel[3],
        prev_pixel[0], prev_pixel[1], prev_pixel[2], prev_pixel[3],
        prev_pixel[0], prev_pixel[1], prev_pixel[2], prev_pixel[3],
        prev_pixel[0], prev_pixel[1], prev_pixel[2], prev_pixel[3],
    ]);
    let result = sum2.wrapping_add(prev);
    result.store(chunk);
}
```

## Top-pixel prediction (batch add)

Predictors that reference the row above (top, top-right, top-left) don't have the serial dependency within a row — each pixel depends only on the row above and itself. These parallelize fully:

```rust
/// Apply top-pixel prediction: pixel[x] += top_row[x].
/// Processes 4 pixels (16 bytes) at a time.
#[inline(always)]
fn predictor_add_top<T: U8x16Backend>(
    token: T,
    data: &mut [u8],
    offset: usize,
    top_row: &[u8],
    top_offset: usize,
) {
    let chunk: &mut [u8; 16] = data[offset..offset + 16]
        .try_into().unwrap();
    let top_chunk: &[u8; 16] = top_row[top_offset..top_offset + 16]
        .try_into().unwrap();

    let delta = u8x16::load(token, chunk);
    let top = u8x16::load(token, top_chunk);
    delta.wrapping_add(top).store(chunk);
}
```

## Dispatch pattern

The entry point uses `#[arcane]` with a concrete token, then calls the generic inner functions:

```rust
#[cfg(target_arch = "x86_64")]
#[arcane]
pub fn add_green_sse2_entry(token: X64V1Token, data: &mut [u8]) {
    add_green_portable(token, data, 0..data.len());
}

// NEON and WASM128 use the same generic function:
#[cfg(target_arch = "aarch64")]
#[arcane]
pub fn add_green_neon_entry(token: NeonToken, data: &mut [u8]) {
    add_green_portable(token, data, 0..data.len());
}
```

## Integer vs float generics

The `u8x16` generic pattern is identical to `f32x8` — same trait bound approach, same `partition_slice`, same `load`/`store`. The available operations differ:

| `f32x8` | `u8x16` |
|---|---|
| `+`, `-`, `*`, `/` | `wrapping_add`, `wrapping_sub`, `saturating_add` |
| `mul_add`, `sqrt`, `abs` | `shuffle`, `shift_left_bytes`, `shift_right_bytes` |
| `simd_ge`, `blend` | `simd_eq`, `bitand`, `bitor` |
| `reduce_add`, `reduce_min` | `any_nonzero`, `count_ones` |

The byte types shine for data movement (shuffles, shifts, masks) where float types shine for arithmetic. Choose the type that matches your data, and the generic pattern works the same.

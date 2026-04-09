+++
title = "Color Conversion"
description = "RGB-to-YCbCr with matrix coefficients, multi-plane output, and incant! dispatch"
weight = 6
+++

Color space conversion is a matrix multiply per pixel — three dot products with constant coefficients. This example from `zenjpeg` converts interleaved RGB float planes to separate Y, Cb, Cr planes using BT.601 coefficients. It demonstrates working with multiple input and output slices, safe load/store helpers, and `incant!` dispatch.

## RGB to YCbCr planes

```rust
use archmage::prelude::*;
use magetypes::simd::generic::f32x8 as GenericF32x8;

/// Load 8 f32 values from a slice at the given offset.
#[inline(always)]
fn load_f32x8<T: magetypes::simd::backends::F32x8Backend>(
    token: T,
    slice: &[f32],
    offset: usize,
) -> GenericF32x8<T> {
    let arr: &[f32; 8] = slice[offset..offset + 8].try_into().unwrap();
    GenericF32x8::<T>::load(token, arr)
}

/// Store 8 f32 values to a slice at the given offset.
#[inline(always)]
fn store_f32x8<T: magetypes::simd::backends::F32x8Backend>(
    result: &mut [f32],
    offset: usize,
    v: GenericF32x8<T>,
) {
    let arr: &mut [f32; 8] = result[offset..offset + 8].try_into().unwrap();
    v.store(arr);
}

// BT.601 coefficients for RGB → YCbCr
const KR: f32 = 0.299;
const KG: f32 = 0.587;
const KB: f32 = 0.114;
const CB_SCALE: f32 = 0.564;   // 0.5 / (1 - KB)
const CR_SCALE: f32 = 0.713;   // 0.5 / (1 - KR)

#[magetypes(v3, neon, wasm128, scalar)]
fn rgb_to_ycbcr_planes(
    token: Token,
    r_plane: &[f32],
    g_plane: &[f32],
    b_plane: &[f32],
    y_out: &mut [f32],
    cb_out: &mut [f32],
    cr_out: &mut [f32],
    count: usize,
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    let kr = f32x8::splat(token, KR);
    let kg = f32x8::splat(token, KG);
    let kb = f32x8::splat(token, KB);
    let cb_s = f32x8::splat(token, CB_SCALE);
    let cr_s = f32x8::splat(token, CR_SCALE);

    // SIMD path: 8 pixels at a time
    let simd_count = count / 8;
    for i in 0..simd_count {
        let off = i * 8;
        let r = load_f32x8(token, r_plane, off);
        let g = load_f32x8(token, g_plane, off);
        let b = load_f32x8(token, b_plane, off);

        // Y = KR*R + KG*G + KB*B
        let y = r * kr + g * kg + b * kb;

        // Cb = CB_SCALE * (B - Y)
        let cb = (b - y) * cb_s;

        // Cr = CR_SCALE * (R - Y)
        let cr = (r - y) * cr_s;

        store_f32x8(y_out, off, y);
        store_f32x8(cb_out, off, cb);
        store_f32x8(cr_out, off, cr);
    }

    // Scalar tail
    for i in (simd_count * 8)..count {
        let r = r_plane[i];
        let g = g_plane[i];
        let b = b_plane[i];
        let y = KR * r + KG * g + KB * b;
        y_out[i] = y;
        cb_out[i] = CB_SCALE * (b - y);
        cr_out[i] = CR_SCALE * (r - y);
    }
}
```

## Dispatch

```rust
/// Public API: converts RGB planes to YCbCr planes.
pub fn rgb_to_ycbcr_planes(
    r: &[f32], g: &[f32], b: &[f32],
    y: &mut [f32], cb: &mut [f32], cr: &mut [f32],
    count: usize,
) {
    incant!(rgb_to_ycbcr_planes(r, g, b, y, cb, cr, count));
}
```

## Safe load/store helpers

The `load_f32x8` and `store_f32x8` helpers above demonstrate a common pattern: converting a `&[f32]` slice + offset into the `&[f32; 8]` reference that magetypes requires. The `try_into().unwrap()` does a bounds check at runtime — if the slice is too short, it panics with a clear message rather than producing undefined behavior.

In release mode with the loop structure above, LLVM often proves the bounds check is unnecessary and removes it entirely. The key is that `simd_count = count / 8` guarantees `off + 8 <= count` for all iterations.

For the production version, `zenfilters` uses `partition_slice`/`partition_slice_mut` instead of manual offset arithmetic — it handles the alignment and splitting in one call. Use manual offsets when you need multiple slices at different offsets (like the multi-plane pattern above).

## Interleaved vs. planar data

This example works on separate R, G, B planes (structure-of-arrays). If your data is interleaved RGBRGB... (array-of-structures), you'd need to deinterleave first, or use `f32x4` per pixel as shown in [Pixel Blending](@/magetypes/examples/pixel-blending.md).

The planar layout is better for SIMD because:
- All 8 R values are contiguous → one load gets 8 R channels
- The color matrix becomes three independent dot products
- No lane-crossing shuffles needed

Most image codecs (JPEG, WebP, AVIF) work internally in planar format for exactly this reason.

+++
title = "Pixel Blending: a Generic Helper"
description = "Complete zenblend SrcOver call chain: public entry, generated context, generic helper"
weight = 2
+++

This is the real SrcOver helper from `zen/zenblend` at
`636c847989a4c662d3a56c7ddd3f7e04387c0b53`.
The production chain is [`blend.rs`'s SrcOver arm](https://github.com/imazen/zenblend/blob/636c847989a4c662d3a56c7ddd3f7e04387c0b53/src/blend.rs#L13)
→ [`simd::blend_src_over_row` / `incant!`](https://github.com/imazen/zenblend/blob/636c847989a4c662d3a56c7ddd3f7e04387c0b53/src/simd/mod.rs#L212)
→ the NEON/WASM/scalar entry →
[`portable::blend_src_over_row<T: F32x4Backend>`](https://github.com/imazen/zenblend/blob/636c847989a4c662d3a56c7ddd3f7e04387c0b53/src/simd/portable.rs#L12).

For this standalone version, the helper is renamed `blend_kernel`, and one
`#[magetypes]` wrapper replaces the separate forwarding functions. The helper
body is unchanged; the production x86 specialization is omitted in favor of
the same generic body. This makes every part of the call chain visible:

```rust
use archmage::prelude::*;
use magetypes::simd::{backends::F32x4Backend, generic::f32x4};

#[inline]
fn blend_kernel<T: F32x4Backend>(token: T, fg: &mut [f32], bg: &[f32]) {
    let (fg_chunks, _) = f32x4::<T>::partition_slice_mut(token, fg);
    let (bg_chunks, _) = f32x4::<T>::partition_slice(token, bg);

    for (fg_chunk, bg_chunk) in fg_chunks.iter_mut().zip(bg_chunks.iter()) {
        let fg_pixel = f32x4::load(token, fg_chunk);
        let bg_pixel = f32x4::load(token, bg_chunk);
        let inv_alpha = f32x4::splat(token, 1.0 - fg_chunk[3]);
        let result = fg_pixel + bg_pixel * inv_alpha;
        result.store(fg_chunk);
    }
}

#[magetypes(v3, neon, wasm128, scalar)]
fn blend_entry(token: Token, fg: &mut [f32], bg: &[f32]) {
    blend_kernel(token, fg, bg);
}

pub fn blend_row(fg: &mut [f32], bg: &[f32]) {
    incant!(blend_entry(fg, bg), [v3, neon, wasm128, scalar])
}


let mut fg = [0.1, 0.2, 0.3, 0.5];
blend_row(&mut fg, &[0.4, 0.4, 0.4, 1.0]);
for (actual, expected) in fg.into_iter().zip([0.3, 0.4, 0.5, 1.0]) {
    assert!((actual - expected).abs() < 1e-6);
}
```

The buffers contain **premultiplied RGBA f32** in the same working color space;
this computes `foreground + background * (1 - foreground_alpha)` in that space.
Use a linear-light working space when linear-light compositing is intended.
One vector is one four-channel pixel. Complete pixels are processed up to the
shorter buffer; unmatched pixels and trailing incomplete pixels are left alone,
matching the source helper. This is not an image-stride or color-conversion API.

The `#[inline]` helper is generic over a statically resolved backend. The
`#[magetypes]` caller supplies the concrete target features. An inline attribute
alone does not do that. `incant!` runs once per row, outside the pixel loop.
A performance-sensitive extracted helper can use `#[inline(always)]` when
required by measured codegen; it still needs the generated caller.

The tests execute empty rows, partial rows, unequal buffer lengths, and multiple
pixels. [ISA quirks](@/magetypes/isa-quirks.md) apply to arithmetic; this is not
a blanket promise of cross-ISA floating-point bit identity.

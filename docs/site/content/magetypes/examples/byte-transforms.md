+++
title = "Byte Transforms: Complete Dispatch"
description = "The actual zenwebp add-green body in a generated target-feature context"
weight = 7
+++

The production chain in `zen/zenwebp` at
`20898b7deaa4740c3584bd74831393ab3e7d6a36` is
[`apply_subtract_green_transform` → `incant!` → NEON/WASM entry](https://github.com/imazen/zenwebp/blob/20898b7deaa4740c3584bd74831393ab3e7d6a36/src/decoder/lossless_transform.rs#L759)
→ [`add_green_portable<T: U8x16Backend>`](https://github.com/imazen/zenwebp/blob/20898b7deaa4740c3584bd74831393ab3e7d6a36/src/decoder/lossless_transform_simd.rs#L682).
The actual portable body is scalar-unrolled Rust; it does not use the vector
shift/reinterpret sequence previously shown on this page. LLVM can vectorize
it within the target-feature context, but the token bound alone does not
establish that context.

This standalone adaptation keeps that body, adds an inline annotation, and
consolidates forwarding functions into a macro for V3/NEON/WASM/scalar. It omits
the production x86 SSE2 specialization. The full callable chain is:

```rust
use archmage::prelude::*;

#[inline(always)]
fn add_green_portable<T: magetypes::simd::backends::U8x16Backend>(
    _token: T,
    image_data: &mut [u8],
) {
    // Process 4 pixels (16 bytes) at a time for autovectorization
    let (chunks, remainder) = image_data.as_chunks_mut::<16>();
    for chunk in chunks {
        let g0 = chunk[1];
        let g1 = chunk[5];
        let g2 = chunk[9];
        let g3 = chunk[13];
        chunk[0] = chunk[0].wrapping_add(g0);
        chunk[2] = chunk[2].wrapping_add(g0);
        chunk[4] = chunk[4].wrapping_add(g1);
        chunk[6] = chunk[6].wrapping_add(g1);
        chunk[8] = chunk[8].wrapping_add(g2);
        chunk[10] = chunk[10].wrapping_add(g2);
        chunk[12] = chunk[12].wrapping_add(g3);
        chunk[14] = chunk[14].wrapping_add(g3);
    }
    for pixel in remainder.as_chunks_mut::<4>().0 {
        pixel[0] = pixel[0].wrapping_add(pixel[1]);
        pixel[2] = pixel[2].wrapping_add(pixel[1]);
    }
}

#[magetypes(v3, neon, wasm128, scalar)]
fn add_green_entry(token: Token, rgba: &mut [u8]) {
    add_green_portable(token, rgba);
}

pub fn add_green(rgba: &mut [u8]) {
    incant!(add_green_entry(rgba), [v3, neon, wasm128, scalar])
}


let mut pixels = [250, 10, 20, 255, 1, 2, 3, 4];
add_green(&mut pixels);
assert_eq!(pixels, [4, 10, 30, 255, 3, 2, 5, 4]);
```

Data is packed RGBA8. Red and blue add green modulo 256; green and alpha remain
unchanged. The last 0–3 bytes do not form a complete pixel and are left untouched,
as in the production helper. Tests cover every length through two chunks and
a tail, including overflow. This is a reversible byte transform, not color
space conversion.

The [intrinsics browser](https://imazen.github.io/archmage/intrinsics/) is useful
when profiling justifies a per-ISA specialization. Keep the generated entry and
test the specialized output against this source algorithm.

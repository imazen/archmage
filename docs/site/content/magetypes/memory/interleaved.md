+++
title = "Interleaved Data"
weight = 3
+++

Choose layout from the operation. `zenblend` SrcOver uses one RGBA pixel per
f32x4 vector; it need not split channels. Gamut matrices in `zenpixels-convert`
operate on separate R/G/B vectors, so loading and deinterleaving matters.
Neither layout is universally best.

For four interleaved channels, `deinterleave_4ch([v0, v1, v2, v3])` returns
`[r, g, b, a]`. `interleave_4ch` reverses that mapping. Each vector retains the
same lane count. This reference exercise pins the channel order:

```rust
use archmage::prelude::*;
#[magetypes(define(f32x4), v3, neon, wasm128, scalar)]
fn channels_impl(token: Token, pixels: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let vectors = pixels.map(|p| f32x4::from_array(token, p));
    f32x4::deinterleave_4ch(vectors).map(|v| v.to_array())
}
pub fn channels(pixels: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    incant!(channels_impl(pixels), [v3, neon, wasm128, scalar])
}
assert_eq!(channels([[1.,2.,3.,4.], [5.,6.,7.,8.], [9.,10.,11.,12.], [13.,14.,15.,16.]]),
           [[1.,5.,9.,13.], [2.,6.,10.,14.], [3.,7.,11.,15.], [4.,8.,12.,16.]]);
```

`f32x4::transpose_4x4(&mut rows)` and `transpose_4x4_copy(rows)` expose the
four-by-four matrix operation directly. Wider channel operations can require
cross-half shuffles; inspect those costs before changing a production layout.

A four-channel primitive does not accept RGB triples by dropping alpha. Use a
three-channel loader or explicitly bounded scalar gather into arrays, as in
[the generic luma example](@/magetypes/dispatch/types-and-dispatch.md). Preserve
row strides and tail pixels; padding reads must be justified by the buffer API.

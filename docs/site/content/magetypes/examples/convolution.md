+++
title = "Convolution: Production Call Chain"
weight = 3
+++

This source walkthrough traces the actual `zen/zenresize` implementation at
`33c7c10c0a5cd07a299b61ce2e239d6cbae57a4d`. Follow the chain in order; these links include the definitions that a
standalone copy would otherwise omit.

1. [Streaming row processing](https://github.com/imazen/zenresize/blob/33c7c10c0a5cd07a299b61ce2e239d6cbae57a4d/src/streaming.rs#L2418).
2. [simd::filter_h_row_f32 and incant!](https://github.com/imazen/zenresize/blob/33c7c10c0a5cd07a299b61ce2e239d6cbae57a4d/src/simd/mod.rs#L48).
3. [Generated NEON/WASM filter_h_row_f32_impl selects channels](https://github.com/imazen/zenresize/blob/33c7c10c0a5cd07a299b61ce2e239d6cbae57a4d/src/simd/wide_kernels.rs#L28).
4. [filter_h_4ch<T: F32x4Backend> accumulates each weighted RGBA pixel](https://github.com/imazen/zenresize/blob/33c7c10c0a5cd07a299b61ce2e239d6cbae57a4d/src/simd/wide_kernels.rs#L50).
5. [F32WeightTable owns the coefficient/extent contract](https://github.com/imazen/zenresize/blob/33c7c10c0a5cd07a299b61ce2e239d6cbae57a4d/src/weights.rs).

Four channels use one f32x4 per pixel; three-channel and general-channel paths are separate algorithms. The bounds of the input, output and weight table must agree. The complete implementation includes those types; the previous shortened snippet omitted its scalar paths. Do not infer that a generic token alone enables features: the generated caller supplies the context.

For a complete runnable macro-plus-generic-helper example, see
[zenblend SrcOver](@/magetypes/examples/pixel-blending.md). For a complete slice
loop and scalar tail, see [zenfilters plane scaling](@/magetypes/examples/generic-kernels.md)
and [linear-srgb gamma decoding](@/magetypes/math/transcendentals.md).

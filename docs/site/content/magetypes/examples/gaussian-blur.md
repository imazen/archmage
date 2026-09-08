+++
title = "Gaussian Blur: Production Call Chain"
weight = 5
+++

This source walkthrough traces the actual `zen/zenpipe` implementation at
`12b468e3e0a8ef46e2eeb1c92d7b1a859e53036a`. Follow the chain in order; these links include the definitions that a
standalone copy would otherwise omit.

1. [SIMD dispatch functions](https://github.com/imazen/zenpipe/blob/12b468e3e0a8ef46e2eeb1c92d7b1a859e53036a/zenfilters/src/simd/mod.rs).
2. [gaussian_blur_plane_dispatch_simd generates NEON/WASM contexts and selects the algorithm](https://github.com/imazen/zenpipe/blob/12b468e3e0a8ef46e2eeb1c92d7b1a859e53036a/zenfilters/src/simd/wide_simd.rs#L266).
3. [gaussian_blur_fir_generic<T: F32x8Backend + F32x8Convert + Copy> performs the FIR passes](https://github.com/imazen/zenpipe/blob/12b468e3e0a8ef46e2eeb1c92d7b1a859e53036a/zenfilters/src/simd/wide_simd.rs#L288).
4. [stackblur_plane_generic handles the selected large-radius path](https://github.com/imazen/zenpipe/blob/12b468e3e0a8ef46e2eeb1c92d7b1a859e53036a/zenfilters/src/simd/wide_simd.rs#L376).
5. [FilterContext owns reusable scratch allocations](https://github.com/imazen/zenpipe/blob/12b468e3e0a8ef46e2eeb1c92d7b1a859e53036a/zenfilters/src/context.rs).

The full chain includes image dimensions, edge padding, kernel selection, and scratch-buffer ownership. These are required for a faithful blur example. The previous partial snippet omitted their definitions. Generated entry functions can call generic helpers directly; nested incant! also supports tier-specific helpers, so explicit suffixed calls are not universally required.

For a complete runnable macro-plus-generic-helper example, see
[zenblend SrcOver](@/magetypes/examples/pixel-blending.md). For a complete slice
loop and scalar tail, see [zenfilters plane scaling](@/magetypes/examples/generic-kernels.md)
and [linear-srgb gamma decoding](@/magetypes/math/transcendentals.md).

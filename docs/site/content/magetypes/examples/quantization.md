+++
title = "Quantization: Production Call Chain"
weight = 4
+++

This source walkthrough traces the actual `zen/zenjpeg` implementation at
`fad6a0afd2efed488f4048efe5112753e20412c7`. Follow the chain in order; these links include the definitions that a
standalone copy would otherwise omit.

1. [QuantTableSimd and ZeroBiasSimd define the 8×8 block and threshold inputs](https://github.com/imazen/zenjpeg/blob/fad6a0afd2efed488f4048efe5112753e20412c7/zenjpeg/src/foundation/simd_types.rs).
2. [quantize_block calls incant!(mage_quantize_block(...))](https://github.com/imazen/zenjpeg/blob/fad6a0afd2efed488f4048efe5112753e20412c7/zenjpeg/src/foundation/simd_types.rs#L476).
3. [The macro generates the f32x8/i32x8 quantization body](https://github.com/imazen/zenjpeg/blob/fad6a0afd2efed488f4048efe5112753e20412c7/zenjpeg/src/foundation/simd_types.rs#L402).

The body scales coefficients, forms a threshold mask, calls to_i32_round(), bitcasts the mask to i32, blends, and stores natural-order coefficients. Check the coefficient range and rounding contract before claiming identical results on all ISAs. A separately named quantize_block_with_zero_bias_simd in quant/mod.rs is currently a scalar loop; its name alone is not evidence of this dispatch chain.

For a complete runnable macro-plus-generic-helper example, see
[zenblend SrcOver](@/magetypes/examples/pixel-blending.md). For a complete slice
loop and scalar tail, see [zenfilters plane scaling](@/magetypes/examples/generic-kernels.md)
and [linear-srgb gamma decoding](@/magetypes/math/transcendentals.md).

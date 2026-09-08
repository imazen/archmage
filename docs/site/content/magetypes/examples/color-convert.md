+++
title = "Color Conversion: Verify the Actual Path"
weight = 6
+++

This source walkthrough traces the actual `zen/zenjpeg` implementation at
`fad6a0afd2efed488f4048efe5112753e20412c7`. Follow the chain in order; these links include the definitions that a
standalone copy would otherwise omit.

1. [rgb_to_ycbcr_planes validates dimensions, allocates output planes and iterates batches](https://github.com/imazen/zenjpeg/blob/fad6a0afd2efed488f4048efe5112753e20412c7/zenjpeg/src/color/ycbcr.rs#L214).
2. [simd::rgb_to_ycbcr_x4 performs scalar FMA and rounds/clamps each lane](https://github.com/imazen/zenjpeg/blob/fad6a0afd2efed488f4048efe5112753e20412c7/zenjpeg/src/color/ycbcr.rs#L150).

The current function converts interleaved RGB8 into separate Y/Cb/Cr byte planes. Despite the helper module name simd, this path is scalar arithmetic. The previous float-plane magetypes example did not describe this production chain and has been removed. For a complete, current dispatched color-transfer example, use the linear-srgb gamma chain in the transcendental guide.

For a complete runnable macro-plus-generic-helper example, see
[zenblend SrcOver](@/magetypes/examples/pixel-blending.md). For a complete slice
loop and scalar tail, see [zenfilters plane scaling](@/magetypes/examples/generic-kernels.md)
and [linear-srgb gamma decoding](@/magetypes/math/transcendentals.md).

+++
title = "Real-World Examples"
description = "Production patterns for writing cross-platform SIMD with magetypes generics"
sort_by = "weight"
weight = 9

[extra]
sidebar = true
+++

Start with complete, compiled call chains adapted from real crates in `zen/`:

- [zenfilters: image-plane scaling](@/magetypes/examples/generic-kernels.md) — generated contexts, array chunks and scalar tail.
- [zenblend: SrcOver](@/magetypes/examples/pixel-blending.md) — a generated caller and a reusable generic backend helper.
- [zenwebp: add-green](@/magetypes/examples/byte-transforms.md) — the actual scalar-unrolled integer body inside generated contexts.
- [linear-srgb: gamma decoding](@/magetypes/math/transcendentals.md) — vector approximation and scalar tail, with their accuracy distinction.

Each page gives a pinned source revision, the production call chain, and any
adaptations required to make the excerpt standalone. The snippets are compiled
and exercised in `magetypes/tests/doc_examples.rs`.

The [convolution](@/magetypes/examples/convolution.md),
[Gaussian blur](@/magetypes/examples/gaussian-blur.md),
[quantization](@/magetypes/examples/quantization.md), and
[color conversion](@/magetypes/examples/color-convert.md) pages are production
source walkthroughs. They link the complete types and call sites instead of
presenting partial algorithms with undefined helpers as runnable examples.

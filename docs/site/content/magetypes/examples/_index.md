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
adaptations required to make the excerpt standalone. Rust fences are compiled directly by `xtask/check_docs.py`; the existing
`magetypes/tests/doc_examples.rs` adds numerical regression cases.

- [Generic types and const modes](@/magetypes/dispatch/types-and-dispatch.md) — zenanalyze, zenavif, and zenpixels-convert specialization.
- [Production codec and filter walkthroughs](@/magetypes/examples/convolution.md) — convolution, blur, quantization, and color conversion in one place.
- [Coverage and suspicious gaps](@/magetypes/examples/coverage.md) — what actual zen usage supports, and what it does not.

+++
title = "1. Attune a Kernel"
description = "Generate a callable dispatcher around a complete loop"
weight = 1
+++

`#[attune(dispatch)]` gives your function a normal call interface and generates
its CPU-specific implementations. Put the whole loop in that function.

*Legacy equivalent:* `#[autoversion(v3, neon, wasm128, scalar)]` provides the
same dispatcher role; generated variant names and visibility differ.

## Install the beta

This guide uses the 0.9.31-beta API. Select the prerelease explicitly:

```toml
[dependencies]
archmage = "=0.9.31-beta"
```

## Write and call a kernel

```rust
#![forbid(unsafe_code)]
use archmage::prelude::*;

#[attune(dispatch)]
pub fn scale(values: &mut [f32], factor: f32) {
    for value in values {
        *value *= factor;
    }
}

let mut values = [2.0, 4.0, 6.0];
scale(&mut values, 0.5);
assert_eq!(values, [1.0, 2.0, 3.0]);
```

Callers pass ordinary data. They do not need to select a tier, summon a token,
or write `unsafe`.

[Compare the source with its expansion](@/attunement/expansions.md#dispatcher)
to see the public dispatcher and private feature bodies.

Archmage generates implementations for V3 (AVX2 and FMA), NEON, WASM SIMD128,
and scalar, omitting variants for other architectures. The dispatcher chooses an
available tier and calls its body. Scalar is the fallback. AVX-512 is an
explicit addition, not part of this default set.

The loop is still ordinary Rust: LLVM decides whether to vectorize it.
[Explicit vectors](@/attunement/vectors.md) give you lane-wise operations when
you want to write them yourself. Neither form guarantees a speedup; measure the
complete kernel on your inputs.

## Keep the boundary around the work

Call `scale` once for the slice. Calling a dispatcher once per element adds
selection and call boundaries to the inner loop. Helpers inside the selected
body can keep that context and be eligible for inlining.

**Next:** [Stay in context](@/attunement/helpers.md).

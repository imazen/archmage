+++
title = "3. Add Vectors"
description = "Use Magetypes inside an attuned kernel and handle its tail"
weight = 3
+++

Add Magetypes from the same beta branch so both crates use the same token types:

```toml
[dependencies]
archmage = { git = "https://github.com/imazen/archmage", branch = "release/0.9.31-beta" }
magetypes = { git = "https://github.com/imazen/archmage", branch = "release/0.9.31-beta" }
```

`define(f32x8)` gives each generated body an eight-lane vector alias specialized
for its tier. Obtain that tier's proof locally with `Token::from_context()`.
The caller still passes only data.

*Legacy equivalent:* `#[magetypes(define(f32x8), ...)]` supplies the aliases and
variants; an `incant!` dispatcher provides the ordinary calling interface.

```rust
#![forbid(unsafe_code)]
use archmage::prelude::*;

#[attune(dispatch, define(f32x8))]
pub fn scale(values: &mut [f32], factor: f32) {
    let token = Token::from_context();
    let multiplier = f32x8::splat_t(token, factor);
    let (chunks, tail) = f32x8::partition_slice_mut_t(token, values);

    for chunk in chunks {
        (f32x8::load_t(token, chunk) * multiplier).store(chunk);
    }
    for value in tail {
        *value *= factor;
    }
}

// Exercise an empty slice, a short slice, an exact chunk, and a chunk with a tail.
for len in [0, 3, 8, 11] {
    let mut values = vec![2.0; len];
    scale(&mut values, 0.5);
    assert_eq!(values, vec![1.0; len]);
}
```

`Token` is a placeholder that the generated family replaces with its concrete
tier. `from_context()` uses the enclosing feature guarantee without another CPU
probe. Magetypes' `_t(token, ...)` constructors still require that proof in this
beta; short constructor names have not become token-free.

The slice partition gives you complete eight-element arrays and a remaining
slice. Process both. `f32x8` always has eight logical lanes here, including on the
scalar path; it does not automatically widen when the CPU supports more lanes.

For an image with row padding, call the kernel on each row's active slice. Keep
stride and padding handling in the row traversal rather than treating the whole
allocation as contiguous pixels.

## Continue when you need more

- [Memory operations](@/magetypes/memory/load-store.md): arrays, chunks, and bounds.
- [Numerical precision](@/magetypes/math/precision.md): choose an operation's numerical contract.
- [Full attunement reference](https://github.com/imazen/archmage/blob/release/0.9.31-beta/docs/ATTUNE.md): explicit AVX-512 gates, generic signatures, proof wrappers, and output visibility.

The shared Magetypes chapters include examples using the existing spellcraft
attributes. The vector operations are the same; the attuned kernel above is the
starting pattern for this path.

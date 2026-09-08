+++
title = "#[autoversion]"
weight = 6
+++

`#[autoversion]` generates feature-enabled variants and a callable dispatcher
for an ordinary Rust loop. Use it when you want LLVM to vectorize scalar source;
use `#[magetypes]` when the body explicitly names token-gated vector types.

This is the four-channel swizzle body from
`zenpixels-convert/src/convert_kernels.rs` (`swizzle_bgra_rgba_2bytes`), adapted
to use fixed array chunks and an explicit scalar fallback tier. It moves channel
bits, so the same operation applies to u16 samples and f16 bit representations.

```rust
use archmage::autoversion;

#[autoversion(v3, neon, wasm128, scalar)]
fn swizzle(src: &[[u16; 4]], dst: &mut [[u16; 4]]) {
    assert_eq!(src.len(), dst.len());
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = [s[2], s[1], s[0], s[3]];
    }
}
let src = [[1, 2, 3, 4], [5, 6, 7, 8]];
let mut dst = [[0; 4]; 2];
swizzle(&src, &mut dst);
assert_eq!(dst, [[3, 2, 1, 4], [7, 6, 5, 8]]);
```

The public function keeps the original name. Unlike `#[magetypes]`, you do not
need to write its public `incant!` dispatcher yourself. The macro offers LLVM
more features; it does not guarantee vectorization, reassociate floating-point
reductions, or automatically turn a sum into FMA. Inspect assembly for the
actual data layout and supported baseline.

The other major zen use is `zenanalyze`'s palette/grayscale/statistics loops.
Keep feature selection and input-shape decisions outside inner loops. Const
specialization can remove mode checks, but multiplies compiled variants.

Explicit tier lists, feature gates, and modifiers follow the common tier
syntax. Advanced token-parameter forms exist for composition; the ordinary
application pattern shown here is tokenless. Do not copy tokenless variant
signatures into a token-first `incant!` family without checking its convention.

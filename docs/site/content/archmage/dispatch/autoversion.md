+++
title = "Auto-Vectorized Dispatch (#[attune(dispatch)])"
description = "Compile plain Rust loops across SIMD tiers with #[attune(dispatch)] (replaces #[autoversion])"
weight = 6
+++

`#[attune(dispatch)]` replaces `#[autoversion]`, which is superseded and queued
for deprecation in the next release.

`#[attune(dispatch)]` compiles an ordinary Rust loop across the portable SIMD
tiers (`v3`, `neon`, `wasm128`, `scalar`) and emits a callable dispatcher under
the function's original name. Use it when you want LLVM to auto-vectorize plain
scalar source; add `define(...)` (or use `#[attune(_*_t, define(...))]`) when
the body names Magetypes vectors explicitly.

This is the four-channel swizzle body from
`zenpixels-convert/src/convert_kernels.rs` (`swizzle_bgra_rgba_2bytes`), adapted
to fixed array chunks. Because it moves channel bits without arithmetic, the
same loop handles both `u16` samples and `f16` bit patterns:

```rust
use archmage::prelude::*;

#[attune(dispatch)]
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

No `SimdToken` placeholder parameter is required. The generated dispatcher keeps
the original signature and routes to private per-tier bodies at runtime.
`#[attune(dispatch)]` gives LLVM higher target features per copy; it does not
guarantee vectorization, reassociate floating-point reductions, or turn
sequential sums into FMA. Inspect assembly for your actual data layout and
supported baseline.

The other major zen use is `zenanalyze`'s palette, grayscale, and statistics
loops. Keep feature selection and input-shape checks outside inner loops. Const
specialization can remove mode branches inside the loop at the cost of extra
compiled copies.

## Methods, tiers, and exposed variants

Inherent methods with `&self` or `&mut self` work as written. A receiverless
associated function in an inherent `impl` needs `#[attune(dispatch, in_impl)]`.
Family and dispatcher generation is rejected inside trait implementations
because trait `impl` blocks cannot hold extra sibling functions; delegate from
the trait method to a free `#[attune(dispatch)]` function instead. An
`impl Trait` return type is also rejected on dispatchers because each tier body
would produce a distinct opaque type.

Default `dispatch` omits AVX-512; opt in with
`#[attune(dispatch, +v4(cfg(avx512)))]` when assembly inspection and benchmarks
justify a 512-bit copy. When tests or same-module callers also invoke suffixed
proof entries directly (`swizzle_v3_t(token, ...)`, `swizzle_scalar_t(ScalarToken, ...)`),
select them alongside the dispatcher with `#[attune(_*_t, dispatch)]` (see
[preserving callers of suffixed variants](@/attunement/reference.md#preserve-callers-of-suffixed-variants)).

## Legacy `#[autoversion]` spelling (queued for deprecation)

| Legacy spelling | `#[attune]` replacement | Notes |
|---|---|---|
| `#[autoversion] fn sum(data: &[f32]) -> f32` | `#[attune(dispatch)] fn sum(data: &[f32]) -> f32` | Default tiers are `v3`, `neon`, `wasm128`, `scalar` (legacy defaults also included unconditional `v4`) |
| `#[autoversion(v3, neon, wasm128, scalar)]` | `#[attune(dispatch)]` | Matches the unified default portable tier set |
| `#[autoversion] fn sum(_token: SimdToken, data: &[f32])` | `#[attune(dispatch)] fn sum(data: &[f32])` | Drop the `SimdToken` placeholder parameter |
| `#[autoversion(v4, v3, neon, wasm128, scalar)]` | `#[attune(dispatch, +v4(cfg(avx512)))]` | Gate V4 on the calling crate's `avx512` Cargo feature |
| Direct calls to legacy `sum_v3(token, data)` | `#[attune(_*_t, dispatch)]` calling `sum_v3_t(token, data)` (or `names(...)`) | `dispatch` alone keeps per-tier bodies private |

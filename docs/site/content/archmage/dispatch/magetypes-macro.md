+++
title = "#[magetypes] Macro"
weight = 3
+++

`#[magetypes]` generates concrete target-feature variants from one function.
`incant!` selects a variant at the public boundary. The macro belongs to
archmage; the vector types belong to magetypes. Start with the complete
[zenfilters gain kernel](@/archmage/getting-started/first-simd.md).

## Generic types, with optional aliases

The generic vector implementation is the same with either spelling:

| Inside a `#[magetypes]` body | Meaning |
|---|---|
| `f32x8::<Token>::load(token, chunk)` | Explicit generic type; import `magetypes::simd::generic::f32x8` |
| `f32x8::load(token, chunk)` with `define(f32x8)` | Macro injects a local alias to that same concrete vector |
| `type V = magetypes::simd::generic::f32x8<Token>;` | Explicit local alias, as used in older zen kernels |

Only the identifier `Token` is replaced. The macro does not change [`f32x8`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html)
into [`f32x16`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x16.html), invent `LANES`, or choose a different algorithm. [`f32x8`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html) always
has eight lanes, including on NEON and WASM. Use an explicit tier list whose
tokens implement the vector backend; the portable eight-lane family is
`v3, neon, wasm128, scalar`.

`define(f32x8, i32x8)` injects both aliases inside each function body. These
aliases do not introduce types in the function signature. For a vector
parameter or result, spell the generic type with `Token` explicitly. Empty
`define()` is a no-op. Aliases add no runtime operation.

## Function generics are preserved

Type and const parameters are independent of ISA specialization. Real uses:

- `zenavif`'s `yuv420_strip_kernel<S: YuvSample, P: StripPixel>` specializes
  sample depth and output representation.
- `zenanalyze`'s `accumulate_row_simd<const BT601: bool, const FULL: bool,
  const SKIN: bool, R: ChunkInput>` combines three mode switches with input types.
- `zenpixels-convert`'s HDR measurement uses `<const N: usize>` for RGB/RGBA.

See [generic and const-generic kernels](@/magetypes/dispatch/types-and-dispatch.md)
for a complete, tested call chain with both kinds of parameter. `incant!`
forwards the turbofish, for example `kernel::<3, Pixel>(args)`.
The macro specializes `Token`; an unrelated `T: F32x8Backend` remains generic.

## Choose the boundary and helpers

| Need | Pattern |
|---|---|
| Public entry from ordinary Rust | Default `#[magetypes(...)]`, dispatched by `incant!` |
| Reuse one algorithm across several kernels | Generated entry calling an inline backend-generic helper |
| Per-tier internal helper with explicit features | `#[magetypes(rite, ...)]`, called inside a matching feature context |
| One hand-tuned ISA variant | `#[arcane] fn name_<tier>(...)`, alongside the other generated variants |
| Ordinary scalar loop to offer LLVM for vectorization | `#[autoversion]` |

Default non-scalar variants use `#[arcane]`: a safe wrapper enters a function
with the tier's target features. `rite` instead applies `#[target_feature]`
and `#[inline]` directly. It cannot be called safely by a baseline dispatcher;
reserve it for feature-enabled internal callers. A `ScalarToken` variant
needs no target-feature boundary.

Backend-generic helpers are statically dispatched. Inlining brings their body
into the generated caller's feature context; an inline attribute alone does
not establish that context. See [boundaries](@/archmage/concepts/target-feature-boundaries.md).

## Tier selection

Use the same explicit list in generation and public dispatch. `v4(cfg(avx512))`
gates a tier on the **calling crate's** Cargo feature, which must forward the
dependency features. `v4(avx512)` is shorthand. `+tier` and `-tier` modify defaults;
prefer full lists in tutorials so the selected shapes and fallbacks are visible.
The `_v3` spelling is accepted as an alias for `v3`.

Use `scalar` for vector bodies: it supplies `ScalarToken`. `default` is a
fallback without a token and cannot substitute `Token` in a vector body.
Do not rely on the compatibility behavior that appends a fallback when omitted.

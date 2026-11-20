+++
title = "Vector Families (#[attune(_*_t, define(...))])"
description = "Per-tier vector specialization with #[attune(_*_t, define(...))] and #[attune(dispatch, define(...))] (replaces #[magetypes])"
weight = 3
+++

`#[attune(_*_t, define(...))]` and `#[attune(dispatch, define(...))]` replace
`#[magetypes]`, which is superseded and queued for deprecation in the next
release.

`#[attune]` generates concrete target-feature variants from one function and
specializes `Token` and `define(...)` vector aliases per tier. The attribute
belongs to `archmage`; the vector types belong to `magetypes`. Start with the
complete [zenfilters gain kernel](@/archmage/getting-started/first-simd.md).

```rust
#![forbid(unsafe_code)]
use archmage::prelude::*;

// Proof-entry family: generates scale_chunk_v3_t, _neon_t, _wasm128_t, _scalar_t.
#[attune(_*_t, define(f32x8))]
fn scale_chunk(token: Token, data: &mut [f32; 8], factor: f32) {
    (f32x8::load_t(token, data) * f32x8::splat_t(token, factor)).store(data);
}

// Combined dispatcher: callers pass only data; Token::from_context() proves each body.
#[attune(dispatch, define(f32x8))]
pub fn scale_slice(values: &mut [f32], factor: f32) {
    let token = Token::from_context();
    let v_factor = f32x8::splat_t(token, factor);
    let (chunks, tail) = f32x8::partition_slice_mut_t(token, values);
    for chunk in chunks {
        (f32x8::load_t(token, chunk) * v_factor).store(chunk);
    }
    for value in tail {
        *value *= factor;
    }
}

let mut block = [1.0f32; 8];
attuned!(scale_chunk(&mut block, 3.0));
assert_eq!(block, [3.0; 8]);

let mut values = [2.0f32; 11];
scale_slice(&mut values, 0.5);
assert_eq!(values, [1.0; 11]);
```

## Generic types, with optional aliases

The vector implementation is identical across all three body spellings:

| Inside an `#[attune]` family body | Meaning |
|---|---|
| `f32x8::load_t(token, chunk)` with `define(f32x8)` | Macro injects a body-local alias to `magetypes::simd::generic::f32x8<ConcreteToken>` |
| `f32x8::<Token>::load_t(token, chunk)` | Explicit generic type; import `magetypes::simd::generic::f32x8` |
| `type V = magetypes::simd::generic::f32x8<Token>;` | Explicit local alias |

Only the identifier `Token` is replaced per tier. `#[attune]` does not rewrite
[`f32x8`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html)
into [`f32x16`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x16.html),
invent `LANES`, or choose a different algorithm; `f32x8` always has eight lanes,
including on NEON and WASM. For a 16-lane AVX-512 tier alongside an 8-lane
default family, write two disjoint `#[attune]` declarations as shown in the
[per-tier width split](@/attunement/reference.md#per-tier-width-split).

`define(f32x8, i32x8)` injects both aliases inside each function body. Body
aliases are not in scope in the function signature; for a vector parameter or
return type on `_*` or `_*_t`, spell `magetypes::simd::generic::f32x8<Token>`
explicitly. Aliases add no runtime cost.

## Function generics are preserved

Type and const parameters are independent of ISA specialization:

- `zenavif`'s `yuv420_strip_kernel<S: YuvSample, P: StripPixel>` specializes
  sample depth and output representation.
- `zenanalyze`'s `accumulate_row_simd<const BT601: bool, const FULL: bool,
  const SKIN: bool, R: ChunkInput>` combines three mode switches with input types.
- `zenpixels-convert`'s HDR measurement uses `<const N: usize>` for RGB/RGBA.

See [generic and const-generic kernels](@/magetypes/dispatch/types-and-dispatch.md)
for a complete, tested call chain with both kinds of parameter. `attuned!`
forwards the turbofish (`attuned!(kernel::<3, Pixel>(args))`). The macro
specializes `Token`; an unrelated `T: F32x8Backend` remains generic.

## Choose the boundary and helpers

| Need | Pattern |
|---|---|
| Single public function called from ordinary Rust | `#[attune(dispatch, define(...))]` with `let token = Token::from_context();` |
| Separate `_t` proof entries dispatched with `attuned!` | `#[attune(_*_t, define(...))]` + `attuned!(kernel(args))` |
| Reuse one algorithm across several kernels | Generated entry calling an inline backend-generic helper |
| Per-tier internal helper with direct features | `#[attune(_*, define(...))]`, called inside an attuned caller via `attuned!` |
| One hand-tuned ISA variant | `#[attune] fn name_<tier>_t(...)` alongside `#[attune(_*_t, -_<tier>, ...)]` |
| Ordinary scalar loop for LLVM auto-vectorization | [`#[attune(dispatch)]`](@/archmage/dispatch/autoversion.md) |

`_t` proof wrappers enter a private `#[target_feature]` function after
authenticating the token. `_*` direct variants apply `#[target_feature]` and
`#[inline]` without a wrapper and can only be called from a covering feature
context. `ScalarToken` variants require no target-feature boundary.

Backend-generic helpers are statically dispatched. Inlining brings their body
into the generated caller's feature context; an `#[inline]` attribute alone does
not establish that context. See
[target-feature boundaries](@/archmage/concepts/target-feature-boundaries.md).

## Tier selection

Wildcards (`_*`, `_*_t`, `dispatch`, `all`) select the default portable tiers:
`v3`, `neon`, `wasm128`, and `scalar`. AVX-512 is never added implicitly; opt in
with `+v4(cfg(avx512))` (or `_v4_t(cfg(avx512))` for a standalone V4 proof
wrapper) and forward `avx512` to `archmage` and `magetypes` in `Cargo.toml`.
Remove unneeded tiers from a wildcard with `-_neon` or `-_scalar`.

## Legacy `#[magetypes]` spelling (queued for deprecation)

| Legacy spelling | `#[attune]` replacement | Notes |
|---|---|---|
| `#[magetypes(define(f32x8), v3, neon, wasm128, scalar)] fn f(token: Token, ...)` | `#[attune(_*_t, define(f32x8))] fn f(token: Token, ...)` | Proof variants are named `f_<tier>_t`; dispatch with `attuned!(f(...))` |
| `#[magetypes(...)]` + `pub fn f(...) { incant!(...) }` | `#[attune(dispatch, define(f32x8))] pub fn f(...)` | Omit `token: Token` from the signature and bind `let token = Token::from_context();` inside |
| `#[magetypes(rite, define(f32x8), v3, neon, wasm128, scalar)]` | `#[attune(_*, define(f32x8))]` | Emits direct `f_<tier>` target-feature helpers |
| `#[magetypes(define(i32x8), v3, neon, wasm128, -scalar)]` | `#[attune(_*_t, -_scalar, define(i32x8))]` | Pair with a hand-written `f_scalar_t(ScalarToken, ...)` |
| `#[magetypes(define(f32x16), v4(cfg(avx512)), v3, neon, wasm128, scalar)]` | `#[attune(_*_t, +v4(cfg(avx512)), define(f32x16))]` | Match with `[_v4(cfg(avx512)), _v3, _neon, _wasm128, _scalar]` in `attuned!` |

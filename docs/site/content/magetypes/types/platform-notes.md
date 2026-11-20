+++
title = "Platform Notes"
weight = 2
+++

[`f32x8<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html) denotes eight logical lanes. The token selects the implementation,
not the lane count. Use `#[attune]` to bind `Token` per tier and `attuned!` (or
`dispatch`) to enter it, as in the [first kernel](@/archmage/getting-started/first-simd.md).

| Shape | Common implementations |
|---|---|
| [`f32x4`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x4.html) | V2/V3, NEON, WASM, scalar |
| [`f32x8`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html) | V3, split NEON/WASM, scalar |
| [`f32x16`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x16.html) (`w512`) | Split V3/NEON/WASM, scalar; native V4 with `avx512` |

A stronger token does not automatically implement every backend trait.
Hardware feature implication and Rust trait implementation are separate facts.
When using a V3 shape from a V4 context, extract `token.v3()` explicitly.
Do not place V4 into an eight-lane tier list and assume the macro widens it;
pair `#[attune(_*_t, define(f32x8))]` with a separate
`#[attune(_v4_t(cfg(avx512)), define(f32x16))]` declaration and list
`[_v4(cfg(avx512)), _v3, _neon, _wasm128, _scalar]` in `attuned!`.

`w512` is enabled by default and provides logical 512-bit shapes. `avx512`
adds native AVX-512 support and implies `w512`. With default features disabled,
request `w512` explicitly if you use those shapes. Wider ARM/WASM polyfills
also depend on this logical-width feature.

Generic imports are portable; explicit calls to architecture-specific functions
may still require cfg guards. `attuned!` and `#[attune(dispatch)]` supply those
guards. Bare architecture-bound aliases can be convenient locally, but tutorials
use explicit generic types or `define` so the backend selection remains visible.

Use `-Ctarget-cpu=native` only for a known deployment CPU. `summon()` returns
`Option<Token>`, not `Some(true)`; `compiled_with()` is the separate query about
compile-time guarantees. A global feature guarantee can eliminate detection,
but is not a replacement for testing your baseline-dispatched build.

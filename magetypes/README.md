# magetypes [![CI](https://img.shields.io/github/actions/workflow/status/imazen/archmage/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/archmage/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/magetypes?style=flat-square)](https://crates.io/crates/magetypes) [![lib.rs](https://img.shields.io/crates/v/magetypes?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/magetypes) [![docs.rs](https://img.shields.io/docsrs/magetypes?style=flat-square)](https://docs.rs/magetypes) [![MSRV](https://img.shields.io/badge/MSRV-1.89-blue?style=flat-square)](https://github.com/imazen/archmage/blob/main/MSRV.md) [![license](https://img.shields.io/crates/l/magetypes?style=flat-square)](#license)

[Guide](https://imazen.github.io/archmage/) · [Intrinsics browser](https://imazen.github.io/archmage/intrinsics/) · [Archmage API](https://docs.rs/archmage/latest/archmage/) · [Magetypes API](https://docs.rs/magetypes/latest/magetypes/)

See [reusable generic kernels](https://imazen.github.io/archmage/magetypes/examples/generic-kernels/) and [ISA quirks and fixup costs](https://imazen.github.io/archmage/magetypes/isa-quirks/).

## Quick start

```toml
[dependencies]
magetypes = "0.9.27"
archmage  = "0.9.27"   # required: provides the macros + tokens magetypes uses
```

**Why both?** The vector *types* ([`f32x8`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html), [`u8x16`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.u8x16.html), …) come from `magetypes`, but the macros (`#[magetypes]`, `incant!`, `#[arcane]`, `#[autoversion]`, `#[rite]`) and the tokens (`Token`, `X64V3Token`, `NeonToken`, `ScalarToken`, …) come from `archmage` — every example here opens with `use archmage::prelude::*;`. So add `archmage` as a direct dependency. (`magetypes` does re-export it, so `magetypes::archmage::prelude::*` works with `magetypes` alone — but a direct `archmage` dep is the idiomatic path and what the examples assume.)

Default features (`std`, `w512`) are on. For `no_std + alloc`, use `default-features = false`. For native AVX-512 impls on x86-64, add `features = ["avx512"]` (implies `w512`). See [Cargo features](#cargo-features).

Then write **one** kernel that runs on AVX2, AVX-512, NEON, WASM SIMD128, or scalar — `#[magetypes]` generates the per-tier `#[target_feature]` contexts and `incant!` picks the best at runtime, all `#![forbid(unsafe_code)]`-compatible:

Adapted from the `zenfilters` plane-scaling kernel; the [complete production call chain and adaptation notes](https://imazen.github.io/archmage/magetypes/examples/generic-kernels/) include pinned source links.

```rust
use archmage::prelude::*;

#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
fn scale_plane_impl(token: Token, plane: &mut [f32], factor: f32) {
    // `f32x8` is in scope via `define` — resolves to `f32x8<X64V3Token>` in
    // the v3 variant, `f32x8<NeonToken>` in neon, etc. `Token` is likewise
    // substituted per tier for parameters and return types.
    let factor_v = f32x8::splat(token, factor);
    let (chunks, tail) = f32x8::partition_slice_mut(token, plane);
    for chunk in chunks {
        (f32x8::load(token, chunk) * factor_v).store(chunk);
    }
    for v in tail { *v *= factor; }
}

pub fn scale_plane(plane: &mut [f32], factor: f32) {
    incant!(scale_plane_impl(plane, factor))
}
```

That's it. One algorithm, every platform. `#[magetypes]` generates five `#[arcane]`-wrapped variants (`_v4`, `_v3`, `_neon`, `_wasm128`, `_scalar`), each with its own `#[target_feature]`. `define(f32x8)` injects `type f32x8 = ::magetypes::simd::generic::f32x8<Token>;` at the top of each variant — no boilerplate alias line. Multiple types: `define(f32x8, u8x16, i16x8)`. `incant!` picks the highest available at runtime.

**`#[magetypes]` IS the `#[arcane]` wrapper generator.** Do not write per-tier `#[arcane]` wrappers around a generic kernel by hand — the macro already does that. This is the single biggest source of confusion, so it bears repeating.

### Imports at a glance

| You want… | Import |
|---|---|
| The macros + tokens (`#[magetypes]`, `incant!`, `Token`, `X64V3Token`, …) | `use archmage::prelude::*;` |
| A fixed-width type explicitly | `use magetypes::simd::f32x8;` (8 lanes everywhere, polyfilled off-x86) |
| Inside a `#[magetypes]` body | nothing extra — `define(f32x8)` injects the alias per tier |
| The generic SIMD types + `SimdToken` | `use magetypes::prelude::*;` — then name the token: [`f32x8::<X64V3Token>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html) |

The vector types live under `magetypes::simd::*` — there are no root re-exports (the surface is kept stable during development), so reach for `magetypes::simd::f32x8`, not `magetypes::f32x8`.

### Load / store are **unaligned**

`f32x8::load(token, &[f32; 8])` and `.store(&mut [f32; 8])` take **fixed-size array references** and perform an **unaligned** transfer — on x86-64 they lower to `_mm256_loadu_ps` / `_mm256_storeu_ps`. A `&[f32; 8]` only guarantees 4-byte (`f32`) alignment, and that's all that's required; there is **no 32-byte SIMD-alignment precondition**. Consequently `partition_slice_mut` (which reinterprets an arbitrary `&mut [f32]` as `&mut [[f32; 8]]` chunks) is sound on any slice — the bulk chunks feed straight into `load`/`store`. Don't reach for aligned allocators or hand-padded buffers; window your slice and go.


## Generics and generated variants

`#[magetypes]` is an [archmage attribute](https://docs.rs/archmage/latest/archmage/attr.magetypes.html).
The [magetypes crate](https://docs.rs/magetypes/latest/magetypes/) supplies vector
types such as [`f32x4<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x4.html) and [`f32x8<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html). `define(f32x8)` creates a local alias;
explicit [`f32x8::<Token>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html) uses the same implementation. Type and const generics
can remain on the generated function, as in zenavif's sample/pixel kernels and
zenanalyze's const-mode/input-type kernels.

Read the [complete generic specialization example](https://imazen.github.io/archmage/magetypes/dispatch/types-and-dispatch/).
For reusable backend-generic helpers, keep the generated feature-enabled caller:
an inline attribute or token argument alone does not enable that context.

| Work | Pattern |
|---|---|
| Portable vector kernel | `#[magetypes]` + public `incant!` |
| Reusable algorithm | Generated entry → inline backend-generic helper |
| Ordinary loop offered to LLVM for vectorization | `#[autoversion]` |
| Hand-tuned ISA entry | `#[arcane]` |
| Matched internal intrinsic helper | `#[rite]` |

`stub` has been removed. `incant!` handles cross-architecture call-site guards.
The reference forms `with token` and `without token` remain implemented; they
respectively select by the held token's exact type and call a tokenless variant
in a matching macro-managed context. See [dispatch](https://imazen.github.io/archmage/archmage/dispatch/incant/).

## Tokens from an existing feature context

When a helper already has target features, `from_context()` constructs a token
without runtime detection. Rust checks that the caller's features cover the
token's requirements. It is not a baseline-callable unchecked constructor.

```rust
use archmage::prelude::*;
#[rite(v3)]
fn helper() -> bool {
    let _token = X64V3Token::from_context();
    true
}
#[arcane]
fn entry(_token: X64V3Token) -> bool { helper() }
#[cfg(target_arch = "x86_64")]
if let Some(token) = X64V3Token::summon() { assert!(entry(token)); }
```

This is a repository addition after 0.9.28. See
[from_context and token extraction](https://imazen.github.io/archmage/archmage/getting-started/tokens/).
Use `.v3()` to extract a V3 token from a stronger proof; `as_x64v3()` instead
checks whether the held token is exactly a V3 token.

## Features and numerical contracts

Rust 1.89 is the minimum supported version. Archmage macros are always included;
its `macros` feature is a compatibility no-op. `std` is enabled by default.
Magetypes also defaults to `w512`, which supplies logical 512-bit types and
polyfills. Optional `avx512` adds native AVX-512 support; it does not detect the
running CPU. Follow the [feature-forwarding example](https://imazen.github.io/archmage/archmage/getting-started/installation/)
when exposing features from your own crate.

Logical width does not change with the selected ISA: [`f32x8`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html) stays eight lanes.
Use supported backend lists; do not assume every stronger token implements
every narrower backend. The [ISA quirks and fixups](https://imazen.github.io/archmage/magetypes/isa-quirks/)
explain NaNs, rounding, saturation, lane ordering, and measured repair costs.
[Transcendentals](https://imazen.github.io/archmage/magetypes/math/transcendentals/)
have a separate domain and precision discussion.

Compile the complete call chain, test supported tiers and scalar tails, and
inspect optimized code under your supported baseline. See
[testing](https://imazen.github.io/archmage/archmage/testing/dispatch-testing/) and
[production coverage](https://imazen.github.io/archmage/magetypes/examples/coverage/).

## License

MIT OR Apache-2.0

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenjxl-decoder] · [jxl-encoder] · [zenrav1e] · [rav1d-safe] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [zenmetrics] · [resamplescope-rs] |
| Pickers & ML | [zenanalyze] · [zenpredict] · [zenpicker] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · **magetypes** · [enough] · [whereat] · [cargo-copter]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zentiff
[zenpdf]: https://github.com/imazen/zenpdf
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenfilters
[zensally]: https://github.com/imazen/zensally
[zentone]: https://github.com/imazen/zentone
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[zenwasm]: https://github.com/imazen/zenwasm
[zentract]: https://github.com/imazen/zentract
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenmetrics]: https://github.com/imazen/zenmetrics
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[zenanalyze]: https://github.com/imazen/zenanalyze
[zenpredict]: https://github.com/imazen/zenanalyze
[zenpicker]: https://github.com/imazen/zenanalyze
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go

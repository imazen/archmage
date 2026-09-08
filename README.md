# archmage [![CI](https://img.shields.io/github/actions/workflow/status/imazen/archmage/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/archmage/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/archmage?style=flat-square)](https://crates.io/crates/archmage) [![lib.rs](https://img.shields.io/crates/v/archmage?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/archmage) [![docs.rs](https://img.shields.io/docsrs/archmage?style=flat-square)](https://docs.rs/archmage) [![MSRV](https://img.shields.io/badge/MSRV-1.89-blue?style=flat-square)](https://github.com/imazen/archmage/blob/main/MSRV.md) [![license](https://img.shields.io/crates/l/archmage?style=flat-square)](#license)

[Guide](https://imazen.github.io/archmage/) · [Intrinsics browser](https://imazen.github.io/archmage/intrinsics/) · [Archmage API](https://docs.rs/archmage/latest/archmage/) · [Magetypes API](https://docs.rs/magetypes/latest/magetypes/)

Archmage lets you write SIMD code in Rust **without `unsafe`** — your crate keeps `#![forbid(unsafe_code)]` while calling intrinsics directly. It works on x86-64, AArch64, and WASM, is `no_std + alloc` (with `std` on by default for runtime CPU detection), and depends only on [`archmage-macros`](https://crates.io/crates/archmage-macros) and [`safe_unaligned_simd`](https://crates.io/crates/safe_unaligned_simd). You pick a CPU tier, prove it's present once with `summon()`, and the type system keeps every intrinsic call sound.

## Image planes and audio buffers

Use `archmage` with the [magetypes vector crate](https://docs.rs/magetypes/latest/magetypes/) for portable vector kernels:

```toml
[dependencies]
archmage = "0.9.28"
magetypes = "0.9.28"
```

Process an image plane (exposure) or an audio buffer (gain), including a short
scalar tail. The vector type is generic over the token selected by `#[magetypes]`;
`incant!` chooses the CPU tier once outside the loop. No manual per-tier wrappers
or raw pointers are needed.

Adapted from the `zenfilters` plane-scaling kernel; the [complete production call chain and adaptation notes](https://imazen.github.io/archmage/magetypes/examples/generic-kernels/) include pinned source links.

```rust
#![forbid(unsafe_code)]
use archmage::prelude::*;

#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn gain_impl(token: Token, plane: &mut [f32], gain: f32) {
    let factor = f32x8::splat(token, gain);
    let (chunks, tail) = f32x8::partition_slice_mut(token, plane);
    for chunk in chunks {
        (f32x8::load(token, chunk) * factor).store(chunk);
    }
    for value in tail {
        *value *= gain;
    }
}

pub fn apply_gain(plane: &mut [f32], gain: f32) {
    incant!(gain_impl(plane, gain), [v3, neon, wasm128, scalar])
}
```

For ISA-specific kernels, use `#[arcane(import_intrinsics)]` at the entry and
`#[rite(import_intrinsics)]` for helpers. The [intrinsics browser](https://imazen.github.io/archmage/intrinsics/)
lists available reference-based memory operations. See the [guide](https://imazen.github.io/archmage/)
for both approaches, and [reusable generic kernels](https://imazen.github.io/archmage/magetypes/examples/generic-kernels/)
for sharing algorithms across vector backends.


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

[zenbench] · **archmage** · [magetypes] · [enough] · [whereat] · [cargo-copter]

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
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go

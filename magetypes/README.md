# magetypes [![CI](https://img.shields.io/github/actions/workflow/status/imazen/archmage/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/archmage/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/magetypes?style=flat-square)](https://crates.io/crates/magetypes) [![lib.rs](https://img.shields.io/crates/v/magetypes?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/magetypes) [![docs.rs](https://img.shields.io/docsrs/magetypes?style=flat-square)](https://docs.rs/magetypes) [![MSRV](https://img.shields.io/badge/MSRV-1.89-blue?style=flat-square)](https://github.com/imazen/archmage/blob/main/MSRV.md) [![license](https://img.shields.io/crates/l/magetypes?style=flat-square)](#license)

[Guide](https://imazen.github.io/archmage/) · [Intrinsics browser](https://imazen.github.io/archmage/intrinsics/) · [Archmage API](https://docs.rs/archmage/latest/archmage/) · [Magetypes API](https://docs.rs/magetypes/latest/magetypes/)

magetypes provides SIMD vector types with ordinary Rust operators: `f32x8`,
`i32x4`, `u8x16` and the rest of the 128-, 256- and 512-bit vectors of floats and
integers. Each vector carries an [archmage](https://crates.io/crates/archmage)
token proving its CPU features, so one kernel compiles for AVX2, AVX-512, NEON,
WASM SIMD128 and scalar, and using it takes no `unsafe` in your code.

See [reusable generic kernels](https://imazen.github.io/archmage/magetypes/examples/generic-kernels/) and [ISA quirks and fixup costs](https://imazen.github.io/archmage/magetypes/isa-quirks/).

## Quick start

```toml
[dependencies]
magetypes = "0.9.30"
archmage  = "0.9.30"   # provides the macros and tokens magetypes uses
```

The vector types ([`f32x8`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html), [`u8x16`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.u8x16.html), …) come from magetypes; the macros (`#[magetypes]`, `incant!`, `#[arcane]`, `#[autoversion]`, `#[rite]`) and the tokens (`X64V3Token`, `NeonToken`, `ScalarToken`, …) come from archmage. magetypes re-exports archmage, but the macros expand to `archmage::` paths, so add it as a direct dependency.

Default features (`std`, `w512`) are on. For `no_std + alloc`, set `default-features = false` on both crates. For native AVX-512 on x86-64, give your crate an `avx512` feature that forwards to `magetypes/avx512` (which implies `w512` and `archmage/avx512`): `#[magetypes]` and `incant!` compile `v4` variants and dispatch arms only when your crate's `avx512` feature is on. See [Features and numerical contracts](#features-and-numerical-contracts).

Write the kernel once: `#[magetypes]` generates a feature-enabled variant per
tier, and `incant!` picks the best one at run time. This is adapted from the
`zenfilters` plane-scaling kernel; the [complete production call chain and adaptation notes](https://imazen.github.io/archmage/magetypes/examples/generic-kernels/) include pinned source links.

```rust
use archmage::prelude::*;

#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
fn scale_plane_impl(token: Token, plane: &mut [f32], factor: f32) {
    // `define(f32x8)` makes `f32x8` mean `f32x8<X64V3Token>` in the v3
    // variant, `f32x8<NeonToken>` in neon, and so on. `Token` is replaced
    // the same way.
    let factor_v = f32x8::splat_t(token, factor);
    let (chunks, tail) = f32x8::partition_slice_mut_t(token, plane);
    for chunk in chunks {
        (f32x8::load_t(token, chunk) * factor_v).store(chunk);
    }
    for v in tail { *v *= factor; }
}

pub fn scale_plane(plane: &mut [f32], factor: f32) {
    incant!(scale_plane_impl(plane, factor))
}

let mut plane = [2.0; 11];
scale_plane(&mut plane, 0.5);
assert_eq!(plane, [1.0; 11]);
```

`#[magetypes]` generates `#[arcane]`-wrapped `_v3`, `_neon` and `_wasm128`
variants, each compiled with its tier's features, a plain `_scalar` fallback, and
`_v4` when your crate's `avx512` feature is on. `define(f32x8)` adds
`type f32x8 = ::magetypes::simd::generic::f32x8<Token>;` to each variant; list
several types as `define(f32x8, u8x16, i16x8)`. Don't write per-tier `#[arcane]`
wrappers around a `#[magetypes]` kernel: the macro already generates them.

Constructors take the token first and end in `_t` (`splat_t`, `load_t`).
magetypes 0.9.30 deprecates the older names ahead of a planned 0.10 change; the
[migration guide](https://github.com/imazen/archmage/blob/main/docs/TOKEN-CONSTRUCTOR-MIGRATION.md)
has the mapping.

### Imports at a glance

| You want… | Import |
|---|---|
| The macros and tokens (`#[magetypes]`, `incant!`, `X64V3Token`, …) | `use archmage::prelude::*;` |
| Vectors inside a `#[magetypes]` body | Nothing extra: `define(f32x8)` adds the alias per tier |
| Generic vectors for a token you name | `use magetypes::prelude::*;`, then [`f32x8::<X64V3Token>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html) |
| The platform's natural-width vector | `use magetypes::simd::f32x8;`, an alias for `f32x8<X64V3Token>` on x86-64, `f32x8<NeonToken>` on AArch64 and `f32x8<Wasm128Token>` on WASM |

The vector types live under `magetypes::simd`; there are none at the crate root.

### Loads and stores are unaligned

`f32x8::load_t(token, &[f32; 8])` and `.store(&mut [f32; 8])` take array
references and do unaligned transfers (`_mm256_loadu_ps` and `_mm256_storeu_ps`
on x86-64). A `&[f32; 8]` only guarantees `f32` alignment, and that is all they
need, so `partition_slice_mut_t`, which splits any `&mut [f32]` into `[f32; 8]`
chunks and a tail, works on any slice. No aligned allocator or padding is needed.

## What's included

- 30 vector types: `f32`, `f64`, and signed and unsigned 8- to 64-bit integers,
  at 128, 256 and 512 bits (`f32x4` through `u64x8`).
- Arithmetic and bitwise operators (`/` on float vectors only), comparisons and
  `blend`, `min`, `max`, `abs`, rounding, `sqrt`, `mul_add`, and reductions such
  as `reduce_add`.
- Reciprocals and transcendentals (`exp2`, `exp`, `ln`, `log2`, `log10`, `pow`,
  `cbrt`) in documented precision tiers.
- Conversions between float and integer lanes, widening and narrowing, f16, and
  pixel helpers such as `to_u8` and the RGBA stores.
- Interleave, deinterleave and transpose.
- Raw register interop with `core::arch` (`raw()`, `from_raw_t`), and
  bounds-checked AVX-512 gather and scatter.

## Generics and generated variants

`#[magetypes]` is an [archmage attribute](https://docs.rs/archmage/latest/archmage/attr.magetypes.html).
`define(f32x8)` is shorthand for [`f32x8<Token>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html), and
type and const generics can stay on the generated function, as in zenavif's
sample and pixel kernels and zenanalyze's mode and input-type kernels. A
backend-generic helper still has to be called from a generated, feature-enabled
function: marking it `#[inline]` or passing it a token doesn't compile it with
the features. Read the [complete generic specialization example](https://imazen.github.io/archmage/magetypes/dispatch/types-and-dispatch/).

| Work | Pattern |
|---|---|
| Portable vector kernel | `#[magetypes]` + public `incant!` |
| Reusable algorithm | Generated entry → inline backend-generic helper |
| Ordinary loop offered to LLVM for vectorization | `#[autoversion]` |
| Hand-tuned ISA entry | `#[arcane]` |
| Matched internal intrinsic helper | `#[rite]` |

## Features and numerical contracts

Rust 1.89 is the minimum supported version. `std` is on by default and provides
runtime CPU detection; without it, archmage's `summon()` knows only the features
enabled at compile time. Magetypes also defaults to `w512`, which supplies logical
512-bit types and polyfills. Optional `avx512` adds native AVX-512 support; it
does not detect the running CPU. `incant!` and `#[magetypes]` compile their `v4`
and `v4x` variants and dispatch arms only when your own crate has a feature named
`avx512` (`#[autoversion]` always generates its `v4` variant); follow the
[feature-forwarding example](https://imazen.github.io/archmage/archmage/getting-started/installation/)
to define one.

Logical width does not change with the selected ISA: [`f32x8`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html) stays eight lanes.
List only tiers whose tokens implement the vector types you use; a stronger token
does not implement every narrower backend. `mul_add` rounds once where the hardware fuses (x86
v3/v4, NEON) and twice on the scalar backend and strict WASM;
`mul_add_portable` rounds once everywhere, in software where needed. The [ISA quirks and fixups](https://imazen.github.io/archmage/magetypes/isa-quirks/)
explain NaNs, rounding, saturation, lane ordering, and measured repair costs.
[Transcendentals](https://imazen.github.io/archmage/magetypes/math/transcendentals/)
have a separate domain and precision discussion.

Every native-backend vector has `raw()`, `from_raw_t(token, raw)` and
`from_raw(raw)` for handing registers to and from `core::arch` intrinsics;
`from_raw` needs a matching target-feature context. With `X64V4Token` and the
`avx512` feature, `u32x16`, `i32x16` and `f32x16` also have bounds-checked
[gather and scatter](https://imazen.github.io/archmage/magetypes/memory/gather-scatter/).

Compile the complete call chain, test supported tiers and scalar tails, and
inspect optimized code under your supported baseline. See
[testing](https://imazen.github.io/archmage/archmage/testing/dispatch-testing/) and
[production coverage](https://imazen.github.io/archmage/magetypes/examples/coverage/).

## Safety

Using magetypes takes no `unsafe` in your code, so your crate can keep
`#![forbid(unsafe_code)]`. Every vector carries the archmage token it was built
with, so its operations only run where the CPU's features are proven, and no
constructor works without that proof.

Inside, magetypes stacks its own proofs on archmage's and on Rust's. The compiler
checks each x86 and NEON intrinsic in its backends against the features of the
token it runs under, and compile-time assertions check the size and layout of
every reinterpretation of memory. What the compiler can't check is memory access
through a pointer: a few one-line `unsafe` blocks, all in one internal module,
that load, store, gather and scatter vector storage, each stating what it relies
on. The [magetypes safety model](https://imazen.github.io/archmage/magetypes/safety/)
has the details.

## Limits

- Vector types wider than the hardware run as two or four native operations; an
  `f32x8` on NEON is two `f32x4`s.
- SIMD tiers cover x86-64, AArch64 and WASM; other targets run the scalar
  backend. Native AVX-512 needs your crate's `avx512` feature.
- Transcendentals are approximations with documented error, and some
  floating-point results differ between backends; the ISA quirks page lists each
  difference.

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

# magetypes [![CI](https://img.shields.io/github/actions/workflow/status/imazen/archmage/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/archmage/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/magetypes?style=flat-square)](https://crates.io/crates/magetypes) [![lib.rs](https://img.shields.io/crates/v/magetypes?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/magetypes) [![docs.rs](https://img.shields.io/docsrs/magetypes?style=flat-square)](https://docs.rs/magetypes) [![MSRV](https://img.shields.io/badge/MSRV-1.89-blue?style=flat-square)](https://github.com/imazen/archmage/blob/main/MSRV.md) [![license](https://img.shields.io/crates/l/magetypes?style=flat-square)](#license)

[Guide](https://imazen.github.io/archmage/) · [Intrinsics browser](https://imazen.github.io/archmage/intrinsics/) · [Archmage API](https://docs.rs/archmage/latest/archmage/) · [Magetypes API](https://docs.rs/magetypes/latest/magetypes/)

magetypes provides SIMD vector types with ordinary Rust operators: `f32x8`,
`i32x4`, `u8x16` and the rest of the 128-, 256- and 512-bit float and integer
vectors.

You write a kernel once, and it compiles for AVX2, AVX-512, NEON, WASM SIMD128
and scalar. Each vector carries an [archmage](https://crates.io/crates/archmage)
token proving its CPU features, so using it takes no `unsafe` in your code.

## Quick start

```toml
[dependencies]
magetypes = "0.9.30"
archmage  = "0.9.30"   # the macros and tokens

[features]
avx512 = ["archmage/avx512", "magetypes/avx512"]   # opt in to AVX-512
```

Multiply a buffer by a factor, using AVX-512, AVX2, NEON or WASM SIMD where
available:

```rust
use archmage::prelude::*;

#[magetypes(define(f32x8), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
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
    incant!(scale_plane_impl(plane, factor), [v4(cfg(avx512)), v3, neon, wasm128, scalar])
}

let mut plane = [2.0; 11];
scale_plane(&mut plane, 0.5);
assert_eq!(plane, [1.0; 11]);
```

- `#[magetypes]` compiles `scale_plane_impl` once per tier in its list: `v4`
  (AVX-512), `v3` (AVX2 and FMA), `neon`, `wasm128` and `scalar`. List several
  vector types as `define(f32x8, u8x16, i16x8)`.
- `v4(cfg(avx512))` compiles the AVX-512 copy only when your crate's `avx512`
  feature is on: the opt-in from the `Cargo.toml` above. `f32x8` stays eight
  lanes in that copy. For 512-bit vectors, use `f32x16`.
- `incant!` calls `summon()` for each tier, best first, and runs the first copy
  the CPU supports. Call it around your loop, as here, not inside it.
- Constructors ending in `_t` take the token as their first argument.

Don't write per-tier `#[arcane]` wrappers around a `#[magetypes]` kernel: the
macro already generates them.

The kernel is adapted from `zenfilters`;
[Reusable generic kernels](https://imazen.github.io/archmage/magetypes/examples/generic-kernels/)
links the production source. If you are upgrading, 0.9.30 deprecates the older
constructor names (`splat`, `load`, …) ahead of a planned 0.10 change: the
[migration guide](https://github.com/imazen/archmage/blob/main/docs/TOKEN-CONSTRUCTOR-MIGRATION.md)
has the mapping.

### Imports

| You want | Import |
|---|---|
| The macros and tokens (`#[magetypes]`, `incant!`, `X64V3Token`, …) | `use archmage::prelude::*;` |
| Vectors inside a `#[magetypes]` body | Nothing extra: `define(f32x8)` supplies them |
| Vectors for a token you name, as in [`f32x8::<X64V3Token>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html) | `use magetypes::prelude::*;` |
| The platform's own `f32x8` | `use magetypes::simd::f32x8;`, an alias for `f32x8<X64V3Token>` on x86-64, `f32x8<NeonToken>` on AArch64 and `f32x8<Wasm128Token>` on WASM |

The vector types live under `magetypes::simd`; there are none at the crate root.

### Loads and stores are unaligned

`load_t` and `store` take array references such as `&[f32; 8]` and do unaligned
transfers. Any slice works: `partition_slice_mut_t` splits it into `[f32; 8]`
chunks and a tail. You need no aligned allocator and no padding.

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
- Raw register interop with `core::arch` (`raw()`, `from_raw_t`).
- Bounds-checked AVX-512
  [gather and scatter](https://imazen.github.io/archmage/magetypes/memory/gather-scatter/)
  on `u32x16`, `i32x16` and `f32x16`.

## Which macro

The macros come from archmage:

| You want to | Use |
|---|---|
| Write one kernel with vector types, for every CPU | [`#[magetypes]`](https://imazen.github.io/archmage/archmage/dispatch/magetypes-macro/) + [`incant!`](https://imazen.github.io/archmage/archmage/dispatch/incant/) |
| Let the compiler vectorize a plain loop for each tier | [`#[autoversion]`](https://imazen.github.io/archmage/archmage/dispatch/autoversion/) |
| Call the intrinsics of one instruction set | [`#[arcane]`](https://imazen.github.io/archmage/archmage/concepts/arcane/) at the entry, [`#[rite]`](https://imazen.github.io/archmage/archmage/concepts/rite/) for helpers |
| Share an algorithm between kernels | [A generic helper](https://imazen.github.io/archmage/magetypes/examples/generic-kernels/), inlined into a `#[magetypes]` kernel |

[Types and dispatch](https://imazen.github.io/archmage/magetypes/dispatch/types-and-dispatch/)
covers kernels that are also generic over a pixel type or a constant.

## Features

| Feature | Default | Effect |
|---|---|---|
| `std` | on | Runtime CPU detection. Without it, `summon()` sees only the features enabled at compile time. |
| `w512` | on | The 512-bit vector types. They run as narrower vectors where native AVX-512 is not in use. |
| `avx512` | off | Native AVX-512 vectors. Implies `w512` and `archmage/avx512`. |

AVX-512 is opt-in from your own crate, as in the quick start. Give your crate
an `avx512` feature that forwards to both crates, and write the tier as
`v4(cfg(avx512))` or `v4x(cfg(avx512))`.

For `no_std + alloc`, set `default-features = false` on both crates.
[Installation](https://imazen.github.io/archmage/archmage/getting-started/installation/)
has the full setup.

## Safety

Using magetypes takes no `unsafe` in your code, so your crate can keep
`#![forbid(unsafe_code)]`.

Every vector carries the archmage token it was built with. Its operations run
only where the CPU's features are proven, and no constructor works without that
proof.

Inside, magetypes stacks its own proofs on archmage's and on Rust's. The
compiler checks each x86 and NEON intrinsic against the features of the token
it runs under. Compile-time assertions check the size and layout of every
reinterpretation of memory. What the compiler can't check is memory access
through a pointer. That is left to a few one-line `unsafe` blocks in one
internal module, which load, store, gather and scatter vector storage.

The [magetypes safety model](https://imazen.github.io/archmage/magetypes/safety/)
has the details.

## Limits

- A vector wider than the CPU's registers runs as two or four native
  operations: an `f32x8` on NEON is two `f32x4`s.
- SIMD tiers cover x86-64, AArch64 and WASM. Other targets run the scalar
  backend.
- With `v4` in a `#[magetypes]` tier list, the body can use the 512-bit types,
  `f32x4` and `f32x8`. The AVX-512 tokens don't implement the other 128- and
  256-bit types.
- Transcendentals are approximations with documented error:
  [Transcendentals](https://imazen.github.io/archmage/magetypes/math/transcendentals/)
  gives each function's domain and precision.
- Some floating-point results differ between backends. `mul_add` rounds once
  where the hardware fuses (x86 v3/v4, NEON) and twice on the scalar backend and
  strict WASM; `mul_add_portable` rounds once everywhere, in software where
  needed. [ISA quirks and fixups](https://imazen.github.io/archmage/magetypes/isa-quirks/)
  lists every difference.

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

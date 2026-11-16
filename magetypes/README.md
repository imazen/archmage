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
magetypes = "=0.9.32"
archmage  = "=0.9.32"   # the macros and tokens
```

Multiply a buffer by a factor, using AVX2, NEON or WASM SIMD where available:

```rust
use archmage::prelude::*;

#[attune(_*_t, define(f32x8))]
fn scale_plane_impl(token: Token, plane: &mut [f32], factor: f32) {
    // `define(f32x8)` makes `f32x8` mean `f32x8<X64V3Token>` in `_v3_t`,
    // `f32x8<NeonToken>` in `_neon_t`, and so on. `Token` is replaced too.
    let factor_v = f32x8::splat_t(token, factor);
    let (chunks, tail) = f32x8::partition_slice_mut_t(token, plane);
    for chunk in chunks {
        (f32x8::load_t(token, chunk) * factor_v).store(chunk);
    }
    for v in tail { *v *= factor; }
}

pub fn scale_plane(plane: &mut [f32], factor: f32) {
    attuned!(scale_plane_impl(plane, factor))
}

let mut plane = [2.0; 11];
scale_plane(&mut plane, 0.5);
assert_eq!(plane, [1.0; 11]);
```

- `#[attune(_*_t, define(f32x8))]` compiles `scale_plane_impl` once per
  default portable tier (`scale_plane_impl_v3_t`, `_neon_t`, `_wasm128_t`,
  `_scalar_t`). List several vector types as `define(f32x8, u8x16, i16x8)`.
- `attuned!(scale_plane_impl(plane, factor))` calls `summon()` for each tier,
  best first, and falls back to scalar. Dispatch once around your loop, as
  here, not inside it.
- Constructors ending in `_t` take the token as their first argument.

`0.9.32` is the matching release for `archmage = "=0.9.32"`:
existing vector and constructor signatures are preserved, `_t(token, ...)`
remains token-first, and the flat transcendental methods keep their behavior
(fixing `NaN` propagation in `log2_midp`/`ln_midp`/`log10_midp` and `-0.0` at
`silu_midp(-inf)`). Configurable domain-checked `powf`/`expf`/`logf`/`cbrtf`
with matching scalar tail helpers live in [Magemath](../magemath). Unified
macros (`#[attune]`, `attuned!`, `reattune!`) are documented in the
[unified macro guide](https://github.com/imazen/archmage/blob/main/docs/ATTUNE.md).

Don't write per-tier wrappers around a `#[attune(_*_t)]` kernel: the macro
already generates them.

The kernel is adapted from `zenfilters`;
[Reusable generic kernels](https://imazen.github.io/archmage/magetypes/examples/generic-kernels/)
links the production source. If you are upgrading, 0.9.30 deprecates the older
constructor names (`splat`, `load`, …) ahead of a planned 0.10 change: the
[migration guide](https://github.com/imazen/archmage/blob/main/docs/TOKEN-CONSTRUCTOR-MIGRATION.md)
has the mapping.

### Imports

| You want | Import |
|---|---|
| The macros and tokens (`#[attune]`, `attuned!`, `X64V3Token`, …) | `use archmage::prelude::*;` |
| Vectors inside a `#[attune(..., define(f32x8))]` body | Nothing extra: `define(f32x8)` supplies them |
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

## Naming: `<name>_<tier>` and `<name>_<tier>_t`

- Direct feature bodies end in `_<tier>` (`scale_plane_impl_v3`) and are safe to
  call only inside a feature context covering `<tier>`.
- Proof wrappers end in `_<tier>_t` (`scale_plane_impl_v3_t`) and take that
  tier's token as their first argument so ordinary callers with a summoned token
  can call them safely.

Name your own functions the same way:

- Leave `v3` out and write `scale_plane_impl_v3_t` by hand, and `attuned!`
  still finds it in the same family.
- Inside an annotated feature body, `attuned!(helper(…), [_v3, _scalar])`
  calls direct `_v3` helpers with no runtime probe and no token check.
- The suffix shows, at every call site, which CPU features a function needs.

## Which `#[attune]` form

The macros come from archmage:

| You want | Write | Call it |
|---|---|---|
| One kernel with vector types, for every CPU | `#[attune(_*_t, define(f32x8))]` on `fn kernel(token: Token, …)` | `attuned!(kernel(…))` (or explicit tier list) |
| A plain loop the compiler vectorizes for each tier | `#[attune(dispatch)]` on `fn sum(data: &[f32]) -> f32` | `sum(data)` (macro writes the dispatcher) |
| The intrinsics of one instruction set | `#[attune(import_intrinsics)]` on `fn kernel_v3_t(token: X64V3Token, …)` (or `#[attune(wrap)]`) | `attuned!(kernel(…), [_v3, _scalar])` |
| A helper called inside SIMD code | `#[attune(v3)]` on `fn helper(…)` | `helper(…)` from a context covering `v3` |
| An algorithm shared across widths/backends | [A generic helper](https://imazen.github.io/archmage/magetypes/examples/generic-kernels/): `#[inline(always)] fn helper<T: F32x8Backend>(token: T, …)` | `helper(token, …)` from a `#[attune]` kernel |

Earlier attributes (`#[magetypes]`, `#[arcane]`, `#[rite]`, `#[autoversion]`,
`incant!`) remain supported; see [docs/ATTUNE.md](https://github.com/imazen/archmage/blob/main/docs/ATTUNE.md)
for the mapping.
[Types and dispatch](https://imazen.github.io/archmage/magetypes/dispatch/types-and-dispatch/)
covers kernels that are also generic over a pixel type or a constant.

## Features

| Feature | Default | Effect |
|---|---|---|
| `std` | on | Runtime CPU detection. Without it, `summon()` sees only the features enabled at compile time. |
| `w512` | on | The 512-bit vector types. They run as narrower vectors where native AVX-512 is not in use. |
| `avx512` | off | Native AVX-512 vectors. Implies `w512` and `archmage/avx512`. |

For `no_std + alloc`, set `default-features = false` on both crates.
[Installation](https://imazen.github.io/archmage/archmage/getting-started/installation/)
has the full setup.

### AVX-512

AVX-512 is opt-in from your own crate. Give it an `avx512` feature that
forwards to both crates:

```toml
[features]
avx512 = ["archmage/avx512", "magetypes/avx512"]
```

Then write the kernel with a 512-bit type and opt in `+v4(cfg(avx512))` (or
keep 8 lanes on V3/NEON/WASM via a per-tier width split):

```rust
use archmage::prelude::*;

#[attune(_*_t, +v4(cfg(avx512)), define(f32x16))]
fn scale_wide_impl(token: Token, plane: &mut [f32], factor: f32) {
    let factor_v = f32x16::splat_t(token, factor);
    let (chunks, tail) = f32x16::partition_slice_mut_t(token, plane);
    for chunk in chunks {
        (f32x16::load_t(token, chunk) * factor_v).store(chunk);
    }
    for v in tail { *v *= factor; }
}

pub fn scale_wide(plane: &mut [f32], factor: f32) {
    attuned!(scale_wide_impl(plane, factor), [_v4(cfg(avx512)), _v3, _neon, _wasm128, _scalar])
}

let mut plane = [2.0; 37];
scale_wide(&mut plane, 0.5);
assert_eq!(plane, [1.0; 37]);
```

The `v4` copy runs each `f32x16` operation as one 512-bit instruction when your
crate's `avx512` feature is on. Listing `_v4` on an `f32x8` kernel does not
widen it: `f32x8` stays eight lanes there.

### Polyfills

Every other tier runs the same `f32x16` as a polyfill: two 256-bit operations
on AVX2, four 128-bit operations on NEON and WASM. An `f32x8` is two 128-bit
operations on NEON and WASM.

In the kernel above and a sum kernel, each polyfilled operation compiled to
those native operations and nothing else: no stack traffic and no extra moves
in the AVX2 and NEON loops ([assembly results](https://github.com/imazen/archmage/blob/main/benchmarks/polyfill_asm_2026-10-06.md)).

[Timed](https://github.com/imazen/archmage/blob/main/benchmarks/polyfill_timing_2026-10-06.md) on Zen 5 and a Neoverse-N1 from 1,024 floats up, the multiply
ran within 9% of the native width. The sum ran 1.4 to 3 times faster with the
wider type, because each part keeps its own accumulator. At 64 floats the
`f32x16` versions took up to 20% longer.

Three costs remain. Each `f32x16` value takes two registers on AVX2 and four on
NEON. Reductions such as `reduce_add` run once per part. A wider chunk leaves a
longer scalar tail. [Polyfills](https://imazen.github.io/archmage/magetypes/cross-platform/polyfills/) has the
numbers.

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

- A vector wider than the CPU's registers is a polyfill. It runs as two or
  four native operations, and each value takes that many registers: see
  [Polyfills](#polyfills).
- SIMD tiers cover x86-64, AArch64 and WASM. Other targets run the scalar
  backend.
- With the `avx512` feature, AVX-512 tokens also support the 128- and
  256-bit float and integer types. These narrower backend, conversion and
  bitcast implementations delegate to V3; 512-bit types use AVX-512.
- Transcendentals are approximations with documented error:
  [Transcendentals](https://imazen.github.io/archmage/magetypes/math/transcendentals/)
  gives each function's domain and precision.
- Some floating-point results differ between backends. `mul_add_relaxed`
  rounds once where the hardware fuses (x86 v3/v4, NEON) and twice on the scalar
  backend and strict WASM; `mul_add_fused` rounds once everywhere, in software
  where needed. `mul_add` is deprecated and becomes the fused form in 0.11. [ISA quirks and fixups](https://imazen.github.io/archmage/magetypes/isa-quirks/)
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

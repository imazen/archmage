# archmage

[Guide](https://imazen.github.io/archmage/) · [Intrinsics browser](https://imazen.github.io/archmage/intrinsics/) · [Archmage API](https://docs.rs/archmage/latest/archmage/) · [Magetypes API](https://docs.rs/magetypes/latest/magetypes/)

Archmage lets you write SIMD code in Rust **without `unsafe`** — your crate can keep `#![forbid(unsafe_code)]` while calling intrinsics directly.

You pick a CPU tier, prove it's present once with `summon()`, and the type system keeps every intrinsic call sound. Rust already makes most intrinsics safe inside a function compiled for their CPU features; archmage supplies the proof that the CPU has them.

It runs on x86-64, AArch64 and WASM, is `no_std + alloc`, and needs Rust 1.89 or later.

## Quick start

```toml
[dependencies]
archmage = "0.9.30"
magetypes = "0.9.30"   # vector types such as f32x8
```

Multiply a buffer by a gain, using AVX2, NEON or WASM SIMD where available:

```rust
#![forbid(unsafe_code)]
use archmage::prelude::*;

#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn gain_impl(token: Token, plane: &mut [f32], gain: f32) {
    let factor = f32x8::splat_t(token, gain);
    let (chunks, tail) = f32x8::partition_slice_mut_t(token, plane);
    for chunk in chunks {
        (f32x8::load_t(token, chunk) * factor).store(chunk);
    }
    for value in tail {
        *value *= gain;
    }
}

pub fn apply_gain(plane: &mut [f32], gain: f32) {
    incant!(gain_impl(plane, gain), [v3, neon, wasm128, scalar])
}

let mut plane = [2.0; 11];
apply_gain(&mut plane, 0.5);
assert_eq!(plane, [1.0; 11]);
```

- `#[magetypes]` compiles `gain_impl` once per tier in its list and names each
  copy for its tier: `gain_impl_v3` (AVX2 and FMA), `gain_impl_neon`,
  `gain_impl_wasm128` and `gain_impl_scalar`. In each copy, `Token` is that
  tier's token type and `f32x8` is that tier's vector.
- `incant!` finds the copies by those names. It calls `summon()` for each tier,
  best first, and runs the first copy the CPU supports. Call it around your
  loop, as here, not inside it.
- Constructors ending in `_t` take the token as their first argument.

The kernel is adapted from `zenfilters`;
[Reusable generic kernels](https://imazen.github.io/archmage/magetypes/examples/generic-kernels/)
links the production source. If you are upgrading, 0.9.30 deprecates the older
constructor names (`splat`, `load`, …): the
[migration guide](https://github.com/imazen/archmage/blob/main/docs/TOKEN-CONSTRUCTOR-MIGRATION.md)
has the mapping.

## Calling intrinsics directly

For code tied to one instruction set, write a function per tier and let
`incant!` choose:

```rust
use archmage::prelude::*;

#[arcane(import_intrinsics)]
fn multiply_v3(_token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let v = _mm256_loadu_ps(data);
    let mut out = [0.0; 8];
    _mm256_storeu_ps(&mut out, _mm256_mul_ps(v, _mm256_set1_ps(2.0)));
    out
}

fn multiply_scalar(_token: ScalarToken, data: &[f32; 8]) -> [f32; 8] {
    data.map(|v| v * 2.0)
}

pub fn multiply(data: &[f32; 8]) -> [f32; 8] {
    incant!(multiply(data), [v3, scalar])
}

assert_eq!(multiply(&[3.0; 8]), [6.0; 8]);
```

- `#[arcane]` compiles the function with its token's CPU features and makes it
  safe to call from ordinary code.
- `import_intrinsics` brings the architecture's intrinsics into scope. Loads and
  stores take references instead of raw pointers.
- The names do the dispatch. `incant!(multiply(data), [v3, scalar])` calls
  `multiply_v3` where the CPU has the `v3` features and `multiply_scalar`
  everywhere else.

Mark helper functions with `#[rite]` instead. They get the same features and
inline into the kernel that calls them. The
[intrinsics browser](https://imazen.github.io/archmage/intrinsics/) shows which
intrinsics each token unlocks.

## Naming: `<name>_<tier>`

`incant!(gain_impl(…), [v3, scalar])` calls `gain_impl_v3(token, …)` or
`gain_impl_scalar(token, …)`. That is the whole contract: the tier as a suffix,
and that tier's token as the first argument. `#[magetypes]` and `#[autoversion]`
generate functions of that shape.

Name your own tier functions the same way:

- They join the same family. Leave `v3` out of a `#[magetypes]` list, write
  `gain_impl_v3` by hand, and `incant!` still finds it:
  [hand-tuned variants](https://imazen.github.io/archmage/archmage/dispatch/incant/#hand-tuned-variants).
- Inside a tier function, `incant!` needs no CPU check. In a `v3` function,
  `incant!(helper(…), [v3, scalar])` compiles to a direct call to `helper_v3`:
  [calls inside a tier](https://imazen.github.io/archmage/archmage/dispatch/incant/#calls-inside-a-tier).
- The suffix shows, at every call site, which CPU features a function needs.

## Which macro

| You want | Write | Call it |
|---|---|---|
| One kernel with vector types, for every CPU | [`#[magetypes(v3, neon, scalar)]`](https://imazen.github.io/archmage/archmage/dispatch/magetypes-macro/) on `fn kernel(token: Token, …)` | [`incant!(kernel(…), [v3, neon, scalar])`](https://imazen.github.io/archmage/archmage/dispatch/incant/) |
| A plain loop the compiler vectorizes for each tier | [`#[autoversion]`](https://imazen.github.io/archmage/archmage/dispatch/autoversion/) on `fn sum(data: &[f32]) -> f32` | `sum(data)`. The macro writes the dispatcher |
| The intrinsics of one instruction set | [`#[arcane]`](https://imazen.github.io/archmage/archmage/concepts/arcane/) on `fn kernel_v3(token: X64V3Token, …)` | `incant!(kernel(…), [v3, scalar])` |
| A helper inside SIMD code | [`#[rite(v3)]`](https://imazen.github.io/archmage/archmage/concepts/rite/) on `fn helper(…)` | `helper(…)`, from a function that has the `v3` features |
| An algorithm shared between kernels | [A generic helper](https://imazen.github.io/archmage/magetypes/examples/generic-kernels/): `#[inline(always)] fn helper<T: F32x8Backend>(token: T, …)` | `helper(token, …)`, from a `#[magetypes]` kernel |
| The tier chosen by hand | `X64V3Token::summon()` and an `if let` | `kernel_v3(token, …)`, behind `#[cfg(target_arch = "x86_64")]`: [manual dispatch](https://imazen.github.io/archmage/archmage/dispatch/manual/) |

The guide also covers
[tokens and tiers](https://imazen.github.io/archmage/archmage/getting-started/tokens/),
[testing every tier](https://imazen.github.io/archmage/archmage/testing/dispatch-testing/)
and [AVX-512](https://imazen.github.io/archmage/archmage/advanced/avx512/).

## Features

| Feature | Crate | Default | Effect |
|---|---|---|---|
| `std` | both | on | Runtime CPU detection. Without it, `summon()` sees only the features enabled at compile time. |
| `avx512` | both | off | archmage: AVX-512 intrinsics through `import_intrinsics`. magetypes: native AVX-512 vectors. |
| `w512` | magetypes | on | The 512-bit vector types. They run as narrower vectors where native AVX-512 is not in use. |

AVX-512 is opt-in from your own crate. Give it an `avx512` feature:

```toml
[features]
avx512 = ["archmage/avx512", "magetypes/avx512"]
```

Then name the tier in both lists. Its copy is compiled only when that feature
is on:

```text
#[magetypes(define(f32x8), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
incant!(gain_impl(plane, gain), [v4(cfg(avx512)), v3, neon, wasm128, scalar])
```

`f32x8` stays eight lanes in the `v4` copy. For 512-bit vectors, use `f32x16`.

[Installation](https://imazen.github.io/archmage/archmage/getting-started/installation/)
has the `no_std` setup and the remaining features.

## Safety

You write no `unsafe`, so your crate can keep `#![forbid(unsafe_code)]`.

Rust does most of the work. It checks every call into a `#[target_feature]`
function, and since 1.87 it lets safe code call most intrinsics inside one.
Archmage adds the proof that the CPU has the features: a token such as
`X64V3Token`. Safe code gets one only once the features are confirmed, normally
by `summon()`. The macros generate the one `unsafe` call into feature-enabled
code, and the token is what makes it sound.

magetypes builds on the same tokens and adds its own compile-time checks. What
is left for `unsafe` there is a few one-line blocks that load, store, gather and
scatter vector storage.

The [safety model](https://imazen.github.io/archmage/archmage/concepts/safety/)
has the details, and
[SOUNDNESS.md](https://github.com/imazen/archmage/blob/main/docs/SOUNDNESS.md)
lists every `unsafe` in both crates.

## Limits

- There is no SVE tier: stable Rust has no SVE intrinsics yet. Targets other
  than x86-64, AArch64 and WASM, including 32-bit x86, run the scalar fallback.
- Calling an `#[arcane]` function once per loop iteration is slow. On two small
  kernels it measured 4.1× and 6.2× slower than one call around the whole loop
  ([docs/PERFORMANCE.md](https://github.com/imazen/archmage/blob/main/docs/PERFORMANCE.md)).
  Dispatch once, outside the loop.
- A vector wider than the CPU's registers runs as two or four native
  operations: an `f32x8` on NEON is two `f32x4`s.
- With `v4` in a `#[magetypes]` tier list, the body can use the 512-bit types,
  `f32x4` and `f32x8`. The AVX-512 tokens don't implement the other 128- and
  256-bit types.
- Some floating-point results differ between backends. `mul_add` rounds once
  where the hardware fuses (x86 v3/v4, NEON) and twice on the scalar backend and
  strict WASM; `mul_add_portable` rounds once everywhere, in software where
  needed. [ISA quirks and fixups](https://imazen.github.io/archmage/magetypes/isa-quirks/)
  lists every difference, and
  [Transcendentals](https://imazen.github.io/archmage/magetypes/math/transcendentals/)
  gives each function's domain and precision.

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

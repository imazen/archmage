# archmage

[Guide](https://imazen.github.io/archmage/) · [Intrinsics browser](https://imazen.github.io/archmage/intrinsics/) · [Archmage API](https://docs.rs/archmage/latest/archmage/) · [Magetypes API](https://docs.rs/magetypes/latest/magetypes/)

Archmage lets you write SIMD code in Rust **without `unsafe`** — your crate can keep `#![forbid(unsafe_code)]` while calling intrinsics directly. You pick a CPU tier, prove it's present once with `summon()`, and the type system keeps every intrinsic call sound. It builds on Rust's own rules: since Rust 1.87, most intrinsics are safe to call inside a function compiled for their CPU features, and archmage supplies the proof that the CPU has them. It runs on x86-64, AArch64 and WASM, and is `no_std + alloc`.

## Portable kernels with magetypes

Use `archmage` with the [magetypes vector crate](https://docs.rs/magetypes/latest/magetypes/) for portable vector kernels:

```toml
[dependencies]
archmage = "0.9.30"
magetypes = "0.9.30"
```

This scales an image plane or an audio buffer, including a short scalar tail.
`#[magetypes]` compiles `gain_impl` once per CPU tier, with the vector type bound
to that tier's token, and `incant!` picks the tier once, outside the loop. No
per-tier wrappers or raw pointers are needed. It is adapted from the `zenfilters`
plane-scaling kernel; the [complete production call chain and adaptation notes](https://imazen.github.io/archmage/magetypes/examples/generic-kernels/) include pinned source links.

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

Constructors take the token first and end in `_t`. magetypes 0.9.30 deprecates
the older names (`splat`, `load`, …); the
[migration guide](https://github.com/imazen/archmage/blob/main/docs/TOKEN-CONSTRUCTOR-MIGRATION.md)
has the mapping.

## Direct intrinsics

For ISA-specific code, write the kernel with `#[arcane]` and reach it through
`incant!` or after `summon()`:

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

`#[arcane]` compiles the function with `X64V3Token`'s features and is the safe
entry from ordinary code. `import_intrinsics` brings the intrinsics into scope,
with memory operations that take references instead of pointers. Helpers called
from inside belong in `#[rite]` functions, which get the same features without a
wrapper, so they inline. The [intrinsics browser](https://imazen.github.io/archmage/intrinsics/)
lists what each token unlocks.

## Choosing a pattern

| Work | Pattern |
|---|---|
| Portable vector kernel | `#[magetypes]` + public `incant!` |
| Reusable algorithm | Generated entry → inline backend-generic helper |
| Ordinary loop offered to LLVM for vectorization | `#[autoversion]` |
| Hand-tuned ISA entry | `#[arcane]` |
| Matched internal intrinsic helper | `#[rite]` |

`#[magetypes]` is an [archmage attribute](https://docs.rs/archmage/latest/archmage/attr.magetypes.html);
the [magetypes crate](https://docs.rs/magetypes/latest/magetypes/) supplies the vector types.
`define(f32x8)` is shorthand for [`f32x8<Token>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html), and type and const
generics can stay on the generated function. A backend-generic helper still has
to be called from a generated, feature-enabled function: marking it `#[inline]`
or passing it a token doesn't compile it with the features. See the
[generic specialization example](https://imazen.github.io/archmage/magetypes/dispatch/types-and-dispatch/)
and [reusable generic kernels](https://imazen.github.io/archmage/magetypes/examples/generic-kernels/).

Inside a function that already has the features, `X64V3Token::from_context()`
makes a token without runtime detection, and rustc checks the context; `.v3()`
narrows a stronger token. See [tokens](https://imazen.github.io/archmage/archmage/getting-started/tokens/)
and [dispatch](https://imazen.github.io/archmage/archmage/dispatch/incant/) for more, including `incant!`'s
`with token` and `without token` forms.

## Features and numerical contracts

Rust 1.89 is the minimum supported version. Archmage depends on
[`archmage-macros`](https://crates.io/crates/archmage-macros) and
[`safe_unaligned_simd`](https://crates.io/crates/safe_unaligned_simd), plus
[`winarm-cpufeatures`](https://crates.io/crates/winarm-cpufeatures) on Windows on
ARM. `std` is on by default and provides runtime CPU detection; without it,
`summon()` knows only the features enabled at compile time. The macros are always
included; the `macros` feature is a compatibility no-op. Magetypes also defaults
to `w512`, which supplies logical 512-bit types and polyfills. Optional `avx512`
adds native AVX-512 support; it does not detect the running CPU. `incant!` and
`#[magetypes]` compile their `v4` and `v4x` variants and dispatch arms only when
your own crate has a feature named `avx512` (`#[autoversion]` always generates its
`v4` variant); follow the
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

Compile the complete call chain, test supported tiers and scalar tails, and
inspect optimized code under your supported baseline. See
[testing](https://imazen.github.io/archmage/archmage/testing/dispatch-testing/) and
[production coverage](https://imazen.github.io/archmage/magetypes/examples/coverage/).

## Safety

Rust does most of the work here. It checks every call into a `#[target_feature]`
function, and since 1.87 it lets safe code call most intrinsics inside one.
Archmage supplies the missing piece: a token such as `X64V3Token` is the proof
that the CPU has its features, and safe code gets one only once those features
are confirmed, normally by `summon()`. `#[arcane]`, `#[magetypes]` and `incant!`
enter feature-enabled code only with that proof in hand. The one `unsafe` block
that crosses into such code is generated by the macro, and the token is what
makes it sound. None of that `unsafe` is yours, so your crate can keep
`#![forbid(unsafe_code)]`.

magetypes builds on the same tokens and adds its own compile-time checks; what
is left for `unsafe` is a few one-line blocks that load, store, gather and
scatter vector storage. The
[safety model](https://imazen.github.io/archmage/archmage/concepts/safety/)
walks through the expansion, what the guarantee relies on, and how it is
checked; [SOUNDNESS.md](https://github.com/imazen/archmage/blob/main/docs/SOUNDNESS.md)
inventories every `unsafe` in archmage and magetypes.

## Limits

- SIMD tiers cover x86-64, AArch64 and WASM. Other targets, including 32-bit x86,
  run the scalar fallback. There is no SVE tier: stable Rust has no SVE
  intrinsics yet.
- Native AVX-512 needs the `avx512` feature in your own crate, as above.
- Each call into an `#[arcane]` function from ordinary code crosses an
  optimization boundary. Calling one per loop iteration measured 4.1× and 6.2×
  slower than one call around the whole loop on two small kernels
  ([docs/PERFORMANCE.md](https://github.com/imazen/archmage/blob/main/docs/PERFORMANCE.md)),
  so dispatch once, outside the loop.
- Vector types wider than the hardware run as two or four native operations; an
  `f32x8` on NEON is two `f32x4`s.
- Some floating-point results differ between backends; the ISA quirks page lists
  each difference.

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

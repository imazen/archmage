# archmage

[Guide](https://imazen.github.io/archmage/) · [Intrinsics browser](https://imazen.github.io/archmage/intrinsics/) · [Archmage API](https://docs.rs/archmage/latest/archmage/) · [Magetypes API](https://docs.rs/magetypes/latest/magetypes/) · [Magemath API](https://docs.rs/magemath/latest/magemath/)

Archmage lets you write SIMD code in Rust **without `unsafe`** — your crate can keep `#![forbid(unsafe_code)]` while calling intrinsics directly.

You pick a CPU tier, prove it's present once with `summon()`, and the type system keeps every intrinsic call sound. Rust already makes most intrinsics safe inside a function compiled for their CPU features; archmage supplies the proof that the CPU has them.

It runs on x86-64, AArch64 and WASM, is `no_std + alloc`, and needs Rust 1.89 or later.

## 0.9.32: Cargo 1.100, attune proof opt-out, #154/#193 macro fixes, and magemath

This release follows `0.9.31` with Cargo 1.100 build-dir compatibility,
`#[attune(..., suppress_const_test)]`, safe-Rust callee shadowing and manual
`#[target_feature]` soundness hardening (#193), preserved author body brace
spans for line coverage (#154), and the new [Magemath crate](https://github.com/imazen/archmage/tree/main/magemath)
alongside bug fixes in `magetypes` midp logs and `silu_midp(-inf)`. Earlier
macro names (`#[arcane]`, `#[rite]`, `#[magetypes]`, `#[autoversion]`,
`incant!`) remain warning-free in `0.9.32` and are queued for deprecation in
the next release; use matching versions of archmage and magetypes so their
tokens come from the same archmage package.

```rust
use archmage::prelude::*;

// Generate direct tier functions and an ordinary callable dispatcher.
#[attune(_*, dispatch)]
fn scale(values: &mut [f32], factor: f32) {
    for value in values {
        *value *= factor;
    }
}

let mut values = [2.0, 4.0];
scale(&mut values, 0.5);
assert_eq!(values, [1.0, 2.0]);
```

`#[attune(v3)]` adds features to one function; `#[attune(wrap)]` creates a proof
wrapper using its token parameter. Bare `#[attune]` infers a direct `_v3` or
proof-wrapper `_v3_t` suffix. A token parameter alone does not imply `wrap`.
The [unified macro guide](https://github.com/imazen/archmage/blob/main/docs/ATTUNE.md)
covers the selector grammar, call and proof rules, inline policies, a legacy-attribute
mapping, and current restrictions.

Configurable f32 power, exponential, logarithm and cube-root approximations,
with explicit domains and scalar entry points for SIMD tails, live in the
[Magemath crate](https://github.com/imazen/archmage/tree/main/magemath). The flat
Magetypes transcendental methods remain supported; the
[Magemath guide](https://github.com/imazen/archmage/blob/main/docs/MAGEMATH-MIGRATION.md)
says when to switch.

## Quick start

```toml
[dependencies]
archmage = "=0.9.32"
magetypes = "=0.9.32"   # vector types such as f32x8
```

Multiply a buffer by a gain, using AVX2, NEON or WASM SIMD where available:

```rust
#![forbid(unsafe_code)]
use archmage::prelude::*;

#[attune(_*_t, define(f32x8))]
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
    attuned!(gain_impl(plane, gain))
}

let mut plane = [2.0; 11];
apply_gain(&mut plane, 0.5);
assert_eq!(plane, [1.0; 11]);
```

- `#[attune(_*_t, define(f32x8))]` compiles `gain_impl` once per default
  portable tier and emits proof-taking wrappers: `gain_impl_v3_t` (AVX2 and
  FMA), `gain_impl_neon_t`, `gain_impl_wasm128_t` and `gain_impl_scalar_t`. In
  each copy, `Token` is that tier's token type and `f32x8` is `f32x8<Token>`.
- `attuned!(gain_impl(plane, gain))` dispatches across those default tiers, best
  first via `summon()`, and falls back to scalar. Dispatch once around your
  loop, as here, not inside it.
- Constructors ending in `_t` take the token as their first argument.

The kernel is adapted from `zenfilters`;
[Reusable generic kernels](https://imazen.github.io/archmage/magetypes/examples/generic-kernels/)
links the production source. If you are upgrading, 0.9.30 deprecates the older
constructor names (`splat`, `load`, …): the
[migration guide](https://github.com/imazen/archmage/blob/main/docs/TOKEN-CONSTRUCTOR-MIGRATION.md)
has the mapping.

## Calling intrinsics directly

For code tied to one instruction set, write a proof wrapper per tier and let
`attuned!` choose:

```rust
use archmage::prelude::*;

#[attune(import_intrinsics)]
fn multiply_v3_t(_token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let v = _mm256_loadu_ps(data);
    let mut out = [0.0; 8];
    _mm256_storeu_ps(&mut out, _mm256_mul_ps(v, _mm256_set1_ps(2.0)));
    out
}

fn multiply_scalar_t(_token: ScalarToken, data: &[f32; 8]) -> [f32; 8] {
    data.map(|v| v * 2.0)
}

pub fn multiply(data: &[f32; 8]) -> [f32; 8] {
    attuned!(multiply(data), [_v3, _scalar])
}

assert_eq!(multiply(&[3.0; 8]), [6.0; 8]);
```

- The `_v3_t` suffix tells `#[attune]` to emit a safe `X64V3Token` proof wrapper
  that enables AVX2 + FMA features inside the inner implementation.
- `import_intrinsics` brings the architecture's intrinsics into scope, with safe
  loads and stores taking references instead of raw pointers.
- `attuned!(multiply(data), [_v3, _scalar])` calls `multiply_v3_t` where the CPU
  supports `v3` and `multiply_scalar_t` everywhere else.

For helper functions called from code that already holds matching target
features, mark them `#[attune(v3)]` (zero wrapper overhead; use
`X64V3Token::from_context()` if you need a token inside the body). The
[intrinsics browser](https://imazen.github.io/archmage/intrinsics/) shows which
intrinsics each token unlocks.

## Naming: `<name>_<tier>` and `<name>_<tier>_t`

- Direct feature bodies end in `_<tier>` (`gain_impl_v3`) and are safe to call
  only inside a function whose features already cover `<tier>`.
- Proof wrappers end in `_<tier>_t` (`gain_impl_v3_t`) and take that tier's token
  as their first argument so any ordinary caller with a summoned token can call
  them safely.

Name your own functions the same way:

- Leave `v3` out and write `gain_impl_v3_t` by hand, and `attuned!` still finds
  it in the same family.
- Inside an annotated feature body, `attuned!(helper(…), [_v3, _scalar])`
  resolves direct `_v3` helpers with no runtime probe and no token check.
- The suffix shows, at every call site, which CPU features a function needs.

## Which `#[attune]` form

| You want | Write | Call it |
|---|---|---|
| One kernel with vector types, for every CPU | `#[attune(_*_t, define(f32x8))]` on `fn kernel(token: Token, …)` | `attuned!(kernel(…))` (or explicit tier list) |
| A plain loop the compiler vectorizes per tier | `#[attune(dispatch)]` on `fn sum(data: &[f32]) -> f32` | `sum(data)` (macro writes the dispatcher) |
| The intrinsics of one instruction set | `#[attune(import_intrinsics)]` on `fn kernel_v3_t(token: X64V3Token, …)` (or `#[attune(wrap)]`) | `attuned!(kernel(…), [_v3, _scalar])` |
| A helper called inside SIMD code | `#[attune(v3)]` on `fn helper(…)` | `helper(…)` from a context covering `v3` |
| An algorithm shared across widths/backends | [A generic helper](https://imazen.github.io/archmage/magetypes/examples/generic-kernels/): `#[inline(always)] fn helper<T: F32x8Backend>(token: T, …)` | `helper(token, …)` from a `#[attune]` kernel |
| The tier chosen by hand | `X64V3Token::summon()` and an `if let` | `kernel_v3_t(token, …)` behind `#[cfg(target_arch = "x86_64")]`: [manual dispatch](https://imazen.github.io/archmage/archmage/dispatch/manual/) |

Earlier attributes (`#[magetypes]`, `#[arcane]`, `#[rite]`, `#[autoversion]`,
`incant!`) are superseded by `#[attune]` and `attuned!` and queued for
deprecation in the next release; see [docs/ATTUNE.md](https://github.com/imazen/archmage/blob/main/docs/ATTUNE.md)
for the spelling-by-spelling migration table, [tokens and tiers](https://imazen.github.io/archmage/archmage/getting-started/tokens/),
[testing every tier](https://imazen.github.io/archmage/archmage/testing/dispatch-testing/),
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

Then write the kernel with a 512-bit type (or use a per-tier width split to
keep 8 lanes on V3/NEON/WASM) and list the gated AVX-512 tier explicitly:

```text
#[attune(_*_t, +v4(cfg(avx512)), define(f32x16))]
attuned!(gain_impl(plane, gain), [_v4(cfg(avx512)), _v3, _neon, _wasm128, _scalar])
```

The other tiers run the same `f32x16` as a polyfill: two 256-bit operations on
AVX2, four 128-bit operations on NEON and WASM. In the two kernels we
inspected, each polyfilled operation compiled to those native operations and
nothing else ([assembly results](https://github.com/imazen/archmage/blob/main/benchmarks/polyfill_asm_2026-10-06.md)). The
[magetypes README](https://github.com/imazen/archmage/tree/main/magetypes#avx-512)
has the runnable example and what polyfills cost.

Listing `v4` on an `f32x8` kernel does not widen it: `f32x8` stays eight lanes
there.

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
- Safe Rust calls into `#[target_feature]` helpers (`#[attune(v3)]`) inline with zero wrapper overhead inside a matching tier context.
- Dispatching or crossing a proof wrapper once per loop iteration is slow. On two small
  kernels a proof-wrapper boundary (`#[attune(wrap)]` / `_<tier>_t`) per iteration measured 4.1× and 6.2× slower than one call around the whole loop
  ([docs/PERFORMANCE.md](https://github.com/imazen/archmage/blob/main/docs/PERFORMANCE.md)).
  Dispatch once, outside the loop.
- A vector wider than the CPU's registers is a polyfill. It runs as two or
  four native operations, and each value takes that many registers: an `f32x8`
  on NEON is two `f32x4`s.
  [Polyfills](https://imazen.github.io/archmage/magetypes/cross-platform/polyfills/) has the measured costs.
- With the `avx512` feature, AVX-512 tokens also support the 128- and
  256-bit float and integer types. Narrower integer, f64, conversion and
  bitcast operations delegate to V3 (with AVX-512VL overrides where faster on
  `f32x4`/`f32x8`); 512-bit types use AVX-512.
- Some floating-point results differ between backends. `mul_add_relaxed`
  rounds once where the hardware fuses (x86 v3/v4, NEON) and twice on the scalar
  backend and strict WASM; `mul_add_fused` rounds once everywhere, in software
  where needed. `mul_add` is deprecated and becomes the fused form in 0.11. [ISA quirks and fixups](https://imazen.github.io/archmage/magetypes/isa-quirks/)
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

[zenbench] · **archmage** · [magetypes] · [magemath] · [enough] · [whereat] · [cargo-copter]

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
[magetypes]: https://github.com/imazen/archmage/tree/main/magetypes
[magemath]: https://github.com/imazen/archmage/tree/main/magemath
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go

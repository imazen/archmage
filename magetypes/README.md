# magetypes ![CI](https://img.shields.io/github/actions/workflow/status/imazen/archmage/ci.yml?style=flat-square&label=CI) ![crates.io](https://img.shields.io/crates/v/magetypes?style=flat-square) [![lib.rs](https://img.shields.io/crates/v/magetypes?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/magetypes) ![docs.rs](https://img.shields.io/docsrs/magetypes?style=flat-square) ![License](https://img.shields.io/crates/l/magetypes?style=flat-square)

Token-gated SIMD vector types. Write one kernel, run on AVX2, AVX-512, NEON, WASM SIMD128, or scalar — the `#[magetypes]` macro generates the per-tier target-feature contexts; `incant!` dispatches at runtime. No `unsafe`, `#![forbid(unsafe_code)]`-compatible.

**[Intrinsics Browser](https://imazen.github.io/archmage/intrinsics/)** · [Tutorial Book](https://imazen.github.io/archmage/) · [API Docs](https://docs.rs/magetypes)

## The canonical pattern

```rust
use archmage::prelude::*;
use magetypes::simd::generic::f32x8 as GenericF32x8;

#[magetypes(v4, v3, neon, wasm128, scalar)]
fn scale_plane_impl(token: Token, plane: &mut [f32], factor: f32) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;   // Token replaced per tier

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

That's it. One algorithm, every platform. `#[magetypes]` generates five `#[arcane]`-wrapped variants (`_v4`, `_v3`, `_neon`, `_wasm128`, `_scalar`), each with its own `#[target_feature]`. `incant!` picks the highest available at runtime. `partition_slice_mut` gives you aligned `&mut [f32; 8]` chunks plus a scalar tail with no interior bounds checks.

**`#[magetypes]` IS the `#[arcane]` wrapper generator.** Do not write per-tier `#[arcane]` wrappers around a generic kernel by hand — the macro already does that. This is the single biggest source of confusion, so it bears repeating.

## Pick the width your algorithm wants; polyfills handle the hardware split

`f32x8` on AVX2 → one 256-bit op. On NEON → two 128-bit ops. On WASM SIMD128 → two 128-bit ops. Same source, no code change.

| Width | x86-64 | AArch64 | WASM |
|---|---|---|---|
| 128-bit (`f32x4`, `u8x16`, …) | Native (SSE/AVX) | Native (NEON) | Native (SIMD128) |
| 256-bit (`f32x8`, `u8x32`, …) | Native (AVX2) | Polyfill (2× NEON) | Polyfill (2× SIMD128) |
| 512-bit (`f32x16`, `u8x64`, …, feature `w512`) | Native (AVX-512) | Polyfill (4× NEON) | Polyfill (4× SIMD128) |

Default to the width your algorithm is shaped for. Processing 8 floats at a time? Write `f32x8`. Per-pixel RGBA? Write `f32x4`. Don't downshift to 128-bit "because NEON is 128-bit" — the polyfill is equivalent to hand-rolling the split and runs substantially faster than scalar.

## Runnable reference: `examples/idiomatic_patterns_all.rs`

A single file that exercises and self-tests every pattern below. Runs on x86-64 (±`avx512`), aarch64 (via `cross`/QEMU), and wasm32-wasip1 (via `wasmtime`).

```bash
cargo run --release --example idiomatic_patterns_all
cargo run --release --example idiomatic_patterns_all --features avx512
```

Read it when in doubt about what correct magetypes code looks like.

## Pattern catalog

### A. Inline `#[magetypes]` — the default

Shown above. Algorithm inside the macro body, `GenericF32x8<Token>` with a local `type f32x8` alias. Dispatched by `incant!`.

### B. Extracted generic kernel for reuse

When the same kernel is called from multiple entry points, extract it as a generic function bounded on a backend trait. `T` is inferred at each `#[magetypes]`-generated variant's call site:

```rust
use magetypes::simd::{backends::F32x8Backend, generic::f32x8 as GenericF32x8};

#[inline(always)]   // MANDATORY — inherits target features through inlining
fn dot_kernel<T: F32x8Backend>(token: T, a: &[f32], b: &[f32]) -> f32 {
    let mut acc = GenericF32x8::<T>::zero(token);
    for i in 0..a.len() / 8 {
        let va = GenericF32x8::<T>::load(token, a[i*8..][..8].try_into().unwrap());
        let vb = GenericF32x8::<T>::load(token, b[i*8..][..8].try_into().unwrap());
        acc = va.mul_add(vb, acc);
    }
    acc.reduce_add() /* + scalar tail */
}

#[magetypes(v4, v3, neon, wasm128, scalar)]
fn dot_impl(token: Token, a: &[f32], b: &[f32]) -> f32 {
    dot_kernel(token, a, b)      // T inferred from concrete Token
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 { incant!(dot_impl(a, b)) }
```

**Why `#[inline(always)]` is non-negotiable:** the generic function has no `#[target_feature]` of its own. It inherits the caller's features through inlining. Without it, intrinsics become function calls and the path regresses ~18× even inside a `#[magetypes]`-generated variant. See `benches/generic_vs_concrete.rs`.

### C. Slot a hand-tuned tier into an existing `#[magetypes]` family

When one tier benefits from something the generic API can't express (AVX-512 mask shuffles, cross-lane NEON permutes), write a standalone `#[arcane]` named to match the suffix convention. `incant!` doesn't care which macro or which hand-written function produced which variant — it resolves by name:

```rust
use magetypes::simd::generic::f32x16 as GenericF32x16;

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
fn scale_plane_impl_v4x(token: X64V4xToken, plane: &mut [f32], factor: f32) {
    let factor_v = GenericF32x16::<X64V4xToken>::splat(token, factor);
    let (chunks, tail) = GenericF32x16::<X64V4xToken>::partition_slice_mut(token, plane);
    for chunk in chunks {
        (GenericF32x16::<X64V4xToken>::load(token, chunk) * factor_v).store(chunk);
    }
    for v in tail { *v *= factor; }
}

pub fn scale_plane_fast(plane: &mut [f32], factor: f32) {
    incant!(scale_plane_impl(plane, factor),
            [v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar])
}
```

The `_v4x` name slots into the tier list. Everything else is served by `#[magetypes]`'s generated variants.

### D. `#[autoversion]` for scalar loops the compiler vectorizes

Plain scalar body, recompiled per tier with different `#[target_feature]`; LLVM auto-vectorizes each copy. `#[autoversion]` is the only generator that emits its own dispatcher — call it directly, no `incant!` needed:

```rust
#[autoversion]
fn apply_color_matrix(rgb: &mut [f32], mat: [[f32; 3]; 3]) {
    for pixel in rgb.chunks_exact_mut(3) {
        let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
        pixel[0] = mat[0][0] * r + mat[0][1] * g + mat[0][2] * b;
        pixel[1] = mat[1][0] * r + mat[1][1] * g + mat[1][2] * b;
        pixel[2] = mat[2][0] * r + mat[2][1] * g + mat[2][2] * b;
    }
}
// apply_color_matrix(&mut rgb, mat);  // dispatcher is built in
```

### E. Nested `incant!` is zero-overhead

Inside any tier-annotated body (`#[magetypes]`, `#[autoversion]`, `#[arcane]`, token-based `#[rite]`), `incant!(foo(args))` is rewritten to the direct tier-matching call at compile time. No dispatcher branch, no cache probe — the callee inlines into the caller's `#[target_feature]` region.

```rust
#[magetypes(v4, v3, neon, wasm128, scalar)]
fn pipeline_impl(token: Token, plane: &mut [f32], bias: f32, factor: f32) {
    // ... bias each lane ...

    // These two calls are rewritten per tier:
    //   V3 variant: incant!(clamp01_impl(plane)) → clamp01_impl_v3(token, plane)
    //   NEON:                                     → clamp01_impl_neon(token, plane)
    // Zero dispatcher hops.
    incant!(clamp01_impl(plane));
    incant!(scale_plane_impl(plane, factor));
}
```

See [SPEC-INCANT-REWRITING.md](https://github.com/imazen/archmage/blob/main/docs/SPEC-INCANT-REWRITING.md) for the rules and measurements (0.94 ns vs 5.6 ns with redispatch).

### F. `#[rite]` for explicit target-feature inner helpers

`#[rite(v3, v4, neon, wasm128)]` generates per-tier copies of a function with `#[target_feature]` + `#[inline]` attached directly — no wrapper, no optimization boundary. Callable by suffix from matching contexts.

- `#[rite]` does NOT substitute `Token`. Each variant is the same body with a different `#[target_feature]`; the signature is verbatim (tokenless or concrete token type).
- `#[rite]` has **no `scalar` tier** — scalar has no features to enable. For scalar fallback, use `#[magetypes]` or a plain `fn foo_scalar(_: ScalarToken, ...)`.

Reach for `#[rite]` when you want explicit target-feature control inside an `#[arcane]`/`#[magetypes]` body without delegating through the generic-bound pattern. For most magetypes code, the extracted generic kernel (Pattern B) is cleaner.

## The five macros

| Macro | What it does | Generates | Own dispatcher? |
|---|---|---|---|
| `#[arcane]` | Wraps a function with `#[target_feature]` via a safe outer `fn` + `unsafe { ... }` inner call | 1 function (wrapper + sibling) | No |
| `#[rite]` | Applies `#[target_feature]` + `#[inline]` directly, no wrapper | 1 (single-tier) or N (multi-tier) | No |
| `#[magetypes]` | Per-tier copies with `Token` substituted, each wrapped like `#[arcane]` | N per listed tier | No |
| `#[autoversion]` | Per-tier copies of scalar body, bundled dispatcher | N + 1 dispatcher | **Yes** |
| `incant!` | Runtime dispatch to pre-existing `_v3`/`_v4`/`_neon`/… variants; rewrites to direct calls inside tier-annotated bodies | 0 (just dispatches) | The dispatcher itself |

All five share the `_<tier>` suffix convention, so `incant!` routes across variants generated by any combination of them. A single `incant!` call can dispatch to: a `#[magetypes]`-generated `_v3`, a hand-written `#[arcane]` `_v4x`, a `#[rite(v3, v4, neon)]` `_neon`, and a manually-written `_scalar` — it doesn't care who authored which.

## Types

| Category | Types |
|---|---|
| 128-bit | `f32x4`, `f64x2`, `i8x16`, `i16x8`, `i32x4`, `i64x2`, `u8x16`, `u16x8`, `u32x4`, `u64x2` |
| 256-bit | `f32x8`, `f64x4`, `i8x32`, `i16x16`, `i32x8`, `i64x4`, `u8x32`, `u16x16`, `u32x8`, `u64x4` |
| 512-bit (feature `w512`, default) | `f32x16`, `f64x8`, `i8x64`, `i16x32`, `i32x16`, `i64x8`, `u8x64`, `u16x32`, `u32x16`, `u64x8` |

Native on x86-64 (AVX2, AVX-512 with `avx512` feature); polyfilled on NEON/WASM via pairs or quads of 128-bit ops. Same API on every architecture.

## Cargo features

| Feature | Default | Effect |
|---|---|---|
| `std` | yes | Enable std (disable for `no_std + alloc`) |
| `w512` | yes | 512-bit types (`f32x16`, …), polyfilled on non-AVX-512 targets |
| `avx512` | no | Native AVX-512 impls on x86-64 (implies `w512`) |

## Platform support

| Target | Status | Tokens |
|---|---|---|
| x86-64 | Full | `X64V3Token` (AVX2+FMA), `X64V4Token`/`X64V4xToken` (AVX-512) |
| aarch64 | Full | `NeonToken`, `Arm64V2Token`, `Arm64V3Token` |
| wasm32 | Full | `Wasm128Token` (build with `-C target-feature=+simd128`) |

Same generic API on every platform. Cross-architecture behavioral differences (signed-shift semantics, NaN propagation, signed-zero negation, FMA rounding) are documented in `CLAUDE.md` under "Known Cross-Architecture Behavioral Differences."

## Relationship to archmage

`magetypes` depends on [`archmage`](https://crates.io/crates/archmage) for tokens, the `#[arcane]` / `#[rite]` / `#[magetypes]` / `#[autoversion]` macros, `incant!`, and runtime CPU feature detection.

Use `archmage` directly when you want raw intrinsics behind a token. Use `magetypes` when you want portable SIMD types with natural operators. Both are compatible with `#![forbid(unsafe_code)]` — the `unsafe` lives inside proc-macro output, not in your source.

## License

MIT OR Apache-2.0

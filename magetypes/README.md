# magetypes

Token-gated SIMD types with natural operators.

[![Crates.io](https://img.shields.io/crates/v/magetypes.svg)](https://crates.io/crates/magetypes)
[![Documentation](https://docs.rs/magetypes/badge.svg)](https://docs.rs/magetypes)
[![License](https://img.shields.io/crates/l/magetypes.svg)](LICENSE)

**[Intrinsics Browser](https://imazen.github.io/archmage/intrinsics/)** · [Tutorial Book](https://imazen.github.io/archmage/) · [API Docs](https://docs.rs/magetypes)

## Overview

`magetypes` provides SIMD vector types (`f32x8`, `i32x4`, etc.) that require [archmage](https://crates.io/crates/archmage) tokens for safe construction. This ensures SIMD operations are only performed when CPU features have been verified at runtime.

**Key features:**
- Natural operators (`+`, `-`, `*`, `/`, `&`, `|`, `^`)
- Token-gated construction (safe by design)
- Zero-cost abstractions (compiles to raw SIMD instructions)
- Cross-platform (x86-64 with AVX2/AVX-512, AArch64 with NEON, WASM with SIMD128)

## Quick Start

```rust
use archmage::{X64V3Token, SimdToken};
use magetypes::simd::f32x8;

fn main() {
    // Token proves CPU supports AVX2+FMA
    if let Some(token) = X64V3Token::summon() {
        let a = f32x8::splat(token, 1.0);
        let b = f32x8::splat(token, 2.0);
        let c = a + b;  // Natural operators!

        println!("Result: {:?}", c.to_array());
    }
}
```

## Available Types

### x86-64 (x86-64-v3 - 128-bit)
`f32x4`, `f64x2`, `i8x16`, `i16x8`, `i32x4`, `i64x2`, `u8x16`, `u16x8`, `u32x4`, `u64x2`

### x86-64 (x86-64-v3 - 256-bit)
`f32x8`, `f64x4`, `i8x32`, `i16x16`, `i32x8`, `i64x4`, `u8x32`, `u16x16`, `u32x8`, `u64x4`

### x86-64 (AVX-512 - 512-bit, requires `avx512` feature)
`f32x16`, `f64x8`, `i8x64`, `i16x32`, `i32x16`, `i64x8`, `u8x64`, `u16x32`, `u32x16`, `u64x8`

### AArch64 (NEON - 128-bit)
`f32x4`, `f64x2`, `i8x16`, `i16x8`, `i32x4`, `i64x2`, `u8x16`, `u16x8`, `u32x4`, `u64x2`

### WASM (SIMD128 - 128-bit)
`f32x4`, `f64x2`, `i8x16`, `i16x8`, `i32x4`, `i64x2`, `u8x16`, `u16x8`, `u32x4`, `u64x2`

Build with `RUSTFLAGS="-C target-feature=+simd128"` for WASM targets.

```rust
// WASM example - no runtime detection needed
use archmage::{Wasm128Token, SimdToken};
use magetypes::simd::f32x4;

// When compiled with +simd128, token is always available
let token = Wasm128Token::summon().unwrap();
let a = f32x4::splat(token, 1.0);
let b = f32x4::splat(token, 2.0);
let c = a + b;
```

## Token-Gated Construction

All constructors require a token proving CPU support:

```rust
// Load from array
let v = f32x8::load(token, &data);

// Broadcast scalar
let v = f32x8::splat(token, 42.0);

// Zero vector
let v = f32x8::zero(token);

// From array (zero-cost transmute)
let v = f32x8::from_array(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

// From bytes
let v = f32x8::from_bytes(token, &bytes);
```

## Per-Token Namespaces

Each token level has a namespace with `f32xN` type aliases at the natural width, a `Token` type alias, and `LANES_*` constants:

| Namespace | Token | `f32xN` | `LANES_F32` |
|-----------|-------|---------|-------------|
| `magetypes::simd::v3` | `X64V3Token` | `f32x8` | 8 |
| `magetypes::simd::v4` | `X64V4Token` | `f32x16` | 16 |
| `magetypes::simd::neon` | `NeonToken` | `f32x4` | 4 |
| `magetypes::simd::wasm128` | `Wasm128Token` | `f32x4` | 4 |

Each namespace also includes narrower native types and wider polyfilled types. For example, `v3` includes native 128-bit types and polyfilled 512-bit types (emulated via 2x256-bit ops).

### Scalar polyfills

`magetypes::simd::scalar` provides `f32x1`, `f64x1`, `i32x1`, etc. — single-element types with the same API as SIMD types, taking `ScalarToken`. These are used for scalar fallback code.

## Platform Support

| Platform | Status | Token | Vector Sizes |
|----------|--------|-------|--------------|
| x86-64 | **Full** | `X64V3Token`, `X64V4Token` | 128, 256, 512-bit |
| AArch64 | **Full** | `NeonToken` | 128-bit |
| WASM | **Full** | `Wasm128Token` | 128-bit |

## Features

- **`std`** (default): Enable std library support
- **`avx512`**: Enable 512-bit types for AVX-512

## Mixed Dispatch: Generic + Specialized + Auto-Vectorized

The full pattern: `#[magetypes]` generates generic variants from one function body, a manual `#[arcane]` adds a hand-tuned specialization for one tier, and `#[autoversion]` auto-vectorizes scalar code. All three produce the same `_suffix` naming convention, so one `incant!` dispatches to them all.

```rust
use archmage::prelude::*;
use magetypes::simd::generic::f32x8 as GenericF32x8;

// 1. Generic algorithm — #[magetypes] generates _v4, _v3, _neon, _wasm128, _scalar
#[magetypes(v4, v3, neon, wasm128, scalar)]
fn process_impl(token: Token, data: &mut [f32], scale: f32) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let scale_v = f32x8::splat(token, scale);
    let (chunks, tail) = f32x8::partition_slice_mut(token, data);
    for chunk in chunks {
        let v = f32x8::load(token, chunk);
        (v * scale_v).store(chunk);
    }
    for v in tail { *v *= scale; }
}

// 2. Manual specialization for v4x — uses safe AVX-512 intrinsics
//    not available in the generic f32x8 API. The _v4x suffix matches
//    incant!'s naming convention.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
fn process_impl_v4x(_token: X64V4xToken, data: &mut [f32], scale: f32) {
    let scale_v = _mm512_set1_ps(scale);
    for chunk in data.chunks_exact_mut(16) {
        let v = _mm512_loadu_ps(chunk.as_ptr());
        _mm512_storeu_ps(chunk.as_mut_ptr(), _mm512_mul_ps(v, scale_v));
    }
    // ... scalar tail
}

// 3. One incant! dispatches to ALL variants — generated + manual.
//    v4x(cfg(avx512)) feature-gates that tier: excluded if the avx512
//    cargo feature isn't enabled by the downstream crate.
pub fn process(data: &mut [f32], scale: f32) {
    incant!(process_impl(data, scale),
        [v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar]);
}
```

`incant!` doesn't know or care which macro generated each variant — it just looks for functions named `process_impl_v4x`, `process_impl_v4`, etc.

### `#[rite]` multi-tier — suffixed inner helpers

`#[rite(v3, v4, neon)]` generates `_v3`, `_v4`, `_neon` suffixed copies of a helper function, each with `#[target_feature]` + `#[inline]`. Use for inner functions called from `#[arcane]` entry points — zero dispatch overhead, just inlining:

```rust
// Generates accumulate_v3, accumulate_v4, accumulate_neon — all inlined
#[rite(v3, v4, neon, import_intrinsics)]
fn accumulate(data: &[f32; 8], acc: f32) -> f32 {
    let v = _mm256_loadu_ps(data.as_ptr());
    // ...
    acc
}
```

### `#[autoversion]` — Auto-Vectorized Scalar Code

For loops that LLVM auto-vectorizes well, skip magetypes entirely. `#[autoversion]` generates tier variants AND a dispatcher from plain scalar code:

```rust
use archmage::autoversion;

/// Compiles to vfmadd231ps (AVX2), fmla (NEON), etc. — zero manual SIMD.
#[autoversion]
fn apply_color_matrix(rgb: &mut [f32], mat: [[f32; 3]; 3]) {
    for pixel in rgb.chunks_exact_mut(3) {
        let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
        pixel[0] = mat[0][0] * r + mat[0][1] * g + mat[0][2] * b;
        pixel[1] = mat[1][0] * r + mat[1][1] * g + mat[1][2] * b;
        pixel[2] = mat[2][0] * r + mat[2][1] * g + mat[2][2] * b;
    }
}

// Call directly — autoversion generated the dispatcher too:
apply_color_matrix(&mut pixels, matrix);
```

### When to use which

| Approach | Use when | Dispatch |
|---|---|---|
| `#[magetypes]` + `incant!` | You need explicit SIMD types (`f32x8`, `i32x4`) | Manual `incant!` |
| `#[arcane]` + `incant!` | One tier needs hand-tuned intrinsics | Manual `incant!` |
| `#[autoversion]` | Scalar code that LLVM auto-vectorizes well | Built-in |
| All three mixed | Most tiers are generic, one needs intrinsics, entry is auto-vectorized | One `incant!` handles all |

### Attribute parameter reference

| Parameter | `#[arcane]` | `#[rite]` | `#[magetypes]` | `#[autoversion]` |
|---|---|---|---|---|
| Tier names (`v3`, `neon`, ...) | — | **Yes** (suffixed variants) | **Yes** (suffixed variants) | **Yes** (suffixed + dispatcher) |
| `+tier` / `-tier` modifiers | — | — | **Yes** | **Yes** |
| `tier(cfg(feature))` gate | — | — | **Yes** | **Yes** |
| `import_intrinsics` | **Yes** | **Yes** | auto | auto |
| `import_magetypes` | **Yes** | **Yes** | auto | auto |
| `cfg(feature)` | **Yes** | **Yes** | — | **Yes** |
| `_self = Type` | **Yes** | — | — | **Yes** |
| `nested` | **Yes** | — | — | — |
| `inline_always` | **Yes** (nightly) | — | — | — |

**Tier suffixes:** `_v1`, `_v2`, `_x64_crypto`, `_v3`, `_v3_crypto`, `_v4`, `_v4x`, `_neon`, `_neon_aes`, `_neon_sha3`, `_neon_crc`, `_arm_v2`, `_arm_v3`, `_wasm128`, `_wasm128_relaxed`, `_scalar`, `_default`.

**Default tiers** (when no list given): `v4(avx512)`, `v3`, `neon`, `wasm128`, `scalar`.

### Plain `incant!` dispatch

For simpler cases without mixing, write suffixed variants directly:

```rust
use archmage::incant;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8;

#[cfg(target_arch = "x86_64")]
fn dot_product_v3(token: archmage::X64V3Token, a: &[f32], b: &[f32]) -> f32 {
    let mut acc = f32x8::zero(token);
    for (a_chunk, b_chunk) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        let va = f32x8::from_array(token, a_chunk.try_into().unwrap());
        let vb = f32x8::from_array(token, b_chunk.try_into().unwrap());
        acc = va.mul_add(vb, acc);
    }
    acc.reduce_add()
}

fn dot_product_scalar(_token: archmage::ScalarToken, a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    incant!(dot_product(a, b))
}
```

Explicit tier lists, feature gates, and modifiers all work: `incant!(dot_product(a, b), [v3, neon, scalar])`, `incant!(dot_product(a, b), [v4(cfg(avx512)), v3, scalar])`, `incant!(dot_product(a, b), [+arm_v2])`.

This works with `#![forbid(unsafe_code)]` — magetypes methods handle unsafe internally via `#[inline(always)]`.

## Using with #[arcane] and #[rite]

`#[arcane]` reads the token type from the signature to emit `#[target_feature]`. It generates a wrapper that crosses the boundary without `unsafe` at the call site — but the wrapper creates an LLVM optimization boundary. `#[rite]` applies `#[target_feature]` + `#[inline]` directly, with no wrapper and no boundary. It works in three modes: token-based (`#[rite]`), tier-based (`#[rite(v3)]` — no token needed), or multi-tier (`#[rite(v3, v4, neon)]` — generates suffixed variants).

**`#[rite]` should be your default.** Use `#[arcane]` only at the entry point (the first call from non-SIMD code), and `#[rite]` for everything called from within SIMD code. Both are compatible with `#![forbid(unsafe_code)]`.

```rust
use archmage::prelude::*;
use magetypes::simd::f32x8;

#[arcane(import_intrinsics)]
pub fn dot_product(token: X64V3Token, a: &[f32], b: &[f32]) -> f32 {
    let mut acc = f32x8::zero(token);
    for (a_chunk, b_chunk) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        acc = accumulate(token, acc, a_chunk, b_chunk);
    }
    acc.reduce_add()
}

#[rite(import_intrinsics)]
fn accumulate(token: X64V3Token, acc: f32x8, a: &[f32], b: &[f32]) -> f32x8 {
    let va = f32x8::from_array(token, a.try_into().unwrap());
    let vb = f32x8::from_array(token, b.try_into().unwrap());
    va.mul_add(vb, acc)
}
```

`#[rite]` inlines with zero overhead. `#[arcane]` creates a wrapper (and an optimization boundary). Use `#[rite]` for everything called from SIMD code.

## Relationship to archmage

`magetypes` depends on `archmage` for:
- Token types (`X64V3Token`, `Arm64`, etc.)
- The `#[arcane]` and `#[rite]` macros
- Runtime CPU feature detection

Use `archmage` directly when you need raw intrinsics. Use `magetypes` when you want ergonomic SIMD types with operators.

## License

MIT OR Apache-2.0

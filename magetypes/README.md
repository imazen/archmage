# magetypes

Token-gated SIMD types with natural operators.

[![Crates.io](https://img.shields.io/crates/v/magetypes.svg)](https://crates.io/crates/magetypes)
[![Documentation](https://docs.rs/magetypes/badge.svg)](https://docs.rs/magetypes)
[![License](https://img.shields.io/crates/l/magetypes.svg)](LICENSE)

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

## Using with `incant!` for runtime dispatch

The recommended pattern for multi-platform SIMD: write a `_v3` variant with concrete SIMD types and a `_scalar` fallback, then dispatch with `incant!`:

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

This works with `#![forbid(unsafe_code)]` — magetypes methods handle unsafe internally via `#[inline(always)]`.

## Using with #[arcane] and #[rite]

Both macros read the token type from your function signature to decide which `#[target_feature]` to emit. `Desktop64` → `avx2,fma,...`. `X64V4Token` → `avx512f,...`. The token type is the feature selector.

`#[arcane]` generates a wrapper that crosses the `#[target_feature]` boundary without `unsafe` at the call site — but the wrapper itself creates an LLVM optimization boundary. `#[rite]` applies `#[target_feature]` + `#[inline]` directly, with no wrapper and no boundary.

**`#[rite]` should be your default.** Use `#[arcane]` only at the entry point (the first call from non-SIMD code), and `#[rite]` for everything called from within SIMD code. Passing the same token through your call hierarchy keeps features consistent, so LLVM inlines freely. Both are compatible with `#![forbid(unsafe_code)]`.

```rust
use archmage::prelude::*;
use magetypes::simd::f32x8;

#[arcane]
pub fn dot_product(token: Desktop64, a: &[f32], b: &[f32]) -> f32 {
    let mut acc = f32x8::zero(token);
    for (a_chunk, b_chunk) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        acc = accumulate(token, acc, a_chunk, b_chunk);
    }
    acc.reduce_add()
}

#[rite]
fn accumulate(token: Desktop64, acc: f32x8, a: &[f32], b: &[f32]) -> f32x8 {
    let va = f32x8::from_array(token, a.try_into().unwrap());
    let vb = f32x8::from_array(token, b.try_into().unwrap());
    va.mul_add(vb, acc)
}
```

`#[rite]` inlines with zero overhead. `#[arcane]` creates a wrapper (and an optimization boundary). Use `#[rite]` for everything called from SIMD code.

## Relationship to archmage

`magetypes` depends on `archmage` for:
- Token types (`Desktop64`, `Arm64`, etc.)
- The `#[arcane]` and `#[rite]` macros
- Runtime CPU feature detection

Use `archmage` directly when you need raw intrinsics. Use `magetypes` when you want ergonomic SIMD types with operators.

## License

MIT OR Apache-2.0

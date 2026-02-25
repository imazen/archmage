+++
title = "Type Overview"
weight = 1
+++

Magetypes provides SIMD vector types with natural Rust operators. Each type wraps platform intrinsics and requires an [archmage token](@/archmage/getting-started/tokens.md) for construction.

## Available Types

### x86-64

| Type | Elements | Width | Min Token |
|------|----------|-------|-----------|
| `f32x4` | 4 x f32 | 128-bit | `X64V2Token` |
| `f32x8` | 8 x f32 | 256-bit | `X64V3Token` |
| `f32x16` | 16 x f32 | 512-bit | `X64V4Token`* |
| `f64x2` | 2 x f64 | 128-bit | `X64V2Token` |
| `f64x4` | 4 x f64 | 256-bit | `X64V3Token` |
| `f64x8` | 8 x f64 | 512-bit | `X64V4Token`* |
| `i8x16` | 16 x i8 | 128-bit | `X64V2Token` |
| `i8x32` | 32 x i8 | 256-bit | `X64V3Token` |
| `i16x8` | 8 x i16 | 128-bit | `X64V2Token` |
| `i16x16` | 16 x i16 | 256-bit | `X64V3Token` |
| `i32x4` | 4 x i32 | 128-bit | `X64V2Token` |
| `i32x8` | 8 x i32 | 256-bit | `X64V3Token` |
| `i32x16` | 16 x i32 | 512-bit | `X64V4Token`* |
| `i64x2` | 2 x i64 | 128-bit | `X64V2Token` |
| `i64x4` | 4 x i64 | 256-bit | `X64V3Token` |
| `u8x16` | 16 x u8 | 128-bit | `X64V2Token` |
| `u8x32` | 32 x u8 | 256-bit | `X64V3Token` |
| `u16x8` | 8 x u16 | 128-bit | `X64V2Token` |
| `u16x16` | 16 x u16 | 256-bit | `X64V3Token` |
| `u32x4` | 4 x u32 | 128-bit | `X64V2Token` |
| `u32x8` | 8 x u32 | 256-bit | `X64V3Token` |
| `u64x2` | 2 x u64 | 128-bit | `X64V2Token` |
| `u64x4` | 4 x u64 | 256-bit | `X64V3Token` |

*Requires the `avx512` feature flag.

### AArch64 (NEON)

| Type | Elements | Width | Token |
|------|----------|-------|-------|
| `f32x4` | 4 x f32 | 128-bit | `NeonToken` |
| `f64x2` | 2 x f64 | 128-bit | `NeonToken` |
| `i8x16` | 16 x i8 | 128-bit | `NeonToken` |
| `i16x8` | 8 x i16 | 128-bit | `NeonToken` |
| `i32x4` | 4 x i32 | 128-bit | `NeonToken` |
| `i64x2` | 2 x i64 | 128-bit | `NeonToken` |
| `u8x16` | 16 x u8 | 128-bit | `NeonToken` |
| `u16x8` | 8 x u16 | 128-bit | `NeonToken` |
| `u32x4` | 4 x u32 | 128-bit | `NeonToken` |
| `u64x2` | 2 x u64 | 128-bit | `NeonToken` |

NEON registers are 128-bit. Wider types (`f32x8`, etc.) are available as [polyfills](@/magetypes/cross-platform/polyfills.md) using pairs of NEON operations.

### WASM (SIMD128)

| Type | Elements | Width | Token |
|------|----------|-------|-------|
| `f32x4` | 4 x f32 | 128-bit | `Wasm128Token` |
| `f64x2` | 2 x f64 | 128-bit | `Wasm128Token` |
| `i8x16` | 16 x i8 | 128-bit | `Wasm128Token` |
| `i16x8` | 8 x i16 | 128-bit | `Wasm128Token` |
| `i32x4` | 4 x i32 | 128-bit | `Wasm128Token` |
| `i64x2` | 2 x i64 | 128-bit | `Wasm128Token` |
| `u8x16` | 16 x u8 | 128-bit | `Wasm128Token` |
| `u16x8` | 8 x u16 | 128-bit | `Wasm128Token` |
| `u32x4` | 4 x u32 | 128-bit | `Wasm128Token` |
| `u64x2` | 2 x u64 | 128-bit | `Wasm128Token` |

Wider types are available as polyfills, same as ARM.

## Basic Usage

```rust
use archmage::{Desktop64, SimdToken};
use magetypes::simd::f32x8;

fn example() {
    if let Some(token) = Desktop64::summon() {
        // Construct from array
        let a = f32x8::from_array(token, [1.0; 8]);

        // Splat a single value
        let b = f32x8::splat(token, 2.0);

        // Natural operators
        let c = a + b;
        let d = c * c;

        // Extract result
        let result: [f32; 8] = d.to_array();
    }
}
```

## Type Properties

All magetypes SIMD types are:

- **Copy** — pass by value freely
- **Clone** — explicit cloning works
- **Debug** — print for debugging
- **Send + Sync** — thread-safe

```rust
// Zero-cost copies
let a = f32x8::splat(token, 1.0);
let b = a;  // Copy, not move
let c = a + b;  // Both still valid
```

**Why no `Pod`/`Zeroable`?** Implementing bytemuck traits would let users bypass token-gated construction (e.g., `bytemuck::zeroed::<f32x8>()`), creating vectors without proving CPU support. Use the token-gated [cast_slice and from_bytes](@/magetypes/conversions/slice-casting.md) methods instead.

## Using the Prelude

Import everything at once:

```rust
use magetypes::prelude::*;

// All types + archmage re-exports available
if let Some(token) = Desktop64::summon() {
    let v = f32x8::splat(token, 1.0);
}
```

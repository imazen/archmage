+++
title = "Type Overview"
weight = 1
+++

Magetypes provides generic SIMD vector types parameterized by a backend token. Each type is written as `f32x8<T>` where `T` is a token type that determines the platform implementation. A function generic over `T: F32x8Backend` works on any backend that supports 8-lane f32 operations.

## Available Types

### x86-64

| Type | Elements | Width | Min Token |
|------|----------|-------|-----------|
| `f32x4<T>` | 4 x f32 | 128-bit | `X64V2Token` |
| `f32x8<T>` | 8 x f32 | 256-bit | `X64V3Token` |
| `f32x16<T>` | 16 x f32 | 512-bit | `X64V4Token`* |
| `f64x2<T>` | 2 x f64 | 128-bit | `X64V2Token` |
| `f64x4<T>` | 4 x f64 | 256-bit | `X64V3Token` |
| `f64x8<T>` | 8 x f64 | 512-bit | `X64V4Token`* |
| `i8x16<T>` | 16 x i8 | 128-bit | `X64V2Token` |
| `i8x32<T>` | 32 x i8 | 256-bit | `X64V3Token` |
| `i16x8<T>` | 8 x i16 | 128-bit | `X64V2Token` |
| `i16x16<T>` | 16 x i16 | 256-bit | `X64V3Token` |
| `i32x4<T>` | 4 x i32 | 128-bit | `X64V2Token` |
| `i32x8<T>` | 8 x i32 | 256-bit | `X64V3Token` |
| `i32x16<T>` | 16 x i32 | 512-bit | `X64V4Token`* |
| `i64x2<T>` | 2 x i64 | 128-bit | `X64V2Token` |
| `i64x4<T>` | 4 x i64 | 256-bit | `X64V3Token` |
| `u8x16<T>` | 16 x u8 | 128-bit | `X64V2Token` |
| `u8x32<T>` | 32 x u8 | 256-bit | `X64V3Token` |
| `u16x8<T>` | 8 x u16 | 128-bit | `X64V2Token` |
| `u16x16<T>` | 16 x u16 | 256-bit | `X64V3Token` |
| `u32x4<T>` | 4 x u32 | 128-bit | `X64V2Token` |
| `u32x8<T>` | 8 x u32 | 256-bit | `X64V3Token` |
| `u64x2<T>` | 2 x u64 | 128-bit | `X64V2Token` |
| `u64x4<T>` | 4 x u64 | 256-bit | `X64V3Token` |

*Requires the `avx512` feature flag.

### AArch64 (NEON)

| Type | Elements | Width | Token |
|------|----------|-------|-------|
| `f32x4<T>` | 4 x f32 | 128-bit | `NeonToken` |
| `f64x2<T>` | 2 x f64 | 128-bit | `NeonToken` |
| `i8x16<T>` | 16 x i8 | 128-bit | `NeonToken` |
| `i16x8<T>` | 8 x i16 | 128-bit | `NeonToken` |
| `i32x4<T>` | 4 x i32 | 128-bit | `NeonToken` |
| `i64x2<T>` | 2 x i64 | 128-bit | `NeonToken` |
| `u8x16<T>` | 16 x u8 | 128-bit | `NeonToken` |
| `u16x8<T>` | 8 x u16 | 128-bit | `NeonToken` |
| `u32x4<T>` | 4 x u32 | 128-bit | `NeonToken` |
| `u64x2<T>` | 2 x u64 | 128-bit | `NeonToken` |

NEON registers are 128-bit. Wider types (`f32x8<T>`, etc.) are available as [polyfills](@/magetypes/cross-platform/polyfills.md) using pairs of NEON operations.

### WASM (SIMD128)

| Type | Elements | Width | Token |
|------|----------|-------|-------|
| `f32x4<T>` | 4 x f32 | 128-bit | `Wasm128Token` |
| `f64x2<T>` | 2 x f64 | 128-bit | `Wasm128Token` |
| `i8x16<T>` | 16 x i8 | 128-bit | `Wasm128Token` |
| `i16x8<T>` | 8 x i16 | 128-bit | `Wasm128Token` |
| `i32x4<T>` | 4 x i32 | 128-bit | `Wasm128Token` |
| `i64x2<T>` | 2 x i64 | 128-bit | `Wasm128Token` |
| `u8x16<T>` | 16 x u8 | 128-bit | `Wasm128Token` |
| `u16x8<T>` | 8 x u16 | 128-bit | `Wasm128Token` |
| `u32x4<T>` | 4 x u32 | 128-bit | `Wasm128Token` |
| `u64x2<T>` | 2 x u64 | 128-bit | `Wasm128Token` |

Wider types are available as polyfills, same as ARM.

## Basic Usage

The correct pattern is a generic function bounded by the appropriate backend trait. The type parameter `T` is satisfied at the call site by whichever token the caller holds.

```rust
use archmage::{Desktop64, SimdToken};
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Backend,
};

#[inline(always)]
fn example<T: F32x8Backend>(token: T) {
    // Construct from array
    let a = f32x8::<T>::from_array(token, [1.0; 8]);

    // Splat a single value
    let b = f32x8::<T>::splat(token, 2.0);

    // Natural operators
    let c = a + b;
    let d = c * c;

    // Extract result
    let result: [f32; 8] = d.to_array();
}

// Call site: summon the token, then call the generic function
if let Some(token) = Desktop64::summon() {
    example(token);
}
```

## Type Properties

All magetypes SIMD types are:

- **Copy** — pass by value freely
- **Clone** — explicit cloning works
- **Debug** — print for debugging
- **Send + Sync** — thread-safe

```rust
use magetypes::simd::{generic::f32x8, backends::F32x8Backend};

#[inline(always)]
fn copy_demo<T: F32x8Backend>(token: T) {
    let a = f32x8::<T>::splat(token, 1.0);
    let b = a;  // Copy, not move
    let c = a + b;  // Both still valid
}
```

**Why no `Pod`/`Zeroable`?** Implementing bytemuck traits would let users bypass token-gated construction (e.g., `bytemuck::zeroed::<f32x8<T>>()`), creating vectors without proving CPU support. Use the token-gated [cast_slice and from_bytes](@/magetypes/conversions/slice-casting.md) methods instead.

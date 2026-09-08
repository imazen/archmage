+++
title = "Type Overview"
weight = 1
+++

Magetypes provides generic SIMD vector types parameterized by a backend token. Each type is written as [`f32x8<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html) where `T` is a token type that determines the platform implementation. A function generic over `T: F32x8Backend` works on any backend that supports 8-lane f32 operations.

## Available Types

### Native x86-64 shapes

| Type | Elements | Width | Native Token |
|------|----------|-------|-----------|
| [`f32x4<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x4.html) | 4 x f32 | 128-bit | `X64V2Token` |
| [`f32x8<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html) | 8 x f32 | 256-bit | `X64V3Token` |
| [`f32x16<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x16.html) | 16 x f32 | 512-bit | `X64V4Token`* |
| [`f64x2<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f64x2.html) | 2 x f64 | 128-bit | `X64V2Token` |
| [`f64x4<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f64x4.html) | 4 x f64 | 256-bit | `X64V3Token` |
| [`f64x8<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f64x8.html) | 8 x f64 | 512-bit | `X64V4Token`* |
| [`i8x16<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.i8x16.html) | 16 x i8 | 128-bit | `X64V2Token` |
| [`i8x32<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.i8x32.html) | 32 x i8 | 256-bit | `X64V3Token` |
| [`i16x8<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.i16x8.html) | 8 x i16 | 128-bit | `X64V2Token` |
| [`i16x16<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.i16x16.html) | 16 x i16 | 256-bit | `X64V3Token` |
| [`i32x4<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.i32x4.html) | 4 x i32 | 128-bit | `X64V2Token` |
| [`i32x8<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.i32x8.html) | 8 x i32 | 256-bit | `X64V3Token` |
| [`i32x16<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.i32x16.html) | 16 x i32 | 512-bit | `X64V4Token`* |
| [`i64x2<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.i64x2.html) | 2 x i64 | 128-bit | `X64V2Token` |
| [`i64x4<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.i64x4.html) | 4 x i64 | 256-bit | `X64V3Token` |
| [`u8x16<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.u8x16.html) | 16 x u8 | 128-bit | `X64V2Token` |
| [`u8x32<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.u8x32.html) | 32 x u8 | 256-bit | `X64V3Token` |
| [`u16x8<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.u16x8.html) | 8 x u16 | 128-bit | `X64V2Token` |
| [`u16x16<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.u16x16.html) | 16 x u16 | 256-bit | `X64V3Token` |
| [`u32x4<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.u32x4.html) | 4 x u32 | 128-bit | `X64V2Token` |
| [`u32x8<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.u32x8.html) | 8 x u32 | 256-bit | `X64V3Token` |
| [`u64x2<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.u64x2.html) | 2 x u64 | 128-bit | `X64V2Token` |
| [`u64x4<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.u64x4.html) | 4 x u64 | 256-bit | `X64V3Token` |

*Native 512-bit implementations require `avx512`. Logical 512-bit types also have polyfills under the default `w512` feature. This table lists native implementations, not every supported token/shape combination.

### AArch64 (NEON)

| Type | Elements | Width | Token |
|------|----------|-------|-------|
| [`f32x4<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x4.html) | 4 x f32 | 128-bit | `NeonToken` |
| [`f64x2<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f64x2.html) | 2 x f64 | 128-bit | `NeonToken` |
| [`i8x16<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.i8x16.html) | 16 x i8 | 128-bit | `NeonToken` |
| [`i16x8<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.i16x8.html) | 8 x i16 | 128-bit | `NeonToken` |
| [`i32x4<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.i32x4.html) | 4 x i32 | 128-bit | `NeonToken` |
| [`i64x2<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.i64x2.html) | 2 x i64 | 128-bit | `NeonToken` |
| [`u8x16<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.u8x16.html) | 16 x u8 | 128-bit | `NeonToken` |
| [`u16x8<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.u16x8.html) | 8 x u16 | 128-bit | `NeonToken` |
| [`u32x4<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.u32x4.html) | 4 x u32 | 128-bit | `NeonToken` |
| [`u64x2<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.u64x2.html) | 2 x u64 | 128-bit | `NeonToken` |

NEON registers are 128-bit. Wider types ([`f32x8<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html), etc.) are available as [polyfills](@/magetypes/cross-platform/polyfills.md) using pairs of NEON operations.

### WASM (SIMD128)

| Type | Elements | Width | Token |
|------|----------|-------|-------|
| [`f32x4<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x4.html) | 4 x f32 | 128-bit | `Wasm128Token` |
| [`f64x2<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f64x2.html) | 2 x f64 | 128-bit | `Wasm128Token` |
| [`i8x16<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.i8x16.html) | 16 x i8 | 128-bit | `Wasm128Token` |
| [`i16x8<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.i16x8.html) | 8 x i16 | 128-bit | `Wasm128Token` |
| [`i32x4<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.i32x4.html) | 4 x i32 | 128-bit | `Wasm128Token` |
| [`i64x2<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.i64x2.html) | 2 x i64 | 128-bit | `Wasm128Token` |
| [`u8x16<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.u8x16.html) | 16 x u8 | 128-bit | `Wasm128Token` |
| [`u16x8<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.u16x8.html) | 8 x u16 | 128-bit | `Wasm128Token` |
| [`u32x4<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.u32x4.html) | 4 x u32 | 128-bit | `Wasm128Token` |
| [`u64x2<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.u64x2.html) | 2 x u64 | 128-bit | `Wasm128Token` |

Wider types are available as polyfills, same as ARM.

## Using these types

Start with [the complete generated gain kernel](@/archmage/getting-started/first-simd.md),
then [generic input types and const modes](@/magetypes/dispatch/types-and-dispatch.md).
The macro supplies each tier's feature context. A bare generic helper called
from baseline code does not get that context merely from its token argument.

The vector values are `Copy`, `Clone`, `Debug`, `Send`, and `Sync`. Constructors
require tokens. `define(...)` is optional shorthand for explicit generic types.
See [slice casting](@/magetypes/conversions/slice-casting.md) for why unrestricted
`Pod`/`Zeroable` construction is not exposed.

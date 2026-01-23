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
- Cross-platform (x86-64 with AVX2/AVX-512, AArch64 with NEON)

## Quick Start

```rust
use archmage::{Avx2FmaToken, SimdToken};
use magetypes::f32x8;

fn main() {
    // Token proves CPU supports AVX2+FMA
    if let Some(token) = Avx2FmaToken::summon() {
        let a = f32x8::splat(token, 1.0);
        let b = f32x8::splat(token, 2.0);
        let c = a + b;  // Natural operators!

        println!("Result: {:?}", c.to_array());
    }
}
```

## Available Types

### x86-64 (SSE4.1 - 128-bit)
`f32x4`, `f64x2`, `i8x16`, `i16x8`, `i32x4`, `i64x2`, `u8x16`, `u16x8`, `u32x4`, `u64x2`

### x86-64 (AVX2 - 256-bit)
`f32x8`, `f64x4`, `i8x32`, `i16x16`, `i32x8`, `i64x4`, `u8x32`, `u16x16`, `u32x8`, `u64x4`

### x86-64 (AVX-512 - 512-bit, requires `avx512` feature)
`f32x16`, `f64x8`, `i8x64`, `i16x32`, `i32x16`, `i64x8`, `u8x64`, `u16x32`, `u32x16`, `u64x8`

### AArch64 (NEON - 128-bit)
`f32x4`, `f64x2`, `i8x16`, `i16x8`, `i32x4`, `i64x2`, `u8x16`, `u16x8`, `u32x4`, `u64x2`

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

## Safe Bytemuck Replacements

Instead of using `bytemuck` (which bypasses token safety), use the token-gated methods:

```rust
// Instead of: bytemuck::cast_slice(&floats)
let vectors = f32x8::cast_slice(token, &floats);

// Instead of: bytemuck::zeroed()
let v = f32x8::zero(token);

// View as bytes (no token needed, type already exists)
let bytes = v.as_bytes();
```

## Features

- **`std`** (default): Enable std library support
- **`bytemuck`**: Enable bytemuck Pod/Zeroable traits (bypasses token safety)
- **`avx512`**: Enable 512-bit types for AVX-512

## Relationship to archmage

`magetypes` depends on `archmage` for:
- Token types (`Avx2FmaToken`, `NeonToken`, etc.)
- The `#[arcane]` macro for writing SIMD kernels
- Runtime CPU feature detection

Use `archmage` directly when you need raw intrinsics. Use `magetypes` when you want ergonomic SIMD types with operators.

## License

MIT OR Apache-2.0

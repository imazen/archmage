# magetypes Type Overview

magetypes provides SIMD vector types with natural Rust operators. Each type wraps platform intrinsics and requires an archmage token for construction.

## Available Types

### x86-64 Types

| Type | Elements | Width | Min Token |
|------|----------|-------|-----------|
| `f32x4` | 4 × f32 | 128-bit | `X64V2Token` |
| `f32x8` | 8 × f32 | 256-bit | `X64V3Token` |
| `f32x16` | 16 × f32 | 512-bit | `X64V4Token`* |
| `f64x2` | 2 × f64 | 128-bit | `X64V2Token` |
| `f64x4` | 4 × f64 | 256-bit | `X64V3Token` |
| `f64x8` | 8 × f64 | 512-bit | `X64V4Token`* |
| `i32x4` | 4 × i32 | 128-bit | `X64V2Token` |
| `i32x8` | 8 × i32 | 256-bit | `X64V3Token` |
| `i32x16` | 16 × i32 | 512-bit | `X64V4Token`* |
| `i8x16` | 16 × i8 | 128-bit | `X64V2Token` |
| `i8x32` | 32 × i8 | 256-bit | `X64V3Token` |
| ... | ... | ... | ... |

*Requires `avx512` feature

### AArch64 Types (NEON)

| Type | Elements | Width | Token |
|------|----------|-------|-------|
| `f32x4` | 4 × f32 | 128-bit | `NeonToken` |
| `f64x2` | 2 × f64 | 128-bit | `NeonToken` |
| `i32x4` | 4 × i32 | 128-bit | `NeonToken` |
| `i16x8` | 8 × i16 | 128-bit | `NeonToken` |
| `i8x16` | 16 × i8 | 128-bit | `NeonToken` |
| `u32x4` | 4 × u32 | 128-bit | `NeonToken` |
| ... | ... | ... | ... |

### WASM Types (SIMD128)

| Type | Elements | Width | Token |
|------|----------|-------|-------|
| `f32x4` | 4 × f32 | 128-bit | `Wasm128Token` |
| `f64x2` | 2 × f64 | 128-bit | `Wasm128Token` |
| `i32x4` | 4 × i32 | 128-bit | `Wasm128Token` |
| ... | ... | ... | ... |

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

- **Copy** — Pass by value freely
- **Clone** — Explicit cloning works
- **Debug** — Print for debugging
- **Send + Sync** — Thread-safe
- **Token-gated construction** — Cannot create without proving CPU support

```rust
// Zero-cost copies
let a = f32x8::splat(token, 1.0);
let b = a;  // Copy, not move
let c = a + b;  // Both still valid
```

**Why no `Pod`/`Zeroable`?** Implementing bytemuck traits would let users bypass token-gated construction (e.g., `bytemuck::zeroed::<f32x8>()`), creating vectors without proving CPU support. Use the token-gated `cast_slice` and `from_bytes` methods instead.

## Using the Prelude

For convenience, import everything:

```rust
use magetypes::prelude::*;

// Now you have all types and archmage re-exports
if let Some(token) = Desktop64::summon() {
    let v = f32x8::splat(token, 1.0);
}
```

## Platform-Specific Imports

If you need just one platform:

```rust
// These platform modules only exist on their target architecture,
// so #[cfg] guards are required here. Prefer `use magetypes::simd::f32x8`
// (the top-level re-exports) for cross-platform code.
#[cfg(target_arch = "x86_64")]
use magetypes::simd::x86::*;

#[cfg(target_arch = "aarch64")]
use magetypes::simd::arm::*;

#[cfg(target_arch = "wasm32")]
use magetypes::simd::wasm::*;
```

## Feature Flags

| Feature | Effect |
|---------|--------|
| `avx512` | Enable 512-bit types on x86-64 |
| `std` | Standard library support (default) |

+++
title = "Platform Notes"
weight = 2
+++

Magetypes uses a generic strategy-pattern design: `f32x8<T>` where `T` is the backend token. The same type name works on all platforms — the token determines which intrinsics get used.

## The Generic Pattern

Import from `magetypes::simd::generic` and `magetypes::simd::backends`:

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Backend,
};

#[inline(always)]
fn process<T: F32x8Backend>(token: T, data: &[f32; 8]) -> f32 {
    let v = f32x8::<T>::load(token, data);
    v.reduce_add()
}
```

The function above compiles correctly on x86-64 (backed by `__m256`), AArch64 (polyfilled with two `float32x4_t`), and WASM (polyfilled with two `v128`). The token type determines the backend.

## Platform-Specific Imports

The platform submodules only exist on their target architecture, so `#[cfg]` guards are required:

```rust
#[cfg(target_arch = "x86_64")]
use magetypes::simd::x86::*;

#[cfg(target_arch = "aarch64")]
use magetypes::simd::arm::*;

#[cfg(target_arch = "wasm32")]
use magetypes::simd::wasm::*;
```

You rarely need these. The generic types from `magetypes::simd::generic` work on all platforms. Use platform-specific imports only when you need a type that doesn't exist in the generic namespace.

## Feature Flags

| Feature | Effect |
|---------|--------|
| `avx512` | Enables 512-bit types on x86-64 (`f32x16`, `i32x16`, `f64x8`, etc.) |
| `std` | Standard library support (on by default) |

Without `avx512`, 512-bit types are not available at all. 128-bit and 256-bit types are always available on x86-64.

On ARM and WASM, only 128-bit native types exist. Wider types are polyfilled regardless of feature flags.

## Token Requirements by Width

Each width tier requires a minimum token:

| Width | x86-64 Token | ARM Token | WASM Token |
|-------|-------------|-----------|------------|
| 128-bit | `X64V2Token` | `NeonToken` | `Wasm128Token` |
| 256-bit | `X64V3Token` | `NeonToken` (polyfill) | `Wasm128Token` (polyfill) |
| 512-bit | `X64V4Token` | `NeonToken` (polyfill) | `Wasm128Token` (polyfill) |

Higher tokens also work — `X64V3Token` accepts any function expecting `X64V2Token` because V3 is a superset of V2.

## Backend Type Aliases

The `backends` module provides lowercase type aliases for use in generic bounds and explicit turbofish:

```rust
use magetypes::simd::backends::{x64v3, neon, wasm128};

// Concrete instantiations for specific platforms
type MyF32x8 = magetypes::simd::generic::f32x8<x64v3>;
```

The aliases: `x64v1`, `x64v2`, `x64v3`, `x64v4`, `x86_v4x`, `neon`, `wasm128`, `scalar`.

## Compile-Time Optimization

When building for a known CPU, detection compiles away:

```bash
# On x86-64:
RUSTFLAGS="-Ctarget-cpu=native" cargo build --release
# summon() becomes a no-op when the target guarantees the features

# On WASM:
RUSTFLAGS="-Ctarget-feature=+simd128" cargo build --target wasm32-unknown-unknown
```

With `-Ctarget-cpu=haswell`, `X64V3Token::summon()` returns `Some(true)` at compile time. The runtime check is elided entirely.

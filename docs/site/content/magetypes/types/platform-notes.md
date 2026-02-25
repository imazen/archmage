+++
title = "Platform Notes"
weight = 2
+++

For cross-platform code, import types from the top-level `magetypes::simd` module. The platform-specific submodules exist for when you need direct access.

## Cross-Platform Imports (Recommended)

```rust
use magetypes::simd::f32x8;
use magetypes::simd::i32x4;
```

These re-exports resolve to the correct platform implementation at compile time. Use them for all portable code.

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

You rarely need these. The top-level `magetypes::simd::f32x8` works on all platforms (via polyfills on ARM and WASM). Use platform-specific imports only when you need a type that doesn't exist in the cross-platform namespace.

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

Higher tokens also work — `Desktop64` (alias for `X64V3Token`) accepts any function expecting `X64V2Token` because V3 is a superset of V2.

## Compile-Time Optimization

When building for a known CPU, detection compiles away:

```bash
# On x86-64:
RUSTFLAGS="-Ctarget-cpu=native" cargo build --release
# summon() becomes a no-op when the target guarantees the features

# On WASM:
RUSTFLAGS="-Ctarget-feature=+simd128" cargo build --target wasm32-unknown-unknown
```

With `-Ctarget-cpu=haswell`, `Desktop64::summon()` returns `Some(true)` at compile time. The runtime check is elided entirely.

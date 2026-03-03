+++
title = "WASM SIMD"
weight = 4
+++

WebAssembly SIMD128 provides 128-bit vectors in the browser and WASI environments.

## Setup

Enable SIMD128 in your build:

```bash
RUSTFLAGS="-Ctarget-feature=+simd128" cargo build --target wasm32-unknown-unknown
```

Or in `.cargo/config.toml`:

```toml
[target.wasm32-unknown-unknown]
rustflags = ["-Ctarget-feature=+simd128"]
```

## No Dynamic Dispatch on WASM

WASM is fundamentally different from x86/ARM: there is **no runtime feature detection**. Whether SIMD128 is available is decided entirely at compile time via `-Ctarget-feature=+simd128`.

This means:
- `Wasm128Token::summon()` always returns `Some` if the binary was compiled with SIMD128, and always `None` if it wasn't. The check compiles away entirely.
- There is no need for `incant!` dispatch on WASM — you either have SIMD128 or you don't, and that's known at compile time.
- `#[arcane]` still works and generates the correct `#[target_feature]` annotation, but since wasm32 target features are safe (the validation model traps deterministically), it skips the `unsafe` wrapper entirely and emits the function directly. No sibling function, no `unsafe` block -- just `#[target_feature]` + `#[inline]` on your original function.

The token still serves its purpose: it proves at the type level that SIMD128 is available, which makes intrinsics safe inside `#[arcane]`. But the dispatch story is purely compile-time.

For cross-platform code that also targets x86/ARM, use `incant!` as normal — on WASM, the macro just emits a direct call to the `_wasm128` variant (or `_scalar` if SIMD128 wasn't enabled).

## The Token

```rust
use archmage::{Wasm128Token, SimdToken};

// On WASM: always Some (if compiled with simd128) or always None
// No runtime check — this compiles away
if let Some(token) = Wasm128Token::summon() {
    process_simd(token, &data);
} else {
    process_scalar(&data);
}
```

## Basic Usage with Raw Intrinsics

```rust
use archmage::{Wasm128Token, arcane};
use std::arch::wasm32::*;

#[arcane]
fn dot_product(_token: Wasm128Token, a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let va = v128_load(a.as_ptr() as *const v128);
    let vb = v128_load(b.as_ptr() as *const v128);
    let mul = f32x4_mul(va, vb);
    // Horizontal sum via shuffle+add
    let shuf = i32x4_shuffle::<1, 0, 3, 2>(mul, mul);
    let sum = f32x4_add(mul, shuf);
    let shuf2 = i32x4_shuffle::<2, 3, 0, 1>(sum, sum);
    let final_sum = f32x4_add(sum, shuf2);
    f32x4_extract_lane::<0>(final_sum)
}
```

> **With magetypes**, the same function becomes:
> ```rust
> use magetypes::simd::{generic::f32x4, backends::wasm128};
>
> #[arcane]
> fn dot_product(token: Wasm128Token, a: &[f32; 4], b: &[f32; 4]) -> f32 {
>     let va = f32x4::<wasm128>::from_array(token, *a);
>     let vb = f32x4::<wasm128>::from_array(token, *b);
>     (va * vb).reduce_add()
> }
> ```

## Browser Compatibility

WASM SIMD is supported in:
- Chrome 91+ (May 2021)
- Firefox 89+ (June 2021)
- Safari 16.4+ (March 2023)
- Node.js 16.4+

For older browsers, provide a non-SIMD fallback WASM binary.

## Relaxed SIMD

WASM relaxed-simd is standardized (Wasm 3.0), stable in Rust since 1.82, and supported by Chrome, Firefox 145+, and Wasmtime. It provides 28 intrinsics (FMA, relaxed lane-select, relaxed min/max, dot products, relaxed truncation) that trade strict cross-platform determinism for performance.

Archmage provides `Wasm128RelaxedToken` for these:

```rust
use archmage::{Wasm128RelaxedToken, SimdToken};

if let Some(token) = Wasm128RelaxedToken::summon() {
    // Relaxed SIMD intrinsics available
}
```

Enable with:

```bash
RUSTFLAGS="-Ctarget-feature=+simd128,+relaxed-simd" cargo build
```

## Cross-Platform Code

For code that runs on x86, ARM, and WASM, use `incant!` with explicit tiers. On WASM, the dispatch compiles down to a direct call:

```rust
use archmage::{arcane, incant};

#[arcane]
fn sum_v3(_token: X64V3Token, data: &[f32; 8]) -> f32 {
    // AVX2: raw intrinsics or magetypes
    let v = _mm256_loadu_ps(data);
    // ... horizontal sum ...
}

#[arcane]
fn sum_neon(_token: NeonToken, data: &[f32; 8]) -> f32 {
    // Process as two halves on 128-bit NEON
}

#[arcane]
fn sum_wasm128(_token: Wasm128Token, data: &[f32; 8]) -> f32 {
    // Process as two halves on 128-bit WASM SIMD
}

fn sum_scalar(_token: ScalarToken, data: &[f32; 8]) -> f32 {
    data.iter().sum()
}

pub fn sum(data: &[f32; 8]) -> f32 {
    incant!(sum(data), [v3, neon, wasm128])
}
```

## Testing WASM Code

Use `wasm-pack test`:

```bash
wasm-pack test --node
```

Or test natively with the scalar fallback:

```rust
#[test]
fn test_sum() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let result = sum(&data);
    assert_eq!(result, 36.0);
}
```

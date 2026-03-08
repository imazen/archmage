# Prelude

`use archmage::prelude::*` gives you everything needed for SIMD work without hunting for individual imports.

## What's Included

### Tokens

`X64V3Token`, `Arm64`, `ScalarToken`, `X64V1Token`, `Sse2Token`, `X64V2Token`, `X64CryptoToken`, `X64V3CryptoToken`, `NeonToken`, `NeonAesToken`, `NeonSha3Token`, `NeonCrcToken`, `Arm64V2Token`, `Arm64V3Token`, `Wasm128Token`, `Wasm128RelaxedToken`

With `avx512` feature: `X64V4Token`, `Server64`, `Avx512Token`, `X64V4xToken`, `Avx512Fp16Token`

### Traits

`SimdToken`, `IntoConcreteToken`, `HasX64V2`, `HasNeon`, `HasNeonAes`, `HasNeonSha3`, `HasArm64V2`, `HasArm64V3`

With `avx512` feature: `HasX64V4`

### Macros

`#[arcane]`, `#[rite]` (token-based, tier-based, and multi-tier), `#[magetypes]`, `incant!`

### Platform Intrinsics

`core::arch::x86_64::*` on x86-64, `core::arch::aarch64::*` on AArch64, `core::arch::wasm32::*` on WASM — the standard Rust intrinsics for your platform.

### Memory Operations

Safe memory operations that take references instead of raw pointers -- `_mm256_loadu_ps` takes `&[f32; 8]`, not `*const f32`. These are also available inside `#[arcane(import_intrinsics)]` and `#[rite(import_intrinsics)]` functions.

## Usage

```rust
use archmage::prelude::*;

#[arcane(import_intrinsics)]
fn add(_token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    let va = _mm256_loadu_ps(a);
    let vb = _mm256_loadu_ps(b);
    let sum = _mm256_add_ps(va, vb);
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, sum);
    out
}
```

No individual `use` statements needed — `X64V3Token`, `_mm256_loadu_ps`, `_mm256_add_ps`, and `_mm256_storeu_ps` all come from the prelude.

## magetypes Prelude

`use magetypes::prelude::*` re-exports the archmage prelude plus all SIMD types:

```rust
use magetypes::prelude::*;

#[arcane(import_intrinsics)]
fn dot(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let va = f32x8::from_array(token, *a);
    let vb = f32x8::from_array(token, *b);
    va.mul_add(vb, f32x8::zero(token)).reduce_add()
}
```

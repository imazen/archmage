# Prelude

`use archmage::prelude::*` gives you everything needed for SIMD work without hunting for individual imports.

## What's Included

### Tokens

`Desktop64`, `Arm64`, `ScalarToken`, `X64V2Token`, `X64V3Token`, `NeonToken`, `Wasm128Token`

With `avx512` feature: `X64V4Token`, `Server64`, `Avx512Token`, `X64V4xToken`, `Avx512Fp16Token`

### Traits

`SimdToken`, `IntoConcreteToken`, `HasX64V2`, `HasNeon`, `HasNeonAes`, `HasNeonSha3`

With `avx512` feature: `HasX64V4`

### Macros

`#[arcane]`, `#[rite]`, `#[magetypes]`, `incant!`

### Platform Intrinsics

`core::arch::x86_64::*` on x86-64, `core::arch::aarch64::*` on AArch64, `core::arch::wasm32::*` on WASM — the standard Rust intrinsics for your platform.

### Memory Operations

`safe_unaligned_simd` functions (with the `safe_unaligned_simd` feature, enabled by default). These take references instead of raw pointers — `_mm256_loadu_ps` takes `&[f32; 8]`, not `*const f32`.

## Usage

```rust
use archmage::prelude::*;

#[arcane]
fn add(_token: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    let va = _mm256_loadu_ps(a);
    let vb = _mm256_loadu_ps(b);
    let sum = _mm256_add_ps(va, vb);
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, sum);
    out
}
```

No individual `use` statements needed — `Desktop64`, `_mm256_loadu_ps`, `_mm256_add_ps`, and `_mm256_storeu_ps` all come from the prelude.

## magetypes Prelude

`use magetypes::prelude::*` re-exports the archmage prelude plus all SIMD types:

```rust
use magetypes::prelude::*;

#[arcane]
fn dot(token: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let va = f32x8::from_array(token, *a);
    let vb = f32x8::from_array(token, *b);
    va.mul_add(vb, f32x8::zero(token)).reduce_add()
}
```

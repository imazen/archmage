# archmage

[![Crates.io](https://img.shields.io/crates/v/archmage.svg)](https://crates.io/crates/archmage)
[![Documentation](https://docs.rs/archmage/badge.svg)](https://docs.rs/archmage)
[![CI](https://github.com/imazen/archmage/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/archmage/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/imazen/archmage/branch/main/graph/badge.svg)](https://codecov.io/gh/imazen/archmage)
[![License](https://img.shields.io/crates/l/archmage.svg)](https://github.com/imazen/archmage#license)

> Safely invoke your intrinsic power, using the tokens granted to you by the CPU. Cast primitive magics faster than any mage alive.

**archmage** provides capability tokens that prove CPU feature availability at runtime, making raw SIMD intrinsics safe to call via the `#[arcane]` macro.

## Quick Start

```rust
use archmage::{Desktop64, HasAvx2, SimdToken, arcane};
use archmage::mem::avx;  // safe load/store (enabled by default)
use std::arch::x86_64::*;

#[arcane]
fn multiply_add(token: impl HasAvx2, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    // Safe memory operations - references, not raw pointers!
    let va = avx::_mm256_loadu_ps(token, a);
    let vb = avx::_mm256_loadu_ps(token, b);

    // Value-based intrinsics are SAFE inside #[arcane]!
    let result = _mm256_add_ps(va, vb);
    let result = _mm256_mul_ps(result, result);

    let mut out = [0.0f32; 8];
    avx::_mm256_storeu_ps(token, &mut out, result);
    out
}

fn main() {
    // Desktop64 is the recommended starting point:
    // - AVX2 + FMA + BMI2
    // - Works on Intel Haswell (2013+) and AMD Zen 1 (2017+)
    // - Covers ~95% of desktop/server CPUs in use today
    if let Some(token) = Desktop64::summon() {
        let result = multiply_add(token, &[1.0; 8], &[2.0; 8]);
        println!("{:?}", result);
    }
}
```

## How It Works

### The Problem

Raw SIMD intrinsics have two safety concerns:

1. **Feature availability**: Calling `_mm256_add_ps` on a CPU without AVX is undefined behavior
2. **Memory safety**: `_mm256_loadu_ps(ptr)` dereferences a raw pointer

Rust 1.85+ made value-based intrinsics safe inside `#[target_feature]` functions, but calling those functions is still `unsafe` because the compiler can't verify the CPU supports the features.

### The Solution: Tokens + `#[arcane]`

archmage solves this with two components:

**1. Capability Tokens** - Zero-sized proof types created after runtime CPU detection:

```rust
use archmage::{Desktop64, SimdToken};

// summon() checks CPUID and returns Some only if features are available
// (check is elided if compiled with -C target-cpu=native or similar)
if let Some(token) = Desktop64::summon() {
    // Token exists = CPU definitely has AVX2 + FMA + BMI2
}
```

**2. The `#[arcane]` Macro** - Transforms your function to enable `#[target_feature]`:

```rust
#[arcane]
fn my_kernel(token: impl HasAvx2, data: &[f32; 8]) -> [f32; 8] {
    // Intrinsics are safe here!
    let v = _mm256_setzero_ps();
    // ...
}
```

The macro generates:

```rust
fn my_kernel(token: impl HasAvx2, data: &[f32; 8]) -> [f32; 8] {
    #[target_feature(enable = "avx2")]
    unsafe fn inner(data: &[f32; 8]) -> [f32; 8] {
        let v = _mm256_setzero_ps();  // Safe inside #[target_feature]!
        // ...
    }
    // SAFETY: The token parameter proves the caller verified CPU support
    unsafe { inner(data) }
}
```

**Why is this safe?**

1. `inner()` has `#[target_feature(enable = "avx2")]`, so Rust allows intrinsics without `unsafe`
2. Calling `inner()` is unsafe, but we know it's valid because:
   - The function requires a token parameter
   - Tokens can only be created via `summon()` which checks CPU features
   - Therefore, if you have a token, the CPU supports the features

### Generic Token Bounds

Functions accept any token that provides the required capabilities:

```rust
use archmage::{HasAvx2, HasFma, arcane};
use archmage::mem::avx;
use std::arch::x86_64::*;

// Accept any token with AVX2 (Avx2Token, Desktop64, Server64, etc.)
#[arcane]
fn double(token: impl HasAvx2, data: &[f32; 8]) -> [f32; 8] {
    let v = avx::_mm256_loadu_ps(token, data);
    let doubled = _mm256_add_ps(v, v);
    let mut out = [0.0f32; 8];
    avx::_mm256_storeu_ps(token, &mut out, doubled);
    out
}

// Require multiple features with inline bounds
#[arcane]
fn fma_kernel<T: HasAvx2 + HasFma>(token: T, a: &[f32; 8], b: &[f32; 8], c: &[f32; 8]) -> [f32; 8] {
    let va = avx::_mm256_loadu_ps(token, a);
    let vb = avx::_mm256_loadu_ps(token, b);
    let vc = avx::_mm256_loadu_ps(token, c);
    let result = _mm256_fmadd_ps(va, vb, vc);  // a * b + c
    let mut out = [0.0f32; 8];
    avx::_mm256_storeu_ps(token, &mut out, result);
    out
}

// Where clause syntax
#[arcane]
fn square<T>(token: T, data: &mut [f32; 8])
where
    T: HasAvx2
{
    let v = avx::_mm256_loadu_ps(token, data);
    let squared = _mm256_mul_ps(v, v);
    avx::_mm256_storeu_ps(token, data, squared);
}
```

The trait hierarchy means broader tokens satisfy narrower bounds:
- `Desktop64` implements `HasAvx2`, `HasFma`, `HasSse42`, etc.
- `Server64` implements everything `Desktop64` does, plus `HasAvx512f`, etc.

## Choosing a Token

**Start with `Desktop64`** - it's the sweet spot for modern x86-64:

| Token | Features | Hardware Coverage |
|-------|----------|-------------------|
| `Desktop64` | AVX2 + FMA + BMI2 | Intel Haswell 2013+, AMD Zen 1 2017+ (~95% of x86-64) |
| `Server64` | + AVX-512 | Intel Skylake-X 2017+, AMD Zen 4 2022+ |
| `X64V2Token` | SSE4.2 + POPCNT | Intel Nehalem 2008+, AMD Bulldozer 2011+ |

**For specific features:**

| Token | Use Case |
|-------|----------|
| `Avx2Token` | Need AVX2 but not FMA |
| `Avx2FmaToken` | AVX2 + FMA (most floating-point SIMD) |
| `FmaToken` | FMA only |
| `Sse2Token` | Baseline x86-64 (always available) |

**ARM tokens:**

| Token | Features | Hardware |
|-------|----------|----------|
| `NeonToken` | NEON | All AArch64 (baseline, including Apple M-series) |
| `SveToken` | SVE | Graviton 3, A64FX |
| `Sve2Token` | SVE2 | ARMv9: Graviton 4, Cortex-X2+ |

## Cross-Architecture Tokens

All token types are available on all architectures. This makes cross-platform code easier to write without `#[cfg]` guards everywhere:

```rust
use archmage::{Desktop64, NeonToken, SimdToken};

// This compiles on ARM, x86, WASM - no #[cfg] needed!
fn process_data(data: &mut [f32]) {
    if let Some(token) = Desktop64::summon() {
        // AVX2 path (only succeeds on x86 with AVX2)
        process_x86(token, data);
    } else if let Some(token) = NeonToken::summon() {
        // NEON path (only succeeds on AArch64)
        process_arm(token, data);
    } else {
        // Scalar fallback
        process_scalar(data);
    }
}
```

- `summon()` returns `None` on unsupported architectures
- Rust's type system ensures intrinsic methods don't exist on the wrong arch
- You get compile errors if you try to use x86 intrinsics in ARM code

## Safe Memory Operations (`mem` module)

The `mem` module (enabled by default) provides safe load/store using references instead of raw pointers:

```rust
use archmage::{Desktop64, SimdToken};
use archmage::mem::avx;

if let Some(token) = Desktop64::summon() {
    let data = [1.0f32; 8];
    let v = avx::_mm256_loadu_ps(token, &data);  // Safe! Reference, not pointer

    let mut out = [0.0f32; 8];
    avx::_mm256_storeu_ps(token, &mut out, v);   // Safe!
}
```

**Available submodules:**

| Module | Functions | Token Required |
|--------|-----------|----------------|
| `mem::sse` | `_mm_loadu_ps`, `_mm_storeu_ps`, etc. | `impl HasSse` |
| `mem::sse2` | `_mm_loadu_pd`, `_mm_loadu_si128`, etc. | `impl HasSse2` |
| `mem::avx` | `_mm256_loadu_ps`, `_mm256_storeu_ps`, etc. | `impl HasAvx` |
| `mem::avx2` | `_mm256_loadu_si256`, etc. | `impl HasAvx2` |
| `mem::avx512f` | `_mm512_loadu_ps`, etc. | `impl HasAvx512f` |
| `mem::neon` | `vld1q_f32`, `vst1q_f32`, etc. | `impl HasNeon` |

The wrappers accept any compatible token (e.g., `Desktop64` works with `mem::avx` because it implements `HasAvx`).

## When to Use archmage

archmage is for when you need **specific instructions** that autovectorization won't produce:

- Complex shuffles and permutes
- Exact FMA sequences for numerical precision
- DCT butterflies and signal processing
- Gather/scatter operations
- Bit manipulation (BMI1/BMI2)

For portable SIMD without manual intrinsics, use the `wide` crate instead.

| Approach | When to Use |
|----------|-------------|
| **wide** | Portable code, let the compiler choose instructions |
| **archmage** | Need specific instructions, complex algorithms |

## Feature Flags

```toml
[dependencies]
archmage = "0.1"
```

| Feature | Description |
|---------|-------------|
| `std` (default) | Enable std library support |
| `macros` (default) | Enable `#[arcane]` macro (alias: `#[simd_fn]`) |
| `safe_unaligned_simd` (default) | Safe load/store via references (exposed as `mem` module) |
**Unstable features** (API may change):

| Feature | Description |
|---------|-------------|
| `__composite` | Higher-level ops (transpose, dot product) |
| `__wide` | Integration with the `wide` crate |

### Testing Scalar Fallbacks

Set the `ARCHMAGE_DISABLE` environment variable to force scalar code paths:

```bash
ARCHMAGE_DISABLE=1 cargo test
ARCHMAGE_DISABLE=1 cargo run --release
```

```rust
// With ARCHMAGE_DISABLE set, this always takes the fallback path
if let Some(token) = Desktop64::summon() {
    simd_path(token, &mut data);
} else {
    scalar_fallback(&mut data);  // Always runs with ARCHMAGE_DISABLE
}
```

## Methods with Self Receivers

Methods with `self`, `&self`, `&mut self` receivers are supported via the `_self = Type` argument.
Use `_self` in the function body instead of `self`:

```rust
use archmage::{HasAvx2, arcane};

trait SimdOps {
    fn double(&self, token: impl HasAvx2) -> Self;
    fn scale(&mut self, token: impl HasAvx2, factor: f32);
}

impl SimdOps for [f32; 8] {
    #[arcane(_self = [f32; 8])]
    fn double(&self, _token: impl HasAvx2) -> Self {
        // Use _self instead of self in the body
        let v = unsafe { _mm256_loadu_ps(_self.as_ptr()) };
        let doubled = _mm256_add_ps(v, v);
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
        out
    }

    #[arcane(_self = [f32; 8])]
    fn scale(&mut self, _token: impl HasAvx2, factor: f32) {
        let v = unsafe { _mm256_loadu_ps(_self.as_ptr()) };
        let scale = _mm256_set1_ps(factor);
        let scaled = _mm256_mul_ps(v, scale);
        unsafe { _mm256_storeu_ps(_self.as_mut_ptr(), scaled) };
    }
}
```

**Why `_self`?** The macro generates an inner function where `self` becomes a regular
parameter named `_self`. Using `_self` in your code reminds you that you're not using
the normal `self` keyword.

All receiver types are supported: `self` (move), `&self` (ref), `&mut self` (mut ref)

## License

MIT OR Apache-2.0

## AI-Generated Code Notice

Developed with Claude (Anthropic). Not all code manually reviewed. Review critical paths before production use.

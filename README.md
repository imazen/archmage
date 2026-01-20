# archmage

> Safely invoke your intrinsic power, using the tokens granted to you by the CPU. Cast primitive magics faster than any mage alive.

**archmage** provides capability tokens that prove CPU feature availability at runtime, making raw SIMD intrinsics safe to call via the `#[arcane]` macro.

## Quick Start

```rust
use archmage::{X64V3Token, HasAvx2, SimdToken, arcane};
use archmage::mem::avx;  // Safe load/store (requires safe_unaligned_simd feature)
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
    // X64V3Token is the recommended starting point:
    // - AVX2 + FMA + BMI2
    // - Works on Intel Haswell (2013+) and AMD Zen 1 (2017+)
    // - Covers ~95% of desktop/server CPUs in use today
    if let Some(token) = X64V3Token::try_new() {
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

**1. Capability Tokens** - Zero-sized proof types created only after runtime CPU detection:

```rust
// try_new() checks CPUID and returns Some only if features are available
if let Some(token) = X64V3Token::try_new() {
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
   - Tokens can only be created via `try_new()` which checks CPU features
   - Therefore, if you have a token, the CPU supports the features

### Generic Token Bounds

Functions accept any token that provides the required capabilities:

```rust
use archmage::{HasAvx2, HasFma, arcane};

// Accept any token with AVX2 (Avx2Token, X64V3Token, X64V4Token, etc.)
#[arcane]
fn process(token: impl HasAvx2, data: &[f32; 8]) -> [f32; 8] {
    // Works with any AVX2-capable token
}

// Require multiple features
#[arcane]
fn fma_kernel<T: HasAvx2 + HasFma>(token: T, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    // Has access to both AVX2 and FMA intrinsics
}

// Where clause syntax also works
#[arcane]
fn complex_kernel<T>(token: T, data: &mut [f32])
where
    T: HasAvx2 + HasFma
{
    // ...
}
```

The trait hierarchy means broader tokens satisfy narrower bounds:
- `X64V3Token` implements `HasAvx2`, `HasFma`, `HasSse42`, etc.
- `X64V4Token` implements everything `X64V3Token` does, plus `HasAvx512f`, etc.

## Choosing a Token

**Start with `X64V3Token`** - it's the sweet spot for modern x86-64:

| Token | Features | Hardware Coverage |
|-------|----------|-------------------|
| `X64V3Token` | AVX2 + FMA + BMI2 | Intel Haswell 2013+, AMD Zen 1 2017+ (~95% of x86-64) |
| `X64V4Token` | + AVX-512 | Intel Skylake-X 2017+, AMD Zen 4 2022+ |
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
| `NeonToken` | NEON | All AArch64 (baseline) |
| `SveToken` | SVE | Graviton 3, Apple M-series |
| `Sve2Token` | SVE2 | ARMv9: Graviton 4, Cortex-X2+ |

## Safe Memory Operations

With the `safe_unaligned_simd` feature, load/store uses references instead of raw pointers:

```rust
use archmage::mem::avx;

if let Some(token) = X64V3Token::try_new() {
    let data = [1.0f32; 8];
    let v = avx::_mm256_loadu_ps(token, &data);  // Safe! Reference, not pointer

    let mut out = [0.0f32; 8];
    avx::_mm256_storeu_ps(token, &mut out, v);   // Safe!
}
```

The `mem` module wrappers accept `impl HasAvx`, `impl HasSse2`, etc., so any compatible token works.

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
| `safe_unaligned_simd` | Safe load/store via references (exposed as `mem` module) |

**Unstable features** (API may change):

| Feature | Description |
|---------|-------------|
| `__composite` | Higher-level ops (transpose, dot product) |
| `__wide` | Integration with the `wide` crate |

## Limitations

**Self receivers not supported in `#[arcane]`:**

```rust
// This won't work - inner functions can't have `self`
#[arcane]
fn process(&self, token: impl HasAvx2) { ... }

// Instead, take self as a regular parameter or use free functions
#[arcane]
fn process(state: &MyStruct, token: impl HasAvx2) { ... }
```

## License

MIT OR Apache-2.0

## AI-Generated Code Notice

Developed with Claude (Anthropic). Not all code manually reviewed. Review critical paths before production use.

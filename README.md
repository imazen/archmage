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
use std::arch::x86_64::*;

#[arcane]
fn multiply_add(_token: impl HasAvx2, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    // safe_unaligned_simd calls are SAFE inside #[arcane] - no unsafe needed!
    let va = safe_unaligned_simd::x86_64::_mm256_loadu_ps(a);
    let vb = safe_unaligned_simd::x86_64::_mm256_loadu_ps(b);

    // Value-based intrinsics are also SAFE inside #[arcane]!
    let result = _mm256_add_ps(va, vb);
    let result = _mm256_mul_ps(result, result);

    let mut out = [0.0f32; 8];
    safe_unaligned_simd::x86_64::_mm256_storeu_ps(&mut out, result);
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
use std::arch::x86_64::*;

// Accept any token with AVX2 (Avx2Token, Desktop64, Server64, etc.)
#[arcane]
fn double(_token: impl HasAvx2, data: &[f32; 8]) -> [f32; 8] {
    let v = safe_unaligned_simd::x86_64::_mm256_loadu_ps(data);
    let doubled = _mm256_add_ps(v, v);
    let mut out = [0.0f32; 8];
    safe_unaligned_simd::x86_64::_mm256_storeu_ps(&mut out, doubled);
    out
}

// Require multiple features with inline bounds
#[arcane]
fn fma_kernel<T: HasAvx2 + HasFma>(_token: T, a: &[f32; 8], b: &[f32; 8], c: &[f32; 8]) -> [f32; 8] {
    let va = safe_unaligned_simd::x86_64::_mm256_loadu_ps(a);
    let vb = safe_unaligned_simd::x86_64::_mm256_loadu_ps(b);
    let vc = safe_unaligned_simd::x86_64::_mm256_loadu_ps(c);
    let result = _mm256_fmadd_ps(va, vb, vc);  // a * b + c
    let mut out = [0.0f32; 8];
    safe_unaligned_simd::x86_64::_mm256_storeu_ps(&mut out, result);
    out
}
```

The trait hierarchy means broader tokens satisfy narrower bounds:
- `Desktop64` implements `HasAvx2`, `HasFma`, `HasSse42`, etc.
- `Server64` implements everything `Desktop64` does, plus `HasAvx512f`, etc.

## Zero-Overhead Inlining

Everything inlines completely. No function call overhead, no abstraction cost.

### Rust 1.85+ Contextual Safety

Inside `#[arcane]` functions, the generated inner function has `#[target_feature]`. This enables **contextual safety** for intrinsics:

```rust
#[arcane]
fn example(_token: impl HasAvx2, data: &[f32; 8]) -> __m256 {
    // Value intrinsics are SAFE - no unsafe needed!
    let va = safe_unaligned_simd::x86_64::_mm256_loadu_ps(data);  // SAFE!
    let sum = _mm256_add_ps(va, va);       // SAFE! Value-based intrinsic
    let prod = _mm256_mul_ps(sum, sum);    // SAFE!
    _mm256_sqrt_ps(prod)                   // SAFE!
}
```

**What's safe inside `#[arcane]`:**
- All value-based intrinsics: arithmetic, shuffles, comparisons, bitwise ops
- Calls to `safe_unaligned_simd` functions (they have matching `#[target_feature]`)

**What still needs `unsafe`:**
- Raw pointer loads/stores (`_mm256_loadu_ps(ptr)`) - use `safe_unaligned_simd` instead

### `safe_unaligned_simd` Inlines Completely

Inside `#[arcane]`, `safe_unaligned_simd` calls are **safe** (matching target features) and the optimizer inlines everything:

```rust
#[arcane]
pub fn process(_token: impl Has256BitSimd, data: &mut [[f32; 8]]) {
    for chunk in data.iter_mut() {
        let v = safe_unaligned_simd::x86_64::_mm256_loadu_ps(chunk);  // Safe call!
        let r = _mm256_mul_ps(v, v);
        safe_unaligned_simd::x86_64::_mm256_storeu_ps(chunk, r);
    }
}
```

Compiles to (with `-C opt-level=3`):

```asm
vmovups   (%rdi), %ymm0       # load from chunk[0]
vmulps    %ymm0, %ymm0, %ymm0 # square
vmovups   %ymm0, (%rdi)       # store
vmovups   32(%rdi), %ymm0     # load from chunk[1] (loop unrolled!)
vmulps    %ymm0, %ymm0, %ymm0
vmovups   %ymm0, 32(%rdi)
# ... loop continues unrolled
```

**Zero function calls.** Everything collapses into straight-line SIMD instructions.

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

## Safe Memory Operations with `safe_unaligned_simd`

For safe load/store operations, use the `safe_unaligned_simd` crate directly inside `#[arcane]` functions:

```toml
[dependencies]
archmage = "0.2"
safe_unaligned_simd = "0.2"
```

```rust
use archmage::{Desktop64, SimdToken, arcane, HasAvx2};
use std::arch::x86_64::*;

#[arcane]
fn process(_token: impl HasAvx2, data: &[f32; 8]) -> [f32; 8] {
    // safe_unaligned_simd uses references, not raw pointers
    let v = safe_unaligned_simd::x86_64::_mm256_loadu_ps(data);  // Safe!
    let squared = _mm256_mul_ps(v, v);
    let mut out = [0.0f32; 8];
    safe_unaligned_simd::x86_64::_mm256_storeu_ps(&mut out, squared);  // Safe!
    out
}
```

Inside `#[arcane]`, these calls are **safe** because the `#[target_feature]` attributes match. Outside `#[arcane]`, you would need `unsafe` blocks.

## SIMD Types

archmage provides token-gated SIMD types with `wide`-like ergonomics. Construction requires a capability token, ensuring CPU support at compile time.

### Available Types

| Width | Float | Signed Int | Unsigned Int | Token Required |
|-------|-------|------------|--------------|----------------|
| **128-bit** | `f32x4`, `f64x2` | `i8x16`, `i16x8`, `i32x4`, `i64x2` | `u8x16`, `u16x8`, `u32x4`, `u64x2` | `Sse41Token` |
| **256-bit** | `f32x8`, `f64x4` | `i8x32`, `i16x16`, `i32x8`, `i64x4` | `u8x32`, `u16x16`, `u32x8`, `u64x4` | `Avx2FmaToken` |
| **512-bit** | `f32x16`, `f64x8` | `i8x64`, `i16x32`, `i32x16`, `i64x8` | `u8x64`, `u16x32`, `u32x16`, `u64x8` | `Avx512Token` |

### Operations

**Construction** (requires token):
- `splat(token, value)` - broadcast scalar to all lanes
- `load(token, &array)` - load from aligned array
- `from_array(token, array)` - construct from array
- `zero(token)` - all zeros

**Conversion** (no token needed):
- `store(&mut array)` - store to array
- `to_array() -> [T; N]` - extract to array
- `as_array() -> &[T; N]` - view as array reference
- `raw() -> __m256` - extract raw intrinsic type

**Arithmetic** (operators): `+`, `-`, `*`, `/`, `+=`, `-=`, `*=`, `/=`

**Bitwise** (operators): `&`, `|`, `^`, `&=`, `|=`, `^=`

**Math** (float types):
- `min`, `max`, `clamp` - element-wise bounds
- `sqrt`, `abs` - square root, absolute value
- `floor`, `ceil`, `round` - rounding
- `mul_add(a, b)` - fused multiply-add: `self * a + b`
- `mul_sub(a, b)` - fused multiply-sub: `self * a - b`
- `rcp_approx`, `recip` - reciprocal (fast/accurate)
- `rsqrt_approx`, `rsqrt` - reciprocal sqrt (fast/accurate)

**Transcendentals** (float types):
- `log2_lowp`, `log2_midp` - base-2 logarithm
- `exp2_lowp`, `exp2_midp` - base-2 exponential
- `ln_lowp`, `ln_midp` - natural logarithm
- `exp_lowp`, `exp_midp` - natural exponential
- `pow_lowp(n)`, `pow_midp(n)` - power function
- `cbrt_midp` - cube root

**Comparison** (returns mask):
- `simd_eq`, `simd_ne` - equality
- `simd_lt`, `simd_le`, `simd_gt`, `simd_ge` - ordering

**Blending & Reduction**:
- `blend(mask, if_true, if_false)` - conditional select
- `reduce_add`, `reduce_min`, `reduce_max` - horizontal ops

**Integer-specific**:
- `shl::<N>`, `shr::<N>` - shift by constant
- `shr_arithmetic::<N>` - arithmetic right shift
- `extend_lo_*`, `extend_hi_*` - widen to larger type
- `pack_*` - narrow to smaller type

**Block operations** (f32x4, f32x8):
- `transpose_4x4`, `transpose_8x8` - matrix transpose
- `interleave_4ch`, `deinterleave_4ch` - AoS ↔ SoA for RGBA
- `load_4_rgba_u8`, `store_4_rgba_u8` - packed RGBA u8 conversion

### Example

```rust
use archmage::{Avx2FmaToken, SimdToken, simd::f32x8};

fn main() {
    if let Some(token) = Avx2FmaToken::try_new() {
        let a = f32x8::splat(token, 2.0);
        let b = f32x8::from_array(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let c = a * b + a;  // Operators work naturally
        let result = c.sqrt().pow_midp(0.5);  // Method chaining
        println!("{:?}", result.to_array());
    }
}
```

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
archmage = "0.2"
```

| Feature | Description |
|---------|-------------|
| `std` (default) | Enable std library support |
| `macros` (default) | Enable `#[arcane]` macro (alias: `#[simd_fn]`) |
| `avx512` | AVX-512 token support (`Avx512Token`, `X64V4Token`, etc.) |

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
use std::arch::x86_64::*;

trait SimdOps {
    fn double(&self, token: impl HasAvx2) -> Self;
    fn scale(&mut self, token: impl HasAvx2, factor: f32);
}

impl SimdOps for [f32; 8] {
    #[arcane(_self = [f32; 8])]
    fn double(&self, _token: impl HasAvx2) -> Self {
        // Use _self instead of self in the body
        let v = safe_unaligned_simd::x86_64::_mm256_loadu_ps(_self);
        let doubled = _mm256_add_ps(v, v);
        let mut out = [0.0f32; 8];
        safe_unaligned_simd::x86_64::_mm256_storeu_ps(&mut out, doubled);
        out
    }

    #[arcane(_self = [f32; 8])]
    fn scale(&mut self, _token: impl HasAvx2, factor: f32) {
        let v = safe_unaligned_simd::x86_64::_mm256_loadu_ps(_self);
        let scale = _mm256_set1_ps(factor);
        let scaled = _mm256_mul_ps(v, scale);
        safe_unaligned_simd::x86_64::_mm256_storeu_ps(_self, scaled);
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

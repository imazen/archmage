# archmage

Type-safe SIMD capability tokens for Rust. Isolates `unsafe` to token construction, enabling safe raw intrinsic usage.

## Understanding the SIMD Landscape

### The Approaches

**wide crate**: Portable SIMD types (`f32x8`, `i32x8`, etc.) with compile-time implementation selection via `cfg!`. No runtime dispatch. Without global compile flags, wide uses 128-bit operations (two `f32x4`), but this is still **4-6x faster than scalar** because:
- Parallelism is explicit - no reliance on LLVM autovectorization
- Memory access patterns are predictable and aligned
- You get a SIMD floor even when LLVM's autovectorizer would fail

With global flags (`-C target-feature=+avx2`), wide's `cfg!` selects native 256-bit implementations at compile time.

**Scalar + `#[target_feature]`**: LLVM can autovectorize scalar loops to 256-bit inside `#[target_feature]` functions. Works well for simple patterns, but LLVM may not vectorize complex code.

**archmage + raw intrinsics**: Direct control over exact instructions. Use when you need specific instruction sequences that neither wide nor LLVM autovectorization will produce.

### When to Use What

| Approach | Reliability | Performance | Use When |
|----------|-------------|-------------|----------|
| **wide (default)** | Always SIMD | 4-6x over scalar | Portable code, don't want to think about flags |
| **wide + global flags** | Always 256-bit | 8-12x over scalar | Single target CPU, want maximum wide perf |
| **Scalar + `#[target_feature]`** | LLVM decides | Varies | Simple loops, trust LLVM |
| **pulp** | Runtime dispatch | Good | Want abstraction + runtime dispatch |
| **archmage + raw intrinsics** | Exact control | Maximum | Need specific instructions (shuffles, FMA, DCT) |

### archmage's Niche

archmage is for when you need **specific instructions** that neither wide nor LLVM autovectorization will produce:

- Complex shuffles and permutes
- Exact FMA sequences for numerical precision
- DCT butterflies and other signal processing
- Gather/scatter operations
- Bit manipulation (BMI1/BMI2)

It makes raw `_mm256_*` intrinsics safe via capability tokens, and integrates with multiversion crates.

## Quick Start

```rust
use archmage::{Avx2Token, SimdToken, simd_fn};
use std::arch::x86_64::*;

#[simd_fn]
fn double_avx2(token: Avx2Token, data: &[f32; 8]) -> [f32; 8] {
    // Loads/stores need unsafe (raw pointers)
    let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };

    // Arithmetic intrinsics are safe - token proves AVX2 available!
    let doubled = _mm256_add_ps(v, v);

    let mut out = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
    out
}

fn main() {
    if let Some(token) = Avx2Token::try_new() {
        let result = double_avx2(token, &[1.0; 8]);
        // result = [2.0; 8]
    }
}
```

The `#[simd_fn]` macro expands to:
```rust
fn double_avx2(token: Avx2Token, data: &[f32; 8]) -> [f32; 8] {
    #[target_feature(enable = "avx2")]
    unsafe fn __inner(data: &[f32; 8]) -> [f32; 8] { /* body */ }
    let _ = &token;  // prove we have the token
    unsafe { __inner(data) }  // SAFETY: token proves avx2 available
}
```

## Why Tokens + Raw Intrinsics?

With raw intrinsics inside `#[target_feature]` functions, the compiler **will** generate the correct instructions:

```asm
; Without #[target_feature] - wide uses 128-bit ops
movaps (%rsi),%xmm0        ; Load 128-bit
movaps 0x10(%rsi),%xmm1    ; Load another 128-bit
mulps  (%rdx),%xmm0        ; Multiply first half
mulps  0x10(%rdx),%xmm1    ; Multiply second half

; With #[target_feature(enable = "avx2")] - raw intrinsics use 256-bit
vmovaps (%rsi),%ymm0       ; Load 256-bit
vmulps (%rdx),%ymm0,%ymm0  ; Multiply all 8 floats at once
```

The token proves you're in the right context to call the function safely.

## Token Types

### Feature Tokens

| Token | Features | Runtime Check |
|-------|----------|---------------|
| `Sse2Token` | SSE2 | Always on x86-64 |
| `Sse41Token` | SSE4.1 | `is_x86_feature_detected!("sse4.1")` |
| `AvxToken` | AVX | `is_x86_feature_detected!("avx")` |
| `Avx2Token` | AVX2 | `is_x86_feature_detected!("avx2")` |
| `FmaToken` | FMA | `is_x86_feature_detected!("fma")` |
| `Avx2FmaToken` | AVX2 + FMA | Both checks |

### Profile Tokens (x86-64 microarchitecture levels)

| Token | Features | Hardware |
|-------|----------|----------|
| `X64V2Token` | SSE4.2 + POPCNT | Nehalem 2008+, Bulldozer 2011+ |
| `X64V3Token` | AVX2 + FMA + BMI2 | Haswell 2013+, Zen 1 2017+ |
| `X64V4Token` | AVX-512 (F/BW/CD/DQ/VL) | Xeon 2017+, Zen 4 2022+ |

### ARM Tokens

| Token | Features | Hardware |
|-------|----------|----------|
| `NeonToken` | NEON | All AArch64 (always available) |
| `SveToken` | SVE | Graviton 3, Apple M-series, A64FX |
| `Sve2Token` | SVE2 | ARMv9: Cortex-X2+, Graviton 4 |

## Usage Patterns

### Pattern 1: `#[simd_fn]` (Recommended)

```rust
use archmage::{Avx2FmaToken, simd_fn};
use std::arch::x86_64::*;

#[simd_fn]
fn fma_kernel(token: Avx2FmaToken, a: &[f32; 8], b: &[f32; 8], c: &[f32; 8]) -> [f32; 8] {
    let va = unsafe { _mm256_loadu_ps(a.as_ptr()) };
    let vb = unsafe { _mm256_loadu_ps(b.as_ptr()) };
    let vc = unsafe { _mm256_loadu_ps(c.as_ptr()) };

    let result = _mm256_fmadd_ps(va, vb, vc);  // Safe! Token proves FMA available

    let mut out = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(out.as_mut_ptr(), result) };
    out
}
```

### Pattern 2: Profile Tokens

```rust
#[simd_fn]
fn v3_kernel(token: X64V3Token, data: &mut [f32]) {
    // AVX2 + FMA + BMI1 + BMI2 all enabled automatically!
    // ...
}
```

### Pattern 3: With Multiversion Crates

```rust
use archmage::{x64v3_token, SimdToken};
use multiversed::multiversed;

#[multiversed]
fn process(data: &mut [f32]) {
    let token = x64v3_token!();  // Safe inside multiversed context
    // Use token for operations...
}
```

### Pattern 4: Token-Gated Operations

archmage provides wrapped operations that require tokens:

```rust
use archmage::{Avx2Token, SimdToken, ops};

fn example() {
    if let Some(token) = Avx2Token::try_new() {
        let a = [1.0f32; 8];
        let va = ops::load_f32x8(token, &a);
        let doubled = ops::add_f32x8(token, va, va);
        let mut out = [0.0f32; 8];
        ops::store_f32x8(token, &mut out, doubled);
    }
}
```

## Feature Flags

```toml
[dependencies]
archmage = { version = "0.1", features = ["wide"] }
```

| Feature | Description |
|---------|-------------|
| `std` (default) | Enable std library support |
| `macros` (default) | Enable `#[simd_fn]` attribute macro |
| `wide` | Integration with `wide` crate |
| `safe-simd` | Integration with `safe_unaligned_simd` |

## License

MIT OR Apache-2.0

## AI-Generated Code Notice

Developed with Claude (Anthropic). Not all code manually reviewed. Review critical paths before production use.

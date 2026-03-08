+++
title = "The #[arcane] Macro"
weight = 3
+++

<sub>(alias: `#[token_target_features_boundary]`)</sub>

`#[arcane(import_intrinsics)]` creates a safe wrapper around SIMD code and auto-imports architecture intrinsics. Use it at **entry points**—functions called from non-SIMD code (after `summon()`, from tests, public APIs).

For functions called from other SIMD code, use [`#[rite]`](@/archmage/concepts/rite.md) instead — it inlines into the caller, avoiding the target-feature boundary. Use `#[rite(v3, import_intrinsics)]` with a tier name (no token needed) or `#[rite(import_intrinsics)]` with a token parameter.

> **Rust 1.85+ safety**: Inside the generated `#[target_feature]` function, value-based SIMD intrinsics (arithmetic, shuffle, compare, bitwise) are safe — no `unsafe` needed. Only pointer-based memory operations remain unsafe; use `import_intrinsics` to get safe memory ops that take references instead of raw pointers.

## How It Works

`#[arcane]` generates a sibling function with `#[target_feature]` at the same scope, plus a safe wrapper that calls it. Both functions live in the same scope, so `self` and `Self` work naturally in methods.

{% mermaid() %}
flowchart LR
    A["Your code:<br/>#[arcane]<br/>fn kernel(token: X64V3Token, ...)"] --> B["Macro generates:<br/>__arcane_kernel (unsafe, #[target_feature])<br/>kernel (safe wrapper)"]
    B --> C["Wrapper calls sibling<br/>via unsafe { __arcane_kernel(...) }"]
    C --> D["SAFETY: token proves<br/>CPU support exists"]

    style A fill:#2d5a27,color:#fff
    style B fill:#1a4a6e,color:#fff
    style D fill:#5a3d1e,color:#fff
{% end %}

{% mermaid() %}
flowchart TD
    S["summon() returns Some(token)"] --> A["#[arcane] fn (entry point)"]
    A --> R1["#[rite] fn<br/>(inlines fully)"]
    A --> R2["#[rite] fn<br/>(inlines fully)"]
    R1 --> R3["#[rite] fn<br/>(inlines fully)"]

    style S fill:#5a3d1e,color:#fff
    style A fill:#2d5a27,color:#fff
    style R1 fill:#1a4a6e,color:#fff
    style R2 fill:#1a4a6e,color:#fff
    style R3 fill:#1a4a6e,color:#fff
{% end %}

`#[arcane]` sits at the boundary between non-SIMD and SIMD code. Everything below it in the call tree uses `#[rite]`.

## Basic Usage

```rust
use archmage::prelude::*;

#[arcane(import_intrinsics)]
fn add_vectors(_token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    // Memory ops take references — fully safe inside #[arcane]!
    let va = _mm256_loadu_ps(a);
    let vb = _mm256_loadu_ps(b);
    let sum = _mm256_add_ps(va, vb);

    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, sum);
    out
}
```

## What It Generates

**Sibling mode (default):** Two functions at the same scope. `self`/`Self` work naturally.

<details>
<summary>Sibling expansion (click to expand)</summary>

```rust
// Your code:
#[arcane(import_intrinsics)]
fn add(token: X64V3Token, a: __m256, b: __m256) -> __m256 {
    _mm256_add_ps(a, b)  // In scope from import_intrinsics
}

// Generated (x86_64 only — cfg'd out on other architectures):
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
#[target_feature(enable = "avx2,fma,bmi1,bmi2,...")]
fn __arcane_add(token: X64V3Token, a: __m256, b: __m256) -> __m256 {
    use archmage::intrinsics::x86_64::*;
    _mm256_add_ps(a, b)  // Safe inside #[target_feature]!
}

#[cfg(target_arch = "x86_64")]
fn add(token: X64V3Token, a: __m256, b: __m256) -> __m256 {
    // SAFETY: Token proves CPU support was verified
    unsafe { __arcane_add(token, a, b) }
}
```

</details>

**Nested mode** (`#[arcane(nested, import_intrinsics)]` or `#[arcane(_self = Type, import_intrinsics)]`): Inner function inside the original. **Required for trait impls** — sibling expansion would add `__arcane_fn` to the impl block, which isn't in the trait definition and causes a compile error.

<details>
<summary>Nested expansion (click to expand)</summary>

```rust
// Your code (trait impl — must use nested):
impl SimdOps for MyType {
    #[arcane(_self = MyType, import_intrinsics)]
    fn compute(&self, token: X64V3Token) -> f32 {
        _self.data.iter().sum()  // Use _self, not self
    }
}

// Generated:
impl SimdOps for MyType {
    fn compute(&self, token: X64V3Token) -> f32 {
        #[target_feature(enable = "avx2,fma,bmi1,bmi2,...")]
        #[inline]
        fn __inner(_self: &MyType, token: X64V3Token) -> f32 {
            use archmage::intrinsics::x86_64::*;
            _self.data.iter().sum()
        }
        unsafe { __inner(self, token) }
    }
}
```

The inner `fn` can't have a `self` receiver (Rust doesn't allow that in inner functions), so the macro renames `self` → `_self` with the concrete type you specified.

</details>

## Token-to-Features Mapping

| Token | Enabled Features |
|-------|------------------|
| `X64V1Token` / `Sse2Token` | sse, sse2 (x86-64 baseline) |
| `X64V2Token` | + sse3, ssse3, sse4.1, sse4.2, popcnt |
| `X64CryptoToken` | V2 + pclmulqdq, aes |
| `X64V3Token` | + avx, avx2, fma, bmi1, bmi2, f16c |
| `X64V3CryptoToken` | V3 + vpclmulqdq, vaes |
| `X64V4Token` / `Server64` | + avx512f, avx512bw, avx512cd, avx512dq, avx512vl |
| `X64V4xToken` | V4 + vpopcntdq, ifma, vbmi, vnni, vbmi2, bitalg, vpclmulqdq, gfni, vaes |
| `Avx512Fp16Token` | V4 + avx512fp16 |
| `NeonToken` / `Arm64` | neon |
| `Arm64V2Token` | neon + crc, rdm, dotprod, fp16, aes, sha2 |
| `Arm64V3Token` | V2 + fhm, fcma, sha3, i8mm, bf16 |
| `NeonAesToken` | neon + aes |
| `NeonSha3Token` | neon + sha3 |
| `NeonCrcToken` | neon + crc |
| `Wasm128Token` | simd128 |
| `Wasm128RelaxedToken` | simd128 + relaxed-simd |

See [`token-registry.toml`](https://github.com/imazen/archmage/blob/main/token-registry.toml) for the complete mapping.

## Nesting #[arcane] Functions

Functions with the same token type inline into each other:

```rust
use archmage::prelude::*;

#[arcane(import_intrinsics)]
fn outer(token: X64V3Token, data: &[f32; 8]) -> f32 {
    let sum = inner(token, data);  // Inlines!
    sum * 2.0
}

#[rite(import_intrinsics)]
fn inner(_token: X64V3Token, data: &[f32; 8]) -> f32 {
    // Inlines into outer — same #[target_feature] region
    let v = _mm256_loadu_ps(data);
    let sum = _mm256_hadd_ps(v, v);
    let sum = _mm256_hadd_ps(sum, sum);
    let low = _mm256_castps256_ps128(sum);
    let high = _mm256_extractf128_ps::<1>(sum);
    _mm_cvtss_f32(_mm_add_ss(low, high))
}
```

## Downcasting Tokens

Higher tokens can call functions expecting lower tokens:

```rust
#[arcane(import_intrinsics)]
fn v4_kernel(token: X64V4Token, data: &[f32; 8]) -> f32 {
    // V4 ⊃ V3, so this works and inlines properly
    v3_sum(token, data)
}

#[rite(import_intrinsics)]
fn v3_sum(_token: X64V3Token, data: &[f32; 8]) -> f32 {
    let v = _mm256_loadu_ps(data);
    let sum = _mm256_hadd_ps(v, v);
    let sum = _mm256_hadd_ps(sum, sum);
    let low = _mm256_castps256_ps128(sum);
    let high = _mm256_extractf128_ps::<1>(sum);
    _mm_cvtss_f32(_mm_add_ss(low, high))
}
```

## Cross-Architecture Behavior

**Default (cfg-out):** On non-matching architectures, no function is emitted at all — less dead code. Code referencing the function must use `#[cfg]` guards or `incant!`.

**With `stub`:** `#[arcane(stub)]` generates an `unreachable!()` stub on wrong architectures. Use when cross-arch dispatch references the function without cfg guards.

```rust
// Default: cfg'd out on non-x86 — doesn't exist
#[arcane(import_intrinsics)]
fn process_avx2(token: X64V3Token, data: &[f32]) -> f32 { ... }

// With stub: unreachable stub exists on non-x86
#[arcane(stub, import_intrinsics)]
fn process_avx2_stubbed(token: X64V3Token, data: &[f32]) -> f32 { ... }
```

See [Cross-Platform](@/archmage/concepts/cross-platform.md) for dispatch patterns.

## Options

| Option | Effect |
|--------|--------|
| `#[arcane(import_intrinsics)]` | **Recommended default.** Sibling expansion + auto-import intrinsics |
| `#[arcane(import_intrinsics, import_magetypes)]` | Also auto-import magetypes SIMD types |
| `#[arcane(_self = Type, import_intrinsics)]` | **For trait impls.** Nested mode, replaces `self`→`_self` |
| `#[arcane]` | Sibling expansion, no auto-imports (intrinsics must be in scope) |
| `#[arcane(stub, import_intrinsics)]` | Sibling + unreachable stub on wrong arch |
| `#[arcane(nested, import_intrinsics)]` | Nested inner function (for trait impls without `self`) |
| `#[arcane(nested, stub, import_intrinsics)]` | Nested + stub |
| `#[arcane(inline_always)]` | Force `#[inline(always)]` (nightly only) |

### `stub`

Generate an `unreachable!()` stub on non-matching architectures:

```rust
#[arcane(stub, import_intrinsics)]
fn process(token: X64V3Token, data: &[f32]) -> f32 {
    // Real implementation on x86, unreachable stub on ARM/WASM
    data.iter().sum()
}
```

### `nested` and `_self = Type`

Use nested mode for **trait implementations**. Sibling expansion generates `__arcane_fn` as a separate method in the impl block — but that method isn't in the trait definition, so the compiler rejects it. Nested mode puts the `#[target_feature]` function *inside* the original method instead.

**`_self = Type` is the recommended way** to use nested mode. It implies `nested` and also renames `self` → `_self` with the concrete type, which is required because inner `fn` items can't have a `self` receiver.

```rust
trait SimdProcessor {
    fn process(&self, token: X64V3Token, data: &[f32; 8]) -> [f32; 8];
    fn transform(&mut self, token: X64V3Token);
}

struct MyType {
    scale: f32,
    data: [f32; 8],
}

impl SimdProcessor for MyType {
    #[arcane(_self = MyType, import_intrinsics)]
    fn process(&self, token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
        // Use _self instead of self
        let v = _mm256_loadu_ps(data);
        let scale = _mm256_set1_ps(_self.scale);
        let result = _mm256_mul_ps(v, scale);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(&mut out, result);
        out
    }

    #[arcane(_self = MyType, import_intrinsics)]
    fn transform(&mut self, token: X64V3Token) {
        // _self is &mut MyType here
        let v = _mm256_loadu_ps(&_self.data);
        let doubled = _mm256_add_ps(v, v);
        _mm256_storeu_ps(&mut _self.data, doubled);
    }
}
```

**When do you need `_self`?** Only for trait impls that access `self`. For free functions and inherent methods (`impl MyType { ... }`), use the default sibling mode — `self` and `Self` work naturally.

| Context | What to use |
|---------|-------------|
| Free function | `#[arcane(import_intrinsics)]` |
| Inherent method (`impl MyType`) | `#[arcane(import_intrinsics)]` — `self`/`Self` just work |
| Trait impl (`impl Trait for Type`) | `#[arcane(_self = Type, import_intrinsics)]` — use `_self` |

See [Methods with #[arcane]](@/archmage/advanced/methods.md) for comprehensive examples including all receiver types (`&self`, `&mut self`, `self`).

### `inline_always`

Force aggressive inlining (requires nightly):

```rust
#![feature(target_feature_inline_always)]

#[arcane(inline_always)]
fn hot_path(token: X64V3Token, data: &[f32]) -> f32 {
    // Uses #[inline(always)] instead of #[inline]
}
```

### `import_intrinsics`

Auto-imports architecture intrinsics and safe memory operations into the function body:

```rust
// No manual `use core::arch::x86_64::*` needed!
#[arcane(import_intrinsics)]
fn kernel(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let v = _mm256_loadu_ps(data);     // Safe! Takes &[f32; 8]
    let doubled = _mm256_add_ps(v, v);
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, doubled);
    out
}
```

The macro injects:
- `use archmage::intrinsics::{arch}::*;` — types, value intrinsics, and safe memory ops

This single import combines `core::arch` types and value ops with reference-based safe memory ops. Safe versions shadow unsafe pointer-based ones automatically.

The architecture is derived from the token type: `X64V3Token` → `x86_64`, `NeonToken` → `aarch64`, `Wasm128Token` → `wasm32`.

### `import_magetypes`

Auto-imports magetypes SIMD types for the token's namespace:

```rust
// No manual `use magetypes::simd::v3::*` needed!
#[arcane(import_magetypes)]
fn process(token: X64V3Token, data: &[f32; 8]) -> f32 {
    let v = f32x8::load(token, data);  // In scope from auto-import
    v.reduce_add()
}
```

The macro injects:
- `use magetypes::simd::{ns}::*;` — pre-specialized types for the token (f32x8, i32x8, etc.)
- `use magetypes::simd::backends::*;` — backend traits (F32x8Backend, etc.)

The namespace is token-driven:

| Token | Namespace | Types |
|-------|-----------|-------|
| `X64V1..V3Token` | `v3` | 128/256-bit native, 512-bit polyfill |
| `X64V4Token` | `v4` | 128/256/512-bit native |
| `X64V4xToken` | `v4x` | 128/256/512-bit native |
| `NeonToken` / ARM | `neon` | 128-bit native, wider polyfills |
| `Wasm128Token` | `wasm128` | 128-bit native, wider polyfills |

Both options can be combined: `#[arcane(import_intrinsics, import_magetypes)]`.

Works with trait bounds (`impl HasX64V2`) and generic parameters (`<T: HasNeon>`) too.

## Common Patterns

### Public API with Internal Implementation

```rust
pub fn process(data: &mut [f32]) {
    if let Some(token) = X64V3Token::summon() {
        process_simd(token, data);
    } else {
        process_scalar(data);
    }
}

#[arcane(import_intrinsics)]
fn process_simd(token: X64V3Token, data: &mut [f32]) {
    // SIMD implementation — intrinsics in scope from import_intrinsics
}

fn process_scalar(data: &mut [f32]) {
    // Fallback
}
```

### Generic Over Tokens

```rust
use archmage::HasX64V2;

#[arcane(import_intrinsics)]
fn generic_impl<T: HasX64V2>(token: T, a: __m128, b: __m128) -> __m128 {
    // Works with X64V2Token, X64V3Token, X64V4Token
    // Note: generic bounds create optimization boundaries
    _mm_add_ps(a, b)
}
```

**Warning**: Generic bounds prevent inlining across the boundary. Prefer concrete tokens for hot paths.

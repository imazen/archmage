# Magetypes V2 Specification

> Cross-platform SIMD with zero magic, maximum clarity.

## Overview

This spec defines the archmage SIMD ecosystem:

1. **Tokens** — Zero-sized proofs of CPU capability
2. **`#[arcane]`** — Enable SIMD intrinsics safely via token proof
3. **`#[magetypes]`** — Generate platform variants from generic code
4. **`incant!`** — Dispatch to the right variant at runtime
5. **Generic dispatch** — Trait-based compile-time routing
6. **Prelude** — Platform-appropriate types without cfg blocks
7. **Cross-tier casting** — Safe upcasting inside `#[arcane]` context

## Naming Convention

| Fun | Boring (deprecated alias) | Purpose |
|-----|---------------------------|---------|
| `#[arcane]` | `#[simd_fn]` | Enable target features via token |
| `summon()` | `try_new()` | Runtime feature detection |
| `incant!` | `simd_route!` | Dispatch to platform variants |
| `#[magetypes]` | `#[multiwidth]` | Generate platform variants |

---

## 1. Tokens

Tokens are zero-sized types that prove CPU features are available.

### Token Types

| Token | Platform | Features |
|-------|----------|----------|
| `X64V2Token` | x86_64 | SSE4.2, POPCNT |
| `X64V3Token` / `Desktop64` | x86_64 | AVX2, FMA, BMI2 |
| `X64V4Token` / `Server64` | x86_64 | AVX-512 F/BW/CD/DQ/VL |
| `Avx512ModernToken` | x86_64 | + VNNI, VBMI2, etc. |
| `NeonToken` / `Arm64` | aarch64 | NEON (always available) |
| `NeonAesToken` | aarch64 | + AES |
| `Simd128Token` | wasm32 | WASM SIMD |
| `ScalarToken` | all | No SIMD (fallback) |

### SimdToken Trait

```rust
pub trait SimdToken: Copy + Clone + Send + Sync + 'static {
    const NAME: &'static str;

    /// Check compile-time availability.
    /// - Some(true): compiler guarantees this feature
    /// - Some(false): wrong architecture (stub)
    /// - None: might be available, call summon()
    fn guaranteed() -> Option<bool>;

    /// Attempt to create token with runtime detection.
    fn summon() -> Option<Self>;

    /// Legacy alias for summon().
    #[deprecated]
    fn try_new() -> Option<Self> { Self::summon() }
}
```

### ScalarToken

Always-available token for fallback paths:

```rust
#[derive(Clone, Copy, Debug)]
pub struct ScalarToken;

impl SimdToken for ScalarToken {
    const NAME: &'static str = "Scalar";
    fn guaranteed() -> Option<bool> { Some(true) }
    fn summon() -> Option<Self> { Some(Self) }
}
```

---

## 2. `#[arcane]` Macro

Enables SIMD intrinsics by generating `#[target_feature]` inner function.

### Basic Usage

```rust
use archmage::{arcane, X64V3Token};

#[arcane]
fn dot(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    // AVX2 + FMA intrinsics safe here
    unsafe {
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let prod = _mm256_mul_ps(va, vb);
        // ... reduce ...
    }
}
```

### Cross-Architecture Stubs

When using a concrete token on the wrong architecture, `#[arcane]` generates a stub:

```rust
// On x86, this compiles and works
// On ARM, this compiles to a stub that panics if called
#[arcane]
fn x86_kernel(token: X64V3Token, data: &[f32]) -> f32 {
    // x86 intrinsics...
}
```

Generated on ARM:
```rust
#[cfg(not(target_arch = "x86_64"))]
fn x86_kernel(_token: X64V3Token, _data: &[f32]) -> f32 {
    unreachable!("X64V3Token cannot be summoned on this architecture")
}
```

This allows cross-platform crates to compile everywhere.

### Self Receivers

For trait implementations, use `_self`:

```rust
impl SimdOps for MyType {
    #[arcane(_self = MyType)]
    fn process(&self, token: X64V3Token) -> f32 {
        // Use _self instead of self in body
        _self.data.iter().sum()
    }
}
```

### Supported Token Forms

| Signature | Works? | Features Enabled |
|-----------|--------|------------------|
| `token: X64V3Token` | ✅ | AVX2, FMA, BMI2, etc. |
| `token: impl HasX64V2` | ✅ | SSE4.2, POPCNT |
| `token: impl HasX64V4` | ✅ | AVX-512 |
| `token: impl HasNeon` | ✅ | NEON |
| `token: impl SimdToken` | ❌ | **No features** — too generic |
| `token: T where T: SimdToken` | ❌ | **No features** — too generic |

**Rule:** `#[arcane]` requires compile-time-known features. Use concrete tokens or feature-level trait bounds.

---

## 3. `#[magetypes]` Macro

Generates platform-specific variants from generic code.

### Function-Level Usage

```rust
use archmage::magetypes;

#[magetypes]
pub fn dot(token: Token, a: &[f32], b: &[f32]) -> f32 {
    let mut sum = f32xN::zero(token);
    for (a_chunk, b_chunk) in a.chunks_exact(LANES).zip(b.chunks_exact(LANES)) {
        let va = f32xN::load(token, a_chunk);
        let vb = f32xN::load(token, b_chunk);
        sum = sum.add(va.mul(vb, token), token);
    }
    sum.reduce_add(token)
}
```

Generates:
- `dot_v3(token: X64V3Token, ...)` — AVX2, f32x8
- `dot_v4(token: X64V4Token, ...)` — AVX-512, f32x16
- `dot_neon(token: NeonToken, ...)` — NEON, f32x4
- `dot_wasm128(token: Simd128Token, ...)` — WASM, f32x4
- `dot_scalar(token: ScalarToken, ...)` — No SIMD

### Type Aliases

Inside `#[magetypes]`, these aliases resolve per-variant:

| Alias | v3 (AVX2) | v4 (AVX-512) | neon | wasm128 | scalar |
|-------|-----------|--------------|------|---------|--------|
| `Token` | `X64V3Token` | `X64V4Token` | `NeonToken` | `Simd128Token` | `ScalarToken` |
| `f32xN` | `f32x8` | `f32x16` | `f32x4` | `f32x4` | `f32` |
| `i32xN` | `i32x8` | `i32x16` | `i32x4` | `i32x4` | `i32` |
| `LANES` | `8` | `16` | `4` | `4` | `1` |

### No Magic Call Rewriting

`#[magetypes]` does **NOT** transform function calls. This keeps the macro simple and predictable.

```rust
#[magetypes]
mod kernels {
    pub fn outer(token: Token, data: &[f32]) -> f32 {
        // WRONG: inner() won't be transformed to inner_v3()
        inner(token, data)  // compile error - inner() doesn't exist

        // RIGHT: be explicit
        inner_v3(token, data)  // if you know the variant
        // OR use incant! for dispatch
    }
    pub fn inner(token: Token, data: &[f32]) -> f32 { ... }
}
```

For calling sibling functions, either:
1. Call the suffixed version directly: `inner_v3(token, data)`
2. Use `incant!` with passthrough: `incant!(inner(data) with token)`

### Module-Level Usage

For functions that share types:

```rust
#[magetypes]
mod kernels {
    pub fn dot(token: Token, a: &[f32], b: &[f32]) -> f32 { ... }
    pub fn norm(token: Token, data: &[f32]) -> f32 { ... }
}
```

---

## 4. `incant!` Macro

Dispatches to the best available SIMD variant.

### Entry Point (No Token Yet)

```rust
pub fn public_api(data: &[f32]) -> f32 {
    incant!(dot(data))
}
```

Expands to:
```rust
pub fn public_api(data: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = X64V3Token::summon() {
            return dot_v3(token, data);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if let Some(token) = NeonToken::summon() {
            return dot_neon(token, data);
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(token) = Simd128Token::summon() {
            return dot_wasm128(token, data);
        }
    }
    dot_scalar(ScalarToken, data)
}
```

### Passthrough (Already Have Token)

```rust
#[arcane]
fn outer(token: X64V3Token, data: &[f32]) -> f32 {
    incant!(inner(data) with token)
}
```

Uses trait-based dispatch (see Section 5).

---

## 5. Generic Dispatch via Traits

For advanced use: dispatch at compile-time via monomorphization.

### IntoConcreteToken Trait

```rust
pub trait IntoConcreteToken: SimdToken + Sized {
    fn as_x64v2(self) -> Option<X64V2Token> { None }
    fn as_x64v3(self) -> Option<X64V3Token> { None }
    fn as_x64v4(self) -> Option<X64V4Token> { None }
    fn as_neon(self) -> Option<NeonToken> { None }
    fn as_wasm128(self) -> Option<Simd128Token> { None }
    fn as_scalar(self) -> Option<ScalarToken> { None }
}

impl IntoConcreteToken for X64V3Token {
    fn as_x64v3(self) -> Option<X64V3Token> { Some(self) }
}

impl IntoConcreteToken for NeonToken {
    fn as_neon(self) -> Option<NeonToken> { Some(self) }
}

impl IntoConcreteToken for ScalarToken {
    fn as_scalar(self) -> Option<ScalarToken> { Some(self) }
}
// ... etc for all tokens
```

### Generic Dispatch Function

```rust
fn invoke<T: IntoConcreteToken>(token: T, data: &[f32]) -> f32 {
    if let Some(t) = token.as_x64v3() { return dot_v3(t, data); }
    if let Some(t) = token.as_x64v4() { return dot_v4(t, data); }
    if let Some(t) = token.as_neon() { return dot_neon(t, data); }
    if let Some(t) = token.as_wasm128() { return dot_wasm128(t, data); }
    if let Some(t) = token.as_scalar() { return dot_scalar(t, data); }
    unreachable!()
}
```

The compiler monomorphizes this — for `X64V3Token`, only the first branch survives.

### `incant!` with Passthrough

`incant!(fn(args) with token)` generates this pattern automatically.

---

## 6. Prelude System

Platform-appropriate types without manual `#[cfg]` blocks.

### Usage

```rust
use magetypes::prelude::*;

// Gets the "best" types for the compile target:
// - x86_64: f32x8, i32x8, etc. (256-bit, X64V3Token)
// - x86_64 + avx512: f32x16, i32x16, etc. (512-bit, X64V4Token)
// - aarch64: f32x4, i32x4, etc. (128-bit, NeonToken)
// - wasm32: f32x4, i32x4, etc. (128-bit, Simd128Token)
```

### Prelude Contents

```rust
// magetypes/src/prelude.rs

#[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
pub use crate::simd::x86::w256::*;

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub use crate::simd::x86::w512::*;

#[cfg(target_arch = "aarch64")]
pub use crate::simd::arm::w128::*;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub use crate::simd::wasm::w128::*;

// Also export the recommended token
#[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
pub use archmage::X64V3Token as RecommendedToken;

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub use archmage::X64V4Token as RecommendedToken;

#[cfg(target_arch = "aarch64")]
pub use archmage::NeonToken as RecommendedToken;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub use archmage::Simd128Token as RecommendedToken;
```

### Prelude vs `#[magetypes]`

- **Prelude**: For non-generic code targeting a single platform
- **`#[magetypes]`**: For generating multiple platform variants

Code inside `#[magetypes]` doesn't use the prelude — it uses the macro's aliases.

---

## 7. Cross-Tier Casting

Safe upcasting inside `#[arcane]` context, following Rust 1.85+ target_feature rules.

### Upcast (Requires Context)

```rust
impl x86::w128::f32x4 {
    /// Upgrade to AVX2-context f32x4.
    /// Safe inside #[arcane] with X64V3Token, requires unsafe otherwise.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub fn upcast_v3(self) -> x86::w256::f32x4 {
        unsafe { core::mem::transmute(self) }
    }
}
```

Usage:
```rust
#[arcane]
fn process(token: X64V3Token, input: f32x4_sse) -> f32x4_avx2 {
    // SAFE: Inside #[arcane] with X64V3Token
    input.upcast_v3()
}
```

### Downcast (Always Safe)

```rust
impl x86::w256::f32x4 {
    /// Downcast to SSE-context. Always safe.
    #[inline(always)]
    pub const fn downcast(self) -> x86::w128::f32x4 {
        unsafe { core::mem::transmute(self) }
    }
}
```

### Cast Methods

| From | To | Method | Required Features |
|------|-----|--------|-------------------|
| `w128::f32x4` | `w256::f32x4` | `upcast_v3()` | avx2, fma |
| `w128::f32x4` | `w512::f32x4` | `upcast_v4()` | avx512f, avx512vl |
| `w256::*` | `w128::*` | `downcast()` | (none) |
| `w512::*` | `w256::*` | `downcast()` | (none) |

---

## 8. Magetypes SIMD Types

The `magetypes` crate provides SIMD types with safe methods:

```rust
use magetypes::simd::*;

#[arcane]
fn example(token: X64V3Token, data: &[f32; 8]) -> f32 {
    let v = f32x8::load(token, data);
    let squared = v.mul(v, token);
    squared.reduce_add(token)
}
```

### SimdTypes Trait

For generic programming over token types:

```rust
pub trait SimdTypes: SimdToken {
    type F32: SimdFloat;
    type I32: SimdInt;
    // ...
}

impl SimdTypes for X64V3Token {
    type F32 = f32x8;
    type I32 = i32x8;
}

impl SimdTypes for NeonToken {
    type F32 = f32x4;
    type I32 = i32x4;
}
```

Usage:
```rust
fn generic<T: SimdToken + SimdTypes>(token: T, data: &[f32]) -> f32 {
    let v = <T as SimdTypes>::F32::load(token, data);
    v.reduce_add(token)
}
```

---

## 9. Idiomatic Patterns

**See `tests/idiomatic_patterns.rs` for executable examples.**

### Quick Reference

| Pattern | Syntax | Recommended? | Notes |
|---------|--------|--------------|-------|
| Concrete token | `fn f(t: X64V3Token)` | ✅ | Best for most code |
| Friendly alias | `fn f(t: Desktop64)` | ✅ | Same as X64V3Token |
| Feature trait | `fn f(t: impl HasX64V2)` | ⚠️ | Only SSE4.2, not AVX2 |
| Feature trait | `fn f(t: impl HasX64V4)` | ✅ | For AVX-512 code |
| Generic token | `fn f<T: SimdToken>(t: T)` | ❌ | **Cannot work** |
| `_self` pattern | `#[arcane(_self = T)]` | ✅ | For trait methods |
| Token passthrough | Nested `#[arcane]` calls | ✅ | Natural composition |

### Trait Implementation Patterns

**Pattern 1: Separate methods per platform**
```rust
trait SimdOps {
    fn process_v3(&self, token: X64V3Token) -> Self;
    fn process_neon(&self, token: NeonToken) -> Self;
    fn process_scalar(&self) -> Self;
}
```

**Pattern 2: Generic dispatches to concrete helpers**
```rust
trait SimdOps {
    fn process(&self) -> Self;  // No token
}

impl SimdOps for Data {
    fn process(&self) -> Self {
        incant!(self.process_impl())
    }
}
```

**Pattern 3: Use `#[magetypes]` for trait impls**
```rust
#[magetypes]
impl SimdKernel for [f32; LANES] {
    type Token = Token;
    fn kernel(&self, token: Token) -> Self { ... }
}
// Generates impls for [f32; 8], [f32; 4], etc.
```

---

## 10. Complete Example

```rust
use archmage::{magetypes, incant, SimdToken, ScalarToken};

// Generate all platform variants
#[magetypes]
pub fn dot(token: Token, a: &[f32], b: &[f32]) -> f32 {
    if a.len() < LANES {
        return a.iter().zip(b).map(|(a, b)| a * b).sum();
    }

    let mut sum = f32xN::zero(token);
    let chunks = a.len() / LANES;

    for i in 0..chunks {
        let va = f32xN::load(token, &a[i * LANES..]);
        let vb = f32xN::load(token, &b[i * LANES..]);
        sum = sum.add(va.mul(vb, token), token);
    }

    let mut result = sum.reduce_add(token);
    for i in (chunks * LANES)..a.len() {
        result += a[i] * b[i];
    }
    result
}

// Public API with automatic dispatch
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    incant!(dot(a, b))
}
```

---

## 11. Migration from V1

| V1 | V2 |
|----|-----|
| `#[simd_fn]` | `#[arcane]` |
| `Token::try_new()` | `Token::summon()` |
| `#[multiwidth]` | `#[magetypes]` |
| Manual dispatch | `incant!()` |
| `impl Has256BitSimd` | Use concrete tokens or `#[magetypes]` |
| `conjure()` | `summon().unwrap()` |

### Width Traits (Removed)

`Has128BitSimd`, `Has256BitSimd`, `Has512BitSimd` are removed — they didn't map to specific features.

Replace with:
- Concrete tokens: `X64V3Token`, `NeonToken`
- Feature traits: `HasX64V2`, `HasX64V4`, `HasNeon`
- Code generation: `#[magetypes]`

---

## 12. Implementation Status

| Feature | Status |
|---------|--------|
| `#[arcane]` | ✅ Implemented |
| `#[arcane]` cross-arch stubs | ✅ Implemented |
| `summon()` / `try_new()` | ✅ Implemented |
| `guaranteed()` | ✅ Implemented |
| `ScalarToken` | ✅ Implemented |
| `IntoConcreteToken` trait | ✅ Implemented |
| `#[magetypes]` function-level | ⬜ Not yet (module-level as `#[multiwidth]`) |
| `incant!` entry point | ⬜ Not yet |
| `incant!` passthrough | ⬜ Not yet |
| `SimdTypes` trait | ⬜ Not yet |
| Prelude system | ⬜ Not yet |
| Cross-tier casting | ⬜ Not yet |

---

## 13. Design Principles

1. **No magic** — Explicit is better than implicit. No hidden call rewriting.
2. **Compile everywhere** — Cross-arch stubs prevent compile errors.
3. **Zero cost** — Monomorphization eliminates dispatch overhead.
4. **Tokens are proof** — If you have a token, features are available.
5. **Scalar is always available** — `ScalarToken` ensures fallback exists.
6. **Fun names, boring aliases** — Thematic naming with corporate fallbacks.

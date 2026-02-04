# Magetypes V2 Specification

**Status:** Draft
**Author:** Design session 2026-02-04

This document specifies major ergonomic and safety improvements to the archmage ecosystem.

## Executive Summary

This spec proposes:

1. **Drop width traits** (`Has128BitSimd`, `Has256BitSimd`, `Has512BitSimd`) — they prevent `#[arcane]` from enabling features
2. **`incant!` macro** — cross-platform dispatch that summons tokens and routes to suffixed `#[arcane]` functions
3. **Rename `#[multiwidth]` → `#[magetypes]`** — with auto-imports, type shadowing, and zero-cost casting
4. **Document `_self` pattern** — token goes 2nd, use `_self` in body for trait impls
5. **Prelude system** — compile-time platform-appropriate types without type conflicts
6. **Safe cross-tier casting** — `.from()` that's safe inside token context, unsafe outside

---

## 1. Drop Width Traits (BREAKING)

### Problem

The traits `Has128BitSimd`, `Has256BitSimd`, `Has512BitSimd` cause `#[arcane]` to fail optimization because they don't map to specific CPU features:

```rust
// Current: trait_to_features("Has256BitSimd") returns ["sse", "sse2", "avx"]
// This is WRONG — AVX alone doesn't enable useful 256-bit ops
// Real 256-bit code needs AVX2 + FMA (x86-64-v3)

#[arcane]
fn bad(token: impl Has256BitSimd, data: &[f32; 8]) -> [f32; 8] {
    // PROBLEM: Only AVX enabled, not AVX2/FMA
    // _mm256_fmadd_ps will NOT be optimized properly
    let v = _mm256_loadu_ps(data.as_ptr());
    _mm256_fmadd_ps(v, v, v)  // FMA intrinsic, but FMA feature not enabled!
}
```

The width traits exist to provide generic bounds like "accepts any 256-bit token" but:
- They don't enable the right features for `#[target_feature]`
- `Has256BitSimd` on x86 could mean AVX (useless for most ops) or AVX2+FMA (useful)
- ARM NEON is 128-bit only but has different features than x86 SSE4.2

### Solution

**Remove width traits entirely.** Use concrete tokens or feature-level traits instead:

```rust
// Before (broken):
fn process(token: impl Has256BitSimd, ...) { }

// After (works):
fn process(token: X64V3Token, ...) { }  // Concrete: gets AVX2+FMA
fn process(token: impl HasX64V2, ...) { }  // Trait: gets SSE4.2
```

For cross-platform code, use the `#[magetypes]` macro (see Section 4) which generates per-platform variants.

### Migration

1. Replace `impl Has128BitSimd` → `impl HasX64V2` (x86) or `impl HasNeon` (ARM) or concrete token
2. Replace `impl Has256BitSimd` → `X64V3Token` (there's no equivalent trait; v3 is the only 256-bit option)
3. Replace `impl Has512BitSimd` → `impl HasX64V4` or concrete `X64V4Token`

### Token Registry Changes

```toml
# REMOVE these traits from token-registry.toml:
# - Has128BitSimd
# - Has256BitSimd
# - Has512BitSimd

# UPDATE tokens to remove width trait impls:
[[token]]
name = "X64V3Token"
traits = ["HasX64V2"]  # was ["HasX64V2", "Has128BitSimd", "Has256BitSimd"]
```

---

## 2. The `incant!` Macro

### Problem

Writing cross-platform SIMD code requires tedious boilerplate:

```rust
// Current: manual dispatch for every call site
#[cfg(target_arch = "x86_64")]
if let Some(token) = X64V3Token::summon() {
    return process_x86_v3(token, data);
}
#[cfg(target_arch = "x86_64")]
if let Some(token) = X64V4Token::summon() {
    return process_x86_v4(token, data);
}
#[cfg(target_arch = "aarch64")]
if let Some(token) = NeonToken::summon() {
    return process_neon(token, data);
}
#[cfg(target_arch = "wasm32")]
#[cfg(target_feature = "wasm128")]
{
    let token = Wasm128Token::new();
    return process_wasm(token, data);
}
// Fallback...
```

### Solution

The `incant!` macro generates optimal dispatch:

```rust
/// Dispatch to the best available SIMD implementation.
///
/// Generates token summoning for all relevant platforms and calls
/// the appropriately-suffixed function variant.
///
/// # Naming Convention
///
/// Given `incant!(my_func(args...))`, the macro expects:
/// - `my_func_v3(token: X64V3Token, args...)` — x86 AVX2+FMA
/// - `my_func_v4(token: X64V4Token, args...)` — x86 AVX-512 (optional, feature-gated)
/// - `my_func_neon(token: NeonToken, args...)` — ARM NEON
/// - `my_func_wasm128(token: Wasm128Token, args...)` — WASM WASM128
/// - `my_func_scalar(args...)` — fallback (no token)
///
/// # Example
///
/// ```rust
/// use archmage::{incant, arcane, X64V3Token, NeonToken};
///
/// #[arcane]
/// fn dot_v3(token: X64V3Token, a: &[f32], b: &[f32]) -> f32 {
///     // AVX2+FMA implementation
/// }
///
/// #[arcane]
/// fn dot_neon(token: NeonToken, a: &[f32], b: &[f32]) -> f32 {
///     // NEON implementation
/// }
///
/// fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
///     a.iter().zip(b).map(|(x, y)| x * y).sum()
/// }
///
/// // Dispatch to best available:
/// let result = incant!(dot(a, b));
/// ```
#[macro_export]
macro_rules! incant {
    ($fn_name:ident ($($args:expr),* $(,)?)) => {{
        // x86_64: try v4 first (if avx512 feature), then v3
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(feature = "avx512")]
            if let Some(token) = $crate::X64V4Token::summon() {
                return paste::paste! { [<$fn_name _v4>](token, $($args),*) };
            }
            if let Some(token) = $crate::X64V3Token::summon() {
                return paste::paste! { [<$fn_name _v3>](token, $($args),*) };
            }
        }

        // aarch64: NEON always available
        #[cfg(target_arch = "aarch64")]
        {
            let token = $crate::NeonToken::summon().unwrap();
            return paste::paste! { [<$fn_name _neon>](token, $($args),*) };
        }

        // wasm32: compile-time only (Rust uses "simd128" as the feature name)
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            let token = $crate::Wasm128Token::new();
            return paste::paste! { [<$fn_name _wasm128>](token, $($args),*) };
        }

        // Fallback
        paste::paste! { [<$fn_name _scalar>]($($args),*) }
    }};
}
```

### Compile-Time Elision

When a target is statically known (via `#[cfg]` or `-C target-cpu`), unreachable branches are eliminated:

```rust
// Compiled for ARM64 specifically:
// incant!(dot(a, b)) expands to just:
let token = NeonToken::summon().unwrap();
dot_neon(token, a, b)
// No x86 or WASM code generated
```

### Optional Suffixes

Not all functions need all variants. Missing suffixes compile but fall through:

```rust
// Only x86 and scalar defined:
#[arcane] fn compress_v3(token: X64V3Token, data: &[u8]) -> Vec<u8> { ... }
fn compress_scalar(data: &[u8]) -> Vec<u8> { ... }

// On ARM: falls through to scalar (compile warning suggested)
incant!(compress(data))
```

### Return Type

`incant!` returns the function's return type. All variants must have the same return type.

### Integration with `#[magetypes]`

The `incant!` macro is designed to call functions generated by `#[magetypes]`:

```rust
use archmage::magetypes;

// Define kernels once, generate platform variants
#[magetypes]
mod kernels {
    pub fn dot(token: Token, a: &[f32], b: &[f32]) -> f32 {
        let mut sum = f32xN::zero(token);
        for chunk in a.chunks_exact(LANES_F32).zip(b.chunks_exact(LANES_F32)) {
            let (a_chunk, b_chunk) = chunk;
            let va = f32xN::load(token, a_chunk.try_into().unwrap());
            let vb = f32xN::load(token, b_chunk.try_into().unwrap());
            sum = va.mul_add(vb, sum);
        }
        sum.reduce_add()
    }
}

// Scalar fallback (required by incant!)
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// Public API: dispatches to best available
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    incant!(kernels::dot(a, b))
}
```

**What `#[magetypes]` generates:**
- `kernels::dot_v3(token: X64V3Token, a: &[f32], b: &[f32]) -> f32`
- `kernels::dot_v4(token: X64V4Token, a: &[f32], b: &[f32]) -> f32` (if `avx512` feature)
- `kernels::dot_neon(token: NeonToken, a: &[f32], b: &[f32]) -> f32`
- `kernels::dot_wasm128(token: Wasm128Token, a: &[f32], b: &[f32]) -> f32`

**What `incant!` does at the call site:**
1. On x86_64: tries `dot_v4` (if feature enabled), then `dot_v3`, then `dot_scalar`
2. On aarch64: calls `dot_neon` (NEON always available)
3. On wasm32: calls `dot_wasm128` (if simd128 target feature) or `dot_scalar`
4. Elsewhere: calls `dot_scalar`

---

## 3. Document `_self` Pattern for Trait Impls

### The Pattern

When implementing traits with `#[arcane]`, put the token **second** and use `_self` in the body:

```rust
trait SimdOps {
    fn double(&self, token: impl HasX64V2) -> Self;
}

impl SimdOps for MyVector {
    #[arcane(_self = MyVector)]
    fn double(&self, _token: impl HasX64V2) -> Self {
        // Use _self, not self
        *_self + *_self
    }
}
```

### Why Token Second?

Trait methods have `self` as the receiver. The token must come after:

```rust
// Trait definition:
trait Ops {
    fn process(&self, token: T) -> Self;
    //         ^^^^   ^^^^^
    //         1st    2nd (token MUST be 2nd)
}
```

### Why `_self`?

The `#[arcane]` macro generates an inner function where `self` becomes a regular parameter. The leading underscore:
1. Reminds you it's not the real `self` keyword
2. Follows Rust convention for "used but unusual" names
3. Avoids shadowing warnings

### Receiver Types

All receiver types work:

| Trait signature | `#[arcane]` arg | In body |
|-----------------|-----------------|---------|
| `fn op(self, ...)` | `_self = Type` | `_self` (owned) |
| `fn op(&self, ...)` | `_self = Type` | `_self` (is `&Type`) |
| `fn op(&mut self, ...)` | `_self = Type` | `_self` (is `&mut Type`) |

### Documentation Update

Add to `#[arcane]` docs:

```rust
/// ## Methods with Self Receivers
///
/// For trait implementations, put the token parameter **second** (after self)
/// and use `_self` in the function body:
///
/// ```rust
/// trait SimdOps {
///     fn scale(&mut self, token: impl HasX64V2, factor: f32);
/// }
///
/// impl SimdOps for f32x8 {
///     #[arcane(_self = f32x8)]
///     fn scale(&mut self, _token: impl HasX64V2, factor: f32) {
///         // Use _self instead of self in the body
///         *_self = *_self * f32x8::splat(factor);
///     }
/// }
/// ```
///
/// **Token position:** Always second. The trait defines `&self` as first.
///
/// **Body reference:** Use `_self` (with underscore) to refer to the receiver.
/// This is a regular parameter after macro expansion, not the `self` keyword.
```

---

## 4. The `#[magetypes]` Macro (Replaces `#[multiwidth]`)

### Overview

`#[magetypes]` is the new name for `#[multiwidth]`, with enhanced functionality:

1. **Auto-imports** — Automatically imports `core::arch::*` and `safe_unaligned_simd::*` for the target
2. **Type shadowing** — Uses the highest-capability types for the current token
3. **Generates variants** — Creates `_v3`, `_v4`, `_neon`, `_wasm128` suffixed functions
4. **Works with `incant!`** — The generated functions follow the naming convention

### Basic Usage

```rust
use archmage::magetypes;

#[magetypes]
mod kernels {
    // Inside this module:
    // - `Token` is a type alias for the current platform's token
    // - `f32xN`, `i32xN` etc. are type aliases for the current width
    // - `core::arch::{x86_64,aarch64,wasm32}::*` is auto-imported
    // - `safe_unaligned_simd::{x86_64,aarch64,wasm32}::*` is auto-imported

    pub fn dot(token: Token, a: &[f32], b: &[f32]) -> f32 {
        let mut sum = f32xN::zero(token);
        // ...use SIMD ops...
        sum.reduce_add()
    }
}

// Generated:
// - kernels::dot_v3(X64V3Token, ...) — 256-bit f32x8
// - kernels::dot_v4(X64V4Token, ...) — 512-bit f32x16 (if avx512 feature)
// - kernels::dot_neon(NeonToken, ...) — 128-bit f32x4
// - kernels::dot_wasm128(Wasm128Token, ...) — 128-bit f32x4
```

### Type Aliases

Inside `#[magetypes]` modules, these type aliases exist:

| Alias | v3 (x86 AVX2) | v4 (x86 AVX-512) | neon (ARM) | wasm128 (WASM) |
|-------|---------------|------------------|------------|----------------|
| `Token` | `X64V3Token` | `X64V4Token` | `NeonToken` | `Wasm128Token` |
| `f32xN` | `f32x8` | `f32x16` | `f32x4` | `f32x4` |
| `f64xN` | `f64x4` | `f64x8` | `f64x2` | `f64x2` |
| `i8xN` | `i8x32` | `i8x64` | `i8x16` | `i8x16` |
| `i16xN` | `i16x16` | `i16x32` | `i16x8` | `i16x8` |
| `i32xN` | `i32x8` | `i32x16` | `i32x4` | `i32x4` |
| `i64xN` | `i64x4` | `i64x8` | `i64x2` | `i64x2` |
| `u8xN` | `u8x32` | `u8x64` | `u8x16` | `u8x16` |
| `u16xN` | `u16x16` | `u16x32` | `u16x8` | `u16x8` |
| `u32xN` | `u32x8` | `u32x16` | `u32x4` | `u32x4` |
| `u64xN` | `u64x4` | `u64x8` | `u64x2` | `u64x2` |
| `LANES_F32` | `8` | `16` | `4` | `4` |
| `LANES_F64` | `4` | `8` | `2` | `2` |
| `LANES_I8` | `32` | `64` | `16` | `16` |
| ... | ... | ... | ... | ... |

### Auto-Imports

Each variant auto-imports the relevant namespaces:

```rust
// For _v3 variant, macro inserts:
use core::arch::x86_64::*;
use safe_unaligned_simd::x86_64::*;

// For _neon variant:
use core::arch::aarch64::*;
use safe_unaligned_simd::aarch64::*;

// For _wasm128 variant:
use core::arch::wasm32::*;
use safe_unaligned_simd::wasm32::*;
```

This means intrinsics like `_mm256_add_ps` are available without qualification inside the function body.

### Conditional Code

Platform-specific code uses `cfg_if!` or platform constants:

```rust
#[magetypes]
mod kernels {
    pub fn special(token: Token, data: &[f32]) -> f32 {
        if IS_X86 {
            // x86-specific optimization
        } else if IS_ARM {
            // ARM-specific optimization
        } else {
            // Generic path
        }
    }
}
```

Constants provided:
- `IS_X86: bool` — true for `_v3`, `_v4` variants
- `IS_ARM: bool` — true for `_neon` variant
- `IS_WASM: bool` — true for `_wasm128` variant
- `HAS_FMA: bool` — true if FMA available
- `HAS_AVX512: bool` — true for `_v4` variant
- `WIDTH: usize` — SIMD width in bits (128, 256, or 512)

**Limitation:** Platform constants (`IS_X86`, etc.) cannot conditionally remove code branches at the source level. Dead branches are included but will be optimized away by LLVM. For truly platform-specific code paths, use separate functions.

### Compile-Time Feature Elision

**Critical:** `#[magetypes]` MUST use `#[cfg(not(target_feature = "..."))]` to elide unnecessary variants when higher-tier features are compile-time available.

When compiling with `-C target-cpu=haswell` (or equivalent), AVX2+FMA are **compile-time guaranteed**. Lower-tier variants become dead code:

```rust
// Generated by #[magetypes] with proper elision:

// Only compiled when AVX2 is NOT compile-time available
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
pub fn kernel_v2(token: X64V2Token, data: &[f32]) -> f32 { ... }

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
pub fn kernel_scalar(data: &[f32]) -> f32 { ... }

// Always compiled for x86_64 (it's the preferred tier)
#[cfg(target_arch = "x86_64")]
pub fn kernel_v3(token: X64V3Token, data: &[f32]) -> f32 { ... }

// Only compiled with avx512 cargo feature
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub fn kernel_v4(token: X64V4Token, data: &[f32]) -> f32 { ... }

// Only compiled for aarch64
#[cfg(target_arch = "aarch64")]
pub fn kernel_neon(token: NeonToken, data: &[f32]) -> f32 { ... }
```

**Result:**
- `-C target-cpu=haswell`: Only `kernel_v3` compiled (v2, scalar elided)
- `-C target-cpu=skylake-avx512` + `avx512` feature: Only `kernel_v3`, `kernel_v4` compiled
- Generic x86_64 build: All x86 variants compiled, runtime dispatch needed
- aarch64 build: Only `kernel_neon` compiled

**The `incant!` macro must also respect this:**

```rust
// incant! generates:
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
{
    // AVX2 is COMPILE-TIME guaranteed - skip runtime check!
    let token = unsafe { X64V3Token::forge_token_dangerously() };
    return kernel_v3(token, data);
}

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
{
    // Need runtime check
    if let Some(token) = X64V3Token::summon() {
        return kernel_v3(token, data);
    }
    // ... fallback to v2, then scalar
}
```

**See `tests/cfg_elision.rs` for executable verification of this behavior.**

### Why "magetypes"?

The name reflects what it does:
- **mage** — continues the archmage naming theme
- **types** — provides platform-appropriate type aliases
- Avoids confusion with `multiwidth` which implied width-polymorphism (confusing)

`#[multiwidth]` remains as a deprecated alias.

---

## 5. Prelude System

### Problem

Users want to write code that uses the "best" types for their platform without manual `#[cfg]`:

```rust
// Current: verbose cfg blocks
#[cfg(target_arch = "x86_64")]
use magetypes::simd::x86::w256::*;
#[cfg(target_arch = "aarch64")]
use magetypes::simd::arm::w128::*;
```

### Solution: `magetypes::prelude`

```rust
// New: single import
use magetypes::prelude::*;

// Gets the "best" types for the compile target:
// - x86_64: 256-bit types (f32x8, etc.) requiring X64V3Token
// - x86_64 + avx512: 512-bit types (f32x16, etc.) requiring X64V4Token
// - aarch64: 128-bit types (f32x4, etc.) requiring NeonToken
// - wasm32: 128-bit types requiring Wasm128Token
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

#[cfg(all(target_arch = "wasm32", target_feature = "wasm128"))]
pub use crate::simd::wasm::w128::*;

// Also export the recommended token
#[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
pub use archmage::X64V3Token as RecommendedToken;

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub use archmage::X64V4Token as RecommendedToken;

#[cfg(target_arch = "aarch64")]
pub use archmage::NeonToken as RecommendedToken;

#[cfg(all(target_arch = "wasm32", target_feature = "wasm128"))]
pub use archmage::Wasm128Token as RecommendedToken;
```

### No Conflict with `#[magetypes]`

The prelude is for **non-generic code** that targets a single platform. It doesn't conflict with `#[magetypes]` because:

1. `#[magetypes]` generates multiple variants, each with its own type aliases
2. The prelude provides types for direct use outside `#[magetypes]` modules
3. Code inside `#[magetypes]` doesn't use the prelude — it uses the macro's aliases

---

## 6. Safe Cross-Tier Casting via `#[target_feature]`

### Problem

Sometimes you have a lower-tier type (e.g., `f32x4` from SSE) but you're in a context with higher features enabled (e.g., inside an `#[arcane]` function with AVX2). You want to "upgrade" to use the higher-tier type's optimizations.

### Solution: `#[target_feature]`-Gated Methods

As of Rust 1.85, functions with `#[target_feature]` can safely call intrinsics requiring those features. We extend this pattern to cross-tier casting:

```rust
impl x86::w128::f32x4 {
    /// Upgrade to AVX2-context f32x4.
    ///
    /// # Safety
    ///
    /// This method is safe to call inside a function with
    /// `#[target_feature(enable = "avx2", enable = "fma")]` (or `#[arcane]`
    /// with an appropriate token). Outside such a context, calling this
    /// method requires `unsafe`.
    ///
    /// This follows the same safety model as `core::arch` intrinsics in
    /// Rust 1.85+: safe in the right context, unsafe otherwise.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub fn upcast_v3(self) -> x86::w256::f32x4 {
        // Zero-cost transmute - same __m128 representation
        // SAFETY: Both types are #[repr(transparent)] over __m128
        unsafe { core::mem::transmute(self) }
    }

    /// Upgrade to AVX-512-context f32x4.
    #[target_feature(enable = "avx512f", enable = "avx512vl")]
    pub fn upcast_v4(self) -> x86::w512::f32x4 {
        unsafe { core::mem::transmute(self) }
    }
}
```

### Usage Inside `#[arcane]`

```rust
use archmage::{arcane, X64V3Token};
use magetypes::simd::x86::w128::f32x4 as f32x4_sse;
use magetypes::simd::x86::w256::f32x4 as f32x4_avx2;

#[arcane]
fn process(token: X64V3Token, input: f32x4_sse) -> f32x4_avx2 {
    // SAFE: Inside #[arcane] with X64V3Token, AVX2+FMA are enabled
    // The upcast_v3() call is safe in this context (no unsafe block needed)
    input.upcast_v3()
}
```

### Usage Outside `#[arcane]` (Requires Unsafe)

```rust
use magetypes::simd::x86::w128::f32x4 as f32x4_sse;

fn process_unsafe(input: f32x4_sse) {
    // Outside target_feature context, must use unsafe
    // SAFETY: Caller guarantees AVX2+FMA are available
    let upgraded = unsafe { input.upcast_v3() };
}
```

### Why No Token Parameter?

The `#[target_feature]` attribute on the method itself is the proof mechanism:

1. **Inside `#[arcane]`**: The outer function has `#[target_feature]` enabled by the token. Calling another `#[target_feature]` function with the same features is safe (Rust 1.85+).

2. **Outside `#[arcane]`**: The call requires `unsafe`, and the caller takes responsibility for ensuring features are available.

This matches how `core::arch` intrinsics work — no token needed, just the right calling context.

### Downcast (Always Safe)

Downcasting (higher to lower tier) is always safe — no features required:

```rust
impl x86::w256::f32x4 {
    /// Downcast to SSE-context f32x4.
    ///
    /// Always safe: SSE2 is the x86-64 baseline.
    #[inline(always)]
    pub const fn downcast(self) -> x86::w128::f32x4 {
        // SAFETY: Both types are #[repr(transparent)] over __m128
        unsafe { core::mem::transmute(self) }
    }
}
```

### Generated Methods

For each type pair where upcasting is valid, xtask generates:

| From | To | Method | Required Features |
|------|-----|--------|-------------------|
| `w128::f32x4` | `w256::f32x4` | `upcast_v3()` | avx2, fma |
| `w128::f32x4` | `w512::f32x4` | `upcast_v4()` | avx512f, avx512vl |
| `w256::f32x4` | `w512::f32x4` | `upcast_v4()` | avx512f, avx512vl |
| `w256::f32x4` | `w128::f32x4` | `downcast()` | (none) |
| `w512::f32x4` | `w256::f32x4` | `downcast()` | (none) |
| `w512::f32x4` | `w128::f32x4` | `downcast()` | (none) |

### Rust Version Requirement

This safety model requires **Rust 1.85+** (target_feature_11 stabilization). Our MSRV is 1.92, so this is fine.

---

## 7. Implementation Plan

### Phase 1: Foundation (Non-Breaking)

1. **Add `incant!` macro** to `archmage`
2. **Add prelude module** to `magetypes`
3. **Add `Upcast`/`Downcast` traits** to `magetypes`
4. **Update `#[arcane]` docs** with `_self` pattern details
5. **Add `#[magetypes]` as alias** for `#[multiwidth]`

### Phase 2: Migration Period

1. **Deprecate width traits** with warning
2. **Deprecate `#[multiwidth]`** in favor of `#[magetypes]`
3. **Update all examples** to use new patterns
4. **Migration guide** in docs

### Phase 3: Breaking Changes (0.5.0)

1. **Remove width traits** (`Has128BitSimd`, `Has256BitSimd`, `Has512BitSimd`)
2. **Remove `#[multiwidth]`** alias
3. **Update token registry** to remove width trait references

### File Changes

```
archmage/
├── src/
│   ├── lib.rs                    # Add incant! macro export
│   └── macros.rs                 # incant! implementation (new)
│
archmage-macros/
├── src/
│   ├── lib.rs                    # Add #[magetypes], deprecate #[multiwidth]
│   └── magetypes.rs              # New macro implementation
│
magetypes/
├── src/
│   ├── lib.rs                    # Add prelude export
│   ├── prelude.rs                # New prelude module
│   ├── cast.rs                   # Upcast/Downcast traits (new)
│   └── simd/
│       └── ...                   # Generated types (unchanged)
│
token-registry.toml               # Remove width traits (Phase 3)
```

---

## 8. Open Questions

### Q1: Platform Constants in `#[magetypes]`

Should `IS_X86`, `IS_ARM`, etc. be:
- **Const bools** — Simple, but dead branches still compiled (LLVM removes them)
- **cfg_if! blocks** — More complex macro, but truly conditional compilation

**Recommendation:** Const bools. LLVM is excellent at dead code elimination, and the simpler approach is more readable.

### Q2: Auto-Import Filtering

Auto-imports bring in all intrinsics from `core::arch::*`. Generated code uses `#[allow(unused_imports)]` to suppress warnings.

### Q3: Trait Methods and `#[arcane]` — The Generic Token Problem

**Problem:** `#[arcane]` needs features at compile time. `fn process<T: SimdToken>(token: T)` doesn't work because `T` could be any token.

```rust
// BROKEN - can't determine features for generic T
#[arcane]  // What features should this enable? Unknown!
fn process<T: SimdToken>(&self, token: T) -> Self { ... }
```

**Solution: Traits must use concrete tokens or platform-specific trait bounds.**

---

**Pattern 1: Separate methods per platform**

The trait defines one method per platform:

```rust
trait SimdProcessable {
    fn process_v3(&self, token: X64V3Token) -> Self;
    fn process_neon(&self, token: NeonToken) -> Self;
    fn process_scalar(&self) -> Self;
}

impl SimdProcessable for [f32; 8] {
    #[arcane]
    fn process_v3(&self, _token: X64V3Token) -> Self {
        // AVX2+FMA intrinsics work here
    }

    #[arcane]
    fn process_neon(&self, _token: NeonToken) -> Self {
        // NEON intrinsics work here
    }

    fn process_scalar(&self) -> Self {
        // Scalar fallback
    }
}
```

**Pattern 2: Generic method dispatches to concrete helpers**

The trait has a generic signature, but the impl dispatches internally:

```rust
trait SimdProcessable {
    fn process(&self) -> Self;  // No token in signature
}

impl SimdProcessable for [f32; 8] {
    fn process(&self) -> Self {
        incant!(self.process_impl())  // Dispatch to best available
    }
}

// Private helpers with concrete tokens
impl [f32; 8] {
    #[arcane]
    fn process_impl_v3(&self, _token: X64V3Token) -> Self { ... }

    #[arcane]
    fn process_impl_neon(&self, _token: NeonToken) -> Self { ... }

    fn process_impl_scalar(&self) -> Self { ... }
}
```

**Pattern 3: Use `#[magetypes]` to generate multiple traits**

`#[magetypes]` can generate platform-specific trait impls:

```rust
/// Trait generated per-platform by #[magetypes]
trait SimdKernel {
    type Token: SimdToken;
    fn kernel(&self, token: Self::Token) -> Self;
}

#[magetypes]
mod impls {
    // Inside #[magetypes], Token is concrete (X64V3Token, NeonToken, etc.)
    impl SimdKernel for [f32; LANES_F32] {
        type Token = Token;  // Concrete per variant

        fn kernel(&self, token: Token) -> Self {
            let v = f32xN::load(token, self);
            (v * v).to_array()
        }
    }
}

// Generated:
// impl SimdKernel for [f32; 8] { type Token = X64V3Token; ... }
// impl SimdKernel for [f32; 4] { type Token = NeonToken; ... }
// etc.
```

**Pattern 4: Platform-specific trait bounds (limited use)**

If dropping width traits, you can still use feature-level traits:

```rust
trait X86SimdOps {
    fn process(&self, token: impl HasX64V2) -> Self;
}

impl X86SimdOps for [f32; 8] {
    #[arcane]
    fn process(&self, _token: impl HasX64V2) -> Self {
        // SSE4.2 features enabled (from HasX64V2)
        // But NOT AVX2/FMA! HasX64V2 only gives SSE-level features.
    }
}
```

**⚠️ Warning:** `impl HasX64V2` only enables SSE4.2-level features. For AVX2+FMA, you MUST use `X64V3Token` directly — there's no `HasX64V3` trait (intentionally, to push users toward concrete tokens).

---

**Summary: Why generic tokens don't work with `#[arcane]`**

| Signature | Works with `#[arcane]`? | Why |
|-----------|-------------------------|-----|
| `token: X64V3Token` | ✅ Yes | Concrete type → known features |
| `token: impl HasX64V2` | ✅ Yes | Trait → known feature set |
| `token: impl HasX64V4` | ✅ Yes | Trait → known feature set |
| `token: impl HasNeon` | ✅ Yes | Trait → known feature set |
| `token: impl SimdToken` | ❌ No | Base trait → no specific features |
| `token: T where T: SimdToken` | ❌ No | Generic → features unknown at compile time |

**The rule:** `#[arcane]` requires compile-time-known features. Use concrete tokens or feature-level trait bounds, not `SimdToken` directly.

---

## 9. Idiomatic Patterns Reference

**See `tests/idiomatic_patterns.rs` for executable examples of every pattern.**

All patterns are tested and verified to compile on x86_64, aarch64, and wasm32.

### Quick Reference Table

| Pattern | Syntax | Works? | Recommended? | Notes |
|---------|--------|--------|--------------|-------|
| Concrete token | `fn f(t: X64V3Token)` | ✅ | ✅ | Best for most code |
| Friendly alias | `fn f(t: Desktop64)` | ✅ | ✅ | Same as X64V3Token |
| Feature trait | `fn f(t: impl HasX64V2)` | ✅ | ⚠️ | Only SSE4.2, not AVX2 |
| Feature trait | `fn f(t: impl HasX64V4)` | ✅ | ✅ | For AVX-512 code |
| Width trait | `fn f(t: impl Has256BitSimd)` | ⚠️ | ❌ | **DEPRECATED** |
| Generic token | `fn f<T: SimdToken>(t: T)` | ❌ | ❌ | **Cannot work** |
| `_self` pattern | `#[arcane(_self = T)]` | ✅ | ✅ | For trait methods |
| Token passthrough | Call nested `#[arcane]` | ✅ | ✅ | Natural composition |
| Token extraction | `token.v3()`, `token.v2()` | ✅ | ✅ | For fallback paths |

### Pattern Categories

1. **Basic usage** — Concrete tokens, friendly aliases
2. **Feature bounds** — `impl HasX64V2`, `impl HasX64V4`, `impl HasNeon`
3. **Trait methods** — `_self` pattern with token as 2nd parameter
4. **Composition** — Token passthrough, nested calls
5. **Dispatch** — Manual platform dispatch (until `incant!` is implemented)
6. **magetypes** — Using high-level SIMD types

---

## 10. Code Examples

### Example 1: Complete Workflow

```rust
use archmage::{arcane, incant, X64V3Token, NeonToken, SimdToken};
use magetypes::prelude::*;  // Gets platform-appropriate types

// Define implementations for each platform
#[arcane]
fn sum_v3(token: X64V3Token, data: &[f32]) -> f32 {
    let mut acc = f32x8::zero(token);
    for chunk in data.chunks_exact(8) {
        acc = acc + f32x8::load(token, chunk.try_into().unwrap());
    }
    acc.reduce_add()
}

#[arcane]
fn sum_neon(token: NeonToken, data: &[f32]) -> f32 {
    let mut acc = f32x4::zero(token);
    for chunk in data.chunks_exact(4) {
        acc = acc + f32x4::load(token, chunk.try_into().unwrap());
    }
    acc.reduce_add()
}

fn sum_scalar(data: &[f32]) -> f32 {
    data.iter().sum()
}

// Dispatch automatically
pub fn sum(data: &[f32]) -> f32 {
    incant!(sum(data))
}
```

### Example 2: Using `#[magetypes]`

```rust
use archmage::magetypes;

#[magetypes]
mod kernels {
    /// Dot product — generates _v3, _v4, _neon, _wasm128 variants
    pub fn dot(token: Token, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut sum = f32xN::zero(token);
        let chunks = a.len() / LANES_F32;

        for i in 0..chunks {
            let start = i * LANES_F32;
            let va = f32xN::load(token, &a[start..][..LANES_F32].try_into().unwrap());
            let vb = f32xN::load(token, &b[start..][..LANES_F32].try_into().unwrap());
            sum = va.mul_add(vb, sum);  // FMA
        }

        // Remainder handled by scalar
        let mut result = sum.reduce_add();
        for i in (chunks * LANES_F32)..a.len() {
            result += a[i] * b[i];
        }
        result
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    incant!(kernels::dot(a, b))
}
```

### Example 3: Trait Implementation with Concrete Tokens

```rust
use archmage::{arcane, X64V3Token, NeonToken};

/// Trait with platform-specific methods
trait FastOps {
    fn double_v3(&self, token: X64V3Token) -> Self;
    fn double_neon(&self, token: NeonToken) -> Self;
    fn double_scalar(&self) -> Self;
}

impl FastOps for [f32; 8] {
    #[arcane(_self = [f32; 8])]
    fn double_v3(&self, _token: X64V3Token) -> Self {
        // _self is &[f32; 8] here
        // AVX2+FMA intrinsics available
        let v = _mm256_loadu_ps(_self.as_ptr());
        let doubled = _mm256_add_ps(v, v);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), doubled);
        out
    }

    #[arcane(_self = [f32; 8])]
    fn double_neon(&self, _token: NeonToken) -> Self {
        // NEON implementation (would use different array size in practice)
        todo!()
    }

    fn double_scalar(&self) -> Self {
        let mut out = [0.0f32; 8];
        for i in 0..8 {
            out[i] = _self[i] * 2.0;
        }
        out
    }
}

// Convenience free function using incant!
pub fn double_f32x8(data: &[f32; 8]) -> [f32; 8] {
    incant!(double_f32x8(data))
}

// Where the suffixed functions are:
#[arcane]
fn double_f32x8_v3(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    data.double_v3(token)
}

#[arcane]
fn double_f32x8_neon(token: NeonToken, data: &[f32; 8]) -> [f32; 8] {
    data.double_neon(token)
}

fn double_f32x8_scalar(data: &[f32; 8]) -> [f32; 8] {
    data.double_scalar()
}
```

---

## 11. Token Renaming: `Simd128Token` → `Wasm128Token`

The WASM token is renamed throughout the codebase for clarity:

| Old Name | New Name | Rationale |
|----------|----------|-----------|
| `Simd128Token` | `Wasm128Token` | "simd128" is confusing — all SIMD is "simd". "wasm128" clearly identifies the platform. |
| `_simd128` suffix | `_wasm128` suffix | Consistent with token rename |

**Note:** Rust's `#[cfg(target_feature = "simd128")]` attribute remains unchanged — that's Rust's name for the WASM SIMD feature, not ours to change. Only our token and function suffix names change.

This requires updating:
- `token-registry.toml`
- All generated code
- Documentation
- Examples
- The `incant!` macro

---

## 12. Summary

| Feature | Status | Breaking? |
|---------|--------|-----------|
| Drop width traits | Phase 3 | Yes |
| `incant!` macro | Phase 1 | No |
| `_self` documentation | Phase 1 | No |
| `#[magetypes]` macro | Phase 1 | No |
| Prelude system | Phase 1 | No |
| `upcast_*`/`downcast` methods | Phase 1 | No |
| Rename `Simd128Token` → `Wasm128Token` | Phase 2 | Yes (minor) |
| Remove `#[multiwidth]` | Phase 3 | Yes |

The design maintains backward compatibility through Phase 1 and 2, with breaking changes deferred to a major version bump.

+++
title = "#[magetypes] Macro"
weight = 3
+++

`#[magetypes]` generates platform-specific function variants by replacing `Token` with each concrete token type. It works with `incant!` to provide both generation and dispatch.

> **Note:** This macro is part of archmage (the `macros` feature), not the [magetypes crate](/magetypes/). The name reflects that it generates code in the style that magetypes types use. Magetypes itself is an exploratory companion crate — its API may change between releases.

## When to Use

Use `#[magetypes]` for functions that don't use platform-specific SIMD types — where `Token` is the only platform-dependent part. If your function body uses `f32x8` or `__m256`, write the variants manually and use `incant!` directly.

## Basic Usage

```rust
use archmage::magetypes;

#[magetypes]
fn process(token: Token, data: &[f32]) -> f32 {
    // Token is replaced with X64V3Token, NeonToken, ScalarToken, etc.
    // Generates: process_v3, process_neon, process_scalar, ...
    data.iter().sum()
}
```

The macro generates one function per platform variant, each with the concrete token type substituted in:

| Generated Function | Token Type |
|-------------------|------------|
| `process_v3` | `X64V3Token` |
| `process_v4` | `X64V4Token` |
| `process_neon` | `NeonToken` |
| `process_wasm128` | `Wasm128Token` |
| `process_scalar` | `ScalarToken` |

These suffixed functions are exactly what `incant!` expects:

```rust
pub fn process(data: &[f32]) -> f32 {
    incant!(process(data), [v3, neon, wasm128, scalar])
}
```

## What Gets Replaced

`#[magetypes]` replaces placeholders in each generated variant. `Token` uses token-level replacement (it understands token boundaries, so `ScalarToken` and `IntoConcreteToken` pass through unmodified). Other placeholders use text substitution on the token stream.

| Placeholder | Replaced With |
|-------------|---------------|
| `Token` | Concrete token type (`X64V3Token`, `NeonToken`, etc.) |
| `f32xN` | Platform-native f32 vector (`f32x8`, `f32x4`, etc.) |
| `LANES` | Lane count for the platform (`8`, `4`, etc.) |

Case-sensitive — `Token` is replaced, `token` is not.

## Tier Lists

`#[magetypes]` accepts the same tier list syntax as `incant!` and `#[autoversion]`. By default it generates variants for v3, v4, neon, wasm128, and scalar.

**Explicit tiers:**

```rust
#[magetypes(v3, neon, scalar)]
fn process(token: Token, data: &[f32]) -> f32 { ... }
```

**Feature-gated tiers:**

```rust
#[magetypes(v4(cfg(avx512)), v3, neon, scalar)]
fn process(token: Token, data: &[f32]) -> f32 { ... }
```

The shorthand `v4(avx512)` also works.

**Tier list modifiers:**

```rust
#[magetypes(+arm_v2)]
fn process(token: Token, data: &[f32]) -> f32 { ... }

#[magetypes(-wasm128, +v1)]
fn process(token: Token, data: &[f32]) -> f32 { ... }
```

Tier names accept the `_` prefix — `_v3` is identical to `v3`.

## The `define(...)` Flag: Inject Magetypes Type Aliases

Without `define`, idiomatic `#[magetypes]` bodies usually start with a boilerplate alias line per magetypes type used:

```rust
use magetypes::simd::generic::f32x8 as GenericF32x8;

#[magetypes(v3, scalar)]
fn scale(token: Token, plane: &mut [f32], factor: f32) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;  // ← boilerplate
    // ...
}
```

`define(...)` takes a list of magetypes type names and injects those alias lines automatically at the top of each variant body:

```rust
#[magetypes(define(f32x8), v3, scalar)]
fn scale(token: Token, plane: &mut [f32], factor: f32) {
    // `f32x8` is already in scope as the matching-tier concrete type.
    let factor_v = f32x8::splat(token, factor);
    let (chunks, tail) = f32x8::partition_slice_mut(token, plane);
    for chunk in chunks {
        (f32x8::load(token, chunk) * factor_v).store(chunk);
    }
    for v in tail { *v *= factor; }
}
```

Multiple types: `define(f32x8, f32x4, u8x16, i16x8)`. Type names must match the magetypes generic types (`f32xN`, `fNxM`, `iNxM`, `uNxM`); typos surface as rustc resolution errors.

Injected alias:

```rust
type <name> = ::magetypes::simd::generic::<name><Token>;
```

`Token` in the RHS is substituted per tier (same as the rest of the body), so each variant gets a correctly-typed alias. The aliases are function-local — they don't leak into outer scope and shadow any outer `f32x8` / etc. within the function body.

Empty list `define()` is accepted as a no-op, so commenting out items doesn't produce a syntax error.

## The `rite` Flag: Direct `#[target_feature]` Variants

By default, non-fallback variants are wrapped with `#[archmage::arcane]` — each variant becomes a safe outer function + `#[target_feature]` inner sibling (the "trampoline" pattern). This lets the variant be called from any context, including non-target-feature dispatchers.

`#[magetypes(rite, ...)]` changes the per-tier wrapping to `#[archmage::rite(import_intrinsics)]` — each variant gets `#[target_feature]` + `#[inline]` applied directly, with no sibling and no trampoline:

```rust
#[magetypes(rite, v3, scalar)]
fn helper(_t: Token, x: f32) -> f32 {
    x * x
}

// Expands to (x86_64):
//   #[target_feature(enable = "avx2,fma,...")] #[inline]
//   fn helper_v3(_t: archmage::X64V3Token, x: f32) -> f32 { x * x }
//   #[inline]
//   fn helper_scalar(_t: archmage::ScalarToken, x: f32) -> f32 { x * x }
```

**When to use `rite`:**
- Inner helpers called from matching-feature contexts — another `#[arcane]` / `#[rite]` / arcane-flavored `#[magetypes]` body at the same tier. Rust 1.86+ allows safe calls between `#[target_feature]` functions when the caller's features ⊇ callee's, so a rite-flavored helper inlines into the caller's region with zero optimization boundary.

**When NOT to use `rite`:**
- Public API entry points dispatched via standalone `incant!`. The dispatcher has no `#[target_feature]`, so calling a bare `#[target_feature]` variant requires `unsafe` — which `incant!`'s current dispatcher does not emit. Use arcane-flavored `#[magetypes]` (default) at public boundaries, and reserve the `rite` flag for internal helpers.

**Token substitution is unchanged** — the `rite` flag only changes the per-tier wrapping, not the `Token` → concrete-type replacement.

## `#[magetypes]` vs Manual Variants

**Use `#[magetypes]`** when the function body is platform-independent (only the token type changes):

```rust
#[magetypes]
fn validate(token: Token, threshold: f32) -> bool {
    // Token is the only platform-dependent part
    threshold > 0.0
}
```

**Write manual variants** when the function body uses platform-specific types or different algorithms per platform:

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::{F32x8Backend, x64v3},
};

#[cfg(target_arch = "x86_64")]
fn process_v3(token: X64V3Token, data: &[f32]) -> f32 {
    // Uses f32x8::<x64v3> — 8-wide AVX2 algorithm
    let v = f32x8::<x64v3>::from_array(token, data[..8].try_into().unwrap());
    v.reduce_add()
}

fn process_scalar(_token: ScalarToken, data: &[f32]) -> f32 {
    // Completely different algorithm
    data.iter().sum()
}
```

Or use the **generic pattern** when the same algorithm works across backends:

```rust
#[inline(always)]
fn process_generic<T: F32x8Backend>(token: T, data: &[f32]) -> f32 {
    let v = f32x8::<T>::from_array(token, data[..8].try_into().unwrap());
    v.reduce_add()
}
```

In practice, SIMD functions almost always need different types and algorithms per platform, so `incant!` with manual variants is the more common pattern. But for algorithms that only differ by backend token, the generic pattern avoids code duplication.

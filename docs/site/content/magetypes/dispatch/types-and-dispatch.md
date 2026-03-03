+++
title = "Types and Dispatch"
weight = 1
+++

Magetypes vectors work with archmage's [`incant!`](@/archmage/dispatch/incant.md) macro for multi-platform dispatch. The generic backend pattern lets you write one function body that works across all architectures — `incant!` selects the best available token at runtime.

## The Generic Pattern

Write your SIMD logic once with a generic backend bound, then wire it up with concrete `#[arcane]` wrappers and `incant!`:

```rust
use archmage::{arcane, incant, ScalarToken, X64V3Token};
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Backend,
};

// 1. Generic function — no #[arcane], just #[inline(always)]
//    This has no #[target_feature] of its own. It inherits the caller's
//    features through inlining — without inlining, intrinsics become
//    function calls (18x slower). Always use #[inline(always)].
#[inline(always)]
fn sum_impl<T: F32x8Backend>(token: T, data: &[f32; 8]) -> f32 {
    f32x8::<T>::from_array(token, *data).reduce_add()
}

// 2. Concrete #[arcane] wrappers — one per tier
//    #[arcane] needs a concrete token to know which #[target_feature] to emit.
//    Generic bounds like F32x8Backend are unknown to it.
#[arcane]
fn sum_impl_v3(token: X64V3Token, data: &[f32; 8]) -> f32 {
    sum_impl(token, data)  // generic inlines here, gets AVX2+FMA
}

fn sum_impl_scalar(token: ScalarToken, data: &[f32; 8]) -> f32 {
    sum_impl(token, data)
}

// 3. incant! dispatches to the best available variant at runtime
pub fn sum(data: &[f32; 8]) -> f32 {
    incant!(sum_impl(data), [v3])
}
```

Why three layers? `#[arcane]` parses the token type from your function signature to choose which `#[target_feature]` attributes to emit. It only recognizes concrete tokens (`X64V3Token`, `NeonToken`, etc.) — not generic bounds like `F32x8Backend`. The generic function carries the algorithm; the `#[arcane]` wrappers provide the target-feature context; `incant!` picks the right wrapper at runtime.

**Polyfills are automatic.** `f32x8::<NeonToken>` compiles to two 128-bit NEON operations. `f32x8::<Wasm128Token>` compiles to two 128-bit SIMD128 operations. The same generic function works everywhere with zero code changes — on AVX2 it's one 256-bit operation, on narrower hardware it's polyfilled.

A complete cross-platform version adds NEON and WASM wrappers:

```rust
use archmage::{arcane, incant, ScalarToken, X64V3Token, NeonToken, Wasm128Token};
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Backend,
};

#[inline(always)]
fn sum_impl<T: F32x8Backend>(token: T, data: &[f32; 8]) -> f32 {
    f32x8::<T>::from_array(token, *data).reduce_add()
}

#[arcane]
fn sum_impl_v3(token: X64V3Token, data: &[f32; 8]) -> f32 {
    sum_impl(token, data)
}

#[arcane]
fn sum_impl_neon(token: NeonToken, data: &[f32; 8]) -> f32 {
    sum_impl(token, data)  // polyfilled: two f32x4 ops
}

#[arcane]
fn sum_impl_wasm128(token: Wasm128Token, data: &[f32; 8]) -> f32 {
    sum_impl(token, data)  // polyfilled: two f32x4 ops
}

fn sum_impl_scalar(token: ScalarToken, data: &[f32; 8]) -> f32 {
    sum_impl(token, data)
}

pub fn sum(data: &[f32; 8]) -> f32 {
    incant!(sum_impl(data), [v3, neon, wasm128])
}
```

## When Algorithms Differ Per Platform

Sometimes you want separate implementations to exploit architecture-specific strengths — different register widths, native instruction sequences, or algorithm shapes:

```rust
use archmage::{arcane, incant, ScalarToken, X64V3Token, NeonToken};
use magetypes::simd::generic::{f32x8, f32x4};

// x86-64: use f32x8 (256-bit AVX2)
#[arcane]
fn dot_product_v3(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let va = f32x8::<X64V3Token>::from_array(token, *a);
    let vb = f32x8::<X64V3Token>::from_array(token, *b);
    (va * vb).reduce_add()
}

// AArch64: use f32x4 (128-bit NEON) — process in two halves
#[arcane]
fn dot_product_neon(token: NeonToken, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let sum1 = {
        let va = f32x4::<NeonToken>::from_slice(token, &a[0..4]);
        let vb = f32x4::<NeonToken>::from_slice(token, &b[0..4]);
        (va * vb).reduce_add()
    };
    let sum2 = {
        let va = f32x4::<NeonToken>::from_slice(token, &a[4..8]);
        let vb = f32x4::<NeonToken>::from_slice(token, &b[4..8]);
        (va * vb).reduce_add()
    };
    sum1 + sum2
}

fn dot_product_scalar(_token: ScalarToken, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn dot_product(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    incant!(dot_product(a, b), [v3, neon])
}
```

When the algorithm is the same on every platform, the generic pattern from the previous section is cleaner. When you need platform-tuned implementations, write concrete variants and list the tiers explicitly. Scalar is always implicit — `incant!` falls back to it automatically.

## The #[magetypes] Macro

When the function body is truly platform-independent (only the token type changes), use [`#[magetypes]`](@/archmage/dispatch/magetypes-macro.md) to generate all variants automatically:

```rust
use archmage::{magetypes, incant};

#[magetypes]
fn validate(token: Token, threshold: f32) -> bool {
    // Token is replaced with X64V3Token, NeonToken, ScalarToken, etc.
    threshold > 0.0
}

// Generates: validate_v3, validate_neon, validate_wasm128, validate_scalar
// Ready for incant!:
pub fn validate(threshold: f32) -> bool {
    incant!(validate(threshold), [v3, neon, wasm128])
}
```

`#[magetypes]` does text substitution — `Token` becomes the concrete token type for each variant, and `#[arcane]` is applied to each generated function. This works when the function body doesn't depend on SIMD types that vary by backend.

**Limitation:** `#[magetypes]` generates variants for all tiers, including `v4`. If your function uses `f32x8::<Token>`, the `v4` variant will fail because `X64V4Token` doesn't implement `F32x8Backend` (it uses `f32x16`). For generic SIMD code, use the three-layer pattern from the first section instead.

## Passthrough Dispatch

When you already have a token and want to dispatch to specialized variants without re-summoning:

```rust
use archmage::{incant, IntoConcreteToken};

fn process_inner<T: IntoConcreteToken>(token: T, data: &[f32]) -> f32 {
    incant!(compute(data) with token, [v3, neon, wasm128])
    // Uses IntoConcreteToken to check what the token actually is
}
```

See [`incant!` Passthrough Mode](@/archmage/dispatch/incant.md) for details.

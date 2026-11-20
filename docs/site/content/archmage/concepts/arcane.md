+++
title = "Proof Wrappers (#[attune(wrap)])"
description = "Token-authenticated #[target_feature] entry points with #[attune(wrap)] and _t suffixes (replaces #[arcane])"
weight = 3
+++

`#[attune(wrap)]` (or `#[attune] fn name_<tier>_t`) replaces `#[arcane]`, which
is superseded and queued for deprecation in the next release.

A proof wrapper creates a safe entry into a function compiled with the CPU
features proved by its token parameter. For portable vector kernels across
tiers, use [`#[attune(_*_t, define(...))]` or `#[attune(dispatch, define(...))]`](@/archmage/dispatch/magetypes-macro.md),
which generate these entries automatically. Use `#[attune(wrap)]` or a
`_<tier>_t` suffix when writing a single ISA-specific entry point.

Start with [the complete portable kernel](@/archmage/getting-started/first-simd.md)
or [direct-intrinsic helpers](@/archmage/concepts/rite.md). Neither needs an
explicit `unsafe` block.

```rust
#![forbid(unsafe_code)]
use archmage::prelude::*;

// Explicit wrap: keeps any function name and derives features from the token.
#[attune(wrap, import_intrinsics)]
fn scale_array(_token: X64V3Token, data: &mut [f32; 8], factor: f32) {
    let v = _mm256_loadu_ps(data);
    _mm256_storeu_ps(data, _mm256_mul_ps(v, _mm256_set1_ps(factor)));
}

// Suffix inference: _v3_t selects a V3 proof wrapper without spelling wrap.
#[attune(import_intrinsics)]
fn add_bias_v3_t(_token: X64V3Token, data: &mut [f32; 8], bias: f32) {
    let v = _mm256_loadu_ps(data);
    _mm256_storeu_ps(data, _mm256_add_ps(v, _mm256_set1_ps(bias)));
}

#[cfg(target_arch = "x86_64")]
if let Some(token) = X64V3Token::summon() {
    let mut data = [2.0f32; 8];
    scale_array(token, &mut data, 3.0);
    add_bias_v3_t(token, &mut data, 1.0);
    assert_eq!(data, [7.0; 8]);
}
```

A token parameter alone does not request a wrapper: bare
`#[attune] fn scale(token: X64V3Token)` without `wrap` or `_v3_t` is a compile
error. Pass `wrap` to keep an unsuffixed name, or end the name in `_<tier>_t`
(which also inserts the tier's concrete token parameter when omitted).

## What the macro establishes

The wrapper authenticates the token, then calls a private `#[target_feature]`
function (`__arcane_*`). Your body can safely call value intrinsics whose
requirements are covered. Safe memory wrappers still enforce Rust reference
size and validity; a CPU token does not justify arbitrary pointer loads,
unchecked indices, or aliasing mutable references.

| Signature or option | Meaning |
|---|---|
| `#[attune(wrap)]` with concrete token (`X64V3Token`) | Keeps function name; uses that token's feature set |
| `#[attune] fn name_<tier>_t(...)` | Infers tier and proof wrapper from the `_<tier>_t` suffix |
| `#[attune(wrap)]` with tier bound (`T: HasX64V2`) | Uses the declared bound's feature set, not a stronger caller's extra features |
| `import_intrinsics` | Imports `archmage::intrinsics::{arch}::*` (including safe reference-based memory ops) |
| `in_impl` | Receiverless associated function in an inherent `impl`: wrapper calls `Self::` |
| `in_trait` (alias `nested`) | Receiverless method in a trait `impl`: feature body nests inside the method |
| `_self = ConcreteType` | Trait `impl` method with a receiver (implies `in_trait`); rewrites `self` and `Self` |
| `suppress_const_test` | Omits the token-name mismatch const check (for trusted code generators) |

Ordinary expansions emit a sibling feature function and an `#[inline(always)]`
outer wrapper. Rust generics and `where` clauses are forwarded. Methods follow
the receiver rules in [methods](@/archmage/advanced/methods.md): inherent
`&self` / `&mut self` methods work directly, receiverless inherent functions
need `in_impl`, and trait `impl` methods use `_self = ConcreteType` (or
`in_trait` without a receiver).

Wrong-architecture definitions are omitted automatically; do not place manual
`#[cfg(target_arch = ...)]` on `#[attune]` definitions. `attuned!` handles
call-site `cfg` guards; [manual callers](@/archmage/dispatch/manual.md) must
gate direct calls to architecture-specific functions (`scale_array`,
`add_bias_v3_t`) explicitly, because a runtime `summon()` check does not remove
an absent symbol from name resolution on other targets.

For internal helpers called inside an attuned body, use direct
[`#[attune(v3)]` or `#[attune(_*)]` functions](@/archmage/concepts/rite.md) or
inline backend-generic helpers. Enter around the whole row, strip, or batch
rather than once per vector; see
[target-feature boundaries](@/archmage/concepts/target-feature-boundaries.md).

## Legacy `#[arcane]` spelling (queued for deprecation)

| Legacy spelling | `#[attune]` replacement | Notes |
|---|---|---|
| `#[arcane] fn kernel(token: X64V3Token, x: f32)` | `#[attune(wrap)] fn kernel(token: X64V3Token, x: f32)` | Identical name, signature, and generated wrapper |
| `#[arcane] fn kernel_v3(token: X64V3Token, x: f32)` | `#[attune] fn kernel_v3_t(token: X64V3Token, x: f32)` | `_v3_t` convention integrates directly with `attuned!(kernel(x), [_v3, _scalar])` |
| `#[arcane(import_intrinsics)]` | `#[attune(wrap, import_intrinsics)]` | Same intrinsic imports; also works on `_<tier>_t` functions |
| `#[arcane(in_impl)]` | `#[attune(wrap, in_impl)]` | Receiverless associated function in an inherent `impl` |
| `#[arcane(in_trait, _self = Type)]` | `#[attune(wrap, _self = Type)]` | `_self = Type` implies `in_trait`; use `#[attune(wrap, in_trait)]` when receiverless |

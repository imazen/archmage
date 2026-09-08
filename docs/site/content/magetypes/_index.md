+++
title = "Magetypes"
description = "SIMD vector types with natural Rust operators"
sort_by = "weight"
weight = 2

[extra]
sidebar = true
+++

Process an image plane (exposure) or an audio buffer (gain), including a short
scalar tail. The vector type is generic over the token selected by `#[magetypes]`;
`incant!` chooses the CPU tier once outside the loop. No manual per-tier wrappers
or raw pointers are needed.

Adapted from the `zenfilters` plane-scaling kernel; the [complete production call chain and adaptation notes](https://imazen.github.io/archmage/magetypes/examples/generic-kernels/) include pinned source links.

```rust
#![forbid(unsafe_code)]
use archmage::prelude::*;

#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn gain_impl(token: Token, plane: &mut [f32], gain: f32) {
    let factor = f32x8::splat(token, gain);
    let (chunks, tail) = f32x8::partition_slice_mut(token, plane);
    for chunk in chunks {
        (f32x8::load(token, chunk) * factor).store(chunk);
    }
    for value in tail {
        *value *= gain;
    }
}

pub fn apply_gain(plane: &mut [f32], gain: f32) {
    incant!(gain_impl(plane, gain), [v3, neon, wasm128, scalar])
}
```

See [reusable generic kernels](@/magetypes/examples/generic-kernels.md) for helper bounds and dispatch.

Magetypes provides SIMD vector types — `f32x8`, `i32x4`, `u8x16`, and friends — with natural Rust operators. Instead of writing raw intrinsics, you write `a + b`, `v * v`, `x.reduce_add()`.

Tested across x86-64, AArch64, WASM SIMD128, and the scalar fallback.

See [ISA Quirks and Fixups](@/magetypes/isa-quirks.md) for concrete edge-case results, portability limits, and the fixups we apply.

## Relationship to Archmage

Magetypes depends on [archmage](@/archmage/_index.md) for capability tokens. You cannot construct a magetypes vector without first proving that the CPU supports the required features — this is what "token-gated construction" means.

The types are generic over a backend token `T`. A generic helper may use
`#[inline(always)]`, but its caller must establish the target-feature context
for efficient inlining. The complete [SrcOver example](@/magetypes/examples/pixel-blending.md)
shows the public slice API, `incant!` dispatch, `#[magetypes]` entry, and generic
helper together. A token alone does not enable target features on the caller.

Every constructor (`from_array`, `splat`, `zero`, `load`) takes a token as its
first argument. Safe construction establishes the vector's capability invariant.

## Choosing an entry point and helpers

### When to use which

| Approach | Generates | Dispatch | Use when |
|---|---|---|---|
| `#[magetypes(tiers)]` | Suffixed variants via `Token` substitution | Manual `incant!` | Explicit SIMD types (`f32x8`, `i32x4`) |
| `#[rite(tiers)]` | Suffixed variants with `#[target_feature]` + `#[inline]` | Called from `#[arcane]` context | Inner helpers that need platform features but no dispatch |
| `#[arcane]` | One function with safe target-feature wrapper | Manual `incant!` | Hand-tuned intrinsics for a single tier |
| `#[autoversion]` | Suffixed variants + dispatcher | Built-in | Scalar loops that LLVM auto-vectorizes well |
| Mixed | Combine any of the above | One `incant!` handles all | Most tiers generic, one or two hand-tuned |

### Attribute parameter reference

| Parameter | `#[arcane]` | `#[rite]` | `#[magetypes]` | `#[autoversion]` |
|---|---|---|---|---|
| Tier names (`v3`, `neon`, ...) | — | **Yes** (suffixed variants) | **Yes** (suffixed variants) | **Yes** (suffixed + dispatcher) |
| `+tier` / `-tier` modifiers | — | — | **Yes** | **Yes** |
| `tier(cfg(feature))` gate | — | — | **Yes** | **Yes** |
| `import_intrinsics` | **Yes** | **Yes** | auto | auto |
| `import_magetypes` | **Yes** | **Yes** | auto | auto |
| `cfg(feature)` | **Yes** | **Yes** | — | **Yes** |
| `_self = Type` | **Yes** | — | — | **Yes** |
| `nested` | **Yes** | — | — | — |
| `inline_always` | **Yes** (nightly) | — | — | — |

**Tier suffixes:** `_v1`, `_v2`, `_x64_crypto`, `_v3`, `_v3_crypto`, `_v4`, `_v4x`, `_neon`, `_neon_aes`, `_neon_sha3`, `_neon_crc`, `_arm_v2`, `_arm_v3`, `_wasm128`, `_wasm128_relaxed`, `_scalar`, `_default`.

**Default tiers** (when no list given): `v4(avx512)`, `v3`, `neon`, `wasm128`, `scalar`.

See [Real-World Examples](@/magetypes/examples/_index.md) for complete runnable adaptations and production source walkthroughs.

## Cross-Platform Polyfills

Types wider than the hardware's native register width work everywhere via polyfills. An `f32x8` on AArch64 (which has 128-bit NEON registers) is implemented internally as two `f32x4` operations. The API is identical — you pick the size that fits your algorithm, and magetypes handles the rest. See [Polyfills](@/magetypes/cross-platform/polyfills.md) for details.

## What's Here

- [Getting Started](@/magetypes/getting-started/_index.md) — Installation and your first types
- [Types](@/magetypes/types/_index.md) — Available types per platform, properties, feature flags
- [Operations](@/magetypes/operations/_index.md) — Construction, arithmetic, reductions, bitwise
- [Conversions](@/magetypes/conversions/_index.md) — Float/int, width, bitcast, slice casting
- [Math](@/magetypes/math/_index.md) — Transcendentals, precision levels, approximations
- [Memory](@/magetypes/memory/_index.md) — Load/store, gather/scatter, interleaved data, chunked processing
- [Cross-Platform](@/magetypes/cross-platform/_index.md) — Polyfill strategy, known behavioral differences
- [Dispatch](@/magetypes/dispatch/_index.md) — Using magetypes with `incant!` and `#[magetypes]`
- [Real-World Examples](@/magetypes/examples/_index.md) — Production patterns from image codecs: plane ops, blending, convolution, quantization, blur, color conversion, byte transforms

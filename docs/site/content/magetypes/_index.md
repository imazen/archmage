+++
title = "Magetypes"
description = "Experimental SIMD vector types with natural Rust operators"
sort_by = "weight"
weight = 2

[extra]
sidebar = true
+++

Magetypes provides SIMD vector types — `f32x8`, `i32x4`, `u8x16`, and friends — with natural Rust operators. Instead of writing raw intrinsics, you write `a + b`, `v * v`, `x.reduce_add()`.

**Status: Experimental.** The API is usable and tested across x86-64, AArch64, and WASM, but it may change between minor versions. Pin your dependency version if stability matters.

## Relationship to Archmage

Magetypes depends on [archmage](@/archmage/_index.md) for capability tokens. You cannot construct a magetypes vector without first proving that the CPU supports the required features — this is what "token-gated construction" means.

The types are generic over a backend token `T`. Write one function, get a correct implementation on every architecture:

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Backend,
};

#[inline(always)]
fn dot_product<T: F32x8Backend>(token: T, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let va = f32x8::<T>::load(token, a);
    let vb = f32x8::<T>::load(token, b);

    // Natural operators — no intrinsics, no unsafe
    let product = va * vb;
    product.reduce_add()
}
```

To call this from concrete code, summon a token and pass it in:

```rust
use archmage::{X64V3Token, SimdToken};

fn main() {
    // Prove CPU supports AVX2+FMA — returns None on unsupported hardware
    if let Some(token) = X64V3Token::summon() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];
        let b = [2.0f32; 8];
        let result = dot_product(token, &a, &b);
        println!("dot: {}", result);  // 72.0
    }
}
```

Every constructor (`from_array`, `splat`, `zero`, `load`) takes a token as its first argument. If you have the vector, the CPU can run the operations on it.

## Mixed Dispatch: Generic + Specialized + Auto-Vectorized

In practice, most tiers share the same algorithm while one or two need hand-tuned intrinsics. Four macros generate suffixed `_v3`/`_neon`/`_scalar` variants — `#[magetypes]`, `#[rite]` (multi-tier), `#[autoversion]`, and manual `#[arcane]` (one at a time). All produce the same naming convention, so one `incant!` dispatches to them all:

```rust
use archmage::prelude::*;
use magetypes::simd::generic::f32x8 as GenericF32x8;

// 1. #[magetypes] — generates _v4, _v3, _neon, _wasm128, _scalar from one body
#[magetypes(v4, v3, neon, wasm128, scalar)]
fn process_impl(token: Token, data: &mut [f32], scale: f32) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let scale_v = f32x8::splat(token, scale);
    let (chunks, tail) = f32x8::partition_slice_mut(token, data);
    for chunk in chunks {
        let v = f32x8::load(token, chunk);
        (v * scale_v).store(chunk);
    }
    for v in tail { *v *= scale; }
}

// 2. #[arcane] — manual specialization for ONE tier (v4x).
//    The _v4x suffix matches incant!'s naming convention.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane(import_intrinsics)]
fn process_impl_v4x(_token: X64V4xToken, data: &mut [f32], scale: f32) {
    let scale_v = _mm512_set1_ps(scale);
    for chunk in data.chunks_exact_mut(16) {
        let v = _mm512_loadu_ps(chunk.as_ptr());
        _mm512_storeu_ps(chunk.as_mut_ptr(), _mm512_mul_ps(v, scale_v));
    }
    // ... scalar tail
}

// 3. One incant! dispatches to ALL variants — generated + manual.
//    v4x(cfg(avx512)) feature-gates that tier: excluded if the avx512
//    cargo feature isn't enabled by the downstream crate.
pub fn process(data: &mut [f32], scale: f32) {
    incant!(process_impl(data, scale),
        [v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar]);
}
```

`incant!` doesn't know or care which macro generated each variant — it just looks for functions named `process_impl_v4x`, `process_impl_v4`, etc.

### `#[rite]` multi-tier — suffixed inner helpers

`#[rite(v3, v4, neon)]` generates `_v3`, `_v4`, `_neon` suffixed copies of a helper function, each with `#[target_feature]` + `#[inline]`. Use this for inner functions called from `#[arcane]` entry points — zero dispatch overhead, just inlining under the right target features:

```rust
// Generates accumulate_v3, accumulate_v4, accumulate_neon — all inlined
#[rite(v3, v4, neon, import_intrinsics)]
fn accumulate(data: &[f32; 8], acc: f32) -> f32 {
    let v = _mm256_loadu_ps(data.as_ptr());  // safe inside #[rite]
    // ...
    acc
}
```

### `#[autoversion]` — auto-vectorized scalar code

For loops that LLVM auto-vectorizes well, skip explicit SIMD types entirely. `#[autoversion]` generates tier variants AND a dispatcher from plain scalar code:

```rust
use archmage::autoversion;

/// LLVM compiles this to vfmadd231ps (AVX2), fmla (NEON), etc.
#[autoversion]
fn apply_color_matrix(rgb: &mut [f32], mat: [[f32; 3]; 3]) {
    for pixel in rgb.chunks_exact_mut(3) {
        let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
        pixel[0] = mat[0][0] * r + mat[0][1] * g + mat[0][2] * b;
        pixel[1] = mat[1][0] * r + mat[1][1] * g + mat[1][2] * b;
        pixel[2] = mat[2][0] * r + mat[2][1] * g + mat[2][2] * b;
    }
}
// Call directly — autoversion generated the dispatcher:
apply_color_matrix(&mut pixels, matrix);
```

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

See [Real-World Examples](@/magetypes/examples/_index.md) for 7 production patterns from image codecs.

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

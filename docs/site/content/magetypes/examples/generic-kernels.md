+++
title = "Reusable Generic Kernels"
description = "Real image-plane and audio-buffer loops with per-tier dispatch and reusable generic helpers"
weight = 0
+++

This runnable example is adapted from `zenfilters` in the `zen/zenpipe`
checkout at commit `12b468e3e0a8ef46e2eeb1c92d7b1a859e53036a`.
It preserves the production multiply/load/store/tail body. For this standalone
example, the separate x86/NEON/WASM forwarding functions are consolidated into
one four-tier macro, and the public function is named `apply_gain`. It is not a
verbatim copy of the production dispatch modules.

| Production call chain | Pinned source |
|---|---|
| `Exposure::apply` scales the Oklab L/a/b planes through `simd::scale_plane` | [filters/exposure.rs](https://github.com/imazen/zenpipe/blob/12b468e3e0a8ef46e2eeb1c92d7b1a859e53036a/zenfilters/src/filters/exposure.rs#L25) |
| `simd::scale_plane` calls `incant!(scale_plane_impl(...), [v3, neon, wasm128, scalar])` | [simd/mod.rs](https://github.com/imazen/zenpipe/blob/12b468e3e0a8ef46e2eeb1c92d7b1a859e53036a/zenfilters/src/simd/mod.rs#L27) |
| NEON wrapper forwards to the macro-generated `scale_plane_simd_neon` | [simd/neon.rs](https://github.com/imazen/zenpipe/blob/12b468e3e0a8ef46e2eeb1c92d7b1a859e53036a/zenfilters/src/simd/neon.rs#L14) |
| `#[magetypes(neon, wasm128)]` binds `GenericF32x8<Token>`, processes chunks, then the scalar tail | [simd/wide_simd.rs](https://github.com/imazen/zenpipe/blob/12b468e3e0a8ef46e2eeb1c92d7b1a859e53036a/zenfilters/src/simd/wide_simd.rs#L54) |
| V3 entry calls its `#[rite]` helper; scalar entry runs the same multiply per element | [simd/x86.rs](https://github.com/imazen/zenpipe/blob/12b468e3e0a8ef46e2eeb1c92d7b1a859e53036a/zenfilters/src/simd/x86.rs#L72), [simd/scalar.rs](https://github.com/imazen/zenpipe/blob/12b468e3e0a8ef46e2eeb1c92d7b1a859e53036a/zenfilters/src/simd/scalar.rs#L8) |

Process an image plane (exposure) or an audio buffer (gain), including a short
scalar tail. The vector type is generic over the token selected by `#[magetypes]`;
`incant!` chooses the CPU tier once outside the loop. No manual per-tier wrappers
or raw pointers are needed.

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

`#[magetypes]` creates the concrete per-tier functions. `#[inline(always)]` alone
does **not** establish a target-feature context and is not a substitute for this
entry point. Use it only when extracting a generic helper that must inline into
the generated context.

## Reuse a generic algorithm

The following is an explicitly refactored form of the same production body,
showing helper reuse. The real `zenfilters` blur chain uses this architecture:
`gaussian_blur_plane_dispatch_simd` (macro-generated context) calls
`gaussian_blur_fir_generic<T: F32x8Backend + F32x8Convert + Copy>` or
`stackblur_plane_generic` ([source](https://github.com/imazen/zenpipe/blob/12b468e3e0a8ef46e2eeb1c92d7b1a859e53036a/zenfilters/src/simd/wide_simd.rs#L266)).
The gain example keeps that entire call chain runnable without copying the
blur allocator, image geometry, and kernel-selection machinery:

```rust
use archmage::prelude::*;
use magetypes::simd::{backends::F32x8Backend, generic::f32x8};

#[inline(always)]
fn gain_kernel<T: F32x8Backend>(token: T, plane: &mut [f32], gain: f32) {
    let factor = f32x8::<T>::splat(token, gain);
    let (chunks, tail) = f32x8::<T>::partition_slice_mut(token, plane);
    for chunk in chunks {
        (f32x8::<T>::load(token, chunk) * factor).store(chunk);
    }
    for value in tail { *value *= gain; }
}

#[magetypes(v3, neon, wasm128, scalar)]
fn gain_entry(token: Token, plane: &mut [f32], gain: f32) {
    gain_kernel(token, plane, gain);
}

pub fn gain(plane: &mut [f32], factor: f32) {
    incant!(gain_entry(plane, factor), [v3, neon, wasm128, scalar])
}
```

The generic bound selects available operations; monomorphization resolves the
backend statically. The generated caller supplies the target features, and
inlining keeps the loop in that region. A generic helper is not inherently an
indirect call. Native V4 currently has a different supported shape set; this
8-lane example deliberately lists V3, NEON, WASM, and scalar.

`partition_slice_mut` returns array chunks and a tail. It does not promise
32-byte alignment; these loads and stores accept ordinary Rust array alignment.
The tail is necessary for arbitrary image widths and audio buffer lengths.

For behavior at NaNs, signed zeros, shifts, and conversion boundaries, see
[ISA quirks and fixups](@/magetypes/isa-quirks.md). These examples are compiled and
run by `magetypes/tests/doc_examples.rs`; the standalone first example is
`magetypes/examples/plane_gain.rs`.

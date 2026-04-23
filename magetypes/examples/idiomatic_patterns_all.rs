//! All the idiomatic magetypes + archmage patterns, combined and self-tested.
//!
//! This example is the reference for "what correct magetypes code looks like."
//! Each section is a distinct pattern; together they exercise every generator
//! and dispatch path. `main()` runs asserts against each — if this binary
//! prints `all patterns OK` on a platform, the patterns work there.
//!
//! Run:
//!   cargo run --release --example idiomatic_patterns_all
//!   cargo run --release --example idiomatic_patterns_all --features avx512
//!
//! Patterns covered:
//!   A. Inline `#[magetypes]`            — algorithm lives directly in the macro
//!   B. Extracted generic kernel         — `fn<T: F32x8Backend>` reused from N entries
//!   C. Hand-tuned `#[arcane]` for one tier — slots into `incant!` by suffix
//!   D. `#[rite]` multi-tier inner helper — internal helpers with per-tier features
//!   E. `#[autoversion]`                 — scalar loop, compiler auto-vectorizes
//!   F. Nested `incant!` rewriting       — zero-overhead cross-variant calls
//!   G. Polyfill defaults                — `f32x8` everywhere; NEON/WASM split to 2×128
//!
//! Non-principles:
//!   - No hand-written `#[arcane]` wrappers "per tier" around a generic kernel.
//!     `#[magetypes]` IS the wrapper generator. `T` is inferred from the
//!     substituted concrete `Token` in each variant's call site.
//!   - Width is an algorithm choice, not a platform choice. `f32x8` on NEON
//!     polyfills to 2×`f32x4` with no overhead vs. hand-rolled splits.

#![allow(dead_code)]

use archmage::prelude::*;
use magetypes::simd::{backends::F32x8Backend, generic::f32x8 as GenericF32x8};

// ============================================================================
// Pattern A — Inline #[magetypes]
// ============================================================================
// The default. The algorithm lives directly inside a `#[magetypes]` function.
// `Token` is substituted per tier; `#[magetypes]` generates an `#[arcane]`-
// wrapped variant for each listed tier. No hand-written wrappers.

#[magetypes(v4, v3, neon, wasm128, scalar)]
fn scale_plane_impl(token: Token, plane: &mut [f32], factor: f32) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    let factor_v = f32x8::splat(token, factor);
    let (chunks, tail) = f32x8::partition_slice_mut(token, plane);
    for chunk in chunks {
        (f32x8::load(token, chunk) * factor_v).store(chunk);
    }
    for v in tail {
        *v *= factor;
    }
}

pub fn scale_plane(plane: &mut [f32], factor: f32) {
    incant!(scale_plane_impl(plane, factor))
}

// ============================================================================
// Pattern B — Extracted generic kernel + thin #[magetypes] entry
// ============================================================================
// Use when one kernel is reused from multiple entry points. The generic kernel
// has no `#[target_feature]` of its own — it inherits the caller's features
// through inlining. `#[inline(always)]` is mandatory; without it, intrinsics
// become function calls (18× slower even inside a target-feature region).
//
// `T` is inferred automatically at each call site because the `#[magetypes]`
// entry passes a concrete token (e.g. `X64V3Token` implements `F32x8Backend`).

#[inline(always)]
fn dot_kernel<T: F32x8Backend>(token: T, a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut acc = GenericF32x8::<T>::zero(token);
    let chunks = a.len() / 8;
    for i in 0..chunks {
        let va = GenericF32x8::<T>::load(token, a[i * 8..][..8].try_into().unwrap());
        let vb = GenericF32x8::<T>::load(token, b[i * 8..][..8].try_into().unwrap());
        acc = va.mul_add(vb, acc);
    }
    let mut total = acc.reduce_add();
    for i in (chunks * 8)..a.len() {
        total += a[i] * b[i];
    }
    total
}

#[magetypes(v4, v3, neon, wasm128, scalar)]
fn dot_impl(token: Token, a: &[f32], b: &[f32]) -> f32 {
    dot_kernel(token, a, b)
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    incant!(dot_impl(a, b))
}

// ============================================================================
// Pattern C — Hand-tuned #[arcane] for one tier, slotted in by suffix
// ============================================================================
// When a single tier genuinely benefits from an algorithm `#[magetypes]` can't
// express uniformly (different width, different instructions), write a
// standalone `#[arcane]` function whose name matches the `_<tier>` suffix
// convention. `incant!` doesn't care which macro authored which variant —
// it resolves by name.
//
// Here `_v4x` uses native 512-bit `f32x16` instead of the f32x8 polyfill.

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
mod v4x_hand_tuned {
    use super::*;
    use magetypes::simd::generic::f32x16 as GenericF32x16;

    #[arcane]
    pub(super) fn scale_plane_hand_v4x(token: X64V4xToken, plane: &mut [f32], factor: f32) {
        let factor_v = GenericF32x16::<X64V4xToken>::splat(token, factor);
        let (chunks, tail) = GenericF32x16::<X64V4xToken>::partition_slice_mut(token, plane);
        for chunk in chunks {
            (GenericF32x16::<X64V4xToken>::load(token, chunk) * factor_v).store(chunk);
        }
        for v in tail {
            *v *= factor;
        }
    }
}

// Public entry that reuses Pattern A's `scale_plane_impl_*` variants for
// v4/v3/neon/wasm128/scalar, and the hand-tuned `_v4x` when available.
// `incant!` picks the highest-matching available tier.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub fn scale_plane_hand_v4x(plane: &mut [f32], factor: f32) {
    if let Some(t) = X64V4xToken::summon() {
        v4x_hand_tuned::scale_plane_hand_v4x(t, plane, factor);
    } else {
        scale_plane(plane, factor);
    }
}

// ============================================================================
// Pattern D — #[rite] multi-tier inner helper + #[magetypes] counterpart
// ============================================================================
// `#[rite(v3, v4, neon, wasm128)]` generates per-tier copies of an inner
// helper, each with its own `#[target_feature]` + `#[inline]`. No wrapper,
// no optimization boundary — the matching-tier copy inlines into the
// caller's target-feature region. `#[rite]` has no `scalar` tier; use
// `#[magetypes]` if you need a scalar variant for `incant!` dispatch.
//
// Because Pattern F dispatches via nested `incant!` — which needs a scalar
// fallback — we use `#[magetypes]` for `clamp01_impl` below. `#[rite]` would
// be right for a helper called only from known-tier contexts (e.g. only from
// inside a concrete `#[arcane(X64V3Token)]` body via its `_v3` suffix).
//
// Multi-tier rite example, for completeness; unused in the self-test.
// `#[rite]` does NOT do `Token` substitution — each variant is the same body
// with a different `#[target_feature]`. Signatures stay as-written.
#[rite(v3, v4, neon, wasm128)]
fn saturate_chunk(chunk: &mut [f32; 8]) {
    for v in chunk {
        if *v < 0.0 {
            *v = 0.0;
        }
        if *v > 1.0 {
            *v = 1.0;
        }
    }
}

#[magetypes(v4, v3, neon, wasm128, scalar)]
fn clamp01_impl(token: Token, plane: &mut [f32]) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let lo = f32x8::splat(token, 0.0);
    let hi = f32x8::splat(token, 1.0);
    let (chunks, tail) = f32x8::partition_slice_mut(token, plane);
    for chunk in chunks {
        f32x8::load(token, chunk).max(lo).min(hi).store(chunk);
    }
    for v in tail {
        *v = v.clamp(0.0, 1.0);
    }
}

// ============================================================================
// Pattern E — #[autoversion] for scalar that auto-vectorizes
// ============================================================================
// Plain scalar body, recompiled under each tier's `#[target_feature]`. LLVM
// auto-vectorizes each copy. `#[autoversion]` is the only generator that
// emits its own dispatcher — call `apply_color_matrix` directly, no `incant!`.

#[autoversion]
fn apply_color_matrix(rgb: &mut [f32], mat: [[f32; 3]; 3]) {
    for pixel in rgb.chunks_exact_mut(3) {
        let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
        pixel[0] = mat[0][0] * r + mat[0][1] * g + mat[0][2] * b;
        pixel[1] = mat[1][0] * r + mat[1][1] * g + mat[1][2] * b;
        pixel[2] = mat[2][0] * r + mat[2][1] * g + mat[2][2] * b;
    }
}

// ============================================================================
// Pattern F — Nested incant! rewriting
// ============================================================================
// When `incant!(foo(args))` appears inside a tier-annotated body, the outer
// macro rewrites it to the direct tier-matching call at compile time:
//   V3 variant: incant!(scale_plane_impl(p, f)) → scale_plane_impl_v3(token, p, f)
// No dispatcher branch, no cache probe — the inner function inlines into the
// outer's `#[target_feature]` region. Verified in SPEC-INCANT-REWRITING.md
// (0.94 ns vs 5.6 ns with dispatcher rehop).

#[magetypes(v4, v3, neon, wasm128, scalar)]
fn pipeline_impl(token: Token, plane: &mut [f32], bias: f32, factor: f32) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    // Step 1: add bias in-place.
    let bias_v = f32x8::splat(token, bias);
    let (chunks, tail) = f32x8::partition_slice_mut(token, plane);
    for chunk in chunks {
        (f32x8::load(token, chunk) + bias_v).store(chunk);
    }
    for v in tail {
        *v += bias;
    }

    // Step 2: clamp via `#[magetypes]`-generated family (Pattern D). Inside
    // a `#[magetypes]` variant, the rewriter picks `clamp01_impl_<tier>`
    // matching this variant. Zero dispatch overhead.
    incant!(clamp01_impl(plane));

    // Step 3: scale via another `#[magetypes]`-generated family. Same
    // rewriting applies: this becomes `scale_plane_impl_<tier>(token, ...)`.
    incant!(scale_plane_impl(plane, factor));
}

pub fn pipeline(plane: &mut [f32], bias: f32, factor: f32) {
    incant!(pipeline_impl(plane, bias, factor))
}

// ============================================================================
// Pattern G — Polyfill defaults (demonstrated by Patterns A/B/F)
// ============================================================================
// `scale_plane_impl` uses `f32x8` on every listed tier including `neon` and
// `wasm128`. The polyfill emits 2×`f32x4` ops internally on 128-bit SIMD;
// no hand-rolled half-split needed. Pick the width your algorithm wants —
// the library handles the hardware split. Native on AVX2, polyfilled
// elsewhere, same source.
//
// Verify what the compiled path actually is at runtime:

fn print_implementations() {
    // `implementation_name()` is defined for every concrete backend that has
    // a `{Type}Backend` impl — native and polyfill. The SAME #[magetypes]
    // body above runs on every platform; the lowering differs per tier.
    use magetypes::simd::generic::{f32x4, f32x8};

    println!(
        "  f32x4 on ScalarToken: {}",
        f32x4::<archmage::ScalarToken>::implementation_name()
    );
    println!(
        "  f32x8 on ScalarToken: {}",
        f32x8::<archmage::ScalarToken>::implementation_name()
    );

    #[cfg(target_arch = "x86_64")]
    {
        println!(
            "  f32x4 on X64V3Token:  {}",
            f32x4::<archmage::X64V3Token>::implementation_name()
        );
        println!(
            "  f32x8 on X64V3Token:  {}",
            f32x8::<archmage::X64V3Token>::implementation_name()
        );
    }
    #[cfg(target_arch = "aarch64")]
    {
        println!(
            "  f32x4 on NeonToken:   {}",
            f32x4::<archmage::NeonToken>::implementation_name()
        );
        println!(
            "  f32x8 on NeonToken:   {}",
            f32x8::<archmage::NeonToken>::implementation_name()
        );
    }
    #[cfg(target_arch = "wasm32")]
    {
        println!(
            "  f32x4 on Wasm128Token: {}",
            f32x4::<archmage::Wasm128Token>::implementation_name()
        );
        println!(
            "  f32x8 on Wasm128Token: {}",
            f32x8::<archmage::Wasm128Token>::implementation_name()
        );
    }
}

// ============================================================================
// Self-test
// ============================================================================

fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

fn main() {
    println!("magetypes idiomatic patterns — self-test");

    // --- Pattern A: inline #[magetypes] ---
    let mut plane = vec![1.0f32; 19]; // 2 full f32x8 chunks + 3 tail
    scale_plane(&mut plane, 2.0);
    assert!(
        approx_eq(&plane, &vec![2.0; 19], 1e-6),
        "[A] scale_plane: {plane:?}"
    );
    println!("  [A] inline #[magetypes] scale_plane           OK");

    // --- Pattern B: extracted generic kernel ---
    let a: Vec<f32> = (1..=17).map(|i| i as f32).collect();
    let b: Vec<f32> = (1..=17).map(|i| (2 * i) as f32).collect();
    let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    let got = dot(&a, &b);
    assert!(
        (got - expected).abs() < 1e-2,
        "[B] dot: got {got}, expected {expected}"
    );
    println!("  [B] extracted generic kernel dot              OK ({got} vs {expected})");

    // --- Pattern C: hand-tuned #[arcane] for v4x ---
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        let mut plane = vec![3.0f32; 37];
        scale_plane_hand_v4x(&mut plane, 0.5);
        assert!(
            approx_eq(&plane, &vec![1.5; 37], 1e-6),
            "[C] hand-tuned v4x: {plane:?}"
        );
        println!("  [C] hand-tuned #[arcane] v4x (f32x16)         OK");
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx512")))]
    {
        println!(
            "  [C] hand-tuned #[arcane] v4x (f32x16)         skipped (needs x86_64 + feature=avx512)"
        );
    }

    // --- Pattern E: #[autoversion] ---
    let mut rgb = vec![0.1f32, 0.5, 0.9, 0.2, 0.4, 0.8];
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let before = rgb.clone();
    apply_color_matrix(&mut rgb, identity);
    assert!(
        approx_eq(&rgb, &before, 1e-6),
        "[E] color matrix identity: {rgb:?}"
    );
    let swap_rgb_to_bgr = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
    apply_color_matrix(&mut rgb, swap_rgb_to_bgr);
    assert!(
        approx_eq(&rgb, &[0.9, 0.5, 0.1, 0.8, 0.4, 0.2], 1e-6),
        "[E] color matrix swap: {rgb:?}"
    );
    println!("  [E] #[autoversion] color matrix               OK");

    // --- Pattern F: nested incant! rewriting via pipeline ---
    // pipeline: x → clamp01(x + bias) * factor
    // Input 0.5, bias 0.3 → 0.8 → clamp → 0.8 → * 2.0 → 1.6
    // Input 2.0, bias -0.5 → 1.5 → clamp → 1.0 → * 2.0 → 2.0
    let mut plane: Vec<f32> = (0..19)
        .map(|i| if i % 2 == 0 { 0.5 } else { 2.0 })
        .collect();
    pipeline(&mut plane, -0.5, 2.0);
    for (i, &v) in plane.iter().enumerate() {
        let expected = if i % 2 == 0 { 0.0f32 } else { 2.0f32 };
        assert!(
            (v - expected).abs() < 1e-5,
            "[F] pipeline idx {i}: got {v}, expected {expected}"
        );
    }
    println!("  [F] nested incant! rewriting pipeline         OK");

    // --- Pattern G: polyfill visibility ---
    println!("  [G] polyfill implementations:");
    print_implementations();

    println!("all patterns OK");
}

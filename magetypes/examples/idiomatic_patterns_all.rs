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
//!   B. Extracted generic kernel         — `fn<T: F32x8Backend>` + thin `#[magetypes]` entry
//!   C. Hand-tuned `#[arcane]` for one tier — slots into `incant!` by suffix
//!   D. `#[autoversion]`                 — scalar loop, compiler auto-vectorizes per tier
//!   E. Nested `incant!` rewriting       — zero-overhead cross-variant calls
//!   F. Polyfill assertions              — confirm platform backend via `implementation_name()`
//!
//! Omitted:
//!   - `#[rite]` multi-tier: generates `fn_v3`/`fn_v4`/… for inner helpers with
//!     per-tier `#[target_feature]`. Useful when you want an inner helper with
//!     explicit target-feature control outside the `#[magetypes]` body. Most
//!     magetypes code uses a generic kernel (Pattern B) instead. No example
//!     here because any contrived use would just duplicate Pattern B.
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
// Pattern A — Inline #[magetypes] with define(...)
// ============================================================================
// The default. The algorithm lives directly inside a `#[magetypes]` function.
// `Token` is substituted per tier; `#[magetypes]` generates an `#[arcane]`-
// wrapped variant for each listed tier. No hand-written wrappers.
//
// `define(f32x8)` injects `type f32x8 = ::magetypes::simd::generic::f32x8<Token>;`
// at the top of each variant body — eliminates the boilerplate alias users
// previously had to write manually. Multiple types: `define(f32x8, u8x16, ...)`.

#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
fn scale_plane_impl(token: Token, plane: &mut [f32], factor: f32) {
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
    // Pattern C below adds a hand-tuned `scale_plane_impl_v4x`. Including
    // it in the tier list here lets `incant!` pick the 512-bit native path
    // when avx512 is available, and fall through to the #[magetypes]-generated
    // _v4/_v3/_neon/_wasm128/_scalar variants otherwise.
    incant!(
        scale_plane_impl(plane, factor),
        [v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar]
    )
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
// express uniformly, hand-write it as a standalone `#[arcane]` function
// whose name is `<base>_<tier_suffix>` — MATCHING the prefix used by the
// `#[magetypes]` family it joins. Then list the tier in `incant!`'s tier
// list (see Pattern A's `scale_plane` above).
//
// The idiomatic form:
//   - Omit the hand-written tier from `#[magetypes]`'s list (so the macro
//     doesn't also generate one for that tier)
//   - Name the hand-written function `<same base>_<tier>` so `incant!`
//     resolves it by suffix alongside the macro-generated variants
//   - Extend the `incant!` tier list to include the new tier
//
// Here `_v4x` uses native 512-bit `f32x16` instead of the `f32x8` polyfill.
// `incant!` in `scale_plane` picks it when avx512 is available, else falls
// through to `_v4` / `_v3` / `_neon` / `_wasm128` / `_scalar` — all of which
// were generated by Pattern A's `#[magetypes]`.

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
fn scale_plane_impl_v4x(token: X64V4xToken, plane: &mut [f32], factor: f32) {
    use magetypes::simd::generic::f32x16 as GenericF32x16;

    let factor_v = GenericF32x16::<X64V4xToken>::splat(token, factor);
    let (chunks, tail) = GenericF32x16::<X64V4xToken>::partition_slice_mut(token, plane);
    for chunk in chunks {
        (GenericF32x16::<X64V4xToken>::load(token, chunk) * factor_v).store(chunk);
    }
    for v in tail {
        *v *= factor;
    }
}

// ============================================================================
// Shared helper used by Pattern E's pipeline
// ============================================================================
// A plain `#[magetypes]` function the pipeline routes through via nested
// `incant!`. Kept outside the numbered patterns — it's the thing being
// dispatched to, not a pattern of its own.

#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
fn clamp01_impl(token: Token, plane: &mut [f32]) {
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
// Pattern D — #[autoversion] for scalar that auto-vectorizes
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
// Pattern E — Nested incant! rewriting
// ============================================================================
// When `incant!(foo(args))` appears inside a tier-annotated body, the outer
// macro rewrites it to the direct tier-matching call at compile time:
//   V3 variant: incant!(scale_plane_impl(p, f)) → scale_plane_impl_v3(token, p, f)
// No dispatcher branch, no cache probe — the inner function inlines into the
// outer's `#[target_feature]` region. Verified in SPEC-INCANT-REWRITING.md
// (0.94 ns vs 5.6 ns with dispatcher rehop).

#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
fn pipeline_impl(token: Token, plane: &mut [f32], bias: f32, factor: f32) {
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
// Pattern F — Polyfill assertions: verify which backend ran per platform
// ============================================================================
// The SAME `#[magetypes]` body above runs on every platform. AVX2 lowers
// `f32x8` to one 256-bit op; NEON and Wasm128 lower it to two 128-bit ops
// (polyfill); scalar runs an array loop. This function asserts the actual
// backend names via `implementation_name()` so a passing run proves the
// polyfill claim — not just that the code compiles.

fn verify_polyfill_implementations() {
    use magetypes::simd::generic::{f32x4, f32x8};

    // Scalar fallback is callable and named consistently on every arch.
    assert_eq!(
        f32x4::<archmage::ScalarToken>::implementation_name(),
        "scalar::f32x4"
    );
    assert_eq!(
        f32x8::<archmage::ScalarToken>::implementation_name(),
        "scalar::f32x8"
    );

    #[cfg(target_arch = "x86_64")]
    {
        assert_eq!(
            f32x4::<archmage::X64V3Token>::implementation_name(),
            "x86::v3::f32x4",
            "f32x4 on x86-64 should be native v3 (SSE)"
        );
        assert_eq!(
            f32x8::<archmage::X64V3Token>::implementation_name(),
            "x86::v3::f32x8",
            "f32x8 on x86-64 should be native v3 (AVX2) — one 256-bit op"
        );
    }
    #[cfg(target_arch = "aarch64")]
    {
        assert_eq!(
            f32x4::<archmage::NeonToken>::implementation_name(),
            "arm::neon::f32x4",
            "f32x4 on aarch64 should be native NEON"
        );
        assert_eq!(
            f32x8::<archmage::NeonToken>::implementation_name(),
            "polyfill::neon::f32x8",
            "f32x8 on aarch64 should polyfill to 2× NEON (128-bit)"
        );
    }
    #[cfg(target_arch = "wasm32")]
    {
        assert_eq!(
            f32x4::<archmage::Wasm128Token>::implementation_name(),
            "wasm::wasm128::f32x4",
            "f32x4 on wasm32 should be native SIMD128"
        );
        assert_eq!(
            f32x8::<archmage::Wasm128Token>::implementation_name(),
            "polyfill::wasm128::f32x8",
            "f32x8 on wasm32 should polyfill to 2× SIMD128"
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
    // Tolerance 1e-2: `dot` sums 17 FMA products; the SIMD order (mul_add
    // across 2 chunks then reduce_add across 8 lanes, plus 1 scalar tail
    // term) differs from the scalar reference's sequential sum. With integer
    // values this range is exact, but 1e-2 gives slack for any floating-point
    // reassociation a backend might introduce.
    let a: Vec<f32> = (1..=17).map(|i| i as f32).collect();
    let b: Vec<f32> = (1..=17).map(|i| (2 * i) as f32).collect();
    let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    let got = dot(&a, &b);
    assert!(
        (got - expected).abs() < 1e-2,
        "[B] dot: got {got}, expected {expected}"
    );
    println!("  [B] extracted generic kernel dot              OK ({got} vs {expected})");

    // --- Pattern C: hand-tuned _v4x slots into Pattern A's family ---
    // We call the SAME public `scale_plane` as Pattern A. On a CPU with
    // avx512, `incant!` routes through `scale_plane_impl_v4x` (hand-written
    // below Pattern A) via the suffix convention — no separate entry point,
    // no manual summon() branching.
    {
        let mut plane = vec![3.0f32; 37];
        scale_plane(&mut plane, 0.5);
        assert!(
            approx_eq(&plane, &vec![1.5; 37], 1e-6),
            "[C] scale_plane (v4x/v4/v3/neon/wasm128/scalar): {plane:?}"
        );
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        println!("  [C] hand-tuned _v4x slotted via incant! suffix OK (avx512 active)");
        #[cfg(not(all(target_arch = "x86_64", feature = "avx512")))]
        println!("  [C] hand-tuned _v4x slotted via incant! suffix OK (fallback tier active)");
    }

    // --- Pattern D: #[autoversion] ---
    let mut rgb = vec![0.1f32, 0.5, 0.9, 0.2, 0.4, 0.8];
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let before = rgb.clone();
    apply_color_matrix(&mut rgb, identity);
    assert!(
        approx_eq(&rgb, &before, 1e-6),
        "[D] color matrix identity: {rgb:?}"
    );
    let swap_rgb_to_bgr = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
    apply_color_matrix(&mut rgb, swap_rgb_to_bgr);
    assert!(
        approx_eq(&rgb, &[0.9, 0.5, 0.1, 0.8, 0.4, 0.2], 1e-6),
        "[D] color matrix swap: {rgb:?}"
    );
    println!("  [D] #[autoversion] color matrix               OK");

    // --- Pattern E: nested incant! rewriting via pipeline ---
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
            "[E] pipeline idx {i}: got {v}, expected {expected}"
        );
    }
    println!("  [E] nested incant! rewriting pipeline         OK");

    // --- Pattern F: polyfill assertions (per-platform implementation_name) ---
    verify_polyfill_implementations();
    println!("  [F] polyfill implementation_name() assertions OK");

    println!("all patterns OK");
}

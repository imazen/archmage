//! Benchmark: target-feature boundary overhead vs wrapper overhead.
//!
//! Run with: cargo bench --bench asm_inspection
//!
//! ## Key finding: the overhead is from target-feature mismatch, NOT wrappers
//!
//! LLVM cannot inline a `#[target_feature(enable = "avx2")]` function into a
//! caller that lacks those features. This creates an optimization boundary per
//! call — LLVM can't hoist loads, sink stores, or vectorize across it.
//!
//! Proof:
//! - Patterns 1, 4, & 7 all cross the boundary per iteration → same speed (~2.2 µs)
//! - Pattern 4 has NO wrapper (calls `#[rite]` directly) — still slow
//! - Pattern 7 is bare `#[target_feature]` with no archmage at all — same speed as 1
//! - Patterns 5 & 6 use wrappers but WITHOUT target-feature mismatch → fast (~545 ns)
//!
//! Conclusion: `#[arcane]`'s overhead equals a bare `#[target_feature]` call.
//! The cost is from LLVM's inability to inline across mismatched target features,
//! not from wrapper functions or archmage abstractions.

#![cfg(target_arch = "x86_64")]

use archmage::{Desktop64, SimdToken, arcane, rite};
use std::arch::x86_64::*;

// ============================================================================
// Pattern 1: #[arcane] in loop — target-feature boundary each iteration
// ============================================================================

#[arcane]
fn process_chunk_arcane(_token: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    unsafe {
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let sum = _mm256_add_ps(va, vb);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), sum);
        out
    }
}

#[inline(never)]
pub fn loop_with_arcane_in_loop(token: Desktop64, data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    let mut total = 0.0f32;
    for (a, b) in data.iter().zip(other.iter()) {
        let result = process_chunk_arcane(token, a, b);
        total += result[0];
    }
    total
}

// ============================================================================
// Pattern 2: Loop inside #[arcane], #[rite] inlines (features match)
// ============================================================================

#[rite]
fn process_chunk_rite(_token: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    unsafe {
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let sum = _mm256_add_ps(va, vb);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), sum);
        out
    }
}

#[arcane]
fn loop_inner_rite(token: Desktop64, data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    let mut total = 0.0f32;
    for (a, b) in data.iter().zip(other.iter()) {
        let result = process_chunk_rite(token, a, b);
        total += result[0];
    }
    total
}

#[inline(never)]
pub fn loop_inside_arcane_with_rite(
    token: Desktop64,
    data: &[[f32; 8]],
    other: &[[f32; 8]],
) -> f32 {
    loop_inner_rite(token, data, other)
}

// ============================================================================
// Pattern 3: Manual inline for comparison (best possible)
// ============================================================================

#[arcane]
fn loop_manual_inline(_token: Desktop64, data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    let mut total = 0.0f32;
    for (a, b) in data.iter().zip(other.iter()) {
        unsafe {
            let va = _mm256_loadu_ps(a.as_ptr());
            let vb = _mm256_loadu_ps(b.as_ptr());
            let sum = _mm256_add_ps(va, vb);
            let mut out = [0.0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), sum);
            total += out[0];
        }
    }
    total
}

#[inline(never)]
pub fn loop_inside_arcane_manual(token: Desktop64, data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    loop_manual_inline(token, data, other)
}

// ============================================================================
// Pattern 4: #[rite] called directly from non-target_feature context (NO wrapper)
// Same speed as pattern 1 — proving the overhead is NOT from the wrapper
// ============================================================================

#[inline(never)]
pub fn loop_calling_rite_directly(token: Desktop64, data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    let mut total = 0.0f32;
    for (a, b) in data.iter().zip(other.iter()) {
        // SAFETY: We have the token, so CPU supports the features
        let result = unsafe { process_chunk_rite(token, a, b) };
        total += result[0];
    }
    total
}

// ============================================================================
// Pattern 5: Wrapper WITHOUT target-feature mismatch (scalar fallback)
// Proves that wrapper call overhead itself is negligible — LLVM inlines it.
// ============================================================================

#[inline]
fn process_chunk_scalar(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    let mut out = [0.0f32; 8];
    for i in 0..8 {
        out[i] = a[i] + b[i];
    }
    out
}

/// Wrapper function (no target_feature) calling scalar helper (no target_feature).
/// LLVM inlines this freely — no feature mismatch to block it.
#[inline(never)]
pub fn loop_scalar_wrapper(data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    let mut total = 0.0f32;
    for (a, b) in data.iter().zip(other.iter()) {
        let result = process_chunk_scalar(a, b);
        total += result[0];
    }
    total
}

// ============================================================================
// Pattern 6: Scalar code inlined directly (baseline for pattern 5 comparison)
// ============================================================================

#[inline(never)]
pub fn loop_scalar_inline(data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    let mut total = 0.0f32;
    for (a, b) in data.iter().zip(other.iter()) {
        let mut out = [0.0f32; 8];
        for i in 0..8 {
            out[i] = a[i] + b[i];
        }
        total += out[0];
    }
    total
}

// ============================================================================
// Pattern 7: Bare #[target_feature] — no archmage, no wrapper, no token
// Identical cost to pattern 1 (#[arcane]). Proves archmage adds zero overhead
// vs hand-written #[target_feature] + unsafe.
// ============================================================================

#[target_feature(enable = "avx2")]
unsafe fn process_chunk_bare_target_feature(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    unsafe {
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let sum = _mm256_add_ps(va, vb);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), sum);
        out
    }
}

#[inline(never)]
pub fn loop_bare_target_feature(data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    let mut total = 0.0f32;
    for (a, b) in data.iter().zip(other.iter()) {
        // SAFETY: benchmark only runs when Desktop64::summon() succeeds
        let result = unsafe { process_chunk_bare_target_feature(a, b) };
        total += result[0];
    }
    total
}

// ============================================================================
// DCT-8: realistic SIMD workload (8 dot products per row, 100 rows)
// Confirms the boundary effect holds with real computational density.
// ============================================================================

/// Compute DCT-II coefficient matrix for N=8.
fn dct8_coefficients() -> [[f32; 8]; 8] {
    let mut c = [[0.0f32; 8]; 8];
    for k in 0..8u32 {
        let alpha: f32 = if k == 0 {
            (1.0 / 8.0_f32).sqrt()
        } else {
            (2.0 / 8.0_f32).sqrt()
        };
        for n in 0..8u32 {
            c[k as usize][n as usize] =
                alpha * (std::f32::consts::PI * (2 * n + 1) as f32 * k as f32 / 16.0).cos();
        }
    }
    c
}

/// Horizontal sum of 8 floats in __m256. Use inside unsafe/target_feature context.
macro_rules! hsum256 {
    ($v:expr) => {{
        let hi = _mm256_extractf128_ps::<1>($v);
        let lo = _mm256_castps256_ps128($v);
        let sum = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum);
        let sums = _mm_add_ps(sum, shuf);
        let hi64 = _mm_movehl_ps(sums, sums);
        _mm_cvtss_f32(_mm_add_ss(sums, hi64))
    }};
}

// DCT pattern A: #[arcane] per row — boundary crossing every row
#[arcane]
fn dct8_row_arcane(_token: Desktop64, coeffs: &[[f32; 8]; 8], input: &[f32; 8]) -> [f32; 8] {
    unsafe {
        let v = _mm256_loadu_ps(input.as_ptr());
        let mut out = [0.0f32; 8];
        for k in 0..8 {
            let c = _mm256_loadu_ps(coeffs[k].as_ptr());
            out[k] = hsum256!(_mm256_mul_ps(v, c));
        }
        out
    }
}

#[inline(never)]
pub fn dct_arcane_per_row(token: Desktop64, coeffs: &[[f32; 8]; 8], rows: &[[f32; 8]]) -> f32 {
    let mut total = 0.0f32;
    for row in rows {
        let out = dct8_row_arcane(token, coeffs, row);
        total += out[0];
    }
    total
}

// DCT pattern B: loop inside #[arcane], #[rite] helper
#[rite]
fn dct8_row_rite(_token: Desktop64, coeffs: &[[f32; 8]; 8], input: &[f32; 8]) -> [f32; 8] {
    unsafe {
        let v = _mm256_loadu_ps(input.as_ptr());
        let mut out = [0.0f32; 8];
        for k in 0..8 {
            let c = _mm256_loadu_ps(coeffs[k].as_ptr());
            out[k] = hsum256!(_mm256_mul_ps(v, c));
        }
        out
    }
}

#[arcane]
fn dct_loop_inner(token: Desktop64, coeffs: &[[f32; 8]; 8], rows: &[[f32; 8]]) -> f32 {
    let mut total = 0.0f32;
    for row in rows {
        let out = dct8_row_rite(token, coeffs, row);
        total += out[0];
    }
    total
}

#[inline(never)]
pub fn dct_rite_in_arcane(token: Desktop64, coeffs: &[[f32; 8]; 8], rows: &[[f32; 8]]) -> f32 {
    dct_loop_inner(token, coeffs, rows)
}

// DCT pattern C: bare #[target_feature], no archmage
#[target_feature(enable = "avx2")]
unsafe fn dct8_row_bare(coeffs: &[[f32; 8]; 8], input: &[f32; 8]) -> [f32; 8] {
    unsafe {
        let v = _mm256_loadu_ps(input.as_ptr());
        let mut out = [0.0f32; 8];
        for k in 0..8 {
            let c = _mm256_loadu_ps(coeffs[k].as_ptr());
            out[k] = hsum256!(_mm256_mul_ps(v, c));
        }
        out
    }
}

#[inline(never)]
pub fn dct_bare_target_feature(coeffs: &[[f32; 8]; 8], rows: &[[f32; 8]]) -> f32 {
    let mut total = 0.0f32;
    for row in rows {
        let out = unsafe { dct8_row_bare(coeffs, row) };
        total += out[0];
    }
    total
}

// ============================================================================
// Cross-token nesting benchmarks
//
// Tests #[arcane]→#[arcane] with same token, downcast (V4→V3 via .as_x64v3()),
// and upcast (V3 entry → summon V4 inside). Each has a bare #[target_feature]
// equivalent proving archmage adds zero overhead.
//
// All #[arcane]→#[arcane] patterns hit the boundary. Use #[rite] instead.
// ============================================================================

#[cfg(feature = "avx512")]
use archmage::X64V4Token;

// -- Helper: V3 #[arcane] chunk (called from nesting patterns) ----------------

#[arcane]
fn add_chunk_v3_arcane(_token: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    unsafe {
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let sum = _mm256_add_ps(va, vb);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), sum);
        out
    }
}

// -- Helper: Bare V3 #[target_feature] (no archmage) -------------------------

#[target_feature(enable = "avx2,fma")]
unsafe fn add_chunk_v3_bare(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    unsafe {
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let sum = _mm256_add_ps(va, vb);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), sum);
        out
    }
}

// -- 1. V3→V3: #[arcane] calling #[arcane] (same token, still boundary) ------

#[arcane]
fn loop_v3_arcane_v3_arcane(token: Desktop64, data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    let mut total = 0.0f32;
    for (a, b) in data.iter().zip(other.iter()) {
        let result = add_chunk_v3_arcane(token, a, b);
        total += result[0];
    }
    total
}

#[inline(never)]
pub fn nest_v3_v3_arcane(token: Desktop64, data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    loop_v3_arcane_v3_arcane(token, data, other)
}

// Bare equivalent: V3 #[target_feature] calling V3 #[target_feature]
#[target_feature(enable = "avx2,fma")]
unsafe fn loop_v3_bare_v3_bare(data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    unsafe {
        let mut total = 0.0f32;
        for (a, b) in data.iter().zip(other.iter()) {
            let result = add_chunk_v3_bare(a, b);
            total += result[0];
        }
        total
    }
}

#[inline(never)]
pub fn nest_v3_v3_bare(data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    unsafe { loop_v3_bare_v3_bare(data, other) }
}

// -- 2. V3→V3: #[arcane] calling #[rite] (inlines — control) -----------------

#[arcane]
fn loop_v3_arcane_v3_rite(token: Desktop64, data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    let mut total = 0.0f32;
    for (a, b) in data.iter().zip(other.iter()) {
        let result = process_chunk_rite(token, a, b);
        total += result[0];
    }
    total
}

#[inline(never)]
pub fn nest_v3_v3_rite(token: Desktop64, data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    loop_v3_arcane_v3_rite(token, data, other)
}

// -- 3. V4 entry → downcast .as_x64v3() → call V3 #[arcane] -----------------

#[cfg(feature = "avx512")]
#[arcane]
fn loop_v4_downcast_v3_arcane(_token: X64V4Token, data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    // IntoConcreteToken is identity-only (V4.as_x64v3() = None), so summon V3 directly.
    // CPU always has V3 if it has V4.
    let v3 = Desktop64::summon().unwrap();
    let mut total = 0.0f32;
    for (a, b) in data.iter().zip(other.iter()) {
        let result = add_chunk_v3_arcane(v3, a, b); // V3 #[arcane] from V4 context
        total += result[0];
    }
    total
}

#[cfg(feature = "avx512")]
#[inline(never)]
pub fn nest_v4_down_v3_arcane(token: X64V4Token, data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    loop_v4_downcast_v3_arcane(token, data, other)
}

// Bare equivalent: V4 #[target_feature] calling V3 #[target_feature]
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512cd,avx512dq,avx512vl")]
unsafe fn loop_v4_bare_v3_bare(data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    unsafe {
        let mut total = 0.0f32;
        for (a, b) in data.iter().zip(other.iter()) {
            let result = add_chunk_v3_bare(a, b);
            total += result[0];
        }
        total
    }
}

#[cfg(feature = "avx512")]
#[inline(never)]
pub fn nest_v4_down_v3_bare(data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    unsafe { loop_v4_bare_v3_bare(data, other) }
}

// -- 4. V3 entry → summon V4 inside → call V4 #[arcane] per iteration --------
// Uses REAL AVX-512 instructions (512-bit zmm) so it's not just AVX2 on V4 token.

#[cfg(feature = "avx512")]
#[arcane]
fn add_chunk_v4_arcane(_token: X64V4Token, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    // Use 512-bit: zero-extend 256-bit inputs into zmm, add, extract back.
    // This exercises real AVX-512 codegen, not just AVX2 under a V4 token.
    unsafe {
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let za = _mm512_castps256_ps512(va);
        let zb = _mm512_castps256_ps512(vb);
        let zsum = _mm512_add_ps(za, zb);
        let sum = _mm512_castps512_ps256(zsum);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), sum);
        out
    }
}

#[cfg(feature = "avx512")]
#[arcane]
fn loop_v3_summon_v4_arcane(_token: Desktop64, data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    let v4 = X64V4Token::summon().unwrap();
    let mut total = 0.0f32;
    for (a, b) in data.iter().zip(other.iter()) {
        let result = add_chunk_v4_arcane(v4, a, b); // V4 #[arcane] from V3 context
        total += result[0];
    }
    total
}

#[cfg(feature = "avx512")]
#[inline(never)]
pub fn nest_v3_up_v4_arcane(token: Desktop64, data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    loop_v3_summon_v4_arcane(token, data, other)
}

// Bare equivalent: V3 #[target_feature] calling V4 #[target_feature]
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512cd,avx512dq,avx512vl")]
unsafe fn add_chunk_v4_bare(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    unsafe {
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let za = _mm512_castps256_ps512(va);
        let zb = _mm512_castps256_ps512(vb);
        let zsum = _mm512_add_ps(za, zb);
        let sum = _mm512_castps512_ps256(zsum);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), sum);
        out
    }
}

#[cfg(feature = "avx512")]
#[target_feature(enable = "avx2,fma")]
unsafe fn loop_v3_bare_v4_bare(data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    unsafe {
        let mut total = 0.0f32;
        for (a, b) in data.iter().zip(other.iter()) {
            let result = add_chunk_v4_bare(a, b);
            total += result[0];
        }
        total
    }
}

#[cfg(feature = "avx512")]
#[inline(never)]
pub fn nest_v3_up_v4_bare(data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    unsafe { loop_v3_bare_v4_bare(data, other) }
}

// ============================================================================
// Real feature-level benchmarks: SSE vs AVX2 (with and without AVX2)
//
// Tests what happens when the inner function uses genuinely different
// instructions (128-bit SSE vs 256-bit AVX2), not just the same ops
// under a different token.
// ============================================================================

use archmage::X64V2Token;

// -- SSE (V2) add: 128-bit, processes 4 floats at a time ---------

#[arcane]
fn add_chunk_v2_arcane(_token: X64V2Token, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    unsafe {
        // Two 128-bit loads per half
        let a_lo = _mm_loadu_ps(a.as_ptr());
        let a_hi = _mm_loadu_ps(a.as_ptr().add(4));
        let b_lo = _mm_loadu_ps(b.as_ptr());
        let b_hi = _mm_loadu_ps(b.as_ptr().add(4));
        let sum_lo = _mm_add_ps(a_lo, b_lo);
        let sum_hi = _mm_add_ps(a_hi, b_hi);
        let mut out = [0.0f32; 8];
        _mm_storeu_ps(out.as_mut_ptr(), sum_lo);
        _mm_storeu_ps(out.as_mut_ptr().add(4), sum_hi);
        out
    }
}

#[target_feature(enable = "sse4.2")]
unsafe fn add_chunk_v2_bare(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    unsafe {
        let a_lo = _mm_loadu_ps(a.as_ptr());
        let a_hi = _mm_loadu_ps(a.as_ptr().add(4));
        let b_lo = _mm_loadu_ps(b.as_ptr());
        let b_hi = _mm_loadu_ps(b.as_ptr().add(4));
        let sum_lo = _mm_add_ps(a_lo, b_lo);
        let sum_hi = _mm_add_ps(a_hi, b_hi);
        let mut out = [0.0f32; 8];
        _mm_storeu_ps(out.as_mut_ptr(), sum_lo);
        _mm_storeu_ps(out.as_mut_ptr().add(4), sum_hi);
        out
    }
}

// -- V3 calling V2 #[arcane] (downgrade: AVX2 context calling SSE) -------
// Note: IntoConcreteToken is identity-only (V3.as_x64v2() = None),
// so we summon V2 directly. The CPU always has V2 if it has V3.

#[arcane]
fn loop_v3_down_v2_arcane(_token: Desktop64, data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    let v2 = X64V2Token::summon().unwrap();
    let mut total = 0.0f32;
    for (a, b) in data.iter().zip(other.iter()) {
        let result = add_chunk_v2_arcane(v2, a, b);
        total += result[0];
    }
    total
}

#[inline(never)]
pub fn nest_v3_down_v2_arcane(token: Desktop64, data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    loop_v3_down_v2_arcane(token, data, other)
}

// Bare: AVX2 context calling SSE
#[target_feature(enable = "avx2,fma")]
unsafe fn loop_v3_bare_down_v2_bare(data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    unsafe {
        let mut total = 0.0f32;
        for (a, b) in data.iter().zip(other.iter()) {
            let result = add_chunk_v2_bare(a, b);
            total += result[0];
        }
        total
    }
}

#[inline(never)]
pub fn nest_v3_down_v2_bare(data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    unsafe { loop_v3_bare_down_v2_bare(data, other) }
}

// -- V2 calling V3 #[arcane] (upgrade: SSE context calling AVX2) -------

#[arcane]
fn loop_v2_up_v3_arcane(_token: X64V2Token, data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    let v3 = Desktop64::summon().unwrap();
    let mut total = 0.0f32;
    for (a, b) in data.iter().zip(other.iter()) {
        let result = add_chunk_v3_arcane(v3, a, b);
        total += result[0];
    }
    total
}

#[inline(never)]
pub fn nest_v2_up_v3_arcane(token: X64V2Token, data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    loop_v2_up_v3_arcane(token, data, other)
}

// Bare: SSE context calling AVX2
#[target_feature(enable = "sse4.2")]
unsafe fn loop_v2_bare_up_v3_bare(data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    unsafe {
        let mut total = 0.0f32;
        for (a, b) in data.iter().zip(other.iter()) {
            let result = add_chunk_v3_bare(a, b);
            total += result[0];
        }
        total
    }
}

#[inline(never)]
pub fn nest_v2_up_v3_bare(data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    unsafe { loop_v2_bare_up_v3_bare(data, other) }
}

// -- V2→V2 same level (SSE only, control for V3→V3) ----------

#[arcane]
fn loop_v2_arcane_v2_arcane(token: X64V2Token, data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    let mut total = 0.0f32;
    for (a, b) in data.iter().zip(other.iter()) {
        let result = add_chunk_v2_arcane(token, a, b);
        total += result[0];
    }
    total
}

#[inline(never)]
pub fn nest_v2_v2_arcane(token: X64V2Token, data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    loop_v2_arcane_v2_arcane(token, data, other)
}

#[target_feature(enable = "sse4.2")]
unsafe fn loop_v2_bare_v2_bare(data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    unsafe {
        let mut total = 0.0f32;
        for (a, b) in data.iter().zip(other.iter()) {
            let result = add_chunk_v2_bare(a, b);
            total += result[0];
        }
        total
    }
}

#[inline(never)]
pub fn nest_v2_v2_bare(data: &[[f32; 8]], other: &[[f32; 8]]) -> f32 {
    unsafe { loop_v2_bare_v2_bare(data, other) }
}

// ============================================================================
// Criterion benchmark
// ============================================================================

use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_patterns(c: &mut Criterion) {
    let data: Vec<[f32; 8]> = (0..1000).map(|i| [i as f32; 8]).collect();
    let other: Vec<[f32; 8]> = (0..1000).map(|i| [i as f32 * 2.0; 8]).collect();

    if let Some(token) = Desktop64::summon() {
        c.bench_function("1_arcane_in_loop", |b| {
            b.iter(|| loop_with_arcane_in_loop(token, black_box(&data), black_box(&other)))
        });

        c.bench_function("2_rite_in_arcane", |b| {
            b.iter(|| loop_inside_arcane_with_rite(token, black_box(&data), black_box(&other)))
        });

        c.bench_function("3_manual_inline", |b| {
            b.iter(|| loop_inside_arcane_manual(token, black_box(&data), black_box(&other)))
        });

        c.bench_function("4_rite_direct_unsafe", |b| {
            b.iter(|| loop_calling_rite_directly(token, black_box(&data), black_box(&other)))
        });

        c.bench_function("7_bare_target_feature", |b| {
            b.iter(|| loop_bare_target_feature(black_box(&data), black_box(&other)))
        });
    } else {
        eprintln!("Desktop64 not available, skipping benchmarks");
    }

    // These don't need a token — proving wrapper overhead is negligible
    c.bench_function("5_scalar_wrapper", |b| {
        b.iter(|| loop_scalar_wrapper(black_box(&data), black_box(&other)))
    });

    c.bench_function("6_scalar_inline", |b| {
        b.iter(|| loop_scalar_inline(black_box(&data), black_box(&other)))
    });
}

fn bench_dct(c: &mut Criterion) {
    let coeffs = dct8_coefficients();
    let rows: Vec<[f32; 8]> = (0..100)
        .map(|i| {
            let mut r = [0.0f32; 8];
            for j in 0..8 {
                r[j] = (i * 8 + j) as f32;
            }
            r
        })
        .collect();

    if let Some(token) = Desktop64::summon() {
        c.bench_function("dct8_arcane_per_row", |b| {
            b.iter(|| dct_arcane_per_row(token, black_box(&coeffs), black_box(&rows)))
        });

        c.bench_function("dct8_rite_in_arcane", |b| {
            b.iter(|| dct_rite_in_arcane(token, black_box(&coeffs), black_box(&rows)))
        });

        c.bench_function("dct8_bare_target_feature", |b| {
            b.iter(|| dct_bare_target_feature(black_box(&coeffs), black_box(&rows)))
        });
    }
}

fn bench_nesting(c: &mut Criterion) {
    let data: Vec<[f32; 8]> = (0..1000).map(|i| [i as f32; 8]).collect();
    let other: Vec<[f32; 8]> = (0..1000).map(|i| [i as f32 * 2.0; 8]).collect();

    let v2_token = X64V2Token::summon();
    let v3_token = Desktop64::summon();

    if let Some(token) = v3_token {
        // -- Same-level V3→V3 ---
        c.bench_function("nest_v3_arcane_v3_arcane", |b| {
            b.iter(|| nest_v3_v3_arcane(token, black_box(&data), black_box(&other)))
        });
        c.bench_function("nest_v3_bare_v3_bare", |b| {
            b.iter(|| nest_v3_v3_bare(black_box(&data), black_box(&other)))
        });
        c.bench_function("nest_v3_arcane_v3_rite", |b| {
            b.iter(|| nest_v3_v3_rite(token, black_box(&data), black_box(&other)))
        });

        // -- V3 downgrade to V2 (AVX2 context calling SSE) ---
        c.bench_function("nest_v3_down_v2_arcane", |b| {
            b.iter(|| nest_v3_down_v2_arcane(token, black_box(&data), black_box(&other)))
        });
        c.bench_function("nest_v3_down_v2_bare", |b| {
            b.iter(|| nest_v3_down_v2_bare(black_box(&data), black_box(&other)))
        });
    }

    if let Some(v2) = v2_token {
        // -- Same-level V2→V2 ---
        c.bench_function("nest_v2_arcane_v2_arcane", |b| {
            b.iter(|| nest_v2_v2_arcane(v2, black_box(&data), black_box(&other)))
        });
        c.bench_function("nest_v2_bare_v2_bare", |b| {
            b.iter(|| nest_v2_v2_bare(black_box(&data), black_box(&other)))
        });

        // -- V2 upgrade to V3 (SSE context calling AVX2) ---
        if v3_token.is_some() {
            c.bench_function("nest_v2_up_v3_arcane", |b| {
                b.iter(|| nest_v2_up_v3_arcane(v2, black_box(&data), black_box(&other)))
            });
            c.bench_function("nest_v2_up_v3_bare", |b| {
                b.iter(|| nest_v2_up_v3_bare(black_box(&data), black_box(&other)))
            });
        }
    }

    #[cfg(feature = "avx512")]
    if let Some(v4_token) = X64V4Token::summon() {
        if let Some(token) = v3_token {
            // -- V4 downcast to V3 ---
            c.bench_function("nest_v4_down_v3_arcane", |b| {
                b.iter(|| nest_v4_down_v3_arcane(v4_token, black_box(&data), black_box(&other)))
            });
            c.bench_function("nest_v4_down_v3_bare", |b| {
                b.iter(|| nest_v4_down_v3_bare(black_box(&data), black_box(&other)))
            });

            // -- V3 upgrade to V4 (uses real AVX-512 zmm instructions) ---
            c.bench_function("nest_v3_up_v4_arcane", |b| {
                b.iter(|| nest_v3_up_v4_arcane(token, black_box(&data), black_box(&other)))
            });
            c.bench_function("nest_v3_up_v4_bare", |b| {
                b.iter(|| nest_v3_up_v4_bare(black_box(&data), black_box(&other)))
            });
        }
    }
}

criterion_group!(benches, bench_patterns, bench_dct, bench_nesting);
criterion_main!(benches);

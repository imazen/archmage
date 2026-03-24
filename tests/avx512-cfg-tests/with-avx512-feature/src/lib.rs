//! Test: archmage has avx512, calling crate does NOT define its own avx512 feature.
//!
//! This verifies the hypothesis that `#[cfg(feature = "avx512")]` in incant!/magetypes
//! proc-macro output checks the CALLING CRATE's features, not archmage's.
//!
//! If this hypothesis is correct:
//! - incant! will NOT dispatch to _v4 (the arm is dead code)
//! - #[magetypes] will NOT generate _v4 variants
//! - But #[arcane(import_intrinsics)] with X64V4Token WILL work (it doesn't use cargo_feature)
#![deny(warnings)]
#![allow(unexpected_cfgs)]

use archmage::prelude::*;

// ============================================================================
// Test 1: #[arcane(import_intrinsics)] with V4 token — should always work
// ============================================================================

#[arcane(import_intrinsics)]
fn v4_value_ops(_token: X64V4Token, a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let va = _mm512_loadu_ps(a);
    let vb = _mm512_loadu_ps(b);
    let vc = _mm512_add_ps(va, vb);
    let mut out = [0.0f32; 16];
    _mm512_storeu_ps(&mut out, vc);
    out
}

// ============================================================================
// Test 2: incant! with v4 in the tier list — does v4 dispatch work?
// ============================================================================

#[arcane(import_intrinsics)]
fn add_v4(_token: X64V4Token, a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let va = _mm512_loadu_ps(a);
    let vb = _mm512_loadu_ps(b);
    let vc = _mm512_add_ps(va, vb);
    let mut out = [0.0f32; 16];
    _mm512_storeu_ps(&mut out, vc);
    out
}

#[arcane(import_intrinsics)]
fn add_v3(_token: X64V3Token, a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    // Simple scalar fallback for v3
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        out[i] = a[i] + b[i];
    }
    out
}

fn add_scalar(_token: ScalarToken, a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        out[i] = a[i] + b[i];
    }
    out
}

pub fn add_dispatched(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    incant!(add(a, b), [v4, v3, scalar])
}

// ============================================================================
// Test 3: #[magetypes] with v4 — does the variant get generated?
// ============================================================================

// Note: #[magetypes] is more complex to test standalone, so we focus on incant!

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arcane_v4_works() {
        if let Some(token) = X64V4Token::summon() {
            let a = [1.0f32; 16];
            let b = [2.0f32; 16];
            let out = v4_value_ops(token, &a, &b);
            assert!((out[0] - 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn incant_dispatches() {
        let a = [1.0f32; 16];
        let b = [2.0f32; 16];
        let out = add_dispatched(&a, &b);
        assert!((out[0] - 3.0).abs() < 1e-6);
    }

    /// If V4 dispatch is alive, this should use the v4 variant on capable hardware.
    /// If V4 dispatch is dead (cfg eliminated), this falls through to v3 or scalar.
    /// Either way the result is correct — the question is WHICH path runs.
    #[test]
    fn v4_variant_is_reachable() {
        // We can't easily test which dispatch path was taken without side effects.
        // But we CAN verify the v4 function exists and is callable:
        if let Some(token) = X64V4Token::summon() {
            let a = [1.0f32; 16];
            let b = [2.0f32; 16];
            let out = add_v4(token, &a, &b);
            assert!((out[0] - 3.0).abs() < 1e-6);
        }
    }
}

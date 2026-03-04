//! Tests for `import_intrinsics` and `import_magetypes` parameters on #[arcane] and #[rite].

#![cfg(target_arch = "x86_64")]
#![allow(unused)]

use archmage::{SimdToken, X64V3Token, arcane, rite};

// =============================================================================
// #[arcane(import_intrinsics)] — auto-imports core::arch + safe_unaligned_simd
// =============================================================================

/// Verify that import_intrinsics brings core::arch::x86_64::* into scope.
/// Without this param, _mm256_setzero_ps would require an explicit `use`.
#[arcane(import_intrinsics)]
fn arcane_intrinsics_in_scope(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    // _mm256_setzero_ps and _mm256_add_ps come from core::arch::x86_64::*
    let v = _mm256_setzero_ps();
    let sum = _mm256_add_ps(v, v);
    // safe_unaligned_simd::x86_64::* provides safe load/store
    let loaded = safe_unaligned_simd::x86_64::_mm256_loadu_ps(data);
    let result = _mm256_add_ps(loaded, sum);
    let mut out = [0.0f32; 8];
    safe_unaligned_simd::x86_64::_mm256_storeu_ps(&mut out, result);
    out
}

#[test]
fn test_arcane_import_intrinsics() {
    if let Some(token) = X64V3Token::summon() {
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let output = arcane_intrinsics_in_scope(token, &input);
        assert_eq!(output, input);
    }
}

// =============================================================================
// #[arcane(import_magetypes)] — auto-imports magetypes::simd::{ns}::* etc.
// =============================================================================

/// Verify that import_magetypes brings the right namespace into scope.
/// X64V3Token maps to the "v3" namespace.
#[arcane(import_magetypes)]
fn arcane_magetypes_in_scope(token: X64V3Token, data: &[f32; 8]) -> f32 {
    // f32x8 comes from magetypes::simd::v3::*
    // F32x8Backend comes from magetypes::simd::backends::*
    let v = f32x8::load(token, data);
    v.reduce_add()
}

#[test]
fn test_arcane_import_magetypes() {
    if let Some(token) = X64V3Token::summon() {
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = arcane_magetypes_in_scope(token, &input);
        assert!((result - 36.0).abs() < 0.001);
    }
}

// =============================================================================
// #[arcane(import_intrinsics, import_magetypes)] — combined
// =============================================================================

#[arcane(import_intrinsics, import_magetypes)]
fn arcane_both_imports(token: X64V3Token, data: &[f32; 8]) -> f32 {
    // From import_intrinsics: core::arch intrinsics
    let zero = _mm256_setzero_ps();
    let _ = _mm256_add_ps(zero, zero);

    // From import_magetypes: magetypes types
    let v = f32x8::load(token, data);
    v.reduce_add()
}

#[test]
fn test_arcane_both_imports() {
    if let Some(token) = X64V3Token::summon() {
        let input = [1.0f32; 8];
        let result = arcane_both_imports(token, &input);
        assert!((result - 8.0).abs() < 0.001);
    }
}

// =============================================================================
// #[rite(import_intrinsics)] — same for rite
// =============================================================================

#[rite(import_intrinsics)]
fn rite_intrinsics_in_scope(_token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let loaded = safe_unaligned_simd::x86_64::_mm256_loadu_ps(data);
    let doubled = _mm256_add_ps(loaded, loaded);
    let mut out = [0.0f32; 8];
    safe_unaligned_simd::x86_64::_mm256_storeu_ps(&mut out, doubled);
    out
}

/// Entry point to call the rite function
#[arcane]
fn call_rite_intrinsics(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    rite_intrinsics_in_scope(token, data)
}

#[test]
fn test_rite_import_intrinsics() {
    if let Some(token) = X64V3Token::summon() {
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let output = call_rite_intrinsics(token, &input);
        assert_eq!(output, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    }
}

// =============================================================================
// #[rite(import_magetypes)] — magetypes in rite
// =============================================================================

#[rite(import_magetypes)]
fn rite_magetypes_in_scope(token: X64V3Token, data: &[f32; 8]) -> f32 {
    let v = f32x8::load(token, data);
    v.reduce_add()
}

#[arcane]
fn call_rite_magetypes(token: X64V3Token, data: &[f32; 8]) -> f32 {
    rite_magetypes_in_scope(token, data)
}

#[test]
fn test_rite_import_magetypes() {
    if let Some(token) = X64V3Token::summon() {
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = call_rite_magetypes(token, &input);
        assert!((result - 36.0).abs() < 0.001);
    }
}

// =============================================================================
// #[rite(import_intrinsics, import_magetypes)] — combined rite
// =============================================================================

#[rite(import_intrinsics, import_magetypes)]
fn rite_both_imports(token: X64V3Token, data: &[f32; 8]) -> f32 {
    let zero = _mm256_setzero_ps();
    let _ = _mm256_add_ps(zero, zero);
    let v = f32x8::load(token, data);
    v.reduce_add()
}

#[arcane]
fn call_rite_both(token: X64V3Token, data: &[f32; 8]) -> f32 {
    rite_both_imports(token, data)
}

#[test]
fn test_rite_both_imports() {
    if let Some(token) = X64V3Token::summon() {
        let input = [1.0f32; 8];
        let result = call_rite_both(token, &input);
        assert!((result - 8.0).abs() < 0.001);
    }
}

// =============================================================================
// Trait-bounded token with imports
// =============================================================================

use archmage::HasX64V2;

#[arcane(import_intrinsics)]
fn arcane_trait_bound_intrinsics(token: impl HasX64V2, data: &[f32; 4]) -> [f32; 4] {
    // SSE2 intrinsics from core::arch::x86_64::*
    let loaded = safe_unaligned_simd::x86_64::_mm_loadu_ps(data);
    let doubled = _mm_add_ps(loaded, loaded);
    let mut out = [0.0f32; 4];
    safe_unaligned_simd::x86_64::_mm_storeu_ps(&mut out, doubled);
    out
}

#[test]
fn test_trait_bound_import_intrinsics() {
    if let Some(token) = X64V3Token::summon() {
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let output = arcane_trait_bound_intrinsics(token, &input);
        assert_eq!(output, [2.0, 4.0, 6.0, 8.0]);
    }
}

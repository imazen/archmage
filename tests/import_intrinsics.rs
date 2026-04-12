//! Tests for the `import_intrinsics` parameter on #[arcane] and #[rite].
//!
//! `import_intrinsics` auto-injects `use archmage::intrinsics::<arch>::*` into
//! SIMD function bodies, eliminating boilerplate imports.
//!
//! For `import_magetypes` tests (which require magetypes), see
//! magetypes' own import_params test.

#![allow(unused)]

#[cfg(target_arch = "x86_64")]
mod x86_tests {
    use archmage::{HasX64V2, SimdToken, X64V2Token, X64V3Token, arcane, rite};

    // =========================================================================
    // Basic: #[arcane(import_intrinsics)]
    // =========================================================================

    /// import_intrinsics brings archmage::intrinsics::x86_64::* into scope,
    /// which includes core::arch types/value ops + safe memory ops.
    #[arcane(import_intrinsics)]
    fn arcane_intrinsics_basic(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
        // Value intrinsics from core::arch — safe inside #[target_feature]
        let v = _mm256_setzero_ps();
        let sum = _mm256_add_ps(v, v);
        // Memory ops resolve to safe reference-based versions automatically
        let loaded = _mm256_loadu_ps(data);
        let result = _mm256_add_ps(loaded, sum);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(&mut out, result);
        out
    }

    #[test]
    fn test_arcane_import_intrinsics() {
        if let Some(token) = X64V3Token::summon() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let output = arcane_intrinsics_basic(token, &input);
            assert_eq!(output, input);
        }
    }

    // =========================================================================
    // #[rite(import_intrinsics)]
    // =========================================================================

    #[rite(import_intrinsics)]
    fn rite_intrinsics(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
        let loaded = _mm256_loadu_ps(data);
        let doubled = _mm256_add_ps(loaded, loaded);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(&mut out, doubled);
        out
    }

    // =========================================================================
    // Trait bounds: impl HasX64V2 with import_intrinsics
    // =========================================================================

    /// Trait-bounded tokens work with import_intrinsics — the macro derives
    /// the architecture from the trait (HasX64V2 → x86_64).
    #[arcane(import_intrinsics)]
    fn trait_bound_impl(token: impl HasX64V2, data: &[f32; 4]) -> [f32; 4] {
        let loaded = _mm_loadu_ps(data);
        let doubled = _mm_add_ps(loaded, loaded);
        let mut out = [0.0f32; 4];
        _mm_storeu_ps(&mut out, doubled);
        out
    }

    #[test]
    fn test_trait_bound_intrinsics() {
        if let Some(token) = X64V3Token::summon() {
            let input = [1.0f32, 2.0, 3.0, 4.0];
            let output = trait_bound_impl(token, &input);
            assert_eq!(output, [2.0, 4.0, 6.0, 8.0]);
        }
    }

    // =========================================================================
    // Generic type parameter: <T: HasX64V2> with import_intrinsics
    // =========================================================================

    #[arcane(import_intrinsics)]
    fn generic_bound_intrinsics<T: HasX64V2>(token: T, data: &[f32; 4]) -> [f32; 4] {
        let loaded = _mm_loadu_ps(data);
        let negated = _mm_sub_ps(_mm_setzero_ps(), loaded);
        let mut out = [0.0f32; 4];
        _mm_storeu_ps(&mut out, negated);
        out
    }

    #[test]
    fn test_generic_bounds() {
        if let Some(token) = X64V3Token::summon() {
            let input = [1.0f32, 2.0, 3.0, 4.0];
            let output = generic_bound_intrinsics(token, &input);
            assert_eq!(output, [-1.0, -2.0, -3.0, -4.0]);
        }
    }

    // =========================================================================
    // Wildcard token: _: X64V3Token
    // =========================================================================

    #[arcane(import_intrinsics)]
    fn wildcard_token(_: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
        let v = _mm256_loadu_ps(data);
        let doubled = _mm256_add_ps(v, v);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(&mut out, doubled);
        out
    }

    #[test]
    fn test_wildcard_token() {
        if let Some(token) = X64V3Token::summon() {
            let input = [1.0f32; 8];
            let output = wildcard_token(token, &input);
            assert_eq!(output, [2.0f32; 8]);
        }
    }

    // =========================================================================
    // X64V2Token: import_intrinsics works
    // =========================================================================

    /// V2 maps to x86_64 arch.
    /// import_intrinsics gives SSE/SSE2/SSE3/etc. intrinsics.
    #[arcane(import_intrinsics)]
    fn v2_intrinsics(token: X64V2Token, data: &[f32; 4]) -> [f32; 4] {
        let loaded = _mm_loadu_ps(data);
        let doubled = _mm_add_ps(loaded, loaded);
        let mut out = [0.0f32; 4];
        _mm_storeu_ps(&mut out, doubled);
        out
    }

    #[test]
    fn test_v2_intrinsics() {
        if let Some(token) = X64V2Token::summon() {
            let data = [1.0f32, 2.0, 3.0, 4.0];
            let result = v2_intrinsics(token, &data);
            assert_eq!(result, [2.0, 4.0, 6.0, 8.0]);
        }
    }

    // =========================================================================
    // AVX-512 tokens: import_intrinsics works
    // =========================================================================

    #[cfg(feature = "avx512")]
    mod avx512_tests {
        use archmage::{SimdToken, X64V4Token, arcane};

        /// V4 maps to v4 namespace. import_intrinsics brings all x86_64 intrinsics
        /// (including AVX-512) into scope.
        #[arcane(import_intrinsics)]
        fn v4_intrinsics(token: X64V4Token, data: &[f32; 8]) -> [f32; 8] {
            let v = _mm256_loadu_ps(data);
            let doubled = _mm256_add_ps(v, v);
            let mut out = [0.0f32; 8];
            _mm256_storeu_ps(&mut out, doubled);
            out
        }

        #[test]
        fn test_v4_intrinsics() {
            if let Some(token) = X64V4Token::summon() {
                let data = [1.0f32; 8];
                let result = v4_intrinsics(token, &data);
                assert_eq!(result, [2.0f32; 8]);
            }
        }
    }

    // =========================================================================
    // Explicit imports coexist: user imports don't conflict with auto-imports
    // =========================================================================

    mod coexist_with_explicit_imports {
        // User already has some arch imports — auto-imports shouldn't conflict
        use std::arch::x86_64::_mm256_setzero_ps;

        use archmage::{SimdToken, X64V3Token, arcane};

        #[arcane(import_intrinsics)]
        fn with_existing_imports(token: X64V3Token) -> bool {
            // _mm256_setzero_ps from user's explicit import AND from auto-import
            // Rust resolves this fine — both refer to the same item
            let v = _mm256_setzero_ps();
            let _ = _mm256_add_ps(v, v); // from auto-import only
            true
        }

        #[test]
        fn test_coexist() {
            if let Some(token) = X64V3Token::summon() {
                assert!(with_existing_imports(token));
            }
        }
    }
}

// =============================================================================
// ARM tests (compile on aarch64)
// =============================================================================

#[cfg(target_arch = "aarch64")]
mod arm_tests {
    use archmage::{NeonToken, SimdToken, arcane};

    #[arcane(import_intrinsics)]
    fn neon_intrinsics(token: NeonToken, data: &[f32; 4]) -> [f32; 4] {
        // core::arch::aarch64::* in scope
        let v = vld1q_f32(data);
        let doubled = vaddq_f32(v, v);
        let mut out = [0.0f32; 4];
        vst1q_f32(&mut out, doubled);
        out
    }

    #[test]
    fn test_neon_import_intrinsics() {
        if let Some(token) = NeonToken::summon() {
            let input = [1.0f32, 2.0, 3.0, 4.0];
            let output = neon_intrinsics(token, &input);
            assert_eq!(output, [2.0, 4.0, 6.0, 8.0]);
        }
    }
}

//! Tests for `import_intrinsics` and `import_magetypes` parameters on #[arcane] and #[rite].
//!
//! These parameters auto-inject `use` statements into SIMD function bodies,
//! eliminating boilerplate imports that the macro already knows how to derive
//! from the token type.
//!
//! Key limitations to understand:
//! - Imports are injected into the function **body**, not the signature.
//!   Types in parameters/return position must be imported normally.
//! - `import_magetypes` imports pre-specialized types (e.g., v3::f32x8 expects X64V3Token).
//!   For trait-bounded functions, use generic types from `magetypes::simd::generic` instead.

#![allow(unused)]

// =============================================================================
// x86_64 tests
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod x86_tests {
    use archmage::{HasX64V2, SimdToken, X64V2Token, X64V3Token, arcane, rite};

    // =========================================================================
    // Basic: #[arcane(import_intrinsics)]
    // =========================================================================

    /// import_intrinsics brings core::arch::x86_64::* and safe_unaligned_simd::x86_64::*
    /// into scope. Without this, you'd need explicit `use std::arch::x86_64::*;`.
    #[arcane(import_intrinsics)]
    fn arcane_intrinsics_basic(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
        // Value intrinsics from core::arch — safe inside #[target_feature]
        let v = _mm256_setzero_ps();
        let sum = _mm256_add_ps(v, v);
        // safe_unaligned_simd for memory ops — safe reference-based API
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
            let output = arcane_intrinsics_basic(token, &input);
            assert_eq!(output, input);
        }
    }

    // =========================================================================
    // Basic: #[arcane(import_magetypes)]
    // =========================================================================

    /// import_magetypes brings the token-appropriate magetypes namespace into scope.
    /// X64V3Token maps to `magetypes::simd::v3::*` which exports f32x8, i32x8, etc.
    #[arcane(import_magetypes)]
    fn arcane_magetypes_basic(token: X64V3Token, data: &[f32; 8]) -> f32 {
        // f32x8 comes from magetypes::simd::v3::* — pre-specialized for X64V3Token
        let v = f32x8::load(token, data);
        v.reduce_add()
    }

    #[test]
    fn test_arcane_import_magetypes() {
        if let Some(token) = X64V3Token::summon() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let result = arcane_magetypes_basic(token, &input);
            assert!((result - 36.0).abs() < 0.001);
        }
    }

    // =========================================================================
    // Combined: #[arcane(import_intrinsics, import_magetypes)]
    // =========================================================================

    /// Both imports together — mix raw intrinsics and magetypes in the same function.
    #[arcane(import_intrinsics, import_magetypes)]
    fn arcane_both_imports(token: X64V3Token, data: &[f32; 8]) -> f32 {
        // Raw intrinsics for fine-grained control
        let zero = _mm256_setzero_ps();
        let _ = _mm256_add_ps(zero, zero);

        // Magetypes for higher-level operations
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

    // =========================================================================
    // #[rite] variants — all three combinations
    // =========================================================================

    #[rite(import_intrinsics)]
    fn rite_intrinsics(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
        let loaded = safe_unaligned_simd::x86_64::_mm256_loadu_ps(data);
        let doubled = _mm256_add_ps(loaded, loaded);
        let mut out = [0.0f32; 8];
        safe_unaligned_simd::x86_64::_mm256_storeu_ps(&mut out, doubled);
        out
    }

    #[rite(import_magetypes)]
    fn rite_magetypes(token: X64V3Token, data: &[f32; 8]) -> f32 {
        let v = f32x8::load(token, data);
        v.reduce_add()
    }

    #[rite(import_intrinsics, import_magetypes)]
    fn rite_both(token: X64V3Token, data: &[f32; 8]) -> f32 {
        let zero = _mm256_setzero_ps();
        let _ = _mm256_add_ps(zero, zero);
        let v = f32x8::load(token, data);
        v.reduce_add()
    }

    /// Entry point for calling #[rite] helpers
    #[arcane]
    fn call_rite_variants(token: X64V3Token, data: &[f32; 8]) -> (f32, f32) {
        let _doubled = rite_intrinsics(token, data);
        let sum = rite_magetypes(token, data);
        let combined = rite_both(token, data);
        (sum, combined)
    }

    #[test]
    fn test_rite_variants() {
        if let Some(token) = X64V3Token::summon() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let (sum, combined) = call_rite_variants(token, &input);
            assert!((sum - 36.0).abs() < 0.001);
            assert!((combined - 36.0).abs() < 0.001);
        }
    }

    // =========================================================================
    // Optimal pattern: #[arcane] entry + #[rite] helpers, all with imports
    // =========================================================================

    /// This is the recommended pattern for a real SIMD module:
    ///
    /// 1. `#[arcane(import_intrinsics, import_magetypes)]` at entry point
    /// 2. `#[rite(import_magetypes)]` on helpers
    /// 3. No manual `use std::arch::x86_64::*` or `use magetypes::simd::v3::*`
    ///
    /// Each function imports only what it needs. Helpers that only use magetypes
    /// don't need import_intrinsics. The entry point might need both for
    /// low-level setup + high-level processing.
    mod optimal_pattern {
        use archmage::{SimdToken, X64V3Token, arcane, rite};

        // Entry point — receives token from summon(), safe wrapper generated
        #[arcane(import_intrinsics, import_magetypes)]
        fn process_batch(token: X64V3Token, data: &[f32]) -> f32 {
            let mut total = 0.0f32;
            for chunk in data.chunks_exact(8) {
                total += scale_chunk(token, chunk.try_into().unwrap());
            }
            total
        }

        // Helper — only needs magetypes, no raw intrinsics
        #[rite(import_magetypes)]
        fn scale_chunk(token: X64V3Token, data: &[f32; 8]) -> f32 {
            let v = f32x8::load(token, data);
            let two = f32x8::splat(token, 2.0);
            (v * two).reduce_add()
        }

        #[test]
        fn test_optimal_pattern() {
            if let Some(token) = X64V3Token::summon() {
                let data = [1.0f32; 16];
                let result = process_batch(token, &data);
                // 16 values * 1.0 * 2.0 = 32.0
                assert!((result - 32.0).abs() < 0.001);
            }
        }
    }

    // =========================================================================
    // Real-world pattern: intrinsics for FMA + magetypes for reduction
    // =========================================================================

    /// Sometimes you need raw intrinsics for operations magetypes doesn't
    /// expose directly, combined with magetypes for ergonomic load/reduce.
    mod mixed_intrinsics_and_magetypes {
        use archmage::{SimdToken, X64V3Token, arcane};

        #[arcane(import_intrinsics, import_magetypes)]
        fn fma_then_reduce(token: X64V3Token, a: &[f32; 8], b: &[f32; 8], c: &[f32; 8]) -> f32 {
            // Use safe_unaligned_simd for loading raw __m256 values
            let va = safe_unaligned_simd::x86_64::_mm256_loadu_ps(a);
            let vb = safe_unaligned_simd::x86_64::_mm256_loadu_ps(b);
            let vc = safe_unaligned_simd::x86_64::_mm256_loadu_ps(c);

            // Raw FMA intrinsic: a * b + c
            let fma_result = _mm256_fmadd_ps(va, vb, vc);

            // Store back, then use magetypes for reduction
            let mut tmp = [0.0f32; 8];
            safe_unaligned_simd::x86_64::_mm256_storeu_ps(&mut tmp, fma_result);
            let v = f32x8::load(token, &tmp);
            v.reduce_add()
        }

        #[test]
        fn test_mixed() {
            if let Some(token) = X64V3Token::summon() {
                let a = [2.0f32; 8];
                let b = [3.0f32; 8];
                let c = [1.0f32; 8];
                let result = fma_then_reduce(token, &a, &b, &c);
                // (2*3+1) * 8 = 56
                assert!((result - 56.0).abs() < 0.001);
            }
        }
    }

    // =========================================================================
    // Multi-width: import_magetypes gives access to ALL widths in the namespace
    // =========================================================================

    /// The v3 namespace exports f32x4, f32x8, i32x4, i32x8, i8x16, i8x32, etc.
    /// import_magetypes brings them all into scope.
    #[arcane(import_magetypes)]
    fn multi_width_types(token: X64V3Token, data4: &[f32; 4], data8: &[f32; 8]) -> f32 {
        // 128-bit (native on V3)
        let v4 = f32x4::load(token, data4);
        let sum4 = v4.reduce_add();

        // 256-bit (native on V3)
        let v8 = f32x8::load(token, data8);
        let sum8 = v8.reduce_add();

        sum4 + sum8
    }

    #[test]
    fn test_multi_width() {
        if let Some(token) = X64V3Token::summon() {
            let d4 = [1.0f32, 2.0, 3.0, 4.0];
            let d8 = [1.0f32; 8];
            let result = multi_width_types(token, &d4, &d8);
            assert!((result - 18.0).abs() < 0.001); // 10 + 8
        }
    }

    // =========================================================================
    // Integer types from import_magetypes
    // =========================================================================

    #[arcane(import_magetypes)]
    fn integer_types(token: X64V3Token, data: &[i32; 8]) -> i32 {
        let v = i32x8::load(token, data);
        v.reduce_add()
    }

    #[test]
    fn test_integer_types() {
        if let Some(token) = X64V3Token::summon() {
            let data = [1i32, 2, 3, 4, 5, 6, 7, 8];
            let result = integer_types(token, &data);
            assert_eq!(result, 36);
        }
    }

    // =========================================================================
    // Trait bounds: impl HasX64V2 with import_intrinsics
    // =========================================================================

    /// Trait-bounded tokens work with import_intrinsics — the macro derives
    /// the architecture from the trait (HasX64V2 → x86_64).
    #[arcane(import_intrinsics)]
    fn trait_bound_impl(token: impl HasX64V2, data: &[f32; 4]) -> [f32; 4] {
        let loaded = safe_unaligned_simd::x86_64::_mm_loadu_ps(data);
        let doubled = _mm_add_ps(loaded, loaded);
        let mut out = [0.0f32; 4];
        safe_unaligned_simd::x86_64::_mm_storeu_ps(&mut out, doubled);
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
        let loaded = safe_unaligned_simd::x86_64::_mm_loadu_ps(data);
        let negated = _mm_sub_ps(_mm_setzero_ps(), loaded);
        let mut out = [0.0f32; 4];
        safe_unaligned_simd::x86_64::_mm_storeu_ps(&mut out, negated);
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
        let v = safe_unaligned_simd::x86_64::_mm256_loadu_ps(data);
        let doubled = _mm256_add_ps(v, v);
        let mut out = [0.0f32; 8];
        safe_unaligned_simd::x86_64::_mm256_storeu_ps(&mut out, doubled);
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
    // Method on struct (sibling mode)
    // =========================================================================

    struct Processor {
        scale: f32,
    }

    impl Processor {
        #[arcane(import_intrinsics, import_magetypes)]
        fn process(&self, token: X64V3Token, data: &[f32; 8]) -> f32 {
            let v = f32x8::load(token, data);
            let scale = f32x8::splat(token, self.scale);
            let scaled = v * scale;
            scaled.reduce_add()
        }
    }

    #[test]
    fn test_method_imports() {
        if let Some(token) = X64V3Token::summon() {
            let proc = Processor { scale: 2.0 };
            let data = [1.0f32; 8];
            let result = proc.process(token, &data);
            assert!((result - 16.0).abs() < 0.001);
        }
    }

    // =========================================================================
    // Trait impl with _self = Type (nested mode)
    // =========================================================================

    trait SimdReduce {
        fn reduce_sum(&self, token: X64V3Token, data: &[f32; 8]) -> f32;
    }

    struct Reducer;

    impl SimdReduce for Reducer {
        #[arcane(_self = Reducer, import_magetypes)]
        fn reduce_sum(&self, token: X64V3Token, data: &[f32; 8]) -> f32 {
            let v = f32x8::load(token, data);
            v.reduce_add()
        }
    }

    #[test]
    fn test_nested_mode_imports() {
        if let Some(token) = X64V3Token::summon() {
            let r = Reducer;
            let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let result = r.reduce_sum(token, &data);
            assert!((result - 36.0).abs() < 0.001);
        }
    }

    // =========================================================================
    // Stub + imports: function compiles on all arches, imports on native
    // =========================================================================

    #[arcane(stub, import_intrinsics, import_magetypes)]
    fn stubbed_with_imports(token: X64V3Token, data: &[f32; 8]) -> f32 {
        let v = f32x8::load(token, data);
        v.reduce_add()
    }

    #[test]
    fn test_stub_with_imports() {
        if let Some(token) = X64V3Token::summon() {
            let data = [1.0f32; 8];
            let result = stubbed_with_imports(token, &data);
            assert!((result - 8.0).abs() < 0.001);
        }
    }

    // =========================================================================
    // X64V2Token: import_intrinsics works, import_magetypes works
    // =========================================================================

    /// V2 maps to "v3" namespace and x86_64 arch.
    /// import_intrinsics gives SSE/SSE2/SSE3/etc. intrinsics.
    #[arcane(import_intrinsics)]
    fn v2_intrinsics(token: X64V2Token, data: &[f32; 4]) -> [f32; 4] {
        let loaded = safe_unaligned_simd::x86_64::_mm_loadu_ps(data);
        let doubled = _mm_add_ps(loaded, loaded);
        let mut out = [0.0f32; 4];
        safe_unaligned_simd::x86_64::_mm_storeu_ps(&mut out, doubled);
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
    // AVX-512 tokens: import_intrinsics works, import_magetypes uses v4 ns
    // =========================================================================

    #[cfg(feature = "avx512")]
    mod avx512_tests {
        use archmage::{SimdToken, X64V4Token, arcane};

        /// V4 maps to v4 namespace. import_intrinsics brings all x86_64 intrinsics
        /// (including AVX-512) into scope.
        #[arcane(import_intrinsics)]
        fn v4_intrinsics(token: X64V4Token, data: &[f32; 8]) -> [f32; 8] {
            let v = safe_unaligned_simd::x86_64::_mm256_loadu_ps(data);
            let doubled = _mm256_add_ps(v, v);
            let mut out = [0.0f32; 8];
            safe_unaligned_simd::x86_64::_mm256_storeu_ps(&mut out, doubled);
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

    // =========================================================================
    // Composability: #[arcane] entry with multiple #[rite] helpers
    // =========================================================================

    /// Real-world composability pattern:
    /// - Entry point imports both intrinsics and magetypes
    /// - Each helper imports only what it needs
    /// - Everything inlines into one optimization region
    mod composable_helpers {
        use archmage::{SimdToken, X64V3Token, arcane, rite};

        #[rite(import_magetypes)]
        fn normalize(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
            let v = f32x8::load(token, data);
            let sum = v.reduce_add();
            let inv = f32x8::splat(token, 1.0 / sum);
            (v * inv).to_array()
        }

        #[rite(import_magetypes)]
        fn scale(token: X64V3Token, data: &[f32; 8], factor: f32) -> [f32; 8] {
            let v = f32x8::load(token, data);
            let s = f32x8::splat(token, factor);
            (v * s).to_array()
        }

        #[arcane(import_magetypes)]
        fn normalize_and_scale(token: X64V3Token, data: &[f32; 8], factor: f32) -> [f32; 8] {
            let normed = normalize(token, data);
            scale(token, &normed, factor)
        }

        #[test]
        fn test_composable() {
            if let Some(token) = X64V3Token::summon() {
                let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
                let result = normalize_and_scale(token, &data, 36.0);
                // Each element: (x / 36.0) * 36.0 = x
                for (i, &v) in result.iter().enumerate() {
                    assert!(
                        (v - data[i]).abs() < 0.01,
                        "element {i}: expected {}, got {v}",
                        data[i]
                    );
                }
            }
        }
    }

    // =========================================================================
    // Backends re-export: import_magetypes brings backend traits too
    // =========================================================================

    /// backends::* exports backend traits (F32x8Backend, etc.) and token aliases
    /// (x64v3, neon, scalar). Useful for writing generic helpers within
    /// a concrete #[arcane] entry point.
    mod backends_in_scope {
        use archmage::{SimdToken, X64V3Token, arcane};

        #[arcane(import_magetypes)]
        fn use_backend_trait(token: X64V3Token, data: &[f32; 8]) -> f32 {
            // F32x8Backend is from backends::*
            // Can write a generic inner helper that's #[inline(always)]
            #[inline(always)]
            fn inner<T: F32x8Backend>(t: T, d: &[f32; 8]) -> f32 {
                magetypes::simd::generic::f32x8::<T>::load(t, d).reduce_add()
            }
            inner(token, data)
        }

        #[test]
        fn test_backend_trait_in_scope() {
            if let Some(token) = X64V3Token::summon() {
                let data = [1.0f32; 8];
                let result = use_backend_trait(token, &data);
                assert!((result - 8.0).abs() < 0.001);
            }
        }
    }

    // =========================================================================
    // Width constants: import_magetypes provides LANES_* constants
    // =========================================================================

    #[arcane(import_magetypes)]
    fn use_width_constants(token: X64V3Token) -> (usize, usize) {
        // v3 namespace defines these based on the token's native width
        (LANES_F32, LANES_F64) // 8, 4 for V3 (256-bit)
    }

    #[test]
    fn test_width_constants() {
        if let Some(token) = X64V3Token::summon() {
            let (f32_lanes, f64_lanes) = use_width_constants(token);
            assert_eq!(f32_lanes, 8);
            assert_eq!(f64_lanes, 4);
        }
    }

    // =========================================================================
    // Natural-width alias: f32xN from import_magetypes
    // =========================================================================

    /// Each namespace exports f32xN/f64xN/i32xN aliases for the token's native
    /// SIMD width. V3's f32xN is f32x8 (256-bit).
    #[arcane(import_magetypes)]
    fn use_natural_width(token: X64V3Token, data: &[f32; 8]) -> f32 {
        // f32xN is a type alias for the natural width — f32x8 on V3
        let v = f32xN::load(token, data);
        v.reduce_add()
    }

    #[test]
    fn test_natural_width_alias() {
        if let Some(token) = X64V3Token::summon() {
            let data = [1.0f32; 8];
            let result = use_natural_width(token, &data);
            assert!((result - 8.0).abs() < 0.001);
        }
    }

    // =========================================================================
    // Token type alias: v3::Token = X64V3Token
    // =========================================================================

    /// The namespace also exports a `Token` type alias for the concrete token.
    /// This is useful with #[magetypes] but also available via import_magetypes.
    #[arcane(import_magetypes)]
    fn use_token_alias(token: X64V3Token) -> &'static str {
        // Token is a type alias for X64V3Token in the v3 namespace
        let _: Token = token; // Proves Token == X64V3Token
        "works"
    }

    #[test]
    fn test_token_alias() {
        if let Some(token) = X64V3Token::summon() {
            assert_eq!(use_token_alias(token), "works");
        }
    }
}

// =============================================================================
// ARM tests (compile on aarch64)
// =============================================================================

#[cfg(target_arch = "aarch64")]
mod arm_tests {
    use archmage::{NeonToken, SimdToken, arcane, rite};

    #[arcane(import_intrinsics)]
    fn neon_intrinsics(token: NeonToken, data: &[f32; 4]) -> [f32; 4] {
        // core::arch::aarch64::* in scope
        let v = safe_unaligned_simd::aarch64::vld1q_f32(data);
        let doubled = vaddq_f32(v, v);
        let mut out = [0.0f32; 4];
        safe_unaligned_simd::aarch64::vst1q_f32(&mut out, doubled);
        out
    }

    #[arcane(import_magetypes)]
    fn neon_magetypes(token: NeonToken, data: &[f32; 4]) -> f32 {
        // magetypes::simd::neon::* in scope
        let v = f32x4::load(token, data);
        v.reduce_add()
    }

    #[rite(import_intrinsics, import_magetypes)]
    fn neon_helper(token: NeonToken, data: &[f32; 4]) -> f32 {
        let v = f32x4::load(token, data);
        v.reduce_add()
    }

    #[test]
    fn test_neon_import_intrinsics() {
        if let Some(token) = NeonToken::summon() {
            let input = [1.0f32, 2.0, 3.0, 4.0];
            let output = neon_intrinsics(token, &input);
            assert_eq!(output, [2.0, 4.0, 6.0, 8.0]);
        }
    }

    #[test]
    fn test_neon_import_magetypes() {
        if let Some(token) = NeonToken::summon() {
            let input = [1.0f32, 2.0, 3.0, 4.0];
            let result = neon_magetypes(token, &input);
            assert!((result - 10.0).abs() < 0.001);
        }
    }
}

// =============================================================================
// WASM tests (compile on wasm32)
// =============================================================================

#[cfg(target_arch = "wasm32")]
mod wasm_tests {
    use archmage::{SimdToken, Wasm128Token, arcane};

    #[arcane(import_magetypes)]
    fn wasm_magetypes(token: Wasm128Token, data: &[f32; 4]) -> f32 {
        // magetypes::simd::wasm128::* in scope
        let v = f32x4::load(token, data);
        v.reduce_add()
    }

    #[test]
    fn test_wasm_import_magetypes() {
        if let Some(token) = Wasm128Token::summon() {
            let data = [1.0f32, 2.0, 3.0, 4.0];
            let result = wasm_magetypes(token, &data);
            assert!((result - 10.0).abs() < 0.001);
        }
    }
}

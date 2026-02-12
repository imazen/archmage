//! Prove that `incant!` works with manually-defined partial variant sets.
//!
//! Key insight: `incant!` generates references to ALL suffixed variants
//! (`_v3`, `_v4`, `_neon`, `_wasm128`, `_scalar`), but each is wrapped in
//! `#[cfg]` gates. On x86_64, the compiler never sees `_neon` or `_wasm128`
//! references. You only need to define the variants for your platform.
//!
//! These tests define variants MANUALLY (no `#[magetypes]`) to prove this.

// =============================================================================
// Test 1: Only _v3, _scalar, and _v4 when avx512 is enabled
// No _neon, no _wasm128 — those cfg blocks are eliminated on x86_64
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod x86_minimal {
    use archmage::{ScalarToken, SimdToken, X64V3Token, arcane, incant};

    #[arcane]
    fn add_one_v3(_t: X64V3Token, x: f32) -> f32 {
        x + 1.0
    }

    // When avx512 feature is enabled, incant! generates a _v4 call
    #[cfg(feature = "avx512")]
    #[arcane]
    fn add_one_v4(_t: archmage::X64V4Token, x: f32) -> f32 {
        x + 1.0
    }

    fn add_one_scalar(_token: ScalarToken, x: f32) -> f32 {
        x + 1.0
    }

    // incant! compiles without _neon or _wasm128 defined — those cfg blocks
    // are eliminated on x86_64
    pub fn add_one(x: f32) -> f32 {
        incant!(add_one(x))
    }

    #[test]
    fn incant_works_without_neon_or_wasm_variants() {
        assert_eq!(add_one(41.0), 42.0);
    }

    #[test]
    fn incant_dispatches_to_v3_when_available() {
        // On any x86_64 CPU with AVX2+FMA, this should use _v3
        if X64V3Token::summon().is_some() {
            assert_eq!(add_one(0.0), 1.0);
        }
    }
}

// =============================================================================
// Test 2: All x86 variants (v3, v4, scalar)
// =============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
mod x86_with_avx512 {
    use archmage::{ScalarToken, X64V3Token, X64V4Token, arcane, incant};

    #[arcane]
    fn double_v3(_t: X64V3Token, x: i32) -> i32 {
        x * 2
    }

    #[arcane]
    fn double_v4(_t: X64V4Token, x: i32) -> i32 {
        x * 2
    }

    fn double_scalar(_token: ScalarToken, x: i32) -> i32 {
        x * 2
    }

    pub fn double(x: i32) -> i32 {
        incant!(double(x))
    }

    #[test]
    fn incant_with_v3_v4_and_scalar() {
        assert_eq!(double(21), 42);
    }
}

// =============================================================================
// Test 3: ScalarToken dispatch (direct call, not via incant!)
// =============================================================================

mod scalar_only {
    use archmage::ScalarToken;

    fn negate_scalar(_token: ScalarToken, x: i32) -> i32 {
        -x
    }

    // On x86_64, incant! requires at least _v3 + _scalar (and _v4 with avx512).
    // So we can't use incant! with only scalar. But ScalarToken itself always works.

    #[test]
    fn scalar_token_dispatch() {
        let result = negate_scalar(ScalarToken, 42);
        assert_eq!(result, -42);
    }
}

// =============================================================================
// Test 4: Passthrough mode with partial variants
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod passthrough_partial {
    use archmage::{IntoConcreteToken, ScalarToken, SimdToken, X64V3Token, arcane, incant};

    #[arcane]
    fn square_v3(_t: X64V3Token, x: f32) -> f32 {
        x * x
    }

    #[cfg(feature = "avx512")]
    #[arcane]
    fn square_v4(_t: archmage::X64V4Token, x: f32) -> f32 {
        x * x
    }

    fn square_scalar(_token: ScalarToken, x: f32) -> f32 {
        x * x
    }

    fn square_with_token<T: IntoConcreteToken>(token: T, x: f32) -> f32 {
        incant!(square(x) with token)
    }

    #[test]
    fn passthrough_with_scalar_token() {
        let result = square_with_token(ScalarToken, 7.0);
        assert_eq!(result, 49.0);
    }

    #[test]
    fn passthrough_with_x64v3_token() {
        if let Some(token) = X64V3Token::summon() {
            let result = square_with_token(token, 7.0);
            assert_eq!(result, 49.0);
        }
    }
}

// =============================================================================
// Test 5: Multiple incant! calls chained with partial variants
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod chained_incant {
    use archmage::{ScalarToken, SimdToken, X64V3Token, arcane, incant};

    #[arcane]
    fn step_a_v3(_t: X64V3Token, x: f32) -> f32 {
        x + 10.0
    }

    #[cfg(feature = "avx512")]
    #[arcane]
    fn step_a_v4(_t: archmage::X64V4Token, x: f32) -> f32 {
        x + 10.0
    }

    fn step_a_scalar(_token: ScalarToken, x: f32) -> f32 {
        x + 10.0
    }

    #[arcane]
    fn step_b_v3(_t: X64V3Token, x: f32) -> f32 {
        x * 2.0
    }

    #[cfg(feature = "avx512")]
    #[arcane]
    fn step_b_v4(_t: archmage::X64V4Token, x: f32) -> f32 {
        x * 2.0
    }

    fn step_b_scalar(_token: ScalarToken, x: f32) -> f32 {
        x * 2.0
    }

    pub fn pipeline(x: f32) -> f32 {
        let intermediate = incant!(step_a(x));
        incant!(step_b(intermediate))
    }

    #[test]
    fn chained_incant_with_partial_variants() {
        // (5 + 10) * 2 = 30
        assert_eq!(pipeline(5.0), 30.0);
    }
}

// =============================================================================
// Test 6: Multiple arguments with partial variants
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod multi_arg_partial {
    use archmage::{ScalarToken, X64V3Token, arcane, incant};

    #[arcane]
    fn weighted_sum_v3(_t: X64V3Token, a: f32, b: f32, weight: f32) -> f32 {
        a * weight + b * (1.0 - weight)
    }

    #[cfg(feature = "avx512")]
    #[arcane]
    fn weighted_sum_v4(_t: archmage::X64V4Token, a: f32, b: f32, weight: f32) -> f32 {
        a * weight + b * (1.0 - weight)
    }

    fn weighted_sum_scalar(_token: ScalarToken, a: f32, b: f32, weight: f32) -> f32 {
        a * weight + b * (1.0 - weight)
    }

    pub fn weighted_sum(a: f32, b: f32, weight: f32) -> f32 {
        incant!(weighted_sum(a, b, weight))
    }

    #[test]
    fn multi_arg_incant_partial_variants() {
        let result = weighted_sum(10.0, 20.0, 0.75);
        // 10 * 0.75 + 20 * 0.25 = 7.5 + 5.0 = 12.5
        assert!((result - 12.5).abs() < 1e-6);
    }
}

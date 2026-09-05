//! `forge_token_dangerously()` is a safe `#[target_feature]` function.
//!
//! rustc decides whether a call site may construct the proof:
//!
//! * a caller whose own `#[target_feature]` attribute lists this tier's
//!   features (or a superset) calls it **safely** — the attribute is the proof;
//! * every other caller needs an `unsafe` block and carries the obligation.
//!
//! The whole file is `#![forbid(unsafe_code)]`, so it only compiles if the safe
//! path really is safe. The negative direction — a plain caller, a weaker
//! context, and function-pointer coercion — is covered by the `forge_*` cases
//! in `tests/compile_fail/`. The `unsafe` route from a feature-free context is
//! exercised by `tests/token_downcast.rs`.

#![forbid(unsafe_code)]

use archmage::{ScalarToken, SimdToken, arcane, rite};

// ============================================================================
// x86-64
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::*;
    use archmage::{X64V1Token, X64V2Token, X64V3Token};

    /// A tokenless `#[rite(v3)]` body materializes the token it was never
    /// handed. This is the case the "rite token forging" open question was
    /// about: no `unsafe` is generated or written anywhere.
    #[rite(v3)]
    fn sum_via_forged_v3(data: &[f32; 8]) -> f32 {
        let token = X64V3Token::forge_token_dangerously();
        // The forged token is consumed by a token-taking SIMD helper, so a
        // token that were somehow bogus would not merely go unused.
        double_and_sum(token, data)
    }

    /// Token-based `#[rite]`: same feature set as the tier-based caller above,
    /// so the cross-call is safe and inlines into one `#[target_feature]`
    /// region.
    #[rite(import_intrinsics)]
    fn double_and_sum(_token: X64V3Token, data: &[f32; 8]) -> f32 {
        let v = _mm256_loadu_ps(data);
        let doubled = _mm256_add_ps(v, v);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(&mut out, doubled);
        out.iter().sum()
    }

    /// A `#[target_feature]` context may also forge every *weaker* tier — v3's
    /// feature set is a superset of v2's and v1's.
    #[rite(v3)]
    fn forge_weaker_tiers() -> (X64V1Token, X64V2Token) {
        (
            X64V1Token::forge_token_dangerously(),
            X64V2Token::forge_token_dangerously(),
        )
    }

    /// `#[arcane]` is the boundary: its body is the `#[target_feature]` region
    /// the `#[rite]` helpers above inline into.
    #[arcane]
    fn entry(_token: X64V3Token, data: &[f32; 8]) -> f32 {
        let (v1, v2) = forge_weaker_tiers();
        // Consume the weaker tokens so they cannot be optimized to nothing.
        assert_eq!(<X64V1Token as SimdToken>::NAME, "x86-64-v1");
        assert_eq!(<X64V2Token as SimdToken>::NAME, "x86-64-v2");
        let _ = (v1, v2);
        sum_via_forged_v3(data)
    }

    #[test]
    fn forge_in_matching_and_superset_context() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        match X64V3Token::summon() {
            Some(token) => assert_eq!(entry(token, &data), 72.0),
            // No graceful skip: the CPU may genuinely lack AVX2, but that has
            // testable consequences either way.
            None => {
                assert_ne!(
                    X64V3Token::compiled_with(),
                    Some(true),
                    "summon() returned None although the features are compile-time guaranteed"
                );
                assert!(
                    X64V3Token::summon().is_none(),
                    "summon() is not deterministic"
                );
            }
        }
    }
}

// ============================================================================
// aarch64
// ============================================================================

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::*;
    use archmage::NeonToken;

    #[rite(neon)]
    fn sum_via_forged_neon(data: &[f32; 4]) -> f32 {
        let token = NeonToken::forge_token_dangerously();
        double_and_sum(token, data)
    }

    #[rite(import_intrinsics)]
    fn double_and_sum(_token: NeonToken, data: &[f32; 4]) -> f32 {
        let v = vld1q_f32(data);
        vaddvq_f32(vaddq_f32(v, v))
    }

    #[arcane]
    fn entry(_token: NeonToken, data: &[f32; 4]) -> f32 {
        sum_via_forged_neon(data)
    }

    #[test]
    fn forge_in_matching_context() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        match NeonToken::summon() {
            Some(token) => assert_eq!(entry(token, &data), 20.0),
            None => {
                assert_ne!(
                    NeonToken::compiled_with(),
                    Some(true),
                    "summon() returned None although the features are compile-time guaranteed"
                );
                assert!(
                    NeonToken::summon().is_none(),
                    "summon() is not deterministic"
                );
            }
        }
    }
}

// ============================================================================
// wasm32
// ============================================================================

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use super::*;
    use archmage::Wasm128Token;

    #[rite(wasm128)]
    fn sum_via_forged_wasm128(data: &[f32; 4]) -> f32 {
        let token = Wasm128Token::forge_token_dangerously();
        double_and_sum(token, data)
    }

    #[rite(import_intrinsics)]
    fn double_and_sum(_token: Wasm128Token, data: &[f32; 4]) -> f32 {
        let v = v128_load(data);
        let doubled = f32x4_add(v, v);
        let mut out = [0.0f32; 4];
        v128_store(&mut out, doubled);
        out.iter().sum()
    }

    #[arcane]
    fn entry(_token: Wasm128Token, data: &[f32; 4]) -> f32 {
        sum_via_forged_wasm128(data)
    }

    #[test]
    fn forge_in_matching_context() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let token = Wasm128Token::summon().expect("simd128 is enabled at compile time");
        assert_eq!(entry(token, &data), 20.0);
    }
}

// ============================================================================
// Architecture-independent
// ============================================================================

/// `ScalarToken` asserts no CPU features, so its constructor carries no
/// `#[target_feature]` gate and is callable from anywhere.
#[test]
fn scalar_forge_needs_no_context() {
    assert_eq!(
        ScalarToken::forge_token_dangerously(),
        ScalarToken::summon().unwrap()
    );
}

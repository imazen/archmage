//! Name mangling validation: verify generated function names match the
//! documented suffix table for all macros and all tiers.
//!
//! Each test calls the generated variant BY ITS EXPECTED NAME. If the name
//! is wrong, the test fails to compile. This is a compile-time guarantee
//! that the documented name mangling table is accurate.
#![allow(deprecated)] // Legacy SimdToken usage

use archmage::prelude::*;

// ============================================================================
// autoversion: verify suffix for every tier reachable on this platform
// ============================================================================

// Default tiers: v4, v3, neon, wasm128, scalar
#[autoversion]
fn av_default(_token: SimdToken, x: f32) -> f32 {
    x + 1.0
}

// The dispatcher strips SimdToken → av_default(x)
#[test]
fn autoversion_default_dispatcher_name() {
    assert_eq!(av_default(1.0), 2.0);
}

#[test]
fn autoversion_scalar_suffix() {
    assert_eq!(av_default_scalar(ScalarToken, 1.0), 2.0);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn autoversion_v3_suffix() {
    if let Some(t) = X64V3Token::summon() {
        assert_eq!(av_default_v3(t, 1.0), 2.0);
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn autoversion_v4_suffix() {
    // v4 variant exists even without avx512 feature (autoversion generates scalar code)
    if let Some(t) = X64V4Token::summon() {
        assert_eq!(av_default_v4(t, 1.0), 2.0);
    }
}

// Explicit tiers to cover the rest
#[cfg(target_arch = "x86_64")]
mod x86_tiers {
    use super::*;

    #[autoversion(v1, v2, x64_crypto, v3, v3_crypto, v4, v4x, scalar)]
    fn av_all_x86(_token: SimdToken, x: f32) -> f32 {
        x + 1.0
    }

    #[test]
    fn v1_suffix() {
        if let Some(t) = archmage::X64V1Token::summon() {
            assert_eq!(av_all_x86_v1(t, 1.0), 2.0);
        }
    }

    #[test]
    fn v2_suffix() {
        if let Some(t) = archmage::X64V2Token::summon() {
            assert_eq!(av_all_x86_v2(t, 1.0), 2.0);
        }
    }

    #[test]
    fn x64_crypto_suffix() {
        if let Some(t) = archmage::X64CryptoToken::summon() {
            assert_eq!(av_all_x86_x64_crypto(t, 1.0), 2.0);
        }
    }

    #[test]
    fn v3_suffix() {
        if let Some(t) = X64V3Token::summon() {
            assert_eq!(av_all_x86_v3(t, 1.0), 2.0);
        }
    }

    #[test]
    fn v3_crypto_suffix() {
        if let Some(t) = archmage::X64V3CryptoToken::summon() {
            assert_eq!(av_all_x86_v3_crypto(t, 1.0), 2.0);
        }
    }

    #[test]
    fn v4_suffix() {
        if let Some(t) = X64V4Token::summon() {
            assert_eq!(av_all_x86_v4(t, 1.0), 2.0);
        }
    }

    #[test]
    fn v4x_suffix() {
        if let Some(t) = archmage::X64V4xToken::summon() {
            assert_eq!(av_all_x86_v4x(t, 1.0), 2.0);
        }
    }

    #[test]
    fn scalar_suffix() {
        assert_eq!(av_all_x86_scalar(ScalarToken, 1.0), 2.0);
    }
}

// ARM tiers (only compile on aarch64)
#[cfg(target_arch = "aarch64")]
mod arm_tiers {
    use super::*;

    #[autoversion(neon, arm_v2, arm_v3, neon_aes, neon_sha3, neon_crc, scalar)]
    fn av_all_arm(_token: SimdToken, x: f32) -> f32 {
        x + 1.0
    }

    #[test]
    fn neon_suffix() {
        if let Some(t) = archmage::NeonToken::summon() {
            assert_eq!(av_all_arm_neon(t, 1.0), 2.0);
        }
    }

    #[test]
    fn arm_v2_suffix() {
        if let Some(t) = archmage::Arm64V2Token::summon() {
            assert_eq!(av_all_arm_arm_v2(t, 1.0), 2.0);
        }
    }

    #[test]
    fn arm_v3_suffix() {
        if let Some(t) = archmage::Arm64V3Token::summon() {
            assert_eq!(av_all_arm_arm_v3(t, 1.0), 2.0);
        }
    }

    #[test]
    fn neon_aes_suffix() {
        if let Some(t) = archmage::NeonAesToken::summon() {
            assert_eq!(av_all_arm_neon_aes(t, 1.0), 2.0);
        }
    }

    #[test]
    fn neon_sha3_suffix() {
        if let Some(t) = archmage::NeonSha3Token::summon() {
            assert_eq!(av_all_arm_neon_sha3(t, 1.0), 2.0);
        }
    }

    #[test]
    fn neon_crc_suffix() {
        if let Some(t) = archmage::NeonCrcToken::summon() {
            assert_eq!(av_all_arm_neon_crc(t, 1.0), 2.0);
        }
    }

    #[test]
    fn scalar_suffix() {
        assert_eq!(av_all_arm_scalar(ScalarToken, 1.0), 2.0);
    }
}

// WASM tiers (only compile on wasm32)
#[cfg(target_arch = "wasm32")]
mod wasm_tiers {
    use super::*;

    #[autoversion(wasm128, wasm128_relaxed, scalar)]
    fn av_all_wasm(_token: SimdToken, x: f32) -> f32 {
        x + 1.0
    }

    #[test]
    fn wasm128_suffix() {
        if let Some(t) = archmage::Wasm128Token::summon() {
            assert_eq!(av_all_wasm_wasm128(t, 1.0), 2.0);
        }
    }

    #[test]
    fn wasm128_relaxed_suffix() {
        if let Some(t) = archmage::Wasm128RelaxedToken::summon() {
            assert_eq!(av_all_wasm_wasm128_relaxed(t, 1.0), 2.0);
        }
    }

    #[test]
    fn scalar_suffix() {
        assert_eq!(av_all_wasm_scalar(ScalarToken, 1.0), 2.0);
    }
}

// ============================================================================
// default tier suffix
// ============================================================================

#[autoversion(v3, neon, default)]
fn av_with_default(x: f32) -> f32 {
    x + 1.0
}

#[test]
fn autoversion_default_suffix() {
    // _default is tokenless
    assert_eq!(av_with_default_default(1.0), 2.0);
}

#[test]
fn autoversion_default_dispatcher() {
    assert_eq!(av_with_default(1.0), 2.0);
}

// ============================================================================
// rite multi-tier: verify suffixes match autoversion's
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod rite_suffixes {
    use archmage::{SimdToken, X64V3Token, arcane, rite};

    // Multi-tier rite: tier-based (no token param), generates suffixed variants
    #[rite(v2, v3)]
    fn rite_multi(x: f32) -> f32 {
        x + 1.0
    }

    // Call rite variants from a matching #[arcane] context to verify names
    #[arcane]
    fn call_v3(_token: X64V3Token) -> f32 {
        rite_multi_v3(1.0)
    }

    #[test]
    fn rite_v3_suffix() {
        if let Some(t) = X64V3Token::summon() {
            assert_eq!(call_v3(t), 2.0);
        }
    }

    // v2 variant exists — verify by calling from v3 context (v3 ⊃ v2 features)
    #[arcane]
    fn call_v2(_token: X64V3Token) -> f32 {
        rite_multi_v2(1.0)
    }

    #[test]
    fn rite_v2_suffix() {
        if let Some(t) = X64V3Token::summon() {
            assert_eq!(call_v2(t), 2.0);
        }
    }
}

// ============================================================================
// magetypes: verify suffixes
// ============================================================================

#[magetypes]
fn mt_compute(_token: Token, x: f32) -> f32 {
    x + 1.0
}

#[test]
fn magetypes_scalar_suffix() {
    assert_eq!(mt_compute_scalar(ScalarToken, 1.0), 2.0);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn magetypes_v3_suffix() {
    if let Some(t) = X64V3Token::summon() {
        assert_eq!(mt_compute_v3(t, 1.0), 2.0);
    }
}

// ============================================================================
// incant! dispatches to correct suffix names
// ============================================================================

// Manually define variants with expected names to prove incant! calls them
fn incant_test_scalar(_: ScalarToken, x: f32) -> f32 {
    x * 10.0
}

#[arcane]
fn incant_test_v3(_: X64V3Token, x: f32) -> f32 {
    x * 30.0
}

fn incant_dispatch(x: f32) -> f32 {
    incant!(incant_test(x), [v3, scalar])
}

#[test]
fn incant_calls_correct_suffix() {
    let result = incant_dispatch(1.0);
    // On x86_64 with AVX2: 30.0 (v3). Otherwise: 10.0 (scalar).
    #[cfg(target_arch = "x86_64")]
    if X64V3Token::summon().is_some() {
        assert_eq!(result, 30.0);
    } else {
        assert_eq!(result, 10.0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    assert_eq!(result, 10.0);
}

// incant! with default tier
fn incant_default_test_default(x: f32) -> f32 {
    x * 99.0
}

#[arcane]
fn incant_default_test_v3(_: X64V3Token, x: f32) -> f32 {
    x * 30.0
}

fn incant_default_dispatch(x: f32) -> f32 {
    incant!(incant_default_test(x), [v3, default])
}

#[test]
fn incant_default_calls_correct_suffix() {
    let result = incant_default_dispatch(1.0);
    #[cfg(target_arch = "x86_64")]
    if X64V3Token::summon().is_some() {
        assert_eq!(result, 30.0);
    } else {
        assert_eq!(result, 99.0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    assert_eq!(result, 99.0);
}

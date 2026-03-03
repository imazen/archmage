//! Tests proving incant! works correctly with cfg'd-out functions.
//!
//! When #[arcane] functions are cfg'd out (no stub), incant! still dispatches
//! correctly because it wraps each tier call in #[cfg(target_arch = ...)] blocks.

#![cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]

use archmage::ScalarToken;

// =============================================================================
// #[arcane] functions WITHOUT stub (default cfg-out)
// =============================================================================

// Scalar: always available, no #[arcane] needed
fn add_scalar(_token: ScalarToken, a: f32, b: f32) -> f32 {
    a + b
}

// x86: cfg'd out on other arches (no stub)
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn add_v3(_token: archmage::X64V3Token, a: f32, b: f32) -> f32 {
    a + b
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
fn add_v4(_token: archmage::X64V4Token, a: f32, b: f32) -> f32 {
    a + b
}

// ARM: cfg'd out on other arches (no stub)
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn add_neon(_token: archmage::NeonToken, a: f32, b: f32) -> f32 {
    a + b
}

// WASM: cfg'd out on other arches (no stub)
#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn add_wasm128(_token: archmage::Wasm128Token, a: f32, b: f32) -> f32 {
    a + b
}

// =============================================================================
// incant! dispatch works
// =============================================================================

mod entry_dispatch {
    use super::*;
    use archmage::incant;

    fn add_dispatched(a: f32, b: f32) -> f32 {
        incant!(add(a, b))
    }

    #[test]
    fn incant_dispatches_with_cfgout_functions() {
        assert_eq!(add_dispatched(3.0, 4.0), 7.0);
    }
}

mod explicit_tiers {
    use super::*;
    use archmage::incant;

    fn add_explicit(a: f32, b: f32) -> f32 {
        incant!(add(a, b), [v3, neon])
    }

    #[test]
    fn incant_explicit_tiers_with_cfgout() {
        assert_eq!(add_explicit(5.0, 6.0), 11.0);
    }
}

mod scalar_fallback {
    use super::*;
    use archmage::incant;

    // Only scalar variant — no SIMD at all
    fn trivial_scalar(_token: ScalarToken, x: f32) -> f32 {
        x * 2.0
    }

    fn trivial_dispatched(x: f32) -> f32 {
        incant!(trivial(x), [])
    }

    #[test]
    fn scalar_only_dispatch() {
        assert_eq!(trivial_dispatched(5.0), 10.0);
    }
}

//! Verify `implementation_name()` is callable on every concrete backend that
//! has a corresponding `{Type}Backend` impl, for every generic type
//! (f32xN, iNxM, uNxM). Regression guard: before this test, only
//! f32xN<X64V3Token> on x86 had the method — calling it on f32xN<NeonToken>
//! or f32xN<Wasm128Token> was a compile error even though NEON/WASM have
//! backend impls via polyfill.

use magetypes::simd::generic::{f32x4, f32x8};

#[cfg(feature = "w512")]
use magetypes::simd::generic::f32x16;

// ============================================================================
// ScalarToken: always available on every architecture
// ============================================================================

#[test]
fn scalar_token_implementation_names_f32() {
    assert_eq!(
        f32x4::<archmage::ScalarToken>::implementation_name(),
        "scalar::f32x4"
    );
    assert_eq!(
        f32x8::<archmage::ScalarToken>::implementation_name(),
        "scalar::f32x8"
    );
    #[cfg(feature = "w512")]
    assert_eq!(
        f32x16::<archmage::ScalarToken>::implementation_name(),
        "scalar::f32x16"
    );
}

#[test]
fn scalar_token_implementation_names_int() {
    use magetypes::simd::generic::{i8x16, i32x4, u32x8};
    assert_eq!(
        i8x16::<archmage::ScalarToken>::implementation_name(),
        "scalar::i8x16"
    );
    assert_eq!(
        i32x4::<archmage::ScalarToken>::implementation_name(),
        "scalar::i32x4"
    );
    assert_eq!(
        u32x8::<archmage::ScalarToken>::implementation_name(),
        "scalar::u32x8"
    );
}

// ============================================================================
// x86_64 — V3 native 128/256; V3 polyfill 512; V4/V4x native 512 (avx512)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[test]
fn x86_v3_implementation_names() {
    assert_eq!(
        f32x4::<archmage::X64V3Token>::implementation_name(),
        "x86::v3::f32x4"
    );
    assert_eq!(
        f32x8::<archmage::X64V3Token>::implementation_name(),
        "x86::v3::f32x8"
    );
    #[cfg(feature = "w512")]
    assert_eq!(
        f32x16::<archmage::X64V3Token>::implementation_name(),
        "polyfill::v3_512::f32x16"
    );
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[test]
fn x86_v4_v4x_implementation_names() {
    assert_eq!(
        f32x16::<archmage::X64V4Token>::implementation_name(),
        "x86::v4::f32x16"
    );
    assert_eq!(
        f32x16::<archmage::X64V4xToken>::implementation_name(),
        "x86::v4x::f32x16"
    );
}

// ============================================================================
// aarch64 — NEON native 128; NEON polyfill 256/512
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_implementation_names() {
    assert_eq!(
        f32x4::<archmage::NeonToken>::implementation_name(),
        "arm::neon::f32x4"
    );
    assert_eq!(
        f32x8::<archmage::NeonToken>::implementation_name(),
        "polyfill::neon::f32x8"
    );
    #[cfg(feature = "w512")]
    assert_eq!(
        f32x16::<archmage::NeonToken>::implementation_name(),
        "polyfill::neon_512::f32x16"
    );
}

// ============================================================================
// wasm32 — Wasm128 native 128; Wasm128 polyfill 256/512
// ============================================================================

#[cfg(target_arch = "wasm32")]
#[test]
fn wasm128_implementation_names() {
    assert_eq!(
        f32x4::<archmage::Wasm128Token>::implementation_name(),
        "wasm::wasm128::f32x4"
    );
    assert_eq!(
        f32x8::<archmage::Wasm128Token>::implementation_name(),
        "polyfill::wasm128::f32x8"
    );
    #[cfg(feature = "w512")]
    assert_eq!(
        f32x16::<archmage::Wasm128Token>::implementation_name(),
        "polyfill::wasm128_512::f32x16"
    );
}

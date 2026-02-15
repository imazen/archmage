//! Backend traits for generic SIMD types.
//!
//! Each trait defines the operations for one vector type (e.g., `F32x8Backend`).
//! Token types implement these traits with platform-native intrinsics.
//! The generic wrapper types in [`super::generic`] delegate to these traits.

#![allow(non_camel_case_types)]

pub(crate) mod sealed;

mod f32x8;
pub use f32x8::F32x8Backend;

// ============================================================================
// Short token aliases for use as generic parameters.
//
// These match the existing namespace names (v3, neon, wasm128, scalar)
// and provide clean ergonomics: `f32x8<x64v3>` instead of `f32x8<X64V3Token>`.
// ============================================================================

/// x86-64-v1 (SSE2 baseline) backend.
pub type x64v1 = archmage::X64V1Token;

/// x86-64-v2 (SSE4.2 + POPCNT) backend.
pub type x64v2 = archmage::X64V2Token;

/// x86-64-v3 (AVX2 + FMA) backend — the recommended desktop baseline.
pub type x64v3 = archmage::X64V3Token;

/// x86-64-v4 (AVX-512) backend.
pub type x64v4 = archmage::X64V4Token;

/// AArch64 NEON backend.
pub type neon = archmage::NeonToken;

/// WebAssembly SIMD128 backend.
pub type wasm128 = archmage::Wasm128Token;

/// Scalar fallback backend — always available.
pub type scalar = archmage::ScalarToken;

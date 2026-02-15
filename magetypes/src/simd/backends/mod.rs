//! Backend traits for generic SIMD types.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#![allow(non_camel_case_types)]

pub(crate) mod sealed;

mod f32x4;
pub use f32x4::F32x4Backend;

mod f32x8;
pub use f32x8::F32x8Backend;

mod f64x2;
pub use f64x2::F64x2Backend;

mod f64x4;
pub use f64x4::F64x4Backend;

mod i32x4;
pub use i32x4::I32x4Backend;

mod i32x8;
pub use i32x8::I32x8Backend;

mod convert;
pub use convert::{F32x4Convert, F32x8Convert};

/// x86-64 baseline (SSE2).
pub type x64v1 = archmage::X64V1Token;
/// x86-64 v2 (SSE4.2 + POPCNT).
pub type x64v2 = archmage::X64V2Token;
/// x86-64 v3 (AVX2 + FMA).
pub type x64v3 = archmage::X64V3Token;
/// x86-64 v4 (AVX-512).
pub type x64v4 = archmage::X64V4Token;
/// AVX-512 with modern extensions.
pub type avx512_modern = archmage::Avx512ModernToken;
/// AArch64 NEON.
pub type neon = archmage::NeonToken;
/// WASM SIMD128.
pub type wasm128 = archmage::Wasm128Token;
/// Scalar fallback.
pub type scalar = archmage::ScalarToken;

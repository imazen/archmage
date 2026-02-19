//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! Marker traits for SIMD capability levels.

use crate::tokens::SimdToken;

/// Marker trait for tokens that provide 128-bit SIMD.
pub trait Has128BitSimd: SimdToken {}

/// Marker trait for tokens that provide 256-bit SIMD.
pub trait Has256BitSimd: Has128BitSimd {}

/// Marker trait for tokens that provide 512-bit SIMD.
pub trait Has512BitSimd: Has256BitSimd {}

/// Marker trait for x86-64-v2 level (Nehalem 2008+).
///
/// v2 includes: SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT, CMPXCHG16B, LAHF-SAHF.
pub trait HasX64V2: SimdToken {}

/// Marker trait for x86-64-v4 level (Skylake-X 2017+, Zen 4 2022+).
///
/// v4 includes all of v3 plus: AVX512F, AVX512BW, AVX512CD, AVX512DQ, AVX512VL.
/// Implies HasX64V2.
pub trait HasX64V4: HasX64V2 {}

/// Marker trait for NEON on AArch64.
///
/// NEON is available on virtually all AArch64 processors.
pub trait HasNeon: SimdToken {}

/// Marker trait for NEON + AES.
///
/// AES extension is common on modern ARM64 devices (ARMv8-A with Crypto).
pub trait HasNeonAes: HasNeon {}

/// Marker trait for NEON + SHA3.
///
/// SHA3 extension is available on ARMv8.2-A and later.
pub trait HasNeonSha3: HasNeon {}

/// Marker trait for Arm64-v2 level.
///
/// Arm64-v2 includes: NEON, CRC, RDM, DotProd, FP16, AES, SHA2.
/// Available on Cortex-A55+, Apple M1+, Graviton 2+.
pub trait HasArm64V2: HasNeon + HasNeonAes {}

/// Marker trait for Arm64-v3 level.
///
/// Arm64-v3 adds FHM, FCMA, SHA3, I8MM, BF16 over Arm64-v2.
/// Available on Cortex-A510+, Apple M2+, Snapdragon X, Graviton 3+.
pub trait HasArm64V3: HasArm64V2 + HasNeonSha3 {}

//! # archmage
//!
//! > Safely invoke your intrinsic power, using the tokens granted to you by the CPU.
//! > Cast primitive magics faster than any mage alive.
//!
//! archmage provides capability tokens that prove CPU feature availability at runtime,
//! making raw SIMD intrinsics safe to call via the `#[arcane]` macro.
//!
//! ## Quick Example
//!
//! ```rust,ignore
//! use archmage::{Desktop64, SimdToken, arcane};
//! use std::arch::x86_64::*;
//!
//! #[arcane]
//! fn multiply_add(_token: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
//!     // safe_unaligned_simd calls are SAFE inside #[arcane] - no unsafe needed!
//!     let va = safe_unaligned_simd::x86_64::_mm256_loadu_ps(a);
//!     let vb = safe_unaligned_simd::x86_64::_mm256_loadu_ps(b);
//!
//!     // Value-based intrinsics are also SAFE inside #[arcane]!
//!     // FMA is available because Desktop64 = X64V3Token = AVX2+FMA
//!     let result = _mm256_fmadd_ps(va, vb, va);
//!
//!     let mut out = [0.0f32; 8];
//!     safe_unaligned_simd::x86_64::_mm256_storeu_ps(&mut out, result);
//!     out
//! }
//!
//! fn main() {
//!     // Desktop64: AVX2 + FMA + BMI2 (Haswell 2013+, Zen 1+)
//!     // CPUID check elided if compiled with -C target-cpu=native
//!     if let Some(token) = Desktop64::summon() {
//!         let result = multiply_add(token, &[1.0; 8], &[2.0; 8]);
//!     }
//! }
//! ```
//!
//! ## The Prelude
//!
//! `use archmage::prelude::*` gives you tokens, traits, macros, platform
//! intrinsics, and SIMD types in one import. Value-based intrinsics like
//! `_mm256_add_ps` are already safe inside `#[arcane]` since Rust 1.85.
//!
//! For safe memory operations (load/store), import them explicitly from
//! `safe_unaligned_simd` — the names overlap with `core::arch` and
//! can't resolve through a glob re-export:
//!
//! ```rust,ignore
//! use archmage::prelude::*;
//! use safe_unaligned_simd::x86_64::{_mm256_loadu_ps, _mm256_storeu_ps};
//!
//! #[arcane]
//! fn load(_token: Desktop64, data: &[f32; 8]) -> __m256 {
//!     _mm256_loadu_ps(data)  // Safe! Takes &[f32; 8], not *const f32.
//! }
//! ```
//!
//! See the [`prelude`] module for full documentation of what's included.
//!
//! ## How It Works
//!
//! **Capability Tokens** are zero-sized proof types created via `summon()`, which
//! checks CPUID at runtime (elided if compiled with target features enabled).
//!
//! **The `#[arcane]` macro** generates an inner function with `#[target_feature]`,
//! making intrinsics safe inside. The token parameter proves CPU support was verified.
//!
//! Use concrete tokens like `Desktop64` (AVX2+FMA) or `X64V4Token` (AVX-512).
//! For generic code, use tier traits like `HasX64V2` or `HasX64V4`.
//!
//! ## Feature Flags
//!
//! - `std` (default): Enable std library support
//! - `macros` (default): Enable `#[arcane]` attribute macro (alias: `#[arcane]`)
//! - `avx512`: AVX-512 token support

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

// Re-export macros from archmage-macros
#[cfg(feature = "macros")]
#[cfg_attr(docsrs, doc(cfg(feature = "macros")))]
pub use archmage_macros::{arcane, incant, magetypes, rite, simd_fn, simd_route};

// Optimized feature detection
#[cfg(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64"))]
pub mod detect;

// Core token types and traits
pub mod tokens;

// Prelude: one import for tokens, traits, macros, core::arch, and safe_unaligned_simd
pub mod prelude;

// SIMD types moved to magetypes crate
// Use `magetypes::simd` for f32x8, i32x4, etc.

// ============================================================================
// Re-exports at crate root for convenience
// ============================================================================

// Core traits
pub use tokens::CompileTimeGuaranteedError;
pub use tokens::IntoConcreteToken;
pub use tokens::SimdToken;

// Width marker traits
pub use tokens::{Has128BitSimd, Has256BitSimd, Has512BitSimd};

// x86 tier marker traits (based on LLVM x86-64 microarchitecture levels)
pub use tokens::HasX64V2;
#[cfg(feature = "avx512")]
pub use tokens::HasX64V4;

// AArch64 tier marker traits
pub use tokens::{HasNeon, HasNeonAes, HasNeonSha3};

// All tokens available on all architectures (summon() returns None on wrong arch)
pub use tokens::{
    // ARM tokens
    Arm64,
    // x86 tier tokens
    Avx2FmaToken,
    Desktop64,
    NeonAesToken,
    NeonCrcToken,
    NeonSha3Token,
    NeonToken,
    // Scalar fallback (always available)
    ScalarToken,
    // WASM tokens
    Wasm128Token,
    X64V2Token,
    X64V3Token,
};

// AVX-512 tokens (requires "avx512" feature)
#[cfg(feature = "avx512")]
pub use tokens::{Avx512Fp16Token, Avx512ModernToken, Avx512Token, Server64, X64V4Token};

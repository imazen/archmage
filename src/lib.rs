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
//! use archmage::{Desktop64, Has256BitSimd, SimdToken, arcane};
//! use std::arch::x86_64::*;
//!
//! #[arcane]
//! fn multiply_add(_token: impl Has256BitSimd, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
//!     // safe_unaligned_simd calls are SAFE inside #[arcane] - no unsafe needed!
//!     let va = safe_unaligned_simd::x86_64::_mm256_loadu_ps(a);
//!     let vb = safe_unaligned_simd::x86_64::_mm256_loadu_ps(b);
//!
//!     // Value-based intrinsics are also SAFE inside #[arcane]!
//!     let result = _mm256_add_ps(va, vb);
//!     let result = _mm256_mul_ps(result, result);
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
//! ## How It Works
//!
//! **Capability Tokens** are zero-sized proof types created via `summon()`, which
//! checks CPUID at runtime (elided if compiled with target features enabled).
//!
//! **The `#[arcane]` macro** generates an inner function with `#[target_feature]`,
//! making intrinsics safe inside. The token parameter proves CPU support was verified.
//!
//! **Generic bounds** like `impl Has256BitSimd` let functions accept any token that
//! provides 256-bit SIMD (e.g., `X64V3Token`, `Desktop64`, `X64V4Token`).
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

// Re-export arcane macro (and arcane alias) from archmage-macros
#[cfg(feature = "macros")]
#[cfg_attr(docsrs, doc(cfg(feature = "macros")))]
pub use archmage_macros::{arcane, multiwidth, simd_fn};

// Optimized feature detection
#[cfg(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64"))]
pub mod detect;

// Core token types and traits
pub mod tokens;

// SIMD types moved to magetypes crate
// Use `magetypes::simd` for f32x8, i32x4, etc.

// ============================================================================
// Re-exports at crate root for convenience
// ============================================================================

// Core trait
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
    // WASM tokens
    Simd128Token,
    X64V2Token,
    X64V3Token,
};

// AVX-512 tokens (requires "avx512" feature)
#[cfg(feature = "avx512")]
pub use tokens::{Avx512Fp16Token, Avx512ModernToken, Avx512Token, Server64, X64V4Token};

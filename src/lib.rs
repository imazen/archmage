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
//! use archmage::{X64V3Token, HasAvx2, SimdToken, arcane};
//! use archmage::mem::avx;  // Safe load/store (requires safe_unaligned_simd feature)
//! use std::arch::x86_64::*;
//!
//! #[arcane]
//! fn multiply_add(token: impl HasAvx2, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
//!     // Safe memory operations - references, not raw pointers!
//!     let va = avx::_mm256_loadu_ps(token, a);
//!     let vb = avx::_mm256_loadu_ps(token, b);
//!
//!     // Value-based intrinsics are SAFE inside #[arcane]!
//!     let result = _mm256_add_ps(va, vb);
//!     let result = _mm256_mul_ps(result, result);
//!
//!     let mut out = [0.0f32; 8];
//!     avx::_mm256_storeu_ps(token, &mut out, result);
//!     out
//! }
//!
//! fn main() {
//!     // X64V3Token: AVX2 + FMA + BMI2 (Haswell 2013+, Zen 1+)
//!     if let Some(token) = X64V3Token::try_new() {
//!         let result = multiply_add(token, &[1.0; 8], &[2.0; 8]);
//!     }
//! }
//! ```
//!
//! ## How It Works
//!
//! **Capability Tokens** are zero-sized proof types created via `try_new()`, which
//! performs runtime CPU detection. If `try_new()` returns `Some(token)`, the CPU
//! definitely supports the required features.
//!
//! **The `#[arcane]` macro** generates an inner function with `#[target_feature]`,
//! making intrinsics safe inside. The outer function calls this inner function -
//! this is safe because the token parameter proves CPU support was verified.
//!
//! **Generic bounds** like `impl HasAvx2` let functions accept any token that
//! provides AVX2 (e.g., `Avx2Token`, `X64V3Token`, `X64V4Token`).
//!
//! ## Feature Flags
//!
//! - `std` (default): Enable std library support
//! - `macros` (default): Enable `#[arcane]` attribute macro (alias: `#[simd_fn]`)
//! - `safe_unaligned_simd`: Safe load/store via references (exposed as `mem` module)
//! - `__composite`: Higher-level ops (transpose, dot product) - unstable API
//! - `__wide`: Integration with the `wide` crate - unstable API

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

// Re-export arcane macro (and simd_fn alias) from archmage-macros
#[cfg(feature = "macros")]
#[cfg_attr(docsrs, doc(cfg(feature = "macros")))]
pub use archmage_macros::{arcane, simd_fn};

// Optimized feature detection
#[cfg(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64"))]
pub mod detect;

// Core token types and traits
pub mod tokens;

// Integration layers
#[cfg(any(feature = "__wide", feature = "safe_unaligned_simd"))]
pub mod integrate;

// Composite operations (requires "__composite" feature, unstable API)
#[cfg(feature = "__composite")]
#[cfg_attr(docsrs, doc(cfg(feature = "__composite")))]
pub mod composite;

// Safe unaligned memory operations (requires "safe_unaligned_simd" feature)
// Wraps safe_unaligned_simd with token-based safety
#[cfg(feature = "safe_unaligned_simd")]
#[cfg_attr(docsrs, doc(cfg(feature = "safe_unaligned_simd")))]
pub mod mem;

// ============================================================================
// Re-exports at crate root for convenience
// ============================================================================

// Core trait
pub use tokens::SimdToken;

// Composite token trait
pub use tokens::CompositeToken;

// Capability marker traits
pub use tokens::{Has128BitSimd, Has256BitSimd, Has512BitSimd, HasFma, HasScalableVectors};

// x86 feature marker traits (for generic bounds)
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub use tokens::{HasSse, HasSse2, HasSse41, HasSse42, HasAvx, HasAvx2, HasAvx512f, HasAvx512vl, HasAvx512bw, HasAvx512vbmi2};

// aarch64 feature marker traits
#[cfg(target_arch = "aarch64")]
pub use tokens::{HasNeon, HasSve, HasSve2};

// x86 tokens (when on x86)
#[cfg(target_arch = "x86_64")]
pub use tokens::x86::{
    Avx2FmaToken, Avx2Token, Avx512Vbmi2Token, Avx512Vbmi2VlToken, Avx512bwToken, Avx512bwVlToken,
    Avx512fToken, Avx512fVlToken, AvxToken, FmaToken, Sse2Token, Sse41Token, Sse42Token, SseToken,
    X64V2Token, X64V3Token, X64V4Token,
    // Friendly aliases
    Desktop64, Server64,
};

// aarch64 tokens (when on ARM)
#[cfg(target_arch = "aarch64")]
pub use tokens::arm::{NeonToken, Sve2Token, SveToken, Arm64};

// wasm tokens (when on WASM)
#[cfg(target_arch = "wasm32")]
pub use tokens::wasm::Simd128Token;


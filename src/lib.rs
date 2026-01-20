//! # archmage
//!
//! > Safely invoke your intrinsic power, using the tokens granted to you by the CPU.
//! > Cast primitive magics faster than any mage alive.
//!
//! archmage provides capability tokens that prove CPU feature availability,
//! making raw SIMD intrinsics safe to call via the `#[arcane]` macro.
//!
//! ## Quick Example
//!
//! ```rust,ignore
//! use archmage::{Avx2Token, SimdToken, arcane};
//! use std::arch::x86_64::*;
//!
//! #[arcane]
//! fn double(token: Avx2Token, data: &[f32; 8]) -> [f32; 8] {
//!     // Memory ops need unsafe (raw pointers)
//!     let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };
//!
//!     // Arithmetic intrinsics are SAFE - token proves AVX2!
//!     let doubled = _mm256_add_ps(v, v);
//!
//!     let mut out = [0.0f32; 8];
//!     unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
//!     out
//! }
//!
//! fn main() {
//!     if let Some(token) = Avx2Token::try_new() {
//!         let result = double(token, &[1.0; 8]);
//!         // result = [2.0; 8]
//!     }
//! }
//! ```
//!
//! ## How It Works
//!
//! The `#[arcane]` macro wraps your function with `#[target_feature]`,
//! making all value-based intrinsics safe. The token parameter proves
//! the caller verified feature availability.
//!
//! ## Feature Flags
//!
//! - `std` (default): Enable std library support
//! - `macros` (default): Enable `#[arcane]` attribute macro (also available as `#[simd_fn]`)
//! - `composite`: Higher-level operations (transpose, dot product, etc.) - implies `safe_unaligned_simd`
//! - `wide`: Integration with the `wide` crate
//! - `safe_unaligned_simd`: Safe load/store via `safe_unaligned_simd` crate (exposed as `mem` module)

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
};

// aarch64 tokens (when on ARM)
#[cfg(target_arch = "aarch64")]
pub use tokens::arm::{NeonToken, Sve2Token, SveToken};

// wasm tokens (when on WASM)
#[cfg(target_arch = "wasm32")]
pub use tokens::wasm::Simd128Token;


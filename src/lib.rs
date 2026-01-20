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
#[cfg(any(feature = "wide", feature = "safe_unaligned_simd"))]
pub mod integrate;

// Composite operations (requires "composite" feature)
#[cfg(feature = "composite")]
#[cfg_attr(docsrs, doc(cfg(feature = "composite")))]
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

// ============================================================================
// Token creation macros for use inside multiversioned functions
// ============================================================================

/// Create an AVX2 token inside a `#[multiversed]` or `#[multiversion]` function.
///
/// # Safety Justification
///
/// This macro contains `unsafe` code, but is safe to use inside multiversioned
/// functions because:
/// 1. The `#[multiversed]` macro generates multiple function versions
/// 2. The AVX2 version only executes on CPUs with AVX2 support
/// 3. Therefore, `Avx2Token::forge_token_dangerously()` is valid in that context
///
/// # Example
///
/// ```rust,ignore
/// use archmage::avx2_token;
/// use multiversed::multiversed;
///
/// #[multiversed]
/// fn my_kernel(data: &mut [f32]) {
///     let token = avx2_token!();  // Single unsafe point
///     // All subsequent operations are safe
/// }
/// ```
#[macro_export]
macro_rules! avx2_token {
    () => {{
        // SAFETY: This macro should only be used inside #[multiversed] functions
        // where the AVX2 version is being compiled with AVX2 enabled.
        // The multiversed macro guarantees the feature is available.
        unsafe { $crate::tokens::x86::Avx2Token::forge_token_dangerously() }
    }};
}

/// Create an FMA token inside a `#[multiversed]` function.
#[macro_export]
macro_rules! fma_token {
    () => {{ unsafe { $crate::tokens::x86::FmaToken::forge_token_dangerously() } }};
}

/// Create a combined AVX2+FMA token inside a `#[multiversed]` function.
///
/// This is the most common token for floating-point SIMD work.
#[macro_export]
macro_rules! avx2_fma_token {
    () => {{ unsafe { $crate::tokens::x86::Avx2FmaToken::forge_token_dangerously() } }};
}

/// Create an SSE2 token inside a `#[multiversed]` function.
#[macro_export]
macro_rules! sse2_token {
    () => {{ unsafe { $crate::tokens::x86::Sse2Token::forge_token_dangerously() } }};
}

/// Create an SSE4.1 token inside a `#[multiversed]` function.
#[macro_export]
macro_rules! sse41_token {
    () => {{ unsafe { $crate::tokens::x86::Sse41Token::forge_token_dangerously() } }};
}

/// Create an AVX token inside a `#[multiversed]` function.
#[macro_export]
macro_rules! avx_token {
    () => {{ unsafe { $crate::tokens::x86::AvxToken::forge_token_dangerously() } }};
}

/// Create an SSE4.2 token inside a `#[multiversed]` function.
#[macro_export]
macro_rules! sse42_token {
    () => {{ unsafe { $crate::tokens::x86::Sse42Token::forge_token_dangerously() } }};
}

/// Create an x86-64-v2 profile token inside a `#[multiversed]` function.
///
/// v2 implies: SSE4.2 + POPCNT (Nehalem 2008+, Bulldozer 2011+).
#[macro_export]
macro_rules! x64v2_token {
    () => {{ unsafe { $crate::tokens::x86::X64V2Token::forge_token_dangerously() } }};
}

/// Create an x86-64-v3 profile token inside a `#[multiversed]` function.
///
/// v3 implies: v2 + AVX2 + FMA + BMI1/2 (Haswell 2013+, Zen 1 2017+).
/// This is the most common target for high-performance SIMD code.
#[macro_export]
macro_rules! x64v3_token {
    () => {{ unsafe { $crate::tokens::x86::X64V3Token::forge_token_dangerously() } }};
}

/// Create an x86-64-v4 profile token inside a `#[multiversed]` function.
///
/// v4 implies: v3 + AVX-512 (F/BW/CD/DQ/VL) (Xeon 2017+, Zen 4 2022+).
#[macro_export]
macro_rules! x64v4_token {
    () => {{ unsafe { $crate::tokens::x86::X64V4Token::forge_token_dangerously() } }};
}

#[cfg(target_arch = "aarch64")]
/// Create a NEON token inside a `#[multiversed]` function.
///
/// NEON is the baseline SIMD for AArch64 - always available on 64-bit ARM.
#[macro_export]
macro_rules! neon_token {
    () => {{ unsafe { $crate::tokens::arm::NeonToken::forge_token_dangerously() } }};
}

#[cfg(target_arch = "aarch64")]
/// Create an SVE token inside a `#[multiversed]` function.
///
/// SVE (Scalable Vector Extension) is available on Graviton 3, Apple M-series,
/// Fujitsu A64FX, and ARMv8.2+ cores with SVE support.
#[macro_export]
macro_rules! sve_token {
    () => {{ unsafe { $crate::tokens::arm::SveToken::forge_token_dangerously() } }};
}

#[cfg(target_arch = "aarch64")]
/// Create an SVE2 token inside a `#[multiversed]` function.
///
/// SVE2 is available on ARMv9 cores (Cortex-X2/A710+, Graviton 4, etc.).
#[macro_export]
macro_rules! sve2_token {
    () => {{ unsafe { $crate::tokens::arm::Sve2Token::forge_token_dangerously() } }};
}

//! # archmage - Type-Safe SIMD Capability Tokens
//!
//! Safe SIMD dispatch through capability proof tokens. Isolates `unsafe` to token
//! construction, enabling safe SIMD code at usage sites.
//!
//! ## Quick Example
//!
//! ```rust,ignore
//! use archmage::{Avx2Token, ops};
//!
//! // Runtime detection path
//! if let Some(token) = Avx2Token::try_new() {
//!     let data = [1.0f32; 8];
//!     let v = ops::load_f32x8(token, &data);  // Safe!
//!     let result = ops::add_f32x8(token, v, v);  // Safe!
//! }
//!
//! // Or with multiversed (single unsafe point)
//! #[multiversed]
//! fn my_kernel(data: &[f32]) -> f32 {
//!     let token = avx2_fma_token!();  // Unsafe isolated here
//!     // ... all operations safe via token
//! }
//! ```
//!
//! ## Feature Flags
//!
//! - `std` (default): Enable std library support
//! - `wide`: Integration with the `wide` crate
//! - `safe-simd`: Integration with `safe_unaligned_simd`
//! - `multiversion`: Integration with `multiversion` crate
//! - `multiversed`: Integration with `multiversed` crate
//! - `full`: Enable all integrations

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

// Optimized feature detection
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub mod detect;

// Core token types and traits
pub mod tokens;

// Token-gated operations
#[cfg(target_arch = "x86_64")]
pub mod ops;

// Integration layers
#[cfg(any(
    feature = "wide",
    feature = "safe-simd",
    feature = "multiversion",
    feature = "multiversed"
))]
pub mod integrate;

// Composite operations (transpose, dot product, etc.)
pub mod composite;

// Re-export main types at crate root
pub use tokens::*;

// Re-export commonly used operations
#[cfg(target_arch = "x86_64")]
pub use ops::x86::*;

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
/// 3. Therefore, `Avx2Token::new_unchecked()` is valid in that context
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
        unsafe { $crate::tokens::x86::Avx2Token::new_unchecked() }
    }};
}

/// Create an FMA token inside a `#[multiversed]` function.
#[macro_export]
macro_rules! fma_token {
    () => {{ unsafe { $crate::tokens::x86::FmaToken::new_unchecked() } }};
}

/// Create a combined AVX2+FMA token inside a `#[multiversed]` function.
///
/// This is the most common token for floating-point SIMD work.
#[macro_export]
macro_rules! avx2_fma_token {
    () => {{ unsafe { $crate::tokens::x86::Avx2FmaToken::new_unchecked() } }};
}

/// Create an SSE2 token inside a `#[multiversed]` function.
#[macro_export]
macro_rules! sse2_token {
    () => {{ unsafe { $crate::tokens::x86::Sse2Token::new_unchecked() } }};
}

/// Create an SSE4.1 token inside a `#[multiversed]` function.
#[macro_export]
macro_rules! sse41_token {
    () => {{ unsafe { $crate::tokens::x86::Sse41Token::new_unchecked() } }};
}

/// Create an AVX token inside a `#[multiversed]` function.
#[macro_export]
macro_rules! avx_token {
    () => {{ unsafe { $crate::tokens::x86::AvxToken::new_unchecked() } }};
}

/// Create an SSE4.2 token inside a `#[multiversed]` function.
#[macro_export]
macro_rules! sse42_token {
    () => {{ unsafe { $crate::tokens::x86::Sse42Token::new_unchecked() } }};
}

/// Create an x86-64-v2 profile token inside a `#[multiversed]` function.
///
/// v2 implies: SSE4.2 + POPCNT (Nehalem 2008+, Bulldozer 2011+).
#[macro_export]
macro_rules! x64v2_token {
    () => {{ unsafe { $crate::tokens::x86::X64V2Token::new_unchecked() } }};
}

/// Create an x86-64-v3 profile token inside a `#[multiversed]` function.
///
/// v3 implies: v2 + AVX2 + FMA + BMI1/2 (Haswell 2013+, Zen 1 2017+).
/// This is the most common target for high-performance SIMD code.
#[macro_export]
macro_rules! x64v3_token {
    () => {{ unsafe { $crate::tokens::x86::X64V3Token::new_unchecked() } }};
}

/// Create an x86-64-v4 profile token inside a `#[multiversed]` function.
///
/// v4 implies: v3 + AVX-512 (F/BW/CD/DQ/VL) (Xeon 2017+, Zen 4 2022+).
#[macro_export]
macro_rules! x64v4_token {
    () => {{ unsafe { $crate::tokens::x86::X64V4Token::new_unchecked() } }};
}

#[cfg(target_arch = "aarch64")]
/// Create a NEON token inside a `#[multiversed]` function.
#[macro_export]
macro_rules! neon_token {
    () => {{ unsafe { $crate::tokens::arm::NeonToken::new_unchecked() } }};
}

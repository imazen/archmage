//! # archmage
//!
//! [Guide](https://imazen.github.io/archmage/) · [Intrinsics browser](https://imazen.github.io/archmage/intrinsics/) · [Archmage API](https://docs.rs/archmage/latest/archmage/) · [Magetypes API](https://docs.rs/magetypes/latest/magetypes/)
//!
//! Archmage provides CPU-capability tokens, feature-enabled functions, and dispatch.
//! The `#[magetypes]` attribute belongs to archmage; the
//! [magetypes crate](https://docs.rs/magetypes/latest/magetypes/) provides the vectors.
//! Start with the [complete portable kernel](https://imazen.github.io/archmage/archmage/getting-started/first-simd/)
//! and [generic specialization](https://imazen.github.io/archmage/magetypes/dispatch/types-and-dispatch/).
//!
//! For a direct-intrinsic specialization, keep the full loop behind an entry:
//!
//! ```rust
//! use archmage::prelude::*;
//! #[arcane(import_intrinsics)]
//! fn multiply_v3(_token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
//!     let v = _mm256_loadu_ps(data);
//!     let mut out = [0.0; 8];
//!     _mm256_storeu_ps(&mut out, _mm256_mul_ps(v, _mm256_set1_ps(2.0)));
//!     out
//! }
//! fn multiply_scalar(_token: ScalarToken, data: &[f32; 8]) -> [f32; 8] {
//!     data.map(|v| v * 2.0)
//! }
//! pub fn multiply(data: &[f32; 8]) -> [f32; 8] {
//!     incant!(multiply(data), [v3, scalar])
//! }
//! assert_eq!(multiply(&[3.0; 8]), [6.0; 8]);
//! ```
//!
//! Use default `#[magetypes]` for generated portable entry variants, `#[arcane]`
//! for individual entries, and matching `#[rite]` or inline generic helpers inside.
//! Generics are statically dispatched; an inline attribute alone does not enable
//! features. A CPU token does not justify arbitrary pointers or unchecked indexing.
//!
//! ## Tokens from an existing feature context
//!
//! When a helper already has target features, `from_context()` constructs a token
//! without runtime detection. Rust checks that the caller's features cover the
//! token's requirements. It is not a baseline-callable unchecked constructor.
//!
//! ```rust
//! use archmage::prelude::*;
//! #[rite(v3)]
//! fn helper() -> bool {
//!     let _token = X64V3Token::from_context();
//!     true
//! }
//! #[arcane]
//! fn entry(_token: X64V3Token) -> bool { helper() }
//! #[cfg(target_arch = "x86_64")]
//! if let Some(token) = X64V3Token::summon() { assert!(entry(token)); }
//! ```
//!
//! This is a repository addition after 0.9.28. See
//! [from_context and token extraction](https://imazen.github.io/archmage/archmage/getting-started/tokens/).
//! Use `.v3()` to extract a V3 token from a stronger proof; `as_x64v3()` instead
//! checks whether the held token is exactly a V3 token.
//!
//! ## Features
//!
//! `std` is enabled by default. `avx512` enables native intrinsic-wrapper and macro
//! support. Token names and macros are always available; unsupported tokens cannot
//! be summoned. `testable_dispatch` enables the tier-testing facilities.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

// Re-export macros from archmage-macros
pub use archmage_macros::{
    arcane, autoversion, dispatch_variant, incant, magetypes, rite, simd_fn, simd_route,
    token_target_features, token_target_features_boundary,
};

// Optimized feature detection
#[cfg(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64"))]
pub mod detect;

// Core token types and traits
pub mod tokens;

// Prelude: one import for tokens, traits, macros, and all intrinsics
pub mod prelude;

// Combined intrinsics namespace (core::arch + safe memory ops, safe wins)
pub mod intrinsics;

// Test utilities for exhaustive token permutation testing
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub mod testing;

// SIMD types moved to magetypes crate
// Use `magetypes::simd` for f32x8, i32x4, etc.

// ============================================================================
// Private module for macro-generated assertions
// ============================================================================

/// Internal module used by macro output. Not part of the public API.
///
/// Token shadowing/aliasing defense: the macros do name-based feature
/// lookup, so a user-defined `struct X64V3Token;` (or `use X64V2Token as
/// X64V3Token`) could otherwise receive `#[target_feature]` + `unsafe`
/// wrappers for features the impostor never proves. `#[arcane]`'s current
/// defense for concrete tokens is a shared associated constant named for the
/// expected tier (shadow types and weaker aliases lack it), plus the sealed
/// [`SimdToken`] bound for trait-generic parameters. Public constants do not
/// prevent deliberate forgery.
/// See `tests/soundness/token_shadowing_exploit.rs`
/// and `token_aliasing_exploit.rs`.
#[doc(hidden)]
pub mod __private {
    /// Compile-time assertion that `T` is a genuine archmage token.
    ///
    /// Kept for older macro-output compatibility (pre-tier-tag expansions
    /// called this from every wrapper). Optimized away entirely — exists
    /// only to produce a compile error if the token type doesn't implement
    /// [`SimdToken`](crate::SimdToken) (which is sealed).
    #[inline(always)]
    pub fn assert_archmage_token<T: crate::SimdToken>(_: &T) {}
}

// ============================================================================
// Re-exports at crate root for convenience
// ============================================================================

// Core traits
pub use tokens::CompileTimeGuaranteedError;
pub use tokens::DisableAllSimdError;
pub use tokens::IntoConcreteToken;
pub use tokens::SimdToken;

// Global SIMD kill switch
pub use tokens::dangerously_disable_tokens_except_wasm;

// Width marker traits (deprecated — use concrete tokens or tier traits)
#[allow(deprecated)]
pub use tokens::{Has128BitSimd, Has256BitSimd, Has512BitSimd};

// x86 tier marker traits (based on LLVM x86-64 microarchitecture levels)
pub use tokens::HasX64V2;
pub use tokens::HasX64V4;

// AArch64 tier marker traits
pub use tokens::{HasArm64V2, HasArm64V3, HasNeon, HasNeonAes, HasNeonSha3};

// All tokens available on all architectures (summon() returns None on wrong arch)
#[allow(deprecated)]
pub use tokens::{
    // ARM tokens
    Arm64,
    Arm64V2Token,
    Arm64V3Token,
    // x86 tier tokens (aliases still exported for backward compat)
    Avx2FmaToken,
    Desktop64,
    NeonAesToken,
    NeonCrcToken,
    NeonSha3Token,
    NeonToken,
    // Scalar fallback (always available)
    ScalarToken,
    Sse2Token,
    // WASM tokens
    Wasm128RelaxedToken,
    Wasm128Token,
    X64CryptoToken,
    X64V1Token,
    X64V2Token,
    X64V3CryptoToken,
    X64V3GfniCryptoToken,
    X64V3Token,
};

// AVX-512 tokens (always available; summon() returns None on unsupported CPUs)
pub use tokens::{Avx512Fp16Token, Avx512Token, Server64, X64V4Token, X64V4xToken};

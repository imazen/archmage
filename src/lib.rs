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
//! use archmage::{X64V3Token, SimdToken, arcane};
//!
//! #[arcane(import_intrinsics)]
//! fn multiply_add(_token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
//!     // import_intrinsics brings all intrinsics + safe memory ops into scope
//!     let va = _mm256_loadu_ps(a);  // Takes &[f32; 8], not *const f32
//!     let vb = _mm256_loadu_ps(b);
//!
//!     // Value-based intrinsics are SAFE inside #[arcane]! (Rust 1.85+)
//!     let result = _mm256_fmadd_ps(va, vb, va);
//!
//!     let mut out = [0.0f32; 8];
//!     _mm256_storeu_ps(&mut out, result);
//!     out
//! }
//!
//! fn main() {
//!     // X64V3Token: AVX2 + FMA + BMI2 (Haswell 2013+, Zen 1+)
//!     // CPUID check elided if compiled with -C target-cpu=native
//!     if let Some(token) = X64V3Token::summon() {
//!         let result = multiply_add(token, &[1.0; 8], &[2.0; 8]);
//!     }
//! }
//! ```
//!
//! ## Auto-Imports
//!
//! `import_intrinsics` is the recommended default — it injects
//! `archmage::intrinsics::{arch}::*` into the function body, giving you all
//! platform types, value intrinsics, and safe memory ops in one import:
//!
//! ```rust,ignore
//! use archmage::{X64V3Token, SimdToken, arcane};
//!
//! #[arcane(import_intrinsics)]
//! fn load(_token: X64V3Token, data: &[f32; 8]) -> __m256 {
//!     _mm256_loadu_ps(data)  // Safe! Takes &[f32; 8], not *const f32.
//! }
//! ```
//!
//! The prelude (`use archmage::prelude::*`) is still available for module-level imports.
//! See the [`prelude`] module for full documentation.
//!
//! ## How It Works
//!
//! **Capability Tokens** are zero-sized proof types created via `summon()`, which
//! checks CPUID at runtime (elided if compiled with target features enabled).
//! See [`token-registry.toml`](https://github.com/imazen/archmage/blob/main/token-registry.toml)
//! for the complete mapping of tokens to CPU features.
//!
//! **The `#[arcane]` and `#[rite]` macros** determine which `#[target_feature]`
//! attributes to emit. `#[arcane]` reads the token type from the function
//! signature. `#[rite]` works in three modes: token-based (reads the token
//! parameter), tier-based (`#[rite(v3)]` — no token needed), or multi-tier
//! (`#[rite(v3, v4, neon)]` — generates suffixed variants `fn_v3`, `fn_v4`,
//! `fn_neon`).
//!
//! Descriptive aliases are available for AI-assisted coding:
//! `#[token_target_features_boundary]` = `#[arcane]`,
//! `#[token_target_features]` = `#[rite]`,
//! `dispatch_variant!` = `incant!`.
//!
//! `#[arcane]` generates a sibling `#[target_feature]` function at the same
//! scope, plus a safe wrapper that calls it. Since both live in the same scope,
//! `self` and `Self` work naturally in methods. For trait impls, use
//! `#[arcane(_self = Type)]` (nested mode). On wrong architectures, functions
//! are cfg'd out by default. Use `incant!` for cross-arch dispatch.
//!
//! `#[rite]` applies `#[target_feature]` + `#[inline]` directly to the
//! function, with no wrapper and no boundary. It works in three modes:
//! - **Token-based** (`#[rite]`): reads the token from the function signature
//! - **Tier-based** (`#[rite(v3)]`): specifies features via tier name, no token needed
//! - **Multi-tier** (`#[rite(v3, v4, neon)]`): generates a suffixed copy for each tier
//!
//! **Use `#[arcane]` for all SIMD functions** — entry points and helpers alike.
//! When one `#[arcane]` function calls another with matching features, LLVM
//! inlines the wrapper away (zero overhead). `#[rite]` is available as an
//! advanced alternative that adds `#[target_feature]` + `#[inline]` directly
//! without a wrapper.
//!
//! Use concrete tokens like `X64V3Token` (AVX2+FMA) or `X64V4Token` (AVX-512).
//! For generic code, use tier traits like `HasX64V2` or `HasX64V4`.
//!
//! ## Safety
//!
//! Since Rust 1.85, value-based SIMD intrinsics (arithmetic, shuffle, compare,
//! bitwise) are safe inside `#[target_feature]` functions. Only pointer-based
//! memory operations remain unsafe — `import_intrinsics` handles this by
//! providing safe reference-based memory ops that shadow the pointer-based ones.
//!
//! Downstream crates can use `#![forbid(unsafe_code)]` when combining archmage
//! tokens + `#[arcane]`/`#[rite]` macros + `import_intrinsics`.
//!
//! ## Feature Flags
//!
//! - `std` (default): Enable std library support
//! - `avx512`: AVX-512 token support
//!
//! Macros (`#[arcane]`, `#[rite]`, `incant!`, etc.) are always available.

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
    X64V3Token,
};

// AVX-512 tokens (always available; summon() returns None on unsupported CPUs)
pub use tokens::{Avx512Fp16Token, Avx512Token, Server64, X64V4Token, X64V4xToken};

//! One import for everything you need to write SIMD with archmage.
//!
//! ```rust,ignore
//! use archmage::prelude::*;
//! ```
//!
//! # What's in the prelude
//!
//! The prelude re-exports five categories of items. Together they let you write
//! complete SIMD code with a single `use` statement and zero `unsafe` blocks.
//!
//! ## 1. Traits
//!
//! - [`SimdToken`] — The core trait. Provides `summon()` for runtime detection
//!   and `compiled_with()` for compile-time queries.
//! - [`IntoConcreteToken`] — Enables generic dispatch. Given an opaque token,
//!   try to downcast it to a concrete platform token.
//! - Tier traits: [`HasX64V2`], [`HasX64V4`], [`HasNeon`],
//!   [`HasNeonAes`], [`HasNeonSha3`], [`HasArm64V2`], [`HasArm64V3`] — For
//!   generic bounds when you want to accept any token at a given capability level.
//!
//! ## 2. Tokens
//!
//! All tokens compile on **all platforms**. On the wrong architecture,
//! `summon()` returns `None` and the token type is a zero-sized stub. This
//! means you rarely need `#[cfg(target_arch)]` in user code.
//!
//! **Friendly aliases:**
//! - [`Desktop64`] = [`X64V3Token`] — AVX2 + FMA (Haswell 2013+, Zen 1+)
//! - [`Server64`] = [`X64V4Token`] — + AVX-512
//! - [`Arm64`] = [`NeonToken`] — NEON (all 64-bit ARM)
//!
//! Also includes: [`ScalarToken`] (always available), [`X64V1Token`], [`X64V2Token`],
//! [`X64CryptoToken`] (V2 + PCLMULQDQ + AES-NI),
//! [`X64V3CryptoToken`] (V3 + VPCLMULQDQ + VAES, Zen 3+/Alder Lake+),
//! [`X64V3GfniCryptoToken`] (V3 Crypto + GFNI, including AVX2 without AVX-512),
//! [`Wasm128Token`], [`Wasm128RelaxedToken`],
//! [`NeonAesToken`], [`NeonSha3Token`], [`NeonCrcToken`],
//! [`Arm64V2Token`], [`Arm64V3Token`],
//! and the AVX-512 tokens ([`Avx512Token`], [`X64V4Token`], [`X64V4xToken`],
//! [`Avx512Fp16Token`]).
//!
//! ## 3. Macros
//!
//! - [`attune`] — Unified SIMD attribute: generates direct `#[target_feature]`
//!   helpers (`#[attune(v3)]`, `#[attune(_*)]`), safe token proof wrappers
//!   (`#[attune(wrap)]`, `#[attune(_*_t)]`), per-tier vector families
//!   (`define(f32x8)`), and auto-vectorized dispatchers (`#[attune(dispatch)]`).
//! - [`attuned!`] — Context-aware dispatch macro. Calls covered direct variants
//!   inside `#[attune]` bodies at zero cost, or summons proof wrappers (`_*_t`)
//!   at runtime outside one.
//! - [`reattune!`] — Explicit runtime re-selection of stronger tiers from inside
//!   an already feature-enabled context.
//! - Legacy spellings (superseded, queued for deprecation): [`arcane`], [`rite`],
//!   [`magetypes`], [`autoversion`], [`incant!`], and their descriptive aliases.
//!
//! ## 4. Platform intrinsics
//!
//! Re-exports `core::arch::{x86_64, aarch64, wasm32}::*` for the current
//! platform. Since Rust 1.87, **value-based intrinsics are safe inside
//! `#[target_feature]` functions**. Arithmetic, shuffle, compare, and bitwise
//! intrinsics need no `unsafe` inside `#[attune]`:
//!
//! ```rust,ignore
//! #[attune(v3)]
//! fn add(a: __m256, b: __m256) -> __m256 {
//!     _mm256_add_ps(a, b)  // Safe! No unsafe needed.
//! }
//! ```
//!
//! ## 5. Safe memory operations
//!
//! The prelude re-exports `safe_unaligned_simd` memory ops that shadow
//! `core::arch`'s pointer-based versions. These take references instead of
//! raw pointers — e.g., `_mm256_loadu_ps` takes `&[f32; 8]` instead of
//! `*const f32`.
//!
//! **Everything works unqualified:**
//!
//! ```rust,ignore
//! use archmage::prelude::*;
//!
//! #[attune(wrap, import_intrinsics)]
//! fn load(_token: X64V3Token, data: &[f32; 8]) -> __m256 {
//!     _mm256_loadu_ps(data)  // Safe! Takes a reference, not a pointer.
//! }
//! ```
//!
//! The combined intrinsics module uses Rust's name resolution rules: explicit
//! `safe_unaligned_simd` re-exports shadow the glob `core::arch` imports, so
//! memory ops always resolve to the safe versions.
//!
//! # How `attuned!` works
//!
//! `attuned!` dispatches across suffixed variants, wrapping each candidate in
//! `#[cfg]` gates that eliminate inapplicable architectures at compile time.
//!
//! By default (`attuned!(func(data))`), it dispatches across the portable tiers
//! `_v3`, `_neon`, `_wasm128`, and `_scalar` (calling `_<tier>_t` proof wrappers
//! from ordinary code, or covered `_<tier>` direct helpers inside `#[attune]`).
//! Specify an explicit list when opting into AVX-512 or a custom tier subset:
//!
//! ```rust,ignore
//! attuned!(func(data), [_v4(cfg(avx512)), _v3, _neon, _wasm128, _scalar])
//! ```
//!
//! Known tiers: `v1`, `v2`, `x64_crypto`, `v3`, `v3_crypto`, `v3_gfni_crypto`,
//! `v4`, `v4x`, `fp16`, `neon`, `neon_aes`, `neon_sha3`, `neon_crc`, `arm_v2`,
//! `arm_v3`, `wasm128`, `wasm128_relaxed`, `scalar`.
//!
//! **Default portable variants per platform:**
//!
//! | Platform | Default `_*` / `_*_t` | Explicit opt-in |
//! |----------|-----------------------|-----------------|
//! | x86_64 | `_v3`, `_scalar` | `+v4(cfg(avx512))`, `+v4x(cfg(avx512))` |
//! | aarch64 | `_neon`, `_scalar` | `+arm_v2`, `+arm_v3` |
//! | wasm32 | `_wasm128`, `_scalar` | `+wasm128_relaxed` |
//!
//! # Import styles
//!
//! ```rust,ignore
//! // Recommended: prelude gives you everything
//! use archmage::prelude::*;
//!
//! // Or use the combined intrinsics module directly:
//! use archmage::intrinsics::x86_64::*;

//! ```

// -- Traits --
pub use crate::tokens::HasX64V2;
pub use crate::tokens::HasX64V4;
pub use crate::tokens::IntoConcreteToken;
pub use crate::tokens::SimdToken;
pub use crate::tokens::{HasArm64V2, HasArm64V3, HasNeon, HasNeonAes, HasNeonSha3};

// -- Tokens (all compile on all platforms; summon() returns None on wrong arch) --
pub use crate::tokens::ScalarToken;
pub use crate::tokens::{
    Arm64, Arm64V2Token, Arm64V3Token, NeonAesToken, NeonCrcToken, NeonSha3Token, NeonToken,
};
#[allow(deprecated)]
pub use crate::tokens::{
    Avx2FmaToken, Desktop64, Sse2Token, X64CryptoToken, X64V1Token, X64V2Token, X64V3CryptoToken,
    X64V3GfniCryptoToken, X64V3Token,
};
pub use crate::tokens::{Avx512Fp16Token, Avx512Token, Server64, X64V4Token, X64V4xToken};
pub use crate::tokens::{Wasm128RelaxedToken, Wasm128Token};

// -- Macros --
pub use archmage_macros::{
    arcane, attune, attuned, autoversion, dispatch_variant, incant, magetypes, reattune, rite,
    simd_fn, simd_route, token_target_features, token_target_features_boundary,
};

// -- Platform intrinsics: core::arch types + value ops + safe memory ops --
// Uses the combined intrinsics module where safe_unaligned_simd's reference-based
// memory ops shadow core::arch's pointer-based versions automatically.
#[cfg(target_arch = "x86_64")]
pub use crate::intrinsics::x86_64::*;

#[cfg(target_arch = "x86")]
pub use crate::intrinsics::x86::*;

#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
pub use crate::intrinsics::aarch64::*;

#[cfg(target_arch = "wasm32")]
pub use crate::intrinsics::wasm32::*;

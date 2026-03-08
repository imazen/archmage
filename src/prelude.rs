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
//! - Tier traits: [`HasX64V2`], [`HasX64V4`] (with `avx512`), [`HasNeon`],
//!   [`HasNeonAes`], [`HasNeonSha3`] — For generic bounds when you want to
//!   accept any token at a given capability level.
//! - Width traits: [`Has128BitSimd`], [`Has256BitSimd`], [`Has512BitSimd`] —
//!   Deprecated; prefer concrete tokens instead.
//!
//! ## 2. Tokens
//!
//! All tokens compile on **all platforms**. On the wrong architecture,
//! `summon()` returns `None` and the token type is a zero-sized stub. This
//! means you rarely need `#[cfg(target_arch)]` in user code.
//!
//! **Friendly aliases:**
//! - [`X64V3Token`] = [`X64V3Token`] — AVX2 + FMA (Haswell 2013+, Zen 1+)
//! - [`Server64`] = [`X64V4Token`] — + AVX-512 (with `avx512` feature)
//! - [`Arm64`] = [`NeonToken`] — NEON (all 64-bit ARM)
//!
//! Also includes: [`ScalarToken`] (always available), [`X64V2Token`],
//! [`X64CryptoToken`] (V2 + PCLMULQDQ + AES-NI),
//! [`X64V3CryptoToken`] (V3 + VPCLMULQDQ + VAES, Zen 3+/Alder Lake+),
//! [`Wasm128Token`], [`Wasm128RelaxedToken`],
//! [`NeonAesToken`], [`NeonSha3Token`], [`NeonCrcToken`],
//! and the AVX-512 tokens ([`Avx512Token`], [`X64V4xToken`],
//! [`Avx512Fp16Token`]) when the `avx512` feature is enabled.
//!
//! ## 3. Macros
//!
//! Requires the `macros` feature (enabled by default).
//!
//! - [`arcane`] — Entry-point macro. Generates `#[target_feature]` wrappers
//!   with cross-architecture stubs. Use at API boundaries after `summon()`.
//! - [`rite`] — Internal helper macro. Adds `#[target_feature]` + `#[inline]`
//!   so LLVM inlines it into callers with matching features. Three modes:
//!   token-based (`#[rite]`), tier-based (`#[rite(v3)]` — no token needed),
//!   or multi-tier (`#[rite(v3, v4, neon)]` — generates suffixed variants).
//!   Use for functions called from `#[arcane]` context.
//! - [`incant!`] — Dispatch macro. Routes to suffixed functions (`_v3`, `_neon`,
//!   `_scalar`, etc.) based on platform. See [How `incant!` works](#how-incant-works).
//! - [`autoversion`] — Auto-vectorization macro. Write plain scalar code with a
//!   `SimdToken` placeholder; generates per-platform variants + runtime dispatcher.
//!   No intrinsics, no SIMD types — the compiler auto-vectorizes each variant.
//! - [`magetypes`] — Type generation macro. Expands a single function into
//!   per-platform variants with matching `#[cfg]` guards.
//!
//! ## 4. Platform intrinsics
//!
//! Re-exports `core::arch::{x86_64, aarch64, wasm32}::*` for the current
//! platform. Since Rust 1.85, **value-based intrinsics are safe inside
//! `#[target_feature]` functions**. This means arithmetic, shuffle, compare,
//! and bitwise intrinsics need no `unsafe` inside `#[arcane]`:
//!
//! ```rust,ignore
//! #[arcane]
//! fn add(_: X64V3Token, a: __m256, b: __m256) -> __m256 {
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
//! #[arcane(import_intrinsics)]
//! fn load(_token: X64V3Token, data: &[f32; 8]) -> __m256 {
//!     _mm256_loadu_ps(data)  // Safe! Takes a reference, not a pointer.
//! }
//! ```
//!
//! The combined intrinsics module uses Rust's name resolution rules: explicit
//! `safe_unaligned_simd` re-exports shadow the glob `core::arch` imports, so
//! memory ops always resolve to the safe versions.
//!
//! # How `incant!` works
//!
//! `incant!` generates calls to suffixed variants, wrapping each in `#[cfg]`
//! gates that eliminate wrong-platform branches at compile time.
//!
//! By default, it dispatches to `_v4`, `_v3`, `_neon`, `_wasm128`, and
//! `_scalar`. You can specify explicit tiers:
//!
//! ```rust,ignore
//! incant!(func(data), [v1, v3, neon])  // only these tiers + scalar
//! ```
//!
//! Known tiers: `v1`, `v2`, `x64_crypto`, `v3`, `v4`, `v4x`, `neon`, `neon_aes`,
//! `neon_sha3`, `neon_crc`, `arm_v2`, `arm_v3`, `wasm128`, `wasm128_relaxed`, `scalar`.
//!
//! **Required variants per platform (default tiers):**
//!
//! | Platform | Required | Optional |
//! |----------|----------|----------|
//! | x86_64 | `_v3`, `_scalar` | `_v4` (with `avx512` feature) |
//! | aarch64 | `_neon`, `_scalar` | — |
//! | wasm32 | `_wasm128`, `_scalar` | — |
//!
//! # Import styles
//!
//! ```rust,ignore
//! // Recommended: prelude gives you everything
//! use archmage::prelude::*;
//!
//! // Or use the combined intrinsics module directly:
//! use archmage::intrinsics::x86_64::*;
//!
//! // If you need the raw unsafe pointer version explicitly:
//! let v = unsafe { core::arch::x86_64::_mm256_loadu_ps(ptr) };
//! ```

// -- Traits --
pub use crate::tokens::HasX64V2;
pub use crate::tokens::HasX64V4;
pub use crate::tokens::IntoConcreteToken;
pub use crate::tokens::SimdToken;
pub use crate::tokens::{Has128BitSimd, Has256BitSimd, Has512BitSimd};
pub use crate::tokens::{HasArm64V2, HasArm64V3, HasNeon, HasNeonAes, HasNeonSha3};

// -- Tokens (all compile on all platforms; summon() returns None on wrong arch) --
pub use crate::tokens::ScalarToken;
pub use crate::tokens::{
    Arm64, Arm64V2Token, Arm64V3Token, NeonAesToken, NeonCrcToken, NeonSha3Token, NeonToken,
};
pub use crate::tokens::{
    Avx2FmaToken, Desktop64, Sse2Token, X64CryptoToken, X64V1Token, X64V2Token, X64V3CryptoToken,
    X64V3Token,
};
pub use crate::tokens::{Avx512Fp16Token, Avx512Token, Server64, X64V4Token, X64V4xToken};
pub use crate::tokens::{Wasm128RelaxedToken, Wasm128Token};

// -- Macros --
#[cfg(feature = "macros")]
pub use archmage_macros::{
    arcane, autoversion, dispatch_variant, incant, magetypes, rite, simd_fn, simd_route,
    token_target_features, token_target_features_boundary,
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

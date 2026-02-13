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
//! - [`Desktop64`] = [`X64V3Token`] — AVX2 + FMA (Haswell 2013+, Zen 1+)
//! - [`Server64`] = [`X64V4Token`] — + AVX-512 (with `avx512` feature)
//! - [`Arm64`] = [`NeonToken`] — NEON (all 64-bit ARM)
//!
//! Also includes: [`ScalarToken`] (always available), [`X64V2Token`],
//! [`Wasm128Token`], [`NeonAesToken`], [`NeonSha3Token`], [`NeonCrcToken`],
//! and the AVX-512 tokens ([`Avx512Token`], [`Avx512ModernToken`],
//! [`Avx512Fp16Token`]) when the `avx512` feature is enabled.
//!
//! ## 3. Macros
//!
//! Requires the `macros` feature (enabled by default).
//!
//! - [`arcane`] — Entry-point macro. Generates `#[target_feature]` wrappers
//!   with cross-architecture stubs. Use at API boundaries after `summon()`.
//! - [`rite`] — Internal helper macro. Adds `#[target_feature]` + `#[inline]`
//!   so LLVM inlines it into callers with matching features. Use for functions called from `#[arcane]` context.
//! - [`incant!`] — Dispatch macro. Routes to suffixed functions (`_v3`, `_neon`,
//!   `_scalar`, etc.) based on platform. See [How `incant!` works](#how-incant-works).
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
//! fn add(_: Desktop64, a: __m256, b: __m256) -> __m256 {
//!     _mm256_add_ps(a, b)  // Safe! No unsafe needed.
//! }
//! ```
//!
//! ## 5. Safe memory operations
//!
//! When the `safe_unaligned_simd` feature is enabled (default), the prelude
//! also re-exports `safe_unaligned_simd::{platform}::*`. These are
//! reference-based replacements for the ~159 pointer-based load/store
//! intrinsics. They take `&[f32; 8]` instead of `*const f32`.
//!
//! **Ambiguous names require explicit import.** Both `core::arch` and
//! `safe_unaligned_simd` export functions with identical names (e.g.,
//! `_mm256_loadu_ps`). Due to Rust's glob re-export rules, these overlapping
//! names are **not usable** through `use archmage::prelude::*` alone. You must
//! import the safe memory ops explicitly:
//!
//! ```rust,ignore
//! use archmage::prelude::*;
//! // Must import safe load/store explicitly — overlapping names are ambiguous
//! use safe_unaligned_simd::x86_64::{_mm256_loadu_ps, _mm256_storeu_ps};
//!
//! #[arcane]
//! fn load(_token: Desktop64, data: &[f32; 8]) -> __m256 {
//!     _mm256_loadu_ps(data)  // Safe! Takes a reference, not a pointer.
//! }
//! ```
//!
//! **What the prelude gives you without explicit import:**
//! - All tokens, traits, and macros
//! - SIMD types (`__m256`, `__m128i`, etc.)
//! - Value intrinsics (`_mm256_add_ps`, `_mm256_mul_ps`, etc.) — these have
//!   no overlap and are already safe inside `#[arcane]`
//!
//! **What needs explicit import:**
//! - Load/store ops from `safe_unaligned_simd` (or use full path:
//!   `safe_unaligned_simd::x86_64::_mm256_loadu_ps(data)`)
//!
//! # How `incant!` works
//!
//! `incant!` does **not** detect which variant functions exist. It's simpler
//! than that.
//!
//! The macro unconditionally generates calls to all suffixed variants (`_v3`,
//! `_v4`, `_neon`, `_wasm128`, `_scalar`), but wraps each in `#[cfg]` gates
//! that eliminate wrong-platform branches at compile time:
//!
//! - `_v4(...)` — inside `#[cfg(feature = "avx512")]`
//! - `_v3(...)` — inside `#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]`
//! - `_neon(...)` — inside `#[cfg(target_arch = "aarch64")]`
//! - `_wasm128(...)` — inside `#[cfg(target_arch = "wasm32")]`
//! - `_scalar(...)` — no gate, always compiled
//!
//! On x86_64 without `avx512`, the compiler never sees `_neon` or `_wasm128`
//! references because those `#[cfg]` blocks are eliminated. You only need to
//! define `_v3` and `_scalar`.
//!
//! **Required variants per platform:**
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
//! // Recommended: prelude + explicit safe memory ops
//! use archmage::prelude::*;
//! use safe_unaligned_simd::x86_64::{_mm256_loadu_ps, _mm256_storeu_ps};
//!
//! // Or use full paths for safe memory ops (no import needed):
//! let v = safe_unaligned_simd::x86_64::_mm256_loadu_ps(data);
//!
//! // If you need the raw unsafe pointer version:
//! let v = unsafe { core::arch::x86_64::_mm256_loadu_ps(ptr) };
//! ```

#![allow(ambiguous_glob_reexports)]

// -- Traits --
pub use crate::tokens::HasX64V2;
pub use crate::tokens::HasX64V4;
pub use crate::tokens::IntoConcreteToken;
pub use crate::tokens::SimdToken;
pub use crate::tokens::{Has128BitSimd, Has256BitSimd, Has512BitSimd};
pub use crate::tokens::{HasNeon, HasNeonAes, HasNeonSha3};

// -- Tokens (all compile on all platforms; summon() returns None on wrong arch) --
pub use crate::tokens::ScalarToken;
pub use crate::tokens::Wasm128Token;
pub use crate::tokens::{Arm64, NeonAesToken, NeonCrcToken, NeonSha3Token, NeonToken};
pub use crate::tokens::{Avx512Fp16Token, Avx512ModernToken, Avx512Token, Server64, X64V4Token};
pub use crate::tokens::{Desktop64, Sse2Token, X64V1Token, X64V2Token, X64V3Token};

// -- Macros --
#[cfg(feature = "macros")]
pub use archmage_macros::{arcane, incant, magetypes, rite, simd_fn, simd_route};

// -- core::arch intrinsics for the current platform --
#[cfg(target_arch = "x86_64")]
pub use core::arch::x86_64::*;

#[cfg(target_arch = "x86")]
pub use core::arch::x86::*;

#[cfg(target_arch = "aarch64")]
pub use core::arch::aarch64::*;

#[cfg(target_arch = "wasm32")]
pub use core::arch::wasm32::*;

// -- safe_unaligned_simd memory operations for the current platform --
// These shadow the core::arch versions with the same names. For example,
// _mm256_loadu_ps takes &[f32; 8] instead of *const f32.
#[cfg(all(feature = "safe_unaligned_simd", target_arch = "x86_64"))]
pub use safe_unaligned_simd::x86_64::*;

#[cfg(all(feature = "safe_unaligned_simd", target_arch = "x86"))]
pub use safe_unaligned_simd::x86::*;

#[cfg(all(
    feature = "safe_unaligned_simd",
    any(target_arch = "aarch64", target_arch = "arm64ec")
))]
pub use safe_unaligned_simd::aarch64::*;

#[cfg(all(feature = "safe_unaligned_simd", target_arch = "wasm32"))]
pub use safe_unaligned_simd::wasm32::*;

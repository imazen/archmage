//! One import for everything you need to write SIMD with archmage.
//!
//! ```rust,ignore
//! use archmage::prelude::*;
//! ```
//!
//! This re-exports tokens, traits, macros, `core::arch` intrinsics,
//! and `safe_unaligned_simd` memory operations for the current platform.
//!
//! The safe_unaligned_simd functions shadow the core::arch versions with the
//! same names — `_mm256_loadu_ps` takes `&[f32; 8]` instead of a raw pointer.

#![allow(ambiguous_glob_reexports)]

// -- Traits --
pub use crate::tokens::SimdToken;
pub use crate::tokens::IntoConcreteToken;
pub use crate::tokens::{Has128BitSimd, Has256BitSimd, Has512BitSimd};
pub use crate::tokens::HasX64V2;
#[cfg(feature = "avx512")]
pub use crate::tokens::HasX64V4;
pub use crate::tokens::{HasNeon, HasNeonAes, HasNeonSha3};

// -- Tokens (all compile on all platforms; summon() returns None on wrong arch) --
pub use crate::tokens::ScalarToken;
pub use crate::tokens::{X64V2Token, X64V3Token, Desktop64};
#[cfg(feature = "avx512")]
pub use crate::tokens::{X64V4Token, Server64, Avx512Token, Avx512ModernToken, Avx512Fp16Token};
pub use crate::tokens::{NeonToken, Arm64, NeonAesToken, NeonSha3Token, NeonCrcToken};
pub use crate::tokens::Simd128Token;

// -- Macros --
#[cfg(feature = "macros")]
pub use archmage_macros::{arcane, incant, magetypes, multiwidth, simd_fn, simd_route};

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
#[cfg(all(feature = "safe_unaligned_simd", target_arch = "x86_64"))]
pub use safe_unaligned_simd::x86_64::*;

#[cfg(all(feature = "safe_unaligned_simd", target_arch = "x86"))]
pub use safe_unaligned_simd::x86::*;

#[cfg(all(feature = "safe_unaligned_simd", any(target_arch = "aarch64", target_arch = "arm64ec")))]
pub use safe_unaligned_simd::aarch64::*;

#[cfg(all(feature = "safe_unaligned_simd", target_arch = "wasm32"))]
pub use safe_unaligned_simd::wasm32::*;

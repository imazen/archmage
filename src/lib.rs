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
//! use archmage::{Desktop64, HasAvx2, SimdToken, arcane};
//! use archmage::mem::avx;  // Safe load/store (enabled by default)
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
//!     // Desktop64: AVX2 + FMA + BMI2 (Haswell 2013+, Zen 1+)
//!     // CPUID check elided if compiled with -C target-cpu=native
//!     if let Some(token) = Desktop64::summon() {
//!         let result = multiply_add(token, &[1.0; 8], &[2.0; 8]);
//!     }
//! }
//! ```
//!
//! ## How It Works
//!
//! **Capability Tokens** are zero-sized proof types created via `summon()`, which
//! checks CPUID at runtime (elided if compiled with target features enabled).
//!
//! **The `#[arcane]` macro** generates an inner function with `#[target_feature]`,
//! making intrinsics safe inside. The token parameter proves CPU support was verified.
//!
//! **Generic bounds** like `impl HasAvx2` let functions accept any token that
//! provides AVX2 (e.g., `Avx2Token`, `Desktop64`, `Server64`).
//!
//! ## Feature Flags
//!
//! - `std` (default): Enable std library support
//! - `macros` (default): Enable `#[arcane]` attribute macro (alias: `#[simd_fn]`)
//! - `safe_unaligned_simd` (default): Safe load/store via references (exposed as `mem` module)
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

// Experimental features (unstable API, broken Shl/Shr needs const generics)
#[cfg(all(target_arch = "x86_64", feature = "__experiments"))]
#[cfg_attr(docsrs, doc(cfg(feature = "__experiments")))]
pub mod experiments;

// ============================================================================
// Re-exports at crate root for convenience
// ============================================================================

// Core trait
pub use tokens::SimdToken;

// Composite token trait
pub use tokens::CompositeToken;

// Capability marker traits
pub use tokens::{Has128BitSimd, Has256BitSimd, Has512BitSimd, HasFma};

// x86 feature marker traits (available on all architectures for cross-platform code)
// Note: SSE4.2 is the baseline - HasSse, HasSse2, HasSse41 have been removed
pub use tokens::{
    HasAvx, HasAvx2, HasAvx512bw, HasAvx512cd, HasAvx512dq, HasAvx512f, HasAvx512vbmi2, HasAvx512vl,
    HasSse42,
};

// aarch64 feature marker traits (available on all architectures)
pub use tokens::{HasArm64, HasArmAes, HasArmFp16, HasArmSha3, HasNeon};

// All tokens available on all architectures (summon() returns None on wrong arch)
pub use tokens::{
    // ARM tokens
    Arm64,
    // x86 tokens
    Avx2FmaToken,
    Avx2Token,
    Avx512Fp16Token,
    Avx512ModernToken,
    Avx512Vbmi2Token,
    Avx512Vbmi2VlToken,
    Avx512bwToken,
    Avx512bwVlToken,
    Avx512fToken,
    Avx512fVlToken,
    AvxToken,
    Desktop64,
    FmaToken,
    NeonAesToken,
    NeonFp16Token,
    NeonSha3Token,
    NeonToken,
    Server64,
    // WASM tokens
    Simd128Token,
    // SSE4.2 is the baseline (SSE, SSE2, SSE4.1 tokens removed)
    Sse42Token,
    X64V2Token,
    X64V3Token,
    X64V4Token,
};


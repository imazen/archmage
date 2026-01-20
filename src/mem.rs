//! Safe unaligned SIMD memory operations with token-based safety.
//!
//! This module provides safe, unaligned load and store operations for SIMD vectors,
//! gated by archmage capability tokens that prove the required CPU features are available.
//!
//! # Credits
//!
//! This module wraps the excellent [`safe_unaligned_simd`](https://crates.io/crates/safe_unaligned_simd)
//! crate by [Hakuyume](https://github.com/Hakuyume). That crate provides memory-safe
//! SIMD load/store operations using Rust references instead of raw pointers, eliminating
//! the need for `unsafe` blocks around memory operations.
//!
//! archmage adds runtime feature detection via capability tokens, ensuring that
//! `#[target_feature]` requirements are satisfied before calling these functions.
//! The combination provides:
//!
//! - **Memory safety** (from `safe_unaligned_simd`) - References guarantee valid memory
//! - **Feature safety** (from archmage) - Tokens prove CPU capabilities at runtime
//!
//! # Example
//!
//! ```rust,ignore
//! use archmage::{Avx2Token, SimdToken};
//! use archmage::mem::avx;
//!
//! fn process(data: &mut [f32]) {
//!     if let Some(token) = Avx2Token::try_new() {
//!         for chunk in data.chunks_exact_mut(8) {
//!             let v = avx::_mm256_loadu_ps(token, chunk.try_into().unwrap());
//!             // ... process v ...
//!             avx::_mm256_storeu_ps(token, chunk.try_into().unwrap(), v);
//!         }
//!     }
//! }
//! ```
//!
//! # Creating Mutable Array References from Slices
//!
//! **Beware of accidentally creating mutable references to temporary arrays.**
//!
//! Rust will implicitly clone an array from a slice and return a mutable reference
//! to that clone if not wrapped properly in parentheses.
//!
//! ```rust,ignore
//! // ✅ Correct: mutable reference into original slice
//! let out: &mut [f32; 8] = (&mut chunk[..8]).try_into().unwrap();
//!
//! // ✅ Also correct: explicit turbofish
//! let out = TryInto::<&mut [f32; 8]>::try_into(&mut chunk[..8]).unwrap();
//!
//! // ❌ WRONG: creates a COPY, modifications won't reflect back!
//! let out: &mut [f32; 8] = &mut chunk[..8].try_into().unwrap();
//! ```
//!
//! The incorrect version clones the data into a temporary array and returns a
//! mutable reference to the copy. Any modifications will be lost.
//!
//! **Future improvement:** Once [`as_mut_array`](https://doc.rust-lang.org/std/primitive.slice.html#method.as_mut_array)
//! stabilizes (currently unstable as of rustc 1.91), prefer using that to sidestep this footgun entirely:
//!
//! ```rust,ignore
//! // 🔮 Future: clean and safe (requires #![feature(slice_as_array)])
//! let out: &mut [f32; 8] = chunk[..8].as_mut_array().unwrap();
//! ```
//!
//! # Feature Sets
//!
//! Operations are organized by the CPU features they require:
//!
//! - [`sse`] - SSE (6 functions)
//! - [`sse2`] - SSE2 (20 functions)
//! - [`avx`] - AVX (17 functions)
//! - [`avx512f`] - AVX-512F (49 functions)
//! - [`avx512f_vl`] - AVX-512F + VL for 128/256-bit ops (86 functions)
//! - [`avx512bw`] - AVX-512BW (13 functions)
//! - [`avx512bw_vl`] - AVX-512BW + VL (26 functions)
//! - [`avx512vbmi2`] - AVX-512 VBMI2 (6 functions)
//! - [`avx512vbmi2_vl`] - AVX-512 VBMI2 + VL (12 functions)
//!
//! # Implementation
//!
//! The wrappers in this module are auto-generated from `safe_unaligned_simd`.
//! See `xtask/src/main.rs` for the generator.
//!
//! ## AArch64 / NEON Support
//!
//! AArch64 wrappers are not yet generated. Since NEON is baseline on AArch64 (always
//! available), you can use `safe_unaligned_simd::aarch64::*` functions directly inside
//! a `#[simd_fn]` function with a `NeonToken`:
//!
//! ```rust,ignore
//! use archmage::{NeonToken, SimdToken, simd_fn};
//!
//! #[simd_fn]
//! fn process(token: NeonToken, data: &[f32; 4]) -> float32x4_t {
//!     safe_unaligned_simd::aarch64::vld1q_f32(data)
//! }
//! ```
//!
//! The generator cannot parse the macro-based aarch64 functions in `safe_unaligned_simd`.
//! Token-gated wrappers may be added manually in the future if needed.

// Re-export auto-generated wrappers
#[path = "generated/mod.rs"]
mod generated;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use generated::x86::*;

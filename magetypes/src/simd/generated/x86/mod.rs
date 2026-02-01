//! x86-64 SIMD types.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

pub mod w128;
pub mod w256;
#[cfg(feature = "avx512")]
pub mod w512;

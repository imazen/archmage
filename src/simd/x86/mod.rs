//! x86-64 SIMD types.

pub mod w128;
pub mod w256;
#[cfg(feature = "avx512")]
pub mod w512;

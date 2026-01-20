//! Token-gated wrappers for safe_unaligned_simd.
//!
//! This module re-exports safe_unaligned_simd functions with archmage token gating.
//! Each function takes a token as its first parameter, proving the required
//! CPU features are available at runtime.
//!
//! **Auto-generated** - do not edit manually. See `xtask/src/main.rs`.
//!
//! ## Feature Coverage
//!
//! - [`avx`]: 17 functions (`avx`)
//! - [`avx512bw`]: 13 functions (`avx512bw`)
//! - [`avx512bw_vl`]: 26 functions (`avx512bw,avx512vl`)
//! - [`avx512f`]: 49 functions (`avx512f`)
//! - [`avx512f_vl`]: 86 functions (`avx512f,avx512vl`)
//! - [`avx512vbmi2`]: 6 functions (`avx512vbmi2`)
//! - [`avx512vbmi2_vl`]: 12 functions (`avx512vbmi2,avx512vl`)
//! - [`sse`]: 6 functions (`sse`)
//! - [`sse2`]: 20 functions (`sse2`)

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use x86::*;

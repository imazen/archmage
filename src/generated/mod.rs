//! Token-gated wrappers for safe_unaligned_simd.
//!
//! This module re-exports safe_unaligned_simd functions with archmage token gating.
//! Each function takes a token as its first parameter, proving the required
//! CPU features are available at runtime.
//!
//! **Auto-generated** - do not edit manually. See `xtask/src/main.rs`.
//!
//! ## x86/x86_64 Feature Coverage
//!
//! - [`x86::avx`]: 17 functions (`avx`)
//! - [`x86::avx512bw`]: 13 functions (`avx512bw`)
//! - [`x86::avx512bw_vl`]: 26 functions (`avx512bw,avx512vl`)
//! - [`x86::avx512f`]: 49 functions (`avx512f`)
//! - [`x86::avx512f_vl`]: 86 functions (`avx512f,avx512vl`)
//! - [`x86::avx512vbmi2`]: 6 functions (`avx512vbmi2`)
//! - [`x86::avx512vbmi2_vl`]: 12 functions (`avx512vbmi2,avx512vl`)
//! - [`x86::sse`]: 6 functions (`sse`)
//! - [`x86::sse2`]: 20 functions (`sse2`)
//!
//! ## AArch64 Feature Coverage
//!
//! - [`aarch64::neon`]: 240 functions (`neon`)

// Note: safe_unaligned_simd only provides x86_64 module, not x86 (i686)
#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

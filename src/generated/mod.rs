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
//! - [`x86::modern`]: 6 functions (`avx512vbmi2`) (requires `avx512` feature)
//! - [`x86::modern_vl`]: 12 functions (`avx512vbmi2,avx512vl`) (requires `avx512` feature)
//! - [`x86::v4`]: 49 functions (`avx512f`) (requires `avx512` feature)
//! - [`x86::v4_bw`]: 13 functions (`avx512bw`) (requires `avx512` feature)
//! - [`x86::v4_bw_vl`]: 26 functions (`avx512bw,avx512vl`) (requires `avx512` feature)
//! - [`x86::v4_vl`]: 86 functions (`avx512f,avx512vl`) (requires `avx512` feature)
//!
//! ## AArch64 Feature Coverage
//!
//! - [`aarch64::neon`]: 240 functions (`neon`)

// Note: safe_unaligned_simd only provides x86_64 module, not x86 (i686)
#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

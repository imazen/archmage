//! x86/x86_64 token-gated wrappers.
//!
//! **Auto-generated** - do not edit manually.

pub mod avx;
#[cfg(feature = "avx512")]
pub mod modern;
#[cfg(feature = "avx512")]
pub mod modern_vl;
#[cfg(feature = "avx512")]
pub mod v4;
#[cfg(feature = "avx512")]
pub mod v4_bw;
#[cfg(feature = "avx512")]
pub mod v4_bw_vl;
#[cfg(feature = "avx512")]
pub mod v4_vl;

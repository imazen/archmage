//! Integration layers for external crates
//!
//! Each integration is optional and enabled by the corresponding feature flag.

#[cfg(feature = "__wide")]
#[cfg_attr(docsrs, doc(cfg(feature = "__wide")))]
pub mod wide_ops;

#[cfg(feature = "safe_unaligned_simd")]
#[cfg_attr(docsrs, doc(cfg(feature = "safe_unaligned_simd")))]
pub mod safe_simd;

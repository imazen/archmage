//! Integration layers for external crates
//!
//! Each integration is optional and enabled by the corresponding feature flag.

#[cfg(feature = "wide")]
#[cfg_attr(docsrs, doc(cfg(feature = "wide")))]
pub mod wide_ops;

#[cfg(feature = "safe-simd")]
#[cfg_attr(docsrs, doc(cfg(feature = "safe-simd")))]
pub mod safe_simd;

#[cfg(any(feature = "multiversion", feature = "multiversed"))]
#[cfg_attr(
    docsrs,
    doc(cfg(any(feature = "multiversion", feature = "multiversed")))
)]
pub mod multiversion_support;

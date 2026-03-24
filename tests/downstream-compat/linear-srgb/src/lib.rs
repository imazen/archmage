//! Downstream compatibility test for linear-srgb.
//!
//! Verifies that published linear-srgb compiles against local archmage in both
//! configurations:
//!   cargo check                      (no avx512)
//!   cargo check --features avx512    (with avx512)
//!
//! linear-srgb uses incant! with explicit [v4, v3, neon] tier lists and
//! cfg-gated _v4 functions — the pattern that motivated the skip_avx512
//! logic in resolve_tiers().
#![deny(warnings)]

// Force linear-srgb to be compiled (not just resolved).
pub use linear_srgb::default;

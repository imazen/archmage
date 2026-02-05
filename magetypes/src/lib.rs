//! # magetypes
//!
//! Token-gated SIMD types with natural operators.
//!
//! This crate provides SIMD vector types (`f32x8`, `i32x4`, etc.) that use
//! [archmage](https://docs.rs/archmage) tokens for safe construction.
//!
//! ## Supported Platforms
//!
//! - **x86-64**: x86-64-v3 (128-bit, 256-bit), AVX-512 (512-bit)
//! - **AArch64**: NEON (128-bit)
//! - **WASM**: SIMD128 (128-bit) - compile with `RUSTFLAGS="-C target-feature=+simd128"`
//!
//! ## Example
//!
//! ```no_run
//! # #[cfg(target_arch = "x86_64")]
//! # fn main() {
//! use archmage::{X64V3Token, SimdToken};
//! use magetypes::simd::f32x8;
//!
//! if let Some(token) = X64V3Token::summon() {
//!     let a = f32x8::splat(token, 1.0);
//!     let b = f32x8::splat(token, 2.0);
//!     let c = a + b;  // Natural operators!
//!     println!("Result: {:?}", c.to_array());
//! }
//! # }
//! # #[cfg(not(target_arch = "x86_64"))]
//! # fn main() {}
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

// Re-export archmage for convenience
pub use archmage;

// SimdTypes trait - associates SIMD types with tokens
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
mod types;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
pub use types::SimdTypes;

// Cross-tier casting utilities
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
pub mod cast;

// Platform-appropriate types via prelude
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
pub mod prelude;

// Auto-generated SIMD types with natural operators
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
#[cfg_attr(
    docsrs,
    doc(cfg(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))
)]
pub mod simd;

// Width dispatch trait for accessing all SIMD sizes from any token
mod width;
pub use width::WidthDispatch;

// Types are accessed via magetypes::simd::* - no root re-exports
// This keeps the API stable during development

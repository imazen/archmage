//! # magetypes
//!
//! [Guide](https://imazen.github.io/archmage/) · [Intrinsics browser](https://imazen.github.io/archmage/intrinsics/) · [Archmage API](https://docs.rs/archmage/latest/archmage/) · [Magetypes API](https://docs.rs/magetypes/latest/magetypes/)
//!
//! Token-gated SIMD vectors with natural operators. This complete kernel is adapted
//! from zenfilters; its generated caller establishes the target-feature context.
//!
//! ```rust
//! #![forbid(unsafe_code)]
//! use archmage::prelude::*;
//!
//! #[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
//! fn gain_impl(token: Token, plane: &mut [f32], gain: f32) {
//!     let factor = f32x8::splat(token, gain);
//!     let (chunks, tail) = f32x8::partition_slice_mut(token, plane);
//!     for chunk in chunks {
//!         (f32x8::load(token, chunk) * factor).store(chunk);
//!     }
//!     for value in tail {
//!         *value *= gain;
//!     }
//! }
//!
//! pub fn apply_gain(plane: &mut [f32], gain: f32) {
//!     incant!(gain_impl(plane, gain), [v3, neon, wasm128, scalar])
//! }
//!
//!
//! let mut plane = [2.0; 11];
//! apply_gain(&mut plane, 0.5);
//! assert_eq!(plane, [1.0; 11]);
//! ```
//!
//! Use [generic functions and const modes](https://imazen.github.io/archmage/magetypes/dispatch/types-and-dispatch/)
//! for reusable data/algorithm specialization. `define(...)` is optional shorthand.
//! Logical [`f32x8<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html) stays eight lanes; `w512` enables wider shapes and `avx512`
//! adds native AVX-512 implementations. See the [ISA contracts](https://imazen.github.io/archmage/magetypes/isa-quirks/)
//! for numerical differences and fixups.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
// Every backend trait method that accepts `self` threads the CPU-feature
// token through the call. The clippy `from_*` / `to_*` self-convention rule
// assumes constructor methods take no `self`; that assumption doesn't hold
// here — `self` is the token (the feature-availability proof), not an
// instance being converted.
#![allow(clippy::wrong_self_convention)]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

// Re-export archmage for convenience
pub use archmage;

// Pure-Rust math functions for no_std scalar backends
#[doc(hidden)]
pub mod nostd_math;

// SimdTypes trait - associates SIMD types with tokens
mod simd_storage;
mod types;
pub use types::SimdTypes;

// Cross-tier casting utilities
pub mod cast;

// Platform-appropriate types via prelude
pub mod prelude;

// Auto-generated SIMD types with natural operators
pub mod simd;

// Width dispatch trait for accessing all SIMD sizes from any token
mod width;
pub use width::WidthDispatch;

// Compile-fail adversarial doctests proving the UFCS-tokenless bypass is
// closed on every backend trait category. Module contents are inert at
// runtime; rustdoc compiles the `//!` doctests under `cargo test --doc`.
#[doc(hidden)]
pub mod bypass_adversarial;

// Types are accessed via magetypes::simd::* - no root re-exports
// This keeps the API stable during development

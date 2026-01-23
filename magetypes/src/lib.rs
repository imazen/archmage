//! # magetypes
//!
//! Token-gated SIMD types with natural operators.
//!
//! This crate provides SIMD vector types (`f32x8`, `i32x4`, etc.) that use
//! [archmage](https://docs.rs/archmage) tokens for safe construction.
//!
//! ## Supported Platforms
//!
//! - **x86-64**: SSE4.1 (128-bit), AVX2 (256-bit), AVX-512 (512-bit)
//! - **AArch64**: NEON (128-bit)
//! - **WASM**: SIMD128 (128-bit) - compile with `RUSTFLAGS="-C target-feature=+simd128"`
//!
//! ## Example
//!
//! ```no_run
//! # #[cfg(target_arch = "x86_64")]
//! # fn main() {
//! use archmage::{Avx2FmaToken, SimdToken};
//! use magetypes::f32x8;
//!
//! if let Some(token) = Avx2FmaToken::try_new() {
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

// Auto-generated SIMD types with natural operators
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32"))]
#[cfg_attr(docsrs, doc(cfg(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32"))))]
pub mod simd;

// Width dispatch trait for accessing all SIMD sizes from any token
mod width;
pub use width::WidthDispatch;

// Re-export common types at crate root for convenience
#[cfg(target_arch = "x86_64")]
pub use simd::{f32x4, f32x8, f64x2, f64x4, i8x16, i8x32, i16x8, i16x16, i32x4, i32x8};
#[cfg(target_arch = "x86_64")]
pub use simd::{u8x16, u8x32, u16x8, u16x16, u32x4, u32x8, u64x2, u64x4};

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub use simd::{f32x16, f64x8, i8x64, i16x32, i32x16, i64x8, u8x64, u16x32, u32x16, u64x8};

#[cfg(target_arch = "aarch64")]
pub use simd::{f32x4, f64x2, i8x16, i16x8, i32x4, i64x2, u8x16, u16x8, u32x4, u64x2};

#[cfg(target_arch = "wasm32")]
pub use simd::{f32x4, f64x2, i8x16, i16x8, i32x4, i64x2, u8x16, u16x8, u32x4, u64x2};

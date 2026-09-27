//! Generic SIMD types parameterized by backend token.
//!
//! These types are the strategy-pattern wrappers: `f32x8<T>` where `T`
//! determines the platform implementation. Write one generic function,
//! get monomorphized per backend at dispatch time.
//!
//! All type definitions are auto-generated in the `generated/` subfolder
//! by `cargo xtask generate`. This file is handwritten and should not be
//! purged during regeneration.

//!
//! # Constructor proofs
//!
//! The default aliases retain token arguments (`f32x8::zero(token)`).
//! [`crate::simd::generic::local`] aliases offer short constructors in matching
//! target-feature contexts (`f32x8::zero()`). Both modes also provide explicit
//! `_with_token` constructors with no caller target-feature requirement:
//!
//! ```
//! use archmage::ScalarToken;
//! use magetypes::simd::{backends::F32x8Backend, generic::local};
//!
//! fn load<T: F32x8Backend>(token: T, values: &[f32; 8]) -> local::f32x8<T> {
//!     local::f32x8::load_with_token(token, values)
//! }
//! assert_eq!(load(ScalarToken, &[2.0; 8]).to_array(), [2.0; 8]);
//! ```
//!
//! This includes byte/slice views and vector conversions where supported.
//! Native raw types have `from_raw_with_token(token, raw)`. The original
//! constructors remain available; selecting a different mode changes the Rust
//! vector type, so owned values crossing a mode boundary use `.into()`.

mod convert_f16;
#[path = "generated/mod.rs"]
pub mod core_types;
mod cross_width;
mod modes;
pub use convert_f16::F16Convert;
pub use cross_width::F32x8FromHalves;
#[cfg(feature = "w512")]
pub use cross_width::F32x16FromHalves;
pub use modes::{ConstructorMode, Context, Explicit};
include!("generated/aliases.rs");

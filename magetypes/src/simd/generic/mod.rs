//! Generic SIMD types parameterized by backend token.
//!
//! These types are the strategy-pattern wrappers: `f32x8<T>` where `T`
//! determines the platform implementation. Write one generic function,
//! get monomorphized per backend at dispatch time.
//!
//! All type definitions are auto-generated in the `generated/` subfolder
//! by `cargo xtask generate`. This file is handwritten and should not be
//! purged during regeneration.

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

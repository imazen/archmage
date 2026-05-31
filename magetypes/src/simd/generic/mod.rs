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
mod cross_width;
mod generated;
pub use convert_f16::{f16_to_f32_slice, f16_to_f32x4, f32_to_f16_slice, f32_to_f16x4};
pub use cross_width::F32x8FromHalves;
#[cfg(feature = "w512")]
pub use cross_width::F32x16FromHalves;
pub use generated::*;

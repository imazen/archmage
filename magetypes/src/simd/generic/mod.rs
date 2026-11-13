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
mod fast_ops;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
mod gather;
mod generated;
mod lane_ops;
mod quant_ops;
mod subtiers;
pub use convert_f16::F16Convert;
pub use cross_width::F32x8FromHalves;
#[cfg(feature = "w512")]
pub use cross_width::F32x16FromHalves;
#[cfg(feature = "w512")]
pub use fast_ops::F32x16ToU32;
pub use fast_ops::{F32x4ToU32, F32x8ToU32};
pub use generated::*;
#[cfg(feature = "w512")]
pub use quant_ops::I32x16DotI8;
pub use quant_ops::{I32x4DotI8, I32x8DotI8};
pub use subtiers::MagetypesToken;

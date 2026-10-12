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
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
mod gather;
mod generated;
pub use convert_f16::F16Convert;
#[cfg(target_arch = "aarch64")]
#[rustversion::since(1.94)]
pub use convert_f16::{f16_to_f32_slice_arm_v2_t, f32_to_f16_slice_arm_v2_t};
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub use convert_f16::{f16_to_f32_slice_v4_t, f32_to_f16_slice_v4_t};
pub use cross_width::F32x8FromHalves;
#[cfg(feature = "w512")]
pub use cross_width::F32x16FromHalves;
pub use generated::*;

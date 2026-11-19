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
mod float_layout;
#[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
pub(crate) use float_layout::half4;
mod float_numerics;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
mod gather;
mod generated;
mod int_numerics;
mod rgb_block_ops;
pub use convert_f16::F16Convert;
pub use cross_width::F32x8FromHalves;
#[cfg(feature = "w512")]
pub use cross_width::F32x16FromHalves;
pub use generated::*;

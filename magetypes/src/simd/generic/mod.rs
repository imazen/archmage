//! Generic SIMD types parameterized by backend token.
//!
//! These types are the strategy-pattern wrappers: `f32x8<T>` where `T`
//! determines the platform implementation. Write one generic function,
//! get monomorphized per backend at dispatch time.
//!
//! All type definitions are auto-generated in the `generated/` subfolder
//! by `cargo xtask generate`. This file is handwritten and should not be
//! purged during regeneration.

// Core vector types and operations
mod generated;

// Float arithmetic policies
mod float_numerics;

// Integer conversion policies
mod int_numerics;

// Packed lane and half-precision conversion
mod float_layout;
#[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
pub(crate) use float_layout::half4;

// Slice conversion
mod convert_f16;

// Packed RGB blocks
mod rgb_block_ops;

// Cross-width conversions
mod cross_width;

// Architecture-specific gathers
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
mod gather;

pub use convert_f16::F16Convert;
pub use cross_width::F32x8FromHalves;
#[cfg(feature = "w512")]
pub use cross_width::F32x16FromHalves;
pub use generated::*;

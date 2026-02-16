//! Platform-appropriate SIMD types without manual `#[cfg]` blocks.
//!
//! Import this module to get the "best" SIMD types for your compile target:
//!
//! - **x86_64**: 256-bit types (f32x8, i32x8, etc.) with `X64V3Token`
//! - **x86_64 + avx512**: 512-bit types (f32x16, i32x16, etc.) with `X64V4Token`
//! - **aarch64**: 128-bit types (f32x4, i32x4, etc.) with `NeonToken`
//! - **wasm32**: 128-bit types (f32x4, i32x4, etc.) with `Wasm128Token`
//!
//! # Example
//!
//! ```no_run
//! use magetypes::prelude::*;
//!
//! # #[cfg(target_arch = "x86_64")]
//! # fn example() {
//! if let Some(token) = RecommendedToken::summon() {
//!     let a = F32Vec::splat(token, 1.0);
//!     let b = F32Vec::splat(token, 2.0);
//!     let c = a + b;
//!     println!("Lanes: {}", LANES);
//! }
//! # }
//! ```
//!
//! # What each alias maps to
//!
//! ## `RecommendedToken`
//!
//! The token type for the platform's recommended SIMD tier:
//!
//! | Platform | `RecommendedToken` | Why |
//! |----------|--------------------|-----|
//! | x86_64 | `X64V3Token` | AVX2+FMA covers 95%+ of x86_64 CPUs since 2013 |
//! | x86_64 + `avx512` | `X64V4Token` | AVX-512 for server workloads |
//! | aarch64 | `NeonToken` | NEON is baseline on all 64-bit ARM |
//! | wasm32 | `Wasm128Token` | SIMD128 is the only WASM SIMD tier |
//!
//! ## Type aliases (`F32Vec`, `I32Vec`, etc.)
//!
//! Platform-appropriate SIMD vector types. The concrete type and lane count
//! vary by platform — `F32Vec` is `f32x8` on x86_64 but `f32x4` on ARM.
//!
//! ## Lane counts (`LANES`, `F32_LANES`, `F64_LANES`, `I32_LANES`)
//!
//! Constants for the number of lanes in each vector type. Use these instead
//! of hardcoding lane counts when writing code with the prelude aliases.
//!
//! - `LANES` / `F32_LANES`: f32 lanes (8 on x86, 16 with avx512, 4 on ARM/WASM)
//! - `F64_LANES`: f64 lanes (4 on x86, 8 with avx512, 2 on ARM/WASM)
//! - `I32_LANES`: i32 lanes (same as `F32_LANES`)
//!
//! # When to use this prelude
//!
//! Use this for **quick prototyping** and in **`#[magetypes]` macro contexts**
//! where the macro substitutes the correct types per platform. The aliases
//! hide which concrete type you're using, which is the point — but it also
//! means the same source code produces different lane counts on different
//! architectures.
//!
//! For production code where you want explicit control over vector widths,
//! import concrete types directly:
//!
//! ```rust,ignore
//! use magetypes::simd::f32x8;  // Always 8 lanes, polyfilled on ARM/WASM
//! ```
//!
//! # Prelude vs `#[magetypes]`
//!
//! - **Prelude**: For non-generic code targeting a single platform
//! - **`#[magetypes]`**: For generating multiple platform variants

// Re-export SimdToken trait for summon()
pub use archmage::SimdToken;

// ============================================================================
// x86_64 without AVX-512: 256-bit (AVX2)
// ============================================================================

#[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
pub use crate::simd::{
    f32x8 as F32Vec, f64x4 as F64Vec, i32x8 as I32Vec,
};
#[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
pub use crate::simd::x86::w256::{
    i8x32 as I8Vec, i16x16 as I16Vec,
    i64x4 as I64Vec, u8x32 as U8Vec, u16x16 as U16Vec, u32x8 as U32Vec, u64x4 as U64Vec,
};

#[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
pub use archmage::X64V3Token as RecommendedToken;

#[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
/// Number of f32 lanes in the recommended vector type.
pub const LANES: usize = 8;

#[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
/// Number of f32 lanes in the recommended vector type.
pub const F32_LANES: usize = 8;

#[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
/// Number of f64 lanes in the recommended vector type.
pub const F64_LANES: usize = 4;

#[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
/// Number of i32 lanes in the recommended vector type.
pub const I32_LANES: usize = 8;

// ============================================================================
// x86_64 with AVX-512: 512-bit
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub use crate::simd::x86::w512::{
    f32x16 as F32Vec, f64x8 as F64Vec, i8x64 as I8Vec, i16x32 as I16Vec, i32x16 as I32Vec,
    i64x8 as I64Vec, u8x64 as U8Vec, u16x32 as U16Vec, u32x16 as U32Vec, u64x8 as U64Vec,
};

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub use archmage::X64V4Token as RecommendedToken;

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
/// Number of f32 lanes in the recommended vector type.
pub const LANES: usize = 16;

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
/// Number of f32 lanes in the recommended vector type.
pub const F32_LANES: usize = 16;

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
/// Number of f64 lanes in the recommended vector type.
pub const F64_LANES: usize = 8;

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
/// Number of i32 lanes in the recommended vector type.
pub const I32_LANES: usize = 16;

// ============================================================================
// aarch64: 128-bit (NEON)
// ============================================================================

#[cfg(target_arch = "aarch64")]
pub use crate::simd::arm::w128::{
    f32x4 as F32Vec, f64x2 as F64Vec, i8x16 as I8Vec, i16x8 as I16Vec, i32x4 as I32Vec,
    i64x2 as I64Vec, u8x16 as U8Vec, u16x8 as U16Vec, u32x4 as U32Vec, u64x2 as U64Vec,
};

#[cfg(target_arch = "aarch64")]
pub use archmage::NeonToken as RecommendedToken;

#[cfg(target_arch = "aarch64")]
/// Number of f32 lanes in the recommended vector type.
pub const LANES: usize = 4;

#[cfg(target_arch = "aarch64")]
/// Number of f32 lanes in the recommended vector type.
pub const F32_LANES: usize = 4;

#[cfg(target_arch = "aarch64")]
/// Number of f64 lanes in the recommended vector type.
pub const F64_LANES: usize = 2;

#[cfg(target_arch = "aarch64")]
/// Number of i32 lanes in the recommended vector type.
pub const I32_LANES: usize = 4;

// ============================================================================
// wasm32: 128-bit (SIMD128)
// ============================================================================

#[cfg(target_arch = "wasm32")]
pub use crate::simd::wasm::w128::{
    f32x4 as F32Vec, f64x2 as F64Vec, i8x16 as I8Vec, i16x8 as I16Vec, i32x4 as I32Vec,
    i64x2 as I64Vec, u8x16 as U8Vec, u16x8 as U16Vec, u32x4 as U32Vec, u64x2 as U64Vec,
};

#[cfg(target_arch = "wasm32")]
pub use archmage::Wasm128Token as RecommendedToken;

#[cfg(target_arch = "wasm32")]
/// Number of f32 lanes in the recommended vector type.
pub const LANES: usize = 4;

#[cfg(target_arch = "wasm32")]
/// Number of f32 lanes in the recommended vector type.
pub const F32_LANES: usize = 4;

#[cfg(target_arch = "wasm32")]
/// Number of f64 lanes in the recommended vector type.
pub const F64_LANES: usize = 2;

#[cfg(target_arch = "wasm32")]
/// Number of i32 lanes in the recommended vector type.
pub const I32_LANES: usize = 4;

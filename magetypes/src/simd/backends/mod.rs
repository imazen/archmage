//! Backend traits for generic SIMD types.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#![allow(non_camel_case_types)]

pub(crate) mod sealed;

mod f32x4;
pub use f32x4::F32x4Backend;

mod f32x8;
pub use f32x8::F32x8Backend;

mod f64x2;
pub use f64x2::F64x2Backend;

mod f64x4;
pub use f64x4::F64x4Backend;

mod i32x4;
pub use i32x4::I32x4Backend;

mod i32x8;
pub use i32x8::I32x8Backend;

mod u32x4;
pub use u32x4::U32x4Backend;

mod u32x8;
pub use u32x8::U32x8Backend;

mod i64x2;
pub use i64x2::I64x2Backend;

mod i64x4;
pub use i64x4::I64x4Backend;

mod i8x16;
pub use i8x16::I8x16Backend;

mod i8x32;
pub use i8x32::I8x32Backend;

mod u8x16;
pub use u8x16::U8x16Backend;

mod u8x32;
pub use u8x32::U8x32Backend;

mod i16x8;
pub use i16x8::I16x8Backend;

mod i16x16;
pub use i16x16::I16x16Backend;

mod u16x8;
pub use u16x8::U16x8Backend;

mod u16x16;
pub use u16x16::U16x16Backend;

mod u64x2;
pub use u64x2::U64x2Backend;

mod u64x4;
pub use u64x4::U64x4Backend;

mod f32x16;
pub use f32x16::F32x16Backend;

mod f64x8;
pub use f64x8::F64x8Backend;

mod i8x64;
pub use i8x64::I8x64Backend;

mod u8x64;
pub use u8x64::U8x64Backend;

mod i16x32;
pub use i16x32::I16x32Backend;

mod u16x32;
pub use u16x32::U16x32Backend;

mod i32x16;
pub use i32x16::I32x16Backend;

mod u32x16;
pub use u32x16::U32x16Backend;

mod i64x8;
pub use i64x8::I64x8Backend;

mod u64x8;
pub use u64x8::U64x8Backend;

mod convert;
pub use convert::{
    F32x4Convert, F32x8Convert, I64x2Bitcast, I64x4Bitcast, U32x4Bitcast, U32x8Bitcast,
};

mod convert_int;
pub use convert_int::{
    I8x16Bitcast, I8x32Bitcast, I16x8Bitcast, I16x16Bitcast, U64x2Bitcast, U64x4Bitcast,
};

/// x86-64 baseline (SSE2).
pub type x64v1 = archmage::X64V1Token;
/// x86-64 v2 (SSE4.2 + POPCNT).
pub type x64v2 = archmage::X64V2Token;
/// x86-64 v3 (AVX2 + FMA).
pub type x64v3 = archmage::X64V3Token;
/// x86-64 v4 (AVX-512).
pub type x64v4 = archmage::X64V4Token;
/// AVX-512 with modern extensions.
pub type avx512_modern = archmage::Avx512ModernToken;
/// AArch64 NEON.
pub type neon = archmage::NeonToken;
/// WASM SIMD128.
pub type wasm128 = archmage::Wasm128Token;
/// Scalar fallback.
pub type scalar = archmage::ScalarToken;

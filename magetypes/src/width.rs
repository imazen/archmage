//! Width dispatch trait for token-based SIMD type construction.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.
//!
//! The `WidthDispatch` trait provides access to ALL SIMD sizes from any token.
//! Native types are used where the hardware supports them; polyfills are used
//! for wider types on narrower hardware.

#![allow(missing_docs)]

use archmage::SimdToken;

/// Trait providing access to all SIMD sizes from a capability token.
///
/// Every token implementing this trait can construct vectors of any size.
/// The associated types determine whether native or polyfilled implementations
/// are used based on the token's hardware capabilities.
pub trait WidthDispatch: SimdToken + Copy {
    // 128-bit types
    type F32x4;
    type F64x2;
    type I8x16;
    type U8x16;
    type I16x8;
    type U16x8;
    type I32x4;
    type U32x4;
    type I64x2;
    type U64x2;

    // 256-bit types
    type F32x8;
    type F64x4;
    type I8x32;
    type U8x32;
    type I16x16;
    type U16x16;
    type I32x8;
    type U32x8;
    type I64x4;
    type U64x4;

    // 512-bit types
    type F32x16;
    type F64x8;
    type I8x64;
    type U8x64;
    type I16x32;
    type U16x32;
    type I32x16;
    type U32x16;
    type I64x8;
    type U64x8;

    fn f32x4_splat(self, v: f32) -> Self::F32x4;
    fn f32x4_zero(self) -> Self::F32x4;
    fn f32x4_load(self, data: &[f32; 4]) -> Self::F32x4;
    fn f64x2_splat(self, v: f64) -> Self::F64x2;
    fn f64x2_zero(self) -> Self::F64x2;
    fn f64x2_load(self, data: &[f64; 2]) -> Self::F64x2;
    fn i8x16_splat(self, v: i8) -> Self::I8x16;
    fn i8x16_zero(self) -> Self::I8x16;
    fn i8x16_load(self, data: &[i8; 16]) -> Self::I8x16;
    fn u8x16_splat(self, v: u8) -> Self::U8x16;
    fn u8x16_zero(self) -> Self::U8x16;
    fn u8x16_load(self, data: &[u8; 16]) -> Self::U8x16;
    fn i16x8_splat(self, v: i16) -> Self::I16x8;
    fn i16x8_zero(self) -> Self::I16x8;
    fn i16x8_load(self, data: &[i16; 8]) -> Self::I16x8;
    fn u16x8_splat(self, v: u16) -> Self::U16x8;
    fn u16x8_zero(self) -> Self::U16x8;
    fn u16x8_load(self, data: &[u16; 8]) -> Self::U16x8;
    fn i32x4_splat(self, v: i32) -> Self::I32x4;
    fn i32x4_zero(self) -> Self::I32x4;
    fn i32x4_load(self, data: &[i32; 4]) -> Self::I32x4;
    fn u32x4_splat(self, v: u32) -> Self::U32x4;
    fn u32x4_zero(self) -> Self::U32x4;
    fn u32x4_load(self, data: &[u32; 4]) -> Self::U32x4;
    fn i64x2_splat(self, v: i64) -> Self::I64x2;
    fn i64x2_zero(self) -> Self::I64x2;
    fn i64x2_load(self, data: &[i64; 2]) -> Self::I64x2;
    fn u64x2_splat(self, v: u64) -> Self::U64x2;
    fn u64x2_zero(self) -> Self::U64x2;
    fn u64x2_load(self, data: &[u64; 2]) -> Self::U64x2;
    fn f32x8_splat(self, v: f32) -> Self::F32x8;
    fn f32x8_zero(self) -> Self::F32x8;
    fn f32x8_load(self, data: &[f32; 8]) -> Self::F32x8;
    fn f64x4_splat(self, v: f64) -> Self::F64x4;
    fn f64x4_zero(self) -> Self::F64x4;
    fn f64x4_load(self, data: &[f64; 4]) -> Self::F64x4;
    fn i8x32_splat(self, v: i8) -> Self::I8x32;
    fn i8x32_zero(self) -> Self::I8x32;
    fn i8x32_load(self, data: &[i8; 32]) -> Self::I8x32;
    fn u8x32_splat(self, v: u8) -> Self::U8x32;
    fn u8x32_zero(self) -> Self::U8x32;
    fn u8x32_load(self, data: &[u8; 32]) -> Self::U8x32;
    fn i16x16_splat(self, v: i16) -> Self::I16x16;
    fn i16x16_zero(self) -> Self::I16x16;
    fn i16x16_load(self, data: &[i16; 16]) -> Self::I16x16;
    fn u16x16_splat(self, v: u16) -> Self::U16x16;
    fn u16x16_zero(self) -> Self::U16x16;
    fn u16x16_load(self, data: &[u16; 16]) -> Self::U16x16;
    fn i32x8_splat(self, v: i32) -> Self::I32x8;
    fn i32x8_zero(self) -> Self::I32x8;
    fn i32x8_load(self, data: &[i32; 8]) -> Self::I32x8;
    fn u32x8_splat(self, v: u32) -> Self::U32x8;
    fn u32x8_zero(self) -> Self::U32x8;
    fn u32x8_load(self, data: &[u32; 8]) -> Self::U32x8;
    fn i64x4_splat(self, v: i64) -> Self::I64x4;
    fn i64x4_zero(self) -> Self::I64x4;
    fn i64x4_load(self, data: &[i64; 4]) -> Self::I64x4;
    fn u64x4_splat(self, v: u64) -> Self::U64x4;
    fn u64x4_zero(self) -> Self::U64x4;
    fn u64x4_load(self, data: &[u64; 4]) -> Self::U64x4;
    fn f32x16_splat(self, v: f32) -> Self::F32x16;
    fn f32x16_zero(self) -> Self::F32x16;
    fn f32x16_load(self, data: &[f32; 16]) -> Self::F32x16;
    fn f64x8_splat(self, v: f64) -> Self::F64x8;
    fn f64x8_zero(self) -> Self::F64x8;
    fn f64x8_load(self, data: &[f64; 8]) -> Self::F64x8;
    fn i8x64_splat(self, v: i8) -> Self::I8x64;
    fn i8x64_zero(self) -> Self::I8x64;
    fn i8x64_load(self, data: &[i8; 64]) -> Self::I8x64;
    fn u8x64_splat(self, v: u8) -> Self::U8x64;
    fn u8x64_zero(self) -> Self::U8x64;
    fn u8x64_load(self, data: &[u8; 64]) -> Self::U8x64;
    fn i16x32_splat(self, v: i16) -> Self::I16x32;
    fn i16x32_zero(self) -> Self::I16x32;
    fn i16x32_load(self, data: &[i16; 32]) -> Self::I16x32;
    fn u16x32_splat(self, v: u16) -> Self::U16x32;
    fn u16x32_zero(self) -> Self::U16x32;
    fn u16x32_load(self, data: &[u16; 32]) -> Self::U16x32;
    fn i32x16_splat(self, v: i32) -> Self::I32x16;
    fn i32x16_zero(self) -> Self::I32x16;
    fn i32x16_load(self, data: &[i32; 16]) -> Self::I32x16;
    fn u32x16_splat(self, v: u32) -> Self::U32x16;
    fn u32x16_zero(self) -> Self::U32x16;
    fn u32x16_load(self, data: &[u32; 16]) -> Self::U32x16;
    fn i64x8_splat(self, v: i64) -> Self::I64x8;
    fn i64x8_zero(self) -> Self::I64x8;
    fn i64x8_load(self, data: &[i64; 8]) -> Self::I64x8;
    fn u64x8_splat(self, v: u64) -> Self::U64x8;
    fn u64x8_zero(self) -> Self::U64x8;
    fn u64x8_load(self, data: &[u64; 8]) -> Self::U64x8;
}

use crate::simd::generic;

impl<T> WidthDispatch for T
where
    T: SimdToken
        + Copy
        + crate::simd::backends::F32x4Backend
        + crate::simd::backends::F64x2Backend
        + crate::simd::backends::I8x16Backend
        + crate::simd::backends::U8x16Backend
        + crate::simd::backends::I16x8Backend
        + crate::simd::backends::U16x8Backend
        + crate::simd::backends::I32x4Backend
        + crate::simd::backends::U32x4Backend
        + crate::simd::backends::I64x2Backend
        + crate::simd::backends::U64x2Backend
        + crate::simd::backends::F32x8Backend
        + crate::simd::backends::F64x4Backend
        + crate::simd::backends::I8x32Backend
        + crate::simd::backends::U8x32Backend
        + crate::simd::backends::I16x16Backend
        + crate::simd::backends::U16x16Backend
        + crate::simd::backends::I32x8Backend
        + crate::simd::backends::U32x8Backend
        + crate::simd::backends::I64x4Backend
        + crate::simd::backends::U64x4Backend,
{
    // 128-bit types
    type F32x4 = generic::f32x4<T>;
    type F64x2 = generic::f64x2<T>;
    type I8x16 = generic::i8x16<T>;
    type U8x16 = generic::u8x16<T>;
    type I16x8 = generic::i16x8<T>;
    type U16x8 = generic::u16x8<T>;
    type I32x4 = generic::i32x4<T>;
    type U32x4 = generic::u32x4<T>;
    type I64x2 = generic::i64x2<T>;
    type U64x2 = generic::u64x2<T>;

    // 256-bit types
    type F32x8 = generic::f32x8<T>;
    type F64x4 = generic::f64x4<T>;
    type I8x32 = generic::i8x32<T>;
    type U8x32 = generic::u8x32<T>;
    type I16x16 = generic::i16x16<T>;
    type U16x16 = generic::u16x16<T>;
    type I32x8 = generic::i32x8<T>;
    type U32x8 = generic::u32x8<T>;
    type I64x4 = generic::i64x4<T>;
    type U64x4 = generic::u64x4<T>;

    // 512-bit types
    type F32x16 = [generic::f32x4<T>; 4];
    type F64x8 = [generic::f64x2<T>; 4];
    type I8x64 = [generic::i8x16<T>; 4];
    type U8x64 = [generic::u8x16<T>; 4];
    type I16x32 = [generic::i16x8<T>; 4];
    type U16x32 = [generic::u16x8<T>; 4];
    type I32x16 = [generic::i32x4<T>; 4];
    type U32x16 = [generic::u32x4<T>; 4];
    type I64x8 = [generic::i64x2<T>; 4];
    type U64x8 = [generic::u64x2<T>; 4];

    #[inline(always)]
    fn f32x4_splat(self, v: f32) -> Self::F32x4 {
        generic::f32x4::splat_t(self, v)
    }

    #[inline(always)]
    fn f32x4_zero(self) -> Self::F32x4 {
        generic::f32x4::zero_t(self)
    }

    #[inline(always)]
    fn f32x4_load(self, data: &[f32; 4]) -> Self::F32x4 {
        generic::f32x4::load_t(self, data)
    }
    #[inline(always)]
    fn f64x2_splat(self, v: f64) -> Self::F64x2 {
        generic::f64x2::splat_t(self, v)
    }

    #[inline(always)]
    fn f64x2_zero(self) -> Self::F64x2 {
        generic::f64x2::zero_t(self)
    }

    #[inline(always)]
    fn f64x2_load(self, data: &[f64; 2]) -> Self::F64x2 {
        generic::f64x2::load_t(self, data)
    }
    #[inline(always)]
    fn i8x16_splat(self, v: i8) -> Self::I8x16 {
        generic::i8x16::splat_t(self, v)
    }

    #[inline(always)]
    fn i8x16_zero(self) -> Self::I8x16 {
        generic::i8x16::zero_t(self)
    }

    #[inline(always)]
    fn i8x16_load(self, data: &[i8; 16]) -> Self::I8x16 {
        generic::i8x16::load_t(self, data)
    }
    #[inline(always)]
    fn u8x16_splat(self, v: u8) -> Self::U8x16 {
        generic::u8x16::splat_t(self, v)
    }

    #[inline(always)]
    fn u8x16_zero(self) -> Self::U8x16 {
        generic::u8x16::zero_t(self)
    }

    #[inline(always)]
    fn u8x16_load(self, data: &[u8; 16]) -> Self::U8x16 {
        generic::u8x16::load_t(self, data)
    }
    #[inline(always)]
    fn i16x8_splat(self, v: i16) -> Self::I16x8 {
        generic::i16x8::splat_t(self, v)
    }

    #[inline(always)]
    fn i16x8_zero(self) -> Self::I16x8 {
        generic::i16x8::zero_t(self)
    }

    #[inline(always)]
    fn i16x8_load(self, data: &[i16; 8]) -> Self::I16x8 {
        generic::i16x8::load_t(self, data)
    }
    #[inline(always)]
    fn u16x8_splat(self, v: u16) -> Self::U16x8 {
        generic::u16x8::splat_t(self, v)
    }

    #[inline(always)]
    fn u16x8_zero(self) -> Self::U16x8 {
        generic::u16x8::zero_t(self)
    }

    #[inline(always)]
    fn u16x8_load(self, data: &[u16; 8]) -> Self::U16x8 {
        generic::u16x8::load_t(self, data)
    }
    #[inline(always)]
    fn i32x4_splat(self, v: i32) -> Self::I32x4 {
        generic::i32x4::splat_t(self, v)
    }

    #[inline(always)]
    fn i32x4_zero(self) -> Self::I32x4 {
        generic::i32x4::zero_t(self)
    }

    #[inline(always)]
    fn i32x4_load(self, data: &[i32; 4]) -> Self::I32x4 {
        generic::i32x4::load_t(self, data)
    }
    #[inline(always)]
    fn u32x4_splat(self, v: u32) -> Self::U32x4 {
        generic::u32x4::splat_t(self, v)
    }

    #[inline(always)]
    fn u32x4_zero(self) -> Self::U32x4 {
        generic::u32x4::zero_t(self)
    }

    #[inline(always)]
    fn u32x4_load(self, data: &[u32; 4]) -> Self::U32x4 {
        generic::u32x4::load_t(self, data)
    }
    #[inline(always)]
    fn i64x2_splat(self, v: i64) -> Self::I64x2 {
        generic::i64x2::splat_t(self, v)
    }

    #[inline(always)]
    fn i64x2_zero(self) -> Self::I64x2 {
        generic::i64x2::zero_t(self)
    }

    #[inline(always)]
    fn i64x2_load(self, data: &[i64; 2]) -> Self::I64x2 {
        generic::i64x2::load_t(self, data)
    }
    #[inline(always)]
    fn u64x2_splat(self, v: u64) -> Self::U64x2 {
        generic::u64x2::splat_t(self, v)
    }

    #[inline(always)]
    fn u64x2_zero(self) -> Self::U64x2 {
        generic::u64x2::zero_t(self)
    }

    #[inline(always)]
    fn u64x2_load(self, data: &[u64; 2]) -> Self::U64x2 {
        generic::u64x2::load_t(self, data)
    }
    #[inline(always)]
    fn f32x8_splat(self, v: f32) -> Self::F32x8 {
        generic::f32x8::splat_t(self, v)
    }

    #[inline(always)]
    fn f32x8_zero(self) -> Self::F32x8 {
        generic::f32x8::zero_t(self)
    }

    #[inline(always)]
    fn f32x8_load(self, data: &[f32; 8]) -> Self::F32x8 {
        generic::f32x8::load_t(self, data)
    }
    #[inline(always)]
    fn f64x4_splat(self, v: f64) -> Self::F64x4 {
        generic::f64x4::splat_t(self, v)
    }

    #[inline(always)]
    fn f64x4_zero(self) -> Self::F64x4 {
        generic::f64x4::zero_t(self)
    }

    #[inline(always)]
    fn f64x4_load(self, data: &[f64; 4]) -> Self::F64x4 {
        generic::f64x4::load_t(self, data)
    }
    #[inline(always)]
    fn i8x32_splat(self, v: i8) -> Self::I8x32 {
        generic::i8x32::splat_t(self, v)
    }

    #[inline(always)]
    fn i8x32_zero(self) -> Self::I8x32 {
        generic::i8x32::zero_t(self)
    }

    #[inline(always)]
    fn i8x32_load(self, data: &[i8; 32]) -> Self::I8x32 {
        generic::i8x32::load_t(self, data)
    }
    #[inline(always)]
    fn u8x32_splat(self, v: u8) -> Self::U8x32 {
        generic::u8x32::splat_t(self, v)
    }

    #[inline(always)]
    fn u8x32_zero(self) -> Self::U8x32 {
        generic::u8x32::zero_t(self)
    }

    #[inline(always)]
    fn u8x32_load(self, data: &[u8; 32]) -> Self::U8x32 {
        generic::u8x32::load_t(self, data)
    }
    #[inline(always)]
    fn i16x16_splat(self, v: i16) -> Self::I16x16 {
        generic::i16x16::splat_t(self, v)
    }

    #[inline(always)]
    fn i16x16_zero(self) -> Self::I16x16 {
        generic::i16x16::zero_t(self)
    }

    #[inline(always)]
    fn i16x16_load(self, data: &[i16; 16]) -> Self::I16x16 {
        generic::i16x16::load_t(self, data)
    }
    #[inline(always)]
    fn u16x16_splat(self, v: u16) -> Self::U16x16 {
        generic::u16x16::splat_t(self, v)
    }

    #[inline(always)]
    fn u16x16_zero(self) -> Self::U16x16 {
        generic::u16x16::zero_t(self)
    }

    #[inline(always)]
    fn u16x16_load(self, data: &[u16; 16]) -> Self::U16x16 {
        generic::u16x16::load_t(self, data)
    }
    #[inline(always)]
    fn i32x8_splat(self, v: i32) -> Self::I32x8 {
        generic::i32x8::splat_t(self, v)
    }

    #[inline(always)]
    fn i32x8_zero(self) -> Self::I32x8 {
        generic::i32x8::zero_t(self)
    }

    #[inline(always)]
    fn i32x8_load(self, data: &[i32; 8]) -> Self::I32x8 {
        generic::i32x8::load_t(self, data)
    }
    #[inline(always)]
    fn u32x8_splat(self, v: u32) -> Self::U32x8 {
        generic::u32x8::splat_t(self, v)
    }

    #[inline(always)]
    fn u32x8_zero(self) -> Self::U32x8 {
        generic::u32x8::zero_t(self)
    }

    #[inline(always)]
    fn u32x8_load(self, data: &[u32; 8]) -> Self::U32x8 {
        generic::u32x8::load_t(self, data)
    }
    #[inline(always)]
    fn i64x4_splat(self, v: i64) -> Self::I64x4 {
        generic::i64x4::splat_t(self, v)
    }

    #[inline(always)]
    fn i64x4_zero(self) -> Self::I64x4 {
        generic::i64x4::zero_t(self)
    }

    #[inline(always)]
    fn i64x4_load(self, data: &[i64; 4]) -> Self::I64x4 {
        generic::i64x4::load_t(self, data)
    }
    #[inline(always)]
    fn u64x4_splat(self, v: u64) -> Self::U64x4 {
        generic::u64x4::splat_t(self, v)
    }

    #[inline(always)]
    fn u64x4_zero(self) -> Self::U64x4 {
        generic::u64x4::zero_t(self)
    }

    #[inline(always)]
    fn u64x4_load(self, data: &[u64; 4]) -> Self::U64x4 {
        generic::u64x4::load_t(self, data)
    }
    #[inline(always)]
    fn f32x16_splat(self, v: f32) -> Self::F32x16 {
        {
            let part = generic::f32x4::splat_t(self, v);
            [part, part, part, part]
        }
    }

    #[inline(always)]
    fn f32x16_zero(self) -> Self::F32x16 {
        {
            let part = generic::f32x4::zero_t(self);
            [part, part, part, part]
        }
    }

    #[inline(always)]
    fn f32x16_load(self, data: &[f32; 16]) -> Self::F32x16 {
        {
            [
                generic::f32x4::load_t(self, data[0..4].try_into().unwrap()),
                generic::f32x4::load_t(self, data[4..8].try_into().unwrap()),
                generic::f32x4::load_t(self, data[8..12].try_into().unwrap()),
                generic::f32x4::load_t(self, data[12..16].try_into().unwrap()),
            ]
        }
    }
    #[inline(always)]
    fn f64x8_splat(self, v: f64) -> Self::F64x8 {
        {
            let part = generic::f64x2::splat_t(self, v);
            [part, part, part, part]
        }
    }

    #[inline(always)]
    fn f64x8_zero(self) -> Self::F64x8 {
        {
            let part = generic::f64x2::zero_t(self);
            [part, part, part, part]
        }
    }

    #[inline(always)]
    fn f64x8_load(self, data: &[f64; 8]) -> Self::F64x8 {
        {
            [
                generic::f64x2::load_t(self, data[0..2].try_into().unwrap()),
                generic::f64x2::load_t(self, data[2..4].try_into().unwrap()),
                generic::f64x2::load_t(self, data[4..6].try_into().unwrap()),
                generic::f64x2::load_t(self, data[6..8].try_into().unwrap()),
            ]
        }
    }
    #[inline(always)]
    fn i8x64_splat(self, v: i8) -> Self::I8x64 {
        {
            let part = generic::i8x16::splat_t(self, v);
            [part, part, part, part]
        }
    }

    #[inline(always)]
    fn i8x64_zero(self) -> Self::I8x64 {
        {
            let part = generic::i8x16::zero_t(self);
            [part, part, part, part]
        }
    }

    #[inline(always)]
    fn i8x64_load(self, data: &[i8; 64]) -> Self::I8x64 {
        {
            [
                generic::i8x16::load_t(self, data[0..16].try_into().unwrap()),
                generic::i8x16::load_t(self, data[16..32].try_into().unwrap()),
                generic::i8x16::load_t(self, data[32..48].try_into().unwrap()),
                generic::i8x16::load_t(self, data[48..64].try_into().unwrap()),
            ]
        }
    }
    #[inline(always)]
    fn u8x64_splat(self, v: u8) -> Self::U8x64 {
        {
            let part = generic::u8x16::splat_t(self, v);
            [part, part, part, part]
        }
    }

    #[inline(always)]
    fn u8x64_zero(self) -> Self::U8x64 {
        {
            let part = generic::u8x16::zero_t(self);
            [part, part, part, part]
        }
    }

    #[inline(always)]
    fn u8x64_load(self, data: &[u8; 64]) -> Self::U8x64 {
        {
            [
                generic::u8x16::load_t(self, data[0..16].try_into().unwrap()),
                generic::u8x16::load_t(self, data[16..32].try_into().unwrap()),
                generic::u8x16::load_t(self, data[32..48].try_into().unwrap()),
                generic::u8x16::load_t(self, data[48..64].try_into().unwrap()),
            ]
        }
    }
    #[inline(always)]
    fn i16x32_splat(self, v: i16) -> Self::I16x32 {
        {
            let part = generic::i16x8::splat_t(self, v);
            [part, part, part, part]
        }
    }

    #[inline(always)]
    fn i16x32_zero(self) -> Self::I16x32 {
        {
            let part = generic::i16x8::zero_t(self);
            [part, part, part, part]
        }
    }

    #[inline(always)]
    fn i16x32_load(self, data: &[i16; 32]) -> Self::I16x32 {
        {
            [
                generic::i16x8::load_t(self, data[0..8].try_into().unwrap()),
                generic::i16x8::load_t(self, data[8..16].try_into().unwrap()),
                generic::i16x8::load_t(self, data[16..24].try_into().unwrap()),
                generic::i16x8::load_t(self, data[24..32].try_into().unwrap()),
            ]
        }
    }
    #[inline(always)]
    fn u16x32_splat(self, v: u16) -> Self::U16x32 {
        {
            let part = generic::u16x8::splat_t(self, v);
            [part, part, part, part]
        }
    }

    #[inline(always)]
    fn u16x32_zero(self) -> Self::U16x32 {
        {
            let part = generic::u16x8::zero_t(self);
            [part, part, part, part]
        }
    }

    #[inline(always)]
    fn u16x32_load(self, data: &[u16; 32]) -> Self::U16x32 {
        {
            [
                generic::u16x8::load_t(self, data[0..8].try_into().unwrap()),
                generic::u16x8::load_t(self, data[8..16].try_into().unwrap()),
                generic::u16x8::load_t(self, data[16..24].try_into().unwrap()),
                generic::u16x8::load_t(self, data[24..32].try_into().unwrap()),
            ]
        }
    }
    #[inline(always)]
    fn i32x16_splat(self, v: i32) -> Self::I32x16 {
        {
            let part = generic::i32x4::splat_t(self, v);
            [part, part, part, part]
        }
    }

    #[inline(always)]
    fn i32x16_zero(self) -> Self::I32x16 {
        {
            let part = generic::i32x4::zero_t(self);
            [part, part, part, part]
        }
    }

    #[inline(always)]
    fn i32x16_load(self, data: &[i32; 16]) -> Self::I32x16 {
        {
            [
                generic::i32x4::load_t(self, data[0..4].try_into().unwrap()),
                generic::i32x4::load_t(self, data[4..8].try_into().unwrap()),
                generic::i32x4::load_t(self, data[8..12].try_into().unwrap()),
                generic::i32x4::load_t(self, data[12..16].try_into().unwrap()),
            ]
        }
    }
    #[inline(always)]
    fn u32x16_splat(self, v: u32) -> Self::U32x16 {
        {
            let part = generic::u32x4::splat_t(self, v);
            [part, part, part, part]
        }
    }

    #[inline(always)]
    fn u32x16_zero(self) -> Self::U32x16 {
        {
            let part = generic::u32x4::zero_t(self);
            [part, part, part, part]
        }
    }

    #[inline(always)]
    fn u32x16_load(self, data: &[u32; 16]) -> Self::U32x16 {
        {
            [
                generic::u32x4::load_t(self, data[0..4].try_into().unwrap()),
                generic::u32x4::load_t(self, data[4..8].try_into().unwrap()),
                generic::u32x4::load_t(self, data[8..12].try_into().unwrap()),
                generic::u32x4::load_t(self, data[12..16].try_into().unwrap()),
            ]
        }
    }
    #[inline(always)]
    fn i64x8_splat(self, v: i64) -> Self::I64x8 {
        {
            let part = generic::i64x2::splat_t(self, v);
            [part, part, part, part]
        }
    }

    #[inline(always)]
    fn i64x8_zero(self) -> Self::I64x8 {
        {
            let part = generic::i64x2::zero_t(self);
            [part, part, part, part]
        }
    }

    #[inline(always)]
    fn i64x8_load(self, data: &[i64; 8]) -> Self::I64x8 {
        {
            [
                generic::i64x2::load_t(self, data[0..2].try_into().unwrap()),
                generic::i64x2::load_t(self, data[2..4].try_into().unwrap()),
                generic::i64x2::load_t(self, data[4..6].try_into().unwrap()),
                generic::i64x2::load_t(self, data[6..8].try_into().unwrap()),
            ]
        }
    }
    #[inline(always)]
    fn u64x8_splat(self, v: u64) -> Self::U64x8 {
        {
            let part = generic::u64x2::splat_t(self, v);
            [part, part, part, part]
        }
    }

    #[inline(always)]
    fn u64x8_zero(self) -> Self::U64x8 {
        {
            let part = generic::u64x2::zero_t(self);
            [part, part, part, part]
        }
    }

    #[inline(always)]
    fn u64x8_load(self, data: &[u64; 8]) -> Self::U64x8 {
        {
            [
                generic::u64x2::load_t(self, data[0..2].try_into().unwrap()),
                generic::u64x2::load_t(self, data[2..4].try_into().unwrap()),
                generic::u64x2::load_t(self, data[4..6].try_into().unwrap()),
                generic::u64x2::load_t(self, data[6..8].try_into().unwrap()),
            ]
        }
    }
}

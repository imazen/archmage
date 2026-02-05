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

#[cfg(target_arch = "x86_64")]
mod x86_impl {
    use super::WidthDispatch;
    use archmage::{SimdToken, X64V3Token};

    use crate::simd::{
        f32x4, f32x8, f64x2, f64x4, i8x16, i8x32, i16x8, i16x16, i32x4, i32x8, i64x2, i64x4, u8x16,
        u8x32, u16x8, u16x16, u32x4, u32x8, u64x2, u64x4,
    };

    impl WidthDispatch for X64V3Token {
        // 128-bit types
        type F32x4 = f32x4;
        type F64x2 = f64x2;
        type I8x16 = i8x16;
        type U8x16 = u8x16;
        type I16x8 = i16x8;
        type U16x8 = u16x8;
        type I32x4 = i32x4;
        type U32x4 = u32x4;
        type I64x2 = i64x2;
        type U64x2 = u64x2;

        // 256-bit types
        type F32x8 = f32x8;
        type F64x4 = f64x4;
        type I8x32 = i8x32;
        type U8x32 = u8x32;
        type I16x16 = i16x16;
        type U16x16 = u16x16;
        type I32x8 = i32x8;
        type U32x8 = u32x8;
        type I64x4 = i64x4;
        type U64x4 = u64x4;

        // 512-bit types
        type F32x16 = crate::simd::polyfill::v3_512::f32x16;
        type F64x8 = crate::simd::polyfill::v3_512::f64x8;
        type I8x64 = crate::simd::polyfill::v3_512::i8x64;
        type U8x64 = crate::simd::polyfill::v3_512::u8x64;
        type I16x32 = crate::simd::polyfill::v3_512::i16x32;
        type U16x32 = crate::simd::polyfill::v3_512::u16x32;
        type I32x16 = crate::simd::polyfill::v3_512::i32x16;
        type U32x16 = crate::simd::polyfill::v3_512::u32x16;
        type I64x8 = crate::simd::polyfill::v3_512::i64x8;
        type U64x8 = crate::simd::polyfill::v3_512::u64x8;

        #[inline(always)]
        fn f32x4_splat(self, v: f32) -> Self::F32x4 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                f32x4::splat(token, v)
            }
        }

        #[inline(always)]
        fn f32x4_zero(self) -> Self::F32x4 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                f32x4::zero(token)
            }
        }

        #[inline(always)]
        fn f32x4_load(self, data: &[f32; 4]) -> Self::F32x4 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                f32x4::load(token, data)
            }
        }
        #[inline(always)]
        fn f64x2_splat(self, v: f64) -> Self::F64x2 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                f64x2::splat(token, v)
            }
        }

        #[inline(always)]
        fn f64x2_zero(self) -> Self::F64x2 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                f64x2::zero(token)
            }
        }

        #[inline(always)]
        fn f64x2_load(self, data: &[f64; 2]) -> Self::F64x2 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                f64x2::load(token, data)
            }
        }
        #[inline(always)]
        fn i8x16_splat(self, v: i8) -> Self::I8x16 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                i8x16::splat(token, v)
            }
        }

        #[inline(always)]
        fn i8x16_zero(self) -> Self::I8x16 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                i8x16::zero(token)
            }
        }

        #[inline(always)]
        fn i8x16_load(self, data: &[i8; 16]) -> Self::I8x16 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                i8x16::load(token, data)
            }
        }
        #[inline(always)]
        fn u8x16_splat(self, v: u8) -> Self::U8x16 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                u8x16::splat(token, v)
            }
        }

        #[inline(always)]
        fn u8x16_zero(self) -> Self::U8x16 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                u8x16::zero(token)
            }
        }

        #[inline(always)]
        fn u8x16_load(self, data: &[u8; 16]) -> Self::U8x16 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                u8x16::load(token, data)
            }
        }
        #[inline(always)]
        fn i16x8_splat(self, v: i16) -> Self::I16x8 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                i16x8::splat(token, v)
            }
        }

        #[inline(always)]
        fn i16x8_zero(self) -> Self::I16x8 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                i16x8::zero(token)
            }
        }

        #[inline(always)]
        fn i16x8_load(self, data: &[i16; 8]) -> Self::I16x8 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                i16x8::load(token, data)
            }
        }
        #[inline(always)]
        fn u16x8_splat(self, v: u16) -> Self::U16x8 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                u16x8::splat(token, v)
            }
        }

        #[inline(always)]
        fn u16x8_zero(self) -> Self::U16x8 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                u16x8::zero(token)
            }
        }

        #[inline(always)]
        fn u16x8_load(self, data: &[u16; 8]) -> Self::U16x8 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                u16x8::load(token, data)
            }
        }
        #[inline(always)]
        fn i32x4_splat(self, v: i32) -> Self::I32x4 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                i32x4::splat(token, v)
            }
        }

        #[inline(always)]
        fn i32x4_zero(self) -> Self::I32x4 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                i32x4::zero(token)
            }
        }

        #[inline(always)]
        fn i32x4_load(self, data: &[i32; 4]) -> Self::I32x4 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                i32x4::load(token, data)
            }
        }
        #[inline(always)]
        fn u32x4_splat(self, v: u32) -> Self::U32x4 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                u32x4::splat(token, v)
            }
        }

        #[inline(always)]
        fn u32x4_zero(self) -> Self::U32x4 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                u32x4::zero(token)
            }
        }

        #[inline(always)]
        fn u32x4_load(self, data: &[u32; 4]) -> Self::U32x4 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                u32x4::load(token, data)
            }
        }
        #[inline(always)]
        fn i64x2_splat(self, v: i64) -> Self::I64x2 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                i64x2::splat(token, v)
            }
        }

        #[inline(always)]
        fn i64x2_zero(self) -> Self::I64x2 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                i64x2::zero(token)
            }
        }

        #[inline(always)]
        fn i64x2_load(self, data: &[i64; 2]) -> Self::I64x2 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                i64x2::load(token, data)
            }
        }
        #[inline(always)]
        fn u64x2_splat(self, v: u64) -> Self::U64x2 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                u64x2::splat(token, v)
            }
        }

        #[inline(always)]
        fn u64x2_zero(self) -> Self::U64x2 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                u64x2::zero(token)
            }
        }

        #[inline(always)]
        fn u64x2_load(self, data: &[u64; 2]) -> Self::U64x2 {
            {
                let token = unsafe { X64V3Token::forge_token_dangerously() };
                u64x2::load(token, data)
            }
        }
        #[inline(always)]
        fn f32x8_splat(self, v: f32) -> Self::F32x8 {
            f32x8::splat(self, v)
        }

        #[inline(always)]
        fn f32x8_zero(self) -> Self::F32x8 {
            f32x8::zero(self)
        }

        #[inline(always)]
        fn f32x8_load(self, data: &[f32; 8]) -> Self::F32x8 {
            f32x8::load(self, data)
        }
        #[inline(always)]
        fn f64x4_splat(self, v: f64) -> Self::F64x4 {
            f64x4::splat(self, v)
        }

        #[inline(always)]
        fn f64x4_zero(self) -> Self::F64x4 {
            f64x4::zero(self)
        }

        #[inline(always)]
        fn f64x4_load(self, data: &[f64; 4]) -> Self::F64x4 {
            f64x4::load(self, data)
        }
        #[inline(always)]
        fn i8x32_splat(self, v: i8) -> Self::I8x32 {
            i8x32::splat(self, v)
        }

        #[inline(always)]
        fn i8x32_zero(self) -> Self::I8x32 {
            i8x32::zero(self)
        }

        #[inline(always)]
        fn i8x32_load(self, data: &[i8; 32]) -> Self::I8x32 {
            i8x32::load(self, data)
        }
        #[inline(always)]
        fn u8x32_splat(self, v: u8) -> Self::U8x32 {
            u8x32::splat(self, v)
        }

        #[inline(always)]
        fn u8x32_zero(self) -> Self::U8x32 {
            u8x32::zero(self)
        }

        #[inline(always)]
        fn u8x32_load(self, data: &[u8; 32]) -> Self::U8x32 {
            u8x32::load(self, data)
        }
        #[inline(always)]
        fn i16x16_splat(self, v: i16) -> Self::I16x16 {
            i16x16::splat(self, v)
        }

        #[inline(always)]
        fn i16x16_zero(self) -> Self::I16x16 {
            i16x16::zero(self)
        }

        #[inline(always)]
        fn i16x16_load(self, data: &[i16; 16]) -> Self::I16x16 {
            i16x16::load(self, data)
        }
        #[inline(always)]
        fn u16x16_splat(self, v: u16) -> Self::U16x16 {
            u16x16::splat(self, v)
        }

        #[inline(always)]
        fn u16x16_zero(self) -> Self::U16x16 {
            u16x16::zero(self)
        }

        #[inline(always)]
        fn u16x16_load(self, data: &[u16; 16]) -> Self::U16x16 {
            u16x16::load(self, data)
        }
        #[inline(always)]
        fn i32x8_splat(self, v: i32) -> Self::I32x8 {
            i32x8::splat(self, v)
        }

        #[inline(always)]
        fn i32x8_zero(self) -> Self::I32x8 {
            i32x8::zero(self)
        }

        #[inline(always)]
        fn i32x8_load(self, data: &[i32; 8]) -> Self::I32x8 {
            i32x8::load(self, data)
        }
        #[inline(always)]
        fn u32x8_splat(self, v: u32) -> Self::U32x8 {
            u32x8::splat(self, v)
        }

        #[inline(always)]
        fn u32x8_zero(self) -> Self::U32x8 {
            u32x8::zero(self)
        }

        #[inline(always)]
        fn u32x8_load(self, data: &[u32; 8]) -> Self::U32x8 {
            u32x8::load(self, data)
        }
        #[inline(always)]
        fn i64x4_splat(self, v: i64) -> Self::I64x4 {
            i64x4::splat(self, v)
        }

        #[inline(always)]
        fn i64x4_zero(self) -> Self::I64x4 {
            i64x4::zero(self)
        }

        #[inline(always)]
        fn i64x4_load(self, data: &[i64; 4]) -> Self::I64x4 {
            i64x4::load(self, data)
        }
        #[inline(always)]
        fn u64x4_splat(self, v: u64) -> Self::U64x4 {
            u64x4::splat(self, v)
        }

        #[inline(always)]
        fn u64x4_zero(self) -> Self::U64x4 {
            u64x4::zero(self)
        }

        #[inline(always)]
        fn u64x4_load(self, data: &[u64; 4]) -> Self::U64x4 {
            u64x4::load(self, data)
        }
        #[inline(always)]
        fn f32x16_splat(self, v: f32) -> Self::F32x16 {
            crate::simd::polyfill::v3_512::f32x16::splat(self, v)
        }

        #[inline(always)]
        fn f32x16_zero(self) -> Self::F32x16 {
            crate::simd::polyfill::v3_512::f32x16::zero(self)
        }

        #[inline(always)]
        fn f32x16_load(self, data: &[f32; 16]) -> Self::F32x16 {
            crate::simd::polyfill::v3_512::f32x16::load(self, data)
        }
        #[inline(always)]
        fn f64x8_splat(self, v: f64) -> Self::F64x8 {
            crate::simd::polyfill::v3_512::f64x8::splat(self, v)
        }

        #[inline(always)]
        fn f64x8_zero(self) -> Self::F64x8 {
            crate::simd::polyfill::v3_512::f64x8::zero(self)
        }

        #[inline(always)]
        fn f64x8_load(self, data: &[f64; 8]) -> Self::F64x8 {
            crate::simd::polyfill::v3_512::f64x8::load(self, data)
        }
        #[inline(always)]
        fn i8x64_splat(self, v: i8) -> Self::I8x64 {
            crate::simd::polyfill::v3_512::i8x64::splat(self, v)
        }

        #[inline(always)]
        fn i8x64_zero(self) -> Self::I8x64 {
            crate::simd::polyfill::v3_512::i8x64::zero(self)
        }

        #[inline(always)]
        fn i8x64_load(self, data: &[i8; 64]) -> Self::I8x64 {
            crate::simd::polyfill::v3_512::i8x64::load(self, data)
        }
        #[inline(always)]
        fn u8x64_splat(self, v: u8) -> Self::U8x64 {
            crate::simd::polyfill::v3_512::u8x64::splat(self, v)
        }

        #[inline(always)]
        fn u8x64_zero(self) -> Self::U8x64 {
            crate::simd::polyfill::v3_512::u8x64::zero(self)
        }

        #[inline(always)]
        fn u8x64_load(self, data: &[u8; 64]) -> Self::U8x64 {
            crate::simd::polyfill::v3_512::u8x64::load(self, data)
        }
        #[inline(always)]
        fn i16x32_splat(self, v: i16) -> Self::I16x32 {
            crate::simd::polyfill::v3_512::i16x32::splat(self, v)
        }

        #[inline(always)]
        fn i16x32_zero(self) -> Self::I16x32 {
            crate::simd::polyfill::v3_512::i16x32::zero(self)
        }

        #[inline(always)]
        fn i16x32_load(self, data: &[i16; 32]) -> Self::I16x32 {
            crate::simd::polyfill::v3_512::i16x32::load(self, data)
        }
        #[inline(always)]
        fn u16x32_splat(self, v: u16) -> Self::U16x32 {
            crate::simd::polyfill::v3_512::u16x32::splat(self, v)
        }

        #[inline(always)]
        fn u16x32_zero(self) -> Self::U16x32 {
            crate::simd::polyfill::v3_512::u16x32::zero(self)
        }

        #[inline(always)]
        fn u16x32_load(self, data: &[u16; 32]) -> Self::U16x32 {
            crate::simd::polyfill::v3_512::u16x32::load(self, data)
        }
        #[inline(always)]
        fn i32x16_splat(self, v: i32) -> Self::I32x16 {
            crate::simd::polyfill::v3_512::i32x16::splat(self, v)
        }

        #[inline(always)]
        fn i32x16_zero(self) -> Self::I32x16 {
            crate::simd::polyfill::v3_512::i32x16::zero(self)
        }

        #[inline(always)]
        fn i32x16_load(self, data: &[i32; 16]) -> Self::I32x16 {
            crate::simd::polyfill::v3_512::i32x16::load(self, data)
        }
        #[inline(always)]
        fn u32x16_splat(self, v: u32) -> Self::U32x16 {
            crate::simd::polyfill::v3_512::u32x16::splat(self, v)
        }

        #[inline(always)]
        fn u32x16_zero(self) -> Self::U32x16 {
            crate::simd::polyfill::v3_512::u32x16::zero(self)
        }

        #[inline(always)]
        fn u32x16_load(self, data: &[u32; 16]) -> Self::U32x16 {
            crate::simd::polyfill::v3_512::u32x16::load(self, data)
        }
        #[inline(always)]
        fn i64x8_splat(self, v: i64) -> Self::I64x8 {
            crate::simd::polyfill::v3_512::i64x8::splat(self, v)
        }

        #[inline(always)]
        fn i64x8_zero(self) -> Self::I64x8 {
            crate::simd::polyfill::v3_512::i64x8::zero(self)
        }

        #[inline(always)]
        fn i64x8_load(self, data: &[i64; 8]) -> Self::I64x8 {
            crate::simd::polyfill::v3_512::i64x8::load(self, data)
        }
        #[inline(always)]
        fn u64x8_splat(self, v: u64) -> Self::U64x8 {
            crate::simd::polyfill::v3_512::u64x8::splat(self, v)
        }

        #[inline(always)]
        fn u64x8_zero(self) -> Self::U64x8 {
            crate::simd::polyfill::v3_512::u64x8::zero(self)
        }

        #[inline(always)]
        fn u64x8_load(self, data: &[u64; 8]) -> Self::U64x8 {
            crate::simd::polyfill::v3_512::u64x8::load(self, data)
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod arm_impl {
    use super::WidthDispatch;
    use archmage::{NeonToken, SimdToken};

    use crate::simd::{f32x4, f64x2, i8x16, i16x8, i32x4, i64x2, u8x16, u16x8, u32x4, u64x2};

    impl WidthDispatch for NeonToken {
        // 128-bit types
        type F32x4 = f32x4;
        type F64x2 = f64x2;
        type I8x16 = i8x16;
        type U8x16 = u8x16;
        type I16x8 = i16x8;
        type U16x8 = u16x8;
        type I32x4 = i32x4;
        type U32x4 = u32x4;
        type I64x2 = i64x2;
        type U64x2 = u64x2;

        // 256-bit types
        type F32x8 = crate::simd::polyfill::neon::f32x8;
        type F64x4 = crate::simd::polyfill::neon::f64x4;
        type I8x32 = crate::simd::polyfill::neon::i8x32;
        type U8x32 = crate::simd::polyfill::neon::u8x32;
        type I16x16 = crate::simd::polyfill::neon::i16x16;
        type U16x16 = crate::simd::polyfill::neon::u16x16;
        type I32x8 = crate::simd::polyfill::neon::i32x8;
        type U32x8 = crate::simd::polyfill::neon::u32x8;
        type I64x4 = crate::simd::polyfill::neon::i64x4;
        type U64x4 = crate::simd::polyfill::neon::u64x4;

        // 512-bit types
        type F32x16 = [f32x4; 4];
        type F64x8 = [f64x2; 4];
        type I8x64 = [i8x16; 4];
        type U8x64 = [u8x16; 4];
        type I16x32 = [i16x8; 4];
        type U16x32 = [u16x8; 4];
        type I32x16 = [i32x4; 4];
        type U32x16 = [u32x4; 4];
        type I64x8 = [i64x2; 4];
        type U64x8 = [u64x2; 4];

        #[inline(always)]
        fn f32x4_splat(self, v: f32) -> Self::F32x4 {
            f32x4::splat(self, v)
        }

        #[inline(always)]
        fn f32x4_zero(self) -> Self::F32x4 {
            f32x4::zero(self)
        }

        #[inline(always)]
        fn f32x4_load(self, data: &[f32; 4]) -> Self::F32x4 {
            f32x4::load(self, data)
        }
        #[inline(always)]
        fn f64x2_splat(self, v: f64) -> Self::F64x2 {
            f64x2::splat(self, v)
        }

        #[inline(always)]
        fn f64x2_zero(self) -> Self::F64x2 {
            f64x2::zero(self)
        }

        #[inline(always)]
        fn f64x2_load(self, data: &[f64; 2]) -> Self::F64x2 {
            f64x2::load(self, data)
        }
        #[inline(always)]
        fn i8x16_splat(self, v: i8) -> Self::I8x16 {
            i8x16::splat(self, v)
        }

        #[inline(always)]
        fn i8x16_zero(self) -> Self::I8x16 {
            i8x16::zero(self)
        }

        #[inline(always)]
        fn i8x16_load(self, data: &[i8; 16]) -> Self::I8x16 {
            i8x16::load(self, data)
        }
        #[inline(always)]
        fn u8x16_splat(self, v: u8) -> Self::U8x16 {
            u8x16::splat(self, v)
        }

        #[inline(always)]
        fn u8x16_zero(self) -> Self::U8x16 {
            u8x16::zero(self)
        }

        #[inline(always)]
        fn u8x16_load(self, data: &[u8; 16]) -> Self::U8x16 {
            u8x16::load(self, data)
        }
        #[inline(always)]
        fn i16x8_splat(self, v: i16) -> Self::I16x8 {
            i16x8::splat(self, v)
        }

        #[inline(always)]
        fn i16x8_zero(self) -> Self::I16x8 {
            i16x8::zero(self)
        }

        #[inline(always)]
        fn i16x8_load(self, data: &[i16; 8]) -> Self::I16x8 {
            i16x8::load(self, data)
        }
        #[inline(always)]
        fn u16x8_splat(self, v: u16) -> Self::U16x8 {
            u16x8::splat(self, v)
        }

        #[inline(always)]
        fn u16x8_zero(self) -> Self::U16x8 {
            u16x8::zero(self)
        }

        #[inline(always)]
        fn u16x8_load(self, data: &[u16; 8]) -> Self::U16x8 {
            u16x8::load(self, data)
        }
        #[inline(always)]
        fn i32x4_splat(self, v: i32) -> Self::I32x4 {
            i32x4::splat(self, v)
        }

        #[inline(always)]
        fn i32x4_zero(self) -> Self::I32x4 {
            i32x4::zero(self)
        }

        #[inline(always)]
        fn i32x4_load(self, data: &[i32; 4]) -> Self::I32x4 {
            i32x4::load(self, data)
        }
        #[inline(always)]
        fn u32x4_splat(self, v: u32) -> Self::U32x4 {
            u32x4::splat(self, v)
        }

        #[inline(always)]
        fn u32x4_zero(self) -> Self::U32x4 {
            u32x4::zero(self)
        }

        #[inline(always)]
        fn u32x4_load(self, data: &[u32; 4]) -> Self::U32x4 {
            u32x4::load(self, data)
        }
        #[inline(always)]
        fn i64x2_splat(self, v: i64) -> Self::I64x2 {
            i64x2::splat(self, v)
        }

        #[inline(always)]
        fn i64x2_zero(self) -> Self::I64x2 {
            i64x2::zero(self)
        }

        #[inline(always)]
        fn i64x2_load(self, data: &[i64; 2]) -> Self::I64x2 {
            i64x2::load(self, data)
        }
        #[inline(always)]
        fn u64x2_splat(self, v: u64) -> Self::U64x2 {
            u64x2::splat(self, v)
        }

        #[inline(always)]
        fn u64x2_zero(self) -> Self::U64x2 {
            u64x2::zero(self)
        }

        #[inline(always)]
        fn u64x2_load(self, data: &[u64; 2]) -> Self::U64x2 {
            u64x2::load(self, data)
        }
        #[inline(always)]
        fn f32x8_splat(self, v: f32) -> Self::F32x8 {
            crate::simd::polyfill::neon::f32x8::splat(self, v)
        }

        #[inline(always)]
        fn f32x8_zero(self) -> Self::F32x8 {
            crate::simd::polyfill::neon::f32x8::zero(self)
        }

        #[inline(always)]
        fn f32x8_load(self, data: &[f32; 8]) -> Self::F32x8 {
            crate::simd::polyfill::neon::f32x8::load(self, data)
        }
        #[inline(always)]
        fn f64x4_splat(self, v: f64) -> Self::F64x4 {
            crate::simd::polyfill::neon::f64x4::splat(self, v)
        }

        #[inline(always)]
        fn f64x4_zero(self) -> Self::F64x4 {
            crate::simd::polyfill::neon::f64x4::zero(self)
        }

        #[inline(always)]
        fn f64x4_load(self, data: &[f64; 4]) -> Self::F64x4 {
            crate::simd::polyfill::neon::f64x4::load(self, data)
        }
        #[inline(always)]
        fn i8x32_splat(self, v: i8) -> Self::I8x32 {
            crate::simd::polyfill::neon::i8x32::splat(self, v)
        }

        #[inline(always)]
        fn i8x32_zero(self) -> Self::I8x32 {
            crate::simd::polyfill::neon::i8x32::zero(self)
        }

        #[inline(always)]
        fn i8x32_load(self, data: &[i8; 32]) -> Self::I8x32 {
            crate::simd::polyfill::neon::i8x32::load(self, data)
        }
        #[inline(always)]
        fn u8x32_splat(self, v: u8) -> Self::U8x32 {
            crate::simd::polyfill::neon::u8x32::splat(self, v)
        }

        #[inline(always)]
        fn u8x32_zero(self) -> Self::U8x32 {
            crate::simd::polyfill::neon::u8x32::zero(self)
        }

        #[inline(always)]
        fn u8x32_load(self, data: &[u8; 32]) -> Self::U8x32 {
            crate::simd::polyfill::neon::u8x32::load(self, data)
        }
        #[inline(always)]
        fn i16x16_splat(self, v: i16) -> Self::I16x16 {
            crate::simd::polyfill::neon::i16x16::splat(self, v)
        }

        #[inline(always)]
        fn i16x16_zero(self) -> Self::I16x16 {
            crate::simd::polyfill::neon::i16x16::zero(self)
        }

        #[inline(always)]
        fn i16x16_load(self, data: &[i16; 16]) -> Self::I16x16 {
            crate::simd::polyfill::neon::i16x16::load(self, data)
        }
        #[inline(always)]
        fn u16x16_splat(self, v: u16) -> Self::U16x16 {
            crate::simd::polyfill::neon::u16x16::splat(self, v)
        }

        #[inline(always)]
        fn u16x16_zero(self) -> Self::U16x16 {
            crate::simd::polyfill::neon::u16x16::zero(self)
        }

        #[inline(always)]
        fn u16x16_load(self, data: &[u16; 16]) -> Self::U16x16 {
            crate::simd::polyfill::neon::u16x16::load(self, data)
        }
        #[inline(always)]
        fn i32x8_splat(self, v: i32) -> Self::I32x8 {
            crate::simd::polyfill::neon::i32x8::splat(self, v)
        }

        #[inline(always)]
        fn i32x8_zero(self) -> Self::I32x8 {
            crate::simd::polyfill::neon::i32x8::zero(self)
        }

        #[inline(always)]
        fn i32x8_load(self, data: &[i32; 8]) -> Self::I32x8 {
            crate::simd::polyfill::neon::i32x8::load(self, data)
        }
        #[inline(always)]
        fn u32x8_splat(self, v: u32) -> Self::U32x8 {
            crate::simd::polyfill::neon::u32x8::splat(self, v)
        }

        #[inline(always)]
        fn u32x8_zero(self) -> Self::U32x8 {
            crate::simd::polyfill::neon::u32x8::zero(self)
        }

        #[inline(always)]
        fn u32x8_load(self, data: &[u32; 8]) -> Self::U32x8 {
            crate::simd::polyfill::neon::u32x8::load(self, data)
        }
        #[inline(always)]
        fn i64x4_splat(self, v: i64) -> Self::I64x4 {
            crate::simd::polyfill::neon::i64x4::splat(self, v)
        }

        #[inline(always)]
        fn i64x4_zero(self) -> Self::I64x4 {
            crate::simd::polyfill::neon::i64x4::zero(self)
        }

        #[inline(always)]
        fn i64x4_load(self, data: &[i64; 4]) -> Self::I64x4 {
            crate::simd::polyfill::neon::i64x4::load(self, data)
        }
        #[inline(always)]
        fn u64x4_splat(self, v: u64) -> Self::U64x4 {
            crate::simd::polyfill::neon::u64x4::splat(self, v)
        }

        #[inline(always)]
        fn u64x4_zero(self) -> Self::U64x4 {
            crate::simd::polyfill::neon::u64x4::zero(self)
        }

        #[inline(always)]
        fn u64x4_load(self, data: &[u64; 4]) -> Self::U64x4 {
            crate::simd::polyfill::neon::u64x4::load(self, data)
        }
        #[inline(always)]
        fn f32x16_splat(self, v: f32) -> Self::F32x16 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                let part = f32x4::splat(token, v);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn f32x16_zero(self) -> Self::F32x16 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                let part = f32x4::zero(token);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn f32x16_load(self, data: &[f32; 16]) -> Self::F32x16 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                [
                    f32x4::load(token, data[0..4].try_into().unwrap()),
                    f32x4::load(token, data[4..8].try_into().unwrap()),
                    f32x4::load(token, data[8..12].try_into().unwrap()),
                    f32x4::load(token, data[12..16].try_into().unwrap()),
                ]
            }
        }
        #[inline(always)]
        fn f64x8_splat(self, v: f64) -> Self::F64x8 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                let part = f64x2::splat(token, v);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn f64x8_zero(self) -> Self::F64x8 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                let part = f64x2::zero(token);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn f64x8_load(self, data: &[f64; 8]) -> Self::F64x8 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                [
                    f64x2::load(token, data[0..2].try_into().unwrap()),
                    f64x2::load(token, data[2..4].try_into().unwrap()),
                    f64x2::load(token, data[4..6].try_into().unwrap()),
                    f64x2::load(token, data[6..8].try_into().unwrap()),
                ]
            }
        }
        #[inline(always)]
        fn i8x64_splat(self, v: i8) -> Self::I8x64 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                let part = i8x16::splat(token, v);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn i8x64_zero(self) -> Self::I8x64 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                let part = i8x16::zero(token);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn i8x64_load(self, data: &[i8; 64]) -> Self::I8x64 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                [
                    i8x16::load(token, data[0..16].try_into().unwrap()),
                    i8x16::load(token, data[16..32].try_into().unwrap()),
                    i8x16::load(token, data[32..48].try_into().unwrap()),
                    i8x16::load(token, data[48..64].try_into().unwrap()),
                ]
            }
        }
        #[inline(always)]
        fn u8x64_splat(self, v: u8) -> Self::U8x64 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                let part = u8x16::splat(token, v);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn u8x64_zero(self) -> Self::U8x64 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                let part = u8x16::zero(token);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn u8x64_load(self, data: &[u8; 64]) -> Self::U8x64 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                [
                    u8x16::load(token, data[0..16].try_into().unwrap()),
                    u8x16::load(token, data[16..32].try_into().unwrap()),
                    u8x16::load(token, data[32..48].try_into().unwrap()),
                    u8x16::load(token, data[48..64].try_into().unwrap()),
                ]
            }
        }
        #[inline(always)]
        fn i16x32_splat(self, v: i16) -> Self::I16x32 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                let part = i16x8::splat(token, v);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn i16x32_zero(self) -> Self::I16x32 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                let part = i16x8::zero(token);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn i16x32_load(self, data: &[i16; 32]) -> Self::I16x32 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                [
                    i16x8::load(token, data[0..8].try_into().unwrap()),
                    i16x8::load(token, data[8..16].try_into().unwrap()),
                    i16x8::load(token, data[16..24].try_into().unwrap()),
                    i16x8::load(token, data[24..32].try_into().unwrap()),
                ]
            }
        }
        #[inline(always)]
        fn u16x32_splat(self, v: u16) -> Self::U16x32 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                let part = u16x8::splat(token, v);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn u16x32_zero(self) -> Self::U16x32 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                let part = u16x8::zero(token);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn u16x32_load(self, data: &[u16; 32]) -> Self::U16x32 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                [
                    u16x8::load(token, data[0..8].try_into().unwrap()),
                    u16x8::load(token, data[8..16].try_into().unwrap()),
                    u16x8::load(token, data[16..24].try_into().unwrap()),
                    u16x8::load(token, data[24..32].try_into().unwrap()),
                ]
            }
        }
        #[inline(always)]
        fn i32x16_splat(self, v: i32) -> Self::I32x16 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                let part = i32x4::splat(token, v);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn i32x16_zero(self) -> Self::I32x16 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                let part = i32x4::zero(token);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn i32x16_load(self, data: &[i32; 16]) -> Self::I32x16 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                [
                    i32x4::load(token, data[0..4].try_into().unwrap()),
                    i32x4::load(token, data[4..8].try_into().unwrap()),
                    i32x4::load(token, data[8..12].try_into().unwrap()),
                    i32x4::load(token, data[12..16].try_into().unwrap()),
                ]
            }
        }
        #[inline(always)]
        fn u32x16_splat(self, v: u32) -> Self::U32x16 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                let part = u32x4::splat(token, v);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn u32x16_zero(self) -> Self::U32x16 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                let part = u32x4::zero(token);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn u32x16_load(self, data: &[u32; 16]) -> Self::U32x16 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                [
                    u32x4::load(token, data[0..4].try_into().unwrap()),
                    u32x4::load(token, data[4..8].try_into().unwrap()),
                    u32x4::load(token, data[8..12].try_into().unwrap()),
                    u32x4::load(token, data[12..16].try_into().unwrap()),
                ]
            }
        }
        #[inline(always)]
        fn i64x8_splat(self, v: i64) -> Self::I64x8 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                let part = i64x2::splat(token, v);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn i64x8_zero(self) -> Self::I64x8 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                let part = i64x2::zero(token);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn i64x8_load(self, data: &[i64; 8]) -> Self::I64x8 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                [
                    i64x2::load(token, data[0..2].try_into().unwrap()),
                    i64x2::load(token, data[2..4].try_into().unwrap()),
                    i64x2::load(token, data[4..6].try_into().unwrap()),
                    i64x2::load(token, data[6..8].try_into().unwrap()),
                ]
            }
        }
        #[inline(always)]
        fn u64x8_splat(self, v: u64) -> Self::U64x8 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                let part = u64x2::splat(token, v);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn u64x8_zero(self) -> Self::U64x8 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                let part = u64x2::zero(token);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn u64x8_load(self, data: &[u64; 8]) -> Self::U64x8 {
            {
                let token = unsafe { NeonToken::forge_token_dangerously() };
                [
                    u64x2::load(token, data[0..2].try_into().unwrap()),
                    u64x2::load(token, data[2..4].try_into().unwrap()),
                    u64x2::load(token, data[4..6].try_into().unwrap()),
                    u64x2::load(token, data[6..8].try_into().unwrap()),
                ]
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm_impl {
    use super::WidthDispatch;
    use archmage::{SimdToken, Wasm128Token};

    use crate::simd::{f32x4, f64x2, i8x16, i16x8, i32x4, i64x2, u8x16, u16x8, u32x4, u64x2};

    impl WidthDispatch for Wasm128Token {
        // 128-bit types
        type F32x4 = f32x4;
        type F64x2 = f64x2;
        type I8x16 = i8x16;
        type U8x16 = u8x16;
        type I16x8 = i16x8;
        type U16x8 = u16x8;
        type I32x4 = i32x4;
        type U32x4 = u32x4;
        type I64x2 = i64x2;
        type U64x2 = u64x2;

        // 256-bit types
        type F32x8 = crate::simd::polyfill::wasm128::f32x8;
        type F64x4 = crate::simd::polyfill::wasm128::f64x4;
        type I8x32 = crate::simd::polyfill::wasm128::i8x32;
        type U8x32 = crate::simd::polyfill::wasm128::u8x32;
        type I16x16 = crate::simd::polyfill::wasm128::i16x16;
        type U16x16 = crate::simd::polyfill::wasm128::u16x16;
        type I32x8 = crate::simd::polyfill::wasm128::i32x8;
        type U32x8 = crate::simd::polyfill::wasm128::u32x8;
        type I64x4 = crate::simd::polyfill::wasm128::i64x4;
        type U64x4 = crate::simd::polyfill::wasm128::u64x4;

        // 512-bit types
        type F32x16 = [f32x4; 4];
        type F64x8 = [f64x2; 4];
        type I8x64 = [i8x16; 4];
        type U8x64 = [u8x16; 4];
        type I16x32 = [i16x8; 4];
        type U16x32 = [u16x8; 4];
        type I32x16 = [i32x4; 4];
        type U32x16 = [u32x4; 4];
        type I64x8 = [i64x2; 4];
        type U64x8 = [u64x2; 4];

        #[inline(always)]
        fn f32x4_splat(self, v: f32) -> Self::F32x4 {
            f32x4::splat(self, v)
        }

        #[inline(always)]
        fn f32x4_zero(self) -> Self::F32x4 {
            f32x4::zero(self)
        }

        #[inline(always)]
        fn f32x4_load(self, data: &[f32; 4]) -> Self::F32x4 {
            f32x4::load(self, data)
        }
        #[inline(always)]
        fn f64x2_splat(self, v: f64) -> Self::F64x2 {
            f64x2::splat(self, v)
        }

        #[inline(always)]
        fn f64x2_zero(self) -> Self::F64x2 {
            f64x2::zero(self)
        }

        #[inline(always)]
        fn f64x2_load(self, data: &[f64; 2]) -> Self::F64x2 {
            f64x2::load(self, data)
        }
        #[inline(always)]
        fn i8x16_splat(self, v: i8) -> Self::I8x16 {
            i8x16::splat(self, v)
        }

        #[inline(always)]
        fn i8x16_zero(self) -> Self::I8x16 {
            i8x16::zero(self)
        }

        #[inline(always)]
        fn i8x16_load(self, data: &[i8; 16]) -> Self::I8x16 {
            i8x16::load(self, data)
        }
        #[inline(always)]
        fn u8x16_splat(self, v: u8) -> Self::U8x16 {
            u8x16::splat(self, v)
        }

        #[inline(always)]
        fn u8x16_zero(self) -> Self::U8x16 {
            u8x16::zero(self)
        }

        #[inline(always)]
        fn u8x16_load(self, data: &[u8; 16]) -> Self::U8x16 {
            u8x16::load(self, data)
        }
        #[inline(always)]
        fn i16x8_splat(self, v: i16) -> Self::I16x8 {
            i16x8::splat(self, v)
        }

        #[inline(always)]
        fn i16x8_zero(self) -> Self::I16x8 {
            i16x8::zero(self)
        }

        #[inline(always)]
        fn i16x8_load(self, data: &[i16; 8]) -> Self::I16x8 {
            i16x8::load(self, data)
        }
        #[inline(always)]
        fn u16x8_splat(self, v: u16) -> Self::U16x8 {
            u16x8::splat(self, v)
        }

        #[inline(always)]
        fn u16x8_zero(self) -> Self::U16x8 {
            u16x8::zero(self)
        }

        #[inline(always)]
        fn u16x8_load(self, data: &[u16; 8]) -> Self::U16x8 {
            u16x8::load(self, data)
        }
        #[inline(always)]
        fn i32x4_splat(self, v: i32) -> Self::I32x4 {
            i32x4::splat(self, v)
        }

        #[inline(always)]
        fn i32x4_zero(self) -> Self::I32x4 {
            i32x4::zero(self)
        }

        #[inline(always)]
        fn i32x4_load(self, data: &[i32; 4]) -> Self::I32x4 {
            i32x4::load(self, data)
        }
        #[inline(always)]
        fn u32x4_splat(self, v: u32) -> Self::U32x4 {
            u32x4::splat(self, v)
        }

        #[inline(always)]
        fn u32x4_zero(self) -> Self::U32x4 {
            u32x4::zero(self)
        }

        #[inline(always)]
        fn u32x4_load(self, data: &[u32; 4]) -> Self::U32x4 {
            u32x4::load(self, data)
        }
        #[inline(always)]
        fn i64x2_splat(self, v: i64) -> Self::I64x2 {
            i64x2::splat(self, v)
        }

        #[inline(always)]
        fn i64x2_zero(self) -> Self::I64x2 {
            i64x2::zero(self)
        }

        #[inline(always)]
        fn i64x2_load(self, data: &[i64; 2]) -> Self::I64x2 {
            i64x2::load(self, data)
        }
        #[inline(always)]
        fn u64x2_splat(self, v: u64) -> Self::U64x2 {
            u64x2::splat(self, v)
        }

        #[inline(always)]
        fn u64x2_zero(self) -> Self::U64x2 {
            u64x2::zero(self)
        }

        #[inline(always)]
        fn u64x2_load(self, data: &[u64; 2]) -> Self::U64x2 {
            u64x2::load(self, data)
        }
        #[inline(always)]
        fn f32x8_splat(self, v: f32) -> Self::F32x8 {
            crate::simd::polyfill::wasm128::f32x8::splat(self, v)
        }

        #[inline(always)]
        fn f32x8_zero(self) -> Self::F32x8 {
            crate::simd::polyfill::wasm128::f32x8::zero(self)
        }

        #[inline(always)]
        fn f32x8_load(self, data: &[f32; 8]) -> Self::F32x8 {
            crate::simd::polyfill::wasm128::f32x8::load(self, data)
        }
        #[inline(always)]
        fn f64x4_splat(self, v: f64) -> Self::F64x4 {
            crate::simd::polyfill::wasm128::f64x4::splat(self, v)
        }

        #[inline(always)]
        fn f64x4_zero(self) -> Self::F64x4 {
            crate::simd::polyfill::wasm128::f64x4::zero(self)
        }

        #[inline(always)]
        fn f64x4_load(self, data: &[f64; 4]) -> Self::F64x4 {
            crate::simd::polyfill::wasm128::f64x4::load(self, data)
        }
        #[inline(always)]
        fn i8x32_splat(self, v: i8) -> Self::I8x32 {
            crate::simd::polyfill::wasm128::i8x32::splat(self, v)
        }

        #[inline(always)]
        fn i8x32_zero(self) -> Self::I8x32 {
            crate::simd::polyfill::wasm128::i8x32::zero(self)
        }

        #[inline(always)]
        fn i8x32_load(self, data: &[i8; 32]) -> Self::I8x32 {
            crate::simd::polyfill::wasm128::i8x32::load(self, data)
        }
        #[inline(always)]
        fn u8x32_splat(self, v: u8) -> Self::U8x32 {
            crate::simd::polyfill::wasm128::u8x32::splat(self, v)
        }

        #[inline(always)]
        fn u8x32_zero(self) -> Self::U8x32 {
            crate::simd::polyfill::wasm128::u8x32::zero(self)
        }

        #[inline(always)]
        fn u8x32_load(self, data: &[u8; 32]) -> Self::U8x32 {
            crate::simd::polyfill::wasm128::u8x32::load(self, data)
        }
        #[inline(always)]
        fn i16x16_splat(self, v: i16) -> Self::I16x16 {
            crate::simd::polyfill::wasm128::i16x16::splat(self, v)
        }

        #[inline(always)]
        fn i16x16_zero(self) -> Self::I16x16 {
            crate::simd::polyfill::wasm128::i16x16::zero(self)
        }

        #[inline(always)]
        fn i16x16_load(self, data: &[i16; 16]) -> Self::I16x16 {
            crate::simd::polyfill::wasm128::i16x16::load(self, data)
        }
        #[inline(always)]
        fn u16x16_splat(self, v: u16) -> Self::U16x16 {
            crate::simd::polyfill::wasm128::u16x16::splat(self, v)
        }

        #[inline(always)]
        fn u16x16_zero(self) -> Self::U16x16 {
            crate::simd::polyfill::wasm128::u16x16::zero(self)
        }

        #[inline(always)]
        fn u16x16_load(self, data: &[u16; 16]) -> Self::U16x16 {
            crate::simd::polyfill::wasm128::u16x16::load(self, data)
        }
        #[inline(always)]
        fn i32x8_splat(self, v: i32) -> Self::I32x8 {
            crate::simd::polyfill::wasm128::i32x8::splat(self, v)
        }

        #[inline(always)]
        fn i32x8_zero(self) -> Self::I32x8 {
            crate::simd::polyfill::wasm128::i32x8::zero(self)
        }

        #[inline(always)]
        fn i32x8_load(self, data: &[i32; 8]) -> Self::I32x8 {
            crate::simd::polyfill::wasm128::i32x8::load(self, data)
        }
        #[inline(always)]
        fn u32x8_splat(self, v: u32) -> Self::U32x8 {
            crate::simd::polyfill::wasm128::u32x8::splat(self, v)
        }

        #[inline(always)]
        fn u32x8_zero(self) -> Self::U32x8 {
            crate::simd::polyfill::wasm128::u32x8::zero(self)
        }

        #[inline(always)]
        fn u32x8_load(self, data: &[u32; 8]) -> Self::U32x8 {
            crate::simd::polyfill::wasm128::u32x8::load(self, data)
        }
        #[inline(always)]
        fn i64x4_splat(self, v: i64) -> Self::I64x4 {
            crate::simd::polyfill::wasm128::i64x4::splat(self, v)
        }

        #[inline(always)]
        fn i64x4_zero(self) -> Self::I64x4 {
            crate::simd::polyfill::wasm128::i64x4::zero(self)
        }

        #[inline(always)]
        fn i64x4_load(self, data: &[i64; 4]) -> Self::I64x4 {
            crate::simd::polyfill::wasm128::i64x4::load(self, data)
        }
        #[inline(always)]
        fn u64x4_splat(self, v: u64) -> Self::U64x4 {
            crate::simd::polyfill::wasm128::u64x4::splat(self, v)
        }

        #[inline(always)]
        fn u64x4_zero(self) -> Self::U64x4 {
            crate::simd::polyfill::wasm128::u64x4::zero(self)
        }

        #[inline(always)]
        fn u64x4_load(self, data: &[u64; 4]) -> Self::U64x4 {
            crate::simd::polyfill::wasm128::u64x4::load(self, data)
        }
        #[inline(always)]
        fn f32x16_splat(self, v: f32) -> Self::F32x16 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                let part = f32x4::splat(token, v);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn f32x16_zero(self) -> Self::F32x16 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                let part = f32x4::zero(token);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn f32x16_load(self, data: &[f32; 16]) -> Self::F32x16 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                [
                    f32x4::load(token, data[0..4].try_into().unwrap()),
                    f32x4::load(token, data[4..8].try_into().unwrap()),
                    f32x4::load(token, data[8..12].try_into().unwrap()),
                    f32x4::load(token, data[12..16].try_into().unwrap()),
                ]
            }
        }
        #[inline(always)]
        fn f64x8_splat(self, v: f64) -> Self::F64x8 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                let part = f64x2::splat(token, v);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn f64x8_zero(self) -> Self::F64x8 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                let part = f64x2::zero(token);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn f64x8_load(self, data: &[f64; 8]) -> Self::F64x8 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                [
                    f64x2::load(token, data[0..2].try_into().unwrap()),
                    f64x2::load(token, data[2..4].try_into().unwrap()),
                    f64x2::load(token, data[4..6].try_into().unwrap()),
                    f64x2::load(token, data[6..8].try_into().unwrap()),
                ]
            }
        }
        #[inline(always)]
        fn i8x64_splat(self, v: i8) -> Self::I8x64 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                let part = i8x16::splat(token, v);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn i8x64_zero(self) -> Self::I8x64 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                let part = i8x16::zero(token);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn i8x64_load(self, data: &[i8; 64]) -> Self::I8x64 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                [
                    i8x16::load(token, data[0..16].try_into().unwrap()),
                    i8x16::load(token, data[16..32].try_into().unwrap()),
                    i8x16::load(token, data[32..48].try_into().unwrap()),
                    i8x16::load(token, data[48..64].try_into().unwrap()),
                ]
            }
        }
        #[inline(always)]
        fn u8x64_splat(self, v: u8) -> Self::U8x64 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                let part = u8x16::splat(token, v);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn u8x64_zero(self) -> Self::U8x64 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                let part = u8x16::zero(token);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn u8x64_load(self, data: &[u8; 64]) -> Self::U8x64 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                [
                    u8x16::load(token, data[0..16].try_into().unwrap()),
                    u8x16::load(token, data[16..32].try_into().unwrap()),
                    u8x16::load(token, data[32..48].try_into().unwrap()),
                    u8x16::load(token, data[48..64].try_into().unwrap()),
                ]
            }
        }
        #[inline(always)]
        fn i16x32_splat(self, v: i16) -> Self::I16x32 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                let part = i16x8::splat(token, v);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn i16x32_zero(self) -> Self::I16x32 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                let part = i16x8::zero(token);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn i16x32_load(self, data: &[i16; 32]) -> Self::I16x32 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                [
                    i16x8::load(token, data[0..8].try_into().unwrap()),
                    i16x8::load(token, data[8..16].try_into().unwrap()),
                    i16x8::load(token, data[16..24].try_into().unwrap()),
                    i16x8::load(token, data[24..32].try_into().unwrap()),
                ]
            }
        }
        #[inline(always)]
        fn u16x32_splat(self, v: u16) -> Self::U16x32 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                let part = u16x8::splat(token, v);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn u16x32_zero(self) -> Self::U16x32 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                let part = u16x8::zero(token);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn u16x32_load(self, data: &[u16; 32]) -> Self::U16x32 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                [
                    u16x8::load(token, data[0..8].try_into().unwrap()),
                    u16x8::load(token, data[8..16].try_into().unwrap()),
                    u16x8::load(token, data[16..24].try_into().unwrap()),
                    u16x8::load(token, data[24..32].try_into().unwrap()),
                ]
            }
        }
        #[inline(always)]
        fn i32x16_splat(self, v: i32) -> Self::I32x16 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                let part = i32x4::splat(token, v);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn i32x16_zero(self) -> Self::I32x16 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                let part = i32x4::zero(token);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn i32x16_load(self, data: &[i32; 16]) -> Self::I32x16 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                [
                    i32x4::load(token, data[0..4].try_into().unwrap()),
                    i32x4::load(token, data[4..8].try_into().unwrap()),
                    i32x4::load(token, data[8..12].try_into().unwrap()),
                    i32x4::load(token, data[12..16].try_into().unwrap()),
                ]
            }
        }
        #[inline(always)]
        fn u32x16_splat(self, v: u32) -> Self::U32x16 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                let part = u32x4::splat(token, v);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn u32x16_zero(self) -> Self::U32x16 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                let part = u32x4::zero(token);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn u32x16_load(self, data: &[u32; 16]) -> Self::U32x16 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                [
                    u32x4::load(token, data[0..4].try_into().unwrap()),
                    u32x4::load(token, data[4..8].try_into().unwrap()),
                    u32x4::load(token, data[8..12].try_into().unwrap()),
                    u32x4::load(token, data[12..16].try_into().unwrap()),
                ]
            }
        }
        #[inline(always)]
        fn i64x8_splat(self, v: i64) -> Self::I64x8 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                let part = i64x2::splat(token, v);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn i64x8_zero(self) -> Self::I64x8 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                let part = i64x2::zero(token);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn i64x8_load(self, data: &[i64; 8]) -> Self::I64x8 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                [
                    i64x2::load(token, data[0..2].try_into().unwrap()),
                    i64x2::load(token, data[2..4].try_into().unwrap()),
                    i64x2::load(token, data[4..6].try_into().unwrap()),
                    i64x2::load(token, data[6..8].try_into().unwrap()),
                ]
            }
        }
        #[inline(always)]
        fn u64x8_splat(self, v: u64) -> Self::U64x8 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                let part = u64x2::splat(token, v);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn u64x8_zero(self) -> Self::U64x8 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                let part = u64x2::zero(token);
                [part, part, part, part]
            }
        }

        #[inline(always)]
        fn u64x8_load(self, data: &[u64; 8]) -> Self::U64x8 {
            {
                let token = unsafe { Wasm128Token::forge_token_dangerously() };
                [
                    u64x2::load(token, data[0..2].try_into().unwrap()),
                    u64x2::load(token, data[2..4].try_into().unwrap()),
                    u64x2::load(token, data[4..6].try_into().unwrap()),
                    u64x2::load(token, data[6..8].try_into().unwrap()),
                ]
            }
        }
    }
}

//! Width dispatch trait for token-based SIMD type construction.
//!
//! See [`WidthDispatch`] for the main trait.

#![allow(missing_docs)]
//!
//! The `WidthDispatch` trait provides access to ALL SIMD sizes from any token.
//! Native types are used where the hardware supports them; polyfills are used
//! for wider types on narrower hardware.
//!
//! # Example
//!
//! ```ignore
//! use archmage::{Avx2FmaToken, SimdToken};
//! use magetypes::WidthDispatch;
//!
//! fn process<T: WidthDispatch>(token: T, data: &[f32; 8]) -> T::F32x8 {
//!     let v = token.f32x8_load(data);
//!     let two = token.f32x8_splat(2.0);
//!     v * two
//! }
//!
//! // On SSE: f32x8 is polyfilled (2x f32x4)
//! // On AVX2: f32x8 is native
//! if let Some(token) = Avx2FmaToken::summon() {
//!     let result = process(token, &[1.0; 8]);
//! }
//! ```

use archmage::SimdToken;

/// Trait providing access to all SIMD sizes from a capability token.
///
/// Every token implementing this trait can construct vectors of any size.
/// The associated types determine whether native or polyfilled implementations
/// are used based on the token's hardware capabilities.
pub trait WidthDispatch: SimdToken + Copy {
    // ========================================================================
    // 128-bit types (4x f32, 2x f64, 16x i8, etc.)
    // ========================================================================
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

    // ========================================================================
    // 256-bit types (8x f32, 4x f64, 32x i8, etc.)
    // ========================================================================
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

    // ========================================================================
    // 512-bit types (16x f32, 8x f64, 64x i8, etc.)
    // ========================================================================
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

    // ========================================================================
    // f32x4 constructors
    // ========================================================================
    fn f32x4_splat(self, v: f32) -> Self::F32x4;
    fn f32x4_zero(self) -> Self::F32x4;
    fn f32x4_load(self, data: &[f32; 4]) -> Self::F32x4;

    // ========================================================================
    // f32x8 constructors
    // ========================================================================
    fn f32x8_splat(self, v: f32) -> Self::F32x8;
    fn f32x8_zero(self) -> Self::F32x8;
    fn f32x8_load(self, data: &[f32; 8]) -> Self::F32x8;

    // ========================================================================
    // f32x16 constructors
    // ========================================================================
    fn f32x16_splat(self, v: f32) -> Self::F32x16;
    fn f32x16_zero(self) -> Self::F32x16;
    fn f32x16_load(self, data: &[f32; 16]) -> Self::F32x16;

    // ========================================================================
    // f64x2 constructors
    // ========================================================================
    fn f64x2_splat(self, v: f64) -> Self::F64x2;
    fn f64x2_zero(self) -> Self::F64x2;
    fn f64x2_load(self, data: &[f64; 2]) -> Self::F64x2;

    // ========================================================================
    // f64x4 constructors
    // ========================================================================
    fn f64x4_splat(self, v: f64) -> Self::F64x4;
    fn f64x4_zero(self) -> Self::F64x4;
    fn f64x4_load(self, data: &[f64; 4]) -> Self::F64x4;

    // ========================================================================
    // f64x8 constructors
    // ========================================================================
    fn f64x8_splat(self, v: f64) -> Self::F64x8;
    fn f64x8_zero(self) -> Self::F64x8;
    fn f64x8_load(self, data: &[f64; 8]) -> Self::F64x8;

    // ========================================================================
    // i32x4 constructors
    // ========================================================================
    fn i32x4_splat(self, v: i32) -> Self::I32x4;
    fn i32x4_zero(self) -> Self::I32x4;
    fn i32x4_load(self, data: &[i32; 4]) -> Self::I32x4;

    // ========================================================================
    // i32x8 constructors
    // ========================================================================
    fn i32x8_splat(self, v: i32) -> Self::I32x8;
    fn i32x8_zero(self) -> Self::I32x8;
    fn i32x8_load(self, data: &[i32; 8]) -> Self::I32x8;

    // ========================================================================
    // i32x16 constructors
    // ========================================================================
    fn i32x16_splat(self, v: i32) -> Self::I32x16;
    fn i32x16_zero(self) -> Self::I32x16;
    fn i32x16_load(self, data: &[i32; 16]) -> Self::I32x16;

    // ========================================================================
    // u8x16 constructors
    // ========================================================================
    fn u8x16_splat(self, v: u8) -> Self::U8x16;
    fn u8x16_zero(self) -> Self::U8x16;
    fn u8x16_load(self, data: &[u8; 16]) -> Self::U8x16;

    // ========================================================================
    // u8x32 constructors
    // ========================================================================
    fn u8x32_splat(self, v: u8) -> Self::U8x32;
    fn u8x32_zero(self) -> Self::U8x32;
    fn u8x32_load(self, data: &[u8; 32]) -> Self::U8x32;

    // ========================================================================
    // u8x64 constructors
    // ========================================================================
    fn u8x64_splat(self, v: u8) -> Self::U8x64;
    fn u8x64_zero(self) -> Self::U8x64;
    fn u8x64_load(self, data: &[u8; 64]) -> Self::U8x64;

    // ========================================================================
    // i8x16 constructors
    // ========================================================================
    fn i8x16_splat(self, v: i8) -> Self::I8x16;
    fn i8x16_zero(self) -> Self::I8x16;
    fn i8x16_load(self, data: &[i8; 16]) -> Self::I8x16;

    // ========================================================================
    // i8x32 constructors
    // ========================================================================
    fn i8x32_splat(self, v: i8) -> Self::I8x32;
    fn i8x32_zero(self) -> Self::I8x32;
    fn i8x32_load(self, data: &[i8; 32]) -> Self::I8x32;

    // ========================================================================
    // i8x64 constructors
    // ========================================================================
    fn i8x64_splat(self, v: i8) -> Self::I8x64;
    fn i8x64_zero(self) -> Self::I8x64;
    fn i8x64_load(self, data: &[i8; 64]) -> Self::I8x64;

    // ========================================================================
    // u16x8 constructors
    // ========================================================================
    fn u16x8_splat(self, v: u16) -> Self::U16x8;
    fn u16x8_zero(self) -> Self::U16x8;
    fn u16x8_load(self, data: &[u16; 8]) -> Self::U16x8;

    // ========================================================================
    // u16x16 constructors
    // ========================================================================
    fn u16x16_splat(self, v: u16) -> Self::U16x16;
    fn u16x16_zero(self) -> Self::U16x16;
    fn u16x16_load(self, data: &[u16; 16]) -> Self::U16x16;

    // ========================================================================
    // u16x32 constructors
    // ========================================================================
    fn u16x32_splat(self, v: u16) -> Self::U16x32;
    fn u16x32_zero(self) -> Self::U16x32;
    fn u16x32_load(self, data: &[u16; 32]) -> Self::U16x32;

    // ========================================================================
    // i16x8 constructors
    // ========================================================================
    fn i16x8_splat(self, v: i16) -> Self::I16x8;
    fn i16x8_zero(self) -> Self::I16x8;
    fn i16x8_load(self, data: &[i16; 8]) -> Self::I16x8;

    // ========================================================================
    // i16x16 constructors
    // ========================================================================
    fn i16x16_splat(self, v: i16) -> Self::I16x16;
    fn i16x16_zero(self) -> Self::I16x16;
    fn i16x16_load(self, data: &[i16; 16]) -> Self::I16x16;

    // ========================================================================
    // i16x32 constructors
    // ========================================================================
    fn i16x32_splat(self, v: i16) -> Self::I16x32;
    fn i16x32_zero(self) -> Self::I16x32;
    fn i16x32_load(self, data: &[i16; 32]) -> Self::I16x32;

    // ========================================================================
    // u32x4 constructors
    // ========================================================================
    fn u32x4_splat(self, v: u32) -> Self::U32x4;
    fn u32x4_zero(self) -> Self::U32x4;
    fn u32x4_load(self, data: &[u32; 4]) -> Self::U32x4;

    // ========================================================================
    // u32x8 constructors
    // ========================================================================
    fn u32x8_splat(self, v: u32) -> Self::U32x8;
    fn u32x8_zero(self) -> Self::U32x8;
    fn u32x8_load(self, data: &[u32; 8]) -> Self::U32x8;

    // ========================================================================
    // u32x16 constructors
    // ========================================================================
    fn u32x16_splat(self, v: u32) -> Self::U32x16;
    fn u32x16_zero(self) -> Self::U32x16;
    fn u32x16_load(self, data: &[u32; 16]) -> Self::U32x16;

    // ========================================================================
    // i64x2 constructors
    // ========================================================================
    fn i64x2_splat(self, v: i64) -> Self::I64x2;
    fn i64x2_zero(self) -> Self::I64x2;
    fn i64x2_load(self, data: &[i64; 2]) -> Self::I64x2;

    // ========================================================================
    // i64x4 constructors
    // ========================================================================
    fn i64x4_splat(self, v: i64) -> Self::I64x4;
    fn i64x4_zero(self) -> Self::I64x4;
    fn i64x4_load(self, data: &[i64; 4]) -> Self::I64x4;

    // ========================================================================
    // i64x8 constructors
    // ========================================================================
    fn i64x8_splat(self, v: i64) -> Self::I64x8;
    fn i64x8_zero(self) -> Self::I64x8;
    fn i64x8_load(self, data: &[i64; 8]) -> Self::I64x8;

    // ========================================================================
    // u64x2 constructors
    // ========================================================================
    fn u64x2_splat(self, v: u64) -> Self::U64x2;
    fn u64x2_zero(self) -> Self::U64x2;
    fn u64x2_load(self, data: &[u64; 2]) -> Self::U64x2;

    // ========================================================================
    // u64x4 constructors
    // ========================================================================
    fn u64x4_splat(self, v: u64) -> Self::U64x4;
    fn u64x4_zero(self) -> Self::U64x4;
    fn u64x4_load(self, data: &[u64; 4]) -> Self::U64x4;

    // ========================================================================
    // u64x8 constructors
    // ========================================================================
    fn u64x8_splat(self, v: u64) -> Self::U64x8;
    fn u64x8_zero(self) -> Self::U64x8;
    fn u64x8_load(self, data: &[u64; 8]) -> Self::U64x8;
}

// ============================================================================
// x86_64 Implementations
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod x86_impl {
    use super::WidthDispatch;
    use archmage::{Avx2FmaToken, SimdToken, Sse41Token};

    // Import native types (re-exported at simd level)
    use crate::simd::{
        f32x4, f32x8, f64x2, f64x4, i8x16, i8x32, i16x8, i16x16, i32x4, i32x8, i64x2, i64x4,
        u8x16, u8x32, u16x8, u16x16, u32x4, u32x8, u64x2, u64x4,
    };

    // Import polyfills for 512-bit on AVX2
    use crate::simd::polyfill::avx2 as poly512;

    // ========================================================================
    // Avx2FmaToken Implementation
    // ========================================================================

    impl WidthDispatch for Avx2FmaToken {
        // 128-bit types (native SSE, but accessible via AVX2 token)
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

        // 256-bit types (native AVX2)
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

        // 512-bit types (polyfilled using 2x 256-bit)
        type F32x16 = poly512::f32x16;
        type F64x8 = poly512::f64x8;
        // For integer 512-bit types, we use simple array wrappers for now
        type I8x64 = [i8x16; 4];
        type U8x64 = [u8x16; 4];
        type I16x32 = [i16x8; 4];
        type U16x32 = [u16x8; 4];
        type I32x16 = poly512::i32x16;
        type U32x16 = [u32x4; 4];
        type I64x8 = [i64x2; 4];
        type U64x8 = [u64x2; 4];

        // f32x4 constructors
        #[inline(always)]
        fn f32x4_splat(self, v: f32) -> Self::F32x4 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            f32x4::splat(token, v)
        }

        #[inline(always)]
        fn f32x4_zero(self) -> Self::F32x4 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            f32x4::zero(token)
        }

        #[inline(always)]
        fn f32x4_load(self, data: &[f32; 4]) -> Self::F32x4 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            f32x4::load(token, data)
        }

        // f32x8 constructors
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

        // f32x16 constructors
        #[inline(always)]
        fn f32x16_splat(self, v: f32) -> Self::F32x16 {
            poly512::f32x16::splat(self, v)
        }

        #[inline(always)]
        fn f32x16_zero(self) -> Self::F32x16 {
            poly512::f32x16::zero(self)
        }

        #[inline(always)]
        fn f32x16_load(self, data: &[f32; 16]) -> Self::F32x16 {
            poly512::f32x16::load(self, data)
        }

        // f64x2 constructors
        #[inline(always)]
        fn f64x2_splat(self, v: f64) -> Self::F64x2 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            f64x2::splat(token, v)
        }

        #[inline(always)]
        fn f64x2_zero(self) -> Self::F64x2 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            f64x2::zero(token)
        }

        #[inline(always)]
        fn f64x2_load(self, data: &[f64; 2]) -> Self::F64x2 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            f64x2::load(token, data)
        }

        // f64x4 constructors
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

        // f64x8 constructors
        #[inline(always)]
        fn f64x8_splat(self, v: f64) -> Self::F64x8 {
            poly512::f64x8::splat(self, v)
        }

        #[inline(always)]
        fn f64x8_zero(self) -> Self::F64x8 {
            poly512::f64x8::zero(self)
        }

        #[inline(always)]
        fn f64x8_load(self, data: &[f64; 8]) -> Self::F64x8 {
            poly512::f64x8::load(self, data)
        }

        // i32x4 constructors
        #[inline(always)]
        fn i32x4_splat(self, v: i32) -> Self::I32x4 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            i32x4::splat(token, v)
        }

        #[inline(always)]
        fn i32x4_zero(self) -> Self::I32x4 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            i32x4::zero(token)
        }

        #[inline(always)]
        fn i32x4_load(self, data: &[i32; 4]) -> Self::I32x4 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            i32x4::load(token, data)
        }

        // i32x8 constructors
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

        // i32x16 constructors
        #[inline(always)]
        fn i32x16_splat(self, v: i32) -> Self::I32x16 {
            poly512::i32x16::splat(self, v)
        }

        #[inline(always)]
        fn i32x16_zero(self) -> Self::I32x16 {
            poly512::i32x16::zero(self)
        }

        #[inline(always)]
        fn i32x16_load(self, data: &[i32; 16]) -> Self::I32x16 {
            poly512::i32x16::load(self, data)
        }

        // u8x16 constructors
        #[inline(always)]
        fn u8x16_splat(self, v: u8) -> Self::U8x16 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            u8x16::splat(token, v)
        }

        #[inline(always)]
        fn u8x16_zero(self) -> Self::U8x16 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            u8x16::zero(token)
        }

        #[inline(always)]
        fn u8x16_load(self, data: &[u8; 16]) -> Self::U8x16 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            u8x16::load(token, data)
        }

        // u8x32 constructors
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

        // u8x64 constructors (array-based polyfill)
        #[inline(always)]
        fn u8x64_splat(self, v: u8) -> Self::U8x64 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            let part = u8x16::splat(token, v);
            [part, part, part, part]
        }

        #[inline(always)]
        fn u8x64_zero(self) -> Self::U8x64 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            let part = u8x16::zero(token);
            [part, part, part, part]
        }

        #[inline(always)]
        fn u8x64_load(self, data: &[u8; 64]) -> Self::U8x64 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            [
                u8x16::load(token, data[0..16].try_into().unwrap()),
                u8x16::load(token, data[16..32].try_into().unwrap()),
                u8x16::load(token, data[32..48].try_into().unwrap()),
                u8x16::load(token, data[48..64].try_into().unwrap()),
            ]
        }

        // i8x16 constructors
        #[inline(always)]
        fn i8x16_splat(self, v: i8) -> Self::I8x16 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            i8x16::splat(token, v)
        }

        #[inline(always)]
        fn i8x16_zero(self) -> Self::I8x16 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            i8x16::zero(token)
        }

        #[inline(always)]
        fn i8x16_load(self, data: &[i8; 16]) -> Self::I8x16 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            i8x16::load(token, data)
        }

        // i8x32 constructors
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

        // i8x64 constructors (array-based polyfill)
        #[inline(always)]
        fn i8x64_splat(self, v: i8) -> Self::I8x64 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            let part = i8x16::splat(token, v);
            [part, part, part, part]
        }

        #[inline(always)]
        fn i8x64_zero(self) -> Self::I8x64 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            let part = i8x16::zero(token);
            [part, part, part, part]
        }

        #[inline(always)]
        fn i8x64_load(self, data: &[i8; 64]) -> Self::I8x64 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            [
                i8x16::load(token, data[0..16].try_into().unwrap()),
                i8x16::load(token, data[16..32].try_into().unwrap()),
                i8x16::load(token, data[32..48].try_into().unwrap()),
                i8x16::load(token, data[48..64].try_into().unwrap()),
            ]
        }

        // u16x8 constructors
        #[inline(always)]
        fn u16x8_splat(self, v: u16) -> Self::U16x8 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            u16x8::splat(token, v)
        }

        #[inline(always)]
        fn u16x8_zero(self) -> Self::U16x8 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            u16x8::zero(token)
        }

        #[inline(always)]
        fn u16x8_load(self, data: &[u16; 8]) -> Self::U16x8 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            u16x8::load(token, data)
        }

        // u16x16 constructors
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

        // u16x32 constructors (array-based polyfill)
        #[inline(always)]
        fn u16x32_splat(self, v: u16) -> Self::U16x32 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            let part = u16x8::splat(token, v);
            [part, part, part, part]
        }

        #[inline(always)]
        fn u16x32_zero(self) -> Self::U16x32 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            let part = u16x8::zero(token);
            [part, part, part, part]
        }

        #[inline(always)]
        fn u16x32_load(self, data: &[u16; 32]) -> Self::U16x32 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            [
                u16x8::load(token, data[0..8].try_into().unwrap()),
                u16x8::load(token, data[8..16].try_into().unwrap()),
                u16x8::load(token, data[16..24].try_into().unwrap()),
                u16x8::load(token, data[24..32].try_into().unwrap()),
            ]
        }

        // i16x8 constructors
        #[inline(always)]
        fn i16x8_splat(self, v: i16) -> Self::I16x8 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            i16x8::splat(token, v)
        }

        #[inline(always)]
        fn i16x8_zero(self) -> Self::I16x8 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            i16x8::zero(token)
        }

        #[inline(always)]
        fn i16x8_load(self, data: &[i16; 8]) -> Self::I16x8 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            i16x8::load(token, data)
        }

        // i16x16 constructors
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

        // i16x32 constructors (array-based polyfill)
        #[inline(always)]
        fn i16x32_splat(self, v: i16) -> Self::I16x32 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            let part = i16x8::splat(token, v);
            [part, part, part, part]
        }

        #[inline(always)]
        fn i16x32_zero(self) -> Self::I16x32 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            let part = i16x8::zero(token);
            [part, part, part, part]
        }

        #[inline(always)]
        fn i16x32_load(self, data: &[i16; 32]) -> Self::I16x32 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            [
                i16x8::load(token, data[0..8].try_into().unwrap()),
                i16x8::load(token, data[8..16].try_into().unwrap()),
                i16x8::load(token, data[16..24].try_into().unwrap()),
                i16x8::load(token, data[24..32].try_into().unwrap()),
            ]
        }

        // u32x4 constructors
        #[inline(always)]
        fn u32x4_splat(self, v: u32) -> Self::U32x4 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            u32x4::splat(token, v)
        }

        #[inline(always)]
        fn u32x4_zero(self) -> Self::U32x4 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            u32x4::zero(token)
        }

        #[inline(always)]
        fn u32x4_load(self, data: &[u32; 4]) -> Self::U32x4 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            u32x4::load(token, data)
        }

        // u32x8 constructors
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

        // u32x16 constructors (array-based polyfill)
        #[inline(always)]
        fn u32x16_splat(self, v: u32) -> Self::U32x16 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            let part = u32x4::splat(token, v);
            [part, part, part, part]
        }

        #[inline(always)]
        fn u32x16_zero(self) -> Self::U32x16 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            let part = u32x4::zero(token);
            [part, part, part, part]
        }

        #[inline(always)]
        fn u32x16_load(self, data: &[u32; 16]) -> Self::U32x16 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            [
                u32x4::load(token, data[0..4].try_into().unwrap()),
                u32x4::load(token, data[4..8].try_into().unwrap()),
                u32x4::load(token, data[8..12].try_into().unwrap()),
                u32x4::load(token, data[12..16].try_into().unwrap()),
            ]
        }

        // i64x2 constructors
        #[inline(always)]
        fn i64x2_splat(self, v: i64) -> Self::I64x2 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            i64x2::splat(token, v)
        }

        #[inline(always)]
        fn i64x2_zero(self) -> Self::I64x2 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            i64x2::zero(token)
        }

        #[inline(always)]
        fn i64x2_load(self, data: &[i64; 2]) -> Self::I64x2 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            i64x2::load(token, data)
        }

        // i64x4 constructors
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

        // i64x8 constructors (array-based polyfill)
        #[inline(always)]
        fn i64x8_splat(self, v: i64) -> Self::I64x8 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            let part = i64x2::splat(token, v);
            [part, part, part, part]
        }

        #[inline(always)]
        fn i64x8_zero(self) -> Self::I64x8 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            let part = i64x2::zero(token);
            [part, part, part, part]
        }

        #[inline(always)]
        fn i64x8_load(self, data: &[i64; 8]) -> Self::I64x8 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            [
                i64x2::load(token, data[0..2].try_into().unwrap()),
                i64x2::load(token, data[2..4].try_into().unwrap()),
                i64x2::load(token, data[4..6].try_into().unwrap()),
                i64x2::load(token, data[6..8].try_into().unwrap()),
            ]
        }

        // u64x2 constructors
        #[inline(always)]
        fn u64x2_splat(self, v: u64) -> Self::U64x2 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            u64x2::splat(token, v)
        }

        #[inline(always)]
        fn u64x2_zero(self) -> Self::U64x2 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            u64x2::zero(token)
        }

        #[inline(always)]
        fn u64x2_load(self, data: &[u64; 2]) -> Self::U64x2 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            u64x2::load(token, data)
        }

        // u64x4 constructors
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

        // u64x8 constructors (array-based polyfill)
        #[inline(always)]
        fn u64x8_splat(self, v: u64) -> Self::U64x8 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            let part = u64x2::splat(token, v);
            [part, part, part, part]
        }

        #[inline(always)]
        fn u64x8_zero(self) -> Self::U64x8 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            let part = u64x2::zero(token);
            [part, part, part, part]
        }

        #[inline(always)]
        fn u64x8_load(self, data: &[u64; 8]) -> Self::U64x8 {
            let token = unsafe { Sse41Token::forge_token_dangerously() };
            [
                u64x2::load(token, data[0..2].try_into().unwrap()),
                u64x2::load(token, data[2..4].try_into().unwrap()),
                u64x2::load(token, data[4..6].try_into().unwrap()),
                u64x2::load(token, data[6..8].try_into().unwrap()),
            ]
        }
    }
}

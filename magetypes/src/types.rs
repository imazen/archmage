//! SimdTypes trait - associates SIMD types with tokens.
//!
//! This trait allows generic programming over token types by providing
//! associated types for each SIMD element type.

use archmage::SimdToken;

/// Trait that associates SIMD vector types with a capability token.
///
/// This enables generic programming where the vector width is determined
/// by the token type.
///
/// # Example
///
/// ```rust,ignore
/// use archmage::{X64V3Token, SimdToken};
/// use magetypes::SimdTypes;
///
/// fn sum<T: SimdToken + SimdTypes>(token: T, data: &[f32]) -> f32 {
///     // Use token's associated F32 type
///     let zero = <T as SimdTypes>::F32::splat(token, 0.0);
///     // ...
/// }
/// ```
pub trait SimdTypes: SimdToken {
    /// 32-bit floating point vector (e.g., f32x8 for AVX2)
    type F32;
    /// 64-bit floating point vector (e.g., f64x4 for AVX2)
    type F64;
    /// 8-bit signed integer vector
    type I8;
    /// 16-bit signed integer vector
    type I16;
    /// 32-bit signed integer vector
    type I32;
    /// 64-bit signed integer vector
    type I64;
    /// 8-bit unsigned integer vector
    type U8;
    /// 16-bit unsigned integer vector
    type U16;
    /// 32-bit unsigned integer vector
    type U32;
    /// 64-bit unsigned integer vector
    type U64;

    /// Number of f32 lanes in the F32 type
    const F32_LANES: usize;
    /// Number of f64 lanes in the F64 type
    const F64_LANES: usize;
    /// Number of i32 lanes in the I32 type
    const I32_LANES: usize;
}

// =============================================================================
// x86_64 implementations
// =============================================================================

#[cfg(target_arch = "x86_64")]
impl SimdTypes for archmage::X64V3Token {
    type F32 = crate::simd::generic::f32x8<archmage::X64V3Token>;
    type F64 = crate::simd::generic::f64x4<archmage::X64V3Token>;
    type I8 = crate::simd::x86::w256::i8x32;
    type I16 = crate::simd::x86::w256::i16x16;
    type I32 = crate::simd::generic::i32x8<archmage::X64V3Token>;
    type I64 = crate::simd::generic::i64x4<archmage::X64V3Token>;
    type U8 = crate::simd::x86::w256::u8x32;
    type U16 = crate::simd::x86::w256::u16x16;
    type U32 = crate::simd::generic::u32x8<archmage::X64V3Token>;
    type U64 = crate::simd::x86::w256::u64x4;

    const F32_LANES: usize = 8;
    const F64_LANES: usize = 4;
    const I32_LANES: usize = 8;
}

#[cfg(target_arch = "x86_64")]
impl SimdTypes for archmage::X64V2Token {
    type F32 = crate::simd::generic::f32x4<archmage::X64V3Token>;
    type F64 = crate::simd::generic::f64x2<archmage::X64V3Token>;
    type I8 = crate::simd::x86::w128::i8x16;
    type I16 = crate::simd::x86::w128::i16x8;
    type I32 = crate::simd::generic::i32x4<archmage::X64V3Token>;
    type I64 = crate::simd::generic::i64x2<archmage::X64V3Token>;
    type U8 = crate::simd::x86::w128::u8x16;
    type U16 = crate::simd::x86::w128::u16x8;
    type U32 = crate::simd::generic::u32x4<archmage::X64V3Token>;
    type U64 = crate::simd::x86::w128::u64x2;

    const F32_LANES: usize = 4;
    const F64_LANES: usize = 2;
    const I32_LANES: usize = 4;
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl SimdTypes for archmage::X64V4Token {
    type F32 = crate::simd::x86::w512::f32x16;
    type F64 = crate::simd::x86::w512::f64x8;
    type I8 = crate::simd::x86::w512::i8x64;
    type I16 = crate::simd::x86::w512::i16x32;
    type I32 = crate::simd::x86::w512::i32x16;
    type I64 = crate::simd::x86::w512::i64x8;
    type U8 = crate::simd::x86::w512::u8x64;
    type U16 = crate::simd::x86::w512::u16x32;
    type U32 = crate::simd::x86::w512::u32x16;
    type U64 = crate::simd::x86::w512::u64x8;

    const F32_LANES: usize = 16;
    const F64_LANES: usize = 8;
    const I32_LANES: usize = 16;
}

// =============================================================================
// aarch64 implementations
// =============================================================================

#[cfg(target_arch = "aarch64")]
impl SimdTypes for archmage::NeonToken {
    type F32 = crate::simd::arm::w128::f32x4;
    type F64 = crate::simd::arm::w128::f64x2;
    type I8 = crate::simd::arm::w128::i8x16;
    type I16 = crate::simd::arm::w128::i16x8;
    type I32 = crate::simd::arm::w128::i32x4;
    type I64 = crate::simd::arm::w128::i64x2;
    type U8 = crate::simd::arm::w128::u8x16;
    type U16 = crate::simd::arm::w128::u16x8;
    type U32 = crate::simd::arm::w128::u32x4;
    type U64 = crate::simd::arm::w128::u64x2;

    const F32_LANES: usize = 4;
    const F64_LANES: usize = 2;
    const I32_LANES: usize = 4;
}

// =============================================================================
// wasm32 implementations
// =============================================================================

#[cfg(target_arch = "wasm32")]
impl SimdTypes for archmage::Wasm128Token {
    type F32 = crate::simd::wasm::w128::f32x4;
    type F64 = crate::simd::wasm::w128::f64x2;
    type I8 = crate::simd::wasm::w128::i8x16;
    type I16 = crate::simd::wasm::w128::i16x8;
    type I32 = crate::simd::wasm::w128::i32x4;
    type I64 = crate::simd::wasm::w128::i64x2;
    type U8 = crate::simd::wasm::w128::u8x16;
    type U16 = crate::simd::wasm::w128::u16x8;
    type U32 = crate::simd::wasm::w128::u32x4;
    type U64 = crate::simd::wasm::w128::u64x2;

    const F32_LANES: usize = 4;
    const F64_LANES: usize = 2;
    const I32_LANES: usize = 4;
}

// =============================================================================
// ScalarToken implementation (no SIMD - uses scalar types)
// =============================================================================

impl SimdTypes for archmage::ScalarToken {
    type F32 = f32;
    type F64 = f64;
    type I8 = i8;
    type I16 = i16;
    type I32 = i32;
    type I64 = i64;
    type U8 = u8;
    type U16 = u16;
    type U32 = u32;
    type U64 = u64;

    const F32_LANES: usize = 1;
    const F64_LANES: usize = 1;
    const I32_LANES: usize = 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_token_types() {
        // ScalarToken associates with scalar types
        fn check<T: SimdTypes>() {
            assert_eq!(T::F32_LANES, 1);
            assert_eq!(T::F64_LANES, 1);
            assert_eq!(T::I32_LANES, 1);
        }
        check::<archmage::ScalarToken>();
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x64v3_token_types() {
        fn check<T: SimdTypes>() {
            assert_eq!(T::F32_LANES, 8);
            assert_eq!(T::F64_LANES, 4);
            assert_eq!(T::I32_LANES, 8);
        }
        check::<archmage::X64V3Token>();
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x64v2_token_types() {
        fn check<T: SimdTypes>() {
            assert_eq!(T::F32_LANES, 4);
            assert_eq!(T::F64_LANES, 2);
            assert_eq!(T::I32_LANES, 4);
        }
        check::<archmage::X64V2Token>();
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn x64v4_token_types() {
        fn check<T: SimdTypes>() {
            assert_eq!(T::F32_LANES, 16);
            assert_eq!(T::F64_LANES, 8);
            assert_eq!(T::I32_LANES, 16);
        }
        check::<archmage::X64V4Token>();
    }
}

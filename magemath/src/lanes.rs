//! Internal vector-lane traits uniting `ScalarF32`, `f32x4<T>`, `f32x8<T>`, and
//! `f32x16<T>` so transcendental graphs are written as ordinary generic Rust
//! functions instead of nested `macro_rules!`.

use archmage::{IntoConcreteToken, ScalarToken};
use core::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Sub};

/// Shared FMA arithmetic mode tags selected once at the operation boundary.
pub(crate) mod arith {
    pub(crate) const NATIVE: usize = 0;
    pub(crate) const SOFTWARE: usize = 1;
    #[cfg(all(feature = "std", not(miri)))]
    pub(crate) const STD: usize = 2;
    #[cfg(target_arch = "wasm32")]
    pub(crate) const WASM_SOFTWARE: usize = 3;
}

/// Companion `i32` lane vector for [`F32Lanes`].
pub(crate) trait I32Lanes:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitXor<Output = Self>
{
    type Token: IntoConcreteToken;
    type F32: F32Lanes<Token = Self::Token, I32 = Self>;

    fn splat(token: Self::Token, value: i32) -> Self;
    fn bitcast_to_f32(self) -> Self::F32;
    fn to_f32(self) -> Self::F32;
    fn max(self, rhs: Self) -> Self;
    fn blend(mask: Self, if_true: Self, if_false: Self) -> Self;
    fn any_true(self) -> bool;
    fn all_true(self) -> bool;
    fn shl_const<const N: i32>(self) -> Self;
    fn shr_arithmetic_const<const N: i32>(self) -> Self;
    fn shr_logical_const<const N: i32>(self) -> Self;
    fn simd_eq(self, rhs: Self) -> Self;
    fn simd_ne(self, rhs: Self) -> Self;
    fn simd_lt(self, rhs: Self) -> Self;
    fn simd_le(self, rhs: Self) -> Self;
    fn simd_gt(self, rhs: Self) -> Self;
    fn simd_ge(self, rhs: Self) -> Self;
    fn lane_mask(self) -> u16;
}

/// `f32` lane vector (1, 4, 8, or 16 lanes) carrying an Archmage capability token.
pub(crate) trait F32Lanes:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitXor<Output = Self>
{
    type Token: IntoConcreteToken;
    type I32: I32Lanes<Token = Self::Token, F32 = Self>;

    fn splat(token: Self::Token, value: f32) -> Self;
    fn bitcast_to_i32(self) -> Self::I32;
    fn to_i32(self) -> Self::I32;
    fn to_i32_round(self) -> Self::I32;
    fn floor(self) -> Self;
    fn round(self) -> Self;
    fn sqrt(self) -> Self;
    fn min_fast(self, rhs: Self) -> Self;
    fn max_fast(self, rhs: Self) -> Self;
    fn max(self, rhs: Self) -> Self;
    fn mul_add_relaxed(self, b: Self, c: Self) -> Self;
    fn mul_add_fused(self, b: Self, c: Self) -> Self;
    #[cfg(target_arch = "wasm32")]
    fn mul_add_wasm_soft(token: Self::Token, a: Self, b: Self, c: Self) -> Self;
    #[cfg(all(feature = "std", not(miri)))]
    fn mul_add_std(token: Self::Token, a: Self, b: Self, c: Self) -> Self;
    fn blend(mask: Self, if_true: Self, if_false: Self) -> Self;
    fn simd_eq(self, rhs: Self) -> Self;
    fn simd_ne(self, rhs: Self) -> Self;
    fn simd_lt(self, rhs: Self) -> Self;
    fn simd_le(self, rhs: Self) -> Self;
    fn simd_gt(self, rhs: Self) -> Self;
    fn simd_ge(self, rhs: Self) -> Self;
}

/// Evaluate `a * b + c` under the compile-time arithmetic mode `MODE`.
#[inline(always)]
pub(crate) fn madd<V: F32Lanes, const MODE: usize>(token: V::Token, a: V, b: V, c: V) -> V {
    let _ = token;
    if MODE == arith::NATIVE {
        a.mul_add_relaxed(b, c)
    } else if MODE == arith::SOFTWARE {
        a.mul_add_fused(b, c)
    } else {
        #[cfg(target_arch = "wasm32")]
        if MODE == arith::WASM_SOFTWARE {
            return V::mul_add_wasm_soft(token, a, b, c);
        }
        #[cfg(all(feature = "std", not(miri)))]
        if MODE == arith::STD {
            return V::mul_add_std(token, a, b, c);
        }
        unreachable!("invalid magemath arithmetic mode")
    }
}

#[inline(always)]
fn mask_from_array<const N: usize>(lanes: [i32; N]) -> u16 {
    let mut mask = 0_u16;
    let mut i = 0;
    while i < N {
        mask |= u16::from(lanes[i] != 0) << i;
        i += 1;
    }
    mask
}

impl I32Lanes for crate::scalar1::ScalarI32 {
    type Token = ScalarToken;
    type F32 = crate::scalar1::ScalarF32;

    #[inline(always)]
    fn splat(token: ScalarToken, value: i32) -> Self {
        Self::splat_t(token, value)
    }
    #[inline(always)]
    fn bitcast_to_f32(self) -> Self::F32 {
        self.bitcast_to_f32()
    }
    #[inline(always)]
    fn to_f32(self) -> Self::F32 {
        self.to_f32()
    }
    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        self.max(rhs)
    }
    #[inline(always)]
    fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self::blend(mask, if_true, if_false)
    }
    #[inline(always)]
    fn any_true(self) -> bool {
        self.any_true()
    }
    #[inline(always)]
    fn all_true(self) -> bool {
        self.all_true()
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self) -> Self {
        self.shl_const::<N>()
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self) -> Self {
        self.shr_arithmetic_const::<N>()
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self) -> Self {
        self.shr_logical_const::<N>()
    }
    #[inline(always)]
    fn simd_eq(self, rhs: Self) -> Self {
        self.simd_eq(rhs)
    }
    #[inline(always)]
    fn simd_ne(self, rhs: Self) -> Self {
        self.simd_ne(rhs)
    }
    #[inline(always)]
    fn simd_lt(self, rhs: Self) -> Self {
        self.simd_lt(rhs)
    }
    #[inline(always)]
    fn simd_le(self, rhs: Self) -> Self {
        self.simd_le(rhs)
    }
    #[inline(always)]
    fn simd_gt(self, rhs: Self) -> Self {
        self.simd_gt(rhs)
    }
    #[inline(always)]
    fn simd_ge(self, rhs: Self) -> Self {
        self.simd_ge(rhs)
    }
    #[inline(always)]
    fn lane_mask(self) -> u16 {
        u16::from(self.to_array()[0] != 0)
    }
}

impl F32Lanes for crate::scalar1::ScalarF32 {
    type Token = ScalarToken;
    type I32 = crate::scalar1::ScalarI32;

    #[inline(always)]
    fn splat(token: ScalarToken, value: f32) -> Self {
        Self::splat_t(token, value)
    }
    #[inline(always)]
    fn bitcast_to_i32(self) -> Self::I32 {
        self.bitcast_to_i32()
    }
    #[inline(always)]
    fn to_i32(self) -> Self::I32 {
        self.to_i32()
    }
    #[inline(always)]
    fn to_i32_round(self) -> Self::I32 {
        self.to_i32_round()
    }
    #[inline(always)]
    fn floor(self) -> Self {
        self.floor()
    }
    #[inline(always)]
    fn round(self) -> Self {
        self.round()
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    #[inline(always)]
    fn min_fast(self, rhs: Self) -> Self {
        self.min_fast(rhs)
    }
    #[inline(always)]
    fn max_fast(self, rhs: Self) -> Self {
        self.max_fast(rhs)
    }
    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        self.max(rhs)
    }
    #[inline(always)]
    fn mul_add_relaxed(self, b: Self, c: Self) -> Self {
        self.mul_add_relaxed(b, c)
    }
    #[inline(always)]
    fn mul_add_fused(self, b: Self, c: Self) -> Self {
        self.mul_add_fused(b, c)
    }
    #[cfg(target_arch = "wasm32")]
    #[inline]
    fn mul_add_wasm_soft(_token: ScalarToken, _a: Self, _b: Self, _c: Self) -> Self {
        unreachable!("ScalarF32 never uses WASM_SOFTWARE")
    }
    #[cfg(all(feature = "std", not(miri)))]
    #[inline]
    fn mul_add_std(token: ScalarToken, a: Self, b: Self, c: Self) -> Self {
        Self::splat_t(token, a.0.mul_add(b.0, c.0))
    }
    #[inline(always)]
    fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self::blend(mask, if_true, if_false)
    }
    #[inline(always)]
    fn simd_eq(self, rhs: Self) -> Self {
        self.simd_eq(rhs)
    }
    #[inline(always)]
    fn simd_ne(self, rhs: Self) -> Self {
        self.simd_ne(rhs)
    }
    #[inline(always)]
    fn simd_lt(self, rhs: Self) -> Self {
        self.simd_lt(rhs)
    }
    #[inline(always)]
    fn simd_le(self, rhs: Self) -> Self {
        self.simd_le(rhs)
    }
    #[inline(always)]
    fn simd_gt(self, rhs: Self) -> Self {
        self.simd_gt(rhs)
    }
    #[inline(always)]
    fn simd_ge(self, rhs: Self) -> Self {
        self.simd_ge(rhs)
    }
}

macro_rules! impl_simd_lanes {
    ($float:ident, $int:ident, $bound:ident) => {
        impl<T: magetypes::simd::backends::$bound> I32Lanes for magetypes::simd::generic::$int<T> {
            type Token = T;
            type F32 = magetypes::simd::generic::$float<T>;

            #[inline(always)]
            fn splat(token: T, value: i32) -> Self {
                Self::splat_t(token, value)
            }
            #[inline(always)]
            fn bitcast_to_f32(self) -> Self::F32 {
                self.bitcast_to_f32()
            }
            #[inline(always)]
            fn to_f32(self) -> Self::F32 {
                self.to_f32()
            }
            #[inline(always)]
            fn max(self, rhs: Self) -> Self {
                self.max(rhs)
            }
            #[inline(always)]
            fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
                Self::blend(mask, if_true, if_false)
            }
            #[inline(always)]
            fn any_true(self) -> bool {
                self.any_true()
            }
            #[inline(always)]
            fn all_true(self) -> bool {
                self.all_true()
            }
            #[inline(always)]
            fn shl_const<const N: i32>(self) -> Self {
                self.shl_const::<N>()
            }
            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(self) -> Self {
                self.shr_arithmetic_const::<N>()
            }
            #[inline(always)]
            fn shr_logical_const<const N: i32>(self) -> Self {
                self.shr_logical_const::<N>()
            }
            #[inline(always)]
            fn simd_eq(self, rhs: Self) -> Self {
                self.simd_eq(rhs)
            }
            #[inline(always)]
            fn simd_ne(self, rhs: Self) -> Self {
                self.simd_ne(rhs)
            }
            #[inline(always)]
            fn simd_lt(self, rhs: Self) -> Self {
                self.simd_lt(rhs)
            }
            #[inline(always)]
            fn simd_le(self, rhs: Self) -> Self {
                self.simd_le(rhs)
            }
            #[inline(always)]
            fn simd_gt(self, rhs: Self) -> Self {
                self.simd_gt(rhs)
            }
            #[inline(always)]
            fn simd_ge(self, rhs: Self) -> Self {
                self.simd_ge(rhs)
            }
            #[inline(always)]
            fn lane_mask(self) -> u16 {
                mask_from_array(self.to_array())
            }
        }

        impl<T: magetypes::simd::backends::$bound> F32Lanes
            for magetypes::simd::generic::$float<T>
        {
            type Token = T;
            type I32 = magetypes::simd::generic::$int<T>;

            #[inline(always)]
            fn splat(token: T, value: f32) -> Self {
                Self::splat_t(token, value)
            }
            #[inline(always)]
            fn bitcast_to_i32(self) -> Self::I32 {
                self.bitcast_to_i32()
            }
            #[inline(always)]
            fn to_i32(self) -> Self::I32 {
                self.to_i32()
            }
            #[inline(always)]
            fn to_i32_round(self) -> Self::I32 {
                self.to_i32_round()
            }
            #[inline(always)]
            fn floor(self) -> Self {
                self.floor()
            }
            #[inline(always)]
            fn round(self) -> Self {
                self.round()
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                self.sqrt()
            }
            #[inline(always)]
            fn min_fast(self, rhs: Self) -> Self {
                self.min_fast(rhs)
            }
            #[inline(always)]
            fn max_fast(self, rhs: Self) -> Self {
                self.max_fast(rhs)
            }
            #[inline(always)]
            fn max(self, rhs: Self) -> Self {
                self.max(rhs)
            }
            #[inline(always)]
            fn mul_add_relaxed(self, b: Self, c: Self) -> Self {
                self.mul_add_relaxed(b, c)
            }
            #[inline(always)]
            fn mul_add_fused(self, b: Self, c: Self) -> Self {
                self.mul_add_fused(b, c)
            }
            #[cfg(target_arch = "wasm32")]
            #[inline]
            fn mul_add_wasm_soft(token: T, a: Self, b: Self, c: Self) -> Self {
                let (a, b, c) = (a.to_array(), b.to_array(), c.to_array());
                let proof = token
                    .try_wasm128()
                    .expect("private WASM mode requires its matching proof");
                Self::from_array_t(token, crate::wasm_fma::software_lanes(proof, a, b, c))
            }
            #[cfg(all(feature = "std", not(miri)))]
            #[inline]
            fn mul_add_std(token: T, a: Self, b: Self, c: Self) -> Self {
                let (a, b, c) = (a.to_array(), b.to_array(), c.to_array());
                Self::from_array_t(token, core::array::from_fn(|i| a[i].mul_add(b[i], c[i])))
            }
            #[inline(always)]
            fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
                Self::blend(mask, if_true, if_false)
            }
            #[inline(always)]
            fn simd_eq(self, rhs: Self) -> Self {
                self.simd_eq(rhs)
            }
            #[inline(always)]
            fn simd_ne(self, rhs: Self) -> Self {
                self.simd_ne(rhs)
            }
            #[inline(always)]
            fn simd_lt(self, rhs: Self) -> Self {
                self.simd_lt(rhs)
            }
            #[inline(always)]
            fn simd_le(self, rhs: Self) -> Self {
                self.simd_le(rhs)
            }
            #[inline(always)]
            fn simd_gt(self, rhs: Self) -> Self {
                self.simd_gt(rhs)
            }
            #[inline(always)]
            fn simd_ge(self, rhs: Self) -> Self {
                self.simd_ge(rhs)
            }
        }
    };
}

impl_simd_lanes!(f32x4, i32x4, F32x4Convert);
impl_simd_lanes!(f32x8, i32x8, F32x8Convert);
#[cfg(feature = "w512")]
impl_simd_lanes!(f32x16, i32x16, F32x16Convert);

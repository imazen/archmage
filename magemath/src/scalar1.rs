//! Single-lane operands for the transcendental graphs. The token parameter
//! lets these types use the same graph macros as Magetypes vectors.
use archmage::ScalarToken;
use core::{marker::PhantomData, ops};

#[derive(Clone, Copy)]
#[repr(transparent)]
pub(crate) struct ScalarF32<T = ScalarToken>(pub(crate) f32, PhantomData<T>);

#[derive(Clone, Copy)]
#[repr(transparent)]
pub(crate) struct ScalarI32<T = ScalarToken>(i32, PhantomData<T>);

macro_rules! comparisons {
    ($($name:ident $op:tt),+ $(,)?) => {
        $(
            #[inline(always)]
            pub(crate) fn $name(self, rhs: Self) -> Self {
                Self::from_mask(self.0 $op rhs.0)
            }
        )+
    };
}

impl ScalarF32 {
    #[inline(always)]
    pub(crate) fn splat_t(_: ScalarToken, value: f32) -> Self {
        Self(value, PhantomData)
    }

    #[inline(always)]
    pub(crate) fn from_array_t(token: ScalarToken, [value]: [f32; 1]) -> Self {
        Self::splat_t(token, value)
    }

    #[inline(always)]
    pub(crate) fn to_array(self) -> [f32; 1] {
        [self.0]
    }

    #[inline(always)]
    pub(crate) fn bitcast_to_i32(self) -> ScalarI32 {
        ScalarI32(self.0.to_bits() as i32, PhantomData)
    }

    #[inline(always)]
    pub(crate) fn to_i32(self) -> ScalarI32 {
        ScalarI32(self.0 as i32, PhantomData)
    }

    #[inline(always)]
    pub(crate) fn to_i32_round(self) -> ScalarI32 {
        self.round().to_i32()
    }

    #[inline(always)]
    pub(crate) fn floor(self) -> Self {
        Self(magetypes::scalar::floor_f32(self.0), PhantomData)
    }

    #[inline(always)]
    pub(crate) fn round(self) -> Self {
        Self(magetypes::scalar::round_f32(self.0), PhantomData)
    }

    #[inline(always)]
    pub(crate) fn sqrt(self) -> Self {
        Self(magetypes::simd::scalar::f32x1(self.0).sqrt().0, PhantomData)
    }

    #[inline(always)]
    pub(crate) fn min_fast(self, rhs: Self) -> Self {
        if self.0 < rhs.0 { self } else { rhs }
    }

    #[inline(always)]
    pub(crate) fn max_fast(self, rhs: Self) -> Self {
        if self.0 > rhs.0 { self } else { rhs }
    }

    #[inline(always)]
    pub(crate) fn max(self, rhs: Self) -> Self {
        Self(self.0.max(rhs.0), PhantomData)
    }

    #[inline(always)]
    pub(crate) fn mul_add_relaxed(self, b: Self, c: Self) -> Self {
        Self(self.0 * b.0 + c.0, PhantomData)
    }

    #[inline(always)]
    pub(crate) fn mul_add_fused(self, b: Self, c: Self) -> Self {
        Self(magetypes::nostd_math::fmaf(self.0, b.0, c.0), PhantomData)
    }

    #[inline(always)]
    pub(crate) fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        if mask.bitcast_to_i32().any_true() {
            if_true
        } else {
            if_false
        }
    }

    #[inline(always)]
    fn from_mask(value: bool) -> Self {
        ScalarI32::from_mask(value).bitcast_to_f32()
    }

    comparisons!(simd_eq ==, simd_ne !=, simd_lt <, simd_le <=, simd_gt >, simd_ge >=);
}

impl ScalarI32 {
    #[inline(always)]
    pub(crate) fn splat_t(_: ScalarToken, value: i32) -> Self {
        Self(value, PhantomData)
    }

    #[inline(always)]
    pub(crate) fn to_array(self) -> [i32; 1] {
        [self.0]
    }

    #[inline(always)]
    pub(crate) fn bitcast_to_f32(self) -> ScalarF32 {
        ScalarF32(f32::from_bits(self.0 as u32), PhantomData)
    }

    #[inline(always)]
    pub(crate) fn max(self, rhs: Self) -> Self {
        Self(self.0.max(rhs.0), PhantomData)
    }

    #[inline(always)]
    pub(crate) fn to_f32(self) -> ScalarF32 {
        ScalarF32(self.0 as f32, PhantomData)
    }

    #[inline(always)]
    pub(crate) fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        if mask.0 != 0 { if_true } else { if_false }
    }

    #[inline(always)]
    pub(crate) fn any_true(self) -> bool {
        self.0 < 0
    }

    #[inline(always)]
    pub(crate) fn all_true(self) -> bool {
        self.any_true()
    }

    #[inline(always)]
    pub(crate) fn shl_const<const N: i32>(self) -> Self {
        Self(self.0 << N, PhantomData)
    }

    #[inline(always)]
    pub(crate) fn shr_arithmetic_const<const N: i32>(self) -> Self {
        Self(self.0 >> N, PhantomData)
    }

    #[inline(always)]
    pub(crate) fn shr_logical_const<const N: i32>(self) -> Self {
        Self(((self.0 as u32) >> N) as i32, PhantomData)
    }

    #[inline(always)]
    fn from_mask(value: bool) -> Self {
        Self(-i32::from(value), PhantomData)
    }

    comparisons!(simd_eq ==, simd_ne !=, simd_lt <, simd_le <=, simd_gt >, simd_ge >=);
}

macro_rules! binary_ops {
    ($ty:ty, $($trait:ident::$method:ident => $operation:expr),+ $(,)?) => {
        $(
            impl ops::$trait for $ty {
                type Output = Self;

                #[inline(always)]
                fn $method(self, rhs: Self) -> Self {
                    Self(($operation)(self.0, rhs.0), PhantomData)
                }
            }
        )+
    };
}

binary_ops!(
    ScalarF32,
    Add::add => |a, b| a + b,
    Sub::sub => |a, b| a - b,
    Mul::mul => |a, b| a * b,
    Div::div => |a, b| a / b,
    BitAnd::bitand => |a: f32, b: f32| f32::from_bits(a.to_bits() & b.to_bits()),
    BitOr::bitor => |a: f32, b: f32| f32::from_bits(a.to_bits() | b.to_bits()),
    BitXor::bitxor => |a: f32, b: f32| f32::from_bits(a.to_bits() ^ b.to_bits()),
);

binary_ops!(
    ScalarI32,
    Add::add => i32::wrapping_add,
    Sub::sub => i32::wrapping_sub,
    Mul::mul => i32::wrapping_mul,
    BitAnd::bitand => |a, b| a & b,
    BitOr::bitor => |a, b| a | b,
    BitXor::bitxor => |a, b| a ^ b,
);

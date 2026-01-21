//! Token-gated SIMD types with natural operators
//!
//! Provides `wide`-like ergonomics with token-gated construction.
//! There is NO way to construct these types without proving CPU support.

#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(missing_docs)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Index, IndexMut, Mul, MulAssign, Neg, Not, Shl, Shr, Sub, SubAssign,
};

// ============================================================================
// Macros
// ============================================================================

macro_rules! impl_arithmetic_ops {
    ($t:ty, $add:ident, $sub:ident, $mul:ident, $div:ident) => {
        impl Add for $t {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                Self(unsafe { $add(self.0, rhs.0) })
            }
        }
        impl Sub for $t {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                Self(unsafe { $sub(self.0, rhs.0) })
            }
        }
        impl Mul for $t {
            type Output = Self;
            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                Self(unsafe { $mul(self.0, rhs.0) })
            }
        }
        impl Div for $t {
            type Output = Self;
            #[inline(always)]
            fn div(self, rhs: Self) -> Self {
                Self(unsafe { $div(self.0, rhs.0) })
            }
        }
    };
}

macro_rules! impl_int_arithmetic_ops {
    ($t:ty, $add:ident, $sub:ident) => {
        impl Add for $t {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                Self(unsafe { $add(self.0, rhs.0) })
            }
        }
        impl Sub for $t {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                Self(unsafe { $sub(self.0, rhs.0) })
            }
        }
    };
}

macro_rules! impl_bitwise_ops {
    ($t:ty, $and:ident, $or:ident, $xor:ident) => {
        impl BitAnd for $t {
            type Output = Self;
            #[inline(always)]
            fn bitand(self, rhs: Self) -> Self {
                Self(unsafe { $and(self.0, rhs.0) })
            }
        }
        impl BitOr for $t {
            type Output = Self;
            #[inline(always)]
            fn bitor(self, rhs: Self) -> Self {
                Self(unsafe { $or(self.0, rhs.0) })
            }
        }
        impl BitXor for $t {
            type Output = Self;
            #[inline(always)]
            fn bitxor(self, rhs: Self) -> Self {
                Self(unsafe { $xor(self.0, rhs.0) })
            }
        }
    };
}

macro_rules! impl_bitwise_assign_ops {
    ($t:ty) => {
        impl BitAndAssign for $t {
            #[inline(always)]
            fn bitand_assign(&mut self, rhs: Self) {
                *self = *self & rhs;
            }
        }
        impl BitOrAssign for $t {
            #[inline(always)]
            fn bitor_assign(&mut self, rhs: Self) {
                *self = *self | rhs;
            }
        }
        impl BitXorAssign for $t {
            #[inline(always)]
            fn bitxor_assign(&mut self, rhs: Self) {
                *self = *self ^ rhs;
            }
        }
    };
}

macro_rules! impl_scalar_ops {
    ($t:ty, $scalar:ty, $set1:ident) => {
        impl Add<$scalar> for $t {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: $scalar) -> Self {
                self + Self(unsafe { $set1(rhs) })
            }
        }
        impl Sub<$scalar> for $t {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: $scalar) -> Self {
                self - Self(unsafe { $set1(rhs) })
            }
        }
        impl Mul<$scalar> for $t {
            type Output = Self;
            #[inline(always)]
            fn mul(self, rhs: $scalar) -> Self {
                self * Self(unsafe { $set1(rhs) })
            }
        }
        impl Div<$scalar> for $t {
            type Output = Self;
            #[inline(always)]
            fn div(self, rhs: $scalar) -> Self {
                self / Self(unsafe { $set1(rhs) })
            }
        }
        impl Add<$t> for $scalar {
            type Output = $t;
            #[inline(always)]
            fn add(self, rhs: $t) -> $t {
                <$t>::from_raw(unsafe { $set1(self) }) + rhs
            }
        }
        impl Sub<$t> for $scalar {
            type Output = $t;
            #[inline(always)]
            fn sub(self, rhs: $t) -> $t {
                <$t>::from_raw(unsafe { $set1(self) }) - rhs
            }
        }
        impl Mul<$t> for $scalar {
            type Output = $t;
            #[inline(always)]
            fn mul(self, rhs: $t) -> $t {
                <$t>::from_raw(unsafe { $set1(self) }) * rhs
            }
        }
        impl Div<$t> for $scalar {
            type Output = $t;
            #[inline(always)]
            fn div(self, rhs: $t) -> $t {
                <$t>::from_raw(unsafe { $set1(self) }) / rhs
            }
        }
    };
}

macro_rules! impl_assign_ops {
    ($t:ty) => {
        impl AddAssign for $t {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }
        impl SubAssign for $t {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }
        impl MulAssign for $t {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }
        impl DivAssign for $t {
            #[inline(always)]
            fn div_assign(&mut self, rhs: Self) {
                *self = *self / rhs;
            }
        }
    };
}

macro_rules! impl_int_scalar_ops {
    ($t:ty, $scalar:ty, $set1:ident) => {
        impl Add<$scalar> for $t {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: $scalar) -> Self {
                self + Self(unsafe { $set1(rhs) })
            }
        }
        impl Sub<$scalar> for $t {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: $scalar) -> Self {
                self - Self(unsafe { $set1(rhs) })
            }
        }
        impl Mul<$scalar> for $t {
            type Output = Self;
            #[inline(always)]
            fn mul(self, rhs: $scalar) -> Self {
                self * Self(unsafe { $set1(rhs) })
            }
        }
        impl Add<$t> for $scalar {
            type Output = $t;
            #[inline(always)]
            fn add(self, rhs: $t) -> $t {
                <$t>::from_raw(unsafe { $set1(self) }) + rhs
            }
        }
        impl Sub<$t> for $scalar {
            type Output = $t;
            #[inline(always)]
            fn sub(self, rhs: $t) -> $t {
                <$t>::from_raw(unsafe { $set1(self) }) - rhs
            }
        }
        impl Mul<$t> for $scalar {
            type Output = $t;
            #[inline(always)]
            fn mul(self, rhs: $t) -> $t {
                <$t>::from_raw(unsafe { $set1(self) }) * rhs
            }
        }
    };
}

// ============================================================================
// Multi-arch dispatch macro (similar to wide's pick!)
// ============================================================================

/// Compile-time architecture dispatch macro.
/// Usage:
/// ```ignore
/// pick! {
///     if #[cfg(target_arch = "x86_64")] { /* x86 code */ }
///     else if #[cfg(target_arch = "aarch64")] { /* arm code */ }
///     else { /* fallback */ }
/// }
/// ```
#[allow(unused_macros)]
macro_rules! pick {
    // x86_64 only
    (if #[cfg(target_arch = "x86_64")] { $($x86:tt)* } else { $($fallback:tt)* }) => {
        #[cfg(target_arch = "x86_64")]
        { $($x86)* }
        #[cfg(not(target_arch = "x86_64"))]
        { $($fallback)* }
    };
    // x86_64 then aarch64
    (if #[cfg(target_arch = "x86_64")] { $($x86:tt)* }
     else if #[cfg(target_arch = "aarch64")] { $($arm:tt)* }
     else { $($fallback:tt)* }) => {
        #[cfg(target_arch = "x86_64")]
        { $($x86)* }
        #[cfg(target_arch = "aarch64")]
        { $($arm)* }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        { $($fallback)* }
    };
    // aarch64 only
    (if #[cfg(target_arch = "aarch64")] { $($arm:tt)* } else { $($fallback:tt)* }) => {
        #[cfg(target_arch = "aarch64")]
        { $($arm)* }
        #[cfg(not(target_arch = "aarch64"))]
        { $($fallback)* }
    };
}

// ============================================================================
// Comparison traits
// ============================================================================

pub trait SimdEq<Rhs = Self> {
    type Output;
    fn simd_eq(self, rhs: Rhs) -> Self::Output;
}
pub trait SimdNe<Rhs = Self> {
    type Output;
    fn simd_ne(self, rhs: Rhs) -> Self::Output;
}
pub trait SimdLt<Rhs = Self> {
    type Output;
    fn simd_lt(self, rhs: Rhs) -> Self::Output;
}
pub trait SimdLe<Rhs = Self> {
    type Output;
    fn simd_le(self, rhs: Rhs) -> Self::Output;
}
pub trait SimdGt<Rhs = Self> {
    type Output;
    fn simd_gt(self, rhs: Rhs) -> Self::Output;
}
pub trait SimdGe<Rhs = Self> {
    type Output;
    fn simd_ge(self, rhs: Rhs) -> Self::Output;
}

// ============================================================================
// f32x8
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct f32x8(__m256);

/// Token-gated constants for f32x8. Cannot be constructed directly.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub struct F32x8Consts(()); // Private field prevents external construction

#[cfg(target_arch = "x86_64")]
impl F32x8Consts {
    #[inline(always)]
    pub fn one(self) -> f32x8 {
        f32x8(unsafe { _mm256_set1_ps(1.0) })
    }
    #[inline(always)]
    pub fn zero(self) -> f32x8 {
        f32x8(unsafe { _mm256_setzero_ps() })
    }
    #[inline(always)]
    pub fn half(self) -> f32x8 {
        f32x8(unsafe { _mm256_set1_ps(0.5) })
    }
    #[inline(always)]
    pub fn neg_one(self) -> f32x8 {
        f32x8(unsafe { _mm256_set1_ps(-1.0) })
    }
    #[inline(always)]
    pub fn pi(self) -> f32x8 {
        f32x8(unsafe { _mm256_set1_ps(core::f32::consts::PI) })
    }
    #[inline(always)]
    pub fn tau(self) -> f32x8 {
        f32x8(unsafe { _mm256_set1_ps(core::f32::consts::TAU) })
    }
    #[inline(always)]
    pub fn e(self) -> f32x8 {
        f32x8(unsafe { _mm256_set1_ps(core::f32::consts::E) })
    }
    #[inline(always)]
    pub fn frac_pi_2(self) -> f32x8 {
        f32x8(unsafe { _mm256_set1_ps(core::f32::consts::FRAC_PI_2) })
    }
    #[inline(always)]
    pub fn frac_pi_4(self) -> f32x8 {
        f32x8(unsafe { _mm256_set1_ps(core::f32::consts::FRAC_PI_4) })
    }
    #[inline(always)]
    pub fn sqrt_2(self) -> f32x8 {
        f32x8(unsafe { _mm256_set1_ps(core::f32::consts::SQRT_2) })
    }
    #[inline(always)]
    pub fn ln_2(self) -> f32x8 {
        f32x8(unsafe { _mm256_set1_ps(core::f32::consts::LN_2) })
    }
    #[inline(always)]
    pub fn ln_10(self) -> f32x8 {
        f32x8(unsafe { _mm256_set1_ps(core::f32::consts::LN_10) })
    }
    #[inline(always)]
    pub fn log2_e(self) -> f32x8 {
        f32x8(unsafe { _mm256_set1_ps(core::f32::consts::LOG2_E) })
    }
    #[inline(always)]
    pub fn log10_e(self) -> f32x8 {
        f32x8(unsafe { _mm256_set1_ps(core::f32::consts::LOG10_E) })
    }
}

#[cfg(target_arch = "x86_64")]
impl f32x8 {
    pub const LANES: usize = 8;

    /// Get token-gated constants. Usage: `f32x8::consts(token).one()`
    #[inline(always)]
    pub fn consts(_: impl crate::HasAvx2) -> F32x8Consts {
        F32x8Consts(())
    }

    // Construction (token-gated)
    #[inline(always)]
    pub fn load(_: impl crate::HasAvx2, data: &[f32; 8]) -> Self {
        Self(unsafe { _mm256_loadu_ps(data.as_ptr()) })
    }
    #[inline(always)]
    pub fn load_slice(_: impl crate::HasAvx2, data: &[f32]) -> Self {
        assert!(data.len() >= 8);
        Self(unsafe { _mm256_loadu_ps(data.as_ptr()) })
    }
    #[inline(always)]
    pub fn splat(_: impl crate::HasAvx2, v: f32) -> Self {
        Self(unsafe { _mm256_set1_ps(v) })
    }
    #[inline(always)]
    pub fn zero(_: impl crate::HasAvx2) -> Self {
        Self(unsafe { _mm256_setzero_ps() })
    }

    // Extraction
    #[inline(always)]
    pub fn to_array(self) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), self.0) };
        out
    }
    #[inline(always)]
    pub fn store(self, out: &mut [f32; 8]) {
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), self.0) };
    }
    #[inline(always)]
    pub fn store_slice(self, out: &mut [f32]) {
        assert!(out.len() >= 8);
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), self.0) };
    }
    #[inline(always)]
    pub fn as_array(&self) -> &[f32; 8] {
        unsafe { &*(self as *const Self as *const [f32; 8]) }
    }
    #[inline(always)]
    pub fn as_mut_array(&mut self) -> &mut [f32; 8] {
        unsafe { &mut *(self as *mut Self as *mut [f32; 8]) }
    }
    #[inline(always)]
    pub fn raw(self) -> __m256 {
        self.0
    }
    #[inline(always)]
    fn from_raw(v: __m256) -> Self {
        Self(v)
    }

    // Math
    #[inline(always)]
    pub fn min(self, o: Self) -> Self {
        Self(unsafe { _mm256_min_ps(self.0, o.0) })
    }
    #[inline(always)]
    pub fn max(self, o: Self) -> Self {
        Self(unsafe { _mm256_max_ps(self.0, o.0) })
    }
    #[inline(always)]
    pub fn fast_min(self, o: Self) -> Self {
        self.min(o)
    }
    #[inline(always)]
    pub fn fast_max(self, o: Self) -> Self {
        self.max(o)
    }
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { _mm256_and_ps(self.0, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF))) })
    }
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(unsafe { _mm256_sqrt_ps(self.0) })
    }
    #[inline(always)]
    pub fn recip(self) -> Self {
        Self(unsafe { _mm256_rcp_ps(self.0) })
    }
    #[inline(always)]
    pub fn recip_sqrt(self) -> Self {
        Self(unsafe { _mm256_rsqrt_ps(self.0) })
    }

    // FMA
    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        Self(unsafe { _mm256_fmadd_ps(self.0, b.0, c.0) })
    }
    #[inline(always)]
    pub fn mul_sub(self, b: Self, c: Self) -> Self {
        Self(unsafe { _mm256_fmsub_ps(self.0, b.0, c.0) })
    }
    #[inline(always)]
    pub fn neg_mul_add(self, b: Self, c: Self) -> Self {
        Self(unsafe { _mm256_fnmadd_ps(self.0, b.0, c.0) })
    }
    #[inline(always)]
    pub fn neg_mul_sub(self, b: Self, c: Self) -> Self {
        Self(unsafe { _mm256_fnmsub_ps(self.0, b.0, c.0) })
    }

    // Rounding
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(unsafe { _mm256_round_ps(self.0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) })
    }
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(unsafe { _mm256_floor_ps(self.0) })
    }
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self(unsafe { _mm256_ceil_ps(self.0) })
    }
    #[inline(always)]
    pub fn trunc(self) -> Self {
        Self(unsafe { _mm256_round_ps(self.0, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC) })
    }

    // Horizontal
    #[inline(always)]
    pub fn reduce_add(self) -> f32 {
        unsafe {
            let hi = _mm256_extractf128_ps(self.0, 1);
            let lo = _mm256_castps256_ps128(self.0);
            let sum = _mm_add_ps(lo, hi);
            let shuf = _mm_movehdup_ps(sum);
            let sums = _mm_add_ps(sum, shuf);
            let shuf = _mm_movehl_ps(sums, sums);
            _mm_cvtss_f32(_mm_add_ss(sums, shuf))
        }
    }

    // Predicates
    #[inline(always)]
    pub fn is_nan(self) -> Self {
        Self(unsafe { _mm256_cmp_ps(self.0, self.0, _CMP_UNORD_Q) })
    }
    #[inline(always)]
    pub fn is_finite(self) -> Self {
        let exp_mask = unsafe { _mm256_set1_epi32(0x7F80_0000) };
        let bits = unsafe { _mm256_castps_si256(self.0) };
        let exp = unsafe { _mm256_and_si256(bits, exp_mask) };
        let not_inf = unsafe { _mm256_cmpeq_epi32(exp, exp_mask) };
        Self(unsafe { _mm256_castsi256_ps(_mm256_xor_si256(not_inf, _mm256_set1_epi32(-1))) })
    }
    #[inline(always)]
    pub fn is_inf(self) -> Self {
        let inf_bits = unsafe { _mm256_set1_epi32(0x7F80_0000) };
        let abs_bits = unsafe {
            _mm256_and_si256(_mm256_castps_si256(self.0), _mm256_set1_epi32(0x7FFF_FFFF))
        };
        Self(unsafe { _mm256_castsi256_ps(_mm256_cmpeq_epi32(abs_bits, inf_bits)) })
    }

    // Sign
    #[inline(always)]
    pub fn sign_bit(self) -> Self {
        Self(unsafe { _mm256_and_ps(self.0, _mm256_set1_ps(-0.0)) })
    }
    #[inline(always)]
    pub fn flip_signs(self, signs: Self) -> Self {
        self ^ (signs & Self::from_raw(unsafe { _mm256_set1_ps(-0.0) }))
    }
    #[inline(always)]
    pub fn copysign(self, sign: Self) -> Self {
        let mag_mask = unsafe { _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF)) };
        let sign_mask = unsafe { _mm256_set1_ps(-0.0) };
        Self(unsafe {
            _mm256_or_ps(
                _mm256_and_ps(self.0, mag_mask),
                _mm256_and_ps(sign.0, sign_mask),
            )
        })
    }

    // Blend & mask
    #[inline(always)]
    pub fn blend(self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm256_blendv_ps(if_false.0, if_true.0, self.0) })
    }
    #[inline(always)]
    pub fn to_bitmask(self) -> u32 {
        unsafe { _mm256_movemask_ps(self.0) as u32 }
    }
    #[inline(always)]
    pub fn any(self) -> bool {
        self.to_bitmask() != 0
    }
    #[inline(always)]
    pub fn all(self) -> bool {
        self.to_bitmask() == 0xFF
    }
    #[inline(always)]
    pub fn none(self) -> bool {
        self.to_bitmask() == 0
    }

    // Int conversion
    #[inline(always)]
    pub fn round_int(self) -> i32x8 {
        i32x8(unsafe { _mm256_cvtps_epi32(self.0) })
    }
    #[inline(always)]
    pub fn trunc_int(self) -> i32x8 {
        i32x8(unsafe { _mm256_cvttps_epi32(self.0) })
    }
    #[inline(always)]
    pub fn from_i32x8(v: i32x8) -> Self {
        Self(unsafe { _mm256_cvtepi32_ps(v.0) })
    }

    /// Fast round to int - same as round_int, no additional checking
    /// For wide parity (wide's fast_ versions just alias the normal ones)
    #[inline(always)]
    pub fn fast_round_int(self) -> i32x8 {
        self.round_int()
    }
    /// Fast truncate to int - same as trunc_int, no additional checking
    #[inline(always)]
    pub fn fast_trunc_int(self) -> i32x8 {
        self.trunc_int()
    }
    /// Fast floor to int - floor then convert
    #[inline(always)]
    pub fn fast_floor_int(self) -> i32x8 {
        self.floor().trunc_int()
    }
    /// Fast ceil to int - ceil then convert
    #[inline(always)]
    pub fn fast_ceil_int(self) -> i32x8 {
        self.ceil().trunc_int()
    }

    // Degrees
    #[inline(always)]
    pub fn to_degrees(self) -> Self {
        self * (180.0 / core::f32::consts::PI)
    }
    #[inline(always)]
    pub fn to_radians(self) -> Self {
        self * (core::f32::consts::PI / 180.0)
    }

    /// Transpose an 8x8 matrix of f32 values.
    /// Takes 8 rows as input, returns 8 columns as output.
    /// Uses AVX2 shuffle and permute instructions for efficiency.
    #[inline(always)]
    pub fn transpose8x8(rows: [Self; 8]) -> [Self; 8] {
        unsafe {
            // Stage 1: interleave 32-bit floats within lanes
            let t0 = _mm256_unpacklo_ps(rows[0].0, rows[1].0);
            let t1 = _mm256_unpackhi_ps(rows[0].0, rows[1].0);
            let t2 = _mm256_unpacklo_ps(rows[2].0, rows[3].0);
            let t3 = _mm256_unpackhi_ps(rows[2].0, rows[3].0);
            let t4 = _mm256_unpacklo_ps(rows[4].0, rows[5].0);
            let t5 = _mm256_unpackhi_ps(rows[4].0, rows[5].0);
            let t6 = _mm256_unpacklo_ps(rows[6].0, rows[7].0);
            let t7 = _mm256_unpackhi_ps(rows[6].0, rows[7].0);

            // Stage 2: interleave 64-bit pairs
            let tt0 = _mm256_shuffle_ps(t0, t2, 0x44); // 01 00 01 00
            let tt1 = _mm256_shuffle_ps(t0, t2, 0xEE); // 11 10 11 10
            let tt2 = _mm256_shuffle_ps(t1, t3, 0x44);
            let tt3 = _mm256_shuffle_ps(t1, t3, 0xEE);
            let tt4 = _mm256_shuffle_ps(t4, t6, 0x44);
            let tt5 = _mm256_shuffle_ps(t4, t6, 0xEE);
            let tt6 = _mm256_shuffle_ps(t5, t7, 0x44);
            let tt7 = _mm256_shuffle_ps(t5, t7, 0xEE);

            // Stage 3: swap 128-bit lanes
            [
                Self(_mm256_permute2f128_ps(tt0, tt4, 0x20)),
                Self(_mm256_permute2f128_ps(tt1, tt5, 0x20)),
                Self(_mm256_permute2f128_ps(tt2, tt6, 0x20)),
                Self(_mm256_permute2f128_ps(tt3, tt7, 0x20)),
                Self(_mm256_permute2f128_ps(tt0, tt4, 0x31)),
                Self(_mm256_permute2f128_ps(tt1, tt5, 0x31)),
                Self(_mm256_permute2f128_ps(tt2, tt6, 0x31)),
                Self(_mm256_permute2f128_ps(tt3, tt7, 0x31)),
            ]
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl Neg for f32x8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(unsafe { _mm256_xor_ps(self.0, _mm256_set1_ps(-0.0)) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Not for f32x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(unsafe { _mm256_xor_ps(self.0, _mm256_castsi256_ps(_mm256_set1_epi32(-1))) })
    }
}

#[cfg(target_arch = "x86_64")]
impl_arithmetic_ops!(
    f32x8,
    _mm256_add_ps,
    _mm256_sub_ps,
    _mm256_mul_ps,
    _mm256_div_ps
);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(f32x8, _mm256_and_ps, _mm256_or_ps, _mm256_xor_ps);
#[cfg(target_arch = "x86_64")]
impl_bitwise_assign_ops!(f32x8);
#[cfg(target_arch = "x86_64")]
impl_scalar_ops!(f32x8, f32, _mm256_set1_ps);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(f32x8);

#[cfg(target_arch = "x86_64")]
impl SimdEq for f32x8 {
    type Output = Self;
    #[inline(always)]
    fn simd_eq(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_cmp_ps(self.0, rhs.0, _CMP_EQ_OQ) })
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdNe for f32x8 {
    type Output = Self;
    #[inline(always)]
    fn simd_ne(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_cmp_ps(self.0, rhs.0, _CMP_NEQ_OQ) })
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdLt for f32x8 {
    type Output = Self;
    #[inline(always)]
    fn simd_lt(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_cmp_ps(self.0, rhs.0, _CMP_LT_OQ) })
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdLe for f32x8 {
    type Output = Self;
    #[inline(always)]
    fn simd_le(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_cmp_ps(self.0, rhs.0, _CMP_LE_OQ) })
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdGt for f32x8 {
    type Output = Self;
    #[inline(always)]
    fn simd_gt(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_cmp_ps(self.0, rhs.0, _CMP_GT_OQ) })
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdGe for f32x8 {
    type Output = Self;
    #[inline(always)]
    fn simd_ge(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_cmp_ps(self.0, rhs.0, _CMP_GE_OQ) })
    }
}

#[cfg(target_arch = "x86_64")]
impl core::fmt::Debug for f32x8 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("f32x8").field(&self.to_array()).finish()
    }
}

#[cfg(target_arch = "x86_64")]
impl PartialEq for f32x8 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

#[cfg(target_arch = "x86_64")]
impl Index<usize> for f32x8 {
    type Output = f32;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_array()[index]
    }
}

#[cfg(target_arch = "x86_64")]
impl IndexMut<usize> for f32x8 {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_array()[index]
    }
}

// ============================================================================
// i32x8
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct i32x8(pub(crate) __m256i);

/// Token-gated constants for i32x8. Cannot be constructed directly.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub struct I32x8Consts(());

#[cfg(target_arch = "x86_64")]
impl I32x8Consts {
    #[inline(always)]
    pub fn zero(self) -> i32x8 {
        i32x8(unsafe { _mm256_setzero_si256() })
    }
    #[inline(always)]
    pub fn one(self) -> i32x8 {
        i32x8(unsafe { _mm256_set1_epi32(1) })
    }
    #[inline(always)]
    pub fn neg_one(self) -> i32x8 {
        i32x8(unsafe { _mm256_set1_epi32(-1) })
    }
    #[inline(always)]
    pub fn min_value(self) -> i32x8 {
        i32x8(unsafe { _mm256_set1_epi32(i32::MIN) })
    }
    #[inline(always)]
    pub fn max_value(self) -> i32x8 {
        i32x8(unsafe { _mm256_set1_epi32(i32::MAX) })
    }
}

#[cfg(target_arch = "x86_64")]
impl i32x8 {
    pub const LANES: usize = 8;

    /// Get token-gated constants. Usage: `i32x8::consts(token).one()`
    #[inline(always)]
    pub fn consts(_: impl crate::HasAvx2) -> I32x8Consts {
        I32x8Consts(())
    }

    #[inline(always)]
    pub fn load(_: impl crate::HasAvx2, data: &[i32; 8]) -> Self {
        Self(unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) })
    }
    #[inline(always)]
    pub fn splat(_: impl crate::HasAvx2, v: i32) -> Self {
        Self(unsafe { _mm256_set1_epi32(v) })
    }
    #[inline(always)]
    pub fn zero(_: impl crate::HasAvx2) -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i32; 8] {
        let mut out = [0i32; 8];
        unsafe { _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, self.0) };
        out
    }
    #[inline(always)]
    pub fn as_array(&self) -> &[i32; 8] {
        unsafe { &*(self as *const Self as *const [i32; 8]) }
    }
    #[inline(always)]
    pub fn as_mut_array(&mut self) -> &mut [i32; 8] {
        unsafe { &mut *(self as *mut Self as *mut [i32; 8]) }
    }
    #[inline(always)]
    pub fn raw(self) -> __m256i {
        self.0
    }
    #[inline(always)]
    fn from_raw(v: __m256i) -> Self {
        Self(v)
    }

    #[inline(always)]
    pub fn min(self, o: Self) -> Self {
        Self(unsafe { _mm256_min_epi32(self.0, o.0) })
    }
    #[inline(always)]
    pub fn max(self, o: Self) -> Self {
        Self(unsafe { _mm256_max_epi32(self.0, o.0) })
    }
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { _mm256_abs_epi32(self.0) })
    }

    #[inline(always)]
    pub fn to_f32x8(self) -> f32x8 {
        f32x8::from_i32x8(self)
    }

    /// Widen 8 i16s to 8 i32s with sign extension (AVX2)
    #[inline(always)]
    pub fn from_i16x8(v: i16x8) -> Self {
        Self(unsafe { _mm256_cvtepi16_epi32(v.0) })
    }

    /// Widen 8 u16s to 8 i32s with zero extension (AVX2)
    #[inline(always)]
    pub fn from_u16x8(v: u16x8) -> Self {
        Self(unsafe { _mm256_cvtepu16_epi32(v.0) })
    }

    /// Convert mask to f32x8 (wide parity: round_float)
    #[inline(always)]
    pub fn round_float(self) -> f32x8 {
        f32x8(unsafe { _mm256_castsi256_ps(self.0) })
    }

    #[inline(always)]
    pub fn to_bitmask(self) -> u32 {
        unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(self.0)) as u32 }
    }
    #[inline(always)]
    pub fn any(self) -> bool {
        self.to_bitmask() != 0
    }
    #[inline(always)]
    pub fn all(self) -> bool {
        self.to_bitmask() == 0xFF
    }
    #[inline(always)]
    pub fn none(self) -> bool {
        self.to_bitmask() == 0
    }

    #[inline(always)]
    pub fn blend(self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm256_blendv_epi8(if_false.0, if_true.0, self.0) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Neg for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(unsafe { _mm256_sub_epi32(_mm256_setzero_si256(), self.0) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Not for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(unsafe { _mm256_xor_si256(self.0, _mm256_set1_epi32(-1)) })
    }
}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(i32x8, _mm256_add_epi32, _mm256_sub_epi32);
#[cfg(target_arch = "x86_64")]
impl Mul for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_mullo_epi32(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(i32x8, _mm256_and_si256, _mm256_or_si256, _mm256_xor_si256);
#[cfg(target_arch = "x86_64")]
impl_bitwise_assign_ops!(i32x8);

#[cfg(target_arch = "x86_64")]
impl Shl<i32> for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: i32) -> Self {
        Self(unsafe { _mm256_sllv_epi32(self.0, _mm256_set1_epi32(rhs)) })
    }
}
#[cfg(target_arch = "x86_64")]
impl Shr<i32> for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: i32) -> Self {
        Self(unsafe { _mm256_srav_epi32(self.0, _mm256_set1_epi32(rhs)) })
    }
}
#[cfg(target_arch = "x86_64")]
impl Shl<u32> for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: u32) -> Self {
        self << (rhs as i32)
    }
}
#[cfg(target_arch = "x86_64")]
impl Shr<u32> for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: u32) -> Self {
        self >> (rhs as i32)
    }
}
#[cfg(target_arch = "x86_64")]
impl Shl<usize> for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: usize) -> Self {
        self << (rhs as i32)
    }
}
#[cfg(target_arch = "x86_64")]
impl Shr<usize> for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: usize) -> Self {
        self >> (rhs as i32)
    }
}

// Per-lane shifts (wide parity)
#[cfg(target_arch = "x86_64")]
impl Shl<i32x8> for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: i32x8) -> Self {
        // Mask to ensure same behavior as wrapping_shl
        let shift = unsafe { _mm256_and_si256(rhs.0, _mm256_set1_epi32(31)) };
        Self(unsafe { _mm256_sllv_epi32(self.0, shift) })
    }
}
#[cfg(target_arch = "x86_64")]
impl Shr<i32x8> for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: i32x8) -> Self {
        let shift = unsafe { _mm256_and_si256(rhs.0, _mm256_set1_epi32(31)) };
        Self(unsafe { _mm256_srav_epi32(self.0, shift) })
    }
}

// Scalar ops (wide parity)
#[cfg(target_arch = "x86_64")]
impl_int_scalar_ops!(i32x8, i32, _mm256_set1_epi32);

#[cfg(target_arch = "x86_64")]
impl SimdEq for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn simd_eq(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_cmpeq_epi32(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdGt for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn simd_gt(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_cmpgt_epi32(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdLt for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn simd_lt(self, rhs: Self) -> Self {
        rhs.simd_gt(self)
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdGe for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn simd_ge(self, rhs: Self) -> Self {
        !rhs.simd_gt(self)
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdLe for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn simd_le(self, rhs: Self) -> Self {
        !self.simd_gt(rhs)
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdNe for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn simd_ne(self, rhs: Self) -> Self {
        !self.simd_eq(rhs)
    }
}

#[cfg(target_arch = "x86_64")]
impl core::fmt::Debug for i32x8 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("i32x8").field(&self.to_array()).finish()
    }
}

#[cfg(target_arch = "x86_64")]
impl PartialEq for i32x8 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

#[cfg(target_arch = "x86_64")]
impl Index<usize> for i32x8 {
    type Output = i32;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_array()[index]
    }
}

#[cfg(target_arch = "x86_64")]
impl IndexMut<usize> for i32x8 {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_array()[index]
    }
}

// ============================================================================
// u32x8
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct u32x8(pub(crate) __m256i);

/// Token-gated constants for u32x8. Cannot be constructed directly.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub struct U32x8Consts(());

#[cfg(target_arch = "x86_64")]
impl U32x8Consts {
    #[inline(always)]
    pub fn zero(self) -> u32x8 {
        u32x8(unsafe { _mm256_setzero_si256() })
    }
    #[inline(always)]
    pub fn one(self) -> u32x8 {
        u32x8(unsafe { _mm256_set1_epi32(1) })
    }
    #[inline(always)]
    pub fn max_value(self) -> u32x8 {
        u32x8(unsafe { _mm256_set1_epi32(-1) })
    } // -1 as i32 = u32::MAX
}

#[cfg(target_arch = "x86_64")]
impl u32x8 {
    pub const LANES: usize = 8;

    /// Get token-gated constants. Usage: `u32x8::consts(token).one()`
    #[inline(always)]
    pub fn consts(_: impl crate::HasAvx2) -> U32x8Consts {
        U32x8Consts(())
    }

    #[inline(always)]
    pub fn load(_: impl crate::HasAvx2, data: &[u32; 8]) -> Self {
        Self(unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) })
    }
    #[inline(always)]
    pub fn splat(_: impl crate::HasAvx2, v: u32) -> Self {
        Self(unsafe { _mm256_set1_epi32(v as i32) })
    }
    #[inline(always)]
    pub fn zero(_: impl crate::HasAvx2) -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }

    #[inline(always)]
    pub fn to_array(self) -> [u32; 8] {
        let mut out = [0u32; 8];
        unsafe { _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, self.0) };
        out
    }
    #[inline(always)]
    pub fn as_array(&self) -> &[u32; 8] {
        unsafe { &*(self as *const Self as *const [u32; 8]) }
    }
    #[inline(always)]
    pub fn as_mut_array(&mut self) -> &mut [u32; 8] {
        unsafe { &mut *(self as *mut Self as *mut [u32; 8]) }
    }
    #[inline(always)]
    fn from_raw(v: __m256i) -> Self {
        Self(v)
    }

    #[inline(always)]
    pub fn min(self, o: Self) -> Self {
        Self(unsafe { _mm256_min_epu32(self.0, o.0) })
    }
    #[inline(always)]
    pub fn max(self, o: Self) -> Self {
        Self(unsafe { _mm256_max_epu32(self.0, o.0) })
    }

    /// Widen 8 u16s to 8 u32s with zero extension (AVX2)
    #[inline(always)]
    pub fn from_u16x8(v: u16x8) -> Self {
        Self(unsafe { _mm256_cvtepu16_epi32(v.0) })
    }

    #[inline(always)]
    pub fn to_bitmask(self) -> u32 {
        unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(self.0)) as u32 }
    }
    #[inline(always)]
    pub fn any(self) -> bool {
        self.to_bitmask() != 0
    }
    #[inline(always)]
    pub fn all(self) -> bool {
        self.to_bitmask() == 0xFF
    }
    #[inline(always)]
    pub fn none(self) -> bool {
        self.to_bitmask() == 0
    }

    #[inline(always)]
    pub fn blend(self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm256_blendv_epi8(if_false.0, if_true.0, self.0) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Not for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(unsafe { _mm256_xor_si256(self.0, _mm256_set1_epi32(-1)) })
    }
}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(u32x8, _mm256_add_epi32, _mm256_sub_epi32);
#[cfg(target_arch = "x86_64")]
impl Mul for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_mullo_epi32(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(u32x8, _mm256_and_si256, _mm256_or_si256, _mm256_xor_si256);
#[cfg(target_arch = "x86_64")]
impl_bitwise_assign_ops!(u32x8);

#[cfg(target_arch = "x86_64")]
impl Shl<i32> for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: i32) -> Self {
        Self(unsafe { _mm256_sllv_epi32(self.0, _mm256_set1_epi32(rhs)) })
    }
}
#[cfg(target_arch = "x86_64")]
impl Shr<i32> for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: i32) -> Self {
        Self(unsafe { _mm256_srlv_epi32(self.0, _mm256_set1_epi32(rhs)) })
    }
}
#[cfg(target_arch = "x86_64")]
impl Shl<u32> for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: u32) -> Self {
        self << (rhs as i32)
    }
}
#[cfg(target_arch = "x86_64")]
impl Shr<u32> for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: u32) -> Self {
        self >> (rhs as i32)
    }
}
#[cfg(target_arch = "x86_64")]
impl Shl<usize> for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: usize) -> Self {
        self << (rhs as i32)
    }
}
#[cfg(target_arch = "x86_64")]
impl Shr<usize> for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: usize) -> Self {
        self >> (rhs as i32)
    }
}

// Per-lane shifts
#[cfg(target_arch = "x86_64")]
impl Shl<u32x8> for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: u32x8) -> Self {
        let shift = unsafe { _mm256_and_si256(rhs.0, _mm256_set1_epi32(31)) };
        Self(unsafe { _mm256_sllv_epi32(self.0, shift) })
    }
}
#[cfg(target_arch = "x86_64")]
impl Shr<u32x8> for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: u32x8) -> Self {
        let shift = unsafe { _mm256_and_si256(rhs.0, _mm256_set1_epi32(31)) };
        Self(unsafe { _mm256_srlv_epi32(self.0, shift) })
    }
}

// Scalar ops for u32x8
#[cfg(target_arch = "x86_64")]
impl Add<u32> for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: u32) -> Self {
        self + Self(unsafe { _mm256_set1_epi32(rhs as i32) })
    }
}
#[cfg(target_arch = "x86_64")]
impl Sub<u32> for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: u32) -> Self {
        self - Self(unsafe { _mm256_set1_epi32(rhs as i32) })
    }
}
#[cfg(target_arch = "x86_64")]
impl Mul<u32> for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: u32) -> Self {
        self * Self(unsafe { _mm256_set1_epi32(rhs as i32) })
    }
}
#[cfg(target_arch = "x86_64")]
impl Add<u32x8> for u32 {
    type Output = u32x8;
    #[inline(always)]
    fn add(self, rhs: u32x8) -> u32x8 {
        u32x8(unsafe { _mm256_set1_epi32(self as i32) }) + rhs
    }
}
#[cfg(target_arch = "x86_64")]
impl Sub<u32x8> for u32 {
    type Output = u32x8;
    #[inline(always)]
    fn sub(self, rhs: u32x8) -> u32x8 {
        u32x8(unsafe { _mm256_set1_epi32(self as i32) }) - rhs
    }
}
#[cfg(target_arch = "x86_64")]
impl Mul<u32x8> for u32 {
    type Output = u32x8;
    #[inline(always)]
    fn mul(self, rhs: u32x8) -> u32x8 {
        u32x8(unsafe { _mm256_set1_epi32(self as i32) }) * rhs
    }
}

// Comparison traits for u32x8
#[cfg(target_arch = "x86_64")]
impl SimdEq for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn simd_eq(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_cmpeq_epi32(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdNe for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn simd_ne(self, rhs: Self) -> Self {
        !self.simd_eq(rhs)
    }
}
// Unsigned comparison: flip sign bits to convert to signed comparison
#[cfg(target_arch = "x86_64")]
impl SimdGt for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn simd_gt(self, rhs: Self) -> Self {
        unsafe {
            let flip = _mm256_set1_epi32(i32::MIN);
            let a = _mm256_xor_si256(self.0, flip);
            let b = _mm256_xor_si256(rhs.0, flip);
            Self(_mm256_cmpgt_epi32(a, b))
        }
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdLt for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn simd_lt(self, rhs: Self) -> Self {
        rhs.simd_gt(self)
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdGe for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn simd_ge(self, rhs: Self) -> Self {
        !rhs.simd_gt(self)
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdLe for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn simd_le(self, rhs: Self) -> Self {
        !self.simd_gt(rhs)
    }
}

#[cfg(target_arch = "x86_64")]
impl core::fmt::Debug for u32x8 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("u32x8").field(&self.to_array()).finish()
    }
}

#[cfg(target_arch = "x86_64")]
impl PartialEq for u32x8 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

#[cfg(target_arch = "x86_64")]
impl Index<usize> for u32x8 {
    type Output = u32;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_array()[index]
    }
}

#[cfg(target_arch = "x86_64")]
impl IndexMut<usize> for u32x8 {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_array()[index]
    }
}

// ============================================================================
// f32x4
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct f32x4(__m128);

/// Token-gated constants for f32x4. Cannot be constructed directly.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub struct F32x4Consts(());

#[cfg(target_arch = "x86_64")]
impl F32x4Consts {
    #[inline(always)]
    pub fn one(self) -> f32x4 {
        f32x4(unsafe { _mm_set1_ps(1.0) })
    }
    #[inline(always)]
    pub fn zero(self) -> f32x4 {
        f32x4(unsafe { _mm_setzero_ps() })
    }
    #[inline(always)]
    pub fn half(self) -> f32x4 {
        f32x4(unsafe { _mm_set1_ps(0.5) })
    }
    #[inline(always)]
    pub fn neg_one(self) -> f32x4 {
        f32x4(unsafe { _mm_set1_ps(-1.0) })
    }
    #[inline(always)]
    pub fn pi(self) -> f32x4 {
        f32x4(unsafe { _mm_set1_ps(core::f32::consts::PI) })
    }
    #[inline(always)]
    pub fn tau(self) -> f32x4 {
        f32x4(unsafe { _mm_set1_ps(core::f32::consts::TAU) })
    }
    #[inline(always)]
    pub fn e(self) -> f32x4 {
        f32x4(unsafe { _mm_set1_ps(core::f32::consts::E) })
    }
}

#[cfg(target_arch = "x86_64")]
impl f32x4 {
    pub const LANES: usize = 4;

    /// Get token-gated constants. Usage: `f32x4::consts(token).one()`
    #[inline(always)]
    pub fn consts(_: impl crate::HasSse2) -> F32x4Consts {
        F32x4Consts(())
    }

    #[inline(always)]
    pub fn load(_: impl crate::HasSse2, data: &[f32; 4]) -> Self {
        Self(unsafe { _mm_loadu_ps(data.as_ptr()) })
    }
    #[inline(always)]
    pub fn splat(_: impl crate::HasSse2, v: f32) -> Self {
        Self(unsafe { _mm_set1_ps(v) })
    }
    #[inline(always)]
    pub fn zero(_: impl crate::HasSse2) -> Self {
        Self(unsafe { _mm_setzero_ps() })
    }

    #[inline(always)]
    pub fn to_array(self) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        unsafe { _mm_storeu_ps(out.as_mut_ptr(), self.0) };
        out
    }
    #[inline(always)]
    pub fn as_array(&self) -> &[f32; 4] {
        unsafe { &*(self as *const Self as *const [f32; 4]) }
    }
    #[inline(always)]
    pub fn as_mut_array(&mut self) -> &mut [f32; 4] {
        unsafe { &mut *(self as *mut Self as *mut [f32; 4]) }
    }
    #[inline(always)]
    fn from_raw(v: __m128) -> Self {
        Self(v)
    }

    #[inline(always)]
    pub fn min(self, o: Self) -> Self {
        Self(unsafe { _mm_min_ps(self.0, o.0) })
    }
    #[inline(always)]
    pub fn max(self, o: Self) -> Self {
        Self(unsafe { _mm_max_ps(self.0, o.0) })
    }
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { _mm_and_ps(self.0, _mm_castsi128_ps(_mm_set1_epi32(0x7FFF_FFFF))) })
    }
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(unsafe { _mm_sqrt_ps(self.0) })
    }
    #[inline(always)]
    pub fn recip(self) -> Self {
        Self(unsafe { _mm_rcp_ps(self.0) })
    }
    #[inline(always)]
    pub fn recip_sqrt(self) -> Self {
        Self(unsafe { _mm_rsqrt_ps(self.0) })
    }

    #[inline(always)]
    pub fn round(self) -> Self {
        Self(unsafe { _mm_round_ps(self.0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) })
    }
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(unsafe { _mm_floor_ps(self.0) })
    }
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self(unsafe { _mm_ceil_ps(self.0) })
    }
    #[inline(always)]
    pub fn trunc(self) -> Self {
        Self(unsafe { _mm_round_ps(self.0, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC) })
    }

    #[inline(always)]
    pub fn reduce_add(self) -> f32 {
        unsafe {
            let shuf = _mm_movehdup_ps(self.0);
            let sums = _mm_add_ps(self.0, shuf);
            let shuf = _mm_movehl_ps(sums, sums);
            _mm_cvtss_f32(_mm_add_ss(sums, shuf))
        }
    }

    #[inline(always)]
    pub fn blend(self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm_blendv_ps(if_false.0, if_true.0, self.0) })
    }
    #[inline(always)]
    pub fn to_bitmask(self) -> u32 {
        unsafe { _mm_movemask_ps(self.0) as u32 }
    }
    #[inline(always)]
    pub fn any(self) -> bool {
        self.to_bitmask() != 0
    }
    #[inline(always)]
    pub fn all(self) -> bool {
        self.to_bitmask() == 0xF
    }
    #[inline(always)]
    pub fn none(self) -> bool {
        self.to_bitmask() == 0
    }

    #[inline(always)]
    pub fn round_int(self) -> i32x4 {
        i32x4(unsafe { _mm_cvtps_epi32(self.0) })
    }
    #[inline(always)]
    pub fn trunc_int(self) -> i32x4 {
        i32x4(unsafe { _mm_cvttps_epi32(self.0) })
    }
    #[inline(always)]
    pub fn from_i32x4(v: i32x4) -> Self {
        Self(unsafe { _mm_cvtepi32_ps(v.0) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Neg for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(unsafe { _mm_xor_ps(self.0, _mm_set1_ps(-0.0)) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Not for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(unsafe { _mm_xor_ps(self.0, _mm_castsi128_ps(_mm_set1_epi32(-1))) })
    }
}

#[cfg(target_arch = "x86_64")]
impl_arithmetic_ops!(f32x4, _mm_add_ps, _mm_sub_ps, _mm_mul_ps, _mm_div_ps);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(f32x4, _mm_and_ps, _mm_or_ps, _mm_xor_ps);
#[cfg(target_arch = "x86_64")]
impl_bitwise_assign_ops!(f32x4);
#[cfg(target_arch = "x86_64")]
impl_scalar_ops!(f32x4, f32, _mm_set1_ps);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(f32x4);

#[cfg(target_arch = "x86_64")]
impl core::fmt::Debug for f32x4 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("f32x4").field(&self.to_array()).finish()
    }
}

#[cfg(target_arch = "x86_64")]
impl PartialEq for f32x4 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

#[cfg(target_arch = "x86_64")]
impl Index<usize> for f32x4 {
    type Output = f32;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_array()[index]
    }
}

#[cfg(target_arch = "x86_64")]
impl IndexMut<usize> for f32x4 {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_array()[index]
    }
}

#[cfg(target_arch = "x86_64")]
impl SimdEq for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn simd_eq(self, rhs: Self) -> Self {
        Self(unsafe { _mm_cmpeq_ps(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdNe for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn simd_ne(self, rhs: Self) -> Self {
        Self(unsafe { _mm_cmpneq_ps(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdLt for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn simd_lt(self, rhs: Self) -> Self {
        Self(unsafe { _mm_cmplt_ps(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdLe for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn simd_le(self, rhs: Self) -> Self {
        Self(unsafe { _mm_cmple_ps(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdGt for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn simd_gt(self, rhs: Self) -> Self {
        Self(unsafe { _mm_cmpgt_ps(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdGe for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn simd_ge(self, rhs: Self) -> Self {
        Self(unsafe { _mm_cmpge_ps(self.0, rhs.0) })
    }
}

// ============================================================================
// i32x4
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct i32x4(pub(crate) __m128i);

/// Token-gated constants for i32x4. Cannot be constructed directly.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub struct I32x4Consts(());

#[cfg(target_arch = "x86_64")]
impl I32x4Consts {
    #[inline(always)]
    pub fn zero(self) -> i32x4 {
        i32x4(unsafe { _mm_setzero_si128() })
    }
    #[inline(always)]
    pub fn one(self) -> i32x4 {
        i32x4(unsafe { _mm_set1_epi32(1) })
    }
    #[inline(always)]
    pub fn neg_one(self) -> i32x4 {
        i32x4(unsafe { _mm_set1_epi32(-1) })
    }
    #[inline(always)]
    pub fn min_value(self) -> i32x4 {
        i32x4(unsafe { _mm_set1_epi32(i32::MIN) })
    }
    #[inline(always)]
    pub fn max_value(self) -> i32x4 {
        i32x4(unsafe { _mm_set1_epi32(i32::MAX) })
    }
}

#[cfg(target_arch = "x86_64")]
impl i32x4 {
    pub const LANES: usize = 4;

    /// Get token-gated constants. Usage: `i32x4::consts(token).one()`
    #[inline(always)]
    pub fn consts(_: impl crate::HasSse2) -> I32x4Consts {
        I32x4Consts(())
    }

    #[inline(always)]
    pub fn load(_: impl crate::HasSse2, data: &[i32; 4]) -> Self {
        Self(unsafe { _mm_loadu_si128(data.as_ptr() as *const __m128i) })
    }
    #[inline(always)]
    pub fn splat(_: impl crate::HasSse2, v: i32) -> Self {
        Self(unsafe { _mm_set1_epi32(v) })
    }
    #[inline(always)]
    pub fn zero(_: impl crate::HasSse2) -> Self {
        Self(unsafe { _mm_setzero_si128() })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i32; 4] {
        let mut out = [0i32; 4];
        unsafe { _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, self.0) };
        out
    }
    #[inline(always)]
    pub fn as_array(&self) -> &[i32; 4] {
        unsafe { &*(self as *const Self as *const [i32; 4]) }
    }
    #[inline(always)]
    pub fn as_mut_array(&mut self) -> &mut [i32; 4] {
        unsafe { &mut *(self as *mut Self as *mut [i32; 4]) }
    }
    #[inline(always)]
    fn from_raw(v: __m128i) -> Self {
        Self(v)
    }

    #[inline(always)]
    pub fn min(self, o: Self) -> Self {
        Self(unsafe { _mm_min_epi32(self.0, o.0) })
    }
    #[inline(always)]
    pub fn max(self, o: Self) -> Self {
        Self(unsafe { _mm_max_epi32(self.0, o.0) })
    }
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { _mm_abs_epi32(self.0) })
    }
    #[inline(always)]
    pub fn to_f32x4(self) -> f32x4 {
        f32x4::from_i32x4(self)
    }
    #[inline(always)]
    pub fn to_bitmask(self) -> u32 {
        unsafe { _mm_movemask_ps(_mm_castsi128_ps(self.0)) as u32 }
    }
    #[inline(always)]
    pub fn any(self) -> bool {
        self.to_bitmask() != 0
    }
    #[inline(always)]
    pub fn all(self) -> bool {
        self.to_bitmask() == 0xF
    }
    #[inline(always)]
    pub fn none(self) -> bool {
        self.to_bitmask() == 0
    }

    #[inline(always)]
    pub fn blend(self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm_blendv_epi8(if_false.0, if_true.0, self.0) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Neg for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(unsafe { _mm_sub_epi32(_mm_setzero_si128(), self.0) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Not for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(unsafe { _mm_xor_si128(self.0, _mm_set1_epi32(-1)) })
    }
}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(i32x4, _mm_add_epi32, _mm_sub_epi32);
#[cfg(target_arch = "x86_64")]
impl Mul for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(unsafe { _mm_mullo_epi32(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(i32x4, _mm_and_si128, _mm_or_si128, _mm_xor_si128);
#[cfg(target_arch = "x86_64")]
impl_bitwise_assign_ops!(i32x4);

// Note: SSE2 doesn't have variable shift intrinsics (_mm_sllv_epi32 requires AVX2)
// Use shift_left/shift_right methods with const generics for SSE, or upgrade to AVX2 token

#[cfg(target_arch = "x86_64")]
impl core::fmt::Debug for i32x4 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("i32x4").field(&self.to_array()).finish()
    }
}

#[cfg(target_arch = "x86_64")]
impl PartialEq for i32x4 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// f32x4 - NEON (aarch64)
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct f32x4(float32x4_t);

/// Token-gated constants for f32x4. Cannot be constructed directly.
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub struct F32x4Consts(());

#[cfg(target_arch = "aarch64")]
impl F32x4Consts {
    #[inline(always)]
    pub fn one(self) -> f32x4 {
        f32x4(unsafe { vdupq_n_f32(1.0) })
    }
    #[inline(always)]
    pub fn zero(self) -> f32x4 {
        f32x4(unsafe { vdupq_n_f32(0.0) })
    }
    #[inline(always)]
    pub fn half(self) -> f32x4 {
        f32x4(unsafe { vdupq_n_f32(0.5) })
    }
    #[inline(always)]
    pub fn neg_one(self) -> f32x4 {
        f32x4(unsafe { vdupq_n_f32(-1.0) })
    }
    #[inline(always)]
    pub fn pi(self) -> f32x4 {
        f32x4(unsafe { vdupq_n_f32(core::f32::consts::PI) })
    }
    #[inline(always)]
    pub fn tau(self) -> f32x4 {
        f32x4(unsafe { vdupq_n_f32(core::f32::consts::TAU) })
    }
    #[inline(always)]
    pub fn e(self) -> f32x4 {
        f32x4(unsafe { vdupq_n_f32(core::f32::consts::E) })
    }
}

#[cfg(target_arch = "aarch64")]
impl f32x4 {
    pub const LANES: usize = 4;

    /// Get token-gated constants. Usage: `f32x4::consts(token).one()`
    #[inline(always)]
    pub fn consts(_: impl crate::HasNeon) -> F32x4Consts {
        F32x4Consts(())
    }

    #[inline(always)]
    pub fn load(_: impl crate::HasNeon, data: &[f32; 4]) -> Self {
        Self(unsafe { vld1q_f32(data.as_ptr()) })
    }
    #[inline(always)]
    pub fn splat(_: impl crate::HasNeon, v: f32) -> Self {
        Self(unsafe { vdupq_n_f32(v) })
    }
    #[inline(always)]
    pub fn zero(_: impl crate::HasNeon) -> Self {
        Self(unsafe { vdupq_n_f32(0.0) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        unsafe { vst1q_f32(out.as_mut_ptr(), self.0) };
        out
    }
    #[inline(always)]
    pub fn as_array(&self) -> &[f32; 4] {
        unsafe { &*(self as *const Self as *const [f32; 4]) }
    }
    #[inline(always)]
    pub fn as_mut_array(&mut self) -> &mut [f32; 4] {
        unsafe { &mut *(self as *mut Self as *mut [f32; 4]) }
    }
    #[inline(always)]
    fn from_raw(v: float32x4_t) -> Self {
        Self(v)
    }

    #[inline(always)]
    pub fn min(self, o: Self) -> Self {
        Self(unsafe { vminq_f32(self.0, o.0) })
    }
    #[inline(always)]
    pub fn max(self, o: Self) -> Self {
        Self(unsafe { vmaxq_f32(self.0, o.0) })
    }
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { vabsq_f32(self.0) })
    }
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(unsafe { vsqrtq_f32(self.0) })
    }
    #[inline(always)]
    pub fn recip(self) -> Self {
        Self(unsafe { vrecpeq_f32(self.0) })
    }
    #[inline(always)]
    pub fn recip_sqrt(self) -> Self {
        Self(unsafe { vrsqrteq_f32(self.0) })
    }

    #[inline(always)]
    pub fn round(self) -> Self {
        Self(unsafe { vrndnq_f32(self.0) })
    }
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(unsafe { vrndmq_f32(self.0) })
    }
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self(unsafe { vrndpq_f32(self.0) })
    }
    #[inline(always)]
    pub fn trunc(self) -> Self {
        Self(unsafe { vrndq_f32(self.0) })
    }

    #[inline(always)]
    pub fn reduce_add(self) -> f32 {
        unsafe { vaddvq_f32(self.0) }
    }

    #[inline(always)]
    pub fn blend(self, if_true: Self, if_false: Self) -> Self {
        // NEON: use vbslq with the mask reinterpreted as u32x4
        let mask = unsafe { vreinterpretq_u32_f32(self.0) };
        Self(unsafe { vbslq_f32(mask, if_true.0, if_false.0) })
    }

    /// Convert mask to bitmask (NEON doesn't have movemask, emulate)
    #[inline(always)]
    pub fn to_bitmask(self) -> u32 {
        unsafe {
            let mask = vreinterpretq_u32_f32(self.0);
            let shifted = vshrq_n_u32(mask, 31);
            let arr: [u32; 4] = core::mem::transmute(shifted);
            arr[0] | (arr[1] << 1) | (arr[2] << 2) | (arr[3] << 3)
        }
    }
    #[inline(always)]
    pub fn any(self) -> bool {
        unsafe { vmaxvq_u32(vreinterpretq_u32_f32(self.0)) != 0 }
    }
    #[inline(always)]
    pub fn all(self) -> bool {
        unsafe { vminvq_u32(vreinterpretq_u32_f32(self.0)) == u32::MAX }
    }
    #[inline(always)]
    pub fn none(self) -> bool {
        unsafe { vmaxvq_u32(vreinterpretq_u32_f32(self.0)) == 0 }
    }

    #[inline(always)]
    pub fn round_int(self) -> i32x4 {
        i32x4(unsafe { vcvtnq_s32_f32(self.0) })
    }
    #[inline(always)]
    pub fn trunc_int(self) -> i32x4 {
        i32x4(unsafe { vcvtq_s32_f32(self.0) })
    }
    #[inline(always)]
    pub fn from_i32x4(v: i32x4) -> Self {
        Self(unsafe { vcvtq_f32_s32(v.0) })
    }
}

#[cfg(target_arch = "aarch64")]
impl Neg for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(unsafe { vnegq_f32(self.0) })
    }
}

#[cfg(target_arch = "aarch64")]
impl Not for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(unsafe { vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(self.0))) })
    }
}

#[cfg(target_arch = "aarch64")]
impl Add for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(unsafe { vaddq_f32(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "aarch64")]
impl Sub for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(unsafe { vsubq_f32(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "aarch64")]
impl Mul for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(unsafe { vmulq_f32(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "aarch64")]
impl Div for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(unsafe { vdivq_f32(self.0, rhs.0) })
    }
}

#[cfg(target_arch = "aarch64")]
impl BitAnd for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(unsafe {
            vreinterpretq_f32_u32(vandq_u32(
                vreinterpretq_u32_f32(self.0),
                vreinterpretq_u32_f32(rhs.0),
            ))
        })
    }
}
#[cfg(target_arch = "aarch64")]
impl BitOr for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(unsafe {
            vreinterpretq_f32_u32(vorrq_u32(
                vreinterpretq_u32_f32(self.0),
                vreinterpretq_u32_f32(rhs.0),
            ))
        })
    }
}
#[cfg(target_arch = "aarch64")]
impl BitXor for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(unsafe {
            vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(self.0),
                vreinterpretq_u32_f32(rhs.0),
            ))
        })
    }
}
#[cfg(target_arch = "aarch64")]
impl_bitwise_assign_ops!(f32x4);
#[cfg(target_arch = "aarch64")]
impl_assign_ops!(f32x4);

// Scalar ops for NEON f32x4
#[cfg(target_arch = "aarch64")]
impl Add<f32> for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: f32) -> Self {
        self + Self(unsafe { vdupq_n_f32(rhs) })
    }
}
#[cfg(target_arch = "aarch64")]
impl Sub<f32> for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: f32) -> Self {
        self - Self(unsafe { vdupq_n_f32(rhs) })
    }
}
#[cfg(target_arch = "aarch64")]
impl Mul<f32> for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        self * Self(unsafe { vdupq_n_f32(rhs) })
    }
}
#[cfg(target_arch = "aarch64")]
impl Div<f32> for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self {
        self / Self(unsafe { vdupq_n_f32(rhs) })
    }
}

#[cfg(target_arch = "aarch64")]
impl core::fmt::Debug for f32x4 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("f32x4").field(&self.to_array()).finish()
    }
}

#[cfg(target_arch = "aarch64")]
impl PartialEq for f32x4 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

#[cfg(target_arch = "aarch64")]
impl Index<usize> for f32x4 {
    type Output = f32;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_array()[index]
    }
}

#[cfg(target_arch = "aarch64")]
impl IndexMut<usize> for f32x4 {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_array()[index]
    }
}

#[cfg(target_arch = "aarch64")]
impl SimdEq for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn simd_eq(self, rhs: Self) -> Self {
        Self(unsafe { vreinterpretq_f32_u32(vceqq_f32(self.0, rhs.0)) })
    }
}
#[cfg(target_arch = "aarch64")]
impl SimdNe for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn simd_ne(self, rhs: Self) -> Self {
        !self.simd_eq(rhs)
    }
}
#[cfg(target_arch = "aarch64")]
impl SimdLt for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn simd_lt(self, rhs: Self) -> Self {
        Self(unsafe { vreinterpretq_f32_u32(vcltq_f32(self.0, rhs.0)) })
    }
}
#[cfg(target_arch = "aarch64")]
impl SimdLe for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn simd_le(self, rhs: Self) -> Self {
        Self(unsafe { vreinterpretq_f32_u32(vcleq_f32(self.0, rhs.0)) })
    }
}
#[cfg(target_arch = "aarch64")]
impl SimdGt for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn simd_gt(self, rhs: Self) -> Self {
        Self(unsafe { vreinterpretq_f32_u32(vcgtq_f32(self.0, rhs.0)) })
    }
}
#[cfg(target_arch = "aarch64")]
impl SimdGe for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn simd_ge(self, rhs: Self) -> Self {
        Self(unsafe { vreinterpretq_f32_u32(vcgeq_f32(self.0, rhs.0)) })
    }
}

// ============================================================================
// i32x4 - NEON (aarch64)
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct i32x4(pub(crate) int32x4_t);

/// Token-gated constants for i32x4. Cannot be constructed directly.
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub struct I32x4Consts(());

#[cfg(target_arch = "aarch64")]
impl I32x4Consts {
    #[inline(always)]
    pub fn zero(self) -> i32x4 {
        i32x4(unsafe { vdupq_n_s32(0) })
    }
    #[inline(always)]
    pub fn one(self) -> i32x4 {
        i32x4(unsafe { vdupq_n_s32(1) })
    }
    #[inline(always)]
    pub fn neg_one(self) -> i32x4 {
        i32x4(unsafe { vdupq_n_s32(-1) })
    }
    #[inline(always)]
    pub fn min_value(self) -> i32x4 {
        i32x4(unsafe { vdupq_n_s32(i32::MIN) })
    }
    #[inline(always)]
    pub fn max_value(self) -> i32x4 {
        i32x4(unsafe { vdupq_n_s32(i32::MAX) })
    }
}

#[cfg(target_arch = "aarch64")]
impl i32x4 {
    pub const LANES: usize = 4;

    /// Get token-gated constants. Usage: `i32x4::consts(token).one()`
    #[inline(always)]
    pub fn consts(_: impl crate::HasNeon) -> I32x4Consts {
        I32x4Consts(())
    }

    #[inline(always)]
    pub fn load(_: impl crate::HasNeon, data: &[i32; 4]) -> Self {
        Self(unsafe { vld1q_s32(data.as_ptr()) })
    }
    #[inline(always)]
    pub fn splat(_: impl crate::HasNeon, v: i32) -> Self {
        Self(unsafe { vdupq_n_s32(v) })
    }
    #[inline(always)]
    pub fn zero(_: impl crate::HasNeon) -> Self {
        Self(unsafe { vdupq_n_s32(0) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i32; 4] {
        let mut out = [0i32; 4];
        unsafe { vst1q_s32(out.as_mut_ptr(), self.0) };
        out
    }
    #[inline(always)]
    pub fn as_array(&self) -> &[i32; 4] {
        unsafe { &*(self as *const Self as *const [i32; 4]) }
    }
    #[inline(always)]
    pub fn as_mut_array(&mut self) -> &mut [i32; 4] {
        unsafe { &mut *(self as *mut Self as *mut [i32; 4]) }
    }
    #[inline(always)]
    fn from_raw(v: int32x4_t) -> Self {
        Self(v)
    }

    #[inline(always)]
    pub fn min(self, o: Self) -> Self {
        Self(unsafe { vminq_s32(self.0, o.0) })
    }
    #[inline(always)]
    pub fn max(self, o: Self) -> Self {
        Self(unsafe { vmaxq_s32(self.0, o.0) })
    }
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { vabsq_s32(self.0) })
    }

    #[inline(always)]
    pub fn to_f32x4(self) -> f32x4 {
        f32x4::from_i32x4(self)
    }

    /// Convert mask to bitmask
    #[inline(always)]
    pub fn to_bitmask(self) -> u32 {
        unsafe {
            let mask = vreinterpretq_u32_s32(self.0);
            let shifted = vshrq_n_u32(mask, 31);
            let arr: [u32; 4] = core::mem::transmute(shifted);
            arr[0] | (arr[1] << 1) | (arr[2] << 2) | (arr[3] << 3)
        }
    }
    #[inline(always)]
    pub fn any(self) -> bool {
        unsafe { vmaxvq_u32(vreinterpretq_u32_s32(self.0)) != 0 }
    }
    #[inline(always)]
    pub fn all(self) -> bool {
        unsafe { vminvq_u32(vreinterpretq_u32_s32(self.0)) == u32::MAX }
    }
    #[inline(always)]
    pub fn none(self) -> bool {
        unsafe { vmaxvq_u32(vreinterpretq_u32_s32(self.0)) == 0 }
    }

    #[inline(always)]
    pub fn blend(self, if_true: Self, if_false: Self) -> Self {
        let mask = unsafe { vreinterpretq_u32_s32(self.0) };
        Self(unsafe {
            vreinterpretq_s32_u32(vbslq_u32(
                mask,
                vreinterpretq_u32_s32(if_true.0),
                vreinterpretq_u32_s32(if_false.0),
            ))
        })
    }
}

#[cfg(target_arch = "aarch64")]
impl Neg for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(unsafe { vnegq_s32(self.0) })
    }
}

#[cfg(target_arch = "aarch64")]
impl Not for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(unsafe { vreinterpretq_s32_u32(vmvnq_u32(vreinterpretq_u32_s32(self.0))) })
    }
}

#[cfg(target_arch = "aarch64")]
impl Add for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(unsafe { vaddq_s32(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "aarch64")]
impl Sub for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(unsafe { vsubq_s32(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "aarch64")]
impl Mul for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(unsafe { vmulq_s32(self.0, rhs.0) })
    }
}

#[cfg(target_arch = "aarch64")]
impl BitAnd for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(unsafe { vandq_s32(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "aarch64")]
impl BitOr for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(unsafe { vorrq_s32(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "aarch64")]
impl BitXor for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(unsafe { veorq_s32(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "aarch64")]
impl_bitwise_assign_ops!(i32x4);

// Shifts for NEON i32x4
#[cfg(target_arch = "aarch64")]
impl Shl<i32> for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: i32) -> Self {
        Self(unsafe { vshlq_s32(self.0, vdupq_n_s32(rhs)) })
    }
}
#[cfg(target_arch = "aarch64")]
impl Shr<i32> for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: i32) -> Self {
        // NEON vshlq_s32 with negative shift = right shift
        Self(unsafe { vshlq_s32(self.0, vdupq_n_s32(-rhs)) })
    }
}

#[cfg(target_arch = "aarch64")]
impl core::fmt::Debug for i32x4 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("i32x4").field(&self.to_array()).finish()
    }
}

#[cfg(target_arch = "aarch64")]
impl PartialEq for i32x4 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

#[cfg(target_arch = "aarch64")]
impl SimdEq for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn simd_eq(self, rhs: Self) -> Self {
        Self(unsafe { vreinterpretq_s32_u32(vceqq_s32(self.0, rhs.0)) })
    }
}
#[cfg(target_arch = "aarch64")]
impl SimdGt for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn simd_gt(self, rhs: Self) -> Self {
        Self(unsafe { vreinterpretq_s32_u32(vcgtq_s32(self.0, rhs.0)) })
    }
}
#[cfg(target_arch = "aarch64")]
impl SimdLt for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn simd_lt(self, rhs: Self) -> Self {
        rhs.simd_gt(self)
    }
}
#[cfg(target_arch = "aarch64")]
impl SimdGe for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn simd_ge(self, rhs: Self) -> Self {
        !rhs.simd_gt(self)
    }
}
#[cfg(target_arch = "aarch64")]
impl SimdLe for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn simd_le(self, rhs: Self) -> Self {
        !self.simd_gt(rhs)
    }
}
#[cfg(target_arch = "aarch64")]
impl SimdNe for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn simd_ne(self, rhs: Self) -> Self {
        !self.simd_eq(rhs)
    }
}

// ============================================================================
// f64x4
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct f64x4(__m256d);

#[cfg(target_arch = "x86_64")]
impl f64x4 {
    pub const LANES: usize = 4;

    #[inline(always)]
    pub fn load(_: impl crate::HasAvx2, data: &[f64; 4]) -> Self {
        Self(unsafe { _mm256_loadu_pd(data.as_ptr()) })
    }
    #[inline(always)]
    pub fn splat(_: impl crate::HasAvx2, v: f64) -> Self {
        Self(unsafe { _mm256_set1_pd(v) })
    }
    #[inline(always)]
    pub fn zero(_: impl crate::HasAvx2) -> Self {
        Self(unsafe { _mm256_setzero_pd() })
    }

    #[inline(always)]
    pub fn to_array(self) -> [f64; 4] {
        let mut out = [0.0f64; 4];
        unsafe { _mm256_storeu_pd(out.as_mut_ptr(), self.0) };
        out
    }
    #[inline(always)]
    pub fn as_array(&self) -> &[f64; 4] {
        unsafe { &*(self as *const Self as *const [f64; 4]) }
    }
    #[inline(always)]
    fn from_raw(v: __m256d) -> Self {
        Self(v)
    }

    #[inline(always)]
    pub fn min(self, o: Self) -> Self {
        Self(unsafe { _mm256_min_pd(self.0, o.0) })
    }
    #[inline(always)]
    pub fn max(self, o: Self) -> Self {
        Self(unsafe { _mm256_max_pd(self.0, o.0) })
    }
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe {
            _mm256_and_pd(
                self.0,
                _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFF_FFFF_FFFF_FFFF)),
            )
        })
    }
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(unsafe { _mm256_sqrt_pd(self.0) })
    }

    #[inline(always)]
    pub fn round(self) -> Self {
        Self(unsafe { _mm256_round_pd(self.0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) })
    }
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(unsafe { _mm256_floor_pd(self.0) })
    }
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self(unsafe { _mm256_ceil_pd(self.0) })
    }

    #[inline(always)]
    pub fn blend(self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm256_blendv_pd(if_false.0, if_true.0, self.0) })
    }
    #[inline(always)]
    pub fn to_bitmask(self) -> u32 {
        unsafe { _mm256_movemask_pd(self.0) as u32 }
    }
}

#[cfg(target_arch = "x86_64")]
impl Neg for f64x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(unsafe { _mm256_xor_pd(self.0, _mm256_set1_pd(-0.0)) })
    }
}

#[cfg(target_arch = "x86_64")]
impl_arithmetic_ops!(
    f64x4,
    _mm256_add_pd,
    _mm256_sub_pd,
    _mm256_mul_pd,
    _mm256_div_pd
);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(f64x4, _mm256_and_pd, _mm256_or_pd, _mm256_xor_pd);
#[cfg(target_arch = "x86_64")]
impl_scalar_ops!(f64x4, f64, _mm256_set1_pd);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(f64x4);

#[cfg(target_arch = "x86_64")]
impl core::fmt::Debug for f64x4 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("f64x4").field(&self.to_array()).finish()
    }
}

#[cfg(target_arch = "x86_64")]
impl PartialEq for f64x4 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// f64x2
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct f64x2(__m128d);

#[cfg(target_arch = "x86_64")]
impl f64x2 {
    pub const LANES: usize = 2;

    #[inline(always)]
    pub fn load(_: impl crate::HasSse2, data: &[f64; 2]) -> Self {
        Self(unsafe { _mm_loadu_pd(data.as_ptr()) })
    }
    #[inline(always)]
    pub fn splat(_: impl crate::HasSse2, v: f64) -> Self {
        Self(unsafe { _mm_set1_pd(v) })
    }
    #[inline(always)]
    pub fn zero(_: impl crate::HasSse2) -> Self {
        Self(unsafe { _mm_setzero_pd() })
    }

    #[inline(always)]
    pub fn to_array(self) -> [f64; 2] {
        let mut out = [0.0f64; 2];
        unsafe { _mm_storeu_pd(out.as_mut_ptr(), self.0) };
        out
    }
    #[inline(always)]
    pub fn as_array(&self) -> &[f64; 2] {
        unsafe { &*(self as *const Self as *const [f64; 2]) }
    }
    #[inline(always)]
    fn from_raw(v: __m128d) -> Self {
        Self(v)
    }

    #[inline(always)]
    pub fn min(self, o: Self) -> Self {
        Self(unsafe { _mm_min_pd(self.0, o.0) })
    }
    #[inline(always)]
    pub fn max(self, o: Self) -> Self {
        Self(unsafe { _mm_max_pd(self.0, o.0) })
    }
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(unsafe { _mm_sqrt_pd(self.0) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Neg for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(unsafe { _mm_xor_pd(self.0, _mm_set1_pd(-0.0)) })
    }
}

#[cfg(target_arch = "x86_64")]
impl_arithmetic_ops!(f64x2, _mm_add_pd, _mm_sub_pd, _mm_mul_pd, _mm_div_pd);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(f64x2, _mm_and_pd, _mm_or_pd, _mm_xor_pd);
#[cfg(target_arch = "x86_64")]
impl_scalar_ops!(f64x2, f64, _mm_set1_pd);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(f64x2);

#[cfg(target_arch = "x86_64")]
impl core::fmt::Debug for f64x2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("f64x2").field(&self.to_array()).finish()
    }
}

#[cfg(target_arch = "x86_64")]
impl PartialEq for f64x2 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// i64x4
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct i64x4(pub(crate) __m256i);

#[cfg(target_arch = "x86_64")]
impl i64x4 {
    pub const LANES: usize = 4;

    #[inline(always)]
    pub fn load(_: impl crate::HasAvx2, data: &[i64; 4]) -> Self {
        Self(unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) })
    }
    #[inline(always)]
    pub fn splat(_: impl crate::HasAvx2, v: i64) -> Self {
        Self(unsafe { _mm256_set1_epi64x(v) })
    }
    #[inline(always)]
    pub fn zero(_: impl crate::HasAvx2) -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i64; 4] {
        let mut out = [0i64; 4];
        unsafe { _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, self.0) };
        out
    }
    #[inline(always)]
    fn from_raw(v: __m256i) -> Self {
        Self(v)
    }
}

#[cfg(target_arch = "x86_64")]
impl Neg for i64x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(unsafe { _mm256_sub_epi64(_mm256_setzero_si256(), self.0) })
    }
}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(i64x4, _mm256_add_epi64, _mm256_sub_epi64);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(i64x4, _mm256_and_si256, _mm256_or_si256, _mm256_xor_si256);

#[cfg(target_arch = "x86_64")]
impl core::fmt::Debug for i64x4 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("i64x4").field(&self.to_array()).finish()
    }
}

#[cfg(target_arch = "x86_64")]
impl PartialEq for i64x4 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// i16x16
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct i16x16(pub(crate) __m256i);

#[cfg(target_arch = "x86_64")]
impl i16x16 {
    pub const LANES: usize = 16;

    #[inline(always)]
    pub fn load(_: impl crate::HasAvx2, data: &[i16; 16]) -> Self {
        Self(unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) })
    }
    #[inline(always)]
    pub fn splat(_: impl crate::HasAvx2, v: i16) -> Self {
        Self(unsafe { _mm256_set1_epi16(v) })
    }
    #[inline(always)]
    pub fn zero(_: impl crate::HasAvx2) -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i16; 16] {
        let mut out = [0i16; 16];
        unsafe { _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, self.0) };
        out
    }
    #[inline(always)]
    fn from_raw(v: __m256i) -> Self {
        Self(v)
    }

    #[inline(always)]
    pub fn min(self, o: Self) -> Self {
        Self(unsafe { _mm256_min_epi16(self.0, o.0) })
    }
    #[inline(always)]
    pub fn max(self, o: Self) -> Self {
        Self(unsafe { _mm256_max_epi16(self.0, o.0) })
    }
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { _mm256_abs_epi16(self.0) })
    }
    #[inline(always)]
    pub fn saturating_add(self, o: Self) -> Self {
        Self(unsafe { _mm256_adds_epi16(self.0, o.0) })
    }
    #[inline(always)]
    pub fn saturating_sub(self, o: Self) -> Self {
        Self(unsafe { _mm256_subs_epi16(self.0, o.0) })
    }
}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(i16x16, _mm256_add_epi16, _mm256_sub_epi16);
#[cfg(target_arch = "x86_64")]
impl Mul for i16x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_mullo_epi16(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(i16x16, _mm256_and_si256, _mm256_or_si256, _mm256_xor_si256);

#[cfg(target_arch = "x86_64")]
impl core::fmt::Debug for i16x16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("i16x16").field(&self.to_array()).finish()
    }
}

#[cfg(target_arch = "x86_64")]
impl PartialEq for i16x16 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// u8x32
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct u8x32(pub(crate) __m256i);

#[cfg(target_arch = "x86_64")]
impl u8x32 {
    pub const LANES: usize = 32;

    #[inline(always)]
    pub fn load(_: impl crate::HasAvx2, data: &[u8; 32]) -> Self {
        Self(unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) })
    }
    #[inline(always)]
    pub fn splat(_: impl crate::HasAvx2, v: u8) -> Self {
        Self(unsafe { _mm256_set1_epi8(v as i8) })
    }
    #[inline(always)]
    pub fn zero(_: impl crate::HasAvx2) -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }

    #[inline(always)]
    pub fn to_array(self) -> [u8; 32] {
        let mut out = [0u8; 32];
        unsafe { _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, self.0) };
        out
    }
    #[inline(always)]
    fn from_raw(v: __m256i) -> Self {
        Self(v)
    }

    #[inline(always)]
    pub fn min(self, o: Self) -> Self {
        Self(unsafe { _mm256_min_epu8(self.0, o.0) })
    }
    #[inline(always)]
    pub fn max(self, o: Self) -> Self {
        Self(unsafe { _mm256_max_epu8(self.0, o.0) })
    }
    #[inline(always)]
    pub fn saturating_add(self, o: Self) -> Self {
        Self(unsafe { _mm256_adds_epu8(self.0, o.0) })
    }
    #[inline(always)]
    pub fn saturating_sub(self, o: Self) -> Self {
        Self(unsafe { _mm256_subs_epu8(self.0, o.0) })
    }
}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(u8x32, _mm256_add_epi8, _mm256_sub_epi8);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(u8x32, _mm256_and_si256, _mm256_or_si256, _mm256_xor_si256);

#[cfg(target_arch = "x86_64")]
impl core::fmt::Debug for u8x32 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("u8x32").field(&self.to_array()).finish()
    }
}

#[cfg(target_arch = "x86_64")]
impl PartialEq for u8x32 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// i16x8 (SSE2 - for widening conversions to i32x8)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct i16x8(pub(crate) __m128i);

#[cfg(target_arch = "x86_64")]
impl i16x8 {
    pub const LANES: usize = 8;

    #[inline(always)]
    pub fn load(_: impl crate::HasSse2, data: &[i16; 8]) -> Self {
        Self(unsafe { _mm_loadu_si128(data.as_ptr() as *const __m128i) })
    }
    #[inline(always)]
    pub fn splat(_: impl crate::HasSse2, v: i16) -> Self {
        Self(unsafe { _mm_set1_epi16(v) })
    }
    #[inline(always)]
    pub fn zero(_: impl crate::HasSse2) -> Self {
        Self(unsafe { _mm_setzero_si128() })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i16; 8] {
        let mut out = [0i16; 8];
        unsafe { _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, self.0) };
        out
    }
    #[inline(always)]
    pub fn as_array(&self) -> &[i16; 8] {
        unsafe { &*(self as *const Self as *const [i16; 8]) }
    }
    #[inline(always)]
    pub fn raw(self) -> __m128i {
        self.0
    }
    #[inline(always)]
    fn from_raw(v: __m128i) -> Self {
        Self(v)
    }

    #[inline(always)]
    pub fn min(self, o: Self) -> Self {
        Self(unsafe { _mm_min_epi16(self.0, o.0) })
    }
    #[inline(always)]
    pub fn max(self, o: Self) -> Self {
        Self(unsafe { _mm_max_epi16(self.0, o.0) })
    }
    #[inline(always)]
    pub fn saturating_add(self, o: Self) -> Self {
        Self(unsafe { _mm_adds_epi16(self.0, o.0) })
    }
    #[inline(always)]
    pub fn saturating_sub(self, o: Self) -> Self {
        Self(unsafe { _mm_subs_epi16(self.0, o.0) })
    }
}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(i16x8, _mm_add_epi16, _mm_sub_epi16);
#[cfg(target_arch = "x86_64")]
impl Mul for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(unsafe { _mm_mullo_epi16(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(i16x8, _mm_and_si128, _mm_or_si128, _mm_xor_si128);

#[cfg(target_arch = "x86_64")]
impl core::fmt::Debug for i16x8 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("i16x8").field(&self.to_array()).finish()
    }
}

#[cfg(target_arch = "x86_64")]
impl PartialEq for i16x8 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// u16x8 (SSE2 - for widening conversions to i32x8/u32x8)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct u16x8(pub(crate) __m128i);

#[cfg(target_arch = "x86_64")]
impl u16x8 {
    pub const LANES: usize = 8;

    #[inline(always)]
    pub fn load(_: impl crate::HasSse2, data: &[u16; 8]) -> Self {
        Self(unsafe { _mm_loadu_si128(data.as_ptr() as *const __m128i) })
    }
    #[inline(always)]
    pub fn splat(_: impl crate::HasSse2, v: u16) -> Self {
        Self(unsafe { _mm_set1_epi16(v as i16) })
    }
    #[inline(always)]
    pub fn zero(_: impl crate::HasSse2) -> Self {
        Self(unsafe { _mm_setzero_si128() })
    }

    #[inline(always)]
    pub fn to_array(self) -> [u16; 8] {
        let mut out = [0u16; 8];
        unsafe { _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, self.0) };
        out
    }
    #[inline(always)]
    pub fn as_array(&self) -> &[u16; 8] {
        unsafe { &*(self as *const Self as *const [u16; 8]) }
    }
    #[inline(always)]
    pub fn raw(self) -> __m128i {
        self.0
    }
    #[inline(always)]
    fn from_raw(v: __m128i) -> Self {
        Self(v)
    }

    #[inline(always)]
    pub fn min(self, o: Self) -> Self {
        Self(unsafe { _mm_min_epu16(self.0, o.0) })
    }
    #[inline(always)]
    pub fn max(self, o: Self) -> Self {
        Self(unsafe { _mm_max_epu16(self.0, o.0) })
    }
    #[inline(always)]
    pub fn saturating_add(self, o: Self) -> Self {
        Self(unsafe { _mm_adds_epu16(self.0, o.0) })
    }
    #[inline(always)]
    pub fn saturating_sub(self, o: Self) -> Self {
        Self(unsafe { _mm_subs_epu16(self.0, o.0) })
    }
}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(u16x8, _mm_add_epi16, _mm_sub_epi16);
#[cfg(target_arch = "x86_64")]
impl Mul for u16x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(unsafe { _mm_mullo_epi16(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(u16x8, _mm_and_si128, _mm_or_si128, _mm_xor_si128);

#[cfg(target_arch = "x86_64")]
impl core::fmt::Debug for u16x8 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("u16x8").field(&self.to_array()).finish()
    }
}

#[cfg(target_arch = "x86_64")]
impl PartialEq for u16x8 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// f32x16 (AVX-512DQ)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct f32x16(__m512);

/// Token-gated constants for f32x16. Cannot be constructed directly.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub struct F32x16Consts(());

#[cfg(target_arch = "x86_64")]
impl F32x16Consts {
    #[inline(always)]
    pub fn one(self) -> f32x16 {
        f32x16(unsafe { _mm512_set1_ps(1.0) })
    }
    #[inline(always)]
    pub fn zero(self) -> f32x16 {
        f32x16(unsafe { _mm512_setzero_ps() })
    }
    #[inline(always)]
    pub fn half(self) -> f32x16 {
        f32x16(unsafe { _mm512_set1_ps(0.5) })
    }
    #[inline(always)]
    pub fn neg_one(self) -> f32x16 {
        f32x16(unsafe { _mm512_set1_ps(-1.0) })
    }
    #[inline(always)]
    pub fn pi(self) -> f32x16 {
        f32x16(unsafe { _mm512_set1_ps(core::f32::consts::PI) })
    }
    #[inline(always)]
    pub fn tau(self) -> f32x16 {
        f32x16(unsafe { _mm512_set1_ps(core::f32::consts::TAU) })
    }
    #[inline(always)]
    pub fn e(self) -> f32x16 {
        f32x16(unsafe { _mm512_set1_ps(core::f32::consts::E) })
    }
}

#[cfg(target_arch = "x86_64")]
impl f32x16 {
    pub const LANES: usize = 16;

    /// Get token-gated constants. Usage: `f32x16::consts(token).one()`
    #[inline(always)]
    pub fn consts(_: impl crate::HasAvx512dq) -> F32x16Consts {
        F32x16Consts(())
    }

    #[inline(always)]
    pub fn load(_: impl crate::HasAvx512dq, data: &[f32; 16]) -> Self {
        Self(unsafe { _mm512_loadu_ps(data.as_ptr()) })
    }
    #[inline(always)]
    pub fn load_slice(_: impl crate::HasAvx512dq, data: &[f32]) -> Self {
        assert!(data.len() >= 16);
        Self(unsafe { _mm512_loadu_ps(data.as_ptr()) })
    }
    #[inline(always)]
    pub fn splat(_: impl crate::HasAvx512dq, v: f32) -> Self {
        Self(unsafe { _mm512_set1_ps(v) })
    }
    #[inline(always)]
    pub fn zero(_: impl crate::HasAvx512dq) -> Self {
        Self(unsafe { _mm512_setzero_ps() })
    }

    #[inline(always)]
    pub fn to_array(self) -> [f32; 16] {
        let mut out = [0.0f32; 16];
        unsafe { _mm512_storeu_ps(out.as_mut_ptr(), self.0) };
        out
    }
    #[inline(always)]
    pub fn store(self, out: &mut [f32; 16]) {
        unsafe { _mm512_storeu_ps(out.as_mut_ptr(), self.0) };
    }
    #[inline(always)]
    pub fn store_slice(self, out: &mut [f32]) {
        assert!(out.len() >= 16);
        unsafe { _mm512_storeu_ps(out.as_mut_ptr(), self.0) };
    }
    #[inline(always)]
    pub fn as_array(&self) -> &[f32; 16] {
        unsafe { &*(self as *const Self as *const [f32; 16]) }
    }
    #[inline(always)]
    pub fn as_mut_array(&mut self) -> &mut [f32; 16] {
        unsafe { &mut *(self as *mut Self as *mut [f32; 16]) }
    }
    #[inline(always)]
    pub fn raw(self) -> __m512 {
        self.0
    }
    #[inline(always)]
    fn from_raw(v: __m512) -> Self {
        Self(v)
    }

    // Math
    #[inline(always)]
    pub fn min(self, o: Self) -> Self {
        Self(unsafe { _mm512_min_ps(self.0, o.0) })
    }
    #[inline(always)]
    pub fn max(self, o: Self) -> Self {
        Self(unsafe { _mm512_max_ps(self.0, o.0) })
    }
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { _mm512_abs_ps(self.0) })
    }
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(unsafe { _mm512_sqrt_ps(self.0) })
    }
    #[inline(always)]
    pub fn recip(self) -> Self {
        Self(unsafe { _mm512_rcp14_ps(self.0) })
    }
    #[inline(always)]
    pub fn recip_sqrt(self) -> Self {
        Self(unsafe { _mm512_rsqrt14_ps(self.0) })
    }

    // FMA
    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        Self(unsafe { _mm512_fmadd_ps(self.0, b.0, c.0) })
    }
    #[inline(always)]
    pub fn mul_sub(self, b: Self, c: Self) -> Self {
        Self(unsafe { _mm512_fmsub_ps(self.0, b.0, c.0) })
    }
    #[inline(always)]
    pub fn neg_mul_add(self, b: Self, c: Self) -> Self {
        Self(unsafe { _mm512_fnmadd_ps(self.0, b.0, c.0) })
    }
    #[inline(always)]
    pub fn neg_mul_sub(self, b: Self, c: Self) -> Self {
        Self(unsafe { _mm512_fnmsub_ps(self.0, b.0, c.0) })
    }

    // Rounding
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(unsafe { _mm512_roundscale_ps(self.0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) })
    }
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(unsafe { _mm512_roundscale_ps(self.0, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC) })
    }
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self(unsafe { _mm512_roundscale_ps(self.0, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC) })
    }
    #[inline(always)]
    pub fn trunc(self) -> Self {
        Self(unsafe { _mm512_roundscale_ps(self.0, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC) })
    }

    // Horizontal
    #[inline(always)]
    pub fn reduce_add(self) -> f32 {
        unsafe { _mm512_reduce_add_ps(self.0) }
    }

    // Mask operations - AVX-512 uses k-masks
    #[inline(always)]
    pub fn to_bitmask(self) -> u16 {
        // Extract sign bits (avx512dq)
        unsafe { _mm512_movepi32_mask(_mm512_castps_si512(self.0)) }
    }
    #[inline(always)]
    pub fn any(self) -> bool {
        self.to_bitmask() != 0
    }
    #[inline(always)]
    pub fn all(self) -> bool {
        self.to_bitmask() == 0xFFFF
    }
    #[inline(always)]
    pub fn none(self) -> bool {
        self.to_bitmask() == 0
    }

    // Int conversion
    #[inline(always)]
    pub fn round_int(self) -> i32x16 {
        i32x16(unsafe { _mm512_cvtps_epi32(self.0) })
    }
    #[inline(always)]
    pub fn trunc_int(self) -> i32x16 {
        i32x16(unsafe { _mm512_cvttps_epi32(self.0) })
    }
    #[inline(always)]
    pub fn from_i32x16(v: i32x16) -> Self {
        Self(unsafe { _mm512_cvtepi32_ps(v.0) })
    }

    // Degrees
    #[inline(always)]
    pub fn to_degrees(self) -> Self {
        self * (180.0 / core::f32::consts::PI)
    }
    #[inline(always)]
    pub fn to_radians(self) -> Self {
        self * (core::f32::consts::PI / 180.0)
    }
}

#[cfg(target_arch = "x86_64")]
impl Neg for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(unsafe { _mm512_xor_ps(self.0, _mm512_set1_ps(-0.0)) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Add for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(unsafe { _mm512_add_ps(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "x86_64")]
impl Sub for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(unsafe { _mm512_sub_ps(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "x86_64")]
impl Mul for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(unsafe { _mm512_mul_ps(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "x86_64")]
impl Div for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(unsafe { _mm512_div_ps(self.0, rhs.0) })
    }
}

#[cfg(target_arch = "x86_64")]
impl BitAnd for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(unsafe { _mm512_and_ps(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "x86_64")]
impl BitOr for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(unsafe { _mm512_or_ps(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "x86_64")]
impl BitXor for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(unsafe { _mm512_xor_ps(self.0, rhs.0) })
    }
}

#[cfg(target_arch = "x86_64")]
impl_bitwise_assign_ops!(f32x16);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(f32x16);

// Scalar ops
#[cfg(target_arch = "x86_64")]
impl Add<f32> for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: f32) -> Self {
        self + Self(unsafe { _mm512_set1_ps(rhs) })
    }
}
#[cfg(target_arch = "x86_64")]
impl Sub<f32> for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: f32) -> Self {
        self - Self(unsafe { _mm512_set1_ps(rhs) })
    }
}
#[cfg(target_arch = "x86_64")]
impl Mul<f32> for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        self * Self(unsafe { _mm512_set1_ps(rhs) })
    }
}
#[cfg(target_arch = "x86_64")]
impl Div<f32> for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self {
        self / Self(unsafe { _mm512_set1_ps(rhs) })
    }
}

#[cfg(target_arch = "x86_64")]
impl SimdEq for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn simd_eq(self, rhs: Self) -> Self {
        let mask = unsafe { _mm512_cmp_ps_mask(self.0, rhs.0, _CMP_EQ_OQ) };
        Self(unsafe { _mm512_maskz_mov_ps(mask, _mm512_set1_ps(f32::from_bits(u32::MAX))) })
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdNe for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn simd_ne(self, rhs: Self) -> Self {
        let mask = unsafe { _mm512_cmp_ps_mask(self.0, rhs.0, _CMP_NEQ_OQ) };
        Self(unsafe { _mm512_maskz_mov_ps(mask, _mm512_set1_ps(f32::from_bits(u32::MAX))) })
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdLt for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn simd_lt(self, rhs: Self) -> Self {
        let mask = unsafe { _mm512_cmp_ps_mask(self.0, rhs.0, _CMP_LT_OQ) };
        Self(unsafe { _mm512_maskz_mov_ps(mask, _mm512_set1_ps(f32::from_bits(u32::MAX))) })
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdLe for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn simd_le(self, rhs: Self) -> Self {
        let mask = unsafe { _mm512_cmp_ps_mask(self.0, rhs.0, _CMP_LE_OQ) };
        Self(unsafe { _mm512_maskz_mov_ps(mask, _mm512_set1_ps(f32::from_bits(u32::MAX))) })
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdGt for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn simd_gt(self, rhs: Self) -> Self {
        let mask = unsafe { _mm512_cmp_ps_mask(self.0, rhs.0, _CMP_GT_OQ) };
        Self(unsafe { _mm512_maskz_mov_ps(mask, _mm512_set1_ps(f32::from_bits(u32::MAX))) })
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdGe for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn simd_ge(self, rhs: Self) -> Self {
        let mask = unsafe { _mm512_cmp_ps_mask(self.0, rhs.0, _CMP_GE_OQ) };
        Self(unsafe { _mm512_maskz_mov_ps(mask, _mm512_set1_ps(f32::from_bits(u32::MAX))) })
    }
}

#[cfg(target_arch = "x86_64")]
impl core::fmt::Debug for f32x16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("f32x16").field(&self.to_array()).finish()
    }
}

#[cfg(target_arch = "x86_64")]
impl PartialEq for f32x16 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

#[cfg(target_arch = "x86_64")]
impl Index<usize> for f32x16 {
    type Output = f32;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_array()[index]
    }
}

#[cfg(target_arch = "x86_64")]
impl IndexMut<usize> for f32x16 {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_array()[index]
    }
}

// ============================================================================
// i32x16 (AVX-512DQ)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct i32x16(pub(crate) __m512i);

/// Token-gated constants for i32x16. Cannot be constructed directly.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub struct I32x16Consts(());

#[cfg(target_arch = "x86_64")]
impl I32x16Consts {
    #[inline(always)]
    pub fn zero(self) -> i32x16 {
        i32x16(unsafe { _mm512_setzero_si512() })
    }
    #[inline(always)]
    pub fn one(self) -> i32x16 {
        i32x16(unsafe { _mm512_set1_epi32(1) })
    }
    #[inline(always)]
    pub fn neg_one(self) -> i32x16 {
        i32x16(unsafe { _mm512_set1_epi32(-1) })
    }
    #[inline(always)]
    pub fn min_value(self) -> i32x16 {
        i32x16(unsafe { _mm512_set1_epi32(i32::MIN) })
    }
    #[inline(always)]
    pub fn max_value(self) -> i32x16 {
        i32x16(unsafe { _mm512_set1_epi32(i32::MAX) })
    }
}

#[cfg(target_arch = "x86_64")]
impl i32x16 {
    pub const LANES: usize = 16;

    /// Get token-gated constants. Usage: `i32x16::consts(token).one()`
    #[inline(always)]
    pub fn consts(_: impl crate::HasAvx512dq) -> I32x16Consts {
        I32x16Consts(())
    }

    #[inline(always)]
    pub fn load(_: impl crate::HasAvx512dq, data: &[i32; 16]) -> Self {
        Self(unsafe { _mm512_loadu_si512(data.as_ptr() as *const __m512i) })
    }
    #[inline(always)]
    pub fn splat(_: impl crate::HasAvx512dq, v: i32) -> Self {
        Self(unsafe { _mm512_set1_epi32(v) })
    }
    #[inline(always)]
    pub fn zero(_: impl crate::HasAvx512dq) -> Self {
        Self(unsafe { _mm512_setzero_si512() })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i32; 16] {
        let mut out = [0i32; 16];
        unsafe { _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, self.0) };
        out
    }
    #[inline(always)]
    pub fn as_array(&self) -> &[i32; 16] {
        unsafe { &*(self as *const Self as *const [i32; 16]) }
    }
    #[inline(always)]
    pub fn as_mut_array(&mut self) -> &mut [i32; 16] {
        unsafe { &mut *(self as *mut Self as *mut [i32; 16]) }
    }
    #[inline(always)]
    pub fn raw(self) -> __m512i {
        self.0
    }
    #[inline(always)]
    fn from_raw(v: __m512i) -> Self {
        Self(v)
    }

    #[inline(always)]
    pub fn min(self, o: Self) -> Self {
        Self(unsafe { _mm512_min_epi32(self.0, o.0) })
    }
    #[inline(always)]
    pub fn max(self, o: Self) -> Self {
        Self(unsafe { _mm512_max_epi32(self.0, o.0) })
    }
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { _mm512_abs_epi32(self.0) })
    }

    #[inline(always)]
    pub fn to_f32x16(self) -> f32x16 {
        f32x16::from_i32x16(self)
    }

    #[inline(always)]
    pub fn to_bitmask(self) -> u16 {
        // Extract sign bits (avx512dq)
        unsafe { _mm512_movepi32_mask(self.0) }
    }
    #[inline(always)]
    pub fn any(self) -> bool {
        self.to_bitmask() != 0
    }
    #[inline(always)]
    pub fn all(self) -> bool {
        self.to_bitmask() == 0xFFFF
    }
    #[inline(always)]
    pub fn none(self) -> bool {
        self.to_bitmask() == 0
    }
}

#[cfg(target_arch = "x86_64")]
impl Neg for i32x16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(unsafe { _mm512_sub_epi32(_mm512_setzero_si512(), self.0) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Not for i32x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(unsafe { _mm512_xor_si512(self.0, _mm512_set1_epi32(-1)) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Add for i32x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(unsafe { _mm512_add_epi32(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "x86_64")]
impl Sub for i32x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(unsafe { _mm512_sub_epi32(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "x86_64")]
impl Mul for i32x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(unsafe { _mm512_mullo_epi32(self.0, rhs.0) })
    }
}

#[cfg(target_arch = "x86_64")]
impl BitAnd for i32x16 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(unsafe { _mm512_and_si512(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "x86_64")]
impl BitOr for i32x16 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(unsafe { _mm512_or_si512(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "x86_64")]
impl BitXor for i32x16 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(unsafe { _mm512_xor_si512(self.0, rhs.0) })
    }
}
#[cfg(target_arch = "x86_64")]
impl_bitwise_assign_ops!(i32x16);

// Shifts - AVX-512 has variable shifts
#[cfg(target_arch = "x86_64")]
impl Shl<i32> for i32x16 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: i32) -> Self {
        Self(unsafe { _mm512_sllv_epi32(self.0, _mm512_set1_epi32(rhs)) })
    }
}
#[cfg(target_arch = "x86_64")]
impl Shr<i32> for i32x16 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: i32) -> Self {
        Self(unsafe { _mm512_srav_epi32(self.0, _mm512_set1_epi32(rhs)) })
    }
}

// Per-lane shifts
#[cfg(target_arch = "x86_64")]
impl Shl<i32x16> for i32x16 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: i32x16) -> Self {
        let shift = unsafe { _mm512_and_si512(rhs.0, _mm512_set1_epi32(31)) };
        Self(unsafe { _mm512_sllv_epi32(self.0, shift) })
    }
}
#[cfg(target_arch = "x86_64")]
impl Shr<i32x16> for i32x16 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: i32x16) -> Self {
        let shift = unsafe { _mm512_and_si512(rhs.0, _mm512_set1_epi32(31)) };
        Self(unsafe { _mm512_srav_epi32(self.0, shift) })
    }
}

#[cfg(target_arch = "x86_64")]
impl SimdEq for i32x16 {
    type Output = Self;
    #[inline(always)]
    fn simd_eq(self, rhs: Self) -> Self {
        let mask = unsafe { _mm512_cmpeq_epi32_mask(self.0, rhs.0) };
        Self(unsafe { _mm512_maskz_mov_epi32(mask, _mm512_set1_epi32(-1)) })
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdGt for i32x16 {
    type Output = Self;
    #[inline(always)]
    fn simd_gt(self, rhs: Self) -> Self {
        let mask = unsafe { _mm512_cmpgt_epi32_mask(self.0, rhs.0) };
        Self(unsafe { _mm512_maskz_mov_epi32(mask, _mm512_set1_epi32(-1)) })
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdLt for i32x16 {
    type Output = Self;
    #[inline(always)]
    fn simd_lt(self, rhs: Self) -> Self {
        rhs.simd_gt(self)
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdGe for i32x16 {
    type Output = Self;
    #[inline(always)]
    fn simd_ge(self, rhs: Self) -> Self {
        !rhs.simd_gt(self)
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdLe for i32x16 {
    type Output = Self;
    #[inline(always)]
    fn simd_le(self, rhs: Self) -> Self {
        !self.simd_gt(rhs)
    }
}
#[cfg(target_arch = "x86_64")]
impl SimdNe for i32x16 {
    type Output = Self;
    #[inline(always)]
    fn simd_ne(self, rhs: Self) -> Self {
        !self.simd_eq(rhs)
    }
}

#[cfg(target_arch = "x86_64")]
impl core::fmt::Debug for i32x16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("i32x16").field(&self.to_array()).finish()
    }
}

#[cfg(target_arch = "x86_64")]
impl PartialEq for i32x16 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

#[cfg(target_arch = "x86_64")]
impl Index<usize> for i32x16 {
    type Output = i32;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_array()[index]
    }
}

#[cfg(target_arch = "x86_64")]
impl IndexMut<usize> for i32x16 {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_array()[index]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::tokens::SimdToken;
    use crate::{Avx2Token, Sse2Token};

    #[test]
    fn test_f32x8_basic() {
        if let Some(t) = Avx2Token::summon() {
            let a = f32x8::splat(t, 2.0);
            let b = f32x8::splat(t, 3.0);
            assert_eq!((a + b).to_array(), [5.0f32; 8]);
            assert_eq!((a * b).to_array(), [6.0f32; 8]);
            assert_eq!((-a).to_array(), [-2.0f32; 8]);
        }
    }

    #[test]
    fn test_f32x8_fma() {
        if let Some(t) = Avx2Token::summon() {
            let a = f32x8::splat(t, 2.0);
            let b = f32x8::splat(t, 3.0);
            let c = f32x8::splat(t, 1.0);
            assert_eq!(a.mul_add(b, c).to_array(), [7.0f32; 8]);
        }
    }

    #[test]
    fn test_f32x8_comparison_blend() {
        if let Some(t) = Avx2Token::summon() {
            let a = f32x8::load(t, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            let thresh = f32x8::splat(t, 4.5);
            let mask = a.simd_gt(thresh);
            let big = f32x8::splat(t, 100.0);
            let result = mask.blend(big, a);
            assert_eq!(
                result.to_array(),
                [1.0, 2.0, 3.0, 4.0, 100.0, 100.0, 100.0, 100.0]
            );
        }
    }

    #[test]
    fn test_f32x8_mask_helpers() {
        if let Some(t) = Avx2Token::summon() {
            let a = f32x8::load(t, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            let mask = a.simd_gt(f32x8::splat(t, 4.0));
            assert!(mask.any());
            assert!(!mask.all());
            assert!(!mask.none());
            assert_eq!(mask.to_bitmask(), 0b11110000);
        }
    }

    #[test]
    fn test_f32x8_predicates() {
        if let Some(t) = Avx2Token::summon() {
            let nan = f32x8::splat(t, f32::NAN);
            let inf = f32x8::splat(t, f32::INFINITY);
            let normal = f32x8::splat(t, 1.0);

            assert!(nan.is_nan().all());
            assert!(inf.is_inf().all());
            assert!(normal.is_finite().all());
            assert!(!normal.is_nan().any());
        }
    }

    #[test]
    fn test_f32x8_int_conversion() {
        if let Some(t) = Avx2Token::summon() {
            let f = f32x8::load(t, &[1.4, 2.5, 3.6, 4.0, -1.4, -2.5, -3.6, -4.0]);
            let rounded = f.round_int().to_array();
            assert_eq!(rounded, [1, 2, 4, 4, -1, -2, -4, -4]); // round to nearest even
            let trunced = f.trunc_int().to_array();
            assert_eq!(trunced, [1, 2, 3, 4, -1, -2, -3, -4]);
        }
    }

    #[test]
    fn test_i32x8_basic() {
        if let Some(t) = Avx2Token::summon() {
            let a = i32x8::splat(t, 10);
            let b = i32x8::splat(t, 3);
            assert_eq!((a + b).to_array(), [13i32; 8]);
            assert_eq!((a - b).to_array(), [7i32; 8]);
            assert_eq!((a * b).to_array(), [30i32; 8]);
            assert_eq!((-a).to_array(), [-10i32; 8]);
        }
    }

    #[test]
    fn test_i32x8_shifts() {
        if let Some(t) = Avx2Token::summon() {
            let a = i32x8::splat(t, 16);
            assert_eq!((a << 2i32).to_array(), [64i32; 8]);
            assert_eq!((a >> 2i32).to_array(), [4i32; 8]);
        }
    }

    #[test]
    fn test_i32x8_to_f32x8() {
        if let Some(t) = Avx2Token::summon() {
            let i = i32x8::load(t, &[1, 2, 3, 4, 5, 6, 7, 8]);
            let f = i.to_f32x8();
            assert_eq!(f.to_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        }
    }

    #[test]
    fn test_f32x4_basic() {
        if let Some(t) = Sse2Token::summon() {
            let a = f32x4::splat(t, 2.0);
            let b = f32x4::splat(t, 3.0);
            assert_eq!((a + b).to_array(), [5.0f32; 4]);
        }
    }

    #[test]
    fn test_f64x4_basic() {
        if let Some(t) = Avx2Token::summon() {
            let a = f64x4::splat(t, 2.0);
            let b = f64x4::splat(t, 3.0);
            assert_eq!((a + b).to_array(), [5.0f64; 4]);
            assert_eq!(a.sqrt().to_array()[0], 2.0f64.sqrt());
        }
    }

    #[test]
    fn test_i16x16_saturating() {
        if let Some(t) = Avx2Token::summon() {
            let a = i16x16::splat(t, 32000);
            let b = i16x16::splat(t, 1000);
            assert_eq!(a.saturating_add(b).to_array(), [i16::MAX; 16]);
        }
    }

    #[test]
    fn test_u8x32_basic() {
        if let Some(t) = Avx2Token::summon() {
            let a = u8x32::splat(t, 100);
            let b = u8x32::splat(t, 200);
            assert_eq!(a.saturating_add(b).to_array(), [255u8; 32]);
            assert_eq!(a.min(b).to_array(), [100u8; 32]);
        }
    }

    #[test]
    fn test_as_array() {
        if let Some(t) = Avx2Token::summon() {
            let mut v = f32x8::load(t, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            assert_eq!(v.as_array()[0], 1.0);
            v.as_mut_array()[0] = 100.0;
            assert_eq!(v.to_array()[0], 100.0);
        }
    }

    #[test]
    fn test_no_escape_hatches() {
        // Compile-time guarantees - these would fail to compile:
        // let _: f32x8 = Default::default();
        // let _: f32x8 = [1.0f32; 8].into();
        // let _ = f32x8::from_raw(reg);
        if let Some(t) = Avx2Token::summon() {
            let _ = f32x8::zero(t); // Only way
        }
    }

    // ========================================================================
    // Wide parity tests - comprehensive coverage
    // ========================================================================

    #[test]
    fn test_f32x8_constants() {
        if let Some(t) = Avx2Token::summon() {
            let c = f32x8::consts(t);
            assert_eq!(c.one().to_array(), [1.0f32; 8]);
            assert_eq!(c.zero().to_array(), [0.0f32; 8]);
            assert_eq!(c.half().to_array(), [0.5f32; 8]);
            assert_eq!(c.neg_one().to_array(), [-1.0f32; 8]);
            assert_eq!(c.pi().to_array()[0], core::f32::consts::PI);
            assert_eq!(c.e().to_array()[0], core::f32::consts::E);
        }
    }

    #[test]
    fn test_i32x8_constants() {
        if let Some(t) = Avx2Token::summon() {
            let c = i32x8::consts(t);
            assert_eq!(c.one().to_array(), [1i32; 8]);
            assert_eq!(c.zero().to_array(), [0i32; 8]);
            assert_eq!(c.neg_one().to_array(), [-1i32; 8]);
            assert_eq!(c.max_value().to_array(), [i32::MAX; 8]);
            assert_eq!(c.min_value().to_array(), [i32::MIN; 8]);
        }
    }

    #[test]
    fn test_u32x8_constants() {
        if let Some(t) = Avx2Token::summon() {
            let c = u32x8::consts(t);
            assert_eq!(c.one().to_array(), [1u32; 8]);
            assert_eq!(c.zero().to_array(), [0u32; 8]);
            assert_eq!(c.max_value().to_array(), [u32::MAX; 8]);
        }
    }

    #[test]
    fn test_f32x8_scalar_ops() {
        if let Some(t) = Avx2Token::summon() {
            let a = f32x8::splat(t, 10.0);
            // f32x8 + f32
            assert_eq!((a + 5.0).to_array(), [15.0f32; 8]);
            assert_eq!((a - 3.0).to_array(), [7.0f32; 8]);
            assert_eq!((a * 2.0).to_array(), [20.0f32; 8]);
            assert_eq!((a / 2.0).to_array(), [5.0f32; 8]);
            // f32 + f32x8
            assert_eq!((5.0 + a).to_array(), [15.0f32; 8]);
            assert_eq!((30.0 - a).to_array(), [20.0f32; 8]);
            assert_eq!((2.0 * a).to_array(), [20.0f32; 8]);
            assert_eq!((100.0 / a).to_array(), [10.0f32; 8]);
        }
    }

    #[test]
    fn test_i32x8_scalar_ops() {
        if let Some(t) = Avx2Token::summon() {
            let a = i32x8::splat(t, 10);
            // i32x8 + i32
            assert_eq!((a + 5).to_array(), [15i32; 8]);
            assert_eq!((a - 3).to_array(), [7i32; 8]);
            assert_eq!((a * 2).to_array(), [20i32; 8]);
            // i32 + i32x8
            assert_eq!((5 + a).to_array(), [15i32; 8]);
            assert_eq!((30 - a).to_array(), [20i32; 8]);
            assert_eq!((2 * a).to_array(), [20i32; 8]);
        }
    }

    #[test]
    fn test_i32x8_per_lane_shift() {
        if let Some(t) = Avx2Token::summon() {
            let a = i32x8::load(t, &[1, 2, 4, 8, 16, 32, 64, 128]);
            let shifts = i32x8::load(t, &[0, 1, 2, 3, 0, 1, 2, 3]);
            // Per-lane left shift
            let left = a << shifts;
            assert_eq!(left.to_array(), [1, 4, 16, 64, 16, 64, 256, 1024]);
            // Per-lane right shift
            let right = a >> shifts;
            assert_eq!(right.to_array(), [1, 1, 1, 1, 16, 16, 16, 16]);
        }
    }

    #[test]
    fn test_u32x8_per_lane_shift() {
        if let Some(t) = Avx2Token::summon() {
            let a = u32x8::load(t, &[16, 32, 64, 128, 256, 512, 1024, 2048]);
            let shifts = u32x8::load(t, &[1, 2, 3, 4, 1, 2, 3, 4]);
            let left = a << shifts;
            assert_eq!(
                left.to_array(),
                [32, 128, 512, 2048, 512, 2048, 8192, 32768]
            );
            let right = a >> shifts;
            assert_eq!(right.to_array(), [8, 8, 8, 8, 128, 128, 128, 128]);
        }
    }

    #[test]
    fn test_i32x8_round_float() {
        if let Some(t) = Avx2Token::summon() {
            // Create a mask using comparison
            let a = i32x8::load(t, &[1, -1, 1, -1, 1, -1, 1, -1]);
            let zero = i32x8::zero(t);
            let mask = a.simd_gt(zero);
            // round_float reinterprets the bits as f32
            let float_mask = mask.round_float();
            // Mask should have negative sign bits set where mask is true
            let bitmask = float_mask.to_bitmask();
            assert_eq!(bitmask, 0b01010101); // lanes 0,2,4,6 are true (>0)
        }
    }

    #[test]
    fn test_f32x8_all_comparisons() {
        if let Some(t) = Avx2Token::summon() {
            let a = f32x8::load(t, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            let b = f32x8::load(t, &[2.0, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0]);

            // simd_eq: lanes where a == b
            assert_eq!(a.simd_eq(b).to_bitmask(), 0b00100010);
            // simd_ne: lanes where a != b
            assert_eq!(a.simd_ne(b).to_bitmask(), 0b11011101);
            // simd_lt: lanes where a < b
            assert_eq!(a.simd_lt(b).to_bitmask(), 0b00010001);
            // simd_le: lanes where a <= b
            assert_eq!(a.simd_le(b).to_bitmask(), 0b00110011);
            // simd_gt: lanes where a > b
            assert_eq!(a.simd_gt(b).to_bitmask(), 0b11001100);
            // simd_ge: lanes where a >= b
            assert_eq!(a.simd_ge(b).to_bitmask(), 0b11101110);
        }
    }

    #[test]
    fn test_i32x8_all_comparisons() {
        if let Some(t) = Avx2Token::summon() {
            let a = i32x8::load(t, &[1, 2, 3, 4, 5, 6, 7, 8]);
            let b = i32x8::load(t, &[2, 2, 2, 2, 6, 6, 6, 6]);

            assert_eq!(a.simd_eq(b).to_bitmask(), 0b00100010);
            assert_eq!(a.simd_ne(b).to_bitmask(), 0b11011101);
            assert_eq!(a.simd_lt(b).to_bitmask(), 0b00010001);
            assert_eq!(a.simd_le(b).to_bitmask(), 0b00110011);
            assert_eq!(a.simd_gt(b).to_bitmask(), 0b11001100);
            assert_eq!(a.simd_ge(b).to_bitmask(), 0b11101110);
        }
    }

    #[test]
    fn test_f32x8_sign_operations() {
        if let Some(t) = Avx2Token::summon() {
            let a = f32x8::load(t, &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]);
            let b = f32x8::load(t, &[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);

            // sign_bit returns -0.0 or 0.0
            let signs = a.sign_bit();
            assert_eq!(signs.to_bitmask(), 0b10101010);

            // copysign
            let copied = a.abs().copysign(b);
            assert_eq!(
                copied.to_array(),
                [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0]
            );

            // flip_signs
            let flipped = a.flip_signs(b);
            assert_eq!(
                flipped.to_array(),
                [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]
            );
        }
    }

    #[test]
    fn test_f32x8_math_ops() {
        if let Some(t) = Avx2Token::summon() {
            let a = f32x8::load(t, &[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]);

            // sqrt
            let sq = a.sqrt().to_array();
            assert_eq!(sq, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

            // abs
            let neg = f32x8::load(t, &[-1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0]);
            assert_eq!(
                neg.abs().to_array(),
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            );

            // min/max
            let b = f32x8::splat(t, 5.0);
            assert_eq!(
                a.min(b).to_array(),
                [1.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
            );
            assert_eq!(
                a.max(b).to_array(),
                [5.0, 5.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]
            );

            // clamp
            let lo = f32x8::splat(t, 10.0);
            let hi = f32x8::splat(t, 30.0);
            assert_eq!(
                a.clamp(lo, hi).to_array(),
                [10.0, 10.0, 10.0, 16.0, 25.0, 30.0, 30.0, 30.0]
            );
        }
    }

    #[test]
    fn test_f32x8_rounding() {
        if let Some(t) = Avx2Token::summon() {
            let a = f32x8::load(t, &[1.4, 1.5, 1.6, 2.5, -1.4, -1.5, -1.6, -2.5]);

            assert_eq!(
                a.floor().to_array(),
                [1.0, 1.0, 1.0, 2.0, -2.0, -2.0, -2.0, -3.0]
            );
            assert_eq!(
                a.ceil().to_array(),
                [2.0, 2.0, 2.0, 3.0, -1.0, -1.0, -1.0, -2.0]
            );
            assert_eq!(
                a.trunc().to_array(),
                [1.0, 1.0, 1.0, 2.0, -1.0, -1.0, -1.0, -2.0]
            );
            // round uses banker's rounding (round to nearest even)
            assert_eq!(
                a.round().to_array(),
                [1.0, 2.0, 2.0, 2.0, -1.0, -2.0, -2.0, -2.0]
            );
        }
    }

    #[test]
    fn test_f32x8_horizontal() {
        if let Some(t) = Avx2Token::summon() {
            let a = f32x8::load(t, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            assert_eq!(a.reduce_add(), 36.0);
        }
    }

    #[test]
    fn test_f32x8_fma_variants() {
        if let Some(t) = Avx2Token::summon() {
            let a = f32x8::splat(t, 2.0);
            let b = f32x8::splat(t, 3.0);
            let c = f32x8::splat(t, 1.0);

            // mul_add: a*b + c = 2*3 + 1 = 7
            assert_eq!(a.mul_add(b, c).to_array(), [7.0f32; 8]);
            // mul_sub: a*b - c = 2*3 - 1 = 5
            assert_eq!(a.mul_sub(b, c).to_array(), [5.0f32; 8]);
            // neg_mul_add: -(a*b) + c = -6 + 1 = -5
            assert_eq!(a.neg_mul_add(b, c).to_array(), [-5.0f32; 8]);
            // neg_mul_sub: -(a*b) - c = -6 - 1 = -7
            assert_eq!(a.neg_mul_sub(b, c).to_array(), [-7.0f32; 8]);
        }
    }

    #[test]
    fn test_f32x8_bitwise() {
        if let Some(t) = Avx2Token::summon() {
            let c = f32x8::consts(t);
            let all_ones = !c.zero();
            assert!(all_ones.all()); // all bits set = all lanes have sign bit set

            let a = f32x8::splat(t, 1.0);
            let b = f32x8::splat(t, 2.0);
            // These test the bitwise ops compile and run
            let _ = a & b;
            let _ = a | b;
            let _ = a ^ b;
        }
    }

    #[test]
    fn test_i32x8_bitwise() {
        if let Some(t) = Avx2Token::summon() {
            let a = i32x8::splat(t, 0b1100);
            let b = i32x8::splat(t, 0b1010);

            assert_eq!((a & b).to_array(), [0b1000i32; 8]);
            assert_eq!((a | b).to_array(), [0b1110i32; 8]);
            assert_eq!((a ^ b).to_array(), [0b0110i32; 8]);
            assert_eq!((!a).to_array(), [!0b1100i32; 8]);
        }
    }

    #[test]
    fn test_assign_ops() {
        if let Some(t) = Avx2Token::summon() {
            let c = f32x8::consts(t);
            let mut a = f32x8::splat(t, 10.0);
            a += f32x8::splat(t, 5.0);
            assert_eq!(a.to_array(), [15.0f32; 8]);
            a -= f32x8::splat(t, 3.0);
            assert_eq!(a.to_array(), [12.0f32; 8]);
            a *= f32x8::splat(t, 2.0);
            assert_eq!(a.to_array(), [24.0f32; 8]);
            a /= f32x8::splat(t, 4.0);
            assert_eq!(a.to_array(), [6.0f32; 8]);

            let mut b = f32x8::splat(t, 1.0);
            b &= c.one();
            b |= c.zero();
            b ^= c.zero();
        }
    }

    #[test]
    fn test_degrees_radians() {
        if let Some(t) = Avx2Token::summon() {
            let deg = f32x8::splat(t, 180.0);
            let rad = deg.to_radians();
            assert!((rad.to_array()[0] - core::f32::consts::PI).abs() < 1e-6);

            let rad2 = f32x8::splat(t, core::f32::consts::PI);
            let deg2 = rad2.to_degrees();
            assert!((deg2.to_array()[0] - 180.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_blend_i32x8() {
        if let Some(t) = Avx2Token::summon() {
            let a = i32x8::load(t, &[1, 2, 3, 4, 5, 6, 7, 8]);
            let mask = a.simd_gt(i32x8::splat(t, 4));
            let true_val = i32x8::splat(t, 100);
            let false_val = i32x8::splat(t, 0);
            let result = mask.blend(true_val, false_val);
            assert_eq!(result.to_array(), [0, 0, 0, 0, 100, 100, 100, 100]);
        }
    }

    // ========================================================================
    // Safety verification tests: ensure all intrinsics are safe inside
    // #[target_feature] functions (Rust 1.85+ behavior)
    // ========================================================================

    /// Verify all f32x8 intrinsics are safe inside #[target_feature(enable = "avx2,fma")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn f32x8_intrinsics_are_safe() -> f32x8 {
        use core::arch::x86_64::*;

        // Construction intrinsics
        let zero = _mm256_setzero_ps();
        let one = _mm256_set1_ps(1.0);
        let neg_zero = _mm256_set1_ps(-0.0);

        // Arithmetic (all safe inside target_feature)
        let sum = _mm256_add_ps(zero, one);
        let diff = _mm256_sub_ps(sum, zero);
        let prod = _mm256_mul_ps(diff, one);
        let quot = _mm256_div_ps(prod, one);

        // FMA operations
        let fma = _mm256_fmadd_ps(quot, one, zero);
        let fms = _mm256_fmsub_ps(fma, one, zero);
        let fnma = _mm256_fnmadd_ps(fms, one, zero);
        let fnms = _mm256_fnmsub_ps(fnma, one, zero);

        // Math operations
        let min = _mm256_min_ps(fnms, one);
        let max = _mm256_max_ps(min, zero);
        let sqrt = _mm256_sqrt_ps(max);
        let rcp = _mm256_rcp_ps(sqrt);
        let rsqrt = _mm256_rsqrt_ps(rcp);

        // Rounding
        let round = _mm256_round_ps(rsqrt, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let floor = _mm256_floor_ps(round);
        let ceil = _mm256_ceil_ps(floor);
        let trunc = _mm256_round_ps(ceil, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);

        // Bitwise
        let mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
        let and = _mm256_and_ps(trunc, mask);
        let or = _mm256_or_ps(and, zero);
        let xor = _mm256_xor_ps(or, neg_zero);

        // Comparisons
        let cmp_eq = _mm256_cmp_ps(xor, one, _CMP_EQ_OQ);
        let cmp_lt = _mm256_cmp_ps(cmp_eq, one, _CMP_LT_OQ);
        let cmp_le = _mm256_cmp_ps(cmp_lt, one, _CMP_LE_OQ);
        let cmp_gt = _mm256_cmp_ps(cmp_le, one, _CMP_GT_OQ);
        let cmp_ge = _mm256_cmp_ps(cmp_gt, one, _CMP_GE_OQ);
        let cmp_neq = _mm256_cmp_ps(cmp_ge, one, _CMP_NEQ_OQ);
        let cmp_unord = _mm256_cmp_ps(cmp_neq, cmp_neq, _CMP_UNORD_Q);

        // Blend
        let blend = _mm256_blendv_ps(cmp_unord, one, zero);

        // Horizontal reduction helpers
        let hi = _mm256_extractf128_ps(blend, 1);
        let lo = _mm256_castps256_ps128(blend);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf1 = _mm_movehdup_ps(sum128);
        let sum2 = _mm_add_ps(sum128, shuf1);
        let shuf2 = _mm_movehl_ps(sum2, sum2);
        let final_sum = _mm_add_ss(sum2, shuf2);
        let _scalar = _mm_cvtss_f32(final_sum);

        // Mask extraction
        let _bitmask = _mm256_movemask_ps(blend);

        // Int conversion
        let as_int = _mm256_cvtps_epi32(blend);
        let back = _mm256_cvtepi32_ps(as_int);
        let trunc_int = _mm256_cvttps_epi32(back);
        let _back2 = _mm256_cvtepi32_ps(trunc_int);

        // Cast operations
        let as_si = _mm256_castps_si256(blend);
        let _back_ps = _mm256_castsi256_ps(as_si);

        f32x8(blend)
    }

    /// Verify all i32x8 intrinsics are safe inside #[target_feature(enable = "avx2")]
    #[target_feature(enable = "avx2")]
    unsafe fn i32x8_intrinsics_are_safe() -> i32x8 {
        use core::arch::x86_64::*;

        // Construction
        let zero = _mm256_setzero_si256();
        let one = _mm256_set1_epi32(1);
        let neg_one = _mm256_set1_epi32(-1);

        // Arithmetic
        let sum = _mm256_add_epi32(zero, one);
        let diff = _mm256_sub_epi32(sum, zero);
        let prod = _mm256_mullo_epi32(diff, one);

        // Min/max/abs
        let min = _mm256_min_epi32(prod, one);
        let max = _mm256_max_epi32(min, zero);
        let abs = _mm256_abs_epi32(max);

        // Bitwise
        let and = _mm256_and_si256(abs, neg_one);
        let or = _mm256_or_si256(and, zero);
        let xor = _mm256_xor_si256(or, zero);

        // Shifts (variable - AVX2)
        let shift_amt = _mm256_set1_epi32(2);
        let shl = _mm256_sllv_epi32(xor, shift_amt);
        let shr_arith = _mm256_srav_epi32(shl, shift_amt);
        let shr_logic = _mm256_srlv_epi32(shr_arith, shift_amt);

        // Comparisons
        let cmp_eq = _mm256_cmpeq_epi32(shr_logic, one);
        let cmp_gt = _mm256_cmpgt_epi32(cmp_eq, zero);

        // Blend
        let blend = _mm256_blendv_epi8(cmp_gt, one, zero);

        // Mask (via float cast)
        let as_ps = _mm256_castsi256_ps(blend);
        let _bitmask = _mm256_movemask_ps(as_ps);

        i32x8(blend)
    }

    /// Verify all u32x8 intrinsics are safe inside #[target_feature(enable = "avx2")]
    #[target_feature(enable = "avx2")]
    unsafe fn u32x8_intrinsics_are_safe() -> u32x8 {
        use core::arch::x86_64::*;

        // Construction (same as i32x8, just interpreted as unsigned)
        let zero = _mm256_setzero_si256();
        let one = _mm256_set1_epi32(1);

        // Arithmetic (same intrinsics as signed)
        let sum = _mm256_add_epi32(zero, one);
        let diff = _mm256_sub_epi32(sum, zero);

        // Unsigned min/max
        let min = _mm256_min_epu32(diff, one);
        let max = _mm256_max_epu32(min, zero);

        // Shifts (logical for unsigned)
        let shift_amt = _mm256_set1_epi32(2);
        let shl = _mm256_sllv_epi32(max, shift_amt);
        let shr = _mm256_srlv_epi32(shl, shift_amt);

        // Unsigned comparison via sign-flip trick
        let flip = _mm256_set1_epi32(i32::MIN);
        let a_flipped = _mm256_xor_si256(shr, flip);
        let b_flipped = _mm256_xor_si256(one, flip);
        let _cmp_gt = _mm256_cmpgt_epi32(a_flipped, b_flipped);

        u32x8(shr)
    }

    /// Verify all f32x4 intrinsics are safe inside #[target_feature(enable = "sse4.1")]
    #[target_feature(enable = "sse4.1")]
    unsafe fn f32x4_intrinsics_are_safe() -> f32x4 {
        use core::arch::x86_64::*;

        // Construction
        let zero = _mm_setzero_ps();
        let one = _mm_set1_ps(1.0);
        let neg_zero = _mm_set1_ps(-0.0);

        // Arithmetic
        let sum = _mm_add_ps(zero, one);
        let diff = _mm_sub_ps(sum, zero);
        let prod = _mm_mul_ps(diff, one);
        let quot = _mm_div_ps(prod, one);

        // Math
        let min = _mm_min_ps(quot, one);
        let max = _mm_max_ps(min, zero);
        let sqrt = _mm_sqrt_ps(max);
        let rcp = _mm_rcp_ps(sqrt);
        let rsqrt = _mm_rsqrt_ps(rcp);

        // Rounding (SSE4.1)
        let round = _mm_round_ps(rsqrt, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let floor = _mm_floor_ps(round);
        let ceil = _mm_ceil_ps(floor);
        let trunc = _mm_round_ps(ceil, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);

        // Bitwise
        let mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFF_FFFF));
        let and = _mm_and_ps(trunc, mask);
        let or = _mm_or_ps(and, zero);
        let xor = _mm_xor_ps(or, neg_zero);

        // Comparisons
        let cmp_eq = _mm_cmpeq_ps(xor, one);
        let cmp_lt = _mm_cmplt_ps(cmp_eq, one);
        let cmp_le = _mm_cmple_ps(cmp_lt, one);
        let cmp_gt = _mm_cmpgt_ps(cmp_le, one);
        let cmp_ge = _mm_cmpge_ps(cmp_gt, one);
        let cmp_neq = _mm_cmpneq_ps(cmp_ge, one);

        // Blend (SSE4.1)
        let blend = _mm_blendv_ps(cmp_neq, one, zero);

        // Horizontal
        let shuf1 = _mm_movehdup_ps(blend);
        let sum1 = _mm_add_ps(blend, shuf1);
        let shuf2 = _mm_movehl_ps(sum1, sum1);
        let _final = _mm_add_ss(sum1, shuf2);

        // Mask
        let _bitmask = _mm_movemask_ps(blend);

        // Int conversion
        let as_int = _mm_cvtps_epi32(blend);
        let back = _mm_cvtepi32_ps(as_int);
        let trunc_int = _mm_cvttps_epi32(back);
        let _back2 = _mm_cvtepi32_ps(trunc_int);

        f32x4(blend)
    }

    /// Verify all i32x4 intrinsics are safe inside #[target_feature(enable = "sse4.1")]
    #[target_feature(enable = "sse4.1")]
    unsafe fn i32x4_intrinsics_are_safe() -> i32x4 {
        use core::arch::x86_64::*;

        // Construction
        let zero = _mm_setzero_si128();
        let one = _mm_set1_epi32(1);
        let neg_one = _mm_set1_epi32(-1);

        // Arithmetic
        let sum = _mm_add_epi32(zero, one);
        let diff = _mm_sub_epi32(sum, zero);
        let prod = _mm_mullo_epi32(diff, one); // SSE4.1

        // Min/max/abs (SSE4.1)
        let min = _mm_min_epi32(prod, one);
        let max = _mm_max_epi32(min, zero);
        let abs = _mm_abs_epi32(max);

        // Bitwise
        let and = _mm_and_si128(abs, neg_one);
        let or = _mm_or_si128(and, zero);
        let xor = _mm_xor_si128(or, zero);

        // Comparisons
        let cmp_eq = _mm_cmpeq_epi32(xor, one);
        let cmp_gt = _mm_cmpgt_epi32(cmp_eq, zero);

        i32x4(cmp_gt)
    }

    #[test]
    fn test_f32x8_intrinsics_safe_in_target_feature() {
        if Avx2Token::summon().is_some() {
            let result = unsafe { f32x8_intrinsics_are_safe() };
            // Just verify it compiles and runs - exact value doesn't matter
            let _ = result.to_array();
        }
    }

    #[test]
    fn test_i32x8_intrinsics_safe_in_target_feature() {
        if Avx2Token::summon().is_some() {
            let result = unsafe { i32x8_intrinsics_are_safe() };
            let _ = result.to_array();
        }
    }

    #[test]
    fn test_u32x8_intrinsics_safe_in_target_feature() {
        if Avx2Token::summon().is_some() {
            let result = unsafe { u32x8_intrinsics_are_safe() };
            let _ = result.to_array();
        }
    }

    #[test]
    fn test_f32x4_intrinsics_safe_in_target_feature() {
        if Sse2Token::summon().is_some() {
            let result = unsafe { f32x4_intrinsics_are_safe() };
            let _ = result.to_array();
        }
    }

    #[test]
    fn test_i32x4_intrinsics_safe_in_target_feature() {
        if Sse2Token::summon().is_some() {
            let result = unsafe { i32x4_intrinsics_are_safe() };
            let _ = result.to_array();
        }
    }

    // ========================================================================
    // AVX-512 safety verification tests
    // ========================================================================

    /// Verify all f32x16 AVX-512 intrinsics are safe inside #[target_feature(enable = "avx512f,avx512dq")]
    #[target_feature(enable = "avx512f,avx512dq")]
    unsafe fn f32x16_intrinsics_are_safe() -> f32x16 {
        use core::arch::x86_64::*;

        // Construction
        let zero = _mm512_setzero_ps();
        let one = _mm512_set1_ps(1.0);
        let neg_zero = _mm512_set1_ps(-0.0);

        // Arithmetic
        let sum = _mm512_add_ps(zero, one);
        let diff = _mm512_sub_ps(sum, zero);
        let prod = _mm512_mul_ps(diff, one);
        let quot = _mm512_div_ps(prod, one);

        // FMA
        let fma = _mm512_fmadd_ps(quot, one, zero);
        let fms = _mm512_fmsub_ps(fma, one, zero);
        let fnma = _mm512_fnmadd_ps(fms, one, zero);
        let fnms = _mm512_fnmsub_ps(fnma, one, zero);

        // Math
        let min = _mm512_min_ps(fnms, one);
        let max = _mm512_max_ps(min, zero);
        let abs = _mm512_abs_ps(max);
        let sqrt = _mm512_sqrt_ps(abs);
        let rcp = _mm512_rcp14_ps(sqrt);
        let rsqrt = _mm512_rsqrt14_ps(rcp);

        // Rounding
        let round = _mm512_roundscale_ps(rsqrt, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let floor = _mm512_roundscale_ps(round, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        let ceil = _mm512_roundscale_ps(floor, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
        let trunc = _mm512_roundscale_ps(ceil, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);

        // Bitwise (avx512dq)
        let and = _mm512_and_ps(trunc, one);
        let or = _mm512_or_ps(and, zero);
        let xor = _mm512_xor_ps(or, neg_zero);

        // Comparisons (return k-mask)
        let cmp_eq_mask = _mm512_cmp_ps_mask(xor, one, _CMP_EQ_OQ);
        let cmp_lt_mask = _mm512_cmp_ps_mask(xor, one, _CMP_LT_OQ);
        let cmp_gt_mask = _mm512_cmp_ps_mask(xor, one, _CMP_GT_OQ);

        // Masked move
        let masked = _mm512_maskz_mov_ps(cmp_eq_mask, one);
        let _ = cmp_lt_mask;
        let _ = cmp_gt_mask;

        // Horizontal
        let _reduce = _mm512_reduce_add_ps(masked);

        // Mask extraction (avx512dq)
        let as_si = _mm512_castps_si512(masked);
        let _bitmask = _mm512_movepi32_mask(as_si);

        // Int conversion
        let as_int = _mm512_cvtps_epi32(masked);
        let back = _mm512_cvtepi32_ps(as_int);
        let trunc_int = _mm512_cvttps_epi32(back);
        let _back2 = _mm512_cvtepi32_ps(trunc_int);

        f32x16(masked)
    }

    /// Verify all i32x16 AVX-512 intrinsics are safe inside #[target_feature(enable = "avx512f,avx512dq")]
    #[target_feature(enable = "avx512f,avx512dq")]
    unsafe fn i32x16_intrinsics_are_safe() -> i32x16 {
        use core::arch::x86_64::*;

        // Construction
        let zero = _mm512_setzero_si512();
        let one = _mm512_set1_epi32(1);
        let neg_one = _mm512_set1_epi32(-1);

        // Arithmetic
        let sum = _mm512_add_epi32(zero, one);
        let diff = _mm512_sub_epi32(sum, zero);
        let prod = _mm512_mullo_epi32(diff, one);

        // Min/max/abs
        let min = _mm512_min_epi32(prod, one);
        let max = _mm512_max_epi32(min, zero);
        let abs = _mm512_abs_epi32(max);

        // Bitwise
        let and = _mm512_and_si512(abs, neg_one);
        let or = _mm512_or_si512(and, zero);
        let xor = _mm512_xor_si512(or, zero);

        // Shifts (variable)
        let shift_amt = _mm512_set1_epi32(2);
        let shl = _mm512_sllv_epi32(xor, shift_amt);
        let shr = _mm512_srav_epi32(shl, shift_amt);

        // Comparisons (return k-mask)
        let cmp_eq_mask = _mm512_cmpeq_epi32_mask(shr, one);
        let cmp_gt_mask = _mm512_cmpgt_epi32_mask(shr, zero);

        // Masked move
        let masked = _mm512_maskz_mov_epi32(cmp_eq_mask, neg_one);
        let _ = cmp_gt_mask;

        // Mask extraction (avx512dq)
        let _bitmask = _mm512_movepi32_mask(masked);

        i32x16(masked)
    }

    #[test]
    fn test_f32x16_intrinsics_safe_in_target_feature() {
        // X64V4Token includes avx512dq (avx512f-only token doesn't have DQ)
        if crate::X64V4Token::summon().is_some() {
            let result = unsafe { f32x16_intrinsics_are_safe() };
            let _ = result.to_array();
        }
    }

    #[test]
    fn test_i32x16_intrinsics_safe_in_target_feature() {
        // X64V4Token includes avx512dq (avx512f-only token doesn't have DQ)
        if crate::X64V4Token::summon().is_some() {
            let result = unsafe { i32x16_intrinsics_are_safe() };
            let _ = result.to_array();
        }
    }

    // Basic functional tests for AVX-512DQ types
    #[test]
    fn test_f32x16_basic() {
        if let Some(t) = crate::X64V4Token::summon() {
            let a = f32x16::splat(t, 2.0);
            let b = f32x16::splat(t, 3.0);
            assert_eq!((a + b).to_array(), [5.0f32; 16]);
            assert_eq!((a * b).to_array(), [6.0f32; 16]);
            assert_eq!((-a).to_array(), [-2.0f32; 16]);
        }
    }

    #[test]
    fn test_f32x16_fma() {
        if let Some(t) = crate::X64V4Token::summon() {
            let a = f32x16::splat(t, 2.0);
            let b = f32x16::splat(t, 3.0);
            let c = f32x16::splat(t, 1.0);
            assert_eq!(a.mul_add(b, c).to_array(), [7.0f32; 16]);
        }
    }

    #[test]
    fn test_i32x16_basic() {
        if let Some(t) = crate::X64V4Token::summon() {
            let a = i32x16::splat(t, 10);
            let b = i32x16::splat(t, 3);
            assert_eq!((a + b).to_array(), [13i32; 16]);
            assert_eq!((a - b).to_array(), [7i32; 16]);
            assert_eq!((a * b).to_array(), [30i32; 16]);
            assert_eq!((-a).to_array(), [-10i32; 16]);
        }
    }

    #[test]
    fn test_i32x16_shifts() {
        if let Some(t) = crate::X64V4Token::summon() {
            let a = i32x16::splat(t, 16);
            assert_eq!((a << 2i32).to_array(), [64i32; 16]);
            assert_eq!((a >> 2i32).to_array(), [4i32; 16]);
        }
    }

    #[test]
    fn test_f32x16_constants() {
        if let Some(t) = crate::X64V4Token::summon() {
            let c = f32x16::consts(t);
            assert_eq!(c.one().to_array(), [1.0f32; 16]);
            assert_eq!(c.zero().to_array(), [0.0f32; 16]);
            assert_eq!(c.half().to_array(), [0.5f32; 16]);
        }
    }
}

// ============================================================================
// NEON safety verification tests (aarch64)
// ============================================================================

#[cfg(all(test, target_arch = "aarch64"))]
mod neon_tests {
    use super::*;
    use crate::NeonToken;
    use crate::tokens::SimdToken;

    /// Verify all f32x4 NEON intrinsics are safe inside #[target_feature(enable = "neon")]
    #[target_feature(enable = "neon")]
    unsafe fn f32x4_neon_intrinsics_are_safe() -> f32x4 {
        use core::arch::aarch64::*;

        // Construction
        let zero = vdupq_n_f32(0.0);
        let one = vdupq_n_f32(1.0);

        // Arithmetic
        let sum = vaddq_f32(zero, one);
        let diff = vsubq_f32(sum, zero);
        let prod = vmulq_f32(diff, one);
        let quot = vdivq_f32(prod, one);

        // Math
        let min = vminq_f32(quot, one);
        let max = vmaxq_f32(min, zero);
        let abs = vabsq_f32(max);
        let sqrt = vsqrtq_f32(abs);
        let rcp = vrecpeq_f32(sqrt);
        let rsqrt = vrsqrteq_f32(rcp);

        // Rounding
        let round = vrndnq_f32(rsqrt);
        let floor = vrndmq_f32(round);
        let ceil = vrndpq_f32(floor);
        let trunc = vrndq_f32(ceil);

        // Negation
        let neg = vnegq_f32(trunc);

        // Bitwise (via reinterpret)
        let as_u32 = vreinterpretq_u32_f32(neg);
        let and = vandq_u32(as_u32, as_u32);
        let or = vorrq_u32(and, and);
        let xor = veorq_u32(or, or);
        let not = vmvnq_u32(xor);
        let back = vreinterpretq_f32_u32(not);

        // Comparisons
        let cmp_eq = vceqq_f32(back, one);
        let cmp_lt = vcltq_f32(vreinterpretq_f32_u32(cmp_eq), one);
        let cmp_le = vcleq_f32(vreinterpretq_f32_u32(cmp_lt), one);
        let cmp_gt = vcgtq_f32(vreinterpretq_f32_u32(cmp_le), one);
        let cmp_ge = vcgeq_f32(vreinterpretq_f32_u32(cmp_gt), one);

        // Blend
        let blend = vbslq_f32(cmp_ge, one, zero);

        // Horizontal reduction
        let _sum = vaddvq_f32(blend);

        // Mask helpers
        let mask_u32 = vreinterpretq_u32_f32(blend);
        let _max_lane = vmaxvq_u32(mask_u32);
        let _min_lane = vminvq_u32(mask_u32);
        let _shifted = vshrq_n_u32(mask_u32, 31);

        // Int conversion
        let as_int = vcvtnq_s32_f32(blend);
        let back_f = vcvtq_f32_s32(as_int);
        let trunc_int = vcvtq_s32_f32(back_f);
        let _back2 = vcvtq_f32_s32(trunc_int);

        f32x4(blend)
    }

    /// Verify all i32x4 NEON intrinsics are safe inside #[target_feature(enable = "neon")]
    #[target_feature(enable = "neon")]
    unsafe fn i32x4_neon_intrinsics_are_safe() -> i32x4 {
        use core::arch::aarch64::*;

        // Construction
        let zero = vdupq_n_s32(0);
        let one = vdupq_n_s32(1);

        // Arithmetic
        let sum = vaddq_s32(zero, one);
        let diff = vsubq_s32(sum, zero);
        let prod = vmulq_s32(diff, one);

        // Min/max/abs
        let min = vminq_s32(prod, one);
        let max = vmaxq_s32(min, zero);
        let abs = vabsq_s32(max);

        // Negation
        let neg = vnegq_s32(abs);

        // Bitwise
        let and = vandq_s32(neg, one);
        let or = vorrq_s32(and, zero);
        let xor = veorq_s32(or, zero);

        // Bitwise NOT via u32
        let as_u32 = vreinterpretq_u32_s32(xor);
        let not = vmvnq_u32(as_u32);
        let back = vreinterpretq_s32_u32(not);

        // Shifts
        let shift_amt = vdupq_n_s32(2);
        let shl = vshlq_s32(back, shift_amt);
        let neg_shift = vdupq_n_s32(-2);
        let shr = vshlq_s32(shl, neg_shift);

        // Comparisons
        let cmp_eq = vceqq_s32(shr, one);
        let cmp_gt = vcgtq_s32(vreinterpretq_s32_u32(cmp_eq), zero);

        // Blend
        let mask = vreinterpretq_u32_s32(vreinterpretq_s32_u32(cmp_gt));
        let blend = vbslq_u32(
            mask,
            vreinterpretq_u32_s32(one),
            vreinterpretq_u32_s32(zero),
        );

        // Mask helpers
        let _max_lane = vmaxvq_u32(blend);
        let _min_lane = vminvq_u32(blend);

        i32x4(vreinterpretq_s32_u32(blend))
    }

    #[test]
    fn test_f32x4_neon_intrinsics_safe_in_target_feature() {
        if NeonToken::summon().is_some() {
            let result = unsafe { f32x4_neon_intrinsics_are_safe() };
            let _ = result.to_array();
        }
    }

    #[test]
    fn test_i32x4_neon_intrinsics_safe_in_target_feature() {
        if NeonToken::summon().is_some() {
            let result = unsafe { i32x4_neon_intrinsics_are_safe() };
            let _ = result.to_array();
        }
    }
}

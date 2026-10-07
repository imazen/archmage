//! Generic `f64x2<T>` — 2-lane f64 SIMD vector parameterized by backend.
//!
//! `T` is a token type (e.g., `X64V3Token`, `NeonToken`, `ScalarToken`)
//! that determines the platform-native representation and intrinsics used.
//! The struct delegates all operations to the [`F64x2Backend`] trait.

#![allow(clippy::should_implement_trait)]

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::simd::backends::F64x2Backend;

/// 2-lane f64 SIMD vector, generic over backend `T`.
///
/// `T` is a token type that proves CPU support for the required SIMD features.
/// The inner representation is `T::Repr` (e.g., `__m128d` on x86, `float64x2_t` on ARM).
///
/// **The token is stored** (as a zero-sized field) so methods receiving
/// `self: f64x2<T>` can re-supply it to backend operations that
/// require a token value (e.g. `T::splat(token, v)`). This carries the
/// token-as-feature-proof guarantee through every method call without
/// runtime overhead — `T` is ZST, so `sizeof(f64x2<T>) == sizeof(T::Repr)`
/// and `align_of(f64x2<T>) == align_of(T::Repr)` under `#[repr(C)]`.
///
/// # Layout
///
/// `#[repr(C)]` with a ZST trailing field: `T::Repr` lives at offset 0
/// and `T` is a 0-byte tail. Bitcasts between `f64x2<T>` values of
/// different element-types are sound when the Repr types share a layout
/// (e.g. `__m128` and `__m128i` are both 16-byte aligned 128-bit values).
/// `#[repr(transparent)]` cannot be used because Rust cannot prove at
/// the struct definition site that a generic `T` is a 1-ZST.
///
/// Construction requires a token value to prove CPU support at runtime.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct f64x2<T: F64x2Backend>(pub(crate) T::Repr, pub(crate) T);
// The `unsafe impl` and the checks behind it live in `simd_storage`.
crate::simd_storage::impl_token_storage!(f64x2, F64x2Backend);

// Layout invariant: struct is `#[repr(C)]` with a trailing ZST `T`
// field, so `sizeof/alignof(f64x2<T>) == sizeof/alignof(T::Repr)`
// when `T` is a 1-ZST. Every archmage token currently satisfies this;
// if a future refactor adds a non-ZST field to a token, this const
// assert fires at compile time.
const _: () = {
    assert!(
        core::mem::size_of::<f64x2<archmage::ScalarToken>>()
            == core::mem::size_of::<
                <archmage::ScalarToken as crate::simd::backends::F64x2Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<f64x2<archmage::ScalarToken>>()
            == core::mem::align_of::<
                <archmage::ScalarToken as crate::simd::backends::F64x2Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "x86_64")]
const _: () = {
    assert!(
        core::mem::size_of::<f64x2<archmage::X64V3Token>>()
            == core::mem::size_of::<
                <archmage::X64V3Token as crate::simd::backends::F64x2Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<f64x2<archmage::X64V3Token>>()
            == core::mem::align_of::<
                <archmage::X64V3Token as crate::simd::backends::F64x2Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "aarch64")]
const _: () = {
    assert!(
        core::mem::size_of::<f64x2<archmage::NeonToken>>()
            == core::mem::size_of::<
                <archmage::NeonToken as crate::simd::backends::F64x2Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<f64x2<archmage::NeonToken>>()
            == core::mem::align_of::<
                <archmage::NeonToken as crate::simd::backends::F64x2Backend>::Repr,
            >()
    );
};

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const _: () = {
    assert!(
        core::mem::size_of::<f64x2<archmage::Wasm128Token>>()
            == core::mem::size_of::<
                <archmage::Wasm128Token as crate::simd::backends::F64x2Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<f64x2<archmage::Wasm128Token>>()
            == core::mem::align_of::<
                <archmage::Wasm128Token as crate::simd::backends::F64x2Backend>::Repr,
            >()
    );
};

impl<T: F64x2Backend> f64x2<T> {
    /// Number of f64 lanes.
    pub const LANES: usize = 2;

    // ====== Construction (token-gated) ======

    /// Broadcast scalar to all 2 lanes.
    #[inline(always)]
    pub fn splat_t(token: T, v: f64) -> Self {
        Self(T::splat(token, v), token)
    }

    /// All lanes zero.
    #[inline(always)]
    pub fn zero_t(token: T) -> Self {
        Self(T::zero(token), token)
    }

    /// Load from a `[f64; 2]` array.
    #[inline(always)]
    pub fn load_t(token: T, data: &[f64; 2]) -> Self {
        Self(T::load(token, data), token)
    }

    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub fn from_array_t(token: T, arr: [f64; 2]) -> Self {
        Self(T::from_array(token, arr), token)
    }

    /// Create from slice. Panics if `slice.len() < 2`.
    #[inline(always)]
    pub fn from_slice_t(token: T, slice: &[f64]) -> Self {
        let arr: [f64; 2] = slice[..2].try_into().unwrap();
        Self(T::from_array(token, arr), token)
    }

    /// Split a slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&[[f64; 2]], &[f64])` — fixed-size arrays suitable
    /// for [`load_t`](Self::load_t), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice_t(_: T, data: &[f64]) -> (&[[f64; 2]], &[f64]) {
        data.as_chunks::<2>()
    }

    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&mut [[f64; 2]], &mut [f64])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load_t`](Self::load_t), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice_mut_t(_: T, data: &mut [f64]) -> (&mut [[f64; 2]], &mut [f64]) {
        data.as_chunks_mut::<2>()
    }

    // ====== Accessors ======

    /// Store to array.
    #[inline(always)]
    pub fn store(self, out: &mut [f64; 2]) {
        T::store(self.1, self.0, out);
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [f64; 2] {
        T::to_array(self.1, self.0)
    }

    /// Get the underlying platform representation.
    #[inline(always)]
    pub fn into_repr(self) -> T::Repr {
        self.0
    }

    /// Wrap a platform representation (token-gated).
    #[inline(always)]
    pub fn from_repr_t(token: T, repr: T::Repr) -> Self {
        Self(repr, token)
    }

    /// Wrap a repr with a token. Used by cross-type/cross-width helpers
    /// in `simd::generic::*` where the token is already proven by the
    /// caller's wider input type.
    #[inline(always)]
    #[allow(dead_code)]
    pub(crate) fn from_repr_unchecked(token: T, repr: T::Repr) -> Self {
        Self(repr, token)
    }

    // ====== Math ======

    /// Lane-wise minimum.
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(T::min(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise maximum.
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(T::max(self.1, self.0, other.0), self.1)
    }

    /// Clamp between lo and hi.
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        Self(T::clamp(self.1, self.0, lo.0, hi.0), self.1)
    }

    /// Square root.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(T::sqrt(self.1, self.0), self.1)
    }

    /// Absolute value.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(T::abs(self.1, self.0), self.1)
    }

    /// Round toward negative infinity.
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(T::floor(self.1, self.0), self.1)
    }

    /// Round toward positive infinity.
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self(T::ceil(self.1, self.0), self.1)
    }

    /// Round to nearest integer.
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(T::round(self.1, self.0), self.1)
    }

    /// Multiply-add: `self * a + b`, fused where the hardware fuses.
    ///
    /// For one rounding on every backend, use
    /// [`mul_add_portable`](Self::mul_add_portable).
    ///
    /// x86 v3/v4 and NEON use FMA, one rounding: about as fast as
    /// `self * a + b` in streaming loops, and 24–38% less time in
    /// dependency chains such as Horner polynomials. The scalar backend
    /// and WASM built without `relaxed-simd` multiply, round, add and
    /// round again, at the speed of `self * a + b`. Builds with
    /// `relaxed-simd` use the engine's madd, which may round either way.
    /// Results can therefore differ between backends in the last bit.
    /// NaN payload/sign are unspecified.
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(T::mul_add(self.1, self.0, a.0, b.0), self.1)
    }

    /// Multiply-sub: `self * a - b`.
    ///
    /// Same rounding contract as [`mul_add`](Self::mul_add). For one
    /// rounding on every backend, use
    /// [`mul_sub_portable`](Self::mul_sub_portable).
    #[inline(always)]
    pub fn mul_sub(self, a: Self, b: Self) -> Self {
        Self(T::mul_sub(self.1, self.0, a.0, b.0), self.1)
    }

    /// Multiply-add with one rounding on every backend: `self * a + b`.
    ///
    /// The same result on every backend, NaN payload/sign aside, as
    /// `f32::mul_add` and `f64::mul_add` give in std. x86 v3/v4 and NEON
    /// use FMA, at the cost of [`mul_add`](Self::mul_add). The scalar
    /// backend and WASM fuse in software, relaxed SIMD included because
    /// relaxed madd may round twice. That costs 2.7–29.5× the time of
    /// `self * a + b` on the scalar backend and 8.2× (`f32x4`) to 24×
    /// (`f64x2`) under wasmtime.
    /// [Measurements](https://github.com/imazen/archmage/blob/main/benchmarks/mul_add_portable_zen5-m4pro_2026-10-05.md).
    #[inline(always)]
    pub fn mul_add_portable(self, a: Self, b: Self) -> Self {
        Self(T::mul_add_portable(self.1, self.0, a.0, b.0), self.1)
    }

    /// Multiply-sub with one rounding on every backend: `self * a - b`.
    ///
    /// Same contract and cost as
    /// [`mul_add_portable`](Self::mul_add_portable).
    #[inline(always)]
    pub fn mul_sub_portable(self, a: Self, b: Self) -> Self {
        Self(T::mul_sub_portable(self.1, self.0, a.0, b.0), self.1)
    }

    // ====== Comparisons ======

    /// Lane-wise equality (returns mask).
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(T::simd_eq(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise inequality (returns mask).
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(T::simd_ne(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise less-than (returns mask).
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(T::simd_lt(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise less-than-or-equal (returns mask).
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(T::simd_le(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise greater-than (returns mask).
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(T::simd_gt(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise greater-than-or-equal (returns mask).
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(T::simd_ge(self.1, self.0, other.0), self.1)
    }

    /// Select lanes: where mask is all-1s pick `if_true`, else `if_false`.
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(T::blend(mask.1, mask.0, if_true.0, if_false.0), mask.1)
    }

    // ====== Reductions ======

    /// Sum all 2 lanes.
    #[inline(always)]
    pub fn reduce_add(self) -> f64 {
        T::reduce_add(self.1, self.0)
    }

    /// Minimum across all 2 lanes.
    #[inline(always)]
    pub fn reduce_min(self) -> f64 {
        T::reduce_min(self.1, self.0)
    }

    /// Maximum across all 2 lanes.
    #[inline(always)]
    pub fn reduce_max(self) -> f64 {
        T::reduce_max(self.1, self.0)
    }

    // ====== Approximations ======

    /// Fast reciprocal approximation (1/x): the backend's native estimate
    /// (exact division on f64 — no hardware estimate exists).
    #[inline(always)]
    pub fn rcp_approx(self) -> Self {
        Self(T::rcp_approx(self.1, self.0), self.1)
    }

    /// Precise reciprocal (1/x): exact IEEE division on every backend.
    ///
    /// Correctly rounded (0 ULP vs `1.0 / x`), including the rails:
    /// `recip(±0) = ±inf` and `recip(±inf) = ±0` (issue #64). On f64
    /// the working tier and the exact tier coincide.
    #[inline(always)]
    pub fn recip(self) -> Self {
        Self(T::recip(self.1, self.0), self.1)
    }

    /// Fast reciprocal square root approximation (exact division +
    /// sqrt on f64 — no hardware estimate exists).
    #[inline(always)]
    pub fn rsqrt_approx(self) -> Self {
        Self(T::rsqrt_approx(self.1, self.0), self.1)
    }

    /// Precise reciprocal square root (1/sqrt(x)): exact IEEE division
    /// and square root on every backend. Bit-exact vs scalar
    /// `1.0 / x.sqrt()`, rails included.
    #[inline(always)]
    pub fn rsqrt(self) -> Self {
        Self(T::rsqrt(self.1, self.0), self.1)
    }

    // ====== Bitwise ======

    /// Bitwise NOT.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(T::not(self.1, self.0), self.1)
    }
}

// ============================================================================
// Operator implementations
// ============================================================================

impl<T: F64x2Backend> Add for f64x2<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(T::add(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: F64x2Backend> Sub for f64x2<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(T::sub(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: F64x2Backend> Mul for f64x2<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(T::mul(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: F64x2Backend> Div for f64x2<T> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(T::div(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: F64x2Backend> Neg for f64x2<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(T::neg(self.1, self.0), self.1)
    }
}

impl<T: F64x2Backend> BitAnd for f64x2<T> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(T::bitand(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: F64x2Backend> BitOr for f64x2<T> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(T::bitor(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: F64x2Backend> BitXor for f64x2<T> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(T::bitxor(self.1, self.0, rhs.0), self.1)
    }
}

// ============================================================================
// Assign operators
// ============================================================================

impl<T: F64x2Backend> AddAssign for f64x2<T> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: F64x2Backend> SubAssign for f64x2<T> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: F64x2Backend> MulAssign for f64x2<T> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: F64x2Backend> DivAssign for f64x2<T> {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<T: F64x2Backend> BitAndAssign for f64x2<T> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl<T: F64x2Backend> BitOrAssign for f64x2<T> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl<T: F64x2Backend> BitXorAssign for f64x2<T> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

// ============================================================================
// Scalar broadcast operators (v + 2.0, v * 0.5, etc.)
// ============================================================================

impl<T: F64x2Backend> Add<f64> for f64x2<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: f64) -> Self {
        Self(T::add(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<T: F64x2Backend> Sub<f64> for f64x2<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: f64) -> Self {
        Self(T::sub(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<T: F64x2Backend> Mul<f64> for f64x2<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        Self(T::mul(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<T: F64x2Backend> Div<f64> for f64x2<T> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f64) -> Self {
        Self(T::div(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

// ============================================================================
// Index
// ============================================================================

impl<T: F64x2Backend> Index<usize> for f64x2<T> {
    type Output = f64;
    #[inline(always)]
    fn index(&self, i: usize) -> &f64 {
        &crate::simd_storage::view::<_, [f64; 2]>(&self.0)[i]
    }
}

impl<T: F64x2Backend> IndexMut<usize> for f64x2<T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut crate::simd_storage::view_mut::<_, [f64; 2]>(&mut self.0)[i]
    }
}

// ============================================================================
// Conversions
// ============================================================================

impl<T: F64x2Backend> From<f64x2<T>> for [f64; 2] {
    #[inline(always)]
    fn from(v: f64x2<T>) -> [f64; 2] {
        T::to_array(v.1, v.0)
    }
}

// ============================================================================
// Debug
// ============================================================================

impl<T: F64x2Backend> core::fmt::Debug for f64x2<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let arr = T::to_array(self.1, self.0);
        f.debug_tuple("f64x2").field(&arr).finish()
    }
}

// ============================================================================
// Platform-specific concrete impls
// ============================================================================

impl f64x2<archmage::ScalarToken> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "scalar::f64x2"
    }
}

#[cfg(target_arch = "x86_64")]
impl f64x2<archmage::X64V3Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v3::f64x2"
    }

    /// Get the raw `__m128d` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::x86_64::__m128d {
        self.0
    }

    /// Create from a raw `__m128d` (token-gated, zero-cost).
    #[inline(always)]
    pub fn from_m128d_t(token: archmage::X64V3Token, v: core::arch::x86_64::__m128d) -> Self {
        Self(v, token)
    }
}

#[cfg(target_arch = "aarch64")]
impl f64x2<archmage::NeonToken> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "arm::neon::f64x2"
    }
}

#[cfg(target_arch = "wasm32")]
impl f64x2<archmage::Wasm128Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "wasm::wasm128::f64x2"
    }
}
#[cfg(target_arch = "x86_64")]
impl f64x2<archmage::X64V3Token> {
    /// Wrap a raw `__m128d` using an explicit CPU capability token.
    ///
    /// The caller does not need a target-feature annotation.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_raw_t(token: archmage::X64V3Token, value: core::arch::x86_64::__m128d) -> Self {
        Self(value, token)
    }

    /// Wrap a raw `__m128d` in a matching target-feature context.
    ///
    /// Rust requires the caller to enable the `v3` tier's features.
    /// Use an archmage `#[rite(v3)]` helper or `#[arcane]` entry point.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn from_raw(value: core::arch::x86_64::__m128d) -> Self {
        Self(value, archmage::X64V3Token::from_context())
    }
}
#[cfg(target_arch = "aarch64")]
impl f64x2<archmage::NeonToken> {
    /// Get the raw `float64x2_t` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::aarch64::float64x2_t {
        self.0
    }

    /// Wrap a raw `float64x2_t` using an existing CPU capability token.
    #[inline(always)]
    pub fn from_float64x2_t(
        token: archmage::NeonToken,
        value: core::arch::aarch64::float64x2_t,
    ) -> Self {
        Self(value, token)
    }

    /// Wrap a raw `float64x2_t` using an explicit CPU capability token.
    ///
    /// The caller does not need a target-feature annotation.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_raw_t(token: archmage::NeonToken, value: core::arch::aarch64::float64x2_t) -> Self {
        Self(value, token)
    }

    /// Wrap a raw `float64x2_t` in a matching target-feature context.
    ///
    /// Rust requires the caller to enable the `neon` tier's features.
    /// Use an archmage `#[rite(neon)]` helper or `#[arcane]` entry point.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger feature context, which Rust checks.
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn from_raw(value: core::arch::aarch64::float64x2_t) -> Self {
        Self(value, archmage::NeonToken::from_context())
    }
}
#[cfg(target_arch = "wasm32")]
impl f64x2<archmage::Wasm128Token> {
    /// Get the raw `v128` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::wasm32::v128 {
        self.0
    }

    /// Wrap a raw `v128` using an existing CPU capability token.
    #[inline(always)]
    pub fn from_v128_t(token: archmage::Wasm128Token, value: core::arch::wasm32::v128) -> Self {
        Self(value, token)
    }

    /// Wrap a raw `v128` using an explicit CPU capability token.
    ///
    /// The caller does not need a target-feature annotation.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_raw_t(token: archmage::Wasm128Token, value: core::arch::wasm32::v128) -> Self {
        Self(value, token)
    }

    /// Wrap a raw `v128` in a matching target-feature context.
    ///
    /// Rust requires the caller to enable the `wasm128` tier's features.
    /// Use an archmage `#[rite(wasm128)]` helper or `#[arcane]` entry point.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger feature context, which Rust checks.
    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn from_raw(value: core::arch::wasm32::v128) -> Self {
        Self(value, archmage::Wasm128Token::from_context())
    }
}
// Generated deprecated token-constructor forwarders. Do not edit.
impl<T: F64x2Backend> f64x2<T> {
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::splat_t`].\n\nUse `splat_t` to keep explicit-token construction when `splat` becomes tokenless in magetypes 0.10."]
    #[deprecated(note = "Use splat_t(token, v); splat becomes tokenless in magetypes 0.10.")]
    #[forbid(unsafe_code)]
    pub fn splat(token: T, v: f64) -> Self {
        Self::splat_t(token, v)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::zero_t`].\n\nUse `zero_t` to keep explicit-token construction when `zero` becomes tokenless in magetypes 0.10."]
    #[deprecated(note = "Use zero_t(token); zero becomes tokenless in magetypes 0.10.")]
    #[forbid(unsafe_code)]
    pub fn zero(token: T) -> Self {
        Self::zero_t(token)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::load_t`].\n\nUse `load_t` to keep explicit-token construction when `load` becomes tokenless in magetypes 0.10."]
    #[deprecated(note = "Use load_t(token, data); load becomes tokenless in magetypes 0.10.")]
    #[forbid(unsafe_code)]
    pub fn load(token: T, data: &[f64; 2]) -> Self {
        Self::load_t(token, data)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_array_t`].\n\nUse `from_array_t` to keep explicit-token construction when `from_array` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_array_t(token, arr); from_array becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_array(token: T, arr: [f64; 2]) -> Self {
        Self::from_array_t(token, arr)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_slice_t`].\n\nUse `from_slice_t` to keep explicit-token construction when `from_slice` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_slice_t(token, slice); from_slice becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_slice(token: T, slice: &[f64]) -> Self {
        Self::from_slice_t(token, slice)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::partition_slice_t`].\n\nUse `partition_slice_t` to keep explicit-token construction when `partition_slice` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use partition_slice_t(token, data); partition_slice becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn partition_slice(token: T, data: &[f64]) -> (&[[f64; 2]], &[f64]) {
        Self::partition_slice_t(token, data)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::partition_slice_mut_t`].\n\nUse `partition_slice_mut_t` to keep explicit-token construction when `partition_slice_mut` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use partition_slice_mut_t(token, data); partition_slice_mut becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn partition_slice_mut(token: T, data: &mut [f64]) -> (&mut [[f64; 2]], &mut [f64]) {
        Self::partition_slice_mut_t(token, data)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_repr_t`].\n\nUse `from_repr_t` to keep explicit-token construction when `from_repr` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_repr_t(token, repr); from_repr becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_repr(token: T, repr: T::Repr) -> Self {
        Self::from_repr_t(token, repr)
    }
}
#[cfg(target_arch = "x86_64")]
impl f64x2<archmage::X64V3Token> {
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_m128d_t`].\n\nUse `from_m128d_t` to keep explicit-token construction when `from_m128d` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_m128d_t(token, v); from_m128d becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_m128d(token: archmage::X64V3Token, v: core::arch::x86_64::__m128d) -> Self {
        Self::from_m128d_t(token, v)
    }
}
#[cfg(target_arch = "wasm32")]
impl f64x2<archmage::Wasm128Token> {
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_v128_t`].\n\nUse `from_v128_t` to keep explicit-token construction when `from_v128` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_v128_t(token, value); from_v128 becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_v128(token: archmage::Wasm128Token, value: core::arch::wasm32::v128) -> Self {
        Self::from_v128_t(token, value)
    }
}

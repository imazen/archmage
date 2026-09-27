//! Generic `f64x4<T>` — 4-lane f64 SIMD vector parameterized by backend.
//!
//! `T` is a token type (e.g., `X64V3Token`, `NeonToken`, `ScalarToken`)
//! that determines the platform-native representation and intrinsics used.
//! The struct delegates all operations to the [`F64x4Backend`] trait.

#![allow(clippy::should_implement_trait)]

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::simd::backends::F64x4Backend;

/// 4-lane f64 SIMD vector, generic over backend `T`.
///
/// `T` is a token type that proves CPU support for the required SIMD features.
/// The inner representation is `T::Repr` (e.g., `__m256d` on AVX2, `[f64; 4]` on scalar).
///
/// **The token is stored** (as a zero-sized field) so methods receiving
/// `self: f64x4<T>` can re-supply it to backend operations that
/// require a token value (e.g. `T::splat(token, v)`). This carries the
/// token-as-feature-proof guarantee through every method call without
/// runtime overhead — `T` is ZST, so `sizeof(f64x4<T>) == sizeof(T::Repr)`
/// and `align_of(f64x4<T>) == align_of(T::Repr)` under `#[repr(C)]`.
///
/// # Layout
///
/// `#[repr(C)]` with a ZST trailing field: `T::Repr` lives at offset 0
/// and `T` plus the sealed policy marker are zero-sized tails. Bitcasts between `f64x4<T>` values of
/// different element-types are sound when the Repr types share a layout
/// (e.g. `__m128` and `__m128i` are both 16-byte aligned 128-bit values).
/// `#[repr(transparent)]` cannot be used because Rust cannot prove at
/// the struct definition site that a generic `T` is a 1-ZST.
///
/// Fixed-policy aliases select explicit-token or feature-context constructors.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct f64x4<
    T: F64x4Backend,
    M: crate::simd::generic::ConstructorMode = crate::simd::generic::Explicit,
>(
    pub(crate) T::Repr,
    pub(crate) T,
    pub(crate) core::marker::PhantomData<M>,
);
// SAFETY: repr(C) pair of Pod storage and a sealed 1-ZST token.
// A supplied T proves CPU support; the wrapper adds no bit invariants.
// Helpers additionally check token size/alignment at monomorphization.
unsafe impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend>
    crate::simd_storage::TokenStorage for f64x4<T, M>
{
    type Token = T;
}

// Layout invariant: struct is `#[repr(C)]` with a trailing ZST `T`
// field, so `sizeof/alignof(f64x4<T>) == sizeof/alignof(T::Repr)`
// when `T` is a 1-ZST. Every archmage token currently satisfies this;
// if a future refactor adds a non-ZST field to a token, this const
// assert fires at compile time.
const _: () = {
    assert!(
        core::mem::size_of::<f64x4<archmage::ScalarToken>>()
            == core::mem::size_of::<
                <archmage::ScalarToken as crate::simd::backends::F64x4Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<f64x4<archmage::ScalarToken>>()
            == core::mem::align_of::<
                <archmage::ScalarToken as crate::simd::backends::F64x4Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "x86_64")]
const _: () = {
    assert!(
        core::mem::size_of::<f64x4<archmage::X64V3Token>>()
            == core::mem::size_of::<
                <archmage::X64V3Token as crate::simd::backends::F64x4Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<f64x4<archmage::X64V3Token>>()
            == core::mem::align_of::<
                <archmage::X64V3Token as crate::simd::backends::F64x4Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "aarch64")]
const _: () = {
    assert!(
        core::mem::size_of::<f64x4<archmage::NeonToken>>()
            == core::mem::size_of::<
                <archmage::NeonToken as crate::simd::backends::F64x4Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<f64x4<archmage::NeonToken>>()
            == core::mem::align_of::<
                <archmage::NeonToken as crate::simd::backends::F64x4Backend>::Repr,
            >()
    );
};

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const _: () = {
    assert!(
        core::mem::size_of::<f64x4<archmage::Wasm128Token>>()
            == core::mem::size_of::<
                <archmage::Wasm128Token as crate::simd::backends::F64x4Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<f64x4<archmage::Wasm128Token>>()
            == core::mem::align_of::<
                <archmage::Wasm128Token as crate::simd::backends::F64x4Backend>::Repr,
            >()
    );
};

impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend> f64x4<T, M> {
    #[inline(always)]
    pub(crate) fn new_repr(repr: T::Repr, token: T) -> Self {
        Self(repr, token, core::marker::PhantomData)
    }

    /// Number of f64 lanes.
    pub const LANES: usize = 4;

    // ====== Construction (token-gated) ======

    /// Broadcast scalar to all 4 lanes.
    #[inline(always)]
    pub(crate) fn splat_with_token(token: T, v: f64) -> Self {
        Self::new_repr(T::splat(token, v), token)
    }

    /// All lanes zero.
    #[inline(always)]
    pub(crate) fn zero_with_token(token: T) -> Self {
        Self::new_repr(T::zero(token), token)
    }

    /// Load from a `[f64; 4]` array.
    #[inline(always)]
    pub(crate) fn load_with_token(token: T, data: &[f64; 4]) -> Self {
        Self::new_repr(T::load(token, data), token)
    }

    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub(crate) fn from_array_with_token(token: T, arr: [f64; 4]) -> Self {
        Self::new_repr(T::from_array(token, arr), token)
    }

    /// Create from slice. Panics if `slice.len() < 4`.
    #[inline(always)]
    pub(crate) fn from_slice_with_token(token: T, slice: &[f64]) -> Self {
        let arr: [f64; 4] = slice[..4].try_into().unwrap();
        Self::new_repr(T::from_array(token, arr), token)
    }

    /// Split a slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&[[f64; 4]], &[f64])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub(crate) fn partition_slice_with_token(_token: T, data: &[f64]) -> (&[[f64; 4]], &[f64]) {
        data.as_chunks::<4>()
    }

    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&mut [[f64; 4]], &mut [f64])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub(crate) fn partition_slice_mut_with_token(
        _token: T,
        data: &mut [f64],
    ) -> (&mut [[f64; 4]], &mut [f64]) {
        data.as_chunks_mut::<4>()
    }

    // ====== Accessors ======

    /// Store to array.
    #[inline(always)]
    pub fn store(self, out: &mut [f64; 4]) {
        T::store(self.1, self.0, out);
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [f64; 4] {
        T::to_array(self.1, self.0)
    }

    /// Get the underlying platform representation.
    #[inline(always)]
    pub fn into_repr(self) -> T::Repr {
        self.0
    }

    /// Wrap a platform representation (token-gated).
    #[inline(always)]
    pub(crate) fn from_repr_with_token(token: T, repr: T::Repr) -> Self {
        Self::new_repr(repr, token)
    }

    /// Wrap a repr with a token. Used by cross-type/cross-width helpers
    /// in `simd::generic::*` where the token is already proven by the
    /// caller's wider input type.
    #[inline(always)]
    #[allow(dead_code)]
    pub(crate) fn from_repr_unchecked(token: T, repr: T::Repr) -> Self {
        Self::new_repr(repr, token)
    }

    // ====== Math ======

    /// Lane-wise minimum.
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self::new_repr(T::min(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise maximum.
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self::new_repr(T::max(self.1, self.0, other.0), self.1)
    }

    /// Clamp between lo and hi.
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        Self::new_repr(T::clamp(self.1, self.0, lo.0, hi.0), self.1)
    }

    /// Square root.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self::new_repr(T::sqrt(self.1, self.0), self.1)
    }

    /// Absolute value.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self::new_repr(T::abs(self.1, self.0), self.1)
    }

    /// Round toward negative infinity.
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self::new_repr(T::floor(self.1, self.0), self.1)
    }

    /// Round toward positive infinity.
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self::new_repr(T::ceil(self.1, self.0), self.1)
    }

    /// Round to nearest integer.
    #[inline(always)]
    pub fn round(self) -> Self {
        Self::new_repr(T::round(self.1, self.0), self.1)
    }

    /// Multiply-add: `self * a + b`.
    ///
    /// Fused with a single rounding on backends with hardware FMA
    /// (x86 v3/v4, NEON). The scalar and WASM backends compute an
    /// unfused `mul` + `add` (two roundings), so lanes can differ
    /// from the fused backends by 1 ULP.
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self::new_repr(T::mul_add(self.1, self.0, a.0, b.0), self.1)
    }

    /// Multiply-sub: `self * a - b`.
    ///
    /// Same fusion contract as [`mul_add`](Self::mul_add): fused on
    /// x86 v3/v4 and NEON, unfused (two roundings) on scalar and WASM.
    #[inline(always)]
    pub fn mul_sub(self, a: Self, b: Self) -> Self {
        Self::new_repr(T::mul_sub(self.1, self.0, a.0, b.0), self.1)
    }

    // ====== Comparisons ======

    /// Lane-wise equality (returns mask).
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self::new_repr(T::simd_eq(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise inequality (returns mask).
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self::new_repr(T::simd_ne(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise less-than (returns mask).
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self::new_repr(T::simd_lt(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise less-than-or-equal (returns mask).
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self::new_repr(T::simd_le(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise greater-than (returns mask).
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self::new_repr(T::simd_gt(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise greater-than-or-equal (returns mask).
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self::new_repr(T::simd_ge(self.1, self.0, other.0), self.1)
    }

    /// Select lanes: where mask is all-1s pick `if_true`, else `if_false`.
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self::new_repr(T::blend(mask.1, mask.0, if_true.0, if_false.0), mask.1)
    }

    // ====== Reductions ======

    /// Sum all 4 lanes.
    #[inline(always)]
    pub fn reduce_add(self) -> f64 {
        T::reduce_add(self.1, self.0)
    }

    /// Minimum across all 4 lanes.
    #[inline(always)]
    pub fn reduce_min(self) -> f64 {
        T::reduce_min(self.1, self.0)
    }

    /// Maximum across all 4 lanes.
    #[inline(always)]
    pub fn reduce_max(self) -> f64 {
        T::reduce_max(self.1, self.0)
    }

    // ====== Approximations ======

    /// Fast reciprocal approximation (1/x): the backend's native estimate
    /// (exact division on f64 — no hardware estimate exists).
    #[inline(always)]
    pub fn rcp_approx(self) -> Self {
        Self::new_repr(T::rcp_approx(self.1, self.0), self.1)
    }

    /// Precise reciprocal (1/x): exact IEEE division on every backend.
    ///
    /// Correctly rounded (0 ULP vs `1.0 / x`), including the rails:
    /// `recip(±0) = ±inf` and `recip(±inf) = ±0` (issue #64). On f64
    /// the working tier and the exact tier coincide.
    #[inline(always)]
    pub fn recip(self) -> Self {
        Self::new_repr(T::recip(self.1, self.0), self.1)
    }

    /// Fast reciprocal square root approximation (exact division +
    /// sqrt on f64 — no hardware estimate exists).
    #[inline(always)]
    pub fn rsqrt_approx(self) -> Self {
        Self::new_repr(T::rsqrt_approx(self.1, self.0), self.1)
    }

    /// Precise reciprocal square root (1/sqrt(x)): exact IEEE division
    /// and square root on every backend. Bit-exact vs scalar
    /// `1.0 / x.sqrt()`, rails included.
    #[inline(always)]
    pub fn rsqrt(self) -> Self {
        Self::new_repr(T::rsqrt(self.1, self.0), self.1)
    }

    // ====== Bitwise ======

    /// Bitwise NOT.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self::new_repr(T::not(self.1, self.0), self.1)
    }
}

// ============================================================================
// Operator implementations
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend> Add for f64x4<T, M> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new_repr(T::add(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend> Sub for f64x4<T, M> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new_repr(T::sub(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend> Mul for f64x4<T, M> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new_repr(T::mul(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend> Div for f64x4<T, M> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self::new_repr(T::div(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend> Neg for f64x4<T, M> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new_repr(T::neg(self.1, self.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend> BitAnd for f64x4<T, M> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self::new_repr(T::bitand(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend> BitOr for f64x4<T, M> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self::new_repr(T::bitor(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend> BitXor for f64x4<T, M> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self::new_repr(T::bitxor(self.1, self.0, rhs.0), self.1)
    }
}

// ============================================================================
// Assign operators
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend> AddAssign for f64x4<T, M> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend> SubAssign for f64x4<T, M> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend> MulAssign for f64x4<T, M> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend> DivAssign for f64x4<T, M> {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend> BitAndAssign for f64x4<T, M> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend> BitOrAssign for f64x4<T, M> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend> BitXorAssign for f64x4<T, M> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

// ============================================================================
// Scalar broadcast operators (v + 2.0, v * 0.5, etc.)
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend> Add<f64> for f64x4<T, M> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: f64) -> Self {
        Self::new_repr(T::add(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend> Sub<f64> for f64x4<T, M> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: f64) -> Self {
        Self::new_repr(T::sub(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend> Mul<f64> for f64x4<T, M> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        Self::new_repr(T::mul(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend> Div<f64> for f64x4<T, M> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f64) -> Self {
        Self::new_repr(T::div(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

// ============================================================================
// Index
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend> Index<usize> for f64x4<T, M> {
    type Output = f64;
    #[inline(always)]
    fn index(&self, i: usize) -> &f64 {
        &crate::simd_storage::view::<_, [f64; 4]>(&self.0)[i]
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend> IndexMut<usize> for f64x4<T, M> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut crate::simd_storage::view_mut::<_, [f64; 4]>(&mut self.0)[i]
    }
}

// ============================================================================
// Conversions
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend> From<f64x4<T, M>> for [f64; 4] {
    #[inline(always)]
    fn from(v: f64x4<T, M>) -> [f64; 4] {
        T::to_array(v.1, v.0)
    }
}

// ============================================================================
// Debug
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: F64x4Backend> core::fmt::Debug for f64x4<T, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let arr = T::to_array(self.1, self.0);
        f.debug_tuple("f64x4").field(&arr).finish()
    }
}

// ============================================================================
// Platform-specific concrete impls
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode> f64x4<archmage::ScalarToken, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "scalar::f64x4"
    }
}

#[cfg(target_arch = "x86_64")]
impl<M: crate::simd::generic::ConstructorMode> f64x4<archmage::X64V3Token, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v3::f64x4"
    }

    /// Get the raw `__m256d` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::x86_64::__m256d {
        self.0
    }

    /// Create from a raw `__m256d` (token-gated, zero-cost).
    #[inline(always)]
    pub(crate) fn from_m256d_with_token(
        token: archmage::X64V3Token,
        v: core::arch::x86_64::__m256d,
    ) -> Self {
        Self::new_repr(v, token)
    }
}

#[cfg(target_arch = "aarch64")]
impl<M: crate::simd::generic::ConstructorMode> f64x4<archmage::NeonToken, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "polyfill::neon::f64x4"
    }
}

#[cfg(target_arch = "wasm32")]
impl<M: crate::simd::generic::ConstructorMode> f64x4<archmage::Wasm128Token, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "polyfill::wasm128::f64x4"
    }
}
#[cfg(target_arch = "x86_64")]
impl<M: crate::simd::generic::ConstructorMode> f64x4<archmage::X64V3Token, M> {
    /// Wrap a raw `__m256d` in a matching target-feature context.
    ///
    /// Rust requires the caller to enable the `v3` tier's features.
    /// Use an archmage `#[rite(v3)]` helper or `#[arcane]` entry point.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_raw(value: core::arch::x86_64::__m256d) -> Self {
        Self::new_repr(value, archmage::X64V3Token::from_context())
    }
}
impl<T: F64x4Backend> From<f64x4<T, crate::simd::generic::Explicit>>
    for f64x4<T, crate::simd::generic::Context>
{
    #[inline(always)]
    fn from(value: f64x4<T, crate::simd::generic::Explicit>) -> Self {
        Self::new_repr(value.0, value.1)
    }
}
impl<T: F64x4Backend> From<f64x4<T, crate::simd::generic::Context>>
    for f64x4<T, crate::simd::generic::Explicit>
{
    #[inline(always)]
    fn from(value: f64x4<T, crate::simd::generic::Context>) -> Self {
        Self::new_repr(value.0, value.1)
    }
}

#[cfg(target_arch = "x86_64")]
#[cfg(target_arch = "x86_64")]
impl f64x4<archmage::X64V3Token, crate::simd::generic::Context> {
    /// Create from a raw `__m256d` (token-gated, zero-cost).
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_m256d(v: core::arch::x86_64::__m256d) -> Self {
        Self::from_m256d_with_token(archmage::X64V3Token::from_context(), v)
    }
}

#[cfg(target_arch = "x86_64")]
impl f64x4<archmage::X64V3Token, crate::simd::generic::Explicit> {
    /// Create from a raw `__m256d` (token-gated, zero-cost).
    #[inline(always)]
    pub fn from_m256d(token: archmage::X64V3Token, v: core::arch::x86_64::__m256d) -> Self {
        Self::from_m256d_with_token(token, v)
    }
}

#[cfg(target_arch = "aarch64")]
impl f64x4<archmage::NeonToken, crate::simd::generic::Context> {
    /// Broadcast scalar to all 4 lanes.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn splat(v: f64) -> Self {
        Self::splat_with_token(archmage::NeonToken::from_context(), v)
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn zero() -> Self {
        Self::zero_with_token(archmage::NeonToken::from_context())
    }
    /// Load from a `[f64; 4]` array.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn load(data: &[f64; 4]) -> Self {
        Self::load_with_token(archmage::NeonToken::from_context(), data)
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn from_array(arr: [f64; 4]) -> Self {
        Self::from_array_with_token(archmage::NeonToken::from_context(), arr)
    }
    /// Create from slice. Panics if `slice.len() < 4`.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn from_slice(slice: &[f64]) -> Self {
        Self::from_slice_with_token(archmage::NeonToken::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[f64; 4]], &[f64])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn partition_slice(data: &[f64]) -> (&[[f64; 4]], &[f64]) {
        Self::partition_slice_with_token(archmage::NeonToken::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[f64; 4]], &mut [f64])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn partition_slice_mut(data: &mut [f64]) -> (&mut [[f64; 4]], &mut [f64]) {
        Self::partition_slice_mut_with_token(archmage::NeonToken::from_context(), data)
    }
    /// Wrap a platform representation (requires matching target features).
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn from_repr(
        repr: <archmage::NeonToken as crate::simd::backends::F64x4Backend>::Repr,
    ) -> Self {
        Self::from_repr_with_token(archmage::NeonToken::from_context(), repr)
    }
}

#[cfg(target_arch = "wasm32")]
impl f64x4<archmage::Wasm128Token, crate::simd::generic::Context> {
    /// Broadcast scalar to all 4 lanes.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn splat(v: f64) -> Self {
        Self::splat_with_token(archmage::Wasm128Token::from_context(), v)
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn zero() -> Self {
        Self::zero_with_token(archmage::Wasm128Token::from_context())
    }
    /// Load from a `[f64; 4]` array.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn load(data: &[f64; 4]) -> Self {
        Self::load_with_token(archmage::Wasm128Token::from_context(), data)
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn from_array(arr: [f64; 4]) -> Self {
        Self::from_array_with_token(archmage::Wasm128Token::from_context(), arr)
    }
    /// Create from slice. Panics if `slice.len() < 4`.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn from_slice(slice: &[f64]) -> Self {
        Self::from_slice_with_token(archmage::Wasm128Token::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[f64; 4]], &[f64])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn partition_slice(data: &[f64]) -> (&[[f64; 4]], &[f64]) {
        Self::partition_slice_with_token(archmage::Wasm128Token::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[f64; 4]], &mut [f64])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn partition_slice_mut(data: &mut [f64]) -> (&mut [[f64; 4]], &mut [f64]) {
        Self::partition_slice_mut_with_token(archmage::Wasm128Token::from_context(), data)
    }
    /// Wrap a platform representation (requires matching target features).
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn from_repr(
        repr: <archmage::Wasm128Token as crate::simd::backends::F64x4Backend>::Repr,
    ) -> Self {
        Self::from_repr_with_token(archmage::Wasm128Token::from_context(), repr)
    }
}

#[cfg(target_arch = "x86_64")]
impl f64x4<archmage::X64V3Token, crate::simd::generic::Context> {
    /// Broadcast scalar to all 4 lanes.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn splat(v: f64) -> Self {
        Self::splat_with_token(archmage::X64V3Token::from_context(), v)
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn zero() -> Self {
        Self::zero_with_token(archmage::X64V3Token::from_context())
    }
    /// Load from a `[f64; 4]` array.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn load(data: &[f64; 4]) -> Self {
        Self::load_with_token(archmage::X64V3Token::from_context(), data)
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_array(arr: [f64; 4]) -> Self {
        Self::from_array_with_token(archmage::X64V3Token::from_context(), arr)
    }
    /// Create from slice. Panics if `slice.len() < 4`.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_slice(slice: &[f64]) -> Self {
        Self::from_slice_with_token(archmage::X64V3Token::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[f64; 4]], &[f64])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn partition_slice(data: &[f64]) -> (&[[f64; 4]], &[f64]) {
        Self::partition_slice_with_token(archmage::X64V3Token::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[f64; 4]], &mut [f64])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn partition_slice_mut(data: &mut [f64]) -> (&mut [[f64; 4]], &mut [f64]) {
        Self::partition_slice_mut_with_token(archmage::X64V3Token::from_context(), data)
    }
    /// Wrap a platform representation (requires matching target features).
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_repr(
        repr: <archmage::X64V3Token as crate::simd::backends::F64x4Backend>::Repr,
    ) -> Self {
        Self::from_repr_with_token(archmage::X64V3Token::from_context(), repr)
    }
}

impl<T: F64x4Backend> f64x4<T, crate::simd::generic::Explicit> {
    /// Broadcast scalar to all 4 lanes.
    #[inline(always)]
    pub fn splat(token: T, v: f64) -> Self {
        Self::splat_with_token(token, v)
    }
    /// All lanes zero.
    #[inline(always)]
    pub fn zero(token: T) -> Self {
        Self::zero_with_token(token)
    }
    /// Load from a `[f64; 4]` array.
    #[inline(always)]
    pub fn load(token: T, data: &[f64; 4]) -> Self {
        Self::load_with_token(token, data)
    }
    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub fn from_array(token: T, arr: [f64; 4]) -> Self {
        Self::from_array_with_token(token, arr)
    }
    /// Create from slice. Panics if `slice.len() < 4`.
    #[inline(always)]
    pub fn from_slice(token: T, slice: &[f64]) -> Self {
        Self::from_slice_with_token(token, slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[f64; 4]], &[f64])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice(_token: T, data: &[f64]) -> (&[[f64; 4]], &[f64]) {
        Self::partition_slice_with_token(_token, data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[f64; 4]], &mut [f64])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice_mut(_token: T, data: &mut [f64]) -> (&mut [[f64; 4]], &mut [f64]) {
        Self::partition_slice_mut_with_token(_token, data)
    }
    /// Wrap a platform representation (token-gated).
    #[inline(always)]
    pub fn from_repr(token: T, repr: T::Repr) -> Self {
        Self::from_repr_with_token(token, repr)
    }
}

impl f64x4<archmage::ScalarToken, crate::simd::generic::Context> {
    /// Broadcast scalar to all 4 lanes.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn splat(v: f64) -> Self {
        Self::splat_with_token(archmage::ScalarToken, v)
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn zero() -> Self {
        Self::zero_with_token(archmage::ScalarToken)
    }
    /// Load from a `[f64; 4]` array.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn load(data: &[f64; 4]) -> Self {
        Self::load_with_token(archmage::ScalarToken, data)
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_array(arr: [f64; 4]) -> Self {
        Self::from_array_with_token(archmage::ScalarToken, arr)
    }
    /// Create from slice. Panics if `slice.len() < 4`.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_slice(slice: &[f64]) -> Self {
        Self::from_slice_with_token(archmage::ScalarToken, slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[f64; 4]], &[f64])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn partition_slice(data: &[f64]) -> (&[[f64; 4]], &[f64]) {
        Self::partition_slice_with_token(archmage::ScalarToken, data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[f64; 4]], &mut [f64])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn partition_slice_mut(data: &mut [f64]) -> (&mut [[f64; 4]], &mut [f64]) {
        Self::partition_slice_mut_with_token(archmage::ScalarToken, data)
    }
    /// Wrap a platform representation (requires matching target features).
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_repr(
        repr: <archmage::ScalarToken as crate::simd::backends::F64x4Backend>::Repr,
    ) -> Self {
        Self::from_repr_with_token(archmage::ScalarToken, repr)
    }
}

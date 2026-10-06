//! Generic `f32x16<T>` — 16-lane f32 SIMD vector parameterized by backend.
//!
//! `T` is a token type (e.g., `X64V4Token`, `ScalarToken`)
//! that determines the platform-native representation and intrinsics used.
//! The struct delegates all operations to the [`F32x16Backend`] trait.
//!
//! # Example
//!
//! ```ignore
//! use magetypes::simd::backends::{F32x16Backend, x64v4};
//! use magetypes::simd::generic::f32x16;
//!
//! fn sum<T: F32x16Backend>(token: T, data: &[f32]) -> f32 {
//!     let mut acc = f32x16::<T>::zero_t(token);
//!     for chunk in data.chunks_exact(16) {
//!         acc = acc + f32x16::<T>::load_t(token, chunk.try_into().unwrap());
//!     }
//!     acc.reduce_add()
//! }
//! ```

#![allow(clippy::should_implement_trait)]

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::simd::backends::F32x16Backend;

/// 16-lane f32 SIMD vector, generic over backend `T`.
///
/// `T` is a token type that proves CPU support for the required SIMD features.
/// The inner representation is `T::Repr` (e.g., `__m512` on AVX-512, `[f32; 16]` on scalar).
///
/// **The token is stored** (as a zero-sized field) so methods receiving
/// `self: f32x16<T>` can re-supply it to backend operations that
/// require a token value (e.g. `T::splat(token, v)`). This carries the
/// token-as-feature-proof guarantee through every method call without
/// runtime overhead — `T` is ZST, so `sizeof(f32x16<T>) == sizeof(T::Repr)`
/// and `align_of(f32x16<T>) == align_of(T::Repr)` under `#[repr(C)]`.
///
/// # Layout
///
/// `#[repr(C)]` with a ZST trailing field: `T::Repr` lives at offset 0
/// and `T` is a 0-byte tail. Bitcasts between `f32x16<T>` values of
/// different element-types are sound when the Repr types share a layout
/// (e.g. `__m128` and `__m128i` are both 16-byte aligned 128-bit values).
/// `#[repr(transparent)]` cannot be used because Rust cannot prove at
/// the struct definition site that a generic `T` is a 1-ZST.
///
/// Construction requires a token value to prove CPU support at runtime.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct f32x16<T: F32x16Backend>(pub(crate) T::Repr, pub(crate) T);
// SAFETY: repr(C) pair of Pod storage and a sealed 1-ZST token.
// A supplied T proves CPU support; the wrapper adds no bit invariants.
// Helpers additionally check token size/alignment at monomorphization.
unsafe impl<T: F32x16Backend> crate::simd_storage::TokenStorage for f32x16<T> {
    type Token = T;
}

// PhantomData is ZST, so f32x16<T> has the same size as T::Repr.

// Layout invariant: struct is `#[repr(C)]` with a trailing ZST `T`
// field, so `sizeof/alignof(f32x16<T>) == sizeof/alignof(T::Repr)`
// when `T` is a 1-ZST. Every archmage token currently satisfies this;
// if a future refactor adds a non-ZST field to a token, this const
// assert fires at compile time.
const _: () = {
    assert!(
        core::mem::size_of::<f32x16<archmage::ScalarToken>>()
            == core::mem::size_of::<
                <archmage::ScalarToken as crate::simd::backends::F32x16Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<f32x16<archmage::ScalarToken>>()
            == core::mem::align_of::<
                <archmage::ScalarToken as crate::simd::backends::F32x16Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "x86_64")]
const _: () = {
    assert!(
        core::mem::size_of::<f32x16<archmage::X64V3Token>>()
            == core::mem::size_of::<
                <archmage::X64V3Token as crate::simd::backends::F32x16Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<f32x16<archmage::X64V3Token>>()
            == core::mem::align_of::<
                <archmage::X64V3Token as crate::simd::backends::F32x16Backend>::Repr,
            >()
    );
};

// Native AVX-512 (`__m512`/`__m512d`/`__m512i`) — gated on the
// `avx512` feature, which is how archmage exposes X64V4Token's
// 512-bit backend impls.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
const _: () = {
    assert!(
        core::mem::size_of::<f32x16<archmage::X64V4Token>>()
            == core::mem::size_of::<
                <archmage::X64V4Token as crate::simd::backends::F32x16Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<f32x16<archmage::X64V4Token>>()
            == core::mem::align_of::<
                <archmage::X64V4Token as crate::simd::backends::F32x16Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "aarch64")]
const _: () = {
    assert!(
        core::mem::size_of::<f32x16<archmage::NeonToken>>()
            == core::mem::size_of::<
                <archmage::NeonToken as crate::simd::backends::F32x16Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<f32x16<archmage::NeonToken>>()
            == core::mem::align_of::<
                <archmage::NeonToken as crate::simd::backends::F32x16Backend>::Repr,
            >()
    );
};

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const _: () = {
    assert!(
        core::mem::size_of::<f32x16<archmage::Wasm128Token>>()
            == core::mem::size_of::<
                <archmage::Wasm128Token as crate::simd::backends::F32x16Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<f32x16<archmage::Wasm128Token>>()
            == core::mem::align_of::<
                <archmage::Wasm128Token as crate::simd::backends::F32x16Backend>::Repr,
            >()
    );
};

impl<T: F32x16Backend> f32x16<T> {
    /// Number of f32 lanes.
    pub const LANES: usize = 16;

    // ====== Construction (token-gated) ======

    /// Broadcast scalar to all 16 lanes.
    #[inline(always)]
    pub fn splat_t(token: T, v: f32) -> Self {
        Self(T::splat(token, v), token)
    }

    /// All lanes zero.
    #[inline(always)]
    pub fn zero_t(token: T) -> Self {
        Self(T::zero(token), token)
    }

    /// Load from a `[f32; 16]` array.
    #[inline(always)]
    pub fn load_t(token: T, data: &[f32; 16]) -> Self {
        Self(T::load(token, data), token)
    }

    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub fn from_array_t(token: T, arr: [f32; 16]) -> Self {
        Self(T::from_array(token, arr), token)
    }

    /// Create from slice. Panics if `slice.len() < 16`.
    #[inline(always)]
    pub fn from_slice_t(token: T, slice: &[f32]) -> Self {
        let arr: [f32; 16] = slice[..16].try_into().unwrap();
        Self(T::from_array(token, arr), token)
    }

    /// Split a slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&[[f32; 16]], &[f32])` — fixed-size arrays suitable
    /// for [`load_t`](Self::load_t), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice_t(_: T, data: &[f32]) -> (&[[f32; 16]], &[f32]) {
        data.as_chunks::<16>()
    }

    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&mut [[f32; 16]], &mut [f32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load_t`](Self::load_t), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice_mut_t(_: T, data: &mut [f32]) -> (&mut [[f32; 16]], &mut [f32]) {
        data.as_chunks_mut::<16>()
    }

    // ====== Accessors ======

    /// Store to array.
    #[inline(always)]
    pub fn store(self, out: &mut [f32; 16]) {
        T::store(self.1, self.0, out);
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [f32; 16] {
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

    /// Sum all 16 lanes.
    #[inline(always)]
    pub fn reduce_add(self) -> f32 {
        T::reduce_add(self.1, self.0)
    }

    /// Minimum across all 16 lanes.
    #[inline(always)]
    pub fn reduce_min(self) -> f32 {
        T::reduce_min(self.1, self.0)
    }

    /// Maximum across all 16 lanes.
    #[inline(always)]
    pub fn reduce_max(self) -> f32 {
        T::reduce_max(self.1, self.0)
    }

    // ====== Approximations ======

    /// Fast reciprocal approximation (1/x): the backend's native estimate.
    #[inline(always)]
    pub fn rcp_approx(self) -> Self {
        Self(T::rcp_approx(self.1, self.0), self.1)
    }

    /// Reciprocal (1/x), the working tier: ≤4 ULP **with exact IEEE
    /// rails** on every backend (native AVX-512 uses rcp14 + Newton +
    /// a one-instruction VFIXUPIMM rail patch; polyfill widths
    /// inherit their sub-vector's rescue). Subnormal inputs
    /// unspecified; [`recip_portable`](Self::recip_portable) for
    /// 0 ULP + subnormals + bit-identical.
    #[inline(always)]
    pub fn recip(self) -> Self {
        Self(T::recip(self.1, self.0), self.1)
    }

    /// Fast reciprocal square root approximation: the backend's native
    /// estimate. `±0`/`±inf` lanes can come back NaN on estimate-refine
    /// backends.
    #[inline(always)]
    pub fn rsqrt_approx(self) -> Self {
        Self(T::rsqrt_approx(self.1, self.0), self.1)
    }

    /// Reciprocal square root (1/sqrt(x)), the working tier: ≤4 ULP
    /// with exact IEEE rails — see [`recip`](Self::recip) for the
    /// contract shape; [`rsqrt_portable`](Self::rsqrt_portable) adds
    /// 0 ULP + subnormals + bit-identical.
    #[inline(always)]
    pub fn rsqrt(self) -> Self {
        Self(T::rsqrt(self.1, self.0), self.1)
    }

    // ====== Deterministic (cross-platform bit-identical) reciprocals ======
    //
    // The `_portable` family returns **the same bits on every architecture**
    // (x86 / ARM / WASM). Two ingredients make that hold:
    //   * the seed is an integer bit-trick — pure integer math, so it is
    //     identical by construction (the shift is logical, not arch-dependent);
    //   * the Newton steps use plain IEEE-754 `mul`/`sub`, never `mul_add`
    //     (FMA): one-rounding arithmetic differs from this two-rounding
    //     formulation, so fused ops are avoided on purpose.
    //
    // Hardware estimate instructions (`rsqrtps`, `vrsqrte`) are deliberately
    // NOT used here — their bits differ across vendors and generations. For
    // the faster, per-platform (non-deterministic) variants see
    // [`rsqrt_approx`](Self::rsqrt_approx) / [`recip`](Self::recip).

    /// Deterministic reciprocal-sqrt estimate (~8-bit), bit-identical on
    /// every platform.
    ///
    /// Seed (integer bit-trick) plus one non-FMA Newton step. Opt into more
    /// accuracy with [`rsqrt_newton_portable`](Self::rsqrt_newton_portable)
    /// (one more step ≈ 16-bit), or use
    /// [`rsqrt_portable`](Self::rsqrt_portable) for full precision.
    #[inline(always)]
    pub fn rsqrt_approx_portable(self) -> Self
    where
        T: crate::simd::backends::F32x16Convert,
    {
        let t = self.1;
        let bits = self.bitcast_to_i32();
        let seed = Self::from_i32_bitcast_t(
            t,
            super::i32x16::splat_t(t, 0x5f3759df_i32) - bits.shr_logical_const::<1>(),
        );
        seed.rsqrt_newton_portable(self)
    }

    /// One deterministic Newton refinement step for `1/sqrt(x)`:
    /// `y * (1.5 - 0.5*x*y*y)`, where `self` is the current estimate `y`
    /// and `x` is the original input. Non-FMA → bit-identical everywhere.
    #[inline(always)]
    pub fn rsqrt_newton_portable(self, x: Self) -> Self {
        let t = self.1;
        let half = Self::splat_t(t, 0.5);
        let three_halves = Self::splat_t(t, 1.5);
        self * (three_halves - half * x * self * self)
    }

    /// Precise reciprocal square root: exact IEEE sqrt + division —
    /// **the 0 ULP tier**, with IEEE rails (`rsqrt(+0) = +inf`,
    /// `rsqrt(+inf) = +0`, negatives give NaN) and bit-identical on
    /// every arch. Costs ~3.6x the working-tier [`rsqrt`](Self::rsqrt)
    /// on Zen-class x86 (and is the faster form on Apple Silicon).
    #[inline(always)]
    pub fn rsqrt_portable(self) -> Self {
        Self::splat_t(self.1, 1.0) / self.sqrt()
    }

    /// Deterministic reciprocal estimate (~8-bit), bit-identical on every
    /// platform.
    ///
    /// Seed (integer bit-trick) plus one non-FMA Newton step. Opt into more
    /// accuracy with [`recip_newton_portable`](Self::recip_newton_portable),
    /// or use [`recip_portable`](Self::recip_portable) for full precision.
    #[inline(always)]
    pub fn rcp_approx_portable(self) -> Self
    where
        T: crate::simd::backends::F32x16Convert,
    {
        let t = self.1;
        let bits = self.bitcast_to_i32();
        let seed = Self::from_i32_bitcast_t(t, super::i32x16::splat_t(t, 0x7ef127ea_i32) - bits);
        seed.recip_newton_portable(self)
    }

    /// One deterministic Newton refinement step for `1/x`: `y * (2 - x*y)`,
    /// where `self` is the current estimate `y` and `x` is the input.
    #[inline(always)]
    pub fn recip_newton_portable(self, x: Self) -> Self {
        let t = self.1;
        let two = Self::splat_t(t, 2.0);
        self * (two - x * self)
    }

    /// Precise reciprocal: exact IEEE division — **the 0 ULP tier**.
    ///
    /// Correctly rounded on every backend, with the IEEE rails
    /// (`recip(±0) = ±inf`, `recip(±inf) = ±0` — no NaN after a
    /// saturating `exp_midp`, issue #64) and, because correctly-rounded
    /// division is uniquely defined, bit-identical on every arch — the
    /// portable property is free. Costs ~1.9x the working-tier
    /// [`recip`](Self::recip) on Zen-class x86 (and is the FASTER form
    /// on Apple Silicon).
    #[inline(always)]
    pub fn recip_portable(self) -> Self {
        Self::splat_t(self.1, 1.0) / self
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

impl<T: F32x16Backend> Add for f32x16<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(T::add(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: F32x16Backend> Sub for f32x16<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(T::sub(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: F32x16Backend> Mul for f32x16<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(T::mul(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: F32x16Backend> Div for f32x16<T> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(T::div(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: F32x16Backend> Neg for f32x16<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(T::neg(self.1, self.0), self.1)
    }
}

impl<T: F32x16Backend> BitAnd for f32x16<T> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(T::bitand(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: F32x16Backend> BitOr for f32x16<T> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(T::bitor(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: F32x16Backend> BitXor for f32x16<T> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(T::bitxor(self.1, self.0, rhs.0), self.1)
    }
}

// ============================================================================
// Assign operators
// ============================================================================

impl<T: F32x16Backend> AddAssign for f32x16<T> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: F32x16Backend> SubAssign for f32x16<T> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: F32x16Backend> MulAssign for f32x16<T> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: F32x16Backend> DivAssign for f32x16<T> {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<T: F32x16Backend> BitAndAssign for f32x16<T> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl<T: F32x16Backend> BitOrAssign for f32x16<T> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl<T: F32x16Backend> BitXorAssign for f32x16<T> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

// ============================================================================
// Scalar broadcast operators (v + 2.0, v * 0.5, etc.)
// ============================================================================

impl<T: F32x16Backend> Add<f32> for f32x16<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: f32) -> Self {
        Self(T::add(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<T: F32x16Backend> Sub<f32> for f32x16<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: f32) -> Self {
        Self(T::sub(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<T: F32x16Backend> Mul<f32> for f32x16<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        Self(T::mul(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<T: F32x16Backend> Div<f32> for f32x16<T> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self {
        Self(T::div(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

// ============================================================================
// Index
// ============================================================================

impl<T: F32x16Backend> Index<usize> for f32x16<T> {
    type Output = f32;
    #[inline(always)]
    fn index(&self, i: usize) -> &f32 {
        &crate::simd_storage::view::<_, [f32; 16]>(&self.0)[i]
    }
}

impl<T: F32x16Backend> IndexMut<usize> for f32x16<T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut f32 {
        &mut crate::simd_storage::view_mut::<_, [f32; 16]>(&mut self.0)[i]
    }
}

// ============================================================================
// Conversions
// ============================================================================

impl<T: F32x16Backend> From<f32x16<T>> for [f32; 16] {
    #[inline(always)]
    fn from(v: f32x16<T>) -> [f32; 16] {
        T::to_array(v.1, v.0)
    }
}

// ============================================================================
// Debug
// ============================================================================

impl<T: F32x16Backend> core::fmt::Debug for f32x16<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let arr = T::to_array(self.1, self.0);
        f.debug_tuple("f32x16").field(&arr).finish()
    }
}

// ============================================================================
// Cross-type conversions (available when T implements conversion traits)
// ============================================================================

impl<T: crate::simd::backends::F32x16Convert> f32x16<T> {
    /// Bitcast to i32x16 (reinterpret bits, no conversion).
    #[inline(always)]
    pub fn bitcast_to_i32(self) -> super::i32x16<T> {
        super::i32x16::from_repr_unchecked(self.1, T::bitcast_f32_to_i32(self.1, self.0))
    }

    /// Create from i32x16 via bitcast (reinterpret bits, no conversion).
    #[inline(always)]
    pub fn from_i32_bitcast_t(token: T, v: super::i32x16<T>) -> Self {
        Self(T::bitcast_i32_to_f32(token, v.into_repr()), token)
    }

    /// Convert to i32x16 with truncation toward zero.
    ///
    /// **Out-of-range and NaN lanes DIVERGE per backend** (issue #80,
    /// `docs/CROSS-ISA-DIVERGENCES.md` §3): x86's `cvttps` yields the
    /// `i32::MIN` integer-indefinite sentinel for overflow in BOTH
    /// directions and for NaN, while NEON/WASM/scalar saturate with
    /// NaN→0 — `3e9_f32` converts to −2147483648 on x86 and
    /// +2147483647 everywhere else. Keep lanes in range, or use
    /// [`to_i32_saturating`](Self::to_i32_saturating) for uniform
    /// semantics.
    #[inline(always)]
    pub fn to_i32(self) -> super::i32x16<T> {
        super::i32x16::from_repr_unchecked(self.1, T::convert_f32_to_i32(self.1, self.0))
    }

    /// Convert to i32x16 with truncation and UNIFORM saturation:
    /// out-of-range lanes clamp to `i32::MIN`/`i32::MAX` and NaN
    /// lanes become 0, identically on every backend — the semantics
    /// of Rust scalar `as`. Native on NEON/WASM/scalar; on x86 a
    /// 4-op compare/blend fixup over `cvttps` (issue #80).
    #[inline(always)]
    pub fn to_i32_saturating(self) -> super::i32x16<T> {
        super::i32x16::from_repr_unchecked(self.1, T::convert_f32_to_i32_saturating(self.1, self.0))
    }

    /// Convert to i32x16 with rounding to nearest.
    #[inline(always)]
    pub fn to_i32_round(self) -> super::i32x16<T> {
        super::i32x16::from_repr_unchecked(self.1, T::convert_f32_to_i32_round(self.1, self.0))
    }

    /// Create from i32x16 via numeric conversion.
    #[inline(always)]
    pub fn from_i32_t(token: T, v: super::i32x16<T>) -> Self {
        Self(T::convert_i32_to_f32(token, v.into_repr()), token)
    }

    // ====== Backward-compatible aliases (old generated API names) ======

    /// Alias for [`bitcast_to_i32`](Self::bitcast_to_i32).
    #[inline(always)]
    pub fn bitcast_i32x16(self) -> super::i32x16<T> {
        self.bitcast_to_i32()
    }

    /// Alias for [`to_i32`](Self::to_i32).
    #[inline(always)]
    pub fn to_i32x16(self) -> super::i32x16<T> {
        self.to_i32()
    }

    /// Alias for [`to_i32_round`](Self::to_i32_round).
    #[inline(always)]
    pub fn to_i32x16_round(self) -> super::i32x16<T> {
        self.to_i32_round()
    }

    /// Alias for [`from_i32_t`](Self::from_i32_t).
    #[inline(always)]
    pub fn from_i32x16_t(token: T, v: super::i32x16<T>) -> Self {
        Self::from_i32_t(token, v)
    }
}

// ============================================================================
// Platform-specific concrete impls
// ============================================================================

impl f32x16<archmage::ScalarToken> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "scalar::f32x16"
    }
}

#[cfg(target_arch = "x86_64")]
impl f32x16<archmage::X64V3Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "polyfill::v3_512::f32x16"
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl f32x16<archmage::X64V4Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v4::f32x16"
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl f32x16<archmage::X64V4xToken> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v4x::f32x16"
    }
}

#[cfg(target_arch = "aarch64")]
impl f32x16<archmage::NeonToken> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "polyfill::neon_512::f32x16"
    }
}

#[cfg(target_arch = "wasm32")]
impl f32x16<archmage::Wasm128Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "polyfill::wasm128_512::f32x16"
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl f32x16<archmage::X64V4Token> {
    /// Get the raw `__m512` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::x86_64::__m512 {
        self.0
    }

    /// Wrap a raw `__m512` using an existing CPU capability token.
    #[inline(always)]
    pub fn from_m512_t(token: archmage::X64V4Token, value: core::arch::x86_64::__m512) -> Self {
        Self(value, token)
    }

    /// Wrap a raw `__m512` using an explicit CPU capability token.
    ///
    /// The caller does not need a target-feature annotation.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_raw_t(token: archmage::X64V4Token, value: core::arch::x86_64::__m512) -> Self {
        Self(value, token)
    }

    /// Wrap a raw `__m512` in a matching target-feature context.
    ///
    /// Rust requires the caller to enable the `v4` tier's features.
    /// Use an archmage `#[rite(v4)]` helper or `#[arcane]` entry point.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    pub fn from_raw(value: core::arch::x86_64::__m512) -> Self {
        Self(value, archmage::X64V4Token::from_context())
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl f32x16<archmage::X64V4xToken> {
    /// Get the raw `__m512` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::x86_64::__m512 {
        self.0
    }

    /// Wrap a raw `__m512` using an existing CPU capability token.
    #[inline(always)]
    pub fn from_m512_t(token: archmage::X64V4xToken, value: core::arch::x86_64::__m512) -> Self {
        Self(value, token)
    }

    /// Wrap a raw `__m512` using an explicit CPU capability token.
    ///
    /// The caller does not need a target-feature annotation.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_raw_t(token: archmage::X64V4xToken, value: core::arch::x86_64::__m512) -> Self {
        Self(value, token)
    }

    /// Wrap a raw `__m512` in a matching target-feature context.
    ///
    /// Rust requires the caller to enable the `v4x` tier's features.
    /// Use an archmage `#[rite(v4x)]` helper or `#[arcane]` entry point.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes"
    )]
    #[inline]
    pub fn from_raw(value: core::arch::x86_64::__m512) -> Self {
        Self(value, archmage::X64V4xToken::from_context())
    }
}
// Generated deprecated token-constructor forwarders. Do not edit.
impl<T: F32x16Backend> f32x16<T> {
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::splat_t`].\n\nUse `splat_t` to keep explicit-token construction when `splat` becomes tokenless in magetypes 0.10."]
    #[deprecated(note = "Use splat_t(token, v); splat becomes tokenless in magetypes 0.10.")]
    #[forbid(unsafe_code)]
    pub fn splat(token: T, v: f32) -> Self {
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
    pub fn load(token: T, data: &[f32; 16]) -> Self {
        Self::load_t(token, data)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_array_t`].\n\nUse `from_array_t` to keep explicit-token construction when `from_array` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_array_t(token, arr); from_array becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_array(token: T, arr: [f32; 16]) -> Self {
        Self::from_array_t(token, arr)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_slice_t`].\n\nUse `from_slice_t` to keep explicit-token construction when `from_slice` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_slice_t(token, slice); from_slice becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_slice(token: T, slice: &[f32]) -> Self {
        Self::from_slice_t(token, slice)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::partition_slice_t`].\n\nUse `partition_slice_t` to keep explicit-token construction when `partition_slice` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use partition_slice_t(token, data); partition_slice becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn partition_slice(token: T, data: &[f32]) -> (&[[f32; 16]], &[f32]) {
        Self::partition_slice_t(token, data)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::partition_slice_mut_t`].\n\nUse `partition_slice_mut_t` to keep explicit-token construction when `partition_slice_mut` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use partition_slice_mut_t(token, data); partition_slice_mut becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn partition_slice_mut(token: T, data: &mut [f32]) -> (&mut [[f32; 16]], &mut [f32]) {
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
impl<T: crate::simd::backends::F32x16Convert> f32x16<T> {
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_i32_bitcast_t`].\n\nUse `from_i32_bitcast_t` to keep explicit-token construction when `from_i32_bitcast` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_i32_bitcast_t(token, v); from_i32_bitcast becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_i32_bitcast(token: T, v: super::i32x16<T>) -> Self {
        Self::from_i32_bitcast_t(token, v)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_i32_t`].\n\nUse `from_i32_t` to keep explicit-token construction when `from_i32` becomes tokenless in magetypes 0.10."]
    #[deprecated(note = "Use from_i32_t(token, v); from_i32 becomes tokenless in magetypes 0.10.")]
    #[forbid(unsafe_code)]
    pub fn from_i32(token: T, v: super::i32x16<T>) -> Self {
        Self::from_i32_t(token, v)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_i32x16_t`].\n\nUse `from_i32x16_t` to keep explicit-token construction when `from_i32x16` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_i32x16_t(token, v); from_i32x16 becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_i32x16(token: T, v: super::i32x16<T>) -> Self {
        Self::from_i32x16_t(token, v)
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl f32x16<archmage::X64V4Token> {
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_m512_t`].\n\nUse `from_m512_t` to keep explicit-token construction when `from_m512` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_m512_t(token, value); from_m512 becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_m512(token: archmage::X64V4Token, value: core::arch::x86_64::__m512) -> Self {
        Self::from_m512_t(token, value)
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl f32x16<archmage::X64V4xToken> {
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_m512_t`].\n\nUse `from_m512_t` to keep explicit-token construction when `from_m512` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_m512_t(token, value); from_m512 becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_m512(token: archmage::X64V4xToken, value: core::arch::x86_64::__m512) -> Self {
        Self::from_m512_t(token, value)
    }
}

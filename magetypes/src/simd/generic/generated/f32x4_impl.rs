//! Generic `f32x4<T>` — 4-lane f32 SIMD vector parameterized by backend.
//!
//! `T` is a token type (e.g., `X64V3Token`, `NeonToken`, `ScalarToken`)
//! that determines the platform-native representation and intrinsics used.
//! The struct delegates all operations to the [`F32x4Backend`] trait.

#![allow(clippy::should_implement_trait)]

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::simd::backends::F32x4Backend;

/// 4-lane f32 SIMD vector, generic over backend `T`.
///
/// `T` is a token type that proves CPU support for the required SIMD features.
/// The inner representation is `T::Repr` (e.g., `__m128` on x86, `float32x4_t` on ARM).
///
/// **The token is stored** (as a zero-sized field) so methods receiving
/// `self: f32x4<T>` can re-supply it to backend operations that
/// require a token value (e.g. `T::splat(token, v)`). This carries the
/// token-as-feature-proof guarantee through every method call without
/// runtime overhead — `T` is ZST, so `sizeof(f32x4<T>) == sizeof(T::Repr)`
/// and `align_of(f32x4<T>) == align_of(T::Repr)` under `#[repr(C)]`.
///
/// # Layout
///
/// `#[repr(C)]` with a ZST trailing field: `T::Repr` lives at offset 0
/// and `T` plus the sealed policy marker are zero-sized tails. Bitcasts between `f32x4<T>` values of
/// different element-types are sound when the Repr types share a layout
/// (e.g. `__m128` and `__m128i` are both 16-byte aligned 128-bit values).
/// `#[repr(transparent)]` cannot be used because Rust cannot prove at
/// the struct definition site that a generic `T` is a 1-ZST.
///
/// Fixed-policy aliases select explicit-token or feature-context constructors.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct f32x4<
    T: F32x4Backend,
    M: crate::simd::generic::ConstructorMode = crate::simd::generic::Explicit,
>(
    pub(crate) T::Repr,
    pub(crate) T,
    pub(crate) core::marker::PhantomData<M>,
);
// SAFETY: repr(C) pair of Pod storage and a sealed 1-ZST token.
// A supplied T proves CPU support; the wrapper adds no bit invariants.
// Helpers additionally check token size/alignment at monomorphization.
unsafe impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend>
    crate::simd_storage::TokenStorage for f32x4<T, M>
{
    type Token = T;
}

// Layout invariant: struct is `#[repr(C)]` with a trailing ZST `T`
// field, so `sizeof/alignof(f32x4<T>) == sizeof/alignof(T::Repr)`
// when `T` is a 1-ZST. Every archmage token currently satisfies this;
// if a future refactor adds a non-ZST field to a token, this const
// assert fires at compile time.
const _: () = {
    assert!(
        core::mem::size_of::<f32x4<archmage::ScalarToken>>()
            == core::mem::size_of::<
                <archmage::ScalarToken as crate::simd::backends::F32x4Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<f32x4<archmage::ScalarToken>>()
            == core::mem::align_of::<
                <archmage::ScalarToken as crate::simd::backends::F32x4Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "x86_64")]
const _: () = {
    assert!(
        core::mem::size_of::<f32x4<archmage::X64V3Token>>()
            == core::mem::size_of::<
                <archmage::X64V3Token as crate::simd::backends::F32x4Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<f32x4<archmage::X64V3Token>>()
            == core::mem::align_of::<
                <archmage::X64V3Token as crate::simd::backends::F32x4Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "aarch64")]
const _: () = {
    assert!(
        core::mem::size_of::<f32x4<archmage::NeonToken>>()
            == core::mem::size_of::<
                <archmage::NeonToken as crate::simd::backends::F32x4Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<f32x4<archmage::NeonToken>>()
            == core::mem::align_of::<
                <archmage::NeonToken as crate::simd::backends::F32x4Backend>::Repr,
            >()
    );
};

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const _: () = {
    assert!(
        core::mem::size_of::<f32x4<archmage::Wasm128Token>>()
            == core::mem::size_of::<
                <archmage::Wasm128Token as crate::simd::backends::F32x4Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<f32x4<archmage::Wasm128Token>>()
            == core::mem::align_of::<
                <archmage::Wasm128Token as crate::simd::backends::F32x4Backend>::Repr,
            >()
    );
};

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> f32x4<T, M> {
    #[inline(always)]
    pub(crate) fn new_repr(repr: T::Repr, token: T) -> Self {
        Self(repr, token, core::marker::PhantomData)
    }

    /// Number of f32 lanes.
    pub const LANES: usize = 4;

    // ====== Construction (token-gated) ======

    /// Broadcast scalar to all 4 lanes.
    #[inline(always)]
    pub(crate) fn splat_with_token(token: T, v: f32) -> Self {
        Self::new_repr(T::splat(token, v), token)
    }

    /// All lanes zero.
    #[inline(always)]
    pub(crate) fn zero_with_token(token: T) -> Self {
        Self::new_repr(T::zero(token), token)
    }

    /// Load from a `[f32; 4]` array.
    #[inline(always)]
    pub(crate) fn load_with_token(token: T, data: &[f32; 4]) -> Self {
        Self::new_repr(T::load(token, data), token)
    }

    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub(crate) fn from_array_with_token(token: T, arr: [f32; 4]) -> Self {
        Self::new_repr(T::from_array(token, arr), token)
    }

    /// Create from slice. Panics if `slice.len() < 4`.
    #[inline(always)]
    pub(crate) fn from_slice_with_token(token: T, slice: &[f32]) -> Self {
        let arr: [f32; 4] = slice[..4].try_into().unwrap();
        Self::new_repr(T::from_array(token, arr), token)
    }

    /// Split a slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&[[f32; 4]], &[f32])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub(crate) fn partition_slice_with_token(_token: T, data: &[f32]) -> (&[[f32; 4]], &[f32]) {
        data.as_chunks::<4>()
    }

    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&mut [[f32; 4]], &mut [f32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub(crate) fn partition_slice_mut_with_token(
        _token: T,
        data: &mut [f32],
    ) -> (&mut [[f32; 4]], &mut [f32]) {
        data.as_chunks_mut::<4>()
    }

    // ====== Accessors ======

    /// Store to array.
    #[inline(always)]
    pub fn store(self, out: &mut [f32; 4]) {
        T::store(self.1, self.0, out);
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [f32; 4] {
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
    pub fn reduce_add(self) -> f32 {
        T::reduce_add(self.1, self.0)
    }

    /// Minimum across all 4 lanes.
    #[inline(always)]
    pub fn reduce_min(self) -> f32 {
        T::reduce_min(self.1, self.0)
    }

    /// Maximum across all 4 lanes.
    #[inline(always)]
    pub fn reduce_max(self) -> f32 {
        T::reduce_max(self.1, self.0)
    }

    // ====== Approximations ======

    /// Fast reciprocal (1/x), ≥~12-bit floor by the cheapest path per
    /// platform: x86 ~12-bit (raw hardware estimate), ARM ~16-bit
    /// (estimate + one Newton step), WASM/scalar ~24-bit (exact division —
    /// no hardware estimate exists to undercut it). For the same bits on
    /// *every* machine use [`rcp_approx_portable`](Self::rcp_approx_portable);
    /// for ≤4 ULP use [`recip`](Self::recip); for 0 ULP + IEEE rails
    /// use [`recip_portable`](Self::recip_portable).
    ///
    /// Rails are UNSPECIFIED at this tier (the current lowerings
    /// happen to return IEEE values on x86/NEON/WASM, but only
    /// [`recip`](Self::recip) and up contract it).
    #[inline(always)]
    pub fn rcp_approx(self) -> Self {
        // Each backend owns its >=12-bit estimate (x86 raw rcpps; ARM
        // raw vrecpe + 1 fused FRECPS; WASM/scalar exact division).
        Self::new_repr(T::rcp_approx(self.1, self.0), self.1)
    }

    /// Reciprocal (1/x), the working tier: at least ~22 correct bits
    /// (≤4 ULP) **with exact IEEE rails** — `recip(±0) = ±inf`,
    /// `recip(±inf) = ±0` (signed), NaN propagates — on every backend.
    ///
    /// Fastest conforming path per backend: estimate + Newton + a
    /// branchless rail rescue on x86 (+2.6% over the raw Newton form,
    /// vs +21% for exact division); NEON's FRECPS special-cases
    /// `inf·0` in hardware so its refinement passes rails through for
    /// free; WASM/scalar divide (0 ULP there). The #64 footgun
    /// (`(exp_midp() + one).recip()` NaN'ing on saturated lanes)
    /// cannot occur at this tier.
    ///
    /// **Known accuracy limitation:** V3 reciprocal estimates also flush
    /// subnormal outputs: `recip(f32::MAX)` can return zero rather
    /// than the nonzero subnormal division result. This violates the
    /// working precision target even though the input is normal.
    /// Use `recip_portable()` when this range matters.
    ///
    /// **Subnormal inputs are unspecified** — for 0 ULP
    /// everywhere including subnormals, plus bit-identical
    /// cross-arch results, use
    /// [`recip_portable`](Self::recip_portable).
    #[inline(always)]
    pub fn recip(self) -> Self {
        Self::new_repr(T::recip(self.1, self.0), self.1)
    }

    /// Fast reciprocal square root (1/sqrt(x)), ≥~12-bit floor — see
    /// [`rcp_approx`](Self::rcp_approx) for the per-platform strategy.
    ///
    /// Rails are UNSPECIFIED at this tier (the scalar bit-hack in
    /// particular returns garbage at `±0`); [`rsqrt`](Self::rsqrt)
    /// and up contract them.
    #[inline(always)]
    pub fn rsqrt_approx(self) -> Self {
        // ARM uses raw vrsqrte + 1 fused FRSQRTS; WASM/scalar a bit-hack
        // seed + 2 Newton steps; x86 the raw rsqrtps estimate.
        Self::new_repr(T::rsqrt_approx(self.1, self.0), self.1)
    }

    /// Reciprocal square root (1/sqrt(x)), the working tier: ≤4 ULP
    /// **with exact IEEE rails** (`rsqrt(±0) = ±inf`,
    /// `rsqrt(+inf) = +0`, negatives and NaN give NaN) on every
    /// backend — estimate + Newton + rail rescue on x86 (still ~2.8x
    /// faster than exact), hardware-special-cased FRSQRTS on NEON
    /// (free), division on WASM/scalar. Deep-subnormal inputs
    /// unspecified; use [`rsqrt_portable`](Self::rsqrt_portable) for
    /// 0 ULP + subnormals + bit-identical.
    #[inline(always)]
    pub fn rsqrt(self) -> Self {
        Self::new_repr(T::rsqrt(self.1, self.0), self.1)
    }

    // ====== Deterministic (cross-platform bit-identical) reciprocals ======
    //
    // The `_portable` family returns **the same bits on every architecture**
    // (x86 / ARM / WASM). Two ingredients make that hold:
    //   * the seed is an integer bit-trick — pure integer math, so it is
    //     identical by construction (the shift is logical, not arch-dependent);
    //   * the Newton steps use plain IEEE-754 `mul`/`sub`, never `mul_add`
    //     (FMA): WASM SIMD has no fused FMA, and FMA-vs-non-FMA itself
    //     diverges, so fused ops are avoided on purpose.
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
        T: crate::simd::backends::F32x4Convert,
    {
        let t = self.1;
        let bits = self.bitcast_to_i32();
        let seed = Self::from_i32_bitcast_with_token(
            t,
            super::i32x4::splat_with_token(t, 0x5f3759df_i32) - bits.shr_logical_const::<1>(),
        );
        seed.rsqrt_newton_portable(self)
    }

    /// One deterministic Newton refinement step for `1/sqrt(x)`:
    /// `y * (1.5 - 0.5*x*y*y)`, where `self` is the current estimate `y`
    /// and `x` is the original input. Non-FMA → bit-identical everywhere.
    #[inline(always)]
    pub fn rsqrt_newton_portable(self, x: Self) -> Self {
        let t = self.1;
        let half = Self::splat_with_token(t, 0.5);
        let three_halves = Self::splat_with_token(t, 1.5);
        self * (three_halves - half * x * self * self)
    }

    /// Precise reciprocal square root: exact IEEE sqrt + division —
    /// **the 0 ULP tier**, with IEEE rails (`rsqrt(+0) = +inf`,
    /// `rsqrt(+inf) = +0`, negatives give NaN) and bit-identical on
    /// every arch. Costs ~3.6x the working-tier [`rsqrt`](Self::rsqrt)
    /// on Zen-class x86 (and is the faster form on Apple Silicon).
    #[inline(always)]
    pub fn rsqrt_portable(self) -> Self {
        Self::splat_with_token(self.1, 1.0) / self.sqrt()
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
        T: crate::simd::backends::F32x4Convert,
    {
        let t = self.1;
        let bits = self.bitcast_to_i32();
        let seed = Self::from_i32_bitcast_with_token(
            t,
            super::i32x4::splat_with_token(t, 0x7ef127ea_i32) - bits,
        );
        seed.recip_newton_portable(self)
    }

    /// One deterministic Newton refinement step for `1/x`: `y * (2 - x*y)`,
    /// where `self` is the current estimate `y` and `x` is the input.
    #[inline(always)]
    pub fn recip_newton_portable(self, x: Self) -> Self {
        let t = self.1;
        let two = Self::splat_with_token(t, 2.0);
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
        Self::splat_with_token(self.1, 1.0) / self
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

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> Add for f32x4<T, M> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new_repr(T::add(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> Sub for f32x4<T, M> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new_repr(T::sub(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> Mul for f32x4<T, M> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new_repr(T::mul(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> Div for f32x4<T, M> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self::new_repr(T::div(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> Neg for f32x4<T, M> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new_repr(T::neg(self.1, self.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> BitAnd for f32x4<T, M> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self::new_repr(T::bitand(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> BitOr for f32x4<T, M> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self::new_repr(T::bitor(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> BitXor for f32x4<T, M> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self::new_repr(T::bitxor(self.1, self.0, rhs.0), self.1)
    }
}

// ============================================================================
// Assign operators
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> AddAssign for f32x4<T, M> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> SubAssign for f32x4<T, M> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> MulAssign for f32x4<T, M> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> DivAssign for f32x4<T, M> {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> BitAndAssign for f32x4<T, M> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> BitOrAssign for f32x4<T, M> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> BitXorAssign for f32x4<T, M> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

// ============================================================================
// Scalar broadcast operators (v + 2.0, v * 0.5, etc.)
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> Add<f32> for f32x4<T, M> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: f32) -> Self {
        Self::new_repr(T::add(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> Sub<f32> for f32x4<T, M> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: f32) -> Self {
        Self::new_repr(T::sub(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> Mul<f32> for f32x4<T, M> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        Self::new_repr(T::mul(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> Div<f32> for f32x4<T, M> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self {
        Self::new_repr(T::div(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

// ============================================================================
// Index
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> Index<usize> for f32x4<T, M> {
    type Output = f32;
    #[inline(always)]
    fn index(&self, i: usize) -> &f32 {
        &crate::simd_storage::view::<_, [f32; 4]>(&self.0)[i]
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> IndexMut<usize> for f32x4<T, M> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut f32 {
        &mut crate::simd_storage::view_mut::<_, [f32; 4]>(&mut self.0)[i]
    }
}

// ============================================================================
// Conversions
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> From<f32x4<T, M>> for [f32; 4] {
    #[inline(always)]
    fn from(v: f32x4<T, M>) -> [f32; 4] {
        T::to_array(v.1, v.0)
    }
}

// ============================================================================
// Debug
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> core::fmt::Debug for f32x4<T, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let arr = T::to_array(self.1, self.0);
        f.debug_tuple("f32x4").field(&arr).finish()
    }
}

// ============================================================================
// Cross-type conversions (available when T implements conversion traits)
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: crate::simd::backends::F32x4Convert> f32x4<T, M> {
    /// Bitcast to i32x4 (reinterpret bits, no conversion).
    #[inline(always)]
    pub fn bitcast_to_i32(self) -> super::i32x4<T, M> {
        super::i32x4::from_repr_unchecked(self.1, T::bitcast_f32_to_i32(self.1, self.0))
    }

    /// Create from i32x4 via bitcast (reinterpret bits, no conversion).
    #[inline(always)]
    pub(crate) fn from_i32_bitcast_with_token(token: T, v: super::i32x4<T, M>) -> Self {
        Self::new_repr(T::bitcast_i32_to_f32(token, v.into_repr()), token)
    }

    /// Convert to i32x4 with truncation toward zero.
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
    pub fn to_i32(self) -> super::i32x4<T, M> {
        super::i32x4::from_repr_unchecked(self.1, T::convert_f32_to_i32(self.1, self.0))
    }

    /// Convert to i32x4 with truncation and UNIFORM saturation:
    /// out-of-range lanes clamp to `i32::MIN`/`i32::MAX` and NaN
    /// lanes become 0, identically on every backend — the semantics
    /// of Rust scalar `as`. Native on NEON/WASM/scalar; on x86 a
    /// 4-op compare/blend fixup over `cvttps` (issue #80).
    #[inline(always)]
    pub fn to_i32_saturating(self) -> super::i32x4<T, M> {
        super::i32x4::from_repr_unchecked(self.1, T::convert_f32_to_i32_saturating(self.1, self.0))
    }

    /// Convert to i32x4 with rounding to nearest.
    #[inline(always)]
    pub fn to_i32_round(self) -> super::i32x4<T, M> {
        super::i32x4::from_repr_unchecked(self.1, T::convert_f32_to_i32_round(self.1, self.0))
    }

    /// Create from i32x4 via numeric conversion.
    #[inline(always)]
    pub(crate) fn from_i32_with_token(token: T, v: super::i32x4<T, M>) -> Self {
        Self::new_repr(T::convert_i32_to_f32(token, v.into_repr()), token)
    }

    // ====== Backward-compatible aliases (old generated API names) ======

    /// Alias for [`bitcast_to_i32`](Self::bitcast_to_i32).
    #[inline(always)]
    pub fn bitcast_i32x4(self) -> super::i32x4<T, M> {
        self.bitcast_to_i32()
    }

    /// Alias for [`to_i32`](Self::to_i32).
    #[inline(always)]
    pub fn to_i32x4(self) -> super::i32x4<T, M> {
        self.to_i32()
    }

    /// Alias for [`to_i32_round`](Self::to_i32_round).
    #[inline(always)]
    pub fn to_i32x4_round(self) -> super::i32x4<T, M> {
        self.to_i32_round()
    }

    /// Alias for [`from_i32`](Self::from_i32).
    #[inline(always)]
    pub(crate) fn from_i32x4_with_token(token: T, v: super::i32x4<T, M>) -> Self {
        Self::from_i32_with_token(token, v)
    }

    /// Alias for [`bitcast_ref_i32`](Self::bitcast_ref_i32) (from block_ops).
    #[inline(always)]
    pub fn bitcast_ref_i32x4(&self) -> &super::i32x4<T, M> {
        self.bitcast_ref_i32()
    }

    /// Alias for [`bitcast_mut_i32`](Self::bitcast_mut_i32) (from block_ops).
    #[inline(always)]
    pub fn bitcast_mut_i32x4(&mut self) -> &mut super::i32x4<T, M> {
        self.bitcast_mut_i32()
    }
}

// ============================================================================
// Platform-specific concrete impls
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode> f32x4<archmage::ScalarToken, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "scalar::f32x4"
    }
}

#[cfg(target_arch = "x86_64")]
impl<M: crate::simd::generic::ConstructorMode> f32x4<archmage::X64V3Token, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v3::f32x4"
    }

    /// Get the raw `__m128` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::x86_64::__m128 {
        self.0
    }

    /// Create from a raw `__m128` (token-gated, zero-cost).
    #[inline(always)]
    pub(crate) fn from_m128_with_token(
        token: archmage::X64V3Token,
        v: core::arch::x86_64::__m128,
    ) -> Self {
        Self::new_repr(v, token)
    }
}

#[cfg(target_arch = "aarch64")]
impl<M: crate::simd::generic::ConstructorMode> f32x4<archmage::NeonToken, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "arm::neon::f32x4"
    }
}

#[cfg(target_arch = "wasm32")]
impl<M: crate::simd::generic::ConstructorMode> f32x4<archmage::Wasm128Token, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "wasm::wasm128::f32x4"
    }
}
#[cfg(target_arch = "x86_64")]
impl<M: crate::simd::generic::ConstructorMode> f32x4<archmage::X64V3Token, M> {
    /// Wrap a raw `__m128` in a matching target-feature context.
    ///
    /// Rust requires the caller to enable the `v3` tier's features.
    /// Use an archmage `#[rite(v3)]` helper or `#[arcane]` entry point.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_raw(value: core::arch::x86_64::__m128) -> Self {
        Self::new_repr(value, archmage::X64V3Token::from_context())
    }
}
#[cfg(target_arch = "aarch64")]
impl<M: crate::simd::generic::ConstructorMode> f32x4<archmage::NeonToken, M> {
    /// Get the raw `float32x4_t` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::aarch64::float32x4_t {
        self.0
    }

    /// Wrap a raw `float32x4_t` using an existing CPU capability token.
    #[inline(always)]
    pub(crate) fn from_float32x4_t_with_token(
        token: archmage::NeonToken,
        value: core::arch::aarch64::float32x4_t,
    ) -> Self {
        Self::new_repr(value, token)
    }

    /// Wrap a raw `float32x4_t` in a matching target-feature context.
    ///
    /// Rust requires the caller to enable the `neon` tier's features.
    /// Use an archmage `#[rite(neon)]` helper or `#[arcane]` entry point.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn from_raw(value: core::arch::aarch64::float32x4_t) -> Self {
        Self::new_repr(value, archmage::NeonToken::from_context())
    }
}
#[cfg(target_arch = "wasm32")]
impl<M: crate::simd::generic::ConstructorMode> f32x4<archmage::Wasm128Token, M> {
    /// Get the raw `v128` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::wasm32::v128 {
        self.0
    }

    /// Wrap a raw `v128` using an existing CPU capability token.
    #[inline(always)]
    pub(crate) fn from_v128_with_token(
        token: archmage::Wasm128Token,
        value: core::arch::wasm32::v128,
    ) -> Self {
        Self::new_repr(value, token)
    }

    /// Wrap a raw `v128` in a matching target-feature context.
    ///
    /// Rust requires the caller to enable the `wasm128` tier's features.
    /// Use an archmage `#[rite(wasm128)]` helper or `#[arcane]` entry point.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn from_raw(value: core::arch::wasm32::v128) -> Self {
        Self::new_repr(value, archmage::Wasm128Token::from_context())
    }
}
impl<T: F32x4Backend> From<f32x4<T, crate::simd::generic::Explicit>>
    for f32x4<T, crate::simd::generic::Context>
{
    #[inline(always)]
    fn from(value: f32x4<T, crate::simd::generic::Explicit>) -> Self {
        Self::new_repr(value.0, value.1)
    }
}
impl<T: F32x4Backend> From<f32x4<T, crate::simd::generic::Context>>
    for f32x4<T, crate::simd::generic::Explicit>
{
    #[inline(always)]
    fn from(value: f32x4<T, crate::simd::generic::Context>) -> Self {
        Self::new_repr(value.0, value.1)
    }
}

#[cfg(target_arch = "aarch64")]
#[cfg(target_arch = "aarch64")]
impl f32x4<archmage::NeonToken, crate::simd::generic::Context> {
    /// Wrap a raw `float32x4_t` using an existing CPU capability token.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn from_float32x4_t(value: core::arch::aarch64::float32x4_t) -> Self {
        Self::from_float32x4_t_with_token(archmage::NeonToken::from_context(), value)
    }
}

#[cfg(target_arch = "aarch64")]
impl f32x4<archmage::NeonToken, crate::simd::generic::Explicit> {
    /// Wrap a raw `float32x4_t` using an existing CPU capability token.
    #[inline(always)]
    pub fn from_float32x4_t(
        token: archmage::NeonToken,
        value: core::arch::aarch64::float32x4_t,
    ) -> Self {
        Self::from_float32x4_t_with_token(token, value)
    }
}

#[cfg(target_arch = "wasm32")]
#[cfg(target_arch = "wasm32")]
impl f32x4<archmage::Wasm128Token, crate::simd::generic::Context> {
    /// Wrap a raw `v128` using an existing CPU capability token.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn from_v128(value: core::arch::wasm32::v128) -> Self {
        Self::from_v128_with_token(archmage::Wasm128Token::from_context(), value)
    }
}

#[cfg(target_arch = "wasm32")]
impl f32x4<archmage::Wasm128Token, crate::simd::generic::Explicit> {
    /// Wrap a raw `v128` using an existing CPU capability token.
    #[inline(always)]
    pub fn from_v128(token: archmage::Wasm128Token, value: core::arch::wasm32::v128) -> Self {
        Self::from_v128_with_token(token, value)
    }
}

#[cfg(target_arch = "x86_64")]
#[cfg(target_arch = "x86_64")]
impl f32x4<archmage::X64V3Token, crate::simd::generic::Context> {
    /// Create from a raw `__m128` (token-gated, zero-cost).
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_m128(v: core::arch::x86_64::__m128) -> Self {
        Self::from_m128_with_token(archmage::X64V3Token::from_context(), v)
    }
}

#[cfg(target_arch = "x86_64")]
impl f32x4<archmage::X64V3Token, crate::simd::generic::Explicit> {
    /// Create from a raw `__m128` (token-gated, zero-cost).
    #[inline(always)]
    pub fn from_m128(token: archmage::X64V3Token, v: core::arch::x86_64::__m128) -> Self {
        Self::from_m128_with_token(token, v)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl f32x4<archmage::Avx512Fp16Token, crate::simd::generic::Context> {
    /// Broadcast scalar to all 4 lanes.
    #[forbid(unsafe_code)]
    #[archmage::rite(fp16)]
    pub fn splat(v: f32) -> Self {
        Self::splat_with_token(archmage::Avx512Fp16Token::from_context(), v)
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    #[archmage::rite(fp16)]
    pub fn zero() -> Self {
        Self::zero_with_token(archmage::Avx512Fp16Token::from_context())
    }
    /// Load from a `[f32; 4]` array.
    #[forbid(unsafe_code)]
    #[archmage::rite(fp16)]
    pub fn load(data: &[f32; 4]) -> Self {
        Self::load_with_token(archmage::Avx512Fp16Token::from_context(), data)
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    #[archmage::rite(fp16)]
    pub fn from_array(arr: [f32; 4]) -> Self {
        Self::from_array_with_token(archmage::Avx512Fp16Token::from_context(), arr)
    }
    /// Create from slice. Panics if `slice.len() < 4`.
    #[forbid(unsafe_code)]
    #[archmage::rite(fp16)]
    pub fn from_slice(slice: &[f32]) -> Self {
        Self::from_slice_with_token(archmage::Avx512Fp16Token::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[f32; 4]], &[f32])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(fp16)]
    pub fn partition_slice(data: &[f32]) -> (&[[f32; 4]], &[f32]) {
        Self::partition_slice_with_token(archmage::Avx512Fp16Token::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[f32; 4]], &mut [f32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(fp16)]
    pub fn partition_slice_mut(data: &mut [f32]) -> (&mut [[f32; 4]], &mut [f32]) {
        Self::partition_slice_mut_with_token(archmage::Avx512Fp16Token::from_context(), data)
    }
    /// Wrap a platform representation (requires matching target features).
    #[forbid(unsafe_code)]
    #[archmage::rite(fp16)]
    pub fn from_repr(
        repr: <archmage::Avx512Fp16Token as crate::simd::backends::F32x4Backend>::Repr,
    ) -> Self {
        Self::from_repr_with_token(archmage::Avx512Fp16Token::from_context(), repr)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl f32x4<archmage::X64V4Token, crate::simd::generic::Context> {
    /// Broadcast scalar to all 4 lanes.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4)]
    pub fn splat(v: f32) -> Self {
        Self::splat_with_token(archmage::X64V4Token::from_context(), v)
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4)]
    pub fn zero() -> Self {
        Self::zero_with_token(archmage::X64V4Token::from_context())
    }
    /// Load from a `[f32; 4]` array.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4)]
    pub fn load(data: &[f32; 4]) -> Self {
        Self::load_with_token(archmage::X64V4Token::from_context(), data)
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    #[archmage::rite(v4)]
    pub fn from_array(arr: [f32; 4]) -> Self {
        Self::from_array_with_token(archmage::X64V4Token::from_context(), arr)
    }
    /// Create from slice. Panics if `slice.len() < 4`.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4)]
    pub fn from_slice(slice: &[f32]) -> Self {
        Self::from_slice_with_token(archmage::X64V4Token::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[f32; 4]], &[f32])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4)]
    pub fn partition_slice(data: &[f32]) -> (&[[f32; 4]], &[f32]) {
        Self::partition_slice_with_token(archmage::X64V4Token::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[f32; 4]], &mut [f32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4)]
    pub fn partition_slice_mut(data: &mut [f32]) -> (&mut [[f32; 4]], &mut [f32]) {
        Self::partition_slice_mut_with_token(archmage::X64V4Token::from_context(), data)
    }
    /// Wrap a platform representation (requires matching target features).
    #[forbid(unsafe_code)]
    #[archmage::rite(v4)]
    pub fn from_repr(
        repr: <archmage::X64V4Token as crate::simd::backends::F32x4Backend>::Repr,
    ) -> Self {
        Self::from_repr_with_token(archmage::X64V4Token::from_context(), repr)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl f32x4<archmage::X64V4xToken, crate::simd::generic::Context> {
    /// Broadcast scalar to all 4 lanes.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4x)]
    pub fn splat(v: f32) -> Self {
        Self::splat_with_token(archmage::X64V4xToken::from_context(), v)
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4x)]
    pub fn zero() -> Self {
        Self::zero_with_token(archmage::X64V4xToken::from_context())
    }
    /// Load from a `[f32; 4]` array.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4x)]
    pub fn load(data: &[f32; 4]) -> Self {
        Self::load_with_token(archmage::X64V4xToken::from_context(), data)
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    #[archmage::rite(v4x)]
    pub fn from_array(arr: [f32; 4]) -> Self {
        Self::from_array_with_token(archmage::X64V4xToken::from_context(), arr)
    }
    /// Create from slice. Panics if `slice.len() < 4`.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4x)]
    pub fn from_slice(slice: &[f32]) -> Self {
        Self::from_slice_with_token(archmage::X64V4xToken::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[f32; 4]], &[f32])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4x)]
    pub fn partition_slice(data: &[f32]) -> (&[[f32; 4]], &[f32]) {
        Self::partition_slice_with_token(archmage::X64V4xToken::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[f32; 4]], &mut [f32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4x)]
    pub fn partition_slice_mut(data: &mut [f32]) -> (&mut [[f32; 4]], &mut [f32]) {
        Self::partition_slice_mut_with_token(archmage::X64V4xToken::from_context(), data)
    }
    /// Wrap a platform representation (requires matching target features).
    #[forbid(unsafe_code)]
    #[archmage::rite(v4x)]
    pub fn from_repr(
        repr: <archmage::X64V4xToken as crate::simd::backends::F32x4Backend>::Repr,
    ) -> Self {
        Self::from_repr_with_token(archmage::X64V4xToken::from_context(), repr)
    }
}

#[cfg(target_arch = "aarch64")]
impl f32x4<archmage::NeonToken, crate::simd::generic::Context> {
    /// Broadcast scalar to all 4 lanes.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn splat(v: f32) -> Self {
        Self::splat_with_token(archmage::NeonToken::from_context(), v)
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn zero() -> Self {
        Self::zero_with_token(archmage::NeonToken::from_context())
    }
    /// Load from a `[f32; 4]` array.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn load(data: &[f32; 4]) -> Self {
        Self::load_with_token(archmage::NeonToken::from_context(), data)
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn from_array(arr: [f32; 4]) -> Self {
        Self::from_array_with_token(archmage::NeonToken::from_context(), arr)
    }
    /// Create from slice. Panics if `slice.len() < 4`.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn from_slice(slice: &[f32]) -> Self {
        Self::from_slice_with_token(archmage::NeonToken::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[f32; 4]], &[f32])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn partition_slice(data: &[f32]) -> (&[[f32; 4]], &[f32]) {
        Self::partition_slice_with_token(archmage::NeonToken::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[f32; 4]], &mut [f32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn partition_slice_mut(data: &mut [f32]) -> (&mut [[f32; 4]], &mut [f32]) {
        Self::partition_slice_mut_with_token(archmage::NeonToken::from_context(), data)
    }
    /// Wrap a platform representation (requires matching target features).
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn from_repr(
        repr: <archmage::NeonToken as crate::simd::backends::F32x4Backend>::Repr,
    ) -> Self {
        Self::from_repr_with_token(archmage::NeonToken::from_context(), repr)
    }
    /// Create from i32x4 via bitcast (reinterpret bits, no conversion).
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn from_i32_bitcast(
        v: super::i32x4<archmage::NeonToken, crate::simd::generic::Context>,
    ) -> Self {
        Self::from_i32_bitcast_with_token(archmage::NeonToken::from_context(), v)
    }
    /// Create from i32x4 via numeric conversion.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn from_i32(v: super::i32x4<archmage::NeonToken, crate::simd::generic::Context>) -> Self {
        Self::from_i32_with_token(archmage::NeonToken::from_context(), v)
    }
    /// Alias for [`from_i32`](Self::from_i32).
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn from_i32x4(v: super::i32x4<archmage::NeonToken, crate::simd::generic::Context>) -> Self {
        Self::from_i32x4_with_token(archmage::NeonToken::from_context(), v)
    }
}

#[cfg(target_arch = "wasm32")]
impl f32x4<archmage::Wasm128Token, crate::simd::generic::Context> {
    /// Broadcast scalar to all 4 lanes.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn splat(v: f32) -> Self {
        Self::splat_with_token(archmage::Wasm128Token::from_context(), v)
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn zero() -> Self {
        Self::zero_with_token(archmage::Wasm128Token::from_context())
    }
    /// Load from a `[f32; 4]` array.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn load(data: &[f32; 4]) -> Self {
        Self::load_with_token(archmage::Wasm128Token::from_context(), data)
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn from_array(arr: [f32; 4]) -> Self {
        Self::from_array_with_token(archmage::Wasm128Token::from_context(), arr)
    }
    /// Create from slice. Panics if `slice.len() < 4`.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn from_slice(slice: &[f32]) -> Self {
        Self::from_slice_with_token(archmage::Wasm128Token::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[f32; 4]], &[f32])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn partition_slice(data: &[f32]) -> (&[[f32; 4]], &[f32]) {
        Self::partition_slice_with_token(archmage::Wasm128Token::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[f32; 4]], &mut [f32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn partition_slice_mut(data: &mut [f32]) -> (&mut [[f32; 4]], &mut [f32]) {
        Self::partition_slice_mut_with_token(archmage::Wasm128Token::from_context(), data)
    }
    /// Wrap a platform representation (requires matching target features).
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn from_repr(
        repr: <archmage::Wasm128Token as crate::simd::backends::F32x4Backend>::Repr,
    ) -> Self {
        Self::from_repr_with_token(archmage::Wasm128Token::from_context(), repr)
    }
    /// Create from i32x4 via bitcast (reinterpret bits, no conversion).
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn from_i32_bitcast(
        v: super::i32x4<archmage::Wasm128Token, crate::simd::generic::Context>,
    ) -> Self {
        Self::from_i32_bitcast_with_token(archmage::Wasm128Token::from_context(), v)
    }
    /// Create from i32x4 via numeric conversion.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn from_i32(
        v: super::i32x4<archmage::Wasm128Token, crate::simd::generic::Context>,
    ) -> Self {
        Self::from_i32_with_token(archmage::Wasm128Token::from_context(), v)
    }
    /// Alias for [`from_i32`](Self::from_i32).
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn from_i32x4(
        v: super::i32x4<archmage::Wasm128Token, crate::simd::generic::Context>,
    ) -> Self {
        Self::from_i32x4_with_token(archmage::Wasm128Token::from_context(), v)
    }
}

#[cfg(target_arch = "x86_64")]
impl f32x4<archmage::X64V3Token, crate::simd::generic::Context> {
    /// Broadcast scalar to all 4 lanes.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn splat(v: f32) -> Self {
        Self::splat_with_token(archmage::X64V3Token::from_context(), v)
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn zero() -> Self {
        Self::zero_with_token(archmage::X64V3Token::from_context())
    }
    /// Load from a `[f32; 4]` array.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn load(data: &[f32; 4]) -> Self {
        Self::load_with_token(archmage::X64V3Token::from_context(), data)
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_array(arr: [f32; 4]) -> Self {
        Self::from_array_with_token(archmage::X64V3Token::from_context(), arr)
    }
    /// Create from slice. Panics if `slice.len() < 4`.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_slice(slice: &[f32]) -> Self {
        Self::from_slice_with_token(archmage::X64V3Token::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[f32; 4]], &[f32])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn partition_slice(data: &[f32]) -> (&[[f32; 4]], &[f32]) {
        Self::partition_slice_with_token(archmage::X64V3Token::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[f32; 4]], &mut [f32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn partition_slice_mut(data: &mut [f32]) -> (&mut [[f32; 4]], &mut [f32]) {
        Self::partition_slice_mut_with_token(archmage::X64V3Token::from_context(), data)
    }
    /// Wrap a platform representation (requires matching target features).
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_repr(
        repr: <archmage::X64V3Token as crate::simd::backends::F32x4Backend>::Repr,
    ) -> Self {
        Self::from_repr_with_token(archmage::X64V3Token::from_context(), repr)
    }
    /// Create from i32x4 via bitcast (reinterpret bits, no conversion).
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_i32_bitcast(
        v: super::i32x4<archmage::X64V3Token, crate::simd::generic::Context>,
    ) -> Self {
        Self::from_i32_bitcast_with_token(archmage::X64V3Token::from_context(), v)
    }
    /// Create from i32x4 via numeric conversion.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_i32(v: super::i32x4<archmage::X64V3Token, crate::simd::generic::Context>) -> Self {
        Self::from_i32_with_token(archmage::X64V3Token::from_context(), v)
    }
    /// Alias for [`from_i32`](Self::from_i32).
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_i32x4(
        v: super::i32x4<archmage::X64V3Token, crate::simd::generic::Context>,
    ) -> Self {
        Self::from_i32x4_with_token(archmage::X64V3Token::from_context(), v)
    }
}

impl<T: F32x4Backend> f32x4<T, crate::simd::generic::Explicit> {
    /// Broadcast scalar to all 4 lanes.
    #[inline(always)]
    pub fn splat(token: T, v: f32) -> Self {
        Self::splat_with_token(token, v)
    }
    /// All lanes zero.
    #[inline(always)]
    pub fn zero(token: T) -> Self {
        Self::zero_with_token(token)
    }
    /// Load from a `[f32; 4]` array.
    #[inline(always)]
    pub fn load(token: T, data: &[f32; 4]) -> Self {
        Self::load_with_token(token, data)
    }
    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub fn from_array(token: T, arr: [f32; 4]) -> Self {
        Self::from_array_with_token(token, arr)
    }
    /// Create from slice. Panics if `slice.len() < 4`.
    #[inline(always)]
    pub fn from_slice(token: T, slice: &[f32]) -> Self {
        Self::from_slice_with_token(token, slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[f32; 4]], &[f32])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice(_token: T, data: &[f32]) -> (&[[f32; 4]], &[f32]) {
        Self::partition_slice_with_token(_token, data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[f32; 4]], &mut [f32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice_mut(_token: T, data: &mut [f32]) -> (&mut [[f32; 4]], &mut [f32]) {
        Self::partition_slice_mut_with_token(_token, data)
    }
    /// Wrap a platform representation (token-gated).
    #[inline(always)]
    pub fn from_repr(token: T, repr: T::Repr) -> Self {
        Self::from_repr_with_token(token, repr)
    }
}

impl<T: crate::simd::backends::F32x4Convert> f32x4<T, crate::simd::generic::Explicit> {
    /// Create from i32x4 via bitcast (reinterpret bits, no conversion).
    #[inline(always)]
    pub fn from_i32_bitcast(token: T, v: super::i32x4<T, crate::simd::generic::Explicit>) -> Self {
        Self::from_i32_bitcast_with_token(token, v)
    }
    /// Create from i32x4 via numeric conversion.
    #[inline(always)]
    pub fn from_i32(token: T, v: super::i32x4<T, crate::simd::generic::Explicit>) -> Self {
        Self::from_i32_with_token(token, v)
    }
    /// Alias for [`from_i32`](Self::from_i32).
    #[inline(always)]
    pub fn from_i32x4(token: T, v: super::i32x4<T, crate::simd::generic::Explicit>) -> Self {
        Self::from_i32x4_with_token(token, v)
    }
}

impl f32x4<archmage::ScalarToken, crate::simd::generic::Context> {
    /// Broadcast scalar to all 4 lanes.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn splat(v: f32) -> Self {
        Self::splat_with_token(archmage::ScalarToken, v)
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn zero() -> Self {
        Self::zero_with_token(archmage::ScalarToken)
    }
    /// Load from a `[f32; 4]` array.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn load(data: &[f32; 4]) -> Self {
        Self::load_with_token(archmage::ScalarToken, data)
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_array(arr: [f32; 4]) -> Self {
        Self::from_array_with_token(archmage::ScalarToken, arr)
    }
    /// Create from slice. Panics if `slice.len() < 4`.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_slice(slice: &[f32]) -> Self {
        Self::from_slice_with_token(archmage::ScalarToken, slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[f32; 4]], &[f32])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn partition_slice(data: &[f32]) -> (&[[f32; 4]], &[f32]) {
        Self::partition_slice_with_token(archmage::ScalarToken, data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[f32; 4]], &mut [f32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn partition_slice_mut(data: &mut [f32]) -> (&mut [[f32; 4]], &mut [f32]) {
        Self::partition_slice_mut_with_token(archmage::ScalarToken, data)
    }
    /// Wrap a platform representation (requires matching target features).
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_repr(
        repr: <archmage::ScalarToken as crate::simd::backends::F32x4Backend>::Repr,
    ) -> Self {
        Self::from_repr_with_token(archmage::ScalarToken, repr)
    }
    /// Create from i32x4 via bitcast (reinterpret bits, no conversion).
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_i32_bitcast(
        v: super::i32x4<archmage::ScalarToken, crate::simd::generic::Context>,
    ) -> Self {
        Self::from_i32_bitcast_with_token(archmage::ScalarToken, v)
    }
    /// Create from i32x4 via numeric conversion.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_i32(v: super::i32x4<archmage::ScalarToken, crate::simd::generic::Context>) -> Self {
        Self::from_i32_with_token(archmage::ScalarToken, v)
    }
    /// Alias for [`from_i32`](Self::from_i32).
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_i32x4(
        v: super::i32x4<archmage::ScalarToken, crate::simd::generic::Context>,
    ) -> Self {
        Self::from_i32x4_with_token(archmage::ScalarToken, v)
    }
}

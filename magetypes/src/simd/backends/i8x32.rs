//! Backend trait for `i8x32<T>` — 32-lane i8 SIMD vector.
//!
//! Each token type implements this trait with its platform-native representation.
//! The generic wrapper `i8x32<T>` delegates all operations to these trait methods.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

use super::sealed::Sealed;
use archmage::SimdToken;

/// Backend implementation for 32-lane i8 SIMD vectors.
///
/// Trait methods take `self` (the token) as receiver — the token value
/// is the proof of CPU support, and requiring it as the receiver means
/// the methods cannot be invoked via UFCS without holding one. The
/// implementing type `Self` (a token type) determines which platform
/// intrinsics are used. All methods are `#[inline(always)]` in
/// implementations.
///
/// # Sealed
///
/// This trait is sealed — only archmage token types can implement it.
/// The token proves CPU support was verified via `summon()`.
pub trait I8x32Backend: SimdToken + Sealed + Copy + 'static {
    /// Platform-native SIMD representation.
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;

    // ====== Construction ======

    /// Broadcast scalar to all 32 lanes.
    fn splat(self, v: i8) -> Self::Repr;

    /// All lanes zero.
    fn zero(self) -> Self::Repr;

    /// Load from an aligned array.
    fn load(self, data: &[i8; 32]) -> Self::Repr;

    /// Create from array (zero-cost transmute where possible).
    fn from_array(self, arr: [i8; 32]) -> Self::Repr;

    /// Store to array.
    fn store(self, repr: Self::Repr, out: &mut [i8; 32]);

    /// Convert to array.
    fn to_array(self, repr: Self::Repr) -> [i8; 32];

    // ====== Arithmetic ======

    /// Lane-wise addition (wrapping).
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise subtraction (wrapping).
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    /// Lane-wise negation.
    fn neg(self, a: Self::Repr) -> Self::Repr;

    // ====== Math ======

    /// Lane-wise minimum.
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise maximum.
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    /// Lane-wise absolute value.
    fn abs(self, a: Self::Repr) -> Self::Repr;

    // ====== Comparisons ======

    /// Lane-wise equality.
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise inequality.
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise less-than.
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise less-than-or-equal.
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise greater-than.
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise greater-than-or-equal.
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Select lanes: where mask is all-1s pick `if_true`, else `if_false`.
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;

    // ====== Reductions ======

    /// Sum all 32 lanes (wrapping).
    fn reduce_add(self, a: Self::Repr) -> i8;

    // ====== Bitwise ======

    /// Bitwise NOT.
    fn not(self, a: Self::Repr) -> Self::Repr;

    /// Bitwise AND.
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Bitwise OR.
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Bitwise XOR.
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    // ====== Shifts ======

    /// Shift left by constant.
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;

    /// Logical shift right by constant (zero-filling).
    /// `N` must be in `0..=lane_bits-1`; the generic front-ends reject out-of-range `N` at compile time.
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    /// Arithmetic shift right by constant (sign-extending).
    /// `N` must be in `0..=lane_bits-1`; the generic front-ends reject out-of-range `N` at compile time.
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;

    // ====== Saturating arithmetic ======

    /// Lane-wise addition that clamps to the element range instead of
    /// wrapping (`core`'s `saturating_add`, per lane).
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise subtraction that clamps to the element range instead
    /// of wrapping (`core`'s `saturating_sub`, per lane).
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    // ====== Boolean ======

    /// True if all lanes have their sign bit set (all-1s mask).
    fn all_true(self, a: Self::Repr) -> bool;

    /// True if any lane has its sign bit set.
    fn any_true(self, a: Self::Repr) -> bool;

    /// Extract the high bit of each lane as a bitmask.
    fn bitmask(self, a: Self::Repr) -> u32;

    // ====== Default implementations ======

    /// Clamp values between lo and hi.
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as I8x32Backend>::min(self, <Self as I8x32Backend>::max(self, a, lo), hi)
    }

    /// Widen in natural lane order: result[i] = a[i + 0] as i16.
    fn widen_low_i8_to_i16(
        self,
        a: <Self as super::I8x32Backend>::Repr,
    ) -> <Self as super::I16x16Backend>::Repr
    where
        Self: super::I16x16Backend;
    /// Widen in natural lane order: result[i] = a[i + 16] as i16.
    fn widen_high_i8_to_i16(
        self,
        a: <Self as super::I8x32Backend>::Repr,
    ) -> <Self as super::I16x16Backend>::Repr
    where
        Self: super::I16x16Backend;

    // ====== Cross-vector element shift ======

    /// Lanes `N..N+32` of the concatenation `[lo, hi]`.
    ///
    /// `N == 0` returns `lo`; `N == 32` would return `hi` and is rejected,
    /// since a caller that wants `hi` should just use it. This is the
    /// "funnel shift" — `valignd` on AVX-512, `vperm2f128` + `vpalignr` on
    /// AVX2, `EXT` on NEON, `i8x16.shuffle` on wasm — and it is what a
    /// 3-tap horizontal filter needs to derive the `x-1` and `x+1` vectors
    /// from two loads instead of three, and what a byte-shuffling kernel
    /// needs to slide a window.
    ///
    /// `N` is `i32` because that is the type of the ISA immediates it
    /// forwards to; a const generic cannot be cast in a const position.
    ///
    /// The default body is a lane gather, which LLVM does **not** recover
    /// into a funnel shift (measured 2026-09-08: 6-7 scalar moves where the
    /// native form is 1-2 instructions). It is the correctness fallback and
    /// the differential-test reference; every backend whose ISA has the
    /// instruction overrides it.
    #[inline(always)]
    fn concat_shift<const N: i32>(self, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        const { assert!(N >= 0 && N < 32, "concat_shift: N must be in 0..32") };
        let n = N as usize;
        let a = <Self as I8x32Backend>::to_array(self, lo);
        let b = <Self as I8x32Backend>::to_array(self, hi);
        <Self as I8x32Backend>::from_array(
            self,
            core::array::from_fn(|i| if n + i < 32 { a[n + i] } else { b[n + i - 32] }),
        )
    }
}

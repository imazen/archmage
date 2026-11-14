//! Compile-time lane broadcast, zero-filling whole-vector lane shifts, and
//! lane-indexed multiply-add on `f32x4<T>` and `f32x8<T>`.
//!
//! All operations use **whole-vector** lane indexing (`0..4` on `f32x4<T>`,
//! `0..8` on `f32x8<T>`), never per-128-bit-block indexing:
//! - [`f32x4::broadcast_lane`] / [`f32x8::broadcast_lane`]: splat lane `N` across all lanes.
//! - [`f32x4::shift_left_lanes`] / [`f32x8::shift_left_lanes`]: shift lanes toward higher indices by `N`, zero-filling low lanes `0..N` (`[a0, a1, a2, a3].shift_left_lanes::<1>() == [0.0, a0, a1, a2]`).
//! - [`f32x4::shift_right_lanes`] / [`f32x8::shift_right_lanes`]: shift lanes toward lower indices by `N`, zero-filling high lanes (`[a0, a1, a2, a3].shift_right_lanes::<1>() == [a1, a2, a3, 0.0]`).
//! - [`f32x4::mul_add_lane_relaxed`] / [`f32x8::mul_add_lane_relaxed`]: compute `self * v_in[LANE] + acc` (`vfmaq_laneq_f32` on ARM64 NEON with zero `dup` instructions). For `f32x8<T>` indexing a 4-lane `f32x4<T>` input group directly, see [`f32x8::mul_add_lane4_relaxed`].

#![forbid(unsafe_code)]

use crate::simd::backends::{F32x4Backend, F32x8Backend};
use crate::simd::generic::{f32x4, f32x8};

impl<T: F32x4Backend> f32x4<T> {
    /// Broadcast lane `N` (`0..4`) across all 4 lanes.
    ///
    /// Fails to compile if `N >= 4`.
    #[inline(always)]
    pub fn broadcast_lane<const N: usize>(self) -> Self {
        const { assert!(N < 4, "f32x4::broadcast_lane index N must be in 0..4") };
        Self(T::broadcast_lane::<N>(self.1, self.0), self.1)
    }

    /// Shift lanes toward higher indices by `N` (`0..=4`), filling vacated low
    /// lanes with `0.0`.
    ///
    /// `[a0, a1, a2, a3].shift_left_lanes::<1>()` yields `[0.0, a0, a1, a2]`.
    /// Fails to compile if `N > 4`.
    #[inline(always)]
    pub fn shift_left_lanes<const N: usize>(self) -> Self {
        const { assert!(N <= 4, "f32x4::shift_left_lanes count N must be in 0..=4") };
        Self(T::shift_left_lanes::<N>(self.1, self.0), self.1)
    }

    /// Shift lanes toward lower indices by `N` (`0..=4`), filling vacated high
    /// lanes with `0.0`.
    ///
    /// `[a0, a1, a2, a3].shift_right_lanes::<1>()` yields `[a1, a2, a3, 0.0]`.
    /// Fails to compile if `N > 4`.
    #[inline(always)]
    pub fn shift_right_lanes<const N: usize>(self) -> Self {
        const { assert!(N <= 4, "f32x4::shift_right_lanes count N must be in 0..=4") };
        Self(T::shift_right_lanes::<N>(self.1, self.0), self.1)
    }

    /// Lane-indexed multiply-add: `self * v_in[LANE] + acc` (`LANE` in `0..4`).
    ///
    /// Uses single-instruction `vfmaq_laneq_f32::<LANE>` (`fmla v_acc.4s, v_w.4s, v_in.s[LANE]`)
    /// on ARM64 NEON, `_mm_permute_ps` + `_mm_fmadd_ps` on x86 AVX2/FMA, and
    /// `i32x4.shuffle` + FMA/madd on WASM. Rounding matches [`Self::mul_add`].
    /// Fails to compile if `LANE` is outside `0..4`.
    #[inline(always)]
    pub fn mul_add_lane_relaxed<const LANE: i32>(self, v_in: Self, acc: Self) -> Self {
        const {
            assert!(
                LANE >= 0 && LANE < 4,
                "f32x4::mul_add_lane_relaxed LANE must be in 0..4"
            )
        };
        Self(
            T::mul_add_lane_relaxed::<LANE>(self.1, self.0, v_in.0, acc.0),
            self.1,
        )
    }
}

impl<T: F32x8Backend> f32x8<T> {
    /// Broadcast whole-vector lane `N` (`0..8`) across all 8 lanes.
    ///
    /// Unlike C++ Highway's per-128-bit-block `Broadcast`, `N` indexes the full
    /// 8-lane vector identically on x86 AVX2, ARM64 NEON, WASM, and Scalar.
    /// Fails to compile if `N >= 8`.
    #[inline(always)]
    pub fn broadcast_lane<const N: usize>(self) -> Self {
        const { assert!(N < 8, "f32x8::broadcast_lane index N must be in 0..8") };
        Self(T::broadcast_lane::<N>(self.1, self.0), self.1)
    }

    /// Shift whole-vector lanes toward higher indices by `N` (`0..=8`), filling
    /// vacated low lanes with `0.0` (crossing the 128-bit lane boundary on AVX2).
    ///
    /// `[a0, a1, a2, a3, a4, a5, a6, a7].shift_left_lanes::<1>()` yields
    /// `[0.0, a0, a1, a2, a3, a4, a5, a6]`. Fails to compile if `N > 8`.
    #[inline(always)]
    pub fn shift_left_lanes<const N: usize>(self) -> Self {
        const { assert!(N <= 8, "f32x8::shift_left_lanes count N must be in 0..=8") };
        Self(T::shift_left_lanes::<N>(self.1, self.0), self.1)
    }

    /// Shift whole-vector lanes toward lower indices by `N` (`0..=8`), filling
    /// vacated high lanes with `0.0` (crossing the 128-bit lane boundary on AVX2).
    ///
    /// `[a0, a1, a2, a3, a4, a5, a6, a7].shift_right_lanes::<1>()` yields
    /// `[a1, a2, a3, a4, a5, a6, a7, 0.0]`. Fails to compile if `N > 8`.
    #[inline(always)]
    pub fn shift_right_lanes<const N: usize>(self) -> Self {
        const { assert!(N <= 8, "f32x8::shift_right_lanes count N must be in 0..=8") };
        Self(T::shift_right_lanes::<N>(self.1, self.0), self.1)
    }

    /// Whole-vector lane-indexed multiply-add: `self * v_in[LANE] + acc` (`LANE` in `0..8`).
    ///
    /// On ARM64 NEON this lowers to two `vfmaq_laneq_f32` instructions sharing the
    /// selected half of `v_in` with zero `dup` instructions. When `v_in` is a 4-lane
    /// [`f32x4<T>`] input group, prefer [`Self::mul_add_lane4_relaxed`].
    /// Fails to compile if `LANE` is outside `0..8`.
    #[inline(always)]
    pub fn mul_add_lane_relaxed<const LANE: i32>(self, v_in: Self, acc: Self) -> Self {
        const {
            assert!(
                LANE >= 0 && LANE < 8,
                "f32x8::mul_add_lane_relaxed LANE must be in 0..8"
            )
        };
        Self(
            T::mul_add_lane_relaxed::<LANE>(self.1, self.0, v_in.0, acc.0),
            self.1,
        )
    }
}

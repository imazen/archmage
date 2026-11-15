//! Blanket implementations of all 45 public backend, conversion, and bitcast
//! traits (plus `F64x8Convert`) over every sealed archmage token.
//!
//! Routes through `T::Narrow` / `T::W512` (`backends/ops.rs`) and optional
//! hardware overrides on `Sealed`.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.
//!
//! # Safety (audit contract — checked backend boundaries)
//!
//! `Sealed` is crate-private and implemented only for archmage tokens whose
//! `Narrow` / `W512` projections invoke `.v3()` / `.v4()` / `.neon()` /
//! `.wasm128()` safe extractors or `ScalarToken`. Hardware override hooks
//! forward to concrete `#[arcane]` / target-feature-proven functions.
//! Every method delegates safely without `unsafe` or raw pointers.

#![allow(clippy::too_many_arguments, unused_imports)]

use crate::simd::backends::sealed::Sealed;
use crate::simd::backends::*;

impl<T: archmage::SimdToken + Sealed + Copy + 'static> F32x4Backend for T {
    type Repr = <T::Narrow as ops::F32x4Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: f32) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::splat(self.narrow(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::zero(self.narrow())
    }
    #[inline(always)]
    fn load(self, data: &[f32; 4]) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::load(self.narrow(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [f32; 4]) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::from_array(self.narrow(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [f32; 4]) {
        <T::Narrow as ops::F32x4Ops>::store(self.narrow(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [f32; 4] {
        <T::Narrow as ops::F32x4Ops>::to_array(self.narrow(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::mul(self.narrow(), a, b)
    }
    #[inline(always)]
    fn div(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::div(self.narrow(), a, b)
    }
    #[inline(always)]
    fn neg(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::neg(self.narrow(), a)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::min(self.narrow(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::max(self.narrow(), a, b)
    }
    #[inline(always)]
    fn min_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        self.wasm_fma_f32x4_min_fast(a, b)
    }
    #[inline(always)]
    fn max_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        self.wasm_fma_f32x4_max_fast(a, b)
    }
    #[inline(always)]
    fn sqrt(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::sqrt(self.narrow(), a)
    }
    #[inline(always)]
    fn abs(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::abs(self.narrow(), a)
    }
    #[inline(always)]
    fn floor(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::floor(self.narrow(), a)
    }
    #[inline(always)]
    fn ceil(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::ceil(self.narrow(), a)
    }
    #[inline(always)]
    fn round(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::round(self.narrow(), a)
    }
    #[inline(always)]
    fn mul_add(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        self.wasm_fma_f32x4_mul_add(a, b, c)
    }
    #[inline(always)]
    fn mul_sub(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        self.wasm_fma_f32x4_mul_sub(a, b, c)
    }
    #[inline(always)]
    fn mul_add_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        self.wasm_fma_f32x4_mul_add_portable(a, b, c)
    }
    #[inline(always)]
    fn mul_sub_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        self.wasm_fma_f32x4_mul_sub_portable(a, b, c)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::simd_eq(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::simd_ne(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::simd_lt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::simd_le(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::simd_gt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::simd_ge(self.narrow(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::blend(self.narrow(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> f32 {
        <T::Narrow as ops::F32x4Ops>::reduce_add(self.narrow(), a)
    }
    #[inline(always)]
    fn reduce_min(self, a: Self::Repr) -> f32 {
        <T::Narrow as ops::F32x4Ops>::reduce_min(self.narrow(), a)
    }
    #[inline(always)]
    fn reduce_max(self, a: Self::Repr) -> f32 {
        <T::Narrow as ops::F32x4Ops>::reduce_max(self.narrow(), a)
    }
    #[inline(always)]
    fn rcp_approx(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::rcp_approx(self.narrow(), a)
    }
    #[inline(always)]
    fn rsqrt_approx(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::rsqrt_approx(self.narrow(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::not(self.narrow(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::bitand(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::bitor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::bitxor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::clamp(self.narrow(), a, lo, hi)
    }
    #[inline(always)]
    fn recip(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::recip(self.narrow(), a)
    }
    #[inline(always)]
    fn rsqrt(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::rsqrt(self.narrow(), a)
    }
    #[inline(always)]
    fn broadcast_lane<const N: usize>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::broadcast_lane::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shift_left_lanes<const N: usize>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::shift_left_lanes::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shift_right_lanes<const N: usize>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x4Ops>::shift_right_lanes::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn mul_add_lane_relaxed<const LANE: i32>(
        self,
        a: Self::Repr,
        v_in: Self::Repr,
        acc: Self::Repr,
    ) -> Self::Repr {
        self.wasm_fma_f32x4_mul_add_lane_relaxed::<LANE>(a, v_in, acc)
    }
    #[inline(always)]
    fn to_u8_bytes(self, a: Self::Repr) -> [u8; 4] {
        self.v4_f32x4_to_u8_bytes(a)
    }
    #[inline(always)]
    fn store_rgba_bytes(
        self,
        r: Self::Repr,
        g: Self::Repr,
        b: Self::Repr,
        a: Self::Repr,
    ) -> [u8; 16] {
        self.v4_f32x4_store_rgba_bytes(r, g, b, a)
    }
    #[inline(always)]
    fn deinterleave_3ch_repr(self, rgb: [Self::Repr; 3]) -> [Self::Repr; 3] {
        <T::Narrow as ops::F32x4Ops>::deinterleave_3ch_repr(self.narrow(), rgb)
    }
    #[inline(always)]
    fn interleave_3ch_repr(self, channels: [Self::Repr; 3]) -> [Self::Repr; 3] {
        <T::Narrow as ops::F32x4Ops>::interleave_3ch_repr(self.narrow(), channels)
    }
    #[inline(always)]
    fn load_rgb_bytes(self, rgb: &[u8; 12]) -> (Self::Repr, Self::Repr, Self::Repr) {
        <T::Narrow as ops::F32x4Ops>::load_rgb_bytes(self.narrow(), rgb)
    }
    #[inline(always)]
    fn store_rgb_bytes(self, r: Self::Repr, g: Self::Repr, b: Self::Repr) -> [u8; 12] {
        self.v4_f32x4_store_rgb_bytes(r, g, b)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> F32x8Backend for T {
    type Repr = <T::Narrow as ops::F32x8Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: f32) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::splat(self.narrow(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::zero(self.narrow())
    }
    #[inline(always)]
    fn load(self, data: &[f32; 8]) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::load(self.narrow(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [f32; 8]) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::from_array(self.narrow(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [f32; 8]) {
        <T::Narrow as ops::F32x8Ops>::store(self.narrow(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [f32; 8] {
        <T::Narrow as ops::F32x8Ops>::to_array(self.narrow(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::mul(self.narrow(), a, b)
    }
    #[inline(always)]
    fn div(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::div(self.narrow(), a, b)
    }
    #[inline(always)]
    fn neg(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::neg(self.narrow(), a)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::min(self.narrow(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::max(self.narrow(), a, b)
    }
    #[inline(always)]
    fn min_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        self.wasm_fma_f32x8_min_fast(a, b)
    }
    #[inline(always)]
    fn max_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        self.wasm_fma_f32x8_max_fast(a, b)
    }
    #[inline(always)]
    fn sqrt(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::sqrt(self.narrow(), a)
    }
    #[inline(always)]
    fn abs(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::abs(self.narrow(), a)
    }
    #[inline(always)]
    fn floor(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::floor(self.narrow(), a)
    }
    #[inline(always)]
    fn ceil(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::ceil(self.narrow(), a)
    }
    #[inline(always)]
    fn round(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::round(self.narrow(), a)
    }
    #[inline(always)]
    fn mul_add(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        self.wasm_fma_f32x8_mul_add(a, b, c)
    }
    #[inline(always)]
    fn mul_sub(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        self.wasm_fma_f32x8_mul_sub(a, b, c)
    }
    #[inline(always)]
    fn mul_add_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        self.wasm_fma_f32x8_mul_add_portable(a, b, c)
    }
    #[inline(always)]
    fn mul_sub_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        self.wasm_fma_f32x8_mul_sub_portable(a, b, c)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::simd_eq(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::simd_ne(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::simd_lt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::simd_le(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::simd_gt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::simd_ge(self.narrow(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::blend(self.narrow(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> f32 {
        <T::Narrow as ops::F32x8Ops>::reduce_add(self.narrow(), a)
    }
    #[inline(always)]
    fn reduce_min(self, a: Self::Repr) -> f32 {
        <T::Narrow as ops::F32x8Ops>::reduce_min(self.narrow(), a)
    }
    #[inline(always)]
    fn reduce_max(self, a: Self::Repr) -> f32 {
        <T::Narrow as ops::F32x8Ops>::reduce_max(self.narrow(), a)
    }
    #[inline(always)]
    fn rcp_approx(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::rcp_approx(self.narrow(), a)
    }
    #[inline(always)]
    fn rsqrt_approx(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::rsqrt_approx(self.narrow(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::not(self.narrow(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::bitand(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::bitor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::bitxor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::clamp(self.narrow(), a, lo, hi)
    }
    #[inline(always)]
    fn recip(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::recip(self.narrow(), a)
    }
    #[inline(always)]
    fn rsqrt(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::rsqrt(self.narrow(), a)
    }
    #[inline(always)]
    fn broadcast_lane<const N: usize>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::broadcast_lane::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shift_left_lanes<const N: usize>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::shift_left_lanes::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shift_right_lanes<const N: usize>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F32x8Ops>::shift_right_lanes::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn mul_add_lane_relaxed<const LANE: i32>(
        self,
        a: Self::Repr,
        v_in: Self::Repr,
        acc: Self::Repr,
    ) -> Self::Repr {
        self.wasm_fma_f32x8_mul_add_lane_relaxed::<LANE>(a, v_in, acc)
    }
    #[inline(always)]
    fn to_u8_bytes(self, a: Self::Repr) -> [u8; 8] {
        self.v4_f32x8_to_u8_bytes(a)
    }
    #[inline(always)]
    fn store_rgba_bytes(
        self,
        r: Self::Repr,
        g: Self::Repr,
        b: Self::Repr,
        a: Self::Repr,
    ) -> [u8; 32] {
        <T::Narrow as ops::F32x8Ops>::store_rgba_bytes(self.narrow(), r, g, b, a)
    }
    #[inline(always)]
    fn deinterleave_3ch_repr(self, rgb: [Self::Repr; 3]) -> [Self::Repr; 3] {
        <T::Narrow as ops::F32x8Ops>::deinterleave_3ch_repr(self.narrow(), rgb)
    }
    #[inline(always)]
    fn interleave_3ch_repr(self, channels: [Self::Repr; 3]) -> [Self::Repr; 3] {
        <T::Narrow as ops::F32x8Ops>::interleave_3ch_repr(self.narrow(), channels)
    }
    #[inline(always)]
    fn load_rgb_bytes(self, rgb: &[u8; 24]) -> (Self::Repr, Self::Repr, Self::Repr) {
        <T::Narrow as ops::F32x8Ops>::load_rgb_bytes(self.narrow(), rgb)
    }
    #[inline(always)]
    fn store_rgb_bytes(self, r: Self::Repr, g: Self::Repr, b: Self::Repr) -> [u8; 24] {
        <T::Narrow as ops::F32x8Ops>::store_rgb_bytes(self.narrow(), r, g, b)
    }
    #[inline(always)]
    fn transpose_8x8_repr(self, rows: [Self::Repr; 8]) -> [Self::Repr; 8] {
        <T::Narrow as ops::F32x8Ops>::transpose_8x8_repr(self.narrow(), rows)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> F64x2Backend for T {
    type Repr = <T::Narrow as ops::F64x2Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: f64) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::splat(self.narrow(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::zero(self.narrow())
    }
    #[inline(always)]
    fn load(self, data: &[f64; 2]) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::load(self.narrow(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [f64; 2]) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::from_array(self.narrow(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [f64; 2]) {
        <T::Narrow as ops::F64x2Ops>::store(self.narrow(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [f64; 2] {
        <T::Narrow as ops::F64x2Ops>::to_array(self.narrow(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::mul(self.narrow(), a, b)
    }
    #[inline(always)]
    fn div(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::div(self.narrow(), a, b)
    }
    #[inline(always)]
    fn neg(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::neg(self.narrow(), a)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::min(self.narrow(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::max(self.narrow(), a, b)
    }
    #[inline(always)]
    fn min_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        self.wasm_fma_f64x2_min_fast(a, b)
    }
    #[inline(always)]
    fn max_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        self.wasm_fma_f64x2_max_fast(a, b)
    }
    #[inline(always)]
    fn sqrt(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::sqrt(self.narrow(), a)
    }
    #[inline(always)]
    fn abs(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::abs(self.narrow(), a)
    }
    #[inline(always)]
    fn floor(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::floor(self.narrow(), a)
    }
    #[inline(always)]
    fn ceil(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::ceil(self.narrow(), a)
    }
    #[inline(always)]
    fn round(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::round(self.narrow(), a)
    }
    #[inline(always)]
    fn mul_add(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        self.wasm_fma_f64x2_mul_add(a, b, c)
    }
    #[inline(always)]
    fn mul_sub(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        self.wasm_fma_f64x2_mul_sub(a, b, c)
    }
    #[inline(always)]
    fn mul_add_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        self.wasm_fma_f64x2_mul_add_portable(a, b, c)
    }
    #[inline(always)]
    fn mul_sub_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        self.wasm_fma_f64x2_mul_sub_portable(a, b, c)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::simd_eq(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::simd_ne(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::simd_lt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::simd_le(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::simd_gt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::simd_ge(self.narrow(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::blend(self.narrow(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> f64 {
        <T::Narrow as ops::F64x2Ops>::reduce_add(self.narrow(), a)
    }
    #[inline(always)]
    fn reduce_min(self, a: Self::Repr) -> f64 {
        <T::Narrow as ops::F64x2Ops>::reduce_min(self.narrow(), a)
    }
    #[inline(always)]
    fn reduce_max(self, a: Self::Repr) -> f64 {
        <T::Narrow as ops::F64x2Ops>::reduce_max(self.narrow(), a)
    }
    #[inline(always)]
    fn rcp_approx(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::rcp_approx(self.narrow(), a)
    }
    #[inline(always)]
    fn rsqrt_approx(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::rsqrt_approx(self.narrow(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::not(self.narrow(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::bitand(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::bitor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::bitxor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::clamp(self.narrow(), a, lo, hi)
    }
    #[inline(always)]
    fn recip(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::recip(self.narrow(), a)
    }
    #[inline(always)]
    fn rsqrt(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x2Ops>::rsqrt(self.narrow(), a)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> F64x4Backend for T {
    type Repr = <T::Narrow as ops::F64x4Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: f64) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::splat(self.narrow(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::zero(self.narrow())
    }
    #[inline(always)]
    fn load(self, data: &[f64; 4]) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::load(self.narrow(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [f64; 4]) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::from_array(self.narrow(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [f64; 4]) {
        <T::Narrow as ops::F64x4Ops>::store(self.narrow(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [f64; 4] {
        <T::Narrow as ops::F64x4Ops>::to_array(self.narrow(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::mul(self.narrow(), a, b)
    }
    #[inline(always)]
    fn div(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::div(self.narrow(), a, b)
    }
    #[inline(always)]
    fn neg(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::neg(self.narrow(), a)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::min(self.narrow(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::max(self.narrow(), a, b)
    }
    #[inline(always)]
    fn min_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        self.wasm_fma_f64x4_min_fast(a, b)
    }
    #[inline(always)]
    fn max_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        self.wasm_fma_f64x4_max_fast(a, b)
    }
    #[inline(always)]
    fn sqrt(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::sqrt(self.narrow(), a)
    }
    #[inline(always)]
    fn abs(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::abs(self.narrow(), a)
    }
    #[inline(always)]
    fn floor(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::floor(self.narrow(), a)
    }
    #[inline(always)]
    fn ceil(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::ceil(self.narrow(), a)
    }
    #[inline(always)]
    fn round(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::round(self.narrow(), a)
    }
    #[inline(always)]
    fn mul_add(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        self.wasm_fma_f64x4_mul_add(a, b, c)
    }
    #[inline(always)]
    fn mul_sub(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        self.wasm_fma_f64x4_mul_sub(a, b, c)
    }
    #[inline(always)]
    fn mul_add_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        self.wasm_fma_f64x4_mul_add_portable(a, b, c)
    }
    #[inline(always)]
    fn mul_sub_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        self.wasm_fma_f64x4_mul_sub_portable(a, b, c)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::simd_eq(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::simd_ne(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::simd_lt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::simd_le(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::simd_gt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::simd_ge(self.narrow(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::blend(self.narrow(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> f64 {
        <T::Narrow as ops::F64x4Ops>::reduce_add(self.narrow(), a)
    }
    #[inline(always)]
    fn reduce_min(self, a: Self::Repr) -> f64 {
        <T::Narrow as ops::F64x4Ops>::reduce_min(self.narrow(), a)
    }
    #[inline(always)]
    fn reduce_max(self, a: Self::Repr) -> f64 {
        <T::Narrow as ops::F64x4Ops>::reduce_max(self.narrow(), a)
    }
    #[inline(always)]
    fn rcp_approx(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::rcp_approx(self.narrow(), a)
    }
    #[inline(always)]
    fn rsqrt_approx(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::rsqrt_approx(self.narrow(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::not(self.narrow(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::bitand(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::bitor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::bitxor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::clamp(self.narrow(), a, lo, hi)
    }
    #[inline(always)]
    fn recip(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::recip(self.narrow(), a)
    }
    #[inline(always)]
    fn rsqrt(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::F64x4Ops>::rsqrt(self.narrow(), a)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> I32x4Backend for T {
    type Repr = <T::Narrow as ops::I32x4Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: i32) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::splat(self.narrow(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::zero(self.narrow())
    }
    #[inline(always)]
    fn load(self, data: &[i32; 4]) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::load(self.narrow(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [i32; 4]) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::from_array(self.narrow(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [i32; 4]) {
        <T::Narrow as ops::I32x4Ops>::store(self.narrow(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [i32; 4] {
        <T::Narrow as ops::I32x4Ops>::to_array(self.narrow(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::mul(self.narrow(), a, b)
    }
    #[inline(always)]
    fn neg(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::neg(self.narrow(), a)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::min(self.narrow(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::max(self.narrow(), a, b)
    }
    #[inline(always)]
    fn abs(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::abs(self.narrow(), a)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::simd_eq(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::simd_ne(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::simd_lt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::simd_le(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::simd_gt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::simd_ge(self.narrow(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::blend(self.narrow(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> i32 {
        <T::Narrow as ops::I32x4Ops>::reduce_add(self.narrow(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::not(self.narrow(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::bitand(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::bitor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::bitxor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::shl_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::shr_arithmetic_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::shr_logical_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::shl_uniform(self.narrow(), a, count)
    }
    #[inline(always)]
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::shr_logical_uniform(self.narrow(), a, count)
    }
    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::shr_arithmetic_uniform(self.narrow(), a, count)
    }
    #[inline(always)]
    fn all_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::I32x4Ops>::all_true(self.narrow(), a)
    }
    #[inline(always)]
    fn any_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::I32x4Ops>::any_true(self.narrow(), a)
    }
    #[inline(always)]
    fn bitmask(self, a: Self::Repr) -> u32 {
        <T::Narrow as ops::I32x4Ops>::bitmask(self.narrow(), a)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x4Ops>::clamp(self.narrow(), a, lo, hi)
    }
    #[inline(always)]
    fn narrow_saturating_i32_to_i16(
        self,
        a: <Self as I32x4Backend>::Repr,
        b: <Self as I32x4Backend>::Repr,
    ) -> <Self as I16x8Backend>::Repr {
        <T::Narrow as ops::I32x4Ops>::narrow_saturating_i32_to_i16(self.narrow(), a, b)
    }
    #[inline(always)]
    fn narrow_saturating_i32_to_u16(
        self,
        a: <Self as I32x4Backend>::Repr,
        b: <Self as I32x4Backend>::Repr,
    ) -> <Self as U16x8Backend>::Repr {
        <T::Narrow as ops::I32x4Ops>::narrow_saturating_i32_to_u16(self.narrow(), a, b)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> I32x8Backend for T {
    type Repr = <T::Narrow as ops::I32x8Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: i32) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::splat(self.narrow(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::zero(self.narrow())
    }
    #[inline(always)]
    fn load(self, data: &[i32; 8]) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::load(self.narrow(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [i32; 8]) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::from_array(self.narrow(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [i32; 8]) {
        <T::Narrow as ops::I32x8Ops>::store(self.narrow(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [i32; 8] {
        <T::Narrow as ops::I32x8Ops>::to_array(self.narrow(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::mul(self.narrow(), a, b)
    }
    #[inline(always)]
    fn neg(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::neg(self.narrow(), a)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::min(self.narrow(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::max(self.narrow(), a, b)
    }
    #[inline(always)]
    fn abs(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::abs(self.narrow(), a)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::simd_eq(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::simd_ne(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::simd_lt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::simd_le(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::simd_gt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::simd_ge(self.narrow(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::blend(self.narrow(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> i32 {
        <T::Narrow as ops::I32x8Ops>::reduce_add(self.narrow(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::not(self.narrow(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::bitand(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::bitor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::bitxor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::shl_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::shr_arithmetic_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::shr_logical_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::shl_uniform(self.narrow(), a, count)
    }
    #[inline(always)]
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::shr_logical_uniform(self.narrow(), a, count)
    }
    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::shr_arithmetic_uniform(self.narrow(), a, count)
    }
    #[inline(always)]
    fn all_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::I32x8Ops>::all_true(self.narrow(), a)
    }
    #[inline(always)]
    fn any_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::I32x8Ops>::any_true(self.narrow(), a)
    }
    #[inline(always)]
    fn bitmask(self, a: Self::Repr) -> u32 {
        <T::Narrow as ops::I32x8Ops>::bitmask(self.narrow(), a)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I32x8Ops>::clamp(self.narrow(), a, lo, hi)
    }
    #[inline(always)]
    fn narrow_saturating_i32_to_i16(
        self,
        a: <Self as I32x8Backend>::Repr,
        b: <Self as I32x8Backend>::Repr,
    ) -> <Self as I16x16Backend>::Repr {
        <T::Narrow as ops::I32x8Ops>::narrow_saturating_i32_to_i16(self.narrow(), a, b)
    }
    #[inline(always)]
    fn narrow_saturating_i32_to_u16(
        self,
        a: <Self as I32x8Backend>::Repr,
        b: <Self as I32x8Backend>::Repr,
    ) -> <Self as U16x16Backend>::Repr {
        <T::Narrow as ops::I32x8Ops>::narrow_saturating_i32_to_u16(self.narrow(), a, b)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> U32x4Backend for T {
    type Repr = <T::Narrow as ops::U32x4Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: u32) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::splat(self.narrow(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::zero(self.narrow())
    }
    #[inline(always)]
    fn load(self, data: &[u32; 4]) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::load(self.narrow(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [u32; 4]) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::from_array(self.narrow(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [u32; 4]) {
        <T::Narrow as ops::U32x4Ops>::store(self.narrow(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [u32; 4] {
        <T::Narrow as ops::U32x4Ops>::to_array(self.narrow(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::mul(self.narrow(), a, b)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::min(self.narrow(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::max(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::simd_eq(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::simd_ne(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::simd_lt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::simd_le(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::simd_gt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::simd_ge(self.narrow(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::blend(self.narrow(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> u32 {
        <T::Narrow as ops::U32x4Ops>::reduce_add(self.narrow(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::not(self.narrow(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::bitand(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::bitor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::bitxor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::shl_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::shr_logical_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::shl_uniform(self.narrow(), a, count)
    }
    #[inline(always)]
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::shr_logical_uniform(self.narrow(), a, count)
    }
    #[inline(always)]
    fn all_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::U32x4Ops>::all_true(self.narrow(), a)
    }
    #[inline(always)]
    fn any_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::U32x4Ops>::any_true(self.narrow(), a)
    }
    #[inline(always)]
    fn bitmask(self, a: Self::Repr) -> u32 {
        <T::Narrow as ops::U32x4Ops>::bitmask(self.narrow(), a)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x4Ops>::clamp(self.narrow(), a, lo, hi)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> U32x8Backend for T {
    type Repr = <T::Narrow as ops::U32x8Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: u32) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::splat(self.narrow(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::zero(self.narrow())
    }
    #[inline(always)]
    fn load(self, data: &[u32; 8]) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::load(self.narrow(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [u32; 8]) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::from_array(self.narrow(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [u32; 8]) {
        <T::Narrow as ops::U32x8Ops>::store(self.narrow(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [u32; 8] {
        <T::Narrow as ops::U32x8Ops>::to_array(self.narrow(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::mul(self.narrow(), a, b)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::min(self.narrow(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::max(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::simd_eq(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::simd_ne(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::simd_lt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::simd_le(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::simd_gt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::simd_ge(self.narrow(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::blend(self.narrow(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> u32 {
        <T::Narrow as ops::U32x8Ops>::reduce_add(self.narrow(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::not(self.narrow(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::bitand(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::bitor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::bitxor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::shl_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::shr_logical_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::shl_uniform(self.narrow(), a, count)
    }
    #[inline(always)]
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::shr_logical_uniform(self.narrow(), a, count)
    }
    #[inline(always)]
    fn all_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::U32x8Ops>::all_true(self.narrow(), a)
    }
    #[inline(always)]
    fn any_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::U32x8Ops>::any_true(self.narrow(), a)
    }
    #[inline(always)]
    fn bitmask(self, a: Self::Repr) -> u32 {
        <T::Narrow as ops::U32x8Ops>::bitmask(self.narrow(), a)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U32x8Ops>::clamp(self.narrow(), a, lo, hi)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> I64x2Backend for T {
    type Repr = <T::Narrow as ops::I64x2Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: i64) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::splat(self.narrow(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::zero(self.narrow())
    }
    #[inline(always)]
    fn load(self, data: &[i64; 2]) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::load(self.narrow(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [i64; 2]) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::from_array(self.narrow(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [i64; 2]) {
        <T::Narrow as ops::I64x2Ops>::store(self.narrow(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [i64; 2] {
        <T::Narrow as ops::I64x2Ops>::to_array(self.narrow(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn neg(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::neg(self.narrow(), a)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::min(self.narrow(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::max(self.narrow(), a, b)
    }
    #[inline(always)]
    fn abs(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::abs(self.narrow(), a)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::simd_eq(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::simd_ne(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::simd_lt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::simd_le(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::simd_gt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::simd_ge(self.narrow(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::blend(self.narrow(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> i64 {
        <T::Narrow as ops::I64x2Ops>::reduce_add(self.narrow(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::not(self.narrow(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::bitand(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::bitor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::bitxor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::shl_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::shr_arithmetic_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::shr_logical_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn all_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::I64x2Ops>::all_true(self.narrow(), a)
    }
    #[inline(always)]
    fn any_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::I64x2Ops>::any_true(self.narrow(), a)
    }
    #[inline(always)]
    fn bitmask(self, a: Self::Repr) -> u32 {
        <T::Narrow as ops::I64x2Ops>::bitmask(self.narrow(), a)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x2Ops>::clamp(self.narrow(), a, lo, hi)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> I64x4Backend for T {
    type Repr = <T::Narrow as ops::I64x4Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: i64) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::splat(self.narrow(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::zero(self.narrow())
    }
    #[inline(always)]
    fn load(self, data: &[i64; 4]) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::load(self.narrow(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [i64; 4]) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::from_array(self.narrow(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [i64; 4]) {
        <T::Narrow as ops::I64x4Ops>::store(self.narrow(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [i64; 4] {
        <T::Narrow as ops::I64x4Ops>::to_array(self.narrow(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn neg(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::neg(self.narrow(), a)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::min(self.narrow(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::max(self.narrow(), a, b)
    }
    #[inline(always)]
    fn abs(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::abs(self.narrow(), a)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::simd_eq(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::simd_ne(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::simd_lt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::simd_le(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::simd_gt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::simd_ge(self.narrow(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::blend(self.narrow(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> i64 {
        <T::Narrow as ops::I64x4Ops>::reduce_add(self.narrow(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::not(self.narrow(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::bitand(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::bitor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::bitxor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::shl_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::shr_arithmetic_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::shr_logical_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn all_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::I64x4Ops>::all_true(self.narrow(), a)
    }
    #[inline(always)]
    fn any_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::I64x4Ops>::any_true(self.narrow(), a)
    }
    #[inline(always)]
    fn bitmask(self, a: Self::Repr) -> u32 {
        <T::Narrow as ops::I64x4Ops>::bitmask(self.narrow(), a)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I64x4Ops>::clamp(self.narrow(), a, lo, hi)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> I8x16Backend for T {
    type Repr = <T::Narrow as ops::I8x16Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: i8) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::splat(self.narrow(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::zero(self.narrow())
    }
    #[inline(always)]
    fn load(self, data: &[i8; 16]) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::load(self.narrow(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [i8; 16]) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::from_array(self.narrow(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [i8; 16]) {
        <T::Narrow as ops::I8x16Ops>::store(self.narrow(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [i8; 16] {
        <T::Narrow as ops::I8x16Ops>::to_array(self.narrow(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn neg(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::neg(self.narrow(), a)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::min(self.narrow(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::max(self.narrow(), a, b)
    }
    #[inline(always)]
    fn abs(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::abs(self.narrow(), a)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::simd_eq(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::simd_ne(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::simd_lt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::simd_le(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::simd_gt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::simd_ge(self.narrow(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::blend(self.narrow(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> i8 {
        <T::Narrow as ops::I8x16Ops>::reduce_add(self.narrow(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::not(self.narrow(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::bitand(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::bitor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::bitxor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::shl_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::shr_logical_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::shr_arithmetic_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::saturating_add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::saturating_sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn all_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::I8x16Ops>::all_true(self.narrow(), a)
    }
    #[inline(always)]
    fn any_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::I8x16Ops>::any_true(self.narrow(), a)
    }
    #[inline(always)]
    fn bitmask(self, a: Self::Repr) -> u32 {
        <T::Narrow as ops::I8x16Ops>::bitmask(self.narrow(), a)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x16Ops>::clamp(self.narrow(), a, lo, hi)
    }
    #[inline(always)]
    fn widen_low_i8_to_i16(self, a: <Self as I8x16Backend>::Repr) -> <Self as I16x8Backend>::Repr {
        <T::Narrow as ops::I8x16Ops>::widen_low_i8_to_i16(self.narrow(), a)
    }
    #[inline(always)]
    fn widen_high_i8_to_i16(self, a: <Self as I8x16Backend>::Repr) -> <Self as I16x8Backend>::Repr {
        <T::Narrow as ops::I8x16Ops>::widen_high_i8_to_i16(self.narrow(), a)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> I8x32Backend for T {
    type Repr = <T::Narrow as ops::I8x32Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: i8) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::splat(self.narrow(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::zero(self.narrow())
    }
    #[inline(always)]
    fn load(self, data: &[i8; 32]) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::load(self.narrow(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [i8; 32]) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::from_array(self.narrow(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [i8; 32]) {
        <T::Narrow as ops::I8x32Ops>::store(self.narrow(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [i8; 32] {
        <T::Narrow as ops::I8x32Ops>::to_array(self.narrow(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn neg(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::neg(self.narrow(), a)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::min(self.narrow(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::max(self.narrow(), a, b)
    }
    #[inline(always)]
    fn abs(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::abs(self.narrow(), a)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::simd_eq(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::simd_ne(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::simd_lt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::simd_le(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::simd_gt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::simd_ge(self.narrow(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::blend(self.narrow(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> i8 {
        <T::Narrow as ops::I8x32Ops>::reduce_add(self.narrow(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::not(self.narrow(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::bitand(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::bitor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::bitxor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::shl_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::shr_logical_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::shr_arithmetic_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::saturating_add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::saturating_sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn all_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::I8x32Ops>::all_true(self.narrow(), a)
    }
    #[inline(always)]
    fn any_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::I8x32Ops>::any_true(self.narrow(), a)
    }
    #[inline(always)]
    fn bitmask(self, a: Self::Repr) -> u32 {
        <T::Narrow as ops::I8x32Ops>::bitmask(self.narrow(), a)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I8x32Ops>::clamp(self.narrow(), a, lo, hi)
    }
    #[inline(always)]
    fn widen_low_i8_to_i16(self, a: <Self as I8x32Backend>::Repr) -> <Self as I16x16Backend>::Repr {
        <T::Narrow as ops::I8x32Ops>::widen_low_i8_to_i16(self.narrow(), a)
    }
    #[inline(always)]
    fn widen_high_i8_to_i16(
        self,
        a: <Self as I8x32Backend>::Repr,
    ) -> <Self as I16x16Backend>::Repr {
        <T::Narrow as ops::I8x32Ops>::widen_high_i8_to_i16(self.narrow(), a)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> U8x16Backend for T {
    type Repr = <T::Narrow as ops::U8x16Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: u8) -> Self::Repr {
        <T::Narrow as ops::U8x16Ops>::splat(self.narrow(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::Narrow as ops::U8x16Ops>::zero(self.narrow())
    }
    #[inline(always)]
    fn load(self, data: &[u8; 16]) -> Self::Repr {
        <T::Narrow as ops::U8x16Ops>::load(self.narrow(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [u8; 16]) -> Self::Repr {
        <T::Narrow as ops::U8x16Ops>::from_array(self.narrow(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [u8; 16]) {
        <T::Narrow as ops::U8x16Ops>::store(self.narrow(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [u8; 16] {
        <T::Narrow as ops::U8x16Ops>::to_array(self.narrow(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x16Ops>::add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x16Ops>::sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x16Ops>::min(self.narrow(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x16Ops>::max(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x16Ops>::simd_eq(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x16Ops>::simd_ne(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x16Ops>::simd_lt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x16Ops>::simd_le(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x16Ops>::simd_gt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x16Ops>::simd_ge(self.narrow(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x16Ops>::blend(self.narrow(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> u8 {
        <T::Narrow as ops::U8x16Ops>::reduce_add(self.narrow(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x16Ops>::not(self.narrow(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x16Ops>::bitand(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x16Ops>::bitor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x16Ops>::bitxor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x16Ops>::shl_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x16Ops>::shr_logical_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x16Ops>::saturating_add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x16Ops>::saturating_sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn all_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::U8x16Ops>::all_true(self.narrow(), a)
    }
    #[inline(always)]
    fn any_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::U8x16Ops>::any_true(self.narrow(), a)
    }
    #[inline(always)]
    fn bitmask(self, a: Self::Repr) -> u32 {
        <T::Narrow as ops::U8x16Ops>::bitmask(self.narrow(), a)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x16Ops>::clamp(self.narrow(), a, lo, hi)
    }
    #[inline(always)]
    fn widen_low_u8_to_u16(self, a: <Self as U8x16Backend>::Repr) -> <Self as U16x8Backend>::Repr {
        <T::Narrow as ops::U8x16Ops>::widen_low_u8_to_u16(self.narrow(), a)
    }
    #[inline(always)]
    fn widen_high_u8_to_u16(self, a: <Self as U8x16Backend>::Repr) -> <Self as U16x8Backend>::Repr {
        <T::Narrow as ops::U8x16Ops>::widen_high_u8_to_u16(self.narrow(), a)
    }
    #[inline(always)]
    fn abs_diff(
        self,
        a: <Self as U8x16Backend>::Repr,
        b: <Self as U8x16Backend>::Repr,
    ) -> <Self as U8x16Backend>::Repr {
        <T::Narrow as ops::U8x16Ops>::abs_diff(self.narrow(), a, b)
    }
    #[inline(always)]
    fn reduce_add_u32(self, a: <Self as U8x16Backend>::Repr) -> u32 {
        <T::Narrow as ops::U8x16Ops>::reduce_add_u32(self.narrow(), a)
    }
    #[inline(always)]
    fn sum_abs_diff(self, a: <Self as U8x16Backend>::Repr, b: <Self as U8x16Backend>::Repr) -> u32 {
        <T::Narrow as ops::U8x16Ops>::sum_abs_diff(self.narrow(), a, b)
    }
    #[inline(always)]
    fn pairwise_widen_add(self, a: <Self as U8x16Backend>::Repr) -> <Self as U16x8Backend>::Repr {
        <T::Narrow as ops::U8x16Ops>::pairwise_widen_add(self.narrow(), a)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> U8x32Backend for T {
    type Repr = <T::Narrow as ops::U8x32Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: u8) -> Self::Repr {
        <T::Narrow as ops::U8x32Ops>::splat(self.narrow(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::Narrow as ops::U8x32Ops>::zero(self.narrow())
    }
    #[inline(always)]
    fn load(self, data: &[u8; 32]) -> Self::Repr {
        <T::Narrow as ops::U8x32Ops>::load(self.narrow(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [u8; 32]) -> Self::Repr {
        <T::Narrow as ops::U8x32Ops>::from_array(self.narrow(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [u8; 32]) {
        <T::Narrow as ops::U8x32Ops>::store(self.narrow(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [u8; 32] {
        <T::Narrow as ops::U8x32Ops>::to_array(self.narrow(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x32Ops>::add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x32Ops>::sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x32Ops>::min(self.narrow(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x32Ops>::max(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x32Ops>::simd_eq(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x32Ops>::simd_ne(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x32Ops>::simd_lt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x32Ops>::simd_le(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x32Ops>::simd_gt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x32Ops>::simd_ge(self.narrow(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x32Ops>::blend(self.narrow(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> u8 {
        <T::Narrow as ops::U8x32Ops>::reduce_add(self.narrow(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x32Ops>::not(self.narrow(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x32Ops>::bitand(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x32Ops>::bitor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x32Ops>::bitxor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x32Ops>::shl_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x32Ops>::shr_logical_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x32Ops>::saturating_add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x32Ops>::saturating_sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn all_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::U8x32Ops>::all_true(self.narrow(), a)
    }
    #[inline(always)]
    fn any_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::U8x32Ops>::any_true(self.narrow(), a)
    }
    #[inline(always)]
    fn bitmask(self, a: Self::Repr) -> u32 {
        <T::Narrow as ops::U8x32Ops>::bitmask(self.narrow(), a)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U8x32Ops>::clamp(self.narrow(), a, lo, hi)
    }
    #[inline(always)]
    fn widen_low_u8_to_u16(self, a: <Self as U8x32Backend>::Repr) -> <Self as U16x16Backend>::Repr {
        <T::Narrow as ops::U8x32Ops>::widen_low_u8_to_u16(self.narrow(), a)
    }
    #[inline(always)]
    fn widen_high_u8_to_u16(
        self,
        a: <Self as U8x32Backend>::Repr,
    ) -> <Self as U16x16Backend>::Repr {
        <T::Narrow as ops::U8x32Ops>::widen_high_u8_to_u16(self.narrow(), a)
    }
    #[inline(always)]
    fn abs_diff(
        self,
        a: <Self as U8x32Backend>::Repr,
        b: <Self as U8x32Backend>::Repr,
    ) -> <Self as U8x32Backend>::Repr {
        <T::Narrow as ops::U8x32Ops>::abs_diff(self.narrow(), a, b)
    }
    #[inline(always)]
    fn reduce_add_u32(self, a: <Self as U8x32Backend>::Repr) -> u32 {
        <T::Narrow as ops::U8x32Ops>::reduce_add_u32(self.narrow(), a)
    }
    #[inline(always)]
    fn sum_abs_diff(self, a: <Self as U8x32Backend>::Repr, b: <Self as U8x32Backend>::Repr) -> u32 {
        <T::Narrow as ops::U8x32Ops>::sum_abs_diff(self.narrow(), a, b)
    }
    #[inline(always)]
    fn pairwise_widen_add(self, a: <Self as U8x32Backend>::Repr) -> <Self as U16x16Backend>::Repr {
        <T::Narrow as ops::U8x32Ops>::pairwise_widen_add(self.narrow(), a)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> I16x8Backend for T {
    type Repr = <T::Narrow as ops::I16x8Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: i16) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::splat(self.narrow(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::zero(self.narrow())
    }
    #[inline(always)]
    fn load(self, data: &[i16; 8]) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::load(self.narrow(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [i16; 8]) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::from_array(self.narrow(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [i16; 8]) {
        <T::Narrow as ops::I16x8Ops>::store(self.narrow(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [i16; 8] {
        <T::Narrow as ops::I16x8Ops>::to_array(self.narrow(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::mul(self.narrow(), a, b)
    }
    #[inline(always)]
    fn mul_hrs_q15(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        self.wasm_fma_i16x8_mul_hrs_q15(a, b)
    }
    #[inline(always)]
    fn neg(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::neg(self.narrow(), a)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::min(self.narrow(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::max(self.narrow(), a, b)
    }
    #[inline(always)]
    fn abs(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::abs(self.narrow(), a)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::simd_eq(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::simd_ne(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::simd_lt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::simd_le(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::simd_gt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::simd_ge(self.narrow(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::blend(self.narrow(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> i16 {
        <T::Narrow as ops::I16x8Ops>::reduce_add(self.narrow(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::not(self.narrow(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::bitand(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::bitor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::bitxor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::shl_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::shr_logical_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::shr_arithmetic_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::shl_uniform(self.narrow(), a, count)
    }
    #[inline(always)]
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::shr_logical_uniform(self.narrow(), a, count)
    }
    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::shr_arithmetic_uniform(self.narrow(), a, count)
    }
    #[inline(always)]
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::saturating_add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::saturating_sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn all_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::I16x8Ops>::all_true(self.narrow(), a)
    }
    #[inline(always)]
    fn any_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::I16x8Ops>::any_true(self.narrow(), a)
    }
    #[inline(always)]
    fn bitmask(self, a: Self::Repr) -> u32 {
        <T::Narrow as ops::I16x8Ops>::bitmask(self.narrow(), a)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x8Ops>::clamp(self.narrow(), a, lo, hi)
    }
    #[inline(always)]
    fn widen_low_i16_to_i32(self, a: <Self as I16x8Backend>::Repr) -> <Self as I32x4Backend>::Repr {
        <T::Narrow as ops::I16x8Ops>::widen_low_i16_to_i32(self.narrow(), a)
    }
    #[inline(always)]
    fn widen_high_i16_to_i32(
        self,
        a: <Self as I16x8Backend>::Repr,
    ) -> <Self as I32x4Backend>::Repr {
        <T::Narrow as ops::I16x8Ops>::widen_high_i16_to_i32(self.narrow(), a)
    }
    #[inline(always)]
    fn narrow_saturating_i16_to_i8(
        self,
        a: <Self as I16x8Backend>::Repr,
        b: <Self as I16x8Backend>::Repr,
    ) -> <Self as I8x16Backend>::Repr {
        <T::Narrow as ops::I16x8Ops>::narrow_saturating_i16_to_i8(self.narrow(), a, b)
    }
    #[inline(always)]
    fn narrow_saturating_i16_to_u8(
        self,
        a: <Self as I16x8Backend>::Repr,
        b: <Self as I16x8Backend>::Repr,
    ) -> <Self as U8x16Backend>::Repr {
        <T::Narrow as ops::I16x8Ops>::narrow_saturating_i16_to_u8(self.narrow(), a, b)
    }
    #[inline(always)]
    fn abs_diff(
        self,
        a: <Self as I16x8Backend>::Repr,
        b: <Self as I16x8Backend>::Repr,
    ) -> <Self as U16x8Backend>::Repr {
        <T::Narrow as ops::I16x8Ops>::abs_diff(self.narrow(), a, b)
    }
    #[inline(always)]
    fn madd_adjacent(
        self,
        a: <Self as I16x8Backend>::Repr,
        b: <Self as I16x8Backend>::Repr,
    ) -> <Self as I32x4Backend>::Repr {
        <T::Narrow as ops::I16x8Ops>::madd_adjacent(self.narrow(), a, b)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> I16x16Backend for T {
    type Repr = <T::Narrow as ops::I16x16Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: i16) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::splat(self.narrow(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::zero(self.narrow())
    }
    #[inline(always)]
    fn load(self, data: &[i16; 16]) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::load(self.narrow(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [i16; 16]) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::from_array(self.narrow(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [i16; 16]) {
        <T::Narrow as ops::I16x16Ops>::store(self.narrow(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [i16; 16] {
        <T::Narrow as ops::I16x16Ops>::to_array(self.narrow(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::mul(self.narrow(), a, b)
    }
    #[inline(always)]
    fn mul_hrs_q15(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        self.wasm_fma_i16x16_mul_hrs_q15(a, b)
    }
    #[inline(always)]
    fn neg(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::neg(self.narrow(), a)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::min(self.narrow(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::max(self.narrow(), a, b)
    }
    #[inline(always)]
    fn abs(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::abs(self.narrow(), a)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::simd_eq(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::simd_ne(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::simd_lt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::simd_le(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::simd_gt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::simd_ge(self.narrow(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::blend(self.narrow(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> i16 {
        <T::Narrow as ops::I16x16Ops>::reduce_add(self.narrow(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::not(self.narrow(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::bitand(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::bitor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::bitxor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::shl_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::shr_logical_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::shr_arithmetic_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::shl_uniform(self.narrow(), a, count)
    }
    #[inline(always)]
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::shr_logical_uniform(self.narrow(), a, count)
    }
    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::shr_arithmetic_uniform(self.narrow(), a, count)
    }
    #[inline(always)]
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::saturating_add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::saturating_sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn all_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::I16x16Ops>::all_true(self.narrow(), a)
    }
    #[inline(always)]
    fn any_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::I16x16Ops>::any_true(self.narrow(), a)
    }
    #[inline(always)]
    fn bitmask(self, a: Self::Repr) -> u32 {
        <T::Narrow as ops::I16x16Ops>::bitmask(self.narrow(), a)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::I16x16Ops>::clamp(self.narrow(), a, lo, hi)
    }
    #[inline(always)]
    fn widen_low_i16_to_i32(
        self,
        a: <Self as I16x16Backend>::Repr,
    ) -> <Self as I32x8Backend>::Repr {
        <T::Narrow as ops::I16x16Ops>::widen_low_i16_to_i32(self.narrow(), a)
    }
    #[inline(always)]
    fn widen_high_i16_to_i32(
        self,
        a: <Self as I16x16Backend>::Repr,
    ) -> <Self as I32x8Backend>::Repr {
        <T::Narrow as ops::I16x16Ops>::widen_high_i16_to_i32(self.narrow(), a)
    }
    #[inline(always)]
    fn narrow_saturating_i16_to_i8(
        self,
        a: <Self as I16x16Backend>::Repr,
        b: <Self as I16x16Backend>::Repr,
    ) -> <Self as I8x32Backend>::Repr {
        <T::Narrow as ops::I16x16Ops>::narrow_saturating_i16_to_i8(self.narrow(), a, b)
    }
    #[inline(always)]
    fn narrow_saturating_i16_to_u8(
        self,
        a: <Self as I16x16Backend>::Repr,
        b: <Self as I16x16Backend>::Repr,
    ) -> <Self as U8x32Backend>::Repr {
        <T::Narrow as ops::I16x16Ops>::narrow_saturating_i16_to_u8(self.narrow(), a, b)
    }
    #[inline(always)]
    fn abs_diff(
        self,
        a: <Self as I16x16Backend>::Repr,
        b: <Self as I16x16Backend>::Repr,
    ) -> <Self as U16x16Backend>::Repr {
        <T::Narrow as ops::I16x16Ops>::abs_diff(self.narrow(), a, b)
    }
    #[inline(always)]
    fn madd_adjacent(
        self,
        a: <Self as I16x16Backend>::Repr,
        b: <Self as I16x16Backend>::Repr,
    ) -> <Self as I32x8Backend>::Repr {
        <T::Narrow as ops::I16x16Ops>::madd_adjacent(self.narrow(), a, b)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> U16x8Backend for T {
    type Repr = <T::Narrow as ops::U16x8Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: u16) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::splat(self.narrow(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::zero(self.narrow())
    }
    #[inline(always)]
    fn load(self, data: &[u16; 8]) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::load(self.narrow(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [u16; 8]) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::from_array(self.narrow(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [u16; 8]) {
        <T::Narrow as ops::U16x8Ops>::store(self.narrow(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [u16; 8] {
        <T::Narrow as ops::U16x8Ops>::to_array(self.narrow(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::mul(self.narrow(), a, b)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::min(self.narrow(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::max(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::simd_eq(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::simd_ne(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::simd_lt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::simd_le(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::simd_gt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::simd_ge(self.narrow(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::blend(self.narrow(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> u16 {
        <T::Narrow as ops::U16x8Ops>::reduce_add(self.narrow(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::not(self.narrow(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::bitand(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::bitor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::bitxor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::shl_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::shr_logical_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::shl_uniform(self.narrow(), a, count)
    }
    #[inline(always)]
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::shr_logical_uniform(self.narrow(), a, count)
    }
    #[inline(always)]
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::saturating_add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::saturating_sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn all_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::U16x8Ops>::all_true(self.narrow(), a)
    }
    #[inline(always)]
    fn any_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::U16x8Ops>::any_true(self.narrow(), a)
    }
    #[inline(always)]
    fn bitmask(self, a: Self::Repr) -> u32 {
        <T::Narrow as ops::U16x8Ops>::bitmask(self.narrow(), a)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x8Ops>::clamp(self.narrow(), a, lo, hi)
    }
    #[inline(always)]
    fn widen_low_u16_to_u32(self, a: <Self as U16x8Backend>::Repr) -> <Self as U32x4Backend>::Repr {
        <T::Narrow as ops::U16x8Ops>::widen_low_u16_to_u32(self.narrow(), a)
    }
    #[inline(always)]
    fn widen_high_u16_to_u32(
        self,
        a: <Self as U16x8Backend>::Repr,
    ) -> <Self as U32x4Backend>::Repr {
        <T::Narrow as ops::U16x8Ops>::widen_high_u16_to_u32(self.narrow(), a)
    }
    #[inline(always)]
    fn pairwise_widen_add(self, a: <Self as U16x8Backend>::Repr) -> <Self as U32x4Backend>::Repr {
        <T::Narrow as ops::U16x8Ops>::pairwise_widen_add(self.narrow(), a)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> U16x16Backend for T {
    type Repr = <T::Narrow as ops::U16x16Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: u16) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::splat(self.narrow(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::zero(self.narrow())
    }
    #[inline(always)]
    fn load(self, data: &[u16; 16]) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::load(self.narrow(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [u16; 16]) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::from_array(self.narrow(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [u16; 16]) {
        <T::Narrow as ops::U16x16Ops>::store(self.narrow(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [u16; 16] {
        <T::Narrow as ops::U16x16Ops>::to_array(self.narrow(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::mul(self.narrow(), a, b)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::min(self.narrow(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::max(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::simd_eq(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::simd_ne(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::simd_lt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::simd_le(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::simd_gt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::simd_ge(self.narrow(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::blend(self.narrow(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> u16 {
        <T::Narrow as ops::U16x16Ops>::reduce_add(self.narrow(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::not(self.narrow(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::bitand(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::bitor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::bitxor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::shl_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::shr_logical_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::shl_uniform(self.narrow(), a, count)
    }
    #[inline(always)]
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::shr_logical_uniform(self.narrow(), a, count)
    }
    #[inline(always)]
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::saturating_add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::saturating_sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn all_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::U16x16Ops>::all_true(self.narrow(), a)
    }
    #[inline(always)]
    fn any_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::U16x16Ops>::any_true(self.narrow(), a)
    }
    #[inline(always)]
    fn bitmask(self, a: Self::Repr) -> u32 {
        <T::Narrow as ops::U16x16Ops>::bitmask(self.narrow(), a)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U16x16Ops>::clamp(self.narrow(), a, lo, hi)
    }
    #[inline(always)]
    fn widen_low_u16_to_u32(
        self,
        a: <Self as U16x16Backend>::Repr,
    ) -> <Self as U32x8Backend>::Repr {
        <T::Narrow as ops::U16x16Ops>::widen_low_u16_to_u32(self.narrow(), a)
    }
    #[inline(always)]
    fn widen_high_u16_to_u32(
        self,
        a: <Self as U16x16Backend>::Repr,
    ) -> <Self as U32x8Backend>::Repr {
        <T::Narrow as ops::U16x16Ops>::widen_high_u16_to_u32(self.narrow(), a)
    }
    #[inline(always)]
    fn pairwise_widen_add(self, a: <Self as U16x16Backend>::Repr) -> <Self as U32x8Backend>::Repr {
        <T::Narrow as ops::U16x16Ops>::pairwise_widen_add(self.narrow(), a)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> U64x2Backend for T {
    type Repr = <T::Narrow as ops::U64x2Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: u64) -> Self::Repr {
        <T::Narrow as ops::U64x2Ops>::splat(self.narrow(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::Narrow as ops::U64x2Ops>::zero(self.narrow())
    }
    #[inline(always)]
    fn load(self, data: &[u64; 2]) -> Self::Repr {
        <T::Narrow as ops::U64x2Ops>::load(self.narrow(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [u64; 2]) -> Self::Repr {
        <T::Narrow as ops::U64x2Ops>::from_array(self.narrow(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [u64; 2]) {
        <T::Narrow as ops::U64x2Ops>::store(self.narrow(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [u64; 2] {
        <T::Narrow as ops::U64x2Ops>::to_array(self.narrow(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x2Ops>::add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x2Ops>::sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x2Ops>::min(self.narrow(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x2Ops>::max(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x2Ops>::simd_eq(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x2Ops>::simd_ne(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x2Ops>::simd_lt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x2Ops>::simd_le(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x2Ops>::simd_gt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x2Ops>::simd_ge(self.narrow(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x2Ops>::blend(self.narrow(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> u64 {
        <T::Narrow as ops::U64x2Ops>::reduce_add(self.narrow(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x2Ops>::not(self.narrow(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x2Ops>::bitand(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x2Ops>::bitor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x2Ops>::bitxor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x2Ops>::shl_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x2Ops>::shr_logical_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn all_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::U64x2Ops>::all_true(self.narrow(), a)
    }
    #[inline(always)]
    fn any_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::U64x2Ops>::any_true(self.narrow(), a)
    }
    #[inline(always)]
    fn bitmask(self, a: Self::Repr) -> u32 {
        <T::Narrow as ops::U64x2Ops>::bitmask(self.narrow(), a)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x2Ops>::clamp(self.narrow(), a, lo, hi)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> U64x4Backend for T {
    type Repr = <T::Narrow as ops::U64x4Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: u64) -> Self::Repr {
        <T::Narrow as ops::U64x4Ops>::splat(self.narrow(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::Narrow as ops::U64x4Ops>::zero(self.narrow())
    }
    #[inline(always)]
    fn load(self, data: &[u64; 4]) -> Self::Repr {
        <T::Narrow as ops::U64x4Ops>::load(self.narrow(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [u64; 4]) -> Self::Repr {
        <T::Narrow as ops::U64x4Ops>::from_array(self.narrow(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [u64; 4]) {
        <T::Narrow as ops::U64x4Ops>::store(self.narrow(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [u64; 4] {
        <T::Narrow as ops::U64x4Ops>::to_array(self.narrow(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x4Ops>::add(self.narrow(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x4Ops>::sub(self.narrow(), a, b)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x4Ops>::min(self.narrow(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x4Ops>::max(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x4Ops>::simd_eq(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x4Ops>::simd_ne(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x4Ops>::simd_lt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x4Ops>::simd_le(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x4Ops>::simd_gt(self.narrow(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x4Ops>::simd_ge(self.narrow(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x4Ops>::blend(self.narrow(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> u64 {
        <T::Narrow as ops::U64x4Ops>::reduce_add(self.narrow(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x4Ops>::not(self.narrow(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x4Ops>::bitand(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x4Ops>::bitor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x4Ops>::bitxor(self.narrow(), a, b)
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x4Ops>::shl_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x4Ops>::shr_logical_const::<N>(self.narrow(), a)
    }
    #[inline(always)]
    fn all_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::U64x4Ops>::all_true(self.narrow(), a)
    }
    #[inline(always)]
    fn any_true(self, a: Self::Repr) -> bool {
        <T::Narrow as ops::U64x4Ops>::any_true(self.narrow(), a)
    }
    #[inline(always)]
    fn bitmask(self, a: Self::Repr) -> u32 {
        <T::Narrow as ops::U64x4Ops>::bitmask(self.narrow(), a)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::Narrow as ops::U64x4Ops>::clamp(self.narrow(), a, lo, hi)
    }
}

#[cfg(feature = "w512")]
impl<T: archmage::SimdToken + Sealed + Copy + 'static> F32x16Backend for T {
    type Repr = <T::W512 as ops::F32x16Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: f32) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::splat(self.w512(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::zero(self.w512())
    }
    #[inline(always)]
    fn load(self, data: &[f32; 16]) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::load(self.w512(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [f32; 16]) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::from_array(self.w512(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [f32; 16]) {
        <T::W512 as ops::F32x16Ops>::store(self.w512(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [f32; 16] {
        <T::W512 as ops::F32x16Ops>::to_array(self.w512(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::add(self.w512(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::sub(self.w512(), a, b)
    }
    #[inline(always)]
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::mul(self.w512(), a, b)
    }
    #[inline(always)]
    fn div(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::div(self.w512(), a, b)
    }
    #[inline(always)]
    fn neg(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::neg(self.w512(), a)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::min(self.w512(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::max(self.w512(), a, b)
    }
    #[inline(always)]
    fn min_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        self.wasm_fma_f32x16_min_fast(a, b)
    }
    #[inline(always)]
    fn max_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        self.wasm_fma_f32x16_max_fast(a, b)
    }
    #[inline(always)]
    fn sqrt(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::sqrt(self.w512(), a)
    }
    #[inline(always)]
    fn abs(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::abs(self.w512(), a)
    }
    #[inline(always)]
    fn floor(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::floor(self.w512(), a)
    }
    #[inline(always)]
    fn ceil(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::ceil(self.w512(), a)
    }
    #[inline(always)]
    fn round(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::round(self.w512(), a)
    }
    #[inline(always)]
    fn mul_add(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        self.wasm_fma_f32x16_mul_add(a, b, c)
    }
    #[inline(always)]
    fn mul_sub(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        self.wasm_fma_f32x16_mul_sub(a, b, c)
    }
    #[inline(always)]
    fn mul_add_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        self.wasm_fma_f32x16_mul_add_portable(a, b, c)
    }
    #[inline(always)]
    fn mul_sub_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        self.wasm_fma_f32x16_mul_sub_portable(a, b, c)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::simd_eq(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::simd_ne(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::simd_lt(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::simd_le(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::simd_gt(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::simd_ge(self.w512(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::blend(self.w512(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> f32 {
        <T::W512 as ops::F32x16Ops>::reduce_add(self.w512(), a)
    }
    #[inline(always)]
    fn reduce_min(self, a: Self::Repr) -> f32 {
        <T::W512 as ops::F32x16Ops>::reduce_min(self.w512(), a)
    }
    #[inline(always)]
    fn reduce_max(self, a: Self::Repr) -> f32 {
        <T::W512 as ops::F32x16Ops>::reduce_max(self.w512(), a)
    }
    #[inline(always)]
    fn rcp_approx(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::rcp_approx(self.w512(), a)
    }
    #[inline(always)]
    fn rsqrt_approx(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::rsqrt_approx(self.w512(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::not(self.w512(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::bitand(self.w512(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::bitor(self.w512(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::bitxor(self.w512(), a, b)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::clamp(self.w512(), a, lo, hi)
    }
    #[inline(always)]
    fn recip(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::recip(self.w512(), a)
    }
    #[inline(always)]
    fn rsqrt(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F32x16Ops>::rsqrt(self.w512(), a)
    }
    #[inline(always)]
    fn to_u8_bytes(self, a: Self::Repr) -> [u8; 16] {
        <T::W512 as ops::F32x16Ops>::to_u8_bytes(self.w512(), a)
    }
    #[inline(always)]
    fn store_rgba_bytes(
        self,
        r: Self::Repr,
        g: Self::Repr,
        b: Self::Repr,
        a: Self::Repr,
    ) -> [u8; 64] {
        <T::W512 as ops::F32x16Ops>::store_rgba_bytes(self.w512(), r, g, b, a)
    }
    #[inline(always)]
    fn transpose_dual_8x8_repr(self, rows: [Self::Repr; 8]) -> [Self::Repr; 8] {
        <T::W512 as ops::F32x16Ops>::transpose_dual_8x8_repr(self.w512(), rows)
    }
    #[inline(always)]
    fn transpose_16x16_repr(self, rows: [Self::Repr; 16]) -> [Self::Repr; 16] {
        <T::W512 as ops::F32x16Ops>::transpose_16x16_repr(self.w512(), rows)
    }
}

#[cfg(feature = "w512")]
impl<T: archmage::SimdToken + Sealed + Copy + 'static> F64x8Backend for T {
    type Repr = <T::W512 as ops::F64x8Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: f64) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::splat(self.w512(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::zero(self.w512())
    }
    #[inline(always)]
    fn load(self, data: &[f64; 8]) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::load(self.w512(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [f64; 8]) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::from_array(self.w512(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [f64; 8]) {
        <T::W512 as ops::F64x8Ops>::store(self.w512(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [f64; 8] {
        <T::W512 as ops::F64x8Ops>::to_array(self.w512(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::add(self.w512(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::sub(self.w512(), a, b)
    }
    #[inline(always)]
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::mul(self.w512(), a, b)
    }
    #[inline(always)]
    fn div(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::div(self.w512(), a, b)
    }
    #[inline(always)]
    fn neg(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::neg(self.w512(), a)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::min(self.w512(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::max(self.w512(), a, b)
    }
    #[inline(always)]
    fn min_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        self.wasm_fma_f64x8_min_fast(a, b)
    }
    #[inline(always)]
    fn max_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        self.wasm_fma_f64x8_max_fast(a, b)
    }
    #[inline(always)]
    fn sqrt(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::sqrt(self.w512(), a)
    }
    #[inline(always)]
    fn abs(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::abs(self.w512(), a)
    }
    #[inline(always)]
    fn floor(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::floor(self.w512(), a)
    }
    #[inline(always)]
    fn ceil(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::ceil(self.w512(), a)
    }
    #[inline(always)]
    fn round(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::round(self.w512(), a)
    }
    #[inline(always)]
    fn mul_add(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        self.wasm_fma_f64x8_mul_add(a, b, c)
    }
    #[inline(always)]
    fn mul_sub(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        self.wasm_fma_f64x8_mul_sub(a, b, c)
    }
    #[inline(always)]
    fn mul_add_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        self.wasm_fma_f64x8_mul_add_portable(a, b, c)
    }
    #[inline(always)]
    fn mul_sub_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        self.wasm_fma_f64x8_mul_sub_portable(a, b, c)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::simd_eq(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::simd_ne(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::simd_lt(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::simd_le(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::simd_gt(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::simd_ge(self.w512(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::blend(self.w512(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> f64 {
        <T::W512 as ops::F64x8Ops>::reduce_add(self.w512(), a)
    }
    #[inline(always)]
    fn reduce_min(self, a: Self::Repr) -> f64 {
        <T::W512 as ops::F64x8Ops>::reduce_min(self.w512(), a)
    }
    #[inline(always)]
    fn reduce_max(self, a: Self::Repr) -> f64 {
        <T::W512 as ops::F64x8Ops>::reduce_max(self.w512(), a)
    }
    #[inline(always)]
    fn rcp_approx(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::rcp_approx(self.w512(), a)
    }
    #[inline(always)]
    fn rsqrt_approx(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::rsqrt_approx(self.w512(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::not(self.w512(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::bitand(self.w512(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::bitor(self.w512(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::bitxor(self.w512(), a, b)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::clamp(self.w512(), a, lo, hi)
    }
    #[inline(always)]
    fn recip(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::recip(self.w512(), a)
    }
    #[inline(always)]
    fn rsqrt(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::F64x8Ops>::rsqrt(self.w512(), a)
    }
}

#[cfg(feature = "w512")]
impl<T: archmage::SimdToken + Sealed + Copy + 'static> I8x64Backend for T {
    type Repr = <T::W512 as ops::I8x64Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: i8) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::splat(self.w512(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::zero(self.w512())
    }
    #[inline(always)]
    fn load(self, data: &[i8; 64]) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::load(self.w512(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [i8; 64]) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::from_array(self.w512(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [i8; 64]) {
        <T::W512 as ops::I8x64Ops>::store(self.w512(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [i8; 64] {
        <T::W512 as ops::I8x64Ops>::to_array(self.w512(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::add(self.w512(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::sub(self.w512(), a, b)
    }
    #[inline(always)]
    fn neg(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::neg(self.w512(), a)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::min(self.w512(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::max(self.w512(), a, b)
    }
    #[inline(always)]
    fn abs(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::abs(self.w512(), a)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::simd_eq(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::simd_ne(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::simd_lt(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::simd_le(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::simd_gt(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::simd_ge(self.w512(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::blend(self.w512(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> i8 {
        <T::W512 as ops::I8x64Ops>::reduce_add(self.w512(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::not(self.w512(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::bitand(self.w512(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::bitor(self.w512(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::bitxor(self.w512(), a, b)
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::shl_const::<N>(self.w512(), a)
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::shr_arithmetic_const::<N>(self.w512(), a)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::shr_logical_const::<N>(self.w512(), a)
    }
    #[inline(always)]
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::saturating_add(self.w512(), a, b)
    }
    #[inline(always)]
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::saturating_sub(self.w512(), a, b)
    }
    #[inline(always)]
    fn all_true(self, a: Self::Repr) -> bool {
        <T::W512 as ops::I8x64Ops>::all_true(self.w512(), a)
    }
    #[inline(always)]
    fn any_true(self, a: Self::Repr) -> bool {
        <T::W512 as ops::I8x64Ops>::any_true(self.w512(), a)
    }
    #[inline(always)]
    fn bitmask(self, a: Self::Repr) -> u64 {
        <T::W512 as ops::I8x64Ops>::bitmask(self.w512(), a)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I8x64Ops>::clamp(self.w512(), a, lo, hi)
    }
    #[inline(always)]
    fn widen_low_i8_to_i16(self, a: <Self as I8x64Backend>::Repr) -> <Self as I16x32Backend>::Repr {
        <T::W512 as ops::I8x64Ops>::widen_low_i8_to_i16(self.w512(), a)
    }
    #[inline(always)]
    fn widen_high_i8_to_i16(
        self,
        a: <Self as I8x64Backend>::Repr,
    ) -> <Self as I16x32Backend>::Repr {
        <T::W512 as ops::I8x64Ops>::widen_high_i8_to_i16(self.w512(), a)
    }
}

#[cfg(feature = "w512")]
impl<T: archmage::SimdToken + Sealed + Copy + 'static> U8x64Backend for T {
    type Repr = <T::W512 as ops::U8x64Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: u8) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::splat(self.w512(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::zero(self.w512())
    }
    #[inline(always)]
    fn load(self, data: &[u8; 64]) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::load(self.w512(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [u8; 64]) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::from_array(self.w512(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [u8; 64]) {
        <T::W512 as ops::U8x64Ops>::store(self.w512(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [u8; 64] {
        <T::W512 as ops::U8x64Ops>::to_array(self.w512(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::add(self.w512(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::sub(self.w512(), a, b)
    }
    #[inline(always)]
    fn neg(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::neg(self.w512(), a)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::min(self.w512(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::max(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::simd_eq(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::simd_ne(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::simd_lt(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::simd_le(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::simd_gt(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::simd_ge(self.w512(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::blend(self.w512(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> u8 {
        <T::W512 as ops::U8x64Ops>::reduce_add(self.w512(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::not(self.w512(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::bitand(self.w512(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::bitor(self.w512(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::bitxor(self.w512(), a, b)
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::shl_const::<N>(self.w512(), a)
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::shr_arithmetic_const::<N>(self.w512(), a)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::shr_logical_const::<N>(self.w512(), a)
    }
    #[inline(always)]
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::saturating_add(self.w512(), a, b)
    }
    #[inline(always)]
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::saturating_sub(self.w512(), a, b)
    }
    #[inline(always)]
    fn all_true(self, a: Self::Repr) -> bool {
        <T::W512 as ops::U8x64Ops>::all_true(self.w512(), a)
    }
    #[inline(always)]
    fn any_true(self, a: Self::Repr) -> bool {
        <T::W512 as ops::U8x64Ops>::any_true(self.w512(), a)
    }
    #[inline(always)]
    fn bitmask(self, a: Self::Repr) -> u64 {
        <T::W512 as ops::U8x64Ops>::bitmask(self.w512(), a)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U8x64Ops>::clamp(self.w512(), a, lo, hi)
    }
    #[inline(always)]
    fn widen_low_u8_to_u16(self, a: <Self as U8x64Backend>::Repr) -> <Self as U16x32Backend>::Repr {
        <T::W512 as ops::U8x64Ops>::widen_low_u8_to_u16(self.w512(), a)
    }
    #[inline(always)]
    fn widen_high_u8_to_u16(
        self,
        a: <Self as U8x64Backend>::Repr,
    ) -> <Self as U16x32Backend>::Repr {
        <T::W512 as ops::U8x64Ops>::widen_high_u8_to_u16(self.w512(), a)
    }
    #[inline(always)]
    fn abs_diff(
        self,
        a: <Self as U8x64Backend>::Repr,
        b: <Self as U8x64Backend>::Repr,
    ) -> <Self as U8x64Backend>::Repr {
        <T::W512 as ops::U8x64Ops>::abs_diff(self.w512(), a, b)
    }
    #[inline(always)]
    fn reduce_add_u32(self, a: <Self as U8x64Backend>::Repr) -> u32 {
        <T::W512 as ops::U8x64Ops>::reduce_add_u32(self.w512(), a)
    }
    #[inline(always)]
    fn sum_abs_diff(self, a: <Self as U8x64Backend>::Repr, b: <Self as U8x64Backend>::Repr) -> u32 {
        <T::W512 as ops::U8x64Ops>::sum_abs_diff(self.w512(), a, b)
    }
    #[inline(always)]
    fn pairwise_widen_add(self, a: <Self as U8x64Backend>::Repr) -> <Self as U16x32Backend>::Repr {
        <T::W512 as ops::U8x64Ops>::pairwise_widen_add(self.w512(), a)
    }
}

#[cfg(feature = "w512")]
impl<T: archmage::SimdToken + Sealed + Copy + 'static> I16x32Backend for T {
    type Repr = <T::W512 as ops::I16x32Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: i16) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::splat(self.w512(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::zero(self.w512())
    }
    #[inline(always)]
    fn load(self, data: &[i16; 32]) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::load(self.w512(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [i16; 32]) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::from_array(self.w512(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [i16; 32]) {
        <T::W512 as ops::I16x32Ops>::store(self.w512(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [i16; 32] {
        <T::W512 as ops::I16x32Ops>::to_array(self.w512(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::add(self.w512(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::sub(self.w512(), a, b)
    }
    #[inline(always)]
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::mul(self.w512(), a, b)
    }
    #[inline(always)]
    fn mul_hrs_q15(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        self.wasm_fma_i16x32_mul_hrs_q15(a, b)
    }
    #[inline(always)]
    fn neg(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::neg(self.w512(), a)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::min(self.w512(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::max(self.w512(), a, b)
    }
    #[inline(always)]
    fn abs(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::abs(self.w512(), a)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::simd_eq(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::simd_ne(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::simd_lt(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::simd_le(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::simd_gt(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::simd_ge(self.w512(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::blend(self.w512(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> i16 {
        <T::W512 as ops::I16x32Ops>::reduce_add(self.w512(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::not(self.w512(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::bitand(self.w512(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::bitor(self.w512(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::bitxor(self.w512(), a, b)
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::shl_const::<N>(self.w512(), a)
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::shr_arithmetic_const::<N>(self.w512(), a)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::shr_logical_const::<N>(self.w512(), a)
    }
    #[inline(always)]
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::shl_uniform(self.w512(), a, count)
    }
    #[inline(always)]
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::shr_logical_uniform(self.w512(), a, count)
    }
    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::shr_arithmetic_uniform(self.w512(), a, count)
    }
    #[inline(always)]
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::saturating_add(self.w512(), a, b)
    }
    #[inline(always)]
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::saturating_sub(self.w512(), a, b)
    }
    #[inline(always)]
    fn all_true(self, a: Self::Repr) -> bool {
        <T::W512 as ops::I16x32Ops>::all_true(self.w512(), a)
    }
    #[inline(always)]
    fn any_true(self, a: Self::Repr) -> bool {
        <T::W512 as ops::I16x32Ops>::any_true(self.w512(), a)
    }
    #[inline(always)]
    fn bitmask(self, a: Self::Repr) -> u64 {
        <T::W512 as ops::I16x32Ops>::bitmask(self.w512(), a)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I16x32Ops>::clamp(self.w512(), a, lo, hi)
    }
    #[inline(always)]
    fn widen_low_i16_to_i32(
        self,
        a: <Self as I16x32Backend>::Repr,
    ) -> <Self as I32x16Backend>::Repr {
        <T::W512 as ops::I16x32Ops>::widen_low_i16_to_i32(self.w512(), a)
    }
    #[inline(always)]
    fn widen_high_i16_to_i32(
        self,
        a: <Self as I16x32Backend>::Repr,
    ) -> <Self as I32x16Backend>::Repr {
        <T::W512 as ops::I16x32Ops>::widen_high_i16_to_i32(self.w512(), a)
    }
    #[inline(always)]
    fn narrow_saturating_i16_to_i8(
        self,
        a: <Self as I16x32Backend>::Repr,
        b: <Self as I16x32Backend>::Repr,
    ) -> <Self as I8x64Backend>::Repr {
        <T::W512 as ops::I16x32Ops>::narrow_saturating_i16_to_i8(self.w512(), a, b)
    }
    #[inline(always)]
    fn narrow_saturating_i16_to_u8(
        self,
        a: <Self as I16x32Backend>::Repr,
        b: <Self as I16x32Backend>::Repr,
    ) -> <Self as U8x64Backend>::Repr {
        <T::W512 as ops::I16x32Ops>::narrow_saturating_i16_to_u8(self.w512(), a, b)
    }
    #[inline(always)]
    fn abs_diff(
        self,
        a: <Self as I16x32Backend>::Repr,
        b: <Self as I16x32Backend>::Repr,
    ) -> <Self as U16x32Backend>::Repr {
        <T::W512 as ops::I16x32Ops>::abs_diff(self.w512(), a, b)
    }
    #[inline(always)]
    fn madd_adjacent(
        self,
        a: <Self as I16x32Backend>::Repr,
        b: <Self as I16x32Backend>::Repr,
    ) -> <Self as I32x16Backend>::Repr {
        <T::W512 as ops::I16x32Ops>::madd_adjacent(self.w512(), a, b)
    }
}

#[cfg(feature = "w512")]
impl<T: archmage::SimdToken + Sealed + Copy + 'static> U16x32Backend for T {
    type Repr = <T::W512 as ops::U16x32Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: u16) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::splat(self.w512(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::zero(self.w512())
    }
    #[inline(always)]
    fn load(self, data: &[u16; 32]) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::load(self.w512(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [u16; 32]) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::from_array(self.w512(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [u16; 32]) {
        <T::W512 as ops::U16x32Ops>::store(self.w512(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [u16; 32] {
        <T::W512 as ops::U16x32Ops>::to_array(self.w512(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::add(self.w512(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::sub(self.w512(), a, b)
    }
    #[inline(always)]
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::mul(self.w512(), a, b)
    }
    #[inline(always)]
    fn neg(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::neg(self.w512(), a)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::min(self.w512(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::max(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::simd_eq(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::simd_ne(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::simd_lt(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::simd_le(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::simd_gt(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::simd_ge(self.w512(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::blend(self.w512(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> u16 {
        <T::W512 as ops::U16x32Ops>::reduce_add(self.w512(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::not(self.w512(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::bitand(self.w512(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::bitor(self.w512(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::bitxor(self.w512(), a, b)
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::shl_const::<N>(self.w512(), a)
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::shr_arithmetic_const::<N>(self.w512(), a)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::shr_logical_const::<N>(self.w512(), a)
    }
    #[inline(always)]
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::shl_uniform(self.w512(), a, count)
    }
    #[inline(always)]
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::shr_logical_uniform(self.w512(), a, count)
    }
    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::shr_arithmetic_uniform(self.w512(), a, count)
    }
    #[inline(always)]
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::saturating_add(self.w512(), a, b)
    }
    #[inline(always)]
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::saturating_sub(self.w512(), a, b)
    }
    #[inline(always)]
    fn all_true(self, a: Self::Repr) -> bool {
        <T::W512 as ops::U16x32Ops>::all_true(self.w512(), a)
    }
    #[inline(always)]
    fn any_true(self, a: Self::Repr) -> bool {
        <T::W512 as ops::U16x32Ops>::any_true(self.w512(), a)
    }
    #[inline(always)]
    fn bitmask(self, a: Self::Repr) -> u64 {
        <T::W512 as ops::U16x32Ops>::bitmask(self.w512(), a)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U16x32Ops>::clamp(self.w512(), a, lo, hi)
    }
    #[inline(always)]
    fn widen_low_u16_to_u32(
        self,
        a: <Self as U16x32Backend>::Repr,
    ) -> <Self as U32x16Backend>::Repr {
        <T::W512 as ops::U16x32Ops>::widen_low_u16_to_u32(self.w512(), a)
    }
    #[inline(always)]
    fn widen_high_u16_to_u32(
        self,
        a: <Self as U16x32Backend>::Repr,
    ) -> <Self as U32x16Backend>::Repr {
        <T::W512 as ops::U16x32Ops>::widen_high_u16_to_u32(self.w512(), a)
    }
    #[inline(always)]
    fn pairwise_widen_add(self, a: <Self as U16x32Backend>::Repr) -> <Self as U32x16Backend>::Repr {
        <T::W512 as ops::U16x32Ops>::pairwise_widen_add(self.w512(), a)
    }
}

#[cfg(feature = "w512")]
impl<T: archmage::SimdToken + Sealed + Copy + 'static> I32x16Backend for T {
    type Repr = <T::W512 as ops::I32x16Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: i32) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::splat(self.w512(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::zero(self.w512())
    }
    #[inline(always)]
    fn load(self, data: &[i32; 16]) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::load(self.w512(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [i32; 16]) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::from_array(self.w512(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [i32; 16]) {
        <T::W512 as ops::I32x16Ops>::store(self.w512(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [i32; 16] {
        <T::W512 as ops::I32x16Ops>::to_array(self.w512(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::add(self.w512(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::sub(self.w512(), a, b)
    }
    #[inline(always)]
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::mul(self.w512(), a, b)
    }
    #[inline(always)]
    fn neg(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::neg(self.w512(), a)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::min(self.w512(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::max(self.w512(), a, b)
    }
    #[inline(always)]
    fn abs(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::abs(self.w512(), a)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::simd_eq(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::simd_ne(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::simd_lt(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::simd_le(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::simd_gt(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::simd_ge(self.w512(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::blend(self.w512(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> i32 {
        <T::W512 as ops::I32x16Ops>::reduce_add(self.w512(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::not(self.w512(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::bitand(self.w512(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::bitor(self.w512(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::bitxor(self.w512(), a, b)
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::shl_const::<N>(self.w512(), a)
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::shr_arithmetic_const::<N>(self.w512(), a)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::shr_logical_const::<N>(self.w512(), a)
    }
    #[inline(always)]
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::shl_uniform(self.w512(), a, count)
    }
    #[inline(always)]
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::shr_logical_uniform(self.w512(), a, count)
    }
    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::shr_arithmetic_uniform(self.w512(), a, count)
    }
    #[inline(always)]
    fn all_true(self, a: Self::Repr) -> bool {
        <T::W512 as ops::I32x16Ops>::all_true(self.w512(), a)
    }
    #[inline(always)]
    fn any_true(self, a: Self::Repr) -> bool {
        <T::W512 as ops::I32x16Ops>::any_true(self.w512(), a)
    }
    #[inline(always)]
    fn bitmask(self, a: Self::Repr) -> u64 {
        <T::W512 as ops::I32x16Ops>::bitmask(self.w512(), a)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I32x16Ops>::clamp(self.w512(), a, lo, hi)
    }
    #[inline(always)]
    fn narrow_saturating_i32_to_i16(
        self,
        a: <Self as I32x16Backend>::Repr,
        b: <Self as I32x16Backend>::Repr,
    ) -> <Self as I16x32Backend>::Repr {
        <T::W512 as ops::I32x16Ops>::narrow_saturating_i32_to_i16(self.w512(), a, b)
    }
    #[inline(always)]
    fn narrow_saturating_i32_to_u16(
        self,
        a: <Self as I32x16Backend>::Repr,
        b: <Self as I32x16Backend>::Repr,
    ) -> <Self as U16x32Backend>::Repr {
        <T::W512 as ops::I32x16Ops>::narrow_saturating_i32_to_u16(self.w512(), a, b)
    }
}

#[cfg(feature = "w512")]
impl<T: archmage::SimdToken + Sealed + Copy + 'static> U32x16Backend for T {
    type Repr = <T::W512 as ops::U32x16Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: u32) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::splat(self.w512(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::zero(self.w512())
    }
    #[inline(always)]
    fn load(self, data: &[u32; 16]) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::load(self.w512(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [u32; 16]) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::from_array(self.w512(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [u32; 16]) {
        <T::W512 as ops::U32x16Ops>::store(self.w512(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [u32; 16] {
        <T::W512 as ops::U32x16Ops>::to_array(self.w512(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::add(self.w512(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::sub(self.w512(), a, b)
    }
    #[inline(always)]
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::mul(self.w512(), a, b)
    }
    #[inline(always)]
    fn neg(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::neg(self.w512(), a)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::min(self.w512(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::max(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::simd_eq(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::simd_ne(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::simd_lt(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::simd_le(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::simd_gt(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::simd_ge(self.w512(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::blend(self.w512(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> u32 {
        <T::W512 as ops::U32x16Ops>::reduce_add(self.w512(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::not(self.w512(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::bitand(self.w512(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::bitor(self.w512(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::bitxor(self.w512(), a, b)
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::shl_const::<N>(self.w512(), a)
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::shr_arithmetic_const::<N>(self.w512(), a)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::shr_logical_const::<N>(self.w512(), a)
    }
    #[inline(always)]
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::shl_uniform(self.w512(), a, count)
    }
    #[inline(always)]
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::shr_logical_uniform(self.w512(), a, count)
    }
    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::shr_arithmetic_uniform(self.w512(), a, count)
    }
    #[inline(always)]
    fn all_true(self, a: Self::Repr) -> bool {
        <T::W512 as ops::U32x16Ops>::all_true(self.w512(), a)
    }
    #[inline(always)]
    fn any_true(self, a: Self::Repr) -> bool {
        <T::W512 as ops::U32x16Ops>::any_true(self.w512(), a)
    }
    #[inline(always)]
    fn bitmask(self, a: Self::Repr) -> u64 {
        <T::W512 as ops::U32x16Ops>::bitmask(self.w512(), a)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U32x16Ops>::clamp(self.w512(), a, lo, hi)
    }
}

#[cfg(feature = "w512")]
impl<T: archmage::SimdToken + Sealed + Copy + 'static> I64x8Backend for T {
    type Repr = <T::W512 as ops::I64x8Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: i64) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::splat(self.w512(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::zero(self.w512())
    }
    #[inline(always)]
    fn load(self, data: &[i64; 8]) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::load(self.w512(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [i64; 8]) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::from_array(self.w512(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [i64; 8]) {
        <T::W512 as ops::I64x8Ops>::store(self.w512(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [i64; 8] {
        <T::W512 as ops::I64x8Ops>::to_array(self.w512(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::add(self.w512(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::sub(self.w512(), a, b)
    }
    #[inline(always)]
    fn neg(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::neg(self.w512(), a)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::min(self.w512(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::max(self.w512(), a, b)
    }
    #[inline(always)]
    fn abs(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::abs(self.w512(), a)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::simd_eq(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::simd_ne(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::simd_lt(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::simd_le(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::simd_gt(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::simd_ge(self.w512(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::blend(self.w512(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> i64 {
        <T::W512 as ops::I64x8Ops>::reduce_add(self.w512(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::not(self.w512(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::bitand(self.w512(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::bitor(self.w512(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::bitxor(self.w512(), a, b)
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::shl_const::<N>(self.w512(), a)
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::shr_arithmetic_const::<N>(self.w512(), a)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::shr_logical_const::<N>(self.w512(), a)
    }
    #[inline(always)]
    fn all_true(self, a: Self::Repr) -> bool {
        <T::W512 as ops::I64x8Ops>::all_true(self.w512(), a)
    }
    #[inline(always)]
    fn any_true(self, a: Self::Repr) -> bool {
        <T::W512 as ops::I64x8Ops>::any_true(self.w512(), a)
    }
    #[inline(always)]
    fn bitmask(self, a: Self::Repr) -> u64 {
        <T::W512 as ops::I64x8Ops>::bitmask(self.w512(), a)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::W512 as ops::I64x8Ops>::clamp(self.w512(), a, lo, hi)
    }
}

#[cfg(feature = "w512")]
impl<T: archmage::SimdToken + Sealed + Copy + 'static> U64x8Backend for T {
    type Repr = <T::W512 as ops::U64x8Ops>::Repr;
    #[inline(always)]
    fn splat(self, v: u64) -> Self::Repr {
        <T::W512 as ops::U64x8Ops>::splat(self.w512(), v)
    }
    #[inline(always)]
    fn zero(self) -> Self::Repr {
        <T::W512 as ops::U64x8Ops>::zero(self.w512())
    }
    #[inline(always)]
    fn load(self, data: &[u64; 8]) -> Self::Repr {
        <T::W512 as ops::U64x8Ops>::load(self.w512(), data)
    }
    #[inline(always)]
    fn from_array(self, arr: [u64; 8]) -> Self::Repr {
        <T::W512 as ops::U64x8Ops>::from_array(self.w512(), arr)
    }
    #[inline(always)]
    fn store(self, repr: Self::Repr, out: &mut [u64; 8]) {
        <T::W512 as ops::U64x8Ops>::store(self.w512(), repr, out)
    }
    #[inline(always)]
    fn to_array(self, repr: Self::Repr) -> [u64; 8] {
        <T::W512 as ops::U64x8Ops>::to_array(self.w512(), repr)
    }
    #[inline(always)]
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U64x8Ops>::add(self.w512(), a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U64x8Ops>::sub(self.w512(), a, b)
    }
    #[inline(always)]
    fn neg(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U64x8Ops>::neg(self.w512(), a)
    }
    #[inline(always)]
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U64x8Ops>::min(self.w512(), a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U64x8Ops>::max(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U64x8Ops>::simd_eq(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U64x8Ops>::simd_ne(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U64x8Ops>::simd_lt(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U64x8Ops>::simd_le(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U64x8Ops>::simd_gt(self.w512(), a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U64x8Ops>::simd_ge(self.w512(), a, b)
    }
    #[inline(always)]
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U64x8Ops>::blend(self.w512(), mask, if_true, if_false)
    }
    #[inline(always)]
    fn reduce_add(self, a: Self::Repr) -> u64 {
        <T::W512 as ops::U64x8Ops>::reduce_add(self.w512(), a)
    }
    #[inline(always)]
    fn not(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U64x8Ops>::not(self.w512(), a)
    }
    #[inline(always)]
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U64x8Ops>::bitand(self.w512(), a, b)
    }
    #[inline(always)]
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U64x8Ops>::bitor(self.w512(), a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U64x8Ops>::bitxor(self.w512(), a, b)
    }
    #[inline(always)]
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U64x8Ops>::shl_const::<N>(self.w512(), a)
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U64x8Ops>::shr_arithmetic_const::<N>(self.w512(), a)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U64x8Ops>::shr_logical_const::<N>(self.w512(), a)
    }
    #[inline(always)]
    fn all_true(self, a: Self::Repr) -> bool {
        <T::W512 as ops::U64x8Ops>::all_true(self.w512(), a)
    }
    #[inline(always)]
    fn any_true(self, a: Self::Repr) -> bool {
        <T::W512 as ops::U64x8Ops>::any_true(self.w512(), a)
    }
    #[inline(always)]
    fn bitmask(self, a: Self::Repr) -> u64 {
        <T::W512 as ops::U64x8Ops>::bitmask(self.w512(), a)
    }
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <T::W512 as ops::U64x8Ops>::clamp(self.w512(), a, lo, hi)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> F32x4Convert for T {
    #[inline(always)]
    fn bitcast_f32_to_i32(self, a: <Self as F32x4Backend>::Repr) -> <Self as I32x4Backend>::Repr {
        <T::Narrow as ops::F32x4ConvertOps>::bitcast_f32_to_i32(self.narrow(), a)
    }
    #[inline(always)]
    fn bitcast_i32_to_f32(self, a: <Self as I32x4Backend>::Repr) -> <Self as F32x4Backend>::Repr {
        <T::Narrow as ops::F32x4ConvertOps>::bitcast_i32_to_f32(self.narrow(), a)
    }
    #[inline(always)]
    fn convert_f32_to_i32(self, a: <Self as F32x4Backend>::Repr) -> <Self as I32x4Backend>::Repr {
        <T::Narrow as ops::F32x4ConvertOps>::convert_f32_to_i32(self.narrow(), a)
    }
    #[inline(always)]
    fn convert_f32_to_i32_saturating(
        self,
        a: <Self as F32x4Backend>::Repr,
    ) -> <Self as I32x4Backend>::Repr {
        <T::Narrow as ops::F32x4ConvertOps>::convert_f32_to_i32_saturating(self.narrow(), a)
    }
    #[inline(always)]
    fn convert_f32_to_i32_fast(
        self,
        a: <Self as F32x4Backend>::Repr,
    ) -> <Self as I32x4Backend>::Repr {
        self.wasm_fma_f32x4_cvt_convert_f32_to_i32_fast(a)
    }
    #[inline(always)]
    fn convert_f32_to_i32_round(
        self,
        a: <Self as F32x4Backend>::Repr,
    ) -> <Self as I32x4Backend>::Repr {
        <T::Narrow as ops::F32x4ConvertOps>::convert_f32_to_i32_round(self.narrow(), a)
    }
    #[inline(always)]
    fn convert_i32_to_f32(self, a: <Self as I32x4Backend>::Repr) -> <Self as F32x4Backend>::Repr {
        <T::Narrow as ops::F32x4ConvertOps>::convert_i32_to_f32(self.narrow(), a)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> F32x8Convert for T {
    #[inline(always)]
    fn bitcast_f32_to_i32(self, a: <Self as F32x8Backend>::Repr) -> <Self as I32x8Backend>::Repr {
        <T::Narrow as ops::F32x8ConvertOps>::bitcast_f32_to_i32(self.narrow(), a)
    }
    #[inline(always)]
    fn bitcast_i32_to_f32(self, a: <Self as I32x8Backend>::Repr) -> <Self as F32x8Backend>::Repr {
        <T::Narrow as ops::F32x8ConvertOps>::bitcast_i32_to_f32(self.narrow(), a)
    }
    #[inline(always)]
    fn convert_f32_to_i32(self, a: <Self as F32x8Backend>::Repr) -> <Self as I32x8Backend>::Repr {
        <T::Narrow as ops::F32x8ConvertOps>::convert_f32_to_i32(self.narrow(), a)
    }
    #[inline(always)]
    fn convert_f32_to_i32_saturating(
        self,
        a: <Self as F32x8Backend>::Repr,
    ) -> <Self as I32x8Backend>::Repr {
        <T::Narrow as ops::F32x8ConvertOps>::convert_f32_to_i32_saturating(self.narrow(), a)
    }
    #[inline(always)]
    fn convert_f32_to_i32_fast(
        self,
        a: <Self as F32x8Backend>::Repr,
    ) -> <Self as I32x8Backend>::Repr {
        self.wasm_fma_f32x8_cvt_convert_f32_to_i32_fast(a)
    }
    #[inline(always)]
    fn convert_f32_to_i32_round(
        self,
        a: <Self as F32x8Backend>::Repr,
    ) -> <Self as I32x8Backend>::Repr {
        <T::Narrow as ops::F32x8ConvertOps>::convert_f32_to_i32_round(self.narrow(), a)
    }
    #[inline(always)]
    fn convert_i32_to_f32(self, a: <Self as I32x8Backend>::Repr) -> <Self as F32x8Backend>::Repr {
        <T::Narrow as ops::F32x8ConvertOps>::convert_i32_to_f32(self.narrow(), a)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> F64x2Convert for T {
    #[inline(always)]
    fn convert_f64_to_i32_low(
        self,
        a: <Self as F64x2Backend>::Repr,
    ) -> <Self as I32x4Backend>::Repr {
        <T::Narrow as ops::F64x2ConvertOps>::convert_f64_to_i32_low(self.narrow(), a)
    }
    #[inline(always)]
    fn convert_f64_to_i32_low_saturating(
        self,
        a: <Self as F64x2Backend>::Repr,
    ) -> <Self as I32x4Backend>::Repr {
        < T :: Narrow as crate :: simd :: backends :: ops :: F64x2ConvertOps > :: convert_f64_to_i32_low_saturating (self . narrow () , a)
    }
    #[inline(always)]
    fn convert_f64_to_i32_low_round(
        self,
        a: <Self as F64x2Backend>::Repr,
    ) -> <Self as I32x4Backend>::Repr {
        <T::Narrow as ops::F64x2ConvertOps>::convert_f64_to_i32_low_round(self.narrow(), a)
    }
    #[inline(always)]
    fn convert_i32_low_to_f64(
        self,
        a: <Self as I32x4Backend>::Repr,
    ) -> <Self as F64x2Backend>::Repr {
        <T::Narrow as ops::F64x2ConvertOps>::convert_i32_low_to_f64(self.narrow(), a)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> F64x4Convert for T {
    #[inline(always)]
    fn convert_f64_to_i32(self, a: <Self as F64x4Backend>::Repr) -> <Self as I32x4Backend>::Repr {
        <T::Narrow as ops::F64x4ConvertOps>::convert_f64_to_i32(self.narrow(), a)
    }
    #[inline(always)]
    fn convert_f64_to_i32_saturating(
        self,
        a: <Self as F64x4Backend>::Repr,
    ) -> <Self as I32x4Backend>::Repr {
        <T::Narrow as ops::F64x4ConvertOps>::convert_f64_to_i32_saturating(self.narrow(), a)
    }
    #[inline(always)]
    fn convert_f64_to_i32_round(
        self,
        a: <Self as F64x4Backend>::Repr,
    ) -> <Self as I32x4Backend>::Repr {
        <T::Narrow as ops::F64x4ConvertOps>::convert_f64_to_i32_round(self.narrow(), a)
    }
    #[inline(always)]
    fn convert_i32_to_f64(self, a: <Self as I32x4Backend>::Repr) -> <Self as F64x4Backend>::Repr {
        <T::Narrow as ops::F64x4ConvertOps>::convert_i32_to_f64(self.narrow(), a)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> U32x4Bitcast for T {
    #[inline(always)]
    fn bitcast_u32_to_i32(self, a: <Self as U32x4Backend>::Repr) -> <Self as I32x4Backend>::Repr {
        <T::Narrow as ops::U32x4BitcastOps>::bitcast_u32_to_i32(self.narrow(), a)
    }
    #[inline(always)]
    fn bitcast_i32_to_u32(self, a: <Self as I32x4Backend>::Repr) -> <Self as U32x4Backend>::Repr {
        <T::Narrow as ops::U32x4BitcastOps>::bitcast_i32_to_u32(self.narrow(), a)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> U32x8Bitcast for T {
    #[inline(always)]
    fn bitcast_u32_to_i32(self, a: <Self as U32x8Backend>::Repr) -> <Self as I32x8Backend>::Repr {
        <T::Narrow as ops::U32x8BitcastOps>::bitcast_u32_to_i32(self.narrow(), a)
    }
    #[inline(always)]
    fn bitcast_i32_to_u32(self, a: <Self as I32x8Backend>::Repr) -> <Self as U32x8Backend>::Repr {
        <T::Narrow as ops::U32x8BitcastOps>::bitcast_i32_to_u32(self.narrow(), a)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> I64x2Bitcast for T {
    #[inline(always)]
    fn bitcast_i64_to_f64(self, a: <Self as I64x2Backend>::Repr) -> <Self as F64x2Backend>::Repr {
        <T::Narrow as ops::I64x2BitcastOps>::bitcast_i64_to_f64(self.narrow(), a)
    }
    #[inline(always)]
    fn bitcast_f64_to_i64(self, a: <Self as F64x2Backend>::Repr) -> <Self as I64x2Backend>::Repr {
        <T::Narrow as ops::I64x2BitcastOps>::bitcast_f64_to_i64(self.narrow(), a)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> I64x4Bitcast for T {
    #[inline(always)]
    fn bitcast_i64_to_f64(self, a: <Self as I64x4Backend>::Repr) -> <Self as F64x4Backend>::Repr {
        <T::Narrow as ops::I64x4BitcastOps>::bitcast_i64_to_f64(self.narrow(), a)
    }
    #[inline(always)]
    fn bitcast_f64_to_i64(self, a: <Self as F64x4Backend>::Repr) -> <Self as I64x4Backend>::Repr {
        <T::Narrow as ops::I64x4BitcastOps>::bitcast_f64_to_i64(self.narrow(), a)
    }
}

#[cfg(feature = "w512")]
impl<T: archmage::SimdToken + Sealed + Copy + 'static> F32x16Convert for T {
    #[inline(always)]
    fn bitcast_f32_to_i32(self, a: <Self as F32x16Backend>::Repr) -> <Self as I32x16Backend>::Repr {
        <T::W512 as ops::F32x16ConvertOps>::bitcast_f32_to_i32(self.w512(), a)
    }
    #[inline(always)]
    fn bitcast_i32_to_f32(self, a: <Self as I32x16Backend>::Repr) -> <Self as F32x16Backend>::Repr {
        <T::W512 as ops::F32x16ConvertOps>::bitcast_i32_to_f32(self.w512(), a)
    }
    #[inline(always)]
    fn convert_f32_to_i32(self, a: <Self as F32x16Backend>::Repr) -> <Self as I32x16Backend>::Repr {
        <T::W512 as ops::F32x16ConvertOps>::convert_f32_to_i32(self.w512(), a)
    }
    #[inline(always)]
    fn convert_f32_to_i32_saturating(
        self,
        a: <Self as F32x16Backend>::Repr,
    ) -> <Self as I32x16Backend>::Repr {
        <T::W512 as ops::F32x16ConvertOps>::convert_f32_to_i32_saturating(self.w512(), a)
    }
    #[inline(always)]
    fn convert_f32_to_i32_fast(
        self,
        a: <Self as F32x16Backend>::Repr,
    ) -> <Self as I32x16Backend>::Repr {
        self.wasm_fma_f32x16_cvt_convert_f32_to_i32_fast(a)
    }
    #[inline(always)]
    fn convert_f32_to_i32_round(
        self,
        a: <Self as F32x16Backend>::Repr,
    ) -> <Self as I32x16Backend>::Repr {
        <T::W512 as ops::F32x16ConvertOps>::convert_f32_to_i32_round(self.w512(), a)
    }
    #[inline(always)]
    fn convert_i32_to_f32(self, a: <Self as I32x16Backend>::Repr) -> <Self as F32x16Backend>::Repr {
        <T::W512 as ops::F32x16ConvertOps>::convert_i32_to_f32(self.w512(), a)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> I8x16Bitcast for T {
    #[inline(always)]
    fn bitcast_i8_to_u8(self, a: <Self as I8x16Backend>::Repr) -> <Self as U8x16Backend>::Repr {
        <T::Narrow as ops::I8x16BitcastOps>::bitcast_i8_to_u8(self.narrow(), a)
    }
    #[inline(always)]
    fn bitcast_u8_to_i8(self, a: <Self as U8x16Backend>::Repr) -> <Self as I8x16Backend>::Repr {
        <T::Narrow as ops::I8x16BitcastOps>::bitcast_u8_to_i8(self.narrow(), a)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> I8x32Bitcast for T {
    #[inline(always)]
    fn bitcast_i8_to_u8(self, a: <Self as I8x32Backend>::Repr) -> <Self as U8x32Backend>::Repr {
        <T::Narrow as ops::I8x32BitcastOps>::bitcast_i8_to_u8(self.narrow(), a)
    }
    #[inline(always)]
    fn bitcast_u8_to_i8(self, a: <Self as U8x32Backend>::Repr) -> <Self as I8x32Backend>::Repr {
        <T::Narrow as ops::I8x32BitcastOps>::bitcast_u8_to_i8(self.narrow(), a)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> I16x8Bitcast for T {
    #[inline(always)]
    fn bitcast_i16_to_u16(self, a: <Self as I16x8Backend>::Repr) -> <Self as U16x8Backend>::Repr {
        <T::Narrow as ops::I16x8BitcastOps>::bitcast_i16_to_u16(self.narrow(), a)
    }
    #[inline(always)]
    fn bitcast_u16_to_i16(self, a: <Self as U16x8Backend>::Repr) -> <Self as I16x8Backend>::Repr {
        <T::Narrow as ops::I16x8BitcastOps>::bitcast_u16_to_i16(self.narrow(), a)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> I16x16Bitcast for T {
    #[inline(always)]
    fn bitcast_i16_to_u16(self, a: <Self as I16x16Backend>::Repr) -> <Self as U16x16Backend>::Repr {
        <T::Narrow as ops::I16x16BitcastOps>::bitcast_i16_to_u16(self.narrow(), a)
    }
    #[inline(always)]
    fn bitcast_u16_to_i16(self, a: <Self as U16x16Backend>::Repr) -> <Self as I16x16Backend>::Repr {
        <T::Narrow as ops::I16x16BitcastOps>::bitcast_u16_to_i16(self.narrow(), a)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> U64x2Bitcast for T {
    #[inline(always)]
    fn bitcast_u64_to_i64(self, a: <Self as U64x2Backend>::Repr) -> <Self as I64x2Backend>::Repr {
        <T::Narrow as ops::U64x2BitcastOps>::bitcast_u64_to_i64(self.narrow(), a)
    }
    #[inline(always)]
    fn bitcast_i64_to_u64(self, a: <Self as I64x2Backend>::Repr) -> <Self as U64x2Backend>::Repr {
        <T::Narrow as ops::U64x2BitcastOps>::bitcast_i64_to_u64(self.narrow(), a)
    }
}

impl<T: archmage::SimdToken + Sealed + Copy + 'static> U64x4Bitcast for T {
    #[inline(always)]
    fn bitcast_u64_to_i64(self, a: <Self as U64x4Backend>::Repr) -> <Self as I64x4Backend>::Repr {
        <T::Narrow as ops::U64x4BitcastOps>::bitcast_u64_to_i64(self.narrow(), a)
    }
    #[inline(always)]
    fn bitcast_i64_to_u64(self, a: <Self as I64x4Backend>::Repr) -> <Self as U64x4Backend>::Repr {
        <T::Narrow as ops::U64x4BitcastOps>::bitcast_i64_to_u64(self.narrow(), a)
    }
}

#[cfg(feature = "w512")]
impl<T: archmage::SimdToken + Sealed + Copy + 'static> F64x8Convert for T {
    #[inline(always)]
    fn convert_f64_to_i32(self, a: <Self as F64x8Backend>::Repr) -> <Self as I32x8Backend>::Repr {
        self.f64x8_convert_to_i32(a)
    }
    #[inline(always)]
    fn convert_f64_to_i32_saturating(
        self,
        a: <Self as F64x8Backend>::Repr,
    ) -> <Self as I32x8Backend>::Repr {
        self.f64x8_convert_to_i32_saturating(a)
    }
    #[inline(always)]
    fn convert_f64_to_i32_round(
        self,
        a: <Self as F64x8Backend>::Repr,
    ) -> <Self as I32x8Backend>::Repr {
        self.f64x8_convert_to_i32_round(a)
    }
    #[inline(always)]
    fn convert_i32_to_f64(self, a: <Self as I32x8Backend>::Repr) -> <Self as F64x8Backend>::Repr {
        self.f64x8_convert_from_i32(a)
    }
}

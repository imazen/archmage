//! Marker trait sealing all SIMD backend traits to validated archmage tokens
//! and routing all 21 tokens through their canonical base `*Ops` traits.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#[allow(unused_imports)]
use archmage::SimdToken as _;

/// Seals backend traits so only validated archmage tokens can implement them.
///
/// Also supplies the canonical `Narrow` / `W512` projections and optional
/// hardware override hooks (AVX-512VL pixel packing, WASM `relaxed-simd`
/// single-rounding ops, AVX-512F `to_u32_fast` / `dot_i8` / `f16` slice,
/// `F32x16FromHalves`).
pub trait Sealed: Copy + 'static {
    /// Canonical 128/256-bit backend token implementing [`super::ops::NarrowOps`].
    type Narrow: super::ops::NarrowOps;

    /// Canonical 512-bit backend token implementing [`super::ops::W512Ops`].
    #[cfg(feature = "w512")]
    type W512: super::ops::W512Ops;

    /// Project `self` to its canonical narrow backend proof.
    fn narrow(self) -> Self::Narrow;

    /// Project `self` to its canonical 512-bit backend proof.
    #[cfg(feature = "w512")]
    fn w512(self) -> Self::W512;

    #[inline(always)]
    fn v4_f32x4_to_u8_bytes(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> [u8; 4] {
        <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::to_u8_bytes(self.narrow(), a)
    }
    #[inline(always)]
    fn v4_f32x4_store_rgba_bytes(
        self,
        r: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        g: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        a: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> [u8; 16] {
        <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::store_rgba_bytes(
            self.narrow(),
            r,
            g,
            b,
            a,
        )
    }
    #[inline(always)]
    fn v4_f32x4_store_rgb_bytes(
        self,
        r: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        g: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> [u8; 12] {
        <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::store_rgb_bytes(
            self.narrow(),
            r,
            g,
            b,
        )
    }
    #[inline(always)]
    fn v4_f32x8_to_u8_bytes(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
    ) -> [u8; 8] {
        <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::to_u8_bytes(self.narrow(), a)
    }
    #[inline(always)]
    fn wasm_fma_f32x4_mul_add(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::mul_add(self.narrow(), a, b, c)
    }
    #[inline(always)]
    fn wasm_fma_f32x4_mul_sub(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::mul_sub(self.narrow(), a, b, c)
    }
    #[inline(always)]
    fn wasm_fma_f32x4_mul_add_portable(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::mul_add_portable(
            self.narrow(),
            a,
            b,
            c,
        )
    }
    #[inline(always)]
    fn wasm_fma_f32x4_mul_sub_portable(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::mul_sub_portable(
            self.narrow(),
            a,
            b,
            c,
        )
    }
    #[inline(always)]
    fn wasm_fma_f32x4_min_fast(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::min_fast(self.narrow(), a, b)
    }
    #[inline(always)]
    fn wasm_fma_f32x4_max_fast(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::max_fast(self.narrow(), a, b)
    }
    #[inline(always)]
    fn wasm_fma_f32x4_mul_add_lane_relaxed<const LANE: i32>(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        v_in: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        acc: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::mul_add_lane_relaxed::<LANE>(
            self.narrow(),
            a,
            v_in,
            acc,
        )
    }
    #[inline(always)]
    fn wasm_fma_f32x8_mul_add(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::mul_add(self.narrow(), a, b, c)
    }
    #[inline(always)]
    fn wasm_fma_f32x8_mul_sub(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::mul_sub(self.narrow(), a, b, c)
    }
    #[inline(always)]
    fn wasm_fma_f32x8_mul_add_portable(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::mul_add_portable(
            self.narrow(),
            a,
            b,
            c,
        )
    }
    #[inline(always)]
    fn wasm_fma_f32x8_mul_sub_portable(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::mul_sub_portable(
            self.narrow(),
            a,
            b,
            c,
        )
    }
    #[inline(always)]
    fn wasm_fma_f32x8_min_fast(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::min_fast(self.narrow(), a, b)
    }
    #[inline(always)]
    fn wasm_fma_f32x8_max_fast(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::max_fast(self.narrow(), a, b)
    }
    #[inline(always)]
    fn wasm_fma_f32x8_mul_add_lane_relaxed<const LANE: i32>(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
        v_in: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
        acc: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::mul_add_lane_relaxed::<LANE>(
            self.narrow(),
            a,
            v_in,
            acc,
        )
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f32x16_mul_add(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
        c: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr {
        <Self::W512 as crate::simd::backends::ops::F32x16Ops>::mul_add(self.w512(), a, b, c)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f32x16_mul_sub(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
        c: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr {
        <Self::W512 as crate::simd::backends::ops::F32x16Ops>::mul_sub(self.w512(), a, b, c)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f32x16_mul_add_portable(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
        c: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr {
        <Self::W512 as crate::simd::backends::ops::F32x16Ops>::mul_add_portable(
            self.w512(),
            a,
            b,
            c,
        )
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f32x16_mul_sub_portable(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
        c: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr {
        <Self::W512 as crate::simd::backends::ops::F32x16Ops>::mul_sub_portable(
            self.w512(),
            a,
            b,
            c,
        )
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f32x16_min_fast(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr {
        <Self::W512 as crate::simd::backends::ops::F32x16Ops>::min_fast(self.w512(), a, b)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f32x16_max_fast(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr {
        <Self::W512 as crate::simd::backends::ops::F32x16Ops>::max_fast(self.w512(), a, b)
    }
    #[inline(always)]
    fn wasm_fma_f64x2_mul_add(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::mul_add(self.narrow(), a, b, c)
    }
    #[inline(always)]
    fn wasm_fma_f64x2_mul_sub(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::mul_sub(self.narrow(), a, b, c)
    }
    #[inline(always)]
    fn wasm_fma_f64x2_mul_add_portable(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::mul_add_portable(
            self.narrow(),
            a,
            b,
            c,
        )
    }
    #[inline(always)]
    fn wasm_fma_f64x2_mul_sub_portable(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::mul_sub_portable(
            self.narrow(),
            a,
            b,
            c,
        )
    }
    #[inline(always)]
    fn wasm_fma_f64x2_min_fast(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::min_fast(self.narrow(), a, b)
    }
    #[inline(always)]
    fn wasm_fma_f64x2_max_fast(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::max_fast(self.narrow(), a, b)
    }
    #[inline(always)]
    fn wasm_fma_f64x4_mul_add(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::mul_add(self.narrow(), a, b, c)
    }
    #[inline(always)]
    fn wasm_fma_f64x4_mul_sub(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::mul_sub(self.narrow(), a, b, c)
    }
    #[inline(always)]
    fn wasm_fma_f64x4_mul_add_portable(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::mul_add_portable(
            self.narrow(),
            a,
            b,
            c,
        )
    }
    #[inline(always)]
    fn wasm_fma_f64x4_mul_sub_portable(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::mul_sub_portable(
            self.narrow(),
            a,
            b,
            c,
        )
    }
    #[inline(always)]
    fn wasm_fma_f64x4_min_fast(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::min_fast(self.narrow(), a, b)
    }
    #[inline(always)]
    fn wasm_fma_f64x4_max_fast(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::max_fast(self.narrow(), a, b)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f64x8_mul_add(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
        c: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr {
        <Self::W512 as crate::simd::backends::ops::F64x8Ops>::mul_add(self.w512(), a, b, c)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f64x8_mul_sub(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
        c: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr {
        <Self::W512 as crate::simd::backends::ops::F64x8Ops>::mul_sub(self.w512(), a, b, c)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f64x8_mul_add_portable(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
        c: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr {
        <Self::W512 as crate::simd::backends::ops::F64x8Ops>::mul_add_portable(self.w512(), a, b, c)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f64x8_mul_sub_portable(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
        c: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr {
        <Self::W512 as crate::simd::backends::ops::F64x8Ops>::mul_sub_portable(self.w512(), a, b, c)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f64x8_min_fast(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr {
        <Self::W512 as crate::simd::backends::ops::F64x8Ops>::min_fast(self.w512(), a, b)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f64x8_max_fast(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr {
        <Self::W512 as crate::simd::backends::ops::F64x8Ops>::max_fast(self.w512(), a, b)
    }
    #[inline(always)]
    fn wasm_fma_f32x4_cvt_convert_f32_to_i32_fast(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::I32x4Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F32x4ConvertOps>::convert_f32_to_i32_fast(
            self.narrow(),
            a,
        )
    }
    #[inline(always)]
    fn wasm_fma_f32x8_cvt_convert_f32_to_i32_fast(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::I32x8Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::F32x8ConvertOps>::convert_f32_to_i32_fast(
            self.narrow(),
            a,
        )
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f32x16_cvt_convert_f32_to_i32_fast(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::I32x16Ops>::Repr {
        <Self::W512 as crate::simd::backends::ops::F32x16ConvertOps>::convert_f32_to_i32_fast(
            self.w512(),
            a,
        )
    }
    #[inline(always)]
    fn wasm_fma_i16x8_mul_hrs_q15(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::I16x8Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::I16x8Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::I16x8Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::I16x8Ops>::mul_hrs_q15(self.narrow(), a, b)
    }
    #[inline(always)]
    fn wasm_fma_i16x16_mul_hrs_q15(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::I16x16Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::I16x16Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::I16x16Ops>::Repr {
        <Self::Narrow as crate::simd::backends::ops::I16x16Ops>::mul_hrs_q15(self.narrow(), a, b)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_i16x32_mul_hrs_q15(
        self,
        a: <Self::W512 as crate::simd::backends::ops::I16x32Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::I16x32Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::I16x32Ops>::Repr {
        <Self::W512 as crate::simd::backends::ops::I16x32Ops>::mul_hrs_q15(self.w512(), a, b)
    }

    // Extension-trait override hooks (used by `simd::generic::*` blanket impls).

    #[inline(always)]
    fn f32x8_from_halves(
        self,
        lo: <Self::Narrow as super::ops::F32x4Ops>::Repr,
        hi: <Self::Narrow as super::ops::F32x4Ops>::Repr,
    ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr {
        <Self::Narrow as crate::simd::generic::cross_width::NarrowCrossWidthOps>::from_halves_x8(
            self.narrow(),
            lo,
            hi,
        )
    }

    #[inline(always)]
    fn f32x8_low(
        self,
        wide: <Self::Narrow as super::ops::F32x8Ops>::Repr,
    ) -> <Self::Narrow as super::ops::F32x4Ops>::Repr {
        <Self::Narrow as crate::simd::generic::cross_width::NarrowCrossWidthOps>::low_x8(
            self.narrow(),
            wide,
        )
    }

    #[inline(always)]
    fn f32x8_high(
        self,
        wide: <Self::Narrow as super::ops::F32x8Ops>::Repr,
    ) -> <Self::Narrow as super::ops::F32x4Ops>::Repr {
        <Self::Narrow as crate::simd::generic::cross_width::NarrowCrossWidthOps>::high_x8(
            self.narrow(),
            wide,
        )
    }

    #[inline(always)]
    fn mul_add_lane4_relaxed_x8<const LANE: i32>(
        self,
        a: <Self::Narrow as super::ops::F32x8Ops>::Repr,
        v_in: <Self::Narrow as super::ops::F32x4Ops>::Repr,
        acc: <Self::Narrow as super::ops::F32x8Ops>::Repr,
    ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr {
        <Self::Narrow as crate::simd::generic::cross_width::NarrowCrossWidthOps>::mul_add_lane4_relaxed_x8::<LANE>(
            self.narrow(),
            a,
            v_in,
            acc,
        )
    }

    #[cfg(feature = "w512")]
    fn f32x16_from_halves(
        self,
        lo: <Self::Narrow as super::ops::F32x8Ops>::Repr,
        hi: <Self::Narrow as super::ops::F32x8Ops>::Repr,
    ) -> <Self::W512 as super::ops::F32x16Ops>::Repr;

    #[cfg(feature = "w512")]
    fn f32x16_low(
        self,
        wide: <Self::W512 as super::ops::F32x16Ops>::Repr,
    ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr;

    #[cfg(feature = "w512")]
    fn f32x16_high(
        self,
        wide: <Self::W512 as super::ops::F32x16Ops>::Repr,
    ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr;

    #[inline(always)]
    fn f32x4_to_u32_fast(
        self,
        a: <Self::Narrow as super::ops::F32x4Ops>::Repr,
    ) -> <Self::Narrow as super::ops::U32x4Ops>::Repr {
        <Self::Narrow as crate::simd::generic::fast_ops::NarrowToU32Ops>::f32x4_to_u32_fast(
            self.narrow(),
            a,
        )
    }

    #[inline(always)]
    fn f32x8_to_u32_fast(
        self,
        a: <Self::Narrow as super::ops::F32x8Ops>::Repr,
    ) -> <Self::Narrow as super::ops::U32x8Ops>::Repr {
        <Self::Narrow as crate::simd::generic::fast_ops::NarrowToU32Ops>::f32x8_to_u32_fast(
            self.narrow(),
            a,
        )
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_to_u32_fast(
        self,
        a: <Self::W512 as super::ops::F32x16Ops>::Repr,
    ) -> <Self::W512 as super::ops::U32x16Ops>::Repr {
        <Self::W512 as crate::simd::generic::fast_ops::W512ToU32Ops>::f32x16_to_u32_fast(
            self.w512(),
            a,
        )
    }

    #[inline(always)]
    fn i32x4_dot_i8_i7_add(
        self,
        acc: <Self::Narrow as super::ops::I32x4Ops>::Repr,
        a: <Self::Narrow as super::ops::I8x16Ops>::Repr,
        b: <Self::Narrow as super::ops::I8x16Ops>::Repr,
    ) -> <Self::Narrow as super::ops::I32x4Ops>::Repr {
        <Self::Narrow as crate::simd::generic::quant_ops::NarrowDotI8Ops>::dot_i8_i7_add_x4(
            self.narrow(),
            acc,
            a,
            b,
        )
    }

    #[inline(always)]
    fn i32x8_dot_i8_i7_add(
        self,
        acc: <Self::Narrow as super::ops::I32x8Ops>::Repr,
        a: <Self::Narrow as super::ops::I8x32Ops>::Repr,
        b: <Self::Narrow as super::ops::I8x32Ops>::Repr,
    ) -> <Self::Narrow as super::ops::I32x8Ops>::Repr {
        <Self::Narrow as crate::simd::generic::quant_ops::NarrowDotI8Ops>::dot_i8_i7_add_x8(
            self.narrow(),
            acc,
            a,
            b,
        )
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn i32x16_dot_i8_i7_add(
        self,
        acc: <Self::W512 as super::ops::I32x16Ops>::Repr,
        a: <Self::W512 as super::ops::I8x64Ops>::Repr,
        b: <Self::W512 as super::ops::I8x64Ops>::Repr,
    ) -> <Self::W512 as super::ops::I32x16Ops>::Repr {
        <Self::W512 as crate::simd::generic::quant_ops::W512DotI8Ops>::dot_i8_i7_add_x16(
            self.w512(),
            acc,
            a,
            b,
        )
    }

    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]);

    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]);

    #[cfg(feature = "w512")]
    fn f64x8_convert_to_i32(
        self,
        a: <Self::W512 as super::ops::F64x8Ops>::Repr,
    ) -> <Self::Narrow as super::ops::I32x8Ops>::Repr;

    #[cfg(feature = "w512")]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: <Self::W512 as super::ops::F64x8Ops>::Repr,
    ) -> <Self::Narrow as super::ops::I32x8Ops>::Repr;

    #[cfg(feature = "w512")]
    fn f64x8_convert_to_i32_round(
        self,
        a: <Self::W512 as super::ops::F64x8Ops>::Repr,
    ) -> <Self::Narrow as super::ops::I32x8Ops>::Repr;

    #[cfg(feature = "w512")]
    fn f64x8_convert_from_i32(
        self,
        a: <Self::Narrow as super::ops::I32x8Ops>::Repr,
    ) -> <Self::W512 as super::ops::F64x8Ops>::Repr;
}

impl Sealed for archmage::ScalarToken {
    type Narrow = archmage::ScalarToken;
    #[cfg(feature = "w512")]
    type W512 = archmage::ScalarToken;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        self
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        self
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(self, lo: [f32; 8], hi: [f32; 8]) -> [f32; 16] {
        crate::simd::generic::cross_width::scalar_f32x16_from_halves(lo, hi)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(self, wide: [f32; 16]) -> [f32; 8] {
        crate::simd::generic::cross_width::scalar_f32x16_low(wide)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(self, wide: [f32; 16]) -> [f32; 8] {
        crate::simd::generic::cross_width::scalar_f32x16_high(wide)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(self, a: [f64; 8]) -> [i32; 8] {
        crate::simd::impls::scalar::scalar_f64x8_to_i32(a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(self, a: [f64; 8]) -> [i32; 8] {
        crate::simd::impls::scalar::scalar_f64x8_to_i32(a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(self, a: [f64; 8]) -> [i32; 8] {
        crate::simd::impls::scalar::scalar_f64x8_to_i32_round(a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(self, a: [i32; 8]) -> [f64; 8] {
        crate::simd::impls::scalar::scalar_f64x8_from_i32(a)
    }

    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::f16_to_f32_slice_soft(self.narrow(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::f32_to_f16_slice_soft(self.narrow(), input, output);
    }
}

#[cfg(target_arch = "x86_64")]
impl Sealed for archmage::X64V1Token {
    type Narrow = archmage::ScalarToken;
    #[cfg(feature = "w512")]
    type W512 = archmage::ScalarToken;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        archmage::ScalarToken
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        archmage::ScalarToken
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(self, lo: [f32; 8], hi: [f32; 8]) -> [f32; 16] {
        crate::simd::generic::cross_width::scalar_f32x16_from_halves(lo, hi)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(self, wide: [f32; 16]) -> [f32; 8] {
        crate::simd::generic::cross_width::scalar_f32x16_low(wide)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(self, wide: [f32; 16]) -> [f32; 8] {
        crate::simd::generic::cross_width::scalar_f32x16_high(wide)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(self, a: [f64; 8]) -> [i32; 8] {
        crate::simd::impls::scalar::scalar_f64x8_to_i32(a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(self, a: [f64; 8]) -> [i32; 8] {
        crate::simd::impls::scalar::scalar_f64x8_to_i32(a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(self, a: [f64; 8]) -> [i32; 8] {
        crate::simd::impls::scalar::scalar_f64x8_to_i32_round(a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(self, a: [i32; 8]) -> [f64; 8] {
        crate::simd::impls::scalar::scalar_f64x8_from_i32(a)
    }

    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::f16_to_f32_slice_soft(self.narrow(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::f32_to_f16_slice_soft(self.narrow(), input, output);
    }
}

#[cfg(target_arch = "x86_64")]
impl Sealed for archmage::X64V2Token {
    type Narrow = archmage::ScalarToken;
    #[cfg(feature = "w512")]
    type W512 = archmage::ScalarToken;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        archmage::ScalarToken
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        archmage::ScalarToken
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(self, lo: [f32; 8], hi: [f32; 8]) -> [f32; 16] {
        crate::simd::generic::cross_width::scalar_f32x16_from_halves(lo, hi)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(self, wide: [f32; 16]) -> [f32; 8] {
        crate::simd::generic::cross_width::scalar_f32x16_low(wide)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(self, wide: [f32; 16]) -> [f32; 8] {
        crate::simd::generic::cross_width::scalar_f32x16_high(wide)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(self, a: [f64; 8]) -> [i32; 8] {
        crate::simd::impls::scalar::scalar_f64x8_to_i32(a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(self, a: [f64; 8]) -> [i32; 8] {
        crate::simd::impls::scalar::scalar_f64x8_to_i32(a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(self, a: [f64; 8]) -> [i32; 8] {
        crate::simd::impls::scalar::scalar_f64x8_to_i32_round(a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(self, a: [i32; 8]) -> [f64; 8] {
        crate::simd::impls::scalar::scalar_f64x8_from_i32(a)
    }

    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::f16_to_f32_slice_soft(self.narrow(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::f32_to_f16_slice_soft(self.narrow(), input, output);
    }
}

#[cfg(target_arch = "x86_64")]
impl Sealed for archmage::X64CryptoToken {
    type Narrow = archmage::ScalarToken;
    #[cfg(feature = "w512")]
    type W512 = archmage::ScalarToken;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        archmage::ScalarToken
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        archmage::ScalarToken
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(self, lo: [f32; 8], hi: [f32; 8]) -> [f32; 16] {
        crate::simd::generic::cross_width::scalar_f32x16_from_halves(lo, hi)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(self, wide: [f32; 16]) -> [f32; 8] {
        crate::simd::generic::cross_width::scalar_f32x16_low(wide)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(self, wide: [f32; 16]) -> [f32; 8] {
        crate::simd::generic::cross_width::scalar_f32x16_high(wide)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(self, a: [f64; 8]) -> [i32; 8] {
        crate::simd::impls::scalar::scalar_f64x8_to_i32(a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(self, a: [f64; 8]) -> [i32; 8] {
        crate::simd::impls::scalar::scalar_f64x8_to_i32(a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(self, a: [f64; 8]) -> [i32; 8] {
        crate::simd::impls::scalar::scalar_f64x8_to_i32_round(a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(self, a: [i32; 8]) -> [f64; 8] {
        crate::simd::impls::scalar::scalar_f64x8_from_i32(a)
    }

    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::f16_to_f32_slice_soft(self.narrow(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::f32_to_f16_slice_soft(self.narrow(), input, output);
    }
}

#[cfg(target_arch = "x86_64")]
impl Sealed for archmage::X64V3Token {
    type Narrow = archmage::X64V3Token;
    #[cfg(feature = "w512")]
    type W512 = archmage::X64V3Token;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        self
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        self
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: <Self::Narrow as super::ops::F32x8Ops>::Repr,
        hi: <Self::Narrow as super::ops::F32x8Ops>::Repr,
    ) -> [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2] {
        [lo, hi]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(
        self,
        wide: [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2],
    ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr {
        wide[0]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(
        self,
        wide: [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2],
    ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr {
        wide[1]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32_saturating(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32_round(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(
        self,
        a: core::arch::x86_64::__m256i,
    ) -> [core::arch::x86_64::__m256d; 2] {
        crate::simd::impls::x86_v3::v3_f64x8_from_i32(self.narrow(), a)
    }

    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::f16c_decode_select(self, input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::f16c_encode_select(self, input, output);
    }
}

#[cfg(target_arch = "x86_64")]
impl Sealed for archmage::X64V3CryptoToken {
    type Narrow = archmage::X64V3Token;
    #[cfg(feature = "w512")]
    type W512 = archmage::X64V3Token;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        self.v3()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        self.v3()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: <Self::Narrow as super::ops::F32x8Ops>::Repr,
        hi: <Self::Narrow as super::ops::F32x8Ops>::Repr,
    ) -> [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2] {
        [lo, hi]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(
        self,
        wide: [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2],
    ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr {
        wide[0]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(
        self,
        wide: [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2],
    ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr {
        wide[1]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32_saturating(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32_round(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(
        self,
        a: core::arch::x86_64::__m256i,
    ) -> [core::arch::x86_64::__m256d; 2] {
        crate::simd::impls::x86_v3::v3_f64x8_from_i32(self.narrow(), a)
    }

    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::f16c_decode_select(self.v3(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::f16c_encode_select(self.v3(), input, output);
    }
}

#[cfg(target_arch = "x86_64")]
impl Sealed for archmage::X64V3GfniCryptoToken {
    type Narrow = archmage::X64V3Token;
    #[cfg(feature = "w512")]
    type W512 = archmage::X64V3Token;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        self.v3()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        self.v3()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: <Self::Narrow as super::ops::F32x8Ops>::Repr,
        hi: <Self::Narrow as super::ops::F32x8Ops>::Repr,
    ) -> [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2] {
        [lo, hi]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(
        self,
        wide: [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2],
    ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr {
        wide[0]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(
        self,
        wide: [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2],
    ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr {
        wide[1]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32_saturating(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32_round(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(
        self,
        a: core::arch::x86_64::__m256i,
    ) -> [core::arch::x86_64::__m256d; 2] {
        crate::simd::impls::x86_v3::v3_f64x8_from_i32(self.narrow(), a)
    }

    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::f16c_decode_select(self.v3(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::f16c_encode_select(self.v3(), input, output);
    }
}

#[cfg(target_arch = "x86_64")]
impl Sealed for archmage::X64V3VnniToken {
    type Narrow = archmage::X64V3Token;
    #[cfg(feature = "w512")]
    type W512 = archmage::X64V3Token;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        self.v3()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        self.v3()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: <Self::Narrow as super::ops::F32x8Ops>::Repr,
        hi: <Self::Narrow as super::ops::F32x8Ops>::Repr,
    ) -> [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2] {
        [lo, hi]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(
        self,
        wide: [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2],
    ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr {
        wide[0]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(
        self,
        wide: [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2],
    ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr {
        wide[1]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32_saturating(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32_round(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(
        self,
        a: core::arch::x86_64::__m256i,
    ) -> [core::arch::x86_64::__m256d; 2] {
        crate::simd::impls::x86_v3::v3_f64x8_from_i32(self.narrow(), a)
    }

    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::f16c_decode_select(self.v3(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::f16c_encode_select(self.v3(), input, output);
    }
}

#[cfg(target_arch = "x86_64")]
impl Sealed for archmage::X64V3Vnni2Token {
    type Narrow = archmage::X64V3Token;
    #[cfg(feature = "w512")]
    type W512 = archmage::X64V3Token;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        self.v3()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        self.v3()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: <Self::Narrow as super::ops::F32x8Ops>::Repr,
        hi: <Self::Narrow as super::ops::F32x8Ops>::Repr,
    ) -> [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2] {
        [lo, hi]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(
        self,
        wide: [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2],
    ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr {
        wide[0]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(
        self,
        wide: [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2],
    ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr {
        wide[1]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32_saturating(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32_round(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(
        self,
        a: core::arch::x86_64::__m256i,
    ) -> [core::arch::x86_64::__m256d; 2] {
        crate::simd::impls::x86_v3::v3_f64x8_from_i32(self.narrow(), a)
    }

    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::f16c_decode_select(self.v3(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::f16c_encode_select(self.v3(), input, output);
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Sealed for archmage::X64V4Token {
    type Narrow = archmage::X64V3Token;
    #[cfg(feature = "w512")]
    type W512 = archmage::X64V4Token;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        self.v3()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        self
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: core::arch::x86_64::__m256,
        hi: core::arch::x86_64::__m256,
    ) -> core::arch::x86_64::__m512 {
        crate::simd::generic::cross_width::v4_f32x16_from_halves(self, lo, hi)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(self, wide: core::arch::x86_64::__m512) -> core::arch::x86_64::__m256 {
        crate::simd::generic::cross_width::v4_f32x16_low(self, wide)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(self, wide: core::arch::x86_64::__m512) -> core::arch::x86_64::__m256 {
        crate::simd::generic::cross_width::v4_f32x16_high(self, wide)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(self, a: core::arch::x86_64::__m512d) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v4::v4_f64x8_to_i32(self, a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: core::arch::x86_64::__m512d,
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v4::v4_f64x8_to_i32_saturating(self, a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(
        self,
        a: core::arch::x86_64::__m512d,
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v4::v4_f64x8_to_i32_round(self, a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(self, a: core::arch::x86_64::__m256i) -> core::arch::x86_64::__m512d {
        crate::simd::impls::x86_v4::v4_f64x8_from_i32(self, a)
    }
    #[inline(always)]
    fn v4_f32x4_to_u8_bytes(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> [u8; 4] {
        crate::simd::impls::x86_v4_f32_overrides::to_u8_bytes_x4_v4(self, a)
    }
    #[inline(always)]
    fn v4_f32x4_store_rgba_bytes(
        self,
        r: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        g: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        a: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> [u8; 16] {
        crate::simd::impls::x86_v4_f32_overrides::store_rgba_bytes_x4_v4(self, r, g, b, a)
    }
    #[inline(always)]
    fn v4_f32x4_store_rgb_bytes(
        self,
        r: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        g: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> [u8; 12] {
        crate::simd::impls::x86_v4_f32_overrides::store_rgb_bytes_x4_v4(self, r, g, b)
    }
    #[inline(always)]
    fn v4_f32x8_to_u8_bytes(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
    ) -> [u8; 8] {
        crate::simd::impls::x86_v4_f32_overrides::to_u8_bytes_x8_v4(self, a)
    }
    #[inline(always)]
    fn f32x4_to_u32_fast(self, a: core::arch::x86_64::__m128) -> core::arch::x86_64::__m128i {
        crate::simd::generic::fast_ops::v4_f32x4_to_u32_fast(self, a)
    }
    #[inline(always)]
    fn f32x8_to_u32_fast(self, a: core::arch::x86_64::__m256) -> core::arch::x86_64::__m256i {
        crate::simd::generic::fast_ops::v4_f32x8_to_u32_fast(self, a)
    }
    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::v4_f16_to_f32_into(self, input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::v4_f32_to_f16_into(self, input, output);
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Sealed for archmage::X64V4xToken {
    type Narrow = archmage::X64V3Token;
    #[cfg(feature = "w512")]
    type W512 = archmage::X64V4Token;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        self.v3()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        self.v4()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: core::arch::x86_64::__m256,
        hi: core::arch::x86_64::__m256,
    ) -> core::arch::x86_64::__m512 {
        crate::simd::generic::cross_width::v4_f32x16_from_halves(self.v4(), lo, hi)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(self, wide: core::arch::x86_64::__m512) -> core::arch::x86_64::__m256 {
        crate::simd::generic::cross_width::v4_f32x16_low(self.v4(), wide)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(self, wide: core::arch::x86_64::__m512) -> core::arch::x86_64::__m256 {
        crate::simd::generic::cross_width::v4_f32x16_high(self.v4(), wide)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(self, a: core::arch::x86_64::__m512d) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v4::v4_f64x8_to_i32(self.v4(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: core::arch::x86_64::__m512d,
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v4::v4_f64x8_to_i32_saturating(self.v4(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(
        self,
        a: core::arch::x86_64::__m512d,
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v4::v4_f64x8_to_i32_round(self.v4(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(self, a: core::arch::x86_64::__m256i) -> core::arch::x86_64::__m512d {
        crate::simd::impls::x86_v4::v4_f64x8_from_i32(self.v4(), a)
    }
    #[inline(always)]
    fn v4_f32x4_to_u8_bytes(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> [u8; 4] {
        crate::simd::impls::x86_v4_f32_overrides::to_u8_bytes_x4_v4(self.v4(), a)
    }
    #[inline(always)]
    fn v4_f32x4_store_rgba_bytes(
        self,
        r: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        g: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        a: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> [u8; 16] {
        crate::simd::impls::x86_v4_f32_overrides::store_rgba_bytes_x4_v4(self.v4(), r, g, b, a)
    }
    #[inline(always)]
    fn v4_f32x4_store_rgb_bytes(
        self,
        r: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        g: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> [u8; 12] {
        crate::simd::impls::x86_v4_f32_overrides::store_rgb_bytes_x4_v4(self.v4(), r, g, b)
    }
    #[inline(always)]
    fn v4_f32x8_to_u8_bytes(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
    ) -> [u8; 8] {
        crate::simd::impls::x86_v4_f32_overrides::to_u8_bytes_x8_v4(self.v4(), a)
    }
    #[inline(always)]
    fn f32x4_to_u32_fast(self, a: core::arch::x86_64::__m128) -> core::arch::x86_64::__m128i {
        crate::simd::generic::fast_ops::v4_f32x4_to_u32_fast(self.v4(), a)
    }
    #[inline(always)]
    fn f32x8_to_u32_fast(self, a: core::arch::x86_64::__m256) -> core::arch::x86_64::__m256i {
        crate::simd::generic::fast_ops::v4_f32x8_to_u32_fast(self.v4(), a)
    }
    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::v4_f16_to_f32_into(self.v4(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::v4_f32_to_f16_into(self.v4(), input, output);
    }

    #[inline(always)]
    fn i32x4_dot_i8_i7_add(
        self,
        acc: core::arch::x86_64::__m128i,
        a: core::arch::x86_64::__m128i,
        b: core::arch::x86_64::__m128i,
    ) -> core::arch::x86_64::__m128i {
        crate::simd::generic::quant_ops::v4x_i32x4_dot_i8(self, acc, a, b)
    }
    #[inline(always)]
    fn i32x8_dot_i8_i7_add(
        self,
        acc: core::arch::x86_64::__m256i,
        a: core::arch::x86_64::__m256i,
        b: core::arch::x86_64::__m256i,
    ) -> core::arch::x86_64::__m256i {
        crate::simd::generic::quant_ops::v4x_i32x8_dot_i8(self, acc, a, b)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn i32x16_dot_i8_i7_add(
        self,
        acc: core::arch::x86_64::__m512i,
        a: core::arch::x86_64::__m512i,
        b: core::arch::x86_64::__m512i,
    ) -> core::arch::x86_64::__m512i {
        crate::simd::generic::quant_ops::v4x_i32x16_dot_i8(self, acc, a, b)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Sealed for archmage::Avx512Fp16Token {
    type Narrow = archmage::X64V3Token;
    #[cfg(feature = "w512")]
    type W512 = archmage::X64V4Token;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        self.v3()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        self.v4()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: core::arch::x86_64::__m256,
        hi: core::arch::x86_64::__m256,
    ) -> core::arch::x86_64::__m512 {
        crate::simd::generic::cross_width::v4_f32x16_from_halves(self.v4(), lo, hi)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(self, wide: core::arch::x86_64::__m512) -> core::arch::x86_64::__m256 {
        crate::simd::generic::cross_width::v4_f32x16_low(self.v4(), wide)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(self, wide: core::arch::x86_64::__m512) -> core::arch::x86_64::__m256 {
        crate::simd::generic::cross_width::v4_f32x16_high(self.v4(), wide)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(self, a: core::arch::x86_64::__m512d) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v4::v4_f64x8_to_i32(self.v4(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: core::arch::x86_64::__m512d,
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v4::v4_f64x8_to_i32_saturating(self.v4(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(
        self,
        a: core::arch::x86_64::__m512d,
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v4::v4_f64x8_to_i32_round(self.v4(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(self, a: core::arch::x86_64::__m256i) -> core::arch::x86_64::__m512d {
        crate::simd::impls::x86_v4::v4_f64x8_from_i32(self.v4(), a)
    }
    #[inline(always)]
    fn v4_f32x4_to_u8_bytes(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> [u8; 4] {
        crate::simd::impls::x86_v4_f32_overrides::to_u8_bytes_x4_v4(self.v4(), a)
    }
    #[inline(always)]
    fn v4_f32x4_store_rgba_bytes(
        self,
        r: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        g: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        a: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> [u8; 16] {
        crate::simd::impls::x86_v4_f32_overrides::store_rgba_bytes_x4_v4(self.v4(), r, g, b, a)
    }
    #[inline(always)]
    fn v4_f32x4_store_rgb_bytes(
        self,
        r: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        g: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> [u8; 12] {
        crate::simd::impls::x86_v4_f32_overrides::store_rgb_bytes_x4_v4(self.v4(), r, g, b)
    }
    #[inline(always)]
    fn v4_f32x8_to_u8_bytes(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
    ) -> [u8; 8] {
        crate::simd::impls::x86_v4_f32_overrides::to_u8_bytes_x8_v4(self.v4(), a)
    }
    #[inline(always)]
    fn f32x4_to_u32_fast(self, a: core::arch::x86_64::__m128) -> core::arch::x86_64::__m128i {
        crate::simd::generic::fast_ops::v4_f32x4_to_u32_fast(self.v4(), a)
    }
    #[inline(always)]
    fn f32x8_to_u32_fast(self, a: core::arch::x86_64::__m256) -> core::arch::x86_64::__m256i {
        crate::simd::generic::fast_ops::v4_f32x8_to_u32_fast(self.v4(), a)
    }
    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::v4_f16_to_f32_into(self.v4(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::v4_f32_to_f16_into(self.v4(), input, output);
    }
}

#[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
impl Sealed for archmage::X64V4Token {
    type Narrow = archmage::X64V3Token;
    #[cfg(feature = "w512")]
    type W512 = archmage::X64V3Token;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        self.v3()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        self.v3()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: <Self::Narrow as super::ops::F32x8Ops>::Repr,
        hi: <Self::Narrow as super::ops::F32x8Ops>::Repr,
    ) -> [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2] {
        [lo, hi]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(
        self,
        wide: [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2],
    ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr {
        wide[0]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(
        self,
        wide: [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2],
    ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr {
        wide[1]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32_saturating(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32_round(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(
        self,
        a: core::arch::x86_64::__m256i,
    ) -> [core::arch::x86_64::__m256d; 2] {
        crate::simd::impls::x86_v3::v3_f64x8_from_i32(self.narrow(), a)
    }

    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::f16c_decode_select(self.v3(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::f16c_encode_select(self.v3(), input, output);
    }
}

#[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
impl Sealed for archmage::X64V4xToken {
    type Narrow = archmage::X64V3Token;
    #[cfg(feature = "w512")]
    type W512 = archmage::X64V3Token;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        self.v3()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        self.v3()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: <Self::Narrow as super::ops::F32x8Ops>::Repr,
        hi: <Self::Narrow as super::ops::F32x8Ops>::Repr,
    ) -> [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2] {
        [lo, hi]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(
        self,
        wide: [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2],
    ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr {
        wide[0]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(
        self,
        wide: [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2],
    ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr {
        wide[1]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32_saturating(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32_round(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(
        self,
        a: core::arch::x86_64::__m256i,
    ) -> [core::arch::x86_64::__m256d; 2] {
        crate::simd::impls::x86_v3::v3_f64x8_from_i32(self.narrow(), a)
    }

    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::f16c_decode_select(self.v3(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::f16c_encode_select(self.v3(), input, output);
    }
}

#[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
impl Sealed for archmage::Avx512Fp16Token {
    type Narrow = archmage::X64V3Token;
    #[cfg(feature = "w512")]
    type W512 = archmage::X64V3Token;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        self.v3()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        self.v3()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: <Self::Narrow as super::ops::F32x8Ops>::Repr,
        hi: <Self::Narrow as super::ops::F32x8Ops>::Repr,
    ) -> [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2] {
        [lo, hi]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(
        self,
        wide: [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2],
    ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr {
        wide[0]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(
        self,
        wide: [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2],
    ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr {
        wide[1]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32_saturating(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32_round(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(
        self,
        a: core::arch::x86_64::__m256i,
    ) -> [core::arch::x86_64::__m256d; 2] {
        crate::simd::impls::x86_v3::v3_f64x8_from_i32(self.narrow(), a)
    }

    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::f16c_decode_select(self.v3(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::f16c_encode_select(self.v3(), input, output);
    }
}

#[cfg(target_arch = "aarch64")]
impl Sealed for archmage::NeonToken {
    type Narrow = archmage::NeonToken;
    #[cfg(feature = "w512")]
    type W512 = archmage::NeonToken;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        self
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        self
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2],
        hi: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4] {
        [lo[0], lo[1], hi[0], hi[1]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(
        self,
        wide: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2] {
        [wide[0], wide[1]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(
        self,
        wide: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2] {
        [wide[2], wide[3]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(
        self,
        a: [core::arch::aarch64::float64x2_t; 4],
    ) -> [core::arch::aarch64::int32x4_t; 2] {
        crate::simd::impls::arm_neon::neon_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: [core::arch::aarch64::float64x2_t; 4],
    ) -> [core::arch::aarch64::int32x4_t; 2] {
        crate::simd::impls::arm_neon::neon_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(
        self,
        a: [core::arch::aarch64::float64x2_t; 4],
    ) -> [core::arch::aarch64::int32x4_t; 2] {
        crate::simd::impls::arm_neon::neon_f64x8_to_i32_round(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(
        self,
        a: [core::arch::aarch64::int32x4_t; 2],
    ) -> [core::arch::aarch64::float64x2_t; 4] {
        crate::simd::impls::arm_neon::neon_f64x8_from_i32(self.narrow(), a)
    }

    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::neon_f16_decode(self, input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::neon_f16_encode(self, input, output);
    }
}

#[cfg(target_arch = "aarch64")]
impl Sealed for archmage::NeonAesToken {
    type Narrow = archmage::NeonToken;
    #[cfg(feature = "w512")]
    type W512 = archmage::NeonToken;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        self.neon()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        self.neon()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2],
        hi: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4] {
        [lo[0], lo[1], hi[0], hi[1]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(
        self,
        wide: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2] {
        [wide[0], wide[1]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(
        self,
        wide: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2] {
        [wide[2], wide[3]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(
        self,
        a: [core::arch::aarch64::float64x2_t; 4],
    ) -> [core::arch::aarch64::int32x4_t; 2] {
        crate::simd::impls::arm_neon::neon_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: [core::arch::aarch64::float64x2_t; 4],
    ) -> [core::arch::aarch64::int32x4_t; 2] {
        crate::simd::impls::arm_neon::neon_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(
        self,
        a: [core::arch::aarch64::float64x2_t; 4],
    ) -> [core::arch::aarch64::int32x4_t; 2] {
        crate::simd::impls::arm_neon::neon_f64x8_to_i32_round(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(
        self,
        a: [core::arch::aarch64::int32x4_t; 2],
    ) -> [core::arch::aarch64::float64x2_t; 4] {
        crate::simd::impls::arm_neon::neon_f64x8_from_i32(self.narrow(), a)
    }

    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::neon_f16_decode(self.neon(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::neon_f16_encode(self.neon(), input, output);
    }
}

#[cfg(target_arch = "aarch64")]
impl Sealed for archmage::NeonSha3Token {
    type Narrow = archmage::NeonToken;
    #[cfg(feature = "w512")]
    type W512 = archmage::NeonToken;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        self.neon()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        self.neon()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2],
        hi: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4] {
        [lo[0], lo[1], hi[0], hi[1]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(
        self,
        wide: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2] {
        [wide[0], wide[1]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(
        self,
        wide: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2] {
        [wide[2], wide[3]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(
        self,
        a: [core::arch::aarch64::float64x2_t; 4],
    ) -> [core::arch::aarch64::int32x4_t; 2] {
        crate::simd::impls::arm_neon::neon_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: [core::arch::aarch64::float64x2_t; 4],
    ) -> [core::arch::aarch64::int32x4_t; 2] {
        crate::simd::impls::arm_neon::neon_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(
        self,
        a: [core::arch::aarch64::float64x2_t; 4],
    ) -> [core::arch::aarch64::int32x4_t; 2] {
        crate::simd::impls::arm_neon::neon_f64x8_to_i32_round(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(
        self,
        a: [core::arch::aarch64::int32x4_t; 2],
    ) -> [core::arch::aarch64::float64x2_t; 4] {
        crate::simd::impls::arm_neon::neon_f64x8_from_i32(self.narrow(), a)
    }

    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::neon_f16_decode(self.neon(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::neon_f16_encode(self.neon(), input, output);
    }
}

#[cfg(target_arch = "aarch64")]
impl Sealed for archmage::NeonCrcToken {
    type Narrow = archmage::NeonToken;
    #[cfg(feature = "w512")]
    type W512 = archmage::NeonToken;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        self.neon()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        self.neon()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2],
        hi: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4] {
        [lo[0], lo[1], hi[0], hi[1]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(
        self,
        wide: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2] {
        [wide[0], wide[1]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(
        self,
        wide: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2] {
        [wide[2], wide[3]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(
        self,
        a: [core::arch::aarch64::float64x2_t; 4],
    ) -> [core::arch::aarch64::int32x4_t; 2] {
        crate::simd::impls::arm_neon::neon_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: [core::arch::aarch64::float64x2_t; 4],
    ) -> [core::arch::aarch64::int32x4_t; 2] {
        crate::simd::impls::arm_neon::neon_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(
        self,
        a: [core::arch::aarch64::float64x2_t; 4],
    ) -> [core::arch::aarch64::int32x4_t; 2] {
        crate::simd::impls::arm_neon::neon_f64x8_to_i32_round(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(
        self,
        a: [core::arch::aarch64::int32x4_t; 2],
    ) -> [core::arch::aarch64::float64x2_t; 4] {
        crate::simd::impls::arm_neon::neon_f64x8_from_i32(self.narrow(), a)
    }

    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::neon_f16_decode(self.neon(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::neon_f16_encode(self.neon(), input, output);
    }
}

#[cfg(target_arch = "aarch64")]
impl Sealed for archmage::Arm64V2Token {
    type Narrow = archmage::NeonToken;
    #[cfg(feature = "w512")]
    type W512 = archmage::NeonToken;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        self.neon()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        self.neon()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2],
        hi: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4] {
        [lo[0], lo[1], hi[0], hi[1]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(
        self,
        wide: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2] {
        [wide[0], wide[1]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(
        self,
        wide: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2] {
        [wide[2], wide[3]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(
        self,
        a: [core::arch::aarch64::float64x2_t; 4],
    ) -> [core::arch::aarch64::int32x4_t; 2] {
        crate::simd::impls::arm_neon::neon_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: [core::arch::aarch64::float64x2_t; 4],
    ) -> [core::arch::aarch64::int32x4_t; 2] {
        crate::simd::impls::arm_neon::neon_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(
        self,
        a: [core::arch::aarch64::float64x2_t; 4],
    ) -> [core::arch::aarch64::int32x4_t; 2] {
        crate::simd::impls::arm_neon::neon_f64x8_to_i32_round(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(
        self,
        a: [core::arch::aarch64::int32x4_t; 2],
    ) -> [core::arch::aarch64::float64x2_t; 4] {
        crate::simd::impls::arm_neon::neon_f64x8_from_i32(self.narrow(), a)
    }

    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::neon_f16_decode(self.neon(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::neon_f16_encode(self.neon(), input, output);
    }
}

#[cfg(target_arch = "aarch64")]
impl Sealed for archmage::Arm64V3Token {
    type Narrow = archmage::NeonToken;
    #[cfg(feature = "w512")]
    type W512 = archmage::NeonToken;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        self.neon()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        self.neon()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2],
        hi: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4] {
        [lo[0], lo[1], hi[0], hi[1]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(
        self,
        wide: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2] {
        [wide[0], wide[1]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(
        self,
        wide: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2] {
        [wide[2], wide[3]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(
        self,
        a: [core::arch::aarch64::float64x2_t; 4],
    ) -> [core::arch::aarch64::int32x4_t; 2] {
        crate::simd::impls::arm_neon::neon_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: [core::arch::aarch64::float64x2_t; 4],
    ) -> [core::arch::aarch64::int32x4_t; 2] {
        crate::simd::impls::arm_neon::neon_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(
        self,
        a: [core::arch::aarch64::float64x2_t; 4],
    ) -> [core::arch::aarch64::int32x4_t; 2] {
        crate::simd::impls::arm_neon::neon_f64x8_to_i32_round(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(
        self,
        a: [core::arch::aarch64::int32x4_t; 2],
    ) -> [core::arch::aarch64::float64x2_t; 4] {
        crate::simd::impls::arm_neon::neon_f64x8_from_i32(self.narrow(), a)
    }

    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::neon_f16_decode(self.neon(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::neon_f16_encode(self.neon(), input, output);
    }
}

#[cfg(target_arch = "wasm32")]
impl Sealed for archmage::Wasm128Token {
    type Narrow = archmage::Wasm128Token;
    #[cfg(feature = "w512")]
    type W512 = archmage::Wasm128Token;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        self
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        self
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2],
        hi: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4] {
        [lo[0], lo[1], hi[0], hi[1]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(
        self,
        wide: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2] {
        [wide[0], wide[1]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(
        self,
        wide: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2] {
        [wide[2], wide[3]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(
        self,
        a: [core::arch::wasm32::v128; 4],
    ) -> [core::arch::wasm32::v128; 2] {
        crate::simd::impls::wasm128::wasm_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: [core::arch::wasm32::v128; 4],
    ) -> [core::arch::wasm32::v128; 2] {
        crate::simd::impls::wasm128::wasm_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(
        self,
        a: [core::arch::wasm32::v128; 4],
    ) -> [core::arch::wasm32::v128; 2] {
        crate::simd::impls::wasm128::wasm_f64x8_to_i32_round(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(
        self,
        a: [core::arch::wasm32::v128; 2],
    ) -> [core::arch::wasm32::v128; 4] {
        crate::simd::impls::wasm128::wasm_f64x8_from_i32(self.narrow(), a)
    }

    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::f16_to_f32_slice_soft(self.narrow(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::f32_to_f16_slice_soft(self.narrow(), input, output);
    }
}

#[cfg(target_arch = "wasm32")]
impl Sealed for archmage::Wasm128RelaxedToken {
    type Narrow = archmage::Wasm128Token;
    #[cfg(feature = "w512")]
    type W512 = archmage::Wasm128Token;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        self.wasm128()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        self.wasm128()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2],
        hi: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4] {
        [lo[0], lo[1], hi[0], hi[1]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(
        self,
        wide: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2] {
        [wide[0], wide[1]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(
        self,
        wide: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2] {
        [wide[2], wide[3]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(
        self,
        a: [core::arch::wasm32::v128; 4],
    ) -> [core::arch::wasm32::v128; 2] {
        crate::simd::impls::wasm128::wasm_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: [core::arch::wasm32::v128; 4],
    ) -> [core::arch::wasm32::v128; 2] {
        crate::simd::impls::wasm128::wasm_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(
        self,
        a: [core::arch::wasm32::v128; 4],
    ) -> [core::arch::wasm32::v128; 2] {
        crate::simd::impls::wasm128::wasm_f64x8_to_i32_round(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(
        self,
        a: [core::arch::wasm32::v128; 2],
    ) -> [core::arch::wasm32::v128; 4] {
        crate::simd::impls::wasm128::wasm_f64x8_from_i32(self.narrow(), a)
    }

    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::f16_to_f32_slice_soft(self.narrow(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::f32_to_f16_slice_soft(self.narrow(), input, output);
    }
}

#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    target_feature = "relaxed-simd"
))]
impl Sealed for archmage::Wasm128RelaxedFmaToken {
    type Narrow = archmage::Wasm128Token;
    #[cfg(feature = "w512")]
    type W512 = archmage::Wasm128Token;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        self.wasm128()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        self.wasm128()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2],
        hi: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4] {
        [lo[0], lo[1], hi[0], hi[1]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(
        self,
        wide: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2] {
        [wide[0], wide[1]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(
        self,
        wide: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2] {
        [wide[2], wide[3]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(
        self,
        a: [core::arch::wasm32::v128; 4],
    ) -> [core::arch::wasm32::v128; 2] {
        crate::simd::impls::wasm128::wasm_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: [core::arch::wasm32::v128; 4],
    ) -> [core::arch::wasm32::v128; 2] {
        crate::simd::impls::wasm128::wasm_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(
        self,
        a: [core::arch::wasm32::v128; 4],
    ) -> [core::arch::wasm32::v128; 2] {
        crate::simd::impls::wasm128::wasm_f64x8_to_i32_round(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(
        self,
        a: [core::arch::wasm32::v128; 2],
    ) -> [core::arch::wasm32::v128; 4] {
        crate::simd::impls::wasm128::wasm_f64x8_from_i32(self.narrow(), a)
    }
    #[inline(always)]
    fn wasm_fma_f32x4_mul_add(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_add_f32x4(self, a, b, c)
    }
    #[inline(always)]
    fn wasm_fma_f32x4_mul_sub(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_sub_f32x4(self, a, b, c)
    }
    #[inline(always)]
    fn wasm_fma_f32x4_mul_add_portable(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_add_f32x4(self, a, b, c)
    }
    #[inline(always)]
    fn wasm_fma_f32x4_mul_sub_portable(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_sub_f32x4(self, a, b, c)
    }
    #[inline(always)]
    fn wasm_fma_f32x4_min_fast(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::min_fast_f32x4(self, a, b)
    }
    #[inline(always)]
    fn wasm_fma_f32x4_max_fast(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::max_fast_f32x4(self, a, b)
    }
    #[inline(always)]
    fn wasm_fma_f32x4_mul_add_lane_relaxed<const LANE: i32>(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        v_in: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
        acc: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_add_lane_f32x4::<LANE>(self, a, v_in, acc)
    }
    #[inline(always)]
    fn wasm_fma_f32x8_mul_add(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_add_f32x8(self, a, b, c)
    }
    #[inline(always)]
    fn wasm_fma_f32x8_mul_sub(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_sub_f32x8(self, a, b, c)
    }
    #[inline(always)]
    fn wasm_fma_f32x8_mul_add_portable(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_add_f32x8(self, a, b, c)
    }
    #[inline(always)]
    fn wasm_fma_f32x8_mul_sub_portable(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_sub_f32x8(self, a, b, c)
    }
    #[inline(always)]
    fn wasm_fma_f32x8_min_fast(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::min_fast_f32x8(self, a, b)
    }
    #[inline(always)]
    fn wasm_fma_f32x8_max_fast(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::max_fast_f32x8(self, a, b)
    }
    #[inline(always)]
    fn wasm_fma_f32x8_mul_add_lane_relaxed<const LANE: i32>(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
        v_in: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
        acc: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_add_lane_f32x8::<LANE>(self, a, v_in, acc)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f32x16_mul_add(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
        c: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_add_f32x16(self, a, b, c)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f32x16_mul_sub(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
        c: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_sub_f32x16(self, a, b, c)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f32x16_mul_add_portable(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
        c: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_add_f32x16(self, a, b, c)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f32x16_mul_sub_portable(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
        c: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_sub_f32x16(self, a, b, c)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f32x16_min_fast(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::min_fast_f32x16(self, a, b)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f32x16_max_fast(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::max_fast_f32x16(self, a, b)
    }
    #[inline(always)]
    fn wasm_fma_f64x2_mul_add(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_add_f64x2(self, a, b, c)
    }
    #[inline(always)]
    fn wasm_fma_f64x2_mul_sub(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_sub_f64x2(self, a, b, c)
    }
    #[inline(always)]
    fn wasm_fma_f64x2_mul_add_portable(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_add_f64x2(self, a, b, c)
    }
    #[inline(always)]
    fn wasm_fma_f64x2_mul_sub_portable(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_sub_f64x2(self, a, b, c)
    }
    #[inline(always)]
    fn wasm_fma_f64x2_min_fast(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::min_fast_f64x2(self, a, b)
    }
    #[inline(always)]
    fn wasm_fma_f64x2_max_fast(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F64x2Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::max_fast_f64x2(self, a, b)
    }
    #[inline(always)]
    fn wasm_fma_f64x4_mul_add(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_add_f64x4(self, a, b, c)
    }
    #[inline(always)]
    fn wasm_fma_f64x4_mul_sub(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_sub_f64x4(self, a, b, c)
    }
    #[inline(always)]
    fn wasm_fma_f64x4_mul_add_portable(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_add_f64x4(self, a, b, c)
    }
    #[inline(always)]
    fn wasm_fma_f64x4_mul_sub_portable(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
        c: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_sub_f64x4(self, a, b, c)
    }
    #[inline(always)]
    fn wasm_fma_f64x4_min_fast(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::min_fast_f64x4(self, a, b)
    }
    #[inline(always)]
    fn wasm_fma_f64x4_max_fast(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::F64x4Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::max_fast_f64x4(self, a, b)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f64x8_mul_add(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
        c: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_add_f64x8(self, a, b, c)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f64x8_mul_sub(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
        c: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_sub_f64x8(self, a, b, c)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f64x8_mul_add_portable(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
        c: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_add_f64x8(self, a, b, c)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f64x8_mul_sub_portable(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
        c: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_sub_f64x8(self, a, b, c)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f64x8_min_fast(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::min_fast_f64x8(self, a, b)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f64x8_max_fast(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::F64x8Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::max_fast_f64x8(self, a, b)
    }
    #[inline(always)]
    fn wasm_fma_f32x4_cvt_convert_f32_to_i32_fast(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x4Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::I32x4Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::convert_f32_to_i32_fast_x4(self, a)
    }
    #[inline(always)]
    fn wasm_fma_f32x8_cvt_convert_f32_to_i32_fast(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::F32x8Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::I32x8Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::convert_f32_to_i32_fast_x8(self, a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_f32x16_cvt_convert_f32_to_i32_fast(
        self,
        a: <Self::W512 as crate::simd::backends::ops::F32x16Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::I32x16Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::convert_f32_to_i32_fast_x16(self, a)
    }
    #[inline(always)]
    fn wasm_fma_i16x8_mul_hrs_q15(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::I16x8Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::I16x8Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::I16x8Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_hrs_q15_x8(self, a, b)
    }
    #[inline(always)]
    fn wasm_fma_i16x16_mul_hrs_q15(
        self,
        a: <Self::Narrow as crate::simd::backends::ops::I16x16Ops>::Repr,
        b: <Self::Narrow as crate::simd::backends::ops::I16x16Ops>::Repr,
    ) -> <Self::Narrow as crate::simd::backends::ops::I16x16Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_hrs_q15_x16(self, a, b)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn wasm_fma_i16x32_mul_hrs_q15(
        self,
        a: <Self::W512 as crate::simd::backends::ops::I16x32Ops>::Repr,
        b: <Self::W512 as crate::simd::backends::ops::I16x32Ops>::Repr,
    ) -> <Self::W512 as crate::simd::backends::ops::I16x32Ops>::Repr {
        crate::simd::impls::wasm128_fma_overrides::mul_hrs_q15_x32(self, a, b)
    }
    #[inline(always)]
    fn mul_add_lane4_relaxed_x8<const LANE: i32>(
        self,
        a: [core::arch::wasm32::v128; 2],
        v_in: core::arch::wasm32::v128,
        acc: [core::arch::wasm32::v128; 2],
    ) -> [core::arch::wasm32::v128; 2] {
        crate::simd::generic::cross_width::wasm_fma_mul_add_lane4_relaxed::<LANE>(
            self, a, v_in, acc,
        )
    }
    #[inline(always)]
    fn f32x4_to_u32_fast(self, a: core::arch::wasm32::v128) -> core::arch::wasm32::v128 {
        crate::simd::generic::fast_ops::wasm_fma_f32x4_to_u32_fast(self, a)
    }
    #[inline(always)]
    fn f32x8_to_u32_fast(self, a: [core::arch::wasm32::v128; 2]) -> [core::arch::wasm32::v128; 2] {
        crate::simd::generic::fast_ops::wasm_fma_f32x8_to_u32_fast(self, a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_to_u32_fast(self, a: [core::arch::wasm32::v128; 4]) -> [core::arch::wasm32::v128; 4] {
        crate::simd::generic::fast_ops::wasm_fma_f32x16_to_u32_fast(self, a)
    }
    #[inline(always)]
    fn i32x4_dot_i8_i7_add(
        self,
        acc: core::arch::wasm32::v128,
        a: core::arch::wasm32::v128,
        b: core::arch::wasm32::v128,
    ) -> core::arch::wasm32::v128 {
        crate::simd::generic::quant_ops::wasm_fma_i32x4_dot_i8(self, acc, a, b)
    }
    #[inline(always)]
    fn i32x8_dot_i8_i7_add(
        self,
        acc: [core::arch::wasm32::v128; 2],
        a: [core::arch::wasm32::v128; 2],
        b: [core::arch::wasm32::v128; 2],
    ) -> [core::arch::wasm32::v128; 2] {
        crate::simd::generic::quant_ops::wasm_fma_i32x8_dot_i8(self, acc, a, b)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn i32x16_dot_i8_i7_add(
        self,
        acc: [core::arch::wasm32::v128; 4],
        a: [core::arch::wasm32::v128; 4],
        b: [core::arch::wasm32::v128; 4],
    ) -> [core::arch::wasm32::v128; 4] {
        crate::simd::generic::quant_ops::wasm_fma_i32x16_dot_i8(self, acc, a, b)
    }

    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::f16_to_f32_slice_soft(self.narrow(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::f32_to_f16_slice_soft(self.narrow(), input, output);
    }
}

#[cfg(all(
    target_arch = "wasm32",
    not(all(target_feature = "simd128", target_feature = "relaxed-simd"))
))]
impl Sealed for archmage::Wasm128RelaxedFmaToken {
    type Narrow = archmage::Wasm128Token;
    #[cfg(feature = "w512")]
    type W512 = archmage::Wasm128Token;

    #[inline(always)]
    fn narrow(self) -> Self::Narrow {
        self.wasm128()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn w512(self) -> Self::W512 {
        self.wasm128()
    }

    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2],
        hi: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4] {
        [lo[0], lo[1], hi[0], hi[1]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(
        self,
        wide: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2] {
        [wide[0], wide[1]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(
        self,
        wide: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2] {
        [wide[2], wide[3]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(
        self,
        a: [core::arch::wasm32::v128; 4],
    ) -> [core::arch::wasm32::v128; 2] {
        crate::simd::impls::wasm128::wasm_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: [core::arch::wasm32::v128; 4],
    ) -> [core::arch::wasm32::v128; 2] {
        crate::simd::impls::wasm128::wasm_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(
        self,
        a: [core::arch::wasm32::v128; 4],
    ) -> [core::arch::wasm32::v128; 2] {
        crate::simd::impls::wasm128::wasm_f64x8_to_i32_round(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(
        self,
        a: [core::arch::wasm32::v128; 2],
    ) -> [core::arch::wasm32::v128; 4] {
        crate::simd::impls::wasm128::wasm_f64x8_from_i32(self.narrow(), a)
    }

    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::f16_to_f32_slice_soft(self.narrow(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::f32_to_f16_slice_soft(self.narrow(), input, output);
    }
}

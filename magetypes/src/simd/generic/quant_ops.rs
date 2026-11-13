//! Quantized neural and fixed-point DSP operations: Q15 rounding multiply
//! (`mul_hrs_q15`) and portable 4-way `i8 * i7 -> i32` dot-product accumulate
//! (`dot_i8_i7_add`).
//!
//! ## Fixed-point Q15 rounding multiply (`mul_hrs_q15`)
//!
//! Computes `((a as i32 * b as i32 + 0x4000) >> 15)` per lane in a single hardware
//! instruction across x86 (`pmulhrsw` / `vpmulhrsw`), ARM64 NEON (`sqrdmulh`),
//! WebAssembly (`i16x8.q15mulr_sat_s`), and `Wasm128RelaxedFmaToken`
//! (`i16x8.relaxed_q15mulr_s`):
//! - For every input pair **except** `(i16::MIN, i16::MIN)` (`-32768 * -32768`),
//!   all backends are bit-identical.
//! - At `(i16::MIN, i16::MIN)`, the mathematical result `+32768` overflows `i16`:
//!   x86 `PMULHRSW` and `ScalarToken` wrap to `i16::MIN` (`-32768`), ARM64
//!   `SQRDMULH` and `Wasm128Token` `q15mulr_sat` saturate to `i16::MAX` (`+32767`),
//!   and `Wasm128RelaxedFmaToken` returns either `i16::MIN` or `i16::MAX` depending
//!   on the host architecture. Clamp weights or activations away from `-32768`
//!   (e.g. symmetric `[-32767, +32767]`) when cross-ISA bit-identity at the rail is
//!   required.
//!
//! ## Portable 4-way INT8 dot-product accumulate (`dot_i8_i7_add`)
//!
//! Accumulates groups of 4 `i8 * i7` byte products into each `i32` lane of `self`:
//! `self[lane] + sum_{k=0..4} (a[4*lane + k] as i32) * (b[4*lane + k] as i32)`.
//!
//! Callers must ensure all lanes of `b` lie in `0..=127` (`i7`, non-negative signed
//! bytes — standard for unsigned/ReLU activations shifted to `[0, 127]` or 7-bit
//! quantized activations). Under that invariant:
//! - **x86 (`X64V3Token`, `X64V4Token`, `Avx512Fp16Token`):** `vpmaddubsw(b, a)` + `vpmaddwd(1)` + `vpaddd` (cannot saturate `i16` because `2 * 127 * -128 = -32512 >= -32768`).
//! - **x86 VNNI (`X64V4xToken`):** single-instruction `vpdpbusd` (`_mm*_dpbusd_epi32(acc, b, a)`).
//! - **ARM64 (`NeonToken`):** `vmull_s8` + `vpaddlq_s16` + `vpaddq_s32` + `vaddq_s32` (for single-instruction `SDOT` under `Arm64V2Token`, see [`i32x4::dot_i8`]).
//! - **WASM (`Wasm128Token`):** `i16x8.extmul_{low,high}_i8x16_s` + `i32x4.extadd_pairwise_i16x8_s` + `i32x4.shuffle` + `i32x4.add`.
//! - **WASM (`Wasm128RelaxedFmaToken`):** single-instruction `i32x4.relaxed_dot_i8x16_i7x16_add`.
//! - **Scalar (`ScalarToken`):** exact 4-way `i32` wrapping accumulate.
//!
//! Every backend produces **bit-identical** `i32` results whenever all bytes of `b`
//! are in `0..=127`!

#![forbid(unsafe_code)]

use archmage::SimdToken;

use crate::simd::backends::sealed::Sealed;
use crate::simd::backends::{
    I8x16Backend, I8x32Backend, I16x8Backend, I16x16Backend, I32x4Backend, I32x8Backend,
};
#[cfg(feature = "w512")]
use crate::simd::backends::{I8x64Backend, I16x32Backend, I32x16Backend};
use crate::simd::generic::{i8x16, i8x32, i16x8, i16x16, i32x4, i32x8};
#[cfg(feature = "w512")]
use crate::simd::generic::{i8x64, i16x32, i32x16};

// ============================================================================
// mul_hrs_q15
// ============================================================================

impl<T: I16x8Backend> i16x8<T> {
    /// Fixed-point Q15 rounding multiply: `((self * other + 0x4000) >> 15)` per lane.
    ///
    /// Bit-identical across all backends for every input pair except
    /// `(i16::MIN, i16::MIN)`, where x86/Scalar wrap to `i16::MIN` (`-32768`) and
    /// NEON/WASM saturate (or relaxed-select) to `i16::MAX` (`+32767`).
    #[inline(always)]
    pub fn mul_hrs_q15(self, other: Self) -> Self {
        Self(T::mul_hrs_q15(self.1, self.0, other.0), self.1)
    }
}

impl<T: I16x16Backend> i16x16<T> {
    /// Fixed-point Q15 rounding multiply: `((self * other + 0x4000) >> 15)` per lane.
    /// See [`i16x8::mul_hrs_q15`].
    #[inline(always)]
    pub fn mul_hrs_q15(self, other: Self) -> Self {
        Self(T::mul_hrs_q15(self.1, self.0, other.0), self.1)
    }
}

#[cfg(feature = "w512")]
impl<T: I16x32Backend> i16x32<T> {
    /// Fixed-point Q15 rounding multiply: `((self * other + 0x4000) >> 15)` per lane.
    /// See [`i16x8::mul_hrs_q15`].
    #[inline(always)]
    pub fn mul_hrs_q15(self, other: Self) -> Self {
        Self(T::mul_hrs_q15(self.1, self.0, other.0), self.1)
    }
}

// ============================================================================
// I32x{4,8,16}DotI8 & dot_i8_i7_add
// ============================================================================

/// Portable 4-way `i8 * i7 -> i32` dot-product accumulate for 128-bit vectors.
///
/// Sealed trait implemented by every magetypes backend token that supports
/// [`I32x4Backend`] and [`I8x16Backend`].
pub trait I32x4DotI8: I32x4Backend + I8x16Backend + SimdToken + Sealed + Copy + 'static {
    /// Accumulate 4-way signed `a` (`i8`) × non-negative `b` (`0..=127`, `i7`)
    /// products into `acc`.
    fn dot_i8_i7_add(
        self,
        acc: <Self as I32x4Backend>::Repr,
        a: <Self as I8x16Backend>::Repr,
        b: <Self as I8x16Backend>::Repr,
    ) -> <Self as I32x4Backend>::Repr;
}

/// Portable 4-way `i8 * i7 -> i32` dot-product accumulate for 256-bit vectors.
///
/// Sealed trait implemented by every magetypes backend token that supports
/// [`I32x8Backend`] and [`I8x32Backend`].
pub trait I32x8DotI8: I32x8Backend + I8x32Backend + SimdToken + Sealed + Copy + 'static {
    /// Accumulate 4-way signed `a` (`i8`) × non-negative `b` (`0..=127`, `i7`)
    /// products into `acc`.
    fn dot_i8_i7_add(
        self,
        acc: <Self as I32x8Backend>::Repr,
        a: <Self as I8x32Backend>::Repr,
        b: <Self as I8x32Backend>::Repr,
    ) -> <Self as I32x8Backend>::Repr;
}

/// Portable 4-way `i8 * i7 -> i32` dot-product accumulate for 512-bit vectors.
///
/// Sealed trait implemented by every magetypes backend token that supports
/// [`I32x16Backend`] and [`I8x64Backend`].
#[cfg(feature = "w512")]
pub trait I32x16DotI8: I32x16Backend + I8x64Backend + SimdToken + Sealed + Copy + 'static {
    /// Accumulate 4-way signed `a` (`i8`) × non-negative `b` (`0..=127`, `i7`)
    /// products into `acc`.
    fn dot_i8_i7_add(
        self,
        acc: <Self as I32x16Backend>::Repr,
        a: <Self as I8x64Backend>::Repr,
        b: <Self as I8x64Backend>::Repr,
    ) -> <Self as I32x16Backend>::Repr;
}

impl I32x4DotI8 for archmage::ScalarToken {
    #[inline(always)]
    fn dot_i8_i7_add(self, acc: [i32; 4], a: [i8; 16], b: [i8; 16]) -> [i32; 4] {
        let _ = self;
        core::array::from_fn(|lane| {
            let base = lane * 4;
            let sum = (a[base] as i32) * (b[base] as i32)
                + (a[base + 1] as i32) * (b[base + 1] as i32)
                + (a[base + 2] as i32) * (b[base + 2] as i32)
                + (a[base + 3] as i32) * (b[base + 3] as i32);
            acc[lane].wrapping_add(sum)
        })
    }
}

impl I32x8DotI8 for archmage::ScalarToken {
    #[inline(always)]
    fn dot_i8_i7_add(self, acc: [i32; 8], a: [i8; 32], b: [i8; 32]) -> [i32; 8] {
        let _ = self;
        core::array::from_fn(|lane| {
            let base = lane * 4;
            let sum = (a[base] as i32) * (b[base] as i32)
                + (a[base + 1] as i32) * (b[base + 1] as i32)
                + (a[base + 2] as i32) * (b[base + 2] as i32)
                + (a[base + 3] as i32) * (b[base + 3] as i32);
            acc[lane].wrapping_add(sum)
        })
    }
}

#[cfg(feature = "w512")]
impl I32x16DotI8 for archmage::ScalarToken {
    #[inline(always)]
    fn dot_i8_i7_add(self, acc: [i32; 16], a: [i8; 64], b: [i8; 64]) -> [i32; 16] {
        let _ = self;
        core::array::from_fn(|lane| {
            let base = lane * 4;
            let sum = (a[base] as i32) * (b[base] as i32)
                + (a[base + 1] as i32) * (b[base + 1] as i32)
                + (a[base + 2] as i32) * (b[base + 2] as i32)
                + (a[base + 3] as i32) * (b[base + 3] as i32);
            acc[lane].wrapping_add(sum)
        })
    }
}

#[cfg(target_arch = "aarch64")]
impl I32x4DotI8 for archmage::NeonToken {
    #[archmage::arcane(suppress_const_test, _self = archmage::NeonToken)]
    fn dot_i8_i7_add(
        self,
        acc: core::arch::aarch64::int32x4_t,
        a: core::arch::aarch64::int8x16_t,
        b: core::arch::aarch64::int8x16_t,
    ) -> core::arch::aarch64::int32x4_t {
        use core::arch::aarch64::{
            vaddq_s32, vget_low_s8, vmull_high_s8, vmull_s8, vpaddlq_s16, vpaddq_s32,
        };
        let lo = vpaddlq_s16(vmull_s8(vget_low_s8(a), vget_low_s8(b)));
        let hi = vpaddlq_s16(vmull_high_s8(a, b));
        vaddq_s32(acc, vpaddq_s32(lo, hi))
    }
}

#[cfg(target_arch = "aarch64")]
impl I32x8DotI8 for archmage::NeonToken {
    #[inline(always)]
    fn dot_i8_i7_add(
        self,
        acc: [core::arch::aarch64::int32x4_t; 2],
        a: [core::arch::aarch64::int8x16_t; 2],
        b: [core::arch::aarch64::int8x16_t; 2],
    ) -> [core::arch::aarch64::int32x4_t; 2] {
        [
            <Self as I32x4DotI8>::dot_i8_i7_add(self, acc[0], a[0], b[0]),
            <Self as I32x4DotI8>::dot_i8_i7_add(self, acc[1], a[1], b[1]),
        ]
    }
}

#[cfg(all(target_arch = "aarch64", feature = "w512"))]
impl I32x16DotI8 for archmage::NeonToken {
    #[inline(always)]
    fn dot_i8_i7_add(
        self,
        acc: [core::arch::aarch64::int32x4_t; 4],
        a: [core::arch::aarch64::int8x16_t; 4],
        b: [core::arch::aarch64::int8x16_t; 4],
    ) -> [core::arch::aarch64::int32x4_t; 4] {
        core::array::from_fn(|i| <Self as I32x4DotI8>::dot_i8_i7_add(self, acc[i], a[i], b[i]))
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
impl I32x4DotI8 for archmage::Wasm128Token {
    #[inline(always)]
    fn dot_i8_i7_add(
        self,
        acc: core::arch::wasm32::v128,
        a: core::arch::wasm32::v128,
        b: core::arch::wasm32::v128,
    ) -> core::arch::wasm32::v128 {
        use core::arch::wasm32::{
            i16x8_extmul_high_i8x16, i16x8_extmul_low_i8x16, i32x4_add,
            i32x4_extadd_pairwise_i16x8, i32x4_shuffle,
        };
        let _ = self;
        let lo32 = i32x4_extadd_pairwise_i16x8(i16x8_extmul_low_i8x16(a, b));
        let hi32 = i32x4_extadd_pairwise_i16x8(i16x8_extmul_high_i8x16(a, b));
        let evens = i32x4_shuffle::<0, 2, 4, 6>(lo32, hi32);
        let odds = i32x4_shuffle::<1, 3, 5, 7>(lo32, hi32);
        i32x4_add(acc, i32x4_add(evens, odds))
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
impl I32x8DotI8 for archmage::Wasm128Token {
    #[inline(always)]
    fn dot_i8_i7_add(
        self,
        acc: [core::arch::wasm32::v128; 2],
        a: [core::arch::wasm32::v128; 2],
        b: [core::arch::wasm32::v128; 2],
    ) -> [core::arch::wasm32::v128; 2] {
        [
            <Self as I32x4DotI8>::dot_i8_i7_add(self, acc[0], a[0], b[0]),
            <Self as I32x4DotI8>::dot_i8_i7_add(self, acc[1], a[1], b[1]),
        ]
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "w512"))]
impl I32x16DotI8 for archmage::Wasm128Token {
    #[inline(always)]
    fn dot_i8_i7_add(
        self,
        acc: [core::arch::wasm32::v128; 4],
        a: [core::arch::wasm32::v128; 4],
        b: [core::arch::wasm32::v128; 4],
    ) -> [core::arch::wasm32::v128; 4] {
        core::array::from_fn(|i| <Self as I32x4DotI8>::dot_i8_i7_add(self, acc[i], a[i], b[i]))
    }
}

#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    target_feature = "relaxed-simd"
))]
impl I32x4DotI8 for archmage::Wasm128RelaxedFmaToken {
    #[archmage::arcane(suppress_const_test, _self = archmage::Wasm128RelaxedFmaToken)]
    fn dot_i8_i7_add(
        self,
        acc: core::arch::wasm32::v128,
        a: core::arch::wasm32::v128,
        b: core::arch::wasm32::v128,
    ) -> core::arch::wasm32::v128 {
        core::arch::wasm32::i32x4_relaxed_dot_i8x16_i7x16_add(a, b, acc)
    }
}

#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    target_feature = "relaxed-simd"
))]
impl I32x8DotI8 for archmage::Wasm128RelaxedFmaToken {
    #[inline(always)]
    fn dot_i8_i7_add(
        self,
        acc: [core::arch::wasm32::v128; 2],
        a: [core::arch::wasm32::v128; 2],
        b: [core::arch::wasm32::v128; 2],
    ) -> [core::arch::wasm32::v128; 2] {
        [
            <Self as I32x4DotI8>::dot_i8_i7_add(self, acc[0], a[0], b[0]),
            <Self as I32x4DotI8>::dot_i8_i7_add(self, acc[1], a[1], b[1]),
        ]
    }
}

#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    target_feature = "relaxed-simd",
    feature = "w512"
))]
impl I32x16DotI8 for archmage::Wasm128RelaxedFmaToken {
    #[inline(always)]
    fn dot_i8_i7_add(
        self,
        acc: [core::arch::wasm32::v128; 4],
        a: [core::arch::wasm32::v128; 4],
        b: [core::arch::wasm32::v128; 4],
    ) -> [core::arch::wasm32::v128; 4] {
        core::array::from_fn(|i| <Self as I32x4DotI8>::dot_i8_i7_add(self, acc[i], a[i], b[i]))
    }
}

#[cfg(target_arch = "x86_64")]
impl I32x4DotI8 for archmage::X64V3Token {
    #[archmage::arcane(suppress_const_test, _self = archmage::X64V3Token)]
    fn dot_i8_i7_add(
        self,
        acc: core::arch::x86_64::__m128i,
        a: core::arch::x86_64::__m128i,
        b: core::arch::x86_64::__m128i,
    ) -> core::arch::x86_64::__m128i {
        use core::arch::x86_64::{
            _mm_add_epi32, _mm_madd_epi16, _mm_maddubs_epi16, _mm_set1_epi16,
        };
        let pairs = _mm_maddubs_epi16(b, a);
        let quads = _mm_madd_epi16(pairs, _mm_set1_epi16(1));
        _mm_add_epi32(acc, quads)
    }
}

#[cfg(target_arch = "x86_64")]
impl I32x8DotI8 for archmage::X64V3Token {
    #[archmage::arcane(suppress_const_test, _self = archmage::X64V3Token)]
    fn dot_i8_i7_add(
        self,
        acc: core::arch::x86_64::__m256i,
        a: core::arch::x86_64::__m256i,
        b: core::arch::x86_64::__m256i,
    ) -> core::arch::x86_64::__m256i {
        use core::arch::x86_64::{
            _mm256_add_epi32, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi16,
        };
        let pairs = _mm256_maddubs_epi16(b, a);
        let quads = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
        _mm256_add_epi32(acc, quads)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "w512"))]
impl I32x16DotI8 for archmage::X64V3Token {
    #[inline(always)]
    fn dot_i8_i7_add(
        self,
        acc: [core::arch::x86_64::__m256i; 2],
        a: [core::arch::x86_64::__m256i; 2],
        b: [core::arch::x86_64::__m256i; 2],
    ) -> [core::arch::x86_64::__m256i; 2] {
        [
            <Self as I32x8DotI8>::dot_i8_i7_add(self, acc[0], a[0], b[0]),
            <Self as I32x8DotI8>::dot_i8_i7_add(self, acc[1], a[1], b[1]),
        ]
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl I32x4DotI8 for archmage::X64V4Token {
    #[inline(always)]
    fn dot_i8_i7_add(
        self,
        acc: core::arch::x86_64::__m128i,
        a: core::arch::x86_64::__m128i,
        b: core::arch::x86_64::__m128i,
    ) -> core::arch::x86_64::__m128i {
        <archmage::X64V3Token as I32x4DotI8>::dot_i8_i7_add(self.v3(), acc, a, b)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl I32x4DotI8 for archmage::Avx512Fp16Token {
    #[inline(always)]
    fn dot_i8_i7_add(
        self,
        acc: core::arch::x86_64::__m128i,
        a: core::arch::x86_64::__m128i,
        b: core::arch::x86_64::__m128i,
    ) -> core::arch::x86_64::__m128i {
        <archmage::X64V3Token as I32x4DotI8>::dot_i8_i7_add(self.v3(), acc, a, b)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl I32x4DotI8 for archmage::X64V4xToken {
    #[archmage::arcane(suppress_const_test, _self = archmage::X64V4xToken)]
    fn dot_i8_i7_add(
        self,
        acc: core::arch::x86_64::__m128i,
        a: core::arch::x86_64::__m128i,
        b: core::arch::x86_64::__m128i,
    ) -> core::arch::x86_64::__m128i {
        core::arch::x86_64::_mm_dpbusd_epi32(acc, b, a)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl I32x8DotI8 for archmage::X64V4Token {
    #[inline(always)]
    fn dot_i8_i7_add(
        self,
        acc: core::arch::x86_64::__m256i,
        a: core::arch::x86_64::__m256i,
        b: core::arch::x86_64::__m256i,
    ) -> core::arch::x86_64::__m256i {
        <archmage::X64V3Token as I32x8DotI8>::dot_i8_i7_add(self.v3(), acc, a, b)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl I32x8DotI8 for archmage::Avx512Fp16Token {
    #[inline(always)]
    fn dot_i8_i7_add(
        self,
        acc: core::arch::x86_64::__m256i,
        a: core::arch::x86_64::__m256i,
        b: core::arch::x86_64::__m256i,
    ) -> core::arch::x86_64::__m256i {
        <archmage::X64V3Token as I32x8DotI8>::dot_i8_i7_add(self.v3(), acc, a, b)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl I32x8DotI8 for archmage::X64V4xToken {
    #[archmage::arcane(suppress_const_test, _self = archmage::X64V4xToken)]
    fn dot_i8_i7_add(
        self,
        acc: core::arch::x86_64::__m256i,
        a: core::arch::x86_64::__m256i,
        b: core::arch::x86_64::__m256i,
    ) -> core::arch::x86_64::__m256i {
        core::arch::x86_64::_mm256_dpbusd_epi32(acc, b, a)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "w512"))]
impl I32x16DotI8 for archmage::X64V4Token {
    #[archmage::arcane(suppress_const_test, _self = archmage::X64V4Token)]
    fn dot_i8_i7_add(
        self,
        acc: core::arch::x86_64::__m512i,
        a: core::arch::x86_64::__m512i,
        b: core::arch::x86_64::__m512i,
    ) -> core::arch::x86_64::__m512i {
        use core::arch::x86_64::{
            _mm512_add_epi32, _mm512_madd_epi16, _mm512_maddubs_epi16, _mm512_set1_epi16,
        };
        let pairs = _mm512_maddubs_epi16(b, a);
        let quads = _mm512_madd_epi16(pairs, _mm512_set1_epi16(1));
        _mm512_add_epi32(acc, quads)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "w512"))]
impl I32x16DotI8 for archmage::X64V4xToken {
    #[archmage::arcane(suppress_const_test, _self = archmage::X64V4xToken)]
    fn dot_i8_i7_add(
        self,
        acc: core::arch::x86_64::__m512i,
        a: core::arch::x86_64::__m512i,
        b: core::arch::x86_64::__m512i,
    ) -> core::arch::x86_64::__m512i {
        core::arch::x86_64::_mm512_dpbusd_epi32(acc, b, a)
    }
}

impl<T: I32x4DotI8> i32x4<T> {
    /// Portable 4-way `i8 * i7` dot-product accumulate into `self`:
    /// `self[lane] + sum_{k=0..4} (a[4*lane + k] as i32) * (b[4*lane + k] as i32)`.
    ///
    /// All bytes of `b` must be in `0..=127` (`i7`, non-negative signed bytes) so
    /// unsigned-by-signed hardware instructions (`vpmaddubsw`, `vpdpbusd`,
    /// `i32x4.relaxed_dot_i8x16_i7x16_add`) and signed widening instructions
    /// (`vmull_s8`, `i16x8.extmul`) produce bit-identical `i32` sums.
    #[inline(always)]
    pub fn dot_i8_i7_add(self, a: i8x16<T>, b: i8x16<T>) -> Self {
        Self(T::dot_i8_i7_add(self.1, self.0, a.0, b.0), self.1)
    }
}

impl<T: I32x8DotI8> i32x8<T> {
    /// Portable 4-way `i8 * i7` dot-product accumulate into `self`.
    /// See [`i32x4::dot_i8_i7_add`].
    #[inline(always)]
    pub fn dot_i8_i7_add(self, a: i8x32<T>, b: i8x32<T>) -> Self {
        Self(T::dot_i8_i7_add(self.1, self.0, a.0, b.0), self.1)
    }
}

#[cfg(feature = "w512")]
impl<T: I32x16DotI8> i32x16<T> {
    /// Portable 4-way `i8 * i7` dot-product accumulate into `self`.
    /// See [`i32x4::dot_i8_i7_add`].
    #[inline(always)]
    pub fn dot_i8_i7_add(self, a: i8x64<T>, b: i8x64<T>) -> Self {
        Self(T::dot_i8_i7_add(self.1, self.0, a.0, b.0), self.1)
    }
}

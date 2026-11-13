//! Token-gated sub-tier operations on base representation vectors.
//!
//! Sub-tier proof tokens (`X64V3VnniToken`, `Arm64V2Token`, `Wasm128RelaxedToken`,
//! etc.) share the exact hardware register representation of their base width
//! token (`X64V3Token`, `NeonToken`, `Wasm128Token`). Rather than duplicating
//! 40+ backend traits per sub-tier, `#[attune(..., define(...))]` aliases generic
//! vector names to the base representation token at zero KLOC / `.rmeta` cost,
//! while sub-tier-specific instructions live here as explicit, token-gated
//! inherent methods on those base vectors.

#![forbid(unsafe_code)]

#[cfg(target_arch = "x86_64")]
mod x86_vnni {
    use super::super::{i8x16, i8x32, i16x8, i16x16, i32x4, i32x8, u8x16, u8x32};
    use archmage::{X64V3Token, X64V3VnniToken};
    use core::arch::x86_64::{
        _mm_dpbusd_avx_epi32, _mm_dpbusds_avx_epi32, _mm_dpwssd_avx_epi32, _mm_dpwssds_avx_epi32,
        _mm256_dpbusd_avx_epi32, _mm256_dpbusds_avx_epi32, _mm256_dpwssd_avx_epi32,
        _mm256_dpwssds_avx_epi32,
    };

    impl i32x4<X64V3Token> {
        /// Multiply groups of 4 unsigned `u8` lanes in `a` by signed `i8` lanes in `b`
        /// and wrapping-accumulate each 4-product sum into the corresponding `i32` lane
        /// of `self` (VEX `VPDPBUSD`, `_mm_dpbusd_avx_epi32`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn mul_add_u8_i8_vnni(
            self,
            _token: X64V3VnniToken,
            a: u8x16<X64V3Token>,
            b: i8x16<X64V3Token>,
        ) -> Self {
            Self(_mm_dpbusd_avx_epi32(self.0, a.0, b.0), self.1)
        }

        /// Signed-saturating `u8 * i8` 4-way dot-product accumulate into `self`
        /// (VEX `VPDPBUSDS`, `_mm_dpbusds_avx_epi32`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn mul_add_u8_i8_sat_vnni(
            self,
            _token: X64V3VnniToken,
            a: u8x16<X64V3Token>,
            b: i8x16<X64V3Token>,
        ) -> Self {
            Self(_mm_dpbusds_avx_epi32(self.0, a.0, b.0), self.1)
        }

        /// Multiply pairs of signed `i16` lanes in `a` and `b` and wrapping-accumulate
        /// each 2-product sum into the corresponding `i32` lane of `self`
        /// (VEX `VPDPWSSD`, `_mm_dpwssd_avx_epi32`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn mul_add_i16_vnni(
            self,
            _token: X64V3VnniToken,
            a: i16x8<X64V3Token>,
            b: i16x8<X64V3Token>,
        ) -> Self {
            Self(_mm_dpwssd_avx_epi32(self.0, a.0, b.0), self.1)
        }

        /// Signed-saturating `i16 * i16` 2-way dot-product accumulate into `self`
        /// (VEX `VPDPWSSDS`, `_mm_dpwssds_avx_epi32`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn mul_add_i16_sat_vnni(
            self,
            _token: X64V3VnniToken,
            a: i16x8<X64V3Token>,
            b: i16x8<X64V3Token>,
        ) -> Self {
            Self(_mm_dpwssds_avx_epi32(self.0, a.0, b.0), self.1)
        }
    }

    impl i32x8<X64V3Token> {
        /// Multiply groups of 4 unsigned `u8` lanes in `a` by signed `i8` lanes in `b`
        /// and wrapping-accumulate each 4-product sum into the corresponding `i32` lane
        /// of `self` (VEX `VPDPBUSD`, `_mm256_dpbusd_avx_epi32`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn mul_add_u8_i8_vnni(
            self,
            _token: X64V3VnniToken,
            a: u8x32<X64V3Token>,
            b: i8x32<X64V3Token>,
        ) -> Self {
            Self(_mm256_dpbusd_avx_epi32(self.0, a.0, b.0), self.1)
        }

        /// Signed-saturating `u8 * i8` 4-way dot-product accumulate into `self`
        /// (VEX `VPDPBUSDS`, `_mm256_dpbusds_avx_epi32`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn mul_add_u8_i8_sat_vnni(
            self,
            _token: X64V3VnniToken,
            a: u8x32<X64V3Token>,
            b: i8x32<X64V3Token>,
        ) -> Self {
            Self(_mm256_dpbusds_avx_epi32(self.0, a.0, b.0), self.1)
        }

        /// Multiply pairs of signed `i16` lanes in `a` and `b` and wrapping-accumulate
        /// each 2-product sum into the corresponding `i32` lane of `self`
        /// (VEX `VPDPWSSD`, `_mm256_dpwssd_avx_epi32`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn mul_add_i16_vnni(
            self,
            _token: X64V3VnniToken,
            a: i16x16<X64V3Token>,
            b: i16x16<X64V3Token>,
        ) -> Self {
            Self(_mm256_dpwssd_avx_epi32(self.0, a.0, b.0), self.1)
        }

        /// Signed-saturating `i16 * i16` 2-way dot-product accumulate into `self`
        /// (VEX `VPDPWSSDS`, `_mm256_dpwssds_avx_epi32`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn mul_add_i16_sat_vnni(
            self,
            _token: X64V3VnniToken,
            a: i16x16<X64V3Token>,
            b: i16x16<X64V3Token>,
        ) -> Self {
            Self(_mm256_dpwssds_avx_epi32(self.0, a.0, b.0), self.1)
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod arm_v2 {
    use super::super::{i8x16, i16x8, i32x4, u8x16, u32x4};
    use archmage::{Arm64V2Token, NeonToken};
    use core::arch::aarch64::{
        vaddq_s32, vaddq_u32, vget_low_s8, vget_low_u8, vmull_high_s8, vmull_high_u8, vmull_s8,
        vmull_u8, vpaddlq_s16, vpaddlq_u16, vpaddq_s32, vpaddq_u32, vqrdmlahq_s16, vqrdmlshq_s16,
    };

    impl i32x4<NeonToken> {
        /// Signed 4-way `i8 * i8` dot-product accumulate into `self` under
        /// `Arm64V2Token`'s `+dotprod` target-feature context (MSRV 1.89-compatible;
        /// `stdarch_neon_dotprod` `vdotq_s32` requires Rust 1.98+).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn dot_i8(
            self,
            _token: Arm64V2Token,
            a: i8x16<NeonToken>,
            b: i8x16<NeonToken>,
        ) -> Self {
            let lo = vpaddlq_s16(vmull_s8(vget_low_s8(a.0), vget_low_s8(b.0)));
            let hi = vpaddlq_s16(vmull_high_s8(a.0, b.0));
            Self(vaddq_s32(self.0, vpaddq_s32(lo, hi)), self.1)
        }
    }

    impl u32x4<NeonToken> {
        /// Unsigned 4-way `u8 * u8` dot-product accumulate into `self` under
        /// `Arm64V2Token`'s `+dotprod` target-feature context (MSRV 1.89-compatible;
        /// `stdarch_neon_dotprod` `vdotq_u32` requires Rust 1.98+).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn dot_u8(
            self,
            _token: Arm64V2Token,
            a: u8x16<NeonToken>,
            b: u8x16<NeonToken>,
        ) -> Self {
            let lo = vpaddlq_u16(vmull_u8(vget_low_u8(a.0), vget_low_u8(b.0)));
            let hi = vpaddlq_u16(vmull_high_u8(a.0, b.0));
            Self(vaddq_u32(self.0, vpaddq_u32(lo, hi)), self.1)
        }
    }

    impl i16x8<NeonToken> {
        /// Signed saturating rounding doubling multiply-accumulate high half
        /// (ARMv8.1-A RDMA `SQRDMLAH`, `vqrdmlahq_s16`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn qrdmlah(self, _token: Arm64V2Token, a: Self, b: Self) -> Self {
            Self(vqrdmlahq_s16(self.0, a.0, b.0), self.1)
        }

        /// Signed saturating rounding doubling multiply-subtract high half
        /// (ARMv8.1-A RDMA `SQRDMLSH`, `vqrdmlshq_s16`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn qrdmlsh(self, _token: Arm64V2Token, a: Self, b: Self) -> Self {
            Self(vqrdmlshq_s16(self.0, a.0, b.0), self.1)
        }
    }
}

#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    target_feature = "relaxed-simd"
))]
mod wasm_relaxed {
    use super::super::{f32x4, f64x2, i8x16, i32x4};
    use archmage::{Wasm128RelaxedToken, Wasm128Token};
    use core::arch::wasm32::{
        f32x4_relaxed_max, f32x4_relaxed_min, f64x2_relaxed_max, f64x2_relaxed_min,
        i32x4_relaxed_dot_i8x16_i7x16_add,
    };

    impl f32x4<Wasm128Token> {
        /// Lane-wise relaxed minimum (`f32x4.relaxed_min`; exact behavior on
        /// NaN or `-0.0`/`+0.0` is implementation-defined).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn min_relaxed(self, _token: Wasm128RelaxedToken, other: Self) -> Self {
            Self(f32x4_relaxed_min(self.0, other.0), self.1)
        }

        /// Lane-wise relaxed maximum (`f32x4.relaxed_max`; exact behavior on
        /// NaN or `-0.0`/`+0.0` is implementation-defined).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn max_relaxed(self, _token: Wasm128RelaxedToken, other: Self) -> Self {
            Self(f32x4_relaxed_max(self.0, other.0), self.1)
        }
    }

    impl f64x2<Wasm128Token> {
        /// Lane-wise relaxed minimum (`f64x2.relaxed_min`; exact behavior on
        /// NaN or `-0.0`/`+0.0` is implementation-defined).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn min_relaxed(self, _token: Wasm128RelaxedToken, other: Self) -> Self {
            Self(f64x2_relaxed_min(self.0, other.0), self.1)
        }

        /// Lane-wise relaxed maximum (`f64x2.relaxed_max`; exact behavior on
        /// NaN or `-0.0`/`+0.0` is implementation-defined).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn max_relaxed(self, _token: Wasm128RelaxedToken, other: Self) -> Self {
            Self(f64x2_relaxed_max(self.0, other.0), self.1)
        }
    }

    impl i32x4<Wasm128Token> {
        /// Relaxed 4-way `i8 * i7` dot-product accumulate into `self`
        /// (`i32x4.relaxed_dot_i8x16_i7x16_add`).
        ///
        /// Bytes of `b` must be in `0..=127` (`i7`) so signed and unsigned
        /// hardware interpretations agree.
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn dot_i8_i7_add_relaxed(
            self,
            _token: Wasm128RelaxedToken,
            a: i8x16<Wasm128Token>,
            b: i8x16<Wasm128Token>,
        ) -> Self {
            Self(i32x4_relaxed_dot_i8x16_i7x16_add(a.0, b.0, self.0), self.1)
        }
    }
}

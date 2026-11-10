//! Scalar CORE-MATH arithmetic strategy on caller-provided proofs.
use crate::core_math::PowMath;
use archmage::prelude::*;
impl PowMath for ScalarToken {
    #[inline(always)]
    fn fma(self, a: f64, b: f64, c: f64) -> f64 {
        super::scalar::fma_scalar_t(self, a, b, c)
    }
    #[inline(always)]
    fn floor(self, x: f64) -> f64 {
        super::scalar::floor_scalar_t(self, x)
    }
}
#[cfg(target_arch = "x86_64")]
impl PowMath for X64V3Token {
    #[inline(always)]
    fn fma(self, a: f64, b: f64, c: f64) -> f64 {
        super::scalar::fma_v3_t(self, a, b, c)
    }
    #[inline(always)]
    fn floor(self, x: f64) -> f64 {
        super::scalar::floor_v2_t(self.v2(), x)
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl PowMath for X64V4xToken {
    #[inline(always)]
    fn fma(self, a: f64, b: f64, c: f64) -> f64 {
        super::scalar::fma_v3_t(self.v3(), a, b, c)
    }
    #[inline(always)]
    fn floor(self, x: f64) -> f64 {
        super::scalar::floor_v2_t(self.v3().v2(), x)
    }
}
#[cfg(target_arch = "aarch64")]
impl PowMath for NeonToken {
    #[inline(always)]
    fn fma(self, a: f64, b: f64, c: f64) -> f64 {
        super::scalar::fma_neon_t(self, a, b, c)
    }
    #[inline(always)]
    fn floor(self, x: f64) -> f64 {
        super::scalar::floor_neon_t(self, x)
    }
}
#[cfg(target_arch = "wasm32")]
impl PowMath for Wasm128Token {
    #[inline(always)]
    fn fma(self, a: f64, b: f64, c: f64) -> f64 {
        super::scalar::fma_wasm128_t(self, a, b, c)
    }
    #[inline(always)]
    fn floor(self, x: f64) -> f64 {
        super::scalar::floor_wasm128_t(self, x)
    }
}

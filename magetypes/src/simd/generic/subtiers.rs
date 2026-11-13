//! Token-gated sub-tier operations and [`MagetypesToken`] representation mapping.
//!
//! Sub-tier proof tokens (`X64V3VnniToken`, `Arm64V2Token`, `Arm64V3Token`,
//! `Wasm128RelaxedToken`, etc.) share the exact hardware register representation
//! of their base width token (`X64V3Token`, `NeonToken`, `Wasm128Token`). Rather
//! than duplicating 40+ backend traits per sub-tier, `#[magetypes(..., define(...))]`
//! and `#[attune(..., define(...))]` alias generic vector names to the base
//! representation token at zero KLOC / `.rmeta` cost, [`MagetypesToken::magetypes`]
//! extracts that representation token uniformly across tiers, and sub-tier-specific
//! instructions live here as explicit, token-gated inherent methods.

#![forbid(unsafe_code)]

mod magetypes_token_sealed {
    pub trait Sealed {}
}

/// Maps any [`archmage::SimdToken`] (including sub-tier and crypto/specialized
/// tokens) to the token type that implements `magetypes` width backend traits
/// on the active target.
///
/// Inside a `#[magetypes(..., define(...))]` or `#[attune(..., define(...))]`
/// kernel that mixes base tiers (`v3`, `v4x`, `neon`, `wasm128`) and sub-tiers
/// (`v3_vnni`, `fp16`, `arm_v2`, `arm_v3`, `wasm128_relaxed`), call
/// `let mt = token.magetypes();` to pass the representation token to `_t`
/// constructors (`f32x8::load_t(mt, ...)`, `i32x8::zero_t(mt)`), while keeping
/// `token` itself for sub-tier-specific intrinsics (`acc.mul_add_u8_i8_vnni(token, a, b)`,
/// `acc.dot_i8(token, a, b)`, `acc.fmlal_low_f16(token, a, b)`).
///
/// This trait is sealed and implemented only for `archmage` tokens whose
/// `magetypes` representation backend is available on the current target.
pub trait MagetypesToken: archmage::SimdToken + magetypes_token_sealed::Sealed {
    /// The `archmage` token type that implements `magetypes` backend traits for `Self`.
    type MagetypesBackend: archmage::SimdToken;

    /// Extract the `magetypes` representation backend token (`self` for direct
    /// backend tokens, or zero-cost downgrade such as `.v3()`, `.v4()`, `.neon()`,
    /// `.wasm128()` for sub-tier tokens).
    fn magetypes(self) -> Self::MagetypesBackend;
}

impl magetypes_token_sealed::Sealed for archmage::ScalarToken {}
impl MagetypesToken for archmage::ScalarToken {
    type MagetypesBackend = archmage::ScalarToken;

    #[inline(always)]
    fn magetypes(self) -> Self::MagetypesBackend {
        self
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_magetypes_tokens {
    use super::{MagetypesToken, magetypes_token_sealed::Sealed};
    use archmage::{X64V3CryptoToken, X64V3GfniCryptoToken, X64V3Token, X64V3VnniToken};

    impl Sealed for X64V3Token {}
    impl MagetypesToken for X64V3Token {
        type MagetypesBackend = X64V3Token;

        #[inline(always)]
        fn magetypes(self) -> Self::MagetypesBackend {
            self
        }
    }

    impl Sealed for X64V3CryptoToken {}
    impl MagetypesToken for X64V3CryptoToken {
        type MagetypesBackend = X64V3Token;

        #[inline(always)]
        fn magetypes(self) -> Self::MagetypesBackend {
            self.v3()
        }
    }

    impl Sealed for X64V3GfniCryptoToken {}
    impl MagetypesToken for X64V3GfniCryptoToken {
        type MagetypesBackend = X64V3Token;

        #[inline(always)]
        fn magetypes(self) -> Self::MagetypesBackend {
            self.v3()
        }
    }

    impl Sealed for X64V3VnniToken {}
    impl MagetypesToken for X64V3VnniToken {
        type MagetypesBackend = X64V3Token;

        #[inline(always)]
        fn magetypes(self) -> Self::MagetypesBackend {
            self.v3()
        }
    }

    #[cfg(feature = "avx512")]
    mod avx512_tokens {
        use super::{MagetypesToken, Sealed};
        use archmage::{Avx512Fp16Token, X64V4Token, X64V4xToken};

        impl Sealed for X64V4Token {}
        impl MagetypesToken for X64V4Token {
            type MagetypesBackend = X64V4Token;

            #[inline(always)]
            fn magetypes(self) -> Self::MagetypesBackend {
                self
            }
        }

        impl Sealed for X64V4xToken {}
        impl MagetypesToken for X64V4xToken {
            type MagetypesBackend = X64V4xToken;

            #[inline(always)]
            fn magetypes(self) -> Self::MagetypesBackend {
                self
            }
        }

        impl Sealed for Avx512Fp16Token {}
        impl MagetypesToken for Avx512Fp16Token {
            type MagetypesBackend = X64V4Token;

            #[inline(always)]
            fn magetypes(self) -> Self::MagetypesBackend {
                self.v4()
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod arm_magetypes_tokens {
    use super::{MagetypesToken, magetypes_token_sealed::Sealed};
    use archmage::{
        Arm64V2Token, Arm64V3Token, NeonAesToken, NeonCrcToken, NeonSha3Token, NeonToken,
    };

    impl Sealed for NeonToken {}
    impl MagetypesToken for NeonToken {
        type MagetypesBackend = NeonToken;

        #[inline(always)]
        fn magetypes(self) -> Self::MagetypesBackend {
            self
        }
    }

    impl Sealed for NeonAesToken {}
    impl MagetypesToken for NeonAesToken {
        type MagetypesBackend = NeonToken;

        #[inline(always)]
        fn magetypes(self) -> Self::MagetypesBackend {
            self.neon()
        }
    }

    impl Sealed for NeonSha3Token {}
    impl MagetypesToken for NeonSha3Token {
        type MagetypesBackend = NeonToken;

        #[inline(always)]
        fn magetypes(self) -> Self::MagetypesBackend {
            self.neon()
        }
    }

    impl Sealed for NeonCrcToken {}
    impl MagetypesToken for NeonCrcToken {
        type MagetypesBackend = NeonToken;

        #[inline(always)]
        fn magetypes(self) -> Self::MagetypesBackend {
            self.neon()
        }
    }

    impl Sealed for Arm64V2Token {}
    impl MagetypesToken for Arm64V2Token {
        type MagetypesBackend = NeonToken;

        #[inline(always)]
        fn magetypes(self) -> Self::MagetypesBackend {
            self.neon()
        }
    }

    impl Sealed for Arm64V3Token {}
    impl MagetypesToken for Arm64V3Token {
        type MagetypesBackend = NeonToken;

        #[inline(always)]
        fn magetypes(self) -> Self::MagetypesBackend {
            self.neon()
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_magetypes_tokens {
    use super::{MagetypesToken, magetypes_token_sealed::Sealed};
    use archmage::Wasm128Token;

    impl Sealed for Wasm128Token {}
    impl MagetypesToken for Wasm128Token {
        type MagetypesBackend = Wasm128Token;

        #[inline(always)]
        fn magetypes(self) -> Self::MagetypesBackend {
            self
        }
    }

    #[cfg(target_feature = "relaxed-simd")]
    mod relaxed_tokens {
        use super::{MagetypesToken, Sealed, Wasm128Token};
        use archmage::{Wasm128RelaxedFmaToken, Wasm128RelaxedToken};

        impl Sealed for Wasm128RelaxedToken {}
        impl MagetypesToken for Wasm128RelaxedToken {
            type MagetypesBackend = Wasm128Token;

            #[inline(always)]
            fn magetypes(self) -> Self::MagetypesBackend {
                self.wasm128()
            }
        }

        impl Sealed for Wasm128RelaxedFmaToken {}
        impl MagetypesToken for Wasm128RelaxedFmaToken {
            type MagetypesBackend = Wasm128RelaxedFmaToken;

            #[inline(always)]
            fn magetypes(self) -> Self::MagetypesBackend {
                self
            }
        }
    }
}

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

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
mod x86_avx512_vnni {
    use super::super::{i8x16, i8x32, i16x8, i16x16, i32x4, i32x8, u8x16, u8x32};
    #[cfg(feature = "w512")]
    use super::super::{i8x64, i16x32, i32x16, u8x64};
    use archmage::X64V4xToken;
    use core::arch::x86_64::{
        _mm_dpbusd_epi32, _mm_dpbusds_epi32, _mm_dpwssd_epi32, _mm_dpwssds_epi32,
        _mm256_dpbusd_epi32, _mm256_dpbusds_epi32, _mm256_dpwssd_epi32, _mm256_dpwssds_epi32,
    };
    #[cfg(feature = "w512")]
    use core::arch::x86_64::{
        _mm512_dpbusd_epi32, _mm512_dpbusds_epi32, _mm512_dpwssd_epi32, _mm512_dpwssds_epi32,
    };

    impl i32x4<X64V4xToken> {
        /// Multiply groups of 4 unsigned `u8` lanes in `a` by signed `i8` lanes in `b`
        /// and wrapping-accumulate each 4-product sum into the corresponding `i32` lane
        /// of `self` (EVEX `VPDPBUSD`, `_mm_dpbusd_epi32`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn mul_add_u8_i8_vnni(
            self,
            _token: X64V4xToken,
            a: u8x16<X64V4xToken>,
            b: i8x16<X64V4xToken>,
        ) -> Self {
            Self(_mm_dpbusd_epi32(self.0, a.0, b.0), self.1)
        }

        /// Signed-saturating `u8 * i8` 4-way dot-product accumulate into `self`
        /// (EVEX `VPDPBUSDS`, `_mm_dpbusds_epi32`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn mul_add_u8_i8_sat_vnni(
            self,
            _token: X64V4xToken,
            a: u8x16<X64V4xToken>,
            b: i8x16<X64V4xToken>,
        ) -> Self {
            Self(_mm_dpbusds_epi32(self.0, a.0, b.0), self.1)
        }

        /// Multiply pairs of signed `i16` lanes in `a` and `b` and wrapping-accumulate
        /// each 2-product sum into the corresponding `i32` lane of `self`
        /// (EVEX `VPDPWSSD`, `_mm_dpwssd_epi32`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn mul_add_i16_vnni(
            self,
            _token: X64V4xToken,
            a: i16x8<X64V4xToken>,
            b: i16x8<X64V4xToken>,
        ) -> Self {
            Self(_mm_dpwssd_epi32(self.0, a.0, b.0), self.1)
        }

        /// Signed-saturating `i16 * i16` 2-way dot-product accumulate into `self`
        /// (EVEX `VPDPWSSDS`, `_mm_dpwssds_epi32`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn mul_add_i16_sat_vnni(
            self,
            _token: X64V4xToken,
            a: i16x8<X64V4xToken>,
            b: i16x8<X64V4xToken>,
        ) -> Self {
            Self(_mm_dpwssds_epi32(self.0, a.0, b.0), self.1)
        }
    }

    impl i32x8<X64V4xToken> {
        /// Multiply groups of 4 unsigned `u8` lanes in `a` by signed `i8` lanes in `b`
        /// and wrapping-accumulate each 4-product sum into the corresponding `i32` lane
        /// of `self` (EVEX `VPDPBUSD`, `_mm256_dpbusd_epi32`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn mul_add_u8_i8_vnni(
            self,
            _token: X64V4xToken,
            a: u8x32<X64V4xToken>,
            b: i8x32<X64V4xToken>,
        ) -> Self {
            Self(_mm256_dpbusd_epi32(self.0, a.0, b.0), self.1)
        }

        /// Signed-saturating `u8 * i8` 4-way dot-product accumulate into `self`
        /// (EVEX `VPDPBUSDS`, `_mm256_dpbusds_epi32`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn mul_add_u8_i8_sat_vnni(
            self,
            _token: X64V4xToken,
            a: u8x32<X64V4xToken>,
            b: i8x32<X64V4xToken>,
        ) -> Self {
            Self(_mm256_dpbusds_epi32(self.0, a.0, b.0), self.1)
        }

        /// Multiply pairs of signed `i16` lanes in `a` and `b` and wrapping-accumulate
        /// each 2-product sum into the corresponding `i32` lane of `self`
        /// (EVEX `VPDPWSSD`, `_mm256_dpwssd_epi32`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn mul_add_i16_vnni(
            self,
            _token: X64V4xToken,
            a: i16x16<X64V4xToken>,
            b: i16x16<X64V4xToken>,
        ) -> Self {
            Self(_mm256_dpwssd_epi32(self.0, a.0, b.0), self.1)
        }

        /// Signed-saturating `i16 * i16` 2-way dot-product accumulate into `self`
        /// (EVEX `VPDPWSSDS`, `_mm256_dpwssds_epi32`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn mul_add_i16_sat_vnni(
            self,
            _token: X64V4xToken,
            a: i16x16<X64V4xToken>,
            b: i16x16<X64V4xToken>,
        ) -> Self {
            Self(_mm256_dpwssds_epi32(self.0, a.0, b.0), self.1)
        }
    }

    #[cfg(feature = "w512")]
    impl i32x16<X64V4xToken> {
        /// Multiply groups of 4 unsigned `u8` lanes in `a` by signed `i8` lanes in `b`
        /// and wrapping-accumulate each 4-product sum into the corresponding `i32` lane
        /// of `self` (EVEX `VPDPBUSD`, `_mm512_dpbusd_epi32`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn mul_add_u8_i8_vnni(
            self,
            _token: X64V4xToken,
            a: u8x64<X64V4xToken>,
            b: i8x64<X64V4xToken>,
        ) -> Self {
            Self(_mm512_dpbusd_epi32(self.0, a.0, b.0), self.1)
        }

        /// Signed-saturating `u8 * i8` 4-way dot-product accumulate into `self`
        /// (EVEX `VPDPBUSDS`, `_mm512_dpbusds_epi32`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn mul_add_u8_i8_sat_vnni(
            self,
            _token: X64V4xToken,
            a: u8x64<X64V4xToken>,
            b: i8x64<X64V4xToken>,
        ) -> Self {
            Self(_mm512_dpbusds_epi32(self.0, a.0, b.0), self.1)
        }

        /// Multiply pairs of signed `i16` lanes in `a` and `b` and wrapping-accumulate
        /// each 2-product sum into the corresponding `i32` lane of `self`
        /// (EVEX `VPDPWSSD`, `_mm512_dpwssd_epi32`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn mul_add_i16_vnni(
            self,
            _token: X64V4xToken,
            a: i16x32<X64V4xToken>,
            b: i16x32<X64V4xToken>,
        ) -> Self {
            Self(_mm512_dpwssd_epi32(self.0, a.0, b.0), self.1)
        }

        /// Signed-saturating `i16 * i16` 2-way dot-product accumulate into `self`
        /// (EVEX `VPDPWSSDS`, `_mm512_dpwssds_epi32`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn mul_add_i16_sat_vnni(
            self,
            _token: X64V4xToken,
            a: i16x32<X64V4xToken>,
            b: i16x32<X64V4xToken>,
        ) -> Self {
            Self(_mm512_dpwssds_epi32(self.0, a.0, b.0), self.1)
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod arm_v2 {
    use super::super::{i8x16, i8x32, i16x8, i16x16, i32x4, i32x8, u8x16, u8x32, u32x4, u32x8};
    use archmage::{Arm64V2Token, NeonToken};
    use core::arch::aarch64::{vqrdmlahq_s16, vqrdmlshq_s16};

    impl i32x4<NeonToken> {
        /// Signed 4-way `i8 * i8` dot-product accumulate into `self` (`SDOT`, `vdotq_s32`
        /// on Rust 1.98+; widening `SMULL` + `SADALP` fallback under `+dotprod` on older
        /// toolchains down to MSRV 1.89).
        #[rustversion::since(1.98)]
        #[allow(clippy::incompatible_msrv)]
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn dot_i8(
            self,
            _token: Arm64V2Token,
            a: i8x16<NeonToken>,
            b: i8x16<NeonToken>,
        ) -> Self {
            Self(core::arch::aarch64::vdotq_s32(self.0, a.0, b.0), self.1)
        }

        /// Signed 4-way `i8 * i8` dot-product accumulate into `self` (`SDOT`, `vdotq_s32`
        /// on Rust 1.98+; widening `SMULL` + `SADALP` fallback under `+dotprod` on older
        /// toolchains down to MSRV 1.89).
        #[rustversion::before(1.98)]
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn dot_i8(
            self,
            _token: Arm64V2Token,
            a: i8x16<NeonToken>,
            b: i8x16<NeonToken>,
        ) -> Self {
            use core::arch::aarch64::{
                vaddq_s32, vget_low_s8, vmull_high_s8, vmull_s8, vpaddlq_s16, vpaddq_s32,
            };
            let lo = vpaddlq_s16(vmull_s8(vget_low_s8(a.0), vget_low_s8(b.0)));
            let hi = vpaddlq_s16(vmull_high_s8(a.0, b.0));
            Self(vaddq_s32(self.0, vpaddq_s32(lo, hi)), self.1)
        }
    }

    impl i32x8<NeonToken> {
        /// Two-register 256-bit signed 4-way `i8 * i8` dot-product accumulate into `self`
        /// (`2x SDOT` on Rust 1.98+).
        #[inline(always)]
        pub fn dot_i8(self, token: Arm64V2Token, a: i8x32<NeonToken>, b: i8x32<NeonToken>) -> Self {
            let lo = i32x4::from_repr_unchecked(self.1, self.0[0]).dot_i8(
                token,
                i8x16::from_repr_unchecked(self.1, a.0[0]),
                i8x16::from_repr_unchecked(self.1, b.0[0]),
            );
            let hi = i32x4::from_repr_unchecked(self.1, self.0[1]).dot_i8(
                token,
                i8x16::from_repr_unchecked(self.1, a.0[1]),
                i8x16::from_repr_unchecked(self.1, b.0[1]),
            );
            Self([lo.0, hi.0], self.1)
        }
    }

    impl u32x4<NeonToken> {
        /// Unsigned 4-way `u8 * u8` dot-product accumulate into `self` (`UDOT`, `vdotq_u32`
        /// on Rust 1.98+; widening `UMULL` + `UADALP` fallback under `+dotprod` on older
        /// toolchains down to MSRV 1.89).
        #[rustversion::since(1.98)]
        #[allow(clippy::incompatible_msrv)]
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn dot_u8(
            self,
            _token: Arm64V2Token,
            a: u8x16<NeonToken>,
            b: u8x16<NeonToken>,
        ) -> Self {
            Self(core::arch::aarch64::vdotq_u32(self.0, a.0, b.0), self.1)
        }

        /// Unsigned 4-way `u8 * u8` dot-product accumulate into `self` (`UDOT`, `vdotq_u32`
        /// on Rust 1.98+; widening `UMULL` + `UADALP` fallback under `+dotprod` on older
        /// toolchains down to MSRV 1.89).
        #[rustversion::before(1.98)]
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn dot_u8(
            self,
            _token: Arm64V2Token,
            a: u8x16<NeonToken>,
            b: u8x16<NeonToken>,
        ) -> Self {
            use core::arch::aarch64::{
                vaddq_u32, vget_low_u8, vmull_high_u8, vmull_u8, vpaddlq_u16, vpaddq_u32,
            };
            let lo = vpaddlq_u16(vmull_u8(vget_low_u8(a.0), vget_low_u8(b.0)));
            let hi = vpaddlq_u16(vmull_high_u8(a.0, b.0));
            Self(vaddq_u32(self.0, vpaddq_u32(lo, hi)), self.1)
        }
    }

    impl u32x8<NeonToken> {
        /// Two-register 256-bit unsigned 4-way `u8 * u8` dot-product accumulate into `self`
        /// (`2x UDOT` on Rust 1.98+).
        #[inline(always)]
        pub fn dot_u8(self, token: Arm64V2Token, a: u8x32<NeonToken>, b: u8x32<NeonToken>) -> Self {
            let lo = u32x4::from_repr_unchecked(self.1, self.0[0]).dot_u8(
                token,
                u8x16::from_repr_unchecked(self.1, a.0[0]),
                u8x16::from_repr_unchecked(self.1, b.0[0]),
            );
            let hi = u32x4::from_repr_unchecked(self.1, self.0[1]).dot_u8(
                token,
                u8x16::from_repr_unchecked(self.1, a.0[1]),
                u8x16::from_repr_unchecked(self.1, b.0[1]),
            );
            Self([lo.0, hi.0], self.1)
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

    impl i16x16<NeonToken> {
        /// Two-register 256-bit signed saturating rounding doubling multiply-accumulate
        /// high half (`2x SQRDMLAH`).
        #[inline(always)]
        pub fn qrdmlah(self, token: Arm64V2Token, a: Self, b: Self) -> Self {
            let lo = i16x8::from_repr_unchecked(self.1, self.0[0]).qrdmlah(
                token,
                i16x8::from_repr_unchecked(self.1, a.0[0]),
                i16x8::from_repr_unchecked(self.1, b.0[0]),
            );
            let hi = i16x8::from_repr_unchecked(self.1, self.0[1]).qrdmlah(
                token,
                i16x8::from_repr_unchecked(self.1, a.0[1]),
                i16x8::from_repr_unchecked(self.1, b.0[1]),
            );
            Self([lo.0, hi.0], self.1)
        }

        /// Two-register 256-bit signed saturating rounding doubling multiply-subtract
        /// high half (`2x SQRDMLSH`).
        #[inline(always)]
        pub fn qrdmlsh(self, token: Arm64V2Token, a: Self, b: Self) -> Self {
            let lo = i16x8::from_repr_unchecked(self.1, self.0[0]).qrdmlsh(
                token,
                i16x8::from_repr_unchecked(self.1, a.0[0]),
                i16x8::from_repr_unchecked(self.1, b.0[0]),
            );
            let hi = i16x8::from_repr_unchecked(self.1, self.0[1]).qrdmlsh(
                token,
                i16x8::from_repr_unchecked(self.1, a.0[1]),
                i16x8::from_repr_unchecked(self.1, b.0[1]),
            );
            Self([lo.0, hi.0], self.1)
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod arm_v3 {
    #[rustversion::before(1.94)]
    use super::super::i32x4;
    use super::super::{f32x4, f32x8, u16x8};
    use archmage::{Arm64V3Token, NeonToken};

    impl f32x4<NeonToken> {
        /// Fused half-precision multiply-add long (low 4 `f16` lanes `0..4` of `a` and `b`
        /// widened and fused-accumulated into `self`: `self + f32(a_lo) * f32(b_lo)`).
        ///
        /// Emits `FMLAL` (`vfmlalq_low_f16`, ARMv8.2-A `FHM`) on Rust 1.94+, and falls
        /// back to `f16_to_f32` + NEON `FMLA` on older toolchains down to MSRV 1.89.
        #[rustversion::since(1.94)]
        #[allow(clippy::incompatible_msrv)]
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn fmlal_low_f16(
            self,
            _token: Arm64V3Token,
            a: u16x8<NeonToken>,
            b: u16x8<NeonToken>,
        ) -> Self {
            use core::arch::aarch64::{vfmlalq_low_f16, vreinterpretq_f16_u16};
            Self(
                vfmlalq_low_f16(
                    self.0,
                    vreinterpretq_f16_u16(a.0),
                    vreinterpretq_f16_u16(b.0),
                ),
                self.1,
            )
        }

        /// Fused half-precision multiply-add long (low 4 `f16` lanes `0..4`).
        #[rustversion::before(1.94)]
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn fmlal_low_f16(
            self,
            _token: Arm64V3Token,
            a: u16x8<NeonToken>,
            b: u16x8<NeonToken>,
        ) -> Self {
            let aa = a.to_array();
            let bb = b.to_array();
            let af = i32x4::from_array_t(
                self.1,
                [aa[0] as i32, aa[1] as i32, aa[2] as i32, aa[3] as i32],
            )
            .f16_to_f32();
            let bf = i32x4::from_array_t(
                self.1,
                [bb[0] as i32, bb[1] as i32, bb[2] as i32, bb[3] as i32],
            )
            .f16_to_f32();
            Self(core::arch::aarch64::vfmaq_f32(self.0, af.0, bf.0), self.1)
        }

        /// Fused half-precision multiply-add long (high 4 `f16` lanes `4..8` of `a` and `b`
        /// widened and fused-accumulated into `self`: `self + f32(a_hi) * f32(b_hi)`).
        ///
        /// Emits `FMLAL2` (`vfmlalq_high_f16`, ARMv8.2-A `FHM`) on Rust 1.94+.
        #[rustversion::since(1.94)]
        #[allow(clippy::incompatible_msrv)]
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn fmlal_high_f16(
            self,
            _token: Arm64V3Token,
            a: u16x8<NeonToken>,
            b: u16x8<NeonToken>,
        ) -> Self {
            use core::arch::aarch64::{vfmlalq_high_f16, vreinterpretq_f16_u16};
            Self(
                vfmlalq_high_f16(
                    self.0,
                    vreinterpretq_f16_u16(a.0),
                    vreinterpretq_f16_u16(b.0),
                ),
                self.1,
            )
        }

        /// Fused half-precision multiply-add long (high 4 `f16` lanes `4..8`).
        #[rustversion::before(1.94)]
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn fmlal_high_f16(
            self,
            _token: Arm64V3Token,
            a: u16x8<NeonToken>,
            b: u16x8<NeonToken>,
        ) -> Self {
            let aa = a.to_array();
            let bb = b.to_array();
            let af = i32x4::from_array_t(
                self.1,
                [aa[4] as i32, aa[5] as i32, aa[6] as i32, aa[7] as i32],
            )
            .f16_to_f32();
            let bf = i32x4::from_array_t(
                self.1,
                [bb[4] as i32, bb[5] as i32, bb[6] as i32, bb[7] as i32],
            )
            .f16_to_f32();
            Self(core::arch::aarch64::vfmaq_f32(self.0, af.0, bf.0), self.1)
        }

        /// Fused half-precision multiply-subtract long (low 4 `f16` lanes `0..4` of `a` and `b`:
        /// `self - f32(a_lo) * f32(b_lo)`).
        ///
        /// Emits `FMLSL` (`vfmlslq_low_f16`, ARMv8.2-A `FHM`) on Rust 1.94+.
        #[rustversion::since(1.94)]
        #[allow(clippy::incompatible_msrv)]
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn fmlsl_low_f16(
            self,
            _token: Arm64V3Token,
            a: u16x8<NeonToken>,
            b: u16x8<NeonToken>,
        ) -> Self {
            use core::arch::aarch64::{vfmlslq_low_f16, vreinterpretq_f16_u16};
            Self(
                vfmlslq_low_f16(
                    self.0,
                    vreinterpretq_f16_u16(a.0),
                    vreinterpretq_f16_u16(b.0),
                ),
                self.1,
            )
        }

        /// Fused half-precision multiply-subtract long (low 4 `f16` lanes `0..4`).
        #[rustversion::before(1.94)]
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn fmlsl_low_f16(
            self,
            _token: Arm64V3Token,
            a: u16x8<NeonToken>,
            b: u16x8<NeonToken>,
        ) -> Self {
            let aa = a.to_array();
            let bb = b.to_array();
            let af = i32x4::from_array_t(
                self.1,
                [aa[0] as i32, aa[1] as i32, aa[2] as i32, aa[3] as i32],
            )
            .f16_to_f32();
            let bf = i32x4::from_array_t(
                self.1,
                [bb[0] as i32, bb[1] as i32, bb[2] as i32, bb[3] as i32],
            )
            .f16_to_f32();
            Self(core::arch::aarch64::vfmsq_f32(self.0, af.0, bf.0), self.1)
        }

        /// Fused half-precision multiply-subtract long (high 4 `f16` lanes `4..8` of `a` and `b`:
        /// `self - f32(a_hi) * f32(b_hi)`).
        ///
        /// Emits `FMLSL2` (`vfmlslq_high_f16`, ARMv8.2-A `FHM`) on Rust 1.94+.
        #[rustversion::since(1.94)]
        #[allow(clippy::incompatible_msrv)]
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn fmlsl_high_f16(
            self,
            _token: Arm64V3Token,
            a: u16x8<NeonToken>,
            b: u16x8<NeonToken>,
        ) -> Self {
            use core::arch::aarch64::{vfmlslq_high_f16, vreinterpretq_f16_u16};
            Self(
                vfmlslq_high_f16(
                    self.0,
                    vreinterpretq_f16_u16(a.0),
                    vreinterpretq_f16_u16(b.0),
                ),
                self.1,
            )
        }

        /// Fused half-precision multiply-subtract long (high 4 `f16` lanes `4..8`).
        #[rustversion::before(1.94)]
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn fmlsl_high_f16(
            self,
            _token: Arm64V3Token,
            a: u16x8<NeonToken>,
            b: u16x8<NeonToken>,
        ) -> Self {
            let aa = a.to_array();
            let bb = b.to_array();
            let af = i32x4::from_array_t(
                self.1,
                [aa[4] as i32, aa[5] as i32, aa[6] as i32, aa[7] as i32],
            )
            .f16_to_f32();
            let bf = i32x4::from_array_t(
                self.1,
                [bb[4] as i32, bb[5] as i32, bb[6] as i32, bb[7] as i32],
            )
            .f16_to_f32();
            Self(core::arch::aarch64::vfmsq_f32(self.0, af.0, bf.0), self.1)
        }
    }

    impl f32x8<NeonToken> {
        /// Fused 8-lane half-precision multiply-add long (`FMLAL` + `FMLAL2` across
        /// all 8 `f16` bit patterns in `a` and `b` into `self`).
        #[inline(always)]
        pub fn fmlal_f16(
            self,
            token: Arm64V3Token,
            a: u16x8<NeonToken>,
            b: u16x8<NeonToken>,
        ) -> Self {
            let lo = f32x4::from_repr_unchecked(self.1, self.0[0]).fmlal_low_f16(token, a, b);
            let hi = f32x4::from_repr_unchecked(self.1, self.0[1]).fmlal_high_f16(token, a, b);
            Self([lo.0, hi.0], self.1)
        }

        /// Fused 8-lane half-precision multiply-subtract long (`FMLSL` + `FMLSL2` across
        /// all 8 `f16` bit patterns in `a` and `b` from `self`).
        #[inline(always)]
        pub fn fmlsl_f16(
            self,
            token: Arm64V3Token,
            a: u16x8<NeonToken>,
            b: u16x8<NeonToken>,
        ) -> Self {
            let lo = f32x4::from_repr_unchecked(self.1, self.0[0]).fmlsl_low_f16(token, a, b);
            let hi = f32x4::from_repr_unchecked(self.1, self.0[1]).fmlsl_high_f16(token, a, b);
            Self([lo.0, hi.0], self.1)
        }
    }
}

#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    target_feature = "relaxed-simd"
))]
mod wasm_relaxed {
    use super::super::{
        f32x4, f32x8, f64x2, f64x4, i8x16, i8x32, i16x8, i16x16, i32x4, i32x8, u32x4, u32x8,
    };
    use archmage::{Wasm128RelaxedToken, Wasm128Token};
    use core::arch::wasm32::{
        f32x4_relaxed_max, f32x4_relaxed_min, f64x2_relaxed_max, f64x2_relaxed_min,
        i16x8_relaxed_dot_i8x16_i7x16, i16x8_relaxed_q15mulr, i32x4_relaxed_dot_i8x16_i7x16_add,
        i32x4_relaxed_trunc_f32x4, u32x4_relaxed_trunc_f32x4,
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

        /// Relaxed float-to-signed-`i32` truncation (`i32x4.relaxed_trunc_f32x4_s`;
        /// out-of-bounds and NaN lanes are implementation-defined).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn to_i32_relaxed(self, _token: Wasm128RelaxedToken) -> i32x4<Wasm128Token> {
            i32x4::from_repr_unchecked(self.1, i32x4_relaxed_trunc_f32x4(self.0))
        }

        /// Relaxed float-to-unsigned-`u32` truncation (`i32x4.relaxed_trunc_f32x4_u`;
        /// negative, out-of-bounds, and NaN lanes are implementation-defined).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn to_u32_relaxed(self, _token: Wasm128RelaxedToken) -> u32x4<Wasm128Token> {
            u32x4::from_repr_unchecked(self.1, u32x4_relaxed_trunc_f32x4(self.0))
        }
    }

    impl f32x8<Wasm128Token> {
        /// Two-register lane-wise relaxed minimum (`2x f32x4.relaxed_min`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn min_relaxed(self, _token: Wasm128RelaxedToken, other: Self) -> Self {
            Self(
                [
                    f32x4_relaxed_min(self.0[0], other.0[0]),
                    f32x4_relaxed_min(self.0[1], other.0[1]),
                ],
                self.1,
            )
        }

        /// Two-register lane-wise relaxed maximum (`2x f32x4.relaxed_max`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn max_relaxed(self, _token: Wasm128RelaxedToken, other: Self) -> Self {
            Self(
                [
                    f32x4_relaxed_max(self.0[0], other.0[0]),
                    f32x4_relaxed_max(self.0[1], other.0[1]),
                ],
                self.1,
            )
        }

        /// Two-register relaxed float-to-signed-`i32` truncation (`2x i32x4.relaxed_trunc_f32x4_s`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn to_i32_relaxed(self, _token: Wasm128RelaxedToken) -> i32x8<Wasm128Token> {
            i32x8::from_repr_unchecked(
                self.1,
                [
                    i32x4_relaxed_trunc_f32x4(self.0[0]),
                    i32x4_relaxed_trunc_f32x4(self.0[1]),
                ],
            )
        }

        /// Two-register relaxed float-to-unsigned-`u32` truncation (`2x i32x4.relaxed_trunc_f32x4_u`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn to_u32_relaxed(self, _token: Wasm128RelaxedToken) -> u32x8<Wasm128Token> {
            u32x8::from_repr_unchecked(
                self.1,
                [
                    u32x4_relaxed_trunc_f32x4(self.0[0]),
                    u32x4_relaxed_trunc_f32x4(self.0[1]),
                ],
            )
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

    impl f64x4<Wasm128Token> {
        /// Two-register lane-wise relaxed minimum (`2x f64x2.relaxed_min`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn min_relaxed(self, _token: Wasm128RelaxedToken, other: Self) -> Self {
            Self(
                [
                    f64x2_relaxed_min(self.0[0], other.0[0]),
                    f64x2_relaxed_min(self.0[1], other.0[1]),
                ],
                self.1,
            )
        }

        /// Two-register lane-wise relaxed maximum (`2x f64x2.relaxed_max`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn max_relaxed(self, _token: Wasm128RelaxedToken, other: Self) -> Self {
            Self(
                [
                    f64x2_relaxed_max(self.0[0], other.0[0]),
                    f64x2_relaxed_max(self.0[1], other.0[1]),
                ],
                self.1,
            )
        }
    }

    impl i16x8<Wasm128Token> {
        /// Relaxed Q15 rounding multiply (`i16x8.relaxed_q15mulr_s`; `(i16::MIN, i16::MIN)`
        /// is implementation-defined).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn mul_hrs_q15_relaxed(self, _token: Wasm128RelaxedToken, other: Self) -> Self {
            Self(i16x8_relaxed_q15mulr(self.0, other.0), self.1)
        }

        /// Relaxed pairwise `i8 * i7 -> i16` dot-product (`i16x8.relaxed_dot_i8x16_i7x16_s`).
        ///
        /// Bytes of `b` must be in `0..=127` (`i7`) so signed and unsigned hardware
        /// interpretations agree.
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn dot_i8_i7_relaxed(
            token: Wasm128RelaxedToken,
            a: i8x16<Wasm128Token>,
            b: i8x16<Wasm128Token>,
        ) -> Self {
            Self(i16x8_relaxed_dot_i8x16_i7x16(a.0, b.0), token.wasm128())
        }
    }

    impl i16x16<Wasm128Token> {
        /// Two-register relaxed Q15 rounding multiply (`2x i16x8.relaxed_q15mulr_s`).
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn mul_hrs_q15_relaxed(self, _token: Wasm128RelaxedToken, other: Self) -> Self {
            Self(
                [
                    i16x8_relaxed_q15mulr(self.0[0], other.0[0]),
                    i16x8_relaxed_q15mulr(self.0[1], other.0[1]),
                ],
                self.1,
            )
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

    impl i32x8<Wasm128Token> {
        /// Two-register relaxed 4-way `i8 * i7` dot-product accumulate into `self`
        /// (`2x i32x4.relaxed_dot_i8x16_i7x16_add`).
        ///
        /// Bytes of `b` must be in `0..=127` (`i7`) so signed and unsigned
        /// hardware interpretations agree.
        #[archmage::attune(wrap, suppress_const_test)]
        pub fn dot_i8_i7_add_relaxed(
            self,
            _token: Wasm128RelaxedToken,
            a: i8x32<Wasm128Token>,
            b: i8x32<Wasm128Token>,
        ) -> Self {
            Self(
                [
                    i32x4_relaxed_dot_i8x16_i7x16_add(a.0[0], b.0[0], self.0[0]),
                    i32x4_relaxed_dot_i8x16_i7x16_add(a.0[1], b.0[1], self.0[1]),
                ],
                self.1,
            )
        }
    }
}

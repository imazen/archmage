//! Canonical token normalization for `magemath` dispatch.
//!
//! Subtier and crypto tokens (`X64V3CryptoToken`, `Arm64V2Token`,
//! `Wasm128RelaxedToken`, etc.) downgrade via zero-cost extractors to the
//! base SIMD proof (`X64V3Token`, `NeonToken`, `Wasm128Token`, or scalar) so
//! `magemath` executes the native path without expanding `Plan` or
//! duplicating `graph::*` monomorphizations.

/// Canonical SIMD execution proof extracted from an erased `SimdToken`.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
#[derive(Clone, Copy)]
pub(crate) enum CanonicalToken {
    #[cfg(target_arch = "x86_64")]
    V3(archmage::X64V3Token),
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    V4(archmage::X64V4Token),
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    V4x(archmage::X64V4xToken),
    #[cfg(target_arch = "aarch64")]
    Neon(archmage::NeonToken),
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    Wasm {
        token: archmage::Wasm128Token,
        /// `true` only when `Wasm128RelaxedFmaToken` has already verified
        /// single-rounding `f32x4` and `f64x2` relaxed FMA at runtime.
        relaxed_fma: bool,
    },
    Scalar,
}

/// Map an erased `SimdToken` onto its canonical native proof, if any.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
#[inline(always)]
pub(crate) fn canonicalize(any: &dyn core::any::Any) -> CanonicalToken {
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = any.downcast_ref::<archmage::X64V3Token>() {
            return CanonicalToken::V3(*t);
        }
        if let Some(t) = any.downcast_ref::<archmage::X64V3CryptoToken>() {
            return CanonicalToken::V3(t.v3());
        }
        if let Some(t) = any.downcast_ref::<archmage::X64V3GfniCryptoToken>() {
            return CanonicalToken::V3(t.v3());
        }
        if let Some(t) = any.downcast_ref::<archmage::X64V3VnniToken>() {
            return CanonicalToken::V3(t.v3());
        }
        if let Some(t) = any.downcast_ref::<archmage::X64V3Vnni2Token>() {
            return CanonicalToken::V3(t.v3());
        }
        #[cfg(feature = "avx512")]
        {
            if let Some(t) = any.downcast_ref::<archmage::X64V4Token>() {
                return CanonicalToken::V4(*t);
            }
            if let Some(t) = any.downcast_ref::<archmage::X64V4xToken>() {
                return CanonicalToken::V4x(*t);
            }
            if let Some(t) = any.downcast_ref::<archmage::Avx512Fp16Token>() {
                return CanonicalToken::V4(t.v4());
            }
        }
        #[cfg(not(feature = "avx512"))]
        {
            if let Some(t) = any.downcast_ref::<archmage::X64V4Token>() {
                return CanonicalToken::V3(t.v3());
            }
            if let Some(t) = any.downcast_ref::<archmage::X64V4xToken>() {
                return CanonicalToken::V3(t.v3());
            }
            if let Some(t) = any.downcast_ref::<archmage::Avx512Fp16Token>() {
                return CanonicalToken::V3(t.v3());
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if let Some(t) = any.downcast_ref::<archmage::NeonToken>() {
            return CanonicalToken::Neon(*t);
        }
        if let Some(t) = any.downcast_ref::<archmage::NeonAesToken>() {
            return CanonicalToken::Neon(t.neon());
        }
        if let Some(t) = any.downcast_ref::<archmage::NeonSha3Token>() {
            return CanonicalToken::Neon(t.neon());
        }
        if let Some(t) = any.downcast_ref::<archmage::NeonCrcToken>() {
            return CanonicalToken::Neon(t.neon());
        }
        if let Some(t) = any.downcast_ref::<archmage::Arm64V2Token>() {
            return CanonicalToken::Neon(t.neon());
        }
        if let Some(t) = any.downcast_ref::<archmage::Arm64V3Token>() {
            return CanonicalToken::Neon(t.neon());
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        if let Some(t) = any.downcast_ref::<archmage::Wasm128Token>() {
            return CanonicalToken::Wasm {
                token: *t,
                relaxed_fma: false,
            };
        }
        #[cfg(target_feature = "relaxed-simd")]
        {
            if let Some(t) = any.downcast_ref::<archmage::Wasm128RelaxedToken>() {
                return CanonicalToken::Wasm {
                    token: t.wasm128(),
                    relaxed_fma: false,
                };
            }
            if let Some(t) = any.downcast_ref::<archmage::Wasm128RelaxedFmaToken>() {
                return CanonicalToken::Wasm {
                    token: t.wasm128(),
                    relaxed_fma: true,
                };
            }
        }
    }
    CanonicalToken::Scalar
}

/// Returns `true` when `T` backs `magetypes` vector ops with `ScalarToken`.
#[inline(always)]
pub(crate) fn is_scalar_backed<T: 'static>() -> bool {
    let id = core::any::TypeId::of::<T>();
    if id == core::any::TypeId::of::<archmage::ScalarToken>() {
        return true;
    }
    #[cfg(target_arch = "x86_64")]
    if id == core::any::TypeId::of::<archmage::X64V1Token>()
        || id == core::any::TypeId::of::<archmage::X64V2Token>()
        || id == core::any::TypeId::of::<archmage::X64CryptoToken>()
    {
        return true;
    }
    false
}

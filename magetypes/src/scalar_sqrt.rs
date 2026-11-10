//! Scalar sqrt selection. Keep the public integer implementation as the fallback
//! and independent regression control; use hardware where the build proves it.
//! Internal summons are intentional: no runtime upgrade, one selection per scalar
//! operation, constant-folded outside testable_dispatch. That feature can force
//! software fallback by disabling the matching baseline token. WASM needs only
//! SIMD128; relaxed-SIMD is not required for correctly rounded sqrt.
#![forbid(unsafe_code)]

#[inline(always)]
pub(crate) fn sqrtf(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        selected::sqrtf(x)
    }
}

#[inline(always)]
pub(crate) fn sqrt(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        selected::sqrt(x)
    }
}

// These are baseline scalar lowerings, not runtime SIMD upgrades. Stable core
// has no scalar sqrt method at our MSRV. Compile-time features admit a matching Archmage
// proof, which gates the native kernel. Test-time token disabling falls back
// to the same software graph. No runtime feature upgrade is attempted.
#[cfg(all(not(feature = "std"), target_arch = "x86_64", target_feature = "sse2"))]
mod selected {
    use archmage::SimdToken;
    use core::arch::x86_64::*;
    #[inline(always)]
    pub(super) fn sqrtf(x: f32) -> f32 {
        if let Some(token) = archmage::X64V1Token::summon() {
            hardware32(token, x)
        } else {
            crate::nostd_math::sqrtf(x)
        }
    }
    #[inline(always)]
    pub(super) fn sqrt(x: f64) -> f64 {
        if let Some(token) = archmage::X64V1Token::summon() {
            hardware64(token, x)
        } else {
            crate::nostd_math::sqrt(x)
        }
    }

    #[archmage::arcane]
    fn hardware32(_token: archmage::X64V1Token, x: f32) -> f32 {
        _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)))
    }
    #[archmage::arcane]
    fn hardware64(_token: archmage::X64V1Token, x: f64) -> f64 {
        let a = _mm_set_sd(x);
        _mm_cvtsd_f64(_mm_sqrt_sd(a, a))
    }
}

#[cfg(all(not(feature = "std"), target_arch = "aarch64", target_feature = "neon"))]
mod selected {
    use archmage::SimdToken;
    use core::arch::aarch64::*;
    #[inline(always)]
    pub(super) fn sqrtf(x: f32) -> f32 {
        if let Some(token) = archmage::NeonToken::summon() {
            hardware32(token, x)
        } else {
            crate::nostd_math::sqrtf(x)
        }
    }
    #[inline(always)]
    pub(super) fn sqrt(x: f64) -> f64 {
        if let Some(token) = archmage::NeonToken::summon() {
            hardware64(token, x)
        } else {
            crate::nostd_math::sqrt(x)
        }
    }

    #[archmage::arcane]
    fn hardware32(_token: archmage::NeonToken, x: f32) -> f32 {
        vget_lane_f32::<0>(vsqrt_f32(vdup_n_f32(x)))
    }
    #[archmage::arcane]
    fn hardware64(_token: archmage::NeonToken, x: f64) -> f64 {
        vget_lane_f64::<0>(vsqrt_f64(vdup_n_f64(x)))
    }
}

#[cfg(all(
    not(feature = "std"),
    target_arch = "wasm32",
    target_feature = "simd128"
))]
mod selected {
    use archmage::SimdToken;
    use core::arch::wasm32::*;
    #[inline(always)]
    pub(super) fn sqrtf(x: f32) -> f32 {
        if let Some(token) = archmage::Wasm128Token::summon() {
            hardware32(token, x)
        } else {
            crate::nostd_math::sqrtf(x)
        }
    }
    #[inline(always)]
    pub(super) fn sqrt(x: f64) -> f64 {
        if let Some(token) = archmage::Wasm128Token::summon() {
            hardware64(token, x)
        } else {
            crate::nostd_math::sqrt(x)
        }
    }

    #[archmage::arcane]
    fn hardware32(_token: archmage::Wasm128Token, x: f32) -> f32 {
        f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(x)))
    }
    #[archmage::arcane]
    fn hardware64(_token: archmage::Wasm128Token, x: f64) -> f64 {
        f64x2_extract_lane::<0>(f64x2_sqrt(f64x2_splat(x)))
    }
}

#[cfg(all(
    not(feature = "std"),
    not(any(
        all(target_arch = "x86_64", target_feature = "sse2"),
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128")
    ))
))]
mod selected {
    pub(super) use crate::nostd_math::{sqrt, sqrtf};
}

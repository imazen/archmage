//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! Regenerate with: cargo xtask generate

#[allow(deprecated)]
use crate::tokens::Has128BitSimd;
use crate::tokens::SimdToken;

/// Proof that WASM SIMD128 is available.
#[derive(Clone, Copy, Debug)]
pub struct Wasm128Token {
    _private: (),
}

impl crate::tokens::Sealed for Wasm128Token {}

impl SimdToken for Wasm128Token {
    const NAME: &'static str = "WASM SIMD128";
    const TARGET_FEATURES: &'static str = "simd128";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+simd128";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-simd128";

    #[inline]
    fn compiled_with() -> Option<bool> {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            Some(true)
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            None
        }
    }

    #[allow(deprecated)]
    #[inline]
    fn summon() -> Option<Self> {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            Some(unsafe { Self::forge_token_dangerously() })
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            None
        }
    }

    #[inline(always)]
    #[allow(deprecated)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

/// Proof that WASM Relaxed SIMD is available.
///
/// Relaxed SIMD (Wasm 3.0) provides 28 instructions that trade strict
/// cross-platform determinism for performance: FMA, relaxed lane-select,
/// relaxed min/max, dot products, and relaxed truncation.
///
/// Supported by Chrome 114+, Firefox 145+, Safari 16.4+, and Wasmtime 14+.
/// Stable in Rust since 1.82.
#[derive(Clone, Copy, Debug)]
pub struct Wasm128RelaxedToken {
    _private: (),
}

impl crate::tokens::Sealed for Wasm128RelaxedToken {}

impl SimdToken for Wasm128RelaxedToken {
    const NAME: &'static str = "WASM Relaxed SIMD";
    const TARGET_FEATURES: &'static str = "simd128,relaxed-simd";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+simd128,+relaxed-simd";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-simd128,-relaxed-simd";

    #[inline]
    fn compiled_with() -> Option<bool> {
        #[cfg(all(
            target_arch = "wasm32",
            target_feature = "simd128",
            target_feature = "relaxed-simd"
        ))]
        {
            Some(true)
        }
        #[cfg(not(all(
            target_arch = "wasm32",
            target_feature = "simd128",
            target_feature = "relaxed-simd"
        )))]
        {
            None
        }
    }

    #[allow(deprecated)]
    #[inline]
    fn summon() -> Option<Self> {
        #[cfg(all(
            target_arch = "wasm32",
            target_feature = "simd128",
            target_feature = "relaxed-simd"
        ))]
        {
            Some(unsafe { Self::forge_token_dangerously() })
        }
        #[cfg(not(all(
            target_arch = "wasm32",
            target_feature = "simd128",
            target_feature = "relaxed-simd"
        )))]
        {
            None
        }
    }

    #[inline(always)]
    #[allow(deprecated)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl Wasm128RelaxedToken {
    /// Extract a Wasm128Token — guaranteed because WASM Relaxed SIMD implies WASM SIMD128.
    ///
    /// Zero-cost: compiles away entirely.
    #[allow(deprecated)]
    #[inline(always)]
    pub fn wasm128(self) -> Wasm128Token {
        unsafe { Wasm128Token::forge_token_dangerously() }
    }
}

#[allow(deprecated)]
impl Has128BitSimd for Wasm128Token {}
#[allow(deprecated)]
impl Has128BitSimd for Wasm128RelaxedToken {}

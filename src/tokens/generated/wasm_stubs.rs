//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! Stub tokens: `summon()` always returns `None`.

#[allow(deprecated)]
use crate::tokens::Has128BitSimd;
use crate::tokens::SimdToken;

/// Stub for WASM SIMD128 token (not available on this architecture).
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
        Some(false) // Wrong architecture
    }

    // Note: guaranteed() has a default impl in the trait that calls compiled_with()

    #[inline]
    fn summon() -> Option<Self> {
        None // Not available on this architecture
    }
}

impl Wasm128Token {
    /// Construct a proof for a foreign architecture. Always `unsafe`.
    ///
    /// On this token's native architecture the constructor is
    /// `from_context()`, a **safe** `#[target_feature]` function that
    /// rustc checks against the caller's feature context. The current
    /// compilation target is a different architecture, so no such
    /// context can exist, there is nothing for rustc to check, and no
    /// `from_context()` is generated — only this `unsafe fn`.
    ///
    /// # Safety
    ///
    /// Unsatisfiable: this token asserts CPU features that the target
    /// architecture does not have. Any token produced here is a lie,
    /// and using it to enter a SIMD region is undefined behavior. It
    /// exists so cross-architecture code compiles, not to be called.
    #[deprecated(
        since = "0.9.29",
        note = "Renamed to from_context() on native architectures; on a foreign architecture the features cannot exist at all"
    )]
    #[inline(always)]
    pub unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl Wasm128Token {
    /// This token is not available on this architecture.
    pub fn dangerously_disable_token_process_wide(
        _disabled: bool,
    ) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {
        Err(crate::tokens::CompileTimeGuaranteedError {
            token_name: Self::NAME,
            target_features: Self::TARGET_FEATURES,
            disable_flags: Self::DISABLE_TARGET_FEATURES,
        })
    }

    /// This token is not available on this architecture.
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        Err(crate::tokens::CompileTimeGuaranteedError {
            token_name: Self::NAME,
            target_features: Self::TARGET_FEATURES,
            disable_flags: Self::DISABLE_TARGET_FEATURES,
        })
    }
}

/// Stub for WASM Relaxed SIMD token (not available on this architecture).
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
        Some(false) // Wrong architecture
    }

    // Note: guaranteed() has a default impl in the trait that calls compiled_with()

    #[inline]
    fn summon() -> Option<Self> {
        None // Not available on this architecture
    }
}

impl Wasm128RelaxedToken {
    /// Construct a proof for a foreign architecture. Always `unsafe`.
    ///
    /// On this token's native architecture the constructor is
    /// `from_context()`, a **safe** `#[target_feature]` function that
    /// rustc checks against the caller's feature context. The current
    /// compilation target is a different architecture, so no such
    /// context can exist, there is nothing for rustc to check, and no
    /// `from_context()` is generated — only this `unsafe fn`.
    ///
    /// # Safety
    ///
    /// Unsatisfiable: this token asserts CPU features that the target
    /// architecture does not have. Any token produced here is a lie,
    /// and using it to enter a SIMD region is undefined behavior. It
    /// exists so cross-architecture code compiles, not to be called.
    #[deprecated(
        since = "0.9.29",
        note = "Renamed to from_context() on native architectures; on a foreign architecture the features cannot exist at all"
    )]
    #[inline(always)]
    pub unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl Wasm128RelaxedToken {
    /// This token is not available on this architecture.
    pub fn dangerously_disable_token_process_wide(
        _disabled: bool,
    ) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {
        Err(crate::tokens::CompileTimeGuaranteedError {
            token_name: Self::NAME,
            target_features: Self::TARGET_FEATURES,
            disable_flags: Self::DISABLE_TARGET_FEATURES,
        })
    }

    /// This token is not available on this architecture.
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        Err(crate::tokens::CompileTimeGuaranteedError {
            token_name: Self::NAME,
            target_features: Self::TARGET_FEATURES,
            disable_flags: Self::DISABLE_TARGET_FEATURES,
        })
    }
}

impl Wasm128Token {
    #[doc(hidden)]
    pub const __ARCHMAGE_TIER_TAG: u32 = 0x1E0DF26B;

    #[doc(hidden)]
    pub const __ARCHMAGE_ASSERT_TIER_1E0DF26B: () =
        [()][!(Self::__ARCHMAGE_TIER_TAG == 0x1E0DF26B) as usize];
}

impl Wasm128RelaxedToken {
    #[doc(hidden)]
    pub const __ARCHMAGE_TIER_TAG: u32 = 0x821D5452;

    #[doc(hidden)]
    pub const __ARCHMAGE_ASSERT_TIER_821D5452: () =
        [()][!(Self::__ARCHMAGE_TIER_TAG == 0x821D5452) as usize];
}

#[allow(deprecated)]
impl Has128BitSimd for Wasm128Token {}
#[allow(deprecated)]
impl Has128BitSimd for Wasm128RelaxedToken {}

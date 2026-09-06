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

    #[inline]
    fn summon() -> Option<Self> {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            // SAFETY: the required wasm features are compile-time
            // enabled; a runtime that validated this module supports
            // them (wasm has no runtime feature detection).
            Some(unsafe { Self::from_context() })
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            None
        }
    }
}

impl Wasm128Token {
    /// Construct a WASM SIMD128 proof from the caller's statically proven
    /// feature context.
    ///
    /// Rust permits safe calls to `#[target_feature]` functions from any
    /// WASM context: the engine validates the required instructions when the
    /// module is loaded, so a module that runs at all has the features.
    ///
    /// No runtime detection happens here, so this also bypasses
    /// process-wide token disabling (including `testable_dispatch`):
    /// the caller's feature context is already the proof. Use
    /// [`SimdToken::summon`](crate::SimdToken::summon) when the
    /// features have to be detected at runtime.
    ///
    /// Being a `#[target_feature]` function, this cannot be coerced to
    /// a safe function pointer — there would be no call site left for
    /// rustc to check.
    ///
    /// # Safety
    ///
    /// When called through an `unsafe` block, the caller must ensure
    /// every feature in this tier is available on the executing CPU.
    /// Safe calls have that obligation discharged by the compiler.
    #[inline]
    #[target_feature(enable = "simd128")]
    pub fn from_context() -> Self {
        Self { _private: () }
    }

    /// Deprecated alias for [`Wasm128Token::from_context`].
    ///
    /// Identical in every respect — same `#[target_feature]` gate, same
    /// safe-call rules. The name predates the compiler-checked design
    /// and describes only the `unsafe` half of it.
    ///
    /// # Safety
    ///
    /// Identical to [`Wasm128Token::from_context`]: when called through an
    /// `unsafe` block, the caller must ensure every feature in this
    /// tier is available on the executing CPU. Safe calls have that
    /// obligation discharged by the compiler.
    #[deprecated(
        since = "0.9.29",
        note = "Renamed to from_context() — the constructor is checked against the caller's target-feature context"
    )]
    #[inline]
    #[target_feature(enable = "simd128")]
    pub fn forge_token_dangerously() -> Self {
        // Matching features: a safe call inside the same region.
        Self::from_context()
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

    #[inline]
    fn summon() -> Option<Self> {
        #[cfg(all(
            target_arch = "wasm32",
            target_feature = "simd128",
            target_feature = "relaxed-simd"
        ))]
        {
            // SAFETY: the required wasm features are compile-time
            // enabled; a runtime that validated this module supports
            // them (wasm has no runtime feature detection).
            Some(unsafe { Self::from_context() })
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
}

impl Wasm128RelaxedToken {
    /// Construct a WASM Relaxed SIMD proof from the caller's statically proven
    /// feature context.
    ///
    /// Rust permits safe calls to `#[target_feature]` functions from any
    /// WASM context: the engine validates the required instructions when the
    /// module is loaded, so a module that runs at all has the features.
    ///
    /// No runtime detection happens here, so this also bypasses
    /// process-wide token disabling (including `testable_dispatch`):
    /// the caller's feature context is already the proof. Use
    /// [`SimdToken::summon`](crate::SimdToken::summon) when the
    /// features have to be detected at runtime.
    ///
    /// Being a `#[target_feature]` function, this cannot be coerced to
    /// a safe function pointer — there would be no call site left for
    /// rustc to check.
    ///
    /// # Safety
    ///
    /// When called through an `unsafe` block, the caller must ensure
    /// every feature in this tier is available on the executing CPU.
    /// Safe calls have that obligation discharged by the compiler.
    #[inline]
    #[target_feature(enable = "simd128,relaxed-simd")]
    pub fn from_context() -> Self {
        Self { _private: () }
    }

    /// Deprecated alias for [`Wasm128RelaxedToken::from_context`].
    ///
    /// Identical in every respect — same `#[target_feature]` gate, same
    /// safe-call rules. The name predates the compiler-checked design
    /// and describes only the `unsafe` half of it.
    ///
    /// # Safety
    ///
    /// Identical to [`Wasm128RelaxedToken::from_context`]: when called through an
    /// `unsafe` block, the caller must ensure every feature in this
    /// tier is available on the executing CPU. Safe calls have that
    /// obligation discharged by the compiler.
    #[deprecated(
        since = "0.9.29",
        note = "Renamed to from_context() — the constructor is checked against the caller's target-feature context"
    )]
    #[inline]
    #[target_feature(enable = "simd128,relaxed-simd")]
    pub fn forge_token_dangerously() -> Self {
        // Matching features: a safe call inside the same region.
        Self::from_context()
    }
}

impl Wasm128RelaxedToken {
    /// Extract a Wasm128Token — guaranteed because WASM Relaxed SIMD implies WASM SIMD128.
    ///
    /// Zero-cost: compiles away entirely.
    #[inline(always)]
    pub fn wasm128(self) -> Wasm128Token {
        // SAFETY: holding `self` proves this CPU has WASM Relaxed SIMD's
        // full feature set, a superset of WASM SIMD128's (registry-
        // verified hierarchy), so the ancestor token's claim holds.
        unsafe { Wasm128Token::from_context() }
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

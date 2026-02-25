//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! Regenerate with: cargo xtask generate

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

impl Wasm128Token {
    /// Invoke a closure within this token's `#[target_feature]` context.
    ///
    /// This is the method form of `#[arcane]` — it creates a single
    /// `#[target_feature]` optimization boundary, then calls your closure
    /// with the token inside that boundary.
    ///
    /// Use this when you want `#[arcane]` semantics without proc macros:
    ///
    /// ```rust,ignore
    /// if let Some(token) = Wasm128Token::summon() {
    ///     token.invoke_rite(|t| process_simd(t, data))
    /// }
    /// ```
    ///
    /// Inside the closure, all value-based SIMD intrinsics for this token's
    /// feature set are safe to use (Rust 1.85+).
    #[inline(always)]
    pub fn invoke_rite<F, R>(self, f: F) -> R
    where
        F: FnOnce(Self) -> R,
    {
        #[target_feature(enable = "simd128")]
        unsafe fn __invoke_rite_inner<F, R>(token: Wasm128Token, f: F) -> R
        where
            F: FnOnce(Wasm128Token) -> R,
        {
            f(token)
        }
        // SAFETY: Token existence proves CPU features are available.
        // The token can only be created via summon() which verified CPUID.
        unsafe { __invoke_rite_inner(self, f) }
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
    /// Get a Wasm128Token (WASM Relaxed SIMD implies WASM SIMD128)
    #[allow(deprecated)]
    #[inline(always)]
    pub fn wasm128(self) -> Wasm128Token {
        unsafe { Wasm128Token::forge_token_dangerously() }
    }
}

impl Wasm128RelaxedToken {
    /// Invoke a closure within this token's `#[target_feature]` context.
    ///
    /// This is the method form of `#[arcane]` — it creates a single
    /// `#[target_feature]` optimization boundary, then calls your closure
    /// with the token inside that boundary.
    ///
    /// Use this when you want `#[arcane]` semantics without proc macros:
    ///
    /// ```rust,ignore
    /// if let Some(token) = Wasm128RelaxedToken::summon() {
    ///     token.invoke_rite(|t| process_simd(t, data))
    /// }
    /// ```
    ///
    /// Inside the closure, all value-based SIMD intrinsics for this token's
    /// feature set are safe to use (Rust 1.85+).
    #[inline(always)]
    pub fn invoke_rite<F, R>(self, f: F) -> R
    where
        F: FnOnce(Self) -> R,
    {
        #[target_feature(enable = "simd128,relaxed-simd")]
        unsafe fn __invoke_rite_inner<F, R>(token: Wasm128RelaxedToken, f: F) -> R
        where
            F: FnOnce(Wasm128RelaxedToken) -> R,
        {
            f(token)
        }
        // SAFETY: Token existence proves CPU features are available.
        // The token can only be created via summon() which verified CPUID.
        unsafe { __invoke_rite_inner(self, f) }
    }
}

impl Has128BitSimd for Wasm128Token {}
impl Has128BitSimd for Wasm128RelaxedToken {}

//! SIMD capability tokens
//!
//! Tokens are zero-sized proof types that demonstrate a CPU feature is available.
//! They should be obtained via `summon()` and passed through function calls.
//!
//! ## Token Availability
//!
//! Use `guaranteed()` to check compile-time availability:
//! - `Some(true)` — Compiler guarantees this feature (use `summon().unwrap()`)
//! - `Some(false)` — Wrong architecture (this token can never be available)
//! - `None` — Might be available, call `summon()` to check at runtime
//!
//! ## Cross-Architecture Design
//!
//! All token types exist on all architectures for easier cross-platform code.
//! On unsupported architectures, `summon()` returns `None` and `guaranteed()`
//! returns `Some(false)`.

/// Marker trait for SIMD capability tokens.
///
/// All tokens implement this trait, enabling generic code over different
/// SIMD feature levels.
///
/// # Token Lifecycle
///
/// 1. Optionally check `guaranteed()` to see if runtime check is needed
/// 2. Call `summon()` at runtime to detect CPU support
/// 3. Pass the token through to `#[arcane]` functions — don't forge new ones
///
/// # Example
///
/// ```rust,ignore
/// use archmage::{X64V3Token, SimdToken};
///
/// fn process(data: &[f32]) -> f32 {
///     if let Some(token) = X64V3Token::summon() {
///         return process_avx2(token, data);
///     }
///     process_scalar(data)
/// }
/// ```
pub trait SimdToken: Copy + Clone + Send + Sync + 'static {
    /// Human-readable name for diagnostics and error messages.
    const NAME: &'static str;

    /// Check if this token is compile-time guaranteed.
    ///
    /// Returns:
    /// - `Some(true)` — Feature is guaranteed by target (e.g., `-C target-cpu=haswell`)
    /// - `Some(false)` — Wrong architecture, token can never be available
    /// - `None` — Might be available, call `summon()` to check at runtime
    ///
    /// When `guaranteed()` returns `Some(true)`, `summon().unwrap()` is safe and
    /// the compiler will elide the runtime check entirely.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// match X64V3Token::guaranteed() {
    ///     Some(true) => { /* summon().unwrap() is safe, no runtime check */ }
    ///     Some(false) => { /* use fallback, this arch can't support it */ }
    ///     None => { /* call summon() to check at runtime */ }
    /// }
    /// ```
    fn guaranteed() -> Option<bool>;

    /// Attempt to create a token with runtime feature detection.
    ///
    /// Returns `Some(token)` if the CPU supports this feature, `None` otherwise.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if let Some(token) = X64V3Token::summon() {
    ///     // Use AVX2+FMA operations
    /// } else {
    ///     // Fallback path
    /// }
    /// ```
    fn summon() -> Option<Self>;

    /// Attempt to create a token with runtime feature detection.
    ///
    /// This is an alias for [`summon()`](Self::summon).
    #[inline(always)]
    fn attempt() -> Option<Self> {
        Self::summon()
    }

    /// Legacy alias for [`summon()`](Self::summon).
    ///
    /// **Deprecated:** Use `summon()` instead.
    #[inline(always)]
    #[doc(hidden)]
    fn try_new() -> Option<Self> {
        Self::summon()
    }

    /// Create a token without any checks.
    ///
    /// # Safety
    ///
    /// Caller must guarantee the CPU feature is available. Using a forged token
    /// when the feature is unavailable causes undefined behavior.
    ///
    /// # Deprecated
    ///
    /// **Do not use in new code.** Pass tokens through from `summon()` instead.
    /// If you're inside a `#[cfg(target_feature = "...")]` block where the
    /// feature is compile-time guaranteed, use `summon().unwrap()`.
    #[deprecated(
        since = "0.5.0",
        note = "Pass tokens through from summon() instead of forging"
    )]
    unsafe fn forge_token_dangerously() -> Self;
}

// All token types, traits, and aliases are generated from token-registry.toml.
mod generated;
pub use generated::*;

/// Scalar fallback token — always available on all platforms.
///
/// Use this for fallback paths when no SIMD is available. `ScalarToken`
/// provides type-level proof that "we've given up on SIMD" and allows
/// consistent API shapes in dispatch code.
///
/// # Example
///
/// ```rust
/// use archmage::{ScalarToken, SimdToken};
///
/// // Always succeeds
/// let token = ScalarToken::summon().unwrap();
///
/// // Or use the shorthand
/// let token = ScalarToken;
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScalarToken;

impl SimdToken for ScalarToken {
    const NAME: &'static str = "Scalar";

    /// Always returns `Some(true)` — scalar fallback is always available.
    #[inline(always)]
    fn guaranteed() -> Option<bool> {
        Some(true)
    }

    /// Always returns `Some(ScalarToken)`.
    #[inline(always)]
    fn summon() -> Option<Self> {
        Some(Self)
    }

    #[allow(deprecated)]
    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self
    }
}

/// Trait for compile-time dispatch via monomorphization.
///
/// Each concrete token implements this trait, returning `Some(self)` for its
/// own type and `None` for others. The compiler monomorphizes away all the
/// `None` branches, leaving only the matching path.
///
/// # Example
///
/// ```rust
/// use archmage::{IntoConcreteToken, SimdToken, ScalarToken};
///
/// fn dispatch<T: IntoConcreteToken>(token: T, data: &[f32]) -> f32 {
///     // Compiler eliminates non-matching branches
///     if let Some(_t) = token.as_scalar() {
///         return data.iter().sum();
///     }
///     // ... other paths for x64v3, neon, etc.
///     0.0
/// }
///
/// let result = dispatch(ScalarToken, &[1.0, 2.0, 3.0]);
/// assert_eq!(result, 6.0);
/// ```
pub trait IntoConcreteToken: SimdToken + Sized {
    /// Try to cast to X64V2Token.
    #[inline(always)]
    fn as_x64v2(self) -> Option<X64V2Token> {
        None
    }

    /// Try to cast to X64V3Token.
    #[inline(always)]
    fn as_x64v3(self) -> Option<X64V3Token> {
        None
    }

    /// Try to cast to X64V4Token (requires `avx512` feature).
    #[cfg(feature = "avx512")]
    #[inline(always)]
    fn as_x64v4(self) -> Option<X64V4Token> {
        None
    }

    /// Try to cast to NeonToken.
    #[inline(always)]
    fn as_neon(self) -> Option<NeonToken> {
        None
    }

    /// Try to cast to Simd128Token.
    #[inline(always)]
    fn as_wasm128(self) -> Option<Simd128Token> {
        None
    }

    /// Try to cast to ScalarToken.
    #[inline(always)]
    fn as_scalar(self) -> Option<ScalarToken> {
        None
    }
}

// Implement IntoConcreteToken for ScalarToken
impl IntoConcreteToken for ScalarToken {
    #[inline(always)]
    fn as_scalar(self) -> Option<ScalarToken> {
        Some(self)
    }
}

// Implement IntoConcreteToken for X64V2Token
impl IntoConcreteToken for X64V2Token {
    #[inline(always)]
    fn as_x64v2(self) -> Option<X64V2Token> {
        Some(self)
    }
}

// Implement IntoConcreteToken for X64V3Token
impl IntoConcreteToken for X64V3Token {
    #[inline(always)]
    fn as_x64v3(self) -> Option<X64V3Token> {
        Some(self)
    }
}

// Implement IntoConcreteToken for X64V4Token
#[cfg(feature = "avx512")]
impl IntoConcreteToken for X64V4Token {
    #[cfg(feature = "avx512")]
    #[inline(always)]
    fn as_x64v4(self) -> Option<X64V4Token> {
        Some(self)
    }
}

// Implement IntoConcreteToken for NeonToken
impl IntoConcreteToken for NeonToken {
    #[inline(always)]
    fn as_neon(self) -> Option<NeonToken> {
        Some(self)
    }
}

// Implement IntoConcreteToken for Simd128Token
impl IntoConcreteToken for Simd128Token {
    #[inline(always)]
    fn as_wasm128(self) -> Option<Simd128Token> {
        Some(self)
    }
}

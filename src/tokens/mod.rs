//! SIMD capability tokens
//!
//! Tokens are zero-sized proof types that demonstrate a CPU feature is available.
//! They should be obtained via `summon()` and passed through function calls.
//!
//! ## Token Availability
//!
//! Use `compiled_with()` to check compile-time availability:
//! - `Some(true)` — Binary was compiled with these features enabled (use `summon().unwrap()`)
//! - `Some(false)` — Wrong architecture (this token can never be available)
//! - `None` — Might be available, call `summon()` to check at runtime
//!
//! ## Cross-Architecture Design
//!
//! All token types exist on all architectures for easier cross-platform code.
//! On unsupported architectures, `summon()` returns `None` and `compiled_with()`
//! returns `Some(false)`.
//!
//! ## Disabling Tokens
//!
//! For testing and benchmarking, tokens can be disabled process-wide:
//!
//! ```rust,ignore
//! // Force scalar fallback for benchmarking
//! X64V3Token::dangerously_disable_token_process_wide(true)?;
//! // All summon() calls now return None (cascades to V4, Modern, Fp16)
//! assert!(X64V3Token::summon().is_none());
//! ```
//!
//! Disabling returns `Err(CompileTimeGuaranteedError)` when the features are
//! compile-time enabled (e.g., via `-Ctarget-cpu=native`), since the compiler
//! has already elided the runtime checks.

mod sealed {
    /// Sealed trait preventing external implementations of [`SimdToken`](super::SimdToken).
    pub trait Sealed {}
}

pub use sealed::Sealed;

/// Marker trait for SIMD capability tokens.
///
/// All tokens implement this trait, enabling generic code over different
/// SIMD feature levels.
///
/// This trait is **sealed** — it cannot be implemented outside of archmage.
/// Only tokens created by `summon()` or downcasting are valid.
///
/// # Token Lifecycle
///
/// 1. Optionally check `compiled_with()` to see if runtime check is needed
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
pub trait SimdToken: sealed::Sealed + Copy + Clone + Send + Sync + 'static {
    /// Human-readable name for diagnostics and error messages.
    const NAME: &'static str;

    /// Check if this binary was compiled with the required target features enabled.
    ///
    /// Returns:
    /// - `Some(true)` — Features are compile-time enabled (e.g., `-C target-cpu=haswell`)
    /// - `Some(false)` — Wrong architecture, token can never be available
    /// - `None` — Might be available, call `summon()` to check at runtime
    ///
    /// When `compiled_with()` returns `Some(true)`, `summon().unwrap()` is safe and
    /// the compiler will elide the runtime check entirely.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// match X64V3Token::compiled_with() {
    ///     Some(true) => { /* summon().unwrap() is safe, no runtime check */ }
    ///     Some(false) => { /* use fallback, this arch can't support it */ }
    ///     None => { /* call summon() to check at runtime */ }
    /// }
    /// ```
    fn compiled_with() -> Option<bool>;

    /// Deprecated alias for [`compiled_with()`](Self::compiled_with).
    #[inline(always)]
    #[deprecated(since = "0.6.0", note = "Use compiled_with() instead")]
    fn guaranteed() -> Option<bool> {
        Self::compiled_with()
    }

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

impl sealed::Sealed for ScalarToken {}

impl SimdToken for ScalarToken {
    const NAME: &'static str = "Scalar";

    /// Always returns `Some(true)` — scalar fallback is always available.
    #[inline(always)]
    fn compiled_with() -> Option<bool> {
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

impl ScalarToken {
    /// Scalar tokens cannot be disabled (they are always available).
    ///
    /// Always returns `Err(CompileTimeGuaranteedError)` because `ScalarToken`
    /// is unconditionally available — there is no runtime check to bypass.
    pub fn dangerously_disable_token_process_wide(
        _disabled: bool,
    ) -> Result<(), CompileTimeGuaranteedError> {
        Err(CompileTimeGuaranteedError {
            token_name: Self::NAME,
        })
    }

    /// Scalar tokens cannot be disabled (they are always available).
    ///
    /// Always returns `Err(CompileTimeGuaranteedError)`.
    pub fn manually_disabled() -> Result<bool, CompileTimeGuaranteedError> {
        Err(CompileTimeGuaranteedError {
            token_name: Self::NAME,
        })
    }
}

/// Error returned when attempting to disable a token whose features are
/// compile-time enabled.
///
/// When all required features are enabled via `-Ctarget-cpu` or
/// `-Ctarget-feature`, the compiler has already elided the runtime
/// detection — `summon()` unconditionally returns `Some`. Disabling
/// the token would have no effect and silently produce incorrect behavior.
///
/// # Resolution
///
/// To use `dangerously_disable_token_process_wide`, compile without
/// the target features enabled. For example:
/// - Remove `-Ctarget-cpu=native` from `RUSTFLAGS`
/// - Use `-Ctarget-feature=-avx2` to disable specific features
#[derive(Debug, Clone)]
pub struct CompileTimeGuaranteedError {
    /// The token type that cannot be disabled.
    pub token_name: &'static str,
}

impl core::fmt::Display for CompileTimeGuaranteedError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Cannot disable {} — all required features are compile-time enabled. \
             Remove `-Ctarget-cpu` from RUSTFLAGS, or use \
             `-Ctarget-feature=-<feature>` to disable specific features.",
            self.token_name
        )
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CompileTimeGuaranteedError {}

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

    /// Try to cast to Avx512ModernToken (requires `avx512` feature).
    #[cfg(feature = "avx512")]
    #[inline(always)]
    fn as_avx512_modern(self) -> Option<Avx512ModernToken> {
        None
    }

    /// Try to cast to Avx512Fp16Token (requires `avx512` feature).
    #[cfg(feature = "avx512")]
    #[inline(always)]
    fn as_avx512_fp16(self) -> Option<Avx512Fp16Token> {
        None
    }

    /// Try to cast to NeonToken.
    #[inline(always)]
    fn as_neon(self) -> Option<NeonToken> {
        None
    }

    /// Try to cast to NeonAesToken.
    #[inline(always)]
    fn as_neon_aes(self) -> Option<NeonAesToken> {
        None
    }

    /// Try to cast to NeonSha3Token.
    #[inline(always)]
    fn as_neon_sha3(self) -> Option<NeonSha3Token> {
        None
    }

    /// Try to cast to NeonCrcToken.
    #[inline(always)]
    fn as_neon_crc(self) -> Option<NeonCrcToken> {
        None
    }

    /// Try to cast to Wasm128Token.
    #[inline(always)]
    fn as_wasm128(self) -> Option<Wasm128Token> {
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
    #[inline(always)]
    fn as_x64v4(self) -> Option<X64V4Token> {
        Some(self)
    }
}

// Implement IntoConcreteToken for Avx512ModernToken
#[cfg(feature = "avx512")]
impl IntoConcreteToken for Avx512ModernToken {
    #[inline(always)]
    fn as_avx512_modern(self) -> Option<Avx512ModernToken> {
        Some(self)
    }
}

// Implement IntoConcreteToken for Avx512Fp16Token
#[cfg(feature = "avx512")]
impl IntoConcreteToken for Avx512Fp16Token {
    #[inline(always)]
    fn as_avx512_fp16(self) -> Option<Avx512Fp16Token> {
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

// Implement IntoConcreteToken for NeonAesToken
impl IntoConcreteToken for NeonAesToken {
    #[inline(always)]
    fn as_neon_aes(self) -> Option<NeonAesToken> {
        Some(self)
    }
}

// Implement IntoConcreteToken for NeonSha3Token
impl IntoConcreteToken for NeonSha3Token {
    #[inline(always)]
    fn as_neon_sha3(self) -> Option<NeonSha3Token> {
        Some(self)
    }
}

// Implement IntoConcreteToken for NeonCrcToken
impl IntoConcreteToken for NeonCrcToken {
    #[inline(always)]
    fn as_neon_crc(self) -> Option<NeonCrcToken> {
        Some(self)
    }
}

// Implement IntoConcreteToken for Wasm128Token
impl IntoConcreteToken for Wasm128Token {
    #[inline(always)]
    fn as_wasm128(self) -> Option<Wasm128Token> {
        Some(self)
    }
}

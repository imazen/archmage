//! SIMD capability tokens
//!
//! Tokens are zero-sized proof types that demonstrate a CPU feature is available.
//! They should be obtained via `summon()` and passed through function calls.
//!
//! ## Token Availability
//!
//! Use `guaranteed()` to check compile-time availability:
//! - `Some(true)` — Compiler guarantees this feature (use directly, no runtime check)
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
/// 1. Check `guaranteed()` at compile time to see if runtime check is needed
/// 2. If `None`, call `summon()` at runtime to detect CPU support
/// 3. Pass the token through to `#[arcane]` functions — don't forge new ones
///
/// # Example
///
/// ```rust,ignore
/// use archmage::{X64V3Token, SimdToken};
///
/// fn process(data: &[f32]) -> f32 {
///     // Check compile-time availability first
///     match X64V3Token::guaranteed() {
///         Some(true) => {
///             // Compiler guarantees AVX2+FMA — no runtime check needed!
///             let token = X64V3Token::conjure();
///             return process_avx2(token, data);
///         }
///         Some(false) => {
///             // Wrong architecture — use scalar
///             return process_scalar(data);
///         }
///         None => {
///             // Need runtime check
///             if let Some(token) = X64V3Token::summon() {
///                 return process_avx2(token, data);
///             }
///             return process_scalar(data);
///         }
///     }
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
    /// # Example
    ///
    /// ```rust,ignore
    /// match X64V3Token::guaranteed() {
    ///     Some(true) => { /* use conjure(), skip runtime check */ }
    ///     Some(false) => { /* use fallback */ }
    ///     None => { /* call summon() */ }
    /// }
    /// ```
    fn guaranteed() -> Option<bool>;

    /// Conjure a token when `guaranteed()` returns `Some(true)`.
    ///
    /// This is the safe way to create a token when the compiler guarantees
    /// the feature is available (via `-C target-cpu` or `#[target_feature]`).
    ///
    /// # Panics
    ///
    /// Panics if called when `guaranteed()` does not return `Some(true)`.
    /// Use `summon()` for runtime detection instead.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Only valid when compiled with -C target-cpu=haswell or similar
    /// if X64V3Token::guaranteed() == Some(true) {
    ///     let token = X64V3Token::conjure();
    ///     // Use token...
    /// }
    /// ```
    #[inline(always)]
    fn conjure() -> Self {
        match Self::guaranteed() {
            Some(true) => {
                // SAFETY: guaranteed() == Some(true) means the feature is
                // compile-time guaranteed by target_feature
                #[allow(deprecated)]
                unsafe { Self::forge_token_dangerously() }
            }
            Some(false) => {
                panic!(
                    "Cannot conjure {} on this architecture (guaranteed() = Some(false))",
                    Self::NAME
                );
            }
            None => {
                panic!(
                    "Cannot conjure {} without compile-time guarantee. Use summon() instead.",
                    Self::NAME
                );
            }
        }
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
    /// # Deprecated
    ///
    /// **Do not use.** Pass tokens through from `summon()` or `conjure()` instead.
    ///
    /// If you need a token inside a `#[target_feature]` function, use `conjure()`
    /// when `guaranteed() == Some(true)`, or accept the token as a parameter.
    ///
    /// # Safety
    ///
    /// Caller must guarantee the CPU feature is available. Using a forged token
    /// when the feature is unavailable causes undefined behavior.
    #[deprecated(
        since = "0.5.0",
        note = "Pass tokens through from summon() or conjure() instead of forging"
    )]
    unsafe fn forge_token_dangerously() -> Self;
}

// All token types, traits, and aliases are generated from token-registry.toml.
mod generated;
pub use generated::*;

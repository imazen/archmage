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

//! SIMD capability tokens
//!
//! Tokens are zero-sized proof types that demonstrate a CPU feature is available.
//! They can only be constructed via unsafe `forge_token_dangerously()` or fallible `try_new()`.
//!
//! ## Cross-Architecture Availability
//!
//! All token types are available on all architectures for easier cross-platform code.
//! However, `summon()` / `try_new()` will return `None` on unsupported architectures.
//! Rust's type system ensures intrinsic methods don't exist on the wrong arch,
//! so you get compile errors if you try to use them incorrectly.

/// Marker trait for SIMD capability tokens.
///
/// All tokens implement this trait, enabling generic code over different
/// SIMD feature levels.
///
/// # Safety
///
/// Implementors must ensure that `forge_token_dangerously()` is only called when
/// the corresponding CPU feature is actually available.
pub trait SimdToken: Copy + Clone + Send + Sync + 'static {
    /// Human-readable name for diagnostics and error messages.
    const NAME: &'static str;

    /// Attempt to create a token with runtime feature detection.
    ///
    /// Returns `Some(token)` if the CPU supports this feature, `None` otherwise.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if let Some(token) = X64V3Token::try_new() {
    ///     // Use AVX2+FMA operations
    /// } else {
    ///     // Fallback path
    /// }
    /// ```
    fn try_new() -> Option<Self>;

    /// Attempt to create a token with runtime feature detection.
    ///
    /// This is an alias for [`try_new()`](Self::try_new).
    #[inline(always)]
    fn attempt() -> Option<Self> {
        Self::try_new()
    }

    /// Summon a token if the CPU supports this feature.
    ///
    /// This is a thematic alias for [`try_new()`](Self::try_new). Summoning may fail
    /// if the required power (CPU features) is not available.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use archmage::{Desktop64, SimdToken};
    ///
    /// if let Some(token) = Desktop64::summon() {
    ///     // The power is yours
    /// }
    /// ```
    #[inline(always)]
    fn summon() -> Option<Self> {
        Self::try_new()
    }

    /// Create a token without runtime checks.
    ///
    /// # Safety
    ///
    /// Caller must guarantee the CPU feature is available via one of:
    /// - Runtime detection (`is_x86_feature_detected!` or equivalent)
    /// - Compile-time guarantee (`#[target_feature]` attribute)
    /// - Target CPU specification (`-C target-cpu=native` or similar)
    /// - Being inside a multiversioned function variant
    ///
    /// # Why "forge_token_dangerously"?
    ///
    /// The name is intentionally scary to discourage casual use. Forging a token
    /// for a feature that isn't actually available leads to undefined behavior
    /// (illegal instructions, crashes, or silent data corruption).
    unsafe fn forge_token_dangerously() -> Self;
}

// All token types, traits, and aliases are generated from token-registry.toml.
mod generated;
pub use generated::*;

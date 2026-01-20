//! SIMD capability tokens
//!
//! Tokens are zero-sized proof types that demonstrate a CPU feature is available.
//! They can only be constructed via unsafe `forge_token_dangerously()` or fallible `try_new()`.

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod arm;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

// Re-export platform-specific tokens
#[cfg(target_arch = "x86_64")]
pub use x86::*;

#[cfg(target_arch = "aarch64")]
pub use arm::*;

#[cfg(target_arch = "wasm32")]
pub use wasm::*;

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
    /// if let Some(token) = Avx2Token::try_new() {
    ///     // Use AVX2 operations
    /// } else {
    ///     // Fallback path
    /// }
    /// ```
    fn try_new() -> Option<Self>;

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

/// Trait for tokens that can be decomposed into sub-tokens.
///
/// Combined tokens like `Avx2FmaToken` implement this to provide
/// access to their component tokens.
pub trait CompositeToken: SimdToken {
    /// The component token types
    type Components;

    /// Decompose into component tokens
    fn components(&self) -> Self::Components;
}

// ============================================================================
// Capability Marker Traits
// ============================================================================
//
// These traits indicate what capabilities a token provides, enabling generic
// code to constrain which tokens are accepted. They don't provide operations
// directly - use raw intrinsics via `#[simd_fn]` for that.

/// Marker trait for tokens that provide 128-bit SIMD.
///
/// Implemented by: `Sse2Token`, `NeonToken`, `Simd128Token`
pub trait Has128BitSimd: SimdToken {}

/// Marker trait for tokens that provide 256-bit SIMD.
///
/// Implemented by: `Avx2Token`, `X64V3Token`, etc.
pub trait Has256BitSimd: Has128BitSimd {}

/// Marker trait for tokens that provide 512-bit SIMD.
///
/// Implemented by: `Avx512fToken`, `X64V4Token`
pub trait Has512BitSimd: Has256BitSimd {}

/// Marker trait for tokens that provide FMA (fused multiply-add).
///
/// Implemented by: `FmaToken`, `Avx2FmaToken`, `X64V3Token`, `NeonToken`
pub trait HasFma: SimdToken {}

/// Marker trait for tokens that provide scalable vectors (variable width).
///
/// Implemented by: `SveToken`, `Sve2Token`
pub trait HasScalableVectors: SimdToken {}

// ============================================================================
// x86 Feature Marker Traits
// ============================================================================
//
// These form a hierarchy matching the x86 feature dependencies.
// Use these as bounds on generic functions to accept any token that
// implies a specific feature.

/// Marker trait for tokens that provide SSE.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub trait HasSse: SimdToken {}

/// Marker trait for tokens that provide SSE2.
///
/// SSE2 is baseline on x86_64.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub trait HasSse2: HasSse {}

/// Marker trait for tokens that provide SSE4.1.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub trait HasSse41: HasSse2 {}

/// Marker trait for tokens that provide SSE4.2.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub trait HasSse42: HasSse41 {}

/// Marker trait for tokens that provide AVX.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub trait HasAvx: HasSse42 {}

/// Marker trait for tokens that provide AVX2.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub trait HasAvx2: HasAvx {}

/// Marker trait for tokens that provide AVX-512F (Foundation).
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub trait HasAvx512f: HasAvx2 {}

/// Marker trait for tokens that provide AVX-512VL (Vector Length extensions).
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub trait HasAvx512vl: HasAvx512f {}

/// Marker trait for tokens that provide AVX-512BW (Byte/Word).
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub trait HasAvx512bw: HasAvx512f {}

/// Marker trait for tokens that provide AVX-512VBMI2.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub trait HasAvx512vbmi2: HasAvx512bw {}

// ============================================================================
// AArch64 Feature Marker Traits
// ============================================================================

/// Marker trait for tokens that provide NEON.
///
/// NEON is baseline on AArch64.
#[cfg(target_arch = "aarch64")]
pub trait HasNeon: SimdToken {}

/// Marker trait for tokens that provide SVE.
#[cfg(target_arch = "aarch64")]
pub trait HasSve: HasNeon {}

/// Marker trait for tokens that provide SVE2.
#[cfg(target_arch = "aarch64")]
pub trait HasSve2: HasSve {}

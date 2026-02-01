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

// Platform-specific implementations
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub mod x86;
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "avx512"))]
pub mod x86_avx512;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub use x86::*;

#[cfg(target_arch = "aarch64")]
pub mod arm;
#[cfg(target_arch = "aarch64")]
pub use arm::*;

#[cfg(target_arch = "wasm32")]
pub mod wasm;
#[cfg(target_arch = "wasm32")]
pub use wasm::*;

// Cross-platform stubs - define types that return None on wrong arch
#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
pub mod x86_stubs;
#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
pub use x86_stubs::*;

#[cfg(not(target_arch = "aarch64"))]
pub mod arm_stubs;
#[cfg(not(target_arch = "aarch64"))]
pub use arm_stubs::*;

#[cfg(not(target_arch = "wasm32"))]
pub mod wasm_stubs;
#[cfg(not(target_arch = "wasm32"))]
pub use wasm_stubs::*;

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

// ============================================================================
// Width Marker Traits
// ============================================================================
//
// These traits indicate what vector widths a token provides.

/// Marker trait for tokens that provide 128-bit SIMD.
pub trait Has128BitSimd: SimdToken {}

/// Marker trait for tokens that provide 256-bit SIMD.
pub trait Has256BitSimd: Has128BitSimd {}

/// Marker trait for tokens that provide 512-bit SIMD.
pub trait Has512BitSimd: Has256BitSimd {}

// ============================================================================
// x86 Tier Marker Traits
// ============================================================================
//
// Based on LLVM x86-64 microarchitecture levels (psABI).
// v1 (baseline): No trait needed - SSE2 always available on x86_64
// v3: No trait - use X64V3Token / Avx2FmaToken / Desktop64 directly
//
// Available on all architectures for cross-platform code.

/// Marker trait for x86-64-v2 level (Nehalem 2008+).
///
/// v2 includes: SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT, CMPXCHG16B, LAHF-SAHF.
pub trait HasX64V2: SimdToken {}

/// Marker trait for x86-64-v4 level (Skylake-X 2017+, Zen 4 2022+).
///
/// v4 includes all of v3 plus: AVX512F, AVX512BW, AVX512CD, AVX512DQ, AVX512VL.
/// Implies HasX64V2.
pub trait HasX64V4: HasX64V2 {}

// ============================================================================
// AArch64 Tier Marker Traits
// ============================================================================
//
// NEON is baseline on AArch64 - no runtime detection needed.
// Extensions like AES and SHA3 are optional.
//
// Available on all architectures for cross-platform code.

/// Marker trait for NEON (baseline on AArch64).
///
/// NEON is always available on AArch64.
pub trait HasNeon: SimdToken {}

/// Marker trait for NEON + AES.
///
/// AES extension is common on modern ARM64 devices (ARMv8-A with Crypto).
pub trait HasNeonAes: HasNeon {}

/// Marker trait for NEON + SHA3.
///
/// SHA3 extension is available on ARMv8.2-A and later.
pub trait HasNeonSha3: HasNeon {}

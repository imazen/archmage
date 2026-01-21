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
//!
//! ## Disabling SIMD (`disable-archmage` feature)
//!
//! Enable the `disable-archmage` feature to force all `summon()` calls to return `None`:
//!
//! ```toml
//! [dependencies]
//! archmage = { version = "0.1", features = ["disable-archmage"] }
//! ```
//!
//! This is useful for:
//! - Testing scalar fallback code paths
//! - Benchmarking SIMD vs scalar performance
//! - Debugging issues that might be SIMD-related
//!
//! Note: `try_new()` is unaffected and still performs actual detection. Use `summon()`
//! for code that should respect the disable flag.

/// Private module for sealing marker traits.
///
/// Marker traits like `HasAvx2` are sealed to prevent external implementations.
/// Only token types defined in this crate can implement them.
pub(crate) mod sealed {
    /// Sealed trait - cannot be implemented outside this crate.
    pub trait Sealed {}
}

// Token definition macros (must be before platform modules that use them)
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[macro_use]
mod macros;

// Platform-specific implementations
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub mod x86;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub mod x86_avx512;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub use x86::*;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub use x86_avx512::*;

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
    /// # Feature: `disable-archmage`
    ///
    /// When the `disable-archmage` feature is enabled, this always returns `None`.
    /// Useful for testing scalar fallback paths or benchmarking without SIMD.
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
        #[cfg(feature = "disable-archmage")]
        {
            None
        }
        #[cfg(not(feature = "disable-archmage"))]
        {
            Self::try_new()
        }
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
// Capability Marker Traits (Sealed)
// ============================================================================
//
// These traits indicate what capabilities a token provides, enabling generic
// code to constrain which tokens are accepted. They don't provide operations
// directly - use raw intrinsics via `#[simd_fn]` for that.
//
// All marker traits are SEALED - they cannot be implemented outside this crate.
// This ensures that only archmage-defined tokens can claim CPU feature support.

/// Marker trait for tokens that provide 128-bit SIMD.
///
/// Implemented by: `Sse2Token`, `NeonToken`, `Simd128Token`
///
/// This trait is sealed and cannot be implemented outside this crate.
pub trait Has128BitSimd: SimdToken + sealed::Sealed {}

/// Marker trait for tokens that provide 256-bit SIMD.
///
/// Implemented by: `Avx2Token`, `X64V3Token`, etc.
///
/// This trait is sealed and cannot be implemented outside this crate.
pub trait Has256BitSimd: Has128BitSimd {}

/// Marker trait for tokens that provide 512-bit SIMD.
///
/// Implemented by: `Avx512fToken`, `X64V4Token`
///
/// This trait is sealed and cannot be implemented outside this crate.
pub trait Has512BitSimd: Has256BitSimd {}

/// Marker trait for tokens that provide FMA (fused multiply-add).
///
/// Implemented by: `FmaToken`, `Avx2FmaToken`, `X64V3Token`, `NeonToken`
///
/// This trait is sealed and cannot be implemented outside this crate.
pub trait HasFma: SimdToken + sealed::Sealed {}

// ============================================================================
// x86 Feature Marker Traits (Sealed)
// ============================================================================
//
// These form a hierarchy matching the x86 feature dependencies.
// Use these as bounds on generic functions to accept any token that
// implies a specific feature.
//
// Available on all architectures for cross-platform code.
// All traits are sealed - cannot be implemented outside this crate.
//
// NOTE: SSE4.2 is the baseline. HasSse, HasSse2, HasSse41 have been removed.
// All x86 tokens assume at least SSE4.2 is available.

/// Marker trait for tokens that provide SSE4.2.
///
/// SSE4.2 is the practical baseline for archmage. All tokens that provide
/// any x86 SIMD capability also provide SSE4.2.
///
/// This trait is sealed and cannot be implemented outside this crate.
pub trait HasSse42: SimdToken + sealed::Sealed {}

/// Marker trait for tokens that provide AVX.
///
/// AVX implies SSE4.2.
///
/// This trait is sealed and cannot be implemented outside this crate.
pub trait HasAvx: HasSse42 {}

/// Marker trait for tokens that provide AVX2.
///
/// This trait is sealed and cannot be implemented outside this crate.
pub trait HasAvx2: HasAvx {}

/// Marker trait for tokens that provide AVX-512F (Foundation).
///
/// This trait is sealed and cannot be implemented outside this crate.
pub trait HasAvx512f: HasAvx2 {}

/// Marker trait for tokens that provide AVX-512VL (Vector Length extensions).
///
/// This trait is sealed and cannot be implemented outside this crate.
pub trait HasAvx512vl: HasAvx512f {}

/// Marker trait for tokens that provide AVX-512BW (Byte/Word).
///
/// This trait is sealed and cannot be implemented outside this crate.
pub trait HasAvx512bw: HasAvx512f {}

/// Marker trait for tokens that provide AVX-512CD (Conflict Detection).
///
/// AVX-512CD provides conflict detection for scatter operations.
/// Present on all practical AVX-512 CPUs (Skylake-X 2017+, Zen 4+).
///
/// This trait is sealed and cannot be implemented outside this crate.
pub trait HasAvx512cd: HasAvx512f {}

/// Marker trait for tokens that provide AVX-512DQ (Doubleword/Quadword).
///
/// AVX-512DQ provides float bitwise ops (`_mm512_or_ps`, etc.) and conversions.
/// Present on all practical AVX-512 CPUs (Skylake-X 2017+, Zen 4+).
///
/// This trait is sealed and cannot be implemented outside this crate.
pub trait HasAvx512dq: HasAvx512f {}

/// Marker trait for tokens that provide AVX-512VBMI2.
///
/// This trait is sealed and cannot be implemented outside this crate.
pub trait HasAvx512vbmi2: HasAvx512bw {}

// ============================================================================
// AArch64 Feature Marker Traits (Sealed)
// ============================================================================
//
// Available on all architectures for cross-platform code.
// All traits are sealed - cannot be implemented outside this crate.

/// Marker trait for tokens that provide NEON.
///
/// NEON is baseline on AArch64.
///
/// This trait is sealed and cannot be implemented outside this crate.
pub trait HasNeon: SimdToken + sealed::Sealed {}

/// Marker trait for tokens that provide ARM AES crypto extensions.
///
/// AES + SHA2 + CRC - available on most ARMv8 CPUs (Cortex-A53+, Graviton, Apple Silicon).
///
/// This trait is sealed and cannot be implemented outside this crate.
pub trait HasArmAes: HasNeon {}

/// Marker trait for tokens that provide ARM SHA3 crypto extensions.
///
/// SHA3 - available on ARMv8.4+ CPUs (Cortex-A76+, Graviton 2+, Apple Silicon).
///
/// This trait is sealed and cannot be implemented outside this crate.
pub trait HasArmSha3: HasArmAes {}

/// Marker trait for tokens that provide ARM FP16 (half-precision floating point).
///
/// FP16 - available on modern ARM (Apple M1+, Graviton 2+, Cortex-A76+).
///
/// This trait is sealed and cannot be implemented outside this crate.
pub trait HasArmFp16: HasNeon {}

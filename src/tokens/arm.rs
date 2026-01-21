//! ARM SIMD capability tokens
//!
//! Provides tokens for NEON and crypto extensions.
//!
//! Token construction uses [`crate::is_aarch64_feature_available!`] which combines
//! compile-time and runtime detection. When compiled with a target feature
//! enabled (e.g., in a `#[multiversed]` function), the runtime check is
//! completely eliminated.
//!
//! On AArch64, NEON is always available (baseline).
//!
//! ## Token Hierarchy
//!
//! - `NeonToken` - baseline, always available
//! - `NeonAesToken` - aes + sha2 + crc (most ARMv8 CPUs)
//! - `NeonSha3Token` - sha3 (ARMv8.4+, orthogonal to AES)
//! - `NeonFp16Token` - fp16 half-precision (Apple M1+, Graviton 2+)
//!
//! Note: AES and SHA3 are orthogonal features - having one does not imply the other.

use super::SimdToken;

// ============================================================================
// NEON Token
// ============================================================================

/// Proof that NEON is available.
///
/// NEON is the baseline SIMD for AArch64 - always available on 64-bit ARM.
#[derive(Clone, Copy, Debug)]
pub struct NeonToken {
    _private: (),
}

impl SimdToken for NeonToken {
    const NAME: &'static str = "NEON";

    #[inline]
    fn try_new() -> Option<Self> {
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is always available on AArch64
            Some(Self { _private: () })
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            None
        }
    }

    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

// ============================================================================
// NEON AES Token (aes + sha2 + crc)
// ============================================================================

/// Proof that ARM crypto extensions are available (AES, SHA2, CRC32).
///
/// This covers the baseline ARMv8 crypto extensions found on most AArch64 CPUs:
/// - Cortex-A53/A55/A72/A73/A75/A76/A77/A78/X1
/// - Graviton 1/2/3/4
/// - Apple M1/M2/M3
/// - Snapdragon 8xx series
#[derive(Clone, Copy, Debug)]
pub struct NeonAesToken {
    _private: (),
}

impl SimdToken for NeonAesToken {
    const NAME: &'static str = "NEON+AES";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        // All three features must be available
        if crate::is_aarch64_feature_available!("aes")
            && crate::is_aarch64_feature_available!("sha2")
            && crate::is_aarch64_feature_available!("crc")
        {
            Some(unsafe { Self::forge_token_dangerously() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl NeonAesToken {
    /// Get a NEON token (crypto implies NEON)
    #[inline(always)]
    pub fn neon(self) -> NeonToken {
        unsafe { NeonToken::forge_token_dangerously() }
    }
}


// ============================================================================
// NEON SHA3 Token
// ============================================================================

/// Proof that ARM SHA3 extensions are available.
///
/// SHA3 provides hardware-accelerated SHA3 hashing on ARMv8.2+ CPUs:
/// - Neoverse N2/V1/V2 (Graviton 2/3/4)
/// - Apple M1/M2/M3
/// - Snapdragon 8 Gen 1+
/// - Cortex-A710/A715/X2/X3
///
/// Note: SHA3 is orthogonal to AES - having one does not imply the other.
/// Use `NeonAesToken` separately if you need AES.
#[derive(Clone, Copy, Debug)]
pub struct NeonSha3Token {
    _private: (),
}

impl SimdToken for NeonSha3Token {
    const NAME: &'static str = "NEON+SHA3";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        if crate::is_aarch64_feature_available!("sha3") {
            Some(unsafe { Self::forge_token_dangerously() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl NeonSha3Token {
    /// Get a NEON token (SHA3 implies NEON)
    #[inline(always)]
    pub fn neon(self) -> NeonToken {
        unsafe { NeonToken::forge_token_dangerously() }
    }
}

// ============================================================================
// NEON FP16 Token
// ============================================================================

/// Proof that ARM FP16 (half-precision floating point) is available.
///
/// FP16 provides native half-precision (16-bit) floating point operations,
/// useful for ML inference and graphics workloads.
///
/// Available on modern ARM CPUs:
/// - Apple M1/M2/M3/M4
/// - AWS Graviton 2/3/4
/// - Cortex-A76/A77/A78/X1 and newer
/// - Snapdragon 8xx series (855+)
///
/// Note: Not available on older ARMv8.0-8.1 cores (Cortex-A53/A55/A72, Graviton 1).
#[derive(Clone, Copy, Debug)]
pub struct NeonFp16Token {
    _private: (),
}

impl SimdToken for NeonFp16Token {
    const NAME: &'static str = "NEON+FP16";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        if crate::is_aarch64_feature_available!("fp16") {
            Some(unsafe { Self::forge_token_dangerously() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl NeonFp16Token {
    /// Get a NEON token (FP16 implies NEON)
    #[inline(always)]
    pub fn neon(self) -> NeonToken {
        unsafe { NeonToken::forge_token_dangerously() }
    }
}

// ============================================================================
// Friendly Aliases
// ============================================================================

/// The recommended starting point for AArch64 (NEON + FP16).
///
/// This is an alias for [`NeonFp16Token`], targeting modern ARM CPUs with
/// half-precision floating point support. This covers:
/// - Apple M1/M2/M3/M4
/// - AWS Graviton 2/3/4
/// - Cortex-A76+ (most phones from 2019+)
///
/// For maximum compatibility on older ARM (Graviton 1, Cortex-A53/A55),
/// use [`NeonToken`] directly instead.
///
/// # Why Arm64 = FP16?
///
/// - **Modern baseline**: FP16 is present on virtually all ARM CPUs from 2019+
/// - **ML acceleration**: Half-precision is essential for efficient inference
/// - **Good coverage**: Apple Silicon, Graviton 2+, and recent Cortex-A all have it
///
/// # Example
///
/// ```rust,ignore
/// use archmage::{Arm64, SimdToken, arcane};
///
/// #[arcane]
/// fn process(token: Arm64, data: &mut [f32; 4]) {
///     // NEON + FP16 intrinsics safe here
/// }
///
/// if let Some(token) = Arm64::summon() {
///     process(token, &mut data);
/// } else {
///     // Fallback for older ARM (Graviton 1, Cortex-A53)
/// }
/// ```
pub type Arm64 = NeonFp16Token;

// ============================================================================
// Capability Marker Trait Implementations
// ============================================================================

use super::Has128BitSimd;
use super::{HasArm64, HasArmAes, HasArmFp16, HasArmSha3, HasNeon};

// NEON provides 128-bit SIMD
// Note: HasFma is x86-specific (requires HasAvx2). ARM has FMA via NEON intrinsics.
impl Has128BitSimd for NeonToken {}
impl Has128BitSimd for NeonAesToken {}
impl Has128BitSimd for NeonSha3Token {}
impl Has128BitSimd for NeonFp16Token {}

// ============================================================================
// AArch64 Feature Marker Trait Implementations
// ============================================================================

// HasNeon: All aarch64 tokens have NEON
impl HasNeon for NeonToken {}
impl HasNeon for NeonAesToken {}
impl HasNeon for NeonSha3Token {}
impl HasNeon for NeonFp16Token {}

// HasArmAes: AES token only
impl HasArmAes for NeonAesToken {}

// HasArmSha3: SHA3 token only (orthogonal to AES)
impl HasArmSha3 for NeonSha3Token {}

// HasArmFp16: FP16 token
impl HasArmFp16 for NeonFp16Token {}

// HasArm64: FP16 token (the Arm64 baseline)
impl HasArm64 for NeonFp16Token {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_is_zst() {
        assert_eq!(core::mem::size_of::<NeonToken>(), 0);
        assert_eq!(core::mem::size_of::<NeonAesToken>(), 0);
        assert_eq!(core::mem::size_of::<NeonSha3Token>(), 0);
        assert_eq!(core::mem::size_of::<NeonFp16Token>(), 0);
    }

    #[test]
    fn test_token_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<NeonToken>();
        assert_copy::<NeonAesToken>();
        assert_copy::<NeonSha3Token>();
        assert_copy::<NeonFp16Token>();
    }

    #[test]
    fn test_token_names() {
        assert_eq!(NeonToken::NAME, "NEON");
        assert_eq!(NeonAesToken::NAME, "NEON+AES");
        assert_eq!(NeonSha3Token::NAME, "NEON+SHA3");
        assert_eq!(NeonFp16Token::NAME, "NEON+FP16");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_always_available() {
        // NEON is baseline for AArch64
        assert!(NeonToken::try_new().is_some());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_token_hierarchy() {
        // SHA3 and AES are orthogonal - SHA3 does NOT imply AES
        if NeonSha3Token::try_new().is_some() {
            assert!(NeonToken::try_new().is_some(), "SHA3 implies NEON");
        }

        // If AES crypto is available, NEON must also be available
        if NeonAesToken::try_new().is_some() {
            assert!(NeonToken::try_new().is_some(), "AES crypto implies NEON");
        }

        // If FP16 is available, NEON must also be available
        if NeonFp16Token::try_new().is_some() {
            assert!(NeonToken::try_new().is_some(), "FP16 implies NEON");
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_sha3_token_extraction() {
        if let Some(sha3) = NeonSha3Token::try_new() {
            let _neon = sha3.neon();
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_aes_token_extraction() {
        if let Some(aes) = NeonAesToken::try_new() {
            let _neon = aes.neon();
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_fp16_token_extraction() {
        if let Some(fp16) = NeonFp16Token::try_new() {
            let _neon = fp16.neon();
        }
    }
}

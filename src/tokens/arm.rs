//! ARM SIMD capability tokens
//!
//! Provides tokens for NEON and optional extensions.
//!
//! On AArch64, NEON is always available (baseline).
//!
//! ## Token Hierarchy
//!
//! - `NeonToken` / `Arm64` - baseline, always available
//! - `NeonAesToken` - NEON + AES
//! - `NeonSha3Token` - NEON + SHA3
//! - `NeonCrcToken` - NEON + CRC

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
        // NEON is always available on AArch64
        Some(Self { _private: () })
    }

    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

// ============================================================================
// NEON + AES Token
// ============================================================================

/// Proof that NEON + AES is available.
///
/// AES extension is common on modern ARM64 devices with crypto extensions.
#[derive(Clone, Copy, Debug)]
pub struct NeonAesToken {
    _private: (),
}

impl SimdToken for NeonAesToken {
    const NAME: &'static str = "NEON+AES";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        if crate::is_aarch64_feature_available!("aes") {
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
    /// Get a NEON token (AES implies NEON)
    #[inline(always)]
    pub fn neon(self) -> NeonToken {
        unsafe { NeonToken::forge_token_dangerously() }
    }
}

// ============================================================================
// NEON + SHA3 Token
// ============================================================================

/// Proof that NEON + SHA3 is available.
///
/// SHA3 extension is available on ARMv8.2-A and later.
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
// NEON + CRC Token
// ============================================================================

/// Proof that NEON + CRC is available.
///
/// CRC32 extension is common on most AArch64 CPUs (part of ARMv8.1-A baseline).
/// Available on: Cortex-A53+, Apple M1+, Graviton 1+, Snapdragon 8xx.
#[derive(Clone, Copy, Debug)]
pub struct NeonCrcToken {
    _private: (),
}

impl SimdToken for NeonCrcToken {
    const NAME: &'static str = "NEON+CRC";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        if crate::is_aarch64_feature_available!("crc") {
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

impl NeonCrcToken {
    /// Get a NEON token (CRC implies NEON on AArch64)
    #[inline(always)]
    pub fn neon(self) -> NeonToken {
        unsafe { NeonToken::forge_token_dangerously() }
    }
}

// ============================================================================
// Friendly Aliases
// ============================================================================

/// The baseline for AArch64 (NEON).
///
/// This is an alias for [`NeonToken`], covering all 64-bit ARM CPUs. NEON is
/// always available on AArch64, making this the universal starting point for
/// ARM code.
pub type Arm64 = NeonToken;

// ============================================================================
// Capability Marker Trait Implementations
// ============================================================================

use super::{Has128BitSimd, HasNeon, HasNeonAes, HasNeonSha3};

// NEON provides 128-bit SIMD
impl Has128BitSimd for NeonToken {}
impl Has128BitSimd for NeonAesToken {}
impl Has128BitSimd for NeonSha3Token {}
impl Has128BitSimd for NeonCrcToken {}

// HasNeon: All aarch64 tokens have NEON
impl HasNeon for NeonToken {}
impl HasNeon for NeonAesToken {}
impl HasNeon for NeonSha3Token {}
impl HasNeon for NeonCrcToken {}

// HasNeonAes: tokens with AES
impl HasNeonAes for NeonAesToken {}

// HasNeonSha3: tokens with SHA3
impl HasNeonSha3 for NeonSha3Token {}

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
        assert_eq!(core::mem::size_of::<NeonCrcToken>(), 0);
    }

    #[test]
    fn test_token_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<NeonToken>();
        assert_copy::<NeonAesToken>();
        assert_copy::<NeonSha3Token>();
        assert_copy::<NeonCrcToken>();
    }

    #[test]
    fn test_token_names() {
        assert_eq!(NeonToken::NAME, "NEON");
        assert_eq!(NeonAesToken::NAME, "NEON+AES");
        assert_eq!(NeonSha3Token::NAME, "NEON+SHA3");
        assert_eq!(NeonCrcToken::NAME, "NEON+CRC");
    }

    #[test]
    fn test_neon_always_available() {
        // NEON is baseline for AArch64
        assert!(NeonToken::try_new().is_some());
    }

    #[test]
    fn test_token_hierarchy() {
        // If NeonAes is available, NEON must also be available
        if NeonAesToken::try_new().is_some() {
            assert!(NeonToken::try_new().is_some(), "NeonAes implies NEON");
        }

        // If NeonSha3 is available, NEON must also be available
        if NeonSha3Token::try_new().is_some() {
            assert!(NeonToken::try_new().is_some(), "NeonSha3 implies NEON");
        }

        // If NeonCrc is available, NEON must also be available
        if NeonCrcToken::try_new().is_some() {
            assert!(NeonToken::try_new().is_some(), "NeonCrc implies NEON");
        }
    }

    #[test]
    fn test_neon_aes_token_extraction() {
        if let Some(aes) = NeonAesToken::try_new() {
            let _neon = aes.neon();
        }
    }

    #[test]
    fn test_neon_sha3_token_extraction() {
        if let Some(sha3) = NeonSha3Token::try_new() {
            let _neon = sha3.neon();
        }
    }

    #[test]
    fn test_neon_crc_token_extraction() {
        if let Some(crc) = NeonCrcToken::try_new() {
            let _neon = crc.neon();
        }
    }
}

//! ARM SIMD capability tokens
//!
//! Provides tokens for NEON, SVE, SVE2, and crypto extensions.
//!
//! Token construction uses [`is_aarch64_feature_available!`] which combines
//! compile-time and runtime detection. When compiled with a target feature
//! enabled (e.g., in a `#[multiversed]` function), the runtime check is
//! completely eliminated.
//!
//! On AArch64, NEON is always available (baseline).
//!
//! ## Token Hierarchy
//!
//! - `NeonToken` - baseline, always available
//! - `ArmCryptoToken` - aes + sha2 + crc (most ARMv8 CPUs)
//! - `ArmCrypto3Token` - + sha3 (ARMv8.4+, implies ArmCryptoToken)
//! - `SveToken` - SVE (Graviton 3, A64FX)
//! - `Sve2Token` - SVE2 (Graviton 4+, implies SveToken)

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
// ARM Crypto Token (aes + sha2 + crc)
// ============================================================================

/// Proof that ARM crypto extensions are available (AES, SHA2, CRC32).
///
/// This covers the baseline ARMv8 crypto extensions found on most AArch64 CPUs:
/// - Cortex-A53/A55/A72/A73/A75/A76/A77/A78/X1
/// - Graviton 1/2/3/4
/// - Apple M1/M2/M3
/// - Snapdragon 8xx series
#[derive(Clone, Copy, Debug)]
pub struct ArmCryptoToken {
    _private: (),
}

impl SimdToken for ArmCryptoToken {
    const NAME: &'static str = "ARM Crypto";

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

impl ArmCryptoToken {
    /// Get a NEON token (crypto implies NEON)
    #[inline(always)]
    pub fn neon(self) -> NeonToken {
        unsafe { NeonToken::forge_token_dangerously() }
    }
}

// ============================================================================
// ARM Crypto3 Token (aes + sha2 + sha3 + crc)
// ============================================================================

/// Proof that ARMv8.4 crypto extensions are available (AES, SHA2, SHA3, CRC32).
///
/// This covers the extended crypto on ARMv8.4+ CPUs:
/// - Neoverse N2/V1/V2 (Graviton 2/3/4)
/// - Apple M1/M2/M3
/// - Snapdragon 8 Gen 1+
/// - Cortex-A710/A715/X2/X3
///
/// Note: Some older ARMv8.0-8.2 cores (Cortex-A53/A55, Graviton 1) do NOT have SHA3.
#[derive(Clone, Copy, Debug)]
pub struct ArmCrypto3Token {
    _private: (),
}

impl SimdToken for ArmCrypto3Token {
    const NAME: &'static str = "ARM Crypto3";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        // All four features must be available
        if crate::is_aarch64_feature_available!("aes")
            && crate::is_aarch64_feature_available!("sha2")
            && crate::is_aarch64_feature_available!("sha3")
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

impl ArmCrypto3Token {
    /// Get an ArmCryptoToken (Crypto3 implies base Crypto)
    #[inline(always)]
    pub fn crypto(self) -> ArmCryptoToken {
        unsafe { ArmCryptoToken::forge_token_dangerously() }
    }

    /// Get a NEON token (crypto implies NEON)
    #[inline(always)]
    pub fn neon(self) -> NeonToken {
        unsafe { NeonToken::forge_token_dangerously() }
    }
}

// ============================================================================
// SVE Token
// ============================================================================

/// Proof that SVE (Scalable Vector Extension) is available.
#[derive(Clone, Copy, Debug)]
pub struct SveToken {
    _private: (),
}

impl SimdToken for SveToken {
    const NAME: &'static str = "SVE";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        if crate::is_aarch64_feature_available!("sve") {
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

impl SveToken {
    /// Get a NEON token (SVE implies NEON)
    #[inline(always)]
    pub fn neon(self) -> NeonToken {
        unsafe { NeonToken::forge_token_dangerously() }
    }
}

// ============================================================================
// SVE2 Token
// ============================================================================

/// Proof that SVE2 is available.
#[derive(Clone, Copy, Debug)]
pub struct Sve2Token {
    _private: (),
}

impl SimdToken for Sve2Token {
    const NAME: &'static str = "SVE2";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        if crate::is_aarch64_feature_available!("sve2") {
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

impl Sve2Token {
    /// Get an SVE token (SVE2 implies SVE)
    #[inline(always)]
    pub fn sve(self) -> SveToken {
        unsafe { SveToken::forge_token_dangerously() }
    }

    /// Get a NEON token (SVE2 implies NEON)
    #[inline(always)]
    pub fn neon(self) -> NeonToken {
        unsafe { NeonToken::forge_token_dangerously() }
    }
}

// ============================================================================
// Capability Marker Trait Implementations
// ============================================================================

use super::{Has128BitSimd, HasFma, HasScalableVectors};

// NEON provides 128-bit SIMD and FMA
impl Has128BitSimd for NeonToken {}
impl HasFma for NeonToken {} // NEON has fused multiply-add instructions

// Crypto tokens inherit NEON capabilities
impl Has128BitSimd for ArmCryptoToken {}
impl HasFma for ArmCryptoToken {}

impl Has128BitSimd for ArmCrypto3Token {}
impl HasFma for ArmCrypto3Token {}

// SVE provides scalable vectors and inherits NEON capabilities
impl Has128BitSimd for SveToken {}
impl HasFma for SveToken {}
impl HasScalableVectors for SveToken {}

// SVE2 extends SVE
impl Has128BitSimd for Sve2Token {}
impl HasFma for Sve2Token {}
impl HasScalableVectors for Sve2Token {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_is_zst() {
        assert_eq!(core::mem::size_of::<NeonToken>(), 0);
        assert_eq!(core::mem::size_of::<ArmCryptoToken>(), 0);
        assert_eq!(core::mem::size_of::<ArmCrypto3Token>(), 0);
        assert_eq!(core::mem::size_of::<SveToken>(), 0);
        assert_eq!(core::mem::size_of::<Sve2Token>(), 0);
    }

    #[test]
    fn test_token_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<NeonToken>();
        assert_copy::<ArmCryptoToken>();
        assert_copy::<ArmCrypto3Token>();
        assert_copy::<SveToken>();
        assert_copy::<Sve2Token>();
    }

    #[test]
    fn test_token_names() {
        assert_eq!(NeonToken::NAME, "NEON");
        assert_eq!(ArmCryptoToken::NAME, "ARM Crypto");
        assert_eq!(ArmCrypto3Token::NAME, "ARM Crypto3");
        assert_eq!(SveToken::NAME, "SVE");
        assert_eq!(Sve2Token::NAME, "SVE2");
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
        // If SVE2 is available, SVE and NEON must also be available
        if Sve2Token::try_new().is_some() {
            assert!(SveToken::try_new().is_some(), "SVE2 implies SVE");
            assert!(NeonToken::try_new().is_some(), "SVE2 implies NEON");
        }

        // If SVE is available, NEON must also be available
        if SveToken::try_new().is_some() {
            assert!(NeonToken::try_new().is_some(), "SVE implies NEON");
        }

        // If Crypto3 is available, base Crypto must also be available
        if ArmCrypto3Token::try_new().is_some() {
            assert!(
                ArmCryptoToken::try_new().is_some(),
                "Crypto3 implies Crypto"
            );
            assert!(NeonToken::try_new().is_some(), "Crypto3 implies NEON");
        }

        // If Crypto is available, NEON must also be available
        if ArmCryptoToken::try_new().is_some() {
            assert!(NeonToken::try_new().is_some(), "Crypto implies NEON");
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_sve2_token_extraction() {
        if let Some(sve2) = Sve2Token::try_new() {
            let _sve = sve2.sve();
            let _neon = sve2.neon();
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_sve_token_extraction() {
        if let Some(sve) = SveToken::try_new() {
            let _neon = sve.neon();
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_crypto3_token_extraction() {
        if let Some(crypto3) = ArmCrypto3Token::try_new() {
            let _crypto = crypto3.crypto();
            let _neon = crypto3.neon();
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_crypto_token_extraction() {
        if let Some(crypto) = ArmCryptoToken::try_new() {
            let _neon = crypto.neon();
        }
    }
}

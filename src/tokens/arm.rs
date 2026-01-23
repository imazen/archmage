//! ARM SIMD capability tokens
//!
//! Provides tokens for NEON and crypto extensions.
//!
//! On AArch64, NEON is always available (baseline).
//!
//! ## Token Hierarchy
//!
//! - `NeonToken` / `Arm64` - baseline, always available
//! - `NeonAesToken` - NEON + AES
//! - `NeonSha3Token` - NEON + SHA3
//! - `ArmCryptoToken` - aes + sha2 + crc (most ARMv8 CPUs)
//! - `ArmCrypto3Token` - + sha3 (ARMv8.4+, implies ArmCryptoToken)

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

    /// Get a NeonAesToken (ArmCrypto implies AES)
    #[inline(always)]
    pub fn neon_aes(self) -> NeonAesToken {
        unsafe { NeonAesToken::forge_token_dangerously() }
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

    /// Get a NeonAesToken (Crypto3 implies AES)
    #[inline(always)]
    pub fn neon_aes(self) -> NeonAesToken {
        unsafe { NeonAesToken::forge_token_dangerously() }
    }

    /// Get a NeonSha3Token (Crypto3 implies SHA3)
    #[inline(always)]
    pub fn neon_sha3(self) -> NeonSha3Token {
        unsafe { NeonSha3Token::forge_token_dangerously() }
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
///
/// # Why Arm64?
///
/// - **Universal**: Every AArch64 CPU has NEON - it's the baseline
/// - **128-bit vectors**: Like SSE2 on x86_64, NEON provides 128-bit SIMD
/// - **FMA included**: ARM NEON includes fused multiply-add instructions
///
/// # Example
///
/// ```rust,ignore
/// use archmage::{Arm64, SimdToken, arcane};
///
/// #[arcane]
/// fn process(token: Arm64, data: &mut [f32; 4]) {
///     // NEON intrinsics safe here
/// }
///
/// // Always succeeds on AArch64
/// if let Some(token) = Arm64::try_new() {
///     process(token, &mut data);
/// }
/// ```
pub type Arm64 = NeonToken;

// ============================================================================
// Capability Marker Trait Implementations
// ============================================================================

use super::{Has128BitSimd, HasNeon, HasNeonAes, HasNeonSha3};

// NEON provides 128-bit SIMD
impl Has128BitSimd for NeonToken {}
impl Has128BitSimd for NeonAesToken {}
impl Has128BitSimd for NeonSha3Token {}
impl Has128BitSimd for ArmCryptoToken {}
impl Has128BitSimd for ArmCrypto3Token {}

// HasNeon: All aarch64 tokens have NEON
impl HasNeon for NeonToken {}
impl HasNeon for NeonAesToken {}
impl HasNeon for NeonSha3Token {}
impl HasNeon for ArmCryptoToken {}
impl HasNeon for ArmCrypto3Token {}

// HasNeonAes: tokens with AES
impl HasNeonAes for NeonAesToken {}
impl HasNeonAes for ArmCryptoToken {}
impl HasNeonAes for ArmCrypto3Token {}

// HasNeonSha3: tokens with SHA3
impl HasNeonSha3 for NeonSha3Token {}
impl HasNeonSha3 for ArmCrypto3Token {}

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
        assert_eq!(core::mem::size_of::<ArmCryptoToken>(), 0);
        assert_eq!(core::mem::size_of::<ArmCrypto3Token>(), 0);
    }

    #[test]
    fn test_token_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<NeonToken>();
        assert_copy::<NeonAesToken>();
        assert_copy::<NeonSha3Token>();
        assert_copy::<ArmCryptoToken>();
        assert_copy::<ArmCrypto3Token>();
    }

    #[test]
    fn test_token_names() {
        assert_eq!(NeonToken::NAME, "NEON");
        assert_eq!(NeonAesToken::NAME, "NEON+AES");
        assert_eq!(NeonSha3Token::NAME, "NEON+SHA3");
        assert_eq!(ArmCryptoToken::NAME, "ARM Crypto");
        assert_eq!(ArmCrypto3Token::NAME, "ARM Crypto3");
    }

    #[test]
    fn test_neon_always_available() {
        // NEON is baseline for AArch64
        assert!(NeonToken::try_new().is_some());
    }

    #[test]
    fn test_token_hierarchy() {
        // If Crypto3 is available, base Crypto must also be available
        if ArmCrypto3Token::try_new().is_some() {
            assert!(
                ArmCryptoToken::try_new().is_some(),
                "Crypto3 implies Crypto"
            );
            assert!(NeonToken::try_new().is_some(), "Crypto3 implies NEON");
            assert!(NeonAesToken::try_new().is_some(), "Crypto3 implies AES");
            assert!(NeonSha3Token::try_new().is_some(), "Crypto3 implies SHA3");
        }

        // If Crypto is available, NEON and AES must also be available
        if ArmCryptoToken::try_new().is_some() {
            assert!(NeonToken::try_new().is_some(), "Crypto implies NEON");
            assert!(NeonAesToken::try_new().is_some(), "Crypto implies AES");
        }

        // If NeonAes is available, NEON must also be available
        if NeonAesToken::try_new().is_some() {
            assert!(NeonToken::try_new().is_some(), "NeonAes implies NEON");
        }

        // If NeonSha3 is available, NEON must also be available
        if NeonSha3Token::try_new().is_some() {
            assert!(NeonToken::try_new().is_some(), "NeonSha3 implies NEON");
        }
    }

    #[test]
    fn test_crypto3_token_extraction() {
        if let Some(crypto3) = ArmCrypto3Token::try_new() {
            let _crypto = crypto3.crypto();
            let _neon = crypto3.neon();
            let _aes = crypto3.neon_aes();
            let _sha3 = crypto3.neon_sha3();
        }
    }

    #[test]
    fn test_crypto_token_extraction() {
        if let Some(crypto) = ArmCryptoToken::try_new() {
            let _neon = crypto.neon();
            let _aes = crypto.neon_aes();
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
}

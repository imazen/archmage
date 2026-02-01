//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! Regenerate with: cargo xtask generate

use crate::tokens::SimdToken;
use crate::tokens::{Has128BitSimd, HasNeon, HasNeonAes, HasNeonSha3};

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
    /// Get a NeonToken (NEON+AES implies NEON)
    #[inline(always)]
    pub fn neon(self) -> NeonToken {
        unsafe { NeonToken::forge_token_dangerously() }
    }
}

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
    /// Get a NeonToken (NEON+SHA3 implies NEON)
    #[inline(always)]
    pub fn neon(self) -> NeonToken {
        unsafe { NeonToken::forge_token_dangerously() }
    }
}

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
    /// Get a NeonToken (NEON+CRC implies NEON)
    #[inline(always)]
    pub fn neon(self) -> NeonToken {
        unsafe { NeonToken::forge_token_dangerously() }
    }
}

/// Type alias for [`NeonToken`].
pub type Arm64 = NeonToken;

impl Has128BitSimd for NeonToken {}
impl Has128BitSimd for NeonAesToken {}
impl Has128BitSimd for NeonSha3Token {}
impl Has128BitSimd for NeonCrcToken {}
impl HasNeon for NeonToken {}
impl HasNeon for NeonAesToken {}
impl HasNeon for NeonSha3Token {}
impl HasNeon for NeonCrcToken {}
impl HasNeonAes for NeonAesToken {}
impl HasNeonSha3 for NeonSha3Token {}

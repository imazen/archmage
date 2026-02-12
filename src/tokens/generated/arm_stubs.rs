//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! Stub tokens: `summon()` always returns `None`.

use crate::tokens::SimdToken;
use crate::tokens::{Has128BitSimd, HasNeon, HasNeonAes, HasNeonSha3};

/// Stub for NEON token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct NeonToken {
    _private: (),
}

impl crate::tokens::Sealed for NeonToken {}

impl SimdToken for NeonToken {
    const NAME: &'static str = "NEON";

    #[inline]
    fn compiled_with() -> Option<bool> {
        Some(false) // Wrong architecture
    }

    #[inline]
    fn summon() -> Option<Self> {
        None // Not available on this architecture
    }

    #[allow(deprecated)]
    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl NeonToken {
    /// This token is not available on this architecture.
    pub fn dangerously_disable_token_process_wide(
        _disabled: bool,
    ) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {
        Err(crate::tokens::CompileTimeGuaranteedError {
            token_name: Self::NAME,
        })
    }

    /// This token is not available on this architecture.
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        Err(crate::tokens::CompileTimeGuaranteedError {
            token_name: Self::NAME,
        })
    }
}

/// Stub for NEON+AES token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct NeonAesToken {
    _private: (),
}

impl crate::tokens::Sealed for NeonAesToken {}

impl SimdToken for NeonAesToken {
    const NAME: &'static str = "NEON+AES";

    #[inline]
    fn compiled_with() -> Option<bool> {
        Some(false) // Wrong architecture
    }

    #[inline]
    fn summon() -> Option<Self> {
        None // Not available on this architecture
    }

    #[allow(deprecated)]
    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl NeonAesToken {
    /// This token is not available on this architecture.
    pub fn dangerously_disable_token_process_wide(
        _disabled: bool,
    ) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {
        Err(crate::tokens::CompileTimeGuaranteedError {
            token_name: Self::NAME,
        })
    }

    /// This token is not available on this architecture.
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        Err(crate::tokens::CompileTimeGuaranteedError {
            token_name: Self::NAME,
        })
    }
}

/// Stub for NEON+SHA3 token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct NeonSha3Token {
    _private: (),
}

impl crate::tokens::Sealed for NeonSha3Token {}

impl SimdToken for NeonSha3Token {
    const NAME: &'static str = "NEON+SHA3";

    #[inline]
    fn compiled_with() -> Option<bool> {
        Some(false) // Wrong architecture
    }

    #[inline]
    fn summon() -> Option<Self> {
        None // Not available on this architecture
    }

    #[allow(deprecated)]
    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl NeonSha3Token {
    /// This token is not available on this architecture.
    pub fn dangerously_disable_token_process_wide(
        _disabled: bool,
    ) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {
        Err(crate::tokens::CompileTimeGuaranteedError {
            token_name: Self::NAME,
        })
    }

    /// This token is not available on this architecture.
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        Err(crate::tokens::CompileTimeGuaranteedError {
            token_name: Self::NAME,
        })
    }
}

/// Stub for NEON+CRC token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct NeonCrcToken {
    _private: (),
}

impl crate::tokens::Sealed for NeonCrcToken {}

impl SimdToken for NeonCrcToken {
    const NAME: &'static str = "NEON+CRC";

    #[inline]
    fn compiled_with() -> Option<bool> {
        Some(false) // Wrong architecture
    }

    #[inline]
    fn summon() -> Option<Self> {
        None // Not available on this architecture
    }

    #[allow(deprecated)]
    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl NeonCrcToken {
    /// This token is not available on this architecture.
    pub fn dangerously_disable_token_process_wide(
        _disabled: bool,
    ) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {
        Err(crate::tokens::CompileTimeGuaranteedError {
            token_name: Self::NAME,
        })
    }

    /// This token is not available on this architecture.
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        Err(crate::tokens::CompileTimeGuaranteedError {
            token_name: Self::NAME,
        })
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

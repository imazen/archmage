//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! Stub tokens: `summon()` always returns `None`.

use crate::tokens::SimdToken;
use crate::tokens::{Has128BitSimd, HasArm64V2, HasArm64V3, HasNeon, HasNeonAes, HasNeonSha3};

/// Stub for NEON token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct NeonToken {
    _private: (),
}

impl crate::tokens::Sealed for NeonToken {}

impl SimdToken for NeonToken {
    const NAME: &'static str = "NEON";
    const TARGET_FEATURES: &'static str = "neon";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+neon";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-neon";

    #[inline]
    fn compiled_with() -> Option<bool> {
        Some(false) // Wrong architecture
    }

    // Note: guaranteed() has a default impl in the trait that calls compiled_with()

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
            target_features: Self::TARGET_FEATURES,
            disable_flags: Self::DISABLE_TARGET_FEATURES,
        })
    }

    /// This token is not available on this architecture.
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        Err(crate::tokens::CompileTimeGuaranteedError {
            token_name: Self::NAME,
            target_features: Self::TARGET_FEATURES,
            disable_flags: Self::DISABLE_TARGET_FEATURES,
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
    const TARGET_FEATURES: &'static str = "neon,aes";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+neon,+aes";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-neon,-aes";

    #[inline]
    fn compiled_with() -> Option<bool> {
        Some(false) // Wrong architecture
    }

    // Note: guaranteed() has a default impl in the trait that calls compiled_with()

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
            target_features: Self::TARGET_FEATURES,
            disable_flags: Self::DISABLE_TARGET_FEATURES,
        })
    }

    /// This token is not available on this architecture.
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        Err(crate::tokens::CompileTimeGuaranteedError {
            token_name: Self::NAME,
            target_features: Self::TARGET_FEATURES,
            disable_flags: Self::DISABLE_TARGET_FEATURES,
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
    const TARGET_FEATURES: &'static str = "neon,sha3";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+neon,+sha3";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-neon,-sha3";

    #[inline]
    fn compiled_with() -> Option<bool> {
        Some(false) // Wrong architecture
    }

    // Note: guaranteed() has a default impl in the trait that calls compiled_with()

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
            target_features: Self::TARGET_FEATURES,
            disable_flags: Self::DISABLE_TARGET_FEATURES,
        })
    }

    /// This token is not available on this architecture.
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        Err(crate::tokens::CompileTimeGuaranteedError {
            token_name: Self::NAME,
            target_features: Self::TARGET_FEATURES,
            disable_flags: Self::DISABLE_TARGET_FEATURES,
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
    const TARGET_FEATURES: &'static str = "neon,crc";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+neon,+crc";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-neon,-crc";

    #[inline]
    fn compiled_with() -> Option<bool> {
        Some(false) // Wrong architecture
    }

    // Note: guaranteed() has a default impl in the trait that calls compiled_with()

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
            target_features: Self::TARGET_FEATURES,
            disable_flags: Self::DISABLE_TARGET_FEATURES,
        })
    }

    /// This token is not available on this architecture.
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        Err(crate::tokens::CompileTimeGuaranteedError {
            token_name: Self::NAME,
            target_features: Self::TARGET_FEATURES,
            disable_flags: Self::DISABLE_TARGET_FEATURES,
        })
    }
}

/// Stub for Arm64-v2 token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct Arm64V2Token {
    _private: (),
}

impl crate::tokens::Sealed for Arm64V2Token {}

impl SimdToken for Arm64V2Token {
    const NAME: &'static str = "Arm64-v2";
    const TARGET_FEATURES: &'static str = "neon,crc,rdm,dotprod,fp16,aes,sha2";
    const ENABLE_TARGET_FEATURES: &'static str =
        "-Ctarget-feature=+neon,+crc,+rdm,+dotprod,+fp16,+aes,+sha2";
    const DISABLE_TARGET_FEATURES: &'static str =
        "-Ctarget-feature=-neon,-crc,-rdm,-dotprod,-fp16,-aes,-sha2";

    #[inline]
    fn compiled_with() -> Option<bool> {
        Some(false) // Wrong architecture
    }

    // Note: guaranteed() has a default impl in the trait that calls compiled_with()

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

impl Arm64V2Token {
    /// This token is not available on this architecture.
    pub fn dangerously_disable_token_process_wide(
        _disabled: bool,
    ) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {
        Err(crate::tokens::CompileTimeGuaranteedError {
            token_name: Self::NAME,
            target_features: Self::TARGET_FEATURES,
            disable_flags: Self::DISABLE_TARGET_FEATURES,
        })
    }

    /// This token is not available on this architecture.
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        Err(crate::tokens::CompileTimeGuaranteedError {
            token_name: Self::NAME,
            target_features: Self::TARGET_FEATURES,
            disable_flags: Self::DISABLE_TARGET_FEATURES,
        })
    }
}

/// Stub for Arm64-v3 token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct Arm64V3Token {
    _private: (),
}

impl crate::tokens::Sealed for Arm64V3Token {}

impl SimdToken for Arm64V3Token {
    const NAME: &'static str = "Arm64-v3";
    const TARGET_FEATURES: &'static str =
        "neon,crc,rdm,dotprod,fp16,aes,sha2,fhm,fcma,sha3,i8mm,bf16";
    const ENABLE_TARGET_FEATURES: &'static str =
        "-Ctarget-feature=+neon,+crc,+rdm,+dotprod,+fp16,+aes,+sha2,+fhm,+fcma,+sha3,+i8mm,+bf16";
    const DISABLE_TARGET_FEATURES: &'static str =
        "-Ctarget-feature=-neon,-crc,-rdm,-dotprod,-fp16,-aes,-sha2,-fhm,-fcma,-sha3,-i8mm,-bf16";

    #[inline]
    fn compiled_with() -> Option<bool> {
        Some(false) // Wrong architecture
    }

    // Note: guaranteed() has a default impl in the trait that calls compiled_with()

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

impl Arm64V3Token {
    /// This token is not available on this architecture.
    pub fn dangerously_disable_token_process_wide(
        _disabled: bool,
    ) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {
        Err(crate::tokens::CompileTimeGuaranteedError {
            token_name: Self::NAME,
            target_features: Self::TARGET_FEATURES,
            disable_flags: Self::DISABLE_TARGET_FEATURES,
        })
    }

    /// This token is not available on this architecture.
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        Err(crate::tokens::CompileTimeGuaranteedError {
            token_name: Self::NAME,
            target_features: Self::TARGET_FEATURES,
            disable_flags: Self::DISABLE_TARGET_FEATURES,
        })
    }
}

/// Type alias for [`NeonToken`].
pub type Arm64 = NeonToken;

impl Has128BitSimd for NeonToken {}
impl Has128BitSimd for NeonAesToken {}
impl Has128BitSimd for NeonSha3Token {}
impl Has128BitSimd for NeonCrcToken {}
impl Has128BitSimd for Arm64V2Token {}
impl Has128BitSimd for Arm64V3Token {}
impl HasArm64V2 for Arm64V2Token {}
impl HasArm64V2 for Arm64V3Token {}
impl HasArm64V3 for Arm64V3Token {}
impl HasNeon for NeonToken {}
impl HasNeon for NeonAesToken {}
impl HasNeon for NeonSha3Token {}
impl HasNeon for NeonCrcToken {}
impl HasNeon for Arm64V2Token {}
impl HasNeon for Arm64V3Token {}
impl HasNeonAes for NeonAesToken {}
impl HasNeonAes for Arm64V2Token {}
impl HasNeonAes for Arm64V3Token {}
impl HasNeonSha3 for NeonSha3Token {}
impl HasNeonSha3 for Arm64V3Token {}

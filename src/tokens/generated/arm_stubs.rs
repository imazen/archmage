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

impl SimdToken for NeonToken {
    const NAME: &'static str = "NEON";

    #[inline]
    fn guaranteed() -> Option<bool> {
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

/// Stub for NEON+AES token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct NeonAesToken {
    _private: (),
}

impl SimdToken for NeonAesToken {
    const NAME: &'static str = "NEON+AES";

    #[inline]
    fn guaranteed() -> Option<bool> {
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

/// Stub for NEON+SHA3 token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct NeonSha3Token {
    _private: (),
}

impl SimdToken for NeonSha3Token {
    const NAME: &'static str = "NEON+SHA3";

    #[inline]
    fn guaranteed() -> Option<bool> {
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

/// Stub for NEON+CRC token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct NeonCrcToken {
    _private: (),
}

impl SimdToken for NeonCrcToken {
    const NAME: &'static str = "NEON+CRC";

    #[inline]
    fn guaranteed() -> Option<bool> {
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

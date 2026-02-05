//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! Stub tokens: `summon()` always returns `None`.

use crate::tokens::SimdToken;
use crate::tokens::{Has128BitSimd, Has256BitSimd, HasX64V2};

/// Stub for x86-64-v2 token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct X64V2Token {
    _private: (),
}

impl crate::tokens::Sealed for X64V2Token {}

impl SimdToken for X64V2Token {
    const NAME: &'static str = "x86-64-v2";

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

/// Stub for x86-64-v3 token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct X64V3Token {
    _private: (),
}

impl crate::tokens::Sealed for X64V3Token {}

impl SimdToken for X64V3Token {
    const NAME: &'static str = "x86-64-v3";

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

/// Type alias for [`X64V3Token`].
pub type Desktop64 = X64V3Token;

/// Type alias for [`X64V3Token`].
pub type Avx2FmaToken = X64V3Token;

impl Has128BitSimd for X64V2Token {}
impl Has128BitSimd for X64V3Token {}
impl Has256BitSimd for X64V3Token {}
impl HasX64V2 for X64V2Token {}
impl HasX64V2 for X64V3Token {}

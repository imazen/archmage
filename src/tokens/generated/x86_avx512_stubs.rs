//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! Stub tokens: `summon()` always returns `None`.

use crate::tokens::SimdToken;
use crate::tokens::{Has128BitSimd, Has256BitSimd, Has512BitSimd, HasX64V2, HasX64V4};

/// Stub for AVX-512 token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct X64V4Token {
    _private: (),
}

impl crate::tokens::Sealed for X64V4Token {}

impl SimdToken for X64V4Token {
    const NAME: &'static str = "AVX-512";

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

impl X64V4Token {
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

/// Stub for AVX-512Modern token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct Avx512ModernToken {
    _private: (),
}

impl crate::tokens::Sealed for Avx512ModernToken {}

impl SimdToken for Avx512ModernToken {
    const NAME: &'static str = "AVX-512Modern";

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

impl Avx512ModernToken {
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

/// Stub for AVX-512FP16 token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct Avx512Fp16Token {
    _private: (),
}

impl crate::tokens::Sealed for Avx512Fp16Token {}

impl SimdToken for Avx512Fp16Token {
    const NAME: &'static str = "AVX-512FP16";

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

impl Avx512Fp16Token {
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

/// Type alias for [`X64V4Token`].
pub type Avx512Token = X64V4Token;

/// Type alias for [`X64V4Token`].
pub type Server64 = X64V4Token;

impl Has128BitSimd for X64V4Token {}
impl Has128BitSimd for Avx512ModernToken {}
impl Has128BitSimd for Avx512Fp16Token {}
impl Has256BitSimd for X64V4Token {}
impl Has256BitSimd for Avx512ModernToken {}
impl Has256BitSimd for Avx512Fp16Token {}
impl Has512BitSimd for X64V4Token {}
impl Has512BitSimd for Avx512ModernToken {}
impl Has512BitSimd for Avx512Fp16Token {}
impl HasX64V2 for X64V4Token {}
impl HasX64V2 for Avx512ModernToken {}
impl HasX64V2 for Avx512Fp16Token {}
impl HasX64V4 for X64V4Token {}
impl HasX64V4 for Avx512ModernToken {}
impl HasX64V4 for Avx512Fp16Token {}

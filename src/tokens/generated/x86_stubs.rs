//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! Stub tokens: `summon()` always returns `None`.

use crate::tokens::SimdToken;
#[allow(deprecated)]
use crate::tokens::{Has128BitSimd, Has256BitSimd, Has512BitSimd, HasX64V2, HasX64V4};

/// Stub for x86-64-v1 token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct X64V1Token {
    _private: (),
}

impl crate::tokens::Sealed for X64V1Token {}

impl SimdToken for X64V1Token {
    const NAME: &'static str = "x86-64-v1";
    const TARGET_FEATURES: &'static str = "sse,sse2";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+sse,+sse2";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-sse,-sse2";

    #[inline]
    fn compiled_with() -> Option<bool> {
        Some(false) // Wrong architecture
    }

    // Note: guaranteed() has a default impl in the trait that calls compiled_with()

    #[inline]
    fn summon() -> Option<Self> {
        None // Not available on this architecture
    }
}

#[cfg(feature = "forge-token-api")]
impl X64V1Token {
    /// Create a token without any checks.
    ///
    /// # Safety
    ///
    /// Caller must guarantee the CPU feature is available.
    #[deprecated(
        since = "0.5.0",
        note = "Pass tokens through from summon() instead of forging"
    )]
    #[inline(always)]
    pub unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl X64V1Token {
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

/// Stub for x86-64-v2 token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct X64V2Token {
    _private: (),
}

impl crate::tokens::Sealed for X64V2Token {}

impl SimdToken for X64V2Token {
    const NAME: &'static str = "x86-64-v2";
    const TARGET_FEATURES: &'static str = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b";
    const ENABLE_TARGET_FEATURES: &'static str =
        "-Ctarget-feature=+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b";
    const DISABLE_TARGET_FEATURES: &'static str =
        "-Ctarget-feature=-sse,-sse2,-sse3,-ssse3,-sse4.1,-sse4.2,-popcnt,-cmpxchg16b";

    #[inline]
    fn compiled_with() -> Option<bool> {
        Some(false) // Wrong architecture
    }

    // Note: guaranteed() has a default impl in the trait that calls compiled_with()

    #[inline]
    fn summon() -> Option<Self> {
        None // Not available on this architecture
    }
}

#[cfg(feature = "forge-token-api")]
impl X64V2Token {
    /// Create a token without any checks.
    ///
    /// # Safety
    ///
    /// Caller must guarantee the CPU feature is available.
    #[deprecated(
        since = "0.5.0",
        note = "Pass tokens through from summon() instead of forging"
    )]
    #[inline(always)]
    pub unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl X64V2Token {
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

/// Stub for x86-64 Crypto token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct X64CryptoToken {
    _private: (),
}

impl crate::tokens::Sealed for X64CryptoToken {}

impl SimdToken for X64CryptoToken {
    const NAME: &'static str = "x86-64 Crypto";
    const TARGET_FEATURES: &'static str =
        "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,pclmulqdq,aes";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b,+pclmulqdq,+aes";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-sse,-sse2,-sse3,-ssse3,-sse4.1,-sse4.2,-popcnt,-cmpxchg16b,-pclmulqdq,-aes";

    #[inline]
    fn compiled_with() -> Option<bool> {
        Some(false) // Wrong architecture
    }

    // Note: guaranteed() has a default impl in the trait that calls compiled_with()

    #[inline]
    fn summon() -> Option<Self> {
        None // Not available on this architecture
    }
}

#[cfg(feature = "forge-token-api")]
impl X64CryptoToken {
    /// Create a token without any checks.
    ///
    /// # Safety
    ///
    /// Caller must guarantee the CPU feature is available.
    #[deprecated(
        since = "0.5.0",
        note = "Pass tokens through from summon() instead of forging"
    )]
    #[inline(always)]
    pub unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl X64CryptoToken {
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

/// Stub for x86-64-v3 token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct X64V3Token {
    _private: (),
}

impl crate::tokens::Sealed for X64V3Token {}

impl SimdToken for X64V3Token {
    const NAME: &'static str = "x86-64-v3";
    const TARGET_FEATURES: &'static str = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b,+avx,+avx2,+fma,+bmi1,+bmi2,+f16c,+lzcnt,+movbe";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-sse,-sse2,-sse3,-ssse3,-sse4.1,-sse4.2,-popcnt,-cmpxchg16b,-avx,-avx2,-fma,-bmi1,-bmi2,-f16c,-lzcnt,-movbe";

    #[inline]
    fn compiled_with() -> Option<bool> {
        Some(false) // Wrong architecture
    }

    // Note: guaranteed() has a default impl in the trait that calls compiled_with()

    #[inline]
    fn summon() -> Option<Self> {
        None // Not available on this architecture
    }
}

#[cfg(feature = "forge-token-api")]
impl X64V3Token {
    /// Create a token without any checks.
    ///
    /// # Safety
    ///
    /// Caller must guarantee the CPU feature is available.
    #[deprecated(
        since = "0.5.0",
        note = "Pass tokens through from summon() instead of forging"
    )]
    #[inline(always)]
    pub unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl X64V3Token {
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

/// Stub for x86-64-v3 Crypto token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct X64V3CryptoToken {
    _private: (),
}

impl crate::tokens::Sealed for X64V3CryptoToken {}

impl SimdToken for X64V3CryptoToken {
    const NAME: &'static str = "x86-64-v3 Crypto";
    const TARGET_FEATURES: &'static str = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,vpclmulqdq,vaes";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b,+avx,+avx2,+fma,+bmi1,+bmi2,+f16c,+lzcnt,+movbe,+pclmulqdq,+aes,+vpclmulqdq,+vaes";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-sse,-sse2,-sse3,-ssse3,-sse4.1,-sse4.2,-popcnt,-cmpxchg16b,-avx,-avx2,-fma,-bmi1,-bmi2,-f16c,-lzcnt,-movbe,-pclmulqdq,-aes,-vpclmulqdq,-vaes";

    #[inline]
    fn compiled_with() -> Option<bool> {
        Some(false) // Wrong architecture
    }

    // Note: guaranteed() has a default impl in the trait that calls compiled_with()

    #[inline]
    fn summon() -> Option<Self> {
        None // Not available on this architecture
    }
}

#[cfg(feature = "forge-token-api")]
impl X64V3CryptoToken {
    /// Create a token without any checks.
    ///
    /// # Safety
    ///
    /// Caller must guarantee the CPU feature is available.
    #[deprecated(
        since = "0.5.0",
        note = "Pass tokens through from summon() instead of forging"
    )]
    #[inline(always)]
    pub unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl X64V3CryptoToken {
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

/// Stub for AVX-512 token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct X64V4Token {
    _private: (),
}

impl crate::tokens::Sealed for X64V4Token {}

impl SimdToken for X64V4Token {
    const NAME: &'static str = "AVX-512";
    const TARGET_FEATURES: &'static str = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b,+avx,+avx2,+fma,+bmi1,+bmi2,+f16c,+lzcnt,+movbe,+pclmulqdq,+aes,+avx512f,+avx512bw,+avx512cd,+avx512dq,+avx512vl";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-sse,-sse2,-sse3,-ssse3,-sse4.1,-sse4.2,-popcnt,-cmpxchg16b,-avx,-avx2,-fma,-bmi1,-bmi2,-f16c,-lzcnt,-movbe,-pclmulqdq,-aes,-avx512f,-avx512bw,-avx512cd,-avx512dq,-avx512vl";

    #[inline]
    fn compiled_with() -> Option<bool> {
        Some(false) // Wrong architecture
    }

    // Note: guaranteed() has a default impl in the trait that calls compiled_with()

    #[inline]
    fn summon() -> Option<Self> {
        None // Not available on this architecture
    }
}

#[cfg(feature = "forge-token-api")]
impl X64V4Token {
    /// Create a token without any checks.
    ///
    /// # Safety
    ///
    /// Caller must guarantee the CPU feature is available.
    #[deprecated(
        since = "0.5.0",
        note = "Pass tokens through from summon() instead of forging"
    )]
    #[inline(always)]
    pub unsafe fn forge_token_dangerously() -> Self {
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

/// Stub for x86-64-v4x token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct X64V4xToken {
    _private: (),
}

impl crate::tokens::Sealed for X64V4xToken {}

impl SimdToken for X64V4xToken {
    const NAME: &'static str = "x86-64-v4x";
    const TARGET_FEATURES: &'static str = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b,+avx,+avx2,+fma,+bmi1,+bmi2,+f16c,+lzcnt,+movbe,+pclmulqdq,+aes,+avx512f,+avx512bw,+avx512cd,+avx512dq,+avx512vl,+avx512vpopcntdq,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512bitalg,+avx512vnni,+vpclmulqdq,+gfni,+vaes";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-sse,-sse2,-sse3,-ssse3,-sse4.1,-sse4.2,-popcnt,-cmpxchg16b,-avx,-avx2,-fma,-bmi1,-bmi2,-f16c,-lzcnt,-movbe,-pclmulqdq,-aes,-avx512f,-avx512bw,-avx512cd,-avx512dq,-avx512vl,-avx512vpopcntdq,-avx512ifma,-avx512vbmi,-avx512vbmi2,-avx512bitalg,-avx512vnni,-vpclmulqdq,-gfni,-vaes";

    #[inline]
    fn compiled_with() -> Option<bool> {
        Some(false) // Wrong architecture
    }

    // Note: guaranteed() has a default impl in the trait that calls compiled_with()

    #[inline]
    fn summon() -> Option<Self> {
        None // Not available on this architecture
    }
}

#[cfg(feature = "forge-token-api")]
impl X64V4xToken {
    /// Create a token without any checks.
    ///
    /// # Safety
    ///
    /// Caller must guarantee the CPU feature is available.
    #[deprecated(
        since = "0.5.0",
        note = "Pass tokens through from summon() instead of forging"
    )]
    #[inline(always)]
    pub unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl X64V4xToken {
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

/// Stub for AVX-512FP16 token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct Avx512Fp16Token {
    _private: (),
}

impl crate::tokens::Sealed for Avx512Fp16Token {}

impl SimdToken for Avx512Fp16Token {
    const NAME: &'static str = "AVX-512FP16";
    const TARGET_FEATURES: &'static str = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512fp16";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b,+avx,+avx2,+fma,+bmi1,+bmi2,+f16c,+lzcnt,+movbe,+pclmulqdq,+aes,+avx512f,+avx512bw,+avx512cd,+avx512dq,+avx512vl,+avx512fp16";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-sse,-sse2,-sse3,-ssse3,-sse4.1,-sse4.2,-popcnt,-cmpxchg16b,-avx,-avx2,-fma,-bmi1,-bmi2,-f16c,-lzcnt,-movbe,-pclmulqdq,-aes,-avx512f,-avx512bw,-avx512cd,-avx512dq,-avx512vl,-avx512fp16";

    #[inline]
    fn compiled_with() -> Option<bool> {
        Some(false) // Wrong architecture
    }

    // Note: guaranteed() has a default impl in the trait that calls compiled_with()

    #[inline]
    fn summon() -> Option<Self> {
        None // Not available on this architecture
    }
}

#[cfg(feature = "forge-token-api")]
impl Avx512Fp16Token {
    /// Create a token without any checks.
    ///
    /// # Safety
    ///
    /// Caller must guarantee the CPU feature is available.
    #[deprecated(
        since = "0.5.0",
        note = "Pass tokens through from summon() instead of forging"
    )]
    #[inline(always)]
    pub unsafe fn forge_token_dangerously() -> Self {
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

/// Type alias for [`X64V1Token`].
#[deprecated(
    since = "0.9.9",
    note = "Use X64V1Token instead. Sse2Token is misleading — this is an x86_64-only token (SSE2 is the x86_64 baseline). On 32-bit x86, summon() returns None."
)]
pub type Sse2Token = X64V1Token;

/// Type alias for [`X64V3Token`].
pub type Desktop64 = X64V3Token;

/// Type alias for [`X64V3Token`].
#[deprecated(
    since = "0.9.9",
    note = "Use X64V3Token or Desktop64 instead. Avx2FmaToken is misleading — V3 includes BMI1/2, F16C, and more, not just AVX2+FMA."
)]
pub type Avx2FmaToken = X64V3Token;

/// Type alias for [`X64V4Token`].
pub type Avx512Token = X64V4Token;

/// Type alias for [`X64V4Token`].
pub type Server64 = X64V4Token;

/// Type alias for [`X64V4xToken`].
pub type Avx512ModernToken = X64V4xToken;

impl X64V1Token {
    #[doc(hidden)]
    pub const __ARCHMAGE_TIER_TAG: u32 = 0x510E0DCD;
}

impl X64V2Token {
    #[doc(hidden)]
    pub const __ARCHMAGE_TIER_TAG: u32 = 0x8CCD97C6;
}

impl X64CryptoToken {
    #[doc(hidden)]
    pub const __ARCHMAGE_TIER_TAG: u32 = 0xF5F0FBF5;
}

impl X64V3Token {
    #[doc(hidden)]
    pub const __ARCHMAGE_TIER_TAG: u32 = 0xF38B284B;
}

impl X64V3CryptoToken {
    #[doc(hidden)]
    pub const __ARCHMAGE_TIER_TAG: u32 = 0x01EAE708;
}

impl X64V4Token {
    #[doc(hidden)]
    pub const __ARCHMAGE_TIER_TAG: u32 = 0xFE1B900C;
}

impl X64V4xToken {
    #[doc(hidden)]
    pub const __ARCHMAGE_TIER_TAG: u32 = 0x8F63232A;
}

impl Avx512Fp16Token {
    #[doc(hidden)]
    pub const __ARCHMAGE_TIER_TAG: u32 = 0x2F39FFC0;
}

#[allow(deprecated)]
impl Has128BitSimd for X64V1Token {}
#[allow(deprecated)]
impl Has128BitSimd for X64V2Token {}
#[allow(deprecated)]
impl Has128BitSimd for X64CryptoToken {}
#[allow(deprecated)]
impl Has128BitSimd for X64V3Token {}
#[allow(deprecated)]
impl Has128BitSimd for X64V3CryptoToken {}
#[allow(deprecated)]
impl Has128BitSimd for X64V4Token {}
#[allow(deprecated)]
impl Has128BitSimd for X64V4xToken {}
#[allow(deprecated)]
impl Has128BitSimd for Avx512Fp16Token {}
#[allow(deprecated)]
impl Has256BitSimd for X64V3Token {}
#[allow(deprecated)]
impl Has256BitSimd for X64V3CryptoToken {}
#[allow(deprecated)]
impl Has256BitSimd for X64V4Token {}
#[allow(deprecated)]
impl Has256BitSimd for X64V4xToken {}
#[allow(deprecated)]
impl Has256BitSimd for Avx512Fp16Token {}
#[allow(deprecated)]
impl Has512BitSimd for X64V4Token {}
#[allow(deprecated)]
impl Has512BitSimd for X64V4xToken {}
#[allow(deprecated)]
impl Has512BitSimd for Avx512Fp16Token {}
impl HasX64V2 for X64V2Token {}
impl HasX64V2 for X64CryptoToken {}
impl HasX64V2 for X64V3Token {}
impl HasX64V2 for X64V3CryptoToken {}
impl HasX64V2 for X64V4Token {}
impl HasX64V2 for X64V4xToken {}
impl HasX64V2 for Avx512Fp16Token {}
impl HasX64V4 for X64V4Token {}
impl HasX64V4 for X64V4xToken {}
impl HasX64V4 for Avx512Fp16Token {}

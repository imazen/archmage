//! AVX-512 capability tokens for x86/x86_64.
//!
//! This module provides:
//! - `Avx512Token` - AVX-512 F+CD+VL+DQ+BW (x86-64-v4 level)
//! - `X64V4Token` - Type alias for `Avx512Token`
//! - `Avx512ModernToken` - Full modern AVX-512 (Ice Lake / Zen 4)
//! - `Avx512Fp16Token` - AVX-512 FP16 (Sapphire Rapids+)

use super::SimdToken;
use super::{Avx2FmaToken, Avx2Token, AvxToken, Sse42Token, X64V3Token};

// ============================================================================
// AVX-512 Token (F + CD + VL + DQ + BW = x86-64-v4)
// ============================================================================

/// Proof that AVX-512 (F + CD + VL + DQ + BW) is available.
///
/// This is the complete x86-64-v4 AVX-512 feature set, available on:
/// - Intel Skylake-X (2017+), Ice Lake, Sapphire Rapids
/// - AMD Zen 4+ (2022+)
///
/// Note: Intel 12th-14th gen consumer CPUs do NOT have AVX-512.
#[derive(Clone, Copy, Debug)]
pub struct Avx512Token {
    _private: (),
}

impl SimdToken for Avx512Token {
    const NAME: &'static str = "x86-64-v4";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        // Explicit cumulative check: all v4 AVX-512 features + v3 baseline
        if crate::is_x86_feature_available!("avx512f")
            && crate::is_x86_feature_available!("avx512cd")
            && crate::is_x86_feature_available!("avx512vl")
            && crate::is_x86_feature_available!("avx512dq")
            && crate::is_x86_feature_available!("avx512bw")
            // v3 baseline
            && crate::is_x86_feature_available!("fma")
            && crate::is_x86_feature_available!("avx2")
            && crate::is_x86_feature_available!("bmi2")
            && crate::is_x86_feature_available!("avx")
            && crate::is_x86_feature_available!("sse4.2")
            && crate::is_x86_feature_available!("popcnt")
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

impl Avx512Token {
    /// Get a v3 token (AVX-512 implies v3)
    #[inline(always)]
    pub fn v3(self) -> X64V3Token {
        unsafe { X64V3Token::forge_token_dangerously() }
    }

    /// Get an AVX2+FMA token
    #[inline(always)]
    pub fn avx2_fma(self) -> Avx2FmaToken {
        unsafe { Avx2FmaToken::forge_token_dangerously() }
    }

    /// Get an AVX2 token
    #[inline(always)]
    pub fn avx2(self) -> Avx2Token {
        unsafe { Avx2Token::forge_token_dangerously() }
    }

    /// Get an AVX token
    #[inline(always)]
    pub fn avx(self) -> AvxToken {
        unsafe { AvxToken::forge_token_dangerously() }
    }

    /// Get an SSE4.2 token
    #[inline(always)]
    pub fn sse42(self) -> Sse42Token {
        unsafe { Sse42Token::forge_token_dangerously() }
    }
}

/// Alias for [`Avx512Token`] using the x86-64-v4 microarchitecture level name.
pub type X64V4Token = Avx512Token;

// ============================================================================
// AVX-512 Modern Token (Ice Lake / Zen 4)
// ============================================================================

/// Proof that modern AVX-512 features are available (Ice Lake / Zen 4 level).
///
/// This includes all of [`Avx512Token`] (F+CD+VL+DQ+BW) plus:
/// - VPOPCNTDQ, IFMA, VBMI, VBMI2, BITALG, VNNI, BF16
/// - VPCLMULQDQ, GFNI, VAES
///
/// Available on Intel Ice Lake (2019+), Sapphire Rapids, AMD Zen 4+.
/// NOT available on Skylake-X (lacks VBMI2, VNNI, BF16, etc.).
#[derive(Clone, Copy, Debug)]
pub struct Avx512ModernToken {
    _private: (),
}

impl SimdToken for Avx512ModernToken {
    const NAME: &'static str = "AVX-512Modern";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        // All modern AVX-512 features
        if crate::is_x86_feature_available!("avx512f")
            && crate::is_x86_feature_available!("avx512cd")
            && crate::is_x86_feature_available!("avx512vl")
            && crate::is_x86_feature_available!("avx512dq")
            && crate::is_x86_feature_available!("avx512bw")
            && crate::is_x86_feature_available!("avx512vpopcntdq")
            && crate::is_x86_feature_available!("avx512ifma")
            && crate::is_x86_feature_available!("avx512vbmi")
            && crate::is_x86_feature_available!("avx512vbmi2")
            && crate::is_x86_feature_available!("avx512bitalg")
            && crate::is_x86_feature_available!("avx512vnni")
            && crate::is_x86_feature_available!("avx512bf16")
            && crate::is_x86_feature_available!("vpclmulqdq")
            && crate::is_x86_feature_available!("gfni")
            && crate::is_x86_feature_available!("vaes")
            // v3 baseline
            && crate::is_x86_feature_available!("fma")
            && crate::is_x86_feature_available!("avx2")
            && crate::is_x86_feature_available!("bmi2")
            && crate::is_x86_feature_available!("avx")
            && crate::is_x86_feature_available!("sse4.2")
            && crate::is_x86_feature_available!("popcnt")
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

impl Avx512ModernToken {
    /// Get an Avx512Token (base AVX-512)
    #[inline(always)]
    pub fn avx512(self) -> Avx512Token {
        unsafe { Avx512Token::forge_token_dangerously() }
    }

    /// Get a v4 token
    #[inline(always)]
    pub fn v4(self) -> X64V4Token {
        unsafe { X64V4Token::forge_token_dangerously() }
    }

    /// Get a v3 token
    #[inline(always)]
    pub fn v3(self) -> X64V3Token {
        unsafe { X64V3Token::forge_token_dangerously() }
    }

    /// Get an AVX2+FMA token
    #[inline(always)]
    pub fn avx2_fma(self) -> Avx2FmaToken {
        unsafe { Avx2FmaToken::forge_token_dangerously() }
    }

    /// Get an AVX2 token
    #[inline(always)]
    pub fn avx2(self) -> Avx2Token {
        unsafe { Avx2Token::forge_token_dangerously() }
    }
}

// ============================================================================
// AVX-512 FP16 Token (Sapphire Rapids+)
// ============================================================================

/// Proof that AVX-512 FP16 (half-precision) is available.
///
/// AVX-512 FP16 provides native 16-bit floating-point arithmetic in 512-bit
/// vectors, enabling efficient ML inference and scientific computing.
///
/// Available on Intel Sapphire Rapids (2023+), Emerald Rapids.
/// NOT available on Skylake-X, Ice Lake, AMD Zen 4.
#[derive(Clone, Copy, Debug)]
pub struct Avx512Fp16Token {
    _private: (),
}

impl SimdToken for Avx512Fp16Token {
    const NAME: &'static str = "AVX-512FP16";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        // FP16 requires the full v4 feature set
        if crate::is_x86_feature_available!("avx512fp16")
            && crate::is_x86_feature_available!("avx512f")
            && crate::is_x86_feature_available!("avx512cd")
            && crate::is_x86_feature_available!("avx512vl")
            && crate::is_x86_feature_available!("avx512dq")
            && crate::is_x86_feature_available!("avx512bw")
            && crate::is_x86_feature_available!("fma")
            && crate::is_x86_feature_available!("avx2")
            && crate::is_x86_feature_available!("avx")
            && crate::is_x86_feature_available!("sse4.2")
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

impl Avx512Fp16Token {
    /// Get an Avx512Token
    #[inline(always)]
    pub fn avx512(self) -> Avx512Token {
        unsafe { Avx512Token::forge_token_dangerously() }
    }

    /// Get a v4 token
    #[inline(always)]
    pub fn v4(self) -> X64V4Token {
        unsafe { X64V4Token::forge_token_dangerously() }
    }

    /// Get an AVX2+FMA token
    #[inline(always)]
    pub fn avx2_fma(self) -> Avx2FmaToken {
        unsafe { Avx2FmaToken::forge_token_dangerously() }
    }

    /// Get an AVX2 token
    #[inline(always)]
    pub fn avx2(self) -> Avx2Token {
        unsafe { Avx2Token::forge_token_dangerously() }
    }
}

// ============================================================================
// Sealed Trait Implementations
// ============================================================================

use super::sealed::Sealed;

impl Sealed for Avx512Token {}
impl Sealed for Avx512ModernToken {}
impl Sealed for Avx512Fp16Token {}

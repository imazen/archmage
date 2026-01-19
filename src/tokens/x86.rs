//! x86_64 SIMD capability tokens
//!
//! Provides tokens for SSE2, SSE4.1, AVX, AVX2, AVX-512, and FMA.
//!
//! Token construction uses [`is_x86_feature_available!`] which combines
//! compile-time and runtime detection. When compiled with a target feature
//! enabled (e.g., in a `#[multiversed]` function), the runtime check is
//! completely eliminated.

use super::{CompositeToken, SimdToken};

// ============================================================================
// SSE2 Token (baseline for x86_64)
// ============================================================================

/// Proof that SSE2 is available.
///
/// SSE2 is the baseline for x86_64 - it's always available on 64-bit x86.
/// This token exists for completeness and generic code.
#[derive(Clone, Copy, Debug)]
pub struct Sse2Token {
    _private: (),
}

impl SimdToken for Sse2Token {
    const NAME: &'static str = "SSE2";

    #[inline]
    fn try_new() -> Option<Self> {
        // SSE2 is always available on x86_64
        Some(Self { _private: () })
    }

    #[inline(always)]
    unsafe fn new_unchecked() -> Self {
        Self { _private: () }
    }
}

// ============================================================================
// SSE4.1 Token
// ============================================================================

/// Proof that SSE4.1 is available.
///
/// SSE4.1 adds blend, round, and other useful instructions.
#[derive(Clone, Copy, Debug)]
pub struct Sse41Token {
    _private: (),
}

impl SimdToken for Sse41Token {
    const NAME: &'static str = "SSE4.1";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        if crate::is_x86_feature_available!("sse4.1") {
            Some(unsafe { Self::new_unchecked() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn new_unchecked() -> Self {
        Self { _private: () }
    }
}

impl Sse41Token {
    /// Get an SSE2 token (SSE4.1 implies SSE2)
    #[inline(always)]
    pub fn sse2(self) -> Sse2Token {
        // SAFETY: SSE4.1 implies SSE2
        unsafe { Sse2Token::new_unchecked() }
    }
}

// ============================================================================
// AVX Token
// ============================================================================

/// Proof that AVX is available.
///
/// AVX provides 256-bit floating-point vectors and VEX encoding.
#[derive(Clone, Copy, Debug)]
pub struct AvxToken {
    _private: (),
}

impl SimdToken for AvxToken {
    const NAME: &'static str = "AVX";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        if crate::is_x86_feature_available!("avx") {
            Some(unsafe { Self::new_unchecked() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn new_unchecked() -> Self {
        Self { _private: () }
    }
}

impl AvxToken {
    /// Get an SSE4.1 token (AVX implies SSE4.1)
    #[inline(always)]
    pub fn sse41(self) -> Sse41Token {
        unsafe { Sse41Token::new_unchecked() }
    }

    /// Get an SSE2 token (AVX implies SSE2)
    #[inline(always)]
    pub fn sse2(self) -> Sse2Token {
        unsafe { Sse2Token::new_unchecked() }
    }
}

// ============================================================================
// AVX2 Token
// ============================================================================

/// Proof that AVX2 is available.
///
/// AVX2 provides 256-bit integer operations and gather instructions.
/// This is the most commonly targeted feature level for SIMD code.
#[derive(Clone, Copy, Debug)]
pub struct Avx2Token {
    _private: (),
}

impl SimdToken for Avx2Token {
    const NAME: &'static str = "AVX2";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        if crate::is_x86_feature_available!("avx2") {
            Some(unsafe { Self::new_unchecked() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn new_unchecked() -> Self {
        Self { _private: () }
    }
}

impl Avx2Token {
    /// Get an AVX token (AVX2 implies AVX)
    #[inline(always)]
    pub fn avx(self) -> AvxToken {
        unsafe { AvxToken::new_unchecked() }
    }

    /// Get an SSE4.1 token (AVX2 implies SSE4.1)
    #[inline(always)]
    pub fn sse41(self) -> Sse41Token {
        unsafe { Sse41Token::new_unchecked() }
    }

    /// Get an SSE2 token (AVX2 implies SSE2)
    #[inline(always)]
    pub fn sse2(self) -> Sse2Token {
        unsafe { Sse2Token::new_unchecked() }
    }
}

// ============================================================================
// FMA Token
// ============================================================================

/// Proof that FMA (Fused Multiply-Add) is available.
///
/// FMA is independent of AVX2 in the feature hierarchy but almost always
/// available together on modern CPUs. Use `Avx2FmaToken` for the common case.
#[derive(Clone, Copy, Debug)]
pub struct FmaToken {
    _private: (),
}

impl SimdToken for FmaToken {
    const NAME: &'static str = "FMA";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        if crate::is_x86_feature_available!("fma") {
            Some(unsafe { Self::new_unchecked() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn new_unchecked() -> Self {
        Self { _private: () }
    }
}

// ============================================================================
// Combined AVX2 + FMA Token
// ============================================================================

/// Combined proof that both AVX2 and FMA are available.
///
/// This is the most common token for floating-point SIMD work.
/// Almost all CPUs with AVX2 also have FMA (Haswell and later).
#[derive(Clone, Copy, Debug)]
pub struct Avx2FmaToken {
    avx2: Avx2Token,
    fma: FmaToken,
}

impl SimdToken for Avx2FmaToken {
    const NAME: &'static str = "AVX2+FMA";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        // Both checks use compile-time optimization when features are known
        if crate::is_x86_feature_available!("avx2") && crate::is_x86_feature_available!("fma") {
            Some(unsafe { Self::new_unchecked() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn new_unchecked() -> Self {
        Self {
            avx2: unsafe { Avx2Token::new_unchecked() },
            fma: unsafe { FmaToken::new_unchecked() },
        }
    }
}

impl CompositeToken for Avx2FmaToken {
    type Components = (Avx2Token, FmaToken);

    #[inline(always)]
    fn components(&self) -> Self::Components {
        (self.avx2, self.fma)
    }
}

impl Avx2FmaToken {
    /// Get the AVX2 component token
    #[inline(always)]
    pub fn avx2(&self) -> Avx2Token {
        self.avx2
    }

    /// Get the FMA component token
    #[inline(always)]
    pub fn fma(&self) -> FmaToken {
        self.fma
    }

    /// Get an AVX token
    #[inline(always)]
    pub fn avx(&self) -> AvxToken {
        self.avx2.avx()
    }

    /// Get an SSE4.1 token
    #[inline(always)]
    pub fn sse41(&self) -> Sse41Token {
        self.avx2.sse41()
    }

    /// Get an SSE2 token
    #[inline(always)]
    pub fn sse2(&self) -> Sse2Token {
        self.avx2.sse2()
    }
}

// ============================================================================
// AVX-512 Tokens
// ============================================================================

/// Proof that AVX-512F (Foundation) is available.
///
/// AVX-512F is the base AVX-512 feature set with 512-bit vectors.
#[derive(Clone, Copy, Debug)]
pub struct Avx512fToken {
    _private: (),
}

impl SimdToken for Avx512fToken {
    const NAME: &'static str = "AVX-512F";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        if crate::is_x86_feature_available!("avx512f") {
            Some(unsafe { Self::new_unchecked() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn new_unchecked() -> Self {
        Self { _private: () }
    }
}

impl Avx512fToken {
    /// Get an AVX2 token (AVX-512F implies AVX2)
    #[inline(always)]
    pub fn avx2(self) -> Avx2Token {
        unsafe { Avx2Token::new_unchecked() }
    }

    /// Get an FMA token (AVX-512F implies FMA)
    #[inline(always)]
    pub fn fma(self) -> FmaToken {
        unsafe { FmaToken::new_unchecked() }
    }

    /// Get a combined AVX2+FMA token
    #[inline(always)]
    pub fn avx2_fma(self) -> Avx2FmaToken {
        unsafe { Avx2FmaToken::new_unchecked() }
    }
}

/// Proof that AVX-512BW (Byte and Word) is available.
#[derive(Clone, Copy, Debug)]
pub struct Avx512bwToken {
    _private: (),
}

impl SimdToken for Avx512bwToken {
    const NAME: &'static str = "AVX-512BW";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        if crate::is_x86_feature_available!("avx512bw") {
            Some(unsafe { Self::new_unchecked() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn new_unchecked() -> Self {
        Self { _private: () }
    }
}

impl Avx512bwToken {
    /// Get an AVX-512F token (AVX-512BW implies AVX-512F)
    #[inline(always)]
    pub fn avx512f(self) -> Avx512fToken {
        unsafe { Avx512fToken::new_unchecked() }
    }
}

// ============================================================================
// Assembly verification helpers
// ============================================================================

/// Helper to verify Avx2Token::try_new assembly.
/// Without +avx2: runtime detection. With +avx2: just returns Some.
#[inline(never)]
pub fn verify_avx2_try_new() -> Option<Avx2Token> {
    Avx2Token::try_new()
}

/// Helper to verify Avx2FmaToken::try_new assembly.
#[inline(never)]
pub fn verify_avx2_fma_try_new() -> Option<Avx2FmaToken> {
    Avx2FmaToken::try_new()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse2_always_available() {
        // SSE2 is baseline for x86_64
        assert!(Sse2Token::try_new().is_some());
    }

    #[test]
    fn test_token_is_zst() {
        // Tokens should be zero-sized
        assert_eq!(core::mem::size_of::<Sse2Token>(), 0);
        assert_eq!(core::mem::size_of::<Avx2Token>(), 0);
        assert_eq!(core::mem::size_of::<FmaToken>(), 0);
        // Combined token is also ZST (contains two ZSTs)
        assert_eq!(core::mem::size_of::<Avx2FmaToken>(), 0);
    }

    #[test]
    fn test_token_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<Sse2Token>();
        assert_copy::<Avx2Token>();
        assert_copy::<FmaToken>();
        assert_copy::<Avx2FmaToken>();
    }

    #[test]
    fn test_runtime_detection() {
        // These may or may not be available depending on CPU
        let _avx2 = Avx2Token::try_new();
        let _fma = FmaToken::try_new();
        let _avx2_fma = Avx2FmaToken::try_new();

        // If AVX2+FMA available, test component access
        if let Some(token) = Avx2FmaToken::try_new() {
            let _avx2 = token.avx2();
            let _fma = token.fma();
            let _avx = token.avx();
            let _sse41 = token.sse41();
            let _sse2 = token.sse2();
        }
    }

    #[test]
    fn test_token_hierarchy() {
        if let Some(avx2) = Avx2Token::try_new() {
            // AVX2 implies AVX, SSE4.1, SSE2
            let _avx = avx2.avx();
            let _sse41 = avx2.sse41();
            let _sse2 = avx2.sse2();
        }
    }
}

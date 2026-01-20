//! ARM SIMD capability tokens
//!
//! Provides tokens for NEON, SVE, and SVE2.
//!
//! Token construction uses runtime feature detection via `std::arch::is_aarch64_feature_detected!`
//! combined with compile-time checks. On AArch64, NEON is always available (baseline).

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
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is always available on AArch64
            Some(Self { _private: () })
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            None
        }
    }

    #[inline(always)]
    unsafe fn new_unchecked() -> Self {
        Self { _private: () }
    }
}

// ============================================================================
// SVE Token
// ============================================================================

/// Proof that SVE (Scalable Vector Extension) is available.
#[derive(Clone, Copy, Debug)]
pub struct SveToken {
    _private: (),
}

impl SimdToken for SveToken {
    const NAME: &'static str = "SVE";

    #[inline]
    fn try_new() -> Option<Self> {
        #[cfg(all(target_arch = "aarch64", feature = "std"))]
        {
            if std::arch::is_aarch64_feature_detected!("sve") {
                return Some(unsafe { Self::new_unchecked() });
            }
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "sve"))]
        {
            // Compile-time guarantee
            return Some(unsafe { Self::new_unchecked() });
        }
        #[allow(unreachable_code)]
        None
    }

    #[inline(always)]
    unsafe fn new_unchecked() -> Self {
        Self { _private: () }
    }
}

impl SveToken {
    /// Get a NEON token (SVE implies NEON)
    #[inline(always)]
    pub fn neon(self) -> NeonToken {
        unsafe { NeonToken::new_unchecked() }
    }
}

// ============================================================================
// SVE2 Token
// ============================================================================

/// Proof that SVE2 is available.
#[derive(Clone, Copy, Debug)]
pub struct Sve2Token {
    _private: (),
}

impl SimdToken for Sve2Token {
    const NAME: &'static str = "SVE2";

    #[inline]
    fn try_new() -> Option<Self> {
        #[cfg(all(target_arch = "aarch64", feature = "std"))]
        {
            if std::arch::is_aarch64_feature_detected!("sve2") {
                return Some(unsafe { Self::new_unchecked() });
            }
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "sve2"))]
        {
            // Compile-time guarantee
            return Some(unsafe { Self::new_unchecked() });
        }
        #[allow(unreachable_code)]
        None
    }

    #[inline(always)]
    unsafe fn new_unchecked() -> Self {
        Self { _private: () }
    }
}

impl Sve2Token {
    /// Get an SVE token (SVE2 implies SVE)
    #[inline(always)]
    pub fn sve(self) -> SveToken {
        unsafe { SveToken::new_unchecked() }
    }

    /// Get a NEON token (SVE2 implies NEON)
    #[inline(always)]
    pub fn neon(self) -> NeonToken {
        unsafe { NeonToken::new_unchecked() }
    }
}

// ============================================================================
// Capability Marker Trait Implementations
// ============================================================================

use super::{Has128BitSimd, HasFma, HasScalableVectors};

// NEON provides 128-bit SIMD and FMA
impl Has128BitSimd for NeonToken {}
impl HasFma for NeonToken {} // NEON has fused multiply-add instructions

// SVE provides scalable vectors and inherits NEON capabilities
impl Has128BitSimd for SveToken {}
impl HasFma for SveToken {}
impl HasScalableVectors for SveToken {}

// SVE2 extends SVE
impl Has128BitSimd for Sve2Token {}
impl HasFma for Sve2Token {}
impl HasScalableVectors for Sve2Token {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_is_zst() {
        assert_eq!(core::mem::size_of::<NeonToken>(), 0);
        assert_eq!(core::mem::size_of::<SveToken>(), 0);
        assert_eq!(core::mem::size_of::<Sve2Token>(), 0);
    }

    #[test]
    fn test_token_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<NeonToken>();
        assert_copy::<SveToken>();
        assert_copy::<Sve2Token>();
    }

    #[test]
    fn test_token_names() {
        assert_eq!(NeonToken::NAME, "NEON");
        assert_eq!(SveToken::NAME, "SVE");
        assert_eq!(Sve2Token::NAME, "SVE2");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_always_available() {
        // NEON is baseline for AArch64
        assert!(NeonToken::try_new().is_some());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_token_hierarchy() {
        // If SVE2 is available, SVE and NEON must also be available
        if Sve2Token::try_new().is_some() {
            assert!(SveToken::try_new().is_some(), "SVE2 implies SVE");
            assert!(NeonToken::try_new().is_some(), "SVE2 implies NEON");
        }

        // If SVE is available, NEON must also be available
        if SveToken::try_new().is_some() {
            assert!(NeonToken::try_new().is_some(), "SVE implies NEON");
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_sve2_token_extraction() {
        if let Some(sve2) = Sve2Token::try_new() {
            let _sve = sve2.sve();
            let _neon = sve2.neon();
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_sve_token_extraction() {
        if let Some(sve) = SveToken::try_new() {
            let _neon = sve.neon();
        }
    }
}

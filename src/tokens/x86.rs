//! x86_64 SIMD capability tokens
//!
//! Provides tokens for SSE4.2, AVX, AVX2, AVX-512, and FMA.
//!
//! Token construction uses [`crate::is_x86_feature_available!`] which combines
//! compile-time and runtime detection. When compiled with a target feature
//! enabled (e.g., in a `#[multiversed]` function), the runtime check is
//! completely eliminated.
//!
//! ## Baseline: SSE4.2
//!
//! All tokens in archmage assume SSE4.2 as the baseline. Tokens below SSE4.2
//! (SSE, SSE2, SSE3, SSSE3, SSE4.1) are not provided - this simplifies the
//! hierarchy and matches practical modern CPU availability (Nehalem 2008+).
//!
//! ## Explicit Feature Verification
//!
//! All tokens explicitly check ALL features they claim to provide. This ensures
//! soundness - a token's trait implementations exactly match its runtime checks.

use super::SimdToken;

// Re-export AVX-512 tokens from the dedicated module
pub use super::x86_avx512::{Avx512Fp16Token, Avx512ModernToken, Avx512Token, X64V4Token};

// ============================================================================
// SSE4.2 Token (baseline for archmage)
// ============================================================================

/// Proof that SSE4.2 is available.
///
/// SSE4.2 is the practical baseline for archmage. It's available on all x86_64
/// CPUs from 2008+ (Nehalem, Bulldozer, and later).
#[derive(Clone, Copy, Debug)]
pub struct Sse42Token {
    _private: (),
}

impl SimdToken for Sse42Token {
    const NAME: &'static str = "SSE4.2";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        // Explicit check - SSE4.2 is our baseline
        if crate::is_x86_feature_available!("sse4.2") {
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
        // Explicit cumulative check: AVX + SSE4.2
        if crate::is_x86_feature_available!("avx")
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

impl AvxToken {
    /// Get an SSE4.2 token (AVX implies SSE4.2)
    #[inline(always)]
    pub fn sse42(self) -> Sse42Token {
        unsafe { Sse42Token::forge_token_dangerously() }
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
        // Explicit cumulative check: AVX2 + AVX + SSE4.2
        if crate::is_x86_feature_available!("avx2")
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

impl Avx2Token {
    /// Get an AVX token (AVX2 implies AVX)
    #[inline(always)]
    pub fn avx(self) -> AvxToken {
        unsafe { AvxToken::forge_token_dangerously() }
    }

    /// Get an SSE4.2 token (AVX2 implies SSE4.2)
    #[inline(always)]
    pub fn sse42(self) -> Sse42Token {
        unsafe { Sse42Token::forge_token_dangerously() }
    }
}

// ============================================================================
// AVX2 + FMA Token
// ============================================================================

/// Proof that AVX2 + FMA are available.
///
/// This is the standard token for floating-point SIMD work.
/// All CPUs with AVX2 also have FMA (Haswell 2013+, Zen 1+).
///
/// In archmage's model, FMA requires AVX2 - use this token for FMA operations.
#[derive(Clone, Copy, Debug)]
pub struct Avx2FmaToken {
    _private: (),
}

impl SimdToken for Avx2FmaToken {
    const NAME: &'static str = "AVX2+FMA";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        // Explicit cumulative check: AVX2 + FMA + AVX + SSE4.2
        if crate::is_x86_feature_available!("avx2")
            && crate::is_x86_feature_available!("fma")
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

impl Avx2FmaToken {
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

// ============================================================================
// x86-64 Microarchitecture Level Tokens (Profiles)
// ============================================================================
//
// These match the x86-64 psABI microarchitecture levels:
// https://gitlab.com/x86-psABIs/x86-64-ABI
//
// | Level | Key Features                           | Hardware              |
// |-------|----------------------------------------|-----------------------|
// | v3    | AVX2, FMA, BMI1, BMI2, F16C, LZCNT     | Haswell 2013+, Zen 1+ |
// | v4    | + AVX-512F/BW/CD/DQ/VL                 | Xeon 2017+, Zen 4+    |
//
// Note: v2 (SSE4.2 + POPCNT) is below our baseline and not provided as a token.

/// Proof that AVX2 + FMA + BMI1/2 are available (x86-64-v3 level).
///
/// x86-64-v3 = AVX2 + FMA + BMI1 + BMI2 + F16C + LZCNT + MOVBE + POPCNT + SSE4.2
/// This is the Haswell (2013) / Zen 1 (2017) baseline.
///
/// This is the recommended baseline for modern desktop SIMD code.
#[derive(Clone, Copy, Debug)]
pub struct X64V3Token {
    _private: (),
}

impl SimdToken for X64V3Token {
    const NAME: &'static str = "x86-64-v3";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        // Explicit cumulative check: all v3 features
        if crate::is_x86_feature_available!("avx2")
            && crate::is_x86_feature_available!("fma")
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

impl X64V3Token {
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

// ============================================================================
// Capability Marker Trait Implementations
// ============================================================================

use super::sealed::Sealed;
use super::{Has128BitSimd, Has256BitSimd, Has512BitSimd};
use super::{
    HasAvx, HasAvx2, HasAvx2Fma, HasAvx512, HasDesktop64, HasModernAvx512, HasSse42, HasX64V3,
    HasX64V4,
};

// Sealed trait implementations (required for marker traits)
impl Sealed for Sse42Token {}
impl Sealed for AvxToken {}
impl Sealed for Avx2Token {}
impl Sealed for Avx2FmaToken {}
impl Sealed for X64V3Token {}
// AVX-512 tokens are sealed in x86_avx512.rs

// HasSse42: All x86 tokens (SSE4.2 is baseline)
impl HasSse42 for Sse42Token {}
impl HasSse42 for AvxToken {}
impl HasSse42 for Avx2Token {}
impl HasSse42 for Avx2FmaToken {}
impl HasSse42 for X64V3Token {}
impl HasSse42 for Avx512Token {}
impl HasSse42 for Avx512ModernToken {}
impl HasSse42 for Avx512Fp16Token {}

// HasAvx: AVX and above
impl HasAvx for AvxToken {}
impl HasAvx for Avx2Token {}
impl HasAvx for Avx2FmaToken {}
impl HasAvx for X64V3Token {}
impl HasAvx for Avx512Token {}
impl HasAvx for Avx512ModernToken {}
impl HasAvx for Avx512Fp16Token {}

// HasAvx2: AVX2 and above
impl HasAvx2 for Avx2Token {}
impl HasAvx2 for Avx2FmaToken {}
impl HasAvx2 for X64V3Token {}
impl HasAvx2 for Avx512Token {}
impl HasAvx2 for Avx512ModernToken {}
impl HasAvx2 for Avx512Fp16Token {}

// HasAvx2Fma: AVX2 + FMA (requires AVX2 in our model)
impl HasAvx2Fma for Avx2FmaToken {}
impl HasAvx2Fma for X64V3Token {}
impl HasAvx2Fma for Avx512Token {}
impl HasAvx2Fma for Avx512ModernToken {}
impl HasAvx2Fma for Avx512Fp16Token {}

// HasX64V3: x86-64-v3 level (AVX2 + FMA + BMI2)
impl HasX64V3 for X64V3Token {}
impl HasX64V3 for Avx512Token {}
impl HasX64V3 for Avx512ModernToken {}
impl HasX64V3 for Avx512Fp16Token {}

// HasDesktop64: alias for HasX64V3
impl HasDesktop64 for X64V3Token {}
impl HasDesktop64 for Avx512Token {}
impl HasDesktop64 for Avx512ModernToken {}
impl HasDesktop64 for Avx512Fp16Token {}

// HasAvx512: AVX-512 F+CD+VL+DQ+BW (x86-64-v4 level)
impl HasAvx512 for Avx512Token {}
impl HasAvx512 for Avx512ModernToken {}
impl HasAvx512 for Avx512Fp16Token {}

// HasX64V4: alias for HasAvx512
impl HasX64V4 for Avx512Token {}
impl HasX64V4 for Avx512ModernToken {}
impl HasX64V4 for Avx512Fp16Token {}

// HasModernAvx512: modern AVX-512 (Ice Lake / Zen 4)
impl HasModernAvx512 for Avx512ModernToken {}
// Note: Avx512Fp16Token does NOT impl HasModernAvx512 (FP16 is separate from modern features)

// ============================================================================
// Width Marker Trait Implementations
// ============================================================================

// 128-bit SIMD: SSE4.2+
impl Has128BitSimd for Sse42Token {}

// 256-bit SIMD: AVX+
impl Has128BitSimd for AvxToken {}
impl Has256BitSimd for AvxToken {}
impl Has128BitSimd for Avx2Token {}
impl Has256BitSimd for Avx2Token {}
impl Has128BitSimd for Avx2FmaToken {}
impl Has256BitSimd for Avx2FmaToken {}
impl Has128BitSimd for X64V3Token {}
impl Has256BitSimd for X64V3Token {}

// 512-bit SIMD: AVX-512+
impl Has128BitSimd for Avx512Token {}
impl Has256BitSimd for Avx512Token {}
impl Has512BitSimd for Avx512Token {}
impl Has128BitSimd for Avx512ModernToken {}
impl Has256BitSimd for Avx512ModernToken {}
impl Has512BitSimd for Avx512ModernToken {}
impl Has128BitSimd for Avx512Fp16Token {}
impl Has256BitSimd for Avx512Fp16Token {}
impl Has512BitSimd for Avx512Fp16Token {}

// ============================================================================
// Friendly Aliases
// ============================================================================

/// The recommended baseline for desktop x86_64 (AVX2 + FMA + BMI2).
///
/// This is an alias for [`X64V3Token`], covering all Intel Haswell (2013+) and
/// AMD Zen 1 (2017+) desktop CPUs. Use this as your starting point for desktop
/// applications.
///
/// # Why Desktop64?
///
/// - **Universal on modern desktops**: Every x86_64 desktop/laptop CPU since 2013
/// - **Best performance/compatibility tradeoff**: AVX2 gives 256-bit vectors, FMA
///   enables fused multiply-add
/// - **Excludes AVX-512**: Intel removed AVX-512 from consumer chips (12th-14th gen)
///   due to hybrid P+E core architecture, making it unreliable for desktop targeting
///
/// # Example
///
/// ```rust,ignore
/// use archmage::{Desktop64, SimdToken, arcane};
///
/// #[arcane]
/// fn process(token: Desktop64, data: &mut [f32; 8]) {
///     // AVX2 + FMA intrinsics safe here
/// }
///
/// if let Some(token) = Desktop64::try_new() {
///     process(token, &mut data);
/// }
/// ```
pub type Desktop64 = X64V3Token;


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
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod tests {
    use super::*;

    #[test]
    fn test_sse42_baseline() {
        // SSE4.2 is the baseline for archmage
        // Most modern x86_64 CPUs have SSE4.2 (Nehalem 2008+)
        let _sse42 = Sse42Token::try_new();
    }

    #[test]
    fn test_token_is_zst() {
        // Tokens should be zero-sized
        assert_eq!(core::mem::size_of::<Sse42Token>(), 0);
        assert_eq!(core::mem::size_of::<Avx2Token>(), 0);
        assert_eq!(core::mem::size_of::<Avx2FmaToken>(), 0);
    }

    #[test]
    fn test_token_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<Sse42Token>();
        assert_copy::<Avx2Token>();
        assert_copy::<Avx2FmaToken>();
    }

    #[test]
    fn test_runtime_detection() {
        // These may or may not be available depending on CPU
        let _avx2 = Avx2Token::try_new();
        let _avx2_fma = Avx2FmaToken::try_new();

        // If AVX2+FMA available, test component access
        if let Some(token) = Avx2FmaToken::try_new() {
            let _avx2 = token.avx2();
            let _avx = token.avx();
            let _sse42 = token.sse42();
        }
    }

    #[test]
    fn test_token_hierarchy() {
        if let Some(avx2) = Avx2Token::try_new() {
            // AVX2 implies AVX, SSE4.2
            let _avx = avx2.avx();
            let _sse42 = avx2.sse42();
        }
    }

    #[test]
    fn test_profile_tokens_zst() {
        // Profile tokens should also be zero-sized
        assert_eq!(core::mem::size_of::<X64V3Token>(), 0);
        assert_eq!(core::mem::size_of::<X64V4Token>(), 0);
        assert_eq!(core::mem::size_of::<Avx512Token>(), 0);
    }

    #[test]
    fn test_v3_token_extraction() {
        if let Some(v3) = X64V3Token::try_new() {
            // v3 can extract AVX2+FMA, AVX2, AVX, SSE4.2
            let _avx2_fma = v3.avx2_fma();
            let _avx2 = v3.avx2();
            let _avx = v3.avx();
            let _sse42 = v3.sse42();
        }
    }

    #[test]
    fn test_v4_token_extraction() {
        if let Some(v4) = X64V4Token::try_new() {
            // v4 can extract v3, AVX2+FMA, etc.
            let _v3 = v4.v3();
            let _avx2_fma = v4.avx2_fma();
            let _avx2 = v4.avx2();
            let _avx = v4.avx();
            let _sse42 = v4.sse42();
        }
    }

    #[test]
    fn test_profile_hierarchy_consistency() {
        // If v4 is available, v3 should also be available
        if X64V4Token::try_new().is_some() {
            assert!(
                X64V3Token::try_new().is_some(),
                "v4 implies v3 should be available"
            );
        }
    }

    #[test]
    fn test_profile_token_names() {
        assert_eq!(X64V3Token::NAME, "x86-64-v3");
        assert_eq!(X64V4Token::NAME, "x86-64-v4");
        assert_eq!(Sse42Token::NAME, "SSE4.2");
    }

    // ========================================================================
    // Operation Trait Tests (require composite feature)
    // ========================================================================
    #[cfg(feature = "__composite")]
    mod simd_ops_tests {
        use super::*;
        use crate::composite::simd_ops::{DotProduct, HorizontalOps, Transpose8x8};

        #[test]
        fn test_transpose_trait() {
            if let Some(token) = Avx2Token::try_new() {
                let original: [f32; 64] = core::array::from_fn(|i| i as f32);
                let mut block = original;

                // Use trait method
                token.transpose_8x8(&mut block);

                // Verify transpose
                for row in 0..8 {
                    for col in 0..8 {
                        assert_eq!(original[row * 8 + col], block[col * 8 + row]);
                    }
                }
            }
        }

        #[test]
        fn test_transpose_trait_via_profile() {
            if let Some(token) = X64V3Token::try_new() {
                let original: [f32; 64] = core::array::from_fn(|i| i as f32);
                let mut block = original;

                // Use trait method via profile token
                token.transpose_8x8(&mut block);

                for row in 0..8 {
                    for col in 0..8 {
                        assert_eq!(original[row * 8 + col], block[col * 8 + row]);
                    }
                }
            }
        }

        #[test]
        fn test_dot_product_trait() {
            if let Some(token) = Avx2FmaToken::try_new() {
                let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
                let b: Vec<f32> = vec![1.0; 64];

                // Use trait method
                let result = token.dot_product_f32(&a, &b);
                let expected: f32 = (0..64).map(|i| i as f32).sum();

                assert!((result - expected).abs() < 0.001);
            }
        }

        #[test]
        fn test_horizontal_ops_trait() {
            if let Some(token) = Avx2Token::try_new() {
                let data: Vec<f32> = (1..=100).map(|i| i as f32).collect();

                // Use trait methods
                let sum = token.sum_f32(&data);
                let max = token.max_f32(&data);
                let min = token.min_f32(&data);

                assert!((sum - 5050.0).abs() < 0.001);
                assert!((max - 100.0).abs() < 0.001);
                assert!((min - 1.0).abs() < 0.001);
            }
        }

        #[test]
        fn test_generic_trait_bounds() {
            // This tests that we can write generic code over tokens
            fn process_transpose<T: Transpose8x8>(token: T) {
                let mut block: [f32; 64] = core::array::from_fn(|i| i as f32);
                token.transpose_8x8(&mut block);
            }

            fn process_dot<T: DotProduct>(token: T, a: &[f32], b: &[f32]) -> f32 {
                token.dot_product_f32(a, b)
            }

            fn process_horizontal<T: HorizontalOps>(token: T, data: &[f32]) -> f32 {
                token.sum_f32(data)
            }

            // These compile, proving the traits work
            if let Some(token) = Avx2Token::try_new() {
                process_transpose(token);
                let data = vec![1.0f32; 16];
                let _sum = process_horizontal(token, &data);
                let _dot = process_dot(token, &data, &data);
            }

            if let Some(token) = X64V3Token::try_new() {
                process_transpose(token);
                let data = vec![1.0f32; 16];
                let _sum = process_horizontal(token, &data);
                let _dot = process_dot(token, &data, &data);
            }
        }
    }
}

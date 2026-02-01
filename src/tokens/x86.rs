//! x86_64 SIMD capability tokens
//!
//! Provides tier-level tokens matching LLVM x86-64 microarchitecture levels.
//!
//! Token construction uses [`crate::is_x86_feature_available!`] which combines
//! compile-time and runtime detection. When compiled with a target feature
//! enabled (e.g., in a `#[multiversed]` function), the runtime check is
//! completely eliminated.

use super::SimdToken;

// Re-export AVX-512 tokens from the dedicated module
#[cfg(feature = "avx512")]
pub use super::x86_avx512::{
    Avx512Fp16Token, Avx512ModernToken, Avx512Token, Server64, X64V4Token,
};

// ============================================================================
// x86-64 Microarchitecture Level Tokens
// ============================================================================
//
// These match the x86-64 psABI microarchitecture levels:
// https://gitlab.com/x86-psABIs/x86-64-ABI
//
// | Level | Key Features                           | Hardware              |
// |-------|----------------------------------------|-----------------------|
// | v1    | SSE, SSE2 (baseline x86_64)            | All x86_64            |
// | v2    | + SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT  | Nehalem 2008+         |
// | v3    | + AVX, AVX2, FMA, BMI1, BMI2, F16C     | Haswell 2013+, Zen 1+ |
// | v4    | + AVX-512F/BW/CD/DQ/VL                 | Xeon 2017+, Zen 4+    |

/// Proof that SSE4.2 + POPCNT are available (x86-64-v2 level).
///
/// x86-64-v2 implies: SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT, CX16, SAHF.
/// This is the Nehalem (2008) / Bulldozer (2011) baseline.
#[derive(Clone, Copy, Debug)]
pub struct X64V2Token {
    _private: (),
}

impl SimdToken for X64V2Token {
    const NAME: &'static str = "x86-64-v2";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        // Explicitly check all v2 features for robustness against broken emulators
        if crate::is_x86_feature_available!("sse3")
            && crate::is_x86_feature_available!("ssse3")
            && crate::is_x86_feature_available!("sse4.1")
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

impl X64V2Token {
    /// Get a v2 token from self (identity, for generic code)
    #[inline(always)]
    pub fn v2(self) -> X64V2Token {
        self
    }
}

/// Proof that AVX2 + FMA + BMI1/2 + F16C + LZCNT are available (x86-64-v3 level).
///
/// x86-64-v3 implies all of v2 plus: AVX, AVX2, FMA, BMI1, BMI2, F16C, LZCNT, MOVBE.
/// This is the Haswell (2013) / Zen 1 (2017) baseline.
///
/// This is the most commonly targeted level for high-performance SIMD code.
#[derive(Clone, Copy, Debug)]
pub struct X64V3Token {
    _private: (),
}

impl SimdToken for X64V3Token {
    const NAME: &'static str = "x86-64-v3";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        // Explicitly check all v3 features for robustness against broken emulators
        if crate::is_x86_feature_available!("sse3")
            && crate::is_x86_feature_available!("ssse3")
            && crate::is_x86_feature_available!("sse4.1")
            && crate::is_x86_feature_available!("sse4.2")
            && crate::is_x86_feature_available!("popcnt")
            && crate::is_x86_feature_available!("avx")
            && crate::is_x86_feature_available!("avx2")
            && crate::is_x86_feature_available!("fma")
            && crate::is_x86_feature_available!("bmi1")
            && crate::is_x86_feature_available!("bmi2")
            && crate::is_x86_feature_available!("f16c")
            && crate::is_x86_feature_available!("lzcnt")
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
    /// Get a v2 token (v3 implies v2)
    #[inline(always)]
    pub fn v2(self) -> X64V2Token {
        unsafe { X64V2Token::forge_token_dangerously() }
    }
}

// ============================================================================
// Tier Marker Trait Implementations
// ============================================================================

use super::{Has128BitSimd, Has256BitSimd, HasX64V2};
#[cfg(feature = "avx512")]
use super::{Has512BitSimd, HasX64V4};

// HasX64V2: v2 and above
impl HasX64V2 for X64V2Token {}
impl HasX64V2 for X64V3Token {}

// Width traits: 128-bit
impl Has128BitSimd for X64V2Token {}
impl Has128BitSimd for X64V3Token {}

// Width traits: 256-bit
impl Has256BitSimd for X64V3Token {}

// ============================================================================
// AVX-512 tier trait implementations (requires "avx512" feature)
// ============================================================================
#[cfg(feature = "avx512")]
mod avx512_tier_impls {
    use super::*;

    // HasX64V2 for AVX-512 tokens (v4 implies v2)
    impl HasX64V2 for X64V4Token {}
    impl HasX64V2 for Avx512ModernToken {}
    impl HasX64V2 for Avx512Fp16Token {}

    // HasX64V4: v4 and above
    impl HasX64V4 for X64V4Token {}
    impl HasX64V4 for Avx512ModernToken {}
    impl HasX64V4 for Avx512Fp16Token {}

    // Width traits for AVX-512 tokens
    impl Has128BitSimd for X64V4Token {}
    impl Has256BitSimd for X64V4Token {}
    impl Has512BitSimd for X64V4Token {}

    impl Has128BitSimd for Avx512ModernToken {}
    impl Has256BitSimd for Avx512ModernToken {}
    impl Has512BitSimd for Avx512ModernToken {}

    impl Has128BitSimd for Avx512Fp16Token {}
    impl Has256BitSimd for Avx512Fp16Token {}
    impl Has512BitSimd for Avx512Fp16Token {}
}

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

/// Type alias for backward compatibility. `Avx2FmaToken` is now [`X64V3Token`].
///
/// The full x86-64-v3 level includes AVX2 + FMA plus BMI1, BMI2, F16C, LZCNT.
pub type Avx2FmaToken = X64V3Token;

// ============================================================================
// Assembly verification helpers
// ============================================================================

/// Helper to verify X64V3Token::try_new assembly.
/// Without +avx2: runtime detection. With +avx2: just returns Some.
#[inline(never)]
pub fn verify_v3_try_new() -> Option<X64V3Token> {
    X64V3Token::try_new()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod tests {
    use super::*;

    #[test]
    fn test_token_is_zst() {
        assert_eq!(core::mem::size_of::<X64V2Token>(), 0);
        assert_eq!(core::mem::size_of::<X64V3Token>(), 0);
        // Avx2FmaToken is X64V3Token
        assert_eq!(core::mem::size_of::<Avx2FmaToken>(), 0);
        #[cfg(feature = "avx512")]
        assert_eq!(core::mem::size_of::<X64V4Token>(), 0);
    }

    #[test]
    fn test_token_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<X64V2Token>();
        assert_copy::<X64V3Token>();
        assert_copy::<Avx2FmaToken>();
    }

    #[test]
    fn test_runtime_detection() {
        // These may or may not be available depending on CPU
        let _v2 = X64V2Token::try_new();
        let _v3 = X64V3Token::try_new();
        let _avx2_fma = Avx2FmaToken::try_new();
    }

    #[test]
    fn test_v3_token_extraction() {
        if let Some(v3) = X64V3Token::try_new() {
            let _v2 = v3.v2();
        }
    }

    #[cfg(feature = "avx512")]
    #[test]
    fn test_v4_token_extraction() {
        if let Some(v4) = X64V4Token::try_new() {
            let _v3 = v4.v3();
        }
    }

    #[test]
    fn test_profile_hierarchy_consistency() {
        // If v3 is available, v2 should also be available
        if X64V3Token::try_new().is_some() {
            assert!(
                X64V2Token::try_new().is_some(),
                "v3 implies v2 should be available"
            );
        }

        // If v4 is available, both v3 and v2 should be available
        #[cfg(feature = "avx512")]
        if X64V4Token::try_new().is_some() {
            assert!(
                X64V3Token::try_new().is_some(),
                "v4 implies v3 should be available"
            );
            assert!(
                X64V2Token::try_new().is_some(),
                "v4 implies v2 should be available"
            );
        }
    }

    #[test]
    fn test_profile_token_names() {
        assert_eq!(X64V2Token::NAME, "x86-64-v2");
        assert_eq!(X64V3Token::NAME, "x86-64-v3");
        #[cfg(feature = "avx512")]
        {
            assert_eq!(X64V4Token::NAME, "AVX-512");
            assert_eq!(Avx512Token::NAME, "AVX-512");
        }
    }

    #[test]
    fn test_avx2fma_is_x64v3() {
        // Avx2FmaToken is now a type alias for X64V3Token
        assert_eq!(
            core::mem::size_of::<Avx2FmaToken>(),
            core::mem::size_of::<X64V3Token>()
        );
        assert_eq!(Avx2FmaToken::NAME, X64V3Token::NAME);
    }

    // ========================================================================
    // Operation Trait Tests (require composite feature)
    // ========================================================================
    #[cfg(feature = "__composite")]
    mod simd_ops_tests {
        use super::*;
        use crate::composite::simd_ops::{DotProduct, HorizontalOps, Transpose8x8};

        #[test]
        fn test_transpose_trait_via_profile() {
            if let Some(token) = X64V3Token::try_new() {
                let original: [f32; 64] = core::array::from_fn(|i| i as f32);
                let mut block = original;

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
            if let Some(token) = X64V3Token::try_new() {
                let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
                let b: Vec<f32> = vec![1.0; 64];

                let result = token.dot_product_f32(&a, &b);
                let expected: f32 = (0..64).map(|i| i as f32).sum();

                assert!((result - expected).abs() < 0.001);
            }
        }

        #[test]
        fn test_horizontal_ops_trait() {
            if let Some(token) = X64V3Token::try_new() {
                let data: Vec<f32> = (1..=100).map(|i| i as f32).collect();

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

            if let Some(token) = X64V3Token::try_new() {
                process_transpose(token);
                let data = vec![1.0f32; 16];
                let _sum = process_horizontal(token, &data);
                let _dot = process_dot(token, &data, &data);
            }
        }
    }
}

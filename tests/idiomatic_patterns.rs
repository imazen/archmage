//! Idiomatic patterns for archmage usage.
//!
//! This file documents ALL supported patterns for using `#[arcane]` and related macros.
//! Each pattern is tested to verify it compiles and runs correctly.
//!
//! ## Pattern Summary
//!
//! | Pattern | Works? | Notes |
//! |---------|--------|-------|
//! | Concrete token (`X64V3Token`) | ✅ | Recommended for most code |
//! | Feature trait (`impl HasX64V2`) | ✅ | Only enables that trait's features |
//! | Width trait (`impl Has256BitSimd`) | ⚠️ | Being removed - don't use |
//! | Generic (`T: SimdToken`) | ❌ | Can't determine features at compile time |
//! | `_self` for trait impls | ✅ | Token must be 2nd parameter |
//! | Token passthrough | ✅ | Call other `#[arcane]` fns with same token |
//! | Nested `#[arcane]` calls | ✅ | Inner fns get same target_feature context |

#![allow(dead_code, unused_variables, unused_imports)]

// =============================================================================
// PATTERN 1: Concrete Token Types (RECOMMENDED)
// =============================================================================
//
// The simplest and most efficient pattern. Use concrete token types like
// X64V3Token, X64V4Token, NeonToken, etc.
//
// Pros:
// - Compiler knows exact features → optimal codegen
// - Clear what features are being used
// - Works with all intrinsics for that feature level
//
// Cons:
// - Must write separate functions per platform

#[cfg(target_arch = "x86_64")]
mod pattern_concrete_token {
    use archmage::{arcane, Desktop64, SimdToken, X64V3Token};
    use core::arch::x86_64::*;

    /// Basic function with concrete X64V3Token
    #[arcane]
    pub fn sum_f32x8(token: X64V3Token, data: &[f32; 8]) -> f32 {
        let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };
        // Horizontal sum using AVX
        let sum1 = _mm256_hadd_ps(v, v);
        let sum2 = _mm256_hadd_ps(sum1, sum1);
        let low = _mm256_castps256_ps128(sum2);
        let high = _mm256_extractf128_ps::<1>(sum2);
        let final_sum = _mm_add_ss(low, high);
        unsafe { _mm_cvtss_f32(final_sum) }
    }

    /// Using Desktop64 alias (same as X64V3Token)
    #[arcane]
    pub fn fma_f32x8(token: Desktop64, a: &[f32; 8], b: &[f32; 8], c: &[f32; 8]) -> [f32; 8] {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr()) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr()) };
        let vc = unsafe { _mm256_loadu_ps(c.as_ptr()) };
        let result = _mm256_fmadd_ps(va, vb, vc);
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), result) };
        out
    }

    #[test]
    fn test_concrete_token() {
        if let Some(token) = X64V3Token::summon() {
            let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let sum = sum_f32x8(token, &data);
            assert_eq!(sum, 36.0);
        }
    }

    #[test]
    fn test_desktop64_alias() {
        if let Some(token) = Desktop64::summon() {
            let a = [2.0f32; 8];
            let b = [3.0f32; 8];
            let c = [1.0f32; 8];
            let result = fma_f32x8(token, &a, &b, &c);
            assert_eq!(result, [7.0f32; 8]); // 2*3 + 1 = 7
        }
    }
}

// =============================================================================
// PATTERN 2: Feature-Level Trait Bounds
// =============================================================================
//
// Use trait bounds like `impl HasX64V2`, `impl HasX64V4`, `impl HasNeon`.
// These map to specific feature sets.
//
// Pros:
// - Accepts any token that has those features
// - X64V4Token works where HasX64V2 is required
//
// Cons:
// - Only enables the trait's features, not the token's full set
// - HasX64V2 only enables SSE4.2, not AVX2/FMA
// - There is NO HasX64V3 trait - use X64V3Token directly for AVX2+FMA

#[cfg(target_arch = "x86_64")]
mod pattern_feature_traits {
    use archmage::{arcane, HasX64V2, SimdToken, X64V2Token, X64V3Token};
    #[cfg(feature = "avx512")]
    use archmage::{HasX64V4, X64V4Token};
    use core::arch::x86_64::*;

    /// Function accepting any token with HasX64V2 (SSE4.2 + POPCNT)
    /// WARNING: This only enables SSE4.2 features, NOT AVX2/FMA!
    #[arcane]
    pub fn popcnt_array(token: impl HasX64V2, data: &[u64; 4]) -> u32 {
        // SSE4.2 popcnt is available
        let mut count = 0u32;
        for &val in data {
            count += _popcnt64(val as i64) as u32;
        }
        count
    }

    /// Generic with inline bounds - same as impl Trait
    #[arcane]
    pub fn sum_sse<T: HasX64V2>(token: T, data: &[f32; 4]) -> f32 {
        // Only SSE features available here, not AVX
        let v = unsafe { _mm_loadu_ps(data.as_ptr()) };
        let sum1 = _mm_hadd_ps(v, v);
        let sum2 = _mm_hadd_ps(sum1, sum1);
        unsafe { _mm_cvtss_f32(sum2) }
    }

    /// Generic with where clause
    #[arcane]
    pub fn dot_sse<T>(token: T, a: &[f32; 4], b: &[f32; 4]) -> f32
    where
        T: HasX64V2,
    {
        let va = unsafe { _mm_loadu_ps(a.as_ptr()) };
        let vb = unsafe { _mm_loadu_ps(b.as_ptr()) };
        // SSE4.1 dot product
        let dp = _mm_dp_ps::<0xFF>(va, vb);
        unsafe { _mm_cvtss_f32(dp) }
    }

    #[test]
    fn test_hasx64v2_with_v2_token() {
        if let Some(token) = X64V2Token::summon() {
            let data = [0xFFFF_FFFF_FFFF_FFFFu64; 4];
            let count = popcnt_array(token, &data);
            assert_eq!(count, 256); // 64 bits * 4
        }
    }

    #[test]
    fn test_hasx64v2_with_v3_token() {
        // X64V3Token also implements HasX64V2
        if let Some(token) = X64V3Token::summon() {
            let data = [1.0f32, 2.0, 3.0, 4.0];
            let sum = sum_sse(token, &data);
            assert_eq!(sum, 10.0);
        }
    }

    #[cfg(feature = "avx512")]
    #[test]
    fn test_hasx64v4_with_v4_token() {
        if let Some(token) = X64V4Token::summon() {
            // X64V4Token implements HasX64V4 (and HasX64V2)
            let a = [1.0f32, 2.0, 3.0, 4.0];
            let b = [2.0f32, 2.0, 2.0, 2.0];
            let dot = dot_sse(token, &a, &b);
            assert_eq!(dot, 20.0); // 1*2 + 2*2 + 3*2 + 4*2 = 20
        }
    }
}

// =============================================================================
// PATTERN 3: Width Traits (DEPRECATED - Being Removed)
// =============================================================================
//
// Has128BitSimd, Has256BitSimd, Has512BitSimd are being REMOVED because:
// - They don't map to useful feature sets
// - Has256BitSimd only enables AVX, not AVX2/FMA
// - This causes #[arcane] to generate suboptimal code
//
// DO NOT USE THESE TRAITS IN NEW CODE.
//
// Migration:
// - impl Has128BitSimd → impl HasX64V2 (x86) or impl HasNeon (ARM)
// - impl Has256BitSimd → X64V3Token directly
// - impl Has512BitSimd → impl HasX64V4 or X64V4Token

#[cfg(target_arch = "x86_64")]
mod pattern_width_traits_deprecated {
    use archmage::{arcane, Has256BitSimd, SimdToken, X64V3Token};
    use core::arch::x86_64::*;

    /// ⚠️ DEPRECATED: Using Has256BitSimd
    /// This only enables AVX (not AVX2/FMA), so FMA intrinsics may not optimize!
    #[arcane]
    pub fn add_f32x8_deprecated(token: impl Has256BitSimd, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr()) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr()) };
        // This works because _mm256_add_ps only needs AVX
        let result = _mm256_add_ps(va, vb);
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), result) };
        out
    }

    /// ✅ CORRECT: Use X64V3Token for AVX2+FMA
    #[arcane]
    pub fn fma_f32x8_correct(token: X64V3Token, a: &[f32; 8], b: &[f32; 8], c: &[f32; 8]) -> [f32; 8] {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr()) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr()) };
        let vc = unsafe { _mm256_loadu_ps(c.as_ptr()) };
        // FMA is guaranteed available with X64V3Token
        let result = _mm256_fmadd_ps(va, vb, vc);
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), result) };
        out
    }

    #[test]
    fn test_deprecated_pattern_still_works_for_now() {
        if let Some(token) = X64V3Token::summon() {
            let a = [1.0f32; 8];
            let b = [2.0f32; 8];
            let result = add_f32x8_deprecated(token, &a, &b);
            assert_eq!(result, [3.0f32; 8]);
        }
    }
}

// =============================================================================
// PATTERN 4: Token as Second Parameter (_self pattern)
// =============================================================================
//
// When implementing traits with #[arcane], the token must be the SECOND parameter
// (after self). Use `_self` in the function body instead of `self`.
//
// This is required because:
// - Trait methods have `self` as the receiver
// - #[arcane] transforms self to a regular parameter named `_self`

#[cfg(target_arch = "x86_64")]
mod pattern_self_receiver {
    use archmage::{arcane, HasX64V2, SimdToken, X64V3Token};
    use core::arch::x86_64::*;

    /// A wrapper type for SIMD operations
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Vec8f32(pub [f32; 8]);

    /// Trait with various self receiver types
    pub trait SimdOps {
        /// Shared reference receiver
        fn double(&self, token: X64V3Token) -> Self;
        /// Owned receiver
        fn square(self, token: X64V3Token) -> Self;
        /// Mutable reference receiver
        fn scale(&mut self, token: X64V3Token, factor: f32);
    }

    impl SimdOps for Vec8f32 {
        /// &self → _self is &Vec8f32
        #[arcane(_self = Vec8f32)]
        fn double(&self, _token: X64V3Token) -> Self {
            let v = unsafe { _mm256_loadu_ps(_self.0.as_ptr()) };
            let doubled = _mm256_add_ps(v, v);
            let mut out = [0.0f32; 8];
            unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
            Vec8f32(out)
        }

        /// self (owned) → _self is Vec8f32
        #[arcane(_self = Vec8f32)]
        fn square(self, _token: X64V3Token) -> Self {
            let v = unsafe { _mm256_loadu_ps(_self.0.as_ptr()) };
            let squared = _mm256_mul_ps(v, v);
            let mut out = [0.0f32; 8];
            unsafe { _mm256_storeu_ps(out.as_mut_ptr(), squared) };
            Vec8f32(out)
        }

        /// &mut self → _self is &mut Vec8f32
        #[arcane(_self = Vec8f32)]
        fn scale(&mut self, _token: X64V3Token, factor: f32) {
            let v = unsafe { _mm256_loadu_ps(_self.0.as_ptr()) };
            let scale = _mm256_set1_ps(factor);
            let scaled = _mm256_mul_ps(v, scale);
            unsafe { _mm256_storeu_ps(_self.0.as_mut_ptr(), scaled) };
        }
    }

    #[test]
    fn test_self_ref() {
        if let Some(token) = X64V3Token::summon() {
            let v = Vec8f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            let result = v.double(token);
            assert_eq!(result.0, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        }
    }

    #[test]
    fn test_self_owned() {
        if let Some(token) = X64V3Token::summon() {
            let v = Vec8f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            let result = v.square(token);
            assert_eq!(result.0, [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]);
        }
    }

    #[test]
    fn test_self_mut_ref() {
        if let Some(token) = X64V3Token::summon() {
            let mut v = Vec8f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            v.scale(token, 3.0);
            assert_eq!(v.0, [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0]);
        }
    }
}

// =============================================================================
// PATTERN 5: Token Passthrough (Nested #[arcane] Calls)
// =============================================================================
//
// You can call other #[arcane] functions from within an #[arcane] function.
// The inner function receives the same target_feature context.

#[cfg(target_arch = "x86_64")]
mod pattern_token_passthrough {
    use archmage::{arcane, SimdToken, X64V3Token};
    use core::arch::x86_64::*;

    /// Low-level helper
    #[arcane]
    fn add_vectors(token: X64V3Token, a: __m256, b: __m256) -> __m256 {
        _mm256_add_ps(a, b)
    }

    /// Low-level helper
    #[arcane]
    fn mul_vectors(token: X64V3Token, a: __m256, b: __m256) -> __m256 {
        _mm256_mul_ps(a, b)
    }

    /// High-level function that calls helpers
    #[arcane]
    pub fn dot_product(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32 {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr()) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr()) };

        // Call helper - token passthrough
        let product = mul_vectors(token, va, vb);

        // Horizontal sum
        let sum1 = _mm256_hadd_ps(product, product);
        let sum2 = _mm256_hadd_ps(sum1, sum1);
        let low = _mm256_castps256_ps128(sum2);
        let high = _mm256_extractf128_ps::<1>(sum2);
        let final_sum = _mm_add_ss(low, high);
        unsafe { _mm_cvtss_f32(final_sum) }
    }

    /// Another example: composing multiple helpers
    #[arcane]
    pub fn polynomial(token: X64V3Token, x: &[f32; 8], a: f32, b: f32, c: f32) -> [f32; 8] {
        let vx = unsafe { _mm256_loadu_ps(x.as_ptr()) };
        let va = _mm256_set1_ps(a);
        let vb = _mm256_set1_ps(b);
        let vc = _mm256_set1_ps(c);

        // ax^2 + bx + c using helpers
        let x_squared = mul_vectors(token, vx, vx);
        let ax2 = mul_vectors(token, va, x_squared);
        let bx = mul_vectors(token, vb, vx);
        let ax2_plus_bx = add_vectors(token, ax2, bx);
        let result = add_vectors(token, ax2_plus_bx, vc);

        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), result) };
        out
    }

    #[test]
    fn test_token_passthrough() {
        if let Some(token) = X64V3Token::summon() {
            let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let b = [2.0f32; 8];
            let dot = dot_product(token, &a, &b);
            // 1*2 + 2*2 + 3*2 + 4*2 + 5*2 + 6*2 + 7*2 + 8*2 = 72
            assert_eq!(dot, 72.0);
        }
    }

    #[test]
    fn test_composed_helpers() {
        if let Some(token) = X64V3Token::summon() {
            let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            // 2x^2 + 3x + 1
            let result = polynomial(token, &x, 2.0, 3.0, 1.0);
            let expected: [f32; 8] = x.map(|xi| 2.0 * xi * xi + 3.0 * xi + 1.0);
            assert_eq!(result, expected);
        }
    }
}

// =============================================================================
// PATTERN 6: Platform Dispatch (Manual incant! Pattern)
// =============================================================================
//
// Until `incant!` macro is implemented, use this pattern for cross-platform code.
// Define suffixed functions for each platform, then dispatch manually.

#[cfg(target_arch = "x86_64")]
mod pattern_manual_dispatch_x86 {
    use archmage::{arcane, SimdToken, X64V3Token};
    #[cfg(feature = "avx512")]
    use archmage::X64V4Token;
    use core::arch::x86_64::*;

    /// AVX2+FMA implementation
    #[arcane]
    pub fn sum_v3(token: X64V3Token, data: &[f32]) -> f32 {
        let mut acc = _mm256_setzero_ps();
        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let v = unsafe { _mm256_loadu_ps(chunk.as_ptr()) };
            acc = _mm256_add_ps(acc, v);
        }

        // Horizontal sum
        let sum1 = _mm256_hadd_ps(acc, acc);
        let sum2 = _mm256_hadd_ps(sum1, sum1);
        let low = _mm256_castps256_ps128(sum2);
        let high = _mm256_extractf128_ps::<1>(sum2);
        let mut result = unsafe { _mm_cvtss_f32(_mm_add_ss(low, high)) };

        // Handle remainder
        for &val in remainder {
            result += val;
        }
        result
    }

    /// Scalar fallback
    pub fn sum_scalar(data: &[f32]) -> f32 {
        data.iter().sum()
    }

    /// Dispatch function
    pub fn sum(data: &[f32]) -> f32 {
        #[cfg(feature = "avx512")]
        if let Some(token) = X64V4Token::summon() {
            // Could use AVX-512 here if implemented
            return sum_v3(token.v3(), data);
        }

        if let Some(token) = X64V3Token::summon() {
            return sum_v3(token, data);
        }

        sum_scalar(data)
    }

    #[test]
    fn test_dispatch() {
        let data: Vec<f32> = (1..=100).map(|x| x as f32).collect();
        let result = sum(&data);
        assert_eq!(result, 5050.0); // 1+2+...+100 = 5050
    }
}

// =============================================================================
// PATTERN 7: Using magetypes Types Directly
// =============================================================================
//
// The magetypes crate provides high-level SIMD types (f32x8, i32x4, etc.)
// that work with archmage tokens.

#[cfg(target_arch = "x86_64")]
mod pattern_magetypes {
    use archmage::{SimdToken, X64V3Token};
    use magetypes::simd::f32x8;

    #[test]
    fn test_magetypes_basic() {
        if let Some(token) = X64V3Token::summon() {
            let a = f32x8::splat(token, 2.0);
            let b = f32x8::splat(token, 3.0);
            let c = a + b;
            assert_eq!(c.to_array(), [5.0f32; 8]);
        }
    }

    #[test]
    fn test_magetypes_load_store() {
        if let Some(token) = X64V3Token::summon() {
            let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let v = f32x8::load(token, &data);
            let doubled = v + v;
            assert_eq!(doubled.to_array(), [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        }
    }

    #[test]
    fn test_magetypes_reduce() {
        if let Some(token) = X64V3Token::summon() {
            let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let v = f32x8::load(token, &data);
            let sum = v.reduce_add();
            assert_eq!(sum, 36.0);
        }
    }
}

// =============================================================================
// PATTERN 8: Separate Platform Implementations
// =============================================================================
//
// For maximum control, write completely separate implementations per platform.
// This is useful when algorithms differ significantly between platforms.

#[cfg(target_arch = "x86_64")]
mod pattern_separate_platforms_x86 {
    use archmage::{arcane, SimdToken, X64V3Token};
    use core::arch::x86_64::*;

    #[arcane]
    pub fn process_x86(token: X64V3Token, data: &mut [f32]) {
        for chunk in data.chunks_exact_mut(8) {
            let v = unsafe { _mm256_loadu_ps(chunk.as_ptr()) };
            let processed = _mm256_mul_ps(v, v); // Square
            unsafe { _mm256_storeu_ps(chunk.as_mut_ptr(), processed) };
        }
    }

    #[test]
    fn test_x86_impl() {
        if let Some(token) = X64V3Token::summon() {
            let mut data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            process_x86(token, &mut data);
            assert_eq!(data, [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]);
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod pattern_separate_platforms_arm {
    use archmage::{arcane, NeonToken, SimdToken};
    use core::arch::aarch64::*;

    #[arcane]
    pub fn process_arm(token: NeonToken, data: &mut [f32]) {
        for chunk in data.chunks_exact_mut(4) {
            let v = unsafe { vld1q_f32(chunk.as_ptr()) };
            let processed = vmulq_f32(v, v); // Square
            unsafe { vst1q_f32(chunk.as_mut_ptr(), processed) };
        }
    }

    #[test]
    fn test_arm_impl() {
        if let Some(token) = NeonToken::summon() {
            let mut data = [1.0f32, 2.0, 3.0, 4.0];
            process_arm(token, &mut data);
            assert_eq!(data, [1.0, 4.0, 9.0, 16.0]);
        }
    }
}

// =============================================================================
// ANTI-PATTERN: Generic T: SimdToken (DOES NOT WORK)
// =============================================================================
//
// This pattern CANNOT work with #[arcane] because the compiler can't know
// what features to enable at compile time.
//
// If you need this pattern, use one of:
// 1. Platform dispatch with concrete tokens
// 2. Feature-level trait bounds (HasX64V2, HasNeon, etc.)
// 3. #[magetypes] macro (generates concrete versions)

// This code is commented out because it SHOULD NOT COMPILE with #[arcane]:
//
// ```rust
// // BROKEN - do not use
// #[arcane]
// fn generic_broken<T: SimdToken>(token: T, data: &[f32]) -> f32 {
//     // What features should be enabled here? Unknown!
//     // T could be X64V2Token (SSE4.2), X64V3Token (AVX2+FMA),
//     // NeonToken (NEON), etc.
//     todo!()
// }
// ```
//
// Instead, write separate functions for each platform or use traits that
// map to specific feature sets.

// =============================================================================
// PATTERN 9: Token Extraction (Higher to Lower Tier)
// =============================================================================
//
// Higher-tier tokens can be converted to lower-tier tokens using extraction
// methods like .v3(), .v2(), etc.

#[cfg(target_arch = "x86_64")]
mod pattern_token_extraction {
    use archmage::{arcane, SimdToken, X64V2Token, X64V3Token};
    #[cfg(feature = "avx512")]
    use archmage::X64V4Token;
    use core::arch::x86_64::*;

    /// Requires only SSE4.2
    #[arcane]
    fn sse_operation(token: X64V2Token, data: &[f32; 4]) -> f32 {
        let v = unsafe { _mm_loadu_ps(data.as_ptr()) };
        let sum = _mm_hadd_ps(v, v);
        let sum = _mm_hadd_ps(sum, sum);
        unsafe { _mm_cvtss_f32(sum) }
    }

    /// Uses AVX2 but can fall back to SSE
    #[arcane]
    pub fn flexible_sum(token: X64V3Token, data: &[f32; 4]) -> f32 {
        // Extract v2 token from v3 token
        let v2_token = token.v2();

        // Can use either SSE or AVX operations
        // Here we call an SSE-only function
        sse_operation(v2_token, data)
    }

    #[cfg(feature = "avx512")]
    #[arcane]
    pub fn avx512_with_fallback(token: X64V4Token, data: &[f32; 4]) -> f32 {
        // X64V4Token can extract to v3 or v2
        let v3_token = token.v3();
        flexible_sum(v3_token, data)
    }

    #[test]
    fn test_token_extraction() {
        if let Some(token) = X64V3Token::summon() {
            let data = [1.0f32, 2.0, 3.0, 4.0];
            let sum = flexible_sum(token, &data);
            assert_eq!(sum, 10.0);
        }
    }
}

// =============================================================================
// SUMMARY TABLE (for reference)
// =============================================================================
//
// | Pattern | Example | Works? | Recommended? |
// |---------|---------|--------|--------------|
// | Concrete token | `fn f(t: X64V3Token)` | ✅ | ✅ Yes |
// | Desktop64 alias | `fn f(t: Desktop64)` | ✅ | ✅ Yes |
// | HasX64V2 trait | `fn f(t: impl HasX64V2)` | ✅ | ⚠️ SSE only |
// | HasX64V4 trait | `fn f(t: impl HasX64V4)` | ✅ | ✅ For AVX-512 |
// | Has256BitSimd | `fn f(t: impl Has256BitSimd)` | ⚠️ | ❌ Deprecated |
// | Generic SimdToken | `fn f<T: SimdToken>(t: T)` | ❌ | ❌ Can't work |
// | _self pattern | `#[arcane(_self = T)]` | ✅ | ✅ For traits |
// | Token passthrough | Call other #[arcane] fns | ✅ | ✅ Yes |
// | Token extraction | `token.v3()`, `token.v2()` | ✅ | ✅ For fallback |

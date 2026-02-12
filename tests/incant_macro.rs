//! Tests for the incant! macro.
//!
//! Tests cover:
//! - Entry point mode (summons tokens)
//! - Passthrough mode (uses existing token)
//! - Scalar fallback
//! - Cross-architecture compilation

use archmage::{IntoConcreteToken, ScalarToken, SimdToken};

// =============================================================================
// Test helper functions with required suffixes
// =============================================================================

// Scalar implementation - always available
fn sum_scalar(_token: ScalarToken, data: &[f32]) -> f32 {
    data.iter().sum()
}

// x86 implementations
#[cfg(target_arch = "x86_64")]
fn sum_v3(_token: archmage::X64V3Token, data: &[f32]) -> f32 {
    // In real code, use SIMD. For tests, just verify dispatch works.
    data.iter().sum::<f32>() * 1.0 // Multiply by 1.0 to distinguish from scalar
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn sum_v4(_token: archmage::X64V4Token, data: &[f32]) -> f32 {
    data.iter().sum::<f32>() * 1.0
}

// ARM implementation
#[cfg(target_arch = "aarch64")]
fn sum_neon(_token: archmage::NeonToken, data: &[f32]) -> f32 {
    data.iter().sum::<f32>() * 1.0
}

// WASM implementation
#[cfg(target_arch = "wasm32")]
fn sum_wasm128(_token: archmage::Wasm128Token, data: &[f32]) -> f32 {
    data.iter().sum::<f32>() * 1.0
}

// =============================================================================
// Entry point mode tests
// =============================================================================

mod entry_point_tests {
    use super::*;
    use archmage::incant;

    /// Public API using incant! for dispatch
    pub fn sum_api(data: &[f32]) -> f32 {
        incant!(sum(data))
    }

    #[test]
    fn entry_point_dispatches() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let result = sum_api(&data);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn entry_point_with_empty_data() {
        let data: [f32; 0] = [];
        let result = sum_api(&data);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn entry_point_with_large_data() {
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let result = sum_api(&data);
        let expected: f32 = (0..1000).map(|i| i as f32).sum();
        assert_eq!(result, expected);
    }
}

// =============================================================================
// Passthrough mode tests
// =============================================================================

mod passthrough_tests {
    use super::*;
    use archmage::incant;

    /// Inner function called via passthrough
    fn inner_sum<T: IntoConcreteToken>(token: T, data: &[f32]) -> f32 {
        incant!(sum(data) with token)
    }

    #[test]
    fn passthrough_with_scalar_token() {
        let token = ScalarToken;
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let result = inner_sum(token, &data);
        assert_eq!(result, 10.0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn passthrough_with_x64v3_token() {
        if let Some(token) = archmage::X64V3Token::summon() {
            let data = [1.0f32, 2.0, 3.0, 4.0];
            let result = inner_sum(token, &data);
            assert_eq!(result, 10.0);
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn passthrough_with_x64v4_token() {
        if let Some(token) = archmage::X64V4Token::summon() {
            let data = [1.0f32, 2.0, 3.0, 4.0];
            let result = inner_sum(token, &data);
            assert_eq!(result, 10.0);
        }
    }
}

// =============================================================================
// Scalar fallback tests
// =============================================================================

mod scalar_fallback_tests {
    use super::*;
    use archmage::incant;

    // All variants must exist for incant! to work.
    // The scalar variant is the fallback when no SIMD is available.
    fn double_scalar(_token: ScalarToken, x: i32) -> i32 {
        x * 2
    }

    #[cfg(target_arch = "x86_64")]
    fn double_v3(_token: archmage::X64V3Token, x: i32) -> i32 {
        x * 2
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    fn double_v4(_token: archmage::X64V4Token, x: i32) -> i32 {
        x * 2
    }

    #[cfg(target_arch = "aarch64")]
    fn double_neon(_token: archmage::NeonToken, x: i32) -> i32 {
        x * 2
    }

    #[cfg(target_arch = "wasm32")]
    fn double_wasm128(_token: archmage::Wasm128Token, x: i32) -> i32 {
        x * 2
    }

    pub fn double_api(x: i32) -> i32 {
        incant!(double(x))
    }

    #[test]
    fn works_with_all_variants() {
        let result = double_api(21);
        assert_eq!(result, 42);
    }

    #[test]
    fn scalar_token_always_available() {
        // ScalarToken itself is always available
        assert!(ScalarToken::summon().is_some());
        assert_eq!(ScalarToken::compiled_with(), Some(true));
    }
}

// =============================================================================
// Multiple arguments tests
// =============================================================================

mod multi_arg_tests {
    use super::*;
    use archmage::incant;

    fn dot_scalar(_token: ScalarToken, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[cfg(target_arch = "x86_64")]
    fn dot_v3(_token: archmage::X64V3Token, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    fn dot_v4(_token: archmage::X64V4Token, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[cfg(target_arch = "aarch64")]
    fn dot_neon(_token: archmage::NeonToken, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[cfg(target_arch = "wasm32")]
    fn dot_wasm128(_token: archmage::Wasm128Token, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    pub fn dot_api(a: &[f32], b: &[f32]) -> f32 {
        incant!(dot(a, b))
    }

    #[test]
    fn multiple_arguments() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [4.0f32, 3.0, 2.0, 1.0];
        let result = dot_api(&a, &b);
        // 1*4 + 2*3 + 3*2 + 4*1 = 4 + 6 + 6 + 4 = 20
        assert_eq!(result, 20.0);
    }
}

// =============================================================================
// Return type tests
// =============================================================================

mod return_type_tests {
    use super::*;
    use archmage::incant;

    fn make_array_scalar(_token: ScalarToken, val: f32) -> [f32; 4] {
        [val, val, val, val]
    }

    #[cfg(target_arch = "x86_64")]
    fn make_array_v3(_token: archmage::X64V3Token, val: f32) -> [f32; 4] {
        [val, val, val, val]
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    fn make_array_v4(_token: archmage::X64V4Token, val: f32) -> [f32; 4] {
        [val, val, val, val]
    }

    #[cfg(target_arch = "aarch64")]
    fn make_array_neon(_token: archmage::NeonToken, val: f32) -> [f32; 4] {
        [val, val, val, val]
    }

    #[cfg(target_arch = "wasm32")]
    fn make_array_wasm128(_token: archmage::Wasm128Token, val: f32) -> [f32; 4] {
        [val, val, val, val]
    }

    pub fn make_array_api(val: f32) -> [f32; 4] {
        incant!(make_array(val))
    }

    #[test]
    fn returns_array() {
        let result = make_array_api(3.14);
        assert_eq!(result, [3.14, 3.14, 3.14, 3.14]);
    }
}

// =============================================================================
// simd_route! alias tests
// =============================================================================

mod alias_tests {
    use super::*;
    use archmage::simd_route;

    fn add_scalar(_token: ScalarToken, a: i32, b: i32) -> i32 {
        a + b
    }

    #[cfg(target_arch = "x86_64")]
    fn add_v3(_token: archmage::X64V3Token, a: i32, b: i32) -> i32 {
        a + b
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    fn add_v4(_token: archmage::X64V4Token, a: i32, b: i32) -> i32 {
        a + b
    }

    #[cfg(target_arch = "aarch64")]
    fn add_neon(_token: archmage::NeonToken, a: i32, b: i32) -> i32 {
        a + b
    }

    #[cfg(target_arch = "wasm32")]
    fn add_wasm128(_token: archmage::Wasm128Token, a: i32, b: i32) -> i32 {
        a + b
    }

    pub fn add_api(a: i32, b: i32) -> i32 {
        simd_route!(add(a, b))
    }

    #[test]
    fn simd_route_alias_works() {
        let result = add_api(20, 22);
        assert_eq!(result, 42);
    }
}

// =============================================================================
// IntoConcreteToken direct usage tests
// =============================================================================

mod into_concrete_token_tests {
    use super::*;

    /// Manual dispatch using IntoConcreteToken (what incant! generates)
    fn manual_dispatch<T: IntoConcreteToken>(token: T, data: &[f32]) -> f32 {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            #[cfg(feature = "avx512")]
            if let Some(t) = token.as_x64v4() {
                return super::sum_v4(t, data);
            }
            if let Some(t) = token.as_x64v3() {
                return super::sum_v3(t, data);
            }
        }

        #[cfg(target_arch = "aarch64")]
        if let Some(t) = token.as_neon() {
            return super::sum_neon(t, data);
        }

        #[cfg(target_arch = "wasm32")]
        if let Some(t) = token.as_wasm128() {
            return super::sum_wasm128(t, data);
        }

        if let Some(t) = token.as_scalar() {
            return super::sum_scalar(t, data);
        }

        unreachable!()
    }

    #[test]
    fn manual_dispatch_scalar() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let result = manual_dispatch(ScalarToken, &data);
        assert_eq!(result, 10.0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn manual_dispatch_x64v3() {
        if let Some(token) = archmage::X64V3Token::summon() {
            let data = [1.0f32, 2.0, 3.0, 4.0];
            let result = manual_dispatch(token, &data);
            assert_eq!(result, 10.0);
        }
    }
}

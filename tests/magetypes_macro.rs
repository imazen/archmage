//! Tests for the #[magetypes] function-level macro.
//!
//! Tests verify:
//! - Function variants are generated with correct suffixes
//! - Token type substitution works
//! - Functions compile on the current platform
//! - Integration with incant! for dispatch
//! - Multiple arguments and return types

use archmage::{ScalarToken, SimdToken};

// =============================================================================
// Basic function generation
// =============================================================================

mod basic_generation {
    use super::*;
    use archmage::magetypes;

    /// A simple function that should generate _v3, _v4, _neon, _wasm128, _scalar variants
    #[magetypes]
    pub fn add_one(token: Token, x: f32) -> f32 {
        let _ = token;
        x + 1.0
    }

    #[test]
    fn scalar_variant_exists() {
        let result = add_one_scalar(ScalarToken, 41.0);
        assert_eq!(result, 42.0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v3_variant_exists() {
        if let Some(token) = archmage::X64V3Token::summon() {
            let result = add_one_v3(token, 41.0);
            assert_eq!(result, 42.0);
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn v4_variant_exists() {
        if let Some(token) = archmage::X64V4Token::summon() {
            let result = add_one_v4(token, 41.0);
            assert_eq!(result, 42.0);
        }
    }
}

// =============================================================================
// Multiple arguments
// =============================================================================

mod multi_args {
    use super::*;
    use archmage::magetypes;

    #[magetypes]
    pub fn multiply(token: Token, a: f32, b: f32) -> f32 {
        let _ = token;
        a * b
    }

    #[test]
    fn scalar_multiply() {
        let result = multiply_scalar(ScalarToken, 6.0, 7.0);
        assert_eq!(result, 42.0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v3_multiply() {
        if let Some(token) = archmage::X64V3Token::summon() {
            let result = multiply_v3(token, 6.0, 7.0);
            assert_eq!(result, 42.0);
        }
    }
}

// =============================================================================
// Slice arguments
// =============================================================================

mod slice_args {
    use super::*;
    use archmage::magetypes;

    #[magetypes]
    pub fn sum_slice(token: Token, data: &[f32]) -> f32 {
        let _ = token;
        data.iter().sum()
    }

    #[test]
    fn scalar_sum() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let result = sum_slice_scalar(ScalarToken, &data);
        assert_eq!(result, 10.0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v3_sum() {
        if let Some(token) = archmage::X64V3Token::summon() {
            let data = [1.0f32, 2.0, 3.0, 4.0];
            let result = sum_slice_v3(token, &data);
            assert_eq!(result, 10.0);
        }
    }
}

// =============================================================================
// Integration with incant!
// =============================================================================

mod incant_integration {
    use archmage::{ScalarToken, SimdToken, incant, magetypes};

    #[magetypes]
    pub fn double(token: Token, x: f32) -> f32 {
        let _ = token;
        x * 2.0
    }

    /// Public API that dispatches via incant!
    pub fn double_api(x: f32) -> f32 {
        incant!(double(x))
    }

    #[test]
    fn incant_dispatches_to_variant() {
        let result = double_api(21.0);
        assert_eq!(result, 42.0);
    }

    #[test]
    fn incant_with_zero() {
        let result = double_api(0.0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn incant_with_negative() {
        let result = double_api(-5.0);
        assert_eq!(result, -10.0);
    }
}

// =============================================================================
// LANES constant substitution
// =============================================================================

mod lanes_substitution {
    use super::*;
    use archmage::magetypes;

    #[magetypes]
    pub fn get_lanes(token: Token) -> usize {
        let _ = token;
        LANES
    }

    #[test]
    fn scalar_has_1_lane() {
        assert_eq!(get_lanes_scalar(ScalarToken), 1);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v3_has_8_lanes() {
        if let Some(token) = archmage::X64V3Token::summon() {
            assert_eq!(get_lanes_v3(token), 8);
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn v4_has_16_lanes() {
        if let Some(token) = archmage::X64V4Token::summon() {
            assert_eq!(get_lanes_v4(token), 16);
        }
    }
}

// =============================================================================
// Mutable reference arguments
// =============================================================================

mod mut_ref_args {
    use super::*;
    use archmage::magetypes;

    #[magetypes]
    pub fn fill_array(token: Token, data: &mut [f32], val: f32) {
        let _ = token;
        for x in data.iter_mut() {
            *x = val;
        }
    }

    #[test]
    fn scalar_fill() {
        let mut data = [0.0f32; 4];
        fill_array_scalar(ScalarToken, &mut data, 3.14);
        assert_eq!(data, [3.14, 3.14, 3.14, 3.14]);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v3_fill() {
        if let Some(token) = archmage::X64V3Token::summon() {
            let mut data = [0.0f32; 4];
            fill_array_v3(token, &mut data, 2.71);
            assert_eq!(data, [2.71, 2.71, 2.71, 2.71]);
        }
    }
}

// =============================================================================
// Return types other than f32
// =============================================================================

mod return_types {
    use super::*;
    use archmage::magetypes;

    #[magetypes]
    pub fn count(token: Token, data: &[f32]) -> usize {
        let _ = token;
        data.len()
    }

    #[test]
    fn scalar_count() {
        let data = [1.0f32, 2.0, 3.0];
        assert_eq!(count_scalar(ScalarToken, &data), 3);
    }

    #[magetypes]
    pub fn is_empty(token: Token, data: &[f32]) -> bool {
        let _ = token;
        data.is_empty()
    }

    #[test]
    fn scalar_is_empty() {
        assert!(is_empty_scalar(ScalarToken, &[]));
        assert!(!is_empty_scalar(ScalarToken, &[1.0]));
    }
}

// =============================================================================
// Visibility preservation
// =============================================================================

mod visibility {
    use super::*;
    use archmage::magetypes;

    #[magetypes]
    pub fn public_fn(token: Token, x: f32) -> f32 {
        let _ = token;
        x
    }

    #[magetypes]
    fn private_fn(token: Token, x: f32) -> f32 {
        let _ = token;
        x
    }

    #[test]
    fn public_variant_accessible() {
        let _ = public_fn_scalar(ScalarToken, 1.0);
    }

    #[test]
    fn private_variant_accessible_from_same_module() {
        let _ = private_fn_scalar(ScalarToken, 1.0);
    }
}

// =============================================================================
// Passthrough with incant! and IntoConcreteToken
// =============================================================================

mod passthrough_integration {
    use super::*;
    use archmage::{IntoConcreteToken, incant, magetypes};

    #[magetypes]
    pub fn inner(token: Token, x: f32) -> f32 {
        let _ = token;
        x * 3.0
    }

    fn dispatch_inner<T: IntoConcreteToken>(token: T, x: f32) -> f32 {
        incant!(inner(x) with token)
    }

    #[test]
    fn passthrough_scalar() {
        let result = dispatch_inner(ScalarToken, 14.0);
        assert_eq!(result, 42.0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn passthrough_v3() {
        if let Some(token) = archmage::X64V3Token::summon() {
            let result = dispatch_inner(token, 14.0);
            assert_eq!(result, 42.0);
        }
    }
}

//! Integration tests: using magetypes types with archmage tokens.
//!
//! Covers basic splat/load/store/reduce operations via the archmage token API.
//! For comprehensive archmage macro pattern tests (concrete tokens, trait bounds,
//! _self pattern, passthrough, dispatch), see archmage's own idiomatic_patterns test.

#![allow(dead_code, unused_variables, unused_imports)]

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
            assert_eq!(
                doubled.to_array(),
                [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
            );
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

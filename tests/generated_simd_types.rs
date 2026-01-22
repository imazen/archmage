//! Auto-generated tests for SIMD types.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#![cfg(target_arch = "x86_64")]

use archmage::simd::*;
use archmage::{SimdToken, Avx2FmaToken};


#[test]
fn test_f32x8_basic() {
    if let Some(token) = Avx2FmaToken::try_new() {
        let a = f32x8::splat(token, 1.0);
        let b = f32x8::splat(token, 2.0);
        let c = a + b;
        let arr = c.to_array();
        for &v in &arr {
            assert_eq!(v, 1.0 + 2.0);
        }
    }
}

#[test]
fn test_f32x8_load_store() {
    if let Some(token) = Avx2FmaToken::try_new() {
        let data: [f32; f32x8::LANES] = [1.0; f32x8::LANES];
        let v = f32x8::load(token, &data);
        let mut out = [f32::default(); f32x8::LANES];
        v.store(&mut out);
        assert_eq!(data, out);
    }
}

#[test]
fn test_i32x8_basic() {
    if let Some(token) = Avx2FmaToken::try_new() {
        let a = i32x8::splat(token, 1);
        let b = i32x8::splat(token, 2);
        let c = a + b;
        let arr = c.to_array();
        for &v in &arr {
            assert_eq!(v, 1 + 2);
        }
    }
}

#[test]
fn test_i32x8_load_store() {
    if let Some(token) = Avx2FmaToken::try_new() {
        let data: [i32; i32x8::LANES] = [1; i32x8::LANES];
        let v = i32x8::load(token, &data);
        let mut out = [i32::default(); i32x8::LANES];
        v.store(&mut out);
        assert_eq!(data, out);
    }
}

//! A backend trait's receiver supplies the feature proof without another token.
#![cfg(target_arch = "x86_64")]

use archmage::{SimdToken, X64V3Token, arcane};
use core::arch::x86_64::*;

trait TokenBackend {
    fn splat(self, value: f32) -> [f32; 8];
    fn rotate<const N: i32>(&self, value: i32) -> [i32; 8];
}

impl TokenBackend for X64V3Token {
    #[arcane(_self = X64V3Token)]
    fn splat(self, value: f32) -> [f32; 8] {
        let vector = _mm256_set1_ps(value);
        let mut output = [0.0; 8];
        unsafe { _mm256_storeu_ps(output.as_mut_ptr(), vector) };
        output
    }

    #[arcane(_self = X64V3Token)]
    fn rotate<const N: i32>(&self, value: i32) -> [i32; 8] {
        let _proof: &X64V3Token = _self;
        let vector = _mm256_slli_epi32::<N>(_mm256_set1_epi32(value));
        unsafe { core::mem::transmute(vector) }
    }
}

#[test]
fn token_receiver_proves_features() {
    if let Some(token) = X64V3Token::summon() {
        assert_eq!(token.splat(3.5), [3.5; 8]);
        assert_eq!(token.rotate::<3>(7), [56; 8]);
    }
}

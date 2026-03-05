//! Test that safe memory ops inline into #[arcane] functions

#![cfg(target_arch = "x86_64")]

use archmage::{Desktop64, SimdToken, arcane};

#[arcane(import_intrinsics)]
pub fn process(_token: Desktop64, data: &mut [[f32; 8]]) {
    for chunk in data.iter_mut() {
        let v = _mm256_loadu_ps(chunk as &[f32; 8]);
        let r = _mm256_mul_ps(v, v);
        _mm256_storeu_ps(chunk, r);
    }
}

#[test]
fn test_it() {
    if let Some(token) = Desktop64::summon() {
        let mut data = [[1.0f32; 8]; 4];
        process(token, &mut data);
        assert_eq!(data[0], [1.0f32; 8]);
    }
}

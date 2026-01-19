//! Test that masked loads require unsafe blocks even in target_feature context
//!
//! This test MUST fail to compile - it verifies our safety guarantees.

#![allow(unused)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn try_maskload_without_unsafe() {
    let data = [1.0f32; 8];
    let mask = _mm256_set1_epi32(-1);  // This is safe

    // This MUST fail - maskload takes a raw pointer and is always unsafe
    let loaded = _mm256_maskload_ps(data.as_ptr(), mask);
}

fn main() {}

//! Test that pointer-based stores require unsafe blocks even in target_feature context
//!
//! This test MUST fail to compile - it verifies our safety guarantees.

#![allow(unused)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn try_store_without_unsafe() {
    let mut data = [0.0f32; 8];
    let zeros = _mm256_setzero_ps();  // This is safe

    // This MUST fail - _mm256_storeu_ps takes a raw pointer and is always unsafe
    _mm256_storeu_ps(data.as_mut_ptr(), zeros);
}

fn main() {}

//! Test that pointer-based loads require unsafe blocks even in target_feature context
//!
//! This test MUST fail to compile - it verifies our safety guarantees.

#![allow(unused)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn try_load_without_unsafe() {
    let data = [1.0f32; 8];

    // This MUST fail - _mm256_loadu_ps takes a raw pointer and is always unsafe
    let loaded = _mm256_loadu_ps(data.as_ptr());
}

fn main() {}

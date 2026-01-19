//! Test that gather operations require unsafe blocks even in target_feature context
//!
//! This test MUST fail to compile - it verifies our safety guarantees.

#![allow(unused)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn try_gather_without_unsafe() {
    let data = [1.0f32; 16];
    let indices = _mm256_set1_epi32(0);  // This is safe

    // This MUST fail - gather takes a raw pointer and is always unsafe
    let gathered = _mm256_i32gather_ps::<4>(data.as_ptr(), indices);
}

fn main() {}

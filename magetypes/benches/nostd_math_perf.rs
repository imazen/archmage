//! Benchmark: nostd_math vs std inherent math methods.
//!
//! Run with: RUSTFLAGS="-C target-cpu=native" cargo +nightly bench -p magetypes --bench nostd_math_perf

#![feature(test)]
extern crate test;

use test::{Bencher, black_box};

// ============================================================================
// f32 sqrt
// ============================================================================

#[bench]
fn f32_sqrt_std(b: &mut Bencher) {
    let values: Vec<f32> = (1..=1024).map(|i| i as f32 * 0.1).collect();
    b.iter(|| {
        let mut sum = 0.0f32;
        for &x in &values {
            sum += black_box(x).sqrt();
        }
        black_box(sum)
    });
}

#[bench]
fn f32_sqrt_nostd(b: &mut Bencher) {
    let values: Vec<f32> = (1..=1024).map(|i| i as f32 * 0.1).collect();
    b.iter(|| {
        let mut sum = 0.0f32;
        for &x in &values {
            sum += magetypes::nostd_math::sqrtf(black_box(x));
        }
        black_box(sum)
    });
}

// ============================================================================
// f64 sqrt
// ============================================================================

#[bench]
fn f64_sqrt_std(b: &mut Bencher) {
    let values: Vec<f64> = (1..=1024).map(|i| i as f64 * 0.1).collect();
    b.iter(|| {
        let mut sum = 0.0f64;
        for &x in &values {
            sum += black_box(x).sqrt();
        }
        black_box(sum)
    });
}

#[bench]
fn f64_sqrt_nostd(b: &mut Bencher) {
    let values: Vec<f64> = (1..=1024).map(|i| i as f64 * 0.1).collect();
    b.iter(|| {
        let mut sum = 0.0f64;
        for &x in &values {
            sum += magetypes::nostd_math::sqrt(black_box(x));
        }
        black_box(sum)
    });
}

// ============================================================================
// f32 floor
// ============================================================================

#[bench]
fn f32_floor_std(b: &mut Bencher) {
    let values: Vec<f32> = (1..=1024).map(|i| i as f32 * 0.1 - 50.0).collect();
    b.iter(|| {
        let mut sum = 0.0f32;
        for &x in &values {
            sum += black_box(x).floor();
        }
        black_box(sum)
    });
}

#[bench]
fn f32_floor_nostd(b: &mut Bencher) {
    let values: Vec<f32> = (1..=1024).map(|i| i as f32 * 0.1 - 50.0).collect();
    b.iter(|| {
        let mut sum = 0.0f32;
        for &x in &values {
            sum += magetypes::nostd_math::floorf(black_box(x));
        }
        black_box(sum)
    });
}

// ============================================================================
// f32 round
// ============================================================================

#[bench]
fn f32_round_std(b: &mut Bencher) {
    let values: Vec<f32> = (1..=1024).map(|i| i as f32 * 0.1 - 50.0).collect();
    b.iter(|| {
        let mut sum = 0.0f32;
        for &x in &values {
            sum += black_box(x).round();
        }
        black_box(sum)
    });
}

#[bench]
fn f32_round_nostd(b: &mut Bencher) {
    let values: Vec<f32> = (1..=1024).map(|i| i as f32 * 0.1 - 50.0).collect();
    b.iter(|| {
        let mut sum = 0.0f32;
        for &x in &values {
            sum += magetypes::nostd_math::roundf(black_box(x));
        }
        black_box(sum)
    });
}

// ============================================================================
// f64 floor
// ============================================================================

#[bench]
fn f64_floor_std(b: &mut Bencher) {
    let values: Vec<f64> = (1..=1024).map(|i| i as f64 * 0.1 - 50.0).collect();
    b.iter(|| {
        let mut sum = 0.0f64;
        for &x in &values {
            sum += black_box(x).floor();
        }
        black_box(sum)
    });
}

#[bench]
fn f64_floor_nostd(b: &mut Bencher) {
    let values: Vec<f64> = (1..=1024).map(|i| i as f64 * 0.1 - 50.0).collect();
    b.iter(|| {
        let mut sum = 0.0f64;
        for &x in &values {
            sum += magetypes::nostd_math::floor(black_box(x));
        }
        black_box(sum)
    });
}

// ============================================================================
// f64 round
// ============================================================================

#[bench]
fn f64_round_std(b: &mut Bencher) {
    let values: Vec<f64> = (1..=1024).map(|i| i as f64 * 0.1 - 50.0).collect();
    b.iter(|| {
        let mut sum = 0.0f64;
        for &x in &values {
            sum += black_box(x).round();
        }
        black_box(sum)
    });
}

#[bench]
fn f64_round_nostd(b: &mut Bencher) {
    let values: Vec<f64> = (1..=1024).map(|i| i as f64 * 0.1 - 50.0).collect();
    b.iter(|| {
        let mut sum = 0.0f64;
        for &x in &values {
            sum += magetypes::nostd_math::round(black_box(x));
        }
        black_box(sum)
    });
}

//! Benchmark: `nostd_math` vs the std inherent math methods.
//!
//! Each family pairs the std method against the `nostd_math` fallback over the
//! same L1-resident input, so the delta is the op cost alone.
//!
//! Run with:
//!   RUSTFLAGS="-C target-cpu=native" cargo bench -p magetypes --bench nostd_math_perf

use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

/// Ascending positives — valid for `sqrt`.
fn positives_f32() -> Vec<f32> {
    (1..=1024).map(|i| i as f32 * 0.1).collect()
}

fn positives_f64() -> Vec<f64> {
    (1..=1024).map(|i| i as f64 * 0.1).collect()
}

/// Straddles zero — exercises both rounding directions of `floor`/`round`.
fn signed_f32() -> Vec<f32> {
    (1..=1024).map(|i| i as f32 * 0.1 - 50.0).collect()
}

fn signed_f64() -> Vec<f64> {
    (1..=1024).map(|i| i as f64 * 0.1 - 50.0).collect()
}

/// Sum `op` over every element, keeping the loop opaque to LLVM.
fn sum_f32(values: &[f32], op: impl Fn(f32) -> f32) -> f32 {
    let mut sum = 0.0f32;
    for &x in values {
        sum += op(black_box(x));
    }
    black_box(sum)
}

fn sum_f64(values: &[f64], op: impl Fn(f64) -> f64) -> f64 {
    let mut sum = 0.0f64;
    for &x in values {
        sum += op(black_box(x));
    }
    black_box(sum)
}

fn bench_sqrt(c: &mut Criterion) {
    let v32 = positives_f32();
    c.bench_function("f32_sqrt/std", |b| b.iter(|| sum_f32(&v32, |x| x.sqrt())));
    c.bench_function("f32_sqrt/nostd", |b| {
        b.iter(|| sum_f32(&v32, magetypes::nostd_math::sqrtf))
    });

    let v64 = positives_f64();
    c.bench_function("f64_sqrt/std", |b| b.iter(|| sum_f64(&v64, |x| x.sqrt())));
    c.bench_function("f64_sqrt/nostd", |b| {
        b.iter(|| sum_f64(&v64, magetypes::nostd_math::sqrt))
    });
}

fn bench_floor(c: &mut Criterion) {
    let v32 = signed_f32();
    c.bench_function("f32_floor/std", |b| b.iter(|| sum_f32(&v32, |x| x.floor())));
    c.bench_function("f32_floor/nostd", |b| {
        b.iter(|| sum_f32(&v32, magetypes::nostd_math::floorf))
    });

    let v64 = signed_f64();
    c.bench_function("f64_floor/std", |b| b.iter(|| sum_f64(&v64, |x| x.floor())));
    c.bench_function("f64_floor/nostd", |b| {
        b.iter(|| sum_f64(&v64, magetypes::nostd_math::floor))
    });
}

fn bench_round(c: &mut Criterion) {
    let v32 = signed_f32();
    c.bench_function("f32_round/std", |b| b.iter(|| sum_f32(&v32, |x| x.round())));
    c.bench_function("f32_round/nostd", |b| {
        b.iter(|| sum_f32(&v32, magetypes::nostd_math::roundf))
    });

    let v64 = signed_f64();
    c.bench_function("f64_round/std", |b| b.iter(|| sum_f64(&v64, |x| x.round())));
    c.bench_function("f64_round/nostd", |b| {
        b.iter(|| sum_f64(&v64, magetypes::nostd_math::round))
    });
}

criterion_group!(benches, bench_sqrt, bench_floor, bench_round);
criterion_main!(benches);

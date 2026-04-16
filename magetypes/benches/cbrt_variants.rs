//! Benchmarks comparing cbrt and pow variants.
//!
//! Run:
//!   cargo bench --bench cbrt_variants --features "std"
//!
//! Also benchmarks scalar cbrt/pow for reference.

#![cfg(target_arch = "x86_64")]

use archmage::{SimdToken, X64V3Token, arcane};
use magetypes::simd::generic::f32x8;
use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

// ============================================================================
// SIMD wrappers (must be #[arcane] for target features)
// ============================================================================

#[arcane]
fn simd_cbrt_lowp(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    f32x8::from_array(token, *data).cbrt_lowp().to_array()
}

#[arcane]
fn simd_cbrt_midp(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    f32x8::from_array(token, *data).cbrt_midp().to_array()
}

#[arcane]
fn simd_exp2_lowp(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    f32x8::from_array(token, *data).exp2_lowp().to_array()
}

#[arcane]
fn simd_exp2_midp(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    f32x8::from_array(token, *data).exp2_midp().to_array()
}

#[arcane]
fn simd_pow_lowp(token: X64V3Token, data: &[f32; 8], n: f32) -> [f32; 8] {
    f32x8::from_array(token, *data).pow_lowp(n).to_array()
}

#[arcane]
fn simd_pow_midp(token: X64V3Token, data: &[f32; 8], n: f32) -> [f32; 8] {
    f32x8::from_array(token, *data).pow_midp(n).to_array()
}

// Scalar reference: std cbrt
fn scalar_cbrt(data: &[f32; 8]) -> [f32; 8] {
    core::array::from_fn(|i| data[i].cbrt())
}

// Scalar Kahan + 1 Halley (matches lowp)
fn scalar_cbrt_lowp(data: &[f32; 8]) -> [f32; 8] {
    core::array::from_fn(|i| magetypes::nostd_math::cbrt_lowp_f32(data[i]))
}

// Scalar Kahan + 2 Halley (matches midp)
fn scalar_cbrt_midp(data: &[f32; 8]) -> [f32; 8] {
    core::array::from_fn(|i| magetypes::nostd_math::cbrt_midp_f32(data[i]))
}

// ============================================================================
// Bulk benchmarks (process 1024 values to amortize overhead)
// ============================================================================

fn make_test_data() -> Vec<[f32; 8]> {
    (0..128)
        .map(|i| {
            let base = i as f32 / 128.0;
            core::array::from_fn(|j| 10.0f32.powf(-4.0 + (base + j as f32 / 1024.0) * 8.0))
        })
        .collect()
}

fn bench_cbrt_variants(c: &mut Criterion) {
    let data = make_test_data();
    let single = [0.001f32, 0.1, 0.5, 2.0, 10.0, 100.0, 1000.0, 1e6];

    let mut group = c.benchmark_group("cbrt_single_8");

    // Single f32x8 benchmarks
    if let Some(token) = X64V3Token::summon() {
        group.bench_function("simd_lowp", |b| {
            b.iter(|| simd_cbrt_lowp(token, black_box(&single)))
        });
        group.bench_function("simd_midp", |b| {
            b.iter(|| simd_cbrt_midp(token, black_box(&single)))
        });
    }

    group.bench_function("scalar_std_cbrt", |b| {
        b.iter(|| scalar_cbrt(black_box(&single)))
    });
    group.bench_function("scalar_lowp", |b| {
        b.iter(|| scalar_cbrt_lowp(black_box(&single)))
    });
    group.bench_function("scalar_midp", |b| {
        b.iter(|| scalar_cbrt_midp(black_box(&single)))
    });
    group.finish();

    // Bulk (1024 values) benchmarks
    let mut group = c.benchmark_group("cbrt_bulk_1024");

    if let Some(token) = X64V3Token::summon() {
        group.bench_function("simd_lowp", |b| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for chunk in &data {
                    let r = simd_cbrt_lowp(token, black_box(chunk));
                    sum += r[0];
                }
                sum
            })
        });
        group.bench_function("simd_midp", |b| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for chunk in &data {
                    let r = simd_cbrt_midp(token, black_box(chunk));
                    sum += r[0];
                }
                sum
            })
        });
    }

    group.bench_function("scalar_std_cbrt", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for chunk in &data {
                let r = scalar_cbrt(black_box(chunk));
                sum += r[0];
            }
            sum
        })
    });
    group.bench_function("scalar_lowp", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for chunk in &data {
                let r = scalar_cbrt_lowp(black_box(chunk));
                sum += r[0];
            }
            sum
        })
    });
    group.bench_function("scalar_midp", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for chunk in &data {
                let r = scalar_cbrt_midp(black_box(chunk));
                sum += r[0];
            }
            sum
        })
    });
    group.finish();
}

fn bench_pow_variants(c: &mut Criterion) {
    let data = make_test_data();
    let single = [0.001f32, 0.1, 0.5, 2.0, 10.0, 100.0, 1000.0, 1e6];
    let exponent = 2.5f32;

    let mut group = c.benchmark_group("pow_single_8");

    if let Some(token) = X64V3Token::summon() {
        group.bench_function("simd_lowp", |b| {
            b.iter(|| simd_pow_lowp(token, black_box(&single), black_box(exponent)))
        });
        group.bench_function("simd_midp", |b| {
            b.iter(|| simd_pow_midp(token, black_box(&single), black_box(exponent)))
        });
    }

    group.bench_function("scalar_powf", |b| {
        b.iter(|| {
            let s = black_box(&single);
            let n = black_box(exponent);
            core::array::from_fn::<f32, 8, _>(|i| s[i].powf(n))
        })
    });
    group.finish();

    let mut group = c.benchmark_group("pow_bulk_1024");

    if let Some(token) = X64V3Token::summon() {
        group.bench_function("simd_lowp", |b| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for chunk in &data {
                    let r = simd_pow_lowp(token, black_box(chunk), black_box(exponent));
                    sum += r[0];
                }
                sum
            })
        });
        group.bench_function("simd_midp", |b| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for chunk in &data {
                    let r = simd_pow_midp(token, black_box(chunk), black_box(exponent));
                    sum += r[0];
                }
                sum
            })
        });
    }

    group.bench_function("scalar_powf", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            let n = black_box(exponent);
            for chunk in &data {
                let s = black_box(chunk);
                sum += s[0].powf(n);
            }
            sum
        })
    });
    group.finish();
}

fn bench_exp2_variants(c: &mut Criterion) {
    let single = [-100.0f32, -10.0, -1.0, -0.01, 0.01, 1.0, 10.0, 100.0];

    let mut group = c.benchmark_group("exp2_single_8");

    if let Some(token) = X64V3Token::summon() {
        group.bench_function("simd_lowp", |b| {
            b.iter(|| simd_exp2_lowp(token, black_box(&single)))
        });
        group.bench_function("simd_midp", |b| {
            b.iter(|| simd_exp2_midp(token, black_box(&single)))
        });
    }

    group.bench_function("scalar_exp2", |b| {
        b.iter(|| {
            let s = black_box(&single);
            core::array::from_fn::<f32, 8, _>(|i| s[i].exp2())
        })
    });
    group.finish();

    // Bulk: 1024 values across [-120, 120]
    let bulk_data: Vec<[f32; 8]> = (0..128)
        .map(|i| core::array::from_fn(|j| -120.0 + (i * 8 + j) as f32 * 240.0 / 1024.0))
        .collect();

    let mut group = c.benchmark_group("exp2_bulk_1024");

    if let Some(token) = X64V3Token::summon() {
        group.bench_function("simd_lowp", |b| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for chunk in &bulk_data {
                    let r = simd_exp2_lowp(token, black_box(chunk));
                    sum += r[0];
                }
                sum
            })
        });
        group.bench_function("simd_midp", |b| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for chunk in &bulk_data {
                    let r = simd_exp2_midp(token, black_box(chunk));
                    sum += r[0];
                }
                sum
            })
        });
    }

    group.bench_function("scalar_exp2", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for chunk in &bulk_data {
                let s = black_box(chunk);
                sum += s[0].exp2();
            }
            sum
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_cbrt_variants,
    bench_pow_variants,
    bench_exp2_variants
);
criterion_main!(benches);

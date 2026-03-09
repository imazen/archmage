//! Benchmarks comparing cbrt variants: lowp (1 Halley), fast (2 Halley), midp (3 Newton)
//!
//! Run:
//!   cargo bench --bench cbrt_variants --features "std"
//!
//! Also benchmarks scalar cbrt for reference.

#![cfg(target_arch = "x86_64")]

use archmage::{SimdToken, X64V3Token, arcane};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use magetypes::simd::generic::f32x8;

// ============================================================================
// SIMD wrappers (must be #[arcane] for target features)
// ============================================================================

#[arcane]
fn simd_cbrt_lowp(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    f32x8::from_array(token, *data).cbrt_lowp().to_array()
}

#[arcane]
fn simd_cbrt_fast(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    f32x8::from_array(token, *data).cbrt_fast().to_array()
}

#[arcane]
fn simd_cbrt_midp(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    f32x8::from_array(token, *data).cbrt_midp().to_array()
}

// Scalar reference: std cbrt
fn scalar_cbrt(data: &[f32; 8]) -> [f32; 8] {
    core::array::from_fn(|i| data[i].cbrt())
}

// Scalar Kahan + 1 Halley (matches lowp)
fn scalar_cbrt_lowp(data: &[f32; 8]) -> [f32; 8] {
    core::array::from_fn(|i| {
        let x = data[i];
        let sign = x.signum();
        let ax = x.abs();
        let mut y = f32::from_bits((ax.to_bits() / 3) + 0x2a508c2d);
        let y3 = y * y * y;
        y *= (y3 + 2.0 * ax) / (2.0 * y3 + ax);
        sign * y
    })
}

// Scalar Kahan + 2 Halley (matches fast)
fn scalar_cbrt_fast(data: &[f32; 8]) -> [f32; 8] {
    core::array::from_fn(|i| {
        let x = data[i];
        let sign = x.signum();
        let ax = x.abs();
        let mut y = f32::from_bits((ax.to_bits() / 3) + 0x2a508c2d);
        for _ in 0..2 {
            let y3 = y * y * y;
            y *= (y3 + 2.0 * ax) / (2.0 * y3 + ax);
        }
        sign * y
    })
}

// Scalar Kahan + 3 Newton (matches midp)
fn scalar_cbrt_midp(data: &[f32; 8]) -> [f32; 8] {
    core::array::from_fn(|i| {
        let x = data[i];
        let sign = x.signum();
        let ax = x.abs();
        let mut y = f32::from_bits((ax.to_bits() / 3) + 0x2a508c2d);
        for _ in 0..3 {
            let y3 = y * y * y;
            y *= 0.666_666_6 + ax / (3.0 * y3);
        }
        sign * y
    })
}

// ============================================================================
// Bulk benchmarks (process 1024 values to amortize overhead)
// ============================================================================

fn make_test_data() -> Vec<[f32; 8]> {
    (0..128)
        .map(|i| {
            let base = i as f32 / 128.0;
            core::array::from_fn(|j| {
                10.0f32.powf(-4.0 + (base + j as f32 / 1024.0) * 8.0)
            })
        })
        .collect()
}

fn bench_cbrt_variants(c: &mut Criterion) {
    let data = make_test_data();
    let single = [0.001f32, 0.1, 0.5, 2.0, 10.0, 100.0, 1000.0, 1e6];

    let mut group = c.benchmark_group("cbrt_single_8");

    // Single f32x8 benchmarks
    if let Some(token) = X64V3Token::summon() {
        group.bench_function("simd_lowp_1halley", |b| {
            b.iter(|| simd_cbrt_lowp(token, black_box(&single)))
        });
        group.bench_function("simd_fast_2halley", |b| {
            b.iter(|| simd_cbrt_fast(token, black_box(&single)))
        });
        group.bench_function("simd_midp_3newton", |b| {
            b.iter(|| simd_cbrt_midp(token, black_box(&single)))
        });
    }

    group.bench_function("scalar_std_cbrt", |b| {
        b.iter(|| scalar_cbrt(black_box(&single)))
    });
    group.bench_function("scalar_lowp_1halley", |b| {
        b.iter(|| scalar_cbrt_lowp(black_box(&single)))
    });
    group.bench_function("scalar_fast_2halley", |b| {
        b.iter(|| scalar_cbrt_fast(black_box(&single)))
    });
    group.bench_function("scalar_midp_3newton", |b| {
        b.iter(|| scalar_cbrt_midp(black_box(&single)))
    });
    group.finish();

    // Bulk (1024 values) benchmarks
    let mut group = c.benchmark_group("cbrt_bulk_1024");

    if let Some(token) = X64V3Token::summon() {
        group.bench_function("simd_lowp_1halley", |b| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for chunk in &data {
                    let r = simd_cbrt_lowp(token, black_box(chunk));
                    sum += r[0];
                }
                sum
            })
        });
        group.bench_function("simd_fast_2halley", |b| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for chunk in &data {
                    let r = simd_cbrt_fast(token, black_box(chunk));
                    sum += r[0];
                }
                sum
            })
        });
        group.bench_function("simd_midp_3newton", |b| {
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
    group.bench_function("scalar_lowp_1halley", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for chunk in &data {
                let r = scalar_cbrt_lowp(black_box(chunk));
                sum += r[0];
            }
            sum
        })
    });
    group.bench_function("scalar_fast_2halley", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for chunk in &data {
                let r = scalar_cbrt_fast(black_box(chunk));
                sum += r[0];
            }
            sum
        })
    });
    group.bench_function("scalar_midp_3newton", |b| {
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

criterion_group!(benches, bench_cbrt_variants);
criterion_main!(benches);

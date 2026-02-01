//! Benchmarks for archmage tokens and operations

use criterion::{Criterion, black_box, criterion_group, criterion_main};

#[cfg(target_arch = "x86_64")]
fn bench_token_overhead(c: &mut Criterion) {
    use archmage::{SimdToken, X64V3Token};

    let mut group = c.benchmark_group("token_overhead");

    // Benchmark token creation
    group.bench_function("x64v3_try_new", |b| {
        b.iter(|| black_box(X64V3Token::try_new()))
    });

    // Benchmark operations with token vs raw intrinsics
    if let Some(_token) = X64V3Token::try_new() {
        let data = [1.0f32; 8];

        group.bench_function("load_with_safe_unaligned_simd", |b| {
            b.iter(|| {
                // safe_unaligned_simd is always safe inside target_feature functions
                // but here we're outside, so we still use unsafe
                let v = unsafe { core::arch::x86_64::_mm256_loadu_ps(black_box(&data).as_ptr()) };
                black_box(v)
            })
        });
    }

    group.finish();
}

#[cfg(all(target_arch = "x86_64", feature = "__composite"))]
fn bench_composite_ops(c: &mut Criterion) {
    use archmage::{SimdToken, X64V3Token, composite};

    let mut group = c.benchmark_group("composite");

    if let Some(token) = X64V3Token::try_new() {
        let mut block: [f32; 64] = core::array::from_fn(|i| i as f32);

        group.bench_function("transpose_8x8", |b| {
            b.iter(|| {
                composite::transpose_8x8(token, black_box(&mut block));
            })
        });

        let a: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![1.0; 1024];

        group.bench_function("dot_product_1024", |b_iter| {
            b_iter.iter(|| {
                black_box(composite::dot_product_f32(
                    token,
                    black_box(&a),
                    black_box(&b),
                ))
            })
        });
    }

    group.finish();
}

#[cfg(all(target_arch = "x86_64", not(feature = "__composite")))]
fn bench_composite_ops(_c: &mut Criterion) {
    // Composite feature not enabled
}

#[cfg(target_arch = "x86_64")]
criterion_group!(benches, bench_token_overhead, bench_composite_ops);

#[cfg(not(target_arch = "x86_64"))]
fn placeholder(_c: &mut Criterion) {}

#[cfg(not(target_arch = "x86_64"))]
criterion_group!(benches, placeholder);

criterion_main!(benches);

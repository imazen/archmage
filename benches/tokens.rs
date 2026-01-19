//! Benchmarks for archmage tokens and operations

use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[cfg(target_arch = "x86_64")]
fn bench_token_overhead(c: &mut Criterion) {
    use archmage::{composite, ops, Avx2FmaToken, Avx2Token};

    let mut group = c.benchmark_group("token_overhead");

    // Benchmark token creation
    group.bench_function("avx2_try_new", |b| {
        b.iter(|| black_box(Avx2Token::try_new()))
    });

    group.bench_function("avx2_fma_try_new", |b| {
        b.iter(|| black_box(Avx2FmaToken::try_new()))
    });

    // Benchmark operations with token vs raw intrinsics
    if let Some(token) = Avx2Token::try_new() {
        let data = [1.0f32; 8];

        group.bench_function("load_with_token", |b| {
            b.iter(|| {
                let v = ops::x86::load_f32x8(token, black_box(&data));
                black_box(v)
            })
        });
    }

    group.finish();
}

#[cfg(target_arch = "x86_64")]
fn bench_composite_ops(c: &mut Criterion) {
    use archmage::{composite, Avx2FmaToken, Avx2Token};

    let mut group = c.benchmark_group("composite");

    if let Some(token) = Avx2Token::try_new() {
        let mut block: [f32; 64] = core::array::from_fn(|i| i as f32);

        group.bench_function("transpose_8x8", |b| {
            b.iter(|| {
                composite::transpose_8x8(token, black_box(&mut block));
            })
        });
    }

    if let Some(token) = Avx2FmaToken::try_new() {
        let a: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![1.0; 1024];

        group.bench_function("dot_product_1024", |b_iter| {
            b_iter.iter(|| {
                black_box(composite::dot_product_f32(token, black_box(&a), black_box(&b)))
            })
        });
    }

    group.finish();
}

#[cfg(target_arch = "x86_64")]
criterion_group!(benches, bench_token_overhead, bench_composite_ops);

#[cfg(not(target_arch = "x86_64"))]
fn placeholder(_c: &mut Criterion) {}

#[cfg(not(target_arch = "x86_64"))]
criterion_group!(benches, placeholder);

criterion_main!(benches);

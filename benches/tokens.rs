//! Benchmarks for archmage tokens and operations

use criterion::{Criterion, black_box, criterion_group, criterion_main};

#[cfg(target_arch = "x86_64")]
fn bench_token_overhead(c: &mut Criterion) {
    use archmage::{SimdToken, X64V3Token};

    let mut group = c.benchmark_group("token_overhead");

    // Benchmark token creation
    group.bench_function("x64v3_summon", |b| {
        b.iter(|| black_box(X64V3Token::summon()))
    });

    // Benchmark operations with token vs raw intrinsics
    if let Some(_token) = X64V3Token::summon() {
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

#[cfg(target_arch = "x86_64")]
criterion_group!(benches, bench_token_overhead);

#[cfg(not(target_arch = "x86_64"))]
fn placeholder(_c: &mut Criterion) {}

#[cfg(not(target_arch = "x86_64"))]
criterion_group!(benches, placeholder);

criterion_main!(benches);

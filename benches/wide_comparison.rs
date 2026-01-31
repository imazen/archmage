//! Benchmarks comparing archmage SIMD types against the wide crate.
//!
//! Run with: cargo bench --bench wide_comparison --features wide

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

// ============================================================================
// f32x8 Comparisons
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "wide"))]
fn bench_f32x8_arithmetic(c: &mut Criterion) {
    use archmage::simd::f32x8 as arch_f32x8;
    use archmage::{SimdToken, X64V3Token};
    use wide::f32x8 as wide_f32x8;

    let Some(token) = X64V3Token::try_new() else {
        eprintln!("AVX2+FMA not available, skipping benchmarks");
        return;
    };

    let mut group = c.benchmark_group("f32x8_arithmetic");
    group.throughput(Throughput::Elements(8));

    // Setup data
    let data_a = [1.5f32, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];
    let data_b = [0.5f32, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];

    let arch_a = arch_f32x8::from_array(token, data_a);
    let arch_b = arch_f32x8::from_array(token, data_b);
    let wide_a = wide_f32x8::from(data_a);
    let wide_b = wide_f32x8::from(data_b);

    // Add
    group.bench_function("archmage_add", |b| {
        b.iter(|| black_box(black_box(arch_a) + black_box(arch_b)))
    });
    group.bench_function("wide_add", |b| {
        b.iter(|| black_box(black_box(wide_a) + black_box(wide_b)))
    });

    // Mul
    group.bench_function("archmage_mul", |b| {
        b.iter(|| black_box(black_box(arch_a) * black_box(arch_b)))
    });
    group.bench_function("wide_mul", |b| {
        b.iter(|| black_box(black_box(wide_a) * black_box(wide_b)))
    });

    // Div
    group.bench_function("archmage_div", |b| {
        b.iter(|| black_box(black_box(arch_a) / black_box(arch_b)))
    });
    group.bench_function("wide_div", |b| {
        b.iter(|| black_box(black_box(wide_a) / black_box(wide_b)))
    });

    group.finish();
}

#[cfg(all(target_arch = "x86_64", feature = "wide"))]
fn bench_f32x8_math(c: &mut Criterion) {
    use archmage::simd::f32x8 as arch_f32x8;
    use archmage::{SimdToken, X64V3Token};
    use wide::f32x8 as wide_f32x8;

    let Some(token) = X64V3Token::try_new() else {
        return;
    };

    let mut group = c.benchmark_group("f32x8_math");
    group.throughput(Throughput::Elements(8));

    let data = [1.5f32, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];
    let arch_v = arch_f32x8::from_array(token, data);
    let wide_v = wide_f32x8::from(data);

    // Sqrt
    group.bench_function("archmage_sqrt", |b| {
        b.iter(|| black_box(black_box(arch_v).sqrt()))
    });
    group.bench_function("wide_sqrt", |b| {
        b.iter(|| black_box(black_box(wide_v).sqrt()))
    });

    // Abs (need negative values)
    let neg_data = [-1.5f32, 2.5, -3.5, 4.5, -5.5, 6.5, -7.5, 8.5];
    let arch_neg = arch_f32x8::from_array(token, neg_data);
    let wide_neg = wide_f32x8::from(neg_data);

    group.bench_function("archmage_abs", |b| {
        b.iter(|| black_box(black_box(arch_neg).abs()))
    });
    group.bench_function("wide_abs", |b| {
        b.iter(|| black_box(black_box(wide_neg).abs()))
    });

    // Floor
    group.bench_function("archmage_floor", |b| {
        b.iter(|| black_box(black_box(arch_v).floor()))
    });
    group.bench_function("wide_floor", |b| {
        b.iter(|| black_box(black_box(wide_v).floor()))
    });

    // Ceil
    group.bench_function("archmage_ceil", |b| {
        b.iter(|| black_box(black_box(arch_v).ceil()))
    });
    group.bench_function("wide_ceil", |b| {
        b.iter(|| black_box(black_box(wide_v).ceil()))
    });

    // Round
    group.bench_function("archmage_round", |b| {
        b.iter(|| black_box(black_box(arch_v).round()))
    });
    group.bench_function("wide_round", |b| {
        b.iter(|| black_box(black_box(wide_v).round()))
    });

    group.finish();
}

#[cfg(all(target_arch = "x86_64", feature = "wide"))]
fn bench_f32x8_minmax(c: &mut Criterion) {
    use archmage::simd::f32x8 as arch_f32x8;
    use archmage::{SimdToken, X64V3Token};
    use wide::f32x8 as wide_f32x8;

    let Some(token) = X64V3Token::try_new() else {
        return;
    };

    let mut group = c.benchmark_group("f32x8_minmax");
    group.throughput(Throughput::Elements(8));

    let data_a = [1.5f32, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];
    let data_b = [8.5f32, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5];

    let arch_a = arch_f32x8::from_array(token, data_a);
    let arch_b = arch_f32x8::from_array(token, data_b);
    let wide_a = wide_f32x8::from(data_a);
    let wide_b = wide_f32x8::from(data_b);

    group.bench_function("archmage_min", |b| {
        b.iter(|| black_box(black_box(arch_a).min(black_box(arch_b))))
    });
    group.bench_function("wide_min", |b| {
        b.iter(|| black_box(black_box(wide_a).min(black_box(wide_b))))
    });

    group.bench_function("archmage_max", |b| {
        b.iter(|| black_box(black_box(arch_a).max(black_box(arch_b))))
    });
    group.bench_function("wide_max", |b| {
        b.iter(|| black_box(black_box(wide_a).max(black_box(wide_b))))
    });

    group.finish();
}

#[cfg(all(target_arch = "x86_64", feature = "wide"))]
fn bench_f32x8_fma(c: &mut Criterion) {
    use archmage::simd::f32x8 as arch_f32x8;
    use archmage::{SimdToken, X64V3Token};
    use wide::f32x8 as wide_f32x8;

    let Some(token) = X64V3Token::try_new() else {
        return;
    };

    let mut group = c.benchmark_group("f32x8_fma");
    group.throughput(Throughput::Elements(8));

    let data_a = [1.5f32, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];
    let data_b = [0.5f32, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];
    let data_c = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

    let arch_a = arch_f32x8::from_array(token, data_a);
    let arch_b = arch_f32x8::from_array(token, data_b);
    let arch_c = arch_f32x8::from_array(token, data_c);
    let wide_a = wide_f32x8::from(data_a);
    let wide_b = wide_f32x8::from(data_b);
    let wide_c = wide_f32x8::from(data_c);

    group.bench_function("archmage_mul_add", |b| {
        b.iter(|| black_box(black_box(arch_a).mul_add(black_box(arch_b), black_box(arch_c))))
    });
    group.bench_function("wide_mul_add", |b| {
        b.iter(|| black_box(black_box(wide_a).mul_add(black_box(wide_b), black_box(wide_c))))
    });

    group.finish();
}

#[cfg(all(target_arch = "x86_64", feature = "wide"))]
fn bench_f32x8_reductions(c: &mut Criterion) {
    use archmage::simd::f32x8 as arch_f32x8;
    use archmage::{SimdToken, X64V3Token};
    use wide::f32x8 as wide_f32x8;

    let Some(token) = X64V3Token::try_new() else {
        return;
    };

    let mut group = c.benchmark_group("f32x8_reductions");
    group.throughput(Throughput::Elements(8));

    let data = [1.5f32, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];
    let arch_v = arch_f32x8::from_array(token, data);
    let wide_v = wide_f32x8::from(data);

    group.bench_function("archmage_reduce_add", |b| {
        b.iter(|| black_box(black_box(arch_v).reduce_add()))
    });
    group.bench_function("wide_reduce_add", |b| {
        b.iter(|| black_box(black_box(wide_v).reduce_add()))
    });

    // Note: wide doesn't have reduce_min/reduce_max for f32x8
    group.bench_function("archmage_reduce_min", |b| {
        b.iter(|| black_box(black_box(arch_v).reduce_min()))
    });

    group.bench_function("archmage_reduce_max", |b| {
        b.iter(|| black_box(black_box(arch_v).reduce_max()))
    });

    group.finish();
}

#[cfg(all(target_arch = "x86_64", feature = "wide"))]
fn bench_f32x8_memory(c: &mut Criterion) {
    use archmage::simd::f32x8 as arch_f32x8;
    use archmage::{SimdToken, X64V3Token};
    use wide::f32x8 as wide_f32x8;

    let Some(token) = X64V3Token::try_new() else {
        return;
    };

    let mut group = c.benchmark_group("f32x8_memory");
    group.throughput(Throughput::Elements(8));

    let data = [1.5f32, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];

    // Memory load (explicit load vs From transmute)
    group.bench_function("archmage_load", |b| {
        b.iter(|| black_box(arch_f32x8::load(token, black_box(&data))))
    });
    group.bench_function("archmage_from", |b| {
        b.iter(|| black_box(arch_f32x8::from(black_box(data))))
    });
    group.bench_function("wide_from", |b| {
        b.iter(|| black_box(wide_f32x8::from(black_box(data))))
    });

    let arch_v = arch_f32x8::from_array(token, data);
    let wide_v = wide_f32x8::from(data);

    // Store / Into array
    group.bench_function("archmage_store", |b| {
        let mut out = [0.0f32; 8];
        b.iter(|| {
            black_box(arch_v).store(&mut out);
            black_box(out)
        })
    });
    group.bench_function("archmage_into", |b| {
        b.iter(|| black_box(<[f32; 8]>::from(black_box(arch_v))))
    });
    group.bench_function("wide_into", |b| {
        b.iter(|| black_box(<[f32; 8]>::from(black_box(wide_v))))
    });

    group.finish();
}

// ============================================================================
// i32x8 Comparisons
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "wide"))]
fn bench_i32x8_arithmetic(c: &mut Criterion) {
    use archmage::simd::i32x8 as arch_i32x8;
    use archmage::{SimdToken, X64V3Token};
    use wide::i32x8 as wide_i32x8;

    let Some(token) = X64V3Token::try_new() else {
        return;
    };

    let mut group = c.benchmark_group("i32x8_arithmetic");
    group.throughput(Throughput::Elements(8));

    let data_a = [1i32, 2, 3, 4, 5, 6, 7, 8];
    let data_b = [8i32, 7, 6, 5, 4, 3, 2, 1];

    let arch_a = arch_i32x8::from_array(token, data_a);
    let arch_b = arch_i32x8::from_array(token, data_b);
    let wide_a = wide_i32x8::from(data_a);
    let wide_b = wide_i32x8::from(data_b);

    group.bench_function("archmage_add", |b| {
        b.iter(|| black_box(black_box(arch_a) + black_box(arch_b)))
    });
    group.bench_function("wide_add", |b| {
        b.iter(|| black_box(black_box(wide_a) + black_box(wide_b)))
    });

    group.bench_function("archmage_sub", |b| {
        b.iter(|| black_box(black_box(arch_a) - black_box(arch_b)))
    });
    group.bench_function("wide_sub", |b| {
        b.iter(|| black_box(black_box(wide_a) - black_box(wide_b)))
    });

    group.finish();
}

#[cfg(all(target_arch = "x86_64", feature = "wide"))]
fn bench_i32x8_reductions(c: &mut Criterion) {
    use archmage::simd::i32x8 as arch_i32x8;
    use archmage::{SimdToken, X64V3Token};
    use wide::i32x8 as wide_i32x8;

    let Some(token) = X64V3Token::try_new() else {
        return;
    };

    let mut group = c.benchmark_group("i32x8_reductions");
    group.throughput(Throughput::Elements(8));

    let data = [1i32, 2, 3, 4, 5, 6, 7, 8];
    let arch_v = arch_i32x8::from_array(token, data);
    let wide_v = wide_i32x8::from(data);

    group.bench_function("archmage_reduce_add", |b| {
        b.iter(|| black_box(black_box(arch_v).reduce_add()))
    });
    group.bench_function("wide_reduce_add", |b| {
        b.iter(|| black_box(black_box(wide_v).reduce_add()))
    });

    group.finish();
}

// ============================================================================
// Batched operations (more realistic workload)
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "wide"))]
fn bench_batch_operations(c: &mut Criterion) {
    use archmage::simd::f32x8 as arch_f32x8;
    use archmage::{SimdToken, X64V3Token};
    use wide::f32x8 as wide_f32x8;

    let Some(token) = X64V3Token::try_new() else {
        return;
    };

    let mut group = c.benchmark_group("batch_ops");

    // Process 1024 floats (128 vectors of 8)
    const N: usize = 128;
    let input: Vec<[f32; 8]> = (0..N)
        .map(|i| {
            let base = (i * 8) as f32;
            [
                base,
                base + 1.0,
                base + 2.0,
                base + 3.0,
                base + 4.0,
                base + 5.0,
                base + 6.0,
                base + 7.0,
            ]
        })
        .collect();

    group.throughput(Throughput::Elements((N * 8) as u64));

    // Batch add using From trait (zero-cost transmute)
    group.bench_function("archmage_batch_add_from", |b| {
        b.iter(|| {
            let mut sum = arch_f32x8::zero(token);
            for arr in &input {
                let v: arch_f32x8 = (*arr).into();
                sum = sum + v;
            }
            black_box(sum.reduce_add())
        })
    });

    group.bench_function("wide_batch_add", |b| {
        b.iter(|| {
            let mut sum = wide_f32x8::ZERO;
            for arr in &input {
                let v = wide_f32x8::from(*arr);
                sum = sum + v;
            }
            black_box(sum.reduce_add())
        })
    });

    // Batch mul_add using From trait (zero-cost transmute)
    group.bench_function("archmage_batch_fma_from", |b| {
        let scale = arch_f32x8::splat(token, 0.5);
        let offset = arch_f32x8::splat(token, 1.0);
        b.iter(|| {
            let mut acc = arch_f32x8::zero(token);
            for arr in &input {
                let v: arch_f32x8 = (*arr).into();
                acc = v.mul_add(scale, offset);
            }
            black_box(acc.reduce_add())
        })
    });

    group.bench_function("wide_batch_fma", |b| {
        let scale = wide_f32x8::splat(0.5);
        let offset = wide_f32x8::splat(1.0);
        b.iter(|| {
            let mut acc = wide_f32x8::ZERO;
            for arr in &input {
                let v = wide_f32x8::from(*arr);
                acc = v.mul_add(scale, offset);
            }
            black_box(acc.reduce_add())
        })
    });

    group.finish();
}

// ============================================================================
// Proper usage: #[target_feature] context (like #[arcane] generates)
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "wide"))]
mod target_feature_bench {
    use archmage::simd::f32x8;
    use core::arch::x86_64::*;

    /// Inner function with target_feature - this is what #[arcane] generates.
    /// Operators WILL inline properly within this context.
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn batch_add_inner(input: &[[f32; 8]], zero: f32x8) -> f32 {
        let mut sum = zero;
        for arr in input {
            let v: f32x8 = (*arr).into();
            sum = sum + v; // This + will inline the vaddps instruction!
        }
        sum.reduce_add()
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn batch_fma_inner(
        input: &[[f32; 8]],
        scale: f32x8,
        offset: f32x8,
        zero: f32x8,
    ) -> f32 {
        let mut acc = zero;
        for arr in input {
            let v: f32x8 = (*arr).into();
            acc = v.mul_add(scale, offset); // FMA will inline too
        }
        acc.reduce_add()
    }
}

/// Benchmark showing proper archmage usage within #[target_feature] context.
/// This is how archmage is designed to be used - operators inline when called
/// from functions that have matching target_feature attributes.
#[cfg(all(target_arch = "x86_64", feature = "wide"))]
fn bench_target_feature_context(c: &mut Criterion) {
    use archmage::simd::f32x8 as arch_f32x8;
    use archmage::{SimdToken, X64V3Token};
    use wide::f32x8 as wide_f32x8;

    let Some(token) = X64V3Token::try_new() else {
        eprintln!("AVX2+FMA not available, skipping benchmarks");
        return;
    };

    let mut group = c.benchmark_group("target_feature_context");

    const N: usize = 128;
    let input: Vec<[f32; 8]> = (0..N)
        .map(|i| {
            let base = (i * 8) as f32;
            [
                base,
                base + 1.0,
                base + 2.0,
                base + 3.0,
                base + 4.0,
                base + 5.0,
                base + 6.0,
                base + 7.0,
            ]
        })
        .collect();

    group.throughput(Throughput::Elements((N * 8) as u64));

    // Archmage with proper #[target_feature] context (like #[arcane])
    // This is how you should use archmage for performance-critical code!
    group.bench_function("archmage_proper_batch_add", |b| {
        let zero = arch_f32x8::zero(token);
        b.iter(|| {
            // SAFETY: token proves CPU supports AVX2+FMA
            unsafe { target_feature_bench::batch_add_inner(black_box(&input), zero) }
        })
    });

    // Wide for comparison
    group.bench_function("wide_batch_add", |b| {
        b.iter(|| {
            let mut sum = wide_f32x8::ZERO;
            for arr in &input {
                let v = wide_f32x8::from(*arr);
                sum = sum + v;
            }
            black_box(sum.reduce_add())
        })
    });

    // FMA comparison
    group.bench_function("archmage_proper_batch_fma", |b| {
        let scale = arch_f32x8::splat(token, 0.5);
        let offset = arch_f32x8::splat(token, 1.0);
        let zero = arch_f32x8::zero(token);
        b.iter(|| unsafe {
            target_feature_bench::batch_fma_inner(black_box(&input), scale, offset, zero)
        })
    });

    group.bench_function("wide_batch_fma", |b| {
        let scale = wide_f32x8::splat(0.5);
        let offset = wide_f32x8::splat(1.0);
        b.iter(|| {
            let mut acc = wide_f32x8::ZERO;
            for arr in &input {
                let v = wide_f32x8::from(*arr);
                acc = v.mul_add(scale, offset);
            }
            black_box(acc.reduce_add())
        })
    });

    group.finish();
}

// ============================================================================
// Criterion groups
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "wide"))]
criterion_group!(
    benches,
    bench_f32x8_arithmetic,
    bench_f32x8_math,
    bench_f32x8_minmax,
    bench_f32x8_fma,
    bench_f32x8_reductions,
    bench_f32x8_memory,
    bench_i32x8_arithmetic,
    bench_i32x8_reductions,
    bench_batch_operations,
    bench_target_feature_context,
);

#[cfg(not(all(target_arch = "x86_64", feature = "wide")))]
fn placeholder(_c: &mut Criterion) {
    eprintln!("Benchmarks require x86_64 and the 'wide' feature");
}

#[cfg(not(all(target_arch = "x86_64", feature = "wide")))]
criterion_group!(benches, placeholder);

criterion_main!(benches);

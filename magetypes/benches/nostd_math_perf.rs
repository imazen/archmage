//! Benchmark: `nostd_math` vs the std inherent math methods.
//!
//! Each family pairs the std method against the `nostd_math` fallback over the
//! same L1-resident input, so the delta is the op cost alone.
//!
//! Run with:
//!   cargo bench -p magetypes --bench nostd_math_perf

use zenbench::prelude::*;

/// Ascending positives — valid for `sqrt`.
fn positives_f32() -> [f32; 1024] {
    core::array::from_fn(|i| (i + 1) as f32 * 0.1)
}

fn positives_f64() -> [f64; 1024] {
    core::array::from_fn(|i| (i + 1) as f64 * 0.1)
}

/// Straddles zero — exercises both rounding directions of `floor`/`round`.
fn signed_f32() -> [f32; 1024] {
    core::array::from_fn(|i| (i + 1) as f32 * 0.1 - 50.0)
}

fn signed_f64() -> [f64; 1024] {
    core::array::from_fn(|i| (i + 1) as f64 * 0.1 - 50.0)
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

fn bench_sqrt(suite: &mut Suite) {
    suite.group("sqrt", |g| {
        let v32 = positives_f32();
        g.bench("f32_sqrt/std", move |b| {
            b.iter(|| sum_f32(&v32, |x| x.sqrt()))
        });
        g.bench("f32_sqrt/nostd", move |b| {
            b.iter(|| sum_f32(&v32, magetypes::nostd_math::sqrtf))
        });

        let v64 = positives_f64();
        g.bench("f64_sqrt/std", move |b| {
            b.iter(|| sum_f64(&v64, |x| x.sqrt()))
        });
        g.bench("f64_sqrt/nostd", move |b| {
            b.iter(|| sum_f64(&v64, magetypes::nostd_math::sqrt))
        });
    });
}

fn bench_floor(suite: &mut Suite) {
    suite.group("floor", |g| {
        let v32 = signed_f32();
        g.bench("f32_floor/std", move |b| {
            b.iter(|| sum_f32(&v32, |x| x.floor()))
        });
        g.bench("f32_floor/nostd", move |b| {
            b.iter(|| sum_f32(&v32, magetypes::nostd_math::floorf))
        });

        let v64 = signed_f64();
        g.bench("f64_floor/std", move |b| {
            b.iter(|| sum_f64(&v64, |x| x.floor()))
        });
        g.bench("f64_floor/nostd", move |b| {
            b.iter(|| sum_f64(&v64, magetypes::nostd_math::floor))
        });
    });
}

fn bench_round(suite: &mut Suite) {
    suite.group("round", |g| {
        let v32 = signed_f32();
        g.bench("f32_round/std", move |b| {
            b.iter(|| sum_f32(&v32, |x| x.round()))
        });
        g.bench("f32_round/nostd", move |b| {
            b.iter(|| sum_f32(&v32, magetypes::nostd_math::roundf))
        });

        let v64 = signed_f64();
        g.bench("f64_round/std", move |b| {
            b.iter(|| sum_f64(&v64, |x| x.round()))
        });
        g.bench("f64_round/nostd", move |b| {
            b.iter(|| sum_f64(&v64, magetypes::nostd_math::round))
        });
    });
}

fn bench_fma(suite: &mut Suite) {
    suite.group("fma", |g| {
        let v32 = signed_f32();
        g.bench("fma/f32_unfused_1024", move |b| {
            b.iter(|| sum_f32(&v32, |x| x * black_box(0.731) + black_box(-0.219)))
        });
        g.bench("fma/f32_fused_software_1024", move |b| {
            b.iter(|| {
                sum_f32(&v32, |x| {
                    magetypes::nostd_math::fmaf(x, black_box(0.731), black_box(-0.219))
                })
            })
        });
        let v64 = signed_f64();
        g.bench("fma/f64_unfused_1024", move |b| {
            b.iter(|| sum_f64(&v64, |x| x * black_box(0.731) + black_box(-0.219)))
        });
        g.bench("fma/f64_fused_software_1024", move |b| {
            b.iter(|| {
                sum_f64(&v64, |x| {
                    magetypes::nostd_math::fma(x, black_box(0.731), black_box(-0.219))
                })
            })
        });
    });
}

zenbench::main!(bench_sqrt, bench_floor, bench_round, bench_fma);

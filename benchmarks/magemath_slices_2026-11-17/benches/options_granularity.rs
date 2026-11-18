//! Measures whole-slice throughput (`len = 4096`, no `-C target-cpu`) across
//! granular domain flags, out-of-bounds policies, correctness fixes, and
//! single-octave zero-reduction ranges.
use magemath::{
    cbrtf::{Cbrtf, CbrtfOptions},
    expf::{Expf, ExpfOptions},
    logf::{Logf, LogfOptions},
    powf::{Powf, PowfOptions},
};
use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

// Logf granularity & single-octave zero-reduction
const LOG_DEFAULT: u128 = LogfOptions::new().bits();
const LOG_ZERO: u128 = LogfOptions::new().admit_zero().bits();
const LOG_SUBNORM: u128 = LogfOptions::new().admit_subnormals().bits();
const LOG_FULL: u128 = LogfOptions::new().admit_all_floats().bits();
const LOG_NEAR_ONE: u128 = LogfOptions::new().near_one_range().bits();
const LOG_NEAR_ONE_ASSUME: u128 = LogfOptions::new()
    .near_one_range()
    .assume_in_bounds()
    .bits();

// Expf clamp elision, 127.5 endpoint fix, zero-reduction & OOB clamp
const EXP_DEFAULT: u128 = ExpfOptions::new().bits();
const EXP_AT_127_5: u128 = ExpfOptions::new().input_range(-126.0, 127.5).bits();
const EXP_UNCLAMPED: u128 = ExpfOptions::new().unclamped_exp2_range().bits();
const EXP_SMALL: u128 = ExpfOptions::new().small_exp2_range().bits();
const EXP_SMALL_CLAMP_OOB: u128 = ExpfOptions::new()
    .small_exp2_range()
    .clamp_out_of_bounds()
    .bits();
const EXP_SMALL_ASSUME: u128 = ExpfOptions::new()
    .small_exp2_range()
    .assume_in_bounds()
    .bits();

// Cbrtf denorm elision vs admit_subnormals vs full_domain
const CBRT_DEFAULT: u128 = CbrtfOptions::new().bits();
const CBRT_STRICT_NORMAL: u128 = CbrtfOptions::new().input_range(0.01, 4.0).bits();
const CBRT_UNIT_SUBNORM: u128 = CbrtfOptions::new().unit_range().bits();
const CBRT_SUBNORM_ASSUME: u128 = CbrtfOptions::new()
    .admit_subnormals()
    .assume_in_bounds()
    .bits();
const CBRT_FULL: u128 = CbrtfOptions::new().admit_all_floats().bits();

// Powf domain granularity, srgb_encode_domain & OOB policies
const POW_GAMMA_ERR: u128 = PowfOptions::new().gamma_domain().bits();
const POW_GAMMA_CLAMP: u128 = PowfOptions::new()
    .gamma_domain()
    .clamp_out_of_bounds()
    .bits();
const POW_GAMMA_ASSUME: u128 = PowfOptions::new().gamma_domain().assume_in_bounds().bits();
const POW_SRGB_ENCODE: u128 = PowfOptions::new().srgb_encode_domain().bits();
const POW_ZERO_ONLY: u128 = PowfOptions::new().admit_zero_base().bits();
const POW_ZERO_AND_SUB: u128 = PowfOptions::new()
    .admit_zero_base()
    .admit_subnormals()
    .bits();

fn bench_granularity(c: &mut Criterion) {
    let len = 4096usize;
    // Strictly inside [0.75, 1.25] and [0.01, 1.0] and [-0.5, 0.5] as needed.
    let near_one: Vec<f32> = (0..len)
        .map(|i| 0.75 + 0.5 * (i as f32) / (len as f32))
        .collect();
    let unit_pos: Vec<f32> = (0..len)
        .map(|i| 0.01 + 0.98 * (i as f32) / (len as f32))
        .collect();
    let small_sym: Vec<f32> = (0..len)
        .map(|i| -0.45 + 0.9 * (i as f32) / (len as f32))
        .collect();
    let mut dst = vec![0.0_f32; len];

    let mut g_log = c.benchmark_group("log2_4096");
    g_log.bench_function("default_normals", |b| {
        b.iter(|| Logf::<LOG_DEFAULT>::log2_slice(black_box(&near_one), &mut dst).unwrap())
    });
    g_log.bench_function("admit_zero", |b| {
        b.iter(|| Logf::<LOG_ZERO>::log2_slice(black_box(&near_one), &mut dst).unwrap())
    });
    g_log.bench_function("admit_subnormals", |b| {
        b.iter(|| Logf::<LOG_SUBNORM>::log2_slice(black_box(&near_one), &mut dst).unwrap())
    });
    g_log.bench_function("admit_all_floats", |b| {
        b.iter(|| Logf::<LOG_FULL>::log2_slice(black_box(&near_one), &mut dst).unwrap())
    });
    g_log.bench_function("near_one_range", |b| {
        b.iter(|| Logf::<LOG_NEAR_ONE>::log2_slice(black_box(&near_one), &mut dst).unwrap())
    });
    g_log.bench_function("near_one_assume", |b| {
        b.iter(|| Logf::<LOG_NEAR_ONE_ASSUME>::log2_slice(black_box(&near_one), &mut dst).unwrap())
    });
    g_log.finish();

    let mut g_exp = c.benchmark_group("exp2_4096");
    g_exp.bench_function("default_finite", |b| {
        b.iter(|| Expf::<EXP_DEFAULT>::exp2_slice(black_box(&small_sym), &mut dst).unwrap())
    });
    g_exp.bench_function("range_at_127_5_clamped", |b| {
        b.iter(|| Expf::<EXP_AT_127_5>::exp2_slice(black_box(&small_sym), &mut dst).unwrap())
    });
    g_exp.bench_function("unclamped_exp2_range", |b| {
        b.iter(|| Expf::<EXP_UNCLAMPED>::exp2_slice(black_box(&small_sym), &mut dst).unwrap())
    });
    g_exp.bench_function("small_exp2_range", |b| {
        b.iter(|| Expf::<EXP_SMALL>::exp2_slice(black_box(&small_sym), &mut dst).unwrap())
    });
    g_exp.bench_function("small_exp2_clamp_oob", |b| {
        b.iter(|| Expf::<EXP_SMALL_CLAMP_OOB>::exp2_slice(black_box(&small_sym), &mut dst).unwrap())
    });
    g_exp.bench_function("small_exp2_assume", |b| {
        b.iter(|| Expf::<EXP_SMALL_ASSUME>::exp2_slice(black_box(&small_sym), &mut dst).unwrap())
    });
    g_exp.finish();

    let mut g_cbrt = c.benchmark_group("cbrt_4096");
    g_cbrt.bench_function("default_bounded", |b| {
        b.iter(|| Cbrtf::<CBRT_DEFAULT>::slice(black_box(&unit_pos), &mut dst).unwrap())
    });
    g_cbrt.bench_function("strict_normal_range", |b| {
        b.iter(|| Cbrtf::<CBRT_STRICT_NORMAL>::slice(black_box(&unit_pos), &mut dst).unwrap())
    });
    g_cbrt.bench_function("unit_range_subnormals", |b| {
        b.iter(|| Cbrtf::<CBRT_UNIT_SUBNORM>::slice(black_box(&unit_pos), &mut dst).unwrap())
    });
    g_cbrt.bench_function("subnormals_assume_in_bounds", |b| {
        b.iter(|| Cbrtf::<CBRT_SUBNORM_ASSUME>::slice(black_box(&unit_pos), &mut dst).unwrap())
    });
    g_cbrt.bench_function("admit_all_floats", |b| {
        b.iter(|| Cbrtf::<CBRT_FULL>::slice(black_box(&unit_pos), &mut dst).unwrap())
    });
    g_cbrt.finish();

    let mut g_pow = c.benchmark_group("powf_4096");
    g_pow.bench_function("gamma_error_on_oob", |b| {
        b.iter(|| {
            Powf::<POW_GAMMA_ERR>::slice(black_box(&unit_pos), &mut dst, black_box(1.0 / 2.4))
                .unwrap()
        })
    });
    g_pow.bench_function("gamma_clamp_oob", |b| {
        b.iter(|| {
            Powf::<POW_GAMMA_CLAMP>::slice(black_box(&unit_pos), &mut dst, black_box(1.0 / 2.4))
                .unwrap()
        })
    });
    g_pow.bench_function("gamma_assume_in_bounds", |b| {
        b.iter(|| {
            Powf::<POW_GAMMA_ASSUME>::slice(black_box(&unit_pos), &mut dst, black_box(1.0 / 2.4))
                .unwrap()
        })
    });
    g_pow.bench_function("srgb_encode_domain", |b| {
        b.iter(|| {
            Powf::<POW_SRGB_ENCODE>::slice(black_box(&unit_pos), &mut dst, black_box(1.0 / 2.4))
                .unwrap()
        })
    });
    g_pow.bench_function("unranged_admit_zero", |b| {
        b.iter(|| {
            Powf::<POW_ZERO_ONLY>::slice(black_box(&unit_pos), &mut dst, black_box(1.0 / 2.4))
                .unwrap()
        })
    });
    g_pow.bench_function("unranged_zero_and_subnormals", |b| {
        b.iter(|| {
            Powf::<POW_ZERO_AND_SUB>::slice(black_box(&unit_pos), &mut dst, black_box(1.0 / 2.4))
                .unwrap()
        })
    });
    g_pow.finish();
}

criterion_group!(benches, bench_granularity);
criterion_main!(benches);

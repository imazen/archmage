//! Measures whole-slice throughput (`len = 4096`, no `-C target-cpu`) across
//! discrete `*Domain` enums, `Subnormals` policies (`Reject`, `FlushOrClamp`,
//! `Normalize`), `bf16` bounds, `const_exponent` specializations, and dedicated
//! `log1pf`.
use magemath::{
    cbrtf::{Cbrtf, CbrtfDomain, CbrtfOptions},
    expf::{Expf, ExpfOptions},
    log1pf::{Log1pf, Log1pfDomain, Log1pfOptions},
    logf::{Logf, LogfOptions},
    powf::{Powf, PowfDomain, PowfOptions},
};
use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

// Logf granularity & single-octave zero-reduction
const LOG_DEFAULT: u128 = LogfOptions::new().bits();
const LOG_ZERO: u128 = LogfOptions::new().admit_zero().bits();
const LOG_FLUSH_SUB: u128 = LogfOptions::new()
    .admit_zero()
    .flush_or_clamp_subnormals()
    .bits();
const LOG_SUBNORM: u128 = LogfOptions::new().admit_subnormals().bits();
const LOG_FULL: u128 = LogfOptions::new().admit_all_floats().bits();
const LOG_NEAR_ONE: u128 = LogfOptions::new().near_one_range().bits();
const LOG_NEAR_ONE_ASSUME: u128 = LogfOptions::new()
    .near_one_range()
    .assume_in_bounds()
    .bits();

// Dedicated Log1pf module: GreaterThanNegOne vs AroundZero ([-0.25, 0.25])
const LOG1P_DEFAULT: u128 = Log1pfOptions::new().bits();
const LOG1P_AROUND_ZERO: u128 = Log1pfOptions::new()
    .domain(Log1pfDomain::AroundZero)
    .bits();
const LOG1P_AROUND_ZERO_ASSUME: u128 = Log1pfOptions::new()
    .domain(Log1pfDomain::AroundZero)
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

// Cbrtf: SignedUnit opt-in subnormals vs FlushOrClamp vs Normalize vs AllFloats
const CBRT_DEFAULT: u128 = CbrtfOptions::new().bits();
const CBRT_STRICT_NORMAL: u128 = CbrtfOptions::new().input_range(0.01, 4.0).bits();
const CBRT_SIGNED_UNIT_REJECT: u128 = CbrtfOptions::new().domain(CbrtfDomain::SignedUnit).bits();
const CBRT_SIGNED_UNIT_FLUSH: u128 = CbrtfOptions::new()
    .domain(CbrtfDomain::SignedUnit)
    .flush_or_clamp_subnormals()
    .bits();
const CBRT_SIGNED_UNIT_NORM: u128 = CbrtfOptions::new()
    .domain(CbrtfDomain::SignedUnit)
    .admit_subnormals()
    .bits();
const CBRT_SIGNED_UNIT_ASSUME: u128 = CbrtfOptions::new()
    .domain(CbrtfDomain::SignedUnit)
    .assume_in_bounds()
    .bits();
const CBRT_FULL: u128 = CbrtfOptions::new().admit_all_floats().bits();

// Powf domain granularity, const_exponent (algebraic & general) & Subnormals::FlushOrClamp
const POW_GAMMA_ERR: u128 = PowfOptions::new().gamma_domain().bits();
const POW_GAMMA_FLUSH_SUB: u128 = PowfOptions::new()
    .gamma_domain()
    .flush_or_clamp_subnormals()
    .bits();
const POW_GAMMA_CLAMP: u128 = PowfOptions::new()
    .gamma_domain()
    .clamp_out_of_bounds()
    .bits();
const POW_GAMMA_ASSUME: u128 = PowfOptions::new().gamma_domain().assume_in_bounds().bits();
const POW_SRGB_ENCODE: u128 = PowfOptions::new().srgb_encode_domain().bits();
const POW_CONST_INV_24: u128 = PowfOptions::new()
    .domain(PowfDomain::Unit)
    .const_exponent(1.0 / 2.4)
    .bits();
const POW_CONST_INV_24_ASSUME: u128 = PowfOptions::new()
    .domain(PowfDomain::Unit)
    .const_exponent(1.0 / 2.4)
    .assume_in_bounds()
    .bits();
const POW_CONST_SQRT: u128 = PowfOptions::new()
    .domain(PowfDomain::Unit)
    .const_exponent(0.5)
    .bits();
const POW_CONST_SQUARE: u128 = PowfOptions::new()
    .domain(PowfDomain::Unit)
    .const_exponent(2.0)
    .bits();
const POW_ZERO_ONLY: u128 = PowfOptions::new().admit_zero_base().bits();
const POW_ZERO_AND_SUB: u128 = PowfOptions::new()
    .admit_zero_base()
    .admit_subnormals()
    .bits();

fn bench_granularity(c: &mut Criterion) {
    let len = 4096usize;
    let near_one: Vec<f32> = (0..len)
        .map(|i| 0.75 + 0.5 * (i as f32) / (len as f32))
        .collect();
    let unit_pos: Vec<f32> = (0..len)
        .map(|i| 0.01 + 0.98 * (i as f32) / (len as f32))
        .collect();
    let small_sym: Vec<f32> = (0..len)
        .map(|i| -0.45 + 0.9 * (i as f32) / (len as f32))
        .collect();
    let around_zero: Vec<f32> = (0..len)
        .map(|i| -0.20 + 0.40 * (i as f32) / (len as f32))
        .collect();
    let mut dst = vec![0.0_f32; len];

    let mut g_log = c.benchmark_group("log2_4096");
    g_log.bench_function("default_normals", |b| {
        b.iter(|| Logf::<LOG_DEFAULT>::log2_slice(black_box(&near_one), &mut dst).unwrap())
    });
    g_log.bench_function("admit_zero", |b| {
        b.iter(|| Logf::<LOG_ZERO>::log2_slice(black_box(&near_one), &mut dst).unwrap())
    });
    g_log.bench_function("zero_flush_subnormals", |b| {
        b.iter(|| Logf::<LOG_FLUSH_SUB>::log2_slice(black_box(&near_one), &mut dst).unwrap())
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

    let mut g_l1p = c.benchmark_group("log1p_4096");
    g_l1p.bench_function("default_gt_neg_one", |b| {
        b.iter(|| Log1pf::<LOG1P_DEFAULT>::slice(black_box(&around_zero), &mut dst).unwrap())
    });
    g_l1p.bench_function("around_zero_checked", |b| {
        b.iter(|| Log1pf::<LOG1P_AROUND_ZERO>::slice(black_box(&around_zero), &mut dst).unwrap())
    });
    g_l1p.bench_function("around_zero_assume", |b| {
        b.iter(|| {
            Log1pf::<LOG1P_AROUND_ZERO_ASSUME>::slice(black_box(&around_zero), &mut dst).unwrap()
        })
    });
    g_l1p.finish();

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
    g_cbrt.bench_function("signed_unit_reject_subnormals", |b| {
        b.iter(|| Cbrtf::<CBRT_SIGNED_UNIT_REJECT>::slice(black_box(&unit_pos), &mut dst).unwrap())
    });
    g_cbrt.bench_function("signed_unit_flush_subnormals", |b| {
        b.iter(|| Cbrtf::<CBRT_SIGNED_UNIT_FLUSH>::slice(black_box(&unit_pos), &mut dst).unwrap())
    });
    g_cbrt.bench_function("signed_unit_admit_subnormals", |b| {
        b.iter(|| Cbrtf::<CBRT_SIGNED_UNIT_NORM>::slice(black_box(&unit_pos), &mut dst).unwrap())
    });
    g_cbrt.bench_function("signed_unit_assume_in_bounds", |b| {
        b.iter(|| Cbrtf::<CBRT_SIGNED_UNIT_ASSUME>::slice(black_box(&unit_pos), &mut dst).unwrap())
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
    g_pow.bench_function("gamma_flush_subnormals", |b| {
        b.iter(|| {
            Powf::<POW_GAMMA_FLUSH_SUB>::slice(black_box(&unit_pos), &mut dst, black_box(1.0 / 2.4))
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
    g_pow.bench_function("srgb_encode_runtime_exp", |b| {
        b.iter(|| {
            Powf::<POW_SRGB_ENCODE>::slice(black_box(&unit_pos), &mut dst, black_box(1.0 / 2.4))
                .unwrap()
        })
    });
    g_pow.bench_function("unit_const_exp_inv_24", |b| {
        b.iter(|| Powf::<POW_CONST_INV_24>::slice_const(black_box(&unit_pos), &mut dst).unwrap())
    });
    g_pow.bench_function("unit_const_exp_inv_24_assume", |b| {
        b.iter(|| {
            Powf::<POW_CONST_INV_24_ASSUME>::slice_const(black_box(&unit_pos), &mut dst).unwrap()
        })
    });
    g_pow.bench_function("unit_const_exp_0_5_sqrt", |b| {
        b.iter(|| Powf::<POW_CONST_SQRT>::slice_const(black_box(&unit_pos), &mut dst).unwrap())
    });
    g_pow.bench_function("unit_const_exp_2_0_mul", |b| {
        b.iter(|| Powf::<POW_CONST_SQUARE>::slice_const(black_box(&unit_pos), &mut dst).unwrap())
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

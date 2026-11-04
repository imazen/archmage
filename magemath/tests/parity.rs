//! Every compiled tier against the `zenmath::scalar` mirrors, with no tolerance.
//!
//! `portable` promises the same bits on every tier, so a difference of one ULP
//! on any input is a failure. Only NaN results are compared loosely, by
//! NaN-ness, because payload and sign are unspecified.

mod common;

use archmage::prelude::*;
use common::*;
use magemath::portable::kernels;
use magetypes::simd::backends::F32x8Convert;
use magetypes::simd::generic::f32x8;

/// Lanes differ within each vector, so a kernel that mixed lanes would fail too.
fn check_array_tier<T: IntoConcreteToken + Copy>(tier: &str, token: T) {
    let input = corpus();
    for op in OPS {
        for chunk in input.chunks(8) {
            let mut a = [0.0f32; 8];
            a[..chunk.len()].copy_from_slice(chunk);
            let got = array_on(token, op.name, a);
            for (&g, &x) in got.iter().zip(&a) {
                assert_same(tier, op.name, x, g, (op.mirror)(x));
            }
        }
    }
}

/// Lengths around each chunk boundary: the tail must match the whole-vector path.
fn check_slice_tier<T: IntoConcreteToken + Copy>(tier: &str, token: T) {
    let input = corpus();
    for op in OPS {
        let want: Vec<f32> = input.iter().map(|&x| (op.mirror)(x)).collect();
        let mut got = input.clone();
        slice_on(token, op.name, &mut got);
        for ((&g, &w), &x) in got.iter().zip(&want).zip(&input) {
            assert_same(tier, op.name, x, g, w);
        }
        // Start inside the corpus's dense region, so short slices see varied values.
        let start = input.len() - (1 << 15);
        for len in (0..=33).chain([63, 64, 65, 100, 1000]) {
            let src = &input[start..start + len];
            let mut got = src.to_vec();
            slice_on(token, op.name, &mut got);
            for (i, (&g, &x)) in got.iter().zip(src).enumerate() {
                let what = format!("{tier} len={len} [{i}]");
                assert_same(&what, op.name, x, g, (op.mirror)(x));
            }
        }
    }
}

/// In its stated domain, each `_unchecked` kernel returns the mirror's bits.
fn check_unchecked_tier<T: F32x8Convert>(tier: &str, t: T) {
    use zenmath::scalar as zs;
    fn run<T: F32x8Convert>(
        tier: &str,
        name: &str,
        t: T,
        input: &[f32],
        kernel: impl Fn(f32x8<T>) -> f32x8<T>,
        mirror: impl Fn(f32) -> f32,
    ) {
        for chunk in input.as_chunks::<8>().0 {
            let got = kernel(f32x8::from_array_t(t, *chunk)).to_array();
            for (&g, &x) in got.iter().zip(chunk) {
                assert_same(tier, name, x, g, mirror(x));
            }
        }
    }
    let in_domain = |lo: f32, hi: f32| -> Vec<f32> {
        let mut v: Vec<f32> = corpus()
            .into_iter()
            .filter(|x| *x >= lo && *x < hi)
            .collect();
        v.extend([lo, hi.next_down()]);
        v
    };
    let lg = in_domain(f32::MIN_POSITIVE, f32::INFINITY);
    let low = in_domain(-126.0, 127.99);
    let mid = in_domain(-126.0, 128.0);
    let exp_low = in_domain(-87.0, 88.5);
    let exp_mid = in_domain(-87.0, 88.5);
    let pow = in_domain(1e-8, 1e8);
    run(
        tier,
        "log2_lowp",
        t,
        &lg,
        |v| kernels::log2_lowp_unchecked(t, v),
        zs::log2_lowp_f32,
    );
    run(
        tier,
        "ln_lowp",
        t,
        &lg,
        |v| kernels::ln_lowp_unchecked(t, v),
        zs::ln_lowp_f32,
    );
    run(
        tier,
        "log10_lowp",
        t,
        &lg,
        |v| kernels::log10_lowp_unchecked(t, v),
        zs::log10_lowp_f32,
    );
    run(
        tier,
        "exp2_lowp",
        t,
        &low,
        |v| kernels::exp2_lowp_unchecked(t, v),
        zs::exp2_lowp_f32,
    );
    run(
        tier,
        "exp_lowp",
        t,
        &exp_low,
        |v| kernels::exp_lowp_unchecked(t, v),
        zs::exp_lowp_f32,
    );
    run(
        tier,
        "pow_lowp",
        t,
        &pow,
        |v| kernels::pow_lowp_unchecked(t, v, 2.4),
        |x| zs::pow_lowp_f32(x, 2.4),
    );
    run(
        tier,
        "log2_midp",
        t,
        &lg,
        |v| kernels::log2_midp_unchecked(t, v),
        zs::log2_midp_f32,
    );
    run(
        tier,
        "ln_midp",
        t,
        &lg,
        |v| kernels::ln_midp_unchecked(t, v),
        zs::ln_midp_f32,
    );
    run(
        tier,
        "log10_midp",
        t,
        &lg,
        |v| kernels::log10_midp_unchecked(t, v),
        zs::log10_midp_f32,
    );
    run(
        tier,
        "exp2_midp",
        t,
        &mid,
        |v| kernels::exp2_midp_unchecked(t, v),
        zs::exp2_midp_f32,
    );
    run(
        tier,
        "exp_midp",
        t,
        &exp_mid,
        |v| kernels::exp_midp_unchecked(t, v),
        zs::exp_midp_f32,
    );
    run(
        tier,
        "pow_midp",
        t,
        &pow,
        |v| kernels::pow_midp_unchecked(t, v, 2.4),
        |x| zs::pow_midp_f32(x, 2.4),
    );
}

macro_rules! tier_tests {
    ($module:ident, $tier:literal, $token:ty) => {
        mod $module {
            use super::*;

            #[test]
            fn arrays_match_mirrors() {
                if let Some(token) = require::<$token>($tier) {
                    check_array_tier($tier, token);
                }
            }

            #[test]
            fn slices_match_mirrors() {
                if let Some(token) = require::<$token>($tier) {
                    check_slice_tier($tier, token);
                }
            }

            #[test]
            fn unchecked_kernels_match_mirrors_in_domain() {
                if let Some(token) = require::<$token>($tier) {
                    check_unchecked_tier($tier, token);
                }
            }
        }
    };
}

tier_tests!(scalar, "scalar", ScalarToken);
#[cfg(target_arch = "x86_64")]
tier_tests!(v3, "v3", X64V3Token);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
tier_tests!(v4x, "v4x", X64V4xToken);
#[cfg(target_arch = "aarch64")]
tier_tests!(neon, "neon", NeonToken);
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
tier_tests!(wasm128, "wasm128", Wasm128Token);

#[test]
fn dispatchers_match_mirrors() {
    let input = corpus();
    for op in OPS {
        let mut got = input.clone();
        slice_dispatched(op.name, &mut got);
        for (&g, &x) in got.iter().zip(&input) {
            assert_same("slice", op.name, x, g, (op.mirror)(x));
        }
        for a in input.as_chunks::<8>().0.iter().take(1 << 10).copied() {
            for (&g, &x) in array_dispatched(op.name, a).iter().zip(&a) {
                assert_same("array", op.name, x, g, (op.mirror)(x));
            }
        }
    }
}

/// Runs both forms on `input` a vector at a time and counts how many results
/// differ, among the lanes where `keep` holds.
fn count_differences<T: F32x8Convert>(
    token: T,
    input: &[f32],
    keep: impl Fn(f32) -> bool,
    portable: impl Fn(f32x8<T>) -> f32x8<T>,
    relaxed: impl Fn(f32x8<T>) -> f32x8<T>,
) -> usize {
    let mut different = 0;
    for chunk in input.as_chunks::<8>().0 {
        let v = f32x8::from_array_t(token, *chunk);
        let (p, r) = (portable(v).to_array(), relaxed(v).to_array());
        for i in 0..8 {
            if keep(chunk[i]) && !same_bits(p[i], r[i]) {
                different += 1;
            }
        }
    }
    different
}

/// Count, per magetypes method, how many in-domain results differ from the
/// portable kernel with the same coefficients.
fn relaxed_differences<T: F32x8Convert>(token: T) -> Vec<(&'static str, usize)> {
    let input = corpus();
    let positive = |x: f32| x.is_finite() && x >= f32::MIN_POSITIVE;
    let not_nan = |x: f32| !x.is_nan();
    let t = token;
    vec![
        (
            "log2_lowp",
            count_differences(
                t,
                &input,
                positive,
                |v| kernels::log2_lowp(t, v),
                |v| v.log2_lowp(),
            ),
        ),
        (
            "exp2_lowp",
            count_differences(
                t,
                &input,
                not_nan,
                |v| kernels::exp2_lowp(t, v),
                |v| v.exp2_lowp(),
            ),
        ),
        (
            "log2_midp",
            count_differences(
                t,
                &input,
                positive,
                |v| kernels::log2_midp(t, v),
                |v| v.log2_midp(),
            ),
        ),
        (
            "exp2_midp",
            count_differences(
                t,
                &input,
                not_nan,
                |v| kernels::exp2_midp(t, v),
                |v| v.exp2_midp(),
            ),
        ),
        (
            "ln_midp",
            count_differences(
                t,
                &input,
                positive,
                |v| kernels::ln_midp(t, v),
                |v| v.ln_midp(),
            ),
        ),
        (
            "exp_midp",
            count_differences(
                t,
                &input,
                not_nan,
                |v| kernels::exp_midp(t, v),
                |v| v.exp_midp(),
            ),
        ),
        (
            "log10_midp",
            count_differences(
                t,
                &input,
                positive,
                |v| kernels::log10_midp(t, v),
                |v| v.log10_midp(),
            ),
        ),
        (
            "pow_midp",
            count_differences(
                t,
                &input,
                positive,
                |v| kernels::pow_midp(t, v, 2.4),
                |v| v.pow_midp(2.4),
            ),
        ),
        (
            "cbrt_midp",
            count_differences(
                t,
                &input,
                not_nan,
                |v| kernels::cbrt_midp(t, v),
                |v| v.cbrt_midp(),
            ),
        ),
        (
            "sigmoid_midp",
            count_differences(
                t,
                &input,
                not_nan,
                |v| kernels::sigmoid_midp(t, v),
                |v| v.sigmoid_midp(),
            ),
        ),
    ]
}

/// Where the hardware fuses, the kernels here must equal the generated magetypes
/// methods: this is what stops the two copies of each polynomial drifting apart.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn kernels_equal_magetypes_where_hardware_fuses() {
    #[cfg(target_arch = "x86_64")]
    let token = require::<X64V3Token>("v3");
    #[cfg(target_arch = "aarch64")]
    let token = require::<NeonToken>("neon");
    if let Some(token) = token {
        for (name, different) in relaxed_differences(token) {
            assert_eq!(different, 0, "{name}: {different} in-domain results differ");
        }
    }
}

/// Positive control. magetypes' relaxed kernels round twice on the scalar tier,
/// so they must differ from the fused kernels there. If they ever stop differing,
/// either the corpus has lost the inputs that show it or the kernels here stopped
/// fusing, and the mirror comparisons need another look.
#[test]
fn relaxed_kernels_differ_on_the_scalar_tier() {
    let differences = relaxed_differences(ScalarToken);
    for name in [
        "log2_midp",
        "exp2_midp",
        "ln_midp",
        "exp_midp",
        "log10_midp",
        "pow_midp",
    ] {
        let (_, different) = differences.iter().find(|(n, _)| *n == name).unwrap();
        assert!(
            *different > 0,
            "{name}: scalar relaxed and fused no longer differ"
        );
    }
    // cbrt has no multiply-add, so the tier makes no difference to it.
    let (_, cbrt) = differences.iter().find(|(n, _)| *n == "cbrt_midp").unwrap();
    assert_eq!(*cbrt, 0);
}

/// One `#[attune]` body chaining kernels: same bits as the composed mirrors.
#[attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn gamma_roundtrip(token: Token, a: [f32; 8]) -> [f32; 8] {
    let x = f32x8::from_array_t(token, a);
    let expanded = kernels::pow_midp(token, x, 2.4);
    kernels::pow_midp(token, expanded, 1.0 / 2.4).to_array()
}

#[test]
fn composed_kernels_match_composed_mirrors() {
    use zenmath::scalar::pow_midp_f32;
    let input = sweep_value(0.0, 1.0, 1 << 12);
    for &a in input.as_chunks::<8>().0 {
        let got = attuned!(
            gamma_roundtrip(a),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
        for (&g, &x) in got.iter().zip(&a) {
            let want = pow_midp_f32(pow_midp_f32(x, 2.4), 1.0 / 2.4);
            assert_same("composed", "gamma", x, g, want);
        }
    }
}

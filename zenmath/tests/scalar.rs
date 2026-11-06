//! Scalar core tests — bit-exactness against `std` where claimed,
//! tolerance checks for approximation tiers.

use zenmath::scalar::*;

fn bits_f32(x: f32) -> u32 {
    x.to_bits()
}

#[test]
fn sqrtf_bit_exact_vs_std() {
    let cases = [
        0.0f32,
        -0.0,
        1.0,
        2.0,
        0.5,
        1e-30,
        1e30,
        f32::from_bits(1),           // min subnormal
        f32::from_bits(0x007f_ffff), // max subnormal
        8388608.0,
        core::f32::consts::PI,
        f32::INFINITY,
        f32::NAN,
        -1.0,
    ];
    for &x in &cases {
        let a = sqrtf(x);
        let b = x.sqrt();
        // NaN outputs only need NaN-ness: payloads/signs may differ per the
        // documented contract (hardware sqrtss returns -NaN for negatives;
        // the software path returns the canonical qNaN).
        if x.is_nan() || x < 0.0 || x == f32::NEG_INFINITY {
            assert!(a.is_nan() && b.is_nan(), "sqrtf({x}) should be NaN");
        } else {
            assert_eq!(bits_f32(a), bits_f32(b), "sqrtf({x}) bits differ from std");
        }
    }
    // Sweep positive normals.
    let mut x = f32::from_bits(0x0080_0001);
    for _ in 0..200_000 {
        assert_eq!(
            bits_f32(sqrtf(x)),
            bits_f32(x.sqrt()),
            "sqrtf({x:e}) mismatch"
        );
        x = f32::from_bits(x.to_bits() + 7919);
    }
}

#[test]
fn sqrt_bit_exact_vs_std() {
    for &x in &[
        0.0f64,
        -0.0,
        1.0,
        2.0,
        0.1,
        1e-300,
        1e300,
        f64::NAN,
        f64::INFINITY,
        -4.0,
    ] {
        let a = sqrt(x);
        let b = x.sqrt();
        if x.is_nan() || x < 0.0 {
            assert!(a.is_nan() && b.is_nan(), "sqrt({x}) should be NaN");
        } else {
            assert_eq!(a.to_bits(), b.to_bits(), "sqrt({x}) bits differ");
        }
    }
}

#[test]
fn rounding_bit_exact_vs_std() {
    let cases = [
        0.0f32,
        -0.0,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        0.499_999_97,
        3.5,
        -3.5,
        8388607.0,
        -8388607.0,
        1e10,
        -1e10,
        f32::NAN,
        f32::INFINITY,
    ];
    for &x in &cases {
        assert_eq!(floorf(x).to_bits(), x.floor().to_bits(), "floorf({x})");
        assert_eq!(ceilf(x).to_bits(), x.ceil().to_bits(), "ceilf({x})");
        assert_eq!(roundf(x).to_bits(), x.round().to_bits(), "roundf({x})");
        assert_eq!(
            roundevenf(x).to_bits(),
            x.round_ties_even().to_bits(),
            "roundevenf({x})"
        );
    }
    for &x in &[0.0f64, -0.0, 0.5, -0.5, 2.5, -2.5, 1e15, -1e15, f64::NAN] {
        assert_eq!(floor(x).to_bits(), x.floor().to_bits(), "floor({x})");
        assert_eq!(ceil(x).to_bits(), x.ceil().to_bits(), "ceil({x})");
        assert_eq!(round(x).to_bits(), x.round().to_bits(), "round({x})");
        assert_eq!(
            roundeven(x).to_bits(),
            x.round_ties_even().to_bits(),
            "roundeven({x})"
        );
    }
}

#[test]
fn lowp_transcendentals_reasonable() {
    // log2/exp2 round-trip within lowp tolerance (~1%).
    for &x in &[0.5f64, 1.0, 1.5, 2.0, 3.7, 100.0, 1e-3, 1e3] {
        let rt = exp2_lowp_f64(log2_lowp_f64(x));
        let rel = ((rt - x) / x).abs();
        assert!(rel < 0.02, "exp2(log2({x})) rel err {rel}");
        let want = x.log2();
        assert!(
            (log2_lowp_f64(x) - want).abs() < 0.02,
            "log2_lowp_f64({x}) vs std"
        );
    }
    // exp/ln consistency.
    for &x in &[-5.0f64, -1.0, 0.0, 0.7, 3.0] {
        let e = exp_lowp_f64(x);
        assert!(
            (e - x.exp()).abs() / x.exp().abs() < 0.02,
            "exp_lowp_f64({x})"
        );
        assert!(
            (ln_lowp_f64(e) - x).abs() < 0.05,
            "ln_lowp_f64(exp_lowp_f64({x}))"
        );
    }
    // Edge cases.
    assert_eq!(log2_lowp_f64(0.0), f64::NEG_INFINITY);
    assert!(log2_lowp_f64(-1.0).is_nan());
    assert_eq!(log2_lowp_f64(f64::INFINITY), f64::INFINITY);
    assert_eq!(exp2_lowp_f64(2000.0), f64::INFINITY);
    assert_eq!(exp2_lowp_f64(-2000.0), 0.0);
    assert!(pow_lowp_f64(-1.0, 0.5).is_nan() || pow_lowp_f64(-1.0, 0.5) >= 0.0);
}

#[test]
fn midp_f32_mirrors_within_contract() {
    // The f32 midp mirrors must land within magetypes' documented midp
    // error bounds vs an f64 reference, and round-trip sanely.
    for &x in &[0.5f32, 1.0, 1.5, 2.0, 3.7, 100.0, 0.001, 1e3, 0.25] {
        let l = log2_midp_f32(x);
        assert!((l - x.log2()).abs() < 1e-4, "log2_midp_f32({x}) vs std");
        let e = exp2_midp_f32(l);
        let rel = ((e - x) / x).abs();
        assert!(rel < 1e-4, "exp2(log2({x})) rel err {rel}");
        let n = ln_midp_f32(x);
        assert!((n - x.ln()).abs() < 1e-4, "ln_midp_f32({x}) vs std");
        let e2 = exp_midp_f32(n);
        let rel2 = ((e2 - x) / x).abs();
        assert!(rel2 < 1e-3, "exp(ln({x})) rel err {rel2}");
    }
    // Edge cases match the vector kernels' blend order.
    assert_eq!(log2_midp_f32(0.0), f32::NEG_INFINITY);
    assert!(log2_midp_f32(-1.0).is_nan());
    assert_eq!(log2_midp_f32(f32::INFINITY), f32::INFINITY);
    assert_eq!(exp2_midp_f32(-200.0), 0.0);
    assert_eq!(exp2_midp_f32(200.0), f32::INFINITY);
}

#[test]
fn nan_input_returns_nan() {
    // transcendentals.md's rule for this layer. magetypes leaves it to min/max
    // and bit tricks, which give a finite answer on some or all backends.
    for f in [
        log2_lowp_f32 as fn(f32) -> f32,
        exp2_lowp_f32,
        ln_lowp_f32,
        exp_lowp_f32,
        log10_lowp_f32,
        cbrt_lowp_f32,
        log2_midp_f32,
        exp2_midp_f32,
        ln_midp_f32,
        exp_midp_f32,
        log10_midp_f32,
        cbrt_midp_f32,
        cbrt_midp_precise_f32,
        sigmoid_midp_f32,
        silu_midp_f32,
    ] {
        assert!(f(f32::NAN).is_nan());
        assert!(f(-f32::NAN).is_nan());
    }
    for n in [0.0f32, 1.0, 2.4, -1.0] {
        assert!(pow_midp_f32(f32::NAN, n).is_nan());
        assert!(pow_lowp_f32(f32::NAN, n).is_nan());
    }
}

#[test]
fn log_specials_on_every_precision() {
    for f in [log2_lowp_f32 as fn(f32) -> f32, log2_midp_f32] {
        assert_eq!(f(0.0), f32::NEG_INFINITY);
        assert_eq!(f(-0.0), f32::NEG_INFINITY);
        assert!(f(-1.0).is_nan());
        assert!(f(f32::NEG_INFINITY).is_nan());
        assert_eq!(f(f32::INFINITY), f32::INFINITY);
        assert!(f(1.0).abs() < 2e-6);
    }
}

#[test]
fn cbrt_precise_covers_the_range() {
    assert_eq!(cbrt_midp_precise_f32(f32::INFINITY), f32::INFINITY);
    assert_eq!(cbrt_midp_precise_f32(f32::NEG_INFINITY), f32::NEG_INFINITY);
    assert_eq!(cbrt_midp_precise_f32(-0.0).to_bits(), (-0.0f32).to_bits());
    for x in [f32::MAX, 1e36, f32::MIN_POSITIVE, f32::from_bits(1), 1e-42] {
        let want = (x as f64).cbrt();
        let rel = ((cbrt_midp_precise_f32(x) as f64 - want) / want).abs();
        assert!(rel < 5e-7, "cbrt_midp_precise_f32({x:e}) rel err {rel:e}");
    }
}

#[test]
fn sigmoid_rails_are_exact() {
    assert_eq!(sigmoid_midp_f32(100.0), 1.0);
    assert_eq!(sigmoid_midp_f32(-100.0), 0.0);
    assert_eq!(sigmoid_midp_f32(0.0), 0.5);
    assert_eq!(silu_midp_f32(100.0), 100.0);
    assert_eq!(silu_midp_f32(0.0), 0.0);
}

#[test]
fn fmaf_bit_exact_vs_mul_add() {
    // `fmaf` is correctly rounded on every target — bit-identical to
    // `f32::mul_add` (hardware FMA or libm) including subnormals and
    // halfway cases that exercise the software path's exact branch.
    let mut seed = 0x1234_5678u32;
    let mut rng = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        f32::from_bits(seed)
    };
    for _ in 0..500_000 {
        let (a, b, c) = (rng(), rng(), rng());
        let got = fmaf(a, b, c);
        let want = a.mul_add(b, c);
        if want.is_nan() {
            assert!(got.is_nan(), "fmaf({a:e},{b:e},{c:e}) want NaN");
        } else {
            assert_eq!(got.to_bits(), want.to_bits(), "fmaf({a:e},{b:e},{c:e})");
        }
    }
    // Targeted halfway/subnormal cases.
    for &(a, b, c) in &[
        (1.0f32, 1.0, f32::from_bits(0x3380_0001)),
        (f32::MIN_POSITIVE, f32::MIN_POSITIVE, -f32::MIN_POSITIVE),
        (1e-30, 1e-30, 1e-30),
        (3.402_823_5e38, 2.0, -3.402_823_5e38),
        (0.1, 0.2, 0.3),
        (f32::NAN, 1.0, 1.0),
        (f32::INFINITY, 0.0, 1.0),
    ] {
        let got = fmaf(a, b, c);
        let want = a.mul_add(b, c);
        if want.is_nan() {
            assert!(got.is_nan());
        } else {
            assert_eq!(got.to_bits(), want.to_bits(), "fmaf({a:e},{b:e},{c:e})");
        }
    }
}

#[test]
fn cbrt_within_documented_error() {
    for &x in &[
        0.0f32,
        1.0,
        -1.0,
        8.0,
        -27.0,
        0.001,
        1e10,
        -1e-10,
        core::f32::consts::PI,
    ] {
        let a = cbrt_midp_f32(x);
        let want = x.cbrt();
        if x == 0.0 {
            assert_eq!(a.to_bits(), x.to_bits());
            continue;
        }
        let rel = ((a - want) / want).abs();
        assert!(rel < 1e-5, "cbrt_midp_f32({x}) rel err {rel}");
        let b = cbrt_lowp_f32(x);
        let relb = ((b - want) / want).abs();
        assert!(relb < 5e-4, "cbrt_lowp_f32({x}) rel err {relb}");
    }
}

// The two-rounding mirrors retain the same special-value contracts.
mod unfused_specials {
    use zenmath::scalar::unfused::*;

    #[test]
    fn nan_input_returns_nan() {
        // transcendentals.md's rule for this layer. magetypes leaves it to min/max
        // and bit tricks, which give a finite answer on some or all backends.
        for f in [
            log2_lowp_f32 as fn(f32) -> f32,
            exp2_lowp_f32,
            ln_lowp_f32,
            exp_lowp_f32,
            log10_lowp_f32,
            cbrt_lowp_f32,
            log2_midp_f32,
            exp2_midp_f32,
            ln_midp_f32,
            exp_midp_f32,
            log10_midp_f32,
            cbrt_midp_f32,
            cbrt_midp_precise_f32,
            sigmoid_midp_f32,
            silu_midp_f32,
        ] {
            assert!(f(f32::NAN).is_nan());
            assert!(f(-f32::NAN).is_nan());
        }
        for n in [0.0f32, 1.0, 2.4, -1.0] {
            assert!(pow_midp_f32(f32::NAN, n).is_nan());
            assert!(pow_lowp_f32(f32::NAN, n).is_nan());
        }
    }

    #[test]
    fn log_specials_on_every_precision() {
        for f in [log2_lowp_f32 as fn(f32) -> f32, log2_midp_f32] {
            assert_eq!(f(0.0), f32::NEG_INFINITY);
            assert_eq!(f(-0.0), f32::NEG_INFINITY);
            assert!(f(-1.0).is_nan());
            assert!(f(f32::NEG_INFINITY).is_nan());
            assert_eq!(f(f32::INFINITY), f32::INFINITY);
            assert!(f(1.0).abs() < 2e-6);
        }
    }

    #[test]
    fn cbrt_precise_covers_the_range() {
        assert_eq!(cbrt_midp_precise_f32(f32::INFINITY), f32::INFINITY);
        assert_eq!(cbrt_midp_precise_f32(f32::NEG_INFINITY), f32::NEG_INFINITY);
        assert_eq!(cbrt_midp_precise_f32(-0.0).to_bits(), (-0.0f32).to_bits());
        for x in [f32::MAX, 1e36, f32::MIN_POSITIVE, f32::from_bits(1), 1e-42] {
            let want = (x as f64).cbrt();
            let rel = ((cbrt_midp_precise_f32(x) as f64 - want) / want).abs();
            assert!(rel < 5e-7, "cbrt_midp_precise_f32({x:e}) rel err {rel:e}");
        }
    }

    #[test]
    fn sigmoid_rails_are_exact() {
        assert_eq!(sigmoid_midp_f32(100.0), 1.0);
        assert_eq!(sigmoid_midp_f32(-100.0), 0.0);
        assert_eq!(sigmoid_midp_f32(0.0), 0.5);
        assert_eq!(silu_midp_f32(100.0), 100.0);
        assert_eq!(silu_midp_f32(0.0), 0.0);
    }
}

#[test]
fn fmaf_independent_rounding_edges() {
    // Expected bits from the exact-rational corpus in
    // magetypes/tests/common/fma_vectors.rs, not the platform std/libm.
    // The first case is the subnormal counterexample documented at
    // https://shnatsel.github.io/implementing-fma-finding-bugs-in-std/
    for [a, b, c, expected] in [
        [0x9700_0800, 0x1cff_f001, 0x0001_0002, 0x0001_0001],
        [0x1700_0800, 0x1cff_f001, 0x8001_0002, 0x8001_0001],
        [0x8000_0001, 0x0000_0001, 0x0000_0000, 0x8000_0000],
        [0x8000_0001, 0x3f00_0000, 0x0000_0000, 0x8000_0000],
        [0xbf80_0001, 0x3f80_0001, 0x3f80_0002, 0xa880_0000],
        [0xff7f_ffff, 0x4000_0000, 0x7f7f_ffff, 0xff7f_ffff],
    ] {
        let result = fmaf(f32::from_bits(a), f32::from_bits(b), f32::from_bits(c));
        assert_eq!(
            result.to_bits(),
            expected,
            "fmaf({a:08x}, {b:08x}, {c:08x})"
        );
    }
}

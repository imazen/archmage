//! Accuracy against an f64 `std` reference, per the bands and maxima published
//! in `docs/transcendentals.md`.
//!
//! The method matches `benchmarks/transcendental_precision_2026-10-05.md`: the
//! error of `y` against reference `r` is `|y - r| / ulp(r as f32)`, skipping
//! inputs whose reference is zero, subnormal or infinite as an f32. Those figures
//! came from every f32 in each band on both rounding behaviors, so no input in
//! the band may exceed them and the bounds carry no headroom. Samples are spaced
//! in bit space across the whole band; narrow bands are covered exhaustively.
//!
//! Every surface runs every bound: the published figures cover both roundings, so
//! none of them gets a looser one.

mod common;

use common::*;

/// Up to `n` values spaced evenly in bit space across `lo..=hi`, for `lo < hi`.
fn band(lo: f32, hi: f32, n: usize) -> Vec<f32> {
    // Map to a monotonic integer line so a band may straddle zero.
    let line = |x: f32| -> i64 {
        let b = i64::from(x.to_bits() & 0x7fff_ffff);
        if x.is_sign_negative() { -b } else { b }
    };
    let unline = |i: i64| -> f32 {
        let b = i.unsigned_abs() as u32;
        f32::from_bits(if i < 0 { b | 0x8000_0000 } else { b })
    };
    let (a, b) = (line(lo), line(hi));
    let span = (b - a) as u64;
    let step = (span / n as u64).max(1) as i64;
    let mut v: Vec<f32> = (a..=b).step_by(step as usize).map(unline).collect();
    v.push(hi);
    v
}

fn ulp_of(r: f64) -> f64 {
    let a = (r as f32).abs();
    if a == f32::MAX {
        // No finite next value: the ULP below is the same size.
        return f64::from(a) - f64::from(a.next_down());
    }
    f64::from(a.next_up()) - f64::from(a)
}

#[derive(Default)]
struct Errors {
    ulp: f64,
    ulp_at: f32,
    rel: f64,
    rel_at: f32,
    abs: f64,
    abs_at: f32,
}

fn measure(input: &[f32], op: impl Fn(&mut [f32]), reference: impl Fn(f64) -> f64) -> Errors {
    let mut got = input.to_vec();
    op(&mut got);
    let mut e = Errors::default();
    for (&y, &x) in got.iter().zip(input) {
        let r = reference(f64::from(x));
        let rounded = (r as f32).abs();
        if rounded == 0.0 || !rounded.is_normal() {
            continue;
        }
        let (y, abs) = (f64::from(y), (f64::from(y) - r).abs());
        assert!(!y.is_nan(), "NaN at {x:e}, reference {r:e}");
        let ulp = abs / ulp_of(r);
        if ulp > e.ulp {
            (e.ulp, e.ulp_at) = (ulp, x);
        }
        if r != 0.0 && abs / r.abs() > e.rel {
            (e.rel, e.rel_at) = (abs / r.abs(), x);
        }
        if abs > e.abs {
            (e.abs, e.abs_at) = (abs, x);
        }
    }
    e
}

#[track_caller]
fn assert_ulp(name: &str, e: &Errors, max: f64) {
    assert!(
        e.ulp <= max,
        "{name}: {:.3} ULP at {:e} (published {max})",
        e.ulp,
        e.ulp_at
    );
}

#[track_caller]
fn assert_rel(name: &str, e: &Errors, max: f64) {
    assert!(
        e.rel <= max,
        "{name}: relative {:.3e} at {:e} (published {max:e})",
        e.rel,
        e.rel_at
    );
}

#[track_caller]
fn assert_abs(name: &str, e: &Errors, max: f64) {
    assert!(
        e.abs <= max,
        "{name}: absolute {:.3e} at {:e} (published {max:e})",
        e.abs,
        e.abs_at
    );
}

const SAMPLES: usize = 1 << 18;

/// `f32::MAX / 3` rounded down: the published top of the `cbrt` domain.
const CBRT_TOP: f32 = 1.134_274_5e38;

fn positive_normals() -> Vec<f32> {
    band(f32::MIN_POSITIVE, f32::MAX, SAMPLES)
}

macro_rules! accuracy_tests {
    ($module:ident, $common:ident, $surface:ident) => {
        mod $module {
            use super::*;
            use magemath::$surface::slice as sm;

            #[test]
            fn exact_primitives_match_std() {
                let input = corpus();
                for (name, f) in [
                    ("sqrt", f32::sqrt as fn(f32) -> f32),
                    ("floor", f32::floor),
                    ("ceil", f32::ceil),
                    ("roundeven", f32::round_ties_even),
                ] {
                    let mut got = input.clone();
                    $common::slice_dispatched(name, &mut got);
                    for (&g, &x) in got.iter().zip(&input) {
                        assert_same("std", name, x, g, f(x));
                    }
                }
            }

            #[test]
            fn midp_logs() {
                let input = positive_normals();
                let e = measure(&input, sm::log2_midp_f32, f64::log2);
                assert_ulp("log2_midp", &e, 4.5);
                assert_rel("log2_midp", &e, 3.3e-7);
                let e = measure(&input, sm::ln_midp_f32, f64::ln);
                assert_ulp("ln_midp", &e, 4.1);
                assert_rel("ln_midp", &e, 3.5e-7);
                let e = measure(&input, sm::log10_midp_f32, f64::log10);
                assert_ulp("log10_midp", &e, 4.5);
                assert_rel("log10_midp", &e, 4.1e-7);
            }

            #[test]
            fn midp_exp2() {
                let below = 127.5f32.next_down();
                let e = measure(&band(-126.0, below, SAMPLES), sm::exp2_midp_f32, f64::exp2);
                assert_ulp("exp2_midp [-126, 127.5)", &e, 1.9);
                assert_rel("exp2_midp [-126, 127.5)", &e, 1.6e-7);
                let e = measure(
                    &band(127.5, 128f32.next_down(), SAMPLES),
                    sm::exp2_midp_f32,
                    f64::exp2,
                );
                assert_ulp("exp2_midp [127.5, 128)", &e, 134.1);
                assert_rel("exp2_midp [127.5, 128)", &e, 8.0e-6);
            }

            #[test]
            fn midp_exp() {
                let ln_max = f32::MAX.ln();
                for (lo, hi, ulp, rel) in [
                    (-1.0, 1.0, 2.0, 1.7e-7),
                    (-10.0, 10.0, 8.2, 5.9e-7),
                    (-40.0, 40.0, 31.3, 2.0e-6),
                    (-87.0, 88.5, 64.1, 4.4e-6),
                    (88.5f32.next_up(), ln_max, 197.1, 1.2e-5),
                ] {
                    let name = format!("exp_midp [{lo}, {hi}]");
                    let e = measure(&band(lo, hi, SAMPLES), sm::exp_midp_f32, f64::exp);
                    assert_ulp(&name, &e, ulp);
                    assert_rel(&name, &e, rel);
                }
            }

            #[test]
            fn midp_cbrt() {
                let e = measure(
                    &band(f32::MIN_POSITIVE, CBRT_TOP, SAMPLES),
                    sm::cbrt_midp_f32,
                    f64::cbrt,
                );
                assert_ulp("cbrt_midp", &e, 3.2);
                assert_rel("cbrt_midp", &e, 2.5e-7);
                let e = measure(
                    &band(f32::from_bits(1), f32::MAX, SAMPLES),
                    sm::cbrt_midp_precise_f32,
                    f64::cbrt,
                );
                assert_ulp("cbrt_midp_precise", &e, 3.2);
                assert_rel("cbrt_midp_precise", &e, 2.5e-7);
            }

            #[test]
            fn midp_pow() {
                for (exponent, ulp, rel) in
                    [(8, 27.8, 1.7e-6), (20, 64.8, 4.0e-6), (50, 144.9, 8.8e-6)]
                {
                    let (lo, hi) = (2f32.powi(-exponent), 2f32.powi(exponent));
                    let name = format!("pow_midp(2.4) [2^-{exponent}, 2^{exponent}]");
                    let e = measure(
                        &band(lo, hi, SAMPLES),
                        |d| sm::pow_midp_f32(d, 2.4),
                        |x| x.powf(f64::from(2.4f32)),
                    );
                    assert_ulp(&name, &e, ulp);
                    assert_rel(&name, &e, rel);
                }
            }

            #[test]
            fn lowp_exp_family() {
                let e = measure(&band(-126.0, 127.99, SAMPLES), sm::exp2_lowp_f32, f64::exp2);
                assert_rel("exp2_lowp [-126, 127.99]", &e, 5.57e-3);
                let e = measure(
                    &band(127.99f32.next_up(), 128f32.next_down(), SAMPLES),
                    sm::exp2_lowp_f32,
                    f64::exp2,
                );
                assert_rel("exp2_lowp (127.99, 128)", &e, 1.23e-2);
                let e = measure(&band(-87.0, 88.5, SAMPLES), sm::exp_lowp_f32, f64::exp);
                assert_rel("exp_lowp [-87, 88.5]", &e, 5.57e-3);
                let (lo, hi) = (2f32.powi(-20), 2f32.powi(20));
                let e = measure(
                    &band(lo, hi, SAMPLES),
                    |d| sm::pow_lowp_f32(d, 2.4),
                    |x| x.powf(f64::from(2.4f32)),
                );
                assert_rel("pow_lowp(2.4)", &e, 5.57e-3);
            }

            #[test]
            fn lowp_logs_and_cbrt() {
                let input = positive_normals();
                assert_abs(
                    "log2_lowp",
                    &measure(&input, sm::log2_lowp_f32, f64::log2),
                    6.4e-6,
                );
                assert_abs(
                    "ln_lowp",
                    &measure(&input, sm::ln_lowp_f32, f64::ln),
                    8.5e-6,
                );
                assert_abs(
                    "log10_lowp",
                    &measure(&input, sm::log10_lowp_f32, f64::log10),
                    5.8e-6,
                );
                let e = measure(
                    &band(f32::MIN_POSITIVE, CBRT_TOP, SAMPLES),
                    sm::cbrt_lowp_f32,
                    f64::cbrt,
                );
                assert_rel("cbrt_lowp", &e, 2.98e-5);
                assert_ulp("cbrt_lowp", &e, 259.0);
            }

            #[test]
            fn every_nan_input_gives_nan() {
                // The portable contract, on every dispatched op; see docs/transcendentals.md
                // for what magetypes' methods return instead.
                let mut nans: Vec<f32> = vec![f32::NAN, -f32::NAN, f32::from_bits(0x7f80_0001)];
                nans.extend(sweep_bits(1 << 16).into_iter().filter(|x| x.is_nan()));
                for op in $common::OPS {
                    let mut got = nans.clone();
                    $common::slice_dispatched(op.name, &mut got);
                    for (&g, &x) in got.iter().zip(&nans) {
                        assert!(g.is_nan(), "{}({:08x}) = {g:e}", op.name, x.to_bits());
                    }
                }
            }
        }
    };
}

accuracy_tests!(portable, fused, portable);
accuracy_tests!(unfused_kernels, unfused, unfused);
accuracy_tests!(relaxed_kernels, relaxed, relaxed);

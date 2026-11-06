//! Shared test support: the input corpus, the op table, and per-tier evaluation.
#![allow(dead_code)] // Each test binary uses part of this module.

use archmage::prelude::*;

/// One op: its name in the golden files and its scalar mirror.
#[derive(Clone, Copy)]
pub struct Op {
    pub name: &'static str,
    pub mirror: fn(f32) -> f32,
}

macro_rules! ops {
    ($( $name:literal, $array:ident, $slice:ident, $mirror:expr $(, $n:expr)?; )*) => {
        /// Every op, with `pow` listed once per pinned exponent.
        pub const OPS: &[Op] = &[$(Op { name: $name, mirror: $mirror }),*];

        /// Run the `[f32; 8]` op on exactly the tier `token` proves.
        pub fn array_on<T: IntoConcreteToken>(token: T, name: &str, a: [f32; 8]) -> [f32; 8] {
            match name {
                $($name => attuned!(
                    arr::$array(a $(, $n)?),
                    [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar],
                    using(token)
                ),)*
                other => panic!("unknown op {other}"),
            }
        }

        /// Run the slice op on exactly the tier `token` proves.
        pub fn slice_on<T: IntoConcreteToken>(token: T, name: &str, d: &mut [f32]) {
            match name {
                $($name => attuned!(
                    sli::$slice(d $(, $n)?),
                    [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar],
                    using(token)
                ),)*
                other => panic!("unknown op {other}"),
            }
        }

        /// Run the slice op through its runtime dispatcher.
        pub fn slice_dispatched(name: &str, d: &mut [f32]) {
            match name {
                $($name => sli::$slice(d $(, $n)?),)*
                other => panic!("unknown op {other}"),
            }
        }

        /// Run the `[f32; 8]` op through its runtime dispatcher.
        pub fn array_dispatched(name: &str, a: [f32; 8]) -> [f32; 8] {
            match name {
                $($name => arr::$array(a $(, $n)?),)*
                other => panic!("unknown op {other}"),
            }
        }
    };
}

/// The op table over one `magemath` surface and one set of mirrors.
macro_rules! surface {
    ($module:ident, $surface:ident, $($zs:ident)::+) => {
        pub mod $module {
            use super::*;
            use magemath::$surface::{array as arr, slice as sli};
            use $($zs)::+ as zs;

            ops! {
                "sqrt", sqrt_f32x8, sqrt_f32, zs::sqrtf;
                "floor", floor_f32x8, floor_f32, zs::floorf;
                "ceil", ceil_f32x8, ceil_f32, zs::ceilf;
                "roundeven", roundeven_f32x8, roundeven_f32, zs::roundevenf;

                "log2_lowp", log2_lowp_f32x8, log2_lowp_f32, zs::log2_lowp_f32;
                "exp2_lowp", exp2_lowp_f32x8, exp2_lowp_f32, zs::exp2_lowp_f32;
                "ln_lowp", ln_lowp_f32x8, ln_lowp_f32, zs::ln_lowp_f32;
                "exp_lowp", exp_lowp_f32x8, exp_lowp_f32, zs::exp_lowp_f32;
                "log10_lowp", log10_lowp_f32x8, log10_lowp_f32, zs::log10_lowp_f32;
                "pow_lowp(2.4)", pow_lowp_f32x8, pow_lowp_f32, |x| zs::pow_lowp_f32(x, 2.4), 2.4;
                "pow_lowp(-1)", pow_lowp_f32x8, pow_lowp_f32, |x| zs::pow_lowp_f32(x, -1.0), -1.0;
                "pow_lowp(0)", pow_lowp_f32x8, pow_lowp_f32, |x| zs::pow_lowp_f32(x, 0.0), 0.0;
                "cbrt_lowp", cbrt_lowp_f32x8, cbrt_lowp_f32, zs::cbrt_lowp_f32;

                "log2_midp", log2_midp_f32x8, log2_midp_f32, zs::log2_midp_f32;
                "exp2_midp", exp2_midp_f32x8, exp2_midp_f32, zs::exp2_midp_f32;
                "ln_midp", ln_midp_f32x8, ln_midp_f32, zs::ln_midp_f32;
                "exp_midp", exp_midp_f32x8, exp_midp_f32, zs::exp_midp_f32;
                "log10_midp", log10_midp_f32x8, log10_midp_f32, zs::log10_midp_f32;
                "pow_midp(2.4)", pow_midp_f32x8, pow_midp_f32, |x| zs::pow_midp_f32(x, 2.4), 2.4;
                "pow_midp(0.4166667)", pow_midp_f32x8, pow_midp_f32, |x| zs::pow_midp_f32(x, 1.0 / 2.4), 1.0 / 2.4;
                "pow_midp(-1)", pow_midp_f32x8, pow_midp_f32, |x| zs::pow_midp_f32(x, -1.0), -1.0;
                "pow_midp(0)", pow_midp_f32x8, pow_midp_f32, |x| zs::pow_midp_f32(x, 0.0), 0.0;
                "cbrt_midp", cbrt_midp_f32x8, cbrt_midp_f32, zs::cbrt_midp_f32;
                "cbrt_midp_precise", cbrt_midp_precise_f32x8, cbrt_midp_precise_f32, zs::cbrt_midp_precise_f32;
                "sigmoid_midp", sigmoid_midp_f32x8, sigmoid_midp_f32, zs::sigmoid_midp_f32;
                "silu_midp", silu_midp_f32x8, silu_midp_f32, zs::silu_midp_f32;
            }
        }
    };
}

surface!(fused, portable, zenmath::scalar);
surface!(unfused, unfused, zenmath::scalar::unfused);
// `relaxed::OPS` carries the fused mirrors only to fill the table: which mirrors
// `relaxed` must equal depends on the tier. See `Rounding`.
surface!(relaxed, relaxed, zenmath::scalar);

/// How a polynomial step rounds, which decides the mirrors `relaxed` must equal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rounding {
    Once,
    Twice,
}

impl Rounding {
    pub fn ops(self) -> &'static [Op] {
        match self {
            Rounding::Once => fused::OPS,
            Rounding::Twice => unfused::OPS,
        }
    }
}

/// What `mul_add_relaxed` does on `tier`. On relaxed WASM the engine chooses; a
/// wrong answer here cannot pass, since the caller then compares every result.
pub fn relaxed_rounding<T: IntoConcreteToken + Copy>(tier: &str, token: T) -> Rounding {
    match tier {
        "v3" | "v4x" | "neon" => Rounding::Once,
        "wasm128" if cfg!(target_feature = "relaxed-simd") => {
            let input = corpus();
            let mut got = input.clone();
            relaxed::slice_on(token, "exp2_midp", &mut got);
            let fused_bits = got
                .iter()
                .zip(&input)
                .all(|(&g, &x)| same_bits(g, zenmath::scalar::exp2_midp_f32(x)));
            let rounding = if fused_bits {
                Rounding::Once
            } else {
                Rounding::Twice
            };
            if let Ok(expected) = std::env::var("MAGEMATH_EXPECT_RELAXED_ROUNDING") {
                let expected = match expected.as_str() {
                    "once" => Rounding::Once,
                    "twice" => Rounding::Twice,
                    _ => panic!("MAGEMATH_EXPECT_RELAXED_ROUNDING must be once or twice"),
                };
                assert_eq!(rounding, expected, "relaxed WASM engine rounding");
            }
            rounding
        }
        _ => Rounding::Twice,
    }
}

/// The hand-picked inputs whose results `tests/golden/edges.txt` spells out.
pub fn named_edges() -> Vec<f32> {
    vec![
        f32::NAN,
        -f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        1.5,
        2.5,
        f32::from_bits(1),
        f32::from_bits(0x007f_ffff),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        1.0e36,
        f32::MAX / 3.0,
        -127.0,
        -126.0,
        -125.5,
        88.5,
        127.5,
        127.99,
        128.0,
        8_388_608.0,
    ]
}

/// `named_edges` plus every power of two with its neighbors and the
/// neighbors of each band boundary the kernels branch on.
pub fn edges() -> Vec<f32> {
    let mut v = named_edges();
    for bits in (0..=254u32).map(|e| e << 23) {
        for b in [bits.saturating_sub(1), bits, bits + 1] {
            for sign in [0, 0x8000_0000] {
                v.push(f32::from_bits(b | sign));
            }
        }
    }
    let boundaries: [f32; 14] = [
        -126.0,
        -125.5,
        88.5,
        127.0,
        127.5,
        127.99,
        128.0,
        1.0e36,
        3.402_823_5e38 / 3.0,
        0.5,
        1.5,
        2.5,
        8_388_607.5,
        8_388_608.0,
    ];
    for b in boundaries {
        let bits = b.to_bits();
        for offset in 0..8 {
            v.push(f32::from_bits(bits.wrapping_add(offset)));
            v.push(f32::from_bits(bits.wrapping_sub(offset)));
        }
    }
    v
}

/// Xorshift over the full bit range, NaN patterns included: tiers must agree
/// on NaN-ness for every payload.
pub fn sweep_bits(n: usize) -> Vec<f32> {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    (0..n)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            f32::from_bits((s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32)
        })
        .collect()
}

/// `n` values evenly spaced by value from `lo` to `hi`, where bit-uniform
/// sampling is too sparse: most of the exp domain has few distinct exponents.
pub fn sweep_value(lo: f64, hi: f64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (lo + (hi - lo) * i as f64 / (n - 1) as f64) as f32)
        .collect()
}

/// The inputs behind the digests in `tests/golden/digests.txt`.
pub fn corpus() -> Vec<f32> {
    let mut v = edges();
    v.extend(sweep_bits(1 << 17));
    v.extend(sweep_value(-160.0, 160.0, 1 << 15));
    v.extend(sweep_value(-1.0, 1.0, 1 << 15));
    v.extend(sweep_value(0.0, 256.0, 1 << 15));
    v
}

/// NaN results compare by NaN-ness; everything else by exact bits.
pub fn same_bits(got: f32, want: f32) -> bool {
    if want.is_nan() {
        got.is_nan()
    } else {
        got.to_bits() == want.to_bits()
    }
}

/// Fails the test, naming the input, unless `got` matches the mirror.
#[track_caller]
pub fn assert_same(what: &str, name: &str, x: f32, got: f32, want: f32) {
    assert!(
        same_bits(got, want),
        "{what} {name}({x:e} = {:08x}): got {got:e} ({:08x}), mirror {want:e} ({:08x})",
        x.to_bits(),
        got.to_bits(),
        want.to_bits()
    );
}

/// `T`'s token, or `None` when the caller listed `tier` in
/// `MAGEMATH_SKIP_TIERS`. A tier that is simply missing fails the test, so
/// a run cannot silently lose coverage.
pub fn require<T: SimdToken>(tier: &str) -> Option<T> {
    if let Some(token) = T::summon() {
        return Some(token);
    }
    let skip = std::env::var("MAGEMATH_SKIP_TIERS").unwrap_or_default();
    assert!(
        skip.split(',').any(|s| s == tier),
        "tier {tier} is unavailable on this CPU; set MAGEMATH_SKIP_TIERS={tier} to skip it"
    );
    None
}

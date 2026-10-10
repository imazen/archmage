//! The `*_midp_portable` transcendentals give the same bits on every backend
//! and token tier.
//!
//! On one machine the test computes every function with the native tier and
//! with the scalar backend, under every token permutation (so the scalar
//! backend's `mul_add_portable` runs both with hardware FMA and fused in
//! software), and requires identical bits. Across machines it folds the output
//! bits into FNV-1a checksums and requires the pinned values, which CI checks
//! on x86-64, aarch64 and WASM. Where the CPU has FMA, each portable form must
//! also equal its plain `*_midp` form, which is fused there too, wherever the
//! result is a number (the plain forms do not carry NaN through).
//!
//!   cargo test -p magetypes --test transcendentals_portable -- --nocapture

#![cfg(feature = "std")]

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use archmage::{ScalarToken, SimdToken};
use magetypes::simd::generic::{f32x4, f32x8};

#[cfg(target_arch = "x86_64")]
type Native = archmage::X64V3Token;
#[cfg(target_arch = "aarch64")]
type Native = archmage::NeonToken;
#[cfg(target_arch = "wasm32")]
type Native = archmage::Wasm128Token;
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
)))]
type Native = archmage::ScalarToken;

/// Whether `Native` fuses `mul_add` (so plain midp equals portable on it).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const NATIVE_FUSES: bool = true;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const NATIVE_FUSES: bool = false;

/// Positive inputs over the whole normal range and the specials, for the
/// logarithms and `pow`; length a multiple of 16.
fn log_inputs() -> Vec<f32> {
    let mut v = Vec::new();
    for exp in 1u32..=254 {
        let mut m = 0u32;
        while m < (1 << 23) {
            v.push(f32::from_bits((exp << 23) | m));
            m += 65_537;
        }
    }
    v.extend([
        0.0,
        -0.0,
        -1.0,
        -1e-30,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1.0,
        f32::MIN_POSITIVE,
        1e-42,
        f32::MAX,
        0.5,
    ]);
    while !v.len().is_multiple_of(16) {
        v.push(1.0);
    }
    v
}

/// Arguments for the exponentials: [-140, 140] in steps that hit every
/// half-integer tie, and the specials; length a multiple of 16.
fn exp_inputs() -> Vec<f32> {
    let mut v: Vec<f32> = (-140_000..=140_000)
        .step_by(7)
        .map(|i| i as f32 * 1e-3)
        .collect();
    v.extend((-280..=280).map(|i| i as f32 * 0.5));
    v.extend([
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        127.5,
        127.99,
        -126.0,
        -125.999,
        128.0,
    ]);
    while !v.len().is_multiple_of(16) {
        v.push(0.0);
    }
    v
}

/// Output bits with every NaN folded to one pattern (NaN payloads are not part
/// of the contract).
fn canon(x: f32) -> u32 {
    if x.is_nan() { 0x7fc0_0000 } else { x.to_bits() }
}

/// Panics with the first input where `got` and `want` differ.
fn same(got: &[u32], want: &[u32], data: &[f32], what: &str) {
    if let Some(i) = (0..want.len()).find(|&i| got[i] != want[i]) {
        panic!(
            "{what}: input {:e} ({:#010x}) gives {:e} ({:#010x}), expected {:e} ({:#010x})",
            data[i],
            data[i].to_bits(),
            f32::from_bits(got[i]),
            got[i],
            f32::from_bits(want[i]),
            want[i]
        );
    }
}

/// [`same`] where `want` is a number: the plain midp forms do not carry NaN
/// through (`log2_midp(NaN)` and `pow_midp(-1, n)` give numbers on some
/// backends), so they match the portable forms everywhere else.
fn same_for_numbers(got: &[u32], want: &[u32], data: &[f32], what: &str) {
    let idx: Vec<usize> = (0..want.len())
        .filter(|&i| !f32::from_bits(want[i]).is_nan())
        .collect();
    let pick = |v: &[u32]| -> Vec<u32> { idx.iter().map(|&i| v[i]).collect() };
    let inputs: Vec<f32> = idx.iter().map(|&i| data[i]).collect();
    same(&pick(got), &pick(want), &inputs, what);
}

fn fnv(bits: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bits {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Every output of `$method` (with `$($arg)*`) over `$data`, `$lanes` at a time
/// through `$ty`.
macro_rules! run {
    ($ty:ty, $lanes:literal, $token:expr, $data:expr, $method:ident $(, $arg:expr)*) => {{
        let mut out: Vec<u32> = Vec::with_capacity($data.len());
        for chunk in $data.chunks_exact($lanes) {
            let arr: [f32; $lanes] = chunk.try_into().unwrap();
            let r = <$ty>::from_array_t($token, arr).$method($($arg),*).to_array();
            out.extend(r.iter().map(|&v| canon(v)));
        }
        out
    }};
}

/// One function: the scalar backend under every permutation and the native
/// tier where it is available must agree; returns the checksum.
macro_rules! check {
    ($name:literal, $data:expr, $portable:ident, $plain:ident $(, $arg:expr)*) => {{
        let data = $data;
        let reference = run!(f32x4<ScalarToken>, 4, ScalarToken, &data, $portable $(, $arg)*);
        let mut runs = 0;
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let scalar = run!(f32x4<ScalarToken>, 4, ScalarToken, &data, $portable $(, $arg)*);
            same(&scalar, &reference, &data, &format!("{}: scalar backend under {perm}", $name));
            if let Some(t) = Native::summon() {
                let native = run!(f32x4<Native>, 4, t, &data, $portable $(, $arg)*);
                same(&native, &reference, &data, &format!("{}: native tier under {perm}", $name));
                let wide = run!(f32x8<Native>, 8, t, &data, $portable $(, $arg)*);
                same(&wide, &reference, &data, &format!("{}: f32x8 under {perm}", $name));
                if NATIVE_FUSES {
                    let plain = run!(f32x4<Native>, 4, t, &data, $plain $(, $arg)*);
                    same_for_numbers(&plain, &reference, &data, &format!("{}: plain midp where it fuses", $name));
                }
                #[cfg(feature = "w512")]
                {
                    let w16 = run!(magetypes::simd::generic::f32x16<Native>, 16, t, &data, $portable $(, $arg)*);
                    same(&w16, &reference, &data, &format!("{}: f32x16 under {perm}", $name));
                }
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            if let Some(t) = archmage::X64V4Token::summon() {
                let v4w = run!(magetypes::simd::generic::f32x16<archmage::X64V4Token>, 16, t, &data, $portable $(, $arg)*);
                same(&v4w, &reference, &data, &format!("{}: x86-64-v4 f32x16 under {perm}", $name));
            }
            runs += 1;
        });
        assert!(runs > 0, "{report}");
        fnv(&reference)
    }};
}

#[test]
fn portable_transcendentals_same_bits_everywhere() {
    let logs = log_inputs();
    let exps = exp_inputs();
    let sums = [
        (
            "log2_midp_portable",
            check!("log2", &logs, log2_midp_portable, log2_midp),
        ),
        (
            "ln_midp_portable",
            check!("ln", &logs, ln_midp_portable, ln_midp),
        ),
        (
            "log10_midp_portable",
            check!("log10", &logs, log10_midp_portable, log10_midp),
        ),
        (
            "exp2_midp_portable",
            check!("exp2", &exps, exp2_midp_portable, exp2_midp),
        ),
        (
            "exp_midp_portable",
            check!("exp", &exps, exp_midp_portable, exp_midp),
        ),
        (
            "pow_midp_portable(2.4)",
            check!("pow 2.4", &logs, pow_midp_portable, pow_midp, 2.4),
        ),
        (
            "pow_midp_portable(-0.65)",
            check!("pow -0.65", &logs, pow_midp_portable, pow_midp, -0.65),
        ),
        (
            "pow_midp_portable(61.4)",
            check!("pow 61.4", &logs, pow_midp_portable, pow_midp, 61.4),
        ),
    ];
    for (name, h) in &sums {
        eprintln!("  {name:<26} {h:#018x}");
    }
    for ((name, h), want) in sums.iter().zip(PINNED) {
        assert_eq!(*h, want, "{name}: checksum {h:#018x}, pinned {want:#018x}");
    }
}

/// Checksums of the outputs above, the same on every machine.
const PINNED: [u64; 8] = [
    0x0712_8576_b413_b706, // log2
    0x91b6_93bf_3f88_65b2, // ln
    0x769b_1f82_10d3_df6b, // log10
    0x0119_075b_ae5d_1287, // exp2
    0x307d_eb8e_1c1f_f7fb, // exp
    0x4b81_0e93_6e24_5b52, // pow 2.4
    0xdb7f_adca_600c_2096, // pow -0.65
    0xacac_fbb0_cca8_bd52, // pow 61.4
];

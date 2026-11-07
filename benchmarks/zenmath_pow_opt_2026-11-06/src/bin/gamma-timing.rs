#![forbid(unsafe_code)]
use archmage::prelude::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    hint::black_box,
    time::Duration,
};
use zenbench::prelude::*;
const Y: [u32; 2] = [0x4019999a, 0x3ed55555];
const MIN: [u32; 2] = [0x3db71074, 0x3b475041];
const A: f32 = f32::from_bits(0x3d6152eb);
const AP: f32 = f32::from_bits(0x3f870a97);
const CUT: [f32; 2] = [f32::from_bits(0x3d20f216), f32::from_bits(0x3b475041)];
const NAMES: [&str; 13] = [
    "std",
    "before/ordinary",
    "current/ordinary",
    "before/policy-correct",
    "current/policy-correct",
    "before/context",
    "current/context",
    "before/packed4",
    "current/packed4",
    "std/context",
    "before/slice-context",
    "current/slice-context",
    "std/slice-context",
];
fn inputs(k: usize, n: usize, distribution: &str, transfer: bool) -> (Vec<f32>, Vec<f32>) {
    let mut state = 0x7e5162b3u32;
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..n {
        let j = i * 719 % n;
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let value = match distribution {
            "bits" => f32::from_bits(if transfer {
                (u64::from(1.0f32.to_bits()) * j as u64 / (n - 1) as u64) as u32
            } else {
                MIN[k] + (u64::from(0x3f800000 - MIN[k]) * j as u64 / (n - 1) as u64) as u32
            }),
            "values" => {
                let min = if transfer {
                    0.0
                } else {
                    f32::from_bits(MIN[k])
                };
                min + (1.0 - min) * (j as f32 / (n - 1) as f32)
            }
            "general" => {
                f32::from_bits(((67 + state % 121) << 23) | (state.rotate_left(7) & 0x7fffff))
            }
            "historical" => 0.01 + 4.0 * (state >> 8) as f32 / 16777216.0,
            _ => panic!("unknown corpus"),
        };
        x.push(value);
        y.push(if distribution == "general" {
            ((state.rotate_left(13) >> 8) as f32 / 16777216.0) * 4.0 - 2.0
        } else {
            f32::from_bits(Y[k])
        });
    }
    (x, y)
}
#[attune(_v3_t, _neon_t, _scalar_t)]
fn contexts<const M: usize>(x: &[f32], y: &[f32], out: &mut [f32]) {
    let t = Token::from_context();
    if M >= 10 {
        out.copy_from_slice(x);
        if M == 10 {
            attuned!(
                pregamma::simd::slice::pow_f32(out, y),
                [_v3, _neon, _scalar],
                using(t)
            );
        } else if M == 11 {
            attuned!(
                zenmath::simd::slice::pow_f32(out, y),
                [_v3, _neon, _scalar],
                using(t)
            );
        } else {
            for (x, &y) in out.iter_mut().zip(y) {
                *x = x.powf(y);
            }
        }
        return;
    }
    for ((x, y), o) in x.iter().zip(y).zip(out) {
        *o = if M == 5 {
            attuned!(
                pregamma::simd::scalar::powf(*x, *y),
                [_v3, _neon, _scalar],
                using(t)
            )
        } else if M == 6 {
            attuned!(
                zenmath::simd::scalar::powf(*x, *y),
                [_v3, _neon, _scalar],
                using(t)
            )
        } else {
            x.powf(*y)
        };
    }
}
fn context<const M: usize>(x: &[f32], y: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        contexts_v3_t::<M>(t, x, y, out);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        contexts_neon_t::<M>(t, x, y, out);
        return;
    }
    contexts_scalar_t::<M>(ScalarToken, x, y, out)
}
fn power<const M: usize>(x: &[f32], y: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len(), y.len());
    if M == 5 || M == 6 || M == 9 || M >= 10 {
        context::<M>(x, y, out);
        return;
    }
    if M == 7 || M == 8 {
        // All registered timing lengths are full blocks. Assert the contract
        // before as_chunks, avoiding per-block variable-length memcpy glue.
        assert_eq!(x.len() % 4, 0);
        for ((a, b), o) in x
            .as_chunks::<4>()
            .0
            .iter()
            .zip(y.as_chunks::<4>().0)
            .zip(out.as_chunks_mut::<4>().0)
        {
            *o = if M == 7 {
                pregamma::simd::array::pow_f32x4(*a, *b)
            } else {
                zenmath::simd::array::pow_f32x4(*a, *b)
            };
        }
        return;
    }
    for ((&x, &y), o) in x.iter().zip(y).zip(out) {
        *o = match M {
            0 => x.powf(y),
            1 => pregamma::scalar::powf(x, y),
            2 => zenmath::scalar::powf(x, y),
            3 => pregamma::scalar::powf_with::<0>(x, y),
            4 => zenmath::scalar::powf_with::<0>(x, y),
            _ => unreachable!(),
        };
    }
}
fn transfer<const K: usize, const M: usize>(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else if x < CUT[K] {
        if K == 0 {
            x * f32::from_bits(0x3d9e8391)
        } else {
            x * 12.92
        }
    } else if x < 1.0 {
        let base = if K == 0 { (x + A) / AP } else { x };
        let y = f32::from_bits(Y[K]);
        let p = match M {
            0 => base.powf(y),
            1 => pregamma::scalar::powf(base, y),
            2 => zenmath::scalar::powf(base, y),
            3 => pregamma::scalar::powf_with::<0>(base, y),
            4 => zenmath::scalar::powf_with::<0>(base, y),
            _ => unreachable!(),
        };
        if K == 0 { p } else { AP * p - A }
    } else {
        1.0
    }
}
fn whole<const K: usize, const M: usize>(x: &[f32], _: &[f32], out: &mut [f32]) {
    for (&x, o) in x.iter().zip(out) {
        *o = transfer::<K, M>(x);
    }
}
fn fixed<const K: usize, const M: usize>(x: &[f32], _: &[f32], out: &mut [f32]) {
    for (&x, o) in x.iter().zip(out) {
        let y = f32::from_bits(Y[K]);
        *o = match M {
            0 => x.powf(y),
            1 => pregamma::scalar::powf(x, y),
            2 => zenmath::scalar::powf(x, y),
            3 => pregamma::scalar::powf_with::<0>(x, y),
            4 => zenmath::scalar::powf_with::<0>(x, y),
            _ => unreachable!(),
        };
    }
}
type Kernel = fn(&[f32], &[f32], &mut [f32]);
fn methods(k: usize, mode: &str) -> Vec<(&'static str, Kernel)> {
    macro_rules! list {
        ($f:ident,$k:literal) => {
            vec![
                (NAMES[0], $f::<$k, 0> as Kernel),
                (NAMES[1], $f::<$k, 1> as Kernel),
                (NAMES[2], $f::<$k, 2> as Kernel),
                (NAMES[3], $f::<$k, 3> as Kernel),
                (NAMES[4], $f::<$k, 4> as Kernel),
            ]
        };
    }
    if mode == "C0" {
        return if k == 0 {
            list!(whole, 0)
        } else {
            list!(whole, 1)
        };
    }
    if mode == "fixed" {
        return if k == 0 {
            list!(fixed, 0)
        } else {
            list!(fixed, 1)
        };
    }
    vec![
        (NAMES[0], power::<0>),
        (NAMES[1], power::<1>),
        (NAMES[2], power::<2>),
        (NAMES[3], power::<3>),
        (NAMES[4], power::<4>),
        (NAMES[5], power::<5>),
        (NAMES[6], power::<6>),
        (NAMES[7], power::<7>),
        (NAMES[8], power::<8>),
        (NAMES[9], power::<9>),
        (NAMES[10], power::<10>),
        (NAMES[11], power::<11>),
        (NAMES[12], power::<12>),
    ]
}
fn main() {
    let required =
        std::env::var("POW_EXPECT_TIER").expect("caller must name expected context tier");
    let available = match required.as_str() {
        "v3" => X64V3Token::summon().is_some(),
        "neon" => NeonToken::summon().is_some(),
        "scalar" => true,
        _ => false,
    };
    assert!(available, "required timing context unavailable: {required}");
    println!("GAMMA_CONTEXT {required}; scalar/packed4, packed4 nativebinary64width4; no512 claim");
    let args: Vec<_> = std::env::args().collect();
    let output = &args[1];
    let mut requested = BTreeSet::new();
    for pair in args[2..].chunks(2) {
        assert_eq!(pair.len(), 2);
        assert_eq!(pair[0], "--study-group");
        assert!(requested.insert(pair[1].clone()));
    }
    let mut expected = BTreeMap::new();
    let results = zenbench::run(|suite| {
        suite.cpu_policy(zenbench::platform::CpuPolicy::Performance);
        for k in 0..2 {
            for n in [64, 4096] {
                for distribution in ["bits", "values"] {
                    for mode in ["dynamic", "fixed", "C0"] {
                        let name = format!(
                            "{mode}/{}/{distribution}/{n}",
                            if k == 0 { "decode" } else { "encode" }
                        );
                        if !requested.is_empty() && !requested.contains(&name) {
                            continue;
                        }
                        let (x, y) = inputs(k, n, distribution, mode == "C0");
                        let cases = methods(k, mode);
                        let mut baseline = vec![0.0; n];
                        (cases[1].1)(&x, &y, &mut baseline);
                        for &(label, f) in &cases {
                            if label.starts_with("std") {
                                continue;
                            }
                            let mut got = vec![0.0; n];
                            f(&x, &y, &mut got);
                            for i in 0..n {
                                assert_eq!(
                                    got[i].to_bits(),
                                    baseline[i].to_bits(),
                                    "{name}/{label}/{i}"
                                );
                            }
                        }
                        assert!(
                            expected
                                .insert(
                                    name.clone(),
                                    cases
                                        .iter()
                                        .map(|c| c.0.to_owned())
                                        .collect::<BTreeSet<_>>()
                                )
                                .is_none()
                        );
                        suite.group(name, |g| {
                            g.throughput(Throughput::Elements(n as u64));
                            g.baseline("before/ordinary");
                            g.config()
                                .warmup_time(Duration::from_millis(50))
                                .min_rounds(60)
                                .max_rounds(100)
                                .max_wall_time(Duration::from_secs(20));
                            for (label, f) in cases {
                                let x = x.clone();
                                let y = y.clone();
                                let mut out = vec![0.0; n];
                                f(&x, &y, &mut out);
                                g.bench(label, move |b| {
                                    b.iter(|| {
                                        f(black_box(&x), black_box(&y), black_box(&mut out));
                                        black_box(&out);
                                    })
                                });
                            }
                        });
                    }
                }
            }
        }
        for distribution in ["general", "historical"] {
            for n in [16, 65536] {
                let name = format!("dynamic/{distribution}/{n}");
                if !requested.is_empty() && !requested.contains(&name) {
                    continue;
                }
                let (x, y) = inputs(0, n, distribution, false);
                let cases = methods(0, "dynamic");
                let mut baseline = vec![0.0; n];
                (cases[1].1)(&x, &y, &mut baseline);
                for &(label, f) in &cases {
                    if label.starts_with("std") {
                        continue;
                    }
                    let mut got = vec![0.0; n];
                    f(&x, &y, &mut got);
                    for i in 0..n {
                        assert_eq!(
                            got[i].to_bits(),
                            baseline[i].to_bits(),
                            "{name}/{label}/{i}"
                        );
                    }
                }
                expected.insert(
                    name.clone(),
                    cases
                        .iter()
                        .map(|c| c.0.to_owned())
                        .collect::<BTreeSet<_>>(),
                );
                suite.group(name, |g| {
                    g.throughput(Throughput::Elements(n as u64));
                    g.baseline("before/ordinary");
                    g.config()
                        .warmup_time(Duration::from_millis(50))
                        .min_rounds(60)
                        .max_rounds(100);
                    for (label, f) in cases {
                        let x = x.clone();
                        let y = y.clone();
                        let mut out = vec![0.0; n];
                        g.bench(label, move |b| {
                            b.iter(|| {
                                f(black_box(&x), black_box(&y), black_box(&mut out));
                                black_box(&out);
                            })
                        });
                    }
                });
            }
        }
    });
    let mut actual = BTreeMap::new();
    for group in &results.comparisons {
        let names: BTreeSet<_> = group.benchmarks.iter().map(|b| b.name.clone()).collect();
        assert_eq!(names.len(), group.benchmarks.len());
        assert!(actual.insert(group.group_name.clone(), names).is_none());
    }
    assert_eq!(actual, expected);
    if !requested.is_empty() {
        assert_eq!(requested, actual.keys().cloned().collect());
    }
    results.save(output).unwrap();
}

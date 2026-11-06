use crate::{
    generated as g,
    kernel::{self, domain, exponent},
    piecewise,
};
use archmage::{SimdToken, X64V3Token};
use std::{hint::black_box, path::Path, time::Duration};
use zenbench::prelude::*;
use zenmath::{PowBehavior as B, PowRange};

type Stream = fn(&[f32], &mut [f32]);
const MIDP_PRE: u64 = B::MIDP.range(PowRange::PositiveNormalPrecondition).to_u64();
#[inline(always)]
fn std_stream<const K: usize>(input: &[f32], out: &mut [f32]) {
    let p = exponent::<K>();
    for (o, &x) in out.iter_mut().zip(input) {
        *o = x.powf(p);
    }
}
fn core_stream<const K: usize>(input: &[f32], out: &mut [f32]) {
    let p = exponent::<K>();
    for (o, &x) in out.iter_mut().zip(input) {
        *o = zenmath::scalar::reference::powf(x, p);
    }
}
fn policy_stream<const K: usize, const P: u64>(input: &[f32], out: &mut [f32]) {
    let p = exponent::<K>();
    for (o, &x) in out.iter_mut().zip(input) {
        *o = zenmath::scalar::powf_with::<P>(x, p);
    }
}
#[inline(always)]
fn cert_stream<const K: usize, const N: usize, const L: usize>(input: &[f32], out: &mut [f32]) {
    for (o, &x) in out.iter_mut().zip(input) {
        *o = kernel::evaluate::<K, N, L>(x).value;
    }
}
#[archmage::arcane]
fn std_context<const K: usize>(token: X64V3Token, input: &[f32], out: &mut [f32]) {
    std_stream::<K>(input, out);
    let _ = token;
}
#[archmage::arcane]
fn core_context<const K: usize>(token: X64V3Token, input: &[f32], out: &mut [f32]) {
    let p = exponent::<K>();
    for (o, &x) in out.iter_mut().zip(input) {
        *o = zenmath::simd::scalar::powf_v3(x, p);
    }
    let _ = token;
}
#[archmage::arcane]
fn cert_context<const K: usize, const N: usize, const L: usize>(
    token: X64V3Token,
    input: &[f32],
    out: &mut [f32],
) {
    cert_stream::<K, N, L>(input, out);
    let _ = token;
}
fn std_v3<const K: usize>(input: &[f32], out: &mut [f32]) {
    std_context::<K>(X64V3Token::summon().expect("V3 required"), input, out);
}
fn core_v3<const K: usize>(input: &[f32], out: &mut [f32]) {
    core_context::<K>(X64V3Token::summon().expect("V3 required"), input, out);
}
fn cert_v3<const K: usize, const N: usize, const L: usize>(input: &[f32], out: &mut [f32]) {
    cert_context::<K, N, L>(X64V3Token::summon().expect("V3 required"), input, out);
}

fn cases<const K: usize>(refine: bool) -> Vec<(String, Stream)> {
    let mut cases: Vec<(String, Stream)> = vec![
        ("std".into(), std_stream::<K>),
        ("std_v3".into(), std_v3::<K>),
        ("core_reference".into(), core_stream::<K>),
        ("core_v3".into(), core_v3::<K>),
        (
            "midp_full_approx".into(),
            policy_stream::<K, { B::MIDP.to_u64() }>,
        ),
        (
            "midp_precondition_approx".into(),
            policy_stream::<K, MIDP_PRE>,
        ),
        (
            "lowp_full_approx".into(),
            policy_stream::<K, { B::LOWP.to_u64() }>,
        ),
    ];
    macro_rules! add {
        ($n:literal) => {
            cases.push((format!("cert_h{}", $n), cert_stream::<K, $n, 0>));
            cases.push((format!("cert_e{}", $n), cert_stream::<K, $n, 1>));
            cases.push((format!("cert_h{}_v3", $n), cert_v3::<K, $n, 0>));
            cases.push((format!("cert_e{}_v3", $n), cert_v3::<K, $n, 1>));
        };
    }
    if refine {
        add!(11);
        add!(12);
        add!(13);
        add!(14);
        add!(15);
        add!(16);
        add!(17);
    } else {
        add!(8);
        add!(12);
        add!(16);
        add!(20);
    }
    cases
}
#[inline(always)]
fn piece_stream<const K: usize, const N: usize, const L: usize>(input: &[f32], out: &mut [f32]) {
    for (o, &x) in out.iter_mut().zip(input) {
        *o = piecewise::evaluate::<K, N, L>(x).value;
    }
}
#[archmage::arcane]
fn piece_context<const K: usize, const N: usize, const L: usize>(
    token: X64V3Token,
    input: &[f32],
    out: &mut [f32],
) {
    piece_stream::<K, N, L>(input, out);
    let _ = token;
}
fn piece_v3<const K: usize, const N: usize, const L: usize>(input: &[f32], out: &mut [f32]) {
    piece_context::<K, N, L>(X64V3Token::summon().expect("V3 required"), input, out);
}
fn piece_cases<const K: usize>() -> Vec<(String, Stream)> {
    let mut methods = cases::<K>(false);
    methods.retain(|(n, _)| !n.starts_with("cert_"));
    macro_rules! add {
        ($n:literal) => {
            methods.push((format!("piece_h{}", $n), piece_stream::<K, $n, 0>));
            methods.push((format!("piece_e{}", $n), piece_stream::<K, $n, 1>));
            methods.push((format!("piece_h{}_v3", $n), piece_v3::<K, $n, 0>));
            methods.push((format!("piece_e{}_v3", $n), piece_v3::<K, $n, 1>));
        };
    }
    add!(3);
    add!(4);
    add!(5);
    if K == 0 {
        methods.push(("cert_e13".into(), cert_stream::<K, 13, 1>));
        methods.push(("cert_e13_v3".into(), cert_v3::<K, 13, 1>));
    } else {
        methods.push(("cert_e15".into(), cert_stream::<K, 15, 1>));
        methods.push(("cert_e15_v3".into(), cert_v3::<K, 15, 1>));
    }
    methods
}
pub(crate) fn input<const K: usize>(n: usize, distribution: &str) -> Vec<f32> {
    let (start, end) = domain::<K>();
    let minimum = f32::from_bits(start);
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let index = (i * 719) % n;
        data.push(if distribution == "bits" {
            f32::from_bits(
                start + ((u64::from(end - start) * index as u64) / (n - 1) as u64) as u32,
            )
        } else {
            minimum + (1.0 - minimum) * (index as f32 / (n - 1) as f32)
        });
    }
    data
}
fn groups<const K: usize>(suite: &mut Suite, refine: bool, piece: bool) {
    for n in [64usize, 4096] {
        for distribution in ["values", "bits"] {
            let name = format!(
                "{}/{distribution}/{n}",
                if K == 0 { "decode" } else { "encode" }
            );
            let input = input::<K>(n, distribution);
            suite.group(name, |group| {
                group.throughput(Throughput::Elements(n as u64));
                group
                    .config()
                    .warmup_time(Duration::from_millis(200))
                    .min_rounds(50)
                    .max_rounds(100)
                    .max_wall_time(Duration::from_secs(30));
                for (label, fun) in if piece {
                    piece_cases::<K>()
                } else {
                    cases::<K>(refine)
                } {
                    let input = input.clone();
                    let mut out = vec![0.0; n];
                    fun(&input, &mut out); // Prefault/warm code and arrays before timed sampling.
                    group.bench(label, move |bench| {
                        bench.iter(|| {
                            fun(black_box(&input), black_box(&mut out));
                            black_box(&out);
                        })
                    });
                }
                group.baseline("std");
            });
        }
    }
}
pub(crate) fn run(
    path: &Path,
    refine: bool,
    piece: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    X64V3Token::summon().ok_or("V3 unavailable")?;
    // Correct rounding and approximation controls intentionally remain distinct.
    let results = zenbench::run(|suite| {
        suite.cpu_policy(zenbench::platform::CpuPolicy::Performance);
        groups::<0>(suite, refine, piece);
        groups::<1>(suite, refine, piece);
    });
    results.save(path)?;
    let _ = g::DECODE_P_BITS;
    Ok(())
}

/// Offline only; never called from timed closures.
pub(crate) fn rescue_counts(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let mut output = std::fs::File::create_new(path)?;
    writeln!(
        output,
        "kind,distribution,length,degree,layout,rescues,count"
    )?;
    fn kind<const K: usize>(output: &mut std::fs::File) -> Result<(), Box<dyn std::error::Error>> {
        for n in [64usize, 4096] {
            for distribution in ["values", "bits"] {
                let mut counts = [0usize; 32];
                for x in input::<K>(n, distribution) {
                    let expected = zenmath::scalar::reference::powf(x, exponent::<K>());
                    kernel::each::<K>(x, |degree, layout, evaluation| {
                        assert_eq!(evaluation.value.to_bits(), expected.to_bits());
                        counts[(degree - 8) * 2 + layout] += usize::from(evaluation.rescued);
                    });
                    piecewise::each::<K>(x, |degree, layout, evaluation| {
                        assert_eq!(evaluation.value.to_bits(), expected.to_bits());
                        counts[26 + (degree - 3) * 2 + layout] += usize::from(evaluation.rescued);
                    });
                }
                for (i, count) in counts.iter().enumerate() {
                    writeln!(
                        output,
                        "{K},{distribution},{n},{},{},{count},{n}",
                        if i < 26 { i / 2 + 8 } else { (i - 26) / 2 + 3 },
                        i % 2
                    )?;
                }
            }
        }
        Ok(())
    }
    kind::<0>(&mut output)?;
    kind::<1>(&mut output)?;
    Ok(())
}

fn vector_cases<const K: usize>() -> Vec<(String, Stream)> {
    let mut methods = piece_cases::<K>();
    methods.retain(|(n, _)| !n.starts_with("piece_h") && !n.starts_with("piece_e3"));
    macro_rules! add {
        ($w:literal) => {
            if K == 0 {
                methods.push((
                    format!("array{}_global_e13", $w),
                    crate::vector::stream::<K, 13, 1, false, $w>,
                ));
            } else {
                methods.push((
                    format!("array{}_global_e15", $w),
                    crate::vector::stream::<K, 15, 1, false, $w>,
                ));
            }
            methods.push((
                format!("array{}_piece_e4", $w),
                crate::vector::stream::<K, 4, 1, true, $w>,
            ));
            methods.push((
                format!("array{}_piece_e5", $w),
                crate::vector::stream::<K, 5, 1, true, $w>,
            ));
        };
    }
    add!(4);
    add!(8);
    methods
}
pub(crate) fn run_vector(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    X64V3Token::summon().ok_or("V3 unavailable")?;
    fn kind<const K: usize>(suite: &mut Suite) {
        for n in [64usize, 4096] {
            for distribution in ["values", "bits"] {
                let name = format!(
                    "{}/{distribution}/{n}",
                    if K == 0 { "decode" } else { "encode" }
                );
                let input = input::<K>(n, distribution);
                suite.group(name, |group| {
                    group.throughput(Throughput::Elements(n as u64));
                    group
                        .config()
                        .warmup_time(Duration::from_millis(200))
                        .min_rounds(50)
                        .max_rounds(100)
                        .max_wall_time(Duration::from_secs(30));
                    for (label, fun) in vector_cases::<K>() {
                        let input = input.clone();
                        let mut out = vec![0.0; n];
                        fun(&input, &mut out);
                        group.bench(label, move |b| {
                            b.iter(|| {
                                fun(black_box(&input), black_box(&mut out));
                                black_box(&out);
                            })
                        });
                    }
                    group.baseline("std");
                });
            }
        }
    }
    let results = zenbench::run(|suite| {
        suite.cpu_policy(zenbench::platform::CpuPolicy::Performance);
        kind::<0>(suite);
        kind::<1>(suite);
    });
    results.save(path)?;
    Ok(())
}

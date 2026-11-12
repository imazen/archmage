//! All canonical primitive configurations; fixed paired workloads.
#![forbid(unsafe_code)]
#[macro_use]
#[path = "../policy_registry.rs"]
mod registry;
#[path = "../policy_check.rs"]
mod check;
#[path = "../policy_workload.rs"]
mod workload;
use archmage::ScalarToken;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
use archmage::SimdToken;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use workload::*;
use zenbench::{Suite, black_box};

struct BenchFilter {
    substr: Option<String>,
    dedup: Option<BTreeMap<String, BTreeSet<String>>>,
    rounds: usize,
}

impl BenchFilter {
    fn from_args(args: &[String], iterate: bool) -> Self {
        let substr = args
            .iter()
            .position(|x| x == "--filter")
            .and_then(|i| args.get(i + 1))
            .cloned();
        let dedup_path = args
            .iter()
            .position(|x| x == "--dedup")
            .and_then(|i| args.get(i + 1))
            .cloned();
        let dedup = dedup_path.map(|p| {
            let mut map: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
            for line in std::fs::read_to_string(p)
                .expect("read --dedup TSV")
                .lines()
            {
                let mut cols = line.split('\t');
                if let (Some(cell), Some(rep)) = (cols.next(), cols.next()) {
                    map.entry(cell.to_owned())
                        .or_default()
                        .insert(rep.to_owned());
                }
            }
            map
        });
        Self {
            substr,
            dedup,
            rounds: if iterate { 30 } else { 50 },
        }
    }
    fn matches_group(&self, name: &str) -> bool {
        self.substr.as_ref().is_none_or(|f| name.contains(f))
    }
    fn keep_arm(&self, group: &str, arm: &str) -> bool {
        let Some(map) = &self.dedup else {
            return true;
        };
        if let Some(reps) = map.get(group) {
            return reps.contains(arm);
        }
        if let Some(prefix) = group
            .strip_suffix("/stream4096")
            .or_else(|| group.strip_suffix("/zero-clamp64"))
        {
            let fallback = format!("{prefix}/stream64");
            if let Some(reps) = map.get(&fallback) {
                return reps.contains(arm);
            }
        }
        true
    }
}

fn add_bench<F: CheckScalar, const N: usize, const W: usize, const DEP: bool>(
    g: &mut zenbench::BenchGroup,
    name: &str,
    data: Arc<([F; N], [F; N])>,
    stream: impl Fn(&[F; N], &[F; N], &mut [F; N]) + Send + 'static,
    chain: impl Fn(F) -> [F; W] + Send + 'static,
) {
    g.bench(name.to_owned(), move |b| {
        let mut out = [F::from_f32(0.0); N];
        b.iter(|| {
            if DEP {
                black_box(chain(black_box(F::from_f32(0.75))));
            } else {
                stream(black_box(&data.0), black_box(&data.1), black_box(&mut out));
                black_box(&out);
            }
        });
    });
}

macro_rules! group {
    ($suite:ident,$flt:ident,$tier:ident,$t:ident,$stream:ident,$chain:ident,$float:ty,$w:literal,$operation:literal,$recip:literal,$op:literal,$n:literal,$dependent:literal,$zero:literal,$cases:ident,$kind:literal) => {{
        let name = format!(
            "{}/{}/{}x{}/{}",
            $tier,
            $operation,
            stringify!($float),
            $w,
            $kind
        );
        if $flt.matches_group(&name) {
            let pair = |i| <$float as CheckScalar>::pair(i, $zero, $recip, $recip && $op);
            let a: [$float; $n] = core::array::from_fn(|i| pair(i).0);
            let b: [$float; $n] = core::array::from_fn(|i| pair(i).1);
            let data = Arc::new((a, b));
            macro_rules! preflight {
                ($flag:literal,$legacy:literal,$id:expr) => {{
                    let error = check::verify::<$float, $n, $w>(
                        $flag,
                        $legacy,
                        $recip,
                        $op,
                        $dependent,
                        &data,
                        |a, b, out| $stream::<$flag, $op, $legacy, $n>($t, a, b, out),
                        |a, b, out| $stream::<$flag, $op, $legacy, $w>($t, a, b, out),
                        |x| $chain::<$flag, $op, $legacy>($t, x),
                    );
                    eprintln!("POLICY_ACCURACY\t{}\t{}\t{:.17e}", name, $id, error);
                }};
            }
            if std::env::args().any(|a| a == "--verify-only") {
                preflight!(0, true, "existing Magetypes");
                macro_rules! verify {
                    ($o:literal,$id:literal) => {
                        preflight!($o, false, $id);
                    };
                }
                $cases!(verify);
            } else {
                $suite.group(name.clone(), |g| {
                    g.config()
                        .min_rounds($flt.rounds)
                        .max_rounds($flt.rounds)
                        .auto_rounds(false);
                    macro_rules! add {
                        ($o:literal,$legacy:literal,$id:expr) => {{
                            preflight!($o, $legacy, $id);
                            if $flt.keep_arm(&name, $id) {
                                add_bench::<$float, $n, $w, $dependent>(
                                    g,
                                    $id,
                                    Arc::clone(&data),
                                    move |a, b, out| $stream::<$o, $op, $legacy, $n>($t, a, b, out),
                                    move |x| $chain::<$o, $op, $legacy>($t, x),
                                );
                            }
                        }};
                    }
                    add!(0, true, concat!("existing Magetypes ", $operation));
                    macro_rules! add_case {
                        ($o:literal,$id:literal) => {
                            add!($o, false, $id);
                        };
                    }
                    $cases!(add_case);
                });
            }
        }
    }};
}
macro_rules! width {
    ($suite:ident,$flt:ident,$tier:ident,$t:ident,$stream:ident,$chain:ident,$float:ty,$w:literal,$operation:literal,$recip:literal,$op:literal,$cases:ident) => {{
        group!(
            $suite, $flt, $tier, $t, $stream, $chain, $float, $w, $operation, $recip, $op, 64,
            false, false, $cases, "stream64"
        );
        group!(
            $suite,
            $flt,
            $tier,
            $t,
            $stream,
            $chain,
            $float,
            $w,
            $operation,
            $recip,
            $op,
            4096,
            false,
            false,
            $cases,
            "stream4096"
        );
        group!(
            $suite, $flt, $tier, $t, $stream, $chain, $float, $w, $operation, $recip, $op, 64,
            true, false, $cases, "chain64"
        );
    }};
}
fn register_mm4_scalar(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: ScalarToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm4_scalar,
        chain_mm4_scalar,
        f32,
        4,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm4_scalar,
        chain_mm4_scalar,
        f32,
        4,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm4_scalar,
        chain_mm4_scalar,
        f32,
        4,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm4_scalar,
        chain_mm4_scalar,
        f32,
        4,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

fn register_mm8_scalar(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: ScalarToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm8_scalar,
        chain_mm8_scalar,
        f32,
        8,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm8_scalar,
        chain_mm8_scalar,
        f32,
        8,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm8_scalar,
        chain_mm8_scalar,
        f32,
        8,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm8_scalar,
        chain_mm8_scalar,
        f32,
        8,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(feature = "w512")]
fn register_mm16_scalar(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: ScalarToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm16_scalar,
        chain_mm16_scalar,
        f32,
        16,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm16_scalar,
        chain_mm16_scalar,
        f32,
        16,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm16_scalar,
        chain_mm16_scalar,
        f32,
        16,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm16_scalar,
        chain_mm16_scalar,
        f32,
        16,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

fn register_mm64x2_scalar(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: ScalarToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x2_scalar,
        chain_mm64x2_scalar,
        f64,
        2,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x2_scalar,
        chain_mm64x2_scalar,
        f64,
        2,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x2_scalar,
        chain_mm64x2_scalar,
        f64,
        2,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x2_scalar,
        chain_mm64x2_scalar,
        f64,
        2,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

fn register_mm64x4_scalar(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: ScalarToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x4_scalar,
        chain_mm64x4_scalar,
        f64,
        4,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x4_scalar,
        chain_mm64x4_scalar,
        f64,
        4,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x4_scalar,
        chain_mm64x4_scalar,
        f64,
        4,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x4_scalar,
        chain_mm64x4_scalar,
        f64,
        4,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(feature = "w512")]
fn register_mm64x8_scalar(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: ScalarToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x8_scalar,
        chain_mm64x8_scalar,
        f64,
        8,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x8_scalar,
        chain_mm64x8_scalar,
        f64,
        8,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x8_scalar,
        chain_mm64x8_scalar,
        f64,
        8,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x8_scalar,
        chain_mm64x8_scalar,
        f64,
        8,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

fn register_rc4_scalar(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: ScalarToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc4_scalar,
        chain_rc4_scalar,
        f32,
        4,
        "recip",
        true,
        false,
        recip_cases
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc4_scalar,
        chain_rc4_scalar,
        f32,
        4,
        "rsqrt",
        true,
        true,
        recip_cases
    );
}

fn register_rc8_scalar(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: ScalarToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc8_scalar,
        chain_rc8_scalar,
        f32,
        8,
        "recip",
        true,
        false,
        recip_cases
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc8_scalar,
        chain_rc8_scalar,
        f32,
        8,
        "rsqrt",
        true,
        true,
        recip_cases
    );
}

#[cfg(feature = "w512")]
fn register_rc16_scalar(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: ScalarToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc16_scalar,
        chain_rc16_scalar,
        f32,
        16,
        "recip",
        true,
        false,
        recip_cases
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc16_scalar,
        chain_rc16_scalar,
        f32,
        16,
        "rsqrt",
        true,
        true,
        recip_cases
    );
}

#[cfg(target_arch = "x86_64")]
fn register_mm4_v3(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V3Token) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm4_v3,
        chain_mm4_v3,
        f32,
        4,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm4_v3,
        chain_mm4_v3,
        f32,
        4,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm4_v3,
        chain_mm4_v3,
        f32,
        4,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm4_v3,
        chain_mm4_v3,
        f32,
        4,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(target_arch = "x86_64")]
fn register_mm8_v3(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V3Token) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm8_v3,
        chain_mm8_v3,
        f32,
        8,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm8_v3,
        chain_mm8_v3,
        f32,
        8,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm8_v3,
        chain_mm8_v3,
        f32,
        8,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm8_v3,
        chain_mm8_v3,
        f32,
        8,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(all(target_arch = "x86_64", feature = "w512"))]
fn register_mm16_v3(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V3Token) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm16_v3,
        chain_mm16_v3,
        f32,
        16,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm16_v3,
        chain_mm16_v3,
        f32,
        16,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm16_v3,
        chain_mm16_v3,
        f32,
        16,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm16_v3,
        chain_mm16_v3,
        f32,
        16,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(target_arch = "x86_64")]
fn register_mm64x2_v3(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V3Token) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x2_v3,
        chain_mm64x2_v3,
        f64,
        2,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x2_v3,
        chain_mm64x2_v3,
        f64,
        2,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x2_v3,
        chain_mm64x2_v3,
        f64,
        2,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x2_v3,
        chain_mm64x2_v3,
        f64,
        2,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(target_arch = "x86_64")]
fn register_mm64x4_v3(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V3Token) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x4_v3,
        chain_mm64x4_v3,
        f64,
        4,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x4_v3,
        chain_mm64x4_v3,
        f64,
        4,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x4_v3,
        chain_mm64x4_v3,
        f64,
        4,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x4_v3,
        chain_mm64x4_v3,
        f64,
        4,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(all(target_arch = "x86_64", feature = "w512"))]
fn register_mm64x8_v3(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V3Token) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x8_v3,
        chain_mm64x8_v3,
        f64,
        8,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x8_v3,
        chain_mm64x8_v3,
        f64,
        8,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x8_v3,
        chain_mm64x8_v3,
        f64,
        8,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x8_v3,
        chain_mm64x8_v3,
        f64,
        8,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(target_arch = "x86_64")]
fn register_rc4_v3(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V3Token) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc4_v3,
        chain_rc4_v3,
        f32,
        4,
        "recip",
        true,
        false,
        recip_cases
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc4_v3,
        chain_rc4_v3,
        f32,
        4,
        "rsqrt",
        true,
        true,
        recip_cases
    );
}

#[cfg(target_arch = "x86_64")]
fn register_rc8_v3(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V3Token) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc8_v3,
        chain_rc8_v3,
        f32,
        8,
        "recip",
        true,
        false,
        recip_cases
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc8_v3,
        chain_rc8_v3,
        f32,
        8,
        "rsqrt",
        true,
        true,
        recip_cases
    );
}

#[cfg(all(target_arch = "x86_64", feature = "w512"))]
fn register_rc16_v3(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V3Token) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc16_v3,
        chain_rc16_v3,
        f32,
        16,
        "recip",
        true,
        false,
        recip_cases
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc16_v3,
        chain_rc16_v3,
        f32,
        16,
        "rsqrt",
        true,
        true,
        recip_cases
    );
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn register_mm4_v4(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V4Token) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm4_v4,
        chain_mm4_v4,
        f32,
        4,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm4_v4,
        chain_mm4_v4,
        f32,
        4,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm4_v4,
        chain_mm4_v4,
        f32,
        4,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm4_v4,
        chain_mm4_v4,
        f32,
        4,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn register_mm8_v4(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V4Token) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm8_v4,
        chain_mm8_v4,
        f32,
        8,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm8_v4,
        chain_mm8_v4,
        f32,
        8,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm8_v4,
        chain_mm8_v4,
        f32,
        8,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm8_v4,
        chain_mm8_v4,
        f32,
        8,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "w512"))]
fn register_mm16_v4(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V4Token) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm16_v4,
        chain_mm16_v4,
        f32,
        16,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm16_v4,
        chain_mm16_v4,
        f32,
        16,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm16_v4,
        chain_mm16_v4,
        f32,
        16,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm16_v4,
        chain_mm16_v4,
        f32,
        16,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn register_mm64x2_v4(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V4Token) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x2_v4,
        chain_mm64x2_v4,
        f64,
        2,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x2_v4,
        chain_mm64x2_v4,
        f64,
        2,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x2_v4,
        chain_mm64x2_v4,
        f64,
        2,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x2_v4,
        chain_mm64x2_v4,
        f64,
        2,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn register_mm64x4_v4(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V4Token) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x4_v4,
        chain_mm64x4_v4,
        f64,
        4,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x4_v4,
        chain_mm64x4_v4,
        f64,
        4,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x4_v4,
        chain_mm64x4_v4,
        f64,
        4,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x4_v4,
        chain_mm64x4_v4,
        f64,
        4,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "w512"))]
fn register_mm64x8_v4(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V4Token) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x8_v4,
        chain_mm64x8_v4,
        f64,
        8,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x8_v4,
        chain_mm64x8_v4,
        f64,
        8,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x8_v4,
        chain_mm64x8_v4,
        f64,
        8,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x8_v4,
        chain_mm64x8_v4,
        f64,
        8,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn register_rc4_v4(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V4Token) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc4_v4,
        chain_rc4_v4,
        f32,
        4,
        "recip",
        true,
        false,
        recip_cases
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc4_v4,
        chain_rc4_v4,
        f32,
        4,
        "rsqrt",
        true,
        true,
        recip_cases
    );
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn register_rc8_v4(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V4Token) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc8_v4,
        chain_rc8_v4,
        f32,
        8,
        "recip",
        true,
        false,
        recip_cases
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc8_v4,
        chain_rc8_v4,
        f32,
        8,
        "rsqrt",
        true,
        true,
        recip_cases
    );
}

#[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "w512"))]
fn register_rc16_v4(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V4Token) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc16_v4,
        chain_rc16_v4,
        f32,
        16,
        "recip",
        true,
        false,
        recip_cases
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc16_v4,
        chain_rc16_v4,
        f32,
        16,
        "rsqrt",
        true,
        true,
        recip_cases
    );
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn register_mm4_v4x(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V4xToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm4_v4x,
        chain_mm4_v4x,
        f32,
        4,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm4_v4x,
        chain_mm4_v4x,
        f32,
        4,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm4_v4x,
        chain_mm4_v4x,
        f32,
        4,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm4_v4x,
        chain_mm4_v4x,
        f32,
        4,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn register_mm8_v4x(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V4xToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm8_v4x,
        chain_mm8_v4x,
        f32,
        8,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm8_v4x,
        chain_mm8_v4x,
        f32,
        8,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm8_v4x,
        chain_mm8_v4x,
        f32,
        8,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm8_v4x,
        chain_mm8_v4x,
        f32,
        8,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "w512"))]
fn register_mm16_v4x(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V4xToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm16_v4x,
        chain_mm16_v4x,
        f32,
        16,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm16_v4x,
        chain_mm16_v4x,
        f32,
        16,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm16_v4x,
        chain_mm16_v4x,
        f32,
        16,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm16_v4x,
        chain_mm16_v4x,
        f32,
        16,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn register_mm64x2_v4x(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V4xToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x2_v4x,
        chain_mm64x2_v4x,
        f64,
        2,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x2_v4x,
        chain_mm64x2_v4x,
        f64,
        2,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x2_v4x,
        chain_mm64x2_v4x,
        f64,
        2,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x2_v4x,
        chain_mm64x2_v4x,
        f64,
        2,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn register_mm64x4_v4x(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V4xToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x4_v4x,
        chain_mm64x4_v4x,
        f64,
        4,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x4_v4x,
        chain_mm64x4_v4x,
        f64,
        4,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x4_v4x,
        chain_mm64x4_v4x,
        f64,
        4,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x4_v4x,
        chain_mm64x4_v4x,
        f64,
        4,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "w512"))]
fn register_mm64x8_v4x(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V4xToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x8_v4x,
        chain_mm64x8_v4x,
        f64,
        8,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x8_v4x,
        chain_mm64x8_v4x,
        f64,
        8,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x8_v4x,
        chain_mm64x8_v4x,
        f64,
        8,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x8_v4x,
        chain_mm64x8_v4x,
        f64,
        8,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn register_rc4_v4x(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V4xToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc4_v4x,
        chain_rc4_v4x,
        f32,
        4,
        "recip",
        true,
        false,
        recip_cases
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc4_v4x,
        chain_rc4_v4x,
        f32,
        4,
        "rsqrt",
        true,
        true,
        recip_cases
    );
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn register_rc8_v4x(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V4xToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc8_v4x,
        chain_rc8_v4x,
        f32,
        8,
        "recip",
        true,
        false,
        recip_cases
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc8_v4x,
        chain_rc8_v4x,
        f32,
        8,
        "rsqrt",
        true,
        true,
        recip_cases
    );
}

#[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "w512"))]
fn register_rc16_v4x(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::X64V4xToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc16_v4x,
        chain_rc16_v4x,
        f32,
        16,
        "recip",
        true,
        false,
        recip_cases
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc16_v4x,
        chain_rc16_v4x,
        f32,
        16,
        "rsqrt",
        true,
        true,
        recip_cases
    );
}

#[cfg(target_arch = "aarch64")]
fn register_mm4_neon(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::NeonToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm4_neon,
        chain_mm4_neon,
        f32,
        4,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm4_neon,
        chain_mm4_neon,
        f32,
        4,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm4_neon,
        chain_mm4_neon,
        f32,
        4,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm4_neon,
        chain_mm4_neon,
        f32,
        4,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(target_arch = "aarch64")]
fn register_mm8_neon(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::NeonToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm8_neon,
        chain_mm8_neon,
        f32,
        8,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm8_neon,
        chain_mm8_neon,
        f32,
        8,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm8_neon,
        chain_mm8_neon,
        f32,
        8,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm8_neon,
        chain_mm8_neon,
        f32,
        8,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(all(target_arch = "aarch64", feature = "w512"))]
fn register_mm16_neon(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::NeonToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm16_neon,
        chain_mm16_neon,
        f32,
        16,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm16_neon,
        chain_mm16_neon,
        f32,
        16,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm16_neon,
        chain_mm16_neon,
        f32,
        16,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm16_neon,
        chain_mm16_neon,
        f32,
        16,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(target_arch = "aarch64")]
fn register_mm64x2_neon(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::NeonToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x2_neon,
        chain_mm64x2_neon,
        f64,
        2,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x2_neon,
        chain_mm64x2_neon,
        f64,
        2,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x2_neon,
        chain_mm64x2_neon,
        f64,
        2,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x2_neon,
        chain_mm64x2_neon,
        f64,
        2,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(target_arch = "aarch64")]
fn register_mm64x4_neon(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::NeonToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x4_neon,
        chain_mm64x4_neon,
        f64,
        4,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x4_neon,
        chain_mm64x4_neon,
        f64,
        4,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x4_neon,
        chain_mm64x4_neon,
        f64,
        4,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x4_neon,
        chain_mm64x4_neon,
        f64,
        4,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(all(target_arch = "aarch64", feature = "w512"))]
fn register_mm64x8_neon(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::NeonToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x8_neon,
        chain_mm64x8_neon,
        f64,
        8,
        "min",
        false,
        false,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x8_neon,
        chain_mm64x8_neon,
        f64,
        8,
        "min",
        false,
        false,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x8_neon,
        chain_mm64x8_neon,
        f64,
        8,
        "max",
        false,
        true,
        minmax_cases
    );
    group!(
        suite,
        flt,
        tier,
        t,
        stream_mm64x8_neon,
        chain_mm64x8_neon,
        f64,
        8,
        "max",
        false,
        true,
        64,
        false,
        true,
        zero_cases,
        "zero-clamp64"
    );
}

#[cfg(target_arch = "aarch64")]
fn register_rc4_neon(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::NeonToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc4_neon,
        chain_rc4_neon,
        f32,
        4,
        "recip",
        true,
        false,
        recip_cases
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc4_neon,
        chain_rc4_neon,
        f32,
        4,
        "rsqrt",
        true,
        true,
        recip_cases
    );
}

#[cfg(target_arch = "aarch64")]
fn register_rc8_neon(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::NeonToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc8_neon,
        chain_rc8_neon,
        f32,
        8,
        "recip",
        true,
        false,
        recip_cases
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc8_neon,
        chain_rc8_neon,
        f32,
        8,
        "rsqrt",
        true,
        true,
        recip_cases
    );
}

#[cfg(all(target_arch = "aarch64", feature = "w512"))]
fn register_rc16_neon(suite: &mut Suite, flt: &BenchFilter, tier: &str, t: archmage::NeonToken) {
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc16_neon,
        chain_rc16_neon,
        f32,
        16,
        "recip",
        true,
        false,
        recip_cases
    );
    width!(
        suite,
        flt,
        tier,
        t,
        stream_rc16_neon,
        chain_rc16_neon,
        f32,
        16,
        "rsqrt",
        true,
        true,
        recip_cases
    );
}

fn main() {
    if std::env::args().any(|x| x == "--cpu-set") {
        let mut classes = zenbench::platform::detect_cpu_topology().classes;
        classes.sort_by(|a, b| {
            let rank = |s: &str| {
                if s.starts_with("super") {
                    0
                } else if s.starts_with("performance") {
                    1
                } else {
                    2
                }
            };
            rank(&a.name)
                .cmp(&rank(&b.name))
                .then_with(|| a.name.cmp(&b.name))
        });
        println!(
            "{}",
            classes[0]
                .cpus
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        return;
    }
    let args: Vec<_> = std::env::args().collect();
    let tier = args
        .iter()
        .position(|x| x == "--policy-tier")
        .and_then(|i| args.get(i + 1))
        .expect("--policy-tier")
        .clone();
    let exploratory = args.iter().any(|x| x == "--exploratory-macos");
    let iterate_flag = args.iter().any(|x| x == "--iterate");
    let flt = BenchFilter::from_args(&args, iterate_flag);
    let register = |suite: &mut Suite| match tier.as_str() {
        "scalar" => {
            let t = ScalarToken;
            register_mm4_scalar(suite, &flt, &tier, t);
            register_mm8_scalar(suite, &flt, &tier, t);
            #[cfg(feature = "w512")]
            register_mm16_scalar(suite, &flt, &tier, t);
            register_mm64x2_scalar(suite, &flt, &tier, t);
            register_mm64x4_scalar(suite, &flt, &tier, t);
            #[cfg(feature = "w512")]
            register_mm64x8_scalar(suite, &flt, &tier, t);
            register_rc4_scalar(suite, &flt, &tier, t);
            register_rc8_scalar(suite, &flt, &tier, t);
            #[cfg(feature = "w512")]
            register_rc16_scalar(suite, &flt, &tier, t);
        }
        #[cfg(target_arch = "x86_64")]
        "v3" => {
            let t = archmage::X64V3Token::summon().expect("V3");
            register_mm4_v3(suite, &flt, &tier, t);
            register_mm8_v3(suite, &flt, &tier, t);
            #[cfg(feature = "w512")]
            register_mm16_v3(suite, &flt, &tier, t);
            register_mm64x2_v3(suite, &flt, &tier, t);
            register_mm64x4_v3(suite, &flt, &tier, t);
            #[cfg(feature = "w512")]
            register_mm64x8_v3(suite, &flt, &tier, t);
            register_rc4_v3(suite, &flt, &tier, t);
            register_rc8_v3(suite, &flt, &tier, t);
            #[cfg(feature = "w512")]
            register_rc16_v3(suite, &flt, &tier, t);
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        "v4" => {
            let t = archmage::X64V4Token::summon().expect("V4");
            register_mm4_v4(suite, &flt, &tier, t);
            register_mm8_v4(suite, &flt, &tier, t);
            #[cfg(feature = "w512")]
            register_mm16_v4(suite, &flt, &tier, t);
            register_mm64x2_v4(suite, &flt, &tier, t);
            register_mm64x4_v4(suite, &flt, &tier, t);
            #[cfg(feature = "w512")]
            register_mm64x8_v4(suite, &flt, &tier, t);
            register_rc4_v4(suite, &flt, &tier, t);
            register_rc8_v4(suite, &flt, &tier, t);
            #[cfg(feature = "w512")]
            register_rc16_v4(suite, &flt, &tier, t);
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        "v4x" => {
            let t = archmage::X64V4xToken::summon().expect("V4x");
            register_mm4_v4x(suite, &flt, &tier, t);
            register_mm8_v4x(suite, &flt, &tier, t);
            #[cfg(feature = "w512")]
            register_mm16_v4x(suite, &flt, &tier, t);
            register_mm64x2_v4x(suite, &flt, &tier, t);
            register_mm64x4_v4x(suite, &flt, &tier, t);
            #[cfg(feature = "w512")]
            register_mm64x8_v4x(suite, &flt, &tier, t);
            register_rc4_v4x(suite, &flt, &tier, t);
            register_rc8_v4x(suite, &flt, &tier, t);
            #[cfg(feature = "w512")]
            register_rc16_v4x(suite, &flt, &tier, t);
        }
        #[cfg(target_arch = "aarch64")]
        "neon" => {
            let t = archmage::NeonToken::summon().expect("NEON");
            register_mm4_neon(suite, &flt, &tier, t);
            register_mm8_neon(suite, &flt, &tier, t);
            #[cfg(feature = "w512")]
            register_mm16_neon(suite, &flt, &tier, t);
            register_mm64x2_neon(suite, &flt, &tier, t);
            register_mm64x4_neon(suite, &flt, &tier, t);
            #[cfg(feature = "w512")]
            register_mm64x8_neon(suite, &flt, &tier, t);
            register_rc4_neon(suite, &flt, &tier, t);
            register_rc8_neon(suite, &flt, &tier, t);
            #[cfg(feature = "w512")]
            register_rc16_neon(suite, &flt, &tier, t);
        }
        _ => panic!("unavailable policy tier"),
    };
    if args.iter().any(|x| x == "--verify-only") {
        register(&mut Suite::new());
        return;
    }
    assert!(
        args.iter().any(|x| x == "--measure"),
        "timing requires --measure"
    );
    assert!(!exploratory || cfg!(target_os = "macos"));
    let fast_iter = exploratory || iterate_flag;
    let mut gate = zenbench::GateConfig::strict();
    if fast_iter {
        gate.strict = false;
    }
    let result = zenbench::run_gated(gate, |suite| {
        let p = suite.plan();
        p.mode = if fast_iter {
            zenbench::BenchmarkMode::Iterate
        } else {
            zenbench::BenchmarkMode::Confirm
        };
        p.processes = if fast_iter { 1 } else { 5 };
        p.rounds = Some(flt.rounds);
        p.margin_pct = 2.0;
        p.confidence = 0.95;
        p.workload_id =
            Some("public-primitive-canonical-flags-stream64-4096-chain64-zero-clamp-v1".into());
        p.scope = Some(format!("primitive-policies-{tier}"));
        register(suite);
    });
    if std::env::var("ZENBENCH_SUBPROCESS").as_deref() == Ok("1") {
        return;
    }
    result
        .save(std::env::var("ARCHMAGE_PRIMITIVE_RESULTS").expect("durable results path"))
        .unwrap();
    result.print_report();
    if exploratory {
        eprintln!("EXPLORATORY M6; no uniform core-class enforcement or promotion");
    } else if iterate_flag || flt.substr.is_some() || flt.dedup.is_some() {
        eprintln!("ITERATE/SLICE RUN; non-archival codegen check");
    } else {
        assert!(
            result.confirmation_issues().is_empty(),
            "retained matrix failed confirmation: {:?}",
            result.confirmation_issues()
        );
    }
}

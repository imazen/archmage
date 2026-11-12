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
use std::sync::Arc;
use workload::*;
use zenbench::{Suite, black_box};

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
    ($suite:ident,$tier:ident,$t:ident,$stream:ident,$chain:ident,$float:ty,$w:literal,$operation:literal,$recip:literal,$op:literal,$n:literal,$dependent:literal,$zero:literal,$cases:ident,$kind:literal) => {{
        let name = format!(
            "{}/{}/{}x{}/{}",
            $tier,
            $operation,
            stringify!($float),
            $w,
            $kind
        );
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
                g.config().min_rounds(50).max_rounds(50).auto_rounds(false);
                macro_rules! add {
                    ($o:literal,$legacy:literal,$id:expr) => {{
                        preflight!($o, $legacy, $id);
                        add_bench::<$float, $n, $w, $dependent>(
                            g,
                            $id,
                            Arc::clone(&data),
                            move |a, b, out| $stream::<$o, $op, $legacy, $n>($t, a, b, out),
                            move |x| $chain::<$o, $op, $legacy>($t, x),
                        );
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
    }};
}
macro_rules! width {
    ($suite:ident,$tier:ident,$t:ident,$stream:ident,$chain:ident,$float:ty,$w:literal,$operation:literal,$recip:literal,$op:literal,$cases:ident) => {{
        group!(
            $suite, $tier, $t, $stream, $chain, $float, $w, $operation, $recip, $op, 64, false,
            false, $cases, "stream64"
        );
        group!(
            $suite,
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
            $suite, $tier, $t, $stream, $chain, $float, $w, $operation, $recip, $op, 64, true,
            false, $cases, "chain64"
        );
    }};
}
macro_rules! matrix {
    ($suite:ident,$tier:ident,$token:expr,$smm4:ident,$cmm4:ident,$smm8:ident,$cmm8:ident,$smm16:ident,$cmm16:ident,$smm64x2:ident,$cmm64x2:ident,$smm64x4:ident,$cmm64x4:ident,$smm64x8:ident,$cmm64x8:ident,$src4:ident,$crc4:ident,$src8:ident,$crc8:ident,$src16:ident,$crc16:ident) => {{
        let t = $token;
        width!(
            $suite,
            $tier,
            t,
            $smm4,
            $cmm4,
            f32,
            4,
            "min",
            false,
            false,
            minmax_cases
        );
        group!(
            $suite,
            $tier,
            t,
            $smm4,
            $cmm4,
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
            $suite,
            $tier,
            t,
            $smm4,
            $cmm4,
            f32,
            4,
            "max",
            false,
            true,
            minmax_cases
        );
        group!(
            $suite,
            $tier,
            t,
            $smm4,
            $cmm4,
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
        width!(
            $suite,
            $tier,
            t,
            $smm8,
            $cmm8,
            f32,
            8,
            "min",
            false,
            false,
            minmax_cases
        );
        group!(
            $suite,
            $tier,
            t,
            $smm8,
            $cmm8,
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
            $suite,
            $tier,
            t,
            $smm8,
            $cmm8,
            f32,
            8,
            "max",
            false,
            true,
            minmax_cases
        );
        group!(
            $suite,
            $tier,
            t,
            $smm8,
            $cmm8,
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
        #[cfg(feature = "w512")]
        width!(
            $suite,
            $tier,
            t,
            $smm16,
            $cmm16,
            f32,
            16,
            "min",
            false,
            false,
            minmax_cases
        );
        #[cfg(feature = "w512")]
        group!(
            $suite,
            $tier,
            t,
            $smm16,
            $cmm16,
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
        #[cfg(feature = "w512")]
        width!(
            $suite,
            $tier,
            t,
            $smm16,
            $cmm16,
            f32,
            16,
            "max",
            false,
            true,
            minmax_cases
        );
        #[cfg(feature = "w512")]
        group!(
            $suite,
            $tier,
            t,
            $smm16,
            $cmm16,
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
        width!(
            $suite,
            $tier,
            t,
            $smm64x2,
            $cmm64x2,
            f64,
            2,
            "min",
            false,
            false,
            minmax_cases
        );
        group!(
            $suite,
            $tier,
            t,
            $smm64x2,
            $cmm64x2,
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
            $suite,
            $tier,
            t,
            $smm64x2,
            $cmm64x2,
            f64,
            2,
            "max",
            false,
            true,
            minmax_cases
        );
        group!(
            $suite,
            $tier,
            t,
            $smm64x2,
            $cmm64x2,
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
        width!(
            $suite,
            $tier,
            t,
            $smm64x4,
            $cmm64x4,
            f64,
            4,
            "min",
            false,
            false,
            minmax_cases
        );
        group!(
            $suite,
            $tier,
            t,
            $smm64x4,
            $cmm64x4,
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
            $suite,
            $tier,
            t,
            $smm64x4,
            $cmm64x4,
            f64,
            4,
            "max",
            false,
            true,
            minmax_cases
        );
        group!(
            $suite,
            $tier,
            t,
            $smm64x4,
            $cmm64x4,
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
        #[cfg(feature = "w512")]
        width!(
            $suite,
            $tier,
            t,
            $smm64x8,
            $cmm64x8,
            f64,
            8,
            "min",
            false,
            false,
            minmax_cases
        );
        #[cfg(feature = "w512")]
        group!(
            $suite,
            $tier,
            t,
            $smm64x8,
            $cmm64x8,
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
        #[cfg(feature = "w512")]
        width!(
            $suite,
            $tier,
            t,
            $smm64x8,
            $cmm64x8,
            f64,
            8,
            "max",
            false,
            true,
            minmax_cases
        );
        #[cfg(feature = "w512")]
        group!(
            $suite,
            $tier,
            t,
            $smm64x8,
            $cmm64x8,
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
        width!(
            $suite,
            $tier,
            t,
            $src4,
            $crc4,
            f32,
            4,
            "recip",
            true,
            false,
            recip_cases
        );
        width!(
            $suite,
            $tier,
            t,
            $src4,
            $crc4,
            f32,
            4,
            "rsqrt",
            true,
            true,
            recip_cases
        );
        width!(
            $suite,
            $tier,
            t,
            $src8,
            $crc8,
            f32,
            8,
            "recip",
            true,
            false,
            recip_cases
        );
        width!(
            $suite,
            $tier,
            t,
            $src8,
            $crc8,
            f32,
            8,
            "rsqrt",
            true,
            true,
            recip_cases
        );
        #[cfg(feature = "w512")]
        width!(
            $suite,
            $tier,
            t,
            $src16,
            $crc16,
            f32,
            16,
            "recip",
            true,
            false,
            recip_cases
        );
        #[cfg(feature = "w512")]
        width!(
            $suite,
            $tier,
            t,
            $src16,
            $crc16,
            f32,
            16,
            "rsqrt",
            true,
            true,
            recip_cases
        );
    }};
}

fn main() {
    if std::env::args().any(|x| x == "--cpu-set") {
        let mut classes = zenbench::platform::detect_cpu_topology().classes;
        classes.sort_by(|a, b| a.name.cmp(&b.name));
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
    let register = |suite: &mut Suite| match tier.as_str() {
        "scalar" => matrix!(
            suite,
            tier,
            ScalarToken,
            stream_mm4_scalar,
            chain_mm4_scalar,
            stream_mm8_scalar,
            chain_mm8_scalar,
            stream_mm16_scalar,
            chain_mm16_scalar,
            stream_mm64x2_scalar,
            chain_mm64x2_scalar,
            stream_mm64x4_scalar,
            chain_mm64x4_scalar,
            stream_mm64x8_scalar,
            chain_mm64x8_scalar,
            stream_rc4_scalar,
            chain_rc4_scalar,
            stream_rc8_scalar,
            chain_rc8_scalar,
            stream_rc16_scalar,
            chain_rc16_scalar
        ),
        #[cfg(target_arch = "x86_64")]
        "v3" => matrix!(
            suite,
            tier,
            archmage::X64V3Token::summon().expect("V3"),
            stream_mm4_v3,
            chain_mm4_v3,
            stream_mm8_v3,
            chain_mm8_v3,
            stream_mm16_v3,
            chain_mm16_v3,
            stream_mm64x2_v3,
            chain_mm64x2_v3,
            stream_mm64x4_v3,
            chain_mm64x4_v3,
            stream_mm64x8_v3,
            chain_mm64x8_v3,
            stream_rc4_v3,
            chain_rc4_v3,
            stream_rc8_v3,
            chain_rc8_v3,
            stream_rc16_v3,
            chain_rc16_v3
        ),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        "v4" => matrix!(
            suite,
            tier,
            archmage::X64V4Token::summon().expect("V4"),
            stream_mm4_v4,
            chain_mm4_v4,
            stream_mm8_v4,
            chain_mm8_v4,
            stream_mm16_v4,
            chain_mm16_v4,
            stream_mm64x2_v4,
            chain_mm64x2_v4,
            stream_mm64x4_v4,
            chain_mm64x4_v4,
            stream_mm64x8_v4,
            chain_mm64x8_v4,
            stream_rc4_v4,
            chain_rc4_v4,
            stream_rc8_v4,
            chain_rc8_v4,
            stream_rc16_v4,
            chain_rc16_v4
        ),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        "v4x" => matrix!(
            suite,
            tier,
            archmage::X64V4xToken::summon().expect("V4x"),
            stream_mm4_v4x,
            chain_mm4_v4x,
            stream_mm8_v4x,
            chain_mm8_v4x,
            stream_mm16_v4x,
            chain_mm16_v4x,
            stream_mm64x2_v4x,
            chain_mm64x2_v4x,
            stream_mm64x4_v4x,
            chain_mm64x4_v4x,
            stream_mm64x8_v4x,
            chain_mm64x8_v4x,
            stream_rc4_v4x,
            chain_rc4_v4x,
            stream_rc8_v4x,
            chain_rc8_v4x,
            stream_rc16_v4x,
            chain_rc16_v4x
        ),
        #[cfg(target_arch = "aarch64")]
        "neon" => matrix!(
            suite,
            tier,
            archmage::NeonToken::summon().expect("NEON"),
            stream_mm4_neon,
            chain_mm4_neon,
            stream_mm8_neon,
            chain_mm8_neon,
            stream_mm16_neon,
            chain_mm16_neon,
            stream_mm64x2_neon,
            chain_mm64x2_neon,
            stream_mm64x4_neon,
            chain_mm64x4_neon,
            stream_mm64x8_neon,
            chain_mm64x8_neon,
            stream_rc4_neon,
            chain_rc4_neon,
            stream_rc8_neon,
            chain_rc8_neon,
            stream_rc16_neon,
            chain_rc16_neon
        ),
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
    let exploratory = args.iter().any(|x| x == "--exploratory-macos");
    assert!(!exploratory || cfg!(target_os = "macos"));
    let mut gate = zenbench::GateConfig::strict();
    if exploratory {
        gate.strict = false;
    }
    let result = zenbench::run_gated(gate, |suite| {
        let p = suite.plan();
        p.mode = if exploratory {
            zenbench::BenchmarkMode::Iterate
        } else {
            zenbench::BenchmarkMode::Confirm
        };
        p.processes = if exploratory { 1 } else { 5 };
        p.rounds = Some(50);
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
    } else {
        assert!(
            result.confirmation_issues().is_empty(),
            "retained matrix failed confirmation: {:?}",
            result.confirmation_issues()
        );
    }
}

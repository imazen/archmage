//! Fixed independent-process qualification matrix. All results are retained.
use crate::recip_workload::*;
use archmage::ScalarToken;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
use archmage::SimdToken;
use zenbench::{GateConfig, black_box};
macro_rules! group {
    ($suite:ident,$name:expr,$t:ident,$stream:ident,$chain:ident,$n:literal,$dependent:literal,$distribution:literal) => {{
        $suite.group($name, |g| {
            g.config().min_rounds(50).max_rounds(50).auto_rounds(false);
            let input: [f32; $n] = core::array::from_fn(|i| input(i, $distribution));
            macro_rules! arm {
                ($label:literal,$mode:literal) => {
                    g.bench($label, move |b| {
                        let mut out = [0.0; $n];
                        b.iter(|| {
                            if $dependent {
                                black_box($chain::<$mode>($t, black_box(0.75)));
                            } else {
                                $stream::<$mode, $n>($t, black_box(&input), black_box(&mut out));
                                black_box(&out);
                            }
                        });
                    });
                };
            }
            arm!("magetypes recip", 0);
            if std::env::args().any(|a| a == "--cold-study") {
                arm!("portable HIGHP division", 3);
                arm!("RN14 native guarded inline", 12);
                arm!("RN14 division guarded inline", 13);
                arm!("RN14 native guarded cold", 14);
                arm!("RN14 division guarded cold", 15);
            } else if std::env::args().any(|a| a == "--certified-study" || a == "--cold-study") {
                arm!("portable serial MIDP", 2);
                arm!("bounded f64 MIDP (same fused bits)", 10);
                arm!("portable HIGHP division", 3);
                arm!("RN14 native always certified", 11);
                arm!("RN14 native guarded", 12);
                arm!("RN14 division guarded", 13);
            } else {
                arm!("portable parallel MIDP", 1);
                arm!("portable serial MIDP", 2);
                arm!("portable HIGHP division", 3);
                arm!("canonical RCP14", 4);
            }
        });
    }};
}
macro_rules! width {
    ($suite:ident,$tier:ident,$t:ident,$stream:ident,$chain:ident,$w:literal) => {{
        group!(
            $suite,
            format!("{}/f32x{}/stream64", $tier, $w),
            $t,
            $stream,
            $chain,
            64,
            false,
            0
        );
        group!(
            $suite,
            format!("{}/f32x{}/stream4096", $tier, $w),
            $t,
            $stream,
            $chain,
            4096,
            false,
            0
        );
        group!(
            $suite,
            format!("{}/f32x{}/chain64", $tier, $w),
            $t,
            $stream,
            $chain,
            64,
            true,
            0
        );
    }};
}
macro_rules! matrix {
    ($suite:ident,$tier:ident,$token:expr,$s4:ident,$c4:ident,$s8:ident,$c8:ident,$s16:ident,$c16:ident) => {{
        let t = $token;
        width!($suite, $tier, t, $s4, $c4, 4);
        width!($suite, $tier, t, $s8, $c8, 8);
        #[cfg(feature = "w512")]
        width!($suite, $tier, t, $s16, $c16, 16);
        if std::env::args().any(|a| a == "--certified-study" || a == "--cold-study") {
            group!(
                $suite,
                format!("{}/f32x8/bits64", $tier),
                t,
                $s8,
                $c8,
                64,
                false,
                1
            );
            group!(
                $suite,
                format!("{}/f32x8/bits4096", $tier),
                t,
                $s8,
                $c8,
                4096,
                false,
                1
            );
            group!(
                $suite,
                format!("{}/f32x8/boundary64", $tier),
                t,
                $s8,
                $c8,
                64,
                false,
                2
            );
            group!(
                $suite,
                format!("{}/f32x8/boundary4096", $tier),
                t,
                $s8,
                $c8,
                4096,
                false,
                2
            );
        }
    }};
}
pub(crate) fn run() {
    let args: Vec<_> = std::env::args().collect();
    let tier = args
        .iter()
        .position(|a| a == "--qual-tier")
        .and_then(|i| args.get(i + 1))
        .expect("--qual-tier required")
        .clone();
    let exploratory = args.iter().any(|a| a == "--exploratory-macos");
    assert!(
        !exploratory || cfg!(target_os = "macos"),
        "exploratory mode is only for the explicitly unconfined macOS study"
    );
    let mut gate = GateConfig::strict();
    // macOS cannot enforce heterogeneous core classes. Retain all monitoring and
    // dirty rounds, but classify these user-authorized timings as exploratory.
    if exploratory {
        gate.strict = false;
    }
    let result = zenbench::run_gated(gate, |suite| {
        let plan = suite.plan();
        plan.mode = if exploratory {
            zenbench::BenchmarkMode::Iterate
        } else {
            zenbench::BenchmarkMode::Confirm
        };
        plan.processes = if exploratory { 1 } else { 5 };
        plan.rounds = Some(50);
        plan.margin_pct = 2.0;
        plan.confidence = 0.95;
        plan.workload_id = Some(
            if std::env::args().any(|a| a == "--certified-study" || a == "--cold-study") {
                "recip-certified-normal-signed-ordinary-bits-boundary64-4096-chain64-v1"
            } else {
                "recip-normal-signed-period64-stream64-stream4096-positive-chain64-v1"
            }
            .into(),
        );
        plan.scope = Some(format!("recip-qualification-{tier}"));
        match tier.as_str() {
            "scalar" => matrix!(
                suite,
                tier,
                ScalarToken,
                stream4_scalar,
                chain4_scalar,
                stream8_scalar,
                chain8_scalar,
                stream16_scalar,
                chain16_scalar
            ),
            #[cfg(target_arch = "x86_64")]
            "v3" => matrix!(
                suite,
                tier,
                archmage::X64V3Token::summon().expect("V3 required"),
                stream4_v3,
                chain4_v3,
                stream8_v3,
                chain8_v3,
                stream16_v3,
                chain16_v3
            ),
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            "v4" => matrix!(
                suite,
                tier,
                archmage::X64V4Token::summon().expect("V4 required"),
                stream4_v4,
                chain4_v4,
                stream8_v4,
                chain8_v4,
                stream16_v4,
                chain16_v4
            ),
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            "v4x" => matrix!(
                suite,
                tier,
                archmage::X64V4xToken::summon().expect("V4x required"),
                stream4_v4x,
                chain4_v4x,
                stream8_v4x,
                chain8_v4x,
                stream16_v4x,
                chain16_v4x
            ),
            #[cfg(target_arch = "aarch64")]
            "neon" => matrix!(
                suite,
                tier,
                archmage::NeonToken::summon().expect("NEON required"),
                stream4_neon,
                chain4_neon,
                stream8_neon,
                chain8_neon,
                stream16_neon,
                chain16_neon
            ),
            _ => panic!("qualification tier unavailable"),
        }
    });
    if std::env::var("ZENBENCH_SUBPROCESS").as_deref() == Ok("1") {
        return;
    }
    result
        .save(std::env::var("ARCHMAGE_PRIMITIVE_RESULTS").expect("durable result path"))
        .unwrap();
    result.print_report();
    if exploratory {
        eprintln!("EXPLORATORY: macOS scheduler chooses core class; excluded from promotion gate");
        return;
    }
    let issues = result.confirmation_issues();
    assert!(
        issues.is_empty(),
        "qualification measurement invalid: {issues:?}"
    );
}

//! Equal-work controls for the bounded powf API. See MATRIX.md before running.
#![forbid(unsafe_code)]
use archmage::SimdToken;
use magetypes::powf::{PowfDomain, PowfExecution, PowfOptions};
#[allow(dead_code)]
mod soft {
    include!(concat!(env!("OUT_DIR"), "/soft.rs"));
}
#[allow(dead_code, unused_variables)]
mod prevalidated {
    include!(concat!(env!("OUT_DIR"), "/prevalidated.rs"));
}
include!(concat!(env!("OUT_DIR"), "/cost_kernels.rs"));
fn matches_filter(name: &str) -> bool {
    std::env::var("POWF_FILTER").map_or(true, |f| name.contains(&f))
}
fn value(i: usize, distribution: &str) -> f32 {
    if distribution == "ordinary" {
        0.5 + (i % 193) as f32 / 128.0
    } else {
        f32::from_bits(0x3f00_0000 + ((i as u32).wrapping_mul(0x9e37_79b9) % 0x0100_0000))
    }
}
fn scalar(arm: usize, input: &[f32], output: &mut [f32], chain: bool, options: PowfOptions) {
    let n = std::hint::black_box(2.4);
    let mut acc = input[0];
    for (&base, out) in input.iter().zip(output) {
        let x = if chain { acc } else { base };
        let y = match arm {
            0 => prevalidated_call4(archmage::ScalarToken, [x; 4], n, options.domain)[0],
            1 => magetypes::powf::try_powf_runtime(
                x,
                n,
                PowfOptions {
                    execution: PowfExecution::Caller,
                    ..options
                },
            )
            .unwrap(),
            2 => magetypes::powf::try_powf_runtime(
                x,
                n,
                PowfOptions {
                    execution: PowfExecution::Automatic,
                    ..options
                },
            )
            .unwrap(),
            _ => unreachable!(),
        };
        acc = if chain { y.abs() * 0.125 + 0.5 } else { y };
        *out = acc;
    }
}
fn register1(
    suite: &mut zenbench::Suite,
    size: usize,
    chain: bool,
    domain: PowfDomain,
    distribution: &str,
) {
    let name = format!(
        "powf/x1/scalar/{size}/{}/{domain:?}/{distribution}",
        if chain { "chain" } else { "stream" }
    );
    if !matches_filter(&name) {
        return;
    }
    let input: Vec<_> = (0..size).map(|i| value(i, distribution)).collect();
    let options = PowfOptions {
        domain,
        execution: PowfExecution::Caller,
    };
    let mut expected = vec![0.0; size];
    let mut output = expected.clone();
    scalar(0, &input, &mut expected, chain, options);
    for arm in 1..3 {
        scalar(arm, &input, &mut output, chain, options);
        assert_eq!(
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "pre-timing graph equality {name} arm{arm}"
        );
    }
    suite.group(name, |g| {
        g.config()
            .min_rounds(50)
            .max_rounds(50)
            .auto_rounds(false)
            .warmup_time(std::time::Duration::from_millis(10))
            .max_time(std::time::Duration::from_millis(100))
            .max_wall_time(std::time::Duration::from_secs(60));
        for (arm, label) in ["prevalidated_graph", "checked_caller", "checked_auto"]
            .into_iter()
            .enumerate()
        {
            let input = input.clone();
            g.bench(label, move |b| {
                let mut output = vec![0.0; size];
                scalar(arm, &input, &mut output, chain, options);
                b.iter(|| {
                    scalar(
                        arm,
                        std::hint::black_box(&input),
                        std::hint::black_box(&mut output),
                        chain,
                        options,
                    )
                });
                std::hint::black_box(output);
            });
        }
    });
}
fn main() {
    assert!(!std::env::args().any(|a| a == "--no-busy-gate"));
    assert!(std::env::var("ZENBENCH_NO_BUSY_GATE").map_or(true, |v| v.is_empty() || v == "0"));
    let output = std::env::var("POWF_RESULT").expect("set POWF_RESULT to a new result path");
    assert!(
        !std::path::Path::new(&output).exists(),
        "preserve prior evidence"
    );
    let result = zenbench::run_gated(zenbench::GateConfig::strict(), |s| {
        let native = if cfg!(target_arch = "aarch64") {
            "neon"
        } else {
            "v3"
        };
        for size in [64, 4096] {
            for context in ["scalar", native] {
                for dense in [false, true] {
                    for subnormal in [false, true] {
                        reject_register4(s, context, size, dense, subnormal);
                        reject_register8(s, context, size, dense, subnormal);
                        reject_register16(s, context, size, dense, subnormal);
                    }
                }
            }
        }
        for size in [64, 4096] {
            for chain in [false, true] {
                for domain in [
                    PowfDomain::PositiveNormal,
                    PowfDomain::NonNegativeNormal,
                    PowfDomain::Gamma,
                ] {
                    for distribution in ["ordinary", "bits"] {
                        register1(s, size, chain, domain, distribution);
                        for context in ["scalar", native] {
                            register4(s, context, size, chain, domain, distribution);
                            register8(s, context, size, chain, domain, distribution);
                            register16(s, context, size, chain, domain, distribution);
                        }
                    }
                }
            }
        }
    });
    result.save(std::path::Path::new(&output)).unwrap();
}

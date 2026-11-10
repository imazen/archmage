//! Interleaved comparison of retained software, scalar APIs and std hardware.
//! Each invocation transforms 64 values; chain groups have 64 dependent steps.
#![forbid(unsafe_code)]
use archmage::ScalarToken;
use magetypes::simd::generic::{f32x4, f64x2};
use magetypes::simd::scalar::{f32x1, f64x1};
use zenbench::{GateConfig, Suite, black_box};

const N: usize = 64;

// Named probes also make the scalar instruction selection inspectable.
#[inline(never)]
pub fn scalar32_one(x: f32) -> f32 {
    f32x1(x).sqrt().0
}
#[inline(never)]
pub fn scalar64_one(x: f64) -> f64 {
    f64x1(x).sqrt().0
}

macro_rules! kernels {
    ($stream:ident,$chain:ident,$float:ident,$one:ident,$vec:ident,$width:literal,$soft:ident) => {
        #[inline(never)]
        fn $stream<const ARM: u8>(input: &[$float; N], output: &mut [$float; N]) {
            if ARM == 3 {
                for (x, y) in input
                    .as_chunks::<$width>()
                    .0
                    .iter()
                    .zip(output.as_chunks_mut::<$width>().0.iter_mut())
                {
                    *y = $vec::from_array_t(ScalarToken, *x).sqrt().to_array();
                }
            } else {
                for (x, y) in input.iter().zip(output) {
                    *y = match ARM {
                        0 => magetypes::nostd_math::$soft(*x),
                        1 => $one(*x).sqrt().0,
                        2 => x.sqrt(),
                        _ => unreachable!(),
                    };
                }
            }
        }
        #[inline(never)]
        fn $chain<const ARM: u8>(mut x: $float) -> $float {
            for _ in 0..N {
                x = match ARM {
                    0 => magetypes::nostd_math::$soft(x),
                    1 => $one(x).sqrt().0,
                    2 => x.sqrt(),
                    _ => unreachable!(),
                } + 0.5;
            }
            x
        }
    };
}
kernels!(stream32, chain32, f32, f32x1, f32x4, 4, sqrtf);
kernels!(stream64, chain64, f64, f64x1, f64x2, 2, sqrt);

fn register(suite: &mut Suite) {
    macro_rules! groups {
        ($float:ident,$stream:ident,$chain:ident,$packed:literal) => {
            suite.group(concat!(stringify!($float), "/stream64"), |g| {
                g.config().min_rounds(50).max_rounds(50).auto_rounds(false);
                let input: [$float; N] = core::array::from_fn(|i| (i + 1) as $float * 0.125);
                macro_rules! arm {
                    ($name:literal,$arm:literal) => {
                        g.bench($name, move |b| {
                            let mut output = [0.; N];
                            b.iter(|| $stream::<$arm>(black_box(&input), black_box(&mut output)));
                            black_box(output);
                        });
                    };
                }
                arm!("retained_software", 0);
                arm!("scalar_x1_selected", 1);
                arm!("std_sqrt", 2);
                arm!($packed, 3);
            });
            suite.group(concat!(stringify!($float), "/chain64"), |g| {
                g.config().min_rounds(50).max_rounds(50).auto_rounds(false);
                macro_rules! arm {
                    ($name:literal,$arm:literal) => {
                        g.bench($name, |b| {
                            b.iter(|| black_box($chain::<$arm>(black_box(0.75))));
                        });
                    };
                }
                arm!("retained_software", 0);
                arm!("scalar_x1_selected", 1);
                arm!("std_sqrt", 2);
            });
        };
    }
    groups!(f32, stream32, chain32, "scalar_x4_selected");
    groups!(f64, stream64, chain64, "scalar_x2_selected");
    // Keep the named one-value probes materialized for assembly inspection.
    black_box(scalar32_one as fn(f32) -> f32);
    black_box(scalar64_one as fn(f64) -> f64);
}

fn measure() {
    let results = zenbench::run_gated(GateConfig::strict(), register);
    if let Ok(path) = std::env::var("ARCHMAGE_SQRT_RESULTS") {
        results.save(path).unwrap();
    }
    assert!(!results.unreliable, "sqrt timing admission failed");
    for group in &results.comparisons {
        assert_eq!(group.samples.len(), 50);
        assert!(
            group.samples.iter().all(|s| s.gate_clean == Some(true)),
            "sqrt timing has flagged or unqualified rounds"
        );
    }
}

fn main() {
    // Measurement is opt-in at the caller. Cargo invokes harness-free benches
    // without a distinguishing test argument under `cargo test --all-targets`.
    if std::env::var("ARCHMAGE_SQRT_MEASURE").as_deref() != Ok("1") {
        let input32 = core::array::from_fn(|i| (i + 1) as f32 * 0.125);
        let input64 = core::array::from_fn(|i| (i + 1) as f64 * 0.125);
        macro_rules! check {
            ($stream:ident,$input:ident,$chain:ident) => {{
                let mut reference = [0.; N];
                $stream::<0>(&$input, &mut reference);
                let mut output = [0.; N];
                $stream::<1>(&$input, &mut output);
                assert_eq!(output, reference);
                $stream::<2>(&$input, &mut output);
                assert_eq!(output, reference);
                $stream::<3>(&$input, &mut output);
                assert_eq!(output, reference);
                assert_eq!($chain::<0>(0.75), $chain::<1>(0.75));
                assert_eq!($chain::<0>(0.75), $chain::<2>(0.75));
            }};
        }
        check!(stream32, input32, chain32);
        check!(stream64, input64, chain64);
        eprintln!("sqrt kernel checks passed; use `just sqrt-bench` for timings");
    } else {
        measure();
    }
}

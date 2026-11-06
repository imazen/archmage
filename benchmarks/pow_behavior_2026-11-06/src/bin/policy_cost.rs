#![forbid(unsafe_code)]
use magemath::{PowBehavior as B, PowSign};
use std::{hint::black_box, time::Duration};
use zenbench::prelude::*;
const SIGNED: u64 = B::CORRECT.sign(PowSign::SignedMagnitude).to_u64();
fn scalar<const P: u64>(x: &mut [f32], y: &[f32]) {
    for (x, y) in x.iter_mut().zip(y) {
        *x = zenmath::scalar::powf_with::<P>(*x, *y);
    }
}
fn main() {
    zenbench::run(|s| {
        s.cpu_policy(zenbench::platform::CpuPolicy::Performance);
        for (contract, scalar, packed) in [
            (
                "signed_magnitude",
                scalar::<SIGNED> as fn(&mut [f32], &[f32]),
                magemath::portable::slice::pow_f32_with::<SIGNED> as fn(&mut [f32], &[f32]),
            ),
            (
                "midp_full",
                scalar::<{ B::MIDP.to_u64() }>,
                magemath::portable::slice::pow_f32_with::<{ B::MIDP.to_u64() }>,
            ),
            (
                "lowp_full",
                scalar::<{ B::LOWP.to_u64() }>,
                magemath::portable::slice::pow_f32_with::<{ B::LOWP.to_u64() }>,
            ),
        ] {
            for workload in ["positive", "negative_integer", "mixed", "exceptional"] {
                s.group(format!("{contract}/{workload}"), |g| {
                    g.throughput(Throughput::Elements(1024));
                    g.config()
                        .warmup_time(Duration::from_millis(100))
                        .min_rounds(30)
                        .max_rounds(60)
                        .max_wall_time(Duration::from_secs(2));
                    let mut input: Vec<f32> = (0..1024)
                        .map(|i| 0.01 + 4.0 * ((i * 719) % 1024) as f32 / 1024.0)
                        .collect();
                    let mut exponent: Vec<f32> = (0..1024)
                        .map(|i| [2.0, 3.0, -2.0, -3.0, 4.0, 5.0][i % 6])
                        .collect();
                    for i in 0..1024 {
                        if workload == "negative_integer" || workload != "positive" && i % 2 == 0 {
                            input[i] = -input[i];
                        }
                        if workload == "mixed" && i % 3 == 0 {
                            exponent[i] = 2.4;
                        }
                        if workload == "exceptional" && i % 8 == 0 {
                            input[i] = [
                                f32::NAN,
                                0.0,
                                -0.0,
                                f32::INFINITY,
                                f32::NEG_INFINITY,
                                f32::from_bits(1),
                                -1.0,
                                1.0,
                            ][i / 8 % 8];
                        }
                    }
                    for (label, f) in [("scalar_policy", scalar), ("packed_policy", packed)] {
                        let input = input.clone();
                        let exponent = exponent.clone();
                        let mut out = input.clone();
                        g.bench(label, move |b| {
                            b.iter(|| {
                                out.copy_from_slice(black_box(&input));
                                f(black_box(&mut out), black_box(&exponent));
                                black_box(&out);
                            })
                        });
                    }
                    g.baseline("scalar_policy");
                });
            }
        }
    })
    .save(std::env::args().nth(1).expect("output JSON"))
    .unwrap();
}

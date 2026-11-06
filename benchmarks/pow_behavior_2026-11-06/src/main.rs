#![forbid(unsafe_code)]
mod baseline;
use archmage::prelude::*;
#[attune(_*, dispatch)]
fn branch_baseline(x: &mut [f32], y: &[f32]) {
    let token = Token::from_context();
    for (chunk, ys) in x
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(y.as_chunks::<4>().0)
    {
        *chunk = baseline::pow(token, *chunk, *ys);
    }
}
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use std::{hint::black_box, time::Duration};
    use zenbench::prelude::*;
    zenbench::run(|s| {
        s.cpu_policy(zenbench::platform::CpuPolicy::Performance);
        for workload in [
            "positive",
            "negative_integer",
            "mixed_negative",
            "exceptional",
        ] {
            s.group(workload, |g| {
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
                    if workload == "mixed_negative" && i % 3 == 0 {
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
                        exponent[i] = [0.0, -3.0, 0.5, f32::INFINITY][i / 8 % 4];
                    }
                }
                for (name, f) in [
                    ("branch_baseline", branch_baseline as fn(&mut [f32], &[f32])),
                    ("packed_sign", magemath::portable::slice::pow_f32),
                ] {
                    let input = input.clone();
                    let exponent = exponent.clone();
                    let mut out = input.clone();
                    g.bench(name, move |b| {
                        b.iter(|| {
                            out.copy_from_slice(black_box(&input));
                            f(black_box(&mut out), black_box(&exponent));
                            black_box(&out);
                        })
                    });
                }
                g.baseline("branch_baseline");
            });
        }
    })
    .save(std::env::args().nth(1).expect("output JSON"))
    .unwrap();
}
#[cfg(target_arch = "wasm32")]
fn main() {
    panic!("native timing harness");
}

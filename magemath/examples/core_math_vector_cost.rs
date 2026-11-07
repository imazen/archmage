//! Compare correctly rounded vector/scalar streams, not approximation contracts.
#[cfg(not(target_arch = "wasm32"))]
mod bench {
    use magemath::portable::slice;
    use std::{hint::black_box, time::Duration};
    use zenbench::prelude::*;
    const N: usize = 1024;
    type Op = fn(&mut [f32], &[f32]);
    fn groups(s: &mut Suite) {
        for name in ["expf", "logf", "log2f", "powf"] {
            for mixed in [false, true] {
                s.group(
                    format!("{name}/{}", if mixed { "mixed" } else { "normal" }),
                    |g| {
                        g.throughput(Throughput::Elements(N as u64));
                        g.config()
                            .warmup_time(Duration::from_millis(100))
                            .min_rounds(30)
                            .max_rounds(60)
                            .max_wall_time(Duration::from_secs(2));
                        let input: Vec<f32> = (0..N)
                            .map(|i| {
                                let t = ((i * 719) % N) as f32 / N as f32;
                                if mixed && i % 8 == 0 {
                                    f32::NAN
                                } else if name == "expf" {
                                    -8.0 + 16.0 * t
                                } else {
                                    0.01 + 4.0 * t
                                }
                            })
                            .collect();
                        let y: Vec<f32> = (0..N)
                            .map(|i| if i % 2 == 0 { 2.4 } else { -2.4 })
                            .collect();
                        let vector: Op = match name {
                            "expf" => |x, _| slice::exp_f32(x),
                            "logf" => |x, _| slice::ln_f32(x),
                            "log2f" => |x, _| slice::log2_f32(x),
                            "powf" => slice::pow_f32,
                            _ => unreachable!(),
                        };
                        let scalar: Op = match name {
                            "expf" => |x, _| {
                                for a in x {
                                    *a = zenmath::scalar::expf(*a)
                                }
                            },
                            "logf" => |x, _| {
                                for a in x {
                                    *a = zenmath::scalar::logf(*a)
                                }
                            },
                            "log2f" => |x, _| {
                                for a in x {
                                    *a = zenmath::scalar::log2f(*a)
                                }
                            },
                            "powf" => |x, y| {
                                for (a, b) in x.iter_mut().zip(y) {
                                    *a = zenmath::scalar::powf(*a, *b)
                                }
                            },
                            _ => unreachable!(),
                        };
                        for (label, f) in [("scalar_core", scalar), ("vector_core", vector)] {
                            let input = input.clone();
                            let y = y.clone();
                            let mut out = input.clone();
                            g.bench(label, move |b| {
                                b.iter(|| {
                                    out.copy_from_slice(black_box(&input));
                                    f(black_box(&mut out), black_box(&y));
                                    black_box(&out);
                                })
                            });
                        }
                        g.baseline("scalar_core");
                    },
                );
            }
        }
    }
    pub fn run() {
        let path = std::env::args().nth(1).expect("output JSON required");
        zenbench::run(groups).save(path).unwrap();
    }
}
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    bench::run();
}
#[cfg(target_arch = "wasm32")]
fn main() {
    panic!("native benchmark; WASM correctness is checked separately");
}

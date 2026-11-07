//! Compare distinct contracts: CORE-MATH, Rust std and magetypes approximations.
//! Uniform pow exponent 2.4 gives all implementations the same task.
#[cfg(not(target_arch = "wasm32"))]
mod bench {
    use archmage::prelude::*;
    use magemath::portable::slice;
    use std::{hint::black_box, time::Duration};
    use zenbench::prelude::*;
    macro_rules! method {
        ($name:ident, $op:ident $(, $arg:expr)?) => {
            // This control deliberately measures the deprecated inherent methods.
            #[attune(dispatch, define(f32x8))]
            #[allow(deprecated)]
            fn $name(data: &mut [f32], _exponents: &[f32]) {
                let token = Token::from_context();
                let (chunks, tail) = data.as_chunks_mut::<8>();
                assert!(tail.is_empty(), "benchmark requires whole vectors");
                for a in chunks {
                    *a = f32x8::from_array_t(token, *a).$op($($arg)?).to_array();
                }
            }
        };
    }
    method!(exp_mid, exp_midp);
    method!(ln_mid, ln_midp);
    method!(log2_mid, log2_midp);
    method!(pow_mid, pow_midp, 2.4);
    method!(exp_low, exp_lowp);
    method!(ln_low, ln_lowp);
    method!(log2_low, log2_lowp);
    method!(pow_low, pow_lowp, 2.4);
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
                        let y: Vec<f32> = (0..N).map(|_| 2.4).collect();
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
                        let std_op: Op = match name {
                            "expf" => |x, _| x.iter_mut().for_each(|a| *a = a.exp()),
                            "logf" => |x, _| x.iter_mut().for_each(|a| *a = a.ln()),
                            "log2f" => |x, _| x.iter_mut().for_each(|a| *a = a.log2()),
                            "powf" => |x, y| x.iter_mut().zip(y).for_each(|(a, b)| *a = a.powf(*b)),
                            _ => unreachable!(),
                        };
                        let (mid, low): (Op, Op) = match name {
                            "expf" => (exp_mid, exp_low),
                            "logf" => (ln_mid, ln_low),
                            "log2f" => (log2_mid, log2_low),
                            "powf" => (pow_mid, pow_low),
                            _ => unreachable!(),
                        };
                        for (label, f) in [
                            ("scalar_core", scalar),
                            ("vector_core", vector),
                            ("rust_std", std_op),
                            ("magetypes_midp", mid),
                            ("magetypes_lowp", low),
                        ] {
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

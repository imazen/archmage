#![forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use portable_f32_study::{CANDIDATES, eval};
    use std::{hint::black_box, time::Duration};
    use zenbench::prelude::*;
    let suite = |s: &mut Suite| {
        for (group, op, fixed) in [
            ("exp", "exp", true),
            ("ln", "ln", true),
            ("log2", "log2", true),
            ("pow_fixed_2.4", "pow", true),
            ("pow_general", "pow", false),
        ] {
            s.group(group, |g| {
                g.throughput(Throughput::Elements(1024));
                g.config()
                    .warmup_time(Duration::from_millis(100))
                    .min_rounds(30)
                    .max_rounds(60)
                    .max_wall_time(Duration::from_secs(2));
                for candidate in CANDIDATES {
                    let input: Vec<f32> = (0..1024)
                        .map(|i| {
                            let t = ((i * 719) % 1024) as f32 / 1024.0;
                            if op == "exp" {
                                -8.0 + 16.0 * t
                            } else {
                                0.01 + 4.0 * t
                            }
                        })
                        .collect();
                    let y: Vec<f32> = (0..1024)
                        .map(|i| {
                            if fixed {
                                2.4
                            } else {
                                -4.0 + 8.0 * ((i * 313) % 1024) as f32 / 1024.0
                            }
                        })
                        .collect();
                    let mut out = input.clone();
                    g.bench(*candidate, move |b| {
                        b.iter(|| {
                            out.copy_from_slice(black_box(&input));
                            eval(candidate, op, black_box(&mut out), black_box(&y));
                            black_box(&out);
                        })
                    });
                }
                g.baseline("std");
            });
        }
    };
    zenbench::run(suite)
        .save(std::env::args().nth(1).expect("output JSON"))
        .unwrap();
}
#[cfg(target_arch = "wasm32")]
fn main() {
    panic!("Use native zenbench timing, WASM results executable checks numerical behavior");
}

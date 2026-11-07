#![forbid(unsafe_code)]
use math_matrix_study::*;
use std::{hint::black_box, time::Duration};
use zenbench::prelude::*;
fn main() {
    tier_inventory();
    let args: Vec<_> = std::env::args().collect();
    let filter = args.get(2).map(String::as_str).unwrap_or("");
    zenbench::run(|s| {
        s.cpu_policy(zenbench::platform::CpuPolicy::Performance);
        for op in [
            "powf",
            "pow_fixed",
            "exp",
            "ln",
            "log2",
            "log10",
            "sqrt",
            "floor",
            "ceil",
            "round",
            "roundeven",
            "log2_f64",
        ] {
            if !op.contains(filter) {
                continue;
            }
            for band in ["ordinary", "edge", "special"] {
                s.group(format!("{op}/{band}"), |g| {
                    g.throughput(Throughput::Elements(1024));
                    g.config()
                        .warmup_time(Duration::from_millis(50))
                        .min_rounds(30)
                        .max_rounds(60)
                        .max_wall_time(Duration::from_secs(30));
                    if op == "log2_f64" {
                        for c in cases64() {
                            let (x, y) = inputs64(band, 1024);
                            let mut out = vec![0.; 1024];
                            g.bench(c.label, move |b| {
                                b.iter(|| {
                                    (c.run)(black_box(&x), black_box(&y), black_box(&mut out));
                                    black_box(&out);
                                })
                            });
                        }
                    } else {
                        for c in cases32().into_iter().filter(|c| c.op == op) {
                            let (x, y) = inputs(op, band, 1024);
                            let mut out = vec![0.; 1024];
                            g.bench(c.label, move |b| {
                                b.iter(|| {
                                    (c.run)(black_box(&x), black_box(&y), black_box(&mut out));
                                    black_box(&out);
                                })
                            });
                        }
                    }
                    g.baseline("std/baseline");
                });
            }
        }
    })
    .save(&args[1])
    .unwrap();
}

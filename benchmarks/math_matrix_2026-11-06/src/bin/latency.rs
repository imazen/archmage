#![forbid(unsafe_code)]
use math_matrix_study::{feedback, tier_inventory};
use std::{hint::black_box, time::Duration};
use zenbench::prelude::*;
fn main() {
    tier_inventory();
    let out = std::env::args().nth(1).expect("output JSON");
    zenbench::run(|s| {
        s.cpu_policy(zenbench::platform::CpuPolicy::Performance);
        for op in ["exp", "ln", "log2", "sqrt"] {
            s.group(format!("feedback/{op}"), |g| {
                g.throughput(Throughput::Elements(1024));
                g.config()
                    .warmup_time(Duration::from_millis(50))
                    .min_rounds(30)
                    .max_rounds(60)
                    .max_wall_time(Duration::from_secs(30));
                for c in feedback::cases().into_iter().filter(|c| c.op == op) {
                    let x = vec![0.5; 1024];
                    let y = x.clone();
                    let mut o = vec![0.; 1024];
                    g.bench(c.label, move |b| {
                        b.iter(|| {
                            (c.run)(black_box(&x), black_box(&y), black_box(&mut o));
                            black_box(&o);
                        })
                    });
                }
                g.baseline("std/baseline/x1");
            });
        }
    })
    .save(out)
    .unwrap();
}

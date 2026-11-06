#![forbid(unsafe_code)]
use math_matrix_study::{algebraic::*, tier_inventory};
use std::{hint::black_box, time::Duration};
use zenbench::prelude::*;
fn main() {
    tier_inventory();
    for (name, values) in [
        ("cancellation", [1e20_f32, 1.0, -1e20, 1.0]),
        ("mixed_magnitudes", [1e-20_f32, 1e20, 1.0, -1e20]),
    ] {
        let x: Vec<_> = (0..1024).map(|i| values[i % 4]).collect();
        let y = vec![1.0; 1024];
        for c in cases_f32().into_iter().filter(|c| c.op.starts_with("dot")) {
            let mut o = vec![0.; 1024];
            (c.run)(&x, &y, &mut o);
            println!(
                "SNAPSHOT f32 {name} {} {:08x} {}",
                c.label,
                o[0].to_bits(),
                o[0]
            );
        }
        let x: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let y = vec![1.0; 1024];
        for c in cases_f64().into_iter().filter(|c| c.op.starts_with("dot")) {
            let mut o = vec![0.; 1024];
            (c.run)(&x, &y, &mut o);
            println!(
                "SNAPSHOT f64 {name} {} {:016x} {}",
                c.label,
                o[0].to_bits(),
                o[0]
            );
        }
    }
    let output = std::env::args().nth(1).expect("output JSON");
    zenbench::run(|s| {
        s.cpu_policy(zenbench::platform::CpuPolicy::Performance);
        for ty in ["f32", "f64"] {
            for op in ["dot", "poly"] {
                s.group(format!("{op}/{ty}"), |g| {
                    g.throughput(Throughput::Elements(1024));
                    g.config()
                        .warmup_time(Duration::from_millis(100))
                        .min_rounds(30)
                        .max_rounds(60)
                        .max_wall_time(Duration::from_secs(30));
                    if ty == "f32" {
                        for c in cases_f32().into_iter().filter(|c| c.op.starts_with(op)) {
                            let x: Vec<_> = (0..1024).map(|i| i as f32 / 1024.).collect();
                            let y = x.clone();
                            let mut out = vec![0.; 1024];
                            g.bench(c.label, move |b| {
                                b.iter(|| {
                                    (c.run)(black_box(&x), black_box(&y), black_box(&mut out));
                                    black_box(&out);
                                })
                            });
                        }
                    } else {
                        for c in cases_f64().into_iter().filter(|c| c.op.starts_with(op)) {
                            let x: Vec<_> = (0..1024).map(|i| i as f64 / 1024.).collect();
                            let y = x.clone();
                            let mut out = vec![0.; 1024];
                            g.bench(c.label, move |b| {
                                b.iter(|| {
                                    (c.run)(black_box(&x), black_box(&y), black_box(&mut out));
                                    black_box(&out);
                                })
                            });
                        }
                    }
                    g.baseline(format!("{op}_ordinary/scalar"));
                });
            }
        }
    })
    .save(output)
    .unwrap();
}

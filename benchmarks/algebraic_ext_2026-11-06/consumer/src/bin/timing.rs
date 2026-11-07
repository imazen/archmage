#![forbid(unsafe_code)]
use algebraic_ext_consumer::*;
use std::{hint::black_box, time::Duration};
use zenbench::prelude::*;
fn main() {
    tier_inventory();
    println!("exact fixture operations {}", fixtures::check_all());
    let out = std::env::args().nth(1).expect("output JSON");
    zenbench::run(|s| {
        s.cpu_policy(zenbench::platform::CpuPolicy::Performance);
        for ty in ["f32", "f64"] {
            for op in ["poly", "dot", "div", "rem"] {
                s.group(format!("{op}/{ty}"), |g| {
                    g.throughput(Throughput::Elements(1024));
                    g.config()
                        .warmup_time(Duration::from_millis(50))
                        .min_rounds(30)
                        .max_rounds(60)
                        .max_wall_time(Duration::from_secs(30));
                    if ty == "f32" {
                        for c in cases_f32()
                            .into_iter()
                            .filter(|c| c.group == format!("{op}/{ty}"))
                        {
                            let (x, y) = inputs_f32(op);
                            let mut o = vec![0.; 1024];
                            g.bench(c.name, move |b| {
                                b.iter(|| {
                                    (c.run)(black_box(&x), black_box(&y), black_box(&mut o));
                                    black_box(&o);
                                })
                            });
                        }
                    } else {
                        for c in cases_f64()
                            .into_iter()
                            .filter(|c| c.group == format!("{op}/{ty}"))
                        {
                            let (x, y) = inputs_f64(op);
                            let mut o = vec![0.; 1024];
                            g.bench(c.name, move |b| {
                                b.iter(|| {
                                    (c.run)(black_box(&x), black_box(&y), black_box(&mut o));
                                    black_box(&o);
                                })
                            });
                        }
                    }
                    g.baseline("ordinary/scalar");
                });
            }
        }
    })
    .save(out)
    .unwrap();
}

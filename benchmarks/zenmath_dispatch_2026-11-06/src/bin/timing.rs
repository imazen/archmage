#![forbid(unsafe_code)]
use std::{hint::black_box, time::Duration};
use zenbench::prelude::*;
use zenmath_dispatch_study::*;
fn main() {
    verify();
    let output = std::env::args().nth(1).expect("output JSON");
    zenbench::run(|s| {
        s.cpu_policy(zenbench::platform::CpuPolicy::Performance);
        for op in OPS32.iter().copied() {
            for mode in [
                "stream1",
                "stream16",
                "stream1024",
                "stream65536",
                "feedback",
            ] {
                if op == "pow_rescue" && mode == "feedback" {
                    continue;
                }
                let feedback = mode == "feedback";
                let n = if feedback {
                    1024
                } else {
                    mode.strip_prefix("stream").unwrap().parse().unwrap()
                };
                s.group(format!("{op}/f32/{mode}"), |g| {
                    g.throughput(Throughput::Elements(if feedback { 1024 } else { n as u64 }));
                    g.config()
                        .warmup_time(Duration::from_millis(10))
                        .min_rounds(30)
                        .max_rounds(60)
                        .max_wall_time(Duration::from_secs(20));
                    g.baseline("reference/baseline");
                    for c in cases32(op, feedback) {
                        let (x, y, z) = inputs32(op, n);
                        let mut out = vec![0.; n];
                        g.bench(c.name, move |b| {
                            b.iter(|| {
                                (c.run)(
                                    black_box(&x),
                                    black_box(&y),
                                    black_box(&z),
                                    black_box(&mut out),
                                );
                                black_box(&out);
                            })
                        });
                    }
                });
            }
        }
        for op in OPS64.iter().copied() {
            for mode in [
                "stream1",
                "stream16",
                "stream1024",
                "stream65536",
                "feedback",
            ] {
                let feedback = mode == "feedback";
                let n = if feedback {
                    1024
                } else {
                    mode.strip_prefix("stream").unwrap().parse().unwrap()
                };
                s.group(format!("{op}/f64/{mode}"), |g| {
                    g.throughput(Throughput::Elements(if feedback { 1024 } else { n as u64 }));
                    g.config()
                        .warmup_time(Duration::from_millis(10))
                        .min_rounds(30)
                        .max_rounds(60)
                        .max_wall_time(Duration::from_secs(20));
                    g.baseline("reference/baseline");
                    for c in cases64(op, feedback) {
                        let (x, y, z) = inputs64(op, n);
                        let mut out = vec![0.; n];
                        g.bench(c.name, move |b| {
                            b.iter(|| {
                                (c.run)(
                                    black_box(&x),
                                    black_box(&y),
                                    black_box(&z),
                                    black_box(&mut out),
                                );
                                black_box(&out);
                            })
                        });
                    }
                });
            }
        }
    })
    .save(output)
    .unwrap();
}

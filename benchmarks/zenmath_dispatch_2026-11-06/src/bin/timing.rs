#![forbid(unsafe_code)]
use std::{hint::black_box, time::Duration};
use zenbench::prelude::*;
use zenmath_dispatch_study::*;
fn main() {
    verify();
    let output = std::env::args().nth(1).expect("output JSON");
    let args: Vec<String> = std::env::args().collect();
    let mut filters = Vec::new();
    let mut i = 2;
    while i < args.len() {
        assert_eq!(args[i], "--study-group", "expected explicit --study-group filter");
        filters.push(
            args.get(i + 1)
                .expect("--study-group needs a full group name")
                .clone(),
        );
        i += 2;
    }
    let mut matched = std::collections::BTreeSet::new();
    let enabled = |name: &str| filters.is_empty() || filters.iter().any(|f| f == name);
    let result = zenbench::run(|s| {
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
                let group_name = format!("{op}/f32/{mode}");
                if !enabled(&group_name) {
                    continue;
                }
                matched.insert(group_name.clone());
                s.group(group_name, |g| {
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
                let group_name = format!("{op}/f64/{mode}");
                if !enabled(&group_name) {
                    continue;
                }
                matched.insert(group_name.clone());
                s.group(group_name, |g| {
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
    });
    for name in &filters {
        assert!(
            matched.contains(name),
            "requested group not measured: {name}"
        );
    }
    let measured: std::collections::BTreeSet<_> = result
        .comparisons
        .iter()
        .map(|g| g.group_name.clone())
        .collect();
    assert_eq!(measured, matched, "registered groups differ from measured groups");
    if !filters.is_empty() {
        let requested = filters.iter().cloned().collect();
        assert_eq!(measured, requested, "requested groups differ from measured groups");
    }
    result.save(output).unwrap();
}

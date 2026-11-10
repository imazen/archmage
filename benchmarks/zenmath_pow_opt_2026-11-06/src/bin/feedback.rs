#![forbid(unsafe_code)]
use std::{
    collections::{BTreeMap, BTreeSet},
    hint::black_box,
    time::Duration,
};
use zenbench::prelude::*;
use zenmath_pow_opt_study::*;
fn main() {
    verify();
    let a: Vec<_> = std::env::args().collect();
    let mut expected = BTreeMap::new();
    let result = zenbench::run(|s| {
        s.cpu_policy(zenbench::platform::CpuPolicy::Performance);
        for d in [Domain::Decode, Domain::Encode] {
            for n in [1, 8] {
                let name = format!("feedback/{}/chains{n}", d.name());
                let cases = domain_cases(d, true);
                let names: BTreeSet<_> = cases.iter().map(|c| c.name.clone()).collect();
                assert_eq!(names.len(), cases.len());
                expected.insert(name.clone(), names);
                s.group(name, |g| {
                    g.baseline("old-ordinary/correct");
                    g.throughput(Throughput::Elements(1024));
                    g.config()
                        .warmup_time(Duration::from_millis(10))
                        .min_rounds(50)
                        .max_rounds(80)
                        .max_wall_time(Duration::from_secs(30));
                    for c in cases {
                        let (initial, y) = inputs(d, n);
                        let mut x = initial.clone();
                        let mut out = vec![0.0; n];
                        g.bench(c.name, move |b| {
                            b.iter(|| {
                                x.copy_from_slice(&initial);
                                for _ in 0..1024 / n {
                                    (c.run)(black_box(&x), black_box(&y), &mut out);
                                    feedback::step(d, &out, &mut x);
                                }
                                black_box(&out);
                            })
                        });
                    }
                });
            }
        }
    });
    let measured: BTreeMap<_, _> = result
        .comparisons
        .iter()
        .map(|g| {
            let names: BTreeSet<_> = g.benchmarks.iter().map(|b| b.name.clone()).collect();
            assert_eq!(names.len(), g.benchmarks.len());
            (g.group_name.clone(), names)
        })
        .collect();
    assert_eq!(measured.len(), result.comparisons.len());
    assert_eq!(expected, measured);
    result.save(&a[1]).unwrap();
}

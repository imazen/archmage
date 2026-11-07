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
    let args: Vec<_> = std::env::args().collect();
    let output = &args[1];
    let mut correct_only = false;
    let mut requested = BTreeSet::new();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--correct-only" => {
                correct_only = true;
                i += 1;
            }
            "--study-group" => {
                requested.insert(args.get(i + 1).expect("group name").clone());
                i += 2;
            }
            _ => panic!("unknown study argument"),
        }
    }
    let mut expected = BTreeMap::<String, BTreeSet<String>>::new();
    let result = zenbench::run(|s| {
        s.cpu_policy(zenbench::platform::CpuPolicy::Performance);
        for d in DOMAINS {
            for &distribution in distributions(d) {
                for uniform in [false, true] {
                    if uniform && !matches!(d, Domain::Decode | Domain::Encode) {
                        continue;
                    }
                    for n in [1, 16, 1024, 65536] {
                        let name = format!(
                            "{}/{}/stream{n}",
                            if distribution == Distribution::Uniform {
                                d.name().to_owned()
                            } else {
                                format!("{}@{}", d.name(), distribution.name())
                            },
                            if uniform { "uniform" } else { "per-value" }
                        );
                        if !requested.is_empty() && !requested.contains(&name) {
                            continue;
                        }
                        let cases: Vec<_> = domain_cases(d, uniform)
                            .into_iter()
                            .filter(|c| !correct_only || c.policy == 0)
                            .collect();
                        let names: BTreeSet<_> = cases.iter().map(|c| c.name.clone()).collect();
                        assert_eq!(names.len(), cases.len(), "duplicate case");
                        assert!(
                            expected.insert(name.clone(), names).is_none(),
                            "duplicate group"
                        );
                        s.group(name, |g| {
                            g.throughput(Throughput::Elements(n as u64));
                            g.baseline("old-ordinary/correct");
                            g.config()
                                .warmup_time(Duration::from_millis(10))
                                .min_rounds(50)
                                .max_rounds(80)
                                .max_wall_time(Duration::from_secs(30));
                            for c in cases {
                                let (x, y) = distributed_inputs(d, distribution, n);
                                let mut out = vec![0.0; n];
                                g.bench(c.name, move |b| {
                                    b.iter(|| {
                                        (c.run)(black_box(&x), black_box(&y), black_box(&mut out));
                                        black_box(&out);
                                    })
                                });
                            }
                        });
                    }
                }
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
    assert_eq!(
        result.comparisons.len(),
        measured.len(),
        "duplicate measured groups"
    );
    assert_eq!(expected, measured, "measured inventory mismatch");
    if !requested.is_empty() {
        assert_eq!(
            requested,
            measured.keys().cloned().collect(),
            "requested groups missing"
        );
    }
    result.save(output).unwrap();
}

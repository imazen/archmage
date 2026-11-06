#![forbid(unsafe_code)]
use std::{
    collections::{BTreeMap, BTreeSet},
    hint::black_box,
    time::Duration,
};
use zenbench::prelude::*;
use zenmath_pow_opt_study::{transfer::*, *};
fn main() {
    verify();
    let a: Vec<_> = std::env::args().collect();
    let output = &a[1];
    let requested: BTreeSet<_> = a[2..].iter().cloned().collect();
    let mut expected = BTreeMap::new();
    let result = zenbench::run(|s| {
        s.cpu_policy(zenbench::platform::CpuPolicy::Performance);
        for d in [Domain::Decode, Domain::Encode] {
            for &distribution in distributions(d) {
                for composition in [Composition::Separate, Composition::Fused] {
                    for n in [1, 16, 1024, 65536] {
                        let name = format!(
                            "transfer/{}/{}/{}/stream{n}",
                            d.name(),
                            distribution.name(),
                            composition.name()
                        );
                        if !requested.is_empty() && !requested.contains(&name) {
                            continue;
                        }
                        let cases = domain_cases(d, false);
                        let mut names: BTreeSet<_> = cases.iter().map(|c| c.name.clone()).collect();
                        assert_eq!(names.len(), cases.len());
                        names.insert("linear-srgb-0.6.12/scalar-rational".to_owned());
                        names.insert("linear-srgb-0.6.12/slice-rational".to_owned());
                        assert!(expected.insert(name.clone(), names).is_none());
                        s.group(name, |g| {
                            g.throughput(Throughput::Elements(n as u64));
                            g.baseline("old-ordinary/correct");
                            g.config()
                                .warmup_time(Duration::from_millis(10))
                                .min_rounds(50)
                                .max_rounds(80)
                                .max_wall_time(Duration::from_secs(30));
                            for c in cases {
                                let input = transfer_inputs(d, distribution, n);
                                let (mut x, mut y, mut p, mut out) =
                                    (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
                                g.bench(c.name, move |b| {
                                    b.iter(|| {
                                        prepare(d, black_box(&input), &mut x, &mut y);
                                        (c.run)(black_box(&x), black_box(&y), &mut p);
                                        compose(d, composition, &input, &p, &mut out);
                                        black_box(&out);
                                    })
                                });
                            }
                            for slice in [false, true] {
                                let input = transfer_inputs(d, distribution, n);
                                let mut out = vec![0.0; n];
                                let label = if slice {
                                    "linear-srgb-0.6.12/slice-rational"
                                } else {
                                    "linear-srgb-0.6.12/scalar-rational"
                                };
                                g.bench(label, move |b| {
                                    b.iter(|| {
                                        if slice {
                                            rational_slice(d, black_box(&input), &mut out)
                                        } else {
                                            rational_scalar(d, black_box(&input), &mut out)
                                        };
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
    assert_eq!(measured.len(), result.comparisons.len());
    assert_eq!(expected, measured);
    if !requested.is_empty() {
        assert_eq!(requested, measured.keys().cloned().collect());
    }
    result.save(output).unwrap();
}

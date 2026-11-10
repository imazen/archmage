#![forbid(unsafe_code)]
use std::{
    collections::{BTreeMap, BTreeSet},
    hint::black_box,
    time::Duration,
};
use zenbench::{prelude::*, wasm::WasmBench};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(args.len() >= 4);
    let requested: BTreeSet<_> = args[4..].iter().cloned().collect();
    let w = WasmBench::from_file(&args[1]).unwrap();
    let catalog = std::fs::read_to_string(&args[2]).unwrap();
    let exports: Vec<_> = w
        .module()
        .exports()
        .filter(|e| e.name().contains("pow_study_kernel"))
        .map(|e| e.name().to_owned())
        .collect();
    assert_eq!(exports.len(), 1);
    let export = &exports[0];
    let mut groups = BTreeMap::<(i32, i32, i32, String), Vec<(i32, String)>>::new();
    for line in catalog.lines() {
        let p: Vec<_> = line.split('\t').collect();
        if p.first() == Some(&"C") {
            assert_eq!(p.len(), 7);
            groups
                .entry((
                    p[1].parse().unwrap(),
                    p[2].parse().unwrap(),
                    p[3].parse().unwrap(),
                    p[5].to_owned(),
                ))
                .or_default()
                .push((p[4].parse().unwrap(), p[6].to_owned()));
        }
    }
    assert_eq!(
        groups.len(),
        15,
        "expected domain/distribution/uniform catalog"
    );
    for cases in groups.values() {
        let names: BTreeSet<_> = cases.iter().map(|(_, s)| s).collect();
        assert_eq!(names.len(), cases.len());
        assert!(names.contains(&"current/correct".to_owned()));
        assert!(names.contains(&"old-ordinary/correct".to_owned()));
        assert!(names.iter().any(|s| s.starts_with("packed-wasm128/")));
    }
    let mut expected = BTreeMap::new();
    let result = zenbench::run(|s| {
        s.cpu_policy(zenbench::platform::CpuPolicy::Performance);
        for ((d, dist, uniform, name), cases) in groups {
            for n in [1, 16, 1024, 65536] {
                let group = format!(
                    "{}/{}/stream{n}",
                    name,
                    if uniform != 0 { "uniform" } else { "per-value" }
                );
                if !requested.is_empty() && !requested.contains(&group) {
                    continue;
                }
                expected.insert(
                    group.clone(),
                    cases
                        .iter()
                        .map(|(_, s)| s.clone())
                        .collect::<BTreeSet<_>>(),
                );
                let repeats = if n >= 1024 { 262144 / n } else { 256 };
                s.group(group, |g| {
                    g.baseline("old-ordinary/correct");
                    g.throughput(Throughput::Elements((n * repeats) as u64));
                    g.config()
                        .warmup_time(Duration::from_millis(10))
                        .min_rounds(50)
                        .max_rounds(80)
                        .max_wall_time(Duration::from_secs(30));
                    for (id, label) in &cases {
                        let mut instance = w.wasi_instance().unwrap();
                        let f = instance
                            .typed_func::<(i32, i32, i32, i32, i32, i32), f64>(export)
                            .unwrap();
                        let id = *id;
                        g.bench(label.clone(), move |b| {
                            b.iter(|| {
                                black_box(
                                    f.call(&mut instance.store, (d, dist, uniform, id, n, repeats))
                                        .unwrap(),
                                );
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
    if !requested.is_empty() {
        assert_eq!(requested, measured.keys().cloned().collect());
    } else {
        assert_eq!(measured.len(), 60);
    }
    result.save(&args[3]).unwrap();
}

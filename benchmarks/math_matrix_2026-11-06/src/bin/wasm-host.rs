#![forbid(unsafe_code)]
use std::{hint::black_box, time::Duration};
use zenbench::{prelude::*, wasm::WasmBench};
fn main() {
    let a: Vec<_> = std::env::args().collect();
    let w = WasmBench::from_file(&a[1]).unwrap();
    let catalog = std::fs::read_to_string(&a[2]).unwrap();
    let matrix = w
        .module()
        .exports()
        .find(|e| e.name().contains("matrix_kernel"))
        .unwrap()
        .name()
        .to_owned();
    let algebraic = w
        .module()
        .exports()
        .find(|e| e.name().contains("algebraic_kernel"))
        .unwrap()
        .name()
        .to_owned();
    let mut groups = std::collections::BTreeMap::<String, Vec<(i32, String, i32, bool)>>::new();
    for line in catalog.lines() {
        let p: Vec<_> = line.split('\t').collect();
        if p.len() != 4 {
            continue;
        }
        let id: i32 = p[1].parse().unwrap();
        if p[0] == "M" {
            for (b, band) in ["ordinary", "edge", "special"].iter().enumerate() {
                groups.entry(format!("{}/{band}", p[2])).or_default().push((
                    id,
                    p[3].to_owned(),
                    b as i32,
                    false,
                ));
            }
        } else if p[0] == "A" {
            groups
                .entry(format!("algebraic/{}", p[2]))
                .or_default()
                .push((id, p[3].to_owned(), 0, true));
        }
    }
    let traps = std::fs::read_to_string(&a[4]).expect("explicit candidate-trap manifest");
    for (group, cases) in groups.iter_mut() {
        cases.retain(|(_, label, _, _)| !traps.lines().any(|l| l == format!("{group}\t{label}")));
    }
    zenbench::run(|s| {
        s.cpu_policy(zenbench::platform::CpuPolicy::Performance);
        for (group, cases) in groups {
            let baseline = if group.starts_with("algebraic/") {
                format!("{}_ordinary/scalar", group.split('/').nth(1).unwrap())
            } else {
                "std/baseline".to_owned()
            };
            s.group(group, |g| {
                g.baseline(baseline);
                g.throughput(Throughput::Elements(1024 * 256));
                g.config()
                    .warmup_time(Duration::from_millis(50))
                    .min_rounds(30)
                    .max_rounds(60)
                    .max_wall_time(Duration::from_secs(30));
                for (id, label, band, alg) in cases {
                    let mut inst = w.wasi_instance().unwrap();
                    if alg {
                        let f = inst.typed_func::<(i32, i32), f64>(&algebraic).unwrap();
                        g.bench(label, move |b| {
                            b.iter(|| {
                                black_box(f.call(&mut inst.store, (id, 256)).unwrap());
                            })
                        });
                    } else {
                        let f = inst.typed_func::<(i32, i32, i32), f64>(&matrix).unwrap();
                        g.bench(label, move |b| {
                            b.iter(|| {
                                black_box(f.call(&mut inst.store, (id, band, 256)).unwrap());
                            })
                        });
                    }
                }
            });
        }
    })
    .save(&a[3])
    .unwrap();
}

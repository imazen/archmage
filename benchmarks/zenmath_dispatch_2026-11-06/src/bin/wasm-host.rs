#![forbid(unsafe_code)]
use std::{hint::black_box, time::Duration};
use zenbench::{prelude::*, wasm::WasmBench};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let w = WasmBench::from_file(&args[1]).unwrap();
    let catalog = std::fs::read_to_string(&args[2]).unwrap();
    let export = w
        .module()
        .exports()
        .find(|e| e.name().contains("dispatch_kernel"))
        .unwrap()
        .name()
        .to_owned();
    let mut groups =
        std::collections::BTreeMap::<(i32, i32, String, String), Vec<(i32, String)>>::new();
    for line in catalog.lines() {
        let p: Vec<_> = line.split('\t').collect();
        if p.len() == 7 && p[0] == "C" {
            groups
                .entry((
                    p[1].parse().unwrap(),
                    p[2].parse().unwrap(),
                    p[4].to_owned(),
                    p[5].to_owned(),
                ))
                .or_default()
                .push((p[3].parse().unwrap(), p[6].to_owned()));
        }
    }
    assert_eq!(groups.len(), 37, "incomplete validated WASM catalog");
    let result = zenbench::run(|s| {
        s.cpu_policy(zenbench::platform::CpuPolicy::Performance);
        for ((group, feedback, ty, op), cases) in groups {
            let sizes: Vec<i32> = if feedback != 0 {
                vec![1024]
            } else {
                vec![1, 16, 1024, 65536]
            };
            for n in sizes {
                let repeats = if feedback != 0 {
                    32
                } else if n >= 1024 {
                    262144 / n
                } else {
                    256
                };
                let steps = if feedback != 0 { 1024 } else { n };
                let mode = if feedback != 0 {
                    "feedback".to_owned()
                } else {
                    format!("stream{n}")
                };
                s.group(format!("{op}/{ty}/{mode}"), |g| {
                    g.throughput(Throughput::Elements((steps * repeats) as u64));
                    g.config()
                        .warmup_time(Duration::from_millis(10))
                        .min_rounds(30)
                        .max_rounds(60)
                        .max_wall_time(Duration::from_secs(40));
                    g.baseline("reference/baseline");
                    for (id, label) in &cases {
                        let mut inst = w.wasi_instance().unwrap();
                        let f = inst
                            .typed_func::<(i32, i32, i32, i32, i32), f64>(&export)
                            .unwrap();
                        let id = *id;
                        g.bench(label.clone(), move |b| {
                            b.iter(|| {
                                black_box(
                                    f.call(&mut inst.store, (group, feedback, id, n, repeats))
                                        .unwrap(),
                                );
                            })
                        });
                    }
                });
            }
        }
    });
    assert_eq!(result.comparisons.len(), 94, "incomplete WASM timing matrix");
    result.save(&args[3])
    .unwrap();
}

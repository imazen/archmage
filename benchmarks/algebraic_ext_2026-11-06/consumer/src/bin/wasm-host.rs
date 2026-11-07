#![forbid(unsafe_code)]
use std::{hint::black_box, time::Duration};
use zenbench::{prelude::*, wasm::WasmBench};
fn main() {
    let a: Vec<_> = std::env::args().collect();
    let w = WasmBench::from_file(&a[1]).unwrap();
    let text = std::fs::read_to_string(&a[2]).unwrap();
    let export = w
        .module()
        .exports()
        .find(|e| e.name().contains("ext_kernel"))
        .unwrap()
        .name()
        .to_owned();
    let mut groups = std::collections::BTreeMap::<String, Vec<(i32, String)>>::new();
    for l in text.lines() {
        let p: Vec<_> = l.split('\t').collect();
        if p.len() == 4 && p[0] == "C" {
            groups
                .entry(p[2].to_owned())
                .or_default()
                .push((p[1].parse().unwrap(), p[3].to_owned()));
        }
    }
    assert!(text.contains("index/wasm128"));
    assert!(text.contains("token/scalar"));
    zenbench::run(|s| {
        s.cpu_policy(zenbench::platform::CpuPolicy::Performance);
        for (group, cases) in groups {
            s.group(group, |g| {
                g.throughput(Throughput::Elements(1024 * 256));
                g.config()
                    .warmup_time(Duration::from_millis(50))
                    .min_rounds(30)
                    .max_rounds(60)
                    .max_wall_time(Duration::from_secs(30));
                g.baseline("ordinary/scalar");
                for (id, label) in cases {
                    let mut inst = w.wasi_instance().unwrap();
                    let f = inst.typed_func::<(i32, i32), f64>(&export).unwrap();
                    g.bench(label, move |b| {
                        b.iter(|| {
                            black_box(f.call(&mut inst.store, (id, 256)).unwrap());
                        })
                    });
                }
            });
        }
    })
    .save(&a[3])
    .unwrap();
}

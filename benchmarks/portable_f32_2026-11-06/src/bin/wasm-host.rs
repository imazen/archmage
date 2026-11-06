#![forbid(unsafe_code)]
use portable_f32_study::{CANDIDATES, supports};
use std::{hint::black_box, time::Duration};
use zenbench::{prelude::*, wasm::WasmBench};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "wasm-host MODULE OUTPUT_JSON");
    let wasm = WasmBench::from_file(&args[1]).unwrap();
    let export = wasm
        .module()
        .exports()
        .find(|e| e.name().contains("portable_bench_kernel"))
        .expect("kernel export")
        .name()
        .to_owned();
    println!(
        "Export: {export}; 256 copies/1024-value kernel per WASM call; wasmtime version recorded in Cargo.lock"
    );
    zenbench::run(|s| {
        s.cpu_policy(zenbench::platform::CpuPolicy::Performance);
        for (operation, group) in ["exp", "ln", "log2", "pow_fixed_2.4", "pow_general"]
            .iter()
            .enumerate()
        {
            let op = if operation >= 3 { "pow" } else { group };
            s.group(*group, |g| {
                g.throughput(Throughput::Elements(1024 * 256));
                g.config()
                    .warmup_time(Duration::from_millis(100))
                    .min_rounds(30)
                    .max_rounds(60)
                    .max_wall_time(Duration::from_secs(2));
                for (candidate, name) in CANDIDATES.iter().enumerate() {
                    if !supports(name, op) {
                        continue;
                    }
                    let mut inst = wasm.wasi_instance().unwrap();
                    let f = inst.typed_func::<(i32, i32, i32), f64>(&export).unwrap();
                    g.bench(*name, move |b| {
                        b.iter(|| {
                            black_box(
                                f.call(&mut inst.store, (candidate as i32, operation as i32, 256))
                                    .unwrap(),
                            );
                        })
                    });
                }
                g.baseline("std");
            });
        }
    })
    .save(&args[2])
    .unwrap();
}

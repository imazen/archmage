//! Wasmtime host; primitive-level probing versus whole-operation selection.
#![forbid(unsafe_code)]
use std::time::Duration;
use zenbench::wasm::{WasmBench, wasmtime};
fn main() {
    assert!(!std::env::args().any(|a| a == "--no-busy-gate"));
    assert!(std::env::var("ZENBENCH_NO_BUSY_GATE").map_or(true, |v| v.is_empty() || v == "0"));
    let path = std::env::var("POWF_WASM").expect("POWF_WASM");
    let mode = std::env::var("POWF_WASM_MODE").expect("strict/fusing/nonfusing");
    assert!(matches!(mode.as_str(), "strict" | "fusing" | "nonfusing"));
    let expected = if mode == "fusing" { 1 } else { 0 };
    let mut config = wasmtime::Config::new();
    config.wasm_simd(true).wasm_relaxed_simd(true);
    if mode == "nonfusing" {
        config.target("x86_64-unknown-linux-gnu").unwrap();
    }
    let wasm = WasmBench::from_bytes_with_config(&std::fs::read(path).unwrap(), config).unwrap();
    let find = |part: &str| {
        let names: Vec<_> = wasm
            .module()
            .exports()
            .filter(|e| e.name().contains(part))
            .map(|e| e.name().to_string())
            .collect();
        assert_eq!(names.len(), 1);
        names[0].clone()
    };
    let prepare = find("guest_prepare");
    let run = find("guest_run");
    let probe = find("guest_probe");
    let verify = find("guest_verify");
    let mut check = wasm.wasi_instance().unwrap();
    assert_eq!(
        check.call::<_, i32>(&probe, ()).unwrap(),
        expected,
        "actual engine fusion projection"
    );
    println!("WASM mode={mode} actual_f32_fusion={expected}");
    if std::env::var("POWF_VERIFY_ONLY").is_ok_and(|v| v == "1") {
        let mut verified = 0;
        for size in [64, 4096] {
            for chain in [0, 1] {
                for domain in 0..3 {
                    for distribution in 0..2 {
                        for (width, context) in
                            [(1, 0), (4, 0), (8, 0), (16, 0), (4, 1), (8, 1), (16, 1)]
                        {
                            check
                                .call::<_, ()>(&prepare, (size, distribution, domain))
                                .unwrap();
                            check
                                .call::<_, ()>(&verify, (width, chain, context))
                                .unwrap();
                            verified += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(verified, 168);
        println!("verified {verified} whole-graph cells, every arm and lane bit");
        return;
    }
    let output = std::env::var("POWF_RESULT").expect("POWF_RESULT");
    assert!(!std::path::Path::new(&output).exists());
    let result = zenbench::run_gated(zenbench::GateConfig::strict(), |s| {
        for size in [64, 4096] {
            for chain in [0, 1] {
                for domain in 0..3 {
                    for distribution in 0..2 {
                        for (width, context) in
                            [(1, 0), (4, 0), (8, 0), (16, 0), (4, 1), (8, 1), (16, 1)]
                        {
                            let name = format!(
                                "wasm/{mode}/x{width}/{}/{size}/{}/{}/{distribution}",
                                if context == 0 { "scalar" } else { "wasm128" },
                                if chain == 0 { "stream" } else { "chain" },
                                ["PositiveNormal", "NonNegativeNormal", "Gamma"][domain as usize]
                            );
                            if !matches_filter(&name) {
                                continue;
                            }
                            check
                                .call::<_, ()>(&prepare, (size, distribution, domain))
                                .unwrap();
                            check
                                .call::<_, ()>(&verify, (width, chain, context))
                                .unwrap();
                            s.group(name, |g| {
                                g.config()
                                    .min_rounds(50)
                                    .max_rounds(50)
                                    .auto_rounds(false)
                                    .warmup_time(Duration::from_millis(10))
                                    .max_time(Duration::from_millis(100))
                                    .max_wall_time(Duration::from_secs(60));
                                for (arm, label) in [
                                    "prevalidated_primitive_graph",
                                    "checked_caller",
                                    "checked_auto",
                                ]
                                .into_iter()
                                .enumerate()
                                {
                                    let wasm = wasm.clone();
                                    let prepare = prepare.clone();
                                    let run = run.clone();
                                    g.bench(label, move |b| {
                                        let mut instance = wasm.wasi_instance().unwrap();
                                        instance
                                            .call::<_, ()>(&prepare, (size, distribution, domain))
                                            .unwrap();
                                        let f = instance
                                            .typed_func::<(i32, i32, i32, i32, i32), i64>(&run)
                                            .unwrap();
                                        f.call(
                                            &mut instance.store,
                                            (width, arm as i32, chain, 1, context),
                                        )
                                        .unwrap();
                                        b.iter(|| {
                                            f.call(
                                                &mut instance.store,
                                                (width, arm as i32, chain, 128, context),
                                            )
                                            .unwrap()
                                        });
                                    });
                                }
                            });
                        }
                    }
                }
            }
        }
        let name = format!("wasm/{mode}/first-use/x4/64/Gamma");
        if matches_filter(&name) {
            check.call::<_, ()>(&prepare, (64, 0, 2)).unwrap();
            check.call::<_, ()>(&verify, (4, 0, 1)).unwrap();
            s.group(name, |g| {
                g.config()
                    .min_rounds(50)
                    .max_rounds(50)
                    .auto_rounds(false)
                    .warmup_time(Duration::from_millis(10))
                    .max_time(Duration::from_millis(20))
                    .max_wall_time(Duration::from_secs(60));
                for (arm, label) in [
                    "prevalidated_primitive_graph",
                    "checked_caller",
                    "checked_auto",
                ]
                .into_iter()
                .enumerate()
                {
                    let wasm = wasm.clone();
                    let prepare = prepare.clone();
                    let run = run.clone();
                    g.bench(label, move |b| {
                        let wasm = wasm.clone();
                        let prepare = prepare.clone();
                        let run = run.clone();
                        b.with_input(move || {
                            let mut instance = wasm.wasi_instance().unwrap();
                            instance.call::<_, ()>(&prepare, (64, 0, 2)).unwrap();
                            let f = instance
                                .typed_func::<(i32, i32, i32, i32, i32), i64>(&run)
                                .unwrap();
                            (instance, f)
                        })
                        .run(move |(mut instance, f)| {
                            let result = f
                                .call(&mut instance.store, (4, arm as i32, 0, 1, 1))
                                .unwrap();
                            (instance, result)
                        });
                    });
                }
            });
        }
    });
    result.save(std::path::Path::new(&output)).unwrap();
}
fn matches_filter(name: &str) -> bool {
    std::env::var("POWF_FILTER").map_or(true, |v| name.contains(&v))
}

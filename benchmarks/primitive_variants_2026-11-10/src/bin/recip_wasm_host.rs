//! Patched zenbench hosts the same fixed WASM kernels; no guest wall-clock timer.
#![forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use zenbench::wasm::{WasmBench, wasmtime};
    if std::env::args().any(|x| x == "--cpu-set") {
        let mut c = zenbench::platform::detect_cpu_topology().classes;
        c.sort_by(|a, b| a.name.cmp(&b.name));
        println!(
            "{}",
            c[0].cpus
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        return;
    }
    let path = std::env::var("ARCHMAGE_WASM_MODULE").expect("frozen module path");
    let mode = std::env::var("ARCHMAGE_WASM_MODE").expect("strict/fusing/nonfusing");
    assert!(["strict", "fusing", "nonfusing"].contains(&mode.as_str()));
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
    let verify = find("guest_verify");
    let probe = find("guest_probe");
    let mut check = wasm.wasi_instance().unwrap();
    let fusion = check.call::<_, i32>(&probe, ()).unwrap();
    assert_eq!(fusion, i32::from(mode == "fusing"), "actual engine fusion");
    eprintln!(
        "WASM mode={mode} actual relaxed fusion={fusion}; timed calls batch complete workloads"
    );
    let mut cells = Vec::new();
    for w in [4, 8, 16] {
        for (kind, n, dist, chain) in [
            ("stream64", 64, 0, 0),
            ("stream4096", 4096, 0, 0),
            ("chain64", 64, 0, 1),
        ] {
            cells.push((w, kind, n, dist, chain));
        }
    }
    for (kind, n, dist) in [
        ("bits64", 64, 1),
        ("bits4096", 4096, 1),
        ("boundary64", 64, 2),
        ("boundary4096", 4096, 2),
    ] {
        cells.push((8, kind, n, dist, 0));
    }
    let arms = vec![
        (0, "magetypes recip"),
        (2, "portable serial MIDP"),
        (10, "bounded f64 MIDP (same fused bits)"),
        (3, "portable HIGHP division"),
        (11, "RN14 native always certified"),
        (12, "RN14 native guarded"),
        (13, "RN14 division guarded"),
    ];
    let arms=if std::env::args().any(|x|x=="--cold-study") {vec![(0,"magetypes recip"),(3,"portable HIGHP division"),(12,"RN14 native guarded inline"),(13,"RN14 division guarded inline"),(14,"RN14 native guarded cold"),(15,"RN14 division guarded cold")]} else {arms};
    for &(w, _, n, dist, chain) in &cells {
        check.call::<_, ()>(&prepare, (n, dist)).unwrap();
        for &(arm, _) in &arms {
            check.call::<_, ()>(&verify, (w, arm, chain)).unwrap();
        }
    }
    if std::env::args().any(|x| x == "--verify-only") {
        return;
    }
    let result = zenbench::run_gated(zenbench::GateConfig::strict(), |suite| {
        let p = suite.plan();
        p.mode = zenbench::BenchmarkMode::Confirm;
        p.processes = 5;
        p.rounds = Some(50);
        p.margin_pct = 2.0;
        p.confidence = 0.95;
        p.workload_id = Some("recip-certified-wasm-batch4096-short-batch256-long-chain-v1".into());
        p.scope = Some(format!("recip-certified-wasm-{mode}"));
        for &(w, kind, n, dist, chain) in &cells {
            suite.group(format!("wasm-{mode}/f32x{w}/{kind}"), |g| {
                g.config().min_rounds(50).max_rounds(50).auto_rounds(false);
                for &(arm, label) in &arms {
                    let wasm = wasm.clone();
                    let prepare = prepare.clone();
                    let run = run.clone();
                    g.bench(label, move |b| {
                        let mut inst = wasm.wasi_instance().unwrap();
                        inst.call::<_, ()>(&prepare, (n, dist)).unwrap();
                        let f = inst.typed_func::<(i32, i32, i32, i32), i64>(&run).unwrap();
                        // These batch counts are also recorded in the analysis manifest.
                        let reps = if n == 64 && chain == 0 { 4096 } else { 256 };
                        b.iter(|| f.call(&mut inst.store, (w, arm, chain, reps)).unwrap());
                    });
                }
            });
        }
    });
    if std::env::var("ZENBENCH_SUBPROCESS").as_deref() == Ok("1") {
        return;
    }
    result
        .save(std::env::var("ARCHMAGE_PRIMITIVE_RESULTS").expect("durable result path"))
        .unwrap();
    result.print_report();
    assert!(
        result.confirmation_issues().is_empty(),
        "retained WASM timing fails confirmation: {:?}",
        result.confirmation_issues()
    );
}
#[cfg(target_arch = "wasm32")]
fn main() {
    panic!("host runs natively");
}

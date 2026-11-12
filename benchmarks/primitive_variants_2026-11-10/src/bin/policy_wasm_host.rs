//! Host-side zenbench timing of the complete primitive policy WASI matrix.
#![forbid(unsafe_code)]
#[macro_use]
#[path = "../policy_registry.rs"]
mod registry;
use std::sync::{Arc, Mutex};
use zenbench::wasm::{WasmBench, wasmtime};

fn main() {
    if std::env::args().any(|a| a == "--cpu-set") {
        let mut classes = zenbench::platform::detect_cpu_topology().classes;
        classes.sort_by(|a, b| a.name.cmp(&b.name));
        println!(
            "{}",
            classes[0]
                .cpus
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        return;
    }
    let mode = std::env::var("ARCHMAGE_WASM_MODE").expect("strict/fusing/nonfusing");
    assert!(["strict", "fusing", "nonfusing"].contains(&mode.as_str()));
    let mut config = wasmtime::Config::new();
    config.wasm_simd(true).wasm_relaxed_simd(true);
    if mode == "nonfusing" {
        config.target("x86_64-unknown-linux-gnu").unwrap();
    }
    let bytes =
        std::fs::read(std::env::var("ARCHMAGE_WASM_MODULE").expect("frozen module")).unwrap();
    let wasm = WasmBench::from_bytes_with_config(&bytes, config).unwrap();
    let find = |part: &str| {
        let names: Vec<_> = wasm
            .module()
            .exports()
            .filter(|e| e.name().ends_with(part))
            .map(|e| e.name().to_owned())
            .collect();
        assert_eq!(names.len(), 1);
        names[0].clone()
    };
    let prepare = find("guest_prepare");
    let run = find("guest_run");
    let verify = find("guest_verify");
    let probe = find("guest_probe");
    let mut inst = wasm.wasi_instance().unwrap();
    let fusion = inst.call::<_, i32>(&probe, ()).unwrap();
    assert_eq!(
        fusion,
        i32::from(mode == "fusing"),
        "actual relaxed fusion mode"
    );
    eprintln!("WASM_POLICY_MODE\t{mode}\tactual_fusion={fusion}");
    let mut cells = Vec::new();
    for (op, name, ty, widths) in [
        (0, "min", "f32", [4, 8, 16]),
        (1, "max", "f32", [4, 8, 16]),
        (2, "min", "f64", [2, 4, 8]),
        (3, "max", "f64", [2, 4, 8]),
        (4, "recip", "f32", [4, 8, 16]),
        (5, "rsqrt", "f32", [4, 8, 16]),
    ] {
        for width in widths {
            for (kind, label) in [(0, "stream64"), (1, "stream4096"), (2, "chain64")] {
                cells.push((op, name, ty, width, kind, label));
            }
            if op < 4 {
                cells.push((op, name, ty, width, 3, "zero-clamp64"));
            }
        }
    }
    assert_eq!(cells.len(), 66);
    let mut inventory = Vec::new();
    for &(op, name, ty, width, kind, label) in &cells {
        let group = format!("wasm-{mode}/{name}/{ty}x{width}/{label}");
        let mut arms = vec![(-1, format!("existing Magetypes {name}"))];
        macro_rules! add {
            ($flag:literal,$id:literal) => {
                arms.push(($flag, String::from($id)));
            };
        }
        if op >= 4 {
            recip_cases!(add);
        } else if kind == 3 {
            zero_cases!(add);
        } else {
            minmax_cases!(add);
        }
        inst.call::<_, ()>(&prepare, (op, kind)).unwrap();
        for (flag, id) in &arms {
            let error = inst
                .call::<_, f64>(&verify, (width, *flag, op, kind))
                .unwrap();
            eprintln!("POLICY_ACCURACY\t{group}\t{id}\t{error:.17e}");
        }
        inventory.push((op, width, kind, group, arms));
    }
    assert_eq!(inventory.iter().map(|v| v.4.len()).sum::<usize>(), 1566);
    if std::env::args().any(|a| a == "--verify-only") {
        return;
    }
    // One instance is shared by the sequential paired arms. Preparation/locking
    // happens outside b.iter; guest input/output addresses are identical for all
    // arms. Reusing it avoids per-round instantiation and an instance per cell.
    let shared = Arc::new(Mutex::new(inst));
    let result = zenbench::run_gated(zenbench::GateConfig::strict(), |suite| {
        let p = suite.plan();
        p.mode = zenbench::BenchmarkMode::Confirm;
        p.processes = 5;
        p.rounds = Some(50);
        p.margin_pct = 2.0;
        p.confidence = 0.95;
        p.workload_id =
            Some("public-primitive-policy-wasm-shared-instance-batch4096-256-v1".into());
        p.scope = Some(format!("primitive-policies-wasm-{mode}"));
        for (op, width, kind, group, arms) in inventory {
            suite.group(group, |g| {
                g.config().min_rounds(50).max_rounds(50).auto_rounds(false);
                for (flag, id) in arms {
                    let shared = Arc::clone(&shared);
                    let prepare = prepare.clone();
                    let run = run.clone();
                    g.bench(id, move |b| {
                        let mut inst = shared.lock().unwrap();
                        inst.call::<_, ()>(&prepare, (op, kind)).unwrap();
                        let f = inst
                            .typed_func::<(i32, i32, i32, i32, i32), i64>(&run)
                            .unwrap();
                        let reps = if kind == 0 || kind == 3 { 4096 } else { 256 };
                        b.iter(|| {
                            f.call(&mut inst.store, (width, flag, op, kind, reps))
                                .unwrap()
                        });
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
        "retained WASM policy matrix fails confirmation: {:?}",
        result.confirmation_issues()
    );
}

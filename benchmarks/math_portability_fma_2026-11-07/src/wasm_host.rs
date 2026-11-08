use std::time::Duration;
use zenbench::wasm::{WasmBench, wasmtime};
pub fn register(suite: &mut zenbench::Suite, path: &str) {
    let mut config = wasmtime::Config::new();
    config.wasm_simd(true).wasm_relaxed_simd(true);
    let mode = std::env::var("STUDY_WASM_MODE").unwrap_or_else(|_| "fusing".into());
    if mode == "nonfusing" {
        // Clearing host autodetection safely selects the x86-64 baseline ISA,
        // which has no FMA. This also disables other optional ISA features;
        // the report must not attribute the entire time difference to FMA.
        #[cfg(target_arch = "x86_64")]
        config.target("x86_64-unknown-linux-gnu").unwrap();
        #[cfg(not(target_arch = "x86_64"))]
        panic!("nonfusing baseline engine control requires x86-64");
    }
    if mode == "deterministic" {
        config.relaxed_simd_deterministic(true);
    }
    let wasm = WasmBench::from_bytes_with_config(&std::fs::read(path).unwrap(), config).unwrap();
    let find = |name: &str| {
        let found: Vec<_> = wasm
            .module()
            .exports()
            .filter(|e| e.name().contains(name))
            .map(|e| e.name().to_string())
            .collect();
        assert_eq!(found.len(), 1, "unique export {name}");
        found[0].clone()
    };
    let prepare = find("guest_prepare");
    let run = find("guest_run");
    let probe = find("guest_probe");
    let mut instance = wasm.wasi_instance().unwrap();
    let projection = instance.call::<_, i32>(&probe, ()).unwrap();
    let expected: i32 = std::env::var("STUDY_WASM_EXPECT_FUSION")
        .expect("explicit expected f32/f64 projection bits required")
        .parse()
        .unwrap();
    assert_eq!(
        projection, expected,
        "actually selected engine FMA projection"
    );
    println!("WASM mode={mode} projection={projection} (bit0=f32 fused, bit1=f64 fused)");
    let ops = [
        "exp2_lowp",
        "exp_lowp",
        "pow_lowp",
        "exp2_midp",
        "exp_midp",
        "pow_midp",
        "sigmoid_midp",
        "silu_midp",
        "log2_lowp",
        "ln_lowp",
        "log10_lowp",
        "log2_midp",
        "ln_midp",
        "log10_midp",
        "combined_patch",
    ];
    for (op, name) in ops.iter().enumerate() {
        for width in [4, 8, 16] {
            for n in [64, 4096] {
                for chain in [false, true] {
                    for (dist, distribution) in
                        ["ordinary", "bits", "sparse", "dense"].iter().enumerate()
                    {
                        let id = format!(
                            "wasm/{mode}/{name}/x{width}/{n}/{}/{distribution}",
                            if chain { "chain" } else { "stream" }
                        );
                        if !super::matches_filter(&id) {
                            continue;
                        }
                        suite.group(id, |g| {
                            g.config()
                                .min_rounds(50)
                                .max_rounds(50)
                                .auto_rounds(false)
                                .warmup_time(Duration::from_millis(10))
                                .max_time(Duration::from_millis(100))
                                .max_wall_time(Duration::from_secs(60));
                            for (arm, label) in ["baseline", "exceptional", "fixed_fused"]
                                .iter()
                                .enumerate()
                            {
                                let wasm = wasm.clone();
                                let prepare = prepare.clone();
                                let run = run.clone();
                                g.bench(*label, move |b| {
                                    let mut instance = wasm.wasi_instance().unwrap();
                                    instance
                                        .call::<_, ()>(&prepare, (n, dist as i32, op as i32))
                                        .unwrap();
                                    let f = instance
                                        .typed_func::<(i32, i32, i32, i32, i32), i64>(&run)
                                        .unwrap();
                                    b.iter(|| {
                                        f.call(
                                            &mut instance.store,
                                            (op as i32, arm as i32, width, chain as i32, 128),
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
    let fma = find("guest_fma");
    for (kind, width) in [(32, 1), (32, 4), (64, 1), (64, 2)] {
        for n in [64, 4096] {
            for chain in [false, true] {
                for (dist, distribution) in
                    ["ordinary", "bits", "sparse", "dense"].iter().enumerate()
                {
                    let id = format!(
                        "wasm/{mode}/fma/f{kind}x{width}/{n}/{}/{distribution}",
                        if chain { "chain" } else { "stream" }
                    );
                    if !super::matches_filter(&id) {
                        continue;
                    }
                    suite.group(id, |g| {
                        g.config()
                            .min_rounds(50)
                            .max_rounds(50)
                            .auto_rounds(false)
                            .warmup_time(Duration::from_millis(10))
                            .max_time(Duration::from_millis(100))
                            .max_wall_time(Duration::from_secs(60));
                        for (arm, label) in
                            ["current_fused", "array_software", "whole_buffer_hardware"]
                                .iter()
                                .enumerate()
                        {
                            let wasm = wasm.clone();
                            let prepare = prepare.clone();
                            let fma = fma.clone();
                            g.bench(*label, move |b| {
                                let mut instance = wasm.wasi_instance().unwrap();
                                instance
                                    .call::<_, ()>(&prepare, (n, dist as i32, -kind))
                                    .unwrap();
                                let f = instance
                                    .typed_func::<(i32, i32, i32, i32, i32), i64>(&fma)
                                    .unwrap();
                                b.iter(|| {
                                    f.call(
                                        &mut instance.store,
                                        (kind, width, arm as i32, chain as i32, 128),
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
    // A cold guest instance keeps the fused primitive's probe uninitialized.
    // Instance construction, input preparation and destruction stay outside
    // the timer; the measured unit is the first complete 64-element buffer.
    let id = format!("wasm/{mode}/first-use/exp2_midp/x4/64");
    if super::matches_filter(&id) {
        suite.group(id, |g| {
            g.config()
                .min_rounds(50)
                .max_rounds(50)
                .auto_rounds(false)
                .warmup_time(Duration::from_millis(10))
                .max_time(Duration::from_millis(20))
                .max_wall_time(Duration::from_secs(60));
            for (arm, label) in ["baseline", "exceptional", "fixed_fused"]
                .iter()
                .enumerate()
            {
                let wasm = wasm.clone();
                let prepare = prepare.clone();
                let run = run.clone();
                g.bench(*label, move |b| {
                    let wasm = wasm.clone();
                    let prepare = prepare.clone();
                    let run = run.clone();
                    b.with_input(move || {
                        let mut instance = wasm.wasi_instance().unwrap();
                        instance.call::<_, ()>(&prepare, (64, 0, 3)).unwrap();
                        let f = instance
                            .typed_func::<(i32, i32, i32, i32, i32), i64>(&run)
                            .unwrap();
                        (instance, f)
                    })
                    .run(move |(mut instance, f)| {
                        let result = f
                            .call(&mut instance.store, (3, arm as i32, 4, 0, 1))
                            .unwrap();
                        (instance, result)
                    });
                });
            }
        });
    }
}

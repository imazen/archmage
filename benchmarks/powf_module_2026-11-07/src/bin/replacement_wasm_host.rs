//! Actual legacy/std and const caller comparisons inside one WASM module.
#![forbid(unsafe_code)]
use std::time::Duration;
use zenbench::wasm::{WasmBench, wasmtime};
const LABELS: [&str; 15] = [
    "pow_midp",
    "pow_midp_unchecked",
    "pow_midp_precise",
    "pow_lowp",
    "pow_lowp_unchecked",
    "std_f32_powf",
    "std_promoted_f64_powf",
    "runtime_literal_caller",
    "runtime_dynamic_caller",
    "runtime_literal_auto",
    "runtime_dynamic_auto",
    "const_caller",
    "const_auto",
    "prevalidated_graph",
    "std_f32_powi",
];
#[derive(Clone)]
struct Cell {
    name: String,
    size: usize,
    context: &'static str,
    chain: bool,
    domain: u32,
    exponent: u32,
    dynamic: bool,
    integer: bool,
    distribution: &'static str,
    width: usize,
}
fn matches_filter(name: &str) -> bool {
    std::env::var("POWF_FILTER").map_or(true, |filter| name == filter)
}
fn cells() -> Vec<Cell> {
    let native = "wasm128";
    let contexts = [
        (1, "scalar"),
        (4, "scalar"),
        (8, "scalar"),
        (16, "scalar"),
        (4, native),
        (8, native),
        (16, native),
    ];
    let mut configurations = Vec::new();
    for domain in 0..3 {
        for exponent in [0x4019_999a, 0x3ed5_5555, 0x3f00_0000] {
            for distribution in ["unorm8", "wide"] {
                configurations.push((domain, exponent, false, false, distribution));
            }
        }
    }
    for exponent in [0x3e23_2000, 0x429d_b000] {
        configurations.push((0, exponent, false, false, "ordinary"));
    }
    for distribution in ["unorm8", "wide"] {
        configurations.push((2, 0x4019_999a, true, false, distribution));
    }
    for exponent in [0x4000_0000, 0x4080_0000] {
        configurations.push((2, exponent, false, true, "unorm8"));
    }
    let mut cells = Vec::new();
    for (domain, exponent, dynamic, integer, distribution) in configurations {
        for size in [64, 4096] {
            for chain in [false, true] {
                for (width, context) in contexts {
                    let name = format!(
                        "replacement/x{width}/{context}/{size}/{}/{domain}/{exponent:08x}/{}/{distribution}",
                        if chain { "chain" } else { "stream" },
                        if dynamic { "dynamic-n" } else { "literal-n" }
                    );
                    cells.push(Cell {
                        name,
                        size,
                        context,
                        chain,
                        domain,
                        exponent,
                        dynamic,
                        integer,
                        distribution,
                        width,
                    });
                }
            }
        }
    }
    assert_eq!(cells.len(), 672);
    cells
}

fn main() {
    if std::env::var("POWF_LIST").is_ok() {
        for c in cells() {
            println!("{}", c.name);
        }
        return;
    }
    assert!(!std::env::args().any(|a| a == "--no-busy-gate"));
    assert!(std::env::var("ZENBENCH_NO_BUSY_GATE").map_or(true, |v| v.is_empty() || v == "0"));
    let path = std::env::var("POWF_WASM").expect("module path");
    let mode = std::env::var("POWF_WASM_MODE").expect("strict/fusing/nonfusing");
    assert!(matches!(mode.as_str(), "strict" | "fusing" | "nonfusing"));
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
    assert_eq!(
        fusion,
        i32::from(mode == "fusing"),
        "actual engine fusion projection"
    );
    println!("mode={mode} actual fusion={fusion}");
    let verify_only = std::env::var("POWF_VERIFY_ONLY").is_ok();
    let mut verified = 0;
    for c in cells().iter().filter(|c| matches_filter(&c.name)) {
        check.call::<_, ()>(&prepare, prepare_args(c)).unwrap();
        check
            .call::<_, ()>(
                &verify,
                (
                    c.width as i32,
                    i32::from(c.chain),
                    i32::from(c.context != "scalar"),
                ),
            )
            .unwrap();
        verified += 1;
    }
    println!("verified {verified} selected replacement cells, every arm/lane");
    if verify_only {
        return;
    }
    let output = std::env::var("POWF_RESULT").expect("fresh result path");
    assert!(!std::path::Path::new(&output).exists());
    zenbench::run_gated(zenbench::GateConfig::strict(), |suite| {
        for c in cells().into_iter().filter(|c| matches_filter(&c.name)) {
            suite.group(&c.name, |g| {
                g.config()
                    .min_rounds(50)
                    .max_rounds(50)
                    .auto_rounds(false)
                    .warmup_time(Duration::from_millis(10))
                    .max_time(Duration::from_millis(100))
                    .max_wall_time(Duration::from_secs(120));
                for (arm, label) in LABELS
                    .iter()
                    .take(if c.integer { 15 } else { 14 })
                    .enumerate()
                {
                    let wasm = wasm.clone();
                    let prepare = prepare.clone();
                    let run = run.clone();
                    let c = c.clone();
                    g.bench(*label, move |b| {
                        let mut instance = wasm.wasi_instance().unwrap();
                        instance.call::<_, ()>(&prepare, prepare_args(&c)).unwrap();
                        let function = instance
                            .typed_func::<(i32, i32, i32, i32, i32), i64>(&run)
                            .unwrap();
                        let args = (
                            c.width as i32,
                            arm as i32,
                            i32::from(c.chain),
                            1,
                            i32::from(c.context != "scalar"),
                        );
                        function.call(&mut instance.store, args).unwrap();
                        b.iter(|| {
                            function
                                .call(&mut instance.store, (args.0, args.1, args.2, 128, args.4))
                                .unwrap()
                        });
                    });
                }
            });
        }
    })
    .save(std::path::Path::new(&output))
    .unwrap();
}
fn prepare_args(c: &Cell) -> (i32, i32, i32, i32, i32) {
    (
        c.size as i32,
        match c.distribution {
            "unorm8" => 0,
            "wide" => 1,
            "ordinary" => 2,
            _ => panic!("distribution"),
        },
        c.domain as i32,
        c.exponent as i32,
        i32::from(c.dynamic),
    )
}

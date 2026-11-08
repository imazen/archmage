//! Replacement comparisons: actual legacy/std methods, runtime and const APIs.
#![forbid(unsafe_code)]
use archmage::SimdToken;
use magetypes::powf::{PowfDomain, PowfExecution, PowfOptions};
#[allow(dead_code)]
mod soft {
    include!(concat!(env!("OUT_DIR"), "/soft.rs"));
}
#[allow(dead_code, unused_variables)]
mod prevalidated {
    include!(concat!(env!("OUT_DIR"), "/prevalidated.rs"));
}
include!(concat!(env!("OUT_DIR"), "/replacement_kernels.rs"));
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
fn normalize(x: f32) -> f32 {
    f32::from_bits(0x3f00_0000 | (x.to_bits() & 0x007f_ffff))
}
fn matches_filter(name: &str) -> bool {
    std::env::var("POWF_FILTER").map_or(true, |filter| name == filter)
}
fn value(i: usize, distribution: &str) -> f32 {
    match distribution {
        "unorm8" => ((i % 255) + 1) as f32 / 255.0,
        "wide" => f32::from_bits(0x3780_0000 + (i as u32).wrapping_mul(0x9e37_79b9) % 0x1000_0001),
        "ordinary" => 0.5 + (i % 193) as f32 / 128.0,
        _ => panic!("unknown distribution"),
    }
}
fn cells() -> Vec<Cell> {
    let native = if cfg!(target_arch = "aarch64") {
        "neon"
    } else {
        "v3"
    };
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
fn register_all(mut suite: Option<&mut zenbench::Suite>) {
    for cell in cells() {
        match cell.width {
            1 => replacement_dispatch1(suite.as_deref_mut(), &cell),
            4 => replacement_dispatch4(suite.as_deref_mut(), &cell),
            8 => replacement_dispatch8(suite.as_deref_mut(), &cell),
            16 => replacement_dispatch16(suite.as_deref_mut(), &cell),
            _ => unreachable!(),
        }
    }
}
fn main() {
    if std::env::var("POWF_LIST").is_ok() {
        for cell in cells() {
            println!("{}", cell.name);
        }
        return;
    }
    if std::env::var("POWF_VERIFY_ONLY").is_ok() {
        register_all(None);
        println!("all selected replacement outputs checked");
        return;
    }
    assert!(!std::env::args().any(|a| a == "--no-busy-gate"));
    assert!(std::env::var("ZENBENCH_NO_BUSY_GATE").map_or(true, |v| v.is_empty() || v == "0"));
    let output = std::env::var("POWF_RESULT").expect("set a new POWF_RESULT path");
    assert!(
        !std::path::Path::new(&output).exists(),
        "preserve existing evidence"
    );
    zenbench::run_gated(zenbench::GateConfig::strict(), |suite| {
        register_all(Some(suite))
    })
    .save(std::path::Path::new(&output))
    .unwrap();
}

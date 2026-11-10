//! Actual public policy permutations; see POLICY-PLAN.md.
#![forbid(unsafe_code)]
mod highp_reference {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/highp_reference.rs"));
}
use archmage::SimdToken;
struct Cell {
    name: String,
    width: usize,
    context: &'static str,
    size: usize,
    chain: bool,
    exponent: u32,
}
fn normalize(x: f32) -> f32 {
    f32::from_bits(0x3f000000 | (x.to_bits() & 0x007fffff))
}
include!(concat!(env!("OUT_DIR"), "/policy_bench.rs"));
fn cells() -> Vec<Cell> {
    let native = if cfg!(target_arch = "aarch64") {
        "neon"
    } else {
        "v3"
    };
    let mut cells = Vec::new();
    for width in [1, 4, 8, 16] {
        for context in ["scalar", native] {
            if width == 1 && context != "scalar" {
                continue;
            }
            for size in [64, 4096] {
                for chain in [false, true] {
                    for exponent in [0x4019999a, 0x3f000000] {
                        let name = format!(
                            "policy/x{width}/{context}/{size}/{}/{exponent:08x}",
                            if chain { "chain" } else { "stream" }
                        );
                        cells.push(Cell {
                            name,
                            width,
                            context,
                            size,
                            chain,
                            exponent,
                        });
                    }
                }
            }
        }
    }
    assert_eq!(cells.len(), 56);
    cells
}
fn register_all(mut suite: Option<&mut zenbench::Suite>) {
    for cell in cells() {
        if std::env::var("POWF_FILTER").is_ok_and(|f| f != cell.name) {
            continue;
        }
        macro_rules! go {
            ($n:ident) => {
                match cell.exponent {
                    0x4019999a => $n::<0x4019999a>(suite.as_deref_mut(), &cell),
                    0x3f000000 => $n::<0x3f000000>(suite.as_deref_mut(), &cell),
                    _ => unreachable!(),
                }
            };
        }
        match cell.width {
            1 => go!(register1),
            4 => go!(register4),
            8 => go!(register8),
            16 => go!(register16),
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
        println!("all 56 public policy cells verified");
        return;
    }
    assert!(!std::env::args().any(|a| a == "--no-busy-gate"));
    assert!(std::env::var("ZENBENCH_NO_BUSY_GATE").map_or(true, |v| v.is_empty() || v == "0"));
    let output = std::env::var("POWF_RESULT").expect("new POWF_RESULT required");
    assert!(!std::path::Path::new(&output).exists());
    zenbench::run_gated(zenbench::GateConfig::strict(), |s| register_all(Some(s)))
        .save(std::path::Path::new(&output))
        .unwrap();
}

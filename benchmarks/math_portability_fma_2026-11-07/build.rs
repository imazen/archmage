use std::{env, path::PathBuf, process::Command};
fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("../..");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    for script in ["freeze.py", "kernels.py", "fma_kernels.py"] {
        assert!(Command::new("python3").current_dir(&root)
            .arg(root.join("benchmarks/math_portability_fma_2026-11-07").join(script))
            .arg(&out).status().unwrap().success());
        println!("cargo:rerun-if-changed={script}");
    }
    println!("cargo:rerun-if-changed=../../magetypes/src/soft_fma.rs");
    for width in [4, 8, 16] {
        println!("cargo:rerun-if-changed=../../magetypes/src/simd/generic/generated/transcendentals_f32x{width}.rs");
    }
}

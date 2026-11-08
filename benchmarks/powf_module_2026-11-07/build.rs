use std::{env, path::PathBuf, process::Command};

fn main() {
    let directory = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    assert!(
        Command::new("python3")
            .arg(directory.join("prepare.py"))
            .arg(output)
            .status()
            .unwrap()
            .success()
    );
    println!("cargo:rerun-if-changed=prepare.py");
    println!("cargo:rerun-if-changed=../../magetypes/src/powf/graph.rs");
    println!("cargo:rerun-if-changed=../../magetypes/src/soft_fma.rs");
}

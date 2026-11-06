use std::process::Command;
fn main() {
    let compiler = std::env::var("RUSTC").expect("Cargo RUSTC");
    let output = Command::new(compiler)
        .arg("-vV")
        .output()
        .expect("rustc identity");
    assert!(output.status.success());
    let rustc = String::from_utf8(output.stdout).expect("rustc UTF-8");
    let source = Command::new("python3")
        .args(["drive.py", "stamp"])
        .output()
        .expect("source stamp");
    assert!(
        source.status.success(),
        "source stamp failed: {}",
        String::from_utf8_lossy(&source.stderr)
    );
    let source = String::from_utf8(source.stdout).expect("source stamp UTF-8");
    println!("cargo:rustc-env=DIGEST_SOURCE_SHA256={}", source.trim());
    println!(
        "cargo:rustc-env=DIGEST_RUSTC={}",
        rustc.lines().next().unwrap()
    );
    println!(
        "cargo:rustc-env=DIGEST_TARGET={}",
        std::env::var("TARGET").unwrap()
    );
    // Cargo's default package scan reruns this script when its own files change.
    // Explicit external inputs make the stamp track local production dependencies.
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=drive.py");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=Cargo.lock");
    for path in [
        "../../src",
        "../../archmage-macros/src",
        "../../zenmath/src",
        "../../Cargo.toml",
        "../../Cargo.lock",
        "../../archmage-macros/Cargo.toml",
        "../../zenmath/Cargo.toml",
        "../../token-registry.toml",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }
}

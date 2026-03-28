//! Macro expansion snapshot tests.
//!
//! - `tests/expand/*.rs` — all must compile (unexpanded AND expanded)
//! - `tests/expand/should-fail/` — known bugs, may fail unexpanded and/or expanded
//!
//! To update snapshots: `MACROTEST=overwrite cargo test -p archmage --test macro_expand`
//!
//! Requires `cargo-expand` (`cargo install cargo-expand`).
//! Tests that need it are skipped if not installed.

fn has_cargo_expand() -> bool {
    std::process::Command::new("cargo")
        .args(["expand", "--version"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Expand all passing inputs and diff against `.expanded.rs` snapshots.
#[test]
fn macro_expansion_snapshots() {
    if !has_cargo_expand() {
        eprintln!("SKIPPED: cargo-expand not installed");
        return;
    }
    macrotest::expand("tests/expand/*.rs");
}

/// Snapshot tests for known-buggy expansions.
#[test]
fn macro_expansion_snapshots_known_bugs() {
    if !has_cargo_expand() {
        eprintln!("SKIPPED: cargo-expand not installed");
        return;
    }
    macrotest::expand("tests/expand/should-fail/*.rs");
}

/// Every unexpanded input in expand/ must compile with macros applied.
#[test]
fn unexpanded_input_compiles() {
    let t = trybuild::TestCases::new();
    t.pass("tests/expand/*.rs");
}

/// Every expanded output in expand/ must compile as standalone Rust.
#[test]
fn expanded_output_compiles() {
    let t = trybuild::TestCases::new();
    t.pass("tests/expand/*.expanded.rs");
}

/// Known bugs: expanded output that doesn't compile standalone.
/// When fixed, move from should-fail/ back to expand/.
#[test]
fn expanded_output_known_bugs() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/expand/should-fail/*.expanded.rs");
}

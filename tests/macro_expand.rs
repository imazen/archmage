//! Macro expansion snapshot tests.
//!
//! - `tests/expand/*.rs` — all must compile (unexpanded AND expanded)
//! - `tests/expand/should-fail/` — known bugs, may fail unexpanded and/or expanded
//!
//! To update snapshots: `MACROTEST=overwrite cargo test -p archmage --test macro_expand`
//!
//! Requires `cargo-expand` (`cargo install cargo-expand`).

/// Expand all passing inputs and diff against `.expanded.rs` snapshots.
#[test]
fn macro_expansion_snapshots() {
    macrotest::expand("tests/expand/*.rs");
}

/// Snapshot tests for known-buggy expansions.
#[test]
fn macro_expansion_snapshots_known_bugs() {
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

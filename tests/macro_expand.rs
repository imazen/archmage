//! Macro expansion snapshot tests.
//!
//! Generated test files live in `tests/expand/{category}/`.
//! Known bugs live in `tests/expand/should-fail/`.
//!
//! To regenerate test inputs: `cargo run -p xtask -- gen-expand`
//! To update snapshots: `MACROTEST=overwrite cargo test -p archmage --test macro_expand`
//!
//! Requires `cargo-expand` (`cargo install cargo-expand`).

/// Expand all passing inputs and diff against `.expanded.rs` snapshots.
/// x86_64 only — expansion output is arch-dependent.
#[test]
#[cfg(target_arch = "x86_64")]
fn macro_expansion_snapshots() {
    macrotest::expand("tests/expand/**/*.rs");
}

/// Snapshot tests for known-buggy expansions.
#[test]
#[cfg(target_arch = "x86_64")]
fn macro_expansion_snapshots_known_bugs() {
    macrotest::expand("tests/expand/should-fail/*.rs");
}

/// Every unexpanded input must compile with macros applied.
#[test]
#[cfg(target_arch = "x86_64")]
fn unexpanded_input_compiles() {
    let t = trybuild::TestCases::new();
    t.pass("tests/expand/arcane/*.rs");
    t.pass("tests/expand/rite/*.rs");
    t.pass("tests/expand/autoversion/*.rs");
    t.pass("tests/expand/incant/*.rs");
    t.pass("tests/expand/rewrite/*.rs");
    t.pass("tests/expand/deprecated/*.rs");
    t.pass("tests/expand/combinations/*.rs");
}

/// Every expanded output must compile as standalone Rust.
#[test]
#[cfg(target_arch = "x86_64")]
fn expanded_output_compiles() {
    let t = trybuild::TestCases::new();
    t.pass("tests/expand/arcane/*.expanded.rs");
    t.pass("tests/expand/rite/*.expanded.rs");
    t.pass("tests/expand/autoversion/*.expanded.rs");
    t.pass("tests/expand/incant/*.expanded.rs");
    t.pass("tests/expand/rewrite/*.expanded.rs");
    t.pass("tests/expand/deprecated/*.expanded.rs");
    t.pass("tests/expand/combinations/*.expanded.rs");
}

/// Known bugs: expanded output that doesn't compile standalone.
#[test]
#[cfg(target_arch = "x86_64")]
fn expanded_output_known_bugs() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/expand/should-fail/*.expanded.rs");
}

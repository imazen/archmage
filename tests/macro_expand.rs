//! Macro expansion snapshot tests.
//!
//! - `tests/expand/*.rs` — all must compile (unexpanded AND expanded)
//! - `tests/expand/should-fail/` — known bugs, may fail unexpanded and/or expanded
//!
//! Snapshot diffs (macrotest) only run on x86_64 — expansion is arch-dependent.
//! Compilation tests (trybuild) run on all platforms.
//!
//! To update snapshots (on x86_64):
//!   `MACROTEST=overwrite cargo test -p archmage --test macro_expand`
//!
//! Requires `cargo-expand` (`cargo install cargo-expand`).

/// Expand all passing inputs and diff against `.expanded.rs` snapshots.
/// x86_64 only — expansion output is arch-dependent (cfg-gated code differs).
#[test]
#[cfg(target_arch = "x86_64")]
fn macro_expansion_snapshots() {
    macrotest::expand("tests/expand/*.rs");
}

/// Snapshot tests for known-buggy expansions.
#[test]
#[cfg(target_arch = "x86_64")]
fn macro_expansion_snapshots_known_bugs() {
    macrotest::expand("tests/expand/should-fail/*.rs");
}

/// Every unexpanded input in expand/ must compile with macros applied.
/// Runs on all platforms.
#[test]
fn unexpanded_input_compiles() {
    let t = trybuild::TestCases::new();
    t.pass("tests/expand/*.rs");
}

/// Every expanded output in expand/ must compile as standalone Rust.
/// x86_64 only — snapshots contain x86_64-specific imports and types.
#[test]
#[cfg(target_arch = "x86_64")]
fn expanded_output_compiles() {
    let t = trybuild::TestCases::new();
    t.pass("tests/expand/*.expanded.rs");
}

/// Known bugs: expanded output that doesn't compile standalone.
/// When fixed, move from should-fail/ back to expand/.
#[test]
#[cfg(target_arch = "x86_64")]
fn expanded_output_known_bugs() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/expand/should-fail/*.expanded.rs");
}

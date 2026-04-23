//! Macro expansion snapshot tests for magetypes-specific features.
//!
//! The `archmage` crate has its own macro_expand tests covering `#[arcane]`,
//! `#[rite]`, `#[autoversion]`, `#[incant]`, and the generic `#[magetypes]`
//! expansion. THIS test file covers expansions that emit `::magetypes::simd::*`
//! paths — those can't be tested in archmage's harness because archmage has
//! no `magetypes` dependency.
//!
//! Generated test files live in `tests/expand/{category}/`.
//!
//! To update snapshots: `MACROTEST=overwrite cargo test -p magetypes --test macro_expand`
//!
//! Requires `cargo-expand` (`cargo install cargo-expand`).

/// Expand all passing inputs and diff against `.expanded.rs` snapshots.
/// x86_64 only — expansion output is arch-dependent.
#[test]
#[cfg(target_arch = "x86_64")]
fn macro_expansion_snapshots() {
    macrotest::expand("tests/expand/**/*.rs");
}

/// Every unexpanded input must compile with macros applied.
#[test]
#[cfg(target_arch = "x86_64")]
fn unexpanded_input_compiles() {
    let t = trybuild::TestCases::new();
    t.pass("tests/expand/define/*.rs");
    t.pass("tests/expand/rite_flag/*.rs");
}

/// Every expanded output must compile as standalone Rust — this is the key
/// check that the emitted `::magetypes::simd::generic::*` paths resolve.
#[test]
#[cfg(target_arch = "x86_64")]
fn expanded_output_compiles() {
    let t = trybuild::TestCases::new();
    t.pass("tests/expand/define/*.expanded.rs");
    t.pass("tests/expand/rite_flag/*.expanded.rs");
}

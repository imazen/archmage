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

// Miri isolates the filesystem by default; trybuild/macrotest both do
// disk I/O (glob, compile, write snapshots) which Miri rejects. These
// tests are x86_64-only (expansion output is arch-dependent) AND skipped
// under Miri — the actual macro behavior is exercised by the functional
// tests (magetypes_define_flag.rs, magetypes_rite_flag.rs) which Miri
// can run safely.

/// Expand all passing inputs and diff against `.expanded.rs` snapshots.
#[test]
#[cfg(all(target_arch = "x86_64", not(miri)))]
fn macro_expansion_snapshots() {
    macrotest::expand("tests/expand/**/*.rs");
}

/// Every unexpanded input must compile with macros applied.
#[test]
#[cfg(all(target_arch = "x86_64", not(miri)))]
fn unexpanded_input_compiles() {
    let t = trybuild::TestCases::new();
    t.pass("tests/expand/define/*.rs");
    t.pass("tests/expand/rite_flag/*.rs");
}

/// Every expanded output must compile as standalone Rust — this is the key
/// check that the emitted `::magetypes::simd::generic::*` paths resolve.
#[test]
#[cfg(all(target_arch = "x86_64", not(miri)))]
fn expanded_output_compiles() {
    let t = trybuild::TestCases::new();
    t.pass("tests/expand/define/*.expanded.rs");
    t.pass("tests/expand/rite_flag/*.expanded.rs");
}

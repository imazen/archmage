//! Macro expansion snapshot tests.
//!
//! Generated test files live in `tests/expand/{category}/`.
//! Known bugs live in `tests/expand/should-fail/`. Signature shapes (bound
//! placement, parameter patterns, receivers, generics, return types) live in
//! `tests/expand/shapes/`; a shape whose expansion is wrong today is listed in
//! `KNOWN_FAILURES` in `xtask/src/expand_gen.rs` and lands in `should-fail/`.
//!
//! To regenerate test inputs: `cargo run -p xtask -- gen-expand`
//! To update snapshots: `MACROTEST=overwrite cargo test -p archmage --test macro_expand`
//!
//! Requires `cargo-expand` (`cargo install cargo-expand --locked --version 1.0.126`).
//! CI pins this version for macrotest 1.2.1: cargo-expand 1.0.127 returns a
//! nonzero status for our deliberately invalid snapshot inputs. Their rejection
//! is checked separately by soundness_exploits.rs; snapshot expectations stay
//! unchanged.
//!
//! These `.expanded.rs` files are what `cargo expand` prints: the crate after
//! rustc has evaluated every `cfg`, so a false gate removes the item and a true
//! gate loses its attribute, and the foreign-architecture variants are absent
//! on an x86-64 host. The same inputs expanded by the macro implementations
//! themselves, before cfg evaluation (all architectures, gates and diagnostics
//! visible), are the raw snapshots under `tests/expand-raw/`, written and
//! checked by `archmage-macros/src/raw_snapshots.rs`:
//! `ARCHMAGE_RAW_SNAPSHOTS=overwrite cargo test -p archmage-macros raw_snapshots`.

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
    t.pass("tests/expand/shapes/*.rs");
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
    t.pass("tests/expand/shapes/*.expanded.rs");
}

// Known-bug rejection reasons are checked by soundness_exploits.rs using
// stable error codes/symbols rather than line-number and diagnostic-note snapshots.

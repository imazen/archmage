//! Macro expansion snapshot tests.
//!
//! Each `.rs` file in `tests/expand/` uses the real macros as a user would.
//! `macrotest` runs `cargo expand` on each and diffs against `.expanded.rs` snapshots.
//!
//! To update snapshots: `MACROTEST=overwrite cargo +nightly test -p archmage --test macro_expand`
//! To check snapshots: `cargo +nightly test -p archmage --test macro_expand`

#[test]
fn macro_expansion_snapshots() {
    macrotest::expand("tests/expand/*.rs");
}

//! Macro expansion snapshot tests.
//!
//! Each `.rs` file in `tests/expand/` uses the real macros as a user would.
//! `macrotest` runs `cargo expand` on each and diffs against `.expanded.rs` snapshots.
//!
//! To update snapshots: `MACROTEST=overwrite cargo test -p archmage --test macro_expand`

#[test]
fn macro_expansion_snapshots() {
    // This both compiles the unexpanded input (verifying macros produce valid code)
    // and diffs the expanded output against checked-in snapshots.
    macrotest::expand("tests/expand/*.rs");
}

/// Verify that expanded snapshots compile as standalone Rust.
///
/// Excluded:
/// - `*_stub.expanded.rs` — stubs expand `unreachable!()` to internal `core::panicking`
///   which isn't valid standalone Rust (cargo expand artifact, not a macro bug)
/// - `incant_passthrough.expanded.rs` — labeled blocks expand to internal compiler plumbing
/// - `autoversion_unsafe_fn.expanded.rs` — known bug: dispatcher drops `unsafe` from `unsafe fn`
///   (tracked, will be fixed before this exclusion is removed)
#[test]
fn expanded_snapshots_compile() {
    let t = trybuild::TestCases::new();
    // Test all expanded files EXCEPT known cargo-expand artifacts and known bugs
    t.pass("tests/expand/arcane_calls_rite.expanded.rs");
    t.pass("tests/expand/arcane_cfg_feature.expanded.rs");
    t.pass("tests/expand/arcane_import_intrinsics.expanded.rs");
    t.pass("tests/expand/arcane_nested_trait_impl.expanded.rs");
    t.pass("tests/expand/arcane_sibling_concrete.expanded.rs");
    t.pass("tests/expand/arcane_sibling_generic.expanded.rs");
    t.pass("tests/expand/arcane_sibling_self.expanded.rs");
    t.pass("tests/expand/arcane_sibling_trait_bound.expanded.rs");
    t.pass("tests/expand/arcane_unsafe_body.expanded.rs");
    t.pass("tests/expand/arcane_unsafe_fn.expanded.rs");
    t.pass("tests/expand/autoversion_basic.expanded.rs");
    t.pass("tests/expand/autoversion_chain.expanded.rs");
    t.pass("tests/expand/autoversion_explicit_tiers.expanded.rs");
    t.pass("tests/expand/autoversion_scalar_token.expanded.rs");
    t.pass("tests/expand/autoversion_self_type.expanded.rs");
    t.pass("tests/expand/autoversion_tier_modifiers.expanded.rs");
    t.pass("tests/expand/incant_entry.expanded.rs");
    t.pass("tests/expand/incant_explicit_tiers.expanded.rs");
    t.pass("tests/expand/plain_token_fn.expanded.rs");
    t.pass("tests/expand/rite_import_intrinsics.expanded.rs");
    t.pass("tests/expand/rite_multi_tier.expanded.rs");
    t.pass("tests/expand/rite_single_tier.expanded.rs");
    t.pass("tests/expand/rite_single_token.expanded.rs");
    t.pass("tests/expand/rite_unsafe_fn.expanded.rs");
    t.pass("tests/expand/token_downgrade.expanded.rs");
    t.pass("tests/expand/token_upgrade_conditional.expanded.rs");
    // EXCLUDED — cargo expand artifacts (not real bugs):
    // t.pass("tests/expand/arcane_stub.expanded.rs");        // unreachable!() → core::panicking
    // t.pass("tests/expand/rite_multi_tier_stub.expanded.rs"); // same
    // t.pass("tests/expand/incant_passthrough.expanded.rs");   // labeled block internals
    // EXCLUDED — known bug (dispatcher drops unsafe from unsafe fn):
    // t.pass("tests/expand/autoversion_unsafe_fn.expanded.rs");
}

//! Macro expansion snapshot tests.
//!
//! Each `.rs` file in `tests/expand/` uses the real macros as a user would.
//! `macrotest` runs `cargo expand` on each and diffs against `.expanded.rs` snapshots.
//!
//! To update snapshots: `MACROTEST=overwrite cargo test -p archmage --test macro_expand`

#[test]
fn macro_expansion_snapshots() {
    macrotest::expand("tests/expand/*.rs");
}

/// Verify that every expanded snapshot compiles as standalone Rust.
///
/// This catches bugs where macros generate code that only works as tokens
/// in the original context but wouldn't compile if written by hand.
///
/// Known bugs are tested with `compile_fail` — when fixed, they'll fail
/// this test (prompting removal of the compile_fail and addition of pass).
#[test]
fn expanded_snapshots_compile() {
    let t = trybuild::TestCases::new();

    // === All expanded files that should compile ===
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

    // === Fixed bugs — expanded output now compiles ===
    t.pass("tests/expand/autoversion_unsafe_fn.expanded.rs");

    // === Known bugs — expanded output doesn't compile ===
    // When fixed: move to t.pass() and delete the .stderr file
    //
    // Bug: #[rite] on trait impl — #[target_feature] on safe trait method is invalid
    t.compile_fail("tests/expand/rite_trait_impl.expanded.rs");
    // Bug: #[autoversion] on trait impl — variants placed inside trait impl block
    t.compile_fail("tests/expand/autoversion_trait_impl.expanded.rs");

    // === Excluded — cargo expand artifacts (not macro bugs) ===
    // unreachable!() expands to internal core::panicking; labeled blocks to compiler internals
    // t.pass("tests/expand/arcane_stub.expanded.rs");
    // t.pass("tests/expand/rite_multi_tier_stub.expanded.rs");
    // t.pass("tests/expand/incant_passthrough.expanded.rs");
}
